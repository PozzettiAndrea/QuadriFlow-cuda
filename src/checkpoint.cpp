#include "checkpoint.hpp"
#include "parametrizer.hpp"
#include "serialize.hpp"

#include <cstring>
#include <ctime>
#include <sys/stat.h>

namespace qflow {

// ============================================================
// Stage name mapping
// ============================================================

static const char* stage_names[] = {
    "post-init",
    "post-orient",
    "post-field",
    "post-edgeinfo",
    "post-constraints",
    "post-flow",
    "post-subdiv1",
    "pre-ffh",
    "post-ffh",
    "post-subdiv2",
    "post-extract",
    "post-dynamic",
};

PipelineStage stage_from_name(const char* name) {
    for (int i = 0; i < STAGE_COUNT; ++i) {
        if (strcmp(name, stage_names[i]) == 0) return (PipelineStage)i;
    }
    return STAGE_NONE;
}

const char* stage_name(PipelineStage s) {
    if (s >= 0 && s < STAGE_COUNT) return stage_names[s];
    return "unknown";
}

// ============================================================
// DEdge serialization (two ints)
// ============================================================

static void SaveDEdge(FILE* fp, const DEdge& e) {
    fwrite(&e.x, sizeof(int), 1, fp);
    fwrite(&e.y, sizeof(int), 1, fp);
}

static void ReadDEdge(FILE* fp, DEdge& e) {
    fread(&e.x, sizeof(int), 1, fp);
    fread(&e.y, sizeof(int), 1, fp);
}

static void SaveDEdgeVec(FILE* fp, const std::vector<DEdge>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& e : v) SaveDEdge(fp, e);
}

static void ReadDEdgeVec(FILE* fp, std::vector<DEdge>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& e : v) ReadDEdge(fp, e);
}

// Vector3d serialization for std::vector<Vector3d>
static void SaveVec3dVec(FILE* fp, const std::vector<Vector3d>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& x : v) {
        double d[3] = {x[0], x[1], x[2]};
        fwrite(d, sizeof(double), 3, fp);
    }
}

static void ReadVec3dVec(FILE* fp, std::vector<Vector3d>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& x : v) {
        double d[3];
        fread(d, sizeof(double), 3, fp);
        x = Vector3d(d[0], d[1], d[2]);
    }
}

// Vector4i serialization for std::vector<Vector4i>
static void SaveVec4iVec(FILE* fp, const std::vector<Vector4i>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& x : v) {
        int d[4] = {x[0], x[1], x[2], x[3]};
        fwrite(d, sizeof(int), 4, fp);
    }
}

static void ReadVec4iVec(FILE* fp, std::vector<Vector4i>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& x : v) {
        int d[4];
        fread(d, sizeof(int), 4, fp);
        x = Vector4i(d[0], d[1], d[2], d[3]);
    }
}

// Vector2i serialization for std::vector<Vector2i>
static void SaveVec2iVec(FILE* fp, const std::vector<Vector2i>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& x : v) {
        int d[2] = {x[0], x[1]};
        fwrite(d, sizeof(int), 2, fp);
    }
}

static void ReadVec2iVec(FILE* fp, std::vector<Vector2i>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& x : v) {
        int d[2];
        fread(d, sizeof(int), 2, fp);
        x = Vector2i(d[0], d[1]);
    }
}

// Vector3i serialization for std::vector<Vector3i>
static void SaveVec3iVec(FILE* fp, const std::vector<Vector3i>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& x : v) {
        int d[3] = {x[0], x[1], x[2]};
        fwrite(d, sizeof(int), 3, fp);
    }
}

static void ReadVec3iVec(FILE* fp, std::vector<Vector3i>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& x : v) {
        int d[3];
        fread(d, sizeof(int), 3, fp);
        x = Vector3i(d[0], d[1], d[2]);
    }
}

// std::vector<int> serialization
static void SaveIntVec(FILE* fp, const std::vector<int>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    if (n > 0) fwrite(v.data(), sizeof(int), n, fp);
}

static void ReadIntVec(FILE* fp, std::vector<int>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    if (n > 0) fread(v.data(), sizeof(int), n, fp);
}

// std::vector<std::vector<int>> serialization
static void SaveIntVecVec(FILE* fp, const std::vector<std::vector<int>>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& inner : v) SaveIntVec(fp, inner);
}

static void ReadIntVecVec(FILE* fp, std::vector<std::vector<int>>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& inner : v) ReadIntVec(fp, inner);
}

// variables: std::vector<std::pair<Vector2i, int>>
static void SaveVariables(FILE* fp, const std::vector<std::pair<Vector2i, int>>& v) {
    int n = (int)v.size();
    fwrite(&n, sizeof(int), 1, fp);
    for (auto& p : v) {
        int d[3] = {p.first[0], p.first[1], p.second};
        fwrite(d, sizeof(int), 3, fp);
    }
}

static void ReadVariables(FILE* fp, std::vector<std::pair<Vector2i, int>>& v) {
    int n;
    fread(&n, sizeof(int), 1, fp);
    v.resize(n);
    for (auto& p : v) {
        int d[3];
        fread(d, sizeof(int), 3, fp);
        p.first = Vector2i(d[0], d[1]);
        p.second = d[2];
    }
}

// ============================================================
// Checkpoint file path
// ============================================================

static std::string checkpoint_path(const char* dir, PipelineStage stage) {
    return std::string(dir) + "/" + stage_names[stage] + ".qfc";
}

bool checkpoint_exists(const char* dir, PipelineStage stage) {
    struct stat st;
    std::string path = checkpoint_path(dir, stage);
    return stat(path.c_str(), &st) == 0;
}

// ============================================================
// Save checkpoint
// ============================================================

void save_checkpoint(const Parametrizer& p_const, PipelineStage stage,
                     const char* dir, const char* input_mesh, int target_faces) {
    // Need non-const for Hierarchy::SaveToFile (doesn't modify, just not marked const)
    Parametrizer& p = const_cast<Parametrizer&>(p_const);
    // Ensure directory exists
    mkdir(dir, 0755);

    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    // Write header
    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "QFC", 4);
    hdr.version = 1;
    strncpy(hdr.stage, stage_names[stage], sizeof(hdr.stage) - 1);
    hdr.fixflip_strategy = p.hierarchy.fixflip_strategy;
    hdr.dse_strategy = p.hierarchy.dse_strategy;
    hdr.subdiv_strategy = p.hierarchy.subdiv_strategy;
    hdr.flow_strategy = p.hierarchy.flow_strategy;
    hdr.flag_preserve_sharp = p.flag_preserve_sharp;
    hdr.flag_preserve_boundary = p.flag_preserve_boundary;
    hdr.flag_adaptive_scale = p.flag_adaptive_scale;
    hdr.flag_aggresive_sat = p.flag_aggresive_sat;
    hdr.flag_minimum_cost_flow = p.flag_minimum_cost_flow;
    hdr.target_faces = target_faces;
    if (input_mesh) strncpy(hdr.input_mesh, input_mesh, sizeof(hdr.input_mesh) - 1);
    hdr.timestamp = (long long)time(nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);

    // Write stage index so we know what data follows
    int stage_idx = (int)stage;
    fwrite(&stage_idx, sizeof(int), 1, fp);

    // ---- Always save: scalar fields ----
    Save(fp, p.normalize_scale);
    Save(fp, p.normalize_offset);
    Save(fp, p.surface_area);
    Save(fp, p.scale);
    Save(fp, p.average_edge_length);
    Save(fp, p.max_edge_length);

    // ---- Always save: hierarchy (full) ----
    // Save hierarchy strategy flags
    Save(fp, p.hierarchy.fixflip_strategy);
    Save(fp, p.hierarchy.dse_strategy);
    Save(fp, p.hierarchy.subdiv_strategy);
    Save(fp, p.hierarchy.flow_strategy);
    Save(fp, p.hierarchy.rng_seed);
    Save(fp, p.hierarchy.with_scale);
    p.hierarchy.SaveToFile(fp);

    // ---- Always save: base mesh arrays (may have been moved into hierarchy) ----
    // After Initialize(), V/F/N/E2E are moved into hierarchy, so save from hierarchy
    // But V2E, boundary, nonManifold, rho stay in Parametrizer
    Save(fp, p.V2E);
    Save(fp, p.E2E);
    Save(fp, p.boundary);
    Save(fp, p.nonManifold);
    Save(fp, p.rho);
    Save(fp, p.V);
    Save(fp, p.F);
    Save(fp, p.N);
    Save(fp, p.Nf);
    Save(fp, p.A);
    SaveIntVec(fp, p.sharp_edges);

    // ---- Save flags ----
    Save(fp, p.flag_preserve_sharp);
    Save(fp, p.flag_preserve_boundary);
    Save(fp, p.flag_adaptive_scale);
    Save(fp, p.flag_aggresive_sat);
    Save(fp, p.flag_minimum_cost_flow);

    // ---- Singularities (available after post-orient) ----
    if (stage >= STAGE_POST_ORIENT) {
        Save(fp, p.singularities);
    }

    // ---- Position singularities (available after post-field) ----
    if (stage >= STAGE_POST_FIELD) {
        Save(fp, p.pos_sing);
        Save(fp, p.pos_rank);
        Save(fp, p.pos_index);
    }

    // ---- Edge info (available after post-edgeinfo) ----
    if (stage >= STAGE_POST_EDGEINFO) {
        SaveVec2iVec(fp, p.edge_diff);
        SaveDEdgeVec(fp, p.edge_values);
        SaveVec3iVec(fp, p.face_edgeIds);
        SaveVec3iVec(fp, p.face_edgeOrients);
        SaveIntVec(fp, p.allow_changes);
    }

    // ---- Variables (available after post-constraints) ----
    if (stage >= STAGE_POST_CONSTRAINTS) {
        SaveVariables(fp, p.variables);
    }

    // ---- Compact quad mesh (available after post-extract) ----
    if (stage >= STAGE_POST_EXTRACT) {
        SaveVec3dVec(fp, p.O_compact);
        SaveVec3dVec(fp, p.Q_compact);
        SaveVec3dVec(fp, p.N_compact);
        SaveVec4iVec(fp, p.F_compact);
        SaveIntVecVec(fp, p.Vset);
        SaveIntVec(fp, p.V2E_compact);
        SaveIntVec(fp, p.E2E_compact);
        Save(fp, p.boundary_compact);
        Save(fp, p.nonManifold_compact);
    }

    fclose(fp);

    // Print summary
    long file_size = 0;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) file_size = st.st_size;
    printf("[CHECKPOINT] Saved '%s' to %s (%.1f MB)\n",
           stage_names[stage], path.c_str(), file_size / (1024.0 * 1024.0));
    printf("[CHECKPOINT]   strategies: ff=%d subdiv=%d dse=%d flow=%d | flags: sharp=%d boundary=%d adaptive=%d sat=%d mcf=%d\n",
           hdr.fixflip_strategy, hdr.subdiv_strategy, hdr.dse_strategy, hdr.flow_strategy,
           hdr.flag_preserve_sharp, hdr.flag_preserve_boundary,
           hdr.flag_adaptive_scale, hdr.flag_aggresive_sat, hdr.flag_minimum_cost_flow);
}

// ============================================================
// Load checkpoint
// ============================================================

PipelineStage load_checkpoint(Parametrizer& p, const char* dir, PipelineStage stage) {
    std::string path = checkpoint_path(dir, stage);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        printf("[CHECKPOINT] ERROR: Cannot open %s for reading\n", path.c_str());
        return STAGE_NONE;
    }

    // Read header
    CheckpointHeader hdr;
    fread(&hdr, sizeof(hdr), 1, fp);
    if (memcmp(hdr.magic, "QFC", 4) != 0) {
        printf("[CHECKPOINT] ERROR: Invalid magic in %s\n", path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }
    if (hdr.version != 1) {
        printf("[CHECKPOINT] ERROR: Unsupported version %d in %s\n", hdr.version, path.c_str());
        fclose(fp);
        return STAGE_NONE;
    }

    int stage_idx;
    fread(&stage_idx, sizeof(int), 1, fp);
    PipelineStage saved_stage = (PipelineStage)stage_idx;

    printf("[CHECKPOINT] Loading '%s' from %s\n", stage_names[saved_stage], path.c_str());
    printf("[CHECKPOINT]   saved with: ff=%d subdiv=%d dse=%d flow=%d | flags: sharp=%d boundary=%d adaptive=%d sat=%d mcf=%d\n",
           hdr.fixflip_strategy, hdr.subdiv_strategy, hdr.dse_strategy, hdr.flow_strategy,
           hdr.flag_preserve_sharp, hdr.flag_preserve_boundary,
           hdr.flag_adaptive_scale, hdr.flag_aggresive_sat, hdr.flag_minimum_cost_flow);
    printf("[CHECKPOINT]   input mesh: %s | target faces: %d\n", hdr.input_mesh, hdr.target_faces);

    // Read scalars
    Read(fp, p.normalize_scale);
    Read(fp, p.normalize_offset);
    Read(fp, p.surface_area);
    Read(fp, p.scale);
    Read(fp, p.average_edge_length);
    Read(fp, p.max_edge_length);

    // Read hierarchy strategy flags
    Read(fp, p.hierarchy.fixflip_strategy);
    Read(fp, p.hierarchy.dse_strategy);
    Read(fp, p.hierarchy.subdiv_strategy);
    Read(fp, p.hierarchy.flow_strategy);
    Read(fp, p.hierarchy.rng_seed);
    Read(fp, p.hierarchy.with_scale);
    p.hierarchy.LoadFromFile(fp);

    // Read base mesh arrays
    Read(fp, p.V2E);
    Read(fp, p.E2E);
    Read(fp, p.boundary);
    Read(fp, p.nonManifold);
    Read(fp, p.rho);
    Read(fp, p.V);
    Read(fp, p.F);
    Read(fp, p.N);
    Read(fp, p.Nf);
    Read(fp, p.A);
    ReadIntVec(fp, p.sharp_edges);

    // Read flags
    Read(fp, p.flag_preserve_sharp);
    Read(fp, p.flag_preserve_boundary);
    Read(fp, p.flag_adaptive_scale);
    Read(fp, p.flag_aggresive_sat);
    Read(fp, p.flag_minimum_cost_flow);

    // Singularities
    if (saved_stage >= STAGE_POST_ORIENT) {
        Read(fp, p.singularities);
    }

    // Position singularities
    if (saved_stage >= STAGE_POST_FIELD) {
        Read(fp, p.pos_sing);
        Read(fp, p.pos_rank);
        Read(fp, p.pos_index);
    }

    // Edge info
    if (saved_stage >= STAGE_POST_EDGEINFO) {
        ReadVec2iVec(fp, p.edge_diff);
        ReadDEdgeVec(fp, p.edge_values);
        ReadVec3iVec(fp, p.face_edgeIds);
        ReadVec3iVec(fp, p.face_edgeOrients);
        ReadIntVec(fp, p.allow_changes);
    }

    // Variables
    if (saved_stage >= STAGE_POST_CONSTRAINTS) {
        ReadVariables(fp, p.variables);
    }

    // Compact quad mesh
    if (saved_stage >= STAGE_POST_EXTRACT) {
        ReadVec3dVec(fp, p.O_compact);
        ReadVec3dVec(fp, p.Q_compact);
        ReadVec3dVec(fp, p.N_compact);
        ReadVec4iVec(fp, p.F_compact);
        ReadIntVecVec(fp, p.Vset);
        ReadIntVec(fp, p.V2E_compact);
        ReadIntVec(fp, p.E2E_compact);
        Read(fp, p.boundary_compact);
        Read(fp, p.nonManifold_compact);
    }

    fclose(fp);
    return saved_stage;
}

} // namespace qflow
