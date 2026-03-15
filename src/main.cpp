#include "checkpoint.hpp"
#include "config.hpp"
#include "field-math.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"
#include "subdivide.hpp"
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace qflow;

Parametrizer field;

// ============================================================
// Pipeline stage runner with checkpoint support
//
// Stages are run in order. If -run-from is set, stages before
// that point are skipped (state loaded from checkpoint).
// If -run-to is set, stages after that point are skipped.
// If -save-at is set, a checkpoint is saved after that stage.
// If -save-all is set, checkpoints are saved after every stage.
// ============================================================

static void print_usage() {
    printf("Usage: quadriflow [options]\n");
    printf("  -i <input.obj>          Input mesh\n");
    printf("  -o <output.obj>         Output mesh\n");
    printf("  -f <faces>              Target face count\n");
    printf("  -sharp                  Preserve sharp edges\n");
    printf("  -boundary               Preserve boundary\n");
    printf("  -adaptive               Adaptive scale\n");
    printf("  -mcf                    Minimum cost flow\n");
    printf("  -sat                    Aggressive SAT\n");
    printf("  -seed <n>               RNG seed\n");
    printf("  -G, --cuda              Enable CUDA for all strategies\n");
    printf("  -ff <strategy>          FixFlip strategy: cpu, gpu-prefilter, gpu-only\n");
    printf("  -ff-depth <n>           FixFlip max recursion depth (default: 5)\n");
    printf("  -subdiv <strategy>      Subdivide strategy: cpu, cuda\n");
    printf("  -dse <strategy>         DownsampleEdgeGraph strategy: cpu, cuda\n");
    printf("  -flow <strategy>        Max-flow solver: boykov, cuda, lemon, edkarp, dinic\n");
    printf("\n");
    printf("Checkpoint options:\n");
    printf("  -save-dir <dir>         Directory for checkpoint files\n");
    printf("  -save-at <stage>        Save checkpoint after this stage\n");
    printf("  -save-all               Save checkpoint after every stage\n");
    printf("  -run-from <stage>       Load checkpoint and resume from this stage\n");
    printf("  -run-to <stage>         Stop after this stage\n");
    printf("  -list-stages            List all stage names\n");
    printf("\n");
    printf("Legacy benchmark options:\n");
    printf("  -save-ff <path>         Save FixFlip state (legacy)\n");
    printf("  -bench-ff <path>        Bench FixFlipHierarchy only (legacy)\n");
    printf("\n");
    printf("Stages: post-init, post-orient, post-field, post-edgeinfo,\n");
    printf("        post-constraints, post-flow, post-subdiv1, pre-ffh,\n");
    printf("        post-ffh, post-subdiv2, post-extract, post-dynamic\n");
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

#ifdef WITH_CUDA
    cudaFree(0);
#endif
    unsigned long long t1, t2;
    std::string input_obj, output_obj;
    std::string save_ff_path, bench_ff_path;
    std::string save_dir;
    PipelineStage save_at = STAGE_NONE;
    PipelineStage run_from = STAGE_NONE;
    PipelineStage run_to = STAGE_NONE;
    bool save_all = false;
    int faces = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            sscanf(argv[++i], "%d", &faces);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_obj = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_obj = argv[++i];
        } else if (strcmp(argv[i], "-sharp") == 0) {
            field.flag_preserve_sharp = 1;
        } else if (strcmp(argv[i], "-boundary") == 0) {
            field.flag_preserve_boundary = 1;
        } else if (strcmp(argv[i], "-adaptive") == 0) {
            field.flag_adaptive_scale = 1;
        } else if (strcmp(argv[i], "-mcf") == 0) {
            field.flag_minimum_cost_flow = 1;
        } else if (strcmp(argv[i], "-sat") == 0) {
            field.flag_aggresive_sat = 1;
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            field.hierarchy.rng_seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ff") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "cpu") == 0)                 field.hierarchy.fixflip_strategy = 0;
            else if (strcmp(s, "gpu-prefilter") == 0)  field.hierarchy.fixflip_strategy = 1;
            else if (strcmp(s, "gpu-only") == 0)       field.hierarchy.fixflip_strategy = 2;
            else { printf("Unknown -ff strategy: %s\n", s); return 1; }
        } else if (strcmp(argv[i], "-ff-depth") == 0 && i + 1 < argc) {
            field.hierarchy.fixflip_max_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-subdiv") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "cpu") == 0)        field.hierarchy.subdiv_strategy = 0;
            else if (strcmp(s, "cuda") == 0)  field.hierarchy.subdiv_strategy = 1;
            else { printf("Unknown -subdiv strategy: %s\n", s); return 1; }
        } else if (strcmp(argv[i], "-dse") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "cpu") == 0)        field.hierarchy.dse_strategy = 0;
            else if (strcmp(s, "cuda") == 0)  field.hierarchy.dse_strategy = 1;
            else { printf("Unknown -dse strategy: %s\n", s); return 1; }
        } else if (strcmp(argv[i], "-flow") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "boykov") == 0)     field.hierarchy.flow_strategy = 0;
            else if (strcmp(s, "cuda") == 0)  field.hierarchy.flow_strategy = 1;
            else if (strcmp(s, "lemon") == 0) field.hierarchy.flow_strategy = 2;
            else if (strcmp(s, "edkarp") == 0) field.hierarchy.flow_strategy = 3;
            else if (strcmp(s, "dinic") == 0) field.hierarchy.flow_strategy = 4;
            else { printf("Unknown -flow strategy: %s (valid: boykov, cuda, lemon, edkarp, dinic)\n", s); return 1; }
        } else if (strcmp(argv[i], "-G") == 0 || strcmp(argv[i], "--cuda") == 0) {
            field.hierarchy.subdiv_strategy = 1;
            field.hierarchy.dse_strategy = 1;
            field.hierarchy.flow_strategy = 3;  // edkarp (reliable)
            // fixflip stays cpu (GPU doesn't help, see experiments.md §6)
        } else if (strcmp(argv[i], "-save-dir") == 0 && i + 1 < argc) {
            save_dir = argv[++i];
        } else if (strcmp(argv[i], "-save-at") == 0 && i + 1 < argc) {
            save_at = stage_from_name(argv[++i]);
            if (save_at == STAGE_NONE) { printf("Unknown stage: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "-save-all") == 0) {
            save_all = true;
        } else if (strcmp(argv[i], "-run-from") == 0 && i + 1 < argc) {
            run_from = stage_from_name(argv[++i]);
            if (run_from == STAGE_NONE) { printf("Unknown stage: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "-run-to") == 0 && i + 1 < argc) {
            run_to = stage_from_name(argv[++i]);
            if (run_to == STAGE_NONE) { printf("Unknown stage: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "-list-stages") == 0) {
            printf("Pipeline stages:\n");
            for (int s = 0; s < STAGE_COUNT; ++s)
                printf("  %2d. %s\n", s, stage_name((PipelineStage)s));
            return 0;
        } else if (strcmp(argv[i], "-save-ff") == 0 && i + 1 < argc) {
            save_ff_path = argv[++i];
        } else if (strcmp(argv[i], "-bench-ff") == 0 && i + 1 < argc) {
            bench_ff_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    // ---- Legacy bench mode ----
    if (!bench_ff_path.empty()) {
        printf("=== BENCH MODE: FixFlipHierarchy only ===\n");
        field.LoadFFState(bench_ff_path.c_str());
        printf("Strategy: %d (0=cpu, 1=gpu-prefilter, 2=gpu-only)\n",
               field.hierarchy.fixflip_strategy);
        t1 = GetCurrentTime64();
        field.FixFlipHierarchy();
        t2 = GetCurrentTime64();
        printf("[BENCH] FixFlipHierarchy: %.3f s\n", (t2 - t1) * 1e-3);
        return 0;
    }

    // ---- Helper: should we run a stage? ----
    auto should_run = [&](PipelineStage stage) -> bool {
        if (run_from != STAGE_NONE && stage <= run_from) return false;
        if (run_to != STAGE_NONE && stage > run_to) return false;
        return true;
    };

    // ---- Helper: save checkpoint if requested ----
    auto maybe_save = [&](PipelineStage stage) {
        if (!save_dir.empty() && (save_all || stage == save_at)) {
            save_checkpoint(field, stage, save_dir.c_str(), input_obj.c_str(), faces);
        }
    };

    // ---- Helper: stop check ----
    auto should_stop = [&](PipelineStage stage) -> bool {
        return (run_to != STAGE_NONE && stage >= run_to);
    };

    // ---- Load checkpoint if -run-from specified ----
    if (run_from != STAGE_NONE) {
        if (save_dir.empty()) {
            printf("ERROR: -run-from requires -save-dir\n");
            return 1;
        }
        // Save CLI-specified strategies before checkpoint load overwrites them
        int cli_ff = field.hierarchy.fixflip_strategy;
        int cli_subdiv = field.hierarchy.subdiv_strategy;
        int cli_dse = field.hierarchy.dse_strategy;
        int cli_flow = field.hierarchy.flow_strategy;

        PipelineStage loaded = load_checkpoint(field, save_dir.c_str(), run_from);
        if (loaded == STAGE_NONE) {
            printf("ERROR: Failed to load checkpoint for stage '%s'\n", stage_name(run_from));
            return 1;
        }

        // CLI flags override checkpoint-saved strategies
        field.hierarchy.fixflip_strategy = cli_ff;
        field.hierarchy.subdiv_strategy = cli_subdiv;
        field.hierarchy.dse_strategy = cli_dse;
        field.hierarchy.flow_strategy = cli_flow;

        printf("[PIPELINE] Resuming after stage '%s'\n", stage_name(run_from));
    }

    // ================================================================
    // PHASE 1: INITIALIZATION
    // ================================================================

    if (should_run(STAGE_POST_INIT)) {
        if (input_obj.empty()) {
            printf("ERROR: -i <input.obj> required\n");
            return 1;
        }
        printf("%d %s %s\n", faces, input_obj.c_str(), output_obj.c_str());
        field.Load(input_obj.c_str());

        printf("Initialize...\n");
        t1 = GetCurrentTime64();
        field.Initialize(faces);
        t2 = GetCurrentTime64();
        printf("[TIMING] Initialize: %lf s\n", (t2 - t1) * 1e-3);

        if (field.flag_preserve_boundary) {
            printf("Add boundary constraints...\n");
            Hierarchy& mRes = field.hierarchy;
            mRes.clearConstraints();
            for (uint32_t i = 0; i < 3 * mRes.mF.cols(); ++i) {
                if (mRes.mE2E[i] == -1) {
                    uint32_t i0 = mRes.mF(i % 3, i / 3);
                    uint32_t i1 = mRes.mF((i + 1) % 3, i / 3);
                    Vector3d p0 = mRes.mV[0].col(i0), p1 = mRes.mV[0].col(i1);
                    Vector3d edge = p1 - p0;
                    if (edge.squaredNorm() > 0) {
                        edge.normalize();
                        mRes.mCO[0].col(i0) = p0;
                        mRes.mCO[0].col(i1) = p1;
                        mRes.mCQ[0].col(i0) = mRes.mCQ[0].col(i1) = edge;
                        mRes.mCQw[0][i0] = mRes.mCQw[0][i1] = mRes.mCOw[0][i0] = mRes.mCOw[0][i1] = 1.0;
                    }
                }
            }
            mRes.propagateConstraints();
        }

        maybe_save(STAGE_POST_INIT);
        if (should_stop(STAGE_POST_INIT)) goto done;
    }

    // ================================================================
    // PHASE 2: FIELD SOLVING
    // ================================================================

    if (should_run(STAGE_POST_ORIENT)) {
        printf("Solve Orientation Field...\n");
        t1 = GetCurrentTime64();
        Optimizer::optimize_orientations(field.hierarchy);
        field.ComputeOrientationSingularities();
        t2 = GetCurrentTime64();
        printf("[TIMING] Orientation field: %lf s\n", (t2 - t1) * 1e-3);

        maybe_save(STAGE_POST_ORIENT);
        if (should_stop(STAGE_POST_ORIENT)) goto done;
    }

    if (should_run(STAGE_POST_FIELD)) {
        if (field.flag_adaptive_scale == 1) {
            printf("Estimate Slope...\n");
            t1 = GetCurrentTime64();
            field.EstimateSlope();
            t2 = GetCurrentTime64();
            printf("[TIMING] EstimateSlope: %lf s\n", (t2 - t1) * 1e-3);
        }

        printf("Solve for scale...\n");
        t1 = GetCurrentTime64();
        Optimizer::optimize_scale(field.hierarchy, field.rho, field.flag_adaptive_scale);
        field.flag_adaptive_scale = 1;
        t2 = GetCurrentTime64();
        printf("[TIMING] Scale field: %lf s\n", (t2 - t1) * 1e-3);

        printf("Solve for position field...\n");
        t1 = GetCurrentTime64();
        Optimizer::optimize_positions(field.hierarchy, field.flag_adaptive_scale);
        field.ComputePositionSingularities();
        t2 = GetCurrentTime64();
        printf("[TIMING] Position field: %lf s\n", (t2 - t1) * 1e-3);

        maybe_save(STAGE_POST_FIELD);
        if (should_stop(STAGE_POST_FIELD)) goto done;
    }

    // ================================================================
    // PHASE 3: INDEX MAP (ComputeIndexMap broken into stages)
    // ================================================================

    printf("Solve index map...\n");
    {
        unsigned long long t_stage, t_start = GetCurrentTime64();
        t_stage = t_start;

        auto& V = field.hierarchy.mV[0];
        auto& F = field.hierarchy.mF;
        auto& Q = field.hierarchy.mQ[0];
        auto& N = field.hierarchy.mN[0];
        auto& O = field.hierarchy.mO[0];
        auto& S = field.hierarchy.mS[0];

        // ---- BuildEdgeInfo + sharp edge setup ----
        if (should_run(STAGE_POST_EDGEINFO)) {
            field.BuildEdgeInfo();
            printf("[TIMING] BuildEdgeInfo: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            if (field.flag_preserve_sharp) {
                // ComputeSharpO(); // optional
            }
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                if (field.sharp_edges[i]) {
                    int e = field.face_edgeIds[i / 3][i % 3];
                    if (field.edge_diff[e][0] * field.edge_diff[e][1] != 0) {
                        Vector3d d = O.col(field.edge_values[e].y) - O.col(field.edge_values[e].x);
                        Vector3d q = Q.col(field.edge_values[e].x);
                        Vector3d n = N.col(field.edge_values[e].x);
                        Vector3d qy = n.cross(q);
                        if (abs(q.dot(d)) > qy.dot(d))
                            field.edge_diff[e][1] = 0;
                        else
                            field.edge_diff[e][0] = 0;
                    }
                }
            }
            std::map<int, std::pair<Vector3d, Vector3d>> sharp_constraints;
            std::set<int> sharpvert;
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                if (field.sharp_edges[i]) {
                    sharpvert.insert(F(i % 3, i / 3));
                    sharpvert.insert(F((i + 1) % 3, i / 3));
                }
            }
            field.allow_changes.resize(field.edge_diff.size() * 2, 1);
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                int e = field.face_edgeIds[i / 3][i % 3];
                if (sharpvert.count(field.edge_values[e].x) && sharpvert.count(field.edge_values[e].y)) {
                    if (field.sharp_edges[i] != 0) {
                        for (int k = 0; k < 2; ++k) {
                            if (field.edge_diff[e][k] == 0) {
                                field.allow_changes[e * 2 + k] = 0;
                            }
                        }
                    }
                }
            }
            printf("[TIMING] Sharp edge setup: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_EDGEINFO);
            if (should_stop(STAGE_POST_EDGEINFO)) goto done;
        }

        // ---- BuildIntegerConstraints ----
        if (should_run(STAGE_POST_CONSTRAINTS)) {
            field.BuildIntegerConstraints();
            printf("[TIMING] BuildIntegerConstraints: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            // Debug: check edge_diff before max flow
            {
                int bad = 0, max0 = 0, max1 = 0;
                for (int i = 0; i < (int)field.edge_diff.size(); ++i) {
                    int a0 = abs(field.edge_diff[i][0]), a1 = abs(field.edge_diff[i][1]);
                    if (a0 > max0) max0 = a0;
                    if (a1 > max1) max1 = a1;
                    if (a0 > 1 || a1 > 1) bad++;
                }
                printf("[DEBUG pre-flow] edges: %d, bad: %d, max|d0|=%d max|d1|=%d\n",
                       (int)field.edge_diff.size(), bad, max0, max1);
            }

            maybe_save(STAGE_POST_CONSTRAINTS);
            if (should_stop(STAGE_POST_CONSTRAINTS)) goto done;
        }

        // ---- ComputeMaxFlow ----
        if (should_run(STAGE_POST_FLOW)) {
            field.ComputeMaxFlow();
            printf("[TIMING] ComputeMaxFlow: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_FLOW);
            if (should_stop(STAGE_POST_FLOW)) goto done;
        }

        // ---- subdivide_edgeDiff (1st) ----
        if (should_run(STAGE_POST_SUBDIV1)) {
            subdivide_edgeDiff(F, V, N, Q, O, &field.hierarchy.mS[0], field.V2E, field.hierarchy.mE2E,
                               field.boundary, field.nonManifold,
                               field.edge_diff, field.edge_values, field.face_edgeOrients,
                               field.face_edgeIds, field.sharp_edges, field.singularities, 1);
            printf("[TIMING] subdivide_edgeDiff (1st): %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_SUBDIV1);
            if (should_stop(STAGE_POST_SUBDIV1)) goto done;
        }

        // ---- allow_changes setup + FixFlipHierarchy ----
        if (should_run(STAGE_PRE_FFH)) {
            field.allow_changes.clear();
            field.allow_changes.resize(field.edge_diff.size() * 2, 1);
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                if (field.sharp_edges[i] == 0) continue;
                int e = field.face_edgeIds[i / 3][i % 3];
                for (int k = 0; k < 2; ++k) {
                    if (field.edge_diff[e][k] == 0) field.allow_changes[e * 2 + k] = 0;
                }
            }

            // Legacy save-ff
            if (!save_ff_path.empty()) {
                field.SaveFFState(save_ff_path.c_str());
            }

            maybe_save(STAGE_PRE_FFH);
            if (should_stop(STAGE_PRE_FFH)) goto done;
        }

        if (should_run(STAGE_POST_FFH)) {
            field.FixFlipHierarchy();
            printf("[TIMING] FixFlipHierarchy: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_FFH);
            if (should_stop(STAGE_POST_FFH)) goto done;
        }

        // ---- subdivide_edgeDiff (2nd) + FixFlipSat ----
        if (should_run(STAGE_POST_SUBDIV2)) {
            // Check if any edges need splitting (|diff| > 1)
            int bad_count_2 = 0;
            for (int i = 0; i < (int)field.edge_diff.size(); ++i) {
                if (abs(field.edge_diff[i][0]) > 1 || abs(field.edge_diff[i][1]) > 1) bad_count_2++;
            }
            if (bad_count_2 > 0) {
                subdivide_edgeDiff(F, V, N, Q, O, &field.hierarchy.mS[0], field.V2E, field.hierarchy.mE2E,
                                   field.boundary, field.nonManifold,
                                   field.edge_diff, field.edge_values, field.face_edgeOrients,
                                   field.face_edgeIds, field.sharp_edges, field.singularities, 1);
                printf("[TIMING] subdivide_edgeDiff (2nd): %lf s (%d bad edges)\n",
                       (GetCurrentTime64() - t_stage) * 1e-3, bad_count_2);
            } else {
                printf("[TIMING] subdivide_edgeDiff (2nd): SKIPPED (0 bad edges)\n");
            }
            t_stage = GetCurrentTime64();

            field.FixFlipSat();
            printf("[TIMING] FixFlipSat: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_SUBDIV2);
            if (should_stop(STAGE_POST_SUBDIV2)) goto done;
        }

        // ---- optimize_positions_sharp + fixed + AdvancedExtractQuad + FixValence ----
        if (should_run(STAGE_POST_EXTRACT)) {
            std::set<int> sharp_vertices;
            std::map<int, std::pair<Vector3d, Vector3d>> sharp_constraints;
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                if (field.sharp_edges[i] == 1) {
                    sharp_vertices.insert(F(i % 3, i / 3));
                    sharp_vertices.insert(F((i + 1) % 3, i / 3));
                }
            }

            Optimizer::optimize_positions_sharp(field.hierarchy, field.edge_values, field.edge_diff,
                                                field.sharp_edges, sharp_vertices, sharp_constraints,
                                                field.flag_adaptive_scale);
            printf("[TIMING] optimize_positions_sharp: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            Optimizer::optimize_positions_fixed(field.hierarchy, field.edge_values, field.edge_diff,
                                                sharp_vertices, sharp_constraints, field.flag_adaptive_scale);
            printf("[TIMING] optimize_positions_fixed: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            field.AdvancedExtractQuad();
            printf("[TIMING] AdvancedExtractQuad: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            field.FixValence();
            printf("[TIMING] FixValence: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_EXTRACT);
            if (should_stop(STAGE_POST_EXTRACT)) goto done;
        }

        // ---- pre-dynamic setup + optimize_positions_dynamic ----
        if (should_run(STAGE_POST_DYNAMIC)) {
            // Build edge-to-quad mapping and diffs
            unsigned long long _pd0 = GetCurrentTime64();
            std::unordered_map<std::pair<int, int>, int, PairHash> o2e;
            o2e.reserve(field.F_compact.size() * 4);
            for (int i = 0; i < (int)field.F_compact.size(); ++i) {
                for (int j = 0; j < 4; ++j) {
                    int v1 = field.F_compact[i][j];
                    int v2 = field.F_compact[i][(j + 1) % 4];
                    o2e[std::make_pair(v1, v2)] = i * 4 + j;
                }
            }
            unsigned long long _pd1 = GetCurrentTime64();
            std::vector<std::vector<int>> v2o(V.cols());
            for (int i = 0; i < (int)field.Vset.size(); ++i) {
                for (auto v : field.Vset[i]) {
                    v2o[v].push_back(i);
                }
            }
            unsigned long long _pd2 = GetCurrentTime64();
            std::vector<Vector3d> diffs(field.F_compact.size() * 4, Vector3d(0, 0, 0));
            std::vector<int> diff_count(field.F_compact.size() * 4, 0);
            for (int i = 0; i < F.cols(); ++i) {
                for (int j = 0; j < 3; ++j) {
                    int v1 = F(j, i);
                    int v2 = F((j + 1) % 3, i);
                    if (v1 != field.edge_values[field.face_edgeIds[i][j]].x) continue;
                    if (field.edge_diff[field.face_edgeIds[i][j]].array().abs().sum() != 1) continue;
                    if (v2o[v1].size() > 1 || v2o[v2].size() > 1) continue;
                    for (auto o1 : v2o[v1]) {
                        for (auto o2 : v2o[v2]) {
                            auto key = std::make_pair(o1, o2);
                            if (o2e.count(key)) {
                                int dedge = o2e[key];
                                Vector3d q_1 = Q.col(v1);
                                Vector3d q_2 = Q.col(v2);
                                Vector3d n_1 = N.col(v1);
                                Vector3d n_2 = N.col(v2);
                                Vector3d q_1_y = n_1.cross(q_1);
                                Vector3d q_2_y = n_2.cross(q_2);
                                auto index = compat_orientation_extrinsic_index_4(q_1, n_1, q_2, n_2);
                                double s_x1 = S(0, v1), s_y1 = S(1, v1);
                                double s_x2 = S(0, v2), s_y2 = S(1, v2);
                                int rank_diff = (index.second + 4 - index.first) % 4;
                                if (rank_diff % 2 == 1) std::swap(s_x2, s_y2);
                                Vector3d qd_x = 0.5 * (rotate90_by(q_2, n_2, rank_diff) + q_1);
                                Vector3d qd_y = 0.5 * (rotate90_by(q_2_y, n_2, rank_diff) + q_1_y);
                                double scale_x = (field.flag_adaptive_scale ? 0.5 * (s_x1 + s_x2) : 1) * field.hierarchy.mScale;
                                double scale_y = (field.flag_adaptive_scale ? 0.5 * (s_y1 + s_y2) : 1) * field.hierarchy.mScale;
                                Vector2i diff = field.edge_diff[field.face_edgeIds[i][j]];
                                Vector3d C = diff[0] * scale_x * qd_x + diff[1] * scale_y * qd_y;

                                diff_count[dedge] += 1;
                                diffs[dedge] += C;
                                auto key2 = std::make_pair(o2, o1);
                                if (o2e.count(key2)) {
                                    int dedge2 = o2e[key2];
                                    diff_count[dedge2] += 1;
                                    diffs[dedge2] -= C;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < F.cols(); ++i) {
                Vector2i d1 = rshift90(field.edge_diff[field.face_edgeIds[i][0]], field.face_edgeOrients[i][0]);
                Vector2i d2 = rshift90(field.edge_diff[field.face_edgeIds[i][1]], field.face_edgeOrients[i][1]);
                if (d1[0] * d2[1] - d1[1] * d2[0] < 0) {
                    for (int j = 0; j < 3; ++j) {
                        int v1 = F(j, i);
                        int v2 = F((j + 1) % 3, i);
                        for (auto o1 : v2o[v1]) {
                            for (auto o2 : v2o[v2]) {
                                auto key = std::make_pair(o1, o2);
                                if (o2e.count(key)) {
                                    int dedge = o2e[key];
                                    diff_count[dedge] = 0;
                                    diffs[dedge] = Vector3d(0, 0, 0);
                                }
                            }
                        }
                    }
                }
            }

            unsigned long long _pd3 = GetCurrentTime64();
            for (int i = 0; i < (int)diff_count.size(); ++i) {
                if (diff_count[i] != 0) {
                    diffs[i] /= diff_count[i];
                    diff_count[i] = 1;
                }
            }

            // Sharp constraints for compact mesh
            std::map<int, std::pair<Vector3d, Vector3d>> sharp_constraints;
            std::set<int> sharp_vertices;
            for (int i = 0; i < (int)field.sharp_edges.size(); ++i) {
                if (field.sharp_edges[i] == 1) {
                    sharp_vertices.insert(F(i % 3, i / 3));
                    sharp_vertices.insert(F((i + 1) % 3, i / 3));
                }
            }
            std::vector<int> sharp_o(field.O_compact.size(), 0);
            std::map<int, std::pair<Vector3d, Vector3d>> compact_sharp_constraints;
            for (int i = 0; i < (int)field.Vset.size(); ++i) {
                int sharpv = -1;
                for (auto& p : field.Vset[i]) {
                    if (sharp_constraints.count(p)) {
                        sharpv = p;
                        sharp_o[i] = 1;
                        if (compact_sharp_constraints.count(i) == 0 ||
                            compact_sharp_constraints[i].second != Vector3d::Zero()) {
                            compact_sharp_constraints[i] = sharp_constraints[sharpv];
                            field.O_compact[i] = O.col(sharpv);
                            compact_sharp_constraints[i].first = field.O_compact[i];
                        }
                    }
                }
            }

            unsigned long long _pd4 = GetCurrentTime64();
            printf("[TIMING] pre-dynamic: o2e=%.3f v2o=%.3f diffs=%.3f sharp=%.3f s\n",
                   (_pd1-_pd0)*1e-3, (_pd2-_pd1)*1e-3, (_pd3-_pd2)*1e-3, (_pd4-_pd3)*1e-3);
            printf("[TIMING] pre-dynamic setup: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            Optimizer::optimize_positions_dynamic(F, V, N, Q, field.Vset, field.O_compact,
                                                  field.F_compact, field.V2E_compact, field.E2E_compact,
                                                  sqrt(field.surface_area / field.F_compact.size()),
                                                  diffs, diff_count, o2e, sharp_o,
                                                  compact_sharp_constraints, field.flag_adaptive_scale);
            printf("[TIMING] optimize_positions_dynamic: %lf s\n", (GetCurrentTime64() - t_stage) * 1e-3);
            t_stage = GetCurrentTime64();

            maybe_save(STAGE_POST_DYNAMIC);
        }

        printf("[TIMING] === ComputeIndexMap TOTAL: %lf s ===\n", (GetCurrentTime64() - t_start) * 1e-3);
    }

done:
    // Write output mesh if we have one and we ran far enough
    if (!output_obj.empty() && !field.F_compact.empty()) {
        printf("Writing the file...\n");
        field.OutputMesh(output_obj.c_str());
    }

    printf("finish...\n");
    return 0;
}
