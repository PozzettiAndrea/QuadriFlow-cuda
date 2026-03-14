#ifndef HIERARCHY_H_
#define HIERARCHY_H_

#ifdef WITH_CUDA
#    include <glm/glm.hpp>
#endif

#include <map>
#include <vector>
#include "adjacent-matrix.hpp"
#include "config.hpp"
#include "serialize.hpp"
#define RCPOVERFLOW 2.93873587705571876e-39f

using namespace Eigen;

namespace qflow {

class Hierarchy {
   public:
    Hierarchy();
    void Initialize(double scale, int with_scale = 0);
    void DownsampleGraph(const AdjacentMatrix adj, const MatrixXd& V, const MatrixXd& N,
                         const VectorXd& A, MatrixXd& V_p, MatrixXd& N_p, VectorXd& A_p,
                         MatrixXi& to_upper, VectorXi& to_lower, AdjacentMatrix& adj_p);
    void generate_graph_coloring_deterministic(const AdjacentMatrix& adj, int size,
                                               std::vector<std::vector<int>>& phases);
    void FixFlip();
    bool FixFlipAtLevel(int l);  // returns true if any fix was made
    int FixFlipSat(int depth, int threshold = 0);
    void PushDownwardFlip(int depth);
    void PropagateEdge();
    void DownsampleEdgeGraph(std::vector<Vector3i>& FQ, std::vector<Vector3i>& F2E,
                             std::vector<Vector2i>& edge_diff,
                             std::vector<int>& allow_changes, int level);
    void UpdateGraphValue(std::vector<Vector3i>& FQ, std::vector<Vector3i>& F2E,
                          std::vector<Vector2i>& edge_diff);

    enum { MAX_DEPTH = 25 };

    // ============================================================
    // Runtime strategy flags (selectable via CLI flags)
    // ============================================================

    // -ff: FixFlip scan strategy
    //   0 = "cpu"            — Original CPU sequential scan (baseline, correct)
    //   1 = "gpu-prefilter"  — GPU identifies fixable faces, CPU applies in priority order
    //   2 = "gpu-only"       — GPU identifies fixable faces, CPU applies ONLY those
    int fixflip_strategy = 0;

    // -subdiv: subdivide strategy (initial mesh refinement in Initialize)
    //   0 = "cpu"            — Original CPU priority-queue edge splitting
    //   1 = "cuda"           — GPU multi-pass parallel edge splitting
    int subdiv_strategy = 0;  // default: cpu (GPU produces valid but different mesh)

    // -dse: DownsampleEdgeGraph strategy
    //   0 = "cpu"            — Original CPU sequential greedy
    //   1 = "cuda"           — GPU index-priority peeling (matches CPU output)
    int dse_strategy = 1;  // default: cuda when available

    // -flow: max-flow solver strategy
    //   0 = "boykov"         — Boost Boykov-Kolmogorov (CPU, pointer-based graph, slow build)
    //   1 = "cuda"           — GPU push-relabel + local refinement (fast but ~1.5x more bad edges)
    //   2 = "lemon"          — LEMON Preflow (CPU)
    //   3 = "edkarp"         — GPU Edmonds-Karp (GPU BFS + GPU augment, good quality)
    //   4 = "dinic"          — GPU Dinic's (GPU fwd+bwd BFS + GPU multi-augment, fastest)
#ifdef WITH_CUDA
    int flow_strategy = 4;  // default: GPU Dinic's
#else
    int flow_strategy = 0;  // default: boykov when no CUDA
#endif

    void SaveToFile(FILE* fp);
    void LoadFromFile(FILE* fp);

    void clearConstraints();
    void propagateConstraints();

    double mScale;
    int rng_seed;

    MatrixXi mF;    // mF(i, j) i \in [0, 3) ith index in face j
    VectorXi mE2E;  // inverse edge
    std::vector<AdjacentMatrix> mAdj;
    std::vector<MatrixXd> mV;
    std::vector<MatrixXd> mN;
    std::vector<VectorXd> mA;
    std::vector<std::vector<std::vector<int>>> mPhases;
    // parameters
    std::vector<MatrixXd> mQ;
    std::vector<MatrixXd> mO;
    std::vector<VectorXi> mToLower;
    std::vector<MatrixXi> mToUpper;  // mToUpper[h](i, j) \in V; i \in [0, 2); j \in V
    std::vector<MatrixXd> mS;
    std::vector<MatrixXd> mK;

    // constraints
    std::vector<MatrixXd> mCQ;
    std::vector<MatrixXd> mCO;
    std::vector<VectorXd> mCQw;
    std::vector<VectorXd> mCOw;

    int with_scale;

    // upper: fine to coarse
    std::vector<std::vector<int>> mToUpperFaces;  // face correspondance
    std::vector<std::vector<int>> mSing;
    std::vector<std::vector<int>> mToUpperEdges; // edge correspondance
    std::vector<std::vector<int>> mToUpperOrients; // rotation of edges from fine to coarse
    std::vector<std::vector<Vector3i>> mFQ; // face_edgeOrients
    std::vector<std::vector<Vector3i>> mF2E; // face_edgeIds
    std::vector<std::vector<Vector2i>> mE2F; // undirect edges to face ID
    std::vector<std::vector<int> > mAllowChanges;
    std::vector<std::vector<Vector2i>> mEdgeDiff; // face_edgeDiff

#ifdef WITH_CUDA
    std::vector<Link*> cudaAdj;
    std::vector<int*> cudaAdjOffset;
    std::vector<glm::dvec3*> cudaN;
    std::vector<glm::dvec3*> cudaV;
    std::vector<glm::dvec3*> cudaQ;
    std::vector<glm::dvec3*> cudaO;
    std::vector<std::vector<int*>> cudaPhases;
    std::vector<glm::ivec2*> cudaToUpper;
    void CopyToDevice();
    void CopyToHost();
#endif
};

} // namespace qflow

#endif
