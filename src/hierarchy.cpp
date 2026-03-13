#include "hierarchy.hpp"
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include "config.hpp"
#include "field-math.hpp"
#include <queue>
#include "localsat.hpp"
#include "pcg32/pcg32.h"
#ifdef WITH_TBB
#  include "tbb/tbb.h"
#  include "pss/parallel_stable_sort.h"
#endif

#ifdef WITH_CUDA
extern "C" void cuda_sort_entries(void* data, int count);
#include "init_kernels.hpp"

struct DSELevelResult { int numE; int numF; };
extern "C" void cuda_downsample_edge_graph(
    int* h_F2E, int* h_FQ, int* h_EdgeDiff, int* h_Allow,
    int nF, int nE, int max_levels,
    int** out_F2E, int** out_FQ, int** out_E2F,
    int** out_EdgeDiff, int** out_Allow,
    int** out_toUpper, int** out_toUpperOrients, int** out_upperface,
    int** out_sing_flag,
    int* out_nF, int* out_nE, int* out_num_levels);
extern "C" int cuda_find_fixable_faces(
    const int* h_E2E, const int* h_EdgeDiff, const int* h_F2E,
    const int* h_FQ, const int* h_AllowChange,
    int nF, int nE, int max_len,
    int* h_can_fix, int* out_flipped_count);
#endif

namespace qflow {

Hierarchy::Hierarchy() {
    mAdj.resize(MAX_DEPTH + 1);
    mV.resize(MAX_DEPTH + 1);
    mN.resize(MAX_DEPTH + 1);
    mA.resize(MAX_DEPTH + 1);
    mPhases.resize(MAX_DEPTH + 1);
    mToLower.resize(MAX_DEPTH);
    mToUpper.resize(MAX_DEPTH);
    rng_seed = 0;

    mCQ.reserve(MAX_DEPTH + 1);
    mCQw.reserve(MAX_DEPTH + 1);
    mCO.reserve(MAX_DEPTH + 1);
    mCOw.reserve(MAX_DEPTH + 1);
}

#undef max

void Hierarchy::Initialize(double scale, int with_scale) {
    this->with_scale = with_scale;
    unsigned long long th0 = GetCurrentTime64(), th1;
    generate_graph_coloring_deterministic(mAdj[0], mV[0].cols(), mPhases[0]);
    th1 = GetCurrentTime64(); printf("[INIT] graph_coloring level 0 (%d verts): %lf s\n", (int)mV[0].cols(), (th1-th0)*1e-3); th0=th1;

    for (int i = 0; i < MAX_DEPTH; ++i) {
#ifdef WITH_CUDA
        if (mV[i].cols() > 5000) {
            // GPU path for large levels
            int nV = mV[i].cols();
            // Convert AdjacentMatrix to CSR
            int nnz_i = 0;
            for (auto& a : mAdj[i]) nnz_i += a.size();
            std::vector<int> rowPtr(nV + 1), colInd(nnz_i);
            std::vector<double> weights(nnz_i);
            rowPtr[0] = 0;
            for (int v = 0; v < nV; ++v) {
                rowPtr[v + 1] = rowPtr[v] + mAdj[i][v].size();
                for (int k = 0; k < (int)mAdj[i][v].size(); ++k) {
                    colInd[rowPtr[v] + k] = mAdj[i][v][k].id;
                    weights[rowPtr[v] + k] = mAdj[i][v][k].weight;
                }
            }
            // Allocate output
            int vertexCount_p, nnz_p;
            int *adjRowPtr_p, *adjColInd_p;
            double *adjWeights_p;
            // Pre-allocate output geometry at max possible size
            int maxCoarse = nV; // upper bound
            mV[i + 1].resize(3, maxCoarse);
            mN[i + 1].resize(3, maxCoarse);
            mA[i + 1].resize(maxCoarse);
            mToUpper[i].resize(2, maxCoarse);
            mToLower[i].resize(nV);

            cuda_downsample_graph(
                rowPtr.data(), colInd.data(), weights.data(), nV, nnz_i,
                mV[i].data(), mN[i].data(), mA[i].data(),
                mV[i + 1].data(), mN[i + 1].data(), mA[i + 1].data(),
                mToUpper[i].data(), mToLower[i].data(),
                &adjRowPtr_p, &adjColInd_p, &adjWeights_p,
                &vertexCount_p, &nnz_p);

            // Resize to actual
            mV[i + 1].conservativeResize(3, vertexCount_p);
            mN[i + 1].conservativeResize(3, vertexCount_p);
            mA[i + 1].conservativeResize(vertexCount_p);
            mToUpper[i].conservativeResize(2, vertexCount_p);

            // Convert CSR back to AdjacentMatrix
            mAdj[i + 1].resize(vertexCount_p);
            for (int v = 0; v < vertexCount_p; ++v) {
                int start = adjRowPtr_p[v];
                int end = adjRowPtr_p[v + 1];
                mAdj[i + 1][v].resize(end - start);
                for (int k = start; k < end; ++k) {
                    mAdj[i + 1][v][k - start] = Link(adjColInd_p[k], adjWeights_p[k]);
                }
            }
            free(adjRowPtr_p); free(adjColInd_p); free(adjWeights_p);
        } else
#endif
        {
            DownsampleGraph(mAdj[i], mV[i], mN[i], mA[i], mV[i + 1], mN[i + 1], mA[i + 1], mToUpper[i],
                            mToLower[i], mAdj[i + 1]);
        }
        th1 = GetCurrentTime64(); printf("[INIT] DownsampleGraph level %d->%d (%d->%d verts): %lf s\n", i, i+1, (int)mV[i].cols(), (int)mV[i+1].cols(), (th1-th0)*1e-3); th0=th1;
        generate_graph_coloring_deterministic(mAdj[i + 1], mV[i + 1].cols(), mPhases[i + 1]);
        th1 = GetCurrentTime64(); printf("[INIT] graph_coloring level %d (%d verts): %lf s\n", i+1, (int)mV[i+1].cols(), (th1-th0)*1e-3); th0=th1;
        if (mV[i + 1].cols() == 1) {
            mAdj.resize(i + 2);
            mV.resize(i + 2);
            mN.resize(i + 2);
            mA.resize(i + 2);
            mToUpper.resize(i + 1);
            mToLower.resize(i + 1);
            break;
        }
    }
    mQ.resize(mV.size());
    mO.resize(mV.size());
    mS.resize(mV.size());
    mK.resize(mV.size());

    mCO.resize(mV.size());
    mCOw.resize(mV.size());
    mCQ.resize(mV.size());
    mCQw.resize(mV.size());

    //Set random seed
    srand(rng_seed);

    mScale = scale;
    th1 = GetCurrentTime64(); printf("[INIT] field alloc+resize: %lf s\n", (th1-th0)*1e-3); th0=th1;
    for (int i = 0; i < mV.size(); ++i) {
        mQ[i].resize(mN[i].rows(), mN[i].cols());
        mO[i].resize(mN[i].rows(), mN[i].cols());
        mS[i].resize(2, mN[i].cols());
        mK[i].resize(2, mN[i].cols());
        for (int j = 0; j < mN[i].cols(); ++j) {
            Vector3d s, t;
            coordinate_system(mN[i].col(j), s, t);
            //rand() is not thread safe!
            double angle = ((double)rand()) / RAND_MAX * 2 * M_PI;
            double x = ((double)rand()) / RAND_MAX * 2 - 1.f;
            double y = ((double)rand()) / RAND_MAX * 2 - 1.f;
            mQ[i].col(j) = s * std::cos(angle) + t * std::sin(angle);
            mO[i].col(j) = mV[i].col(j) + (s * x + t * y) * scale;
            if (with_scale) {
                mS[i].col(j) = Vector2d(1.0f, 1.0f);
                mK[i].col(j) = Vector2d(0.0, 0.0);
            }
        }
    }
    th1 = GetCurrentTime64(); printf("[INIT] field init (Q,O,S,K): %lf s\n", (th1-th0)*1e-3); th0=th1;
#ifdef WITH_CUDA
    printf("copy to device...\n");
    CopyToDevice();
    printf("copy to device finish...\n");
#endif
}

#ifdef WITH_TBB
void Hierarchy::generate_graph_coloring_deterministic(const AdjacentMatrix& adj, int size,
                                                      std::vector<std::vector<int>>& phases) {
    struct ColorData {
        uint8_t nColors;
        uint32_t nNodes[256];
        ColorData() : nColors(0) {}
    };

    const uint8_t INVALID_COLOR = 0xFF;
    phases.clear();

    /* Generate a permutation */
    std::vector<uint32_t> perm(size);
    std::vector<tbb::spin_mutex> mutex(size);
    for (uint32_t i = 0; i < size; ++i) perm[i] = i;

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0u, size, GRAIN_SIZE),
                      [&](const tbb::blocked_range<uint32_t>& range) {
                          pcg32 rng;
                          rng.advance(range.begin());
                          for (uint32_t i = range.begin(); i != range.end(); ++i) {
                              uint32_t j = i, k = rng.nextUInt(size - i) + i;
                              if (j == k) continue;
                              if (j > k) std::swap(j, k);
                              tbb::spin_mutex::scoped_lock l0(mutex[j]);
                              tbb::spin_mutex::scoped_lock l1(mutex[k]);
                              std::swap(perm[j], perm[k]);
                          }
                      });

    std::vector<uint8_t> color(size, INVALID_COLOR);
    ColorData colorData = tbb::parallel_reduce(
        tbb::blocked_range<uint32_t>(0u, size, GRAIN_SIZE), ColorData(),
        [&](const tbb::blocked_range<uint32_t>& range, ColorData colorData) -> ColorData {
            std::vector<uint32_t> neighborhood;
            bool possible_colors[256];

            for (uint32_t pidx = range.begin(); pidx != range.end(); ++pidx) {
                uint32_t i = perm[pidx];

                neighborhood.clear();
                neighborhood.push_back(i);
                //            for (const Link *link = adj[i]; link != adj[i + 1]; ++link)
                for (auto& link : adj[i]) neighborhood.push_back(link.id);
                std::sort(neighborhood.begin(), neighborhood.end());
                for (uint32_t j : neighborhood) mutex[j].lock();

                std::fill(possible_colors, possible_colors + colorData.nColors, true);

                //            for (const Link *link = adj[i]; link != adj[i + 1]; ++link) {
                for (auto& link : adj[i]) {
                    uint8_t c = color[link.id];
                    if (c != INVALID_COLOR) {
                        while (c >= colorData.nColors) {
                            possible_colors[colorData.nColors] = true;
                            colorData.nNodes[colorData.nColors] = 0;
                            colorData.nColors++;
                        }
                        possible_colors[c] = false;
                    }
                }

                uint8_t chosen_color = INVALID_COLOR;
                for (uint8_t j = 0; j < colorData.nColors; ++j) {
                    if (possible_colors[j]) {
                        chosen_color = j;
                        break;
                    }
                }
                if (chosen_color == INVALID_COLOR) {
                    if (colorData.nColors == INVALID_COLOR - 1)
                        throw std::runtime_error(
                            "Ran out of colors during graph coloring! "
                            "The input mesh is very likely corrupt.");
                    colorData.nNodes[colorData.nColors] = 1;
                    color[i] = colorData.nColors++;
                } else {
                    colorData.nNodes[chosen_color]++;
                    color[i] = chosen_color;
                }

                for (uint32_t j : neighborhood) mutex[j].unlock();
            }
            return colorData;
        },
        [](ColorData c1, ColorData c2) -> ColorData {
            ColorData result;
            result.nColors = std::max(c1.nColors, c2.nColors);
            memset(result.nNodes, 0, sizeof(uint32_t) * result.nColors);
            for (uint8_t i = 0; i < c1.nColors; ++i) result.nNodes[i] += c1.nNodes[i];
            for (uint8_t i = 0; i < c2.nColors; ++i) result.nNodes[i] += c2.nNodes[i];
            return result;
        });

    phases.resize(colorData.nColors);
    for (int i = 0; i < colorData.nColors; ++i) phases[i].reserve(colorData.nNodes[i]);

    for (uint32_t i = 0; i < size; ++i) phases[color[i]].push_back(i);
}
#else
void Hierarchy::generate_graph_coloring_deterministic(const AdjacentMatrix& adj, int size,
                                                      std::vector<std::vector<int>>& phases) {
    phases.clear();

    std::vector<uint32_t> perm(size);
    for (uint32_t i = 0; i < size; ++i) perm[i] = i;
    pcg32 rng;
    rng.shuffle(perm.begin(), perm.end());

    std::vector<int> color(size, -1);
    std::vector<uint8_t> possible_colors;
    std::vector<int> size_per_color;
    int ncolors = 0;

    for (uint32_t i = 0; i < size; ++i) {
        uint32_t ip = perm[i];

        std::fill(possible_colors.begin(), possible_colors.end(), 1);

        for (auto& link : adj[ip]) {
            int c = color[link.id];
            if (c >= 0) possible_colors[c] = 0;
        }

        int chosen_color = -1;
        for (uint32_t j = 0; j < possible_colors.size(); ++j) {
            if (possible_colors[j]) {
                chosen_color = j;
                break;
            }
        }

        if (chosen_color < 0) {
            chosen_color = ncolors++;
            possible_colors.resize(ncolors);
            size_per_color.push_back(0);
        }

        color[ip] = chosen_color;
        size_per_color[chosen_color]++;
    }
    phases.resize(ncolors);
    for (int i = 0; i < ncolors; ++i) phases[i].reserve(size_per_color[i]);
    for (uint32_t i = 0; i < size; ++i) phases[color[i]].push_back(i);
}
#endif

void Hierarchy::DownsampleGraph(const AdjacentMatrix adj, const MatrixXd& V, const MatrixXd& N,
                                const VectorXd& A, MatrixXd& V_p, MatrixXd& N_p, VectorXd& A_p,
                                MatrixXi& to_upper, VectorXi& to_lower, AdjacentMatrix& adj_p) {
    struct Entry {
        int i, j;
        double order;
        inline Entry() { i = j = -1; };
        inline Entry(int i, int j, double order) : i(i), j(j), order(order) {}
        inline bool operator<(const Entry& e) const { return order > e.order; }
        inline bool operator==(const Entry& e) const { return order == e.order; }
    };

    int nLinks = 0;
    for (auto& adj_i : adj) nLinks += adj_i.size();
    std::vector<Entry> entries(nLinks);
    std::vector<int> bases(adj.size());
    for (int i = 1; i < bases.size(); ++i) {
        bases[i] = bases[i - 1] + adj[i - 1].size();
    }

#ifdef WITH_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < V.cols(); ++i) {
        int base = bases[i];
        auto& ad = adj[i];
        auto entry_it = entries.begin() + base;
        for (auto it = ad.begin(); it != ad.end(); ++it, ++entry_it) {
            int k = it->id;
            double dp = N.col(i).dot(N.col(k));
            double ratio = A[i] > A[k] ? (A[i] / A[k]) : (A[k] / A[i]);
            *entry_it = Entry(i, k, dp * ratio);
        }
    }

    unsigned long long tds0 = GetCurrentTime64();
#ifdef WITH_CUDA
    // GPU sort via Thrust
    static_assert(sizeof(Entry) == sizeof(int)*2 + sizeof(double), "Entry layout mismatch for GPU sort");
    cuda_sort_entries(entries.data(), entries.size());
#elif defined(WITH_TBB)
    pss::parallel_stable_sort(entries.begin(), entries.end(), std::less<Entry>());
#else
    std::sort(entries.begin(), entries.end(), std::less<Entry>());
#endif
    printf("[INIT]   sort %d entries: %lf s\n", (int)entries.size(), (GetCurrentTime64()-tds0)*1e-3);

    std::vector<bool> mergeFlag(V.cols(), false);

    int nCollapsed = 0;
    for (int i = 0; i < nLinks; ++i) {
        const Entry& e = entries[i];
        if (mergeFlag[e.i] || mergeFlag[e.j]) continue;
        mergeFlag[e.i] = mergeFlag[e.j] = true;
        entries[nCollapsed++] = entries[i];
    }
    int vertexCount = V.cols() - nCollapsed;

    // Allocate memory for coarsened graph
    V_p.resize(3, vertexCount);
    N_p.resize(3, vertexCount);
    A_p.resize(vertexCount);
    to_upper.resize(2, vertexCount);
    to_lower.resize(V.cols());

#ifdef WITH_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nCollapsed; ++i) {
        const Entry& e = entries[i];
        const double area1 = A[e.i], area2 = A[e.j], surfaceArea = area1 + area2;
        if (surfaceArea > RCPOVERFLOW)
            V_p.col(i) = (V.col(e.i) * area1 + V.col(e.j) * area2) / surfaceArea;
        else
            V_p.col(i) = (V.col(e.i) + V.col(e.j)) * 0.5f;
        Vector3d normal = N.col(e.i) * area1 + N.col(e.j) * area2;
        double norm = normal.norm();
        N_p.col(i) = norm > RCPOVERFLOW ? Vector3d(normal / norm) : Vector3d::UnitX();
        A_p[i] = surfaceArea;
        to_upper.col(i) << e.i, e.j;
        to_lower[e.i] = i;
        to_lower[e.j] = i;
    }

    int offset = nCollapsed;

    for (int i = 0; i < V.cols(); ++i) {
        if (!mergeFlag[i]) {
            int idx = offset++;
            V_p.col(idx) = V.col(i);
            N_p.col(idx) = N.col(i);
            A_p[idx] = A[i];
            to_upper.col(idx) << i, -1;
            to_lower[i] = idx;
        }
    }

    adj_p.resize(V_p.cols());
    std::vector<int> capacity(V_p.cols());
    std::vector<std::vector<Link>> scratches(V_p.cols());
#ifdef WITH_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < V_p.cols(); ++i) {
        int t = 0;
        for (int j = 0; j < 2; ++j) {
            int upper = to_upper(j, i);
            if (upper == -1) continue;
            t += adj[upper].size();
        }
        scratches[i].reserve(t);
        adj_p[i].reserve(t);
    }
#ifdef WITH_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < V_p.cols(); ++i) {
        auto& scratch = scratches[i];
        for (int j = 0; j < 2; ++j) {
            int upper = to_upper(j, i);
            if (upper == -1) continue;
            auto& ad = adj[upper];
            for (auto& link : ad) scratch.push_back(Link(to_lower[link.id], link.weight));
        }
        std::sort(scratch.begin(), scratch.end());
        int id = -1;
        auto& ad = adj_p[i];
        for (auto& link : scratch) {
            if (link.id != i) {
                if (id != link.id) {
                    ad.push_back(link);
                    id = link.id;
                } else {
                    ad.back().weight += link.weight;
                }
            }
        }
    }
}

void Hierarchy::SaveToFile(FILE* fp) {
    Save(fp, mScale);
    Save(fp, mF);
    Save(fp, mE2E);
    Save(fp, mAdj);
    Save(fp, mV);
    Save(fp, mN);
    Save(fp, mA);
    Save(fp, mToLower);
    Save(fp, mToUpper);
    Save(fp, mQ);
    Save(fp, mO);
    Save(fp, mS);
    Save(fp, mK);
    Save(fp, this->mPhases);
}

void Hierarchy::LoadFromFile(FILE* fp) {
    Read(fp, mScale);
    Read(fp, mF);
    Read(fp, mE2E);
    Read(fp, mAdj);
    Read(fp, mV);
    Read(fp, mN);
    Read(fp, mA);
    Read(fp, mToLower);
    Read(fp, mToUpper);
    Read(fp, mQ);
    Read(fp, mO);
    Read(fp, mS);
    Read(fp, mK);
    Read(fp, this->mPhases);
}

void Hierarchy::UpdateGraphValue(std::vector<Vector3i>& FQ, std::vector<Vector3i>& F2E,
                                 std::vector<Vector2i>& edge_diff) {
    FQ = std::move(mFQ[0]);
    F2E = std::move(mF2E[0]);
    edge_diff = std::move(mEdgeDiff[0]);
}

void Hierarchy::DownsampleEdgeGraph(std::vector<Vector3i>& FQ, std::vector<Vector3i>& F2E,
                                    std::vector<Vector2i>& edge_diff,
                                    std::vector<int>& allow_changes, int level) {
#ifdef WITH_CUDA
    {
        int nF = (int)F2E.size();
        int nE = (int)edge_diff.size();
        int max_levels = (level == -1) ? 100 : level;

        // Convert Vector3i/Vector2i to flat int arrays
        // F2E[nF][3], FQ[nF][3] -> flat[nF*3] row-major
        // EdgeDiff[nE][2], Allow[nE*2] -> flat[nE*2]
        std::vector<int> flat_F2E(nF * 3), flat_FQ(nF * 3);
        std::vector<int> flat_EdgeDiff(nE * 2), flat_Allow(nE * 2);
        for (int i = 0; i < nF; ++i) {
            for (int j = 0; j < 3; ++j) {
                flat_F2E[i * 3 + j] = F2E[i][j];
                flat_FQ[i * 3 + j] = FQ[i][j];
            }
        }
        for (int i = 0; i < nE; ++i) {
            flat_EdgeDiff[i * 2] = edge_diff[i][0];
            flat_EdgeDiff[i * 2 + 1] = edge_diff[i][1];
        }
        flat_Allow = allow_changes;  // already flat

        // Output arrays (pointers to per-level malloc'd data)
        const int MAX_GPU_LEVELS = 100;
        int* out_F2E[MAX_GPU_LEVELS] = {};
        int* out_FQ[MAX_GPU_LEVELS] = {};
        int* out_E2F[MAX_GPU_LEVELS] = {};
        int* out_EdgeDiff[MAX_GPU_LEVELS] = {};
        int* out_Allow[MAX_GPU_LEVELS] = {};
        int* out_toUpper[MAX_GPU_LEVELS] = {};
        int* out_toUpperOrients[MAX_GPU_LEVELS] = {};
        int* out_upperface[MAX_GPU_LEVELS] = {};
        int* out_sing_flag[MAX_GPU_LEVELS] = {};
        int out_nF[MAX_GPU_LEVELS] = {};
        int out_nE[MAX_GPU_LEVELS] = {};
        int num_levels = 0;

        unsigned long long _tg0 = GetCurrentTime64();
        cuda_downsample_edge_graph(
            flat_F2E.data(), flat_FQ.data(), flat_EdgeDiff.data(), flat_Allow.data(),
            nF, nE, max_levels,
            out_F2E, out_FQ, out_E2F, out_EdgeDiff, out_Allow,
            out_toUpper, out_toUpperOrients, out_upperface, out_sing_flag,
            out_nF, out_nE, &num_levels);
        unsigned long long _tg1 = GetCurrentTime64();
        printf("[DSE-GPU] %d levels in %.3f s\n", num_levels, (_tg1-_tg0)*1e-3);

        // Convert back to Hierarchy member vectors
        mFQ.resize(num_levels);
        mF2E.resize(num_levels);
        mE2F.resize(num_levels);
        mEdgeDiff.resize(num_levels);
        mAllowChanges.resize(num_levels);
        mSing.resize(num_levels);
        mToUpperEdges.resize(num_levels - 1);
        mToUpperOrients.resize(num_levels - 1);
        mToUpperFaces.clear();

        for (int l = 0; l < num_levels; ++l) {
            int lnF = out_nF[l];
            int lnE = out_nE[l];

            // F2E, FQ: flat[nF*3] -> Vector3i[nF]
            mF2E[l].resize(lnF);
            mFQ[l].resize(lnF);
            for (int i = 0; i < lnF; ++i) {
                mF2E[l][i] = Vector3i(out_F2E[l][i*3], out_F2E[l][i*3+1], out_F2E[l][i*3+2]);
                mFQ[l][i] = Vector3i(out_FQ[l][i*3], out_FQ[l][i*3+1], out_FQ[l][i*3+2]);
            }

            // E2F, EdgeDiff: flat[nE*2] -> Vector2i[nE]
            mE2F[l].resize(lnE);
            mEdgeDiff[l].resize(lnE);
            for (int i = 0; i < lnE; ++i) {
                mE2F[l][i] = Vector2i(out_E2F[l][i*2], out_E2F[l][i*2+1]);
                mEdgeDiff[l][i] = Vector2i(out_EdgeDiff[l][i*2], out_EdgeDiff[l][i*2+1]);
            }

            // AllowChanges: flat[nE*2] -> vector<int>
            mAllowChanges[l].assign(out_Allow[l], out_Allow[l] + lnE * 2);

            // Singularities: scan sing_flag
            mSing[l].clear();
            for (int i = 0; i < lnF; ++i) {
                if (out_sing_flag[l][i]) mSing[l].push_back(i);
            }

            free(out_F2E[l]); free(out_FQ[l]); free(out_E2F[l]);
            free(out_EdgeDiff[l]); free(out_Allow[l]); free(out_sing_flag[l]);
        }

        for (int l = 0; l < num_levels - 1; ++l) {
            int lnE = out_nE[l];
            int lnF = out_nF[l];

            // toUpper, toUpperOrients
            mToUpperEdges[l].assign(out_toUpper[l], out_toUpper[l] + lnE);
            mToUpperOrients[l].assign(out_toUpperOrients[l], out_toUpperOrients[l] + lnE);

            // upperface
            std::vector<int> uf(out_upperface[l], out_upperface[l] + lnF);
            mToUpperFaces.push_back(std::move(uf));

            free(out_toUpper[l]); free(out_toUpperOrients[l]); free(out_upperface[l]);
        }

        return;
    }
#endif
    std::vector<Vector2i> E2F(edge_diff.size(), Vector2i(-1, -1));
    for (int i = 0; i < F2E.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int e = F2E[i][j];
            if (E2F[e][0] == -1)
                E2F[e][0] = i;
            else
                E2F[e][1] = i;
        }
    }
    int levels = (level == -1) ? 100 : level;
    mFQ.resize(levels);
    mF2E.resize(levels);
    mE2F.resize(levels);
    mEdgeDiff.resize(levels);
    mAllowChanges.resize(levels);
    mSing.resize(levels);
    mToUpperEdges.resize(levels - 1);
    mToUpperOrients.resize(levels - 1);
    for (int i = 0; i < FQ.size(); ++i) {
        Vector2i diff(0, 0);
        for (int j = 0; j < 3; ++j) {
            diff += rshift90(edge_diff[F2E[i][j]], FQ[i][j]);
        }
        if (diff != Vector2i::Zero()) {
            mSing[0].push_back(i);
        }
    }
    mAllowChanges[0] = allow_changes;
    mFQ[0] = std::move(FQ);
    mF2E[0] = std::move(F2E);
    mE2F[0] = std::move(E2F);
    mEdgeDiff[0] = std::move(edge_diff);
    for (int l = 0; l < levels - 1; ++l) {
        unsigned long long _dl0 = GetCurrentTime64();
        auto& FQ = mFQ[l];
        auto& E2F = mE2F[l];
        auto& F2E = mF2E[l];
        auto& Allow = mAllowChanges[l];
        auto& EdgeDiff = mEdgeDiff[l];
        auto& Sing = mSing[l];
        std::vector<int> fixed_faces(F2E.size(), 0);
        for (auto& s : Sing) {
            fixed_faces[s] = 1;
        }

        auto& toUpper = mToUpperEdges[l];
        auto& toUpperOrients = mToUpperOrients[l];
        toUpper.resize(E2F.size(), -1);
        toUpperOrients.resize(E2F.size(), 0);

        auto& nFQ = mFQ[l + 1];
        auto& nE2F = mE2F[l + 1];
        auto& nF2E = mF2E[l + 1];
        auto& nAllow = mAllowChanges[l + 1];
        auto& nEdgeDiff = mEdgeDiff[l + 1];
        auto& nSing = mSing[l + 1];

        for (int i = 0; i < E2F.size(); ++i) {
            if (EdgeDiff[i] != Vector2i::Zero()) continue;
            if ((E2F[i][0] >= 0 && fixed_faces[E2F[i][0]]) ||
                (E2F[i][1] >= 0 && fixed_faces[E2F[i][1]])) {
                continue;
            }
            for (int j = 0; j < 2; ++j) {
                int f = E2F[i][j];
                if (f < 0)
                    continue;
                for (int k = 0; k < 3; ++k) {
                    int neighbor_e = F2E[f][k];
                    for (int m = 0; m < 2; ++m) {
                        int neighbor_f = E2F[neighbor_e][m];
                        if (neighbor_f < 0)
                            continue;
                        if (fixed_faces[neighbor_f] == 0) fixed_faces[neighbor_f] = 1;
                    }
                }
            }
            if (E2F[i][0] >= 0)
                fixed_faces[E2F[i][0]] = 2;
            if (E2F[i][1] >= 0)
                fixed_faces[E2F[i][1]] = 2;
            toUpper[i] = -2;
        }
        for (int i = 0; i < E2F.size(); ++i) {
            if (toUpper[i] == -2) continue;
            if ((E2F[i][0] < 0 || fixed_faces[E2F[i][0]] == 2) && (E2F[i][1] < 0 || fixed_faces[E2F[i][1]] == 2)) {
                toUpper[i] = -3;
                continue;
            }
        }
        int numE = 0;
        for (int i = 0; i < toUpper.size(); ++i) {
            if (toUpper[i] == -1) {
                if ((E2F[i][0] < 0 || fixed_faces[E2F[i][0]] < 2) && (E2F[i][1] < 0 || fixed_faces[E2F[i][1]] < 2)) {
                    nE2F.push_back(E2F[i]);
                    toUpperOrients[i] = 0;
                    toUpper[i] = numE++;
                    continue;
                }
                int f0 = (E2F[i][1] < 0 || fixed_faces[E2F[i][0]] < 2) ? E2F[i][0] : E2F[i][1];
                int e = i;
                int f = f0;
                std::vector<std::pair<int, int>> paths;
                paths.push_back(std::make_pair(i, 0));
                while (true) {
                    if (E2F[e][0] == f)
                        f = E2F[e][1];
                    else if (E2F[e][1] == f)
                        f = E2F[e][0];
                    if (f < 0 || fixed_faces[f] < 2) {
                        for (int j = 0; j < paths.size(); ++j) {
                            auto& p = paths[j];
                            toUpper[p.first] = numE;
                            int orient = p.second;
                            if (j > 0) orient = (orient + toUpperOrients[paths[j - 1].first]) % 4;
                            toUpperOrients[p.first] = orient;
                        }
                        nE2F.push_back(Vector2i(f0, f));
                        numE += 1;
                        break;
                    }
                    int ind0 = -1, ind1 = -1;
                    int e0 = e;
                    for (int j = 0; j < 3; ++j) {
                        if (F2E[f][j] == e) {
                            ind0 = j;
                            break;
                        }
                    }
                    for (int j = 0; j < 3; ++j) {
                        int e1 = F2E[f][j];
                        if (e1 != e && toUpper[e1] != -2) {
                            e = e1;
                            ind1 = j;
                            break;
                        }
                    }

                    if (ind1 != -1) {
                        paths.push_back(std::make_pair(e, (FQ[f][ind1] - FQ[f][ind0] + 6) % 4));
                    } else {
                        if (EdgeDiff[e] != Vector2i::Zero()) {
                            printf("Unsatisfied !!!...\n");
                            printf("%d %d %d: %d %d\n", F2E[f][0], F2E[f][1], F2E[f][2], e0, e);
                            //exit(0);
                            throw std::runtime_error("Failed: qflow::Hierarchy::DownsampleEdgeGraph.");
                        }
                        for (auto& p : paths) {
                            toUpper[p.first] = numE;
                            toUpperOrients[p.first] = 0;
                        }
                        numE += 1;
                        nE2F.push_back(Vector2i(f0, f0));
                        break;
                    }
                }
            }
        }
        nEdgeDiff.resize(numE);
        nAllow.resize(numE * 2, 1);
        for (int i = 0; i < toUpper.size(); ++i) {
            if (toUpper[i] >= 0 && toUpperOrients[i] == 0) {
                nEdgeDiff[toUpper[i]] = EdgeDiff[i];
            }
            if (toUpper[i] >= 0) {
                int dimension = toUpperOrients[i] % 2;
                if (Allow[i * 2 + dimension] == 0)
                    nAllow[toUpper[i] * 2] = 0;
                else if (Allow[i * 2 + dimension] == 2)
                    nAllow[toUpper[i] * 2] = 2;
                if (Allow[i * 2 + 1 - dimension] == 0)
                    nAllow[toUpper[i] * 2 + 1] = 0;
                else if (Allow[i * 2 + 1 - dimension] == 2)
                    nAllow[toUpper[i] * 2 + 1] = 2;
            }
        }
        std::vector<int> upperface(F2E.size(), -1);

        for (int i = 0; i < F2E.size(); ++i) {
            Vector3i eid;
            for (int j = 0; j < 3; ++j) {
                eid[j] = toUpper[F2E[i][j]];
            }
            if (eid[0] >= 0 && eid[1] >= 0 && eid[2] >= 0) {
                Vector3i eid_orient;
                for (int j = 0; j < 3; ++j) {
                    eid_orient[j] = (FQ[i][j] + 4 - toUpperOrients[F2E[i][j]]) % 4;
                }
                upperface[i] = nF2E.size();
                nF2E.push_back(eid);
                nFQ.push_back(eid_orient);
            }
        }
        for (int i = 0; i < nE2F.size(); ++i) {
            for (int j = 0; j < 2; ++j) {
                if (nE2F[i][j] >= 0)
                    nE2F[i][j] = upperface[nE2F[i][j]];
            }
        }

        for (auto& s : Sing) {
            if (upperface[s] >= 0) nSing.push_back(upperface[s]);
        }
        mToUpperFaces.push_back(std::move(upperface));
        printf("[DSE] level %d: faces=%zu->%zu edges=%zu->%zu time=%.3f s\n",
               l, F2E.size(), nF2E.size(), EdgeDiff.size(), nEdgeDiff.size(),
               (GetCurrentTime64()-_dl0)*1e-3);

        if (nEdgeDiff.size() == EdgeDiff.size()) {
            levels = l + 1;
            break;
        }
    }

    mFQ.resize(levels);
    mF2E.resize(levels);
    mAllowChanges.resize(levels);
    mE2F.resize(levels);
    mEdgeDiff.resize(levels);
    mSing.resize(levels);
    mToUpperEdges.resize(levels - 1);
    mToUpperOrients.resize(levels - 1);
}

int Hierarchy::FixFlipSat(int depth, int threshold) {
    if (system("which minisat > /dev/null 2>&1")) {
        printf("minisat not found, \"-sat\" will not be used!\n");
        return 0;
    }
    if (system("which timeout > /dev/null 2>&1")) {
        printf("timeout not found, \"-sat\" will not be used!\n");
        return 0;
    }

    auto& F2E = mF2E[depth];
    auto& E2F = mE2F[depth];
    auto& FQ = mFQ[depth];
    auto& EdgeDiff = mEdgeDiff[depth];
    auto& AllowChanges = mAllowChanges[depth];

    // build E2E
    std::vector<int> E2E(F2E.size() * 3, -1);
    for (int i = 0; i < E2F.size(); ++i) {
        int f1 = E2F[i][0];
        int f2 = E2F[i][1];
        int t1 = 0;
        int t2 = 2;
        if (f1 != -1) while (F2E[f1][t1] != i) t1 += 1;
        if (f2 != -1) while (F2E[f2][t2] != i) t2 -= 1;
        t1 += f1 * 3;
        t2 += f2 * 3;
        if (f1 != -1) E2E[t1] = (f2 == -1) ? -1 : t2;
        if (f2 != -1) E2E[t2] = (f1 == -1) ? -1 : t1;
    }

    auto IntegerArea = [&](int f) {
        Vector2i diff1 = rshift90(EdgeDiff[F2E[f][0]], FQ[f][0]);
        Vector2i diff2 = rshift90(EdgeDiff[F2E[f][1]], FQ[f][1]);
        return diff1[0] * diff2[1] - diff1[1] * diff2[0];
    };

    std::deque<std::pair<int, int>> Q;
    std::vector<bool> mark_dedges(F2E.size() * 3, false);
    for (int f = 0; f < F2E.size(); ++f) {
        if (IntegerArea(f) < 0) {
            for (int j = 0; j < 3; ++j) {
                if (mark_dedges[f * 3 + j]) continue;
                Q.push_back(std::make_pair(f * 3 + j, 0));
                mark_dedges[f * 3 + j] = true;
            }
        }
    }

    int mark_count = 0;
    while (!Q.empty()) {
        int e0 = Q.front().first;
        int depth = Q.front().second;
        Q.pop_front();
        mark_count++;

        int e = e0, e1;
        do {
            e1 = E2E[e];
            if (e1 == -1) break;
            int length = EdgeDiff[F2E[e1 / 3][e1 % 3]].array().abs().sum();
            if (length == 0 && !mark_dedges[e1]) {
                mark_dedges[e1] = true;
                Q.push_front(std::make_pair(e1, depth));
            }
            e = (e1 / 3) * 3 + (e1 + 1) % 3;
            mark_dedges[e] = true;
        } while (e != e0);
        if (e1 == -1) {
            do {
                e1 = E2E[e];
                if (e1 == -1) break;
                int length = EdgeDiff[F2E[e1 / 3][e1 % 3]].array().abs().sum();
                if (length == 0 && !mark_dedges[e1]) {
                    mark_dedges[e1] = true;
                    Q.push_front(std::make_pair(e1, depth));
                }
                e = (e1 / 3) * 3 + (e1 + 2) % 3;
                mark_dedges[e] = true;
            } while (e != e0);
        }

        do {
            e1 = E2E[e];
            if (e1 == -1) break;
            int length = EdgeDiff[F2E[e1 / 3][e1 % 3]].array().abs().sum();
            if (length > 0 && depth + length <= threshold && !mark_dedges[e1]) {
                mark_dedges[e1] = true;
                Q.push_back(std::make_pair(e1, depth + length));
            }
            e = e1 / 3 * 3 + (e1 + 1) % 3;
            mark_dedges[e] = true;
        } while (e != e0);
        if (e1 == -1) {
            do {
                e1 = E2E[e];
                if (e1 == -1) break;
                int length = EdgeDiff[F2E[e1 / 3][e1 % 3]].array().abs().sum();
                if (length > 0 && depth + length <= threshold && !mark_dedges[e1]) {
                    mark_dedges[e1] = true;
                    Q.push_back(std::make_pair(e1, depth + length));
                }
                e = e1 / 3 * 3 + (e1 + 2) % 3;
                mark_dedges[e] = true;
            } while (e != e0);
        }
    }
    lprintf("[FlipH] Depth %2d: marked = %d\n", depth, mark_count);

    std::vector<bool> flexible(EdgeDiff.size(), false);
    for (int i = 0; i < F2E.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int dedge = i * 3 + j;
            int edgeid = F2E[i][j];
            if (mark_dedges[dedge]) {
                flexible[edgeid] = true;
            }
        }
    }
    for (int i = 0; i < flexible.size(); ++i) {
        if (E2F[i][0] == E2F[i][1]) flexible[i] = false;
        if (AllowChanges[i] == 0) flexible[i] = false;
    }

    // Reindexing and solve
    int num_group = 0;
    std::vector<int> groups(EdgeDiff.size(), -1);
    std::vector<int> indices(EdgeDiff.size(), -1);
    for (int i = 0; i < EdgeDiff.size(); ++i) {
        if (groups[i] == -1 && flexible[i]) {
            // group it
            std::queue<int> q;
            q.push(i);
            groups[i] = num_group;
            while (!q.empty()) {
                int e = q.front();
                q.pop();
                int f[] = {E2F[e][0], E2F[e][1]};
                for (int j = 0; j < 2; ++j) {
                    if (f[j] == -1) continue;
                    for (int k = 0; k < 3; ++k) {
                        int e1 = F2E[f[j]][k];
                        if (flexible[e1] && groups[e1] == -1) {
                            groups[e1] = num_group;
                            q.push(e1);
                        }
                    }
                }
            }
            num_group += 1;
        }
    }

    std::vector<int> num_edges(num_group);
    std::vector<int> num_flips(num_group);
    std::vector<std::vector<int>> values(num_group);
    std::vector<std::vector<Vector3i>> variable_eq(num_group);
    std::vector<std::vector<Vector3i>> constant_eq(num_group);
    std::vector<std::vector<Vector4i>> variable_ge(num_group);
    std::vector<std::vector<Vector2i>> constant_ge(num_group);
    for (int i = 0; i < groups.size(); ++i) {
        if (groups[i] != -1) {
            indices[i] = num_edges[groups[i]]++;
            values[groups[i]].push_back(EdgeDiff[i][0]);
            values[groups[i]].push_back(EdgeDiff[i][1]);
        }
    }
    std::vector<int> num_edges_flexible = num_edges;
    std::map<std::pair<int, int>, int> fixed_variables;
    for (int i = 0; i < F2E.size(); ++i) {
        Vector2i var[3];
        Vector2i cst[3];
        int gind = 0;
        while (gind < 3 && groups[F2E[i][gind]] == -1) gind += 1;
        if (gind == 3) continue;
        int group = groups[F2E[i][gind]];
        int ind[3] = {-1, -1, -1};
        for (int j = 0; j < 3; ++j) {
            int g = groups[F2E[i][j]];
            if (g != group) {
                if (g == -1) {
                    auto key = std::make_pair(F2E[i][j], group);
                    auto it = fixed_variables.find(key);
                    if (it == fixed_variables.end()) {
                        ind[j] = num_edges[group];
                        values[group].push_back(EdgeDiff[F2E[i][j]][0]);
                        values[group].push_back(EdgeDiff[F2E[i][j]][1]);
                        fixed_variables[key] = num_edges[group]++;
                    } else {
                        ind[j] = it->second;
                    }
                }
            } else {
                ind[j] = indices[F2E[i][j]];
            }
        }
        for (int j = 0; j < 3; ++j) assert(ind[j] != -1);
        for (int j = 0; j < 3; ++j) {
            var[j] = rshift90(Vector2i(ind[j] * 2 + 1, ind[j] * 2 + 2), FQ[i][j]);
            cst[j] = var[j].array().sign();
            var[j] = var[j].array().abs() - 1;
        }

        num_flips[group] += IntegerArea(i) < 0;
        variable_eq[group].push_back(Vector3i(var[0][0], var[1][0], var[2][0]));
        constant_eq[group].push_back(Vector3i(cst[0][0], cst[1][0], cst[2][0]));
        variable_eq[group].push_back(Vector3i(var[0][1], var[1][1], var[2][1]));
        constant_eq[group].push_back(Vector3i(cst[0][1], cst[1][1], cst[2][1]));

        variable_ge[group].push_back(Vector4i(var[0][0], var[1][1], var[0][1], var[1][0]));
        constant_ge[group].push_back(Vector2i(cst[0][0] * cst[1][1], cst[0][1] * cst[1][0]));
    }
    int flip_before = 0, flip_after = 0;
    for (int i = 0; i < F2E.size(); ++i) {
        int area = IntegerArea(i);
        if (area < 0) flip_before++;
    }

    for (int i = 0; i < num_group; ++i) {
        std::vector<bool> flexible(values[i].size(), true);
        for (int j = num_edges_flexible[i] * 2; j < flexible.size(); ++j) {
            flexible[j] = false;
        }
        SolveSatProblem(values[i].size(), values[i], flexible, variable_eq[i], constant_eq[i],
                        variable_ge[i], constant_ge[i]);
    }

    for (int i = 0; i < EdgeDiff.size(); ++i) {
        int group = groups[i];
        if (group == -1) continue;
        EdgeDiff[i][0] = values[group][2 * indices[i] + 0];
        EdgeDiff[i][1] = values[group][2 * indices[i] + 1];
    }
    for (int i = 0; i < F2E.size(); ++i) {
        Vector2i diff(0, 0);
        for (int j = 0; j < 3; ++j) {
            diff += rshift90(EdgeDiff[F2E[i][j]], FQ[i][j]);
        }
        assert(diff == Vector2i::Zero());

        int area = IntegerArea(i);
        if (area < 0) flip_after++;
    }

    lprintf("[FlipH] FlipArea, Before: %d After %d\n", flip_before, flip_after);
    return flip_after;
}

void Hierarchy::PushDownwardFlip(int depth) {
    auto& EdgeDiff = mEdgeDiff[depth];
    auto& nEdgeDiff = mEdgeDiff[depth - 1];
    auto& toUpper = mToUpperEdges[depth - 1];
    auto& toUpperOrients = mToUpperOrients[depth - 1];
    auto& toUpperFaces = mToUpperFaces[depth - 1];
    for (int i = 0; i < toUpper.size(); ++i) {
        if (toUpper[i] >= 0) {
            int orient = (4 - toUpperOrients[i]) % 4;
            nEdgeDiff[i] = rshift90(EdgeDiff[toUpper[i]], orient);
        } else {
            nEdgeDiff[i] = Vector2i(0, 0);
        }
    }
    auto& nF2E = mF2E[depth - 1];
    auto& nFQ = mFQ[depth - 1];
    for (int i = 0; i < nF2E.size(); ++i) {
        Vector2i diff(0, 0);
        for (int j = 0; j < 3; ++j) {
            diff += rshift90(nEdgeDiff[nF2E[i][j]], nFQ[i][j]);
        }
        if (diff != Vector2i::Zero()) {
            printf("Fail!!!!!!! %d\n", i);
            for (int j = 0; j < 3; ++j) {
                Vector2i d = rshift90(nEdgeDiff[nF2E[i][j]], nFQ[i][j]);
                printf("<%d %d %d>\n", nF2E[i][j], nFQ[i][j], toUpperOrients[nF2E[i][j]]);
                printf("%d %d\n", d[0], d[1]);
                printf("%d -> %d\n", nF2E[i][j], toUpper[nF2E[i][j]]);
            }
            printf("%d -> %d\n", i, toUpperFaces[i]);
            exit(1);
        }
    }
}

// FixFlipAtLevel: run CheckShrink at a specific hierarchy level.
// Returns true if any fix was made (EdgeDiff modified).
bool Hierarchy::FixFlipAtLevel(int l) {
    auto& F2E = mF2E[l];
    auto& E2F = mE2F[l];
    auto& FQ = mFQ[l];
    auto& EdgeDiff = mEdgeDiff[l];
    auto& AllowChange = mAllowChanges[l];

    // build E2E
    std::vector<int> E2E(F2E.size() * 3, -1);
    for (int i = 0; i < (int)E2F.size(); ++i) {
        int v1 = E2F[i][0];
        int v2 = E2F[i][1];
        int t1 = 0;
        int t2 = 2;
        if (v1 != -1)
            while (F2E[v1][t1] != i) t1 += 1;
        if (v2 != -1)
            while (F2E[v2][t2] != i) t2 -= 1;
        t1 += v1 * 3;
        t2 += v2 * 3;
        if (v1 != -1)
            E2E[t1] = (v2 == -1) ? -1 : t2;
        if (v2 != -1)
            E2E[t2] = (v1 == -1) ? -1 : t1;
    }

    auto Area = [&](int f) {
        Vector2i diff1 = rshift90(EdgeDiff[F2E[f][0]], FQ[f][0]);
        Vector2i diff2 = rshift90(EdgeDiff[F2E[f][1]], FQ[f][1]);
        return diff1[0] * diff2[1] - diff1[1] * diff2[0];
    };
    std::vector<int> valences(F2E.size() * 3, -10000);
    auto CheckShrink = [&](int deid, int allowed_edge_length) {
        if (deid == -1) {
            return false;
        }
        std::vector<int> corresponding_faces;
        std::vector<int> corresponding_edges;
        std::vector<Vector2i> corresponding_diff;
        int deid0 = deid;
        while (deid != -1) {
            deid = deid / 3 * 3 + (deid + 2) % 3;
            if (E2E[deid] == -1)
                break;
            deid = E2E[deid];
            if (deid == deid0)
                break;
        }
        Vector2i diff = EdgeDiff[F2E[deid / 3][deid % 3]];
        do {
            corresponding_diff.push_back(diff);
            corresponding_edges.push_back(deid);
            corresponding_faces.push_back(deid / 3);
            deid = E2E[deid];
            if (deid == -1) {
                return false;
            }
            diff = -rshift90(diff, FQ[deid / 3][deid % 3]);
            deid = deid / 3 * 3 + (deid + 1) % 3;
            diff = rshift90(diff, (4 - FQ[deid / 3][deid % 3]) % 4);
        } while (deid != corresponding_edges.front());
        if (deid != -1 && diff != corresponding_diff.front()) {
            return false;
        }
        std::unordered_map<int, Vector2i> new_values;
        for (int i = 0; i < (int)corresponding_diff.size(); ++i) {
            int deid = corresponding_edges[i];
            int eid = F2E[deid / 3][deid % 3];
            new_values[eid] = EdgeDiff[eid];
        }
        for (int i = 0; i < (int)corresponding_diff.size(); ++i) {
            int deid = corresponding_edges[i];
            int eid = F2E[deid / 3][deid % 3];
            for (int j = 0; j < 2; ++j) {
                if (corresponding_diff[i][j] != 0 && AllowChange[eid * 2 + j] == 0) return false;
            }
            auto& res = new_values[eid];
            res -= corresponding_diff[i];
            int edge_thres = allowed_edge_length;
            if (abs(res[0]) > edge_thres || abs(res[1]) > edge_thres) {
                return false;
            }
            if ((abs(res[0]) > 1 && abs(res[1]) != 0) || (abs(res[1]) > 1 && abs(res[0]) != 0))
                return false;
        }
        int prev_area = 0, current_area = 0;
        for (int f = 0; f < (int)corresponding_faces.size(); ++f) {
            int area = Area(corresponding_faces[f]);
            if (area < 0) prev_area += 1;
        }
        for (auto& p : new_values) {
            std::swap(EdgeDiff[p.first], p.second);
        }
        for (int f = 0; f < (int)corresponding_faces.size(); ++f) {
            int area = Area(corresponding_faces[f]);
            if (area < 0) {
                current_area += 1;
            }
        }
        if (current_area < prev_area) {
            return true;
        }
        for (auto& p : new_values) {
            std::swap(EdgeDiff[p.first], p.second);
        }
        return false;
    };

    // Count flipped faces
    std::queue<int> flipped;
    int flip_count = 0;
    for (int i = 0; i < (int)F2E.size(); ++i) {
        if (Area(i) < 0) {
            flipped.push(i);
            flip_count++;
        }
    }

    if (flip_count == 0) return false;

    // Try to fix: find first successful CheckShrink
    bool update = false;
    int max_len = 1;
    int checks = 0;
    while (!update && max_len <= 2) {
        while (!flipped.empty()) {
            int f = flipped.front();
            if (Area(f) >= 0) {
                flipped.pop();
                continue;
            }
            for (int i = 0; i < 3; ++i) {
                checks++;
                if (CheckShrink(f * 3 + i, max_len) || CheckShrink(E2E[f * 3 + i], max_len)) {
                    update = true;
                    break;
                }
            }
            flipped.pop();
        }
        max_len += 1;
    }
    printf("[FF]   level=%d faces=%zu flipped=%d checks=%d update=%d\n",
           l, F2E.size(), flip_count, checks, update);
    return update;
}

// Original recursive FixFlip — proven correct, 6.2s baseline
void Hierarchy::FixFlip() {
    int l = mF2E.size() - 1;
    auto& F2E = mF2E[l];
    auto& E2F = mE2F[l];
    auto& FQ = mFQ[l];
    auto& EdgeDiff = mEdgeDiff[l];
    auto& AllowChange = mAllowChanges[l];
    printf("[FF] FixFlip level=%d faces=%zu edges=%zu\n", l, F2E.size(), E2F.size());
    unsigned long long _ff0 = GetCurrentTime64();

    // build E2E
    std::vector<int> E2E(F2E.size() * 3, -1);
    for (int i = 0; i < (int)E2F.size(); ++i) {
        int v1 = E2F[i][0];
        int v2 = E2F[i][1];
        int t1 = 0;
        int t2 = 2;
        if (v1 != -1)
            while (F2E[v1][t1] != i) t1 += 1;
        if (v2 != -1)
            while (F2E[v2][t2] != i) t2 -= 1;
        t1 += v1 * 3;
        t2 += v2 * 3;
        if (v1 != -1)
            E2E[t1] = (v2 == -1) ? -1 : t2;
        if (v2 != -1)
            E2E[t2] = (v1 == -1) ? -1 : t1;
    }

    auto Area = [&](int f) {
        Vector2i diff1 = rshift90(EdgeDiff[F2E[f][0]], FQ[f][0]);
        Vector2i diff2 = rshift90(EdgeDiff[F2E[f][1]], FQ[f][1]);
        return diff1[0] * diff2[1] - diff1[1] * diff2[0];
    };
    std::vector<int> valences(F2E.size() * 3, -10000);
    auto CheckShrink = [&](int deid, int allowed_edge_length) {
        if (deid == -1) {
            return false;
        }
        std::vector<int> corresponding_faces;
        std::vector<int> corresponding_edges;
        std::vector<Vector2i> corresponding_diff;
        int deid0 = deid;
        while (deid != -1) {
            deid = deid / 3 * 3 + (deid + 2) % 3;
            if (E2E[deid] == -1)
                break;
            deid = E2E[deid];
            if (deid == deid0)
                break;
        }
        Vector2i diff = EdgeDiff[F2E[deid / 3][deid % 3]];
        do {
            corresponding_diff.push_back(diff);
            corresponding_edges.push_back(deid);
            corresponding_faces.push_back(deid / 3);
            deid = E2E[deid];
            if (deid == -1) {
                return false;
            }
            diff = -rshift90(diff, FQ[deid / 3][deid % 3]);
            deid = deid / 3 * 3 + (deid + 1) % 3;
            diff = rshift90(diff, (4 - FQ[deid / 3][deid % 3]) % 4);
        } while (deid != corresponding_edges.front());
        if (deid != -1 && diff != corresponding_diff.front()) {
            return false;
        }
        std::unordered_map<int, Vector2i> new_values;
        for (int i = 0; i < (int)corresponding_diff.size(); ++i) {
            int deid = corresponding_edges[i];
            int eid = F2E[deid / 3][deid % 3];
            new_values[eid] = EdgeDiff[eid];
        }
        for (int i = 0; i < (int)corresponding_diff.size(); ++i) {
            int deid = corresponding_edges[i];
            int eid = F2E[deid / 3][deid % 3];
            for (int j = 0; j < 2; ++j) {
                if (corresponding_diff[i][j] != 0 && AllowChange[eid * 2 + j] == 0) return false;
            }
            auto& res = new_values[eid];
            res -= corresponding_diff[i];
            int edge_thres = allowed_edge_length;
            if (abs(res[0]) > edge_thres || abs(res[1]) > edge_thres) {
                return false;
            }
            if ((abs(res[0]) > 1 && abs(res[1]) != 0) || (abs(res[1]) > 1 && abs(res[0]) != 0))
                return false;
        }
        int prev_area = 0, current_area = 0;
        for (int f = 0; f < (int)corresponding_faces.size(); ++f) {
            int area = Area(corresponding_faces[f]);
            if (area < 0) prev_area += 1;
        }
        for (auto& p : new_values) {
            std::swap(EdgeDiff[p.first], p.second);
        }
        for (int f = 0; f < (int)corresponding_faces.size(); ++f) {
            int area = Area(corresponding_faces[f]);
            if (area < 0) {
                current_area += 1;
            }
        }
        if (current_area < prev_area) {
            return true;
        }
        for (auto& p : new_values) {
            std::swap(EdgeDiff[p.first], p.second);
        }
        return false;
    };

    unsigned long long _ff1 = GetCurrentTime64();

    bool update = false;
    int max_len = 1;
    int checks = 0;
    int flip_count = 0;

    // =========================================================================
    // FixFlip scan strategies (selectable via -ff flag)
    //
    // KEY INSIGHT (discovered during optimization):
    //   The CPU scan doesn't stop after the FIRST successful CheckShrink.
    //   The inner while(!flipped.empty()) loop CONTINUES after update=true,
    //   because only the OUTER while checks !update. This means the scan
    //   applies CheckShrink to ALL flipped faces in one pass, producing
    //   ~100K zero-diff edges that the recursive DSE needs for deep coarsening.
    //
    // STRATEGY 0: "cpu" — Original CPU sequential scan [DEFAULT]
    //   Iterates all flipped faces in index order, tries CheckShrink on each.
    //   Each successful CheckShrink permanently modifies EdgeDiff.
    //   Result: ~100K+ zero-diff edges, 8-9 level recursive hierarchies.
    //   Timing: ~0.5s first recursion, ~0.15s second, decreasing.
    //   Total FixFlip: ~3-4s. Best quality, proven correct.
    //
    // STRATEGY 1: "gpu-prefilter" — GPU identifies fixable faces, prioritized queue
    //   GPU parallel scan (0.02s) finds ALL fixable faces. These go to the front
    //   of the queue, remaining flipped faces follow. CPU processes the queue
    //   identically to strategy 0.
    //   Result: Same quality as CPU. Saves ~50ms on first recursion by trying
    //   fixable faces first (fewer wasted CheckShrink calls on non-fixable faces).
    //   Overhead: ~20ms GPU per recursion (malloc/copy/free), ~5ms flatten.
    //   Net: ~10% faster scan, but GPU overhead eats most of the savings.
    //   Total FixFlip: ~3-4s (roughly same as CPU, sometimes slightly better).
    //
    // STRATEGY 2: "gpu-only" — GPU identifies fixable faces, CPU applies ONLY those
    //   GPU parallel scan finds fixable faces. CPU applies CheckShrink ONLY to
    //   those faces, skipping all others.
    //   Result: Fewer fixes per pass (~30K vs ~100K zero-diff edges), requiring
    //   more recursion levels (20+ vs 10). Each recursion is faster but more are
    //   needed. GPU overhead accumulates.
    //   Total FixFlip: ~6-9s (WORSE — more recursions + GPU overhead per recursion).
    //   NOT RECOMMENDED unless GPU CheckShrink apply is implemented.
    // =========================================================================

#ifdef WITH_CUDA
    if (fixflip_strategy == 1 || fixflip_strategy == 2) {
        // --- GPU-assisted strategies ---
        int nF_local = (int)F2E.size();
        int nE_local = (int)E2F.size();

        // Flatten data for GPU upload
        std::vector<int> flat_E2E(E2E);
        std::vector<int> flat_EdgeDiff(nE_local * 2);
        std::vector<int> flat_F2E(nF_local * 3);
        std::vector<int> flat_FQ(nF_local * 3);
        std::vector<int> flat_Allow(nE_local * 2);
        for (int i = 0; i < nE_local; ++i) {
            flat_EdgeDiff[i * 2] = EdgeDiff[i][0];
            flat_EdgeDiff[i * 2 + 1] = EdgeDiff[i][1];
        }
        for (int i = 0; i < nF_local; ++i) {
            for (int j = 0; j < 3; ++j) {
                flat_F2E[i * 3 + j] = F2E[i][j];
                flat_FQ[i * 3 + j] = FQ[i][j];
            }
        }
        flat_Allow = AllowChange;

        // GPU scan: find all fixable faces (read-only, 0.02s)
        std::vector<int> can_fix(nF_local);
        int n_fixable = cuda_find_fixable_faces(
            flat_E2E.data(), flat_EdgeDiff.data(), flat_F2E.data(),
            flat_FQ.data(), flat_Allow.data(),
            nF_local, nE_local, 1, can_fix.data(), &flip_count);

        if (fixflip_strategy == 1) {
            // STRATEGY 1: GPU-prefilter — prioritized queue, full scan
            // Put GPU-fixable faces first, then remaining flipped faces
            std::queue<int> flipped;
            for (int f = 0; f < nF_local; ++f)
                if (can_fix[f]) flipped.push(f);
            for (int f = 0; f < nF_local; ++f)
                if (!can_fix[f] && Area(f) < 0) flipped.push(f);

            while (!update && max_len <= 2) {
                while (!flipped.empty()) {
                    int f = flipped.front();
                    if (Area(f) >= 0) { flipped.pop(); continue; }
                    for (int i = 0; i < 3; ++i) {
                        checks++;
                        if (CheckShrink(f * 3 + i, max_len) || CheckShrink(E2E[f * 3 + i], max_len)) {
                            update = true;
                            break;
                        }
                    }
                    flipped.pop();
                }
                max_len += 1;
            }
        } else {
            // STRATEGY 2: GPU-only — apply ONLY to GPU-identified fixable faces
            while (!update && max_len <= 2) {
                for (int f = 0; f < nF_local; ++f) {
                    if (!can_fix[f]) continue;
                    if (Area(f) >= 0) continue;
                    for (int i = 0; i < 3; ++i) {
                        checks++;
                        if (CheckShrink(f * 3 + i, max_len) || CheckShrink(E2E[f * 3 + i], max_len)) {
                            update = true;
                            break;
                        }
                    }
                }
                max_len += 1;
            }
        }
    } else
#endif
    {
        // STRATEGY 0: CPU sequential scan (default, proven correct)
        std::queue<int> flipped;
        for (int i = 0; i < (int)F2E.size(); ++i) {
            if (Area(i) < 0) {
                flipped.push(i);
                flip_count++;
            }
        }
        while (!update && max_len <= 2) {
            while (!flipped.empty()) {
                int f = flipped.front();
                if (Area(f) >= 0) {
                    flipped.pop();
                    continue;
                }
                for (int i = 0; i < 3; ++i) {
                    checks++;
                    if (CheckShrink(f * 3 + i, max_len) || CheckShrink(E2E[f * 3 + i], max_len)) {
                        update = true;
                        break;
                    }
                }
                flipped.pop();
            }
            max_len += 1;
        }
    }

    unsigned long long _ff2 = GetCurrentTime64();
    printf("[FF]   scan: %.3f s, checks=%d, flipped=%d, update=%d, strategy=%d\n",
           (_ff2-_ff1)*1e-3, checks, flip_count, update, fixflip_strategy);

    if (update) {
        Hierarchy flip_hierarchy;
        flip_hierarchy.fixflip_strategy = fixflip_strategy;  // propagate to recursive calls
        unsigned long long _ff3 = GetCurrentTime64();
        flip_hierarchy.DownsampleEdgeGraph(mFQ.back(), mF2E.back(), mEdgeDiff.back(),
                                           mAllowChanges.back(), -1);
        unsigned long long _ff4 = GetCurrentTime64();
        flip_hierarchy.FixFlip();
        unsigned long long _ff5 = GetCurrentTime64();
        flip_hierarchy.UpdateGraphValue(mFQ.back(), mF2E.back(), mEdgeDiff.back());
        unsigned long long _ff6 = GetCurrentTime64();
        printf("[FF]   recursive: downsample=%.3f fixflip=%.3f update=%.3f s\n",
               (_ff4-_ff3)*1e-3, (_ff5-_ff4)*1e-3, (_ff6-_ff5)*1e-3);
    }
    PropagateEdge();
}

void Hierarchy::PropagateEdge() {
    for (int level = mToUpperEdges.size(); level > 0; --level) {
        auto& EdgeDiff = mEdgeDiff[level];
        auto& nEdgeDiff = mEdgeDiff[level - 1];
        auto& FQ = mFQ[level];
        auto& nFQ = mFQ[level - 1];
        auto& F2E = mF2E[level - 1];
        auto& toUpper = mToUpperEdges[level - 1];
        auto& toUpperFace = mToUpperFaces[level - 1];
        auto& toUpperOrients = mToUpperOrients[level - 1];
        for (int i = 0; i < (int)toUpper.size(); ++i) {
            if (toUpper[i] >= 0) {
                int orient = (4 - toUpperOrients[i]) % 4;
                nEdgeDiff[i] = rshift90(EdgeDiff[toUpper[i]], orient);
            } else {
                nEdgeDiff[i] = Vector2i(0, 0);
            }
        }
        for (int i = 0; i < (int)toUpperFace.size(); ++i) {
            if (toUpperFace[i] == -1) continue;
            Vector3i eid_orient = FQ[toUpperFace[i]];
            for (int j = 0; j < 3; ++j) {
                nFQ[i][j] = (eid_orient[j] + toUpperOrients[F2E[i][j]]) % 4;
            }
        }
    }
}

void Hierarchy::clearConstraints() {
    int levels = mV.size();
    if (levels == 0) return;
    for (int i = 0; i < levels; ++i) {
        int size = mV[i].cols();
        mCQ[i].resize(3, size);
        mCO[i].resize(3, size);
        mCQw[i].resize(size);
        mCOw[i].resize(size);
        mCQw[i].setZero();
        mCOw[i].setZero();
    }
}

void Hierarchy::propagateConstraints() {
    int levels = mV.size();
    if (levels == 0) return;

    for (int l = 0; l < levels - 1; ++l) {
        auto& N = mN[l];
        auto& N_next = mN[l + 1];
        auto& V = mV[l];
        auto& V_next = mV[l + 1];
        auto& CQ = mCQ[l];
        auto& CQ_next = mCQ[l + 1];
        auto& CQw = mCQw[l];
        auto& CQw_next = mCQw[l + 1];
        auto& CO = mCO[l];
        auto& CO_next = mCO[l + 1];
        auto& COw = mCOw[l];
        auto& COw_next = mCOw[l + 1];
        auto& toUpper = mToUpper[l];
        MatrixXd& S = mS[l];

        for (uint32_t i = 0; i != mV[l + 1].cols(); ++i) {
            Vector2i upper = toUpper.col(i);
            Vector3d cq = Vector3d::Zero(), co = Vector3d::Zero();
            float cqw = 0.0f, cow = 0.0f;

            bool has_cq0 = CQw[upper[0]] != 0;
            bool has_cq1 = upper[1] != -1 && CQw[upper[1]] != 0;
            bool has_co0 = COw[upper[0]] != 0;
            bool has_co1 = upper[1] != -1 && COw[upper[1]] != 0;

            if (has_cq0 && !has_cq1) {
                cq = CQ.col(upper[0]);
                cqw = CQw[upper[0]];
            } else if (has_cq1 && !has_cq0) {
                cq = CQ.col(upper[1]);
                cqw = CQw[upper[1]];
            } else if (has_cq1 && has_cq0) {
                Vector3d q_i = CQ.col(upper[0]);
                Vector3d n_i = CQ.col(upper[0]);
                Vector3d q_j = CQ.col(upper[1]);
                Vector3d n_j = CQ.col(upper[1]);
                auto result = compat_orientation_extrinsic_4(q_i, n_i, q_j, n_j);
                cq = result.first * CQw[upper[0]] + result.second * CQw[upper[1]];
                cqw = (CQw[upper[0]] + CQw[upper[1]]);
            }
            if (cq != Vector3d::Zero()) {
                Vector3d n = N_next.col(i);
                cq -= n.dot(cq) * n;
                if (cq.squaredNorm() > RCPOVERFLOW) cq.normalize();
            }

            if (has_co0 && !has_co1) {
                co = CO.col(upper[0]);
                cow = COw[upper[0]];
            } else if (has_co1 && !has_co0) {
                co = CO.col(upper[1]);
                cow = COw[upper[1]];
            } else if (has_co1 && has_co0) {
                double scale_x = mScale;
                double scale_y = mScale;
                if (with_scale) {
                  // FIXME
                    // scale_x *= S(0, i);
                    // scale_y *= S(1, i);
                }
                double inv_scale_x = 1.0f / scale_x;
                double inv_scale_y = 1.0f / scale_y;

                double scale_x_1 = mScale;
                double scale_y_1 = mScale;
                if (with_scale) {
                  // FIXME
                    // scale_x_1 *= S(0, j);
                    // scale_y_1 *= S(1, j);
                }
                double inv_scale_x_1 = 1.0f / scale_x_1;
                double inv_scale_y_1 = 1.0f / scale_y_1;
                auto result = compat_position_extrinsic_4(
                    V.col(upper[0]), N.col(upper[0]), CQ.col(upper[0]), CO.col(upper[0]),
                    V.col(upper[1]), N.col(upper[1]), CQ.col(upper[1]), CO.col(upper[1]), scale_x,
                    scale_y, inv_scale_x, inv_scale_y, scale_x_1, scale_y_1, inv_scale_x_1,
                    inv_scale_y_1);
                cow = COw[upper[0]] + COw[upper[1]];
                co = (result.first * COw[upper[0]] + result.second * COw[upper[1]]) / cow;
            }
            if (co != Vector3d::Zero()) {
                Vector3d n = N_next.col(i), v = V_next.col(i);
                co -= n.dot(cq - v) * n;
            }
#if 0
                        cqw *= 0.5f;
                        cow *= 0.5f;
#else
            if (cqw > 0) cqw = 1;
            if (cow > 0) cow = 1;
#endif

            CQw_next[i] = cqw;
            COw_next[i] = cow;
            CQ_next.col(i) = cq;
            CO_next.col(i) = co;
        }
    }
}

#ifdef WITH_CUDA
} // namespace qflow
#include <cuda_runtime.h>
namespace qflow {

void Hierarchy::CopyToDevice() {
    unsigned long long t_copy_start = GetCurrentTime64();
    if (cudaAdj.empty()) {
        cudaAdj.resize(mAdj.size());
        cudaAdjOffset.resize(mAdj.size());
        for (int i = 0; i < mAdj.size(); ++i) {
            std::vector<int> offset(mAdj[i].size() + 1, 0);
            for (int j = 0; j < mAdj[i].size(); ++j) {
                offset[j + 1] = offset[j] + mAdj[i][j].size();
            }
            cudaMalloc(&cudaAdjOffset[i], sizeof(int) * (mAdj[i].size() + 1));
            cudaMemcpy(cudaAdjOffset[i], offset.data(), sizeof(int) * (mAdj[i].size() + 1),
                       cudaMemcpyHostToDevice);
            //            cudaAdjOffset[i] = (int*)malloc(sizeof(int) * (mAdj[i].size() + 1));
            //            memcpy(cudaAdjOffset[i], offset.data(), sizeof(int) * (mAdj[i].size() +
            //            1));

            cudaMalloc(&cudaAdj[i], sizeof(Link) * offset.back());
            //            cudaAdj[i] = (Link*)malloc(sizeof(Link) * offset.back());
            std::vector<Link> plainlink(offset.back());
            for (int j = 0; j < mAdj[i].size(); ++j) {
                memcpy(plainlink.data() + offset[j], mAdj[i][j].data(),
                       mAdj[i][j].size() * sizeof(Link));
            }
            cudaMemcpy(cudaAdj[i], plainlink.data(), plainlink.size() * sizeof(Link),
                       cudaMemcpyHostToDevice);
        }
    }

    if (cudaN.empty()) {
        cudaN.resize(mN.size());
        for (int i = 0; i < mN.size(); ++i) {
            cudaMalloc(&cudaN[i], sizeof(glm::dvec3) * mN[i].cols());
            //            cudaN[i] = (glm::dvec3*)malloc(sizeof(glm::dvec3) * mN[i].cols());
        }
    }
    for (int i = 0; i < mN.size(); ++i) {
        cudaMemcpy(cudaN[i], mN[i].data(), sizeof(glm::dvec3) * mN[i].cols(),
                   cudaMemcpyHostToDevice);
        //        memcpy(cudaN[i], mN[i].data(), sizeof(glm::dvec3) * mN[i].cols());
    }

    if (cudaV.empty()) {
        cudaV.resize(mV.size());
        for (int i = 0; i < mV.size(); ++i) {
            cudaMalloc(&cudaV[i], sizeof(glm::dvec3) * mV[i].cols());
            //            cudaV[i] = (glm::dvec3*)malloc(sizeof(glm::dvec3) * mV[i].cols());
        }
    }
    for (int i = 0; i < mV.size(); ++i) {
        cudaMemcpy(cudaV[i], mV[i].data(), sizeof(glm::dvec3) * mV[i].cols(),
                   cudaMemcpyHostToDevice);
        //        memcpy(cudaV[i], mV[i].data(), sizeof(glm::dvec3) * mV[i].cols());
    }

    if (cudaQ.empty()) {
        cudaQ.resize(mQ.size());
        for (int i = 0; i < mQ.size(); ++i) {
            cudaMalloc(&cudaQ[i], sizeof(glm::dvec3) * mQ[i].cols());
            //            cudaQ[i] = (glm::dvec3*)malloc(sizeof(glm::dvec3) * mQ[i].cols());
        }
    }
    for (int i = 0; i < mQ.size(); ++i) {
        cudaMemcpy(cudaQ[i], mQ[i].data(), sizeof(glm::dvec3) * mQ[i].cols(),
                   cudaMemcpyHostToDevice);
        //        memcpy(cudaQ[i], mQ[i].data(), sizeof(glm::dvec3) * mQ[i].cols());
    }
    if (cudaO.empty()) {
        cudaO.resize(mO.size());
        for (int i = 0; i < mO.size(); ++i) {
            cudaMalloc(&cudaO[i], sizeof(glm::dvec3) * mO[i].cols());
            //            cudaO[i] = (glm::dvec3*)malloc(sizeof(glm::dvec3) * mO[i].cols());
        }
    }
    for (int i = 0; i < mO.size(); ++i) {
        cudaMemcpy(cudaO[i], mO[i].data(), sizeof(glm::dvec3) * mO[i].cols(),
                   cudaMemcpyHostToDevice);
        //        memcpy(cudaO[i], mO[i].data(), sizeof(glm::dvec3) * mO[i].cols());
    }
    if (cudaPhases.empty()) {
        cudaPhases.resize(mPhases.size());
        for (int i = 0; i < mPhases.size(); ++i) {
            cudaPhases[i].resize(mPhases[i].size());
            for (int j = 0; j < mPhases[i].size(); ++j) {
                cudaMalloc(&cudaPhases[i][j], sizeof(int) * mPhases[i][j].size());
                //                cudaPhases[i][j] = (int*)malloc(sizeof(int) *
                //                mPhases[i][j].size());
            }
        }
    }
    for (int i = 0; i < mPhases.size(); ++i) {
        for (int j = 0; j < mPhases[i].size(); ++j) {
            cudaMemcpy(cudaPhases[i][j], mPhases[i][j].data(), sizeof(int) * mPhases[i][j].size(),
                       cudaMemcpyHostToDevice);
            //            memcpy(cudaPhases[i][j], mPhases[i][j].data(), sizeof(int) *
            //            mPhases[i][j].size());
        }
    }
    if (cudaToUpper.empty()) {
        cudaToUpper.resize(mToUpper.size());
        for (int i = 0; i < mToUpper.size(); ++i) {
            cudaMalloc(&cudaToUpper[i], mToUpper[i].cols() * sizeof(glm::ivec2));
            //            cudaToUpper[i] = (glm::ivec2*)malloc(mToUpper[i].cols() *
            //            sizeof(glm::ivec2));
        }
    }
    for (int i = 0; i < mToUpper.size(); ++i) {
        cudaMemcpy(cudaToUpper[i], mToUpper[i].data(), sizeof(glm::ivec2) * mToUpper[i].cols(),
                   cudaMemcpyHostToDevice);
        //        memcpy(cudaToUpper[i], mToUpper[i].data(), sizeof(glm::ivec2) *
        //        mToUpper[i].cols());
    }
    cudaDeviceSynchronize();
    printf("[TIMING] CopyToDevice actual: %lf s\n", (GetCurrentTime64() - t_copy_start) * 1e-3);
}

void Hierarchy::CopyToHost() {}

#endif

} // namespace qflow
