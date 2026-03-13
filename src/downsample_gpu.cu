// ============================================================
// GPU-accelerated DownsampleEdgeGraph for QuadriFlow
//
// Parallel face merging and edge graph coarsening on GPU.
// Key operations:
//   1. Candidate edge selection (EdgeDiff == 0)
//   2. Greedy independent set via graph coloring
//   3. Face collapse and upper-level graph construction
//   4. Edge diff propagation through collapsed paths
//   5. New E2E/F2E/E2F construction via prefix scan
//
// Called from Hierarchy::DownsampleEdgeGraph() when strategy=1 (cuda).
// ============================================================

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <cstdio>

// rshift90: rotate a 2D integer vector by amount * 90 degrees
__device__ __host__ inline void rshift90(int& x, int& y, int amount) {
    if (amount & 1) { int tmp = x; x = -y; y = tmp; }
    if (amount >= 2) { x = -x; y = -y; }
}

// ============================================================
// Kernel 1: Build E2F from F2E
// Each face writes its ID into the 2 slots of its 3 edges.
// E2F_flat[e*2+0] = first face, E2F_flat[e*2+1] = second face
// ============================================================
__global__ void k_build_E2F(const int* F2E, int nF, int* E2F, int nE) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    for (int j = 0; j < 3; ++j) {
        int e = F2E[f * 3 + j];
        // Try to claim slot 0
        int old = atomicCAS(&E2F[e * 2], -1, f);
        if (old != -1) {
            // Slot 0 taken, write to slot 1
            E2F[e * 2 + 1] = f;
        }
    }
}

// ============================================================
// Kernel 2: Find singularities
// A face is singular if sum of rshift90(EdgeDiff[F2E[f][j]], FQ[f][j]) != (0,0)
// Output: sing_flag[f] = 1 if singular, 0 otherwise
// ============================================================
__global__ void k_find_singularities(const int* F2E, const int* FQ,
                                      const int* EdgeDiff, int nF,
                                      int* sing_flag) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    int sx = 0, sy = 0;
    for (int j = 0; j < 3; ++j) {
        int e = F2E[f * 3 + j];
        int dx = EdgeDiff[e * 2];
        int dy = EdgeDiff[e * 2 + 1];
        rshift90(dx, dy, FQ[f * 3 + j]);
        sx += dx;
        sy += dy;
    }
    sing_flag[f] = (sx != 0 || sy != 0) ? 1 : 0;
}

// ============================================================
// Kernel 3: Mark candidates for collapse
// An edge is a candidate if EdgeDiff == (0,0) and neither adjacent face is singular.
// candidate[i] = 1 if candidate, 0 otherwise
// ============================================================
__global__ void k_mark_candidates(const int* EdgeDiff, const int* E2F,
                                   const int* sing_flag, int nE,
                                   int* candidate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (EdgeDiff[i * 2] != 0 || EdgeDiff[i * 2 + 1] != 0) {
        candidate[i] = 0;
        return;
    }
    int f0 = E2F[i * 2];
    int f1 = E2F[i * 2 + 1];
    if ((f0 >= 0 && sing_flag[f0]) || (f1 >= 0 && sing_flag[f1])) {
        candidate[i] = 0;
        return;
    }
    candidate[i] = 1;
}

// Hash function for priority assignment (Knuth multiplicative + xorshift)
__device__ __host__ inline unsigned int edge_hash(int i) {
    unsigned int x = (unsigned int)i;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

// ============================================================
// Kernel 4: Select independent set (distance-2 in face adjacency)
// Uses index priority: lowest index wins. A candidate is selected
// if no candidate with a LOWER index exists in its distance-2
// face-adjacency neighborhood. Multi-round peeling with this
// priority produces the exact same selection as CPU sequential greedy.
// ============================================================
__global__ void k_select_independent_set(const int* candidate, const int* E2F,
                                          const int* F2E, int nE,
                                          int* selected) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (!candidate[i]) {
        selected[i] = 0;
        return;
    }
    // Check all candidate edges within distance-2 face adjacency
    // Edge i is selected if no candidate j < i exists in neighborhood
    for (int s = 0; s < 2; ++s) {
        int f = E2F[i * 2 + s];
        if (f < 0) continue;
        for (int j = 0; j < 3; ++j) {
            int ne = F2E[f * 3 + j];
            if (ne == i) continue;
            if (candidate[ne] && ne < i) {
                selected[i] = 0;
                return;
            }
            // Distance-2 neighbors: edges of ne's other face
            for (int t = 0; t < 2; ++t) {
                int nf = E2F[ne * 2 + t];
                if (nf < 0 || nf == f) continue;
                for (int k = 0; k < 3; ++k) {
                    int nne = F2E[nf * 3 + k];
                    if (nne == i || nne == ne) continue;
                    if (candidate[nne] && nne < i) {
                        selected[i] = 0;
                        return;
                    }
                }
            }
        }
    }
    selected[i] = 1;
}

// ============================================================
// Kernel 4b: Update candidates after a selection round
// Removes candidates whose adjacent faces are now in fixed_faces
// ============================================================
__global__ void k_update_candidates(int* candidate, const int* fixed_faces,
                                     const int* E2F, int nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (!candidate[i]) return;
    int f0 = E2F[i * 2], f1 = E2F[i * 2 + 1];
    if ((f0 >= 0 && fixed_faces[f0]) || (f1 >= 0 && fixed_faces[f1]))
        candidate[i] = 0;
}

// ============================================================
// Kernel 5: Apply selection — mark faces and toUpper for selected edges
// For each selected edge:
//   - toUpper[i] = -2
//   - fixed_faces[adjacent faces] = 2
//   - fixed_faces[neighbor faces] = max(1, current)
// ============================================================
__global__ void k_apply_selection(const int* selected, const int* E2F,
                                   const int* F2E, int nE,
                                   int* toUpper, int* fixed_faces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (!selected[i]) return;

    toUpper[i] = -2;

    // Mark neighbor faces as fixed=1 (buffer zone)
    for (int s = 0; s < 2; ++s) {
        int f = E2F[i * 2 + s];
        if (f < 0) continue;
        for (int j = 0; j < 3; ++j) {
            int ne = F2E[f * 3 + j];
            for (int t = 0; t < 2; ++t) {
                int nf = E2F[ne * 2 + t];
                if (nf >= 0) {
                    atomicMax(&fixed_faces[nf], 1);
                }
            }
        }
    }

    // Mark adjacent faces as fixed=2 (collapsed)
    for (int s = 0; s < 2; ++s) {
        int f = E2F[i * 2 + s];
        if (f >= 0) {
            atomicMax(&fixed_faces[f], 2);
        }
    }
}

// ============================================================
// Kernel 6: Classify remaining edges
// Edges with toUpper == -1 where both adjacent faces are fixed==2
// get toUpper = -3 (interior of collapsed region, will be removed)
// ============================================================
__global__ void k_classify_edges(const int* E2F, const int* fixed_faces,
                                  int nE, int* toUpper) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (toUpper[i] != -1) return;
    int f0 = E2F[i * 2];
    int f1 = E2F[i * 2 + 1];
    if ((f0 < 0 || fixed_faces[f0] == 2) && (f1 < 0 || fixed_faces[f1] == 2)) {
        toUpper[i] = -3;
    }
}

// ============================================================
// Kernel 7: Identify path starts and simple edges
// For edges with toUpper == -1:
//   - If both faces have fixed < 2: simple edge, produces_edge = 1
//   - If one face has fixed < 2, other has fixed == 2: path endpoint.
//     Trace the path to find the other endpoint. Only the LOWER-indexed
//     endpoint of each path produces an edge (avoids double-counting).
//   - path_start_face stores the non-collapsed face for path starts.
// ============================================================
__global__ void k_find_path_starts(const int* E2F, const int* F2E,
                                    const int* fixed_faces, const int* toUpper_in,
                                    int nE,
                                    int* produces_edge,
                                    int* path_start_face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (toUpper_in[i] != -1) {
        produces_edge[i] = 0;
        return;
    }
    int f0 = E2F[i * 2];
    int f1 = E2F[i * 2 + 1];
    int fix0 = (f0 >= 0) ? fixed_faces[f0] : 0;
    int fix1 = (f1 >= 0) ? fixed_faces[f1] : 0;

    if (fix0 < 2 && fix1 < 2) {
        // Simple edge — both sides non-collapsed
        produces_edge[i] = 1;
        path_start_face[i] = -1;  // sentinel for simple
        return;
    }

    if (fix0 >= 2 && fix1 >= 2) {
        // Both collapsed but not -3 — shouldn't normally happen
        produces_edge[i] = 0;
        return;
    }

    // One side collapsed, one not — this is a path endpoint.
    // Trace through collapsed faces to find the other endpoint edge.
    int start_face = (fix0 < 2) ? f0 : f1;
    int e = i;
    int f = start_face;

    for (int step = 0; step < 100; ++step) {
        // Cross edge e to the other face
        int next_f = (E2F[e * 2] == f) ? E2F[e * 2 + 1] : E2F[e * 2];

        if (next_f < 0 || fixed_faces[next_f] < 2) {
            // Reached the other side. 'e' is the exit edge (the other endpoint).
            // If e == i, this is a single-edge "path" (edge borders collapsed
            // region but exits immediately). It's a path start.
            // If e != i, compare indices: lower index is the canonical start.
            if (e == i || i < e) {
                produces_edge[i] = 1;
                path_start_face[i] = start_face;
            } else {
                produces_edge[i] = 0;
            }
            return;
        }

        // Find exit edge in next_f (an edge != e that has toUpper != -2)
        int next_e = -1;
        for (int j = 0; j < 3; ++j) {
            int e1 = F2E[next_f * 3 + j];
            if (e1 != e && toUpper_in[e1] != -2) {
                next_e = e1;
                break;
            }
        }

        if (next_e == -1) {
            // Dead end — all other edges collapsed. Self-loop path.
            produces_edge[i] = 1;
            path_start_face[i] = start_face;
            return;
        }

        e = next_e;
        f = next_f;
    }

    // Safety fallback
    produces_edge[i] = 1;
    path_start_face[i] = start_face;
}

// ============================================================
// Kernel 8: Follow paths and write toUpper + toUpperOrients
// Each edge that produces_edge == 1 follows its path through collapsed faces.
// edge_id[i] comes from prefix sum of produces_edge.
// ============================================================
__global__ void k_follow_paths(const int* E2F, const int* F2E, const int* FQ,
                                const int* fixed_faces, const int* EdgeDiff,
                                int nE,
                                int* toUpper, int* toUpperOrients,
                                const int* produces_edge, const int* edge_id,
                                const int* path_start_face,
                                int* nE2F_flat,  // output: new E2F[numE][2]
                                int* upperface_starts  // not used yet, placeholder
                                ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    if (!produces_edge[i]) return;

    int my_id = edge_id[i];

    if (path_start_face[i] == -1) {
        // Simple edge: both faces non-collapsed
        toUpper[i] = my_id;
        toUpperOrients[i] = 0;
        nE2F_flat[my_id * 2] = E2F[i * 2];
        nE2F_flat[my_id * 2 + 1] = E2F[i * 2 + 1];
        return;
    }

    // Path following
    int f0 = path_start_face[i];
    int e = i;
    int f = f0;

    // First edge in path
    toUpper[i] = my_id;
    toUpperOrients[i] = 0;
    int prev_orient = 0;

    // Follow through collapsed faces
    for (int step = 0; step < 100; ++step) {  // safety bound
        // Cross edge e to the other face
        int next_f;
        if (E2F[e * 2] == f)
            next_f = E2F[e * 2 + 1];
        else
            next_f = E2F[e * 2];

        if (next_f < 0 || fixed_faces[next_f] < 2) {
            // Reached the other side — done
            nE2F_flat[my_id * 2] = f0;
            nE2F_flat[my_id * 2 + 1] = next_f;
            return;
        }

        // Find entry edge index in face next_f
        int ind0 = -1;
        for (int j = 0; j < 3; ++j) {
            if (F2E[next_f * 3 + j] == e) {
                ind0 = j;
                break;
            }
        }

        // Find exit edge (another edge in next_f with toUpper != -2)
        int ind1 = -1;
        int next_e = -1;
        for (int j = 0; j < 3; ++j) {
            int e1 = F2E[next_f * 3 + j];
            if (e1 != e && toUpper[e1] != -2) {
                next_e = e1;
                ind1 = j;
                break;
            }
        }

        if (ind1 == -1) {
            // Dead end — all other edges are -2 (collapsed)
            // This is the "Unsatisfied" case — self-loop
            toUpper[e] = my_id;
            toUpperOrients[e] = 0;
            nE2F_flat[my_id * 2] = f0;
            nE2F_flat[my_id * 2 + 1] = f0;
            return;
        }

        // Compute orientation delta
        int orient = (FQ[next_f * 3 + ind1] - FQ[next_f * 3 + ind0] + 6) % 4;
        int cumulative = (orient + prev_orient) % 4;

        // Write toUpper for this path edge
        // Paths are disjoint and each traced by exactly one thread, so no races.
        toUpper[next_e] = my_id;
        toUpperOrients[next_e] = cumulative;
        prev_orient = cumulative;

        e = next_e;
        f = next_f;
    }

    // Safety: if we didn't break, set a default
    nE2F_flat[my_id * 2] = f0;
    nE2F_flat[my_id * 2 + 1] = -1;
}

// ============================================================
// Kernel 9: Build nEdgeDiff and nAllow from toUpper mapping
// ============================================================
__global__ void k_build_nEdgeDiff_nAllow(const int* toUpper, const int* toUpperOrients,
                                          const int* EdgeDiff, const int* Allow,
                                          int nE, int* nEdgeDiff, int* nAllow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    int upper = toUpper[i];
    if (upper < 0) return;

    int orient = toUpperOrients[i];
    if (orient == 0) {
        // This edge is the representative — copy EdgeDiff
        nEdgeDiff[upper * 2] = EdgeDiff[i * 2];
        nEdgeDiff[upper * 2 + 1] = EdgeDiff[i * 2 + 1];
    }

    // AllowChanges mapping
    int dim = orient % 2;
    // nAllow[upper*2] corresponds to dimension 'dim' of original
    // nAllow[upper*2+1] corresponds to dimension '1-dim' of original
    if (Allow[i * 2 + dim] == 0)
        atomicMin(&nAllow[upper * 2], 0);
    else if (Allow[i * 2 + dim] == 2)
        atomicMax(&nAllow[upper * 2], 2);
    if (Allow[i * 2 + 1 - dim] == 0)
        atomicMin(&nAllow[upper * 2 + 1], 0);
    else if (Allow[i * 2 + 1 - dim] == 2)
        atomicMax(&nAllow[upper * 2 + 1], 2);
}

// ============================================================
// Kernel 10: Count surviving faces (all 3 edges have toUpper >= 0)
// face_survives[f] = 1 if face survives, 0 otherwise
// ============================================================
__global__ void k_count_surviving_faces(const int* F2E, const int* toUpper,
                                         int nF, int* face_survives) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    int e0 = toUpper[F2E[f * 3]];
    int e1 = toUpper[F2E[f * 3 + 1]];
    int e2 = toUpper[F2E[f * 3 + 2]];
    face_survives[f] = (e0 >= 0 && e1 >= 0 && e2 >= 0) ? 1 : 0;
}

// ============================================================
// Kernel 11: Build nF2E and nFQ for surviving faces
// Uses prefix sum of face_survives to get compact face IDs
// Also writes upperface[old_f] = new_f
// ============================================================
__global__ void k_build_nF2E_nFQ(const int* F2E, const int* FQ,
                                  const int* toUpper, const int* toUpperOrients,
                                  const int* face_survives, const int* face_id,
                                  int nF,
                                  int* nF2E, int* nFQ, int* upperface) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    if (!face_survives[f]) {
        upperface[f] = -1;
        return;
    }
    int nf = face_id[f];
    upperface[f] = nf;
    for (int j = 0; j < 3; ++j) {
        int old_e = F2E[f * 3 + j];
        nF2E[nf * 3 + j] = toUpper[old_e];
        nFQ[nf * 3 + j] = (FQ[f * 3 + j] + 4 - toUpperOrients[old_e]) % 4;
    }
}

// ============================================================
// Kernel 12: Fix nE2F — remap face IDs using upperface
// ============================================================
__global__ void k_fix_nE2F(int* nE2F, const int* upperface, int numE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numE) return;
    for (int j = 0; j < 2; ++j) {
        int f = nE2F[i * 2 + j];
        if (f >= 0) {
            nE2F[i * 2 + j] = upperface[f];
        }
    }
}

// ============================================================
// Kernel 13: Build singularity flags for new level
// ============================================================
__global__ void k_find_singularities_new(const int* nF2E, const int* nFQ,
                                          const int* nEdgeDiff, int nF_new,
                                          int* sing_flag_new) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF_new) return;
    int sx = 0, sy = 0;
    for (int j = 0; j < 3; ++j) {
        int e = nF2E[f * 3 + j];
        int dx = nEdgeDiff[e * 2];
        int dy = nEdgeDiff[e * 2 + 1];
        rshift90(dx, dy, nFQ[f * 3 + j]);
        sx += dx;
        sy += dy;
    }
    sing_flag_new[f] = (sx != 0 || sy != 0) ? 1 : 0;
}

// ============================================================
// Host function: one level of GPU downsampling
// Returns: number of new edges (numE), number of new faces (numF)
// All device pointers are caller-managed.
// ============================================================
static inline int div_up(int a, int b) { return (a + b - 1) / b; }

struct DSELevelResult {
    int numE;
    int numF;
};

extern "C"
DSELevelResult cuda_downsample_edge_graph_level(
    // Input arrays (device pointers, current level)
    int* d_F2E, int* d_FQ, int* d_E2F, int* d_EdgeDiff, int* d_Allow,
    int* d_sing_flag,  // precomputed singularity flags for current level
    int nF, int nE,
    // Output arrays (device pointers, allocated by caller)
    int* d_toUpper, int* d_toUpperOrients,
    int* d_nF2E, int* d_nFQ, int* d_nE2F, int* d_nEdgeDiff, int* d_nAllow,
    int* d_sing_flag_new,
    int* d_upperface
) {
    const int BS = 256;

    // Initialize toUpper = -1, toUpperOrients = 0
    cudaMemset(d_toUpper, 0xFF, nE * sizeof(int));  // -1 in two's complement
    cudaMemset(d_toUpperOrients, 0, nE * sizeof(int));

    // Temp arrays
    thrust::device_vector<int> candidate(nE, 0);
    thrust::device_vector<int> selected(nE, 0);
    thrust::device_vector<int> fixed_faces(nF, 0);

    // Set fixed_faces for singular faces
    // We need to mark faces where sing_flag[f] == 1
    // Simple kernel or thrust transform
    {
        // Mark singular faces
        thrust::device_vector<int> d_ff(nF, 0);
        // Copy sing_flag to fixed_faces where sing_flag == 1
        // Just use sing_flag directly as initial fixed_faces
        cudaMemcpy(thrust::raw_pointer_cast(fixed_faces.data()),
                   d_sing_flag, nF * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Step 1: Mark candidates
    k_mark_candidates<<<div_up(nE, BS), BS>>>(
        d_EdgeDiff, d_E2F, d_sing_flag, nE,
        thrust::raw_pointer_cast(candidate.data()));
    cudaDeviceSynchronize();
    int n_candidates = thrust::count(candidate.begin(), candidate.end(), 1);

    // Steps 2-3: Multi-round index-priority peeling (matches CPU sequential greedy)
    int total_selected = 0;
    int rounds = 0;
    for (;; ++rounds) {
        // Select: local index-minima among remaining candidates
        k_select_independent_set<<<div_up(nE, BS), BS>>>(
            thrust::raw_pointer_cast(candidate.data()), d_E2F, d_F2E, nE,
            thrust::raw_pointer_cast(selected.data()));

        int n_sel = thrust::count(selected.begin(), selected.end(), 1);
        if (n_sel == 0) break;
        total_selected += n_sel;

        // Apply: mark toUpper=-2, fixed_faces=2/1
        k_apply_selection<<<div_up(nE, BS), BS>>>(
            thrust::raw_pointer_cast(selected.data()), d_E2F, d_F2E, nE,
            d_toUpper, thrust::raw_pointer_cast(fixed_faces.data()));

        // Update candidates: remove edges now blocked by fixed_faces
        k_update_candidates<<<div_up(nE, BS), BS>>>(
            thrust::raw_pointer_cast(candidate.data()),
            thrust::raw_pointer_cast(fixed_faces.data()),
            d_E2F, nE);
    }
    cudaDeviceSynchronize();
    printf("[DSE-GPU]   nF=%d nE=%d candidates=%d selected=%d rounds=%d", nF, nE, n_candidates, total_selected, rounds);

    // Step 4: Classify remaining edges
    k_classify_edges<<<div_up(nE, BS), BS>>>(
        d_E2F, thrust::raw_pointer_cast(fixed_faces.data()), nE, d_toUpper);

    // Step 5: Find path starts
    thrust::device_vector<int> produces_edge(nE, 0);
    thrust::device_vector<int> path_start_face(nE, -1);
    k_find_path_starts<<<div_up(nE, BS), BS>>>(
        d_E2F, d_F2E,
        thrust::raw_pointer_cast(fixed_faces.data()), d_toUpper,
        nE,
        thrust::raw_pointer_cast(produces_edge.data()),
        thrust::raw_pointer_cast(path_start_face.data()));

    // Step 6: Prefix sum to assign edge IDs
    thrust::device_vector<int> edge_id(nE);
    thrust::exclusive_scan(produces_edge.begin(), produces_edge.end(), edge_id.begin());
    int numE = 0;
    {
        int last_prod = 0, last_id = 0;
        cudaMemcpy(&last_prod, thrust::raw_pointer_cast(produces_edge.data()) + nE - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_id, thrust::raw_pointer_cast(edge_id.data()) + nE - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        numE = last_id + last_prod;
    }

    if (numE == 0) {
        return {0, 0};
    }

    // Initialize nE2F to -1
    cudaMemset(d_nE2F, 0xFF, numE * 2 * sizeof(int));

    // Step 7: Follow paths and write toUpper
    k_follow_paths<<<div_up(nE, BS), BS>>>(
        d_E2F, d_F2E, d_FQ,
        thrust::raw_pointer_cast(fixed_faces.data()), d_EdgeDiff,
        nE, d_toUpper, d_toUpperOrients,
        thrust::raw_pointer_cast(produces_edge.data()),
        thrust::raw_pointer_cast(edge_id.data()),
        thrust::raw_pointer_cast(path_start_face.data()),
        d_nE2F, nullptr);

    // Step 8: Build nEdgeDiff and nAllow
    // Initialize nAllow to 1 (default)
    {
        thrust::device_ptr<int> ptr(d_nAllow);
        thrust::fill(ptr, ptr + numE * 2, 1);
    }
    cudaMemset(d_nEdgeDiff, 0, numE * 2 * sizeof(int));
    k_build_nEdgeDiff_nAllow<<<div_up(nE, BS), BS>>>(
        d_toUpper, d_toUpperOrients, d_EdgeDiff, d_Allow,
        nE, d_nEdgeDiff, d_nAllow);

    // Step 9: Count surviving faces
    thrust::device_vector<int> face_survives(nF, 0);
    k_count_surviving_faces<<<div_up(nF, BS), BS>>>(
        d_F2E, d_toUpper, nF,
        thrust::raw_pointer_cast(face_survives.data()));

    // Step 10: Prefix sum for face IDs
    thrust::device_vector<int> face_id(nF);
    thrust::exclusive_scan(face_survives.begin(), face_survives.end(), face_id.begin());
    int numF = 0;
    {
        int last_surv = 0, last_fid = 0;
        cudaMemcpy(&last_surv, thrust::raw_pointer_cast(face_survives.data()) + nF - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_fid, thrust::raw_pointer_cast(face_id.data()) + nF - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        numF = last_fid + last_surv;
    }

    // Step 11: Build nF2E, nFQ, upperface
    k_build_nF2E_nFQ<<<div_up(nF, BS), BS>>>(
        d_F2E, d_FQ, d_toUpper, d_toUpperOrients,
        thrust::raw_pointer_cast(face_survives.data()),
        thrust::raw_pointer_cast(face_id.data()),
        nF, d_nF2E, d_nFQ, d_upperface);

    // Step 12: Fix nE2F with upperface remapping
    k_fix_nE2F<<<div_up(numE, BS), BS>>>(d_nE2F, d_upperface, numE);

    // Step 13: Find singularities for new level
    cudaMemset(d_sing_flag_new, 0, numF * sizeof(int));
    if (numF > 0) {
        k_find_singularities_new<<<div_up(numF, BS), BS>>>(
            d_nF2E, d_nFQ, d_nEdgeDiff, numF, d_sing_flag_new);
    }

    cudaDeviceSynchronize();

    printf(" -> numE=%d numF=%d\n", numE, numF);
    return {numE, numF};
}

// ============================================================
// Host function: Full multi-level GPU downsampling
// Takes host arrays, runs all levels on GPU, returns host arrays
// ============================================================
extern "C"
void cuda_downsample_edge_graph(
    // Input (host, moved from caller)
    int* h_F2E, int* h_FQ, int* h_EdgeDiff, int* h_Allow,
    int nF, int nE,
    // Output: arrays of per-level data (caller allocates outer vectors)
    // Returns via callback-style: fills the Hierarchy members
    int max_levels,
    // Output arrays (allocated by this function, caller frees)
    // We use a flat approach: pack all levels into output arrays
    // The caller provides pointers to vectors that we'll fill.
    // For simplicity, we'll output to host vectors.
    int** out_F2E,    int** out_FQ,    int** out_E2F,
    int** out_EdgeDiff, int** out_Allow,
    int** out_toUpper, int** out_toUpperOrients, int** out_upperface,
    int** out_sing_flag,
    int* out_nF,      int* out_nE,
    int* out_num_levels
) {
    // Upload input to GPU
    int* d_F2E, *d_FQ, *d_E2F, *d_EdgeDiff, *d_Allow;
    cudaMalloc(&d_F2E, nF * 3 * sizeof(int));
    cudaMalloc(&d_FQ, nF * 3 * sizeof(int));
    cudaMalloc(&d_E2F, nE * 2 * sizeof(int));
    cudaMalloc(&d_EdgeDiff, nE * 2 * sizeof(int));
    cudaMalloc(&d_Allow, nE * 2 * sizeof(int));

    cudaMemcpy(d_F2E, h_F2E, nF * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FQ, h_FQ, nF * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EdgeDiff, h_EdgeDiff, nE * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Allow, h_Allow, nE * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Build initial E2F on GPU
    cudaMemset(d_E2F, 0xFF, nE * 2 * sizeof(int));  // fill with -1
    k_build_E2F<<<div_up(nF, 256), 256>>>(d_F2E, nF, d_E2F, nE);

    // Find initial singularities
    int* d_sing_flag;
    cudaMalloc(&d_sing_flag, nF * sizeof(int));
    k_find_singularities<<<div_up(nF, 256), 256>>>(d_F2E, d_FQ, d_EdgeDiff, nF, d_sing_flag);
    cudaDeviceSynchronize();

    // Store level 0 on host
    out_nF[0] = nF;
    out_nE[0] = nE;
    out_F2E[0] = (int*)malloc(nF * 3 * sizeof(int));
    out_FQ[0] = (int*)malloc(nF * 3 * sizeof(int));
    out_E2F[0] = (int*)malloc(nE * 2 * sizeof(int));
    out_EdgeDiff[0] = (int*)malloc(nE * 2 * sizeof(int));
    out_Allow[0] = (int*)malloc(nE * 2 * sizeof(int));
    out_sing_flag[0] = (int*)malloc(nF * sizeof(int));
    cudaMemcpy(out_F2E[0], d_F2E, nF * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_FQ[0], d_FQ, nF * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_E2F[0], d_E2F, nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_EdgeDiff[0], d_EdgeDiff, nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_Allow[0], d_Allow, nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_sing_flag[0], d_sing_flag, nF * sizeof(int), cudaMemcpyDeviceToHost);

    int levels = 1;
    int cur_nF = nF, cur_nE = nE;

    for (int l = 0; l < max_levels - 1; ++l) {
        // Allocate output arrays for this level (max size = current)
        int* d_toUpper, *d_toUpperOrients;
        int* d_nF2E, *d_nFQ, *d_nE2F, *d_nEdgeDiff, *d_nAllow;
        int* d_sing_flag_new, *d_upperface;

        cudaMalloc(&d_toUpper, cur_nE * sizeof(int));
        cudaMalloc(&d_toUpperOrients, cur_nE * sizeof(int));
        cudaMalloc(&d_nF2E, cur_nF * 3 * sizeof(int));  // upper bound
        cudaMalloc(&d_nFQ, cur_nF * 3 * sizeof(int));
        cudaMalloc(&d_nE2F, cur_nE * 2 * sizeof(int));
        cudaMalloc(&d_nEdgeDiff, cur_nE * 2 * sizeof(int));
        cudaMalloc(&d_nAllow, cur_nE * 2 * sizeof(int));
        cudaMalloc(&d_sing_flag_new, cur_nF * sizeof(int));
        cudaMalloc(&d_upperface, cur_nF * sizeof(int));

        DSELevelResult result = cuda_downsample_edge_graph_level(
            d_F2E, d_FQ, d_E2F, d_EdgeDiff, d_Allow, d_sing_flag,
            cur_nF, cur_nE,
            d_toUpper, d_toUpperOrients,
            d_nF2E, d_nFQ, d_nE2F, d_nEdgeDiff, d_nAllow,
            d_sing_flag_new, d_upperface);

        int new_nE = result.numE;
        int new_nF = result.numF;

        // Copy level data to host
        out_toUpper[l] = (int*)malloc(cur_nE * sizeof(int));
        out_toUpperOrients[l] = (int*)malloc(cur_nE * sizeof(int));
        out_upperface[l] = (int*)malloc(cur_nF * sizeof(int));
        cudaMemcpy(out_toUpper[l], d_toUpper, cur_nE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_toUpperOrients[l], d_toUpperOrients, cur_nE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_upperface[l], d_upperface, cur_nF * sizeof(int), cudaMemcpyDeviceToHost);

        // Check convergence
        if (new_nE == cur_nE || new_nE == 0) {
            cudaFree(d_toUpper); cudaFree(d_toUpperOrients);
            cudaFree(d_nF2E); cudaFree(d_nFQ); cudaFree(d_nE2F);
            cudaFree(d_nEdgeDiff); cudaFree(d_nAllow);
            cudaFree(d_sing_flag_new); cudaFree(d_upperface);
            break;
        }

        levels++;
        out_nF[l + 1] = new_nF;
        out_nE[l + 1] = new_nE;

        // Copy new level arrays to host
        out_F2E[l + 1] = (int*)malloc(new_nF * 3 * sizeof(int));
        out_FQ[l + 1] = (int*)malloc(new_nF * 3 * sizeof(int));
        out_E2F[l + 1] = (int*)malloc(new_nE * 2 * sizeof(int));
        out_EdgeDiff[l + 1] = (int*)malloc(new_nE * 2 * sizeof(int));
        out_Allow[l + 1] = (int*)malloc(new_nE * 2 * sizeof(int));
        out_sing_flag[l + 1] = (int*)malloc(new_nF * sizeof(int));
        cudaMemcpy(out_F2E[l + 1], d_nF2E, new_nF * 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_FQ[l + 1], d_nFQ, new_nF * 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_E2F[l + 1], d_nE2F, new_nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_EdgeDiff[l + 1], d_nEdgeDiff, new_nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_Allow[l + 1], d_nAllow, new_nE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_sing_flag[l + 1], d_sing_flag_new, new_nF * sizeof(int), cudaMemcpyDeviceToHost);

        // Swap: new level becomes current
        cudaFree(d_F2E); cudaFree(d_FQ); cudaFree(d_E2F);
        cudaFree(d_EdgeDiff); cudaFree(d_Allow); cudaFree(d_sing_flag);

        d_F2E = d_nF2E;
        d_FQ = d_nFQ;
        d_E2F = d_nE2F;
        d_EdgeDiff = d_nEdgeDiff;
        d_Allow = d_nAllow;
        d_sing_flag = d_sing_flag_new;

        cudaFree(d_toUpper); cudaFree(d_toUpperOrients);
        cudaFree(d_upperface);

        cur_nF = new_nF;
        cur_nE = new_nE;
    }

    // Cleanup remaining device memory
    cudaFree(d_F2E); cudaFree(d_FQ); cudaFree(d_E2F);
    cudaFree(d_EdgeDiff); cudaFree(d_Allow); cudaFree(d_sing_flag);

    *out_num_levels = levels;
}

// ============================================================
// GPU CheckShrink: read-only parallel scan to find fixable faces
// ============================================================

#define MAX_LOOP_LEN 32

// Device rshift90 returning (x,y) pair
__device__ inline void rshift90_xy(int ix, int iy, int amount, int& ox, int& oy) {
    ox = ix; oy = iy;
    rshift90(ox, oy, amount);
}

// Device: try one CheckShrink call (read-only, no EdgeDiff modification)
// Returns 1 if shrinking deid would reduce flipped face count, 0 otherwise.
__device__ int try_checkshrink_readonly(
    int deid, int max_len,
    const int* E2E, const int* EdgeDiff, const int* F2E,
    const int* FQ, const int* AllowChange, int nF, int nE)
{
    if (deid < 0) return 0;

    // Navigate backwards to find loop start
    int deid0 = deid;
    while (deid >= 0) {
        int prev = (deid / 3) * 3 + (deid + 2) % 3;
        if (E2E[prev] < 0) { deid = prev; break; }
        deid = E2E[prev];
        if (deid == deid0) break;
    }

    // Walk the loop forward, collect edges and diffs
    int loop_edges[MAX_LOOP_LEN];   // deid values
    int loop_faces[MAX_LOOP_LEN];   // face IDs
    int loop_diff_x[MAX_LOOP_LEN];  // corresponding_diff[i].x
    int loop_diff_y[MAX_LOOP_LEN];  // corresponding_diff[i].y
    int loop_eid[MAX_LOOP_LEN];     // undirected edge IDs
    int loop_len = 0;

    int eid0 = F2E[(deid / 3) * 3 + (deid % 3)];
    int diff_x = EdgeDiff[eid0 * 2];
    int diff_y = EdgeDiff[eid0 * 2 + 1];
    int first_deid = deid;

    do {
        if (loop_len >= MAX_LOOP_LEN) return 0; // safety
        int f = deid / 3;
        int e = deid % 3;
        int eid = F2E[f * 3 + e];

        loop_edges[loop_len] = deid;
        loop_faces[loop_len] = f;
        loop_diff_x[loop_len] = diff_x;
        loop_diff_y[loop_len] = diff_y;
        loop_eid[loop_len] = eid;
        loop_len++;

        // Cross to next face
        deid = E2E[deid];
        if (deid < 0) return 0; // boundary = fail

        // Transform diff: diff = -rshift90(diff, FQ[deid/3][deid%3])
        int fq_val = FQ[(deid / 3) * 3 + (deid % 3)];
        int nd_x, nd_y;
        rshift90_xy(diff_x, diff_y, fq_val, nd_x, nd_y);
        diff_x = -nd_x;
        diff_y = -nd_y;

        // Advance to next edge in face
        deid = (deid / 3) * 3 + (deid + 1) % 3;

        // Transform to local: diff = rshift90(diff, (4 - FQ[deid/3][deid%3]) % 4)
        fq_val = FQ[(deid / 3) * 3 + (deid % 3)];
        rshift90_xy(diff_x, diff_y, (4 - fq_val) % 4, nd_x, nd_y);
        diff_x = nd_x;
        diff_y = nd_y;
    } while (deid != first_deid);

    // Check loop consistency
    if (deid >= 0 && (diff_x != loop_diff_x[0] || diff_y != loop_diff_y[0])) {
        return 0;
    }

    // Compute new_values and check constraints
    int new_val_x[MAX_LOOP_LEN];
    int new_val_y[MAX_LOOP_LEN];
    for (int i = 0; i < loop_len; ++i) {
        int eid = loop_eid[i];
        new_val_x[i] = EdgeDiff[eid * 2];
        new_val_y[i] = EdgeDiff[eid * 2 + 1];
    }

    for (int i = 0; i < loop_len; ++i) {
        int eid = loop_eid[i];
        // Check AllowChange
        if (loop_diff_x[i] != 0 && AllowChange[eid * 2] == 0) return 0;
        if (loop_diff_y[i] != 0 && AllowChange[eid * 2 + 1] == 0) return 0;

        new_val_x[i] -= loop_diff_x[i];
        new_val_y[i] -= loop_diff_y[i];

        int ax = abs(new_val_x[i]);
        int ay = abs(new_val_y[i]);
        if (ax > max_len || ay > max_len) return 0;
        if ((ax > 1 && ay != 0) || (ay > 1 && ax != 0)) return 0;
    }

    // Count negative-area faces before and after
    int prev_neg = 0, new_neg = 0;
    for (int i = 0; i < loop_len; ++i) {
        int f = loop_faces[i];

        // Before: use original EdgeDiff
        int e0 = F2E[f * 3 + 0], e1 = F2E[f * 3 + 1];
        int fq0 = FQ[f * 3 + 0], fq1 = FQ[f * 3 + 1];

        int d1x = EdgeDiff[e0 * 2], d1y = EdgeDiff[e0 * 2 + 1];
        rshift90(d1x, d1y, fq0);
        int d2x = EdgeDiff[e1 * 2], d2y = EdgeDiff[e1 * 2 + 1];
        rshift90(d2x, d2y, fq1);
        if (d1x * d2y - d1y * d2x < 0) prev_neg++;

        // After: substitute new_values where applicable
        d1x = EdgeDiff[e0 * 2]; d1y = EdgeDiff[e0 * 2 + 1];
        d2x = EdgeDiff[e1 * 2]; d2y = EdgeDiff[e1 * 2 + 1];
        // Check if e0 or e1 is in our modified set
        for (int j = 0; j < loop_len; ++j) {
            if (loop_eid[j] == e0) { d1x = new_val_x[j]; d1y = new_val_y[j]; }
            if (loop_eid[j] == e1) { d2x = new_val_x[j]; d2y = new_val_y[j]; }
        }
        rshift90(d1x, d1y, fq0);
        rshift90(d2x, d2y, fq1);
        if (d1x * d2y - d1y * d2x < 0) new_neg++;
    }

    return (new_neg < prev_neg) ? 1 : 0;
}

// Kernel: one thread per face, tries all 6 CheckShrink directions
__global__ void k_find_fixable_faces(
    const int* E2E, const int* EdgeDiff, const int* F2E,
    const int* FQ, const int* AllowChange,
    const int* face_area,  // precomputed: < 0 means flipped
    int nF, int nE, int max_len,
    int* can_fix)  // output: 1 if face can be fixed
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    if (face_area[f] >= 0) { can_fix[f] = 0; return; }

    // Try 3 edges × 2 directions
    for (int i = 0; i < 3; ++i) {
        int deid = f * 3 + i;
        if (try_checkshrink_readonly(deid, max_len, E2E, EdgeDiff, F2E, FQ, AllowChange, nF, nE)) {
            can_fix[f] = 1;
            return;
        }
        int opp = E2E[deid];
        if (try_checkshrink_readonly(opp, max_len, E2E, EdgeDiff, F2E, FQ, AllowChange, nF, nE)) {
            can_fix[f] = 1;
            return;
        }
    }
    can_fix[f] = 0;
}

// Kernel: compute area for each face
__global__ void k_compute_face_area(
    const int* EdgeDiff, const int* F2E, const int* FQ,
    int nF, int* area)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    int e0 = F2E[f * 3 + 0], e1 = F2E[f * 3 + 1];
    int d1x = EdgeDiff[e0 * 2], d1y = EdgeDiff[e0 * 2 + 1];
    rshift90(d1x, d1y, FQ[f * 3 + 0]);
    int d2x = EdgeDiff[e1 * 2], d2y = EdgeDiff[e1 * 2 + 1];
    rshift90(d2x, d2y, FQ[f * 3 + 1]);
    area[f] = d1x * d2y - d1y * d2x;
}

// Host function: find ALL fixable faces on GPU.
// Returns number of fixable faces found. Fills h_can_fix[nF] with 1/0.
// Also returns the number of flipped faces in *out_flipped_count.
extern "C"
int cuda_find_fixable_faces(
    const int* h_E2E,        // [nF * 3]
    const int* h_EdgeDiff,   // [nE * 2]
    const int* h_F2E,        // [nF * 3]
    const int* h_FQ,         // [nF * 3]
    const int* h_AllowChange,// [nE * 2]
    int nF, int nE, int max_len,
    int* h_can_fix,          // OUTPUT: [nF], 1 if fixable
    int* out_flipped_count)
{
    const int BS = 256;

    int *d_E2E, *d_EdgeDiff, *d_F2E, *d_FQ, *d_AllowChange, *d_area, *d_can_fix;
    cudaMalloc(&d_E2E, nF * 3 * sizeof(int));
    cudaMalloc(&d_EdgeDiff, nE * 2 * sizeof(int));
    cudaMalloc(&d_F2E, nF * 3 * sizeof(int));
    cudaMalloc(&d_FQ, nF * 3 * sizeof(int));
    cudaMalloc(&d_AllowChange, nE * 2 * sizeof(int));
    cudaMalloc(&d_area, nF * sizeof(int));
    cudaMalloc(&d_can_fix, nF * sizeof(int));

    cudaMemcpy(d_E2E, h_E2E, nF * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EdgeDiff, h_EdgeDiff, nE * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F2E, h_F2E, nF * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FQ, h_FQ, nF * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AllowChange, h_AllowChange, nE * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Compute areas
    k_compute_face_area<<<(nF + BS - 1) / BS, BS>>>(d_EdgeDiff, d_F2E, d_FQ, nF, d_area);

    // Count flipped
    std::vector<int> h_area(nF);
    cudaMemcpy(h_area.data(), d_area, nF * sizeof(int), cudaMemcpyDeviceToHost);
    int fc = 0;
    for (int i = 0; i < nF; ++i) if (h_area[i] < 0) fc++;
    *out_flipped_count = fc;

    if (fc == 0) {
        memset(h_can_fix, 0, nF * sizeof(int));
        cudaFree(d_E2E); cudaFree(d_EdgeDiff); cudaFree(d_F2E);
        cudaFree(d_FQ); cudaFree(d_AllowChange); cudaFree(d_area); cudaFree(d_can_fix);
        return 0;
    }

    // Find all fixable faces
    k_find_fixable_faces<<<(nF + BS - 1) / BS, BS>>>(
        d_E2E, d_EdgeDiff, d_F2E, d_FQ, d_AllowChange,
        d_area, nF, nE, max_len, d_can_fix);

    // Copy results to host
    cudaMemcpy(h_can_fix, d_can_fix, nF * sizeof(int), cudaMemcpyDeviceToHost);

    // Count fixable
    int n_fixable = 0;
    for (int i = 0; i < nF; ++i) if (h_can_fix[i]) n_fixable++;

    cudaFree(d_E2E); cudaFree(d_EdgeDiff); cudaFree(d_F2E);
    cudaFree(d_FQ); cudaFree(d_AllowChange); cudaFree(d_area); cudaFree(d_can_fix);

    return n_fixable;
}
