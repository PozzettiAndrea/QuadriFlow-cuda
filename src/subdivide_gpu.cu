// ============================================================
// GPU-resident parallel mesh subdivision
//
// Single H→D upload, all passes on GPU, single D→H download.
// Each pass:
//   1. GPU marks long edges
//   2. GPU resolves within-face conflicts (keep longest)
//   3. GPU resolves cross-face conflicts (lower index wins)
//   4. thrust::reduce to count splits. If 0, done.
//   5. thrust::exclusive_scan for vertex/face output offsets
//   6. GPU applies all splits in parallel (double-buffered F)
//   7. GPU rebuilds E2E via sort-based pairing
// ============================================================

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// ---- Kernel: mark edges that are too long ----
__global__ void k_mark_long_edges(
    const double* V,        // [3 x nV] col-major
    const int* F,           // [3 x nF] col-major
    const int* E2E,         // [3*nF]
    const int* nonmanifold, // [nV]
    int nF,
    double maxLengthSq,
    const double* rho,      // [nV]
    int* edge_marks         // [3*nF] output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;

    int f = idx / 3;
    int j = idx % 3;
    int v0 = F[j + f * 3];
    int v1 = F[((j + 1) % 3) + f * 3];

    edge_marks[idx] = 0;
    if (nonmanifold[v0] || nonmanifold[v1]) return;

    // Canonical: only mark if boundary or idx < twin
    int other = E2E[idx];
    if (other != -1 && other < idx) return;

    double d0 = V[0 + v0 * 3] - V[0 + v1 * 3];
    double d1 = V[1 + v0 * 3] - V[1 + v1 * 3];
    double d2 = V[2 + v0 * 3] - V[2 + v1 * 3];
    double lengthSq = d0 * d0 + d1 * d1 + d2 * d2;

    double rho_min = min(rho[v0], rho[v1]);
    if (lengthSq > maxLengthSq || lengthSq > max(maxLengthSq * 0.75, rho_min * 1.0)) {
        edge_marks[idx] = 1;
    }
}

// ---- Kernel: resolve within-face conflicts (keep longest) ----
__global__ void k_resolve_conflicts(
    const double* V,
    const int* F,
    int nF,
    int* edge_marks
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;

    int count = edge_marks[f * 3 + 0] + edge_marks[f * 3 + 1] + edge_marks[f * 3 + 2];
    if (count <= 1) return;

    double best_len = -1.0;
    int best_j = -1;
    for (int j = 0; j < 3; ++j) {
        if (!edge_marks[f * 3 + j]) continue;
        int v0 = F[j + f * 3];
        int v1 = F[((j + 1) % 3) + f * 3];
        double d0 = V[0 + v0 * 3] - V[0 + v1 * 3];
        double d1 = V[1 + v0 * 3] - V[1 + v1 * 3];
        double d2 = V[2 + v0 * 3] - V[2 + v1 * 3];
        double len = d0 * d0 + d1 * d1 + d2 * d2;
        if (len > best_len) {
            best_len = len;
            best_j = j;
        }
    }
    for (int j = 0; j < 3; ++j) {
        if (j != best_j) edge_marks[f * 3 + j] = 0;
    }
}

// ---- Kernel: resolve cross-face conflicts (lower index wins) ----
__global__ void k_resolve_cross_face(
    const int* E2E,
    int nF,
    int* edge_marks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;
    if (!edge_marks[idx]) return;

    int other = E2E[idx];
    if (other != -1 && edge_marks[other] && idx > other) {
        edge_marks[idx] = 0;
    }
}

// ---- Kernel: resolve neighbor conflicts (Luby-like independent set) ----
// A split of edge (idx) rewrites both f0 (containing idx) and f1 (twin's face).
// Two marks that share a face must not both survive.
// Each marked edge checks both faces it touches. If any competing mark is
// *longer* (or same length but higher index), this edge yields.
// Longest-first priority produces near-optimal triangulations matching CPU.
// Proof: if marks A and B both survive and share a face, then neither dominates
// the other: len(A)==len(B) and idx(A)==idx(B), so A==B. Contradiction. QED.
__global__ void k_resolve_neighbor_conflicts(
    const double* V,
    const int* F,
    const int* E2E,
    int nE,
    int* edge_marks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE) return;
    if (!edge_marks[idx]) return;

    // Compute my edge length squared
    int f0 = idx / 3, j0 = idx % 3;
    int va = F[j0 + f0 * 3], vb = F[((j0 + 1) % 3) + f0 * 3];
    double dx = V[va*3] - V[vb*3], dy = V[va*3+1] - V[vb*3+1], dz = V[va*3+2] - V[vb*3+2];
    double my_len = dx*dx + dy*dy + dz*dz;

    // Check if another marked edge beats us (longer wins, higher index breaks ties)
    auto dominated_by = [&](int other_idx) -> bool {
        if (other_idx < 0 || !edge_marks[other_idx]) return false;
        int fo = other_idx / 3, jo = other_idx % 3;
        int ua = F[jo + fo * 3], ub = F[((jo + 1) % 3) + fo * 3];
        double ex = V[ua*3] - V[ub*3], ey = V[ua*3+1] - V[ub*3+1], ez = V[ua*3+2] - V[ub*3+2];
        double other_len = ex*ex + ey*ey + ez*ez;
        return (other_len > my_len) || (other_len == my_len && other_idx > idx);
    };

    // Check f0 (our face): do any of our face-mates' twins bring in a longer mark?
    for (int j = 0; j < 3; j++) {
        int he = f0 * 3 + j;
        if (he == idx) continue;
        int twin = E2E[he];
        if (dominated_by(twin)) { edge_marks[idx] = 0; return; }
    }

    // Check f1 (twin's face)
    int twin_of_idx = E2E[idx];
    if (twin_of_idx == -1) return;  // boundary, no f1
    int f1 = twin_of_idx / 3;
    for (int j = 0; j < 3; j++) {
        int he = f1 * 3 + j;
        if (he == twin_of_idx) continue;  // skip twin (same edge as us)
        if (dominated_by(he)) { edge_marks[idx] = 0; return; }
        int twin2 = E2E[he];
        if (dominated_by(twin2)) { edge_marks[idx] = 0; return; }
    }
}

// ---- Kernel: compute face counts per marked edge ----
// For each marked half-edge: 1 new face if boundary, 2 if interior
__global__ void k_compute_face_counts(
    const int* edge_marks,
    const int* E2E,
    int nE,
    int* face_counts  // [nE] output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE) return;
    if (!edge_marks[idx]) { face_counts[idx] = 0; return; }
    face_counts[idx] = (E2E[idx] == -1) ? 1 : 2;
}

// ---- Kernel: apply splits in parallel ----
// Reads from F_old (snapshot), writes to F, V, rho, nm, bnd
__global__ void k_apply_splits(
    const int* F_old,       // snapshot of F before this pass
    int* F,                 // modified in-place
    double* V,
    double* rho,
    int* nm,
    int* bnd,
    const int* E2E,         // E2E from before this pass
    const int* edge_marks,
    const int* vtx_scan,    // exclusive scan of marks
    const int* face_scan,   // exclusive scan of face_counts
    int nV_old,
    int nF_old,
    int nE_old
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nE_old) return;
    if (!edge_marks[idx]) return;

    int f0 = idx / 3, j0 = idx % 3;
    int e1 = E2E[idx];
    bool is_boundary = (e1 == -1);

    // Read from snapshot
    int v0  = F_old[j0 + f0 * 3];
    int v0p = F_old[((j0 + 2) % 3) + f0 * 3];
    int v1  = F_old[((j0 + 1) % 3) + f0 * 3];

    int vn = nV_old + vtx_scan[idx];

    // New vertex = midpoint
    V[0 + vn * 3] = 0.5 * (V[0 + v0 * 3] + V[0 + v1 * 3]);
    V[1 + vn * 3] = 0.5 * (V[1 + v0 * 3] + V[1 + v1 * 3]);
    V[2 + vn * 3] = 0.5 * (V[2 + v0 * 3] + V[2 + v1 * 3]);
    rho[vn] = 0.5 * (rho[v0] + rho[v1]);
    nm[vn] = 0;
    bnd[vn] = is_boundary ? 1 : 0;

    // Rewrite f0: (vn, v0p, v0)
    F[0 + f0 * 3] = vn;
    F[1 + f0 * 3] = v0p;
    F[2 + f0 * 3] = v0;

    // New face f3: (vn, v1, v0p)
    int f3 = nF_old + face_scan[idx];
    F[0 + f3 * 3] = vn;
    F[1 + f3 * 3] = v1;
    F[2 + f3 * 3] = v0p;

    if (!is_boundary) {
        int f1 = e1 / 3, j1 = e1 % 3;
        int v1p = F_old[((j1 + 2) % 3) + f1 * 3];

        // Rewrite f1: (vn, v0, v1p)
        F[0 + f1 * 3] = vn;
        F[1 + f1 * 3] = v0;
        F[2 + f1 * 3] = v1p;

        // New face f2: (vn, v1p, v1)
        int f2 = nF_old + face_scan[idx] + 1;
        F[0 + f2 * 3] = vn;
        F[1 + f2 * 3] = v1p;
        F[2 + f2 * 3] = v1;
    }
}

// ---- Kernel: build sort keys for E2E reconstruction ----
// key = min(va,vb) * maxV + max(va,vb), value = half-edge index
__global__ void k_build_e2e_keys(
    const int* F,
    int nF,
    int maxV,           // upper bound on vertex index (for key packing)
    long long* keys,    // [3*nF] output
    int* indices        // [3*nF] output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nF * 3) return;

    int f = idx / 3, j = idx % 3;
    int va = F[j + f * 3];
    int vb = F[((j + 1) % 3) + f * 3];
    long long lo = (va < vb) ? va : vb;
    long long hi = (va < vb) ? vb : va;
    keys[idx] = lo * (long long)maxV + hi;
    indices[idx] = idx;
}

// ---- Kernel: pair sorted half-edges to build E2E ----
// After sorting by key, consecutive entries with same key are twins.
// Each thread checks if it's the FIRST of a pair (keys[i]==keys[i+1] && keys[i]!=keys[i-1]).
__global__ void k_pair_e2e(
    const long long* sorted_keys,
    const int* sorted_indices,
    int nE,
    int* E2E   // pre-filled with -1
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;

    long long my_key = sorted_keys[i];
    bool is_first = (i == 0 || sorted_keys[i - 1] != my_key);
    bool has_next = (i + 1 < nE && sorted_keys[i + 1] == my_key);

    if (is_first && has_next) {
        // This is the first of a twin pair — link them
        int h0 = sorted_indices[i];
        int h1 = sorted_indices[i + 1];
        E2E[h0] = h1;
        E2E[h1] = h0;
    }
    // Boundary edges (unpaired) keep E2E = -1 from the fill
}

// ============================================================
// Host function: GPU-resident mesh subdivision
// ============================================================
extern "C"
void cuda_subdivide_mesh(
    const double* V_in, int nV_in,
    const int* F_in, int nF_in,
    const double* rho_in,
    const int* E2E_in,
    const int* boundary_in,
    const int* nonmanifold_in,
    double maxLength,
    double** V_out, int* nV_out,
    int** F_out, int* nF_out,
    double** rho_out,
    int** boundary_out,
    int** nonmanifold_out
) {
    double maxLengthSq = maxLength * maxLength;
    int nV = nV_in, nF = nF_in;

    // Debug: rho and threshold stats
    {
        double rho_min = rho_in[0], rho_max = rho_in[0], rho_sum = 0;
        for (int i = 0; i < nV_in; i++) {
            if (rho_in[i] < rho_min) rho_min = rho_in[i];
            if (rho_in[i] > rho_max) rho_max = rho_in[i];
            rho_sum += rho_in[i];
        }
        printf("[SUBDIV-GPU] pre-split: nV=%d nF=%d maxLength=%.6f maxLengthSq=%.6f\n",
               nV_in, nF_in, maxLength, maxLengthSq);
        printf("[SUBDIV-GPU] rho stats: min=%.6f max=%.6f mean=%.6f\n",
               rho_min, rho_max, rho_sum / nV_in);
    }

    // Capacity: 10x initial size, minimum 1000
    int capV = std::max(nV_in * 10, 1000);
    int capF = std::max(nF_in * 10, 2000);
    int capE = capF * 3;

    printf("[SUBDIV-GPU] Allocating GPU buffers: capV=%d capF=%d (%.0f MB)\n",
           capV, capF,
           (3.0*capV*sizeof(double) + 3.0*capF*sizeof(int) + capE*sizeof(int) +
            capV*(sizeof(int)*2+sizeof(double)) + capE*(sizeof(int)*4+sizeof(long long)) +
            3.0*capF*sizeof(int) + capE*sizeof(int)) / 1e6);

    // Allocate GPU arrays with capacity
    double *d_V, *d_rho;
    int *d_F, *d_E2E, *d_nm, *d_bnd;

    cudaMalloc(&d_V,   3 * capV * sizeof(double));
    cudaMalloc(&d_F,   3 * capF * sizeof(int));
    cudaMalloc(&d_E2E, capE * sizeof(int));
    cudaMalloc(&d_nm,  capV * sizeof(int));
    cudaMalloc(&d_bnd, capV * sizeof(int));
    cudaMalloc(&d_rho, capV * sizeof(double));

    // Work arrays (allocated at max capacity)
    int *d_marks, *d_face_counts, *d_vtx_scan, *d_face_scan;
    int *d_F_old;
    long long *d_sort_keys;
    int *d_sort_indices;

    cudaMalloc(&d_marks,       capE * sizeof(int));
    cudaMalloc(&d_face_counts, capE * sizeof(int));
    cudaMalloc(&d_vtx_scan,    capE * sizeof(int));
    cudaMalloc(&d_face_scan,   capE * sizeof(int));
    cudaMalloc(&d_F_old,       3 * capF * sizeof(int));
    cudaMalloc(&d_sort_keys,   capE * sizeof(long long));
    cudaMalloc(&d_sort_indices,capE * sizeof(int));

    // Upload once
    printf("[XFER] H->D (once): V=%.1f MB, F=%.1f MB, E2E=%.1f MB, rho=%.1f KB, nm=%.1f KB, bnd=%.1f KB\n",
           3.0*nV*sizeof(double)/1e6, 3.0*nF*sizeof(int)/1e6, 3*nF*sizeof(int)/1e6,
           nV*sizeof(double)/1e3, nV*sizeof(int)/1e3, nV*sizeof(int)/1e3);

    cudaMemcpy(d_V,   V_in,           3 * nV * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,   F_in,           3 * nF * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_E2E, E2E_in,         3 * nF * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_nm,  nonmanifold_in,  nV * sizeof(int),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_bnd, boundary_in,     nV * sizeof(int),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho_in,          nV * sizeof(double),    cudaMemcpyHostToDevice);

    int total_splits = 0, pass = 0;
    const int block = 256;

    while (true) {
        int nE = nF * 3;
        int grid_e = (nE + block - 1) / block;
        int grid_f = (nF + block - 1) / block;

        // 1-4. Mark + resolve conflicts
        k_mark_long_edges<<<grid_e, block>>>(d_V, d_F, d_E2E, d_nm, nF, maxLengthSq, d_rho, d_marks);
        k_resolve_conflicts<<<grid_f, block>>>(d_V, d_F, nF, d_marks);
        k_resolve_cross_face<<<grid_e, block>>>(d_E2E, nF, d_marks);
        k_resolve_neighbor_conflicts<<<grid_e, block>>>(d_V, d_F, d_E2E, nE, d_marks);

        // 4. Count splits
        thrust::device_ptr<int> marks_ptr(d_marks);
        int num_splits = thrust::reduce(marks_ptr, marks_ptr + nE);
        if (num_splits == 0) break;
        total_splits += num_splits;

        // 5. Exclusive scan of marks → vertex offsets
        thrust::device_ptr<int> vtx_scan_ptr(d_vtx_scan);
        thrust::exclusive_scan(marks_ptr, marks_ptr + nE, vtx_scan_ptr);

        // 6-7. Compute face counts and scan
        k_compute_face_counts<<<grid_e, block>>>(d_marks, d_E2E, nE, d_face_counts);
        thrust::device_ptr<int> fc_ptr(d_face_counts);
        thrust::device_ptr<int> face_scan_ptr(d_face_scan);
        thrust::exclusive_scan(fc_ptr, fc_ptr + nE, face_scan_ptr);

        // Get total new faces from last element
        int last_fc, last_fs;
        cudaMemcpy(&last_fc, d_face_counts + nE - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_fs, d_face_scan + nE - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int total_new_faces = last_fs + last_fc;

        int new_nV = nV + num_splits;
        int new_nF = nF + total_new_faces;

        // Check capacity
        if (new_nV > capV || new_nF > capF) {
            printf("[SUBDIV-GPU] ERROR: capacity exceeded! nV=%d/%d nF=%d/%d\n",
                   new_nV, capV, new_nF, capF);
            break;
        }

        // 8. Snapshot F
        cudaMemcpy(d_F_old, d_F, 3 * nF * sizeof(int), cudaMemcpyDeviceToDevice);

        // 9. Apply splits
        k_apply_splits<<<grid_e, block>>>(
            d_F_old, d_F, d_V, d_rho, d_nm, d_bnd,
            d_E2E, d_marks, d_vtx_scan, d_face_scan,
            nV, nF, nE);

        // 10. Update counts
        nV = new_nV;
        nF = new_nF;
        int new_nE = nF * 3;
        int new_grid_e = (new_nE + block - 1) / block;

        // 11. Build E2E sort keys
        k_build_e2e_keys<<<new_grid_e, block>>>(d_F, nF, nV, d_sort_keys, d_sort_indices);

        // 12. Sort by key
        thrust::device_ptr<long long> keys_ptr(d_sort_keys);
        thrust::device_ptr<int> idx_ptr(d_sort_indices);
        thrust::sort_by_key(keys_ptr, keys_ptr + new_nE, idx_ptr);

        // 13. Initialize E2E to -1, then pair
        thrust::device_ptr<int> e2e_ptr(d_E2E);
        thrust::fill(e2e_ptr, e2e_ptr + new_nE, -1);

        k_pair_e2e<<<new_grid_e, block>>>(d_sort_keys, d_sort_indices, new_nE, d_E2E);

        pass++;
        printf("[SUBDIV-GPU]   pass %d: %d splits (%d total), nV=%d nF=%d\n",
               pass, num_splits, total_splits, nV, nF);

        // Debug: dump mesh state for tiny meshes
        cudaDeviceSynchronize();
        if (nV_in < 100) {
            double* dbg_V = (double*)malloc(3 * nV * sizeof(double));
            int* dbg_F = (int*)malloc(3 * nF * sizeof(int));
            int* dbg_marks = (int*)malloc(3 * nF * sizeof(int));
            cudaMemcpy(dbg_V, d_V, 3 * nV * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg_F, d_F, 3 * nF * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg_marks, d_marks, 3 * nF * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[SUBDIV-GPU]   vertices after pass %d:\n", pass);
            for (int i = 0; i < nV; i++)
                printf("    v%d: (%.4f, %.4f, %.4f)\n", i, dbg_V[i*3], dbg_V[i*3+1], dbg_V[i*3+2]);
            printf("[SUBDIV-GPU]   faces after pass %d:\n", pass);
            for (int i = 0; i < nF; i++)
                printf("    f%d: v%d v%d v%d\n", i, dbg_F[i*3], dbg_F[i*3+1], dbg_F[i*3+2]);
            free(dbg_V); free(dbg_F); free(dbg_marks);
        }
    }

    printf("[SUBDIV-GPU] Done: %d passes, %d total splits, final nV=%d nF=%d\n",
           pass, total_splits, nV, nF);

    // Download once
    printf("[XFER] D->H (once): V=%.1f MB, F=%.1f MB, rho=%.1f KB\n",
           3.0*nV*sizeof(double)/1e6, 3.0*nF*sizeof(int)/1e6, nV*sizeof(double)/1e3);

    *nV_out = nV;
    *nF_out = nF;
    *V_out = (double*)malloc(3 * nV * sizeof(double));
    *F_out = (int*)malloc(3 * nF * sizeof(int));
    *rho_out = (double*)malloc(nV * sizeof(double));
    *boundary_out = (int*)malloc(nV * sizeof(int));
    *nonmanifold_out = (int*)malloc(nV * sizeof(int));

    cudaMemcpy(*V_out,           d_V,   3 * nV * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(*F_out,           d_F,   3 * nF * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(*rho_out,         d_rho, nV * sizeof(double),     cudaMemcpyDeviceToHost);
    cudaMemcpy(*boundary_out,    d_bnd, nV * sizeof(int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(*nonmanifold_out, d_nm,  nV * sizeof(int),        cudaMemcpyDeviceToHost);

    // Free all GPU memory
    cudaFree(d_V); cudaFree(d_F); cudaFree(d_E2E);
    cudaFree(d_nm); cudaFree(d_bnd); cudaFree(d_rho);
    cudaFree(d_marks); cudaFree(d_face_counts);
    cudaFree(d_vtx_scan); cudaFree(d_face_scan);
    cudaFree(d_F_old);
    cudaFree(d_sort_keys); cudaFree(d_sort_indices);
}
