#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RCPOVERFLOW 2.93873587705571876e-39f
#define INVALID -1

// ============================================================
// Device helper functions
// ============================================================

__device__ __host__ inline int dedge_prev_3(int e) { return (e % 3 == 0) ? e + 2 : e - 1; }
__device__ __host__ inline int dedge_next_3(int e) { return (e % 3 == 2) ? e - 2 : e + 1; }

__device__ __host__ inline double fast_acos_d(double x) {
    double negate = double(x < 0.0);
    x = fabs(x);
    double ret = -0.0187293;
    ret *= x;
    ret = ret + 0.0742610;
    ret *= x;
    ret = ret - 0.2121144;
    ret *= x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0 - x);
    ret = ret - 2.0 * negate * ret;
    return negate * M_PI + ret;
}

// Eigen column-major access: MatrixXd(3, N) → data[row + col * 3]
// F(row, col) = F[row + col * 3], V(row, col) = V[row + col * 3]
__device__ inline void load_vec3(const double* M, int col, double& x, double& y, double& z) {
    int base = col * 3;
    x = M[base];
    y = M[base + 1];
    z = M[base + 2];
}

__device__ inline int load_F(const int* F, int row, int col) {
    return F[row + col * 3];
}

// ============================================================
// Kernel 1: Face normals
// ============================================================

__global__ void kernel_face_normals(
    const int* F, const double* V, double* Nf, int nFaces)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    int i0 = load_F(F, 0, f);
    int i1 = load_F(F, 1, f);
    int i2 = load_F(F, 2, f);

    double v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z;
    load_vec3(V, i0, v0x, v0y, v0z);
    load_vec3(V, i1, v1x, v1y, v1z);
    load_vec3(V, i2, v2x, v2y, v2z);

    // edge vectors
    double e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    double e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // cross product
    double nx = e1y * e2z - e1z * e2y;
    double ny = e1z * e2x - e1x * e2z;
    double nz = e1x * e2y - e1y * e2x;

    double norm = sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < RCPOVERFLOW) {
        nx = 1.0; ny = 0.0; nz = 0.0;
    } else {
        double inv = 1.0 / norm;
        nx *= inv; ny *= inv; nz *= inv;
    }

    int base = f * 3;
    Nf[base]     = nx;
    Nf[base + 1] = ny;
    Nf[base + 2] = nz;
}

// ============================================================
// Kernel 2: Smooth vertex normals (angle-weighted)
// ============================================================

__global__ void kernel_smooth_normals(
    const int* F, const double* V, const double* Nf,
    const int* V2E, const int* E2E,
    const int* nonManifold, const int* sharp_edges,
    double* N, int nVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    int edge = V2E[i];
    if (nonManifold[i] || edge == INVALID) {
        int base = i * 3;
        N[base] = 1.0; N[base + 1] = 0.0; N[base + 2] = 0.0;
        return;
    }

    // Find starting edge (walk to sharp edge or boundary)
    int stop = edge;
    do {
        if (sharp_edges[edge]) break;
        int opp = E2E[edge];
        if (opp == INVALID) { edge = INVALID; break; }
        edge = dedge_next_3(opp);
    } while (edge != stop);

    if (edge == INVALID)
        edge = stop;
    else
        stop = edge;

    // Accumulate angle-weighted face normals
    double normx = 0.0, normy = 0.0, normz = 0.0;
    do {
        int idx = edge % 3;
        int face = edge / 3;

        // d0 = V[F((idx+1)%3, face)] - V[i]
        int vi1 = load_F(F, (idx + 1) % 3, face);
        int vi2 = load_F(F, (idx + 2) % 3, face);

        double vix, viy, viz;
        load_vec3(V, i, vix, viy, viz);

        double d0x, d0y, d0z, d1x, d1y, d1z;
        double tmp_x, tmp_y, tmp_z;
        load_vec3(V, vi1, tmp_x, tmp_y, tmp_z);
        d0x = tmp_x - vix; d0y = tmp_y - viy; d0z = tmp_z - viz;
        load_vec3(V, vi2, tmp_x, tmp_y, tmp_z);
        d1x = tmp_x - vix; d1y = tmp_y - viy; d1z = tmp_z - viz;

        double dot = d0x * d1x + d0y * d1y + d0z * d1z;
        double len2_0 = d0x * d0x + d0y * d0y + d0z * d0z;
        double len2_1 = d1x * d1x + d1y * d1y + d1z * d1z;
        double denom = sqrt(len2_0 * len2_1);
        double angle = (denom > 0.0) ? fast_acos_d(dot / denom) : 0.0;

        if (isfinite(angle)) {
            int nf_base = face * 3;
            normx += Nf[nf_base]     * angle;
            normy += Nf[nf_base + 1] * angle;
            normz += Nf[nf_base + 2] * angle;
        }

        int opp = E2E[edge];
        if (opp == INVALID) break;
        edge = dedge_next_3(opp);
        if (sharp_edges[edge]) break;
    } while (edge != stop);

    double norm = sqrt(normx * normx + normy * normy + normz * normz);
    int base = i * 3;
    if (norm > RCPOVERFLOW) {
        double inv = 1.0 / norm;
        N[base] = normx * inv; N[base + 1] = normy * inv; N[base + 2] = normz * inv;
    } else {
        N[base] = 1.0; N[base + 1] = 0.0; N[base + 2] = 0.0;
    }
}

// ============================================================
// Kernel 3: Vertex area
// ============================================================

__global__ void kernel_vertex_area(
    const int* F, const double* V,
    const int* V2E, const int* E2E,
    const int* nonManifold,
    double* A, int nVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    int edge = V2E[i];
    int stop = edge;
    if (nonManifold[i] || edge == INVALID) {
        A[i] = 0.0;
        return;
    }

    double vertex_area = 0.0;
    do {
        int ep = dedge_prev_3(edge);
        int en = dedge_next_3(edge);

        double vx, vy, vz;
        load_vec3(V, load_F(F, edge % 3, edge / 3), vx, vy, vz);

        double vnx, vny, vnz;
        load_vec3(V, load_F(F, en % 3, en / 3), vnx, vny, vnz);

        double vpx, vpy, vpz;
        load_vec3(V, load_F(F, ep % 3, ep / 3), vpx, vpy, vpz);

        // face_center = (v + vp + vn) / 3
        double fcx = (vx + vpx + vnx) / 3.0;
        double fcy = (vy + vpy + vny) / 3.0;
        double fcz = (vz + vpz + vnz) / 3.0;

        // prev = (v + vp) / 2
        double px = (vx + vpx) * 0.5;
        double py = (vy + vpy) * 0.5;
        double pz = (vz + vpz) * 0.5;

        // next = (v + vn) / 2
        double nx = (vx + vnx) * 0.5;
        double ny = (vy + vny) * 0.5;
        double nz = (vz + vnz) * 0.5;

        // area1 = 0.5 * |cross(v - prev, v - face_center)|
        double a1x = vx - px, a1y = vy - py, a1z = vz - pz;
        double b1x = vx - fcx, b1y = vy - fcy, b1z = vz - fcz;
        double cx1 = a1y * b1z - a1z * b1y;
        double cy1 = a1z * b1x - a1x * b1z;
        double cz1 = a1x * b1y - a1y * b1x;
        double area1 = 0.5 * sqrt(cx1 * cx1 + cy1 * cy1 + cz1 * cz1);

        // area2 = 0.5 * |cross(v - next, v - face_center)|
        double a2x = vx - nx, a2y = vy - ny, a2z = vz - nz;
        double cx2 = a2y * b1z - a2z * b1y;  // reuse b1 = v - face_center
        double cy2 = a2z * b1x - a2x * b1z;
        double cz2 = a2x * b1y - a2y * b1x;
        double area2 = 0.5 * sqrt(cx2 * cx2 + cy2 * cy2 + cz2 * cz2);

        vertex_area += area1 + area2;

        int opp = E2E[edge];
        if (opp == INVALID) break;
        edge = dedge_next_3(opp);
    } while (edge != stop);

    A[i] = vertex_area;
}

// ============================================================
// Kernel 4-6: compute_direct_graph
// ============================================================

// Phase 1: Build half-edge linked lists using atomicCAS
__global__ void kernel_build_halfedge_lists(
    const int* F, int nFaces, int nVerts,
    int* V2E,           // initialized to INVALID
    unsigned int* tmp_next,  // nFaces*3, stores linked-list next pointers
    int* tmp_opposite       // nFaces*3, stores opposite vertex IDs
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    for (int ii = 0; ii < 3; ++ii) {
        int idx_cur = load_F(F, ii, f);
        int idx_next = load_F(F, (ii + 1) % 3, f);
        int edge_id = 3 * f + ii;

        if (idx_cur >= nVerts || idx_next >= nVerts) continue;
        if (idx_cur == idx_next) continue;

        tmp_opposite[edge_id] = idx_next;
        tmp_next[edge_id] = (unsigned int)INVALID;

        // Try to set V2E[idx_cur] = edge_id
        int old = atomicCAS(&V2E[idx_cur], INVALID, edge_id);
        if (old != INVALID) {
            // V2E[idx_cur] was already set. Walk the linked list and append.
            unsigned int idx = (unsigned int)old;
            while (true) {
                unsigned int expected = (unsigned int)INVALID;
                unsigned int prev = atomicCAS(&tmp_next[idx], expected, (unsigned int)edge_id);
                if (prev == expected) break;
                idx = prev;
            }
        }
    }
}

// Phase 2: Match opposite half-edges
__global__ void kernel_match_opposites(
    const int* F, int nFaces, int nVerts,
    const int* V2E,
    const unsigned int* tmp_next,
    const int* tmp_opposite,
    int* E2E       // initialized to INVALID, output
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    for (int ii = 0; ii < 3; ++ii) {
        int edge_id_cur = 3 * f + ii;
        int idx_cur = load_F(F, ii, f);
        int idx_next = load_F(F, (ii + 1) % 3, f);
        if (idx_cur == idx_next) continue;

        // Search linked list at idx_next for an edge pointing back to idx_cur
        int edge_id_opp = V2E[idx_next];
        int found_count = 0;
        int match = INVALID;
        while (edge_id_opp != INVALID) {
            if (tmp_opposite[edge_id_opp] == idx_cur) {
                found_count++;
                if (found_count == 1) match = edge_id_opp;
            }
            edge_id_opp = (int)tmp_next[edge_id_opp];
        }

        if (found_count == 1 && edge_id_cur < match) {
            // Bidirectional link (only the lower ID writes to avoid races)
            E2E[edge_id_cur] = match;
            E2E[match] = edge_id_cur;
        }
        // found_count > 1 means non-manifold (handled in phase 3)
    }
}

// Phase 3: Boundary detection + V2E normalization
__global__ void kernel_detect_boundary(
    const int* F, int nVerts, int nFaces,
    int* V2E,
    const int* E2E,
    int* boundary,
    int* nonManifold
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    int edge = V2E[i];
    boundary[i] = 0;
    nonManifold[i] = 0;

    if (edge == INVALID) return;

    // Walk backwards to find a boundary edge
    int start = edge;
    do {
        // Go to prev edge in face, then to opposite
        int prev = dedge_prev_3(edge);
        int opp = E2E[prev];
        if (opp == INVALID) {
            // Found boundary
            boundary[i] = 1;
            V2E[i] = edge;
            return;
        }
        edge = opp;
    } while (edge != start);

    // No boundary found — interior vertex, V2E stays as is
}

// ============================================================
// Wrapper functions (extern "C")
// ============================================================

extern "C" void cuda_compute_smooth_normal(
    const int* h_F, int nFaces,
    const double* h_V, int nVerts,
    const int* h_V2E,
    const int* h_E2E,
    const int* h_nonManifold,
    const int* h_sharp_edges,
    double* h_N,
    double* h_Nf)
{
    // Allocate device memory
    int *d_F, *d_V2E, *d_E2E, *d_nonManifold, *d_sharp_edges;
    double *d_V, *d_N, *d_Nf;

    cudaMalloc(&d_F, 3 * nFaces * sizeof(int));
    cudaMalloc(&d_V, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_V2E, nVerts * sizeof(int));
    cudaMalloc(&d_E2E, 3 * nFaces * sizeof(int));
    cudaMalloc(&d_nonManifold, nVerts * sizeof(int));
    cudaMalloc(&d_sharp_edges, 3 * nFaces * sizeof(int));
    cudaMalloc(&d_N, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_Nf, 3 * nFaces * sizeof(double));

    // Upload
    cudaMemcpy(d_F, h_F, 3 * nFaces * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 3 * nVerts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2E, h_V2E, nVerts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E2E, h_E2E, 3 * nFaces * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonManifold, h_nonManifold, nVerts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sharp_edges, h_sharp_edges, 3 * nFaces * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel 1: face normals
    int blockSize = 256;
    int gridFaces = (nFaces + blockSize - 1) / blockSize;
    kernel_face_normals<<<gridFaces, blockSize>>>(d_F, d_V, d_Nf, nFaces);

    // Kernel 2: smooth vertex normals
    int gridVerts = (nVerts + blockSize - 1) / blockSize;
    kernel_smooth_normals<<<gridVerts, blockSize>>>(
        d_F, d_V, d_Nf, d_V2E, d_E2E, d_nonManifold, d_sharp_edges, d_N, nVerts);

    // Download results
    cudaMemcpy(h_N, d_N, 3 * nVerts * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Nf, d_Nf, 3 * nFaces * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_F); cudaFree(d_V); cudaFree(d_V2E); cudaFree(d_E2E);
    cudaFree(d_nonManifold); cudaFree(d_sharp_edges); cudaFree(d_N); cudaFree(d_Nf);
}

extern "C" void cuda_compute_vertex_area(
    const int* h_F, int nFaces,
    const double* h_V, int nVerts,
    const int* h_V2E,
    const int* h_E2E,
    const int* h_nonManifold,
    double* h_A)
{
    int *d_F, *d_V2E, *d_E2E, *d_nonManifold;
    double *d_V, *d_A;

    cudaMalloc(&d_F, 3 * nFaces * sizeof(int));
    cudaMalloc(&d_V, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_V2E, nVerts * sizeof(int));
    cudaMalloc(&d_E2E, 3 * nFaces * sizeof(int));
    cudaMalloc(&d_nonManifold, nVerts * sizeof(int));
    cudaMalloc(&d_A, nVerts * sizeof(double));

    cudaMemcpy(d_F, h_F, 3 * nFaces * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 3 * nVerts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2E, h_V2E, nVerts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E2E, h_E2E, 3 * nFaces * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonManifold, h_nonManifold, nVerts * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridVerts = (nVerts + blockSize - 1) / blockSize;
    kernel_vertex_area<<<gridVerts, blockSize>>>(d_F, d_V, d_V2E, d_E2E, d_nonManifold, d_A, nVerts);

    cudaMemcpy(h_A, d_A, nVerts * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_F); cudaFree(d_V); cudaFree(d_V2E); cudaFree(d_E2E);
    cudaFree(d_nonManifold); cudaFree(d_A);
}

extern "C" int cuda_compute_direct_graph(
    const double* h_V, int nVerts,
    const int* h_F, int nFaces,
    int* h_V2E,
    int* h_E2E,
    int* h_boundary,
    int* h_nonManifold)
{
    int *d_F, *d_V2E, *d_E2E, *d_boundary, *d_nonManifold;
    double *d_V;
    unsigned int *d_tmp_next;
    int *d_tmp_opposite;
    int nEdges = 3 * nFaces;

    cudaMalloc(&d_F, nEdges * sizeof(int));
    cudaMalloc(&d_V, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_V2E, nVerts * sizeof(int));
    cudaMalloc(&d_E2E, nEdges * sizeof(int));
    cudaMalloc(&d_boundary, nVerts * sizeof(int));
    cudaMalloc(&d_nonManifold, nVerts * sizeof(int));
    cudaMalloc(&d_tmp_next, nEdges * sizeof(unsigned int));
    cudaMalloc(&d_tmp_opposite, nEdges * sizeof(int));

    cudaMemcpy(d_F, h_F, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 3 * nVerts * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize V2E and E2E to INVALID
    cudaMemset(d_V2E, 0xFF, nVerts * sizeof(int));     // sets to -1 (all bits set)
    cudaMemset(d_E2E, 0xFF, nEdges * sizeof(int));
    cudaMemset(d_tmp_next, 0xFF, nEdges * sizeof(unsigned int));
    cudaMemset(d_tmp_opposite, 0xFF, nEdges * sizeof(int));

    int blockSize = 256;
    int gridFaces = (nFaces + blockSize - 1) / blockSize;
    int gridVerts = (nVerts + blockSize - 1) / blockSize;

    // Phase 1: Build half-edge linked lists
    kernel_build_halfedge_lists<<<gridFaces, blockSize>>>(
        d_F, nFaces, nVerts, d_V2E, d_tmp_next, d_tmp_opposite);
    cudaDeviceSynchronize();

    // Phase 2: Match opposites
    kernel_match_opposites<<<gridFaces, blockSize>>>(
        d_F, nFaces, nVerts, d_V2E, d_tmp_next, d_tmp_opposite, d_E2E);
    cudaDeviceSynchronize();

    // Phase 3: Boundary detection
    kernel_detect_boundary<<<gridVerts, blockSize>>>(
        d_F, nVerts, nFaces, d_V2E, d_E2E, d_boundary, d_nonManifold);

    // Download results
    cudaMemcpy(h_V2E, d_V2E, nVerts * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E2E, d_E2E, nEdges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_boundary, d_boundary, nVerts * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nonManifold, d_nonManifold, nVerts * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_F); cudaFree(d_V); cudaFree(d_V2E); cudaFree(d_E2E);
    cudaFree(d_boundary); cudaFree(d_nonManifold);
    cudaFree(d_tmp_next); cudaFree(d_tmp_opposite);

    return 0;
}

// ============================================================
// Kernel 7: Count adjacency (pass 1 of 2 for generate_adjacency_matrix)
// ============================================================

__global__ void kernel_count_adjacency(
    const int* F, const int* V2E, const int* E2E,
    const int* nonManifold, int* counts, int nVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    int start = V2E[i];
    if (start == INVALID || nonManifold[i]) {
        counts[i] = 0;
        return;
    }

    int count = 0;
    int edge = start;
    do {
        int opp = E2E[edge];
        int next = (opp == INVALID) ? INVALID : dedge_next_3(opp);

        if (count == 0)
            count++; // first neighbor

        if (opp == INVALID || next != start)
            count++; // additional neighbor

        if (opp == INVALID) break;
        edge = next;
    } while (edge != start);

    counts[i] = count;
}

// ============================================================
// Kernel 8: Fill adjacency (pass 2 of 2)
// ============================================================

__global__ void kernel_fill_adjacency(
    const int* F, const int* V2E, const int* E2E,
    const int* nonManifold,
    const int* rowPtr,   // CSR row pointers (prefix sum of counts)
    int* colInd,         // output: neighbor IDs
    double* weights,     // output: all 1.0
    int nVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    int start = V2E[i];
    if (start == INVALID || nonManifold[i]) return;

    int offset = rowPtr[i];
    int edge = start;
    do {
        int base = edge % 3, f = edge / 3;
        int opp = E2E[edge];
        int next = (opp == INVALID) ? INVALID : dedge_next_3(opp);

        if (offset == rowPtr[i]) {
            // first neighbor
            colInd[offset] = load_F(F, (base + 2) % 3, f);
            weights[offset] = 1.0;
            offset++;
        }

        if (opp == INVALID || next != start) {
            colInd[offset] = load_F(F, (base + 1) % 3, f);
            weights[offset] = 1.0;
            offset++;
            if (opp == INVALID) break;
        }
        edge = next;
    } while (edge != start);
}

// ============================================================
// Kernel 9: Rho smoothing (min over neighbors)
// ============================================================

__global__ void kernel_rho_smooth(
    const int* rowPtr, const int* colInd,
    const double* rho_in, double* rho_out, int nVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;

    double val = rho_in[i];
    int start = rowPtr[i];
    int end = rowPtr[i + 1];
    for (int j = start; j < end; ++j) {
        double neighbor_val = rho_in[colInd[j]];
        if (neighbor_val < val) val = neighbor_val;
    }
    rho_out[i] = val;
}

// ============================================================
// Wrapper: generate_adjacency_matrix on GPU
// Returns CSR format: rowPtr (nVerts+1), colInd (nnz), weights (nnz)
// ============================================================

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

extern "C" void cuda_generate_adjacency_matrix(
    const int* h_F, int nFaces,
    const int* h_V2E, const int* h_E2E,
    const int* h_nonManifold, int nVerts,
    int** h_rowPtr_out, int** h_colInd_out, double** h_weights_out, int* nnz_out)
{
    int *d_F, *d_V2E, *d_E2E, *d_nonManifold;
    int nEdges = 3 * nFaces;

    cudaMalloc(&d_F, nEdges * sizeof(int));
    cudaMalloc(&d_V2E, nVerts * sizeof(int));
    cudaMalloc(&d_E2E, nEdges * sizeof(int));
    cudaMalloc(&d_nonManifold, nVerts * sizeof(int));

    cudaMemcpy(d_F, h_F, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2E, h_V2E, nVerts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E2E, h_E2E, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonManifold, h_nonManifold, nVerts * sizeof(int), cudaMemcpyHostToDevice);

    // Pass 1: count neighbors per vertex
    int* d_counts;
    cudaMalloc(&d_counts, nVerts * sizeof(int));

    int blockSize = 256;
    int gridVerts = (nVerts + blockSize - 1) / blockSize;
    kernel_count_adjacency<<<gridVerts, blockSize>>>(d_F, d_V2E, d_E2E, d_nonManifold, d_counts, nVerts);

    // Prefix sum to get CSR row pointers
    int* d_rowPtr;
    cudaMalloc(&d_rowPtr, (nVerts + 1) * sizeof(int));
    cudaMemset(d_rowPtr, 0, sizeof(int)); // first element = 0

    thrust::device_ptr<int> counts_ptr(d_counts);
    thrust::device_ptr<int> rowPtr_ptr(d_rowPtr);
    thrust::exclusive_scan(counts_ptr, counts_ptr + nVerts, rowPtr_ptr, 0);
    // Set last element = total nnz
    int total_nnz;
    int last_count;
    int last_offset;
    cudaMemcpy(&last_count, d_counts + nVerts - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_offset, d_rowPtr + nVerts - 1, sizeof(int), cudaMemcpyDeviceToHost);
    total_nnz = last_offset + last_count;
    cudaMemcpy(d_rowPtr + nVerts, &total_nnz, sizeof(int), cudaMemcpyHostToDevice);

    // Pass 2: fill adjacency
    int* d_colInd;
    double* d_weights;
    cudaMalloc(&d_colInd, total_nnz * sizeof(int));
    cudaMalloc(&d_weights, total_nnz * sizeof(double));

    kernel_fill_adjacency<<<gridVerts, blockSize>>>(
        d_F, d_V2E, d_E2E, d_nonManifold, d_rowPtr, d_colInd, d_weights, nVerts);

    // Download
    *h_rowPtr_out = (int*)malloc((nVerts + 1) * sizeof(int));
    *h_colInd_out = (int*)malloc(total_nnz * sizeof(int));
    *h_weights_out = (double*)malloc(total_nnz * sizeof(double));
    *nnz_out = total_nnz;

    cudaMemcpy(*h_rowPtr_out, d_rowPtr, (nVerts + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*h_colInd_out, d_colInd, total_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*h_weights_out, d_weights, total_nnz * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_F); cudaFree(d_V2E); cudaFree(d_E2E); cudaFree(d_nonManifold);
    cudaFree(d_counts); cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_weights);
}

// ============================================================
// Wrapper: rho smoothing on GPU (5 iterations)
// Uses CSR adjacency
// ============================================================

extern "C" void cuda_rho_smooth(
    const int* h_rowPtr, const int* h_colInd,
    int nVerts, int nnz,
    double* h_rho,  // in/out
    int iterations)
{
    int *d_rowPtr, *d_colInd;
    double *d_rho_a, *d_rho_b;

    cudaMalloc(&d_rowPtr, (nVerts + 1) * sizeof(int));
    cudaMalloc(&d_colInd, nnz * sizeof(int));
    cudaMalloc(&d_rho_a, nVerts * sizeof(double));
    cudaMalloc(&d_rho_b, nVerts * sizeof(double));

    cudaMemcpy(d_rowPtr, h_rowPtr, (nVerts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho_a, h_rho, nVerts * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int grid = (nVerts + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0)
            kernel_rho_smooth<<<grid, blockSize>>>(d_rowPtr, d_colInd, d_rho_a, d_rho_b, nVerts);
        else
            kernel_rho_smooth<<<grid, blockSize>>>(d_rowPtr, d_colInd, d_rho_b, d_rho_a, nVerts);
    }

    // Copy result back (depends on iteration parity)
    if (iterations % 2 == 1)
        cudaMemcpy(h_rho, d_rho_b, nVerts * sizeof(double), cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(h_rho, d_rho_a, nVerts * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_rho_a); cudaFree(d_rho_b);
}

// ============================================================
// DownsampleGraph GPU kernels
// ============================================================

// Kernel 10: Score entries for DownsampleGraph
// Each thread processes one vertex's adjacency list
__global__ void kernel_score_entries(
    const double* N,     // 3 * nVerts (normals, column-major)
    const double* A,     // nVerts (areas)
    const int* adjRowPtr,
    const int* adjColInd,
    int* out_i, int* out_j, double* out_order,
    int nVerts)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nVerts) return;

    double nx, ny, nz;
    load_vec3(N, v, nx, ny, nz);
    double av = A[v];

    int start = adjRowPtr[v];
    int end = adjRowPtr[v + 1];
    for (int idx = start; idx < end; ++idx) {
        int k = adjColInd[idx];
        double knx, kny, knz;
        load_vec3(N, k, knx, kny, knz);
        double dp = nx * knx + ny * kny + nz * knz;
        double ak = A[k];
        double ratio = av > ak ? (av / ak) : (ak / av);
        out_i[idx] = v;
        out_j[idx] = k;
        out_order[idx] = dp * ratio;
    }
}

// Kernel 11: Build to_lower mapping and coarse vertex data for collapsed pairs
__global__ void kernel_build_collapsed(
    const int* collapsed_i,   // nCollapsed pairs (i vertex)
    const int* collapsed_j,   // nCollapsed pairs (j vertex)
    const double* V, const double* N, const double* A,
    double* V_p, double* N_p, double* A_p,
    int* to_upper,    // 2 * vertexCount (column-major)
    int* to_lower,    // nVerts
    int nCollapsed, int vertexCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCollapsed) return;

    int ei = collapsed_i[idx];
    int ej = collapsed_j[idx];
    double area1 = A[ei], area2 = A[ej];
    double surfaceArea = area1 + area2;

    // V_p
    double vix, viy, viz, vjx, vjy, vjz;
    load_vec3(V, ei, vix, viy, viz);
    load_vec3(V, ej, vjx, vjy, vjz);
    int base = idx * 3;
    if (surfaceArea > RCPOVERFLOW) {
        double inv = 1.0 / surfaceArea;
        V_p[base]     = (vix * area1 + vjx * area2) * inv;
        V_p[base + 1] = (viy * area1 + vjy * area2) * inv;
        V_p[base + 2] = (viz * area1 + vjz * area2) * inv;
    } else {
        V_p[base]     = (vix + vjx) * 0.5;
        V_p[base + 1] = (viy + vjy) * 0.5;
        V_p[base + 2] = (viz + vjz) * 0.5;
    }

    // N_p
    double nix, niy, niz, njx, njy, njz;
    load_vec3(N, ei, nix, niy, niz);
    load_vec3(N, ej, njx, njy, njz);
    double nnx = nix * area1 + njx * area2;
    double nny = niy * area1 + njy * area2;
    double nnz = niz * area1 + njz * area2;
    double norm = sqrt(nnx * nnx + nny * nny + nnz * nnz);
    if (norm > RCPOVERFLOW) {
        double inv = 1.0 / norm;
        N_p[base]     = nnx * inv;
        N_p[base + 1] = nny * inv;
        N_p[base + 2] = nnz * inv;
    } else {
        N_p[base] = 1.0; N_p[base + 1] = 0.0; N_p[base + 2] = 0.0;
    }

    A_p[idx] = surfaceArea;

    // to_upper (2, vertexCount) column-major: to_upper(row, col) = data[row + col*2]
    to_upper[0 + idx * 2] = ei;
    to_upper[1 + idx * 2] = ej;

    to_lower[ei] = idx;
    to_lower[ej] = idx;
}

// Kernel 12: Copy unmerged vertices
__global__ void kernel_copy_unmerged(
    const int* mergeFlag,    // nVerts, 1=merged, 0=not
    const int* prefixSum,    // exclusive prefix sum of !mergeFlag
    const double* V, const double* N, const double* A,
    double* V_p, double* N_p, double* A_p,
    int* to_upper, int* to_lower,
    int nVerts, int nCollapsed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVerts) return;
    if (mergeFlag[i]) return;

    int idx = nCollapsed + prefixSum[i];

    // Copy vertex data
    int src_base = i * 3, dst_base = idx * 3;
    V_p[dst_base]     = V[src_base];
    V_p[dst_base + 1] = V[src_base + 1];
    V_p[dst_base + 2] = V[src_base + 2];
    N_p[dst_base]     = N[src_base];
    N_p[dst_base + 1] = N[src_base + 1];
    N_p[dst_base + 2] = N[src_base + 2];
    A_p[idx] = A[i];

    to_upper[0 + idx * 2] = i;
    to_upper[1 + idx * 2] = -1;
    to_lower[i] = idx;
}

// Kernel 13: Count new adjacency per coarse vertex
__global__ void kernel_count_coarse_adj(
    const int* to_upper,     // 2 * vertexCount_p (column-major)
    const int* adjRowPtr,    // fine level CSR
    const int* adjColInd,
    const int* to_lower,     // fine -> coarse mapping
    int* counts,             // output: per coarse vertex neighbor count (upper bound)
    int vertexCount_p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertexCount_p) return;

    int total = 0;
    for (int j = 0; j < 2; ++j) {
        int upper = to_upper[j + i * 2];
        if (upper == -1) continue;
        total += adjRowPtr[upper + 1] - adjRowPtr[upper];
    }
    counts[i] = total;
}

// Kernel 14: Fill coarse adjacency (with dedup)
// This produces a CSR where duplicates are merged by summing weights
__global__ void kernel_fill_coarse_adj(
    const int* to_upper,
    const int* adjRowPtr,
    const int* adjColInd,
    const double* adjWeights,
    const int* to_lower,
    const int* outRowPtr,    // CSR row pointers for output (from prefix sum of counts)
    int* outColInd,
    double* outWeights,
    int* actualCount,        // actual count per coarse vertex after dedup
    int vertexCount_p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vertexCount_p) return;

    int out_start = outRowPtr[i];
    int n = 0; // actual neighbors written

    // Collect all neighbors, remap through to_lower
    // We use the output buffer as scratch, then sort & compact in-place
    for (int j = 0; j < 2; ++j) {
        int upper = to_upper[j + i * 2];
        if (upper == -1) continue;
        int start = adjRowPtr[upper];
        int end = adjRowPtr[upper + 1];
        for (int k = start; k < end; ++k) {
            int mapped = to_lower[adjColInd[k]];
            if (mapped != i) { // skip self-loops
                outColInd[out_start + n] = mapped;
                outWeights[out_start + n] = adjWeights[k];
                n++;
            }
        }
    }

    // Simple insertion sort (small n, typically < 20)
    for (int a = 1; a < n; ++a) {
        int key_id = outColInd[out_start + a];
        double key_w = outWeights[out_start + a];
        int b = a - 1;
        while (b >= 0 && outColInd[out_start + b] > key_id) {
            outColInd[out_start + b + 1] = outColInd[out_start + b];
            outWeights[out_start + b + 1] = outWeights[out_start + b];
            b--;
        }
        outColInd[out_start + b + 1] = key_id;
        outWeights[out_start + b + 1] = key_w;
    }

    // Compact: merge duplicates by summing weights
    int write = 0;
    for (int a = 0; a < n; ++a) {
        if (write > 0 && outColInd[out_start + write - 1] == outColInd[out_start + a]) {
            outWeights[out_start + write - 1] += outWeights[out_start + a];
        } else {
            if (write != a) {
                outColInd[out_start + write] = outColInd[out_start + a];
                outWeights[out_start + write] = outWeights[out_start + a];
            }
            write++;
        }
    }
    actualCount[i] = write;
}

// Sort comparator for entries (descending by order)
struct EntryGpu {
    int i, j;
    double order;
};

struct EntryGpuComp {
    __host__ __device__
    bool operator()(const EntryGpu& a, const EntryGpu& b) const {
        return a.order > b.order;
    }
};

// ============================================================
// Wrapper: DownsampleGraph on GPU
// ============================================================

extern "C" void cuda_downsample_graph(
    // Input fine level (CSR adjacency)
    const int* h_adjRowPtr, const int* h_adjColInd, const double* h_adjWeights,
    int nVerts, int nnz,
    // Input fine level geometry
    const double* h_V, const double* h_N, const double* h_A,
    // Output coarse level geometry
    double* h_V_p, double* h_N_p, double* h_A_p,
    int* h_to_upper,    // 2 * vertexCount_p (column-major)
    int* h_to_lower,    // nVerts
    // Output coarse level CSR adjacency
    int** h_adjRowPtr_p_out, int** h_adjColInd_p_out, double** h_adjWeights_p_out,
    int* vertexCount_p_out, int* nnz_p_out)
{
    // Upload fine level data to GPU
    int *d_adjRowPtr, *d_adjColInd;
    double *d_adjWeights, *d_V, *d_N, *d_A;

    cudaMalloc(&d_adjRowPtr, (nVerts + 1) * sizeof(int));
    cudaMalloc(&d_adjColInd, nnz * sizeof(int));
    cudaMalloc(&d_adjWeights, nnz * sizeof(double));
    cudaMalloc(&d_V, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_N, 3 * nVerts * sizeof(double));
    cudaMalloc(&d_A, nVerts * sizeof(double));

    cudaMemcpy(d_adjRowPtr, h_adjRowPtr, (nVerts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjColInd, h_adjColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjWeights, h_adjWeights, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, 3 * nVerts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, 3 * nVerts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, nVerts * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridVerts = (nVerts + blockSize - 1) / blockSize;

    auto _t0 = std::chrono::high_resolution_clock::now();
    // Step 1: Score entries on GPU
    int *d_entry_i, *d_entry_j;
    double *d_entry_order;
    cudaMalloc(&d_entry_i, nnz * sizeof(int));
    cudaMalloc(&d_entry_j, nnz * sizeof(int));
    cudaMalloc(&d_entry_order, nnz * sizeof(double));

    kernel_score_entries<<<gridVerts, blockSize>>>(
        d_N, d_A, d_adjRowPtr, d_adjColInd,
        d_entry_i, d_entry_j, d_entry_order, nVerts);

    // Download entries for sort + greedy merge on CPU
    // (Sort uses Thrust SoA approach, greedy merge is sequential)
    std::vector<int> h_entry_i(nnz), h_entry_j(nnz);
    std::vector<double> h_entry_order(nnz);
    cudaMemcpy(h_entry_i.data(), d_entry_i, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_entry_j.data(), d_entry_j, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_entry_order.data(), d_entry_order, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_entry_i); cudaFree(d_entry_j); cudaFree(d_entry_order);

    // Step 2: Sort entries (GPU via Thrust)
    std::vector<EntryGpu> entries(nnz);
    for (int i = 0; i < nnz; ++i) {
        entries[i].i = h_entry_i[i];
        entries[i].j = h_entry_j[i];
        entries[i].order = h_entry_order[i];
    }
    {
        thrust::device_vector<EntryGpu> d_entries(entries.begin(), entries.end());
        thrust::sort(d_entries.begin(), d_entries.end(), EntryGpuComp());
        thrust::copy(d_entries.begin(), d_entries.end(), entries.data());
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    // Step 3: Greedy merge (CPU, sequential)
    std::vector<bool> mergeFlag(nVerts, false);
    int nCollapsed = 0;
    std::vector<int> collapsed_i, collapsed_j;
    collapsed_i.reserve(nVerts / 2);
    collapsed_j.reserve(nVerts / 2);
    for (int i = 0; i < nnz; ++i) {
        int ei = entries[i].i, ej = entries[i].j;
        if (mergeFlag[ei] || mergeFlag[ej]) continue;
        mergeFlag[ei] = mergeFlag[ej] = true;
        collapsed_i.push_back(ei);
        collapsed_j.push_back(ej);
        nCollapsed++;
    }
    int vertexCount_p = nVerts - nCollapsed;
    *vertexCount_p_out = vertexCount_p;
    auto _t2 = std::chrono::high_resolution_clock::now();

    // Step 4: Build collapsed vertices + unmerged vertices on GPU
    int *d_collapsed_i, *d_collapsed_j;
    double *d_V_p, *d_N_p, *d_A_p;
    int *d_to_upper, *d_to_lower;

    cudaMalloc(&d_collapsed_i, nCollapsed * sizeof(int));
    cudaMalloc(&d_collapsed_j, nCollapsed * sizeof(int));
    cudaMalloc(&d_V_p, 3 * vertexCount_p * sizeof(double));
    cudaMalloc(&d_N_p, 3 * vertexCount_p * sizeof(double));
    cudaMalloc(&d_A_p, vertexCount_p * sizeof(double));
    cudaMalloc(&d_to_upper, 2 * vertexCount_p * sizeof(int));
    cudaMalloc(&d_to_lower, nVerts * sizeof(int));

    cudaMemcpy(d_collapsed_i, collapsed_i.data(), nCollapsed * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collapsed_j, collapsed_j.data(), nCollapsed * sizeof(int), cudaMemcpyHostToDevice);

    int gridCollapsed = (nCollapsed + blockSize - 1) / blockSize;
    kernel_build_collapsed<<<gridCollapsed, blockSize>>>(
        d_collapsed_i, d_collapsed_j, d_V, d_N, d_A,
        d_V_p, d_N_p, d_A_p, d_to_upper, d_to_lower,
        nCollapsed, vertexCount_p);

    // Unmerged vertices: prefix sum of !mergeFlag
    int *d_mergeFlag, *d_prefixSum, *d_notMerged;
    cudaMalloc(&d_mergeFlag, nVerts * sizeof(int));
    cudaMalloc(&d_notMerged, nVerts * sizeof(int));
    cudaMalloc(&d_prefixSum, nVerts * sizeof(int));

    // Convert bool mergeFlag to int array
    std::vector<int> mergeFlagInt(nVerts);
    std::vector<int> notMerged(nVerts);
    for (int i = 0; i < nVerts; ++i) {
        mergeFlagInt[i] = mergeFlag[i] ? 1 : 0;
        notMerged[i] = mergeFlag[i] ? 0 : 1;
    }
    cudaMemcpy(d_mergeFlag, mergeFlagInt.data(), nVerts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_notMerged, notMerged.data(), nVerts * sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_ptr<int> notMerged_ptr(d_notMerged);
    thrust::device_ptr<int> prefix_ptr(d_prefixSum);
    thrust::exclusive_scan(notMerged_ptr, notMerged_ptr + nVerts, prefix_ptr, 0);

    kernel_copy_unmerged<<<gridVerts, blockSize>>>(
        d_mergeFlag, d_prefixSum, d_V, d_N, d_A,
        d_V_p, d_N_p, d_A_p, d_to_upper, d_to_lower,
        nVerts, nCollapsed);

    // Download results
    cudaMemcpy(h_V_p, d_V_p, 3 * vertexCount_p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N_p, d_N_p, 3 * vertexCount_p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A_p, d_A_p, vertexCount_p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_to_upper, d_to_upper, 2 * vertexCount_p * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_to_lower, d_to_lower, nVerts * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 5: Build coarse adjacency on GPU
    int *d_counts;
    cudaMalloc(&d_counts, vertexCount_p * sizeof(int));

    int gridCoarse = (vertexCount_p + blockSize - 1) / blockSize;
    kernel_count_coarse_adj<<<gridCoarse, blockSize>>>(
        d_to_upper, d_adjRowPtr, d_adjColInd, d_to_lower, d_counts, vertexCount_p);

    // Prefix sum for output CSR
    int *d_outRowPtr;
    cudaMalloc(&d_outRowPtr, (vertexCount_p + 1) * sizeof(int));
    thrust::device_ptr<int> counts_ptr(d_counts);
    thrust::device_ptr<int> outRowPtr_ptr(d_outRowPtr);
    thrust::exclusive_scan(counts_ptr, counts_ptr + vertexCount_p, outRowPtr_ptr, 0);

    // Get total nnz (upper bound before dedup)
    int last_count_val, last_offset_val;
    cudaMemcpy(&last_count_val, d_counts + vertexCount_p - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_offset_val, d_outRowPtr + vertexCount_p - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int nnz_upper = last_offset_val + last_count_val;
    cudaMemcpy(d_outRowPtr + vertexCount_p, &nnz_upper, sizeof(int), cudaMemcpyHostToDevice);

    int *d_outColInd, *d_actualCount;
    double *d_outWeights;
    cudaMalloc(&d_outColInd, nnz_upper * sizeof(int));
    cudaMalloc(&d_outWeights, nnz_upper * sizeof(double));
    cudaMalloc(&d_actualCount, vertexCount_p * sizeof(int));

    kernel_fill_coarse_adj<<<gridCoarse, blockSize>>>(
        d_to_upper, d_adjRowPtr, d_adjColInd, d_adjWeights, d_to_lower,
        d_outRowPtr, d_outColInd, d_outWeights, d_actualCount, vertexCount_p);

    // Download coarse adjacency (with actual counts for compaction on host)
    std::vector<int> h_outRowPtr(vertexCount_p + 1);
    std::vector<int> h_outColInd(nnz_upper);
    std::vector<double> h_outWeights(nnz_upper);
    std::vector<int> h_actualCount(vertexCount_p);

    cudaMemcpy(h_outRowPtr.data(), d_outRowPtr, (vertexCount_p + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outColInd.data(), d_outColInd, nnz_upper * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outWeights.data(), d_outWeights, nnz_upper * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_actualCount.data(), d_actualCount, vertexCount_p * sizeof(int), cudaMemcpyDeviceToHost);

    // Compact CSR on host (remove dedup gaps)
    int actual_nnz = 0;
    for (int i = 0; i < vertexCount_p; ++i) actual_nnz += h_actualCount[i];

    *h_adjRowPtr_p_out = (int*)malloc((vertexCount_p + 1) * sizeof(int));
    *h_adjColInd_p_out = (int*)malloc(actual_nnz * sizeof(int));
    *h_adjWeights_p_out = (double*)malloc(actual_nnz * sizeof(double));
    *nnz_p_out = actual_nnz;

    int write_pos = 0;
    (*h_adjRowPtr_p_out)[0] = 0;
    for (int i = 0; i < vertexCount_p; ++i) {
        int src_start = h_outRowPtr[i];
        int cnt = h_actualCount[i];
        memcpy(*h_adjColInd_p_out + write_pos, h_outColInd.data() + src_start, cnt * sizeof(int));
        memcpy(*h_adjWeights_p_out + write_pos, h_outWeights.data() + src_start, cnt * sizeof(double));
        write_pos += cnt;
        (*h_adjRowPtr_p_out)[i + 1] = write_pos;
    }

    auto _t3 = std::chrono::high_resolution_clock::now();
    printf("[DSE-TIMING] score+sort=%.0f merge=%.0f build+adj=%.0f ms (nV=%d nCollapsed=%d)\n",
           std::chrono::duration<double, std::milli>(_t1 - _t0).count(),
           std::chrono::duration<double, std::milli>(_t2 - _t1).count(),
           std::chrono::duration<double, std::milli>(_t3 - _t2).count(),
           nVerts, nCollapsed);

    // Cleanup
    cudaFree(d_adjRowPtr); cudaFree(d_adjColInd); cudaFree(d_adjWeights);
    cudaFree(d_V); cudaFree(d_N); cudaFree(d_A);
    cudaFree(d_collapsed_i); cudaFree(d_collapsed_j);
    cudaFree(d_V_p); cudaFree(d_N_p); cudaFree(d_A_p);
    cudaFree(d_to_upper); cudaFree(d_to_lower);
    cudaFree(d_mergeFlag); cudaFree(d_notMerged); cudaFree(d_prefixSum);
    cudaFree(d_counts); cudaFree(d_outRowPtr); cudaFree(d_outColInd);
    cudaFree(d_outWeights); cudaFree(d_actualCount);
}
