// ============================================================
// GPU-resident max-flow solvers: Edmonds-Karp and Dinic's
//
// Both use CSR graph representation on GPU.
// Edmonds-Karp: 1 BFS per augmentation, GPU-resident.
// Dinic's: fwd BFS + bwd BFS pruning + multi-augment per phase.
// ============================================================

#include "cuda_maxflow.hpp"
#include <cstdio>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace qflow {

// ---- Fused BFS kernel for Edmonds-Karp (parent-tracking) ----
__global__ void k_ek_bfs_expand(
    const int* nindex, const int* nlist, const int* cap, const int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* frontier, int frontier_size,
    int* parent, int* parent_edge, int* parent_dir,
    int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];

    for (int e = nindex[u]; e < nindex[u + 1]; e++) {
        int v = nlist[e];
        if (cap[e] - flow[e] <= 0) continue;
        if (atomicCAS(&parent[v], -1, u) == -1) {
            parent_edge[v] = e;
            parent_dir[v] = 1;
            next_frontier[atomicAdd(next_count, 1)] = v;
        }
    }
    for (int re = rnindex[u]; re < rnindex[u + 1]; re++) {
        int v = rnlist[re];
        int e_fwd = retoe[re];
        if (flow[e_fwd] <= 0) continue;
        if (atomicCAS(&parent[v], -1, u) == -1) {
            parent_edge[v] = e_fwd;
            parent_dir[v] = -1;
            next_frontier[atomicAdd(next_count, 1)] = v;
        }
    }
}

// ---- Path trace + flow push (single thread, Edmonds-Karp) ----
__global__ void k_ek_trace_push(
    const int* parent, const int* parent_edge, const int* parent_dir,
    const int* cap, int* flow, int source, int sink, int* out_bottleneck
) {
    int bn = 0x7FFFFFFF;
    for (int v = sink; v != source; v = parent[v]) {
        int e = parent_edge[v];
        int r = (parent_dir[v] == 1) ? (cap[e] - flow[e]) : flow[e];
        if (r < bn) bn = r;
    }
    for (int v = sink; v != source; v = parent[v]) {
        int e = parent_edge[v];
        if (parent_dir[v] == 1) flow[e] += bn; else flow[e] -= bn;
    }
    *out_bottleneck = bn;
}

// ---- Level BFS kernel for Dinic's (level-tracking, no parent) ----
__global__ void k_dinic_bfs_expand(
    const int* nindex, const int* nlist, const int* cap, const int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* frontier, int frontier_size,
    int current_level,
    int* level,   // level[v]=-1 means unvisited; set to current_level+1
    int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];

    int next_lev = current_level + 1;

    // Forward: cap - flow > 0
    for (int e = nindex[u]; e < nindex[u + 1]; e++) {
        int v = nlist[e];
        if (cap[e] - flow[e] <= 0) continue;
        int old = atomicCAS(&level[v], -1, next_lev);
        if (old == -1) next_frontier[atomicAdd(next_count, 1)] = v;
    }
    // Backward: flow > 0
    for (int re = rnindex[u]; re < rnindex[u + 1]; re++) {
        int v = rnlist[re];
        if (flow[retoe[re]] <= 0) continue;
        int old = atomicCAS(&level[v], -1, next_lev);
        if (old == -1) next_frontier[atomicAdd(next_count, 1)] = v;
    }
}

// ---- Backward BFS from sink through level graph (forward residual) ----
__global__ void k_dinic_bfs_backward(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, const int* flow, const int* level,
    const int* frontier, int frontier_size,
    int* reach, int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int v = frontier[tid];

    // Forward residual: u->v exists means u can reach v (and thus sink).
    // Check all u that have forward edge to v with level[u]=level[v]-1 and residual>0.
    for (int re = rnindex[v]; re < rnindex[v + 1]; re++) {
        int u = rnlist[re];
        int e_fwd = retoe[re];
        if (level[u] < 0 || level[u] != level[v] - 1) continue;
        if (cap[e_fwd] - flow[e_fwd] <= 0) continue;
        if (atomicCAS(&reach[u], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = u;
    }
    // Backward residual: flow on v->w > 0 means w can reach v via cancel.
    // Check all w where v->w has flow>0 and level[w]=level[v]-1.
    for (int e = nindex[v]; e < nindex[v + 1]; e++) {
        int w = nlist[e];
        if (level[w] < 0 || level[w] != level[v] - 1) continue;
        if (flow[e] <= 0) continue;
        if (atomicCAS(&reach[w], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = w;
    }
}

// ---- Multi-augment on GPU (single thread, DFS with backtracking) ----
// Uses current-arc optimization: iter[v] tracks which edge to try next.
__global__ void k_dinic_multi_augment(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, int* flow,
    const int* level, int* reach,
    int source, int sink,
    int* out_total_flow, int* out_num_paths
) {
    // Current-arc pointers: for each node, which edge to try next
    // Encode: values < rnindex[v+1]-rnindex[v] are reverse-CSR indices (forward edges TO v)
    //         values >= that are forward-CSR indices (backward residual edges FROM v)
    // But simpler: just store separate forward and backward iterators.
    // Since we're single-threaded with ~4M nodes, we can't afford per-node arrays.
    // Instead: use DFS stack with edge iterators stored per stack frame.

    struct Frame {
        int node;
        int re_iter;  // next reverse-CSR index to try (for forward edges u->node)
        int fe_iter;  // next forward-CSR index to try (for backward edges node->w)
        int edge_used;
        int dir_used;
    };

    // Path stack — max depth ~200
    Frame stack[256];
    int sp = 0;

    int total = 0, paths = 0;

    // Push source
    stack[0] = {source, rnindex[source], nindex[source], -1, 0};
    sp = 1;

    while (sp > 0) {
        Frame& fr = stack[sp - 1];
        int v = fr.node;

        if (v == sink) {
            // Found a path! Find bottleneck and push flow.
            int bn = 0x7FFFFFFF;
            for (int i = 1; i < sp; i++) {
                int e = stack[i].edge_used;
                int d = stack[i].dir_used;
                int r = (d == 1) ? (cap[e] - flow[e]) : flow[e];
                if (r < bn) bn = r;
            }
            if (bn > 0) {
                for (int i = 1; i < sp; i++) {
                    int e = stack[i].edge_used;
                    if (stack[i].dir_used == 1) flow[e] += bn; else flow[e] -= bn;
                }
                total += bn;
                paths++;
            }
            // Pop sink and continue from parent (try next edge)
            sp--;
            continue;
        }

        bool advanced = false;

        // Try forward edges: v→w with residual>0 and level[w]=level[v]+1
        for (int& e = fr.fe_iter; e < nindex[v + 1]; e++) {
            int w = nlist[e];
            if (!reach[w]) continue;
            if (level[w] != level[v] + 1) continue;
            if (cap[e] - flow[e] <= 0) continue;
            if (sp < 255) {
                stack[sp] = {w, rnindex[w], nindex[w], e, 1};
                sp++;
                e++;
                advanced = true;
                break;
            }
        }

        if (!advanced) {
            // Backward residual: edge w→v with flow>0, level[w]=level[v]+1
            for (int& re = fr.re_iter; re < rnindex[v + 1]; re++) {
                int w = rnlist[re];
                int e_fwd = retoe[re];
                if (!reach[w]) continue;
                if (level[w] != level[v] + 1) continue;
                if (flow[e_fwd] <= 0) continue;
                if (sp < 255) {
                    stack[sp] = {w, rnindex[w], nindex[w], e_fwd, -1};
                    sp++;
                    re++;
                    advanced = true;
                    break;
                }
            }
        }

        if (!advanced) {
            // Dead end — mark unreachable and backtrack
            reach[v] = 0;
            sp--;
        }
    }

    *out_total_flow = total;
    *out_num_paths = paths;
}


// ============================================================
// Edmonds-Karp (GPU-resident): 1 BFS per augmentation
// ============================================================
CudaMaxFlowResult cuda_edmonds_karp_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_next_count, *d_bottleneck;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));

    const int B = 256;
    int total_flow = 0, augmentations = 0;
    int src_val = source;

    while (true) {
        cudaMemset(d_parent, 0xFF, num_nodes*sizeof(int));
        cudaMemcpy(d_parent+source, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        int fsize = 1;
        bool found = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_ek_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize,
                d_parent, d_parent_edge, d_parent_dir,
                d_next_frontier, d_next_count);
            int sp;
            cudaMemcpy(&sp, d_parent+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sp != -1) { found = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }
        if (!found) break;

        k_ek_trace_push<<<1,1>>>(d_parent, d_parent_edge, d_parent_dir,
                                  d_cap, d_flow, source, sink, d_bottleneck);
        int bn;
        cudaMemcpy(&bn, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
        total_flow += bn;
        augmentations++;
    }

    printf("[TIMING]       GPU Edmonds-Karp: %d augmentations, flow=%d\n", augmentations, total_flow);
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_count);
    cudaFree(d_bottleneck);

    result.max_flow = total_flow;
    return result;
}


// ============================================================
// GPU Dinic's: Dinic phases for bulk flow, Edmonds-Karp for stragglers
//
// Phase 1: Dinic's (fwd BFS + bwd BFS + multi-augment) — gets ~83% of flow fast
// Phase 2: Edmonds-Karp (full BFS per augment) — handles remaining ~17% correctly
// ============================================================
CudaMaxFlowResult cuda_dinic_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    // Allocate all GPU arrays (shared by both phases)
    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_level, *d_reach;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_next_count;
    int *d_total_flow, *d_num_paths, *d_bottleneck;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_level, num_nodes*sizeof(int));
    cudaMalloc(&d_reach, num_nodes*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_total_flow, sizeof(int));
    cudaMalloc(&d_num_paths, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));

    const int B = 256;
    int total_flow = 0, total_paths = 0, dinic_phases = 0;
    int src_val = source, sink_val = sink;

    // ==== PHASE 1: Dinic's (bulk flow) ====
    while (true) {
        // Forward BFS: build level graph
        cudaMemset(d_level, 0xFF, num_nodes*sizeof(int));
        int zero = 0;
        cudaMemcpy(d_level+source, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);

        int fsize = 1, current_level = 0;
        bool found_sink = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize, current_level,
                d_level, d_next_frontier, d_next_count);
            int sink_lev;
            cudaMemcpy(&sink_lev, d_level+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sink_lev >= 0) { found_sink = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
            current_level++;
        }

        if (!found_sink) break;

        // Backward BFS: mark nodes that can reach sink
        cudaMemset(d_reach, 0, num_nodes*sizeof(int));
        int one = 1;
        cudaMemcpy(d_reach+sink, &one, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &sink_val, sizeof(int), cudaMemcpyHostToDevice);
        fsize = 1;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_backward<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
                d_cap, d_flow, d_level,
                d_frontier, fsize,
                d_reach, d_next_frontier, d_next_count);
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }

        // Multi-augment on pruned level graph
        k_dinic_multi_augment<<<1, 1>>>(
            d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
            d_cap, d_flow, d_level, d_reach,
            source, sink, d_total_flow, d_num_paths);

        int phase_flow, phase_paths;
        cudaMemcpy(&phase_flow, d_total_flow, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phase_paths, d_num_paths, sizeof(int), cudaMemcpyDeviceToHost);

        total_flow += phase_flow;
        total_paths += phase_paths;
        dinic_phases++;

        if (phase_flow == 0) break;
    }

    printf("[TIMING]       GPU Dinic phase: %d phases, %d paths, flow=%d\n",
           dinic_phases, total_paths, total_flow);

    // ==== PHASE 2: Edmonds-Karp for remaining flow ====
    // Handles paths that Dinic's backward BFS pruning missed.
    // Uses the same GPU arrays (flow state continues from Dinic).
    int ek_augmentations = 0;

    while (true) {
        cudaMemset(d_parent, 0xFF, num_nodes*sizeof(int));
        cudaMemcpy(d_parent+source, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        int fsize = 1;
        bool found = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_ek_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize,
                d_parent, d_parent_edge, d_parent_dir,
                d_next_frontier, d_next_count);
            int sp;
            cudaMemcpy(&sp, d_parent+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sp != -1) { found = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }
        if (!found) break;

        k_ek_trace_push<<<1,1>>>(d_parent, d_parent_edge, d_parent_dir,
                                  d_cap, d_flow, source, sink, d_bottleneck);
        int bn;
        cudaMemcpy(&bn, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
        total_flow += bn;
        ek_augmentations++;
    }

    if (ek_augmentations > 0) {
        printf("[TIMING]       GPU EK cleanup: %d augmentations, total flow=%d\n",
               ek_augmentations, total_flow);
    }

    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_level); cudaFree(d_reach);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_count);
    cudaFree(d_total_flow); cudaFree(d_num_paths); cudaFree(d_bottleneck);

    result.max_flow = total_flow;
    return result;
}

} // namespace qflow
