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
#include <queue>
#include <algorithm>
#include <chrono>
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

// ---- Backward BFS on full residual graph (no level check) ----
// Used to find all nodes that can reach sink in the residual graph.
__global__ void k_bfs_backward_residual(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, const int* flow,
    const int* frontier, int frontier_size,
    int* reach, int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int v = frontier[tid];

    // Forward residual: edge u->v with cap-flow>0 means u can reach v→sink
    for (int re = rnindex[v]; re < rnindex[v + 1]; re++) {
        int u = rnlist[re];
        int e_fwd = retoe[re];
        if (cap[e_fwd] - flow[e_fwd] <= 0) continue;
        if (atomicCAS(&reach[u], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = u;
    }
    // Backward residual: edge v->w with flow>0 means w can reach v→sink
    for (int e = nindex[v]; e < nindex[v + 1]; e++) {
        int w = nlist[e];
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

    // Download flow after Dinic phase
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    // (debug save removed — use -save-all for checkpoints instead)

    // ==== PHASE 2: Compact subgraph EK ====
    // Two BFS identify the active subgraph (reachable from source AND reaching sink).
    // Build compact CSR, run CPU EK on it. Much faster than full-graph EK.
    {
        auto t_cleanup_start = std::chrono::high_resolution_clock::now();

        // Step 1: Forward BFS from source on residual (reuse level array)
        // level[v] >= 0 means reachable from source (already set by last Dinic BFS)
        // But Dinic may have modified flow since last BFS. Re-do a full forward BFS.
        cudaMemset(d_level, 0xFF, num_nodes * sizeof(int));
        int zero = 0;
        cudaMemcpy(d_level + source, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        int fsize = 1, cur_lev = 0;
        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize, cur_lev,
                d_level, d_next_frontier, d_next_count);
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
            cur_lev++;
        }

        // Step 2: Backward BFS from sink on full residual (no level check)
        cudaMemset(d_reach, 0, num_nodes * sizeof(int));
        int one = 1;
        cudaMemcpy(d_reach + sink, &one, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &sink_val, sizeof(int), cudaMemcpyHostToDevice);
        fsize = 1;
        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_bfs_backward_residual<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
                d_cap, d_flow,
                d_frontier, fsize,
                d_reach, d_next_frontier, d_next_count);
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }

        // Download level and reach arrays
        std::vector<int> h_level(num_nodes), h_reach(num_nodes);
        cudaMemcpy(h_level.data(), d_level, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_reach.data(), d_reach, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // Also need flow on host for compact EK
        // (already downloaded into result.edge_flows above)

        // Step 3: Build compact subgraph
        // Active nodes: level[v] >= 0 AND reach[v] == 1
        std::vector<int> node_map(num_nodes, -1);  // original → compact
        int K = 0;
        for (int v = 0; v < num_nodes; v++) {
            if (h_level[v] >= 0 && h_reach[v]) {
                node_map[v] = K++;
            }
        }
        // Always include source and sink
        if (node_map[source] == -1) node_map[source] = K++;
        if (node_map[sink] == -1) node_map[sink] = K++;

        int compact_src = node_map[source];
        int compact_sink = node_map[sink];

        // Build compact CSR + reverse CSR
        // First pass: count edges per compact node
        std::vector<int> c_nindex(K + 1, 0);
        int* h_flow = result.edge_flows.data();
        for (int u = 0; u < num_nodes; u++) {
            if (node_map[u] == -1) continue;
            int cu = node_map[u];
            for (int e = h_nindex[u]; e < h_nindex[u + 1]; e++) {
                int v = h_nlist[e];
                if (node_map[v] == -1) continue;
                if (h_cap[e] - h_flow[e] <= 0) continue;
                c_nindex[cu + 1]++;
            }
            // Backward residual edges: for each edge w->u with flow>0
            for (int re = h_rnindex[u]; re < h_rnindex[u + 1]; re++) {
                int w = h_rnlist[re];
                if (node_map[w] == -1) continue;
                int ef = h_retoe[re];
                if (h_flow[ef] <= 0) continue;
                c_nindex[cu + 1]++;
            }
        }
        for (int i = 1; i <= K; i++) c_nindex[i] += c_nindex[i - 1];
        int c_num_edges = c_nindex[K];

        std::vector<int> c_nlist(c_num_edges), c_cap(c_num_edges);
        std::vector<int> c_orig_edge(c_num_edges);  // map compact edge → original edge
        std::vector<int> c_dir(c_num_edges);         // +1 forward, -1 backward
        std::vector<int> c_offset(K, 0);

        for (int u = 0; u < num_nodes; u++) {
            if (node_map[u] == -1) continue;
            int cu = node_map[u];
            // Forward edges
            for (int e = h_nindex[u]; e < h_nindex[u + 1]; e++) {
                int v = h_nlist[e];
                if (node_map[v] == -1) continue;
                int res = h_cap[e] - h_flow[e];
                if (res <= 0) continue;
                int pos = c_nindex[cu] + c_offset[cu]++;
                c_nlist[pos] = node_map[v];
                c_cap[pos] = res;
                c_orig_edge[pos] = e;
                c_dir[pos] = 1;
            }
            // Backward residual edges
            for (int re = h_rnindex[u]; re < h_rnindex[u + 1]; re++) {
                int w = h_rnlist[re];
                if (node_map[w] == -1) continue;
                int ef = h_retoe[re];
                if (h_flow[ef] <= 0) continue;
                int pos = c_nindex[cu] + c_offset[cu]++;
                c_nlist[pos] = node_map[w];
                c_cap[pos] = h_flow[ef];
                c_orig_edge[pos] = ef;
                c_dir[pos] = -1;
            }
        }

        printf("[TIMING]       Compact subgraph: %d nodes, %d edges (from %d/%d)\n",
               K, c_num_edges, num_nodes, num_edges);

        // Step 4: CPU Edmonds-Karp on compact graph
        std::vector<int> c_flow(c_num_edges, 0);
        std::vector<int> c_par(K), c_par_e(K), c_par_d(K);
        std::vector<int> c_bfs_q;
        c_bfs_q.reserve(K);
        int ek_augs = 0, ek_flow = 0;

        while (true) {
            std::fill(c_par.begin(), c_par.end(), -1);
            c_par[compact_src] = compact_src;
            c_bfs_q.clear();
            c_bfs_q.push_back(compact_src);

            bool found = false;
            for (int qi = 0; qi < (int)c_bfs_q.size() && !found; qi++) {
                int u = c_bfs_q[qi];
                for (int e = c_nindex[u]; e < c_nindex[u + 1]; e++) {
                    int v = c_nlist[e];
                    if (c_par[v] != -1) continue;
                    if (c_cap[e] - c_flow[e] <= 0) continue;
                    c_par[v] = u; c_par_e[v] = e; c_par_d[v] = 1;
                    c_bfs_q.push_back(v);
                    if (v == compact_sink) { found = true; break; }
                }
            }
            if (!found) break;

            int bn = INT_MAX;
            for (int v = compact_sink; v != compact_src; v = c_par[v]) {
                int r = c_cap[c_par_e[v]] - c_flow[c_par_e[v]];
                if (r < bn) bn = r;
            }
            for (int v = compact_sink; v != compact_src; v = c_par[v]) {
                c_flow[c_par_e[v]] += bn;
            }
            ek_augs++;
            ek_flow += bn;
        }

        // Step 5: Map compact flow back to original edges
        for (int e = 0; e < c_num_edges; e++) {
            if (c_flow[e] <= 0) continue;
            int orig_e = c_orig_edge[e];
            int dir = c_dir[e];
            if (dir == 1)
                result.edge_flows[orig_e] += c_flow[e];
            else
                result.edge_flows[orig_e] -= c_flow[e];
        }
        total_flow += ek_flow;

        auto t_cleanup_end = std::chrono::high_resolution_clock::now();
        double cleanup_s = std::chrono::duration<double>(t_cleanup_end - t_cleanup_start).count();
        printf("[TIMING]       Compact EK cleanup: %d augmentations, flow=%d, total flow=%d (%.3f s)\n",
               ek_augs, ek_flow, total_flow, cleanup_s);
    }

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
