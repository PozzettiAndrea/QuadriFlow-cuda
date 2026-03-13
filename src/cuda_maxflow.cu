/*
 * CUDA Max-Flow integration for QuadriFlow
 * Based on ECL-MaxFlow by Avery VanAusdal and Martin Burtscher (BSD license)
 * https://github.com/burtscher/ECL-MaxFlow
 *
 * Adapted to work as a library call within QuadriFlow's integer constraint solver.
 */

#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits.h>
#include <numeric>

typedef int excess_t;

static const int ThreadsPerBlock = 512;
static const int WarpSize = 32;
static const int maxval = INT_MAX;

template <typename T>
static __device__ inline T atomicRead(T* const addr)
{
  return ((cuda::atomic<T, cuda::thread_scope_device>*)addr)->load(cuda::memory_order_relaxed);
}

template <typename T>
static __device__ inline void atomicWrite(T* const addr, const T val)
{
  ((cuda::atomic<T, cuda::thread_scope_device>*)addr)->store(val, cuda::memory_order_relaxed);
}

// ---- Kernels (from ECL-MaxFlow) ----

struct CSRGraph {
  int nodes;
  int edges;
  int* nindex;
  int* nlist;
};

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_init(CSRGraph g, const int source, const int sink, int* const flow, excess_t* const excess, int* const time, int* const height)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < g.edges) flow[idx] = 0;
  if (idx < g.nodes) {
    excess[idx] = 0;
    time[idx] = 0;
    height[idx] = (idx == sink) ? 0 : g.nodes;
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_initialPush(CSRGraph g, const int source, const int sink, excess_t* const excess, int* const flow, const int* const cap,
                   int* const wl, int* const wlsize)
{
  const int e = g.nindex[source] + threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (e < g.nindex[source + 1]) {
    const int capacity = cap[e];
    const int dst = g.nlist[e];
    flow[e] = capacity;
    atomicAdd(&excess[source], -capacity);
    atomicAdd(&excess[dst], capacity);
    if (dst != sink) {
      wl[atomicAdd(wlsize, 1)] = dst;
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_reverseResidualBFS(CSRGraph g, const int* const flow, const int* const cap,
                          const int* const rnindex, const int* const rnlist, const int* const retoe,
                          const int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                          int* const time, const int iter,
                          const int source, const int sink, int* const height, const int curr_h)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  const int lane = threadIdx.x % WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    const int rbeg = rnindex[v];
    const int rend = rnindex[v + 1];
    for (int re = rbeg + lane; re < rend; re += WarpSize) {
      const int e = retoe[re];
      const int n = rnlist[re];
      if (n != source && n != sink && flow[e] < cap[e]) {
        if (atomicMin(&time[n], iter) != iter) {
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    for (int e = beg + lane; e < end; e += WarpSize) {
      const int n = g.nlist[e];
      if (n != source && n != sink && flow[e] > 0) {
        if (atomicMin(&time[n], iter) != iter) {
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_phase2_reverseResidualBFS(CSRGraph g, const int* const flow, const int* const cap,
                                 const int* const rnindex, const int* const rnlist, const int* const retoe,
                                 const int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                                 int* const time, const int iter,
                                 const int source, const int sink, int* const height, const int curr_h)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  const int lane = threadIdx.x % WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    const int beg = g.nindex[v];
    const int end = g.nindex[v + 1];
    for (int e = beg + lane; e < end; e += WarpSize) {
      const int n = g.nlist[e];
      if (n != source && n != sink && flow[e] > 0) {
        if (atomicMin(&time[n], iter) != iter) {
          height[n] = curr_h;
          wl2[atomicAdd(wl2size, 1)] = n;
        }
      }
    }
  }
}

static __global__ void k_phase1_pushRelabel(CSRGraph g, const int source, const int sink,
                        int* const height, excess_t* const excess,
                        int* const flow, const int* const cap,
                        const int* const rnindex, const int* const rnlist, const int* const retoe,
                        int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                        int* const time, const int iter)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  const int lane = threadIdx.x % WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    const int v_height = atomicRead(&height[v]);
    if (v_height < g.nodes && atomicRead(&excess[v]) > 0) {
      int min_height = INT_MAX;
      int min_dst = 0;
      int min_e = 0;
      bool minIsForward = true;
      const int beg = g.nindex[v];
      const int end = g.nindex[v + 1];
      for (int e = beg + lane; e < end; e += WarpSize) {
        const int dst = g.nlist[e];
        if (atomicRead(&flow[e]) < cap[e]) {
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
          }
        }
      }
      const int rbeg = rnindex[v];
      const int rend = rnindex[v + 1];
      for (int re = rbeg + lane; re < rend; re += WarpSize) {
        const int dst = rnlist[re];
        const int e = retoe[re];
        if (atomicRead(&flow[e]) > 0) {
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
            minIsForward = false;
          }
        }
      }
      for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        int new_h = __shfl_down_sync(~0, min_height, offset);
        int new_dst = __shfl_down_sync(~0, min_dst, offset);
        int new_e = __shfl_down_sync(~0, min_e, offset);
        bool new_isF = __shfl_down_sync(~0, minIsForward, offset);
        if (new_h < min_height) {
          min_height = new_h;
          min_dst = new_dst;
          min_e = new_e;
          minIsForward = new_isF;
        }
      }
      if (lane == 0) {
        if (min_height < v_height) {
          const int dst = min_dst;
          const int e = min_e;
          excess_t delta;
          if (minIsForward) {
            delta = min(atomicRead(&excess[v]), cap[e] - atomicRead(&flow[e]));
            atomicAdd(&flow[e], delta);
          } else {
            delta = min(atomicRead(&excess[v]), atomicRead(&flow[e]));
            atomicAdd(&flow[e], -delta);
          }
          const int old_excess = atomicAdd(&excess[v], -delta);
          if (old_excess > delta && atomicMax(&time[v], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = v;
          }
          atomicAdd(&excess[dst], delta);
          if (dst != source && dst != sink && atomicMax(&time[dst], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = dst;
          }
        } else {
          if (min_height != maxval) {
            atomicWrite(&height[v], min_height + 1);
            if (((min_height + 1) < g.nodes) && atomicMax(&time[v], iter) != iter) {
              wl2[atomicAdd(wl2size, 1)] = v;
            }
          } else {
            atomicWrite(&height[v], g.nodes);
          }
        }
      }
    }
  }
}

static __global__ void k_phase2_pushRelabel(CSRGraph g, const int source, const int sink,
                        int* const height, excess_t* const excess,
                        int* const flow, const int* const cap,
                        const int* const rnindex, const int* const rnlist, const int* const retoe,
                        int* const wl1, const int wl1size, int* const wl2, int* const wl2size,
                        int* const time, const int iter)
{
  const int idx = (threadIdx.x + blockIdx.x * ThreadsPerBlock) / WarpSize;
  const int lane = threadIdx.x % WarpSize;
  if (idx < wl1size) {
    const int v = wl1[idx];
    if (atomicRead(&excess[v]) > 0) {
      int min_height = INT_MAX;
      int min_dst = 0;
      int min_e = 0;
      const int rbeg = rnindex[v];
      const int rend = rnindex[v + 1];
      for (int re = rbeg + lane; re < rend; re += WarpSize) {
        const int dst = rnlist[re];
        const int e = retoe[re];
        if (atomicRead(&flow[e]) > 0) {
          const int dst_height = atomicRead(&height[dst]);
          if (dst_height < min_height) {
            min_height = dst_height;
            min_dst = dst;
            min_e = e;
          }
        }
      }
      for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        int new_h = __shfl_down_sync(~0, min_height, offset);
        int new_dst = __shfl_down_sync(~0, min_dst, offset);
        int new_e = __shfl_down_sync(~0, min_e, offset);
        if (new_h < min_height) {
          min_height = new_h;
          min_dst = new_dst;
          min_e = new_e;
        }
      }
      if (lane == 0) {
        if (min_height < atomicRead(&height[v])) {
          const int dst = min_dst;
          const int e = min_e;
          excess_t delta = min(atomicRead(&excess[v]), atomicRead(&flow[e]));
          atomicAdd(&flow[e], -delta);
          const int old_excess = atomicAdd(&excess[v], -delta);
          if (old_excess > delta && atomicMax(&time[v], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = v;
          }
          atomicAdd(&excess[dst], delta);
          if (dst != source && dst != sink && atomicMax(&time[dst], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = dst;
          }
        } else {
          if (min_height != maxval) {
            atomicWrite(&height[v], min_height + 1);
          }
          if (atomicMax(&time[v], iter) != iter) {
            wl2[atomicAdd(wl2size, 1)] = v;
          }
        }
      }
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_phase1_initGR(CSRGraph g, const int source, const int sink, int* const height, int* const wl)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < g.nodes) {
    height[idx] = (idx == sink) ? 0 : g.nodes;
    if (idx == 0) wl[0] = sink;
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_phase2_initGR(CSRGraph g, const int source, const int sink, int* const height, int* const wl)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < g.nodes) {
    height[idx] = (idx == source) ? 0 : INT_MAX;
    if (idx == 0) wl[0] = source;
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock)
void k_phase2_buildWL(CSRGraph g, const int source, const int sink, int* const excess, int* const wl, int* const wlsize)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < g.nodes && v != source && v != sink) {
    if (excess[v] > 0) {
      wl[atomicAdd(wlsize, 1)] = v;
    }
  }
}

// ---- Host-side solver ----

namespace qflow {

struct CudaMaxFlowResult {
    int max_flow;
    std::vector<int> edge_flows;  // flow[e] for each edge in CSR order
};

CudaMaxFlowResult cuda_maxflow_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex,   // CSR offsets [num_nodes+1]
    const int* h_nlist,    // CSR adjacency [num_edges]
    const int* h_cap,      // capacity per edge [num_edges]
    int num_edges,
    const int* h_rnindex,  // reverse CSR offsets
    const int* h_rnlist,   // reverse CSR adjacency
    const int* h_retoe     // reverse edge to original edge mapping
)
{
    CudaMaxFlowResult result;

    // Device graph struct
    CSRGraph d_g;
    d_g.nodes = num_nodes;
    d_g.edges = num_edges;
    cudaMalloc(&d_g.nindex, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_g.nlist, num_edges * sizeof(int));
    cudaMemcpy(d_g.nindex, h_nindex, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.nlist, h_nlist, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int* d_cap;
    cudaMalloc(&d_cap, num_edges * sizeof(int));
    cudaMemcpy(d_cap, h_cap, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int* d_rnindex;
    cudaMalloc(&d_rnindex, (num_nodes + 1) * sizeof(int));
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int* d_rnlist;
    cudaMalloc(&d_rnlist, num_edges * sizeof(int));
    cudaMemcpy(d_rnlist, h_rnlist, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int* d_retoe;
    cudaMalloc(&d_retoe, num_edges * sizeof(int));
    cudaMemcpy(d_retoe, h_retoe, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate solver state
    int* d_flow;
    cudaMalloc(&d_flow, num_edges * sizeof(int));

    excess_t* d_excess;
    cudaMalloc(&d_excess, num_nodes * sizeof(excess_t));

    int* d_height;
    cudaMalloc(&d_height, num_nodes * sizeof(int));

    int* d_wl1;
    cudaMalloc(&d_wl1, num_nodes * sizeof(int));

    int* d_wl2;
    cudaMalloc(&d_wl2, num_nodes * sizeof(int));

    int* d_wl2size;
    cudaMalloc(&d_wl2size, sizeof(int));

    int* d_wl3;
    cudaMalloc(&d_wl3, num_nodes * sizeof(int));

    int* d_time;
    cudaMalloc(&d_time, num_nodes * sizeof(int));

    // Initialize
    int blocks = (std::max(num_nodes, num_edges) + ThreadsPerBlock - 1) / ThreadsPerBlock;
    k_init<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_flow, d_excess, d_time, d_height);
    cudaMemset(d_wl2size, 0, sizeof(int));

    // GR frequency
    const double avgDeg = (double)num_edges / num_nodes;
    const int GR_frequency = std::max(100, (int)((num_nodes / 1000) / avgDeg));

    // Initial push from source
    int src_degree = h_nindex[source + 1] - h_nindex[source];
    blocks = (src_degree + ThreadsPerBlock - 1) / ThreadsPerBlock;
    k_initialPush<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_excess, d_flow, d_cap, d_wl1, d_wl2size);
    int wl1size = 0;
    cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

    const int node_blocks = (num_nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // Initial BFS from sink for heights
    cudaMemcpy(d_wl3, &sink, sizeof(int), cudaMemcpyHostToDevice);
    int wl3size = 1;
    int bfs_iter_count = 0;
    {
        int bfs_iter = 0;
        do {
            bfs_iter++;
            bfs_iter_count++;
            cudaMemset(d_wl2size, 0, sizeof(int));
            blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
            k_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe,
                d_wl3, wl3size, d_wl2, d_wl2size, d_time, -1, source, sink, d_height, bfs_iter);
            cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_wl3, d_wl2);
        } while (wl3size > 0);
    }
    int gr_iters = bfs_iter_count;

    // Phase 1: push flow to sink
    int iter = 0;
    while (wl1size > 0) {
        iter++;
        cudaMemset(d_wl2size, 0, sizeof(int));
        blocks = ((long)wl1size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
        k_phase1_pushRelabel<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_excess, d_flow, d_cap,
            d_rnindex, d_rnlist, d_retoe, d_wl1, wl1size, d_wl2, d_wl2size, d_time, iter);
        cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_wl1, d_wl2);
        cudaDeviceSynchronize();

        if (wl1size == 0) break;

        // Global relabel
        if ((iter % GR_frequency) == 0) {
            k_phase1_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
            wl3size = 1;
            int bfs_iter = 0;
            do {
                bfs_iter++;
                bfs_iter_count++;
                cudaMemset(d_wl2size, 0, sizeof(int));
                blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
                k_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe,
                    d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter);
                cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(d_wl3, d_wl2);
            } while (wl3size > 0);
        }
    }

    const int phase1_pr_iters = iter;
    const int phase1_bfs_iters = bfs_iter_count;

    // Phase 2: return excess to source
    cudaMemset(d_wl2size, 0, sizeof(int));
    k_phase2_buildWL<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_excess, d_wl1, d_wl2size);
    cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

    if (wl1size > 0) {
        // BFS from source for Phase 2 heights
        k_phase2_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
        wl3size = 1;
        {
            int bfs_iter = 0;
            do {
                bfs_iter++;
                bfs_iter_count++;
                cudaMemset(d_wl2size, 0, sizeof(int));
                blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
                k_phase2_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe,
                    d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter);
                cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(d_wl3, d_wl2);
            } while (wl3size > 0);
        }

        // Phase 2 push-relabel
        while (wl1size > 0) {
            iter++;
            cudaMemset(d_wl2size, 0, sizeof(int));
            blocks = ((long)wl1size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
            k_phase2_pushRelabel<<<blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_excess, d_flow, d_cap,
                d_rnindex, d_rnlist, d_retoe, d_wl1, wl1size, d_wl2, d_wl2size, d_time, iter);
            cudaMemcpy(&wl1size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_wl1, d_wl2);
            cudaDeviceSynchronize();

            if (wl1size == 0) break;

            if (((iter - phase1_pr_iters) % GR_frequency) == 0) {
                k_phase2_initGR<<<node_blocks, ThreadsPerBlock>>>(d_g, source, sink, d_height, d_wl3);
                wl3size = 1;
                int bfs_iter = 0;
                do {
                    bfs_iter++;
                    bfs_iter_count++;
                    cudaMemset(d_wl2size, 0, sizeof(int));
                    blocks = ((long)wl3size * WarpSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
                    k_phase2_reverseResidualBFS<<<blocks, ThreadsPerBlock>>>(d_g, d_flow, d_cap, d_rnindex, d_rnlist, d_retoe,
                        d_wl3, wl3size, d_wl2, d_wl2size, d_time, -iter, source, sink, d_height, bfs_iter);
                    cudaMemcpy(&wl3size, d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);
                    std::swap(d_wl3, d_wl2);
                } while (wl3size > 0);
            }
        }
    }

    // Read result
    excess_t max_flow_val = 0;
    cudaMemcpy(&max_flow_val, d_excess + sink, sizeof(excess_t), cudaMemcpyDeviceToHost);
    result.max_flow = max_flow_val;

    // Copy flow values back
    result.edge_flows.resize(num_edges);
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_g.nindex);
    cudaFree(d_g.nlist);
    cudaFree(d_cap);
    cudaFree(d_rnindex);
    cudaFree(d_rnlist);
    cudaFree(d_retoe);
    cudaFree(d_flow);
    cudaFree(d_excess);
    cudaFree(d_height);
    cudaFree(d_wl1);
    cudaFree(d_wl2);
    cudaFree(d_wl2size);
    cudaFree(d_wl3);
    cudaFree(d_time);

    return result;
}

} // namespace qflow
