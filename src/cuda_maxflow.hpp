#ifndef CUDA_MAXFLOW_HPP_
#define CUDA_MAXFLOW_HPP_

#include <vector>

namespace qflow {

struct CudaMaxFlowResult {
    int max_flow;
    std::vector<int> edge_flows;
};

CudaMaxFlowResult cuda_maxflow_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex,
    const int* h_nlist,
    const int* h_cap,
    int num_edges,
    const int* h_rnindex,
    const int* h_rnlist,
    const int* h_retoe
);

} // namespace qflow

#endif
