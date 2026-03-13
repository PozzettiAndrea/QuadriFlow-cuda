#ifndef FLOW_H_
#define FLOW_H_

#include <Eigen/Core>
#include <list>
#include <map>
#include <vector>

#include "config.hpp"

#ifdef WITH_CUDA
#include "cuda_maxflow.hpp"
#endif

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

#include <lemon/network_simplex.h>
#include <lemon/preflow.h>
#include <lemon/smart_graph.h>

using namespace boost;
using namespace Eigen;

namespace qflow {

class MaxFlowHelper {
   public:
    MaxFlowHelper() {}
    virtual ~MaxFlowHelper(){};
    virtual void resize(int n, int m) = 0;
    virtual void addEdge(int x, int y, int c, int rc, int v, int cost = 1) = 0;
    virtual int compute() = 0;
    virtual void applyTo(std::vector<Vector2i>& edge_diff) = 0;
};

class PushRelabelMaxFlowHelper : public MaxFlowHelper {
   public:
    typedef int EdgeWeightType;
    typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
    typedef adjacency_list < vecS, vecS, directedS,
        property < vertex_name_t, std::string,
        property < vertex_index_t, long,
        property < vertex_color_t, boost::default_color_type,
        property < vertex_distance_t, long,
        property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,
        property < edge_capacity_t, EdgeWeightType,
        property < edge_residual_capacity_t, EdgeWeightType,
        property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

   public:
    PushRelabelMaxFlowHelper() { rev = get(edge_reverse, g); }
    void resize(int n, int m) {
        vertex_descriptors.resize(n);
        for (int i = 0; i < n; ++i) vertex_descriptors[i] = add_vertex(g);
    }
    int compute() {
        EdgeWeightType flow =
            push_relabel_max_flow(g, vertex_descriptors.front(), vertex_descriptors.back());
        return flow;
    }
    void addEdge(int x, int y, int c, int rc, int v, int cost = 1) {
        Traits::edge_descriptor e1, e2;
        e1 = add_edge(vertex_descriptors[x], vertex_descriptors[y], g).first;
        e2 = add_edge(vertex_descriptors[y], vertex_descriptors[x], g).first;
        put(edge_capacity, g, e1, c);
        put(edge_capacity, g, e2, rc);
        rev[e1] = e2;
        rev[e2] = e1;
        if (v != -1) {
            edge_to_variables[e1] = std::make_pair(v, -1);
            edge_to_variables[e2] = std::make_pair(v, 1);
        }
    }
    void applyTo(std::vector<Vector2i>& edge_diff) {
        property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
        property_map<Graph, edge_residual_capacity_t>::type residual_capacity =
            get(edge_residual_capacity, g);
        graph_traits<Graph>::vertex_iterator u_iter, u_end;
        graph_traits<Graph>::out_edge_iterator ei, e_end;
        for (tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
            for (tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
                if (capacity[*ei] > 0) {
                    int flow = (capacity[*ei] - residual_capacity[*ei]);
                    if (flow > 0) {
                        auto it = edge_to_variables.find(*ei);
                        if (it != edge_to_variables.end()) {
                            edge_diff[it->second.first / 2][it->second.first % 2] +=
                                it->second.second * flow;
                        }
                    }
                }
    }

   private:
    Graph g;
    property_map<Graph, edge_reverse_t>::type rev;
    std::vector<Traits::vertex_descriptor> vertex_descriptors;
    std::map<Traits::edge_descriptor, std::pair<int, int>> edge_to_variables;
};

class BoykovMaxFlowHelper : public MaxFlowHelper {
   public:
    typedef int EdgeWeightType;
    typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
    // clang-format off
    typedef adjacency_list < vecS, vecS, directedS,
        property < vertex_name_t, std::string,
        property < vertex_index_t, long,
        property < vertex_color_t, boost::default_color_type,
        property < vertex_distance_t, long,
        property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,

        property < edge_capacity_t, EdgeWeightType,
        property < edge_residual_capacity_t, EdgeWeightType,
        property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;
    // clang-format on

   public:
    BoykovMaxFlowHelper() { rev = get(edge_reverse, g); }
    void resize(int n, int m) {
        vertex_descriptors.resize(n);
        for (int i = 0; i < n; ++i) vertex_descriptors[i] = add_vertex(g);
    }
    int compute() {
        EdgeWeightType flow =
            boykov_kolmogorov_max_flow(g, vertex_descriptors.front(), vertex_descriptors.back());
        return flow;
    }
    void addDirectEdge(Traits::vertex_descriptor& v1, Traits::vertex_descriptor& v2,
                       property_map<Graph, edge_reverse_t>::type& rev, const int capacity,
                       const int inv_capacity, Graph& g, Traits::edge_descriptor& e1,
                       Traits::edge_descriptor& e2) {
        e1 = add_edge(v1, v2, g).first;
        e2 = add_edge(v2, v1, g).first;
        put(edge_capacity, g, e1, capacity);
        put(edge_capacity, g, e2, inv_capacity);

        rev[e1] = e2;
        rev[e2] = e1;
    }
    void addEdge(int x, int y, int c, int rc, int v, int cost = 1) {
        Traits::edge_descriptor e1, e2;
        addDirectEdge(vertex_descriptors[x], vertex_descriptors[y], rev, c, rc, g, e1, e2);
        if (v != -1) {
            edge_to_variables[e1] = std::make_pair(v, -1);
            edge_to_variables[e2] = std::make_pair(v, 1);
        }
    }
    void applyTo(std::vector<Vector2i>& edge_diff) {
        property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
        property_map<Graph, edge_residual_capacity_t>::type residual_capacity =
            get(edge_residual_capacity, g);

        graph_traits<Graph>::vertex_iterator u_iter, u_end;
        graph_traits<Graph>::out_edge_iterator ei, e_end;
        for (tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
            for (tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
                if (capacity[*ei] > 0) {
                    int flow = (capacity[*ei] - residual_capacity[*ei]);
                    if (flow > 0) {
                        auto it = edge_to_variables.find(*ei);
                        if (it != edge_to_variables.end()) {
                            edge_diff[it->second.first / 2][it->second.first % 2] +=
                                it->second.second * flow;
                        }
                    }
                }
    }

   private:
    Graph g;
    property_map<Graph, edge_reverse_t>::type rev;
    std::vector<Traits::vertex_descriptor> vertex_descriptors;
    std::map<Traits::edge_descriptor, std::pair<int, int>> edge_to_variables;
};

class LemonPreflowHelper : public MaxFlowHelper {
   public:
    using Capacity = int;
    using Graph = lemon::SmartDigraph;
    using Node = Graph::Node;
    using Arc = Graph::Arc;
    template <typename ValueType>
    using ArcMap = lemon::SmartDigraph::ArcMap<ValueType>;
    using Preflow = lemon::Preflow<lemon::SmartDigraph, ArcMap<Capacity>>;

   public:
    LemonPreflowHelper() : capacity(graph), flow(graph), variable(graph) {}
    ~LemonPreflowHelper(){};
    void resize(int n, int m) {
        nodes.reserve(n);
        for (int i = 0; i < n; ++i) nodes.push_back(graph.addNode());
    }
    void addEdge(int x, int y, int c, int rc, int v, int cost = 1) {
        assert(x >= 0);
        assert(v >= -1);
        if (c) {
            auto e1 = graph.addArc(nodes[x], nodes[y]);
            capacity[e1] = c;
            variable[e1] = std::make_pair(v, -1);
        }
        if (rc) {
            auto e2 = graph.addArc(nodes[y], nodes[x]);
            capacity[e2] = rc;
            variable[e2] = std::make_pair(v, 1);
        }
    }
    int compute() {
        Preflow pf(graph, capacity, nodes.front(), nodes.back());
        pf.runMinCut();
        // Copy flow values out
        for (Graph::ArcIt e(graph); e != lemon::INVALID; ++e) {
            flow[e] = pf.flow(e);
        }
        return pf.flowValue();
    }
    void applyTo(std::vector<Vector2i>& edge_diff) {
        for (Graph::ArcIt e(graph); e != lemon::INVALID; ++e) {
            int var = variable[e].first;
            if (var == -1) continue;
            int sgn = variable[e].second;
            int f = flow[e];
            if (f > 0) {
                edge_diff[var / 2][var % 2] += sgn * f;
            }
        }
    }

   private:
    Graph graph;
    ArcMap<Capacity> capacity;
    ArcMap<Capacity> flow;
    ArcMap<std::pair<int, int>> variable;
    std::vector<Node> nodes;
};

class NetworkSimplexFlowHelper : public MaxFlowHelper {
   public:
    using Weight = int;
    using Capacity = int;
    using Graph = lemon::SmartDigraph;
    using Node = Graph::Node;
    using Arc = Graph::Arc;
    template <typename ValueType>
    using ArcMap = lemon::SmartDigraph::ArcMap<ValueType>;
    using Preflow = lemon::Preflow<lemon::SmartDigraph, ArcMap<Capacity>>;
    using NetworkSimplex = lemon::NetworkSimplex<lemon::SmartDigraph, Capacity, Weight>;

   public:
    NetworkSimplexFlowHelper() : cost(graph), capacity(graph), flow(graph), variable(graph) {}
    ~NetworkSimplexFlowHelper(){};
    void resize(int n, int m) {
        nodes.reserve(n);
        for (int i = 0; i < n; ++i) nodes.push_back(graph.addNode());
    }
    void addEdge(int x, int y, int c, int rc, int v, int cst = 1) {
        assert(x >= 0);
        assert(v >= -1);
        if (c) {
            auto e1 = graph.addArc(nodes[x], nodes[y]);
            cost[e1] = cst;
            capacity[e1] = c;
            variable[e1] = std::make_pair(v, 1);
        }

        if (rc) {
            auto e2 = graph.addArc(nodes[y], nodes[x]);
            cost[e2] = cst;
            capacity[e2] = rc;
            variable[e2] = std::make_pair(v, -1);
        }
    }
    int compute() {
        Preflow pf(graph, capacity, nodes.front(), nodes.back());
        NetworkSimplex ns(graph);

        // Run preflow to find maximum flow
        lprintf("push-relabel flow... ");
        pf.runMinCut();
        int maxflow = pf.flowValue();

        // Run network simplex to find minimum cost maximum flow
        ns.costMap(cost).upperMap(capacity).stSupply(nodes.front(), nodes.back(), maxflow);
        auto status = ns.run();
        switch (status) {
            case NetworkSimplex::OPTIMAL:
                ns.flowMap(flow);
                break;
            case NetworkSimplex::INFEASIBLE:
                lputs("NetworkSimplex::INFEASIBLE");
                assert(0);
                break;
            default:
                lputs("Unknown: NetworkSimplex::Default");
                assert(0);
                break;
        }

        return maxflow;
    }
    void applyTo(std::vector<Vector2i>& edge_diff) {
        for (Graph::ArcIt e(graph); e != lemon::INVALID; ++e) {
            int var = variable[e].first;
            if (var == -1) continue;
            int sgn = variable[e].second;
            edge_diff[var / 2][var % 2] -= sgn * flow[e];
        }
    }

   private:
    Graph graph;
    ArcMap<Weight> cost;
    ArcMap<Capacity> capacity;
    ArcMap<Capacity> flow;
    ArcMap<std::pair<int, int>> variable;
    std::vector<Node> nodes;
    std::vector<Arc> edges;
};

#ifdef WITH_GUROBI

#include <gurobi_c++.h>

class GurobiFlowHelper : public MaxFlowHelper {
   public:
    GurobiFlowHelper() {}
    virtual ~GurobiFlowHelper(){};
    virtual void resize(int n, int m) {
        nodes.resize(n * 2);
        edges.resize(m);
    }
    virtual void addEdge(int x, int y, int c, int rc, int v, int cost = 1) {
        nodes[x * 2 + 0].push_back(vars.size());
        nodes[y * 2 + 1].push_back(vars.size());
        vars.push_back(model.addVar(0, c, 0, GRB_INTEGER));
        edges.push_back(std::make_pair(v, 1));

        nodes[y * 2 + 0].push_back(vars.size());
        nodes[x * 2 + 1].push_back(vars.size());
        vars.push_back(model.addVar(0, rc, 0, GRB_INTEGER));
        edges.push_back(std::make_pair(v, -1));
    }
    virtual int compute() {
        std::cerr << "compute" << std::endl;
        int ns = nodes.size() / 2;

        int flow;
        for (int i = 1; i < ns - 1; ++i) {
            GRBLinExpr cons = 0;
            for (auto n : nodes[2 * i + 0]) cons += vars[n];
            for (auto n : nodes[2 * i + 1]) cons -= vars[n];
            model.addConstr(cons == 0);
        }

        // first pass, maximum flow
        GRBLinExpr outbound = 0;
        {
            lprintf("first pass\n");
            for (auto& n : nodes[0]) outbound += vars[n];
            for (auto& n : nodes[1]) outbound -= vars[n];
            model.setObjective(outbound, GRB_MAXIMIZE);
            model.optimize();

            flow = (int)model.get(GRB_DoubleAttr_ObjVal);
            lprintf("Gurobi result: %d\n", flow);
        }

        // second pass, minimum cost flow
        {
            lprintf("second pass\n");
            model.addConstr(outbound == flow);
            GRBLinExpr cost = 0;
            for (auto& v : vars) cost += v;
            model.setObjective(cost, GRB_MINIMIZE);
            model.optimize();

            double optimal_cost = (int)model.get(GRB_DoubleAttr_ObjVal);
            lprintf("Gurobi result: %.3f\n", optimal_cost);
        }
        return flow;
    }
    virtual void applyTo(std::vector<Vector2i>& edge_diff) { assert(0); };

   private:
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    std::vector<GRBVar> vars;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> nodes;
};

#endif

class ECMaxFlowHelper : public MaxFlowHelper {
   public:
    struct FlowInfo {
        int id;
        int capacity, flow;
        int v, d;
        int rev_node, rev_idx;  // reverse edge: graph[rev_node][rev_idx]
    };
    struct SearchInfo {
        SearchInfo(int _id, int _prev_id, int _edge_node, int _edge_idx)
            : id(_id), prev_id(_prev_id), edge_node(_edge_node), edge_idx(_edge_idx) {}
        int id;
        int prev_id;
        int edge_node, edge_idx;  // which edge got us here
    };
    ECMaxFlowHelper() { num = 0; }
    int num;
    std::vector<FlowInfo*> variable_to_edge;
    void resize(int n, int m) {
        graph.resize(n);
        variable_to_edge.resize(m, 0);
        num = n;
    }
    void addEdge(int x, int y, int c, int rc, int v, int cost = 0) {
        int xi = (int)graph[x].size();
        int yi = (int)graph[y].size();
        graph[x].push_back({y, c, 0, v, -1, y, yi});
        graph[y].push_back({x, rc, 0, v, 1, x, xi});
    }
    int compute() {
        int total_flow = 0;
        while (true) {
            std::vector<int> vhash(num, 0);
            std::vector<SearchInfo> q;
            q.push_back(SearchInfo(0, -1, -1, -1));
            vhash[0] = 1;
            int q_front = 0;
            bool found = false;
            while (q_front < (int)q.size()) {
                int vert = q[q_front].id;
                for (int ei = 0; ei < (int)graph[vert].size(); ++ei) {
                    auto& l = graph[vert][ei];
                    if (vhash[l.id] || l.capacity <= l.flow) continue;
                    q.push_back(SearchInfo(l.id, q_front, vert, ei));
                    vhash[l.id] = 1;
                    if (l.id == num - 1) {
                        found = true;
                        break;
                    }
                }
                if (found) break;
                q_front += 1;
            }
            if (q_front == (int)q.size()) break;
            int loc = (int)q.size() - 1;
            while (q[loc].prev_id != -1) {
                auto& fwd = graph[q[loc].edge_node][q[loc].edge_idx];
                fwd.flow += 1;
                graph[fwd.rev_node][fwd.rev_idx].flow -= 1;
                loc = q[loc].prev_id;
            }
            total_flow += 1;
        }
        return total_flow;
    }
    void applyTo(std::vector<Vector2i>& edge_diff) {
        for (int i = 0; i < (int)graph.size(); ++i) {
            for (auto& flow : graph[i]) {
                if (flow.flow > 0 && flow.v != -1) {
                    edge_diff[flow.v / 2][flow.v % 2] += flow.d * flow.flow;
                }
            }
        }
    }
    void applyFlow(int v1, int v2, int flow) {
        for (auto& it : graph[v1]) {
            if (it.id == v2) {
                it.flow += flow;
                break;
            }
        }
    }
    std::vector<std::vector<FlowInfo>> graph;
};

#ifdef WITH_CUDA
class CudaMaxFlowHelper : public MaxFlowHelper {
   public:
    struct EdgeInfo {
        int src, dst, cap, var, sign;
    };

    CudaMaxFlowHelper() : num_nodes_(0) {}
    ~CudaMaxFlowHelper() {}

    void resize(int n, int m) {
        num_nodes_ = n;
        edges_.clear();
        edges_.reserve(m * 2);  // forward + reverse
    }

    void addEdge(int x, int y, int c, int rc, int v, int cost = 1) {
        // Forward edge: x -> y with capacity c
        if (c > 0)
            edges_.push_back({x, y, c, v, -1});
        // Reverse edge: y -> x with capacity rc
        if (rc > 0)
            edges_.push_back({y, x, rc, v, 1});
    }

    int compute() {
        // Build CSR from edge list
        int num_edges = (int)edges_.size();

        // Count degree per node
        std::vector<int> nindex(num_nodes_ + 1, 0);
        for (auto& e : edges_) {
            nindex[e.src + 1]++;
        }
        for (int i = 1; i <= num_nodes_; i++) {
            nindex[i] += nindex[i - 1];
        }

        // Fill CSR
        std::vector<int> nlist(num_edges);
        std::vector<int> cap(num_edges);
        edge_csr_index_.resize(edges_.size());  // maps original edge idx to CSR position
        std::vector<int> offset(num_nodes_, 0);
        for (int i = 0; i < (int)edges_.size(); i++) {
            int src = edges_[i].src;
            int pos = nindex[src] + offset[src];
            nlist[pos] = edges_[i].dst;
            cap[pos] = edges_[i].cap;
            edge_csr_index_[i] = pos;
            offset[src]++;
        }

        // Build reverse CSR
        std::vector<int> rnindex(num_nodes_ + 1, 0);
        for (int i = 0; i < num_edges; i++) {
            rnindex[nlist[i] + 1]++;
        }
        for (int i = 1; i <= num_nodes_; i++) {
            rnindex[i] += rnindex[i - 1];
        }

        std::vector<int> rnlist(num_edges);
        std::vector<int> retoe(num_edges);
        std::vector<int> roffset(num_nodes_, 0);
        for (int v = 0; v < num_nodes_; v++) {
            for (int e = nindex[v]; e < nindex[v + 1]; e++) {
                int dst = nlist[e];
                int rpos = rnindex[dst] + roffset[dst];
                rnlist[rpos] = v;
                retoe[rpos] = e;
                roffset[dst]++;
            }
        }

        int source = 0;           // node 0 = source (addEdge uses v1=0 for source edges)
        int sink = num_nodes_ - 1;  // last node = sink

        result_ = cuda_maxflow_solve(
            num_nodes_, source, sink,
            nindex.data(), nlist.data(), cap.data(), num_edges,
            rnindex.data(), rnlist.data(), retoe.data()
        );

        return result_.max_flow;
    }

    void applyTo(std::vector<Vector2i>& edge_diff) {
        for (int i = 0; i < (int)edges_.size(); i++) {
            int v = edges_[i].var;
            if (v == -1) continue;
            int sgn = edges_[i].sign;
            int csr_idx = edge_csr_index_[i];
            int f = result_.edge_flows[csr_idx];
            if (f > 0) {
                edge_diff[v / 2][v % 2] += sgn * f;
            }
        }
    }

   private:
    int num_nodes_;
    std::vector<EdgeInfo> edges_;
    std::vector<int> edge_csr_index_;  // maps edges_[i] to CSR edge index
    CudaMaxFlowResult result_;
};
#endif  // WITH_CUDA

} // namespace qflow

#endif
