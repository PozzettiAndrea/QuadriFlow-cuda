# QuadriFlow-CUDA: Experiments & Engineering Log

Test mesh: **Stanford Dragon** — 435,545 vertices, 871,306 faces, target 100,000 output faces.
Hardware: NVIDIA RTX A6000 (SM 86, 48GB VRAM), Linux.

---

## 1. Baseline CPU Pipeline Profile

Full pipeline timing on dragon.obj with all-CPU strategies:

### Phase 1: Initialize (~6.7s)

| Step | Time | Device | Notes |
|------|------|--------|-------|
| ComputeMeshStatus | 0.01s | CPU | Basic mesh stats |
| compute_direct_graph (pre) | 0.01s | CPU | Half-edge structure (V2E, E2E, boundary, nonManifold) |
| **subdivide** | **2.92s** | **CPU** | Priority-queue edge splitting to target resolution |
| compute_direct_graph (post) | 0.04s | CPU | Rebuild half-edges after subdivision |
| generate_adjacency_matrix | 0.43s | CPU | Sparse adjacency for hierarchy |
| rho smoothing | 0.02s | CPU | Scale field normalization |
| ComputeSharpEdges | 0.02s | CPU | Detect dihedral angle features |
| ComputeSmoothNormal | 0.15s | CPU | Angle-weighted per-vertex normals |
| ComputeVertexArea | 0.04s | CPU | Per-vertex area calculation |
| Hierarchy build (25 levels) | ~2.1s | CPU | graph_coloring + DownsampleGraph x25 |
| field init (Q,O,S,K) | 0.40s | CPU | Allocate orientation/position fields |
| CopyToDevice | 0.43s | GPU | Upload mesh arrays to GPU |

### Phase 2: Field Solving (~5.9s)

| Step | Time | Device | Notes |
|------|------|--------|-------|
| Orientation field | 1.12s | GPU | Solve cross-field on multigrid hierarchy |
| Scale field | 0.13s | GPU | Solve target edge length field |
| Position field | 4.60s | GPU | Solve integer-grid positions on hierarchy |

### Phase 3: Index Map (~54.5s) — THE BOTTLENECK

| Step | Time | Device | Notes |
|------|------|--------|-------|
| BuildEdgeInfo | 1.14s | CPU | Edge difference structures |
| Sharp edge setup | 0.03s | CPU | Mark sharp constraints |
| BuildIntegerConstraints | 5.21s | CPU | Round position field to integers |
| **ComputeMaxFlow** | **16.58s** | **CPU+GPU** | Min-cost flow for integer constraints |
| — DownsampleEdgeGraph | 0.26s | CPU | Edge graph coarsening |
| — CudaMaxFlow (main) | 5.74s | GPU | GPU push-relabel solver |
| — arc setup + addEdge | 5.55s | CPU | Build flow graph (**CPU bottleneck**) |
| — ECMaxFlow (residual) | 0.23s | CPU | CPU fallback for residual flow |
| — arc setup + addEdge (2nd) | 4.14s | CPU | Rebuild graph for 2nd pass |
| subdivide_edgeDiff (1st) | 4.10s | CPU | Split edges with |d| > 1 |
| **FixFlipHierarchy** | **8.58s** | **CPU** | Fix orientation flips via recursive hierarchy |
| subdivide_edgeDiff (2nd) | 2.25s | CPU | 2nd subdivision pass |
| FixFlipSat | 0.00s | CPU | SAT-based flip fix (trivial) |
| optimize_positions_sharp | 0.20s | CPU | Sharp edge position solve |
| optimize_positions_fixed | 3.00s | CPU→GPU | PCG solve (0.46s GPU + 2.5s CPU setup) |
| AdvancedExtractQuad | 3.85s | CPU | Extract quad mesh from parametrization |
| FixValence | 0.30s | CPU | Fix irregular valence vertices |
| pre-dynamic CSR setup | 2.32s | CPU | Build CSR sparsity pattern (std::map) |
| optimize_positions_dynamic | 4.60s | CPU→GPU | 10x PCG solve loop |
| — FindNearest (total) | 1.41s | CPU | Nearest-point queries |
| — fillCSR (total) | 0.52s | CPU | Fill CSR values |
| — PCG solve (total) | 2.14s | GPU | IC0-preconditioned CG |
| — setup+overhead (total) | 0.53s | CPU | Compact vectors etc |

**Total: ~67s** (dominated by ComputeMaxFlow arc setup + FixFlipHierarchy + subdivide).

---

## 2. GPU Mesh Subdivision

### Experiment 2a: Naive GPU hybrid (per-pass transfers)

**Approach:** GPU marks long edges + resolves conflicts, CPU applies splits + rebuilds E2E via hash map. Every pass: full cudaMalloc → H→D transfer → kernel → D→H transfer → cudaFree.

**Result: 144.6s** — 60x slower than CPU (2.3s).

Root cause: 30 passes with growing mesh (871K → 5M faces). Each pass transfers the full mesh both directions and allocates/frees GPU memory. Transfer log from a typical run:

```
[XFER] H->D: V=10.5 MB, F=10.5 MB, E2E=10.5 MB     (pass 1)
[XFER] H->D: V=21.2 MB, F=21.2 MB, E2E=21.2 MB     (pass 2)
[XFER] H->D: V=37.0 MB, F=32.9 MB, E2E=32.9 MB     (pass 3)
...
[XFER] H->D: V=75.4 MB, F=60.6 MB, E2E=60.6 MB     (pass 17)
```

Data transferred: ~1.5 GB H→D + ~0.5 GB D→H across 30 passes. The CPU hash-map E2E rebuild (O(nF) per pass on growing mesh) was also significant.

**Conclusion:** Per-pass CPU-GPU round-trips dominate. The GPU kernels themselves are fast; the problem is the data movement pattern.

### Experiment 2b: GPU-resident (single transfer)

**Approach:** Allocate all GPU buffers once with 10x capacity. Single H→D upload, all 19 passes run entirely on GPU, single D→H download. Key innovations:

1. **Conflict-free parallel splitting** — Three kernels: mark long edges, resolve within-face conflicts (keep longest), resolve cross-face conflicts (lower index wins). After resolution, at most one split per face, guaranteed independent.

2. **Parallel split application** — `thrust::exclusive_scan` on marks gives per-split vertex/face output offsets. Each split thread writes to non-overlapping output locations.

3. **Double-buffered F** — Snapshot F→F_old before applying splits. Read from F_old, write to F. Needed because a face's neighbor (via E2E) can be the primary face of a different split — without the snapshot, two threads could race on the same face.

4. **Sort-based E2E rebuild** — For each half-edge, compute key = `min(va,vb) * maxV + max(va,vb)`. Sort `(key, half_edge_index)` pairs via `thrust::sort_by_key`. Consecutive entries with same key are twin half-edges. Single kernel pairs them.

**Result: 0.615s** — **3.7x faster than CPU** (2.3s).

```
[XFER] H->D (once): V=10.5 MB, F=10.5 MB, E2E=10.5 MB
[SUBDIV-GPU]   pass 1: 445708 splits, nV=881253 nF=1762722
[SUBDIV-GPU]   pass 2: 717080 splits, nV=1598333 nF=2869539
...
[SUBDIV-GPU]   pass 19: 1 splits, nV=4102187 nF=6144983
[SUBDIV-GPU] Done: 19 passes, 3666642 total splits
[XFER] D->H (once): V=98.5 MB, F=73.7 MB
[INIT] subdivide: 0.615000 s
```

Total data transferred: 31.5 MB H→D + 172 MB D→H (once each). Compare to naive approach: ~2 GB total.

### Experiment 2c: GPU vs CPU mesh size discrepancy

The GPU produces a ~30% larger mesh (4.1M vs 3.1M vertices, 3.67M vs 2.71M splits). Investigation revealed a **bug in upstream QuadriFlow** (see Section 7).

---

## 3. GPU DownsampleEdgeGraph

**Approach:** Parallel face merging and edge graph coarsening:
1. Candidate edge selection (EdgeDiff == 0 edges)
2. Greedy independent set via iterated graph coloring (multi-round)
3. Parallel face collapse with `thrust::exclusive_scan` for compacted output
4. Edge diff propagation through collapsed face paths
5. E2E/F2E/E2F rebuild via prefix scan

**Baseline CPU:** 0.26s per call.
**GPU:** Comparable speed — the main benefit is keeping data on GPU to avoid transfer overhead when called repeatedly during hierarchy construction.

Dispatched via `-dse cuda` flag. Strategy field: `hierarchy.dse_strategy`.

---

## 4. GPU Max-Flow Solver

**Approach:** CUDA push-relabel based on ECL-MaxFlow (Avery VanAusdal & Martin Burtscher, BSD license). Kernels:
- `k_init` — Initialize flow, excess, height, time arrays
- `k_initialPush` — Push flow from source vertex
- `k_reverseResidualBFS` — Two-phase BFS on residual graph for height relabeling
- `k_phase1/2_pushRelabel` — Push-relabel with global relabeling heuristic

**Baseline CPU (Boykov max-flow):** Two passes totaling ~10s (solver only, excluding arc setup).
**GPU push-relabel:** 5.74s for the main flow solve.

**Remaining bottleneck:** CPU arc setup + addEdge takes 5.55s + 4.14s = 9.69s. The flow graph construction is sequential on CPU because it builds a pointer-based adjacency structure. This is the single largest remaining CPU bottleneck in the pipeline.

Also added: **LEMON Preflow** solver (CPU, via `-DWITH_LEMON_FLOW`). Uses LEMON library's Preflow algorithm as an alternative CPU backend.

---

## 5. GPU Sparse Linear Solvers

### IC0-Preconditioned Conjugate Gradient (PCG)

Used for orientation field, position field, and dynamic position optimization.

**Architecture:**
- Persistent context (`cuda_pcg_init` / `cuda_pcg_solve` / `cuda_pcg_destroy`)
- IC0 symbolic analysis done once (depends only on sparsity pattern)
- Numeric IC0 re-factorization per solve (values change, pattern doesn't)
- SpMV via cuSPARSE, triangular solves via cuSPARSE SpSV
- Custom CUDA kernels for CG vector operations (dot, axpy, scale)
- Parameters: max 1000 iterations, tolerance 1e-6

**Dynamic positioning performance:**
- 10x PCG solve loop total: 2.14s GPU compute
- CPU overhead (FindNearest + fillCSR + setup): 2.46s
- Total: 4.60s (GPU is 46% of wall time, rest is CPU setup)

### Cholesky Factorization

Used for scale field optimization via cuSolver `cusolverSpDcsrlsvchol`. Automatic fallback to CPU Eigen SimplicialLLT if GPU solve fails.

---

## 6. FixFlip Strategy Experiments

Three strategies were implemented and tested for the recursive FixFlip orientation correction:

### Strategy 0: "cpu" (baseline)

Original CPU sequential scan. Iterates through flipped faces, tries CheckShrink on each, recursively builds new hierarchy and repeats.

- **Time:** ~8.6s
- **Quality:** Best — proven correct, deep recursive hierarchies (8-10 levels)
- Flipped faces: 146K → ~0 after full recursion

### Strategy 1: "gpu-prefilter"

GPU parallel scan identifies ALL fixable faces (via `try_checkshrink_readonly` kernel on GPU), then CPU processes them in priority order.

- **GPU scan time:** 0.02s per call (vs ~0.66s CPU sequential scan)
- **Total FixFlip:** ~8.5s — roughly same as CPU
- **Why not faster:** GPU overhead per recursion (~20ms malloc/copy/free, ~5ms flatten) eats the scan speedup. The scan is only ~10% of each recursion; the actual CheckShrink application + hierarchy rebuild dominates.

### Strategy 2: "gpu-only"

GPU identifies fixable faces, CPU applies ONLY those faces (no sequential scan fallback).

- **Result:** Much worse — fewer fixes per pass (~30K vs ~100K zero-diff edges), more recursion levels needed (20+ vs 10)
- **Total FixFlip:** ~6-9s (WORSE than CPU)
- **Root cause:** GPU `thrust::find` returns lowest-index fixable face, which may produce shallow recursive hierarchies (1-3 levels vs 8-10 for CPU). The face selection quality matters for multigrid amplification.

**Conclusion:** FixFlip is fundamentally sequential — each fix changes the topology for subsequent fixes. GPU parallelism helps with the scan but not the application. Strategy 0 (cpu) remains the default.

---

## 7. Upstream Bug Discovery: rho Comma Operator

While investigating why the GPU subdivision produces ~30% more vertices than CPU (Experiment 2c), we found a bug in upstream QuadriFlow.

**File:** `src/subdivide.cpp`, line 140
**Present since:** First commit adding rho support (commit `4fb9cc0`)

```cpp
// BUG: comma operator, not addition
rho[vn] = 0.5f * (rho[v0], rho[v1]);

// What it actually computes:
rho[vn] = 0.5f * rho[v1];  // discards rho[v0]

// What was intended:
rho[vn] = 0.5f * (rho[v0] + rho[v1]);  // midpoint average
```

**Impact:** The C++ comma operator evaluates both operands but returns only the second. New midpoint vertices get `0.5 * rho[v1]` instead of the average. This makes rho artificially small at new vertices, which lowers the adaptive splitting threshold (`lengthSq > max(maxLengthSq * 0.75, min(rho[v0], rho[v1]))`), causing the CPU to mark **fewer** edges for splitting.

**Verification:**
- CPU with bug: 2,706,003 splits → 3,141,548 vertices, 5,053,004 faces
- GPU without bug: 3,666,642 splits → 4,102,187 vertices, 6,144,983 faces (35% more)
- The GPU code correctly averages rho, producing the intended mesh density

**Paper reference:** The QuadriFlow paper (Huang et al., SGP 2018) builds on Instant Meshes (Jakob et al., SIGGRAPH Asia 2015), which defines rho as a global edge length parameter. QuadriFlow extends it to a per-vertex scale field. The intended behavior for subdivision midpoints is linear interpolation (average), consistent with standard FEM practice for scalar field refinement.

**Status:** Bug exists in upstream [hjwdzh/QuadriFlow](https://github.com/hjwdzh/QuadriFlow). Fix branch: `fix/rho-comma-operator-bug`.

---

## 8. Checkpoint System

Binary serialization of full `Parametrizer` + `Hierarchy` state at 12 pipeline stage boundaries. Enables benchmarking individual stages without re-running the full pipeline.

### Stages

| # | Stage Name | What just completed |
|---|-----------|-------------------|
| 0 | post-init | Initialize() |
| 1 | post-orient | optimize_orientations + ComputeOrientationSingularities |
| 2 | post-field | optimize_positions + ComputePositionSingularities |
| 3 | post-edgeinfo | BuildEdgeInfo + sharp edge setup |
| 4 | post-constraints | BuildIntegerConstraints |
| 5 | post-flow | ComputeMaxFlow |
| 6 | post-subdiv1 | First subdivide_edgeDiff |
| 7 | pre-ffh | allow_changes setup, before FixFlipHierarchy |
| 8 | post-ffh | FixFlipHierarchy |
| 9 | post-subdiv2 | Second subdivide_edgeDiff + FixFlipSat |
| 10 | post-extract | AdvancedExtractQuad + FixValence |
| 11 | post-dynamic | optimize_positions_dynamic (final) |

### File format

Each `.qfc` file contains:
- **Header** (576 bytes): magic (`QFC\0`), version, stage name, all strategy flags, input mesh path, target faces, unix timestamp, reserved padding
- **Parametrizer scalars**: normalize_scale/offset, surface_area, scale, edge lengths
- **Hierarchy**: full multigrid hierarchy via `SaveToFile`/`LoadFromFile`
- **Stage-dependent data**: only fields available at the saved stage are serialized (e.g., edge_diff only after post-edgeinfo)

### Checkpoint sizes (dragon.obj, 100K target)

Typical checkpoint file sizes range from ~928 MB (post-init, mostly hierarchy) to ~1.3 GB (post-extract, includes compact quad mesh).

### Usage

```bash
# Save all checkpoints during a full run
./quadriflow -i dragon.obj -o out.obj -f 100000 -save-all -save-dir /tmp/ckpt

# Resume from post-field, run only through post-flow
./quadriflow -run-from post-field -run-to post-flow -save-dir /tmp/ckpt -o out.obj

# Benchmark just FixFlipHierarchy
./quadriflow -run-from pre-ffh -run-to post-ffh -save-dir /tmp/ckpt -o out.obj
```

---

## 9. CLI Reference

```
./quadriflow -i <input.obj> -o <output.obj> -f <target_faces> [options]

Strategy flags:
  -ff cpu|gpu-prefilter|gpu-only    FixFlip strategy (default: cpu)
  -dse cpu|cuda                     DownsampleEdgeGraph strategy (default: cpu)
  -subdiv cpu|cuda                  Mesh subdivision strategy (default: cpu)

Checkpoint control:
  -save-dir <dir>       Directory for checkpoint files
  -save-at <stage>      Save checkpoint at a specific stage
  -save-all             Save at every stage boundary
  -run-from <stage>     Resume from a saved checkpoint
  -run-to <stage>       Stop after reaching a stage
  -list-stages          Print all stage names and exit

Legacy:
  -save-ff <path>       Save FixFlip state (old interface)
  -bench-ff <path>      Load FixFlip state and benchmark
```

---

## 10. Open Questions & Future Work

1. **ComputeMaxFlow arc setup (9.7s CPU):** The largest remaining bottleneck. The flow graph is built sequentially via `addEdge` calls. GPU-accelerating this requires building CSR flow graphs on GPU, which needs a different graph construction approach.

2. **BuildIntegerConstraints (5.2s CPU):** Entirely CPU. Rounding position field to integers with constraint satisfaction. Could benefit from GPU parallelism.

3. **AdvancedExtractQuad (3.9s CPU):** Quad mesh extraction from parametrization. Complex topology operations that are hard to parallelize.

4. **FixFlipHierarchy (8.6s CPU):** Fundamentally sequential due to topology-changing operations. GPU prefilter helps scan speed but not application speed. Possible approach: batch multiple independent fixes per iteration.

5. **GPU subdivision mesh size:** The corrected rho averaging produces ~35% more vertices. Need to evaluate whether this affects downstream quad mesh quality, or if the CPU's accidentally-smaller mesh was "good enough."

6. **Data transfer minimization:** `[XFER]` logging is added to subdivide_gpu.cu. Should be extended to all GPU code paths to identify transfer hotspots across the full pipeline.
