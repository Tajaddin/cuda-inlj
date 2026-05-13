# cuda-inlj

GPU-accelerated Index Nested-Loop Join with a B+ Tree index, written in CUDA C++. Coursework for COP6527 (Computing Massive Parallel Systems), Fall 2025, USF.

## Hero numbers

| Metric | Value |
|---|---|
| Max dataset | 10M tuples per table |
| Speedup over CPU INLJ | 4.2x |
| Memory bandwidth headroom | under 5 percent of theoretical peak |
| Selectivity sweep | 10, 25, 50, 75, 100 percent overlap |
| Scalability sweep | 100K, 500K, 1M, 2M, 5M, 10M tuples |

The kernel saturates GPU memory bandwidth on a probe-heavy workload. The remaining gap to theoretical peak stays under 5 percent across all six dataset sizes.

## Algorithm

Index Nested-Loop Join on GPU:

1. Build a B+ Tree index on the inner table R in host code, then copy to device with a node layout aligned to warp boundaries.
2. Each GPU thread takes one outer-table S tuple and walks the index. Fan-out is tuned to minimize divergent memory access during traversal.
3. Matches go to a per-block output buffer and aggregate into a single contiguous result array with a prefix-sum pass.

A CPU reference INLJ runs first on each test to verify match counts. The GPU implementation matches the CPU baseline result count on every selectivity level.

## Four tests

The driver in `src/main.cu` runs four scenarios, all selectable from the Makefile.

| Test | What it covers |
|---|---|
| 1 Correctness | GPU result count matches CPU reference across selectivity levels |
| 2 Performance | GPU INLJ vs CPU INLJ wall-clock on the canonical 10M x 10M join |
| 3 Scalability | Throughput from 100K to 10M tuples |
| 4 Selectivity | Throughput at 10, 25, 50, 75, 100 percent overlap |

## Build

```
make
```

You need a CUDA toolkit and an NVIDIA GPU. The Makefile ships a multi-arch build for sm_60 through sm_86 (Pascal through Ampere).

## Run

```
make run        # all four tests
make run1       # correctness only
make run2       # performance only
make run3       # scalability sweep
make run4       # selectivity sweep
```

Direct execution:

```
./bin/inlj_test 2   # test number, 1 through 4
```

## Profile

```
make profile        # nvprof
make ncu-profile    # Nsight Compute, writes profile_report.ncu-rep
```

The Nsight report shows the probe kernel is memory-bound, with achieved DRAM throughput within 5 percent of the device peak.

## Repository layout

```
cuda-inlj/
  include/
    inlj.h            Shared types: Tuple, BTreeIndex, JoinResult, PerformanceMetrics
  src/
    btree_index.cu    B+ Tree build, device copy, traversal kernel
    utils.cu          Test-data generation, timing helpers, CPU reference INLJ
    main.cu           Four-test driver with throughput reporting
  Makefile            Multi-arch build, run targets, profiling targets
```

## Limitations

1. Single-GPU. Multi-GPU partitioning of R is future work.
2. Fixed-size B+ Tree nodes. A dynamic node size would help skewed key distributions.
3. Materialized results. A pipelined output to a downstream kernel would cut a memory pass on selectivity-100 joins.
4. Hash join baseline lives in the COP6527 course notes, not in this repo. Speedup numbers compare to CPU INLJ only.
