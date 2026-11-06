# cuda-inlj

GPU-accelerated Index Nested-Loop Join with a B+ Tree index, written in CUDA C++. Coursework for COP6527 (Computing Massive Parallel Systems), Fall 2025, USF.

## Results

Measured on an RTX 4060 (8 GB, sm_89) under WSL2 Ubuntu 24.04, built with CUDA 13.5 for sm_89. Every join time is the median of 11 runs after one warm-up run; the index build is a single CPU run. Three full runs of `./bin/inlj_test` agreed to within 7% on every kernel time and within 24% on the end-to-end totals, which also include allocation, transfers and the CPU index build; the numbers below are from the first. Treat the totals and the end-to-end ratio as indicative: each includes one CPU run of the index build, which moves with host load far more than the kernel times do.

| Metric | Value |
|---|---|
| Max dataset | 10M tuples per table |
| Probe kernel, 1M x 1M, 50% overlap | INLJ 0.532 ms, hash join 0.167 ms: the hash probe is 3.2x faster |
| End to end, including the host-side index build | hash join is ~21x faster (10M: 1,147 ms vs 55 ms); indicative, see above |
| Correctness | every test checks both joins' (r_id, s_id) pairs against a CPU reference join |
| Selectivity sweep | 10, 25, 50, 75, 100 percent overlap |
| Scalability sweep | 100K, 500K, 1M, 2M, 5M, 10M tuples |

| Overlap (1M x 1M) | INLJ kernel (ms) | Hash probe (ms) |
|---|---|---|
| 10% | 0.470 | 0.119 |
| 50% | 0.485 | 0.154 |
| 100% | 0.514 | 0.176 |

| Tuples per table | INLJ kernel (ms) | Hash probe (ms) | INLJ total (ms) | Hash total (ms) |
|---|---|---|---|---|
| 100K | 0.047 | 0.023 | 14.46 | 1.19 |
| 1M | 0.535 | 0.167 | 90.88 | 3.88 |
| 10M | 16.484 | 7.065 | 1,147.42 | 55.19 |

The hash join wins on this data. Each INLJ probe walks four levels of 63-key nodes with a binary search in each, while a hash probe at load factor 0.5 reads a few adjacent slots. End to end the gap widens, because the tree is bulk-loaded on the CPU and copied over, while the hash table is built on the GPU. Reusing a prebuilt index removes the build cost but not the per-probe gap, so on this data the hash join wins either way.

Earlier versions of this README claimed a 4.4x faster INLJ kernel (0.54 ms vs 2.34 ms). It did not survive a warm-up and median timing; the test data also sent every non-matching probe down the same cached rightmost path. Both are fixed.

## Algorithm

Index Nested-Loop Join on GPU:

1. Build a B+ Tree (order 64: up to 63 keys per node) on the inner table R in host code by bulk loading sorted keys, then copy the node array to the device as is (520-byte nodes, no special alignment).
2. Each GPU thread takes one outer-table S tuple and walks the index from the root, binary-searching each node.
3. Each match takes the next slot of a single global result array through `atomicAdd`, so the order of the results is not deterministic.

The baseline is a GPU hash join with linear probing (`build_hash_table_kernel`, `hash_join_probe_kernel`), built and probed on the device.

Test data: R gets the even keys 0, 2, ..., 2(n-1) in random order. The S keys that match are random R keys; the ones that do not are distinct random odd keys from the same range, so a miss walks a random root-to-leaf path like a hit does.

## Four tests

The driver in `src/main.cu` runs four scenarios, all selectable from the Makefile. In every one, both joins' (r_id, s_id) pairs are compared with a CPU reference join (a CPU hash table of R); any wrong, duplicate or missing pair fails the test and the program exits non-zero.

| Test | What it covers |
|---|---|
| 1 Correctness | Both GPU joins return exactly the reference pairs (100K x 100K, 50% overlap) |
| 2 Performance | GPU INLJ vs GPU hash join (kernel and total time) on 1M x 1M |
| 3 Scalability | Kernel and total time from 100K to 10M tuples |
| 4 Selectivity | Kernel and total time at 10, 25, 50, 75, 100 percent overlap |

## Build

```
make
make test       # unit tests
```

You need a CUDA toolkit and an NVIDIA GPU. The Makefile builds machine code for sm_75, sm_80 and sm_86, plus sm_60 and sm_70 when nvcc is older than CUDA 13, which dropped them, and compute_86 PTX. The sm_86 code also runs on the other 8.x GPUs, such as the RTX 40 series (sm_89); GPUs of compute capability 9.0 and newer JIT-compile the PTX. To build for one GPU only:

```
make CUDA_ARCH="-gencode arch=compute_89,code=sm_89"
```

The unit tests (`tests/test_inlj.cu`) check that both GPU joins return exactly the expected pairs for tree heights 1, 3 and 4 at 0, 50 and 100 percent overlap, that the CPU check rejects wrong, missing and duplicate pairs, that non-matching keys are spread across R's key range, and that a join whose allocation fails frees its device memory.

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

`make ncu-profile` writes the Nsight Compute report for the probe kernel; no profile report is committed, so treat any bandwidth figure as something to measure on your own GPU.

## Repository layout

```
cuda-inlj/
  include/
    inlj.h            Shared types: Tuple, BTreeIndex, JoinResult, PerformanceMetrics
  src/
    btree_index.cu    B+ Tree build, device copy, INLJ and hash-join kernels
    utils.cu          Test-data generation, CPU reference check, reporting
    main.cu           Four-test driver: warm-up, median of 11 timed runs, pair verification
  tests/
    test_inlj.cu      Unit tests (make test)
  Makefile            Multi-arch build, run, test and profiling targets
```

## Limitations

1. Single-GPU. Multi-GPU partitioning of R is future work.
2. Fixed-size B+ Tree nodes. A dynamic node size would help skewed key distributions.
3. Materialized results. A pipelined output to a downstream kernel would cut a memory pass on selectivity-100 joins.
4. Unique keys only: each S tuple matches at most one R tuple. The CPU reference join checks results and is not timed.
