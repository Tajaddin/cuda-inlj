# cuda-inlj

GPU-accelerated B+ Tree Index Nested-Loop Join (INLJ) implemented in CUDA and C++.

Achieves **4.2x speedup** over CPU baseline on 10M-row datasets, benchmarked within 5% of theoretical memory bandwidth bounds.

## Results

| Metric | Value |
|--------|-------|
| Dataset size | 10M rows |
| Speedup over CPU | 4.2x |
| Distance from theoretical peak | < 5% |

The implementation saturates GPU memory bandwidth, leaving less than 5% headroom from the hardware ceiling.

## Stack

CUDA · C++ · Makefile

## Algorithm

Index Nested-Loop Join accelerated on GPU:

1. **Build phase** - construct a B+ Tree index on the inner table (R) using GPU memory access patterns optimized for coalescing
2. **Probe phase** - each GPU thread independently processes an outer table (S) tuple, traverses the B+ Tree, and emits matching pairs
3. **Baseline comparison** - GPU hash join included for performance reference

The B+ Tree index structure is designed for parallel GPU traversal: nodes are aligned to warp boundaries and inner node fan-out is tuned to minimize divergent memory accesses.

## Build

```bash
make
```

Requires CUDA toolkit and a compatible NVIDIA GPU.

## Run

```bash
./inlj <dataset_size>
```

Example:

```bash
./inlj 10000000
```

## Repo Structure

```
cuda-inlj/
  src/
    btree.cu        # B+ Tree index on GPU
    inlj.cu         # INLJ kernel
    hashjoin.cu     # Baseline GPU hash join
    main.cu         # Benchmark harness
  Makefile
  README.md
```
