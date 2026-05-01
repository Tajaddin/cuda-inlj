# GPU Index Nested-Loop Join Implementation

**Course:** COP6527 - Computing Massive Parallel Systems  
**Semester:** Fall 2025  
**Author:** Tajaddin  

## Project Overview

This project implements an **Index Nested-Loop Join (INLJ)** algorithm on NVIDIA GPUs using CUDA. The implementation uses a B+ Tree index structure to accelerate join operations on relational database tables.

### Key Features

- **B+ Tree Index:** Efficient tree-based index structure optimized for GPU memory access patterns
- **Parallel Join Processing:** Each GPU thread independently processes outer table tuples
- **Baseline Comparison:** Includes GPU hash join implementation for performance comparison
- **Comprehensive Testing:** Correctness verification, scalability analysis, and selectivity testing

## Algorithm Description

### Index Nested-Loop Join

The Index Nested-Loop Join algorithm works as follows:

1. **Build Phase:** Construct a B+ Tree index on the inner table (R) based on the join key
2. **Probe Phase:** For each tuple in the outer table (S), use the index to find matching tuples in R
3. **Output Phase:** Emit `<R.id, S.id>` pairs for all matching tuples where `R.key = S.key`

### GPU Parallelization Strategy

- **Thread Assignment:** Each CUDA thread processes one tuple from the outer table S
- **Index Traversal:** Threads traverse the B+ Tree from root to leaf using binary search
- **Result Collection:** Matching pairs are atomically added to the output buffer
- **Memory Coalescing:** B+ Tree nodes are organized to maximize memory access efficiency

## Project Structure

```
cuda_inlj/
├── include/
│   └── inlj.h           # Header file with data structures and function declarations
├── src/
│   ├── btree_index.cu   # B+ Tree construction and GPU join kernels
│   ├── utils.cu         # Data generation and utility functions
│   └── main.cu          # Test program with benchmarks
├── Makefile             # Build system
└── README.md            # This file
```

## Building the Project

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- CUDA Toolkit 11.0 or later
- GCC/G++ compiler
- Make build system

### Build Commands

```bash
# Build the project
make

# Build with debug symbols
make debug

# Clean build artifacts
make clean

# Show help
make help
```

### GPU Architecture Configuration

Edit the `CUDA_ARCH` variable in the Makefile to match your GPU:

| GPU Series | Architecture | Flag |
|------------|--------------|------|
| GTX 10xx   | Pascal       | sm_60 |
| RTX 20xx   | Turing       | sm_75 |
| RTX 30xx   | Ampere       | sm_86 |
| RTX 40xx   | Ada Lovelace | sm_89 |

## Running Tests

```bash
# Run all tests
make run

# Run individual tests
make run1    # Correctness verification
make run2    # Performance comparison
make run3    # Scalability analysis
make run4    # Selectivity analysis

# Profile with NVIDIA tools
make profile       # nvprof
make ncu-profile   # Nsight Compute
```

## Test Descriptions

### Test 1: Correctness Verification
Verifies that the GPU INLJ produces the correct number of join results by comparing against a CPU reference implementation.

### Test 2: Performance Comparison
Compares execution time between Index Nested-Loop Join and a baseline hash join implementation, providing detailed timing breakdowns.

### Test 3: Scalability Analysis
Tests performance across different data sizes (100K to 10M tuples) to analyze how the algorithms scale with increasing data.

### Test 4: Selectivity Analysis
Tests performance with varying join selectivities (10% to 100% match rates) to understand behavior with different join cardinalities.

## Data Structures

### Tuple
```c
typedef struct {
    uint32_t id;    // Sequential tuple identifier
    uint32_t key;   // Join key attribute
} Tuple;
```

### B+ Tree Node
```c
typedef struct {
    uint32_t keys[BTREE_MAX_KEYS];   // Keys in the node
    uint32_t values[BTREE_ORDER];    // Child pointers or tuple IDs
    uint32_t num_keys;               // Current key count
    uint32_t is_leaf;                // Leaf node flag
    uint32_t next_leaf;              // Sibling pointer for range queries
} BTreeNode;
```

### Join Result
```c
typedef struct {
    uint32_t r_id;  // ID from table R
    uint32_t s_id;  // ID from table S
} JoinResult;
```

## Performance Optimization Techniques

1. **Bulk Loading:** B+ Tree is constructed bottom-up from sorted data for optimal structure
2. **Binary Search:** All tree traversals use binary search for O(log n) key comparisons
3. **Coalesced Access:** Node layout optimized for GPU memory coalescing
4. **Atomic Operations:** Thread-safe result collection using atomic operations
5. **Grid-Stride Loops:** Efficient work distribution across variable-size grids

## Sample Output

```
=============================================================
  GPU Index Nested-Loop Join Implementation
  COP6527 - Computing Massive Parallel Systems
  Fall 2025
=============================================================

=============================================================
 CUDA Device Information
=============================================================
 Device:               NVIDIA GeForce RTX 3080
 Compute Capability:   8.6
 Total Global Memory:  10.00 GB
 Multiprocessors:      68
=============================================================

=============================================================
 Performance Comparison: INLJ vs Hash Join
=============================================================
 Metric                    INLJ         Hash       Speedup
-------------------------------------------------------------
 Kernel Time (ms)        12.450       18.230       1.46x
 Total Time (ms)         25.680       32.150       1.25x
 Results                 500000       500000
=============================================================
```

## References

1. Kim, C., et al. "Sort vs. Hash Revisited: Fast Join Implementation on Modern Multi-Core CPUs." VLDB 2009.
2. He, B., et al. "Relational Query Coprocessing on Graphics Processors." TODS 2009.
3. NVIDIA CUDA Programming Guide
4. B+ Tree: Comer, D. "The Ubiquitous B-Tree." ACM Computing Surveys, 1979.

## License

MIT License
