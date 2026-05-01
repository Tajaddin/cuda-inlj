/**
 * @file inlj.h
 * @brief Index Nested-Loop Join on GPUs using CUDA
 * @author Tajaddin
 * @course COP6527 - Computing Massive Parallel Systems
 * @date Fall 2025
 * * This header defines the data structures and function prototypes for
 * implementing an Index Nested-Loop Join algorithm on NVIDIA GPUs.
 * The implementation uses a B+ Tree index for efficient key lookups.
 */

#ifndef INLJ_H
#define INLJ_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// ============================================================================
// Configuration Constants
// ============================================================================

/** Maximum fanout of B+ Tree nodes (must be even for proper splitting) */
#define BTREE_ORDER 64

/** Maximum number of keys in a B+ Tree node */
#define BTREE_MAX_KEYS (BTREE_ORDER - 1)

/** Minimum number of keys in a B+ Tree node (except root) */
#define BTREE_MIN_KEYS (BTREE_ORDER / 2 - 1)

/** Number of CUDA threads per block */
#define THREADS_PER_BLOCK 256

/** Maximum depth of B+ Tree (sufficient for billions of records) */
#define MAX_TREE_DEPTH 20

/** Invalid index marker */
#define INVALID_INDEX 0xFFFFFFFF

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Tuple structure representing a record in tables R and S
 * * Each tuple consists of:
 * - id: Sequential identifier (position in the table)
 * - key: Join attribute value (uniformly distributed, unique)
 */
typedef struct {
    uint32_t id;    /**< Sequential tuple identifier */
    uint32_t key;   /**< Join key attribute */
} Tuple;

/**
 * @brief Output tuple representing a join result
 * * Contains the IDs of matching tuples from tables R and S
 * where R.key = S.key
 */
typedef struct {
    uint32_t r_id;  /**< ID from table R */
    uint32_t s_id;  /**< ID from table S */
} JoinResult;

/**
 * @brief B+ Tree node structure for GPU-based index
 * * This structure represents both internal and leaf nodes:
 * - Internal nodes: keys guide searches, children point to subtrees
 * - Leaf nodes: keys are actual values, values store tuple IDs
 */
typedef struct {
    uint32_t keys[BTREE_MAX_KEYS];      /**< Keys stored in this node */
    uint32_t values[BTREE_ORDER];       /**< For leaves: tuple IDs; for internal: child indices */
    uint32_t num_keys;                   /**< Number of keys currently in node */
    uint32_t is_leaf;                    /**< 1 if leaf node, 0 if internal */
    uint32_t next_leaf;                  /**< Index of next leaf (for range queries) */
} BTreeNode;

/**
 * @brief B+ Tree index structure
 * * Contains the tree metadata and node array for GPU transfer
 */
typedef struct {
    BTreeNode* nodes;       /**< Array of all tree nodes */
    uint32_t num_nodes;     /**< Total number of nodes allocated */
    uint32_t root_idx;      /**< Index of root node */
    uint32_t height;        /**< Height of the tree */
    uint32_t num_keys;      /**< Total number of keys indexed */
} BTreeIndex;

/**
 * @brief Performance metrics structure
 * * Stores timing information for various phases of the join operation
 */
typedef struct {
    float index_build_time_ms;      /**< Time to build B+ Tree index */
    float data_transfer_time_ms;    /**< Time for CPU-GPU data transfer */
    float join_kernel_time_ms;      /**< Time for join kernel execution */
    float result_transfer_time_ms;  /**< Time to transfer results back */
    float total_time_ms;            /**< Total join operation time */
    uint32_t num_results;           /**< Number of join results produced */
} PerformanceMetrics;

// ============================================================================
// Function Prototypes - Index Construction
// ============================================================================

/**
 * @brief Build a B+ Tree index on the CPU
 * * Constructs a B+ Tree index for the given table. The index is built
 * on the CPU and can then be transferred to the GPU for join operations.
 * * @param tuples Array of tuples to index
 * @param num_tuples Number of tuples in the array
 * @param index Pointer to BTreeIndex structure to populate
 * @return 0 on success, -1 on failure
 */
int build_btree_index(const Tuple* tuples, uint32_t num_tuples, BTreeIndex* index);

/**
 * @brief Free memory allocated for B+ Tree index
 * * @param index Pointer to BTreeIndex structure to free
 */
void free_btree_index(BTreeIndex* index);

// ============================================================================
// Function Prototypes - GPU Join Operations
// ============================================================================

/**
 * @brief Execute Index Nested-Loop Join on GPU
 * * Performs the join operation using the B+ Tree index. Table S (outer)
 * is probed against the indexed table R (inner) using GPU parallelism.
 * * @param h_R Host array of table R tuples (indexed/inner table)
 * @param num_R Number of tuples in table R
 * @param h_S Host array of table S tuples (outer table)
 * @param num_S Number of tuples in table S
 * @param h_index Host B+ Tree index for table R
 * @param h_results Host array to store join results
 * @param max_results Maximum capacity of results array
 * @param metrics Pointer to store performance metrics
 * @return Number of join results found, or -1 on error
 */
int gpu_index_nested_loop_join(
    const Tuple* h_R, uint32_t num_R,
    const Tuple* h_S, uint32_t num_S,
    const BTreeIndex* h_index,
    JoinResult* h_results, uint32_t max_results,
    PerformanceMetrics* metrics
);

/**
 * @brief Execute baseline Hash Join on GPU for comparison
 * * Implements a simple hash join as a baseline for performance comparison.
 * Uses a hash table to probe for matching keys.
 * * @param h_R Host array of table R tuples
 * @param num_R Number of tuples in table R
 * @param h_S Host array of table S tuples
 * @param num_S Number of tuples in table S
 * @param h_results Host array to store join results
 * @param max_results Maximum capacity of results array
 * @param metrics Pointer to store performance metrics
 * @return Number of join results found, or -1 on error
 */
int gpu_hash_join_baseline(
    const Tuple* h_R, uint32_t num_R,
    const Tuple* h_S, uint32_t num_S,
    JoinResult* h_results, uint32_t max_results,
    PerformanceMetrics* metrics
);

// ============================================================================
// Function Prototypes - Utility Functions
// ============================================================================

/**
 * @brief Generate test data with uniformly distributed unique keys
 * * Creates tuples with sequential IDs and unique, uniformly distributed keys.
 * Uses Fisher-Yates shuffle for uniform distribution.
 * * @param tuples Array to populate with generated tuples
 * @param num_tuples Number of tuples to generate
 * @param seed Random seed for reproducibility
 */
void generate_test_data(Tuple* tuples, uint32_t num_tuples, uint32_t seed);

/**
 * @brief Generate test data with controlled overlap for join testing
 * * Creates two tables with a specified percentage of overlapping keys
 * to control the selectivity of the join operation.
 * * @param R Array for table R tuples
 * @param num_R Number of tuples in R
 * @param S Array for table S tuples
 * @param num_S Number of tuples in S
 * @param overlap_percentage Percentage of keys that should match (0-100)
 * @param seed Random seed for reproducibility
 */
void generate_join_test_data(
    Tuple* R, uint32_t num_R,
    Tuple* S, uint32_t num_S,
    float overlap_percentage,
    uint32_t seed
);

/**
 * @brief Verify join results using CPU-based nested loop join
 * * Performs a simple nested loop join on CPU to verify GPU results.
 * * @param R Array of table R tuples
 * @param num_R Number of tuples in R
 * @param S Array of table S tuples
 * @param num_S Number of tuples in S
 * @return Number of matching pairs found
 */
uint32_t verify_join_cpu(
    const Tuple* R, uint32_t num_R,
    const Tuple* S, uint32_t num_S
);

/**
 * @brief Print performance metrics
 * * @param metrics Pointer to PerformanceMetrics structure
 * @param label Descriptive label for the output
 */
void print_metrics(const PerformanceMetrics* metrics, const char* label);

/**
 * @brief Print comparative performance metrics
 */
void print_comparison(
    const PerformanceMetrics* inlj_metrics,
    const PerformanceMetrics* hash_metrics
);

/**
 * @brief Print CUDA device information
 */
void print_gpu_info();

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

/**
 * @brief CUDA error checking macro (void return version)
 */
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

#endif // INLJ_H