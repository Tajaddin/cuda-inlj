/**
 * @file btree_index.cu
 * @brief B+ Tree Index Implementation for GPU-based Index Nested-Loop Join
 * @author Tajaddin
 * @course COP6527 - Computing Massive Parallel Systems
 * @date Fall 2025
 * * This file implements a B+ Tree index structure optimized for GPU-based
 * join operations. The index is built on the CPU and transferred to GPU
 * memory for parallel query processing.
 */

#include "inlj.h"
#include <string.h>
#include <algorithm>

// ============================================================================
// Internal Data Structures for Index Construction
// ============================================================================

/**
 * @brief Dynamic array for B+ Tree nodes during construction
 */
typedef struct {
    BTreeNode* nodes;
    uint32_t capacity;
    uint32_t count;
} NodeArray;

// ============================================================================
// Helper Functions for Index Construction
// ============================================================================

/**
 * @brief Initialize a dynamic node array
 */
static int init_node_array(NodeArray* arr, uint32_t initial_capacity) {
    arr->nodes = (BTreeNode*)malloc(initial_capacity * sizeof(BTreeNode));
    if (!arr->nodes) return -1;
    arr->capacity = initial_capacity;
    arr->count = 0;
    return 0;
}

/**
 * @brief Allocate a new node in the array
 */
static uint32_t allocate_node(NodeArray* arr) {
    if (arr->count >= arr->capacity) {
        uint32_t new_capacity = arr->capacity * 2;
        BTreeNode* new_nodes = (BTreeNode*)realloc(arr->nodes, 
                                                    new_capacity * sizeof(BTreeNode));
        if (!new_nodes) return INVALID_INDEX;
        arr->nodes = new_nodes;
        arr->capacity = new_capacity;
    }
    
    uint32_t idx = arr->count++;
    memset(&arr->nodes[idx], 0, sizeof(BTreeNode));
    arr->nodes[idx].next_leaf = INVALID_INDEX;
    return idx;
}

/**
 * @brief Comparison function for sorting tuples by key
 */
static int compare_tuples(const void* a, const void* b) {
    const Tuple* t1 = (const Tuple*)a;
    const Tuple* t2 = (const Tuple*)b;
    if (t1->key < t2->key) return -1;
    if (t1->key > t2->key) return 1;
    return 0;
}

// ============================================================================
// B+ Tree Construction (Bulk Loading)
// ============================================================================

/**
 * @brief Build B+ Tree using bulk loading algorithm
 * * This implementation uses a bottom-up bulk loading approach:
 * 1. Sort all tuples by key
 * 2. Create leaf nodes with sorted keys
 * 3. Build internal nodes level by level
 * * This is more efficient than repeated insertions for large datasets.
 */
int build_btree_index(const Tuple* tuples, uint32_t num_tuples, BTreeIndex* index) {
    if (!tuples || num_tuples == 0 || !index) {
        fprintf(stderr, "Invalid parameters for build_btree_index\n");
        return -1;
    }
    
    // Create sorted copy of tuples
    Tuple* sorted_tuples = (Tuple*)malloc(num_tuples * sizeof(Tuple));
    if (!sorted_tuples) {
        fprintf(stderr, "Failed to allocate memory for sorted tuples\n");
        return -1;
    }
    memcpy(sorted_tuples, tuples, num_tuples * sizeof(Tuple));
    qsort(sorted_tuples, num_tuples, sizeof(Tuple), compare_tuples);
    
    // Initialize node array
    // Estimate: ~2 * (num_tuples / keys_per_leaf) nodes should be sufficient
    uint32_t estimated_nodes = (num_tuples / BTREE_MIN_KEYS) * 2 + 100;
    NodeArray node_arr;
    if (init_node_array(&node_arr, estimated_nodes) != 0) {
        free(sorted_tuples);
        fprintf(stderr, "Failed to initialize node array\n");
        return -1;
    }
    
    // ========================================================================
    // Step 1: Create leaf nodes
    // ========================================================================
    
    uint32_t keys_per_leaf = BTREE_MAX_KEYS;
    uint32_t num_leaves = (num_tuples + keys_per_leaf - 1) / keys_per_leaf;
    
    uint32_t* leaf_indices = (uint32_t*)malloc(num_leaves * sizeof(uint32_t));
    if (!leaf_indices) {
        free(sorted_tuples);
        free(node_arr.nodes);
        fprintf(stderr, "Failed to allocate leaf indices\n");
        return -1;
    }
    
    uint32_t tuple_idx = 0;
    uint32_t prev_leaf = INVALID_INDEX;
    
    for (uint32_t i = 0; i < num_leaves; i++) {
        uint32_t leaf_idx = allocate_node(&node_arr);
        if (leaf_idx == INVALID_INDEX) {
            free(sorted_tuples);
            free(leaf_indices);
            free(node_arr.nodes);
            fprintf(stderr, "Failed to allocate leaf node\n");
            return -1;
        }
        
        leaf_indices[i] = leaf_idx;
        BTreeNode* leaf = &node_arr.nodes[leaf_idx];
        leaf->is_leaf = 1;
        
        // Fill leaf with keys and tuple IDs
        uint32_t keys_in_this_leaf = 0;
        while (tuple_idx < num_tuples && keys_in_this_leaf < keys_per_leaf) {
            leaf->keys[keys_in_this_leaf] = sorted_tuples[tuple_idx].key;
            leaf->values[keys_in_this_leaf] = sorted_tuples[tuple_idx].id;
            keys_in_this_leaf++;
            tuple_idx++;
        }
        leaf->num_keys = keys_in_this_leaf;
        
        // Link leaves for range queries
        if (prev_leaf != INVALID_INDEX) {
            node_arr.nodes[prev_leaf].next_leaf = leaf_idx;
        }
        prev_leaf = leaf_idx;
    }
    
    free(sorted_tuples);
    
    // ========================================================================
    // Step 2: Build internal nodes level by level
    // ========================================================================
    
    uint32_t* current_level = leaf_indices;
    uint32_t current_level_size = num_leaves;
    uint32_t height = 1;
    
    while (current_level_size > 1) {
        uint32_t keys_per_internal = BTREE_MAX_KEYS;
        uint32_t children_per_internal = BTREE_ORDER;
        uint32_t num_internal = (current_level_size + children_per_internal - 1) / children_per_internal;
        
        uint32_t* next_level = (uint32_t*)malloc(num_internal * sizeof(uint32_t));
        if (!next_level) {
            free(current_level);
            free(node_arr.nodes);
            fprintf(stderr, "Failed to allocate next level indices\n");
            return -1;
        }
        
        uint32_t child_idx = 0;
        for (uint32_t i = 0; i < num_internal; i++) {
            uint32_t internal_idx = allocate_node(&node_arr);
            if (internal_idx == INVALID_INDEX) {
                free(current_level);
                free(next_level);
                free(node_arr.nodes);
                fprintf(stderr, "Failed to allocate internal node\n");
                return -1;
            }
            
            next_level[i] = internal_idx;
            BTreeNode* internal = &node_arr.nodes[internal_idx];
            internal->is_leaf = 0;
            
            // Add children and separator keys
            uint32_t num_children = 0;
            while (child_idx < current_level_size && num_children < children_per_internal) {
                internal->values[num_children] = current_level[child_idx];
                
                // Add separator key (first key of child, except for first child)
                if (num_children > 0) {
                    BTreeNode* child = &node_arr.nodes[current_level[child_idx]];
                    internal->keys[num_children - 1] = child->keys[0];
                }
                
                num_children++;
                child_idx++;
            }
            
            internal->num_keys = (num_children > 0) ? num_children - 1 : 0;
        }
        
        free(current_level);
        current_level = next_level;
        current_level_size = num_internal;
        height++;
    }
    
    // ========================================================================
    // Step 3: Finalize index structure
    // ========================================================================
    
    index->nodes = node_arr.nodes;
    index->num_nodes = node_arr.count;
    index->root_idx = current_level[0];
    index->height = height;
    index->num_keys = num_tuples;
    
    free(current_level);
    
    printf("B+ Tree Index Built:\n");
    printf("  - Total nodes: %u\n", index->num_nodes);
    printf("  - Tree height: %u\n", index->height);
    printf("  - Keys indexed: %u\n", index->num_keys);
    
    return 0;
}

/**
 * @brief Free B+ Tree index memory
 */
void free_btree_index(BTreeIndex* index) {
    if (index && index->nodes) {
        free(index->nodes);
        index->nodes = NULL;
        index->num_nodes = 0;
        index->root_idx = INVALID_INDEX;
        index->height = 0;
        index->num_keys = 0;
    }
}

// ============================================================================
// GPU Device Functions for B+ Tree Search
// ============================================================================

/**
 * @brief Device function to search for a key in the B+ Tree
 * * Traverses the B+ Tree from root to leaf to find the tuple ID
 * corresponding to the given key.
 * * @param nodes Device pointer to B+ Tree nodes
 * @param root_idx Index of root node
 * @param search_key Key to search for
 * @return Tuple ID if found, INVALID_INDEX otherwise
 */
__device__ uint32_t btree_search(const BTreeNode* nodes, uint32_t root_idx, uint32_t search_key) {
    uint32_t current_idx = root_idx;
    
    // Traverse to leaf
    while (!nodes[current_idx].is_leaf) {
        const BTreeNode* node = &nodes[current_idx];
        uint32_t child_idx = node->num_keys; // Default to last child
        
        // Binary search for the correct child
        uint32_t left = 0, right = node->num_keys;
        while (left < right) {
            uint32_t mid = (left + right) / 2;
            if (search_key < node->keys[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        child_idx = left;
        
        current_idx = node->values[child_idx];
    }
    
    // Search in leaf node
    const BTreeNode* leaf = &nodes[current_idx];
    
    // Binary search in leaf
    uint32_t left = 0, right = leaf->num_keys;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (leaf->keys[mid] < search_key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    if (left < leaf->num_keys && leaf->keys[left] == search_key) {
        return leaf->values[left];
    }
    
    return INVALID_INDEX;
}

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * @brief CUDA Kernel for Index Nested-Loop Join
 * * Each thread processes one tuple from table S (outer table),
 * searches the B+ Tree index for matching key in table R,
 * and outputs the join result if found.
 * * @param d_S Device pointer to table S tuples
 * @param num_S Number of tuples in S
 * @param d_nodes Device pointer to B+ Tree nodes
 * @param root_idx Root node index
 * @param d_results Device pointer for join results
 * @param d_result_count Device pointer for atomic result counter
 * @param max_results Maximum number of results to store
 */
__global__ void inlj_kernel(
    const Tuple* d_S,
    uint32_t num_S,
    const BTreeNode* d_nodes,
    uint32_t root_idx,
    JoinResult* d_results,
    uint32_t* d_result_count,
    uint32_t max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = tid; i < num_S; i += stride) {
        uint32_t search_key = d_S[i].key;
        uint32_t r_id = btree_search(d_nodes, root_idx, search_key);
        
        if (r_id != INVALID_INDEX) {
            // Found a match, atomically add result
            uint32_t result_idx = atomicAdd(d_result_count, 1);
            if (result_idx < max_results) {
                d_results[result_idx].r_id = r_id;
                d_results[result_idx].s_id = d_S[i].id;
            }
        }
    }
}

/**
 * @brief Hash function for hash join baseline
 */
__device__ __host__ inline uint32_t hash_func(uint32_t key, uint32_t table_size) {
    // MurmurHash3 finalizer
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key % table_size;
}

/**
 * @brief CUDA Kernel for building hash table (baseline comparison)
 */
__global__ void build_hash_table_kernel(
    const Tuple* d_R,
    uint32_t num_R,
    uint32_t* d_hash_table,
    uint32_t* d_hash_values,
    uint32_t table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = tid; i < num_R; i += stride) {
        uint32_t key = d_R[i].key;
        uint32_t hash_idx = hash_func(key, table_size);
        
        // Linear probing
        while (true) {
            uint32_t old = atomicCAS(&d_hash_table[hash_idx], INVALID_INDEX, key);
            if (old == INVALID_INDEX || old == key) {
                d_hash_values[hash_idx] = d_R[i].id;
                break;
            }
            hash_idx = (hash_idx + 1) % table_size;
        }
    }
}

/**
 * @brief CUDA Kernel for hash join probe phase (baseline comparison)
 */
__global__ void hash_join_probe_kernel(
    const Tuple* d_S,
    uint32_t num_S,
    const uint32_t* d_hash_table,
    const uint32_t* d_hash_values,
    uint32_t table_size,
    JoinResult* d_results,
    uint32_t* d_result_count,
    uint32_t max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = tid; i < num_S; i += stride) {
        uint32_t key = d_S[i].key;
        uint32_t hash_idx = hash_func(key, table_size);
        uint32_t start_idx = hash_idx;
        
        // Linear probing to find key
        while (d_hash_table[hash_idx] != INVALID_INDEX) {
            if (d_hash_table[hash_idx] == key) {
                uint32_t result_idx = atomicAdd(d_result_count, 1);
                if (result_idx < max_results) {
                    d_results[result_idx].r_id = d_hash_values[hash_idx];
                    d_results[result_idx].s_id = d_S[i].id;
                }
                break;
            }
            hash_idx = (hash_idx + 1) % table_size;
            if (hash_idx == start_idx) break; // Full loop
        }
    }
}

// ============================================================================
// GPU Join Functions
// ============================================================================

/**
 * @brief Execute Index Nested-Loop Join on GPU
 */
int gpu_index_nested_loop_join(
    const Tuple* h_R, uint32_t num_R,
    const Tuple* h_S, uint32_t num_S,
    const BTreeIndex* h_index,
    JoinResult* h_results, uint32_t max_results,
    PerformanceMetrics* metrics
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device pointers
    Tuple* d_S = NULL;
    BTreeNode* d_nodes = NULL;
    JoinResult* d_results = NULL;
    uint32_t* d_result_count = NULL;
    
    float transfer_time = 0, kernel_time = 0, result_time = 0;
    
    // ========================================================================
    // Allocate and transfer data to GPU
    // ========================================================================
    
    cudaEventRecord(start);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_S, num_S * sizeof(Tuple)));
    CUDA_CHECK(cudaMalloc(&d_nodes, h_index->num_nodes * sizeof(BTreeNode)));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(JoinResult)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    
    // Transfer data
    CUDA_CHECK(cudaMemcpy(d_S, h_S, num_S * sizeof(Tuple), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, h_index->nodes, 
                          h_index->num_nodes * sizeof(BTreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(uint32_t)));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_time, start, stop);
    
    // ========================================================================
    // Execute join kernel
    // ========================================================================
    
    int num_blocks = (num_S + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_blocks = min(num_blocks, 65535); // Limit grid size
    
    cudaEventRecord(start);
    
    inlj_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_S, num_S, d_nodes, h_index->root_idx,
        d_results, d_result_count, max_results
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // ========================================================================
    // Transfer results back
    // ========================================================================
    
    cudaEventRecord(start);
    
    uint32_t result_count;
    CUDA_CHECK(cudaMemcpy(&result_count, d_result_count, sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost));
    
    uint32_t results_to_copy = min(result_count, max_results);
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_to_copy * sizeof(JoinResult),
                          cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result_time, start, stop);
    
    // ========================================================================
    // Cleanup and return
    // ========================================================================
    
    cudaFree(d_S);
    cudaFree(d_nodes);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Record metrics
    if (metrics) {
        metrics->data_transfer_time_ms = transfer_time;
        metrics->join_kernel_time_ms = kernel_time;
        metrics->result_transfer_time_ms = result_time;
        metrics->total_time_ms = transfer_time + kernel_time + result_time;
        metrics->num_results = result_count;
    }
    
    return result_count;
}

/**
 * @brief Execute baseline Hash Join on GPU
 */
int gpu_hash_join_baseline(
    const Tuple* h_R, uint32_t num_R,
    const Tuple* h_S, uint32_t num_S,
    JoinResult* h_results, uint32_t max_results,
    PerformanceMetrics* metrics
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Device pointers
    Tuple* d_R = NULL;
    Tuple* d_S = NULL;
    uint32_t* d_hash_table = NULL;
    uint32_t* d_hash_values = NULL;
    JoinResult* d_results = NULL;
    uint32_t* d_result_count = NULL;
    
    float build_time = 0, transfer_time = 0, kernel_time = 0, result_time = 0;
    
    // Hash table size (load factor ~0.5)
    uint32_t table_size = num_R * 2;
    
    // ========================================================================
    // Allocate and transfer data to GPU
    // ========================================================================
    
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMalloc(&d_R, num_R * sizeof(Tuple)));
    CUDA_CHECK(cudaMalloc(&d_S, num_S * sizeof(Tuple)));
    CUDA_CHECK(cudaMalloc(&d_hash_table, table_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hash_values, table_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(JoinResult)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_R, h_R, num_R * sizeof(Tuple), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, num_S * sizeof(Tuple), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, table_size * sizeof(uint32_t))); // INVALID_INDEX
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(uint32_t)));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_time, start, stop);
    
    // ========================================================================
    // Build hash table
    // ========================================================================
    
    int num_blocks_build = (num_R + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_blocks_build = min(num_blocks_build, 65535);
    
    cudaEventRecord(start);
    
    build_hash_table_kernel<<<num_blocks_build, THREADS_PER_BLOCK>>>(
        d_R, num_R, d_hash_table, d_hash_values, table_size
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&build_time, start, stop);
    
    // ========================================================================
    // Probe phase
    // ========================================================================
    
    int num_blocks_probe = (num_S + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_blocks_probe = min(num_blocks_probe, 65535);
    
    cudaEventRecord(start);
    
    hash_join_probe_kernel<<<num_blocks_probe, THREADS_PER_BLOCK>>>(
        d_S, num_S, d_hash_table, d_hash_values, table_size,
        d_results, d_result_count, max_results
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // ========================================================================
    // Transfer results back
    // ========================================================================
    
    cudaEventRecord(start);
    
    uint32_t result_count;
    CUDA_CHECK(cudaMemcpy(&result_count, d_result_count, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    
    uint32_t results_to_copy = min(result_count, max_results);
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_to_copy * sizeof(JoinResult),
                          cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result_time, start, stop);
    
    // ========================================================================
    // Cleanup and return
    // ========================================================================
    
    cudaFree(d_R);
    cudaFree(d_S);
    cudaFree(d_hash_table);
    cudaFree(d_hash_values);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Record metrics
    if (metrics) {
        metrics->index_build_time_ms = build_time;
        metrics->data_transfer_time_ms = transfer_time;
        metrics->join_kernel_time_ms = kernel_time;
        metrics->result_transfer_time_ms = result_time;
        metrics->total_time_ms = transfer_time + build_time + kernel_time + result_time;
        metrics->num_results = result_count;
    }
    
    return result_count;
}