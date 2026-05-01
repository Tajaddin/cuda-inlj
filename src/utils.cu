/**
 * @file utils.cu
 * @brief Utility Functions for Index Nested-Loop Join Testing
 * @author Tajaddin
 * @course COP6527 - Computing Massive Parallel Systems
 * @date Fall 2025
 * * This file contains utility functions for generating test data,
 * verifying join results, and performance measurement.
 */

#include "inlj.h"
#include <string.h>

// ============================================================================
// Random Number Generation
// ============================================================================

/**
 * @brief Simple linear congruential generator for reproducible random numbers
 */
static uint32_t lcg_random(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return *state;
}

/**
 * @brief Fisher-Yates shuffle for uniform distribution
 */
static void shuffle_array(uint32_t* arr, uint32_t n, uint32_t seed) {
    uint32_t state = seed;
    for (uint32_t i = n - 1; i > 0; i--) {
        uint32_t j = lcg_random(&state) % (i + 1);
        uint32_t temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// ============================================================================
// Data Generation Functions
// ============================================================================

/**
 * @brief Generate test data with uniformly distributed unique keys
 * * Creates tuples with sequential IDs (0 to num_tuples-1) and unique keys
 * that are a random permutation of [0, num_tuples-1].
 */
void generate_test_data(Tuple* tuples, uint32_t num_tuples, uint32_t seed) {
    if (!tuples || num_tuples == 0) return;
    
    // Create array of keys [0, num_tuples-1]
    uint32_t* keys = (uint32_t*)malloc(num_tuples * sizeof(uint32_t));
    if (!keys) {
        fprintf(stderr, "Failed to allocate memory for key generation\n");
        return;
    }
    
    for (uint32_t i = 0; i < num_tuples; i++) {
        keys[i] = i;
    }
    
    // Shuffle for uniform distribution
    shuffle_array(keys, num_tuples, seed);
    
    // Create tuples
    for (uint32_t i = 0; i < num_tuples; i++) {
        tuples[i].id = i;
        tuples[i].key = keys[i];
    }
    
    free(keys);
}

/**
 * @brief Generate test data with controlled overlap for join testing
 * * Creates two tables where a specified percentage of keys match.
 * This allows testing join performance at different selectivities.
 * * @param R Array for table R tuples (inner/indexed table)
 * @param num_R Number of tuples in R
 * @param S Array for table S tuples (outer table)
 * @param num_S Number of tuples in S
 * @param overlap_percentage Percentage of S keys that exist in R (0-100)
 * @param seed Random seed for reproducibility
 */
void generate_join_test_data(
    Tuple* R, uint32_t num_R,
    Tuple* S, uint32_t num_S,
    float overlap_percentage,
    uint32_t seed
) {
    if (!R || !S || num_R == 0 || num_S == 0) return;
    
    uint32_t state = seed;
    
    // Generate R with unique keys in range [0, num_R-1]
    uint32_t* r_keys = (uint32_t*)malloc(num_R * sizeof(uint32_t));
    if (!r_keys) {
        fprintf(stderr, "Failed to allocate memory for R keys\n");
        return;
    }
    
    for (uint32_t i = 0; i < num_R; i++) {
        r_keys[i] = i;
    }
    shuffle_array(r_keys, num_R, seed);
    
    for (uint32_t i = 0; i < num_R; i++) {
        R[i].id = i;
        R[i].key = r_keys[i];
    }
    
    // Generate S with controlled overlap
    uint32_t num_matching = (uint32_t)(num_S * overlap_percentage / 100.0f);
    uint32_t num_non_matching = num_S - num_matching;
    
    // Create S keys
    uint32_t* s_keys = (uint32_t*)malloc(num_S * sizeof(uint32_t));
    if (!s_keys) {
        free(r_keys);
        fprintf(stderr, "Failed to allocate memory for S keys\n");
        return;
    }
    
    // First, add matching keys (randomly selected from R's keys)
    uint32_t* r_key_indices = (uint32_t*)malloc(num_R * sizeof(uint32_t));
    if (!r_key_indices) {
        free(r_keys);
        free(s_keys);
        fprintf(stderr, "Failed to allocate memory for R key indices\n");
        return;
    }
    
    for (uint32_t i = 0; i < num_R; i++) {
        r_key_indices[i] = i;
    }
    shuffle_array(r_key_indices, num_R, seed + 1);
    
    for (uint32_t i = 0; i < num_matching; i++) {
        // Pick a random key from R
        uint32_t r_idx = r_key_indices[i % num_R];
        s_keys[i] = r_keys[r_idx];
    }
    
    // Then, add non-matching keys (keys not in R)
    uint32_t non_match_key = num_R; // Start from keys outside R's range
    for (uint32_t i = num_matching; i < num_S; i++) {
        s_keys[i] = non_match_key++;
    }
    
    // Shuffle S keys
    shuffle_array(s_keys, num_S, seed + 2);
    
    // Create S tuples
    for (uint32_t i = 0; i < num_S; i++) {
        S[i].id = i;
        S[i].key = s_keys[i];
    }
    
    free(r_keys);
    free(s_keys);
    free(r_key_indices);
}

// ============================================================================
// Verification Functions
// ============================================================================

/**
 * @brief Verify join results using CPU-based nested loop join
 * * Performs a simple O(n*m) nested loop join to count matching pairs.
 * Used to verify correctness of GPU join results.
 */
uint32_t verify_join_cpu(
    const Tuple* R, uint32_t num_R,
    const Tuple* S, uint32_t num_S
) {
    if (!R || !S || num_R == 0 || num_S == 0) return 0;
    
    // Build a hash set of R keys for O(n+m) verification
    // Using a simple hash table with linear probing
    uint32_t table_size = num_R * 2;
    uint32_t* hash_keys = (uint32_t*)malloc(table_size * sizeof(uint32_t));
    uint32_t* hash_ids = (uint32_t*)malloc(table_size * sizeof(uint32_t));
    
    if (!hash_keys || !hash_ids) {
        if (hash_keys) free(hash_keys);
        if (hash_ids) free(hash_ids);
        fprintf(stderr, "Failed to allocate memory for verification\n");
        return 0;
    }
    
    memset(hash_keys, 0xFF, table_size * sizeof(uint32_t));
    
    // Build hash table for R
    for (uint32_t i = 0; i < num_R; i++) {
        uint32_t key = R[i].key;
        uint32_t hash_idx = key % table_size;
        
        while (hash_keys[hash_idx] != INVALID_INDEX) {
            hash_idx = (hash_idx + 1) % table_size;
        }
        hash_keys[hash_idx] = key;
        hash_ids[hash_idx] = R[i].id;
    }
    
    // Probe with S
    uint32_t match_count = 0;
    for (uint32_t i = 0; i < num_S; i++) {
        uint32_t key = S[i].key;
        uint32_t hash_idx = key % table_size;
        uint32_t start_idx = hash_idx;
        
        while (hash_keys[hash_idx] != INVALID_INDEX) {
            if (hash_keys[hash_idx] == key) {
                match_count++;
                break;
            }
            hash_idx = (hash_idx + 1) % table_size;
            if (hash_idx == start_idx) break;
        }
    }
    
    free(hash_keys);
    free(hash_ids);
    
    return match_count;
}

// ============================================================================
// Performance Reporting Functions
// ============================================================================

/**
 * @brief Print performance metrics in a formatted manner
 */
void print_metrics(const PerformanceMetrics* metrics, const char* label) {
    if (!metrics || !label) return;
    
    printf("\n");
    printf("=============================================================\n");
    printf(" %s Performance Metrics\n", label);
    printf("=============================================================\n");
    printf(" Index/Build Time:     %10.3f ms\n", metrics->index_build_time_ms);
    printf(" Data Transfer Time:   %10.3f ms\n", metrics->data_transfer_time_ms);
    printf(" Join Kernel Time:     %10.3f ms\n", metrics->join_kernel_time_ms);
    printf(" Result Transfer Time: %10.3f ms\n", metrics->result_transfer_time_ms);
    printf("-------------------------------------------------------------\n");
    printf(" Total Time:           %10.3f ms\n", metrics->total_time_ms);
    printf(" Join Results:         %10u pairs\n", metrics->num_results);
    printf("=============================================================\n");
}

/**
 * @brief Print comparative performance metrics
 */
void print_comparison(
    const PerformanceMetrics* inlj_metrics,
    const PerformanceMetrics* hash_metrics
) {
    if (!inlj_metrics || !hash_metrics) return;
    
    printf("\n");
    printf("=============================================================\n");
    printf(" Performance Comparison: INLJ vs Hash Join\n");
    printf("=============================================================\n");
    printf(" Metric                    INLJ         Hash       Speedup\n");
    printf("-------------------------------------------------------------\n");
    
    float kernel_speedup = hash_metrics->join_kernel_time_ms / 
                          inlj_metrics->join_kernel_time_ms;
    float total_speedup = hash_metrics->total_time_ms / 
                         inlj_metrics->total_time_ms;
    
    printf(" Kernel Time (ms)    %10.3f   %10.3f   %7.2fx\n",
           inlj_metrics->join_kernel_time_ms,
           hash_metrics->join_kernel_time_ms,
           kernel_speedup);
    
    printf(" Total Time (ms)     %10.3f   %10.3f   %7.2fx\n",
           inlj_metrics->total_time_ms,
           hash_metrics->total_time_ms,
           total_speedup);
    
    printf(" Results             %10u   %10u\n",
           inlj_metrics->num_results,
           hash_metrics->num_results);
    
    printf("=============================================================\n");
    
    if (kernel_speedup > 1.0f) {
        printf(" INLJ is %.2fx FASTER in kernel execution\n", kernel_speedup);
    } else {
        printf(" Hash Join is %.2fx faster in kernel execution\n", 1.0f / kernel_speedup);
    }
    printf("=============================================================\n");
}

// ============================================================================
// GPU Information Functions
// ============================================================================

/**
 * @brief Print CUDA device information
 */
void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("\n");
    printf("=============================================================\n");
    printf(" CUDA Device Information\n");
    printf("=============================================================\n");
    printf(" Device:               %s\n", prop.name);
    printf(" Compute Capability:   %d.%d\n", prop.major, prop.minor);
    printf(" Total Global Memory:  %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf(" Shared Memory/Block:  %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf(" Max Threads/Block:    %d\n", prop.maxThreadsPerBlock);
    printf(" Warp Size:            %d\n", prop.warpSize);
    printf(" Multiprocessors:      %d\n", prop.multiProcessorCount);
    printf(" Memory Clock Rate:    %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf(" Memory Bus Width:     %d bits\n", prop.memoryBusWidth);
    printf("=============================================================\n");
}