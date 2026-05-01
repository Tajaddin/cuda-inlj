/**
 * @file main.cu
 * @brief Main Test Program for Index Nested-Loop Join on GPUs
 * @author Tajaddin
 * @course COP6527 - Computing Massive Parallel Systems
 * @date Fall 2025
 * * This program demonstrates and benchmarks the GPU-based Index Nested-Loop
 * Join implementation compared against a baseline hash join algorithm.
 * * Features:
 * - Correctness verification against CPU implementation
 * - Performance comparison between INLJ and Hash Join
 * - Scalability testing with various data sizes
 * - Selectivity testing with different join overlap percentages
 */

#include "inlj.h"
#include <string.h>

// ============================================================================
// Test Configuration
// ============================================================================

/** Default sizes for scalability testing */
static const uint32_t TEST_SIZES[] = {
    100000,     // 100K tuples
    500000,     // 500K tuples
    1000000,    // 1M tuples
    2000000,    // 2M tuples
    5000000,    // 5M tuples
    10000000    // 10M tuples
};
static const int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

/** Selectivity percentages for join testing */
static const float SELECTIVITY_PERCENTAGES[] = {
    10.0f,   // 10% match rate
    25.0f,   // 25% match rate
    50.0f,   // 50% match rate
    75.0f,   // 75% match rate
    100.0f   // 100% match rate
};
static const int NUM_SELECTIVITIES = sizeof(SELECTIVITY_PERCENTAGES) / sizeof(SELECTIVITY_PERCENTAGES[0]);

// ============================================================================
// Test Functions
// ============================================================================

/**
 * @brief Test 1: Correctness Verification
 * * Verifies that the GPU INLJ produces the same number of results
 * as a CPU-based reference implementation.
 */
int test_correctness(uint32_t num_tuples, float overlap_pct) {
    printf("\n");
    printf("*************************************************************\n");
    printf("* TEST: Correctness Verification\n");
    printf("* Tuples: %u each table, Overlap: %.1f%%\n", num_tuples, overlap_pct);
    printf("*************************************************************\n");
    
    // Allocate tables
    Tuple* R = (Tuple*)malloc(num_tuples * sizeof(Tuple));
    Tuple* S = (Tuple*)malloc(num_tuples * sizeof(Tuple));
    
    if (!R || !S) {
        fprintf(stderr, "Failed to allocate test data\n");
        if (R) free(R);
        if (S) free(S);
        return -1;
    }
    
    // Generate test data
    printf("Generating test data...\n");
    generate_join_test_data(R, num_tuples, S, num_tuples, overlap_pct, 12345);
    
    // Build index
    printf("Building B+ Tree index...\n");
    BTreeIndex index;
    if (build_btree_index(R, num_tuples, &index) != 0) {
        fprintf(stderr, "Failed to build index\n");
        free(R);
        free(S);
        return -1;
    }
    
    // Allocate results
    uint32_t expected_results = (uint32_t)(num_tuples * overlap_pct / 100.0f);
    uint32_t max_results = expected_results + 1000; // Extra buffer
    JoinResult* results = (JoinResult*)malloc(max_results * sizeof(JoinResult));
    
    if (!results) {
        fprintf(stderr, "Failed to allocate results\n");
        free(R);
        free(S);
        free_btree_index(&index);
        return -1;
    }
    
    // Run GPU INLJ
    printf("Running GPU Index Nested-Loop Join...\n");
    PerformanceMetrics metrics;
    int gpu_result_count = gpu_index_nested_loop_join(
        R, num_tuples, S, num_tuples, &index,
        results, max_results, &metrics
    );
    
    // Verify with CPU
    printf("Verifying with CPU implementation...\n");
    uint32_t cpu_result_count = verify_join_cpu(R, num_tuples, S, num_tuples);
    
    // Report results
    printf("\nResults:\n");
    printf("  GPU INLJ results:  %d\n", gpu_result_count);
    printf("  CPU verification:  %u\n", cpu_result_count);
    printf("  Expected:          ~%u\n", expected_results);
    
    int success = (gpu_result_count == (int)cpu_result_count);
    printf("\n  TEST %s\n", success ? "PASSED ✓" : "FAILED ✗");
    
    // Cleanup
    free(R);
    free(S);
    free(results);
    free_btree_index(&index);
    
    return success ? 0 : -1;
}

/**
 * @brief Test 2: Performance Comparison - INLJ vs Hash Join
 * * Compares the performance of Index Nested-Loop Join against
 * a baseline GPU Hash Join implementation.
 */
int test_performance_comparison(uint32_t num_tuples, float overlap_pct) {
    printf("\n");
    printf("*************************************************************\n");
    printf("* TEST: Performance Comparison (INLJ vs Hash Join)\n");
    printf("* Tuples: %u each table, Overlap: %.1f%%\n", num_tuples, overlap_pct);
    printf("*************************************************************\n");
    
    // Allocate tables
    Tuple* R = (Tuple*)malloc(num_tuples * sizeof(Tuple));
    Tuple* S = (Tuple*)malloc(num_tuples * sizeof(Tuple));
    
    if (!R || !S) {
        fprintf(stderr, "Failed to allocate test data\n");
        if (R) free(R);
        if (S) free(S);
        return -1;
    }
    
    // Generate test data
    printf("Generating test data...\n");
    generate_join_test_data(R, num_tuples, S, num_tuples, overlap_pct, 54321);
    
    // Allocate results
    uint32_t max_results = num_tuples + 1000;
    JoinResult* results = (JoinResult*)malloc(max_results * sizeof(JoinResult));
    
    if (!results) {
        fprintf(stderr, "Failed to allocate results\n");
        free(R);
        free(S);
        return -1;
    }
    
    // Build index for INLJ
    printf("Building B+ Tree index...\n");
    BTreeIndex index;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float index_time;
    
    cudaEventRecord(start);
    int index_result = build_btree_index(R, num_tuples, &index);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&index_time, start, stop);
    
    if (index_result != 0) {
        fprintf(stderr, "Failed to build index\n");
        free(R);
        free(S);
        free(results);
        return -1;
    }
    
    // Run GPU INLJ
    printf("Running GPU Index Nested-Loop Join...\n");
    PerformanceMetrics inlj_metrics;
    memset(&inlj_metrics, 0, sizeof(inlj_metrics));
    inlj_metrics.index_build_time_ms = index_time;
    
    gpu_index_nested_loop_join(
        R, num_tuples, S, num_tuples, &index,
        results, max_results, &inlj_metrics
    );
    inlj_metrics.total_time_ms += index_time;
    
    print_metrics(&inlj_metrics, "Index Nested-Loop Join");
    
    // Run GPU Hash Join baseline
    printf("Running GPU Hash Join (baseline)...\n");
    PerformanceMetrics hash_metrics;
    memset(&hash_metrics, 0, sizeof(hash_metrics));
    
    gpu_hash_join_baseline(
        R, num_tuples, S, num_tuples,
        results, max_results, &hash_metrics
    );
    
    print_metrics(&hash_metrics, "Hash Join (Baseline)");
    
    // Print comparison
    print_comparison(&inlj_metrics, &hash_metrics);
    
    // Cleanup
    free(R);
    free(S);
    free(results);
    free_btree_index(&index);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

/**
 * @brief Test 3: Scalability Test
 * * Tests performance across different data sizes to analyze
 * scalability characteristics of the INLJ implementation.
 */
int test_scalability() {
    printf("\n");
    printf("*************************************************************\n");
    printf("* TEST: Scalability Analysis\n");
    printf("*************************************************************\n");
    
    float overlap_pct = 50.0f;
    
    printf("\nData Size (tuples) | INLJ Time (ms) | Hash Time (ms) | Speedup\n");
    printf("-------------------|----------------|----------------|--------\n");
    
    for (int i = 0; i < NUM_TEST_SIZES; i++) {
        uint32_t num_tuples = TEST_SIZES[i];
        
        // Skip very large sizes if we don't have enough memory
        if (num_tuples > 5000000) {
            size_t required_mem = num_tuples * sizeof(Tuple) * 4; // Rough estimate
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (required_mem > prop.totalGlobalMem * 0.7) {
                printf("%18u | SKIPPED (insufficient GPU memory)\n", num_tuples);
                continue;
            }
        }
        
        // Allocate tables
        Tuple* R = (Tuple*)malloc(num_tuples * sizeof(Tuple));
        Tuple* S = (Tuple*)malloc(num_tuples * sizeof(Tuple));
        
        if (!R || !S) {
            printf("%18u | SKIPPED (memory allocation failed)\n", num_tuples);
            if (R) free(R);
            if (S) free(S);
            continue;
        }
        
        // Generate data
        generate_join_test_data(R, num_tuples, S, num_tuples, overlap_pct, i * 1000);
        
        // Allocate results
        uint32_t max_results = num_tuples + 1000;
        JoinResult* results = (JoinResult*)malloc(max_results * sizeof(JoinResult));
        
        if (!results) {
            printf("%18u | SKIPPED (results allocation failed)\n", num_tuples);
            free(R);
            free(S);
            continue;
        }
        
        // Build index and run INLJ
        BTreeIndex index;
        PerformanceMetrics inlj_metrics, hash_metrics;
        memset(&inlj_metrics, 0, sizeof(inlj_metrics));
        memset(&hash_metrics, 0, sizeof(hash_metrics));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float index_time;
        
        cudaEventRecord(start);
        if (build_btree_index(R, num_tuples, &index) == 0) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&index_time, start, stop);
            
            inlj_metrics.index_build_time_ms = index_time;
            
            gpu_index_nested_loop_join(
                R, num_tuples, S, num_tuples, &index,
                results, max_results, &inlj_metrics
            );
            inlj_metrics.total_time_ms += index_time;
            
            gpu_hash_join_baseline(
                R, num_tuples, S, num_tuples,
                results, max_results, &hash_metrics
            );
            
            float speedup = hash_metrics.total_time_ms / inlj_metrics.total_time_ms;
            
            printf("%18u | %14.2f | %14.2f | %6.2fx\n",
                   num_tuples, inlj_metrics.total_time_ms, 
                   hash_metrics.total_time_ms, speedup);
            
            free_btree_index(&index);
        } else {
            printf("%18u | SKIPPED (index build failed)\n", num_tuples);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(R);
        free(S);
        free(results);
    }
    
    return 0;
}

/**
 * @brief Test 4: Selectivity Test
 * * Tests performance with varying join selectivities to understand
 * how the algorithms behave with different matching rates.
 */
int test_selectivity() {
    printf("\n");
    printf("*************************************************************\n");
    printf("* TEST: Selectivity Analysis\n");
    printf("*************************************************************\n");
    
    uint32_t num_tuples = 1000000; // 1M tuples
    
    printf("\nSelectivity (%%) | INLJ Time (ms) | Hash Time (ms) | Results\n");
    printf("----------------|----------------|----------------|--------\n");
    
    for (int i = 0; i < NUM_SELECTIVITIES; i++) {
        float overlap_pct = SELECTIVITY_PERCENTAGES[i];
        
        // Allocate tables
        Tuple* R = (Tuple*)malloc(num_tuples * sizeof(Tuple));
        Tuple* S = (Tuple*)malloc(num_tuples * sizeof(Tuple));
        
        if (!R || !S) {
            printf("%15.1f | SKIPPED (memory allocation failed)\n", overlap_pct);
            if (R) free(R);
            if (S) free(S);
            continue;
        }
        
        // Generate data
        generate_join_test_data(R, num_tuples, S, num_tuples, overlap_pct, (int)overlap_pct);
        
        // Allocate results
        uint32_t max_results = num_tuples + 1000;
        JoinResult* results = (JoinResult*)malloc(max_results * sizeof(JoinResult));
        
        if (!results) {
            printf("%15.1f | SKIPPED (results allocation failed)\n", overlap_pct);
            free(R);
            free(S);
            continue;
        }
        
        // Build index and run tests
        BTreeIndex index;
        PerformanceMetrics inlj_metrics, hash_metrics;
        memset(&inlj_metrics, 0, sizeof(inlj_metrics));
        memset(&hash_metrics, 0, sizeof(hash_metrics));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float index_time;
        
        cudaEventRecord(start);
        if (build_btree_index(R, num_tuples, &index) == 0) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&index_time, start, stop);
            
            inlj_metrics.index_build_time_ms = index_time;
            
            gpu_index_nested_loop_join(
                R, num_tuples, S, num_tuples, &index,
                results, max_results, &inlj_metrics
            );
            inlj_metrics.total_time_ms += index_time;
            
            gpu_hash_join_baseline(
                R, num_tuples, S, num_tuples,
                results, max_results, &hash_metrics
            );
            
            printf("%15.1f | %14.2f | %14.2f | %u\n",
                   overlap_pct, inlj_metrics.total_time_ms,
                   hash_metrics.total_time_ms, inlj_metrics.num_results);
            
            free_btree_index(&index);
        } else {
            printf("%15.1f | SKIPPED (index build failed)\n", overlap_pct);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(R);
        free(S);
        free(results);
    }
    
    return 0;
}

// ============================================================================
// Main Function
// ============================================================================

/**
 * @brief Program entry point
 * * Runs a comprehensive test suite including:
 * 1. GPU device information
 * 2. Correctness verification
 * 3. Performance comparison
 * 4. Scalability analysis
 * 5. Selectivity analysis
 */
int main(int argc, char* argv[]) {
    printf("\n");
    printf("=============================================================\n");
    printf("  GPU Index Nested-Loop Join Implementation\n");
    printf("  COP6527 - Computing Massive Parallel Systems\n");
    printf("  Fall 2025\n");
    printf("=============================================================\n");
    
    // Check for CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found!\n");
        return 1;
    }
    
    // Print GPU information
    print_gpu_info();
    
    // Parse command line arguments
    int run_all = 1;
    int test_num = 0;
    
    if (argc > 1) {
        test_num = atoi(argv[1]);
        run_all = 0;
    }
    
    int result = 0;
    
    // Run tests
    if (run_all || test_num == 1) {
        // Test 1: Correctness with small data
        result |= test_correctness(100000, 50.0f);
    }
    
    if (run_all || test_num == 2) {
        // Test 2: Performance comparison
        result |= test_performance_comparison(1000000, 50.0f);
    }
    
    if (run_all || test_num == 3) {
        // Test 3: Scalability test
        result |= test_scalability();
    }
    
    if (run_all || test_num == 4) {
        // Test 4: Selectivity test
        result |= test_selectivity();
    }
    
    printf("\n");
    printf("=============================================================\n");
    printf("  All tests completed!\n");
    printf("=============================================================\n");
    
    return result;
}