/**
 * @file main.cu
 * @brief Main Test Program for Index Nested-Loop Join on GPUs
 * @author Tajaddin
 * @course COP6527 - Computing Massive Parallel Systems
 * @date Fall 2025
 * * This program demonstrates and benchmarks the GPU-based Index Nested-Loop
 * Join implementation compared against a baseline hash join algorithm.
 * * Features:
 * - Pair-level verification of every join against a CPU reference
 * - Performance comparison between INLJ and Hash Join
 * - Scalability testing with various data sizes
 * - Selectivity testing with different join overlap percentages
 */

#include "inlj.h"
#include <string.h>
#include <algorithm>
#include <chrono>

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

/** Untimed runs first: the first launch pays for module loading and for the GPU leaving its idle clocks */
#define WARMUP_RUNS 1

/** Timed runs per join; every reported join time is the median of these */
#define TIMED_RUNS 11

// ============================================================================
// Helpers
// ============================================================================

/**
 * @brief Run one GPU join and check its (r_id, s_id) pairs
 * * Runs the INLJ when index is given, the hash join otherwise, then compares
 * the returned pairs with the CPU reference join.
 * @return 0 if the join ran and returned exactly the expected pairs, -1 otherwise
 */
static int run_join(
    const Tuple* R, const Tuple* S, uint32_t num_tuples,
    const BTreeIndex* index,
    JoinResult* results, uint32_t max_results,
    PerformanceMetrics* metrics
) {
    const char* name = index ? "GPU INLJ" : "GPU hash join";
    int count = index
        ? gpu_index_nested_loop_join(R, num_tuples, S, num_tuples, index,
                                     results, max_results, metrics)
        : gpu_hash_join_baseline(R, num_tuples, S, num_tuples,
                                 results, max_results, metrics);
    if (count < 0) {
        fprintf(stderr, "%s failed\n", name);
        return -1;
    }
    if ((uint32_t)count > max_results) {
        fprintf(stderr, "%s: %d pairs overflow the %u-slot result buffer\n",
                name, count, max_results);
        return -1;
    }
    if (verify_join_results(R, num_tuples, S, num_tuples, results, count) != 0) {
        fprintf(stderr, "%s returned the wrong pairs\n", name);
        return -1;
    }
    return 0;
}

static float median(const float* values, int n) {
    float sorted[TIMED_RUNS];
    memcpy(sorted, values, n * sizeof(float));
    std::sort(sorted, sorted + n);
    return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/**
 * @brief Time one GPU join: warm-up runs, then the median of TIMED_RUNS runs
 * * The first warm-up run's pairs are checked against the CPU reference; the
 * timed runs run back to back and must return the same number of pairs.
 * @return 0 on success, -1 if any run fails or returns wrong pairs
 */
static int benchmark_join(
    const Tuple* R, const Tuple* S, uint32_t num_tuples,
    const BTreeIndex* index,
    JoinResult* results, uint32_t max_results,
    PerformanceMetrics* metrics
) {
    float build[TIMED_RUNS], transfer[TIMED_RUNS], kernel[TIMED_RUNS];
    float result[TIMED_RUNS], total[TIMED_RUNS];
    PerformanceMetrics run;
    memset(&run, 0, sizeof(run));
    
    if (run_join(R, S, num_tuples, index, results, max_results, &run) != 0) return -1;
    uint32_t expected = run.num_results;
    
    for (int i = 1 - WARMUP_RUNS; i < TIMED_RUNS; i++) {
        memset(&run, 0, sizeof(run));
        int count = index
            ? gpu_index_nested_loop_join(R, num_tuples, S, num_tuples, index,
                                         results, max_results, &run)
            : gpu_hash_join_baseline(R, num_tuples, S, num_tuples,
                                     results, max_results, &run);
        if (count < 0 || (uint32_t)count != expected) {
            fprintf(stderr, "Timed run returned %d pairs, expected %u\n", count, expected);
            return -1;
        }
        if (i < 0) continue;
        build[i] = run.index_build_time_ms;
        transfer[i] = run.data_transfer_time_ms;
        kernel[i] = run.join_kernel_time_ms;
        result[i] = run.result_transfer_time_ms;
        total[i] = run.total_time_ms;
    }
    
    metrics->index_build_time_ms = median(build, TIMED_RUNS);
    metrics->data_transfer_time_ms = median(transfer, TIMED_RUNS);
    metrics->join_kernel_time_ms = median(kernel, TIMED_RUNS);
    metrics->result_transfer_time_ms = median(result, TIMED_RUNS);
    metrics->total_time_ms = median(total, TIMED_RUNS);
    metrics->num_results = expected;
    return 0;
}

/**
 * @brief Build the B+ tree on the CPU and return the wall-clock time in ms
 */
static float timed_index_build(const Tuple* R, uint32_t num_tuples, BTreeIndex* index, int* status) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    *status = build_btree_index(R, num_tuples, index);
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * @brief Test 1: Correctness Verification
 * * Verifies that both GPU joins return exactly the (r_id, s_id) pairs
 * of a CPU reference join.
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
    
    PerformanceMetrics inlj_metrics, hash_metrics;
    memset(&inlj_metrics, 0, sizeof(inlj_metrics));
    memset(&hash_metrics, 0, sizeof(hash_metrics));
    
    printf("Running GPU Index Nested-Loop Join and checking its pairs...\n");
    int inlj_ok = run_join(R, S, num_tuples, &index, results, max_results, &inlj_metrics) == 0;
    printf("Running GPU Hash Join and checking its pairs...\n");
    int hash_ok = run_join(R, S, num_tuples, NULL, results, max_results, &hash_metrics) == 0;
    
    // Report results
    printf("\nResults:\n");
    printf("  GPU INLJ pairs:    %u (%s)\n", inlj_metrics.num_results,
           inlj_ok ? "match the CPU reference" : "WRONG");
    printf("  GPU hash pairs:    %u (%s)\n", hash_metrics.num_results,
           hash_ok ? "match the CPU reference" : "WRONG");
    printf("  Expected:          %u\n", expected_results);
    
    int success = inlj_ok && hash_ok;
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
    int index_result;
    float index_time = timed_index_build(R, num_tuples, &index, &index_result);
    
    if (index_result != 0) {
        fprintf(stderr, "Failed to build index\n");
        free(R);
        free(S);
        free(results);
        return -1;
    }
    
    // Run GPU INLJ
    printf("Running GPU Index Nested-Loop Join (%d warm-up + %d timed runs)...\n",
           WARMUP_RUNS, TIMED_RUNS);
    PerformanceMetrics inlj_metrics, hash_metrics;
    memset(&inlj_metrics, 0, sizeof(inlj_metrics));
    memset(&hash_metrics, 0, sizeof(hash_metrics));
    
    int status = benchmark_join(R, S, num_tuples, &index, results, max_results, &inlj_metrics);
    inlj_metrics.index_build_time_ms = index_time;
    inlj_metrics.total_time_ms += index_time;
    
    if (status == 0) {
        print_metrics(&inlj_metrics, "Index Nested-Loop Join");
        printf("Running GPU Hash Join (baseline, %d warm-up + %d timed runs)...\n",
               WARMUP_RUNS, TIMED_RUNS);
        status = benchmark_join(R, S, num_tuples, NULL, results, max_results, &hash_metrics);
    }
    
    if (status == 0) {
        print_metrics(&hash_metrics, "Hash Join (Baseline)");
        print_comparison(&inlj_metrics, &hash_metrics);
        printf(" Join times are medians of %d runs; the index build is one CPU run.\n", TIMED_RUNS);
        printf(" Both joins returned exactly the CPU reference pairs.\n");
    }
    
    // Cleanup
    free(R);
    free(S);
    free(results);
    free_btree_index(&index);
    
    return status;
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
    int status = 0;
    
    printf("\nMedian of %d timed runs after %d warm-up; totals include the CPU index build\n",
           TIMED_RUNS, WARMUP_RUNS);
    printf("Data Size (tuples) | INLJ kernel (ms) | Hash probe (ms) | INLJ total (ms) | Hash total (ms) | Total speedup\n");
    printf("-------------------|------------------|-----------------|-----------------|-----------------|--------------\n");
    
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
        
        int index_result;
        float index_time = timed_index_build(R, num_tuples, &index, &index_result);
        if (index_result == 0) {
            if (benchmark_join(R, S, num_tuples, &index, results, max_results, &inlj_metrics) == 0 &&
                benchmark_join(R, S, num_tuples, NULL, results, max_results, &hash_metrics) == 0) {
                inlj_metrics.index_build_time_ms = index_time;
                inlj_metrics.total_time_ms += index_time;
                float speedup = hash_metrics.total_time_ms / inlj_metrics.total_time_ms;
                
                printf("%18u | %16.3f | %15.3f | %15.2f | %15.2f | %12.2fx\n",
                       num_tuples, inlj_metrics.join_kernel_time_ms,
                       hash_metrics.join_kernel_time_ms, inlj_metrics.total_time_ms,
                       hash_metrics.total_time_ms, speedup);
            } else {
                printf("%18u | FAILED\n", num_tuples);
                status = -1;
            }
            
            free_btree_index(&index);
        } else {
            printf("%18u | SKIPPED (index build failed)\n", num_tuples);
        }
        
        free(R);
        free(S);
        free(results);
    }
    
    return status;
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
    int status = 0;
    
    printf("\nMedian of %d timed runs after %d warm-up; totals include the CPU index build\n",
           TIMED_RUNS, WARMUP_RUNS);
    printf("Selectivity (%%) | INLJ kernel (ms) | Hash probe (ms) | INLJ total (ms) | Hash total (ms) | Results\n");
    printf("----------------|------------------|-----------------|-----------------|-----------------|--------\n");
    
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
        
        int index_result;
        float index_time = timed_index_build(R, num_tuples, &index, &index_result);
        if (index_result == 0) {
            if (benchmark_join(R, S, num_tuples, &index, results, max_results, &inlj_metrics) == 0 &&
                benchmark_join(R, S, num_tuples, NULL, results, max_results, &hash_metrics) == 0) {
                inlj_metrics.index_build_time_ms = index_time;
                inlj_metrics.total_time_ms += index_time;
                
                printf("%15.1f | %16.3f | %15.3f | %15.2f | %15.2f | %u\n",
                       overlap_pct, inlj_metrics.join_kernel_time_ms,
                       hash_metrics.join_kernel_time_ms, inlj_metrics.total_time_ms,
                       hash_metrics.total_time_ms, inlj_metrics.num_results);
            } else {
                printf("%15.1f | FAILED\n", overlap_pct);
                status = -1;
            }
            
            free_btree_index(&index);
        } else {
            printf("%15.1f | SKIPPED (index build failed)\n", overlap_pct);
        }
        
        free(R);
        free(S);
        free(results);
    }
    
    return status;
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
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
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
    printf("  %s\n", result ? "Some tests FAILED" : "All tests passed");
    printf("=============================================================\n");
    
    return result;
}
