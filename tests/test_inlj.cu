/**
 * @file test_inlj.cu
 * @brief Unit tests for the join library. Run with `make test` (needs an NVIDIA GPU).
 */

#include "inlj.h"
#include <string.h>

static int failures = 0;

#define EXPECT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            failures++; \
        } \
    } while(0)

static size_t free_device_memory() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

/**
 * A join whose allocation fails must free what it had already allocated and
 * leave no error behind for the next join.
 */
static void test_failed_join_frees_device_memory() {
    printf("failed join frees device memory\n");
    const uint32_t n = 1000000;
    const long long tolerance = 1 << 20;  // a leak here would be >= 8 MB
    Tuple* R = (Tuple*)malloc(n * sizeof(Tuple));
    Tuple* S = (Tuple*)malloc(n * sizeof(Tuple));
    JoinResult* results = (JoinResult*)malloc(n * sizeof(JoinResult));
    BTreeIndex index;
    generate_join_test_data(R, n, S, n, 50.0f, 1);
    if (build_btree_index(R, n, &index) != 0) {
        EXPECT(0, "index build failed");
        return;
    }

    // UINT32_MAX result slots is about 34 GB, so that allocation fails after
    // the earlier ones succeeded.
    size_t before = free_device_memory();
    EXPECT(gpu_index_nested_loop_join(R, n, S, n, &index, results, UINT32_MAX, NULL) == -1,
           "INLJ with an impossible allocation should fail");
    EXPECT((long long)before - (long long)free_device_memory() < tolerance, "INLJ leaked device memory");
    EXPECT(gpu_hash_join_baseline(R, n, S, n, results, UINT32_MAX, NULL) == -1,
           "hash join with an impossible allocation should fail");
    EXPECT((long long)before - (long long)free_device_memory() < tolerance, "hash join leaked device memory");

    EXPECT(gpu_index_nested_loop_join(R, n, S, n, &index, results, n, NULL) == (int)(n / 2),
           "INLJ after a failed join should succeed");
    EXPECT(gpu_hash_join_baseline(R, n, S, n, results, n, NULL) == (int)(n / 2),
           "hash join after a failed join should succeed");

    free_btree_index(&index);
    free(R);
    free(S);
    free(results);
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA device found\n");
        return 1;
    }

    test_failed_join_frees_device_memory();

    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("All checks passed\n");
    return 0;
}
