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

/**
 * The CPU check must accept the true pairs and reject a missing pair, a
 * duplicate and a wrong partner. The true pairs come from a brute-force join.
 */
static void test_verification_rejects_wrong_missing_and_duplicate_pairs() {
    printf("verification rejects wrong, missing and duplicate pairs\n");
    const uint32_t n = 1000;
    Tuple* R = (Tuple*)malloc(n * sizeof(Tuple));
    Tuple* S = (Tuple*)malloc(n * sizeof(Tuple));
    JoinResult* pairs = (JoinResult*)malloc(n * sizeof(JoinResult));
    uint32_t count = 0;
    generate_join_test_data(R, n, S, n, 50.0f, 3);
    for (uint32_t s = 0; s < n; s++) {
        for (uint32_t r = 0; r < n; r++) {
            if (R[r].key == S[s].key) {
                pairs[count].r_id = R[r].id;
                pairs[count].s_id = S[s].id;
                count++;
            }
        }
    }
    EXPECT(count == n / 2, "half of S should match");
    EXPECT(verify_join_results(R, n, S, n, pairs, count) == 0, "the true pairs should pass");
    EXPECT(verify_join_results(R, n, S, n, pairs, count - 1) != 0, "a missing pair should fail");

    JoinResult last = pairs[count - 1];
    pairs[count - 1] = pairs[0];
    EXPECT(verify_join_results(R, n, S, n, pairs, count) != 0, "a duplicate pair should fail");
    pairs[count - 1] = last;

    pairs[0].r_id = (pairs[0].r_id + 1) % n;
    EXPECT(verify_join_results(R, n, S, n, pairs, count) != 0, "a wrong partner should fail");

    free(R);
    free(S);
    free(pairs);
}

/**
 * Both GPU joins must return exactly the expected pairs for trees of height
 * 1, 3 and 4, and when none, half or all of S matches.
 */
static void test_gpu_joins_return_exact_pairs() {
    printf("GPU joins return exactly the expected pairs\n");
    const uint32_t sizes[] = {50, 5000, 300000};
    const float overlaps[] = {0.0f, 50.0f, 100.0f};
    char what[96];

    for (int i = 0; i < 3; i++) {
        uint32_t n = sizes[i];
        Tuple* R = (Tuple*)malloc(n * sizeof(Tuple));
        Tuple* S = (Tuple*)malloc(n * sizeof(Tuple));
        JoinResult* results = (JoinResult*)malloc(n * sizeof(JoinResult));

        for (int j = 0; j < 3; j++) {
            BTreeIndex index;
            generate_join_test_data(R, n, S, n, overlaps[j], 100 + j);
            if (build_btree_index(R, n, &index) != 0) {
                EXPECT(0, "index build failed");
                continue;
            }

            int count = gpu_index_nested_loop_join(R, n, S, n, &index, results, n, NULL);
            snprintf(what, sizeof(what), "INLJ pairs, n=%u, overlap %.0f%%", n, overlaps[j]);
            EXPECT(count >= 0 && verify_join_results(R, n, S, n, results, count) == 0, what);

            count = gpu_hash_join_baseline(R, n, S, n, results, n, NULL);
            snprintf(what, sizeof(what), "hash join pairs, n=%u, overlap %.0f%%", n, overlaps[j]);
            EXPECT(count >= 0 && verify_join_results(R, n, S, n, results, count) == 0, what);

            free_btree_index(&index);
        }

        free(R);
        free(S);
        free(results);
    }
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA device found\n");
        return 1;
    }

    test_failed_join_frees_device_memory();
    test_verification_rejects_wrong_missing_and_duplicate_pairs();
    test_gpu_joins_return_exact_pairs();

    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("All checks passed\n");
    return 0;
}
