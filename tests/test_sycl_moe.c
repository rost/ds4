/* Correctness tests for the routed-MoE SYCL kernels.  Self-contained (no
 * tests/test_sycl_harness.h on this branch's base commit): defines its own
 * CHECK/CHECK_CLOSE matching the semantics used in tests/test_sycl_kernels.c.
 * Scalar oracles here reimplement the documented ROCm/ds4.c formulas with
 * line numbers cited; ds4.c's static functions cannot be linked directly,
 * matching the existing precedent in test_sycl_kernels.c. Needs no model
 * file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK_CLOSE(got, want, tol, msg)                                    \
    do {                                                                    \
        double d_ = fabs((double)(got) - (double)(want));                   \
        if (!(d_ <= (tol))) {                                               \
            fprintf(stderr, "FAIL: %s (got %.9g want %.9g delta %.3g)\n",   \
                    (msg), (double)(got), (double)(want), d_);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Must match the device-side poison pattern in
 * sycl_moe_build_sorted_pairs's poison_for_test path (0xAB every byte). */
enum { SENTINEL = 0xABABABABu };

/* Test-only side door into sycl/ds4_sycl_moe.hpp, matching the precedent
 * set by sycl/ds4_sycl_streaming.hpp's ds4_sycl_stream_test_* functions.
 * Runs the full counting sort (rocm/ds4_rocm_moe.cuh:1255-1316) plus the
 * expert-tile build (moe.cuh:1327-1350) and reads every intermediate
 * array back to the host. */
extern int ds4_sycl_moe_test_sort(const int32_t *selected, uint32_t pair_count,
                                  uint32_t n_total_expert, uint32_t block_m,
                                  uint32_t *counts_out, uint32_t *offsets_out,
                                  uint32_t *sorted_pairs_out,
                                  uint32_t *tile_total_out,
                                  uint32_t *tile_experts_out,
                                  uint32_t *tile_starts_out,
                                  uint32_t tile_capacity_cap);

static uint32_t sort_tile_capacity(uint32_t pair_count, uint32_t n_total_expert,
                                   uint32_t block_m) {
    return (pair_count + block_m - 1u) / block_m + n_total_expert;
}

/* Scalar oracle for the counting sort, cited to rocm/ds4_rocm_moe.cuh:
 * 1255-1316 and to ds4.c:11126-11166 (layer_routed_moe_batch's own
 * scalar sort, which the GPU sort structurally mirrors). */
static void oracle_sort(const int32_t *selected, uint32_t pair_count,
                        uint32_t n_total_expert, uint32_t *counts,
                        uint32_t *offsets, uint32_t *sorted_pairs) {
    memset(counts, 0, (size_t)n_total_expert * sizeof(uint32_t));
    for (uint32_t p = 0; p < pair_count; p++) {
        int32_t e = selected[p];
        if (e < 0) e = 0;
        if ((uint32_t)e >= n_total_expert) continue;
        counts[(uint32_t)e]++;
    }
    uint32_t sum = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        offsets[e] = sum;
        sum += counts[e];
    }
    offsets[n_total_expert] = sum;
    uint32_t *cursor = malloc((size_t)n_total_expert * sizeof(uint32_t));
    memcpy(cursor, offsets, (size_t)n_total_expert * sizeof(uint32_t));
    for (uint32_t p = 0; p < pair_count; p++) {
        int32_t e = selected[p];
        if (e < 0) e = 0;
        if ((uint32_t)e >= n_total_expert) continue;
        sorted_pairs[cursor[(uint32_t)e]++] = p;
    }
    free(cursor);
}

static int test_sort_basic(void) {
    /* n_total_expert=6, expert 0 selected by no pair, expert 3 selected
     * by every one of the 5 tokens (n_expert slots = 2, so pair index =
     * tok*2+slot), and pair-index order within a bucket must come out
     * strictly ascending (the "_deterministic" guarantee: a scan over
     * pairs 0..pair_count-1 in order, never an atomic-append race). */
    enum { N_TOK = 5, N_SLOT = 2, N_TOTAL_EXPERT = 6 };
    const uint32_t pair_count = N_TOK * N_SLOT;
    int32_t selected[N_TOK * N_SLOT];
    /* Every token selects expert 3 in slot 0, and a rotating expert among
     * {1,2,4,5} in slot 1.  Expert 0 is never selected. */
    const int32_t rotating[N_TOK] = {1, 2, 4, 5, 1};
    for (int t = 0; t < N_TOK; t++) {
        selected[t * N_SLOT + 0] = 3;
        selected[t * N_SLOT + 1] = rotating[t];
    }

    uint32_t oc[N_TOTAL_EXPERT], oo[N_TOTAL_EXPERT + 1], osp[N_TOK * N_SLOT];
    oracle_sort(selected, pair_count, N_TOTAL_EXPERT, oc, oo, osp);
    CHECK(oc[0] == 0u, "sort_basic: expert 0 selected by no pair");
    CHECK(oc[3] == N_TOK, "sort_basic: expert 3 selected by every token");

    const uint32_t block_m = 8u;
    const uint32_t tile_cap = sort_tile_capacity(pair_count, N_TOTAL_EXPERT, block_m);
    uint32_t counts[N_TOTAL_EXPERT], offsets[N_TOTAL_EXPERT + 1];
    uint32_t sorted_pairs[N_TOK * N_SLOT];
    uint32_t tile_total = SENTINEL;
    uint32_t *tile_experts = malloc((size_t)tile_cap * sizeof(uint32_t));
    uint32_t *tile_starts = malloc((size_t)tile_cap * sizeof(uint32_t));
    for (uint32_t i = 0; i < tile_cap; i++) {
        tile_experts[i] = SENTINEL;
        tile_starts[i] = SENTINEL;
    }
    for (uint32_t i = 0; i < pair_count; i++) sorted_pairs[i] = SENTINEL;

    CHECK(ds4_sycl_moe_test_sort(selected, pair_count, N_TOTAL_EXPERT, block_m,
                                 counts, offsets, sorted_pairs, &tile_total,
                                 tile_experts, tile_starts, tile_cap) != 0,
          "sort_basic: ds4_sycl_moe_test_sort failed");

    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sort_basic: counts[%u] mismatch", e);
        CHECK(counts[e] == oc[e], msg);
        snprintf(msg, sizeof(msg), "sort_basic: offsets[%u] mismatch", e);
        CHECK(offsets[e] == oo[e], msg);
    }
    CHECK(offsets[N_TOTAL_EXPERT] == oo[N_TOTAL_EXPERT],
          "sort_basic: final offset mismatch");
    for (uint32_t p = 0; p < pair_count; p++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sort_basic: sorted_pairs[%u] mismatch", p);
        CHECK(sorted_pairs[p] == osp[p], msg);
    }

    /* Deterministic order: within expert 3's bucket (5 entries), the
     * scatter must produce pairs in strictly ascending pair-index order,
     * since a per-expert sequential scan (not a racing atomic append)
     * is what "_deterministic" means. */
    uint32_t prev = 0;
    for (uint32_t i = offsets[3]; i < offsets[3] + counts[3]; i++) {
        if (i > offsets[3]) {
            CHECK(sorted_pairs[i] > prev,
                  "sort_basic: expert 3 bucket not in ascending pair order");
        }
        prev = sorted_pairs[i];
    }

    /* Tile build: verify tile_total, and every filled tile_experts/
     * tile_starts entry against a hand oracle; everything from tile_total
     * up to tile_cap must remain the sentinel (unwritten), proving the
     * kernel does not spill past its own tile count. */
    uint32_t expect_tile_total = 0;
    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        expect_tile_total += (counts[e] + block_m - 1u) / block_m;
    }
    CHECK(tile_total == expect_tile_total, "sort_basic: tile_total mismatch");
    uint32_t tile_idx = 0;
    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
        for (uint32_t t = 0; t < ntiles; t++) {
            char msg[80];
            snprintf(msg, sizeof(msg), "sort_basic: tile_experts[%u] mismatch", tile_idx);
            CHECK(tile_experts[tile_idx] == e, msg);
            snprintf(msg, sizeof(msg), "sort_basic: tile_starts[%u] mismatch", tile_idx);
            CHECK(tile_starts[tile_idx] == t * block_m, msg);
            tile_idx++;
        }
    }
    CHECK(tile_idx == tile_total, "sort_basic: tile walk did not reach tile_total");
    for (uint32_t i = tile_total; i < tile_cap; i++) {
        CHECK(tile_experts[i] == SENTINEL, "sort_basic: tile_experts spilled past tile_total");
        CHECK(tile_starts[i] == SENTINEL, "sort_basic: tile_starts spilled past tile_total");
    }

    free(tile_experts);
    free(tile_starts);
    fprintf(stderr, "  test_sort_basic OK\n");
    return 0;
}

/* A second block_m to prove the tile math is not hardcoded to 8 (Q4_K's
 * own expert_tile_m), and a wider pair_count so at least one expert's
 * bucket spans multiple tiles. */
static int test_sort_tile_sizes(void) {
    enum { N_TOTAL_EXPERT = 4 };
    const uint32_t pair_count = 40;
    int32_t selected[40];
    /* Expert 0 gets 17 pairs (spans 3 tiles at block_m=8, 6 at block_m=3),
     * expert 1 gets 0, expert 2 gets 12, expert 3 gets 11. */
    uint32_t idx = 0;
    for (uint32_t i = 0; i < 17; i++) selected[idx++] = 0;
    for (uint32_t i = 0; i < 12; i++) selected[idx++] = 2;
    for (uint32_t i = 0; i < 11; i++) selected[idx++] = 3;

    const uint32_t block_ms[2] = {8u, 3u};
    for (int bi = 0; bi < 2; bi++) {
        const uint32_t block_m = block_ms[bi];
        const uint32_t tile_cap = sort_tile_capacity(pair_count, N_TOTAL_EXPERT, block_m);
        uint32_t counts[N_TOTAL_EXPERT], offsets[N_TOTAL_EXPERT + 1];
        uint32_t *sorted_pairs = malloc((size_t)pair_count * sizeof(uint32_t));
        uint32_t tile_total = SENTINEL;
        uint32_t *tile_experts = malloc((size_t)tile_cap * sizeof(uint32_t));
        uint32_t *tile_starts = malloc((size_t)tile_cap * sizeof(uint32_t));

        CHECK(ds4_sycl_moe_test_sort(selected, pair_count, N_TOTAL_EXPERT, block_m,
                                     counts, offsets, sorted_pairs, &tile_total,
                                     tile_experts, tile_starts, tile_cap) != 0,
              "sort_tile_sizes: ds4_sycl_moe_test_sort failed");

        uint32_t expect_total = 0;
        for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
            expect_total += (counts[e] + block_m - 1u) / block_m;
        }
        CHECK(tile_total == expect_total, "sort_tile_sizes: tile_total mismatch");
        CHECK(counts[1] == 0u, "sort_tile_sizes: expert 1 must have zero count");

        free(sorted_pairs);
        free(tile_experts);
        free(tile_starts);
    }
    fprintf(stderr, "  test_sort_tile_sizes OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_sort_basic() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sort_tile_sizes() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_moe OK\n");
    return 0;
}
