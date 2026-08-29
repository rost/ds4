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
#include <stdbool.h>
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
/* Q8_K activation quantiser, rocm/ds4_rocm_moe.cuh:750-798.  out_bytes
 * must hold n_rows*(in_dim/256) blocks of 292 bytes each (float d, int8
 * qs[256], int16 bsums[16], no padding between fields). */
extern int ds4_sycl_moe_test_q8_k_quantize(const float *x, uint32_t in_dim,
                                           uint32_t n_rows, uint8_t *out_bytes);

/* Sub-group sum reduction, rocm/ds4_rocm_moe.cuh:732-749 (half_warp_sum_f32
 * for width 16, quarter_warp_sum_f32 for width 8).  in holds
 * n_groups*width values; out[g] is the sum of in[g*width .. g*width+width). */
extern int ds4_sycl_moe_test_subgroup_sum(int width, const float *in,
                                          uint32_t n_groups, float *out);

/* Weighted-sum-across-experts combine, rocm/ds4_rocm_moe.cuh:3877-3918.
 * mode 0 = moe_sum_kernel (f32 down), 1 = moe_sum_f16_kernel (down is raw
 * f16 bit patterns), 2 = moe_sum_f16x2_kernel (same, vectorised path). */
extern int ds4_sycl_moe_test_sum(int mode, const void *down, uint32_t out_dim,
                                 uint32_t n_expert, uint32_t n_tokens, float *out);

enum { Q8_K_BLOCK_BYTES = 292 };

/* d (float, offset 0), qs (int8[256], offset 4), bsums (int16[16], offset 260). */
static float q8k_read_d(const uint8_t *blk) {
    float v;
    memcpy(&v, blk, sizeof(v));
    return v;
}
static int8_t q8k_read_qs(const uint8_t *blk, int i) { return (int8_t)blk[4 + i]; }
static int16_t q8k_read_bsum(const uint8_t *blk, int i) {
    int16_t v;
    memcpy(&v, blk + 260 + i * 2, sizeof(v));
    return v;
}

/* Minimal round-to-nearest-even IEEE754 binary16 encoder, sufficient for
 * the finite, moderate-magnitude test values used below (no inf/NaN/
 * subnormal handling needed). */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign; /* flush to zero, not needed by tests */
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t rounded_mant = (uint16_t)(mant >> 13);
    if (mant & 0x1000u) rounded_mant++; /* round to nearest, ties up */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | rounded_mant);
}

static int test_q8_k_quantize(void) {
    /* Two superblocks (in_dim = 512), two rows.  Row 0 has a wide dynamic
     * range (values spanning several orders of magnitude plus a mix of
     * signs); row 1 is entirely zero, exercising the amax==0 branch. */
    enum { IN_DIM = 512, N_ROWS = 2, N_BLOCKS = IN_DIM / 256 };
    float x[N_ROWS * IN_DIM];
    for (int i = 0; i < IN_DIM; i++) {
        float v = ((float)((i * 37) % 251) - 125.0f) * ((i % 5 == 0) ? 100.0f : 0.01f);
        if (i % 7 == 0) v = -v;
        x[i] = v;
    }
    for (int i = 0; i < IN_DIM; i++) x[IN_DIM + i] = 0.0f;

    uint8_t *out = malloc((size_t)N_ROWS * N_BLOCKS * Q8_K_BLOCK_BYTES);
    CHECK(ds4_sycl_moe_test_q8_k_quantize(x, IN_DIM, N_ROWS, out) != 0,
          "q8_k_quantize: call failed");

    for (int row = 0; row < N_ROWS; row++) {
        for (int b = 0; b < N_BLOCKS; b++) {
            const float *xr = x + row * IN_DIM + b * 256;
            const uint8_t *blk = out + ((size_t)row * N_BLOCKS + b) * Q8_K_BLOCK_BYTES;
            float amax = 0.0f, maxv = 0.0f;
            for (int i = 0; i < 256; i++) {
                if (fabsf(xr[i]) > amax) { amax = fabsf(xr[i]); maxv = xr[i]; }
            }
            char msg[96];
            if (amax == 0.0f) {
                CHECK(q8k_read_d(blk) == 0.0f, "q8_k_quantize: zero block d must be 0");
                for (int i = 0; i < 256; i++) {
                    CHECK(q8k_read_qs(blk, i) == 0, "q8_k_quantize: zero block qs must be 0");
                }
                for (int i = 0; i < 16; i++) {
                    CHECK(q8k_read_bsum(blk, i) == 0, "q8_k_quantize: zero block bsums must be 0");
                }
                continue;
            }
            const float iscale = -127.0f / maxv;
            const float d = q8k_read_d(blk);
            CHECK_CLOSE(d, 1.0f / iscale, fabsf(d) * 1e-5f + 1e-8f, "q8_k_quantize: d mismatch");
            for (int i = 0; i < 256; i++) {
                int qv = (int)lrintf(iscale * xr[i]);
                if (qv > 127) qv = 127;
                if (qv < -128) qv = -128;
                snprintf(msg, sizeof(msg), "q8_k_quantize: qs[%d] mismatch (row %d block %d)", i, row, b);
                CHECK(q8k_read_qs(blk, i) == (int8_t)qv, msg);
            }
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += q8k_read_qs(blk, g * 16 + i);
                snprintf(msg, sizeof(msg), "q8_k_quantize: bsums[%d] mismatch (row %d block %d)", g, row, b);
                CHECK(q8k_read_bsum(blk, g) == (int16_t)sum, msg);
            }
        }
    }
    free(out);
    fprintf(stderr, "  test_q8_k_quantize OK\n");
    return 0;
}

/* A row length that is not a multiple of 256 must be rejected (the
 * quantiser has no defined behaviour for a partial superblock; the
 * dispatcher's own validation, ported in a later task, rejects this
 * shape before ever reaching the kernel). */
static int test_q8_k_quantize_rejects_partial_row(void) {
    float x[300];
    for (int i = 0; i < 300; i++) x[i] = (float)i;
    uint8_t out[2 * Q8_K_BLOCK_BYTES];
    CHECK(ds4_sycl_moe_test_q8_k_quantize(x, 300, 1, out) == 0,
          "q8_k_quantize: non-multiple-of-256 in_dim must be rejected");
    fprintf(stderr, "  test_q8_k_quantize_rejects_partial_row OK\n");
    return 0;
}

/* Both reduction widths, with values chosen so a reduction over the wrong
 * lane count gives a different answer: group g's values are
 * (g*100 + lane + 1), so summing 8 of them differs from summing 16 by a
 * large, easily-distinguished amount (not just "some upper lanes happen
 * to be zero"). */
static int test_subgroup_sum(void) {
    const int widths[2] = {8, 16};
    for (int wi = 0; wi < 2; wi++) {
        const int width = widths[wi];
        enum { N_GROUPS = 6 };
        float in[6 * 16];
        float want[6];
        for (int g = 0; g < N_GROUPS; g++) {
            float sum = 0.0f;
            for (int lane = 0; lane < width; lane++) {
                float v = (float)(g * 100 + lane + 1);
                in[g * width + lane] = v;
                sum += v;
            }
            want[g] = sum;
        }
        float got[N_GROUPS];
        char msg[64];
        snprintf(msg, sizeof(msg), "subgroup_sum: width %d call failed", width);
        CHECK(ds4_sycl_moe_test_subgroup_sum(width, in, N_GROUPS, got) != 0, msg);
        for (int g = 0; g < N_GROUPS; g++) {
            snprintf(msg, sizeof(msg), "subgroup_sum: width %d group %d mismatch", width, g);
            CHECK_CLOSE(got[g], want[g], 1e-3, msg);
        }
    }
    fprintf(stderr, "  test_subgroup_sum OK\n");
    return 0;
}

/* moe_sum_kernel accumulates strictly in ascending slot order
 * (e = 0..n_expert-1 over the pair index tok*n_expert+e).  Floating point
 * addition is not associative, so a wrong accumulation order (as
 * layer_routed_moe_batch's expert-id ordering would produce against this
 * kernel's own down-tensor layout, see ds4-sycl-moe-reference.md section
 * 4(a)) is only actually distinguishable from the correct order when the
 * terms have a wide enough dynamic range for reordering to change
 * rounding.  Each row's four values are {+1e8, -1e8, v1, v2}: summed in
 * ascending order the two huge terms cancel to exactly 0.0 before the
 * small terms are added (result v1+v2); summed in any order that adds a
 * small term between the two huge ones, the small term is lost to
 * rounding against the ~1e8-magnitude partial sum instead. */
static int test_sum_slot_order(void) {
    enum { OUT_DIM = 6, N_EXPERT = 4, N_TOKENS = 3 };
    float down[N_TOKENS * N_EXPERT * OUT_DIM];
    float want[N_TOKENS * OUT_DIM];
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            const float v1 = (float)((t + 1) * 10 + row + 1);
            const float v2 = (float)((t + 1) * 7 + row + 3);
            const float vals[N_EXPERT] = {1.0e8f, -1.0e8f, v1, v2};
            float sum = 0.0f;
            for (int e = 0; e < N_EXPERT; e++) {
                down[(t * N_EXPERT + e) * OUT_DIM + row] = vals[e];
                sum += vals[e];
            }
            want[t * OUT_DIM + row] = sum;
        }
    }
    float got[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(0, down, OUT_DIM, N_EXPERT, N_TOKENS, got) != 0,
          "sum_slot_order: f32 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "sum_slot_order: f32 value mismatch");
    }

    /* f16 and f16x2 variants: half precision cannot represent 1e8 (max
     * finite value ~65504), so these use their own moderate-magnitude
     * data instead of the cancellation values above; the order-sensitive
     * property was already exercised by the f32 case.  OUT_DIM is even,
     * so f16x2 is exercised too. */
    uint16_t down_h[N_TOKENS * N_EXPERT * OUT_DIM];
    float want_h[N_TOKENS * OUT_DIM];
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            for (int e = 0; e < N_EXPERT; e++) {
                float v = (float)((t + 1) * 12 - e * 5 + row);
                down_h[(t * N_EXPERT + e) * OUT_DIM + row] = f32_to_f16_bits(v);
            }
        }
    }
    /* want_h is derived from the actual half-rounded values (decoded by
     * hand below) so the oracle matches what the kernel reads, not full
     * f32 precision. */
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            float sum = 0.0f;
            for (int e = 0; e < N_EXPERT; e++) {
                uint16_t bits = down_h[(t * N_EXPERT + e) * OUT_DIM + row];
                float f;
                /* Round-trip through the same bit-cast the kernel uses;
                 * a plain host-side half decode via ldexp/mantissa. */
                int sign = (bits & 0x8000) ? -1 : 1;
                int exp = (bits >> 10) & 0x1f;
                int mant = bits & 0x3ff;
                if (exp == 0) f = sign * (float)mant * (float)ldexp(1.0, -24);
                else f = sign * (float)(1024 + mant) * (float)ldexp(1.0, exp - 25);
                sum += f;
            }
            want_h[t * OUT_DIM + row] = sum;
        }
    }
    float got_h[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(1, down_h, OUT_DIM, N_EXPERT, N_TOKENS, got_h) != 0,
          "sum_slot_order: f16 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got_h[i], want_h[i], 1e-2, "sum_slot_order: f16 value mismatch");
    }
    float got_h2[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(2, down_h, OUT_DIM, N_EXPERT, N_TOKENS, got_h2) != 0,
          "sum_slot_order: f16x2 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got_h2[i], want_h[i], 1e-2, "sum_slot_order: f16x2 value mismatch");
    }
    fprintf(stderr, "  test_sum_slot_order OK\n");
    return 0;
}

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

/* ---- Dispatcher skeleton and the two entry points --------------------
 *
 * Shapes small enough that the scratch-buffer-reuse precondition
 * (gate/down tensors doubling as Q8_K quantisation scratch) is always
 * satisfied by construction: gate holds n_tokens*n_expert*expert_mid_dim
 * floats, which for these sizes is comfortably larger than the Q8_K
 * scratch it would need to double as. */
enum {
    RM_EXPERT_IN_DIM  = 256,
    RM_EXPERT_MID_DIM = 256,
    RM_OUT_DIM        = 64,
    RM_N_EXPERT       = 2,
    RM_N_TOKENS       = 2,
    RM_N_TOTAL_EXPERT = 4,
    RM_Q4K_BLOCK_BYTES = 144, /* cuda_block_q4_K, ds4_rocm.cu:72-77 */
};

typedef struct {
    unsigned char *model;
    uint64_t       model_size;
    uint64_t       gate_offset, up_offset, down_offset;
    uint64_t       gate_expert_bytes, gate_row_bytes;
    uint64_t       down_expert_bytes, down_row_bytes;
} rm_model;

/* type: 12 = Q4_K (used for gate_row_bytes sizing regardless of the
 * gate_type/down_type the caller then passes; only the byte layout
 * needs to be self-consistent for the model-range bounds checks to
 * pass, and every format the dispatcher can currently reach
 * (implemented or stubbed) uses 256-wide superblocks). */
static rm_model rm_build_model(void) {
    rm_model m;
    m.gate_row_bytes = (RM_EXPERT_IN_DIM / 256u) * RM_Q4K_BLOCK_BYTES;
    m.gate_expert_bytes = (uint64_t)RM_EXPERT_MID_DIM * m.gate_row_bytes;
    m.down_row_bytes = (RM_EXPERT_MID_DIM / 256u) * RM_Q4K_BLOCK_BYTES;
    m.down_expert_bytes = (uint64_t)RM_OUT_DIM * m.down_row_bytes;
    m.gate_offset = 0;
    m.up_offset = m.gate_expert_bytes * RM_N_TOTAL_EXPERT;
    m.down_offset = m.up_offset + m.gate_expert_bytes * RM_N_TOTAL_EXPERT;
    m.model_size = m.down_offset + m.down_expert_bytes * RM_N_TOTAL_EXPERT;
    m.model = calloc(1, (size_t)m.model_size);
    return m;
}

typedef struct {
    ds4_gpu_tensor *out, *gate, *up, *mid, *down, *selected, *weights, *x;
} rm_tensors;

static rm_tensors rm_build_tensors(void) {
    rm_tensors t;
    t.out = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_OUT_DIM * sizeof(float));
    t.gate = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.up = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.mid = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.down = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    t.selected = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    t.weights = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(float));
    t.x = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t sel[RM_N_TOKENS * RM_N_EXPERT];
    float w[RM_N_TOKENS * RM_N_EXPERT];
    for (int i = 0; i < RM_N_TOKENS * RM_N_EXPERT; i++) {
        sel[i] = i % RM_N_TOTAL_EXPERT;
        w[i] = 1.0f;
    }
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    float xv[RM_N_TOKENS * RM_EXPERT_IN_DIM];
    for (int i = 0; i < RM_N_TOKENS * RM_EXPERT_IN_DIM; i++) xv[i] = 0.01f * (float)i;
    ds4_gpu_tensor_write(t.x, 0, xv, sizeof(xv));
    return t;
}

static void rm_free_tensors(rm_tensors *t) {
    ds4_gpu_tensor_free(t->out);
    ds4_gpu_tensor_free(t->gate);
    ds4_gpu_tensor_free(t->up);
    ds4_gpu_tensor_free(t->mid);
    ds4_gpu_tensor_free(t->down);
    ds4_gpu_tensor_free(t->selected);
    ds4_gpu_tensor_free(t->weights);
    ds4_gpu_tensor_free(t->x);
}

static int rm_call_batch(rm_model *m, rm_tensors *t, uint32_t gate_type,
                         uint32_t down_type, uint32_t n_tokens, bool *mid_is_f16) {
    return ds4_gpu_routed_moe_batch_tensor(
            t->out, t->gate, t->up, t->mid, t->down, m->model, m->model_size,
            m->gate_offset, m->up_offset, m->down_offset, gate_type, down_type,
            m->gate_expert_bytes, m->gate_row_bytes, m->down_expert_bytes,
            m->down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
            t->selected, t->weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t->x,
            /*layer_index=*/0u, n_tokens, mid_is_f16, /*force_resident=*/false);
}

/* n_tokens == 0 is a FAILURE (routed_moe_build_plan, moe_launch.cuh:479),
 * unlike several other entries ported earlier in this project where
 * zero-work is a free success. */
static int test_dispatcher_zero_tokens_fails(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 12u, 12u, 0u, NULL) == 0,
          "dispatcher: n_tokens == 0 must fail, not succeed");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_zero_tokens_fails OK\n");
    return 0;
}

static int test_dispatcher_unrecognised_format_fails(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 99u, 99u, RM_N_TOKENS, NULL) == 0,
          "dispatcher: unrecognised gate/down type pair must fail");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_unrecognised_format_fails OK\n");
    return 0;
}

/* Each of the three formats not yet implemented here must fail
 * cleanly with a well-formed call (reaching its own delimited dispatcher
 * block, not failing on validation), and each of these three tests is
 * exactly the test that must flip when that format is implemented. */
static int test_dispatcher_iq2_not_yet_implemented(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 16u, 10u, RM_N_TOKENS, NULL) == 0,
          "dispatcher: iq2_path must fail until it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_iq2_not_yet_implemented OK\n");
    return 0;
}

static int test_dispatcher_q2k_not_yet_implemented(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 10u, 10u, RM_N_TOKENS, NULL) == 0,
          "dispatcher: q2k_path must fail until it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_q2k_not_yet_implemented OK\n");
    return 0;
}

static int test_dispatcher_mxfp4_not_yet_implemented(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 39u, 39u, RM_N_TOKENS, NULL) == 0,
          "dispatcher: mxfp4_path must fail until it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_mxfp4_not_yet_implemented OK\n");
    return 0;
}

static int test_dispatcher_add_in_rejected(void) {
    ds4_gpu_tensor *dummy = ds4_gpu_tensor_alloc(64);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, NULL, NULL, 0, 0, 0.0f, NULL, dummy, 0, false) == 0,
          "dispatcher: non-null add_in must fail on the decode entry");
    ds4_gpu_tensor_free(dummy);
    fprintf(stderr, "  test_dispatcher_add_in_rejected OK\n");
    return 0;
}

static int test_dispatcher_mid_is_f16_written_false(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    bool mid_is_f16 = true;
    /* The call itself fails (mxfp4 not implemented yet), but mid_is_f16
     * must still be written, unconditionally, before that failure. */
    rm_call_batch(&m, &t, 39u, 39u, RM_N_TOKENS, &mid_is_f16);
    CHECK(mid_is_f16 == false,
          "dispatcher: mid_is_f16 must be written false regardless of outcome");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_mid_is_f16_written_false OK\n");
    return 0;
}

/* A representative sample of routed_moe_build_plan's own validation
 * rejections (moe_launch.cuh:479-509), each independent of format. */
static int test_dispatcher_build_plan_validation(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();

    /* expert_in_dim not a multiple of 256: reuse the batch call but with
     * an ad hoc direct call so expert_in_dim can be perturbed. */
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, /*expert_in_dim=*/255u, RM_EXPERT_MID_DIM,
              RM_OUT_DIM, t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT,
              0.0f, t.x, 0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: expert_in_dim not a multiple of 256 must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, /*n_expert=*/9u, 0.0f,
              t.x, 0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: n_expert above DS4_ROCM_N_EXPERT_USED must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              NULL, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: null out tensor must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model,
              /*model_size=*/1u /* too small for gate/up/down ranges */,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: model range exceeding model_size must fail");

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_build_plan_validation OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_q8_k_quantize() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q8_k_quantize_rejects_partial_row() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_subgroup_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sum_slot_order() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sort_basic() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sort_tile_sizes() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_zero_tokens_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_unrecognised_format_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_iq2_not_yet_implemented() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_q2k_not_yet_implemented() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mxfp4_not_yet_implemented() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_add_in_rejected() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mid_is_f16_written_false() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_build_plan_validation() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_moe OK\n");
    return 0;
}
