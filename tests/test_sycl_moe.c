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

/* Drives sycl_moe_q2k_down_direct (the untagged-family Q2_K down kernel,
 * rocm/ds4_rocm_moe.cuh:2909-2936, generalised over n_tokens) directly,
 * for the slot-order regression test below; every ABI-level test in this
 * file uses n_expert == 2, which cannot discriminate summation order. */
extern int ds4_sycl_moe_test_q2k_down_direct(
        const uint8_t *down_bytes, const uint8_t *midq_bytes, const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t n_total_expert,
        float *out);

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

/* Inverse of f32_to_f16_bits, matching the sycl::bit_cast<sycl::half>
 * decode every SYCL-side f16 field in this subsystem uses. */
static float f16_bits_to_f32(uint16_t bits) {
    const int sign = (bits & 0x8000) ? -1 : 1;
    const int exp = (bits >> 10) & 0x1f;
    const int mant = bits & 0x3ff;
    if (exp == 0) return (float)sign * (float)mant * (float)ldexp(1.0, -24);
    return (float)sign * (float)(1024 + mant) * (float)ldexp(1.0, exp - 25);
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
    /* want_h is derived from the actual half-rounded values (decoded via
     * f16_bits_to_f32) so the oracle matches what the kernel reads, not
     * full f32 precision. */
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            float sum = 0.0f;
            for (int e = 0; e < N_EXPERT; e++) {
                uint16_t bits = down_h[(t * N_EXPERT + e) * OUT_DIM + row];
                sum += f16_bits_to_f32(bits);
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

enum {
    RM_IQ2_BLOCK_BYTES = 66, /* cuda_block_iq2_xxs, ds4_rocm.cu:85-88 */
    RM_Q2K_BLOCK_BYTES = 84, /* cuda_block_q2_K, ds4_rocm.cu:65-70 */
};

/* Generalised model builder: gate/up always share one block size (both
 * paired formats implemented here use the same weight type for gate
 * and up), down may use a different one (iq2_path pairs IQ2_XXS gate/up
 * with Q2_K down). */
static rm_model rm_build_model_ex(uint32_t gate_block_bytes, uint32_t down_block_bytes) {
    rm_model m;
    m.gate_row_bytes = (RM_EXPERT_IN_DIM / 256u) * gate_block_bytes;
    m.gate_expert_bytes = (uint64_t)RM_EXPERT_MID_DIM * m.gate_row_bytes;
    m.down_row_bytes = (RM_EXPERT_MID_DIM / 256u) * down_block_bytes;
    m.down_expert_bytes = (uint64_t)RM_OUT_DIM * m.down_row_bytes;
    m.gate_offset = 0;
    m.up_offset = m.gate_expert_bytes * RM_N_TOTAL_EXPERT;
    m.down_offset = m.up_offset + m.gate_expert_bytes * RM_N_TOTAL_EXPERT;
    m.model_size = m.down_offset + m.down_expert_bytes * RM_N_TOTAL_EXPERT;
    m.model = calloc(1, (size_t)m.model_size);
    return m;
}

/* type: 12 = Q4_K (used for gate_row_bytes sizing regardless of the
 * gate_type/down_type the caller then passes; only the byte layout
 * needs to be self-consistent for the model-range bounds checks to
 * pass, and every format the dispatcher can currently reach
 * (implemented or stubbed) uses 256-wide superblocks). */
static rm_model rm_build_model(void) {
    return rm_build_model_ex(RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
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

/* Both iq2_path and q2k_path formats are now implemented: they now
 * reach their own delimited dispatcher block and succeed on a
 * well-formed call, not fail as "not yet implemented". The mxfp4_path
 * sibling test below is unchanged; that format is implemented separately below. */
static int test_dispatcher_iq2_succeeds(void) {
    rm_model m = rm_build_model_ex(RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES);
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 16u, 10u, RM_N_TOKENS, NULL) != 0,
          "dispatcher: iq2_path must succeed now that it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_iq2_succeeds OK\n");
    return 0;
}

static int test_dispatcher_q2k_succeeds(void) {
    rm_model m = rm_build_model_ex(RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES);
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 10u, 10u, RM_N_TOKENS, NULL) != 0,
          "dispatcher: q2k_path must succeed now that it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_q2k_succeeds OK\n");
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

/* Uses a fully well-formed q4k_path call (the only format this test
 * model's byte layout can actually complete successfully) so the add_in check is proven to fire
 * before a call that would otherwise succeed, not merely alongside other
 * validation failures that would return 0 anyway. */
static int test_dispatcher_add_in_rejected(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    ds4_gpu_tensor *dummy = ds4_gpu_tensor_alloc(64);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              dummy, 0u, false) == 0,
          "dispatcher: non-null add_in must fail on the decode entry");
    ds4_gpu_tensor_free(dummy);
    rm_free_tensors(&t);
    free(m.model);
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

/* ---- Q4_K: full decode/batch path against a scalar CPU oracle --------
 *
 * The oracle below reimplements ds4.c's layer_routed_moe_one_prealloc
 * (ds4.c:10984-11092) for the Q4_K case: matvec_q4_k_experts_mid_prequant
 * (ds4.c:8970-9014, via matvec_q4_k_mid_worker at :8946-8965) for gate/up,
 * ds4_quantize_row_q8_K (ds4.c:3336-3370) to requantize the mid
 * activations, and matvec_q4_k_experts_accum_prequant (ds4.c:9042-9078,
 * via matvec_q4_k_accum_worker at :9024-9036) for the down projection,
 * which sums over slots in ascending order inside one worker call -- the
 * same order ds4-sycl-moe-reference.md section 4(a) attributes to a
 * "for i in 0..n_expert_used" loop that this reading of ds4.c did not
 * find; the loop shape differs from the research document's description
 * but the property that matters (ascending slot-order summation,
 * matching moe_sum_kernel) is the same either way, re-confirmed directly
 * against ds4.c rather than trusted from the document.
 *
 * ds4.c's own scalar routines (ds4_vec_dot_q4_K_q8_K, ds4_quantize_row_q8_K)
 * cannot be linked (static, and ds4.c is not part of this build), so
 * every scalar helper below is an independent reimplementation, matching
 * the precedent in tests/test_mxfp4_rocm.c and every oracle earlier in
 * this file. */

/* q4_k_get_scale_min, ds4.c:3514-3521 (and dev_q4_K_get_scale_min,
 * moe.cuh:250-262): unpacks one of 8 6-bit (scale, min) pairs from a
 * Q4_K block's 12-byte scales array. */
static void oracle_q4k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Inverse packing: given 8 (scale, min) pairs (each 0-63), produce the
 * 12-byte scales array a real Q4_K block would carry.  Derived by
 * inverting oracle_q4k_get_scale_min's two branches; self-checked by
 * test_q4k_scale_pack_roundtrip below before being trusted for anything
 * else. */
static void oracle_q4k_pack_scales(uint8_t q[12], const uint8_t sc[8], const uint8_t m[8]) {
    for (int i = 0; i < 4; i++) {
        q[i]     = (uint8_t)((sc[i] & 0x3Fu) | ((uint32_t)(sc[i + 4] >> 4) << 6));
        q[i + 4] = (uint8_t)((m[i] & 0x3Fu) | ((uint32_t)(m[i + 4] >> 4) << 6));
        q[i + 8] = (uint8_t)((sc[i + 4] & 0x0Fu) | ((uint32_t)(m[i + 4] & 0x0Fu) << 4));
    }
}

static int test_q4k_scale_pack_roundtrip(void) {
    uint8_t sc[8], m[8], q[12];
    for (int seed = 0; seed < 5; seed++) {
        for (int j = 0; j < 8; j++) {
            sc[j] = (uint8_t)((seed * 17 + j * 23) % 64);
            m[j] = (uint8_t)((seed * 29 + j * 11) % 64);
        }
        oracle_q4k_pack_scales(q, sc, m);
        for (int j = 0; j < 8; j++) {
            uint8_t got_sc, got_m;
            oracle_q4k_get_scale_min(j, q, &got_sc, &got_m);
            char msg[80];
            snprintf(msg, sizeof(msg), "scale_pack_roundtrip: seed %d j %d sc mismatch", seed, j);
            CHECK(got_sc == sc[j], msg);
            snprintf(msg, sizeof(msg), "scale_pack_roundtrip: seed %d j %d m mismatch", seed, j);
            CHECK(got_m == m[j], msg);
        }
    }
    fprintf(stderr, "  test_q4k_scale_pack_roundtrip OK\n");
    return 0;
}

/* Packs one 144-byte cuda_block_q4_K (ds4_rocm.cu:72-77): d and dmin as
 * f16 bits, 12 bytes of packed scale/min, 128 bytes of packed 4-bit
 * codes (two sub-blocks share each 32-byte qs region, low nibble for the
 * even sub-block, high nibble for the odd one, per moe.cuh:274-287's
 * byte_off/shift math). */
static void oracle_q4k_pack_block(uint8_t out[144], float d, float dmin,
                                  const uint8_t sc[8], const uint8_t m[8],
                                  const uint8_t nib[256]) {
    uint16_t dh = f32_to_f16_bits(d);
    uint16_t dminh = f32_to_f16_bits(dmin);
    memcpy(out, &dh, 2);
    memcpy(out + 2, &dminh, 2);
    oracle_q4k_pack_scales(out + 4, sc, m);
    uint8_t qs[128];
    memset(qs, 0, sizeof(qs));
    for (int j = 0; j < 8; j++) {
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) ? 4 : 0;
        for (int k = 0; k < 32; k++) {
            uint8_t v = nib[j * 32 + k] & 0x0Fu;
            if (shift == 0) qs[byte_off + k] = (uint8_t)((qs[byte_off + k] & 0xF0u) | v);
            else qs[byte_off + k] = (uint8_t)((qs[byte_off + k] & 0x0Fu) | (v << 4));
        }
    }
    memcpy(out + 16, qs, 128);
}

typedef struct {
    int8_t  qs[256];
    int16_t bsums[16];
    float   d;
} oracle_q8k_block;

/* ds4_quantize_row_q8_K / q8_K_quantize_kernel, both cited above
 * test_q8_k_quantize: identical amax scan, iscale = -127/max, round to
 * nearest, clamp to [-128,127], and 16-wide block sums. */
static void oracle_q8k_quantize_block(const float *xr, oracle_q8k_block *out) {
    float amax = 0.0f, maxv = 0.0f;
    for (int i = 0; i < 256; i++) {
        float ax = fabsf(xr[i]);
        if (ax > amax) { amax = ax; maxv = xr[i]; }
    }
    if (amax == 0.0f) {
        out->d = 0.0f;
        memset(out->qs, 0, sizeof(out->qs));
        memset(out->bsums, 0, sizeof(out->bsums));
        return;
    }
    const float iscale = -127.0f / maxv;
    for (int i = 0; i < 256; i++) {
        int v = (int)lrintf(iscale * xr[i]);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out->qs[i] = (int8_t)v;
    }
    for (int g = 0; g < 16; g++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += out->qs[g * 16 + i];
        out->bsums[g] = (int16_t)sum;
    }
    out->d = 1.0f / iscale;
}

/* dev_dot_q4_K_q8_K_block / ds4_vec_dot_q4_K_q8_K, single 256-value block. */
static float oracle_q4k_dot_block(const uint8_t blk[144], const oracle_q8k_block *y) {
    uint16_t dbits, dminbits;
    memcpy(&dbits, blk, 2);
    memcpy(&dminbits, blk + 2, 2);
    const float xd = f16_bits_to_f32(dbits);
    const float xmin = f16_bits_to_f32(dminbits);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int32_t isum = 0, summs = 0;
    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        oracle_q4k_get_scale_min(j, scales, &sc, &m);
        summs += (int32_t)m * (int32_t)(y->bsums[2 * j] + y->bsums[2 * j + 1]);
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) ? 4 : 0;
        int32_t block_sum = 0;
        for (int i = 0; i < 32; i++) {
            block_sum += (int32_t)((qs[byte_off + i] >> shift) & 0x0F) * (int32_t)y->qs[j * 32 + i];
        }
        isum += (int32_t)sc * block_sum;
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

static float oracle_silu(float x) { return x / (1.0f + expf(-x)); }

/* Deterministic per-(expert,row) weight-block generator with a
 * non-linear interaction term (spec 6f: pure affine test data makes
 * every element mutually proportional and hides scale-only bugs). */
static void q4k_fill_row(uint8_t out[144], uint32_t phase, uint32_t expert, uint32_t row) {
    uint8_t sc[8], m[8], nib[256];
    for (int j = 0; j < 8; j++) {
        sc[j] = (uint8_t)(1u + (phase * 5u + expert * 7u + row * 3u + (uint32_t)j * 5u) % 40u);
        m[j] = (uint8_t)((phase * 3u + expert * 11u + row * 2u + (uint32_t)j * 13u) % 20u);
    }
    for (int k = 0; k < 256; k++) {
        nib[k] = (uint8_t)((phase * 19u + expert * 13u + row * 17u + (uint32_t)k +
                            (expert * row) % 5u + ((uint32_t)k * row) % 7u) % 16u);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row) % 7u);
    const float dmin = 0.001f * (float)((phase + expert * 2u + row) % 5u);
    oracle_q4k_pack_block(out, d, dmin, sc, m, nib);
}

static void q4k_fill_x_row(float *x, uint32_t in_dim, uint32_t tok) {
    for (uint32_t k = 0; k < in_dim; k++) {
        x[k] = 0.01f * (float)((int)((tok * 37u + k * 11u + (tok * k) % 13u + 5u) % 200u) - 100);
    }
}

enum { Q4K_PHASE_GATE = 0, Q4K_PHASE_UP = 100, Q4K_PHASE_DOWN = 200 };

/* Fills an rm_model's gate/up/down sections with q4k_fill_row blocks,
 * one block per row (in_dim == 256 for gate/up, mid_dim == 256 for down,
 * so xq_blocks == midq_blocks == 1 for every test in this section). */
static void q4k_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim,
                           uint32_t out_dim) {
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            uint8_t blk[144];
            q4k_fill_row(blk, Q4K_PHASE_GATE, e, row);
            memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
            q4k_fill_row(blk, Q4K_PHASE_UP, e, row);
            memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            uint8_t blk[144];
            q4k_fill_row(blk, Q4K_PHASE_DOWN, e, row);
            memcpy(m->model + m->down_offset + e * m->down_expert_bytes + (uint64_t)row * m->down_row_bytes,
                   blk, sizeof(blk));
        }
    }
}

/* layer_routed_moe_one_prealloc for the Q4_K case, one token at a time. */
static void oracle_q4k_one_token(const rm_model *m, uint32_t in_dim, uint32_t mid_dim,
                                 uint32_t out_dim, uint32_t n_expert,
                                 const int32_t *sel, const float *w, const float *x,
                                 float clamp, float *out) {
    (void)in_dim; /* == 256 assumed throughout (single Q8_K block per row) */
    oracle_q8k_block xq;
    oracle_q8k_quantize_block(x, &xq);

    float *mid = malloc((size_t)n_expert * mid_dim * sizeof(float));
    for (uint32_t s = 0; s < n_expert; s++) {
        const uint32_t expert = (uint32_t)sel[s];
        for (uint32_t row = 0; row < mid_dim; row++) {
            const uint8_t *gate_blk = m->model + m->gate_offset + (uint64_t)expert * m->gate_expert_bytes +
                                      (uint64_t)row * m->gate_row_bytes;
            const uint8_t *up_blk = m->model + m->up_offset + (uint64_t)expert * m->gate_expert_bytes +
                                    (uint64_t)row * m->gate_row_bytes;
            float gate = oracle_q4k_dot_block(gate_blk, &xq);
            float up = oracle_q4k_dot_block(up_blk, &xq);
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid[(size_t)s * mid_dim + row] = oracle_silu(gate) * up * w[s];
        }
    }

    oracle_q8k_block *midq = malloc((size_t)n_expert * sizeof(oracle_q8k_block));
    for (uint32_t s = 0; s < n_expert; s++) {
        oracle_q8k_quantize_block(mid + (size_t)s * mid_dim, &midq[s]); /* mid_dim == 256 */
    }

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_expert; s++) { /* ascending slot order */
            const uint32_t expert = (uint32_t)sel[s];
            const uint8_t *down_blk = m->model + m->down_offset + (uint64_t)expert * m->down_expert_bytes +
                                      (uint64_t)row * m->down_row_bytes;
            acc += oracle_q4k_dot_block(down_blk, &midq[s]);
        }
        out[row] = acc;
    }
    free(mid);
    free(midq);
}

static void q4k_fill_selected(int32_t *sel, float *w, uint32_t tok, uint32_t n_expert,
                              uint32_t n_total_expert) {
    for (uint32_t s = 0; s < n_expert; s++) {
        sel[s] = (int32_t)((tok * 3u + s * 2u + 1u) % n_total_expert);
        w[s] = 0.5f + 0.25f * (float)((tok + s) % 3u);
    }
}

/* n_tokens == 1 (decode) against the per-token oracle, at tight tolerance. */
static int test_q4k_decode(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "q4k_decode: call failed");

    float want[RM_OUT_DIM];
    oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                         sel, w, x, 0.0f, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4, "q4k_decode: value mismatch");
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode OK\n");
    return 0;
}

/* Batched path against N ordered per-token oracle calls, for an n_tokens
 * both below and at/above the sorted-pairs threshold (32), so both the
 * untiled and tiled routes are exercised. */
static int test_q4k_batch(uint32_t n_tokens) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t *sel = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    float *w = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(float));
    float *xv = malloc((size_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        q4k_fill_selected(sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT, t,
                          RM_N_EXPERT, RM_N_TOTAL_EXPERT);
        q4k_fill_x_row(xv + (size_t)t * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, t);
    }
    ds4_gpu_tensor_write(selected, 0, sel, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor_write(weights, 0, w, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    bool mid_is_f16 = true;
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, /*layer_index=*/0u, n_tokens,
              &mid_is_f16, /*force_resident=*/false) != 0,
          "q4k_batch: call failed");
    CHECK(mid_is_f16 == false, "q4k_batch: mid_is_f16 must be written false");

    float *got = malloc((size_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));

    float want_row[RM_OUT_DIM];
    for (uint32_t t = 0; t < n_tokens; t++) {
        oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                             sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT,
                             xv + (size_t)t * RM_EXPERT_IN_DIM, 0.0f, want_row);
        for (int i = 0; i < RM_OUT_DIM; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "q4k_batch(n=%u): token %u value mismatch", n_tokens, t);
            CHECK_CLOSE(got[(size_t)t * RM_OUT_DIM + i], want_row[i],
                       fabs(want_row[i]) * 1e-3 + 1e-4, msg);
        }
    }

    free(sel);
    free(w);
    free(xv);
    free(got);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_batch(n_tokens=%u) OK\n", n_tokens);
    return 0;
}

/* Differential check between the decode path (ds4_gpu_routed_moe_one_tensor,
 * always n_tokens == 1) and the batch path (ds4_gpu_routed_moe_batch_tensor
 * called with n_tokens == 1u): both dispatch to the same "decode" gate/up
 * kernel and the same sum6 down kernel, so they must produce identical
 * output on identical input.  This substitutes for a differential test
 * against the F32 fallback path: that fallback turned out to hardcode
 * IQ2_XXS/Q2_K weight-block casts, so it cannot validate
 * Q4_K at all; comparing decode against batch on identical input is the
 * cheapest available cross-check between two independently dispatched
 * real code paths instead. */
static int test_q4k_decode_matches_batch_of_one(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t1 = rm_build_tensors();
    rm_tensors t2 = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/2, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/2);
    ds4_gpu_tensor_write(t1.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t1.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t1.x, 0, x, sizeof(x));
    ds4_gpu_tensor_write(t2.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t2.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t2.x, 0, x, sizeof(x));

    CHECK(ds4_gpu_routed_moe_one_tensor(
              t1.out, t1.gate, t1.up, t1.mid, t1.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t1.selected, t1.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t1.x,
              NULL, 0u, false) != 0,
          "decode_matches_batch: decode call failed");
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t2.out, t2.gate, t2.up, t2.mid, t2.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t2.selected, t2.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t2.x,
              0u, /*n_tokens=*/1u, NULL, false) != 0,
          "decode_matches_batch: batch(n=1) call failed");

    float got1[RM_OUT_DIM], got2[RM_OUT_DIM];
    ds4_gpu_tensor_read(t1.out, 0, got1, sizeof(got1));
    ds4_gpu_tensor_read(t2.out, 0, got2, sizeof(got2));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got1[i], got2[i], fabs(got1[i]) * 1e-4 + 1e-5,
                   "decode_matches_batch: decode vs batch(n=1) mismatch");
    }

    rm_free_tensors(&t1);
    rm_free_tensors(&t2);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode_matches_batch_of_one OK\n");
    return 0;
}

/* The scratch-buffer-reuse precondition (moe_launch.cuh:746) failing must
 * be a clean failure for Q4_K: ROCm's own
 * fallback for this case cannot be reused (it hardcodes IQ2_XXS/Q2_K
 * weight-block casts, wrong for Q4_K bytes). */
static int test_q4k_scratch_precondition_failure(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    /* gate/down sized only for their literal F32 role, too small to also
     * hold the Q8_K quantisation scratch this path needs. */
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));
    int32_t sel[RM_N_TOKENS * RM_N_EXPERT] = {0, 1, 2, 3};
    float w[RM_N_TOKENS * RM_N_EXPERT] = {1, 1, 1, 1};
    ds4_gpu_tensor_write(selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(weights, 0, w, sizeof(w));

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, 0u, RM_N_TOKENS, NULL,
              false) == 0,
          "q4k_scratch_precondition: undersized gate/down must fail cleanly");

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_scratch_precondition_failure OK\n");
    return 0;
}

/* ---- IQ2_XXS and Q2_K: full decode/batch paths against scalar oracles --
 *
 * Both oracles reimplement layer_routed_moe_one_prealloc's dispatch for
 * these formats: matvec_iq2_xxs_experts_mid_prequant / matvec_q2_k_experts_
 * mid_prequant (ds4.c:8079, 8377) for gate/up, matvec_iq2_xxs_experts_
 * accum_prequant / matvec_q2_k_experts_accum_prequant (ds4.c:8580, 8297)
 * for down, both dispatched from matvec_experts_mid_prequant /
 * matvec_experts_down_accum_prequant (ds4.c:9452, 9480).  ds4_vec_dot_
 * q2_K_q8_K (ds4.c:3376) and ds4_vec_dot_iq2_xxs_q8_K (ds4.c:3933) were
 * read directly (non-NEON branch, since this machine is x86) and verified
 * field-for-field against dev_dot_q2_K_q8_K_block / dev_dot_iq2_xxs_q8_K_
 * block (moe.cuh:622-645, 101-123): identical aux0/aux1/ls decode, scale
 * indexing and accumulation order in both cases.  As with Q4_K, ds4.c's
 * own scalar routines cannot be linked (static, ds4.c not part of this
 * build), so every helper below is an independent reimplementation. */

/* cuda_iq2xxs_grid / cuda_ksigns_iq2xs, ds4_iq2_tables_cuda.inc: the exact
 * same tables sycl/ds4_sycl_moe.hpp embeds for the kernels under test.
 * Duplicated here rather than shared because this is a plain C
 * translation unit and ds4.c's own IQ2_XXS tables (iq2xxs_signed_grid) are
 * precomputed in a different, though mathematically equivalent, shape
 * (grid and sign already combined) that this file has no way to link to
 * either. */
static const uint8_t g_iq2_signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t g_iq2_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

static uint32_t oracle_popcount_u32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0f0f0f0fu;
    return (v * 0x01010101u) >> 24;
}

static int32_t oracle_iq2_grid_dot8(uint8_t grid_idx, uint8_t sign_idx, const int8_t *q8) {
    uint64_t grid = g_iq2_grid[grid_idx];
    uint8_t raw = g_iq2_signs[sign_idx & 127u];
    uint32_t parity = oracle_popcount_u32((uint32_t)raw) & 1u;
    uint8_t s = (uint8_t)(raw ^ (uint8_t)(parity << 7));
    int32_t sum = 0;
    for (int b = 0; b < 8; b++) {
        int32_t v = (int32_t)(int8_t)((grid >> (b * 8)) & 0xffu);
        if ((s >> b) & 1u) v = -v;
        sum += v * (int32_t)q8[b];
    }
    return sum;
}

static int32_t oracle_iq2_pair_dot16(uint8_t g0, uint8_t s0, uint8_t g1, uint8_t s1, const int8_t *q8) {
    return oracle_iq2_grid_dot8(g0, s0, q8) + oracle_iq2_grid_dot8(g1, s1, q8 + 8);
}

/* Packs one ib32 group (32 elements: 4 grid indices, 4 sign indices, 1
 * scale nibble) into the 4 uint16 codes the real format uses, inverting
 * dev_dot_iq2_xxs_q8_K_block's aux0/aux1 decode exactly. */
static void oracle_iq2_pack_ib32(uint16_t out4[4], uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3,
                                 uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                                 uint32_t ls_nibble) {
    uint32_t aux0 = (uint32_t)a0 | ((uint32_t)a1 << 8) | ((uint32_t)a2 << 16) | ((uint32_t)a3 << 24);
    uint32_t aux1 = (s0 & 127u) | ((s1 & 127u) << 7) | ((s2 & 127u) << 14) | ((s3 & 127u) << 21) |
                    ((ls_nibble & 15u) << 28);
    out4[0] = (uint16_t)(aux0 & 0xffffu);
    out4[1] = (uint16_t)(aux0 >> 16);
    out4[2] = (uint16_t)(aux1 & 0xffffu);
    out4[3] = (uint16_t)(aux1 >> 16);
}

/* Packs one 66-byte cuda_block_iq2_xxs (ds4_rocm.cu:85-88): d (f16) plus
 * 32 packed u16 codes, 8 groups of 4. */
static void oracle_iq2_pack_block(uint8_t out[66], float d, const uint16_t qs[32]) {
    uint16_t dh = f32_to_f16_bits(d);
    memcpy(out, &dh, 2);
    memcpy(out + 2, qs, 64);
}

static void iq2_fill_row(uint8_t out[66], uint32_t phase, uint32_t expert, uint32_t row) {
    uint16_t qs[32];
    for (uint32_t g = 0; g < 8u; g++) {
        uint8_t a0 = (uint8_t)((phase * 7u + expert * 13u + row * 5u + g * 3u) % 256u);
        uint8_t a1 = (uint8_t)((phase * 11u + expert * 17u + row * 7u + g * 5u + (expert * row) % 17u) % 256u);
        uint8_t a2 = (uint8_t)((phase * 13u + expert * 19u + row * 11u + g * 7u) % 256u);
        uint8_t a3 = (uint8_t)((phase * 17u + expert * 23u + row * 13u + g * 11u + (g * row) % 13u) % 256u);
        uint32_t s0 = (phase + expert * 3u + row * 2u + g) % 128u;
        uint32_t s1 = (phase * 2u + expert + row * 5u + g * 2u) % 128u;
        uint32_t s2 = (phase * 3u + expert * 5u + row + g * 3u) % 128u;
        uint32_t s3 = (phase * 5u + expert * 2u + row * 3u + g * 5u) % 128u;
        uint32_t ls_nibble = (phase + expert + row + g) % 16u;
        oracle_iq2_pack_ib32(&qs[g * 4u], a0, a1, a2, a3, s0, s1, s2, s3, ls_nibble);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row) % 7u);
    oracle_iq2_pack_block(out, d, qs);
}

static float oracle_iq2_dot_block(const uint8_t *blk, const oracle_q8k_block *y) {
    uint16_t dbits;
    memcpy(&dbits, blk, 2);
    const float xd = f16_bits_to_f32(dbits);
    const uint16_t *q2 = (const uint16_t *)(blk + 2);
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        uint32_t ls = 2u * (aux1 >> 28) + 1u;
        uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = oracle_iq2_pair_dot16(a0, (uint8_t)((aux1 >> 0) & 127u), a1,
                                             (uint8_t)((aux1 >> 7) & 127u), q8);
        q8 += 16;
        sumi += oracle_iq2_pair_dot16(a2, (uint8_t)((aux1 >> 14) & 127u), a3,
                                      (uint8_t)((aux1 >> 21) & 127u), q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

/* Packs one 84-byte cuda_block_q2_K (ds4_rocm.cu:65-70): 16 bytes of
 * packed (scale, min) nibble pairs, 64 bytes of 2-bit codes, d and dmin
 * as f16.  The per-element byte/shift placement is lifted directly from
 * ds4.c's own accessor q2_k_value_f32 (ds4.c:3494-3505: group = idx/16,
 * q_base = 32*(group/8) + 16*(group&1), shift = ((group/2)&3)*2), which
 * this test also uses as ground truth for the roundtrip check below --
 * not derived by hand from the dot-product loop nesting, to avoid
 * transcribing that nesting incorrectly. */
static void oracle_q2k_pack_block(uint8_t out[84], float d, float dmin,
                                  const uint8_t scales[16], const uint8_t nib[256]) {
    memset(out, 0, 84);
    memcpy(out, scales, 16);
    for (uint32_t idx = 0; idx < 256u; idx++) {
        uint32_t group = idx / 16u;
        uint32_t l = idx - group * 16u;
        uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        uint32_t shift = ((group / 2u) & 3u) * 2u;
        out[16u + q_base + l] = (uint8_t)(out[16u + q_base + l] | ((nib[idx] & 3u) << shift));
    }
    uint16_t dh = f32_to_f16_bits(d), dminh = f32_to_f16_bits(dmin);
    memcpy(out + 80, &dh, 2);
    memcpy(out + 82, &dminh, 2);
}

/* Read back one packed element the same way q2_k_value_f32 would (minus
 * the dequant scale), to self-check oracle_q2k_pack_block before trusting
 * it for anything else. */
static uint8_t oracle_q2k_read_nib(const uint8_t blk[84], uint32_t idx) {
    uint32_t group = idx / 16u;
    uint32_t l = idx - group * 16u;
    uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    uint32_t shift = ((group / 2u) & 3u) * 2u;
    return (uint8_t)((blk[16u + q_base + l] >> shift) & 3u);
}

static int test_q2k_pack_roundtrip(void) {
    uint8_t scales[16], nib[256], blk[84];
    for (int i = 0; i < 16; i++) scales[i] = (uint8_t)((i * 13 + 7) & 0xff);
    for (int i = 0; i < 256; i++) nib[i] = (uint8_t)((i * 3 + i / 7) & 3);
    oracle_q2k_pack_block(blk, 0.5f, 0.25f, scales, nib);
    for (uint32_t idx = 0; idx < 256u; idx++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "q2k_pack_roundtrip: idx %u mismatch", idx);
        CHECK(oracle_q2k_read_nib(blk, idx) == nib[idx], msg);
    }
    fprintf(stderr, "  test_q2k_pack_roundtrip OK\n");
    return 0;
}

static void q2k_fill_row(uint8_t out[84], uint32_t phase, uint32_t expert, uint32_t row) {
    uint8_t scales[16], nib[256];
    for (int g = 0; g < 16; g++) {
        uint8_t sc = (uint8_t)(1u + (phase * 5u + expert * 7u + row * 3u + (uint32_t)g * 5u) % 15u);
        uint8_t mn = (uint8_t)((phase * 3u + expert * 11u + row * 2u + (uint32_t)g * 13u) % 15u);
        scales[g] = (uint8_t)((sc & 0x0fu) | (uint8_t)(mn << 4));
    }
    for (int k = 0; k < 256; k++) {
        nib[k] = (uint8_t)((phase * 19u + expert * 13u + row * 17u + (uint32_t)k +
                            (expert * row) % 5u + ((uint32_t)k * row) % 7u) % 4u);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row) % 7u);
    const float dmin = 0.001f * (float)((phase + expert * 2u + row) % 5u);
    oracle_q2k_pack_block(out, d, dmin, scales, nib);
}

static float oracle_q2k_dot_block16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += (int32_t)((q2[i] >> shift) & 3) * (int32_t)q8[i];
    return (float)sum;
}

static float oracle_q2k_dot_block(const uint8_t *blk, const oracle_q8k_block *y) {
    const uint8_t *sc = blk;
    const uint8_t *q2 = blk + 16;
    uint16_t dbits, dminbits;
    memcpy(&dbits, blk + 80, 2);
    memcpy(&dminbits, blk + 82, 2);
    const float dall = y->d * f16_bits_to_f32(dbits);
    const float dmin = y->d * f16_bits_to_f32(dminbits);
    int32_t summs = 0;
    for (int j = 0; j < 16; j++) summs += (int32_t)y->bsums[j] * (int32_t)(sc[j] >> 4);
    int32_t isum = 0;
    int is = 0;
    const int8_t *q8 = y->qs;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * (int32_t)oracle_q2k_dot_block16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * (int32_t)oracle_q2k_dot_block16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

typedef float (*oracle_dot_fn)(const uint8_t *blk, const oracle_q8k_block *y);

/* Generic one-token oracle: layer_routed_moe_one_prealloc's shape (gate/up
 * against Q8_K-quantised x, SwiGLU, requantise mid, down against
 * Q8_K-quantised mid, ascending slot-order sum), parameterised over which
 * dot product gate/up and down each use.  Covers Q4_K (gate_dot==down_dot
 * == oracle_q4k_dot_block would work too, though the existing per-format
 * test above predates this and is left as-is), IQ2_XXS/Q2_K (iq2_path),
 * IQ2_XXS/IQ2_XXS (iq2_iq2_path) and Q2_K/Q2_K (q2k_path). */
static void oracle_moe_one_token(const rm_model *m, uint32_t mid_dim, uint32_t out_dim,
                                 uint32_t n_expert, const int32_t *sel, const float *w,
                                 const float *x, float clamp, oracle_dot_fn gate_dot,
                                 oracle_dot_fn down_dot, float *out) {
    oracle_q8k_block xq;
    oracle_q8k_quantize_block(x, &xq);

    float *mid = malloc((size_t)n_expert * mid_dim * sizeof(float));
    for (uint32_t s = 0; s < n_expert; s++) {
        const uint32_t expert = (uint32_t)sel[s];
        for (uint32_t row = 0; row < mid_dim; row++) {
            const uint8_t *gate_blk = m->model + m->gate_offset + (uint64_t)expert * m->gate_expert_bytes +
                                      (uint64_t)row * m->gate_row_bytes;
            const uint8_t *up_blk = m->model + m->up_offset + (uint64_t)expert * m->gate_expert_bytes +
                                    (uint64_t)row * m->gate_row_bytes;
            float gate = gate_dot(gate_blk, &xq);
            float up = gate_dot(up_blk, &xq);
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid[(size_t)s * mid_dim + row] = oracle_silu(gate) * up * w[s];
        }
    }

    oracle_q8k_block *midq = malloc((size_t)n_expert * sizeof(oracle_q8k_block));
    for (uint32_t s = 0; s < n_expert; s++) {
        oracle_q8k_quantize_block(mid + (size_t)s * mid_dim, &midq[s]);
    }

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_expert; s++) {
            const uint32_t expert = (uint32_t)sel[s];
            const uint8_t *down_blk = m->model + m->down_offset + (uint64_t)expert * m->down_expert_bytes +
                                      (uint64_t)row * m->down_row_bytes;
            acc += down_dot(down_blk, &midq[s]);
        }
        out[row] = acc;
    }
    free(mid);
    free(midq);
}

static void iq2_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            uint8_t blk[RM_IQ2_BLOCK_BYTES];
            iq2_fill_row(blk, Q4K_PHASE_GATE, e, row);
            memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
            iq2_fill_row(blk, Q4K_PHASE_UP, e, row);
            memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            uint8_t blk[RM_Q2K_BLOCK_BYTES];
            q2k_fill_row(blk, Q4K_PHASE_DOWN, e, row);
            memcpy(m->model + m->down_offset + e * m->down_expert_bytes + (uint64_t)row * m->down_row_bytes,
                   blk, sizeof(blk));
        }
    }
}

static void iq2iq2_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            uint8_t blk[RM_IQ2_BLOCK_BYTES];
            iq2_fill_row(blk, Q4K_PHASE_GATE, e, row);
            memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
            iq2_fill_row(blk, Q4K_PHASE_UP, e, row);
            memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            uint8_t blk[RM_IQ2_BLOCK_BYTES];
            iq2_fill_row(blk, Q4K_PHASE_DOWN, e, row);
            memcpy(m->model + m->down_offset + e * m->down_expert_bytes + (uint64_t)row * m->down_row_bytes,
                   blk, sizeof(blk));
        }
    }
}

static void q2k_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            uint8_t blk[RM_Q2K_BLOCK_BYTES];
            q2k_fill_row(blk, Q4K_PHASE_GATE, e, row);
            memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
            q2k_fill_row(blk, Q4K_PHASE_UP, e, row);
            memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + (uint64_t)row * m->gate_row_bytes,
                   blk, sizeof(blk));
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            uint8_t blk[RM_Q2K_BLOCK_BYTES];
            q2k_fill_row(blk, Q4K_PHASE_DOWN, e, row);
            memcpy(m->model + m->down_offset + e * m->down_expert_bytes + (uint64_t)row * m->down_row_bytes,
                   blk, sizeof(blk));
        }
    }
}

/* rm_call_one/batch wrappers per format, mirroring the Q4_K precedent
 * (test_q4k_decode/test_q4k_batch) but parameterised over gate_type/
 * down_type and the pair of oracle dot functions, to avoid tripling the
 * decode/batch/decode-matches-batch boilerplate across the three
 * additional formats. */
static int rm_decode_test(const char *name, uint32_t gate_type, uint32_t down_type,
                          uint32_t gate_block_bytes, uint32_t down_block_bytes,
                          void (*fill_model)(rm_model *, uint32_t, uint32_t, uint32_t),
                          oracle_dot_fn gate_dot, oracle_dot_fn down_dot) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    char msg[96];
    snprintf(msg, sizeof(msg), "%s: call failed", name);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          msg);

    float want[RM_OUT_DIM];
    oracle_moe_one_token(&m, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT, sel, w, x, 0.0f,
                         gate_dot, down_dot, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        snprintf(msg, sizeof(msg), "%s: value mismatch", name);
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4, msg);
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

static int rm_batch_test(const char *name, uint32_t gate_type, uint32_t down_type,
                         uint32_t gate_block_bytes, uint32_t down_block_bytes,
                         uint32_t n_tokens,
                         void (*fill_model)(rm_model *, uint32_t, uint32_t, uint32_t),
                         oracle_dot_fn gate_dot, oracle_dot_fn down_dot) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t *sel = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    float *w = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(float));
    float *xv = malloc((size_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        q4k_fill_selected(sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT, t,
                          RM_N_EXPERT, RM_N_TOTAL_EXPERT);
        q4k_fill_x_row(xv + (size_t)t * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, t);
    }
    ds4_gpu_tensor_write(selected, 0, sel, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor_write(weights, 0, w, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    bool mid_is_f16 = true;
    char msg[96];
    snprintf(msg, sizeof(msg), "%s(n=%u): call failed", name, n_tokens);
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, gate_type, down_type, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, /*layer_index=*/0u, n_tokens,
              &mid_is_f16, /*force_resident=*/false) != 0,
          msg);
    snprintf(msg, sizeof(msg), "%s(n=%u): mid_is_f16 must be written false", name, n_tokens);
    CHECK(mid_is_f16 == false, msg);

    float *got = malloc((size_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));

    float want_row[RM_OUT_DIM];
    for (uint32_t t = 0; t < n_tokens; t++) {
        oracle_moe_one_token(&m, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                             sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT,
                             xv + (size_t)t * RM_EXPERT_IN_DIM, 0.0f, gate_dot, down_dot,
                             want_row);
        for (int i = 0; i < RM_OUT_DIM; i++) {
            snprintf(msg, sizeof(msg), "%s(n=%u): token %u value mismatch", name, n_tokens, t);
            CHECK_CLOSE(got[(size_t)t * RM_OUT_DIM + i], want_row[i],
                       fabs(want_row[i]) * 1e-3 + 1e-4, msg);
        }
    }

    free(sel);
    free(w);
    free(xv);
    free(got);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  %s(n_tokens=%u) OK\n", name, n_tokens);
    return 0;
}

/* Decode versus batch(n=1) cross-check, mirroring test_q4k_decode_matches_
 * batch_of_one: both dispatch to the same decode gate/up kernel and the
 * same direct-to-out down kernel, so they must agree exactly (well within
 * float-reordering tolerance) on identical input. */
static int rm_decode_matches_batch_of_one(const char *name, uint32_t gate_type,
                                          uint32_t down_type, uint32_t gate_block_bytes,
                                          uint32_t down_block_bytes,
                                          void (*fill_model)(rm_model *, uint32_t, uint32_t,
                                                             uint32_t)) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t1 = rm_build_tensors();
    rm_tensors t2 = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/2, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/2);
    ds4_gpu_tensor_write(t1.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t1.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t1.x, 0, x, sizeof(x));
    ds4_gpu_tensor_write(t2.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t2.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t2.x, 0, x, sizeof(x));

    char msg[96];
    snprintf(msg, sizeof(msg), "%s: decode call failed", name);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t1.out, t1.gate, t1.up, t1.mid, t1.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t1.selected, t1.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t1.x,
              NULL, 0u, false) != 0,
          msg);
    snprintf(msg, sizeof(msg), "%s: batch(n=1) call failed", name);
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t2.out, t2.gate, t2.up, t2.mid, t2.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t2.selected, t2.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t2.x,
              0u, /*n_tokens=*/1u, NULL, false) != 0,
          msg);

    float got1[RM_OUT_DIM], got2[RM_OUT_DIM];
    ds4_gpu_tensor_read(t1.out, 0, got1, sizeof(got1));
    ds4_gpu_tensor_read(t2.out, 0, got2, sizeof(got2));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        snprintf(msg, sizeof(msg), "%s: decode vs batch(n=1) mismatch", name);
        CHECK_CLOSE(got1[i], got2[i], fabs(got1[i]) * 1e-4 + 1e-5, msg);
    }

    rm_free_tensors(&t1);
    rm_free_tensors(&t2);
    free(m.model);
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

/* ---- iq2_path: gate_type==16 (IQ2_XXS), down_type==10 (Q2_K) ---------- */

static int test_iq2_decode(void) {
    return rm_decode_test("test_iq2_decode", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES,
                          iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
static int test_iq2_batch_small(void) {
    return rm_batch_test("test_iq2_batch", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 3u,
                         iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
static int test_iq2_batch_large(void) {
    return rm_batch_test("test_iq2_batch", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 40u,
                         iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
static int test_iq2_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_iq2_decode_matches_batch_of_one", 16u, 10u,
                                          RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, iq2_fill_model);
}

/* ---- iq2_iq2_path: gate_type==16, down_type==16 (both IQ2_XXS) -------- */

static int test_iq2iq2_decode(void) {
    return rm_decode_test("test_iq2iq2_decode", 16u, 16u, RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES,
                          iq2iq2_fill_model, oracle_iq2_dot_block, oracle_iq2_dot_block);
}
static int test_iq2iq2_batch(void) {
    return rm_batch_test("test_iq2iq2_batch", 16u, 16u, RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES, 12u,
                         iq2iq2_fill_model, oracle_iq2_dot_block, oracle_iq2_dot_block);
}
static int test_iq2iq2_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_iq2iq2_decode_matches_batch_of_one", 16u, 16u,
                                          RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES,
                                          iq2iq2_fill_model);
}

/* ---- q2k_path: gate_type==10, down_type==10 (both Q2_K) --------------- */

static int test_q2k_decode(void) {
    return rm_decode_test("test_q2k_decode", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES,
                          q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_batch_small(void) {
    return rm_batch_test("test_q2k_batch", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 5u,
                         q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_batch_large(void) {
    return rm_batch_test("test_q2k_batch", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 40u,
                         q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_q2k_decode_matches_batch_of_one", 10u, 10u,
                                          RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, q2k_fill_model);
}

/* An ablation ("route q2k_path inputs through the
 * untagged down kernels and confirm the test catches the difference") is
 * the second, explicitly anticipated outcome rather than the first: in
 * this port, q2k_path's down projection and iq2_path's down projection
 * are the same function, sycl_moe_q2k_down_direct
 * (sycl/ds4_sycl_moe.hpp), because both are validated against the same
 * Q8_K-prequantised-mid CPU oracle (matvec_q2_k_experts_accum_prequant,
 * ds4.c:8297) rather than against ROCm's own kernel selection. There is
 * no separate "route through the other kernel" test to write: both
 * test_q2k_decode/test_q2k_batch above and test_iq2_decode/test_iq2_batch
 * already exercise this identical function (confirmed by reading the
 * dispatch call sites in sycl/ds4_sycl_moe_launch.hpp, not inferred), so
 * running one format's test already is running the other's down kernel.
 * ROCm's own q2k_path and iq2_path down kernels genuinely differ
 * (moe_down_q2K_expert_batch_sharedmid_kernel's raw-float mid versus
 * moe_down_sum6_qwarp32_kernel's Q8_K-quantised mid); this port merges
 * them deliberately, so they are alike by design here, not by an
 * ablation that failed to discriminate. */

/* The down projection's central property, tested directly rather than through the ABI:
 * the down projection sums in ascending SLOT order, matching
 * moe_sum_kernel and layer_routed_moe_one_prealloc, not ascending
 * expert-id order (ds4-sycl-moe-reference.md section 4(a)). Every ABI
 * test above uses RM_N_EXPERT == 2, and IEEE754 addition of exactly two
 * values is exactly commutative (a+b bit-equals b+a always), so no
 * 2-term test can ever catch an order bug. This test uses 3 slots
 * engineered so two extreme, exactly-opposite-sign values cancel to
 * precisely 0.0 only when summed adjacently, and a third, ~6e8 times
 * smaller value survives only when it is not the first term combined
 * with either extreme -- summing in slot order [big+, big-, small]
 * gives exactly `small`; any other order that separates the two extremes
 * gives exactly 0.0 (verified by hand: float addition of two bit-exact
 * negations is exact regardless of what else is in flight, and 6e8 is
 * far past float32's ~1.19e-7 relative precision, so adding `small` to
 * either extreme first always rounds it away). */
static int test_q2k_down_slot_order(void) {
    enum { N_EXPERT = 3, N_TOTAL_EXPERT = 6 };
    uint8_t scales[16], nib[256];
    for (int g = 0; g < 16; g++) scales[g] = 15u; /* scale=15, min=0 */
    /* Constant code (not alternating): x below is also constant, so the
     * per-element products all share one sign and accumulate rather than
     * partially cancelling, which a symmetric-around-zero x would do. */
    for (int k = 0; k < 256; k++) nib[k] = 1u;

    const uint16_t d_pos_bits = f32_to_f16_bits(60000.0f);
    const uint16_t d_neg_bits = (uint16_t)(d_pos_bits ^ 0x8000u); /* exact negation */
    const uint16_t d_small_bits = f32_to_f16_bits(0.001f);
    const uint16_t zero16 = 0;

    uint8_t down[N_TOTAL_EXPERT][84];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) {
        memset(down[e], 0, sizeof(down[e]));
        memcpy(down[e], scales, 16);
        for (uint32_t idx = 0; idx < 256u; idx++) {
            uint32_t group = idx / 16u;
            uint32_t l = idx - group * 16u;
            uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
            uint32_t shift = ((group / 2u) & 3u) * 2u;
            down[e][16u + q_base + l] =
                (uint8_t)(down[e][16u + q_base + l] | ((nib[idx] & 3u) << shift));
        }
        memcpy(down[e] + 82, &zero16, 2); /* dmin = 0 for every expert */
    }
    memcpy(down[0] + 80, &d_pos_bits, 2);   /* expert 0: +big */
    memcpy(down[5] + 80, &d_neg_bits, 2);   /* expert 5: -big, exact negation */
    memcpy(down[2] + 80, &d_small_bits, 2); /* expert 2: small */

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 10.0f; /* constant: no cross-element cancellation */
    oracle_q8k_block xq;
    oracle_q8k_quantize_block(x, &xq);

    uint8_t midq_bytes[3 * Q8_K_BLOCK_BYTES];
    for (int s = 0; s < N_EXPERT; s++) {
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES, &xq.d, 4);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 4, xq.qs, 256);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 260, xq.bsums, 32);
    }

    uint8_t down_flat[N_TOTAL_EXPERT * 84];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) memcpy(down_flat + e * 84, down[e], 84);

    /* Slot order: expert 0 (+big), expert 5 (-big), expert 2 (small). */
    int32_t sel[N_EXPERT] = {0, 5, 2};
    float got = 0.0f;
    CHECK(ds4_sycl_moe_test_q2k_down_direct(down_flat, midq_bytes, sel,
                                            /*down_expert_bytes=*/84u,
                                            /*down_row_bytes=*/84u, /*midq_blocks=*/1u,
                                            /*out_dim=*/1u, N_EXPERT, /*n_tokens=*/1u,
                                            N_TOTAL_EXPERT, &got) != 0,
          "q2k_down_slot_order: call failed");

    float want = oracle_q2k_dot_block(down[2], &xq);
    CHECK(fabsf(want) > 1.0f, "q2k_down_slot_order: test data too weak (small term near zero)");
    CHECK_CLOSE(got, want, fabs(want) * 1e-3 + 1e-4,
               "q2k_down_slot_order: slot-order sum must equal the small term exactly, "
               "not be swallowed by the two cancelling extremes");
    fprintf(stderr, "  test_q2k_down_slot_order OK (got %.6g, want %.6g)\n", (double)got,
            (double)want);
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
    if (test_dispatcher_iq2_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_q2k_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mxfp4_not_yet_implemented() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_add_in_rejected() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mid_is_f16_written_false() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_build_plan_validation() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_scale_pack_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch(8u) != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch(40u) != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_scratch_precondition_failure() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_pack_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_batch_small() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_batch_large() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_batch() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_batch_small() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_batch_large() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_down_slot_order() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_moe OK\n");
    return 0;
}
