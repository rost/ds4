/* Correctness tests for the attention output projection: the grouped Q8_0
 * A-stage kernel family (tested directly through test-only hooks before
 * either ABI entry existed, mirroring
 * ds4_sycl_test_indexed_topk_sort_512_asc's pattern) and the three ABI
 * entries in sycl/ds4_sycl_attention_output.hpp.
 *
 * The oracle is ds4.c: layer_grouped_out_one (:10482-10498, single token)
 * and layer_grouped_out_batch (:10517-10535, batched), both of which
 * quantise the activation to Q8_0 (quantize_q8_0_activation /
 * quantize_q8_0_activation_batch) before the grouped int8 dot product
 * (matvec_q8_0_grouped_rows / matmul_q8_0_grouped_batch), then call the
 * plain Q8_0 matvec/matmul (matvec_q8_0 / matmul_q8_0_batch) for the
 * B-stage. ds4.c's statics cannot be linked, so every oracle below
 * reimplements the documented formula from source, with ds4.c line numbers
 * cited. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same technique as test_sycl_matmul.c's test_float_to_half: plain
 * truncation, safe here because every decode path reads back the same bit
 * pattern it was encoded with. */
static uint16_t test_float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Encodes one Q8_0 row (in_dim wide) whose quantised int8 payload carries an
 * (o, k) product interaction term (not a plain a*i + b, per spec 6f/6i/6n:
 * affine test data makes every element mutually proportional or lands on
 * clean rounding ties far more often than real weights do). Mirrors
 * test_sycl_matmul.c's test_encode_q8_0_row. */
static void encode_q8_0_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t braw = test_float_to_half(0.037f * (float)(o + 3u * blk + 1u));
        bp[0] = (unsigned char)(braw & 0xFFu);
        bp[1] = (unsigned char)((braw >> 8) & 0xFFu);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const int qv = (k < in_dim)
                    ? (int)(((o + 1u) * (k + 5u) + (o * k) % 7u) % 13u) - 6
                    : 0;
            bp[2 + idx] = (unsigned char)(signed char)qv;
        }
    }
}

/* Oracle: quantize_q8_0_activation, ds4.c:7177-7200. One row of length
 * in_dim, rounding to nearest (ties to even, matching lrintf's default
 * mode), clamped to [-128, 127]. */
static void oracle_quantize_q8_0_row(int8_t *xq, float *xscale, const float *x,
                                     uint32_t in_dim) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t i0 = b * 32u;
        const uint32_t bn = (in_dim - i0 < 32u) ? (in_dim - i0) : 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        xscale[b] = d;
        for (uint32_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint32_t i = bn; i < 32u; i++) xq[i0 + i] = 0;
    }
}

/* Oracle: dot_q8_0_row, ds4.c:6940-7001 (scalar branch), applied to one
 * already-quantised row against one Q8_0 weight row. */
static float oracle_dot_q8_0_row(const unsigned char *row, const int8_t *xq,
                                 const float *xscale, uint32_t in_dim) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    double acc = 0.0;
    for (uint32_t b = 0; b < blocks; b++) {
        const unsigned char *bp = row + (size_t)b * 34u;
        const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
        const float wscale = oracle_half_to_float(raw);
        int32_t dot = 0;
        for (uint32_t i = 0; i < 32u; i++) dot += (int32_t)(signed char)bp[2 + i] * (int32_t)xq[b * 32u + i];
        acc += (double)wscale * (double)xscale[b] * (double)dot;
    }
    return (float)acc;
}

/* Oracle: layer_grouped_out_one's first line, matvec_q8_0_grouped_rows
 * (ds4.c:7704-7741) via matvec_q8_0_grouped_worker (:7249-7261): one head
 * vector per group, quantised independently, dotted against its group's
 * weight rows. `w` is n_groups*rank rows of row_bytes each, row layout
 * [group][row_in_group]. */
static void oracle_grouped_out_low(float *low, const float *heads,
                                   const unsigned char *w, uint32_t n_groups,
                                   uint32_t group_dim, uint32_t rank) {
    const uint32_t blocks = (group_dim + 31u) / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    int8_t *xq = malloc((size_t)blocks * 32u * sizeof(int8_t));
    float  *xscale = malloc((size_t)blocks * sizeof(float));
    for (uint32_t g = 0; g < n_groups; g++) {
        oracle_quantize_q8_0_row(xq, xscale, heads + (size_t)g * group_dim, group_dim);
        for (uint32_t r = 0; r < rank; r++) {
            const unsigned char *row = w + ((size_t)g * rank + r) * row_bytes;
            low[(size_t)g * rank + r] = oracle_dot_q8_0_row(row, xq, xscale, group_dim);
        }
    }
    free(xq);
    free(xscale);
}

/* Oracle: layer_grouped_out_batch's first line, matmul_q8_0_grouped_batch
 * (ds4.c:7780-7823): same as above, generalised to n_tok independent
 * token rows, each with its own n_groups activations. */
static void oracle_grouped_out_batch_low(float *low, const float *heads,
                                         const unsigned char *w, uint32_t n_tok,
                                         uint32_t n_groups, uint32_t group_dim,
                                         uint32_t rank) {
    for (uint32_t t = 0; t < n_tok; t++) {
        oracle_grouped_out_low(low + (size_t)t * n_groups * rank,
                               heads + (size_t)t * n_groups * group_dim, w,
                               n_groups, group_dim, rank);
    }
}

/* Oracle for the reused B-stage: matches ds4_gpu_matmul_q8_0_tensor's own
 * documented formula (sycl_q8_0_matmul_launch in ds4_sycl_matmul.hpp,
 * itself ported from matmul_q8_0_f32_warp8_kernel /
 * _batch_warp8_kernel, rocm/ds4_rocm_q8.cuh:440-520): raw float activation
 * times dequantised Q8_0 weight, NOT activation-quantised -- deliberately
 * different from ds4.c's own matvec_q8_0 (which does quantise): this port
 * reuses the already-landed, already-reviewed dense-matmul entry as
 * the B-stage rather than porting a second B-stage that matches ds4.c
 * exactly. Used only for the differential/end-to-end sanity checks below,
 * not as the tight A-stage oracle. */
static void oracle_matmul_q8_0_raw(float *out, const float *x,
                                   const unsigned char *w, uint32_t in_dim,
                                   uint32_t out_dim, uint32_t n_tok,
                                   uint64_t row_bytes) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const unsigned char *row = w + (size_t)o * row_bytes;
            double sum = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) {
                const uint32_t blk = k / 32u;
                const uint32_t idx = k % 32u;
                const unsigned char *bp = row + (size_t)blk * 34u;
                const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
                const float scale = oracle_half_to_float(raw);
                const signed char qv = (signed char)bp[2 + idx];
                sum += (double)x[t * in_dim + k] * (double)scale * (double)qv;
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* Test data helper: an (index-interaction, not affine) activation pattern
 * for one row of length n, distinguishable per (row, n) pair. */
static void fill_activation_row(float *x, uint32_t n, uint32_t row_seed) {
    for (uint32_t i = 0; i < n; i++) {
        x[i] = (float)(((row_seed + 1u) * (i + 3u) + (row_seed * i) % 11u) % 17u) - 8.0f;
    }
}

/* ---- Kernel-family test-only hooks ---- */

extern int ds4_sycl_test_quantize_q8_0_rows(ds4_gpu_tensor *xq, ds4_gpu_tensor *xscale,
                                            const ds4_gpu_tensor *x, uint32_t in_dim,
                                            uint32_t n_rows);
extern int ds4_sycl_test_grouped_q8_0_a_preq(ds4_gpu_tensor *low, const ds4_gpu_tensor *w,
                                             const ds4_gpu_tensor *xq,
                                             const ds4_gpu_tensor *xscale, uint32_t group_dim,
                                             uint32_t rank, uint32_t n_groups,
                                             uint32_t n_tokens);

/* IN_DIM deliberately not a multiple of 32 (ragged last block) and N_ROWS >
 * 1 so the per-row stride is exercised. */
static int test_quantize_q8_0_rows(void) {
    enum { IN_DIM = 41, N_ROWS = 3 };
    const uint32_t blocks = (IN_DIM + 31u) / 32u;

    float   x[N_ROWS * IN_DIM];
    int8_t  want_xq[N_ROWS * 2 * 32];
    float   want_xscale[N_ROWS * 2];
    int8_t  got_xq[N_ROWS * 2 * 32];
    float   got_xscale[N_ROWS * 2];

    for (uint32_t r = 0; r < N_ROWS; r++) fill_activation_row(x + r * IN_DIM, IN_DIM, r);
    for (uint32_t r = 0; r < N_ROWS; r++) {
        oracle_quantize_q8_0_row(want_xq + (size_t)r * blocks * 32u,
                                 want_xscale + (size_t)r * blocks, x + r * IN_DIM, IN_DIM);
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *txq = ds4_gpu_tensor_alloc(sizeof(got_xq));
    ds4_gpu_tensor *txs = ds4_gpu_tensor_alloc(sizeof(got_xscale));
    CHECK(tx && txq && txs, "quantize_q8_0_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "quantize_q8_0_rows: write x");

    CHECK(ds4_sycl_test_quantize_q8_0_rows(txq, txs, tx, IN_DIM, N_ROWS) != 0,
          "quantize_q8_0_rows: call");
    CHECK(ds4_gpu_tensor_read(txq, 0, got_xq, sizeof(got_xq)) != 0,
          "quantize_q8_0_rows: read xq");
    CHECK(ds4_gpu_tensor_read(txs, 0, got_xscale, sizeof(got_xscale)) != 0,
          "quantize_q8_0_rows: read xscale");

    for (uint32_t i = 0; i < N_ROWS * blocks * 32u; i++) {
        CHECK(got_xq[i] == want_xq[i], "quantize_q8_0_rows: xq mismatch");
    }
    for (uint32_t i = 0; i < N_ROWS * blocks; i++) {
        CHECK_CLOSE(got_xscale[i], want_xscale[i], 1e-6, "quantize_q8_0_rows: xscale mismatch");
    }

    CHECK(ds4_sycl_test_quantize_q8_0_rows(txq, txs, tx, 0u, N_ROWS) == 0,
          "quantize_q8_0_rows: zero in_dim must be rejected");
    CHECK(ds4_sycl_test_quantize_q8_0_rows(txq, txs, tx, IN_DIM, 0u) == 0,
          "quantize_q8_0_rows: zero n_rows must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(txq);
    ds4_gpu_tensor_free(txs);
    fprintf(stderr, "  test_quantize_q8_0_rows OK\n");
    return 0;
}

/* GROUP_DIM ragged, RANK and N_GROUPS both > 1 so the row/group stride
 * math is genuinely exercised, N_TOKENS > 1 so the token stride is too. */
static int test_grouped_q8_0_a_preq(void) {
    enum { GROUP_DIM = 37, RANK = 5, N_GROUPS = 3, N_TOKENS = 2 };
    const uint32_t blocks = (GROUP_DIM + 31u) / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint32_t low_dim = RANK * N_GROUPS;

    unsigned char w[N_GROUPS * RANK * 2 * 34];
    float heads[N_TOKENS * N_GROUPS * GROUP_DIM];
    int8_t xq[N_TOKENS * N_GROUPS * 2 * 32];
    float xscale[N_TOKENS * N_GROUPS * 2];
    float want_low[N_TOKENS * N_GROUPS * RANK];
    float got_low[N_TOKENS * N_GROUPS * RANK];

    for (uint32_t g = 0; g < N_GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++) {
            encode_q8_0_row(w + ((size_t)g * RANK + r) * row_bytes, GROUP_DIM, g * 17u + r);
        }
    }
    for (uint32_t t = 0; t < N_TOKENS; t++) {
        for (uint32_t g = 0; g < N_GROUPS; g++) {
            fill_activation_row(heads + ((size_t)t * N_GROUPS + g) * GROUP_DIM, GROUP_DIM,
                                t * 5u + g);
            oracle_quantize_q8_0_row(xq + ((size_t)t * N_GROUPS + g) * blocks * 32u,
                                     xscale + ((size_t)t * N_GROUPS + g) * blocks,
                                     heads + ((size_t)t * N_GROUPS + g) * GROUP_DIM, GROUP_DIM);
        }
    }
    oracle_grouped_out_batch_low(want_low, heads, w, N_TOKENS, N_GROUPS, GROUP_DIM, RANK);

    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(w));
    ds4_gpu_tensor *txq = ds4_gpu_tensor_alloc(sizeof(xq));
    ds4_gpu_tensor *txs = ds4_gpu_tensor_alloc(sizeof(xscale));
    ds4_gpu_tensor *tlow = ds4_gpu_tensor_alloc(sizeof(got_low));
    CHECK(tw && txq && txs && tlow, "grouped_q8_0_a_preq: allocation failed");
    CHECK(ds4_gpu_tensor_write(tw, 0, w, sizeof(w)) != 0 &&
          ds4_gpu_tensor_write(txq, 0, xq, sizeof(xq)) != 0 &&
          ds4_gpu_tensor_write(txs, 0, xscale, sizeof(xscale)) != 0,
          "grouped_q8_0_a_preq: write inputs");

    CHECK(ds4_sycl_test_grouped_q8_0_a_preq(tlow, tw, txq, txs, GROUP_DIM, RANK, N_GROUPS,
                                            N_TOKENS) != 0,
          "grouped_q8_0_a_preq: call");
    CHECK(ds4_gpu_tensor_read(tlow, 0, got_low, sizeof(got_low)) != 0,
          "grouped_q8_0_a_preq: read low");

    for (uint32_t i = 0; i < N_TOKENS * low_dim; i++) {
        CHECK_CLOSE(got_low[i], want_low[i], 1e-2, "grouped_q8_0_a_preq: low mismatch");
    }

    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(txq);
    ds4_gpu_tensor_free(txs);
    ds4_gpu_tensor_free(tlow);
    fprintf(stderr, "  test_grouped_q8_0_a_preq OK\n");
    return 0;
}

/* ---- The low-rank entry ---- */

static int test_attention_output_low_q8(void) {
    enum { GROUP_DIM = 37, RANK = 5, N_GROUPS = 3 };
    const uint32_t blocks = (GROUP_DIM + 31u) / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint32_t low_dim = RANK * N_GROUPS;

    unsigned char w[N_GROUPS * RANK * 2 * 34];
    float heads[N_GROUPS * GROUP_DIM];
    float want_low[N_GROUPS * RANK];
    float got_low[N_GROUPS * RANK];

    for (uint32_t g = 0; g < N_GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++) {
            encode_q8_0_row(w + ((size_t)g * RANK + r) * row_bytes, GROUP_DIM, g * 13u + r + 2u);
        }
    }
    for (uint32_t g = 0; g < N_GROUPS; g++) {
        fill_activation_row(heads + (size_t)g * GROUP_DIM, GROUP_DIM, g + 1u);
    }
    oracle_grouped_out_low(want_low, heads, w, N_GROUPS, GROUP_DIM, RANK);

    const uint64_t out_a_bytes = sizeof(w);
    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc(sizeof(heads));
    ds4_gpu_tensor *tlow = ds4_gpu_tensor_alloc(sizeof(got_low));
    CHECK(theads && tlow, "attention_output_low_q8: allocation failed");
    CHECK(ds4_gpu_tensor_write(theads, 0, heads, sizeof(heads)) != 0,
          "attention_output_low_q8: write heads");

    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, GROUP_DIM, RANK,
                                                 N_GROUPS, theads) != 0,
          "attention_output_low_q8: call");
    CHECK(ds4_gpu_tensor_read(tlow, 0, got_low, sizeof(got_low)) != 0,
          "attention_output_low_q8: read low");
    for (uint32_t i = 0; i < low_dim; i++) {
        CHECK_CLOSE(got_low[i], want_low[i], 1e-2, "attention_output_low_q8: low mismatch");
    }

    /* Swap the A matrix's two dimensions: pass group_dim/rank transposed.
     * The weight buffer is far too small for the transposed shape's byte
     * requirement, so this must be rejected outright. The remaining
     * ablations called for (dropping the Q8_0 scale, using the
     * wrong group for a head) are exercised by mutating the actual kernel
     * source and confirming the oracle comparison above fails, per this
     * project's ablation discipline (design-spec sections 6b/6j/6n): the
     * test data above already gives every group and row a numerically
     * distinct oracle value, so either mutation is guaranteed to produce a
     * visibly wrong result rather than a coincidentally matching one. */
    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, RANK, GROUP_DIM,
                                                 N_GROUPS, theads) == 0,
          "attention_output_low_q8: swapped A dimensions must be rejected");

    /* Standard validation-failure checks. */
    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, 0, RANK, N_GROUPS,
                                                 theads) == 0,
          "attention_output_low_q8: zero group_dim must be rejected");
    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, GROUP_DIM, 0, N_GROUPS,
                                                 theads) == 0,
          "attention_output_low_q8: zero rank must be rejected");
    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, GROUP_DIM, RANK, 0,
                                                 theads) == 0,
          "attention_output_low_q8: zero n_groups must be rejected");
    CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, out_a_bytes, GROUP_DIM,
                                                 RANK, N_GROUPS, theads) == 0,
          "attention_output_low_q8: out-of-range out_a_offset must be rejected");

    ds4_gpu_tensor_free(theads);
    ds4_gpu_tensor_free(tlow);
    fprintf(stderr, "  test_attention_output_low_q8 OK\n");
    return 0;
}

/* Contention-scale regression test for the A-stage's own internal ordering:
 * sycl_attention_output_a_stage (ds4_sycl_attention_output.hpp) submits the
 * quantise kernel and the grouped preq dot-product kernel back to back on
 * the same out-of-order queue, over raw USM with no automatic dependency
 * tracking (design-spec queue-ordering note). The preq kernel reads
 * xq/xscale, which the quantise kernel writes; a wait between the two
 * submissions is what makes that safe. This test's shapes are picked large
 * enough (GROUP_DIM=4096, N_GROUPS=64, RANK=16) to give both kernels many
 * work-groups in flight, per design-spec section 6j: a launch too small to
 * create contention will not expose a missing-ordering race even when one
 * exists. Repeated REPS times per process run, since a race need not
 * reproduce on every call. */
static int test_attention_output_a_stage_contention(void) {
    enum { GROUP_DIM = 4096, RANK = 16, N_GROUPS = 64, REPS = 20 };
    const uint32_t blocks = (GROUP_DIM + 31u) / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint32_t low_dim = RANK * N_GROUPS;

    unsigned char *w = malloc((size_t)N_GROUPS * RANK * row_bytes);
    float *heads = malloc((size_t)N_GROUPS * GROUP_DIM * sizeof(float));
    float *want_low = malloc((size_t)N_GROUPS * RANK * sizeof(float));
    float *got_low = malloc((size_t)N_GROUPS * RANK * sizeof(float));
    CHECK(w && heads && want_low && got_low,
          "attention_output_a_stage_contention: host allocation failed");

    for (uint32_t g = 0; g < N_GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++) {
            encode_q8_0_row(w + ((size_t)g * RANK + r) * row_bytes, GROUP_DIM, g * 13u + r + 2u);
        }
    }
    for (uint32_t g = 0; g < N_GROUPS; g++) {
        fill_activation_row(heads + (size_t)g * GROUP_DIM, GROUP_DIM, g + 1u);
    }
    oracle_grouped_out_low(want_low, heads, w, N_GROUPS, GROUP_DIM, RANK);

    const uint64_t out_a_bytes = (uint64_t)N_GROUPS * RANK * row_bytes;
    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc((size_t)N_GROUPS * GROUP_DIM * sizeof(float));
    ds4_gpu_tensor *tlow = ds4_gpu_tensor_alloc((size_t)low_dim * sizeof(float));
    CHECK(theads && tlow, "attention_output_a_stage_contention: device allocation failed");
    CHECK(ds4_gpu_tensor_write(theads, 0, heads, (size_t)N_GROUPS * GROUP_DIM * sizeof(float)) != 0,
          "attention_output_a_stage_contention: write heads");

    for (int rep = 0; rep < REPS; rep++) {
        CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow, w, out_a_bytes, 0, GROUP_DIM, RANK,
                                                     N_GROUPS, theads) != 0,
              "attention_output_a_stage_contention: call");
        CHECK(ds4_gpu_tensor_read(tlow, 0, got_low, (size_t)low_dim * sizeof(float)) != 0,
              "attention_output_a_stage_contention: read low");
        for (uint32_t i = 0; i < low_dim; i++) {
            CHECK_CLOSE(got_low[i], want_low[i], 1e-2 + 1e-2 * fabsf(want_low[i]),
                        "attention_output_a_stage_contention: low mismatch");
        }
    }

    ds4_gpu_tensor_free(theads);
    ds4_gpu_tensor_free(tlow);
    free(w);
    free(heads);
    free(want_low);
    free(got_low);
    fprintf(stderr, "  test_attention_output_a_stage_contention OK\n");
    return 0;
}

/* ---- The batch entries, F32 and F16 ---- */

extern uint16_t ds4_sycl_test_hip_round_f16_bits(float f);

static int test_attention_output_q8_batch(void) {
    enum { GROUP_DIM = 41, RANK = 6, N_GROUPS = 4, N_TOKENS = 3, OUT_DIM = 9 };
    const uint32_t blocks_a = (GROUP_DIM + 31u) / 32u;
    const uint64_t row_bytes_a = (uint64_t)blocks_a * 34u;
    const uint32_t low_dim = RANK * N_GROUPS;
    const uint32_t blocks_b = (low_dim + 31u) / 32u;
    const uint64_t row_bytes_b = (uint64_t)blocks_b * 34u;

    unsigned char wa[N_GROUPS * RANK * 2 * 34];
    unsigned char wb[OUT_DIM * 34]; /* low_dim=24 here, blocks_b=1, row_bytes_b=34 */
    float heads[N_TOKENS * N_GROUPS * GROUP_DIM];
    float want_low[N_TOKENS * N_GROUPS * RANK];
    float want_out[N_TOKENS * OUT_DIM];
    float got_low[N_TOKENS * N_GROUPS * RANK];
    float got_out[N_TOKENS * OUT_DIM];

    for (uint32_t g = 0; g < N_GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++) {
            encode_q8_0_row(wa + ((size_t)g * RANK + r) * row_bytes_a, GROUP_DIM, g * 19u + r + 3u);
        }
    }
    for (uint32_t o = 0; o < OUT_DIM; o++) {
        encode_q8_0_row(wb + (size_t)o * row_bytes_b, low_dim, o + 41u);
    }
    for (uint32_t t = 0; t < N_TOKENS; t++) {
        for (uint32_t g = 0; g < N_GROUPS; g++) {
            fill_activation_row(heads + ((size_t)t * N_GROUPS + g) * GROUP_DIM, GROUP_DIM,
                                t * 7u + g + 1u);
        }
    }
    oracle_grouped_out_batch_low(want_low, heads, wa, N_TOKENS, N_GROUPS, GROUP_DIM, RANK);
    oracle_matmul_q8_0_raw(want_out, want_low, wb, low_dim, OUT_DIM, N_TOKENS, row_bytes_b);

    unsigned char model[sizeof(wa) + sizeof(wb)];
    memcpy(model, wa, sizeof(wa));
    memcpy(model + sizeof(wa), wb, sizeof(wb));
    const uint64_t out_a_offset = 0;
    const uint64_t out_b_offset = sizeof(wa);
    const uint64_t model_size = sizeof(model);

    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc(sizeof(heads));
    ds4_gpu_tensor *tlow = ds4_gpu_tensor_alloc(sizeof(got_low));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got_out));
    CHECK(theads && tlow && tout, "attention_output_q8_batch: allocation failed");
    CHECK(ds4_gpu_tensor_write(theads, 0, heads, sizeof(heads)) != 0,
          "attention_output_q8_batch: write heads");

    CHECK(ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, NULL, NULL, model, model_size,
                                                   out_a_offset, out_b_offset, GROUP_DIM, RANK,
                                                   N_GROUPS, OUT_DIM, theads, N_TOKENS) != 0,
          "attention_output_q8_batch: call");
    CHECK(ds4_gpu_tensor_read(tlow, 0, got_low, sizeof(got_low)) != 0,
          "attention_output_q8_batch: read low");
    CHECK(ds4_gpu_tensor_read(tout, 0, got_out, sizeof(got_out)) != 0,
          "attention_output_q8_batch: read out");

    /* Tight: the A-stage matches ds4.c's own oracle (activation-quantised)
     * to within Q8_0 int8 rounding noise. */
    for (uint32_t i = 0; i < N_TOKENS * low_dim; i++) {
        CHECK_CLOSE(got_low[i], want_low[i], 1e-2, "attention_output_q8_batch: low mismatch");
    }
    /* Looser: the full pipeline's B-stage deliberately reuses
     * ds4_gpu_matmul_q8_0_tensor's raw-float (non-activation-quantised)
     * kernel rather than porting a second, ds4.c-matching B-stage (see the
     * header comment on ds4_gpu_attention_output_q8_batch_tensor), so
     * `out` is compared against a raw-float oracle built from the true
     * `want_low`, not against ds4.c's own (quantised) matvec_q8_0. */
    for (uint32_t i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got_out[i], want_out[i], 5e-2, "attention_output_q8_batch: out mismatch");
    }

    /* Cross-check: run the
     * low-rank entry once per token, then the same B-stage matmul entry
     * used above, and confirm the result matches the fused batch entry's
     * `out` tightly (both paths share the exact same kernels, so this is
     * far tighter than the raw-float-oracle comparison above -- it would
     * catch drift an oracle comparison alone can miss). */
    {
        float low_loop[N_TOKENS * N_GROUPS * RANK];
        for (uint32_t t = 0; t < N_TOKENS; t++) {
            ds4_gpu_tensor *theads_t = ds4_gpu_tensor_alloc((size_t)N_GROUPS * GROUP_DIM * sizeof(float));
            ds4_gpu_tensor *tlow_t = ds4_gpu_tensor_alloc((size_t)low_dim * sizeof(float));
            CHECK(theads_t && tlow_t, "attention_output_q8_batch: per-token allocation failed");
            CHECK(ds4_gpu_tensor_write(theads_t, 0, heads + (size_t)t * N_GROUPS * GROUP_DIM,
                                       (size_t)N_GROUPS * GROUP_DIM * sizeof(float)) != 0,
                  "attention_output_q8_batch: per-token write heads");
            CHECK(ds4_gpu_attention_output_low_q8_tensor(tlow_t, model, model_size, out_a_offset,
                                                         GROUP_DIM, RANK, N_GROUPS, theads_t) != 0,
                  "attention_output_q8_batch: per-token low-rank call");
            CHECK(ds4_gpu_tensor_read(tlow_t, 0, low_loop + (size_t)t * low_dim,
                                      (size_t)low_dim * sizeof(float)) != 0,
                  "attention_output_q8_batch: per-token read low");
            ds4_gpu_tensor_free(theads_t);
            ds4_gpu_tensor_free(tlow_t);
        }
        ds4_gpu_tensor *tlow_loop = ds4_gpu_tensor_alloc(sizeof(low_loop));
        ds4_gpu_tensor *tout_loop = ds4_gpu_tensor_alloc(sizeof(got_out));
        CHECK(tlow_loop && tout_loop, "attention_output_q8_batch: loop allocation failed");
        CHECK(ds4_gpu_tensor_write(tlow_loop, 0, low_loop, sizeof(low_loop)) != 0,
              "attention_output_q8_batch: write loop low");
        CHECK(ds4_gpu_matmul_q8_0_tensor(tout_loop, model, model_size, out_b_offset, low_dim,
                                         OUT_DIM, tlow_loop, N_TOKENS) != 0,
              "attention_output_q8_batch: loop matmul call");
        float out_loop[N_TOKENS * OUT_DIM];
        CHECK(ds4_gpu_tensor_read(tout_loop, 0, out_loop, sizeof(out_loop)) != 0,
              "attention_output_q8_batch: read loop out");
        for (uint32_t i = 0; i < N_TOKENS * OUT_DIM; i++) {
            CHECK_CLOSE(got_out[i], out_loop[i], 1e-3,
                        "attention_output_q8_batch: differential vs per-token loop mismatch");
        }
        ds4_gpu_tensor_free(tlow_loop);
        ds4_gpu_tensor_free(tout_loop);
    }

    /* The remaining required ablations (a token index held constant across
     * the batch; the B-stage reading `low` before the A-stage has
     * completed) are exercised by mutating the actual kernel source and
     * confirming the checks above fail, per this project's ablation
     * discipline: test data here already gives every token a numerically
     * distinct activation row (fill_activation_row is seeded by `t`), so a
     * constant-token mutation is guaranteed to produce a visibly wrong
     * result at any token other than the one it collapses onto. */

    CHECK(ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, NULL, NULL, model, model_size, 0,
                                                   out_b_offset, 0, RANK, N_GROUPS, OUT_DIM,
                                                   theads, N_TOKENS) == 0,
          "attention_output_q8_batch: zero group_dim must be rejected");
    CHECK(ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, NULL, NULL, model, model_size, 0,
                                                   out_b_offset, GROUP_DIM, RANK, N_GROUPS,
                                                   OUT_DIM, theads, 0) == 0,
          "attention_output_q8_batch: zero n_tokens must be rejected");
    CHECK(ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, NULL, NULL, model, model_size,
                                                   model_size, out_b_offset, GROUP_DIM, RANK,
                                                   N_GROUPS, OUT_DIM, theads, N_TOKENS) == 0,
          "attention_output_q8_batch: out-of-range out_a_offset must be rejected");

    ds4_gpu_tensor_free(theads);
    ds4_gpu_tensor_free(tlow);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_attention_output_q8_batch OK\n");
    return 0;
}

static int test_attention_output_q8_batch_f16(void) {
    enum { GROUP_DIM = 41, RANK = 6, N_GROUPS = 4, N_TOKENS = 3, OUT_DIM = 9 };
    const uint32_t blocks_a = (GROUP_DIM + 31u) / 32u;
    const uint64_t row_bytes_a = (uint64_t)blocks_a * 34u;
    const uint32_t low_dim = RANK * N_GROUPS;
    const uint32_t blocks_b = (low_dim + 31u) / 32u;
    const uint64_t row_bytes_b = (uint64_t)blocks_b * 34u;

    unsigned char wa[N_GROUPS * RANK * 2 * 34];
    unsigned char wb[OUT_DIM * 34];
    float heads[N_TOKENS * N_GROUPS * GROUP_DIM];

    for (uint32_t g = 0; g < N_GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++) {
            encode_q8_0_row(wa + ((size_t)g * RANK + r) * row_bytes_a, GROUP_DIM, g * 23u + r + 5u);
        }
    }
    for (uint32_t o = 0; o < OUT_DIM; o++) {
        encode_q8_0_row(wb + (size_t)o * row_bytes_b, low_dim, o + 71u);
    }
    for (uint32_t t = 0; t < N_TOKENS; t++) {
        for (uint32_t g = 0; g < N_GROUPS; g++) {
            fill_activation_row(heads + ((size_t)t * N_GROUPS + g) * GROUP_DIM, GROUP_DIM,
                                t * 3u + g + 2u);
        }
    }

    unsigned char model[sizeof(wa) + sizeof(wb)];
    memcpy(model, wa, sizeof(wa));
    memcpy(model + sizeof(wa), wb, sizeof(wb));
    const uint64_t out_a_offset = 0;
    const uint64_t out_b_offset = sizeof(wa);
    const uint64_t model_size = sizeof(model);

    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc(sizeof(heads));
    ds4_gpu_tensor *tlow_ref = ds4_gpu_tensor_alloc((size_t)N_TOKENS * low_dim * sizeof(float));
    ds4_gpu_tensor *tout_ref = ds4_gpu_tensor_alloc((size_t)N_TOKENS * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tlow_h = ds4_gpu_tensor_alloc((size_t)N_TOKENS * low_dim * sizeof(float));
    ds4_gpu_tensor *tout_h = ds4_gpu_tensor_alloc((size_t)N_TOKENS * OUT_DIM * sizeof(uint16_t));
    CHECK(theads && tlow_ref && tout_ref && tlow_h && tout_h,
          "attention_output_q8_batch_f16: allocation failed");
    CHECK(ds4_gpu_tensor_write(theads, 0, heads, sizeof(heads)) != 0,
          "attention_output_q8_batch_f16: write heads");

    /* Reference: the already-tested F32 batch entry, which uses the exact
     * same A-stage and B-stage kernels this entry does. */
    CHECK(ds4_gpu_attention_output_q8_batch_tensor(tout_ref, tlow_ref, NULL, NULL, model,
                                                   model_size, out_a_offset, out_b_offset,
                                                   GROUP_DIM, RANK, N_GROUPS, OUT_DIM, theads,
                                                   N_TOKENS) != 0,
          "attention_output_q8_batch_f16: F32 reference call");
    float out_ref[N_TOKENS * OUT_DIM];
    CHECK(ds4_gpu_tensor_read(tout_ref, 0, out_ref, sizeof(out_ref)) != 0,
          "attention_output_q8_batch_f16: read F32 reference out");

    CHECK(ds4_gpu_attention_output_q8_batch_f16_tensor(tout_h, tlow_h, model, model_size,
                                                       out_a_offset, out_b_offset, GROUP_DIM,
                                                       RANK, N_GROUPS, OUT_DIM, theads,
                                                       N_TOKENS) != 0,
          "attention_output_q8_batch_f16: call");
    uint16_t got_bits[N_TOKENS * OUT_DIM];
    CHECK(ds4_gpu_tensor_read(tout_h, 0, got_bits, sizeof(got_bits)) != 0,
          "attention_output_q8_batch_f16: read out bits");

    /* Exact: both entries share the same A-stage and B-stage kernels, so
     * the only difference is the final F32-to-F16 encode. Comparing
     * against the ported bit-manipulation oracle bit-for-bit (rather than
     * a tolerance on the decoded value) is the tightest test available and
     * would catch a fallback to sycl::half immediately at any input whose
     * discarded mantissa bits happen to differ between the two encoders
     * (anything but an exact tie, per spec 6k). */
    for (uint32_t i = 0; i < N_TOKENS * OUT_DIM; i++) {
        const uint16_t want_bits = ds4_sycl_test_hip_round_f16_bits(out_ref[i]);
        CHECK(got_bits[i] == want_bits,
              "attention_output_q8_batch_f16: F16 bits do not match the ported encoder");
    }

    /* Exact-tie distinguishability: per spec 6k, sycl::half and the ported
     * encoder disagree ONLY at an exact tie (13 discarded mantissa bits
     * exactly 0x1000, demonstrated on F32 0x3F801000). None of this test's
     * quantised-arithmetic outputs land on such a tie (probability zero
     * for a sum of many int8 products over essentially arbitrary weights),
     * so this test cannot itself distinguish the two encoders through the
     * full pipeline -- that is a test-data limitation of an end-to-end
     * entry test, not evidence the choice does not matter. The shared
     * helper itself is exercised at the exact tie directly by
     * tests/test_sycl_fp8_kv.c's test_f16_exact_tie_sycl_half_diverges_
     * from_oracle, which this file relies on rather than duplicating. */

    ds4_gpu_tensor_free(theads);
    ds4_gpu_tensor_free(tlow_ref);
    ds4_gpu_tensor_free(tout_ref);
    ds4_gpu_tensor_free(tlow_h);
    ds4_gpu_tensor_free(tout_h);
    fprintf(stderr, "  test_attention_output_q8_batch_f16 OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_quantize_q8_0_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_grouped_q8_0_a_preq() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_attention_output_low_q8() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_attention_output_a_stage_contention() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_attention_output_q8_batch() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_attention_output_q8_batch_f16() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_attention_output OK\n");
    return 0;
}
