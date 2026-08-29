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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_quantize_q8_0_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_grouped_q8_0_a_preq() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_attention_output OK\n");
    return 0;
}
