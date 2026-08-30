/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ROCm source line numbers cited (ROCm, not ds4.c, is the port
 * source for the dense matmul entries covered here).  Needs no model file.
 *
 * This file covers the F16 pair matmul entries (the paired dense
 * projection and the Metal-only fused compressor-store optimisation,
 * which this backend declines on every call) and the core Q8_0 dense
 * matmul entry. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <stdio.h>
#include <string.h>

/* No SYCL half type is available in a plain C test, so weights are built
 * by hand-encoding IEEE754 binary16 bit patterns.  This does plain
 * truncation of the mantissa, not round-to-nearest, so it is not a general
 * float-to-half conversion; some of this file's test inputs (for example
 * test_encode_q8_0_row's 0.05f * (o + blk + 1)) are not exactly
 * representable in binary16.  That is safe here specifically because the
 * paired oracle decode function reads back the identical stored bit
 * pattern, so both sides agree bit-for-bit regardless of whether truncation
 * or rounding produced those bits, not because the inputs are exactly
 * representable. */
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

/* Oracle: out[t][o] = sum_k x[t][k] * f16_to_f32(w[o][k]), cited to
 * rocm/ds4_rocm_matmul.cuh:873-931.  Weight storage is plain IEEE754
 * binary16, row-major, out_dim rows of in_dim half values each (no block
 * structure, unlike Q8_0). */
static void oracle_matmul_f16(float *out, const float *x, const uint16_t *w,
                              uint32_t in_dim, uint32_t out_dim,
                              uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) {
                sum += (double)x[t * in_dim + k] *
                       (double)oracle_half_to_float(w[o * in_dim + k]);
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* Oracle: out[t][o] = sum_k x[t][k] * w[o][k], cited to
 * rocm/ds4_rocm_matmul.cuh:967-991 (matmul_f32_kernel, the non-cuBLAS
 * fallback).  Weight storage is plain IEEE754 binary32, row-major, out_dim
 * rows of in_dim float values each. */
static void oracle_matmul_f32(float *out, const float *x, const float *w,
                              uint32_t in_dim, uint32_t out_dim,
                              uint32_t n_tok) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) {
                sum += (double)x[t * in_dim + k] * (double)w[o * in_dim + k];
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* in_dim deliberately not a multiple of 8/16/32/64 so a tile-remainder bug
 * would be exercised.  n_tok > 1 so the row-stride term over x is
 * exercised too, per rocm/ds4_rocm_matmul.cuh:873-931.  Weight and
 * activation data both carry an index INTERACTION term (a product, not a
 * plain a*i + b), and the two weight tables differ, so an entry that
 * aliased the two weight offsets or dropped a row stride would produce a
 * visibly wrong result rather than a coincidentally matching one. */
static int test_matmul_f16_pair(void) {
    enum { IN_DIM = 37, OUT_DIM = 5, N_TOK = 3 };

    uint16_t weights[2 * OUT_DIM * IN_DIM];
    uint16_t *w0 = weights;
    uint16_t *w1 = weights + (size_t)OUT_DIM * IN_DIM;
    float     x[N_TOK * IN_DIM];
    float     want0[N_TOK * OUT_DIM];
    float     want1[N_TOK * OUT_DIM];
    float     got0[N_TOK * OUT_DIM];
    float     got1[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            float v0 = (float)(((o + 1) * (k + 3)) % 13) - 6.0f;
            float v1 = (float)(((o + 2) * (k + 5)) % 11) - 5.0f;
            w0[o * IN_DIM + k] = test_float_to_half(v0);
            w1[o * IN_DIM + k] = test_float_to_half(v1);
        }
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_f16(want0, x, w0, IN_DIM, OUT_DIM, N_TOK);
    oracle_matmul_f16(want1, x, w1, IN_DIM, OUT_DIM, N_TOK);

    const uint64_t weight_a_offset = 0;
    const uint64_t weight_b_offset = (uint64_t)OUT_DIM * IN_DIM * sizeof(uint16_t);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(got0));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(got1));
    CHECK(tx != NULL && ta != NULL && tb != NULL,
          "matmul_f16_pair: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_f16_pair: write x");

    CHECK(ds4_gpu_matmul_f16_pair_tensor(ta, tb, weights, sizeof(weights),
                                         weight_a_offset, weight_b_offset,
                                         IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_f16_pair: call");
    CHECK(ds4_gpu_tensor_read(ta, 0, got0, sizeof(got0)) != 0,
          "matmul_f16_pair: read out_a");
    CHECK(ds4_gpu_tensor_read(tb, 0, got1, sizeof(got1)) != 0,
          "matmul_f16_pair: read out_b");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got0[i], want0[i], 1e-2, "matmul_f16_pair: out_a mismatch");
        CHECK_CLOSE(got1[i], want1[i], 1e-2, "matmul_f16_pair: out_b mismatch");
    }

    /* A weight offset past the end of the model buffer must be rejected,
     * matching the weight_bytes > model_size - weight_offset check ported
     * from rocm/ds4_rocm_matmul.cuh:873-931. */
    CHECK(ds4_gpu_matmul_f16_pair_tensor(ta, tb, weights, sizeof(weights),
                                         sizeof(weights), weight_b_offset,
                                         IN_DIM, OUT_DIM, tx, N_TOK) == 0,
          "matmul_f16_pair: out-of-range weight_a offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    fprintf(stderr, "  test_matmul_f16_pair OK\n");
    return 0;
}

/* This backend has no fused Metal-style decode path, so this entry's only
 * defined behaviour here is "decline": rocm/ds4_rocm_matmul.cuh:933-965 is
 * `(void)` on all 15 parameters then `return 0;`, a real and complete
 * implementation of the tri-state contract's 0 branch (ds4_gpu.h,
 * ds4_gpu_matmul_f16_pair_compressor_store_tensor), not a placeholder for
 * an unimplemented stub.  The correct, complete test is: returns exactly
 * 0, and touches none of its four output/state tensors. */
static int test_matmul_f16_pair_compressor_store(void) {
    enum { N = 16 };
    unsigned char sentinel[N];
    unsigned char check[N];
    float         x[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    for (int i = 0; i < N; i++) sentinel[i] = (unsigned char)(0xA5 ^ i);

    ds4_gpu_tensor *out_kv      = ds4_gpu_tensor_alloc(N);
    ds4_gpu_tensor *out_score   = ds4_gpu_tensor_alloc(N);
    ds4_gpu_tensor *state_kv    = ds4_gpu_tensor_alloc(N);
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(N);
    ds4_gpu_tensor *tx          = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(out_kv && out_score && state_kv && state_score && tx,
          "matmul_f16_pair_compressor_store: allocation failed");

    CHECK(ds4_gpu_tensor_write(out_kv, 0, sentinel, N) != 0 &&
          ds4_gpu_tensor_write(out_score, 0, sentinel, N) != 0 &&
          ds4_gpu_tensor_write(state_kv, 0, sentinel, N) != 0 &&
          ds4_gpu_tensor_write(state_score, 0, sentinel, N) != 0 &&
          ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_f16_pair_compressor_store: write sentinels");

    unsigned char weight_kv[8] = {0};
    int ret = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            out_kv, out_score, state_kv, state_score, weight_kv,
            sizeof(weight_kv), 0, 0, 0, 1u, 8u, 16u, tx, 4u, 0u);
    CHECK(ret == 0,
          "matmul_f16_pair_compressor_store: must decline (return 0)");

    CHECK(ds4_gpu_tensor_read(out_kv, 0, check, N) != 0 &&
          memcmp(sentinel, check, N) == 0,
          "matmul_f16_pair_compressor_store: out_kv must be untouched");
    CHECK(ds4_gpu_tensor_read(out_score, 0, check, N) != 0 &&
          memcmp(sentinel, check, N) == 0,
          "matmul_f16_pair_compressor_store: out_score must be untouched");
    CHECK(ds4_gpu_tensor_read(state_kv, 0, check, N) != 0 &&
          memcmp(sentinel, check, N) == 0,
          "matmul_f16_pair_compressor_store: state_kv must be untouched");
    CHECK(ds4_gpu_tensor_read(state_score, 0, check, N) != 0 &&
          memcmp(sentinel, check, N) == 0,
          "matmul_f16_pair_compressor_store: state_score must be untouched");

    ds4_gpu_tensor_free(out_kv);
    ds4_gpu_tensor_free(out_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_matmul_f16_pair_compressor_store OK\n");
    return 0;
}

/* Q8_0 row layout, cited to rocm/ds4_rocm_common.cuh:19-63: blocks of 32
 * values, 34 bytes per block, a little-endian F16 scale in bytes 0-1
 * followed by 32 signed int8 values.  Encodes one out_dim row so the
 * per-column dequantised value is scale(k/32) * (float)q8[k], matching
 * sycl_q8_0_dequant in sycl/ds4_sycl_common.hpp.  The int8 payload carries
 * an (o, k) product interaction term (not a plain a*i + b), so a kernel
 * that mixed up rows or dropped the block stride would produce a visibly
 * wrong result rather than a coincidentally matching one. */
static void test_encode_q8_0_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t braw = test_float_to_half(0.05f * (float)(o + blk + 1u));
        bp[0] = (unsigned char)(braw & 0xFFu);
        bp[1] = (unsigned char)((braw >> 8) & 0xFFu);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const int qv = (k < in_dim)
                    ? (int)(((o + 1u) * (k + 3u)) % 13u) - 6
                    : 0;
            bp[2 + idx] = (unsigned char)(signed char)qv;
        }
    }
}

/* Oracle: out[t][o] = sum_k x[t][k] * scale(k/32) * (float)q8[o][k], cited
 * to rocm/ds4_rocm_matmul.cuh:306-525 (matmul_q8_0_f32_warp8_kernel and
 * matmul_q8_0_f32_batch_warp8_kernel, which agree on this formula and
 * differ only in tiling). */
static void oracle_matmul_q8_0(float *out, const float *x,
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

/* in_dim deliberately not a multiple of 32 so the last Q8_0 block is only
 * partially used, exercising the remainder.  n_tok > 1 exercises the
 * per-token row stride over x. */
static int test_matmul_q8_0(void) {
    enum { IN_DIM = 37, OUT_DIM = 5, N_TOK = 3 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights[OUT_DIM * 68]; /* row_bytes <= 2*34 = 68 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_q8_0: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0: write x");

    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes, 0,
                                     IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q8_0: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q8_0: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_q8_0: out mismatch");
    }

    /* Validation ported from rocm/ds4_rocm_matmul.cuh:312-323. */
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes, 0,
                                     0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q8_0: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes, 0,
                                     IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_q8_0: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes, 0,
                                     IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_q8_0: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes, 0,
                                     (uint64_t)UINT32_MAX + 1u, OUT_DIM,
                                     tx, N_TOK) == 0,
          "matmul_q8_0: oversized in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, weights, weight_bytes,
                                     weight_bytes, IN_DIM, OUT_DIM, tx,
                                     N_TOK) == 0,
          "matmul_q8_0: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q8_0 OK\n");
    return 0;
}

/* Oracle for the k-slice family: as oracle_matmul_q8_0 above, but each
 * input row is already restricted to [in_start, in_start+in_count) and
 * the dequant column addressed into the FULL row is in_start+k, not k --
 * the weight table is staged whole (full_row_bytes), only the columns
 * touched are restricted. Matches ds4_gpu_matmul_q8_0_kslice_rows_tensor's
 * own contract exactly (ds4_cuda.cu:14859-14884). */
static void oracle_matmul_q8_0_kslice(float *out, const float *x_slice,
                                      const unsigned char *w,
                                      uint32_t out_dim, uint32_t n_tok,
                                      uint64_t full_row_bytes,
                                      uint32_t in_start, uint32_t in_count) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const unsigned char *row = w + (size_t)o * full_row_bytes;
            double sum = 0.0;
            for (uint32_t k = 0; k < in_count; k++) {
                const uint32_t col = in_start + k;
                const uint32_t blk = col / 32u;
                const uint32_t idx = col % 32u;
                const unsigned char *bp = row + (size_t)blk * 34u;
                const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
                const float scale = oracle_half_to_float(raw);
                const signed char qv = (signed char)bp[2 + idx];
                sum += (double)x_slice[t * in_count + k] * (double)scale * (double)qv;
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* Tensor parallelism: ds4_gpu_matmul_q8_0_kslice_rows_tensor,
 * the split-matmul core. FULL_IN_DIM is not a multiple of 64 (96 is, so
 * split at a non-half boundary too: HALF_A=32, HALF_B=64, both multiples
 * of 32 as the contract requires, summing to 96) to exercise an uneven
 * TP split, not just the symmetric case a real 2-way TP world would
 * mostly use. */
static int test_matmul_q8_0_kslice_rows(void) {
    enum { FULL_IN_DIM = 96, OUT_DIM = 5, N_TOK = 3, HALF_A = 32, HALF_B = 64 };
    const uint64_t full_blocks = (FULL_IN_DIM + 31u) / 32u;
    const uint64_t full_row_bytes = full_blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)OUT_DIM * full_row_bytes;

    unsigned char weights[OUT_DIM * 102]; /* full_row_bytes == 3*34 = 102 here */
    float x_full[N_TOK * FULL_IN_DIM];
    float want_full[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * full_row_bytes, FULL_IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < FULL_IN_DIM; k++) {
            x_full[t * FULL_IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want_full, x_full, weights, FULL_IN_DIM, OUT_DIM, N_TOK,
                       full_row_bytes);

    /* Each rank's input row holds only its own contiguous slice, per this
     * entry's own contract ("each input row contains only the owned
     * contiguous K slice"), not the full FULL_IN_DIM-wide row. */
    float xa[N_TOK * HALF_A], xb[N_TOK * HALF_B];
    for (uint32_t t = 0; t < N_TOK; t++) {
        memcpy(xa + t * HALF_A, x_full + t * FULL_IN_DIM, sizeof(float) * HALF_A);
        memcpy(xb + t * HALF_B, x_full + t * FULL_IN_DIM + HALF_A, sizeof(float) * HALF_B);
    }
    float want_a[N_TOK * OUT_DIM], want_b[N_TOK * OUT_DIM];
    oracle_matmul_q8_0_kslice(want_a, xa, weights, OUT_DIM, N_TOK, full_row_bytes, 0, HALF_A);
    oracle_matmul_q8_0_kslice(want_b, xb, weights, OUT_DIM, N_TOK, full_row_bytes, HALF_A, HALF_B);

    ds4_gpu_tensor *txa = ds4_gpu_tensor_alloc(sizeof(xa));
    ds4_gpu_tensor *txb = ds4_gpu_tensor_alloc(sizeof(xb));
    ds4_gpu_tensor *touta = ds4_gpu_tensor_alloc(sizeof(want_a));
    ds4_gpu_tensor *toutb = ds4_gpu_tensor_alloc(sizeof(want_b));
    CHECK(txa && txb && touta && toutb, "matmul_q8_0_kslice_rows: alloc failed");
    CHECK(ds4_gpu_tensor_write(txa, 0, xa, sizeof(xa)) != 0,
          "matmul_q8_0_kslice_rows: write xa");
    CHECK(ds4_gpu_tensor_write(txb, 0, xb, sizeof(xb)) != 0,
          "matmul_q8_0_kslice_rows: write xb");

    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, 0, HALF_A,
                                                 txa, N_TOK) != 0,
          "matmul_q8_0_kslice_rows: rank-a call");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(toutb, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, HALF_A, HALF_B,
                                                 txb, N_TOK) != 0,
          "matmul_q8_0_kslice_rows: rank-b call");

    float got_a[N_TOK * OUT_DIM], got_b[N_TOK * OUT_DIM];
    CHECK(ds4_gpu_tensor_read(touta, 0, got_a, sizeof(got_a)) != 0,
          "matmul_q8_0_kslice_rows: read rank-a");
    CHECK(ds4_gpu_tensor_read(toutb, 0, got_b, sizeof(got_b)) != 0,
          "matmul_q8_0_kslice_rows: read rank-b");
    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got_a[i], want_a[i], 1e-2, "matmul_q8_0_kslice_rows: rank-a mismatch");
        CHECK_CLOSE(got_b[i], want_b[i], 1e-2, "matmul_q8_0_kslice_rows: rank-b mismatch");
    }

    /* The TP invariant this whole family exists for: the two ranks'
     * partial results, summed, reconstruct the SAME full-width projection
     * a single non-split matmul over the whole row would produce. This is
     * the property ds4_gpu_add_xdev_tensor / ds4_gpu_hc_expand_add_tensor
     * rely on downstream; if this does not hold, no amount of correct
     * cross-rank plumbing on top of it would produce a correct answer. */
    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got_a[i] + got_b[i], want_full[i], 1e-2,
                    "matmul_q8_0_kslice_rows: split halves do not sum to the "
                    "unsplit projection");
    }

    /* Validation ported from ds4_cuda.cu:14870-14884. */
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 0, OUT_DIM, 0, HALF_A, txa, N_TOK) == 0,
          "matmul_q8_0_kslice_rows: zero full_in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, 0, 0, HALF_A, txa, N_TOK) == 0,
          "matmul_q8_0_kslice_rows: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, 0, HALF_A, txa, 0) == 0,
          "matmul_q8_0_kslice_rows: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, 1, HALF_A, txa, N_TOK) == 0,
          "matmul_q8_0_kslice_rows: unaligned in_start must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, 0, HALF_A + 1u, txa,
                                                 N_TOK) == 0,
          "matmul_q8_0_kslice_rows: unaligned in_count must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, FULL_IN_DIM, 32u, txa,
                                                 N_TOK) == 0,
          "matmul_q8_0_kslice_rows: in_start beyond full_in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, HALF_A,
                                                 FULL_IN_DIM, txa, N_TOK) == 0,
          "matmul_q8_0_kslice_rows: in_count exceeding remaining range must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(touta, weights, weight_bytes,
                                                 weight_bytes, FULL_IN_DIM, OUT_DIM, 0,
                                                 HALF_A, txa, N_TOK) == 0,
          "matmul_q8_0_kslice_rows: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(txa);
    ds4_gpu_tensor_free(txb);
    ds4_gpu_tensor_free(touta);
    ds4_gpu_tensor_free(toutb);
    fprintf(stderr, "  test_matmul_q8_0_kslice_rows OK\n");
    return 0;
}

/* Tensor parallelism: ds4_gpu_matmul_q8_0_kslice_tensor, the
 * single-token wrapper. Differential against
 * ds4_gpu_matmul_q8_0_kslice_rows_tensor: a call through the x_elem_off/
 * n_tok=1 wrapper must produce exactly the same result as calling the
 * rows entry directly on an equivalent pre-sliced view, since
 * ds4_cuda.cu:30649-30671 implements the wrapper as exactly that
 * delegation and this port matches it line-for-line. Note: this entry has ZERO call sites anywhere in ds4.c, so
 * this test is the only thing that will ever exercise it. */
static int test_matmul_q8_0_kslice(void) {
    enum { FULL_IN_DIM = 96, OUT_DIM = 5, IN_START = 32, IN_COUNT = 64, X_ELEM_OFF = 7 };
    const uint64_t full_blocks = (FULL_IN_DIM + 31u) / 32u;
    const uint64_t full_row_bytes = full_blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)OUT_DIM * full_row_bytes;

    unsigned char weights[OUT_DIM * 102];
    /* x is wider than X_ELEM_OFF + IN_COUNT so the slice genuinely starts
     * mid-buffer, exercising x_elem_off as a real offset, not a no-op. */
    float x[X_ELEM_OFF + IN_COUNT + 5];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * full_row_bytes, FULL_IN_DIM, o + 3u);
    }
    for (uint32_t k = 0; k < X_ELEM_OFF + IN_COUNT + 5u; k++) {
        x[k] = (float)((k * 5u) % 7u) - 3.0f;
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(float) * OUT_DIM);
    ds4_gpu_tensor *tslice = ds4_gpu_tensor_alloc(sizeof(float) * IN_COUNT);
    ds4_gpu_tensor *tout_ref = ds4_gpu_tensor_alloc(sizeof(float) * OUT_DIM);
    CHECK(tx && tout && tslice && tout_ref, "matmul_q8_0_kslice: alloc failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "matmul_q8_0_kslice: write x");
    CHECK(ds4_gpu_tensor_write(tslice, 0, x + X_ELEM_OFF, sizeof(float) * IN_COUNT) != 0,
          "matmul_q8_0_kslice: write slice");

    CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(tout, weights, weight_bytes, 0, FULL_IN_DIM,
                                            IN_START, IN_COUNT, OUT_DIM, tx,
                                            X_ELEM_OFF) != 0,
          "matmul_q8_0_kslice: call");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(tout_ref, weights, weight_bytes, 0,
                                                 FULL_IN_DIM, OUT_DIM, IN_START, IN_COUNT,
                                                 tslice, 1u) != 0,
          "matmul_q8_0_kslice: reference rows call");

    float got[OUT_DIM], want[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "matmul_q8_0_kslice: read");
    CHECK(ds4_gpu_tensor_read(tout_ref, 0, want, sizeof(want)) != 0,
          "matmul_q8_0_kslice: read reference");
    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-6,
                    "matmul_q8_0_kslice: wrapper diverges from rows delegation");
    }

    /* Validation ported from ds4_cuda.cu:30660-30663. */
    CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(tout, weights, weight_bytes, 0, FULL_IN_DIM,
                                            IN_START, IN_COUNT, OUT_DIM, tx,
                                            (uint64_t)(X_ELEM_OFF + IN_COUNT + 5u) + 1u) == 0,
          "matmul_q8_0_kslice: x_elem_off beyond x must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(tout, weights, weight_bytes, 0, FULL_IN_DIM,
                                            IN_START, (uint64_t)(IN_COUNT + 5u) * 100u,
                                            OUT_DIM, tx, X_ELEM_OFF) == 0,
          "matmul_q8_0_kslice: k_cnt exceeding remaining x must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(NULL, weights, weight_bytes, 0, FULL_IN_DIM,
                                            IN_START, IN_COUNT, OUT_DIM, NULL,
                                            X_ELEM_OFF) == 0,
          "matmul_q8_0_kslice: null x must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tslice);
    ds4_gpu_tensor_free(tout_ref);
    fprintf(stderr, "  test_matmul_q8_0_kslice OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_decode_mpp_tensor is, per ROCm
 * (rocm/ds4_rocm_matmul.cuh:531-549), the exact same computation as
 * ds4_gpu_matmul_q8_0_tensor; it exists as a separate ABI entry only
 * because the graph calls it under a different name for the decode path.
 * Reuses test_encode_q8_0_row/oracle_matmul_q8_0 above, with dimensions
 * distinct from test_matmul_q8_0 and from
 * test_matmul_q8_0_decode_mpp_model_view below so that a copy-paste error
 * wiring this entry to the wrong implementation would be caught. */
static int test_matmul_q8_0_decode_mpp(void) {
    enum { IN_DIM = 41, OUT_DIM = 6, N_TOK = 2 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights[OUT_DIM * 68]; /* row_bytes <= 2*34 = 68 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_q8_0_decode_mpp: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0_decode_mpp: write x");

    CHECK(ds4_gpu_matmul_q8_0_decode_mpp_tensor(tout, weights, weight_bytes, 0,
                                                IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q8_0_decode_mpp: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q8_0_decode_mpp: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_q8_0_decode_mpp: out mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q8_0_decode_mpp OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor is, per ROCm
 * (rocm/ds4_rocm_matmul.cuh:551-569), the exact same computation as
 * ds4_gpu_matmul_q8_0_tensor and ds4_gpu_matmul_q8_0_decode_mpp_tensor
 * above (the three differ only in a diagnostic label string in ROCm, not
 * in behaviour). Reuses test_encode_q8_0_row/oracle_matmul_q8_0 above,
 * with dimensions distinct from both of those so that a copy-paste error
 * wiring this entry to the wrong implementation would be caught. */
static int test_matmul_q8_0_decode_mpp_model_view(void) {
    enum { IN_DIM = 29, OUT_DIM = 4, N_TOK = 4 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights[OUT_DIM * 34]; /* row_bytes <= 1*34 = 34 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL,
          "matmul_q8_0_decode_mpp_model_view: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0_decode_mpp_model_view: write x");

    CHECK(ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
                  tout, weights, weight_bytes, 0, IN_DIM, OUT_DIM, tx,
                  N_TOK) != 0,
          "matmul_q8_0_decode_mpp_model_view: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q8_0_decode_mpp_model_view: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2,
                    "matmul_q8_0_decode_mpp_model_view: out mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q8_0_decode_mpp_model_view OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_rows_scalar_tensor has no ROCm reference behaviour to
 * exercise: ROCm's own entry (rocm/ds4_rocm_matmul.cuh:571-583) casts every
 * parameter to void and unconditionally returns 0, dead code uncalled on
 * any reachable non-Apple path.  This backend implements it as a real Q8_0
 * dense matmul identical in contract to ds4_gpu_matmul_q8_0_tensor, so a
 * light test suffices: one case confirming this entry point independently
 * wires through to a correct result (catching a copy-paste mistake), with
 * dimensions distinct from every other test in this file, plus the
 * standard validation-failure checks. */
static int test_matmul_q8_0_rows_scalar(void) {
    enum { IN_DIM = 23, OUT_DIM = 9, N_TOK = 2 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights[OUT_DIM * 34]; /* row_bytes <= 1*34 = 34 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL,
          "matmul_q8_0_rows_scalar: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0_rows_scalar: write x");

    CHECK(ds4_gpu_matmul_q8_0_rows_scalar_tensor(tout, weights, weight_bytes,
                                                 0, IN_DIM, OUT_DIM, tx,
                                                 N_TOK) != 0,
          "matmul_q8_0_rows_scalar: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q8_0_rows_scalar: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_q8_0_rows_scalar: out mismatch");
    }

    CHECK(ds4_gpu_matmul_q8_0_rows_scalar_tensor(tout, weights, weight_bytes,
                                                 0, 0, OUT_DIM, tx,
                                                 N_TOK) == 0,
          "matmul_q8_0_rows_scalar: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_rows_scalar_tensor(tout, weights, weight_bytes,
                                                 0, IN_DIM, OUT_DIM, tx,
                                                 0) == 0,
          "matmul_q8_0_rows_scalar: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_rows_scalar_tensor(tout, weights, weight_bytes,
                                                 weight_bytes, IN_DIM,
                                                 OUT_DIM, tx, N_TOK) == 0,
          "matmul_q8_0_rows_scalar: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q8_0_rows_scalar OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_pair_tensor computes two independent Q8_0
 * projections of the same activations in one call, cited to
 * rocm/ds4_rocm_matmul.cuh:585-688.  Two distinct weight buffers and
 * distinct out0_dim/out1_dim (so the two outputs cannot coincidentally
 * match each other), n_tok > 1, and dimensions distinct from every other
 * test in this file.  Reuses test_encode_q8_0_row/oracle_matmul_q8_0
 * applied independently to each weight table/output pair. */
static int test_matmul_q8_0_pair(void) {
    enum { IN_DIM = 17, OUT0_DIM = 6, OUT1_DIM = 11, N_TOK = 3 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights0[OUT0_DIM * 34]; /* row_bytes <= 1*34 = 34 here */
    unsigned char weights1[OUT1_DIM * 34];
    float          x[N_TOK * IN_DIM];
    float          want0[N_TOK * OUT0_DIM];
    float          want1[N_TOK * OUT1_DIM];
    float          got0[N_TOK * OUT0_DIM];
    float          got1[N_TOK * OUT1_DIM];

    for (uint32_t o = 0; o < OUT0_DIM; o++) {
        test_encode_q8_0_row(weights0 + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t o = 0; o < OUT1_DIM; o++) {
        test_encode_q8_0_row(weights1 + (size_t)o * row_bytes, IN_DIM,
                             o + 100u);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want0, x, weights0, IN_DIM, OUT0_DIM, N_TOK, row_bytes);
    oracle_matmul_q8_0(want1, x, weights1, IN_DIM, OUT1_DIM, N_TOK, row_bytes);

    ds4_gpu_tensor *tx   = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout0 = ds4_gpu_tensor_alloc(sizeof(got0));
    ds4_gpu_tensor *tout1 = ds4_gpu_tensor_alloc(sizeof(got1));
    CHECK(tx != NULL && tout0 != NULL && tout1 != NULL,
          "matmul_q8_0_pair: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0_pair: write x");

    /* weights0 and weights1 are independent allocations in this test, so
     * the pair call is exercised against a single combined model buffer
     * that concatenates both weight tables, matching how the real graph
     * lays out one mmapped model file. */
    unsigned char combined[sizeof(weights0) + sizeof(weights1)];
    memcpy(combined, weights0, sizeof(weights0));
    memcpy(combined + sizeof(weights0), weights1, sizeof(weights1));
    const uint64_t weight0_offset = 0;
    const uint64_t weight1_offset = sizeof(weights0);
    const uint64_t model_size = sizeof(combined);

    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, weight1_offset,
                                          IN_DIM, OUT0_DIM, OUT1_DIM, tx,
                                          N_TOK) != 0,
          "matmul_q8_0_pair: call");
    CHECK(ds4_gpu_tensor_read(tout0, 0, got0, sizeof(got0)) != 0,
          "matmul_q8_0_pair: read out0");
    CHECK(ds4_gpu_tensor_read(tout1, 0, got1, sizeof(got1)) != 0,
          "matmul_q8_0_pair: read out1");

    for (int i = 0; i < N_TOK * OUT0_DIM; i++) {
        CHECK_CLOSE(got0[i], want0[i], 1e-2, "matmul_q8_0_pair: out0 mismatch");
    }
    for (int i = 0; i < N_TOK * OUT1_DIM; i++) {
        CHECK_CLOSE(got1[i], want1[i], 1e-2, "matmul_q8_0_pair: out1 mismatch");
    }

    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, weight1_offset, 0,
                                          OUT0_DIM, OUT1_DIM, tx,
                                          N_TOK) == 0,
          "matmul_q8_0_pair: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, weight1_offset,
                                          IN_DIM, 0, OUT1_DIM, tx,
                                          N_TOK) == 0,
          "matmul_q8_0_pair: zero out0_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, weight1_offset,
                                          IN_DIM, OUT0_DIM, 0, tx,
                                          N_TOK) == 0,
          "matmul_q8_0_pair: zero out1_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, weight1_offset,
                                          IN_DIM, OUT0_DIM, OUT1_DIM, tx,
                                          0) == 0,
          "matmul_q8_0_pair: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          model_size, weight1_offset, IN_DIM,
                                          OUT0_DIM, OUT1_DIM, tx,
                                          N_TOK) == 0,
          "matmul_q8_0_pair: out-of-range weight0 offset must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tout0, tout1, combined, model_size,
                                          weight0_offset, model_size, IN_DIM,
                                          OUT0_DIM, OUT1_DIM, tx,
                                          N_TOK) == 0,
          "matmul_q8_0_pair: out-of-range weight1 offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout0);
    ds4_gpu_tensor_free(tout1);
    fprintf(stderr, "  test_matmul_q8_0_pair OK\n");
    return 0;
}

/* in_dim deliberately not a multiple of 8/16/32/64, n_tok > 1: same
 * discriminating shape as test_matmul_f16_pair above, applied to the
 * plain (non-cuBLAS) F32 dense matmul, cited to
 * rocm/ds4_rocm_matmul.cuh:967-991.  This backend has no cuBLAS/oneMKL
 * batched-GEMM fast path wired for this entry:
 * ROCm's non-BLAS kernel fallback is genuine and gated the same way the
 * Q8_0 GEMM fast path is, so a hand-written kernel is the correct port
 * here, not a GEMM), so there is exactly one code path to validate. */
static int test_matmul_f32(void) {
    enum { IN_DIM = 37, OUT_DIM = 5, N_TOK = 3 };

    float weights[OUT_DIM * IN_DIM];
    float x[N_TOK * IN_DIM];
    float want[N_TOK * OUT_DIM];
    float got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            weights[o * IN_DIM + k] = (float)(((o + 1) * (k + 3)) % 13) - 6.0f;
        }
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_f32(want, x, weights, IN_DIM, OUT_DIM, N_TOK);

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_f32: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "matmul_f32: write x");

    CHECK(ds4_gpu_matmul_f32_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_f32: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_f32: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-3, "matmul_f32: out mismatch");
    }

    /* Validation ported from rocm/ds4_rocm_matmul.cuh:967-971. */
    CHECK(ds4_gpu_matmul_f32_tensor(tout, weights, sizeof(weights), 0,
                                    0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_f32: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_f32_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_f32: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_f32_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_f32: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_f32_tensor(tout, weights, sizeof(weights),
                                    sizeof(weights), IN_DIM, OUT_DIM, tx,
                                    N_TOK) == 0,
          "matmul_f32: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_f32 OK\n");
    return 0;
}

/* Same discriminating shape as test_matmul_f16_pair, applied to the single
 * (non-paired) F16 dense matmul entry, cited to
 * rocm/ds4_rocm_matmul.cuh:804-814/873-931.  This backend has no
 * cuBLAS/oneMKL batched-GEMM fast path wired for this entry either (same
 * reasoning as test_matmul_f32 above), reusing the general kernel already
 * built for ds4_gpu_matmul_f16_pair_tensor
 * (sycl_matmul_f16_launch, sycl/ds4_sycl_matmul.hpp). */
static int test_matmul_f16(void) {
    enum { IN_DIM = 37, OUT_DIM = 5, N_TOK = 3 };

    uint16_t weights[OUT_DIM * IN_DIM];
    float    x[N_TOK * IN_DIM];
    float    want[N_TOK * OUT_DIM];
    float    got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            const float v = (float)(((o + 1) * (k + 3)) % 13) - 6.0f;
            weights[o * IN_DIM + k] = test_float_to_half(v);
        }
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_f16(want, x, weights, IN_DIM, OUT_DIM, N_TOK);

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_f16: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "matmul_f16: write x");

    CHECK(ds4_gpu_matmul_f16_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_f16: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_f16: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_f16: out mismatch");
    }

    /* Validation ported from rocm/ds4_rocm_matmul.cuh:804-808. */
    CHECK(ds4_gpu_matmul_f16_tensor(tout, weights, sizeof(weights), 0,
                                    0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_f16: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_f16_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_f16: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_f16_tensor(tout, weights, sizeof(weights), 0,
                                    IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_f16: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_f16_tensor(tout, weights, sizeof(weights),
                                    sizeof(weights), IN_DIM, OUT_DIM, tx,
                                    N_TOK) == 0,
          "matmul_f16: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_f16 OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_f16_out_tensor, rocm/ds4_rocm_matmul.cuh:216-291
 * (cuda_matmul_q8_0_tensor_f16_gemm_out_half): the one matmul entry this
 * plan found with NO non-GEMM fallback in ROCm at all (unconditional
 * `if (!g_cublas_ready ...) return 0;`), so oneMKL is genuinely load-
 * bearing here, not merely a performance choice the way it is for
 * ds4_gpu_matmul_f32_tensor/ds4_gpu_matmul_f16_tensor above.  Reuses
 * test_encode_q8_0_row/oracle_matmul_q8_0 (this file) for the weight side.
 *
 * The output tensor is genuinely F16 (the GEMM's own C operand, written by
 * oneMKL/cuBLAS's internal cast, not by this backend's own bit-exact F16
 * encoder), so the tolerance here is looser than the F32-output matmul
 * tests above: both the Q8_0-dequantised weight and the F32 activation are
 * themselves rounded to half before the multiply-accumulate, and the GPU
 * accumulates in a different order than this double-precision oracle. 1e-2
 * relative was measured to hold comfortably (observed deltas were an order
 * of magnitude tighter across this test's whole output), while still being
 * loose enough to absorb the half-precision rounding this path genuinely
 * introduces. */
static int test_matmul_q8_0_f16_out(void) {
    enum { IN_DIM = 37, OUT_DIM = 5, N_TOK = 3 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;

    unsigned char weights[OUT_DIM * 68]; /* row_bytes <= 2*34 = 68 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    uint16_t       got_h[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q8_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx   = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *touth = ds4_gpu_tensor_alloc(sizeof(got_h));
    CHECK(tx != NULL && touth != NULL, "matmul_q8_0_f16_out: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q8_0_f16_out: write x");

    CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(touth, weights, weight_bytes, 0,
                                             IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q8_0_f16_out: call");
    CHECK(ds4_gpu_tensor_read(touth, 0, got_h, sizeof(got_h)) != 0,
          "matmul_q8_0_f16_out: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        const float got = oracle_half_to_float(got_h[i]);
        const double tol = 1e-2 * (fabs((double)want[i]) > 1.0 ? fabs((double)want[i]) : 1.0);
        CHECK_CLOSE(got, want[i], tol, "matmul_q8_0_f16_out: out mismatch");
    }

    /* Validation ported from rocm/ds4_rocm_matmul.cuh:250-261. */
    CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(touth, weights, weight_bytes, 0,
                                             0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q8_0_f16_out: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(touth, weights, weight_bytes, 0,
                                             IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_q8_0_f16_out: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(touth, weights, weight_bytes, 0,
                                             IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_q8_0_f16_out: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(touth, weights, weight_bytes,
                                             weight_bytes, IN_DIM, OUT_DIM,
                                             tx, N_TOK) == 0,
          "matmul_q8_0_f16_out: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(touth);
    fprintf(stderr, "  test_matmul_q8_0_f16_out OK\n");
    return 0;
}

/* ---- ds4_gpu_matmul_quant_tensor / _kslice_tensor: Q4_K and Q4_0 --------
 *
 * ds4_cuda.cu's own ds4_gpu_matmul_quant_tensor dispatches only Q8_0 and
 * F16; ds4_metal.m genuinely supports Q4_K/Q4_0 here but through bespoke
 * Metal compute shaders with no scalar source to transcribe. So these
 * oracles are built directly from the standard block formats: Q4_K's from
 * ds4.c's own q4_k_get_scale_min (ds4.c:3524-3532) and
 * ds4_vec_dot_q4_K_f32 (ds4.c:3709-3736), the same formula the routed-MoE's
 * device dot product (sycl_dev_dot_q4_k_q8_k_block, ds4_sycl_moe.hpp) already
 * uses per sub-block; Q4_0's from the standard GGUF layout
 * (gguf_types[2] in ds4.c: block_elems=32, block_bytes=18), which nothing
 * in this codebase decoded before now. */

/* Inverse of q4_k_get_scale_min (ds4.c:3524-3532): packs 8 (scale, min)
 * pairs, each a 6-bit code (0-63), into the 12-byte array a real Q4_K block
 * carries. Derived by solving q4_k_get_scale_min's two branches for their
 * inputs; cross-checked against that decode by test_q4k_pack_roundtrip
 * below before being trusted for anything else, mirroring the precedent in
 * tests/test_sycl_moe.c's test_q4k_scale_pack_roundtrip. */
static void test_q4k_pack_scales(uint8_t q[12], const uint8_t sc[8], const uint8_t m[8]) {
    for (int i = 0; i < 4; i++) {
        q[i]     = (uint8_t)((sc[i] & 0x3Fu) | ((uint32_t)(sc[i + 4] >> 4) << 6));
        q[i + 4] = (uint8_t)((m[i]  & 0x3Fu) | ((uint32_t)(m[i + 4]  >> 4) << 6));
        q[i + 8] = (uint8_t)((sc[i + 4] & 0x0Fu) | ((uint32_t)(m[i + 4] & 0x0Fu) << 4));
    }
}

static void test_q4k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m  = (uint8_t)((q[j + 4] >> 4)  | ((q[j] >> 6) << 4));
    }
}

static int test_q4k_pack_roundtrip(void) {
    for (int seed = 0; seed < 5; seed++) {
        uint8_t sc[8], m[8], q[12];
        for (int j = 0; j < 8; j++) {
            sc[j] = (uint8_t)((seed * 17 + j * 23) % 64);
            m[j]  = (uint8_t)((seed * 29 + j * 11) % 64);
        }
        test_q4k_pack_scales(q, sc, m);
        for (int j = 0; j < 8; j++) {
            uint8_t got_sc, got_m;
            test_q4k_get_scale_min(j, q, &got_sc, &got_m);
            CHECK(got_sc == sc[j], "q4k_pack_roundtrip: sc mismatch");
            CHECK(got_m == m[j], "q4k_pack_roundtrip: m mismatch");
        }
    }
    fprintf(stderr, "  test_q4k_pack_roundtrip OK\n");
    return 0;
}

/* Packs one 144-byte Q4_K block from an explicit (d, dmin, sc[8], m[8],
 * nib[256]) description: two sub-blocks share each 32-byte qs region, low
 * nibble for the even sub-block, high nibble for the odd one, matching
 * sycl_q4_k_dequant's byte_off = (j>>1)*32 / shift = (j&1)?4:0 (and plan
 * 9's identical moe.cuh-derived math). */
static void test_q4k_pack_block(unsigned char out[144], float d, float dmin,
                                const uint8_t sc[8], const uint8_t m[8],
                                const uint8_t nib[256]) {
    const uint16_t draw = test_float_to_half(d);
    const uint16_t dminraw = test_float_to_half(dmin);
    out[0] = (unsigned char)(draw & 0xFFu);
    out[1] = (unsigned char)((draw >> 8) & 0xFFu);
    out[2] = (unsigned char)(dminraw & 0xFFu);
    out[3] = (unsigned char)((dminraw >> 8) & 0xFFu);
    test_q4k_pack_scales(out + 4, sc, m);
    unsigned char *qs = out + 16;
    memset(qs, 0, 128);
    for (int j = 0; j < 8; j++) {
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) ? 4 : 0;
        for (int k = 0; k < 32; k++) {
            const uint8_t v = nib[j * 32 + k] & 0x0Fu;
            qs[byte_off + k] = (unsigned char)(qs[byte_off + k] | (v << shift));
        }
    }
}

/* Deterministic per-(row, block) Q4_K weight-row generator with an
 * index-interaction nibble term (spec 6f/6i: affine test data hides
 * scale-only bugs and makes adjacent blocks indistinguishable). Columns at
 * or past in_dim within the last, possibly ragged, superblock are zeroed,
 * matching test_encode_q8_0_row's own padding treatment. */
static void test_encode_q4_k_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 255u) / 256u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        uint8_t sc[8], m[8], nib[256];
        for (uint32_t j = 0; j < 8u; j++) {
            sc[j] = (uint8_t)(1u + ((o + j * 3u + blk * 11u) % 60u));
            m[j]  = (uint8_t)(1u + ((o * 2u + j * 5u + blk * 7u) % 60u));
        }
        for (uint32_t idx = 0; idx < 256u; idx++) {
            const uint32_t k = blk * 256u + idx;
            nib[idx] = (k < in_dim)
                    ? (uint8_t)(((o + 1u) * (k + 3u) + (o * k) % 7u) % 16u)
                    : 0u;
        }
        const float d = 0.05f * (float)(o + blk + 1u);
        const float dmin = 0.02f * (float)(o + 2u * blk + 1u);
        test_q4k_pack_block(row + (size_t)blk * 144u, d, dmin, sc, m, nib);
    }
}

/* Oracle: value = d*sc*nib - dmin*m per column, ds4.c:3709-3736
 * (ds4_vec_dot_q4_K_f32), specialised to one column at a time so the
 * matmul oracle below reads exactly like oracle_matmul_q8_0's shape. */
static float oracle_q4_k_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 256u;
    const uint32_t idx = col % 256u;
    const unsigned char *bp = row + (size_t)blk * 144u;
    const uint16_t draw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
    const uint16_t dminraw = (uint16_t)(bp[2] | ((uint16_t)bp[3] << 8));
    const uint8_t *scales = bp + 4;
    const uint8_t *qs = bp + 16;
    const uint32_t j = idx / 32u;
    const uint32_t pos = idx % 32u;
    uint8_t sc, m;
    test_q4k_get_scale_min((int)j, scales, &sc, &m);
    const uint32_t byte_off = (j >> 1u) * 32u + pos;
    const uint8_t raw = qs[byte_off];
    const uint8_t nib = (j & 1u) ? (uint8_t)(raw >> 4u) : (uint8_t)(raw & 0x0Fu);
    const float d = oracle_half_to_float(draw);
    const float dmin = oracle_half_to_float(dminraw);
    return d * (float)sc * (float)nib - dmin * (float)m;
}

static void oracle_matmul_q4_k(float *out, const float *x, const unsigned char *w,
                               uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                               uint64_t row_bytes) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const unsigned char *row = w + (size_t)o * row_bytes;
            double sum = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) {
                sum += (double)x[t * in_dim + k] * (double)oracle_q4_k_dequant(row, k);
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* IN_DIM spans two superblocks, the second ragged, so both the full-block
 * and remainder paths are exercised; weight_type=12u is Q4_K
 * (DS4_TENSOR_Q4_K in ds4.c), passed numerically per this backend's
 * existing convention (sycl/ds4_sycl_moe_launch.hpp's own comment: ds4_gpu.h
 * declares no DS4_TENSOR_* constants for backend use). */
static int test_matmul_q4_k(void) {
    enum { IN_DIM = 293, OUT_DIM = 3, N_TOK = 2 };
    const uint32_t weight_type = 12u; /* Q4_K */
    const uint64_t blocks = (IN_DIM + 255u) / 256u;
    const uint64_t row_bytes = blocks * 144u;

    unsigned char weights[OUT_DIM * 2 * 144]; /* row_bytes <= 2*144 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q4_k_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q4_k(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_q4_k: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "matmul_q4_k: write x");

    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q4_k: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q4_k: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_q4_k: out mismatch");
    }

    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q4_k: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_q4_k: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_q4_k: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, weight_bytes,
                                      weight_type, IN_DIM, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q4_k: out-of-range weight offset must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, 99u,
                                      IN_DIM, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q4_k: unsupported weight_type must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q4_k OK\n");
    return 0;
}

/* Q4_0 row encoder: the standard GGUF layout (gguf_types[2] in ds4.c,
 * block_elems=32, block_bytes=18): an F16 scale followed by 16 bytes of
 * packed 4-bit codes, byte i holding value i in its low nibble and value
 * i+16 in its high nibble, decoded with a zero-point of 8. Nothing in this
 * codebase decoded this format before now (established by grep: no
 * block_q4_0 struct, no dequant function, anywhere in ds4.c or ds4_cuda.cu). */
static void test_encode_q4_0_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 18u;
        const float d = 0.07f * (float)(o + blk + 1u);
        const uint16_t draw = test_float_to_half(d);
        bp[0] = (unsigned char)(draw & 0xFFu);
        bp[1] = (unsigned char)((draw >> 8) & 0xFFu);
        unsigned char *qs = bp + 2;
        memset(qs, 0, 16);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const uint8_t nib = (k < in_dim)
                    ? (uint8_t)(((o + 1u) * (k + 5u) + (o * k) % 7u) % 16u)
                    : 8u; /* zero-point: contributes exactly 0 when padding */
            if (idx < 16u) qs[idx] = (unsigned char)(qs[idx] | nib);
            else qs[idx - 16u] = (unsigned char)(qs[idx - 16u] | (nib << 4));
        }
    }
}

static float oracle_q4_0_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 32u;
    const uint32_t idx = col % 32u;
    const unsigned char *bp = row + (size_t)blk * 18u;
    const uint16_t draw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
    const float d = oracle_half_to_float(draw);
    const unsigned char *qs = bp + 2;
    const uint8_t byte = (idx < 16u) ? qs[idx] : qs[idx - 16u];
    const uint8_t nib = (idx < 16u) ? (uint8_t)(byte & 0x0Fu) : (uint8_t)(byte >> 4u);
    return d * ((float)nib - 8.0f);
}

static void oracle_matmul_q4_0(float *out, const float *x, const unsigned char *w,
                               uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
                               uint64_t row_bytes) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const unsigned char *row = w + (size_t)o * row_bytes;
            double sum = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) {
                sum += (double)x[t * in_dim + k] * (double)oracle_q4_0_dequant(row, k);
            }
            out[t * out_dim + o] = (float)sum;
        }
    }
}

/* IN_DIM spans two blocks, the second ragged (9 of 32 used), distinct from
 * every Q4_K dimension above so a copy-paste between the two dequantisers
 * would produce a visibly wrong shape, not just a wrong value. */
static int test_matmul_q4_0(void) {
    enum { IN_DIM = 41, OUT_DIM = 4, N_TOK = 3 };
    const uint32_t weight_type = 2u; /* Q4_0 */
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 18u;

    unsigned char weights[OUT_DIM * 2 * 18]; /* row_bytes <= 2*18 here */
    float          x[N_TOK * IN_DIM];
    float          want[N_TOK * OUT_DIM];
    float          got[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q4_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }
    oracle_matmul_q4_0(want, x, weights, IN_DIM, OUT_DIM, N_TOK, row_bytes);

    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_q4_0: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "matmul_q4_0: write x");

    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q4_0: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_q4_0: read out");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_q4_0: out mismatch");
    }

    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      0, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q4_0: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, 0, tx, N_TOK) == 0,
          "matmul_q4_0: zero out_dim must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, 0, weight_type,
                                      IN_DIM, OUT_DIM, tx, 0) == 0,
          "matmul_q4_0: zero n_tok must be rejected");
    CHECK(ds4_gpu_matmul_quant_tensor(tout, weights, weight_bytes, weight_bytes,
                                      weight_type, IN_DIM, OUT_DIM, tx, N_TOK) == 0,
          "matmul_q4_0: out-of-range weight offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q4_0 OK\n");
    return 0;
}

/* Differential test against the Q8_0 equivalent (plan guidance: two real
 * implemented paths disagreeing is a stronger signal than either against an
 * oracle alone). Both the Q4_K row and a Q8_0 row are built to encode
 * approximately the SAME underlying float target: the Q4_K row's target is
 * whatever its (d, sc, m, nib) fields evaluate to EXACTLY (the same formula
 * sycl_q4_k_dequant computes), and that exact target is then Q8_0-quantised
 * (per-32-block amax/127 scale, round to nearest, matching
 * quantize_q8_0_activation-style encoding) into a same-shaped Q8_0 row.
 * ds4_gpu_matmul_q8_0_tensor's result should then match
 * ds4_gpu_matmul_quant_tensor's Q4_K result up to Q8_0's OWN quantisation
 * error (a few tenths of a percent), not up to Q4_K's much coarser 4-bit
 * step -- because the Q4_K side carries no approximation error here, only
 * the Q8_0 side does. */
static void test_encode_q8_0_row_matching_q4_k(unsigned char *q8_row,
                                               const unsigned char *q4k_row,
                                               uint32_t in_dim) {
    const uint32_t q8_blocks = (in_dim + 31u) / 32u;
    for (uint32_t b = 0; b < q8_blocks; b++) {
        const uint32_t i0 = b * 32u;
        const uint32_t bn = (in_dim - i0 < 32u) ? (in_dim - i0) : 32u;
        float amax = 0.0f;
        float target[32];
        for (uint32_t i = 0; i < bn; i++) {
            target[i] = oracle_q4_k_dequant(q4k_row, i0 + i);
            const float ax = fabsf(target[i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        unsigned char *bp = q8_row + (size_t)b * 34u;
        const uint16_t draw = test_float_to_half(d);
        bp[0] = (unsigned char)(draw & 0xFFu);
        bp[1] = (unsigned char)((draw >> 8) & 0xFFu);
        for (uint32_t i = 0; i < 32u; i++) {
            int v = 0;
            if (i < bn) {
                v = (int)lrintf(target[i] * id);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
            }
            bp[2 + i] = (unsigned char)(signed char)v;
        }
    }
}

/* Worst-case propagated error bound for one output row of the Q8_0 side of
 * the differential below: each quantised weight can be off by at most half
 * its block's step (amax/127), so the dot product can be off by at most
 * sum_k |x[k]| * step(k)/2. Sized per (token, out-row) rather than as a
 * flat percentage of the answer, because the target data's mixed-sign
 * d*sc*nibble - dmin*m terms can partially cancel, making a tolerance
 * relative to the (possibly small) final sum too tight even when every
 * individual term's quantisation error is unremarkable. */
static double q8_0_row_error_bound(const float *x_row, const unsigned char *row,
                                   uint32_t in_dim) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    double bound = 0.0;
    for (uint32_t b = 0; b < blocks; b++) {
        const unsigned char *bp = row + (size_t)b * 34u;
        const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
        const double step = (double)oracle_half_to_float(raw);
        const uint32_t i0 = b * 32u;
        const uint32_t bn = (in_dim - i0 < 32u) ? (in_dim - i0) : 32u;
        for (uint32_t i = 0; i < bn; i++) {
            bound += fabs((double)x_row[i0 + i]) * step * 0.5;
        }
    }
    return bound;
}

static int test_matmul_q4_k_vs_q8_0_differential(void) {
    enum { IN_DIM = 256 + 40, OUT_DIM = 3, N_TOK = 2 };
    const uint64_t q4k_blocks = (IN_DIM + 255u) / 256u;
    const uint64_t q4k_row_bytes = q4k_blocks * 144u;
    const uint64_t q8_blocks = (IN_DIM + 31u) / 32u;
    const uint64_t q8_row_bytes = q8_blocks * 34u;

    unsigned char q4k_weights[OUT_DIM * 2 * 144];
    unsigned char q8_weights[OUT_DIM * 10 * 34]; /* q8_blocks <= 10 here */
    float          x[N_TOK * IN_DIM];
    float          got_q4k[N_TOK * OUT_DIM];
    float          got_q8[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q4_k_row(q4k_weights + (size_t)o * q4k_row_bytes, IN_DIM, o);
        test_encode_q8_0_row_matching_q4_k(q8_weights + (size_t)o * q8_row_bytes,
                                           q4k_weights + (size_t)o * q4k_row_bytes,
                                           IN_DIM);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            x[t * IN_DIM + k] = (float)(((t + 1) * (k + 2)) % 9) - 4.0f;
        }
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tq4k = ds4_gpu_tensor_alloc(sizeof(got_q4k));
    ds4_gpu_tensor *tq8 = ds4_gpu_tensor_alloc(sizeof(got_q8));
    CHECK(tx && tq4k && tq8, "matmul_q4_k_vs_q8_0_differential: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "matmul_q4_k_vs_q8_0_differential: write x");

    CHECK(ds4_gpu_matmul_quant_tensor(tq4k, q4k_weights, sizeof(q4k_weights), 0, 12u,
                                      IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q4_k_vs_q8_0_differential: Q4_K call");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tq8, q8_weights, sizeof(q8_weights), 0,
                                     IN_DIM, OUT_DIM, tx, N_TOK) != 0,
          "matmul_q4_k_vs_q8_0_differential: Q8_0 call");
    CHECK(ds4_gpu_tensor_read(tq4k, 0, got_q4k, sizeof(got_q4k)) != 0,
          "matmul_q4_k_vs_q8_0_differential: read Q4_K out");
    CHECK(ds4_gpu_tensor_read(tq8, 0, got_q8, sizeof(got_q8)) != 0,
          "matmul_q4_k_vs_q8_0_differential: read Q8_0 out");

    /* Tolerance is the worst-case Q8_0 quantisation error bound for this
     * exact (token, out-row) pair (see q8_0_row_error_bound), with 50%
     * headroom for accumulation-order differences between the GPU's float32
     * sum and this bound's own double-precision arithmetic. */
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t o = 0; o < OUT_DIM; o++) {
            const double bound = q8_0_row_error_bound(
                    x + (size_t)t * IN_DIM, q8_weights + (size_t)o * q8_row_bytes, IN_DIM);
            const double tol = 1.5 * bound + 1e-3;
            CHECK_CLOSE(got_q8[t * OUT_DIM + o], got_q4k[t * OUT_DIM + o], tol,
                        "matmul_q4_k_vs_q8_0_differential: Q8_0 vs Q4_K mismatch");
        }
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tq4k);
    ds4_gpu_tensor_free(tq8);
    fprintf(stderr, "  test_matmul_q4_k_vs_q8_0_differential OK\n");
    return 0;
}

/* ds4_gpu_matmul_quant_kslice_tensor: out[out_dim] = W[:, k_off:k_off+k_cnt]
 * @ x[x_elem_off:+k_cnt], generalising ds4_gpu.h's doc comment on
 * ds4_gpu_matmul_q8_0_kslice_tensor to every dense-quant type. Both call
 * sites of this entry in ds4.c are gated by g->tp_world == 2 (tensor
 * parallelism, out of scope for this port), so it is not reached by
 * single-GPU Flash; tested directly here since it is listed as an
 * interface to implement regardless. Exercises Q4_K specifically (Q8_0's
 * own kslice entry, ds4_gpu_matmul_q8_0_kslice_tensor, is a separate,
 * permanently-TP-only stub not touched here); k_off/k_cnt are
 * chosen NOT aligned to a superblock boundary, unlike every real caller
 * would use, specifically to prove the column-addressing math (k_off+k
 * into the full un-sliced row) rather than a block-count shortcut that
 * would only work for aligned slices. */
static int test_matmul_quant_kslice_q4_k(void) {
    enum { FULL_IN_DIM = 256 + 40, K_OFF = 50, K_CNT = 200, OUT_DIM = 3 };
    const uint64_t row_bytes = ((FULL_IN_DIM + 255u) / 256u) * 144u;

    unsigned char weights[OUT_DIM * 2 * 144];
    float          x_full[FULL_IN_DIM];
    float          want[OUT_DIM];
    float          got[OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q4_k_row(weights + (size_t)o * row_bytes, FULL_IN_DIM, o);
    }
    for (uint32_t k = 0; k < FULL_IN_DIM; k++) {
        x_full[k] = (float)((k + 2) % 9) - 4.0f;
    }
    for (uint32_t o = 0; o < OUT_DIM; o++) {
        const unsigned char *row = weights + (size_t)o * row_bytes;
        double sum = 0.0;
        for (uint32_t k = 0; k < K_CNT; k++) {
            sum += (double)x_full[K_OFF + k] * (double)oracle_q4_k_dequant(row, K_OFF + k);
        }
        want[o] = (float)sum;
    }

    ds4_gpu_tensor *tx  = ds4_gpu_tensor_alloc(sizeof(x_full));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx != NULL && tout != NULL, "matmul_quant_kslice_q4_k: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x_full, sizeof(x_full)) != 0,
          "matmul_quant_kslice_q4_k: write x");

    CHECK(ds4_gpu_matmul_quant_kslice_tensor(tout, weights, sizeof(weights), 0, 12u,
                                             FULL_IN_DIM, K_OFF, K_CNT, OUT_DIM, tx,
                                             K_OFF) != 0,
          "matmul_quant_kslice_q4_k: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0,
          "matmul_quant_kslice_q4_k: read out");

    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "matmul_quant_kslice_q4_k: out mismatch");
    }

    CHECK(ds4_gpu_matmul_quant_kslice_tensor(tout, weights, sizeof(weights), 0, 12u,
                                             FULL_IN_DIM, FULL_IN_DIM + 1u, K_CNT,
                                             OUT_DIM, tx, 0) == 0,
          "matmul_quant_kslice_q4_k: k_off past full_in_dim must be rejected");
    CHECK(ds4_gpu_matmul_quant_kslice_tensor(tout, weights, sizeof(weights), 0, 12u,
                                             FULL_IN_DIM, K_OFF, FULL_IN_DIM, OUT_DIM,
                                             tx, K_OFF) == 0,
          "matmul_quant_kslice_q4_k: k_cnt past full_in_dim must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_quant_kslice_q4_k OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_matmul_f16_pair() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_f16_pair_compressor_store() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_kslice_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_kslice() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_decode_mpp() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_decode_mpp_model_view() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_rows_scalar() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_pair() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_f32() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_f16() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_f16_out() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_pack_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q4_k() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q4_0() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q4_k_vs_q8_0_differential() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_quant_kslice_q4_k() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_matmul OK\n");
    return 0;
}
