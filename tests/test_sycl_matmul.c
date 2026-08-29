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
 * by hand-encoding IEEE754 binary16 bit patterns.  Restricted to exactly
 * representable small integers (the only values this file's test data
 * uses), so plain truncation of the mantissa is exact, not an
 * approximation. */
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
 * `(void)` on all 14 parameters then `return 0;`, a real and complete
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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_matmul_f16_pair() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_f16_pair_compressor_store() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_matmul OK\n");
    return 0;
}
