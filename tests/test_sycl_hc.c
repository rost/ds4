/* Correctness tests for the SYCL hyper-connection (HC) subsystem, validated
 * against scalar CPU oracles implemented here.  The ds4.c CPU references
 * are static and cannot be linked, so each oracle reimplements the
 * documented formula with the ds4.c line numbers cited.  Needs no model
 * file: the "model map" used to exercise mmap staging is a plain host
 * float buffer standing in for one.
 *
 * Shares CHECK/CHECK_CLOSE and oracle_rms_norm_weight with
 * test_sycl_harness.h (the fused norm entry's oracle needs exactly that
 * weighted RMS norm); every other oracle here is specific to this
 * subsystem and lives in this file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_HC = 4, MIX_HC = 2 * N_HC + N_HC * N_HC /* 24 */ };

/* n_embd for the fused matmul-plus-HC-expand tests below: this family
 * requires out_dim == n_embd, so both dimensions share one constant. */
enum { N_HC_EXPAND_EMBD = 12 };

/* Matches hc_split_sinkhorn_one, ds4.c:9718-9814.  n_hc is fixed at 4 to
 * match the GPU kernel's own hardcoding (hc4_split_one never takes n_hc as
 * a parameter). out layout: [0,4) pre, [4,8) post, [8,24) comb, addressed
 * comb[dst + src*4]. */
static void oracle_hc_split_sinkhorn_one(float *out, const float *mix,
                                         const float *scale, const float *base,
                                         int iters, float eps) {
    const float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
    for (int i = 0; i < N_HC; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (int i = 0; i < N_HC; i++) {
        const int off = N_HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }
    float c[N_HC * N_HC];
    for (int dst = 0; dst < N_HC; dst++) {
        float row_max = -INFINITY;
        for (int src = 0; src < N_HC; src++) {
            const int idx = src + dst * N_HC;
            const int off = 2 * N_HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < N_HC; src++) {
            const int idx = src + dst * N_HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < N_HC; src++) {
            const int idx = src + dst * N_HC;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (int src = 0; src < N_HC; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < N_HC; dst++) sum += c[src + dst * N_HC];
        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < N_HC; dst++) c[src + dst * N_HC] *= inv;
    }
    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < N_HC; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < N_HC; src++) sum += c[src + dst * N_HC];
            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < N_HC; src++) c[src + dst * N_HC] *= inv;
        }
        for (int src = 0; src < N_HC; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < N_HC; dst++) sum += c[src + dst * N_HC];
            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < N_HC; dst++) c[src + dst * N_HC] *= inv;
        }
    }
    for (int i = 0; i < N_HC * N_HC; i++) out[2 * N_HC + i] = c[i];
}

/* Matches hc_weighted_sum_one, ds4.c:9816-9828. */
static void oracle_hc_weighted_sum_one(float *out, const float *x, const float *weights,
                                       uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) acc += x[(uint64_t)h * n_embd + d] * weights[h];
        out[d] = acc;
    }
}

/* Matches hc_post_one, ds4.c (around :9920): injects the new block output
 * and mixes the previous HC streams through the learned combine matrix,
 * addressed comb[dst + src * n_hc]. */
static void oracle_hc_post_one(float *out_hc, const float *block_out, const float *residual_hc,
                               const float *post, const float *comb, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }
            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

/* A small table of exact IEEE754 binary16 bit patterns and the float
 * values they represent (positive and negative, several magnitudes),
 * used to build F16 test inputs without needing a float32-to-float16
 * encoder in the test itself: since decode (F16 to F32) is lossless, per
 * design-spec section 6k, any input built from an EXACT encoding
 * round-trips exactly through the kernel under test, so hand-picked exact
 * bit patterns are sufficient and avoid relying on a second, separately-
 * fallible encoder. */
static const uint16_t kHalfBits[] = {
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600,
    0x3800, 0x3400, 0xBC00, 0xC000, 0xC500,
};
static const float kHalfVals[] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 0.5f, 0.25f, -1.0f, -2.0f, -5.0f,
};
enum { N_HALF_TABLE = sizeof(kHalfBits) / sizeof(kHalfBits[0]) };

static ds4_gpu_tensor *alloc_write(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) return NULL;
    if (ds4_gpu_tensor_write(t, 0, data, bytes) == 0) return NULL;
    return t;
}

/* Deterministic pseudo-random floats in a small range, with a non-affine
 * per-element term so no two rows are proportional to each other: per
 * design-spec section 6f, an ablation downstream of a normalisation
 * (Sinkhorn row/column normalisation here) can be laundered invisible by
 * purely affine test data. */
static float fill_val(uint32_t i, uint32_t j) {
    return sinf((float)(i * 7 + j * 13 + 1)) * 0.7f + 0.05f * (float)((i * j) % 11);
}

/* Minimal local Q8_0 support for the fused matmul-plus-HC-expand tests
 * below: an IEEE754 binary16 encoder (truncating, not round-to-nearest,
 * exactly like tests/test_sycl_matmul.c's test_float_to_half; safe here
 * for the same reason that file documents -- the oracle below decodes the
 * identical stored bits via oracle_half_to_float, so both sides agree
 * regardless of how those bits were produced), a row encoder carrying an
 * (o, k) interaction term rather than a plain a*i + b so a row mixup or a
 * dropped block stride shows up as a real mismatch, and a host-side dot
 * product oracle matching sycl_q8_0_dequant in sycl/ds4_sycl_common.hpp. */
static uint16_t hc_test_float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void hc_test_encode_q8_0_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t braw = hc_test_float_to_half(0.03f * (float)(o + blk + 1u));
        bp[0] = (unsigned char)(braw & 0xFFu);
        bp[1] = (unsigned char)((braw >> 8) & 0xFFu);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const int qv = (k < in_dim) ? (int)(((o + 1u) * (k + 5u)) % 11u) - 5 : 0;
            bp[2 + idx] = (unsigned char)(signed char)qv;
        }
    }
}

static float hc_oracle_q8_0_dot(const float *x, const unsigned char *row, uint32_t in_dim) {
    double sum = 0.0;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        const unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
        const float scale = oracle_half_to_float(raw);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            if (k >= in_dim) break;
            const signed char qv = (signed char)bp[2 + idx];
            sum += (double)x[k] * (double)scale * (double)qv;
        }
    }
    return (float)sum;
}

static int test_repeat_hc(void) {
    enum { N_EMBD = 37 };
    float row[N_EMBD], got[N_EMBD * N_HC];
    for (int i = 0; i < N_EMBD; i++) row[i] = fill_val((uint32_t)i, 3);

    ds4_gpu_tensor *trow = alloc_write(row, sizeof(row));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(trow && tout, "repeat_hc: alloc failed");

    CHECK(ds4_gpu_repeat_hc_tensor(tout, trow, N_EMBD, N_HC) != 0, "repeat_hc: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "repeat_hc: read");
    for (int h = 0; h < N_HC; h++)
        for (int i = 0; i < N_EMBD; i++)
            CHECK_CLOSE(got[h * N_EMBD + i], row[i], 1e-7, "repeat_hc: value mismatch");

    /* Zero n_embd or n_hc is a validation FAILURE in this file, matching
     * rocm/ds4_rocm_hc_output_launch.cuh:2-3 (`if (n_embd == 0 || n_hc ==
     * 0 || ...) return 0;`), unlike some entries in sibling files where a
     * zero dimension is a free success. */
    CHECK(ds4_gpu_repeat_hc_tensor(tout, trow, 0, N_HC) == 0, "repeat_hc: n_embd=0 must fail");
    CHECK(ds4_gpu_repeat_hc_tensor(tout, trow, N_EMBD, 0) == 0, "repeat_hc: n_hc=0 must fail");

    ds4_gpu_tensor_free(trow);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_repeat_hc OK\n");
    return 0;
}

static int test_repeat_hc_rows(void) {
    enum { N_TOK = 5, N_EMBD = 11 };
    float rows[N_TOK * N_EMBD], got[N_TOK * N_HC * N_EMBD];
    for (int t = 0; t < N_TOK; t++)
        for (int i = 0; i < N_EMBD; i++) rows[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i);

    ds4_gpu_tensor *trows = alloc_write(rows, sizeof(rows));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(trows && tout, "repeat_hc_rows: alloc failed");

    CHECK(ds4_gpu_repeat_hc_rows_tensor(tout, trows, N_TOK, N_EMBD, N_HC) != 0,
          "repeat_hc_rows: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "repeat_hc_rows: read");
    for (int t = 0; t < N_TOK; t++)
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                CHECK_CLOSE(got[(t * N_HC + h) * N_EMBD + i], rows[t * N_EMBD + i], 1e-7,
                            "repeat_hc_rows: value mismatch");

    CHECK(ds4_gpu_repeat_hc_rows_tensor(tout, trows, 0, N_EMBD, N_HC) == 0,
          "repeat_hc_rows: n_tokens=0 must fail");

    ds4_gpu_tensor_free(trows);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_repeat_hc_rows OK\n");
    return 0;
}

static int test_hc_split_sinkhorn(void) {
    enum { N_ROWS = 6, ITERS = 5 };
    const float eps = 1.0e-6f;
    float mix[N_ROWS * MIX_HC];
    for (int r = 0; r < N_ROWS; r++)
        for (int i = 0; i < MIX_HC; i++) mix[r * MIX_HC + i] = fill_val((uint32_t)r, (uint32_t)i) * 2.0f;

    /* Fake model map: scale (3 floats) then base (24 floats). */
    float model[3 + MIX_HC];
    model[0] = 1.1f; model[1] = 0.9f; model[2] = 1.3f;
    for (int i = 0; i < MIX_HC; i++) model[3 + i] = fill_val(99, (uint32_t)i) * 0.3f;
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = 3 * sizeof(float);

    float want[N_ROWS * MIX_HC], got[N_ROWS * MIX_HC];
    for (int r = 0; r < N_ROWS; r++)
        oracle_hc_split_sinkhorn_one(want + r * MIX_HC, mix + r * MIX_HC, model, model + 3, ITERS, eps);

    ds4_gpu_tensor *tmix = alloc_write(mix, sizeof(mix));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tmix && tout, "hc_split_sinkhorn: alloc failed");

    CHECK(ds4_gpu_hc_split_sinkhorn_tensor(tout, tmix, model, sizeof(model),
                                           scale_offset, base_offset, N_HC, ITERS, eps) != 0,
          "hc_split_sinkhorn: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_split_sinkhorn: read");
    for (int i = 0; i < N_ROWS * MIX_HC; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_split_sinkhorn: value mismatch");

    /* n_hc != 4 must be rejected: hc4_split_one hardcodes n_hc == 4. */
    CHECK(ds4_gpu_hc_split_sinkhorn_tensor(tout, tmix, model, sizeof(model),
                                           scale_offset, base_offset, 3, ITERS, eps) == 0,
          "hc_split_sinkhorn: n_hc != 4 must be rejected");

    /* An out-of-range scale/base offset into the model map must be
     * rejected: this is the mmap-staging validation guard, per design-spec
     * section 6l. */
    CHECK(ds4_gpu_hc_split_sinkhorn_tensor(tout, tmix, model, sizeof(model),
                                           sizeof(model), base_offset, N_HC, ITERS, eps) == 0,
          "hc_split_sinkhorn: out-of-range scale offset must be rejected");
    CHECK(ds4_gpu_hc_split_sinkhorn_tensor(tout, tmix, model, sizeof(model),
                                           scale_offset, sizeof(model), N_HC, ITERS, eps) == 0,
          "hc_split_sinkhorn: out-of-range base offset must be rejected");

    ds4_gpu_tensor_free(tmix);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_split_sinkhorn OK\n");
    return 0;
}

static int test_hc_weighted_sum(void) {
    enum { N_TOK = 5, N_EMBD = 20 };
    float residual[N_TOK * N_HC * N_EMBD], weights[N_TOK * N_HC], got[N_TOK * N_EMBD];
    for (int t = 0; t < N_TOK; t++) {
        for (int h = 0; h < N_HC; h++) {
            weights[t * N_HC + h] = fill_val((uint32_t)t, (uint32_t)h) * 0.5f + 0.6f;
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i);
        }
    }

    float want[N_TOK * N_EMBD];
    for (int t = 0; t < N_TOK; t++)
        oracle_hc_weighted_sum_one(want + t * N_EMBD, residual + (uint64_t)t * N_HC * N_EMBD,
                                   weights + t * N_HC, N_EMBD, N_HC);

    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tw = alloc_write(weights, sizeof(weights));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tres && tw && tout, "hc_weighted_sum: alloc failed");

    CHECK(ds4_gpu_hc_weighted_sum_tensor(tout, tres, tw, N_EMBD, N_HC) != 0,
          "hc_weighted_sum: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_weighted_sum: read");
    for (int i = 0; i < N_TOK * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_weighted_sum: value mismatch");

    ds4_gpu_tensor *tw_small = ds4_gpu_tensor_alloc(sizeof(weights) / 2);
    CHECK(tw_small, "hc_weighted_sum: small alloc failed");
    CHECK(ds4_gpu_hc_weighted_sum_tensor(tout, tres, tw_small, N_EMBD, N_HC) == 0,
          "hc_weighted_sum: undersized weights must be rejected");
    ds4_gpu_tensor_free(tw_small);

    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_weighted_sum OK\n");
    return 0;
}

static int test_hc_weighted_sum_split(void) {
    enum { N_TOK = 4, N_EMBD = 16 };
    float residual[N_TOK * N_HC * N_EMBD], split[N_TOK * MIX_HC], got[N_TOK * N_EMBD];
    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < MIX_HC; i++) split[t * MIX_HC + i] = fill_val((uint32_t)t, (uint32_t)i);
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.0f;
    }

    float want[N_TOK * N_EMBD];
    for (int t = 0; t < N_TOK; t++)
        oracle_hc_weighted_sum_one(want + t * N_EMBD, residual + (uint64_t)t * N_HC * N_EMBD,
                                   split + t * MIX_HC, N_EMBD, N_HC);

    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tres && tsplit && tout, "hc_weighted_sum_split: alloc failed");

    CHECK(ds4_gpu_hc_weighted_sum_split_tensor(tout, tres, tsplit, N_EMBD, N_HC) != 0,
          "hc_weighted_sum_split: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_weighted_sum_split: read");
    for (int i = 0; i < N_TOK * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_weighted_sum_split: value mismatch");

    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_weighted_sum_split OK\n");
    return 0;
}

/* ds4_gpu_hc_expand_tensor always uses the general (runtime n_hc, not
 * unrolled) kernel regardless of n_hc, per rocm/ds4_rocm_hc_output_
 * launch.cuh:235-257, which has no n_hc == 4 branch at all -- unlike the
 * split-family entries below.  HC = 3 here (not 4) to exercise that
 * generality directly rather than only ever running the value DS4 Flash
 * happens to configure. */
static int test_hc_expand(void) {
    enum { N_TOK = 3, N_EMBD = 10, HC = 3 };
    float block_out[N_TOK * N_EMBD];
    float residual[N_TOK * HC * N_EMBD];
    float post[N_TOK * HC];
    float comb[N_TOK * HC * HC];
    float want[N_TOK * HC * N_EMBD], got[N_TOK * HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i);
        for (int h = 0; h < HC; h++) {
            post[t * HC + h] = fill_val((uint32_t)t, (uint32_t)h + 50) * 0.4f;
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 2.0f;
        }
        for (int i = 0; i < HC * HC; i++) comb[t * HC * HC + i] = fill_val((uint32_t)t, (uint32_t)i + 90) * 0.3f;
        oracle_hc_post_one(want + (uint64_t)t * HC * N_EMBD, block_out + t * N_EMBD,
                           residual + (uint64_t)t * HC * N_EMBD, post + t * HC, comb + t * HC * HC,
                           N_EMBD, HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tpost = alloc_write(post, sizeof(post));
    ds4_gpu_tensor *tcomb = alloc_write(comb, sizeof(comb));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tres && tpost && tcomb && tout, "hc_expand: alloc failed");

    CHECK(ds4_gpu_hc_expand_tensor(tout, tblock, tres, tpost, tcomb, N_EMBD, HC) != 0,
          "hc_expand: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand: read");
    for (int i = 0; i < N_TOK * HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand: value mismatch");

    CHECK(ds4_gpu_hc_expand_tensor(tout, tblock, tres, tpost, tcomb, 0, HC) == 0,
          "hc_expand: n_embd=0 must fail");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tpost);
    ds4_gpu_tensor_free(tcomb);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_expand OK\n");
    return 0;
}

/* Tensor parallelism: ds4_gpu_hc_expand_add_tensor, the ATTN-gate
 * combine ds4.c:23625/23629 folds the two TP ranks' partial attention
 * outputs into. Oracle: oracle_hc_post_one fed block_out+block_add as its
 * block_out argument, matching ds4_cuda.cu's shared hc_expand_kernel
 * called with is_add=1 (the kernel adds block_add to block_out internally
 * before the same combine math test_hc_expand above already validates).
 * Differential against ds4_gpu_hc_expand_tensor with a pre-summed
 * block_out and an all-zero block_add is the more direct proof that the
 * "add" step is exactly a sum, not merely that both reach the same oracle
 * formula. */
static int test_hc_expand_add(void) {
    enum { N_TOK = 2, N_EMBD = 8, HC = 3 };
    float block_out[N_TOK * N_EMBD];
    float block_add[N_TOK * N_EMBD];
    float block_sum[N_TOK * N_EMBD];
    float residual[N_TOK * HC * N_EMBD];
    float post[N_TOK * HC];
    float comb[N_TOK * HC * HC];
    float want[N_TOK * HC * N_EMBD], got[N_TOK * HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) {
            block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i);
            block_add[t * N_EMBD + i] = fill_val((uint32_t)i, (uint32_t)t + 20u) * 1.3f;
            block_sum[t * N_EMBD + i] = block_out[t * N_EMBD + i] + block_add[t * N_EMBD + i];
        }
        for (int h = 0; h < HC; h++) {
            post[t * HC + h] = fill_val((uint32_t)t, (uint32_t)h + 50) * 0.4f;
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 2.0f;
        }
        for (int i = 0; i < HC * HC; i++) comb[t * HC * HC + i] = fill_val((uint32_t)t, (uint32_t)i + 90) * 0.3f;
        oracle_hc_post_one(want + (uint64_t)t * HC * N_EMBD, block_sum + t * N_EMBD,
                           residual + (uint64_t)t * HC * N_EMBD, post + t * HC, comb + t * HC * HC,
                           N_EMBD, HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tadd   = alloc_write(block_add, sizeof(block_add));
    ds4_gpu_tensor *tres   = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tpost  = alloc_write(post, sizeof(post));
    ds4_gpu_tensor *tcomb  = alloc_write(comb, sizeof(comb));
    ds4_gpu_tensor *tout   = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tadd && tres && tpost && tcomb && tout, "hc_expand_add: alloc failed");

    CHECK(ds4_gpu_hc_expand_add_tensor(tout, tblock, tadd, tres, tpost, tcomb, N_EMBD, HC) != 0,
          "hc_expand_add: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand_add: read");
    for (int i = 0; i < N_TOK * HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand_add: value mismatch");

    /* Differential: hc_expand_add(block_out, block_add, ...) must equal
     * hc_expand(block_out+block_add, ...) -- the "add" is exactly a sum. */
    ds4_gpu_tensor *tsum = alloc_write(block_sum, sizeof(block_sum));
    ds4_gpu_tensor *tout_ref = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tsum && tout_ref, "hc_expand_add: differential alloc failed");
    CHECK(ds4_gpu_hc_expand_tensor(tout_ref, tsum, tres, tpost, tcomb, N_EMBD, HC) != 0,
          "hc_expand_add: differential call");
    float got_ref[N_TOK * HC * N_EMBD];
    CHECK(ds4_gpu_tensor_read(tout_ref, 0, got_ref, sizeof(got_ref)) != 0,
          "hc_expand_add: differential read");
    for (int i = 0; i < N_TOK * HC * N_EMBD; i++) {
        CHECK_CLOSE(got[i], got_ref[i], 1e-4,
                    "hc_expand_add: diverges from hc_expand(pre-summed block)");
    }

    CHECK(ds4_gpu_hc_expand_add_tensor(tout, tblock, tadd, tres, tpost, tcomb, 0, HC) == 0,
          "hc_expand_add: n_embd=0 must fail");
    CHECK(ds4_gpu_hc_expand_add_tensor(NULL, tblock, tadd, tres, tpost, tcomb, N_EMBD, HC) == 0,
          "hc_expand_add: null out_hc must fail");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tadd);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tpost);
    ds4_gpu_tensor_free(tcomb);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tsum);
    ds4_gpu_tensor_free(tout_ref);
    fprintf(stderr, "  test_hc_expand_add OK\n");
    return 0;
}

/* n_hc == 4: exercises ds4_gpu_hc_expand_split_tensor's specialised
 * (unrolled) path, and differentially cross-checks it against
 * ds4_gpu_hc_expand_tensor's always-general kernel fed the same data via
 * separately-packed post/comb tensors extracted from the same split row.
 * This is the primary defence against a drifted
 * unrolled specialisation: the two entries must agree even though neither
 * is validated against the other as a matter of course. */
static int test_hc_expand_split_n4_and_differential(void) {
    enum { N_TOK = 3, N_EMBD = 12 };
    float block_out[N_TOK * N_EMBD];
    float residual[N_TOK * N_HC * N_EMBD];
    float split[N_TOK * MIX_HC];
    float want[N_TOK * N_HC * N_EMBD];
    float got_specialised[N_TOK * N_HC * N_EMBD];
    float got_general[N_TOK * N_HC * N_EMBD];
    float post_flat[N_TOK * N_HC];
    float comb_flat[N_TOK * N_HC * N_HC];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i);
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 3.0f;
        /* pre section (unused by expand) still filled, to catch an
         * implementation that reads the wrong offset into split. */
        for (int i = 0; i < MIX_HC; i++) split[t * MIX_HC + i] = fill_val((uint32_t)t, (uint32_t)i + 40) * 0.5f;

        const float *post = split + t * MIX_HC + N_HC;
        const float *comb = split + t * MIX_HC + 2 * N_HC;
        oracle_hc_post_one(want + (uint64_t)t * N_HC * N_EMBD, block_out + t * N_EMBD,
                           residual + (uint64_t)t * N_HC * N_EMBD, post, comb, N_EMBD, N_HC);
        memcpy(post_flat + t * N_HC, post, N_HC * sizeof(float));
        memcpy(comb_flat + t * N_HC * N_HC, comb, N_HC * N_HC * sizeof(float));
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got_specialised));
    CHECK(tblock && tres && tsplit && tout, "hc_expand_split: alloc failed");

    CHECK(ds4_gpu_hc_expand_split_tensor(tout, tblock, tres, tsplit, N_EMBD, N_HC) != 0,
          "hc_expand_split: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got_specialised, sizeof(got_specialised)) != 0,
          "hc_expand_split: read");
    for (int i = 0; i < N_TOK * N_HC * N_EMBD; i++)
        CHECK_CLOSE(got_specialised[i], want[i], 1e-4, "hc_expand_split: value mismatch");

    ds4_gpu_tensor *tpost = alloc_write(post_flat, sizeof(post_flat));
    ds4_gpu_tensor *tcomb = alloc_write(comb_flat, sizeof(comb_flat));
    ds4_gpu_tensor *tout2 = ds4_gpu_tensor_alloc(sizeof(got_general));
    CHECK(tpost && tcomb && tout2, "hc_expand_split: differential alloc failed");
    CHECK(ds4_gpu_hc_expand_tensor(tout2, tblock, tres, tpost, tcomb, N_EMBD, N_HC) != 0,
          "hc_expand_split: differential call");
    CHECK(ds4_gpu_tensor_read(tout2, 0, got_general, sizeof(got_general)) != 0,
          "hc_expand_split: differential read");
    for (int i = 0; i < N_TOK * N_HC * N_EMBD; i++)
        CHECK_CLOSE(got_specialised[i], got_general[i], 1e-4,
                    "hc_expand_split: specialised/general differential mismatch");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tpost);
    ds4_gpu_tensor_free(tcomb);
    ds4_gpu_tensor_free(tout2);
    fprintf(stderr, "  test_hc_expand_split_n4_and_differential OK\n");
    return 0;
}

/* n_hc == 3: exercises ds4_gpu_hc_expand_split_tensor's general fallback
 * branch directly (the one taken whenever n_hc != 4), against the oracle. */
static int test_hc_expand_split_general_fallback(void) {
    enum { N_TOK = 2, N_EMBD = 9, HC = 3, MIX = 2 * HC + HC * HC };
    float block_out[N_TOK * N_EMBD];
    float residual[N_TOK * HC * N_EMBD];
    float split[N_TOK * MIX];
    float want[N_TOK * HC * N_EMBD], got[N_TOK * HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i + 5);
        for (int h = 0; h < HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.5f;
        for (int i = 0; i < MIX; i++) split[t * MIX + i] = fill_val((uint32_t)t, (uint32_t)i + 20) * 0.6f;
        const float *post = split + t * MIX + HC;
        const float *comb = split + t * MIX + 2 * HC;
        oracle_hc_post_one(want + (uint64_t)t * HC * N_EMBD, block_out + t * N_EMBD,
                           residual + (uint64_t)t * HC * N_EMBD, post, comb, N_EMBD, HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tres && tsplit && tout, "hc_expand_split(fallback): alloc failed");

    CHECK(ds4_gpu_hc_expand_split_tensor(tout, tblock, tres, tsplit, N_EMBD, HC) != 0,
          "hc_expand_split(fallback): call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand_split(fallback): read");
    for (int i = 0; i < N_TOK * HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand_split(fallback): value mismatch");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_expand_split_general_fallback OK\n");
    return 0;
}

/* n_hc == 4, F16 block_out: decode-only, per design-spec section 6k this
 * direction is lossless regardless of sycl::half vs a ported bit
 * manipulation, so the exact-bit-pattern table above is sufficient. */
static int test_hc_expand_split_half(void) {
    enum { N_TOK = 3, N_EMBD = 8 };
    uint16_t block_out_h[N_TOK * N_EMBD];
    float block_out_f[N_TOK * N_EMBD];
    float residual[N_TOK * N_HC * N_EMBD];
    float split[N_TOK * MIX_HC];
    float want[N_TOK * N_HC * N_EMBD], got[N_TOK * N_HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) {
            const int idx = (t * N_EMBD + i) % N_HALF_TABLE;
            block_out_h[t * N_EMBD + i] = kHalfBits[idx];
            block_out_f[t * N_EMBD + i] = kHalfVals[idx];
        }
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 0.7f;
        for (int i = 0; i < MIX_HC; i++) split[t * MIX_HC + i] = fill_val((uint32_t)t, (uint32_t)i + 60) * 0.4f;
        const float *post = split + t * MIX_HC + N_HC;
        const float *comb = split + t * MIX_HC + 2 * N_HC;
        oracle_hc_post_one(want + (uint64_t)t * N_HC * N_EMBD, block_out_f + t * N_EMBD,
                           residual + (uint64_t)t * N_HC * N_EMBD, post, comb, N_EMBD, N_HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out_h, sizeof(block_out_h));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tres && tsplit && tout, "hc_expand_split_half: alloc failed");

    CHECK(ds4_gpu_hc_expand_split_half_tensor(tout, tblock, tres, tsplit, N_EMBD, N_HC) != 0,
          "hc_expand_split_half: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand_split_half: read");
    for (int i = 0; i < N_TOK * N_HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand_split_half: value mismatch");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_expand_split_half OK\n");
    return 0;
}

/* n_hc == 4, F32 block_out + F32 block_add: specialised add path. */
static int test_hc_expand_add_split(void) {
    enum { N_TOK = 3, N_EMBD = 8 };
    float block_out[N_TOK * N_EMBD], block_add[N_TOK * N_EMBD], block_sum[N_TOK * N_EMBD];
    float residual[N_TOK * N_HC * N_EMBD];
    float split[N_TOK * MIX_HC];
    float want[N_TOK * N_HC * N_EMBD], got[N_TOK * N_HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) {
            block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i + 1);
            block_add[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i + 2) * 0.5f;
            block_sum[t * N_EMBD + i] = block_out[t * N_EMBD + i] + block_add[t * N_EMBD + i];
        }
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 2.5f;
        for (int i = 0; i < MIX_HC; i++) split[t * MIX_HC + i] = fill_val((uint32_t)t, (uint32_t)i + 70) * 0.35f;
        const float *post = split + t * MIX_HC + N_HC;
        const float *comb = split + t * MIX_HC + 2 * N_HC;
        oracle_hc_post_one(want + (uint64_t)t * N_HC * N_EMBD, block_sum + t * N_EMBD,
                           residual + (uint64_t)t * N_HC * N_EMBD, post, comb, N_EMBD, N_HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tadd = alloc_write(block_add, sizeof(block_add));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tadd && tres && tsplit && tout, "hc_expand_add_split: alloc failed");

    CHECK(ds4_gpu_hc_expand_add_split_tensor(tout, tblock, tadd, tres, tsplit, N_EMBD, N_HC) != 0,
          "hc_expand_add_split: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand_add_split: read");
    for (int i = 0; i < N_TOK * N_HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand_add_split: value mismatch");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tadd);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_expand_add_split OK\n");
    return 0;
}

/* n_hc == 4, F32 block_out + F16 block_add: decode-then-add, specialised
 * path.  Decode is lossless (design-spec section 6k), so the exact-bit
 * table is again sufficient. */
static int test_hc_expand_add_split_half_add(void) {
    enum { N_TOK = 3, N_EMBD = 8 };
    float block_out[N_TOK * N_EMBD];
    uint16_t block_add_h[N_TOK * N_EMBD];
    float block_sum[N_TOK * N_EMBD];
    float residual[N_TOK * N_HC * N_EMBD];
    float split[N_TOK * MIX_HC];
    float want[N_TOK * N_HC * N_EMBD], got[N_TOK * N_HC * N_EMBD];

    for (int t = 0; t < N_TOK; t++) {
        for (int i = 0; i < N_EMBD; i++) {
            const int idx = (t * N_EMBD + i + 3) % N_HALF_TABLE;
            block_out[t * N_EMBD + i] = fill_val((uint32_t)t, (uint32_t)i + 4);
            block_add_h[t * N_EMBD + i] = kHalfBits[idx];
            block_sum[t * N_EMBD + i] = block_out[t * N_EMBD + i] + kHalfVals[idx];
        }
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(t * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.2f;
        for (int i = 0; i < MIX_HC; i++) split[t * MIX_HC + i] = fill_val((uint32_t)t, (uint32_t)i + 80) * 0.45f;
        const float *post = split + t * MIX_HC + N_HC;
        const float *comb = split + t * MIX_HC + 2 * N_HC;
        oracle_hc_post_one(want + (uint64_t)t * N_HC * N_EMBD, block_sum + t * N_EMBD,
                           residual + (uint64_t)t * N_HC * N_EMBD, post, comb, N_EMBD, N_HC);
    }

    ds4_gpu_tensor *tblock = alloc_write(block_out, sizeof(block_out));
    ds4_gpu_tensor *tadd = alloc_write(block_add_h, sizeof(block_add_h));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tblock && tadd && tres && tsplit && tout, "hc_expand_add_split_half_add: alloc failed");

    CHECK(ds4_gpu_hc_expand_add_split_half_add_tensor(tout, tblock, tadd, tres, tsplit, N_EMBD, N_HC) != 0,
          "hc_expand_add_split_half_add: call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got, sizeof(got)) != 0, "hc_expand_add_split_half_add: read");
    for (int i = 0; i < N_TOK * N_HC * N_EMBD; i++)
        CHECK_CLOSE(got[i], want[i], 1e-4, "hc_expand_add_split_half_add: value mismatch");

    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tadd);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_hc_expand_add_split_half_add OK\n");
    return 0;
}

/* Fuses hc_split_sinkhorn_one and hc_weighted_sum_one for n_rows tokens in
 * one launch, per rocm/ds4_rocm_hc_output_launch.cuh:101-145.  Verified
 * against: (a) the same composed oracle used by test_hc_split_sinkhorn and
 * test_hc_weighted_sum_split; (b) a differential check that the SAME split
 * scratch tensor this entry writes agrees with what ds4_gpu_hc_split_
 * sinkhorn_tensor alone would have produced (the fused kernel must still
 * emit a genuine split, not just fold it away internally); (c) a rejection
 * of n_hc != 4. */
static int test_hc_split_weighted_sum(void) {
    enum { N_ROWS = 5, N_EMBD = 20, ITERS = 6 };
    const float eps = 1.0e-6f;
    float mix[N_ROWS * MIX_HC];
    float residual[N_ROWS * N_HC * N_EMBD];
    for (int r = 0; r < N_ROWS; r++) {
        for (int i = 0; i < MIX_HC; i++) mix[r * MIX_HC + i] = fill_val((uint32_t)r, (uint32_t)i) * 1.7f;
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(r * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.0f;
    }
    float model[3 + MIX_HC];
    model[0] = 0.95f; model[1] = 1.05f; model[2] = 0.85f;
    for (int i = 0; i < MIX_HC; i++) model[3 + i] = fill_val(77, (uint32_t)i) * 0.25f;
    const uint64_t scale_offset = 0, base_offset = 3 * sizeof(float);

    float want_split[N_ROWS * MIX_HC], want_out[N_ROWS * N_EMBD];
    for (int r = 0; r < N_ROWS; r++) {
        oracle_hc_split_sinkhorn_one(want_split + r * MIX_HC, mix + r * MIX_HC, model, model + 3,
                                     ITERS, eps);
        oracle_hc_weighted_sum_one(want_out + r * N_EMBD, residual + (uint64_t)r * N_HC * N_EMBD,
                                   want_split + r * MIX_HC, N_EMBD, N_HC);
    }

    ds4_gpu_tensor *tmix = alloc_write(mix, sizeof(mix));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = ds4_gpu_tensor_alloc(sizeof(want_split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(want_out));
    CHECK(tmix && tres && tsplit && tout, "hc_split_weighted_sum: alloc failed");

    CHECK(ds4_gpu_hc_split_weighted_sum_tensor(tout, tsplit, tmix, tres, model, sizeof(model),
                                               scale_offset, base_offset, N_EMBD, N_HC, ITERS,
                                               eps) != 0,
          "hc_split_weighted_sum: call");

    float got_split[N_ROWS * MIX_HC], got_out[N_ROWS * N_EMBD];
    CHECK(ds4_gpu_tensor_read(tsplit, 0, got_split, sizeof(got_split)) != 0,
          "hc_split_weighted_sum: read split");
    CHECK(ds4_gpu_tensor_read(tout, 0, got_out, sizeof(got_out)) != 0,
          "hc_split_weighted_sum: read out");
    for (int i = 0; i < N_ROWS * MIX_HC; i++)
        CHECK_CLOSE(got_split[i], want_split[i], 1e-4, "hc_split_weighted_sum: split mismatch");
    for (int i = 0; i < N_ROWS * N_EMBD; i++)
        CHECK_CLOSE(got_out[i], want_out[i], 1e-4, "hc_split_weighted_sum: out mismatch");

    /* Differential: the fused entry's split output must agree with the
     * standalone hc_split_sinkhorn_tensor on the same inputs. */
    ds4_gpu_tensor *tsplit_ref = ds4_gpu_tensor_alloc(sizeof(want_split));
    CHECK(tsplit_ref, "hc_split_weighted_sum: ref alloc failed");
    CHECK(ds4_gpu_hc_split_sinkhorn_tensor(tsplit_ref, tmix, model, sizeof(model), scale_offset,
                                           base_offset, N_HC, ITERS, eps) != 0,
          "hc_split_weighted_sum: reference split call");
    float got_split_ref[N_ROWS * MIX_HC];
    CHECK(ds4_gpu_tensor_read(tsplit_ref, 0, got_split_ref, sizeof(got_split_ref)) != 0,
          "hc_split_weighted_sum: read ref split");
    for (int i = 0; i < N_ROWS * MIX_HC; i++)
        CHECK_CLOSE(got_split[i], got_split_ref[i], 1e-4,
                    "hc_split_weighted_sum: fused/standalone split differential mismatch");

    /* Differential: the fused entry's out output must agree with
     * hc_split_sinkhorn_tensor followed by hc_weighted_sum_split_
     * tensor on the same inputs (split then weighted sum, unfused). */
    ds4_gpu_tensor *tout_ref = ds4_gpu_tensor_alloc(sizeof(want_out));
    CHECK(tout_ref, "hc_split_weighted_sum: ref out alloc failed");
    CHECK(ds4_gpu_hc_weighted_sum_split_tensor(tout_ref, tres, tsplit_ref, N_EMBD, N_HC) != 0,
          "hc_split_weighted_sum: reference weighted sum call");
    float got_out_ref[N_ROWS * N_EMBD];
    CHECK(ds4_gpu_tensor_read(tout_ref, 0, got_out_ref, sizeof(got_out_ref)) != 0,
          "hc_split_weighted_sum: read ref out");
    for (int i = 0; i < N_ROWS * N_EMBD; i++)
        CHECK_CLOSE(got_out[i], got_out_ref[i], 1e-4,
                    "hc_split_weighted_sum: fused/unfused out differential mismatch");

    CHECK(ds4_gpu_hc_split_weighted_sum_tensor(tout, tsplit, tmix, tres, model, sizeof(model),
                                               scale_offset, base_offset, N_EMBD, 3, ITERS, eps) == 0,
          "hc_split_weighted_sum: n_hc != 4 must be rejected");

    ds4_gpu_tensor_free(tmix);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tsplit_ref);
    ds4_gpu_tensor_free(tout_ref);
    fprintf(stderr, "  test_hc_split_weighted_sum OK\n");
    return 0;
}

/* Fuses the split and weighted sum above with a further RMS norm over
 * n_embd, per rocm/ds4_rocm_hc_output_launch.cuh:146-202.  N_EMBD = 37 is
 * deliberately smaller than the kernel's fixed 256-wide work group, so
 * some lanes' per-lane accumulation loop runs zero iterations; per design-
 * spec section 6b those lanes must still contribute their (zero) value to
 * the reduction.  residual_hc's branches are built from fill_val, which is
 * NOT affine in its indices (design-spec section 6f): the branches are not
 * scalar multiples of one another, which is what makes a single-branch
 * weight perturbation change the RESULT'S DIRECTION rather than only its
 * magnitude, and therefore observable through the RMS-normalised output
 * rather than divided out by it. */
static int test_hc_split_weighted_sum_norm(void) {
    enum { N_ROWS = 4, N_EMBD = 37, ITERS = 6 };
    const float eps = 1.0e-6f, norm_eps = 1.0e-5f;
    float mix[N_ROWS * MIX_HC];
    float residual[N_ROWS * N_HC * N_EMBD];
    for (int r = 0; r < N_ROWS; r++) {
        for (int i = 0; i < MIX_HC; i++) mix[r * MIX_HC + i] = fill_val((uint32_t)r, (uint32_t)i + 30) * 1.3f;
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[(r * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.5f;
    }
    float model[3 + MIX_HC];
    model[0] = 1.05f; model[1] = 0.9f; model[2] = 1.1f;
    for (int i = 0; i < MIX_HC; i++) model[3 + i] = fill_val(55, (uint32_t)i) * 0.2f;
    float norm_w[N_EMBD];
    for (int i = 0; i < N_EMBD; i++) norm_w[i] = 0.5f + 0.02f * (float)i;
    const uint64_t scale_offset = 0, base_offset = 3 * sizeof(float);
    const uint64_t norm_weight_offset = base_offset + MIX_HC * sizeof(float);

    float model_buf[3 + MIX_HC + N_EMBD];
    memcpy(model_buf, model, sizeof(model));
    memcpy((char *)model_buf + norm_weight_offset, norm_w, sizeof(norm_w));

    float want_split[N_ROWS * MIX_HC], want_out[N_ROWS * N_EMBD], want_norm[N_ROWS * N_EMBD];
    for (int r = 0; r < N_ROWS; r++) {
        oracle_hc_split_sinkhorn_one(want_split + r * MIX_HC, mix + r * MIX_HC, model, model + 3,
                                     ITERS, eps);
        oracle_hc_weighted_sum_one(want_out + r * N_EMBD, residual + (uint64_t)r * N_HC * N_EMBD,
                                   want_split + r * MIX_HC, N_EMBD, N_HC);
        oracle_rms_norm_weight(want_norm + r * N_EMBD, want_out + r * N_EMBD, norm_w, N_EMBD, norm_eps);
    }

    ds4_gpu_tensor *tmix = alloc_write(mix, sizeof(mix));
    ds4_gpu_tensor *tres = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = ds4_gpu_tensor_alloc(sizeof(want_split));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(want_out));
    ds4_gpu_tensor *tnorm = ds4_gpu_tensor_alloc(sizeof(want_norm));
    CHECK(tmix && tres && tsplit && tout && tnorm, "hc_split_weighted_sum_norm: alloc failed");

    CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(
                  tout, tnorm, tsplit, tmix, tres, model_buf, sizeof(model_buf), scale_offset,
                  base_offset, norm_weight_offset, N_EMBD, N_HC, ITERS, eps, norm_eps) != 0,
          "hc_split_weighted_sum_norm: call");

    float got_out[N_ROWS * N_EMBD], got_norm[N_ROWS * N_EMBD];
    CHECK(ds4_gpu_tensor_read(tout, 0, got_out, sizeof(got_out)) != 0,
          "hc_split_weighted_sum_norm: read out");
    CHECK(ds4_gpu_tensor_read(tnorm, 0, got_norm, sizeof(got_norm)) != 0,
          "hc_split_weighted_sum_norm: read norm");
    for (int i = 0; i < N_ROWS * N_EMBD; i++)
        CHECK_CLOSE(got_out[i], want_out[i], 1e-4, "hc_split_weighted_sum_norm: out mismatch");
    for (int i = 0; i < N_ROWS * N_EMBD; i++)
        CHECK_CLOSE(got_norm[i], want_norm[i], 1e-4, "hc_split_weighted_sum_norm: norm mismatch");

    CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(
                  tout, tnorm, tsplit, tmix, tres, model_buf, sizeof(model_buf), scale_offset,
                  base_offset, norm_weight_offset, N_EMBD, 3, ITERS, eps, norm_eps) == 0,
          "hc_split_weighted_sum_norm: n_hc != 4 must be rejected");

    ds4_gpu_tensor_free(tmix);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tnorm);
    fprintf(stderr, "  test_hc_split_weighted_sum_norm OK\n");
    return 0;
}

/* Large-scale regression case for the fused norm kernel's reduction-barrier
 * ablation.  N_EMBD == 256 == the kernel's fixed work-group width is
 * load-bearing, not just N_ROWS: an earlier attempt at this same ablation
 * used N_EMBD = 64 at up to 131072 rows (8.4M+ work-items) and never failed,
 * because 192 of the 256 lanes in every group then pass the reduction's
 * identity value (0.0f), and combining mostly-zero values out of order is
 * invisible.  That is design-spec section 6i's mechanism (an ablation can
 * pass falsely because of the value you chose), not insufficient
 * contention: holding N_ROWS at 65536 and only changing N_EMBD from 64 to
 * 256 was what made the dropped barrier reproduce wrong output, confirmed
 * by running both shapes.  Kept permanently at this shape, not deleted
 * after the ablation that motivated it, per the sibling barrier audit's
 * own precedent (design-spec section 6j). */
static int test_hc_split_weighted_sum_norm_stress(void) {
    enum { N_ROWS = 65536, N_EMBD = 256, ITERS = 4 };
    const float eps = 1.0e-6f, norm_eps = 1.0e-5f;
    size_t mix_bytes = (size_t)N_ROWS * MIX_HC * sizeof(float);
    size_t res_bytes = (size_t)N_ROWS * N_HC * N_EMBD * sizeof(float);
    size_t out_bytes = (size_t)N_ROWS * N_EMBD * sizeof(float);
    float *mix = malloc(mix_bytes);
    float *residual = malloc(res_bytes);
    float *want_out = malloc(out_bytes);
    float *want_norm = malloc(out_bytes);
    float *got_out = malloc(out_bytes);
    float *got_norm = malloc(out_bytes);
    CHECK(mix && residual && want_out && want_norm && got_out && got_norm,
          "hc_split_weighted_sum_norm(stress): host alloc failed");

    float model[3 + MIX_HC];
    model[0] = 1.02f; model[1] = 0.98f; model[2] = 1.04f;
    for (int i = 0; i < MIX_HC; i++) model[3 + i] = fill_val(33, (uint32_t)i) * 0.2f;
    float norm_w[N_EMBD];
    for (int i = 0; i < N_EMBD; i++) norm_w[i] = 0.6f + 0.01f * (float)i;
    const uint64_t scale_offset = 0, base_offset = 3 * sizeof(float);
    const uint64_t norm_weight_offset = base_offset + MIX_HC * sizeof(float);
    float model_buf[3 + MIX_HC + N_EMBD];
    memcpy(model_buf, model, sizeof(model));
    memcpy((char *)model_buf + norm_weight_offset, norm_w, sizeof(norm_w));

    float split_row[MIX_HC];
    for (int r = 0; r < N_ROWS; r++) {
        for (int i = 0; i < MIX_HC; i++)
            mix[(size_t)r * MIX_HC + i] = fill_val((uint32_t)r, (uint32_t)i + 15) * 1.1f;
        for (int h = 0; h < N_HC; h++)
            for (int i = 0; i < N_EMBD; i++)
                residual[((size_t)r * N_HC + h) * N_EMBD + i] = fill_val((uint32_t)(h * 131 + r), (uint32_t)i) + 1.0f;
        oracle_hc_split_sinkhorn_one(split_row, mix + (size_t)r * MIX_HC, model, model + 3, ITERS, eps);
        oracle_hc_weighted_sum_one(want_out + (size_t)r * N_EMBD,
                                   residual + (size_t)r * N_HC * N_EMBD, split_row, N_EMBD, N_HC);
        oracle_rms_norm_weight(want_norm + (size_t)r * N_EMBD, want_out + (size_t)r * N_EMBD, norm_w,
                               N_EMBD, norm_eps);
    }

    ds4_gpu_tensor *tmix = alloc_write(mix, mix_bytes);
    ds4_gpu_tensor *tres = alloc_write(residual, res_bytes);
    ds4_gpu_tensor *tsplit = ds4_gpu_tensor_alloc((size_t)N_ROWS * MIX_HC * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *tnorm = ds4_gpu_tensor_alloc(out_bytes);
    CHECK(tmix && tres && tsplit && tout && tnorm,
          "hc_split_weighted_sum_norm(stress): device alloc failed");

    CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(
                  tout, tnorm, tsplit, tmix, tres, model_buf, sizeof(model_buf), scale_offset,
                  base_offset, norm_weight_offset, N_EMBD, N_HC, ITERS, eps, norm_eps) != 0,
          "hc_split_weighted_sum_norm(stress): call");
    CHECK(ds4_gpu_tensor_read(tout, 0, got_out, out_bytes) != 0,
          "hc_split_weighted_sum_norm(stress): read out");
    CHECK(ds4_gpu_tensor_read(tnorm, 0, got_norm, out_bytes) != 0,
          "hc_split_weighted_sum_norm(stress): read norm");

    for (size_t i = 0; i < (size_t)N_ROWS * N_EMBD; i++)
        CHECK_CLOSE(got_norm[i], want_norm[i], 2e-3,
                    "hc_split_weighted_sum_norm(stress): norm mismatch");

    ds4_gpu_tensor_free(tmix);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tnorm);
    free(mix); free(residual); free(want_out); free(want_norm); free(got_out); free(got_norm);
    fprintf(stderr, "  test_hc_split_weighted_sum_norm_stress OK\n");
    return 0;
}

/* ds4_gpu_matmul_q8_0_hc_expand_tensor: a Q8_0 dense matmul fused with the
 * HC-expand combine (no addend), matching ROCm's cuda_matmul_q8_0_hc_
 * expand_tensor_labeled (rocm/ds4_rocm_matmul.cuh:690) with block_add ==
 * NULL. IN_DIM deliberately not a multiple of 32, exercising a partial
 * final Q8_0 block, the same reasoning as test_matmul_q8_0 in
 * tests/test_sycl_matmul.c. out_dim == N_EMBD, this fused family's own
 * requirement.
 *
 * Verified two ways: against the composed CPU oracle (Q8_0 dot product,
 * then hc_post_one), and differentially against this backend's own
 * unfused composition, ds4_gpu_matmul_q8_0_tensor followed by
 * ds4_gpu_hc_expand_split_tensor, on identical input -- the stronger test
 * per the plan, since both are real implemented paths rather than one
 * being a hand-transcribed oracle. */
static int test_matmul_q8_0_hc_expand(void) {
    enum { IN_DIM = 37 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)N_HC_EXPAND_EMBD * row_bytes;

    unsigned char weights[N_HC_EXPAND_EMBD * 68]; /* row_bytes <= 2*34 here */
    float x[IN_DIM];
    float block_want[N_HC_EXPAND_EMBD];
    float residual[N_HC * N_HC_EXPAND_EMBD];
    float split[MIX_HC];
    float hc_want[N_HC * N_HC_EXPAND_EMBD];

    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        hc_test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) x[k] = (float)((k * 3u) % 9u) - 4.0f;
    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        block_want[o] = hc_oracle_q8_0_dot(x, weights + (size_t)o * row_bytes, IN_DIM);
    }
    for (int h = 0; h < N_HC; h++) {
        for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
            residual[h * N_HC_EXPAND_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 3.0f;
        }
    }
    for (int i = 0; i < MIX_HC; i++) split[i] = fill_val(9u, (uint32_t)i + 40) * 0.5f;
    oracle_hc_post_one(hc_want, block_want, residual, split + N_HC, split + 2 * N_HC,
                       N_HC_EXPAND_EMBD, N_HC);

    ds4_gpu_tensor *tx     = alloc_write(x, sizeof(x));
    ds4_gpu_tensor *tres   = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tblock = ds4_gpu_tensor_alloc(sizeof(block_want));
    ds4_gpu_tensor *tout   = ds4_gpu_tensor_alloc(sizeof(hc_want));
    CHECK(tx && tres && tsplit && tblock && tout,
          "matmul_q8_0_hc_expand: alloc failed");

    CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
              tout, tblock, weights, weight_bytes, 0, IN_DIM,
              N_HC_EXPAND_EMBD, tx, tres, tsplit, N_HC_EXPAND_EMBD, N_HC) != 0,
          "matmul_q8_0_hc_expand: call");

    float block_got[N_HC_EXPAND_EMBD], hc_got[N_HC * N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tblock, 0, block_got, sizeof(block_got)) != 0,
          "matmul_q8_0_hc_expand: read block_out");
    CHECK(ds4_gpu_tensor_read(tout, 0, hc_got, sizeof(hc_got)) != 0,
          "matmul_q8_0_hc_expand: read out_hc");
    for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(block_got[i], block_want[i], 1e-2,
                    "matmul_q8_0_hc_expand: block_out mismatch");
    }
    for (int i = 0; i < N_HC * N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(hc_got[i], hc_want[i], 1e-2,
                    "matmul_q8_0_hc_expand: out_hc mismatch");
    }

    /* Differential: matmul_q8_0 then hc_expand_split, on the same weights,
     * x, residual and split. */
    ds4_gpu_tensor *tblock2 = ds4_gpu_tensor_alloc(sizeof(block_want));
    ds4_gpu_tensor *tout2   = ds4_gpu_tensor_alloc(sizeof(hc_want));
    CHECK(tblock2 && tout2, "matmul_q8_0_hc_expand: differential alloc failed");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tblock2, weights, weight_bytes, 0,
                                     IN_DIM, N_HC_EXPAND_EMBD, tx, 1) != 0,
          "matmul_q8_0_hc_expand: differential matmul call");
    CHECK(ds4_gpu_hc_expand_split_tensor(tout2, tblock2, tres, tsplit,
                                         N_HC_EXPAND_EMBD, N_HC) != 0,
          "matmul_q8_0_hc_expand: differential expand call");
    float hc_diff[N_HC * N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tout2, 0, hc_diff, sizeof(hc_diff)) != 0,
          "matmul_q8_0_hc_expand: differential read");
    for (int i = 0; i < N_HC * N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(hc_got[i], hc_diff[i], 1e-4,
                    "matmul_q8_0_hc_expand: fused vs unfused composition "
                    "mismatch");
    }

    /* Validation: out_dim must equal n_embd; a zero in_dim, an out-of-range
     * weight offset, and an undersized out_hc must all be rejected. */
    CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
              tout, tblock, weights, weight_bytes, 0, IN_DIM,
              N_HC_EXPAND_EMBD + 1u, tx, tres, tsplit, N_HC_EXPAND_EMBD,
              N_HC) == 0,
          "matmul_q8_0_hc_expand: out_dim != n_embd must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
              tout, tblock, weights, weight_bytes, 0, 0, N_HC_EXPAND_EMBD,
              tx, tres, tsplit, N_HC_EXPAND_EMBD, N_HC) == 0,
          "matmul_q8_0_hc_expand: zero in_dim must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
              tout, tblock, weights, weight_bytes, weight_bytes, IN_DIM,
              N_HC_EXPAND_EMBD, tx, tres, tsplit, N_HC_EXPAND_EMBD,
              N_HC) == 0,
          "matmul_q8_0_hc_expand: out-of-range weight offset must be "
          "rejected");
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 2);
    CHECK(small != NULL, "matmul_q8_0_hc_expand: small allocation failed");
    CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
              small, tblock, weights, weight_bytes, 0, IN_DIM,
              N_HC_EXPAND_EMBD, tx, tres, tsplit, N_HC_EXPAND_EMBD,
              N_HC) == 0,
          "matmul_q8_0_hc_expand: undersized out_hc must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tblock2);
    ds4_gpu_tensor_free(tout2);
    fprintf(stderr, "  test_matmul_q8_0_hc_expand OK\n");
    return 0;
}

/* As hc_oracle_q8_0_dot above, but restricted to [in_start, in_start+
 * in_count) of the row addressed in the FULL row's block layout -- the
 * k-slice matmul's own dequant contract. */
static float hc_oracle_q8_0_dot_kslice(const float *x_slice, const unsigned char *row,
                                       uint32_t in_start, uint32_t in_count) {
    double sum = 0.0;
    for (uint32_t k = 0; k < in_count; k++) {
        const uint32_t col = in_start + k;
        const uint32_t blk = col / 32u;
        const uint32_t idx = col % 32u;
        const unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
        const float scale = oracle_half_to_float(raw);
        const signed char qv = (signed char)bp[2 + idx];
        sum += (double)x_slice[k] * (double)scale * (double)qv;
    }
    return (float)sum;
}

/* Tensor parallelism: ds4_gpu_matmul_q8_0_kslice_hc_expand_add_
 * tensor, the fused split-matmul-plus-combine built by composing
 * ds4_gpu_matmul_q8_0_kslice_rows_tensor (ds4_sycl_matmul.hpp) with
 * ds4_gpu_hc_expand_add_split_tensor (above, already implemented
 * separately). FULL_IN_DIM = 64, split clean in half at IN_START = 32 so
 * IN_COUNT = 32 is exactly this rank's usual half; the k-slice contract
 * itself is already exercised at an uneven split by
 * test_matmul_q8_0_kslice_rows in test_sycl_matmul.c, so this test's job
 * is proving the FUSION composes correctly, not re-proving the slice
 * arithmetic. */
static int test_matmul_q8_0_kslice_hc_expand_add(void) {
    enum { FULL_IN_DIM = 64, IN_START = 32, IN_COUNT = 32 };
    const uint64_t full_blocks = (FULL_IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = full_blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)N_HC_EXPAND_EMBD * row_bytes;

    unsigned char weights[N_HC_EXPAND_EMBD * 68]; /* row_bytes == 2*34 = 68 here */
    float x_slice[IN_COUNT];
    float block_add[N_HC_EXPAND_EMBD];
    float block_want[N_HC_EXPAND_EMBD];
    float residual[N_HC * N_HC_EXPAND_EMBD];
    float split[MIX_HC];
    float hc_want[N_HC * N_HC_EXPAND_EMBD];

    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        hc_test_encode_q8_0_row(weights + (size_t)o * row_bytes, FULL_IN_DIM, o);
    }
    for (uint32_t k = 0; k < IN_COUNT; k++) x_slice[k] = (float)((k * 3u) % 9u) - 4.0f;
    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        block_want[o] = hc_oracle_q8_0_dot_kslice(x_slice, weights + (size_t)o * row_bytes,
                                                  IN_START, IN_COUNT);
        block_add[o] = fill_val(o, 77u) * 0.6f;
    }
    for (int h = 0; h < N_HC; h++) {
        for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
            residual[h * N_HC_EXPAND_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 3.0f;
        }
    }
    for (int i = 0; i < MIX_HC; i++) split[i] = fill_val(11u, (uint32_t)i + 40) * 0.5f;
    float block_sum[N_HC_EXPAND_EMBD];
    for (int i = 0; i < N_HC_EXPAND_EMBD; i++) block_sum[i] = block_want[i] + block_add[i];
    oracle_hc_post_one(hc_want, block_sum, residual, split + N_HC, split + 2 * N_HC,
                       N_HC_EXPAND_EMBD, N_HC);

    ds4_gpu_tensor *tx     = alloc_write(x_slice, sizeof(x_slice));
    ds4_gpu_tensor *tadd   = alloc_write(block_add, sizeof(block_add));
    ds4_gpu_tensor *tres   = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tblock = ds4_gpu_tensor_alloc(sizeof(block_want));
    ds4_gpu_tensor *tout   = ds4_gpu_tensor_alloc(sizeof(hc_want));
    CHECK(tx && tadd && tres && tsplit && tblock && tout,
          "matmul_q8_0_kslice_hc_expand_add: alloc failed");

    CHECK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
              tout, tblock, weights, weight_bytes, 0, FULL_IN_DIM,
              N_HC_EXPAND_EMBD, IN_START, IN_COUNT, tx, tadd, tres, tsplit,
              N_HC_EXPAND_EMBD, N_HC) != 0,
          "matmul_q8_0_kslice_hc_expand_add: call");

    float block_got[N_HC_EXPAND_EMBD], hc_got[N_HC * N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tblock, 0, block_got, sizeof(block_got)) != 0,
          "matmul_q8_0_kslice_hc_expand_add: read block_out");
    CHECK(ds4_gpu_tensor_read(tout, 0, hc_got, sizeof(hc_got)) != 0,
          "matmul_q8_0_kslice_hc_expand_add: read out_hc");
    for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(block_got[i], block_want[i], 1e-2,
                    "matmul_q8_0_kslice_hc_expand_add: block_out mismatch");
    }
    for (int i = 0; i < N_HC * N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(hc_got[i], hc_want[i], 1e-2,
                    "matmul_q8_0_kslice_hc_expand_add: out_hc mismatch");
    }

    /* Validation: out_dim must equal n_embd; unaligned in_start/in_count
     * must be rejected (ported from ds4_cuda.cu:14939-14943). */
    CHECK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
              tout, tblock, weights, weight_bytes, 0, FULL_IN_DIM,
              N_HC_EXPAND_EMBD + 1u, IN_START, IN_COUNT, tx, tadd, tres,
              tsplit, N_HC_EXPAND_EMBD, N_HC) == 0,
          "matmul_q8_0_kslice_hc_expand_add: out_dim != n_embd must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
              tout, tblock, weights, weight_bytes, 0, FULL_IN_DIM,
              N_HC_EXPAND_EMBD, IN_START + 1u, IN_COUNT, tx, tadd, tres,
              tsplit, N_HC_EXPAND_EMBD, N_HC) == 0,
          "matmul_q8_0_kslice_hc_expand_add: unaligned in_start must be rejected");
    CHECK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
              NULL, tblock, weights, weight_bytes, 0, FULL_IN_DIM,
              N_HC_EXPAND_EMBD, IN_START, IN_COUNT, tx, tadd, tres, tsplit,
              N_HC_EXPAND_EMBD, N_HC) == 0,
          "matmul_q8_0_kslice_hc_expand_add: null out_hc must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tadd);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tblock);
    ds4_gpu_tensor_free(tout);
    fprintf(stderr, "  test_matmul_q8_0_kslice_hc_expand_add OK\n");
    return 0;
}

/* ds4_gpu_shared_down_hc_expand_q8_0_tensor: the same fused family with the
 * routed-expert sum added into the matmul result BEFORE the HC-expand
 * combine (ROCm's has_add branch).  Verified against the composed oracle
 * and, per the plan, differentially against ds4_gpu_matmul_q8_0_tensor
 * followed by ds4_gpu_hc_expand_add_split_tensor (the unfused add variant,
 * not the plain one used above). */
static int test_shared_down_hc_expand_q8_0(void) {
    enum { IN_DIM = 20 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)N_HC_EXPAND_EMBD * row_bytes;

    unsigned char weights[N_HC_EXPAND_EMBD * 34]; /* row_bytes == 34 here */
    float x[IN_DIM];
    float routed_out[N_HC_EXPAND_EMBD];
    float raw_want[N_HC_EXPAND_EMBD];
    float block_want[N_HC_EXPAND_EMBD];
    float residual[N_HC * N_HC_EXPAND_EMBD];
    float split[MIX_HC];
    float hc_want[N_HC * N_HC_EXPAND_EMBD];

    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        hc_test_encode_q8_0_row(weights + (size_t)o * row_bytes, IN_DIM, o + 1u);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) x[k] = (float)((k * 5u + 1u) % 7u) - 3.0f;
    for (uint32_t o = 0; o < N_HC_EXPAND_EMBD; o++) {
        routed_out[o] = fill_val(o, o + 2u) * 1.3f;
        const float dot = hc_oracle_q8_0_dot(x, weights + (size_t)o * row_bytes, IN_DIM);
        raw_want[o] = dot;
        block_want[o] = dot + routed_out[o];
    }
    for (int h = 0; h < N_HC; h++) {
        for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
            residual[h * N_HC_EXPAND_EMBD + i] = fill_val((uint32_t)h, (uint32_t)i) + 1.7f;
        }
    }
    for (int i = 0; i < MIX_HC; i++) split[i] = fill_val(3u, (uint32_t)i + 60) * 0.4f;
    oracle_hc_post_one(hc_want, block_want, residual, split + N_HC, split + 2 * N_HC,
                       N_HC_EXPAND_EMBD, N_HC);

    ds4_gpu_tensor *tx      = alloc_write(x, sizeof(x));
    ds4_gpu_tensor *trouted = alloc_write(routed_out, sizeof(routed_out));
    ds4_gpu_tensor *tres    = alloc_write(residual, sizeof(residual));
    ds4_gpu_tensor *tsplit  = alloc_write(split, sizeof(split));
    ds4_gpu_tensor *tshared = ds4_gpu_tensor_alloc(sizeof(block_want));
    ds4_gpu_tensor *tout    = ds4_gpu_tensor_alloc(sizeof(hc_want));
    CHECK(tx && trouted && tres && tsplit && tshared && tout,
          "shared_down_hc_expand: alloc failed");

    CHECK(ds4_gpu_shared_down_hc_expand_q8_0_tensor(
              tout, tshared, weights, weight_bytes, 0, IN_DIM,
              N_HC_EXPAND_EMBD, tx, trouted, tres, tsplit,
              N_HC_EXPAND_EMBD, N_HC) != 0,
          "shared_down_hc_expand: call");

    float hc_got[N_HC * N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tout, 0, hc_got, sizeof(hc_got)) != 0,
          "shared_down_hc_expand: read out_hc");
    for (int i = 0; i < N_HC * N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(hc_got[i], hc_want[i], 1e-2,
                    "shared_down_hc_expand: out_hc mismatch");
    }

    /* shared_out (block_out) must hold the RAW pre-add matmul result, not
     * the value after routed_out is folded in: ds4_sycl_hc.hpp's own
     * comment on sycl_matmul_q8_0_hc_expand_labeled states block_out is
     * always written before the addend is applied, matching ROCm's
     * `block_out[d] = acc;` running before `block_v` picks up the addend.
     * Nothing above exercises this: hc_got only depends on the POST-add
     * value, and the differential path below writes its own separate
     * tensor, so a kernel that wrote the post-add value into shared_out
     * would pass every other check in this test. Found by ablation. */
    float shared_got[N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tshared, 0, shared_got, sizeof(shared_got)) != 0,
          "shared_down_hc_expand: read shared_out");
    for (int i = 0; i < N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(shared_got[i], raw_want[i], 1e-2,
                    "shared_down_hc_expand: shared_out must be the raw "
                    "pre-add matmul result");
    }

    /* Differential: matmul_q8_0 into a scratch buffer, then the unfused
     * add-variant expand entry, on the same inputs. */
    ds4_gpu_tensor *tshared2 = ds4_gpu_tensor_alloc(sizeof(block_want));
    ds4_gpu_tensor *tout2    = ds4_gpu_tensor_alloc(sizeof(hc_want));
    CHECK(tshared2 && tout2, "shared_down_hc_expand: differential alloc failed");
    CHECK(ds4_gpu_matmul_q8_0_tensor(tshared2, weights, weight_bytes, 0,
                                     IN_DIM, N_HC_EXPAND_EMBD, tx, 1) != 0,
          "shared_down_hc_expand: differential matmul call");
    CHECK(ds4_gpu_hc_expand_add_split_tensor(tout2, tshared2, trouted, tres,
                                             tsplit, N_HC_EXPAND_EMBD,
                                             N_HC) != 0,
          "shared_down_hc_expand: differential expand call");
    float hc_diff[N_HC * N_HC_EXPAND_EMBD];
    CHECK(ds4_gpu_tensor_read(tout2, 0, hc_diff, sizeof(hc_diff)) != 0,
          "shared_down_hc_expand: differential read");
    for (int i = 0; i < N_HC * N_HC_EXPAND_EMBD; i++) {
        CHECK_CLOSE(hc_got[i], hc_diff[i], 1e-4,
                    "shared_down_hc_expand: fused vs unfused composition "
                    "mismatch");
    }

    /* Validation: same shape as the plain entry's checks above. */
    CHECK(ds4_gpu_shared_down_hc_expand_q8_0_tensor(
              tout, tshared, weights, weight_bytes, 0, IN_DIM,
              N_HC_EXPAND_EMBD + 1u, tx, trouted, tres, tsplit,
              N_HC_EXPAND_EMBD, N_HC) == 0,
          "shared_down_hc_expand: out_dim != n_embd must be rejected");
    CHECK(ds4_gpu_shared_down_hc_expand_q8_0_tensor(
              tout, tshared, weights, weight_bytes, weight_bytes, IN_DIM,
              N_HC_EXPAND_EMBD, tx, trouted, tres, tsplit,
              N_HC_EXPAND_EMBD, N_HC) == 0,
          "shared_down_hc_expand: out-of-range weight offset must be "
          "rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(trouted);
    ds4_gpu_tensor_free(tres);
    ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tshared);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tshared2);
    ds4_gpu_tensor_free(tout2);
    fprintf(stderr, "  test_shared_down_hc_expand_q8_0 OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_repeat_hc() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_repeat_hc_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_sinkhorn() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum_split() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_add() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_n4_and_differential() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_general_fallback() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_half() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_add_split() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_add_split_half_add() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_weighted_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_weighted_sum_norm() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_weighted_sum_norm_stress() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_hc_expand() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_matmul_q8_0_kslice_hc_expand_add() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_shared_down_hc_expand_q8_0() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_hc OK\n");
    return 0;
}
