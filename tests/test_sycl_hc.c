/* Correctness tests for the SYCL hyper-connection (HC) subsystem, validated
 * against scalar CPU oracles implemented here.  The ds4.c CPU references
 * are static and cannot be linked, so each oracle reimplements the
 * documented formula with the ds4.c line numbers cited.  Needs no model
 * file: the "model map" used to exercise mmap staging is a plain host
 * float buffer standing in for one.
 *
 * This is a self-contained file per this project's convention: it defines
 * its own CHECK/CHECK_CLOSE macros rather than sharing test_sycl_harness.h,
 * and does not touch any other test file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
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

enum { N_HC = 4, MIX_HC = 2 * N_HC + N_HC * N_HC /* 24 */ };

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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_repeat_hc() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_repeat_hc_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_sinkhorn() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum_split() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_n4_and_differential() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_general_fallback() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_split_half() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_add_split() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_expand_add_split_half_add() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_hc OK\n");
    return 0;
}
