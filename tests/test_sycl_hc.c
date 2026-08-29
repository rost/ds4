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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_repeat_hc() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_repeat_hc_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_split_sinkhorn() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_hc_weighted_sum_split() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_hc OK\n");
    return 0;
}
