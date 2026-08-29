/* Correctness tests for sycl/ds4_sycl_indexer.hpp.
 *
 * Covers scoring (ds4_gpu_indexer_score_one_tensor,
 * ds4_gpu_indexer_scores_prefill_tensor,
 * ds4_gpu_indexer_scores_decode_batch_tensor), vocabulary argmax
 * (ds4_gpu_argmax_tensor), the top-k-to-mask conversion
 * (ds4_gpu_dsv4_topk_mask_tensor), and the Hadamard+FP4 QAT round trip
 * (ds4_gpu_dsv4_indexer_qat_tensor), followed further below by the top-k
 * subsystem's tests (ds4_gpu_indexer_topk_tensor and the internal
 * indexed_topk_sort_512_asc_kernel).
 *
 * Oracles are transcribed directly from ds4.c:
 *   - dsv4_e2m1fn_value_cpu / dsv4_e2m1fn_dequant_cpu    (ds4.c:3260-3279)
 *   - dsv4_hadamard128_inplace_cpu                        (ds4.c:3282-3294)
 *   - dsv4_fp4_act_quantize_row_inplace_cpu               (ds4.c:3295-3312)
 *   - indexer_allowed_decode_one's inline scoring math     (ds4.c:12936-12980),
 *     which mirrors indexer_scores_kernel's per-head ReLU'd weighted dot
 *     product exactly, and its top-k selection loop, which breaks ties by
 *     keeping the earlier (lower-index) candidate via a strict `>` scan.
 *   - sample_argmax                                        (ds4.c:38282-38292)
 *
 * Self-contained, like tests/test_sycl_fp8_kv.c: does not use
 * tests/test_sycl_harness.h, since none of its shared oracles (RMS norm,
 * RoPE) apply here.  Needs no model file: every entry in this file takes
 * only already-resident device tensors. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test-only hooks defined directly in sycl/ds4_sycl_indexer.hpp. */
extern float ds4_sycl_test_e2m1fn_value(int code);
extern float ds4_sycl_test_e2m1fn_dequant(float x);

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                      \
        }                                                                   \
    } while (0)

#define CHECK_CLOSE(got, want, tol, msg)                                    \
    do {                                                                    \
        double d_ = fabs((double)(got) - (double)(want));                   \
        if (!(d_ <= (tol))) {                                               \
            fprintf(stderr, "FAIL: %s (got %.9g want %.9g delta %.3g)\n",   \
                    (msg), (double)(got), (double)(want), d_);              \
            return 1;                                                      \
        }                                                                   \
    } while (0)

/* ---- CPU oracles, transcribed from ds4.c --------------------------- */

static float oracle_e2m1fn_value(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

static float oracle_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - oracle_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - oracle_e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * oracle_e2m1fn_value(best);
}

static void oracle_hadamard128_inplace(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

static void oracle_fp4_act_quantize_row_inplace(float *x, uint32_t n) {
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = oracle_e2m1fn_dequant(v) * scale;
        }
    }
}

static void oracle_indexer_qat_row_inplace(float *x, uint32_t head_dim) {
    oracle_hadamard128_inplace(x);
    oracle_fp4_act_quantize_row_inplace(x, head_dim);
}

/* Mirrors indexer_scores_kernel's math (rocm/ds4_rocm_indexer.cuh:41-77)
 * and indexer_allowed_decode_one's inline scoring (ds4.c:12966-12975):
 * per-head ReLU'd dot product, weighted-summed, scaled.  Causal masking
 * follows indexer_scores_launch's own visibility formula. */
static void oracle_indexer_scores(float *scores, const float *q, const float *weights,
                                  const float *index_comp, uint32_t n_comp,
                                  uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
                                  uint32_t head_dim, uint32_t ratio, float scale,
                                  int causal) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t n_visible = n_comp;
        if (causal) n_visible = (pos0 + t + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (causal && c >= n_visible) {
                scores[(uint64_t)t * n_comp + c] = -INFINITY;
                continue;
            }
            float total = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
                const float *kh = index_comp + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                if (dot < 0.0f) dot = 0.0f;
                total += dot * weights[(uint64_t)t * n_head + h];
            }
            scores[(uint64_t)t * n_comp + c] = total * scale;
        }
    }
}

/* Mirrors sample_argmax (ds4.c:38282-38292): strict `>` scan, so the
 * first / lowest-index maximum wins any tie. */
static int32_t oracle_argmax(const float *logits, uint32_t n_vocab) {
    int32_t best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (int32_t)i;
        }
    }
    return best;
}

/* ---- QAT tests ------------------------------------------------------ */

static int test_qat_value_table_matches_oracle(void) {
    for (int i = 0; i < 8; i++) {
        CHECK_CLOSE(ds4_sycl_test_e2m1fn_value(i), oracle_e2m1fn_value(i), 0.0,
                    "e2m1fn_value mismatch");
    }
    fprintf(stderr, "  test_qat_value_table_matches_oracle OK\n");
    return 0;
}

static int test_qat_dequant_matches_oracle(void) {
    /* Includes exact ties (0.25 between 0.0 and 0.5, 5.0 between 4.0 and
     * 6.0) to exercise the even-code tie-break, plus ordinary interior
     * and saturating values. */
    const float xs[] = {0.0f, 0.25f, 0.49f, 0.9f, 1.75f, 2.6f, 5.0f, 6.5f,
                        -0.25f, -3.2f, -7.0f};
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        CHECK_CLOSE(ds4_sycl_test_e2m1fn_dequant(xs[i]), oracle_e2m1fn_dequant(xs[i]),
                    0.0, "e2m1fn_dequant mismatch");
    }
    fprintf(stderr, "  test_qat_dequant_matches_oracle OK\n");
    return 0;
}

/* Hashed-dither test data (not a clean arithmetic progression, per spec
 * 6f/6n) across several independent 128-wide rows. */
static void fill_qat_row(float *row, uint32_t seed) {
    for (uint32_t i = 0; i < 128; i++) {
        uint32_t h = (i * 2654435761u + seed * 40503u) ^ (i << 3);
        float v = (float)((int32_t)(h % 4001) - 2000) * 0.01f;
        row[i] = v;
    }
}

static int test_qat_multi_row_matches_oracle(void) {
    enum { N_ROWS = 5, HEAD_DIM = 128 };
    float x[N_ROWS * HEAD_DIM];
    float want[N_ROWS * HEAD_DIM];
    for (int r = 0; r < N_ROWS; r++) fill_qat_row(&x[r * HEAD_DIM], (uint32_t)r * 97u + 3u);
    memcpy(want, x, sizeof(x));
    for (int r = 0; r < N_ROWS; r++) oracle_indexer_qat_row_inplace(&want[r * HEAD_DIM], HEAD_DIM);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "qat_multi_row: alloc");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "qat_multi_row: write");
    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, N_ROWS, HEAD_DIM) != 0, "qat_multi_row: call");
    float got[N_ROWS * HEAD_DIM];
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "qat_multi_row: read");
    for (int i = 0; i < N_ROWS * HEAD_DIM; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "qat_multi_row: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 1.0e-3, msg);
    }
    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_qat_multi_row_matches_oracle OK\n");
    return 0;
}

static int test_qat_rejections(void) {
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(128 * sizeof(float));
    CHECK(tx != NULL, "qat_rejections: alloc");
    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(NULL, 1, 128) == 0, "qat_rejections: null tensor");
    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, 0, 128) == 0, "qat_rejections: zero rows");
    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, 1, 64) == 0, "qat_rejections: wrong head_dim");
    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, 2, 128) == 0, "qat_rejections: undersized tensor");
    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_qat_rejections OK\n");
    return 0;
}

/* ---- Scoring tests ---------------------------------------------------- */

static void fill_scores_data(float *q, float *weights, float *index_comp,
                             uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
                             uint32_t n_comp, uint32_t seed) {
    for (uint64_t i = 0; i < (uint64_t)n_tokens * n_head * head_dim; i++) {
        uint32_t h = (uint32_t)(i * 2654435761u + seed * 12345u) ^ (uint32_t)(i << 5);
        q[i] = (float)((int32_t)(h % 2001) - 1000) * 0.003f;
    }
    for (uint64_t i = 0; i < (uint64_t)n_tokens * n_head; i++) {
        uint32_t h = (uint32_t)(i * 40503u + seed * 777u + 11u);
        weights[i] = (float)((int32_t)(h % 401) - 200) * 0.005f + 0.5f;
    }
    for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
        uint32_t h = (uint32_t)(i * 2246822519u + seed * 99u) ^ (uint32_t)(i >> 2);
        index_comp[i] = (float)((int32_t)(h % 2001) - 1000) * 0.003f;
    }
}

static int test_scores_prefill_causal_matches_oracle(void) {
    enum { N_TOKENS = 5, N_HEAD = 3, HEAD_DIM = 16, N_COMP = 9, RATIO = 4 };
    float q[N_TOKENS * N_HEAD * HEAD_DIM];
    float weights[N_TOKENS * N_HEAD];
    float index_comp[N_COMP * HEAD_DIM];
    fill_scores_data(q, weights, index_comp, N_TOKENS, N_HEAD, HEAD_DIM, N_COMP, 7);
    const float scale = 0.125f;

    float want[N_TOKENS * N_COMP];
    oracle_indexer_scores(want, q, weights, index_comp, N_COMP, N_TOKENS, 0, N_HEAD,
                          HEAD_DIM, RATIO, scale, 1);

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *tic = ds4_gpu_tensor_alloc(sizeof(index_comp));
    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc(sizeof(want));
    CHECK(tq && tw && tic && tscores, "scores_prefill: alloc");
    CHECK(ds4_gpu_tensor_write(tq, 0, q, sizeof(q)) != 0, "scores_prefill: write q");
    CHECK(ds4_gpu_tensor_write(tw, 0, weights, sizeof(weights)) != 0, "scores_prefill: write w");
    CHECK(ds4_gpu_tensor_write(tic, 0, index_comp, sizeof(index_comp)) != 0,
          "scores_prefill: write index_comp");

    CHECK(ds4_gpu_indexer_scores_prefill_tensor(tscores, tq, tw, tic, N_COMP, N_TOKENS,
                                                N_HEAD, HEAD_DIM, RATIO, scale) != 0,
          "scores_prefill: call");
    float got[N_TOKENS * N_COMP];
    CHECK(ds4_gpu_tensor_read(tscores, 0, got, sizeof(got)) != 0, "scores_prefill: read");
    for (int i = 0; i < N_TOKENS * N_COMP; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "scores_prefill: mismatch at %d", i);
        if (isinf(want[i])) {
            CHECK(isinf(got[i]) && got[i] < 0.0f, msg);
        } else {
            CHECK_CLOSE(got[i], want[i], 5.0e-3, msg);
        }
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tic);
    ds4_gpu_tensor_free(tscores);
    fprintf(stderr, "  test_scores_prefill_causal_matches_oracle OK\n");
    return 0;
}

static int test_scores_decode_batch_matches_oracle(void) {
    enum { N_TOKENS = 4, N_HEAD = 2, HEAD_DIM = 8, N_COMP = 6, RATIO = 4, POS0 = 20 };
    float q[N_TOKENS * N_HEAD * HEAD_DIM];
    float weights[N_TOKENS * N_HEAD];
    float index_comp[N_COMP * HEAD_DIM];
    fill_scores_data(q, weights, index_comp, N_TOKENS, N_HEAD, HEAD_DIM, N_COMP, 19);
    const float scale = 0.2f;

    float want[N_TOKENS * N_COMP];
    oracle_indexer_scores(want, q, weights, index_comp, N_COMP, N_TOKENS, POS0, N_HEAD,
                          HEAD_DIM, RATIO, scale, 1);

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *tic = ds4_gpu_tensor_alloc(sizeof(index_comp));
    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc(sizeof(want));
    CHECK(tq && tw && tic && tscores, "scores_decode_batch: alloc");
    CHECK(ds4_gpu_tensor_write(tq, 0, q, sizeof(q)) != 0, "scores_decode_batch: write q");
    CHECK(ds4_gpu_tensor_write(tw, 0, weights, sizeof(weights)) != 0, "scores_decode_batch: write w");
    CHECK(ds4_gpu_tensor_write(tic, 0, index_comp, sizeof(index_comp)) != 0,
          "scores_decode_batch: write index_comp");

    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(tscores, tq, tw, tic, N_COMP, N_TOKENS,
                                                     POS0, N_HEAD, HEAD_DIM, RATIO, scale) != 0,
          "scores_decode_batch: call");
    float got[N_TOKENS * N_COMP];
    CHECK(ds4_gpu_tensor_read(tscores, 0, got, sizeof(got)) != 0, "scores_decode_batch: read");
    for (int i = 0; i < N_TOKENS * N_COMP; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "scores_decode_batch: mismatch at %d", i);
        if (isinf(want[i])) {
            CHECK(isinf(got[i]) && got[i] < 0.0f, msg);
        } else {
            CHECK_CLOSE(got[i], want[i], 5.0e-3, msg);
        }
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tic);
    ds4_gpu_tensor_free(tscores);
    fprintf(stderr, "  test_scores_decode_batch_matches_oracle OK\n");
    return 0;
}

/* Exercises the decode-direct kernel specifically: n_tokens == 1,
 * head_dim == 128, n_head == 64, exactly DS4 Flash's indexer shape
 * (ds4.c:562-563). */
static int test_score_one_direct_path_matches_oracle(void) {
    enum { N_HEAD = 64, HEAD_DIM = 128, N_COMP = 37 };
    static float q[N_HEAD * HEAD_DIM];
    static float weights[N_HEAD];
    static float index_comp[N_COMP * HEAD_DIM];
    fill_scores_data(q, weights, index_comp, 1, N_HEAD, HEAD_DIM, N_COMP, 31);
    const float scale = 1.0f / 32.0f;

    static float want[N_COMP];
    oracle_indexer_scores(want, q, weights, index_comp, N_COMP, 1, 0, N_HEAD, HEAD_DIM,
                          1, scale, 0);

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *tic = ds4_gpu_tensor_alloc(sizeof(index_comp));
    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc(sizeof(want));
    CHECK(tq && tw && tic && tscores, "score_one_direct: alloc");
    CHECK(ds4_gpu_tensor_write(tq, 0, q, sizeof(q)) != 0, "score_one_direct: write q");
    CHECK(ds4_gpu_tensor_write(tw, 0, weights, sizeof(weights)) != 0, "score_one_direct: write w");
    CHECK(ds4_gpu_tensor_write(tic, 0, index_comp, sizeof(index_comp)) != 0,
          "score_one_direct: write index_comp");

    CHECK(ds4_gpu_indexer_score_one_tensor(tscores, tq, tw, tic, N_COMP, N_HEAD, HEAD_DIM,
                                           scale) != 0,
          "score_one_direct: call");
    float got[N_COMP];
    CHECK(ds4_gpu_tensor_read(tscores, 0, got, sizeof(got)) != 0, "score_one_direct: read");
    for (int i = 0; i < N_COMP; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "score_one_direct: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 5.0e-3, msg);
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tic);
    ds4_gpu_tensor_free(tscores);
    fprintf(stderr, "  test_score_one_direct_path_matches_oracle OK\n");
    return 0;
}

/* Exercises the generic scalar kernel's decode path (n_tokens == 1 but a
 * shape that does not match the direct kernel's fixed 64x128 shape). */
static int test_score_one_generic_path_matches_oracle(void) {
    enum { N_HEAD = 5, HEAD_DIM = 12, N_COMP = 8 };
    float q[N_HEAD * HEAD_DIM];
    float weights[N_HEAD];
    float index_comp[N_COMP * HEAD_DIM];
    fill_scores_data(q, weights, index_comp, 1, N_HEAD, HEAD_DIM, N_COMP, 5);
    const float scale = 0.3f;

    float want[N_COMP];
    oracle_indexer_scores(want, q, weights, index_comp, N_COMP, 1, 0, N_HEAD, HEAD_DIM,
                          1, scale, 0);

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *tic = ds4_gpu_tensor_alloc(sizeof(index_comp));
    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc(sizeof(want));
    CHECK(tq && tw && tic && tscores, "score_one_generic: alloc");
    CHECK(ds4_gpu_tensor_write(tq, 0, q, sizeof(q)) != 0, "score_one_generic: write q");
    CHECK(ds4_gpu_tensor_write(tw, 0, weights, sizeof(weights)) != 0, "score_one_generic: write w");
    CHECK(ds4_gpu_tensor_write(tic, 0, index_comp, sizeof(index_comp)) != 0,
          "score_one_generic: write index_comp");

    CHECK(ds4_gpu_indexer_score_one_tensor(tscores, tq, tw, tic, N_COMP, N_HEAD, HEAD_DIM,
                                           scale) != 0,
          "score_one_generic: call");
    float got[N_COMP];
    CHECK(ds4_gpu_tensor_read(tscores, 0, got, sizeof(got)) != 0, "score_one_generic: read");
    for (int i = 0; i < N_COMP; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "score_one_generic: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 5.0e-3, msg);
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tic);
    ds4_gpu_tensor_free(tscores);
    fprintf(stderr, "  test_score_one_generic_path_matches_oracle OK\n");
    return 0;
}

static int test_scores_rejections(void) {
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(64 * sizeof(float));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(4 * sizeof(float));
    ds4_gpu_tensor *tic = ds4_gpu_tensor_alloc(64 * sizeof(float));
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc(16 * sizeof(float));
    CHECK(tq && tw && tic && ts, "scores_rejections: alloc");
    CHECK(ds4_gpu_indexer_scores_prefill_tensor(ts, tq, tw, tic, 0, 1, 1, 8, 4, 1.0f) == 0,
          "scores_rejections: zero n_comp");
    CHECK(ds4_gpu_indexer_scores_prefill_tensor(ts, tq, tw, tic, 2, 1, 1, 8, 0, 1.0f) == 0,
          "scores_rejections: causal with zero ratio");
    CHECK(ds4_gpu_indexer_scores_prefill_tensor(NULL, tq, tw, tic, 2, 1, 1, 8, 4, 1.0f) == 0,
          "scores_rejections: null scores tensor");
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(tic);
    ds4_gpu_tensor_free(ts);
    fprintf(stderr, "  test_scores_rejections OK\n");
    return 0;
}

/* ---- Argmax tests ------------------------------------------------------ */

static int run_argmax_case(const char *name, uint32_t n_vocab, uint32_t seed,
                           int has_tie, uint32_t tie_a, uint32_t tie_b, float tie_val) {
    float *logits = (float *)malloc((size_t)n_vocab * sizeof(float));
    CHECK(logits != NULL, "argmax: host alloc");
    for (uint32_t i = 0; i < n_vocab; i++) {
        uint32_t h = (i * 2654435761u + seed * 97u) ^ (i << 4);
        logits[i] = (float)((int32_t)(h % 20001) - 10000) * 0.0001f;
    }
    if (has_tie) {
        logits[tie_a] = tie_val;
        logits[tie_b] = tie_val;
    }
    int32_t want = oracle_argmax(logits, n_vocab);

    ds4_gpu_tensor *tl = ds4_gpu_tensor_alloc((uint64_t)n_vocab * sizeof(float));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(int32_t));
    CHECK(tl && to, "argmax: alloc");
    CHECK(ds4_gpu_tensor_write(tl, 0, logits, (uint64_t)n_vocab * sizeof(float)) != 0,
          "argmax: write");
    CHECK(ds4_gpu_argmax_tensor(to, tl, n_vocab) != 0, "argmax: call");
    int32_t got = -1;
    CHECK(ds4_gpu_tensor_read(to, 0, &got, sizeof(got)) != 0, "argmax: read");
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %d want %d\n", name, got, want);
        free(logits);
        return 1;
    }
    ds4_gpu_tensor_free(tl);
    ds4_gpu_tensor_free(to);
    free(logits);
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

static int test_argmax_no_tie(void) {
    return run_argmax_case("test_argmax_no_tie", 1500, 3, 0, 0, 0, 0.0f);
}

/* Adversarial per spec 6i: a small-XOR tie (indices 3 and 2, differing by
 * one bit) and a large-XOR tie (indices 3 and 700, XOR = 703, many bits
 * set) are BOTH tested, so a tie-break defect cannot hide behind a
 * reduction structure that happens to resolve the small-XOR case by
 * accident (the mechanism that once hid a real router tie-break bug). */
static int test_argmax_tie_small_xor(void) {
    return run_argmax_case("test_argmax_tie_small_xor", 1500, 11, 1, 3, 2, 500.0f);
}

static int test_argmax_tie_large_xor(void) {
    return run_argmax_case("test_argmax_tie_large_xor", 1500, 17, 1, 700, 3, 500.0f);
}

static int test_argmax_zero_and_null_rejections(void) {
    ds4_gpu_tensor *tl = ds4_gpu_tensor_alloc(16 * sizeof(float));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(int32_t));
    CHECK(tl && to, "argmax_rejections: alloc");
    CHECK(ds4_gpu_argmax_tensor(to, tl, 0) == 0, "argmax_rejections: zero n_vocab");
    CHECK(ds4_gpu_argmax_tensor(NULL, tl, 4) == 0, "argmax_rejections: null out");
    CHECK(ds4_gpu_argmax_tensor(to, NULL, 4) == 0, "argmax_rejections: null logits");
    ds4_gpu_tensor_free(tl);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_argmax_zero_and_null_rejections OK\n");
    return 0;
}

/* ---- Mask tests -------------------------------------------------------- */

static int test_mask_matches_selected_and_rejects_out_of_range(void) {
    enum { N_TOKENS = 3, N_COMP = 10, TOP_K = 4 };
    /* Token 0's list includes an out-of-range sentinel (0xffffffff, the
     * padding value an underfull upstream sort would produce) alongside
     * three real selections; token 1 and 2 use distinct real selections
     * so a transposed (t, c) decomposition bug is visible per-token. */
    uint32_t topk[N_TOKENS * TOP_K] = {
        0xffffffffu, 2u, 5u, 9u,
        1u, 3u, 4u, 6u,
        0u, 7u, 8u, 2u,
    };
    float want[N_TOKENS * N_COMP];
    for (int t = 0; t < N_TOKENS; t++) {
        for (int c = 0; c < N_COMP; c++) want[t * N_COMP + c] = -INFINITY;
        for (int k = 0; k < TOP_K; k++) {
            uint32_t idx = topk[t * TOP_K + k];
            if (idx < N_COMP) want[t * N_COMP + (int)idx] = 0.0f;
        }
    }

    ds4_gpu_tensor *ttopk = ds4_gpu_tensor_alloc(sizeof(topk));
    ds4_gpu_tensor *tmask = ds4_gpu_tensor_alloc(sizeof(want));
    CHECK(ttopk && tmask, "mask: alloc");
    CHECK(ds4_gpu_tensor_write(ttopk, 0, topk, sizeof(topk)) != 0, "mask: write");
    CHECK(ds4_gpu_dsv4_topk_mask_tensor(tmask, ttopk, N_COMP, N_TOKENS, TOP_K) != 0,
          "mask: call");
    float got[N_TOKENS * N_COMP];
    CHECK(ds4_gpu_tensor_read(tmask, 0, got, sizeof(got)) != 0, "mask: read");
    for (int i = 0; i < N_TOKENS * N_COMP; i++) {
        char msg[80];
        snprintf(msg, sizeof(msg), "mask: mismatch at %d", i);
        if (isinf(want[i])) {
            CHECK(isinf(got[i]) && got[i] < 0.0f, msg);
        } else {
            CHECK_CLOSE(got[i], want[i], 0.0, msg);
        }
    }
    ds4_gpu_tensor_free(ttopk);
    ds4_gpu_tensor_free(tmask);
    fprintf(stderr, "  test_mask_matches_selected_and_rejects_out_of_range OK\n");
    return 0;
}

static int test_mask_rejections(void) {
    ds4_gpu_tensor *ttopk = ds4_gpu_tensor_alloc(16 * sizeof(uint32_t));
    ds4_gpu_tensor *tmask = ds4_gpu_tensor_alloc(64 * sizeof(float));
    CHECK(ttopk && tmask, "mask_rejections: alloc");
    CHECK(ds4_gpu_dsv4_topk_mask_tensor(tmask, ttopk, 0, 4, 4) == 0,
          "mask_rejections: zero n_comp");
    CHECK(ds4_gpu_dsv4_topk_mask_tensor(tmask, ttopk, 4, 4, 0) == 0,
          "mask_rejections: zero top_k");
    CHECK(ds4_gpu_dsv4_topk_mask_tensor(NULL, ttopk, 4, 4, 4) == 0,
          "mask_rejections: null mask");
    ds4_gpu_tensor_free(ttopk);
    ds4_gpu_tensor_free(tmask);
    fprintf(stderr, "  test_mask_rejections OK\n");
    return 0;
}

/* ---- Top-k tests -------------------------------------------------------
 *
 * Oracle mirrors indexer_allowed_decode_one's own selection loop
 * (ds4.c:12977-12986: scan ascending, replace only on a STRICT `>`, so an
 * exact tie keeps the earlier/lower index) and topk_score_better's
 * ascending-index tie-break (rocm/ds4_rocm_indexer.cuh:308-310):
 * repeatedly picks the best remaining candidate in the same order the
 * bitonic network's descending sort would produce it. */
static void oracle_topk(uint32_t *out, const float *scores, uint32_t n_comp, uint32_t top_k) {
    int *used = (int *)calloc(n_comp, sizeof(int));
    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_v = -INFINITY;
        int found = 0;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (used[c]) continue;
            if (!found || scores[c] > best_v) {
                best = c;
                best_v = scores[c];
                found = 1;
            }
        }
        used[best] = 1;
        out[k] = best;
    }
    free(used);
}

static void fill_topk_scores(float *scores, uint32_t n_comp, uint32_t seed) {
    for (uint32_t c = 0; c < n_comp; c++) {
        uint32_t h = (c * 2654435761u + seed * 97531u) ^ (c << 6);
        scores[c] = (float)((int32_t)(h % 200001) - 100000) * 0.0001f;
    }
}

/* Runs ds4_gpu_indexer_topk_tensor for one token against oracle_topk and
 * asserts an exact index-for-index match (order matters: the kernel's
 * descending bitonic sort and the oracle's repeated-best-pick produce the
 * same order given the same tie-break), plus that no padding index
 * (>= n_comp) ever appears in the output (spec 6o: padding must never
 * win a comparison against real data). */
static int run_topk_case(const char *name, uint32_t n_comp, uint32_t top_k, uint32_t seed,
                         int n_ties, const uint32_t *tie_a, const uint32_t *tie_b,
                         const float *tie_val) {
    float *scores = (float *)malloc((size_t)n_comp * sizeof(float));
    CHECK(scores != NULL, "topk: host alloc");
    fill_topk_scores(scores, n_comp, seed);
    for (int i = 0; i < n_ties; i++) {
        scores[tie_a[i]] = tie_val[i];
        scores[tie_b[i]] = tie_val[i];
    }

    uint32_t *want = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    oracle_topk(want, scores, n_comp, top_k);

    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *tsel = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    CHECK(tscores && tsel, "topk: alloc");
    CHECK(ds4_gpu_tensor_write(tscores, 0, scores, (uint64_t)n_comp * sizeof(float)) != 0,
          "topk: write");
    CHECK(ds4_gpu_indexer_topk_tensor(tsel, tscores, n_comp, 1, top_k) != 0, "topk: call");

    uint32_t *got = (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    CHECK(ds4_gpu_tensor_read(tsel, 0, got, (uint64_t)top_k * sizeof(uint32_t)) != 0,
          "topk: read");

    int fail = 0;
    for (uint32_t k = 0; k < top_k; k++) {
        if (got[k] >= n_comp) {
            fprintf(stderr, "FAIL: %s: padding index %u won slot %u\n", name, got[k], k);
            fail = 1;
        } else if (got[k] != want[k]) {
            fprintf(stderr, "FAIL: %s: slot %u got %u want %u\n", name, k, got[k], want[k]);
            fail = 1;
        }
    }
    ds4_gpu_tensor_free(tscores);
    ds4_gpu_tensor_free(tsel);
    free(scores);
    free(want);
    free(got);
    if (fail) return 1;
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

/* Adversarial tie indices per spec 6i: one small-XOR pair and one
 * large-XOR pair, both real (not padding) indices within n_comp, spread
 * so neither collapses into the other's bitonic pairing by chance. */
static int run_topk_tie_case(const char *name, uint32_t n_comp, uint32_t top_k, uint32_t seed,
                             uint32_t small_a, uint32_t small_b, uint32_t large_a,
                             uint32_t large_b) {
    uint32_t tie_a[2] = {small_a, large_a};
    uint32_t tie_b[2] = {small_b, large_b};
    float tie_val[2] = {777.0f, 777.0f};
    return run_topk_case(name, n_comp, top_k, seed, 2, tie_a, tie_b, tie_val);
}

/* SORT_N == 1024 (indexer_topk_1024_kernel's shape, served here by
 * sycl_indexer_topk_pow2_kernel<1024>; see this file's header comment on
 * why those are the same kernel). */
static int test_topk_width_1024(void) {
    int failures = 0;
    failures += run_topk_case("test_topk_width_1024_padded", 777, 512, 101, 0, NULL, NULL, NULL);
    failures += run_topk_case("test_topk_width_1024_exact", 1024, 512, 103, 0, NULL, NULL, NULL);
    failures += run_topk_tie_case("test_topk_width_1024_ties", 1024, 512, 107, 5, 4, 600, 33);
    return failures;
}

static int test_topk_width_2048(void) {
    int failures = 0;
    failures += run_topk_case("test_topk_width_2048_padded", 1500, 512, 211, 0, NULL, NULL, NULL);
    failures += run_topk_case("test_topk_width_2048_exact", 2048, 512, 213, 0, NULL, NULL, NULL);
    failures += run_topk_tie_case("test_topk_width_2048_ties", 2048, 512, 217, 9, 8, 1500, 17);
    return failures;
}

static int test_topk_width_4096(void) {
    int failures = 0;
    failures += run_topk_case("test_topk_width_4096_padded", 3000, 512, 311, 0, NULL, NULL, NULL);
    failures += run_topk_case("test_topk_width_4096_exact", 4096, 512, 313, 0, NULL, NULL, NULL);
    failures += run_topk_tie_case("test_topk_width_4096_ties", 4096, 512, 317, 13, 12, 3000, 45);
    return failures;
}

static int test_topk_width_8192_u16(void) {
    int failures = 0;
    failures += run_topk_case("test_topk_width_8192_padded", 5000, 512, 411, 0, NULL, NULL, NULL);
    failures += run_topk_case("test_topk_width_8192_exact", 8192, 512, 413, 0, NULL, NULL, NULL);
    failures += run_topk_tie_case("test_topk_width_8192_ties", 8192, 512, 417, 21, 20, 6000, 77);
    return failures;
}

/* Fallback brute-force kernel: any top_k outside {512, 1024, 2048}. */
static int test_topk_fallback_kernel(void) {
    int failures = 0;
    failures += run_topk_case("test_topk_fallback_small", 50, 7, 511, 0, NULL, NULL, NULL);
    failures += run_topk_tie_case("test_topk_fallback_ties", 50, 7, 513, 3, 2, 40, 5);
    return failures;
}

static int test_topk_rejections(void) {
    ds4_gpu_tensor *tscores = ds4_gpu_tensor_alloc(64 * sizeof(float));
    ds4_gpu_tensor *tsel = ds4_gpu_tensor_alloc(64 * sizeof(uint32_t));
    CHECK(tscores && tsel, "topk_rejections: alloc");
    CHECK(ds4_gpu_indexer_topk_tensor(tsel, tscores, 0, 1, 4) == 0, "topk_rejections: zero n_comp");
    CHECK(ds4_gpu_indexer_topk_tensor(tsel, tscores, 4, 1, 0) == 0, "topk_rejections: zero top_k");
    CHECK(ds4_gpu_indexer_topk_tensor(tsel, tscores, 4, 1, 5) == 0,
          "topk_rejections: top_k > n_comp");
    CHECK(ds4_gpu_indexer_topk_tensor(NULL, tscores, 4, 1, 4) == 0, "topk_rejections: null selected");
    ds4_gpu_tensor_free(tscores);
    ds4_gpu_tensor_free(tsel);
    fprintf(stderr, "  test_topk_rejections OK\n");
    return 0;
}

int main(void) {
    int failures = 0;

    failures += test_qat_value_table_matches_oracle();
    failures += test_qat_dequant_matches_oracle();

    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");

    failures += test_qat_multi_row_matches_oracle();
    failures += test_qat_rejections();

    failures += test_scores_prefill_causal_matches_oracle();
    failures += test_scores_decode_batch_matches_oracle();
    failures += test_score_one_direct_path_matches_oracle();
    failures += test_score_one_generic_path_matches_oracle();
    failures += test_scores_rejections();

    failures += test_argmax_no_tie();
    failures += test_argmax_tie_small_xor();
    failures += test_argmax_tie_large_xor();
    failures += test_argmax_zero_and_null_rejections();

    failures += test_mask_matches_selected_and_rejects_out_of_range();
    failures += test_mask_rejections();

    failures += test_topk_width_1024();
    failures += test_topk_width_2048();
    failures += test_topk_width_4096();
    failures += test_topk_width_8192_u16();
    failures += test_topk_fallback_kernel();
    failures += test_topk_rejections();

    ds4_gpu_cleanup();

    if (failures == 0) {
        fprintf(stderr, "ALL SYCL INDEXER TESTS PASSED\n");
    } else {
        fprintf(stderr, "%d SYCL INDEXER TEST(S) FAILED\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
