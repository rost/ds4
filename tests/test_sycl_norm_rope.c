/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ds4.c line number cited.  Needs no model file.
 *
 * This file covers the normalisation and RoPE kernels: plain and
 * weighted RMS norm (flat and per-row), the fused add+RMS-norm, per-head
 * RMS norm, the fused DSv4 QKV RMS norm, and RoPE tail rotation (standalone
 * and fused with head RMS norm). */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Oracle mirrors rms_norm_no_weight, ds4.c:6754-6760:
 *   scale = 1/sqrt(mean(x^2) + eps);  out[i] = x[i] * scale
 * The CPU accumulates in double and the GPU tree-reduces in float, so
 * expect ULP-level divergence at these row widths, not more. */
static void oracle_rms_norm_plain(float *out, const float *x, int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

static int test_rms_norm_plain(void) {
    /* WIDTHS deliberately spans below, at, and above the fixed 256 work
     * group.  A width below 256 is the case that catches a port which
     * fails to initialise every local slot. */
    const int widths[] = {1, 17, 255, 256, 257, 1024};
    const float eps = 1e-5f;

    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const int n = widths[wi];
        float *x    = (float *)malloc((size_t)n * sizeof(float));
        float *got  = (float *)malloc((size_t)n * sizeof(float));
        float *want = (float *)malloc((size_t)n * sizeof(float));
        CHECK(x && got && want, "rms_plain: host allocation failed");

        for (int i = 0; i < n; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.25f;
        oracle_rms_norm_plain(want, x, n, eps);

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(tx && to, "rms_plain: device allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, (uint64_t)n * sizeof(float)) != 0,
              "rms_plain: write");
        CHECK(ds4_gpu_rms_norm_plain_tensor(to, tx, (uint32_t)n, eps) != 0,
              "rms_plain: call");
        CHECK(ds4_gpu_tensor_read(to, 0, got, (uint64_t)n * sizeof(float)) != 0,
              "rms_plain: read");
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(got[i], want[i], 1e-4, "rms_plain: value mismatch");
        }

        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(to);
        free(x); free(got); free(want);
    }

    /* Zero length is SUCCESS and returns nonzero, matching
     * rocm/ds4_rocm_norm_rope.cuh:414 `if (n == 0u) return 1;`. */
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(64);
    CHECK(t != NULL, "rms_plain: alloc for n=0 case");
    CHECK(ds4_gpu_rms_norm_plain_tensor(t, t, 0, eps) != 0,
          "rms_plain: n=0 must succeed");
    /* An output too small for n floats must be rejected. */
    CHECK(ds4_gpu_rms_norm_plain_tensor(t, t, 1024, eps) == 0,
          "rms_plain: undersized tensor must be rejected");
    ds4_gpu_tensor_free(t);

    fprintf(stderr, "  test_rms_norm_plain OK\n");
    return 0;
}

static int test_rms_norm_plain_rows(void) {
    enum { N = 300, ROWS = 5 };
    const float eps = 1e-5f;
    float x[N * ROWS], got[N * ROWS], want[N];

    for (int i = 0; i < N * ROWS; i++) x[i] = ((float)(i % 23) - 11.0f) * 0.1f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx && to, "rms_plain_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "rms_plain_rows: write");
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(to, tx, N, ROWS, eps) != 0,
          "rms_plain_rows: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "rms_plain_rows: read");

    /* Each row must be normalised independently.  Rows are given different
     * magnitudes above, so a kernel that normalised across the whole buffer
     * instead of per row would fail here while passing a single-row test. */
    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_plain(want, &x[r * N], N, eps);
        for (int i = 0; i < N; i++) {
            CHECK_CLOSE(got[r * N + i], want[i], 1e-4,
                        "rms_plain_rows: value mismatch");
        }
    }

    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(to, tx, N, 0, eps) != 0,
          "rms_plain_rows: rows=0 must succeed");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_rms_norm_plain_rows OK\n");
    return 0;
}

static int test_rms_norm_weight(void) {
    const int widths[] = {1, 17, 255, 256, 257, 1024};
    const float eps = 1e-5f;

    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const int n = widths[wi];
        float *x    = (float *)malloc((size_t)n * sizeof(float));
        float *w    = (float *)malloc((size_t)n * sizeof(float));
        float *got  = (float *)malloc((size_t)n * sizeof(float));
        float *want = (float *)malloc((size_t)n * sizeof(float));
        CHECK(x && w && got && want, "rms_weight: host allocation failed");

        for (int i = 0; i < n; i++) {
            x[i] = ((float)(i % 13) - 6.0f) * 0.25f;
            w[i] = ((float)(i % 7) + 1.0f) * 0.3f;
        }
        oracle_rms_norm_weight(want, x, w, n, eps);

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(tx && to, "rms_weight: device allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, (uint64_t)n * sizeof(float)) != 0,
              "rms_weight: write");
        CHECK(ds4_gpu_rms_norm_weight_tensor(to, tx, w,
                                             (uint64_t)n * sizeof(float), 0,
                                             (uint32_t)n, eps) != 0,
              "rms_weight: call");
        CHECK(ds4_gpu_tensor_read(to, 0, got, (uint64_t)n * sizeof(float)) != 0,
              "rms_weight: read");
        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(got[i], want[i], 1e-4, "rms_weight: value mismatch");
        }

        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(to);
        free(x); free(w); free(got); free(want);
    }

    /* Zero length is SUCCESS, matching rocm/ds4_rocm_norm_rope.cuh:429. */
    float model_dummy[1] = {1.0f};
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(64);
    CHECK(t != NULL, "rms_weight: alloc for n=0 case");
    CHECK(ds4_gpu_rms_norm_weight_tensor(t, t, model_dummy,
                                         sizeof(model_dummy), 0, 0,
                                         eps) != 0,
          "rms_weight: n=0 must succeed");

    /* An out-of-range weight_offset must be rejected. */
    CHECK(ds4_gpu_rms_norm_weight_tensor(t, t, model_dummy,
                                         sizeof(model_dummy),
                                         sizeof(model_dummy), 1, eps) == 0,
          "rms_weight: out-of-range weight_offset must be rejected");
    ds4_gpu_tensor_free(t);

    fprintf(stderr, "  test_rms_norm_weight OK\n");
    return 0;
}

static int test_rms_norm_weight_rows(void) {
    enum { N = 300, ROWS = 5 };
    const float eps = 1e-5f;
    float x[N * ROWS], w[N], got[N * ROWS], want[N];

    for (int i = 0; i < N * ROWS; i++) x[i] = ((float)(i % 23) - 11.0f) * 0.1f;
    for (int i = 0; i < N; i++) w[i] = ((float)(i % 5) + 1.0f) * 0.2f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tx && to, "rms_weight_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "rms_weight_rows: write");
    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w), 0, N,
                                              ROWS, eps) != 0,
          "rms_weight_rows: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0,
          "rms_weight_rows: read");

    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_weight(want, &x[r * N], w, N, eps);
        for (int i = 0; i < N; i++) {
            CHECK_CLOSE(got[r * N + i], want[i], 1e-4,
                        "rms_weight_rows: value mismatch");
        }
    }

    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w), 0, N, 0,
                                              eps) != 0,
          "rms_weight_rows: rows=0 must succeed");

    /* An out-of-range weight_offset must be rejected. */
    CHECK(ds4_gpu_rms_norm_weight_rows_tensor(to, tx, w, sizeof(w),
                                              sizeof(w), N, ROWS, eps) == 0,
          "rms_weight_rows: out-of-range weight_offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_rms_norm_weight_rows OK\n");
    return 0;
}

/* ds4_gpu_add_rms_norm_weight_tensor has no kernel of its own: it composes
 * ds4_gpu_add_tensor and ds4_gpu_rms_norm_weight_tensor
 * (rocm/ds4_rocm_norm_rope.cuh:451-469).  Validate against those two oracle
 * steps run separately, and assert sum_out as well as norm_out: a kernel
 * that computed the norm correctly but never published the elementwise sum
 * would otherwise pass. */
static int test_add_rms_norm_weight(void) {
    enum { N = 200 };
    const float eps = 1e-5f;
    float a[N], b[N], w[N], want_sum[N], want_norm[N];
    float got_sum[N], got_norm[N];

    for (int i = 0; i < N; i++) {
        a[i] = ((float)(i % 19) - 9.0f) * 0.15f;
        b[i] = ((float)(i % 11) - 5.0f) * 0.1f;
        w[i] = ((float)(i % 6) + 1.0f) * 0.25f;
        want_sum[i] = a[i] + b[i];
    }
    oracle_rms_norm_weight(want_norm, want_sum, w, N, eps);

    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(a));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(b));
    ds4_gpu_tensor *tsum = ds4_gpu_tensor_alloc(sizeof(got_sum));
    ds4_gpu_tensor *tnorm = ds4_gpu_tensor_alloc(sizeof(got_norm));
    CHECK(ta && tb && tsum && tnorm, "add_rms_weight: allocation failed");
    CHECK(ds4_gpu_tensor_write(ta, 0, a, sizeof(a)) != 0,
          "add_rms_weight: write a");
    CHECK(ds4_gpu_tensor_write(tb, 0, b, sizeof(b)) != 0,
          "add_rms_weight: write b");

    CHECK(ds4_gpu_add_rms_norm_weight_tensor(tnorm, tsum, ta, tb, w,
                                             sizeof(w), 0, N, eps) != 0,
          "add_rms_weight: call");
    CHECK(ds4_gpu_tensor_read(tsum, 0, got_sum, sizeof(got_sum)) != 0,
          "add_rms_weight: read sum");
    CHECK(ds4_gpu_tensor_read(tnorm, 0, got_norm, sizeof(got_norm)) != 0,
          "add_rms_weight: read norm");

    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got_sum[i], want_sum[i], 1e-4,
                    "add_rms_weight: sum_out mismatch");
        CHECK_CLOSE(got_norm[i], want_norm[i], 1e-4,
                    "add_rms_weight: norm_out mismatch");
    }

    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(tsum);
    ds4_gpu_tensor_free(tnorm);
    fprintf(stderr, "  test_add_rms_norm_weight OK\n");
    return 0;
}

/* Oracle mirrors head_rms_norm_inplace, ds4.c:6772-6781: normalises each
 * head's head_dim slice independently within one token's n_head*head_dim
 * block. */
static void oracle_head_rms_norm_inplace(float *x, int n_head, int head_dim,
                                         float eps) {
    for (int h = 0; h < n_head; h++) {
        float *head = x + (size_t)h * head_dim;
        double ss = 0.0;
        for (int i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (int i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

static int test_head_rms_norm(void) {
    enum { N_TOK = 3, N_HEAD = 4, HEAD_DIM = 65 };
    const float eps = 1e-5f;
    float x[N_TOK * N_HEAD * HEAD_DIM];
    float want[N_TOK * N_HEAD * HEAD_DIM];
    float got[N_TOK * N_HEAD * HEAD_DIM];

    /* Each (token, head) pair gets a different magnitude, so a kernel that
     * normalised across heads instead of within each, or across the whole
     * buffer instead of per (token, head), fails rather than passing by
     * coincidence. */
    for (int t = 0; t < N_TOK; t++) {
        for (int h = 0; h < N_HEAD; h++) {
            const float mag = (float)(h + 1) * (float)(t + 2);
            float *dst = &x[(t * N_HEAD + h) * HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; i++) {
                dst[i] = ((float)(i % 9) - 4.0f) * 0.1f * mag;
            }
        }
    }
    memcpy(want, x, sizeof(x));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "head_rms_norm: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "head_rms_norm: write");
    CHECK(ds4_gpu_head_rms_norm_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, eps) != 0,
          "head_rms_norm: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
          "head_rms_norm: read");

    for (int t = 0; t < N_TOK; t++) {
        oracle_head_rms_norm_inplace(&want[t * N_HEAD * HEAD_DIM], N_HEAD,
                                     HEAD_DIM, eps);
    }
    for (int i = 0; i < N_TOK * N_HEAD * HEAD_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-4, "head_rms_norm: value mismatch");
    }

    CHECK(ds4_gpu_head_rms_norm_tensor(tx, N_TOK, N_HEAD, 0, eps) != 0,
          "head_rms_norm: head_dim=0 must succeed");
    CHECK(ds4_gpu_head_rms_norm_tensor(tx, 0, N_HEAD, HEAD_DIM, eps) != 0,
          "head_rms_norm: n_tok=0 must succeed");

    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(4);
    CHECK(small != NULL, "head_rms_norm: alloc for undersized case");
    CHECK(ds4_gpu_head_rms_norm_tensor(small, 1, 1, HEAD_DIM, eps) == 0,
          "head_rms_norm: undersized tensor must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_head_rms_norm OK\n");
    return 0;
}

/* ds4_gpu_dsv4_qkv_rms_norm_rows_tensor runs rms_norm_weight
 * (ds4.c:6763-6769) twice per row in one launch, with the Q and KV halves
 * selecting their own width (rocm/ds4_rocm_norm_rope.cuh:47-81).  Q_N and
 * KV_N are deliberately different: a port that computed the row stride
 * before selecting the width, or that reused one width for both sides,
 * would mis-index one half silently if the widths matched. */
static int test_dsv4_qkv_rms_norm_rows(void) {
    enum { ROWS = 4, Q_N = 192, KV_N = 96 };
    const float eps = 1e-5f;
    float q_x[ROWS * Q_N], kv_x[ROWS * KV_N];
    float q_w[Q_N], kv_w[KV_N];
    float q_got[ROWS * Q_N], kv_got[ROWS * KV_N];
    float q_want[Q_N], kv_want[KV_N];

    for (int i = 0; i < ROWS * Q_N; i++) {
        q_x[i] = ((float)(i % 17) - 8.0f) * 0.1f;
    }
    for (int i = 0; i < ROWS * KV_N; i++) {
        kv_x[i] = ((float)(i % 11) - 5.0f) * 0.2f;
    }
    for (int i = 0; i < Q_N; i++) q_w[i] = ((float)(i % 5) + 1.0f) * 0.3f;
    for (int i = 0; i < KV_N; i++) kv_w[i] = ((float)(i % 3) + 1.0f) * 0.4f;

    /* Concatenate both weight vectors into one host "model" buffer so the
     * offsets exercised are real byte offsets rather than both being 0. */
    unsigned char model[sizeof(q_w) + sizeof(kv_w)];
    memcpy(model, q_w, sizeof(q_w));
    memcpy(model + sizeof(q_w), kv_w, sizeof(kv_w));
    const uint64_t q_off = 0;
    const uint64_t kv_off = sizeof(q_w);

    ds4_gpu_tensor *tqx = ds4_gpu_tensor_alloc(sizeof(q_x));
    ds4_gpu_tensor *tqo = ds4_gpu_tensor_alloc(sizeof(q_got));
    ds4_gpu_tensor *tkvx = ds4_gpu_tensor_alloc(sizeof(kv_x));
    ds4_gpu_tensor *tkvo = ds4_gpu_tensor_alloc(sizeof(kv_got));
    CHECK(tqx && tqo && tkvx && tkvo, "dsv4_qkv: allocation failed");
    CHECK(ds4_gpu_tensor_write(tqx, 0, q_x, sizeof(q_x)) != 0,
          "dsv4_qkv: write q");
    CHECK(ds4_gpu_tensor_write(tkvx, 0, kv_x, sizeof(kv_x)) != 0,
          "dsv4_qkv: write kv");

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), q_off, Q_N,
              tkvo, tkvx, kv_off, KV_N, ROWS, eps) != 0,
          "dsv4_qkv: call");

    CHECK(ds4_gpu_tensor_read(tqo, 0, q_got, sizeof(q_got)) != 0,
          "dsv4_qkv: read q");
    CHECK(ds4_gpu_tensor_read(tkvo, 0, kv_got, sizeof(kv_got)) != 0,
          "dsv4_qkv: read kv");

    for (int r = 0; r < ROWS; r++) {
        oracle_rms_norm_weight(q_want, &q_x[r * Q_N], q_w, Q_N, eps);
        for (int i = 0; i < Q_N; i++) {
            CHECK_CLOSE(q_got[r * Q_N + i], q_want[i], 1e-4,
                        "dsv4_qkv: q value mismatch");
        }
        oracle_rms_norm_weight(kv_want, &kv_x[r * KV_N], kv_w, KV_N, eps);
        for (int i = 0; i < KV_N; i++) {
            CHECK_CLOSE(kv_got[r * KV_N + i], kv_want[i], 1e-4,
                        "dsv4_qkv: kv value mismatch");
        }
    }

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), q_off, Q_N,
              tkvo, tkvx, kv_off, KV_N, 0, eps) != 0,
          "dsv4_qkv: rows=0 must succeed");

    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
              tqo, tqx, model, sizeof(model), sizeof(model), Q_N,
              tkvo, tkvx, kv_off, KV_N, ROWS, eps) == 0,
          "dsv4_qkv: out-of-range q weight_offset must be rejected");

    ds4_gpu_tensor_free(tqx);
    ds4_gpu_tensor_free(tqo);
    ds4_gpu_tensor_free(tkvx);
    ds4_gpu_tensor_free(tkvo);
    fprintf(stderr, "  test_dsv4_qkv_rms_norm_rows OK\n");
    return 0;
}

/* ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor: fuses the QKV RMS norm
 * above with the KV rotary tail in one launch.  q_out gets plain
 * norm-with-weight, exactly like the unfused entry's q branch.  kv_out's
 * RMS-norm SCALE is ONE reduction over the WHOLE kv_n_head * kv_head_dim
 * row (not per head), then every head's leading n_nope channels are
 * scaled and weighted, and every head's trailing n_rot channels are
 * scaled, weighted AND rotated in place -- matching CUDA's
 * dsv4_qkv_rms_norm_rows_kv_rope_kernel (ds4_cuda.cu:6565), the only
 * reference for this entry (no ROCm implementation exists at all).
 *
 * KV_N_HEAD = 2 deliberately: Flash always runs this with kv_n_head == 1
 * (ds4.c's DS4_N_HEAD_KV), but a wider test here exercises the per-head
 * loop and the head-relative-vs-absolute weight indexing this entry is
 * new at, which a single-head test could not.  Per-head magnitudes
 * (mag = (h+1)*(r+2), matching test_head_rms_norm's technique) plus a
 * nonlinear (d + r*7 + h*3) % 13 interaction term keep heads and rows
 * genuinely non-proportional to each other: per spec 6f, RMS norm divides
 * a pure scale-only difference straight back out, and affine test data
 * would hide exactly the reduction-scope bug this entry is new at (a
 * per-head reduction masquerading as this kernel's required whole-row
 * one, since equal per-head magnitudes would make both reductions agree). */
static int test_dsv4_qkv_rms_norm_rows_kv_rope(void) {
    enum { ROWS = 3, Q_N = 40, KV_N_HEAD = 2, KV_HEAD_DIM = 24,
           KV_N = KV_N_HEAD * KV_HEAD_DIM, N_ROT = 16 };
    const uint32_t POS0 = 5;
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    const float EPS = 1e-5f;
    const double TOL = 1e-3;
    const double DIFF_TOL = 1e-4;

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0},
        {1.0f, 0.0f, 1},
        {0.5f, 1.0f, 0},
        {0.5f, 1.0f, 1},
    };

    float q_x[ROWS * Q_N], kv_x[ROWS * KV_N];
    float q_w[Q_N], kv_w[KV_N];
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < Q_N; i++) {
            q_x[r * Q_N + i] = ((float)((i + r * 13) % 19) - 9.0f) * 0.1f;
        }
        for (int h = 0; h < KV_N_HEAD; h++) {
            const float mag = (float)(h + 1) * (float)(r + 2);
            for (int d = 0; d < KV_HEAD_DIM; d++) {
                const int idx = h * KV_HEAD_DIM + d;
                kv_x[r * KV_N + idx] =
                        ((float)((d + r * 7 + h * 3) % 13) - 6.0f) * 0.15f * mag;
            }
        }
    }
    for (int i = 0; i < Q_N; i++) q_w[i] = ((float)(i % 5) + 1.0f) * 0.3f;
    for (int i = 0; i < KV_N; i++) kv_w[i] = ((float)(i % 7) + 1.0f) * 0.25f;

    unsigned char model[sizeof(q_w) + sizeof(kv_w)];
    memcpy(model, q_w, sizeof(q_w));
    memcpy(model + sizeof(q_w), kv_w, sizeof(kv_w));
    const uint64_t q_off = 0;
    const uint64_t kv_off = sizeof(q_w);

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float q_got[ROWS * Q_N], kv_got[ROWS * KV_N], kv_diff[ROWS * KV_N];
        float q_want[Q_N], kv_want[KV_N];

        ds4_gpu_tensor *tqx  = ds4_gpu_tensor_alloc(sizeof(q_x));
        ds4_gpu_tensor *tqo  = ds4_gpu_tensor_alloc(sizeof(q_got));
        ds4_gpu_tensor *tkvx = ds4_gpu_tensor_alloc(sizeof(kv_x));
        ds4_gpu_tensor *tkvo = ds4_gpu_tensor_alloc(sizeof(kv_got));
        ds4_gpu_tensor *tdiff = ds4_gpu_tensor_alloc(sizeof(kv_diff));
        CHECK(tqx && tqo && tkvx && tkvo && tdiff,
              "dsv4_qkv_kv_rope: allocation failed");
        CHECK(ds4_gpu_tensor_write(tqx, 0, q_x, sizeof(q_x)) != 0,
              "dsv4_qkv_kv_rope: write q");
        CHECK(ds4_gpu_tensor_write(tkvx, 0, kv_x, sizeof(kv_x)) != 0,
              "dsv4_qkv_kv_rope: write kv");
        CHECK(ds4_gpu_tensor_write(tdiff, 0, kv_x, sizeof(kv_x)) != 0,
              "dsv4_qkv_kv_rope: write kv (differential copy)");

        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  tkvo, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM, N_ROT, POS0, N_CTX_ORIG,
                  cases[ci].inverse, FREQ_BASE, cases[ci].freq_scale,
                  cases[ci].ext_factor, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) != 0,
              "dsv4_qkv_kv_rope: call");
        CHECK(ds4_gpu_tensor_read(tqo, 0, q_got, sizeof(q_got)) != 0,
              "dsv4_qkv_kv_rope: read q");
        CHECK(ds4_gpu_tensor_read(tkvo, 0, kv_got, sizeof(kv_got)) != 0,
              "dsv4_qkv_kv_rope: read kv");

        /* Differential: the unfused norm entry followed by the standalone
         * rope-tail entry, on an identical copy of the input, must agree
         * with the fused kernel's kv_out. tdiff doubles as q's unfused
         * output target below; only its kv-shaped buffer matters here. */
        ds4_gpu_tensor *tqo_unfused = ds4_gpu_tensor_alloc(sizeof(q_got));
        CHECK(tqo_unfused != NULL, "dsv4_qkv_kv_rope: unfused q alloc");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                  tqo_unfused, tqx, model, sizeof(model), q_off, Q_N,
                  tdiff, tkvx, kv_off, KV_N, ROWS, EPS) != 0,
              "dsv4_qkv_kv_rope: unfused norm call");
        CHECK(ds4_gpu_rope_tail_tensor(tdiff, ROWS, KV_N_HEAD, KV_HEAD_DIM,
                                       N_ROT, POS0, N_CTX_ORIG,
                                       cases[ci].inverse, FREQ_BASE,
                                       cases[ci].freq_scale,
                                       cases[ci].ext_factor, ATTN_FACTOR,
                                       BETA_FAST, BETA_SLOW) != 0,
              "dsv4_qkv_kv_rope: unfused rope_tail call");
        CHECK(ds4_gpu_tensor_read(tdiff, 0, kv_diff, sizeof(kv_diff)) != 0,
              "dsv4_qkv_kv_rope: read unfused kv");
        ds4_gpu_tensor_free(tqo_unfused);

        for (int i = 0; i < ROWS * KV_N; i++) {
            CHECK_CLOSE(kv_got[i], kv_diff[i], DIFF_TOL,
                        "dsv4_qkv_kv_rope: fused vs unfused composition "
                        "mismatch");
        }

        /* CPU oracle: whole-row norm-with-weight (spanning every head),
         * then per-head rotation of the now-scaled-and-weighted tail. */
        for (int r = 0; r < ROWS; r++) {
            oracle_rms_norm_weight(q_want, &q_x[r * Q_N], q_w, Q_N, EPS);
            for (int i = 0; i < Q_N; i++) {
                CHECK_CLOSE(q_got[r * Q_N + i], q_want[i], TOL,
                            "dsv4_qkv_kv_rope: q value mismatch");
            }

            oracle_rms_norm_weight(kv_want, &kv_x[r * KV_N], kv_w, KV_N, EPS);
            for (int h = 0; h < KV_N_HEAD; h++) {
                oracle_rope_tail_row(&kv_want[h * KV_HEAD_DIM], KV_HEAD_DIM,
                                     N_ROT, POS0 + (uint32_t)r, N_CTX_ORIG,
                                     FREQ_BASE, cases[ci].freq_scale,
                                     cases[ci].ext_factor, ATTN_FACTOR,
                                     BETA_FAST, BETA_SLOW, cases[ci].inverse);
            }
            for (int i = 0; i < KV_N; i++) {
                CHECK_CLOSE(kv_got[r * KV_N + i], kv_want[i], TOL,
                            "dsv4_qkv_kv_rope: kv value mismatch");
            }
        }

        ds4_gpu_tensor_free(tqx);
        ds4_gpu_tensor_free(tqo);
        ds4_gpu_tensor_free(tkvx);
        ds4_gpu_tensor_free(tkvo);
        ds4_gpu_tensor_free(tdiff);
    }

    /* rows == 0 must succeed without doing any work. */
    {
        ds4_gpu_tensor *tqx  = ds4_gpu_tensor_alloc(sizeof(q_x));
        ds4_gpu_tensor *tqo  = ds4_gpu_tensor_alloc(sizeof(q_x));
        ds4_gpu_tensor *tkvx = ds4_gpu_tensor_alloc(sizeof(kv_x));
        ds4_gpu_tensor *tkvo = ds4_gpu_tensor_alloc(sizeof(kv_x));
        CHECK(tqx && tqo && tkvx && tkvo, "dsv4_qkv_kv_rope: rows=0 alloc");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  tkvo, tkvx, kv_off, KV_N, 0,
                  KV_N_HEAD, KV_HEAD_DIM, N_ROT, POS0, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) != 0,
              "dsv4_qkv_kv_rope: rows=0 must succeed");
        ds4_gpu_tensor_free(tqx);
        ds4_gpu_tensor_free(tqo);
        ds4_gpu_tensor_free(tkvx);
        ds4_gpu_tensor_free(tkvo);
    }

    /* Validation: odd n_rot, n_rot > kv_head_dim, a kv_n that does not
     * factor as kv_n_head * kv_head_dim, an out-of-range weight offset,
     * and an undersized output tensor must all be rejected. */
    {
        ds4_gpu_tensor *tqx  = ds4_gpu_tensor_alloc(sizeof(q_x));
        ds4_gpu_tensor *tqo  = ds4_gpu_tensor_alloc(sizeof(q_x));
        ds4_gpu_tensor *tkvx = ds4_gpu_tensor_alloc(sizeof(kv_x));
        ds4_gpu_tensor *tkvo = ds4_gpu_tensor_alloc(sizeof(kv_x));
        CHECK(tqx && tqo && tkvx && tkvo, "dsv4_qkv_kv_rope: validation alloc");

        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  tkvo, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM, N_ROT - 1, POS0, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "dsv4_qkv_kv_rope: odd n_rot must be rejected");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  tkvo, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM, KV_HEAD_DIM + 2, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, EPS) == 0,
              "dsv4_qkv_kv_rope: n_rot > kv_head_dim must be rejected");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  tkvo, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM + 1, N_ROT, POS0, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "dsv4_qkv_kv_rope: kv_n != kv_n_head*kv_head_dim must be "
              "rejected");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), sizeof(model), Q_N,
                  tkvo, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM, N_ROT, POS0, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "dsv4_qkv_kv_rope: out-of-range q weight_offset must be "
              "rejected");

        ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 4);
        CHECK(small != NULL, "dsv4_qkv_kv_rope: small allocation failed");
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                  tqo, tqx, model, sizeof(model), q_off, Q_N,
                  small, tkvx, kv_off, KV_N, ROWS,
                  KV_N_HEAD, KV_HEAD_DIM, N_ROT, POS0, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "dsv4_qkv_kv_rope: undersized kv_out must be rejected");
        ds4_gpu_tensor_free(small);

        ds4_gpu_tensor_free(tqx);
        ds4_gpu_tensor_free(tqo);
        ds4_gpu_tensor_free(tkvx);
        ds4_gpu_tensor_free(tkvo);
    }

    fprintf(stderr, "  test_dsv4_qkv_rms_norm_rows_kv_rope OK\n");
    return 0;
}

/* ds4_gpu_rope_tail_tensor: the flat, pair-parallel, no-reduction RoPE
 * kernel.  N_HEAD > 1 and N_TOK > 1 both hold so the kernel's own
 * decomposition of the flat id into (pair, head, token) is genuinely
 * exercised; a transposed decomposition would still pass with either at
 * 1.  HEAD_DIM > N_ROT exercises the scale-only... except this kernel has
 * no scale at all, so it exercises the leading channels being left
 * completely untouched, which is the flat path's equivalent property. */
static int test_rope_tail(void) {
    enum { N_TOK = 3, N_HEAD = 2, HEAD_DIM = 24, N_ROT = 16,
           N_NOPE = HEAD_DIM - N_ROT, TOTAL = N_TOK * N_HEAD * HEAD_DIM };
    const uint32_t POS0 = 5;
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    /* c/s divergence between the iterative oracle and the GPU's
     * direct-power form measured at these parameters is on the order of
     * 1e-7; this tolerance leaves two orders of magnitude of margin. */
    const double TOL = 1e-4;

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0}, /* plain interpolation, forward */
        {1.0f, 0.0f, 1}, /* plain interpolation, inverse */
        {0.5f, 1.0f, 0}, /* YaRN correction active, forward */
        {0.5f, 1.0f, 1}, /* YaRN correction active, inverse */
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float x[TOTAL], got[TOTAL], want_row[HEAD_DIM];
        for (int i = 0; i < TOTAL; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.2f;

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "rope_tail: allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "rope_tail: write");

        CHECK(ds4_gpu_rope_tail_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
                                       POS0, N_CTX_ORIG, cases[ci].inverse,
                                       FREQ_BASE, cases[ci].freq_scale,
                                       cases[ci].ext_factor, ATTN_FACTOR,
                                       BETA_FAST, BETA_SLOW) != 0,
              "rope_tail: call");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "rope_tail: read");

        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const int base = (t * N_HEAD + h) * HEAD_DIM;
                const float *row_x   = &x[base];
                const float *row_got = &got[base];

                /* No `scale`: the leading n_nope channels must be
                 * BYTE-IDENTICAL to the input.  A kernel that scaled them
                 * (borrowing the fused kernel's behaviour by mistake)
                 * would fail here. */
                for (int c = 0; c < N_NOPE; c++) {
                    CHECK(row_got[c] == row_x[c],
                          "rope_tail: leading channels must be untouched");
                }

                memcpy(want_row, row_x, sizeof(want_row));
                oracle_rope_tail_row(want_row, HEAD_DIM, N_ROT, POS0 + (uint32_t)t,
                                     N_CTX_ORIG, FREQ_BASE, cases[ci].freq_scale,
                                     cases[ci].ext_factor, ATTN_FACTOR,
                                     BETA_FAST, BETA_SLOW, cases[ci].inverse);
                for (int c = N_NOPE; c < HEAD_DIM; c++) {
                    CHECK_CLOSE(row_got[c], want_row[c], TOL,
                                "rope_tail: rotated tail mismatch");
                }
            }
        }

        ds4_gpu_tensor_free(tx);
    }

    /* n_rot == 0 has zero pairs and must succeed without modifying x,
     * matching the pairs == 0 early-out. */
    {
        float x[TOTAL], got[TOTAL];
        for (int i = 0; i < TOTAL; i++) x[i] = ((float)(i % 13) - 6.0f) * 0.2f;
        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "rope_tail: n_rot=0 allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "rope_tail: n_rot=0 write");
        CHECK(ds4_gpu_rope_tail_tensor(tx, N_TOK, N_HEAD, HEAD_DIM, 0, POS0,
                                       N_CTX_ORIG, false, FREQ_BASE, 1.0f,
                                       0.0f, 1.0f, BETA_FAST, BETA_SLOW) != 0,
              "rope_tail: n_rot=0 must succeed");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "rope_tail: n_rot=0 read");
        CHECK(memcmp(x, got, sizeof(x)) == 0,
              "rope_tail: n_rot=0 must leave the tensor untouched");
        ds4_gpu_tensor_free(tx);
    }

    /* An odd n_rot must be rejected: the kernel forms pairs, and an odd
     * count would read one element past the tail. */
    {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)TOTAL * sizeof(float));
        CHECK(t != NULL, "rope_tail: odd n_rot allocation failed");
        CHECK(ds4_gpu_rope_tail_tensor(t, N_TOK, N_HEAD, HEAD_DIM, N_ROT - 1,
                                       POS0, N_CTX_ORIG, false, FREQ_BASE,
                                       1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: odd n_rot must be rejected");

        /* n_rot > head_dim must also be rejected. */
        CHECK(ds4_gpu_rope_tail_tensor(t, N_TOK, N_HEAD, HEAD_DIM,
                                       HEAD_DIM + 2, POS0, N_CTX_ORIG, false,
                                       FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: n_rot > head_dim must be rejected");
        ds4_gpu_tensor_free(t);
    }

    /* A tensor smaller than n_tok * n_head * head_dim floats must be
     * rejected. */
    {
        ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 4);
        CHECK(small != NULL, "rope_tail: small allocation failed");
        CHECK(ds4_gpu_rope_tail_tensor(small, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
                                       POS0, N_CTX_ORIG, false, FREQ_BASE,
                                       1.0f, 0.0f, 1.0f, BETA_FAST,
                                       BETA_SLOW) == 0,
              "rope_tail: undersized tensor must be rejected");
        ds4_gpu_tensor_free(small);
    }

    fprintf(stderr, "  test_rope_tail OK\n");
    return 0;
}

/* ds4_gpu_rope_tail_decode_rows_tensor: the row-batched form of
 * ds4_gpu_rope_tail_tensor above, used by metal_graph_encode_qkv_session_
 * batch (ds4.c:64836 on the KV row, :64855 on the Q row) to RoPE-rotate
 * several concurrent sessions' rows in one launch, each at its own
 * absolute position.
 *
 * Primary check is DIFFERENTIAL, preferred here
 * over an oracle alone: each row's output must match calling the already-
 * landed single-row ds4_gpu_rope_tail_tensor once for that row alone, at
 * that row's own position. Per design spec 6u a relational check like
 * that cannot catch an error that moves every row identically (for
 * instance, applying row 0's angle to every row disagrees with the
 * single-row entry too, in fact, but a broken oracle comparison could
 * still get lucky), so one specific row's tail is additionally checked
 * against oracle_rope_tail_row directly, anchoring it to something
 * outside the differential relation.
 *
 * N_ROWS = 5 is deliberately not a multiple of any power-of-two tile or
 * work-group width this kernel touches (kRmsNormGroup = 256, or any
 * smaller launch granularity), and the five positions are not an affine
 * sequence (spec 6f/6n): {5, 130, 7, 999, 42} mixes small and large
 * magnitudes with no shared arithmetic progression. */
static int test_rope_tail_decode_rows(void) {
    enum { N_ROWS = 5, N_HEAD = 3, HEAD_DIM = 32, N_ROT = 24,
           N_NOPE = HEAD_DIM - N_ROT, ROW_ELEMS = N_HEAD * HEAD_DIM,
           TOTAL = N_ROWS * ROW_ELEMS };
    const uint32_t positions[N_ROWS] = {5u, 130u, 7u, 999u, 42u};
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    const double TOL_DIFFERENTIAL = 1e-6; /* same kernel math, same order */
    const double TOL_ORACLE = 1e-4;       /* iterative vs direct-power angle */

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0},
        {1.0f, 0.0f, 1},
        {0.5f, 1.0f, 0},
        {0.5f, 1.0f, 1},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float x[TOTAL];
        for (int i = 0; i < TOTAL; i++) {
            /* Non-affine: a linear term plus an interaction term, per
             * spec 6f/6n, so no two elements are proportional. */
            x[i] = ((float)(i % 13) - 6.0f) * 0.2f +
                   (float)((i * 7) % 5) * 0.03f;
        }

        ds4_gpu_attention_decode_row rows[N_ROWS];
        memset(rows, 0, sizeof(rows));
        for (int r = 0; r < N_ROWS; r++) rows[r].pos = positions[r];

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "rope_tail_decode_rows: allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "rope_tail_decode_rows: write");

        CHECK(ds4_gpu_rope_tail_decode_rows_tensor(
                  tx, rows, N_ROWS, N_HEAD, HEAD_DIM, N_ROT, N_CTX_ORIG,
                  cases[ci].inverse, FREQ_BASE, cases[ci].freq_scale,
                  cases[ci].ext_factor, ATTN_FACTOR, BETA_FAST, BETA_SLOW) != 0,
              "rope_tail_decode_rows: call");

        float got[TOTAL];
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "rope_tail_decode_rows: read");
        ds4_gpu_tensor_free(tx);

        for (int r = 0; r < N_ROWS; r++) {
            /* Differential: the single-row entry, called alone with
             * n_tok=1 and this row's own position, on a fresh copy of
             * this row's untouched input. */
            float row_x[ROW_ELEMS];
            memcpy(row_x, &x[r * ROW_ELEMS], sizeof(row_x));
            ds4_gpu_tensor *trow = ds4_gpu_tensor_alloc(sizeof(row_x));
            CHECK(trow != NULL, "rope_tail_decode_rows: row allocation failed");
            CHECK(ds4_gpu_tensor_write(trow, 0, row_x, sizeof(row_x)) != 0,
                  "rope_tail_decode_rows: row write");
            CHECK(ds4_gpu_rope_tail_tensor(
                      trow, 1, N_HEAD, HEAD_DIM, N_ROT, positions[r],
                      N_CTX_ORIG, cases[ci].inverse, FREQ_BASE,
                      cases[ci].freq_scale, cases[ci].ext_factor, ATTN_FACTOR,
                      BETA_FAST, BETA_SLOW) != 0,
                  "rope_tail_decode_rows: single-row reference call");
            float want_row[ROW_ELEMS];
            CHECK(ds4_gpu_tensor_read(trow, 0, want_row, sizeof(want_row)) != 0,
                  "rope_tail_decode_rows: single-row reference read");
            ds4_gpu_tensor_free(trow);

            const float *row_got = &got[r * ROW_ELEMS];
            for (int c = 0; c < ROW_ELEMS; c++) {
                CHECK_CLOSE(row_got[c], want_row[c], TOL_DIFFERENTIAL,
                            "rope_tail_decode_rows: differs from single-row entry");
            }
        }

        /* Anchor (spec 6u): row 2's tail checked against the oracle
         * directly, independent of the differential relation above. */
        {
            const int anchor_row = 2;
            float want_row[HEAD_DIM];
            memcpy(want_row, &x[anchor_row * ROW_ELEMS + 0 * HEAD_DIM],
                   sizeof(want_row));
            oracle_rope_tail_row(want_row, HEAD_DIM, N_ROT, positions[anchor_row],
                                 N_CTX_ORIG, FREQ_BASE, cases[ci].freq_scale,
                                 cases[ci].ext_factor, ATTN_FACTOR, BETA_FAST,
                                 BETA_SLOW, cases[ci].inverse);
            const float *row_got = &got[anchor_row * ROW_ELEMS + 0 * HEAD_DIM];
            for (int c = N_NOPE; c < HEAD_DIM; c++) {
                CHECK_CLOSE(row_got[c], want_row[c], TOL_ORACLE,
                            "rope_tail_decode_rows: anchor row disagrees with independent oracle");
            }
        }
    }

    /* n_rows == 0 must be rejected (matches ds4_cuda.cu's own launcher). */
    {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)ROW_ELEMS * sizeof(float));
        CHECK(t != NULL, "rope_tail_decode_rows: n_rows=0 allocation failed");
        CHECK(ds4_gpu_rope_tail_decode_rows_tensor(
                  t, NULL, 0u, N_HEAD, HEAD_DIM, N_ROT, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW) == 0,
              "rope_tail_decode_rows: n_rows=0 must be rejected");
        ds4_gpu_tensor_free(t);
    }

    /* n_rows beyond DS4_GPU_ATTENTION_DECODE_BATCH_MAX must be rejected:
     * the third ablation below targets exactly this bound. */
    {
        enum { OVER = DS4_GPU_ATTENTION_DECODE_BATCH_MAX + 1u };
        ds4_gpu_attention_decode_row rows[OVER];
        memset(rows, 0, sizeof(rows));
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)OVER * ROW_ELEMS * sizeof(float));
        CHECK(t != NULL, "rope_tail_decode_rows: oversized allocation failed");
        CHECK(ds4_gpu_rope_tail_decode_rows_tensor(
                  t, rows, OVER, N_HEAD, HEAD_DIM, N_ROT, N_CTX_ORIG, false,
                  FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW) == 0,
              "rope_tail_decode_rows: n_rows over the batch cap must be rejected");
        ds4_gpu_tensor_free(t);
    }

    fprintf(stderr, "  test_rope_tail_decode_rows OK\n");
    return 0;
}

/* ds4_gpu_head_rms_norm_rope_tail_tensor: the fused, one-work-group-per-row
 * kernel.  Validated against the COMPOSED oracle: head RMS-norm first
 * (oracle_head_rms_norm_inplace with n_head=1, applied to one row at a
 * time), then RoPE on the now-scaled row, matching the GPU's own "scale
 * everything, then rotate the scaled tail" ordering -- the fused kernel
 * never stores an unscaled tail anywhere, so composing the two oracles in
 * this order is exactly equivalent to the kernel's single fused pass. */
static int test_head_rms_norm_rope_tail(void) {
    enum { N_TOK = 2, N_HEAD = 3, HEAD_DIM = 24, N_ROT = 16,
           TOTAL = N_TOK * N_HEAD * HEAD_DIM };
    const uint32_t POS0 = 5;
    const uint32_t N_CTX_ORIG = 4096;
    const float FREQ_BASE = 10000.0f;
    const float BETA_FAST = 32.0f;
    const float BETA_SLOW = 1.0f;
    const float ATTN_FACTOR = 1.0f;
    const float EPS = 1e-5f;
    const double TOL = 1e-4;

    struct { float freq_scale; float ext_factor; int inverse; } cases[] = {
        {1.0f, 0.0f, 0},
        {1.0f, 0.0f, 1},
        {0.5f, 1.0f, 0},
        {0.5f, 1.0f, 1},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        float x[TOTAL], got[TOTAL], want_row[HEAD_DIM];

        /* Each (token, head) pair gets a different magnitude, exactly as
         * test_head_rms_norm does, so a kernel that mixed up rows would
         * fail rather than pass by coincidence. */
        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const float mag = (float)(h + 1) * (float)(t + 2);
                float *dst = &x[(t * N_HEAD + h) * HEAD_DIM];
                for (int i = 0; i < HEAD_DIM; i++) {
                    dst[i] = ((float)(i % 9) - 4.0f) * 0.1f * mag;
                }
            }
        }

        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
        CHECK(tx != NULL, "head_rms_rope: allocation failed");
        CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
              "head_rms_rope: write");

        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  tx, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0, N_CTX_ORIG,
                  cases[ci].inverse, FREQ_BASE, cases[ci].freq_scale,
                  cases[ci].ext_factor, ATTN_FACTOR, BETA_FAST, BETA_SLOW,
                  EPS) != 0,
              "head_rms_rope: call");
        CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0,
              "head_rms_rope: read");

        for (int t = 0; t < N_TOK; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                const int base = (t * N_HEAD + h) * HEAD_DIM;
                memcpy(want_row, &x[base], sizeof(want_row));
                oracle_head_rms_norm_inplace(want_row, 1, HEAD_DIM, EPS);
                oracle_rope_tail_row(want_row, HEAD_DIM, N_ROT, POS0 + (uint32_t)t,
                                     N_CTX_ORIG, FREQ_BASE, cases[ci].freq_scale,
                                     cases[ci].ext_factor, ATTN_FACTOR,
                                     BETA_FAST, BETA_SLOW, cases[ci].inverse);
                for (int c = 0; c < HEAD_DIM; c++) {
                    CHECK_CLOSE(got[base + c], want_row[c], TOL,
                                "head_rms_rope: value mismatch");
                }
            }
        }

        ds4_gpu_tensor_free(tx);
    }

    /* An odd n_rot must be rejected. */
    {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)TOTAL * sizeof(float));
        CHECK(t != NULL, "head_rms_rope: odd n_rot allocation failed");
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  t, N_TOK, N_HEAD, HEAD_DIM, N_ROT - 1, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: odd n_rot must be rejected");

        /* n_rot > head_dim must also be rejected. */
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  t, N_TOK, N_HEAD, HEAD_DIM, HEAD_DIM + 2, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: n_rot > head_dim must be rejected");
        ds4_gpu_tensor_free(t);
    }

    /* A tensor smaller than n_tok * n_head * head_dim floats must be
     * rejected. */
    {
        ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(float) * 4);
        CHECK(small != NULL, "head_rms_rope: small allocation failed");
        CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                  small, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0, N_CTX_ORIG,
                  false, FREQ_BASE, 1.0f, 0.0f, 1.0f, BETA_FAST, BETA_SLOW,
                  EPS) == 0,
              "head_rms_rope: undersized tensor must be rejected");
        ds4_gpu_tensor_free(small);
    }

    fprintf(stderr, "  test_head_rms_norm_rope_tail OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_rms_norm_plain() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_plain_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_weight() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rms_norm_weight_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_add_rms_norm_weight() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_head_rms_norm() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dsv4_qkv_rms_norm_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dsv4_qkv_rms_norm_rows_kv_rope() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rope_tail() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_rope_tail_decode_rows() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_head_rms_norm_rope_tail() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_norm_rope OK\n");
    return 0;
}
