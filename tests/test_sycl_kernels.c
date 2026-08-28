/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ds4.c line number cited.  Needs no model file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Kernels accumulate in a different order than the scalar oracle, so
 * exact equality is not required for reductions.  Elementwise ops should
 * still match to within a tight tolerance. */
#define CHECK_CLOSE(got, want, tol, msg)                                    \
    do {                                                                    \
        double d_ = fabs((double)(got) - (double)(want));                   \
        if (!(d_ <= (tol))) {                                               \
            fprintf(stderr, "FAIL: %s (got %.9g want %.9g delta %.3g)\n",   \
                    (msg), (double)(got), (double)(want), d_);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int test_add(void) {
    enum { N = 1024 };
    float a[N], b[N], c[N], got[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(N - i) * 0.25f;
        c[i] = (float)(i % 7) - 3.0f;
    }

    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(a));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(b));
    ds4_gpu_tensor *tc = ds4_gpu_tensor_alloc(sizeof(c));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(ta && tb && tc && to, "add: allocation failed");

    CHECK(ds4_gpu_tensor_write(ta, 0, a, sizeof(a)) != 0, "add: write a");
    CHECK(ds4_gpu_tensor_write(tb, 0, b, sizeof(b)) != 0, "add: write b");
    CHECK(ds4_gpu_tensor_write(tc, 0, c, sizeof(c)) != 0, "add: write c");

    CHECK(ds4_gpu_add_tensor(to, ta, tb, N) != 0, "add: ds4_gpu_add_tensor");
    memset(got, 0, sizeof(got));
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "add: read");
    for (int i = 0; i < N; i++) {
        /* Oracle: out = a + b, ds4.c:11631 and the other residual sites. */
        CHECK_CLOSE(got[i], a[i] + b[i], 1e-6, "add: value mismatch");
    }

    CHECK(ds4_gpu_add3_tensor(to, ta, tb, tc, N) != 0, "add3: ds4_gpu_add3_tensor");
    memset(got, 0, sizeof(got));
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "add3: read");
    for (int i = 0; i < N; i++) {
        /* Oracle: out = a + b + c.  NOTE: ds4 has no CPU implementation of
         * a three-term add; every CPU site does two sequential two-term
         * adds, and ds4_gpu_add3_tensor's only caller is the GLM decode
         * path (ds4.c:42284), which this project does not support.  This
         * is a synthetic-only oracle by necessity, recorded deliberately
         * rather than skipped. */
        CHECK_CLOSE(got[i], a[i] + b[i] + c[i], 1e-6, "add3: value mismatch");
    }

    /* Zero-length work is SUCCESS and returns nonzero, matching
     * rocm/ds4_rocm_misc_launch.cuh:3 `if (n == 0u) return 1;`. */
    CHECK(ds4_gpu_add_tensor(to, ta, tb, 0) != 0, "add: n=0 must succeed");

    /* Undersized output must be rejected, returning 0. */
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(16);
    CHECK(small != NULL, "add: small allocation failed");
    CHECK(ds4_gpu_add_tensor(small, ta, tb, N) == 0,
          "add: undersized output must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(tc);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_add OK\n");
    return 0;
}

/* Oracle mirrors swiglu at ds4.c:10556-10567 with silu at ds4.c:10546.
 * The GPU kernel additionally multiplies by `weight`, which the CPU
 * function has no parameter for; we implement it faithfully rather than
 * assuming callers always pass 1.0. */
static float oracle_swiglu(float g, float u, float clamp, float weight) {
    if (clamp > 1e-6f) {
        if (u >  clamp) u =  clamp;
        if (u < -clamp) u = -clamp;
        if (g >  clamp) g =  clamp;
    }
    float s = g / (1.0f + expf(-g));
    return s * u * weight;
}

static int test_swiglu(void) {
    enum { N = 512 };
    float gate[N], up[N], got[N];
    for (int i = 0; i < N; i++) {
        gate[i] = ((float)i / (float)N) * 8.0f - 4.0f;
        up[i]   = ((float)(N - i) / (float)N) * 6.0f - 3.0f;
    }

    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc(sizeof(gate));
    ds4_gpu_tensor *tu = ds4_gpu_tensor_alloc(sizeof(up));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tg && tu && to, "swiglu: allocation failed");
    CHECK(ds4_gpu_tensor_write(tg, 0, gate, sizeof(gate)) != 0, "swiglu: write gate");
    CHECK(ds4_gpu_tensor_write(tu, 0, up, sizeof(up)) != 0, "swiglu: write up");

    /* Unclamped, unit weight. */
    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, N, 0.0f, 1.0f) != 0, "swiglu: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "swiglu: read");
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got[i], oracle_swiglu(gate[i], up[i], 0.0f, 1.0f), 1e-5,
                    "swiglu: unclamped mismatch");
    }

    /* Clamped, non-unit weight.  A weight other than 1.0 is what proves
     * the parameter is actually applied rather than ignored. */
    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, N, 2.0f, 0.75f) != 0, "swiglu: clamped call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "swiglu: clamped read");
    for (int i = 0; i < N; i++) {
        CHECK_CLOSE(got[i], oracle_swiglu(gate[i], up[i], 2.0f, 0.75f), 1e-5,
                    "swiglu: clamped mismatch");
    }

    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, 0, 0.0f, 1.0f) != 0,
          "swiglu: n=0 must succeed");

    ds4_gpu_tensor_free(tg);
    ds4_gpu_tensor_free(tu);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_swiglu OK\n");
    return 0;
}

/* Oracle mirrors the gate loop in output_hc_head_one, ds4.c:14018-14020.
 * IMPORTANT NUMERICAL NOTE: ds4's CPU path uses sigmoid_stable
 * (ds4.c:10419-10427), which branches on the sign of x to avoid exp()
 * overflow.  The GPU kernels on every backend use the unstable
 * 1/(1+exp(-z)) form.  The two are mathematically equal and differ only
 * for large-magnitude negative z.  This oracle uses the STABLE form and
 * the test inputs are kept in a bounded range so the difference cannot
 * appear.  If a future change feeds large negative values here and this
 * test starts failing, that is a pre-existing CPU/GPU divergence, not a
 * SYCL regression. */
static float oracle_sigmoid_stable(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static int test_output_hc_weights(void) {
    enum { N_HC = 4, N_TOK = 64, N = N_HC * N_TOK };
    const float eps = 0.01f;
    const size_t row_bytes = (size_t)N_HC * sizeof(float);
    float pre[N], got[N];
    for (int i = 0; i < N; i++) pre[i] = ((float)i / (float)N) * 4.0f - 2.0f;

    /* Model buffer: one f32 scale, then n_hc f32 base values. */
    float model[1 + N_HC];
    model[0] = 1.25f;
    for (int h = 0; h < N_HC; h++) model[1 + h] = (float)h * 0.5f - 0.75f;

    ds4_gpu_tensor *tp = ds4_gpu_tensor_alloc(sizeof(pre));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(tp && to, "hc_weights: allocation failed");
    CHECK(ds4_gpu_tensor_write(tp, 0, pre, sizeof(pre)) != 0, "hc_weights: write pre");

    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp, model, sizeof(model),
                                           0, sizeof(float), N_HC, eps) != 0,
          "hc_weights: call");
    CHECK(ds4_gpu_tensor_read(to, 0, got, sizeof(got)) != 0, "hc_weights: read");
    for (int i = 0; i < N; i++) {
        float z = pre[i] * model[0] + model[1 + (i % N_HC)];
        CHECK_CLOSE(got[i], oracle_sigmoid_stable(z) + eps, 1e-5,
                    "hc_weights: value mismatch");
    }

    /* A scale offset past the end of the model buffer must be rejected. */
    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp, model, sizeof(model),
                                           sizeof(model), sizeof(float),
                                           N_HC, eps) == 0,
          "hc_weights: out-of-range scale offset must be rejected");

    /* An out tensor whose byte size is not an exact multiple of
     * n_hc * sizeof(float) must be rejected: it cannot hold a whole
     * number of HC rows.  rocm/ds4_rocm_hc_output_launch.cuh:213-218 is
     * the authority for this guard. */
    ds4_gpu_tensor *to_unaligned = ds4_gpu_tensor_alloc(row_bytes + 4);
    CHECK(to_unaligned != NULL, "hc_weights: unaligned allocation failed");
    CHECK(ds4_gpu_output_hc_weights_tensor(to_unaligned, tp, model,
                                           sizeof(model), 0, sizeof(float),
                                           N_HC, eps) == 0,
          "hc_weights: non-row-aligned out size must be rejected");
    ds4_gpu_tensor_free(to_unaligned);

    /* A pre tensor smaller than out must be rejected: there is not
     * enough input to cover every output element. */
    ds4_gpu_tensor *tp_small = ds4_gpu_tensor_alloc(sizeof(pre) / 2);
    CHECK(tp_small != NULL, "hc_weights: small pre allocation failed");
    CHECK(ds4_gpu_output_hc_weights_tensor(to, tp_small, model, sizeof(model),
                                           0, sizeof(float), N_HC, eps) == 0,
          "hc_weights: undersized pre must be rejected");
    ds4_gpu_tensor_free(tp_small);

    ds4_gpu_tensor_free(tp);
    ds4_gpu_tensor_free(to);
    fprintf(stderr, "  test_output_hc_weights OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_add() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_swiglu() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_output_hc_weights() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_kernels OK\n");
    return 0;
}
