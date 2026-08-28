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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_add() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_kernels OK\n");
    return 0;
}
