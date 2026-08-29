/* Correctness tests for the SYCL E4M3 value table and nearest rounding in
 * sycl/ds4_sycl_fp8_kv.hpp, validated against host oracles transcribed from
 * ds4.c:3196-3205 (dsv4_e4m3fn_value_cpu) and ds4.c:3211-3235
 * (dsv4_e4m3fn_dequant_cpu).  Deliberately self-contained: it does not
 * include tests/test_sycl_kernels.c or any shared harness header, since a
 * later integration pass folds every SYCL test file onto one harness.
 * Needs no model file. */

#include <math.h>
#include <stdio.h>

/* Test-only hooks defined directly in sycl/ds4_sycl_fp8_kv.hpp.  They are
 * not part of the production ABI (not declared in ds4_gpu.h or
 * ds4_gpu_mgpu.h), so they are forward-declared here instead. */
extern float ds4_sycl_test_e4m3fn_value(int code);
extern float ds4_sycl_test_e4m3fn_dequant(float x);

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

/* Transcribed verbatim from ds4.c:3196-3205. */
static float dsv4_e4m3fn_value_cpu(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };

    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

/* Transcribed verbatim from ds4.c:3211-3235. */
static float dsv4_e4m3fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * dsv4_e4m3fn_value_cpu(best);
}

static int test_value_table_all_codes(void) {
    for (int code = 0; code <= 126; code++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "value table mismatch at code %d", code);
        CHECK_CLOSE(ds4_sycl_test_e4m3fn_value(code), dsv4_e4m3fn_value_cpu(code), 0.0, msg);
    }
    return 0;
}

static int test_value_table_subnormals(void) {
    /* exp field == 0: codes 0..7 (mant 0..7), value is mant * 2^-9. */
    for (int mant = 0; mant <= 7; mant++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "subnormal mismatch at mant %d", mant);
        CHECK_CLOSE(ds4_sycl_test_e4m3fn_value(mant), (float)mant * 0.001953125f, 0.0, msg);
    }
    return 0;
}

static int test_round_trip_all_codes(void) {
    for (int code = 0; code <= 126; code++) {
        const float value = ds4_sycl_test_e4m3fn_value(code);
        const float want = dsv4_e4m3fn_dequant_cpu(value);
        const float got = ds4_sycl_test_e4m3fn_dequant(value);
        char msg[80];
        snprintf(msg, sizeof(msg), "round trip mismatch at code %d (value %.9g)", code, (double)value);
        CHECK_CLOSE(got, want, 0.0, msg);
        /* Every representable magnitude must round-trip to itself exactly. */
        CHECK_CLOSE(got, value, 0.0, msg);
    }
    return 0;
}

static int test_saturation_above_448(void) {
    const float inputs[] = {448.0f, 500.0f, 1000.0f, 1.0e6f, 3.4e38f};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        CHECK_CLOSE(ds4_sycl_test_e4m3fn_dequant(inputs[i]), 448.0f, 0.0, "saturation above 448 failed");
    }
    return 0;
}

static int test_saturation_below_negative_448(void) {
    const float inputs[] = {-448.0f, -500.0f, -1000.0f, -1.0e6f, -3.4e38f};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        CHECK_CLOSE(ds4_sycl_test_e4m3fn_dequant(inputs[i]), -448.0f, 0.0, "saturation below -448 failed");
    }
    return 0;
}

static int test_negative_values(void) {
    for (int code = 1; code <= 126; code++) {
        const float value = ds4_sycl_test_e4m3fn_value(code);
        const float got = ds4_sycl_test_e4m3fn_dequant(-value);
        char msg[64];
        snprintf(msg, sizeof(msg), "negative value mismatch at code %d", code);
        CHECK_CLOSE(got, -value, 0.0, msg);
    }
    return 0;
}

static int test_negative_zero_maps_positive(void) {
    /* x < 0.0f is false for -0.0f in IEEE 754, so the sign branch takes
     * the positive path: dequant(-0.0f) must be +0.0f, not -0.0f. */
    const float got = ds4_sycl_test_e4m3fn_dequant(-0.0f);
    CHECK(got == 0.0f, "negative zero did not dequant to zero");
    CHECK(!signbit(got), "negative zero dequanted to a negative-signed zero");
    return 0;
}

static int test_exact_tie_odd_best(void) {
    /* Find adjacent codes (best, best+1) with best odd, take the exact
     * midpoint of their values as input, and confirm the ties-to-even
     * rule rounds up (since best+1 is then even). */
    int best = -1;
    for (int code = 1; code < 126; code += 2) {
        best = code;
        break;
    }
    CHECK(best > 0, "no odd code found below 126");
    const float lo_val = dsv4_e4m3fn_value_cpu(best);
    const float hi_val = dsv4_e4m3fn_value_cpu(best + 1);
    const float midpoint = lo_val + (hi_val - lo_val) * 0.5f;
    /* Confirm this really is an exact tie under the oracle's own
     * difference computation before asserting anything about it. */
    CHECK(fabsf(midpoint - lo_val) == fabsf(midpoint - hi_val), "constructed midpoint is not an exact tie (odd best)");
    const float want = dsv4_e4m3fn_dequant_cpu(midpoint);
    CHECK_CLOSE(want, hi_val, 0.0, "oracle did not round up on odd-best tie");
    CHECK_CLOSE(ds4_sycl_test_e4m3fn_dequant(midpoint), hi_val, 0.0, "tie with odd best did not round up to even neighbour");
    return 0;
}

static int test_exact_tie_even_best(void) {
    /* Same construction, but with best even: ties-to-even then keeps
     * best (the lower, even neighbour) instead of stepping up. */
    int best = -1;
    for (int code = 2; code < 126; code += 2) {
        best = code;
        break;
    }
    CHECK(best > 0, "no even code found below 126");
    const float lo_val = dsv4_e4m3fn_value_cpu(best);
    const float hi_val = dsv4_e4m3fn_value_cpu(best + 1);
    const float midpoint = lo_val + (hi_val - lo_val) * 0.5f;
    CHECK(fabsf(midpoint - lo_val) == fabsf(midpoint - hi_val), "constructed midpoint is not an exact tie (even best)");
    const float want = dsv4_e4m3fn_dequant_cpu(midpoint);
    CHECK_CLOSE(want, lo_val, 0.0, "oracle did not stay on even-best tie");
    CHECK_CLOSE(ds4_sycl_test_e4m3fn_dequant(midpoint), lo_val, 0.0, "tie with even best did not stay on even neighbour");
    return 0;
}

int main(void) {
    int failures = 0;

    failures += test_value_table_all_codes();
    failures += test_value_table_subnormals();
    failures += test_round_trip_all_codes();
    failures += test_saturation_above_448();
    failures += test_saturation_below_negative_448();
    failures += test_negative_values();
    failures += test_negative_zero_maps_positive();
    failures += test_exact_tie_odd_best();
    failures += test_exact_tie_even_best();

    if (failures == 0) {
        printf("test_sycl_fp8_kv: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_sycl_fp8_kv: %d test group(s) failed\n", failures);
    return 1;
}
