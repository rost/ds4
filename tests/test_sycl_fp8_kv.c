/* Correctness tests for sycl/ds4_sycl_fp8_kv.hpp: the E4M3 value table and
 * nearest rounding helpers, validated against host oracles transcribed from
 * ds4.c:3196-3205 (dsv4_e4m3fn_value_cpu) and ds4.c:3211-3235
 * (dsv4_e4m3fn_dequant_cpu); and the production entry point
 * ds4_gpu_dsv4_fp8_kv_quantize_tensor, validated against
 * dsv4_fp8_kv_quantize_row_inplace_cpu (ds4.c:3241-3261) reached through
 * the normal ds4_gpu.h/ds4_gpu_mgpu.h ABI.  Deliberately self-contained: it
 * does not include tests/test_sycl_kernels.c or any shared harness header,
 * since a later integration pass folds every SYCL test file onto one
 * harness.  Needs no model file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

/* Transcribed verbatim from ds4.c:3241-3261.  NOTE: the inner loops run a
 * fixed `for (i = 0; i < 64; i++)` with no bound against n_nope, so this is
 * only a valid oracle for a row whose n_nope is an exact multiple of 64.
 * DSV4's real shapes are always such a multiple; this divergence from the
 * GPU kernel (which correctly guards every access with off + tid < n_nope)
 * is deliberate upstream behaviour, not a bug for this test to work
 * around.  A partial final group is covered separately below by
 * oracle_fp8_kv_group_bounded, which is NOT a transcription of anything in
 * ds4.c. */
static void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
        }
    }
}

/* Correctly bounded by an explicit group length, unlike the unbounded
 * transcription above.  Used only for the partial-final-group test, where
 * a group shorter than 64 makes the unbounded transcription invalid (it
 * would read past n_nope).  DSV4 never produces a partial final group in
 * practice, so ds4.c has no bounded variant of its own to transcribe. */
static void oracle_fp8_kv_group_bounded(float *group, uint32_t len) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < len; i++) {
        const float av = fabsf(group[i]);
        if (av > amax) amax = av;
    }
    if (amax < 1.0e-4f) amax = 1.0e-4f;
    const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
    for (uint32_t i = 0; i < len; i++) {
        float v = group[i] / scale;
        if (v > 448.0f) v = 448.0f;
        if (v < -448.0f) v = -448.0f;
        group[i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
    }
}

/* n_tok > 1 and n_nope > 64 so both grid dimensions (row, group) are
 * exercised.  n_nope is an exact multiple of 64, so the unbounded
 * transcribed oracle applies directly to a copy of each full row. */
static int test_multi_row_multi_group(void) {
    enum { N_TOK = 3, N_ROT = 8, N_NOPE = 128, HEAD_DIM = N_NOPE + N_ROT };
    float x[N_TOK * HEAD_DIM];
    float want[N_TOK * HEAD_DIM];
    float got[N_TOK * HEAD_DIM];

    for (int r = 0; r < N_TOK; r++) {
        for (int i = 0; i < HEAD_DIM; i++) {
            x[r * HEAD_DIM + i] = (float)(((i * 7 + r * 13) % 23) - 11) * (0.3f + 0.1f * (float)r);
        }
    }
    memcpy(want, x, sizeof(x));
    for (int r = 0; r < N_TOK; r++) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(&want[r * HEAD_DIM], HEAD_DIM, N_ROT);
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "multi_row_multi_group: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "multi_row_multi_group: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "multi_row_multi_group: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "multi_row_multi_group: read");

    for (int i = 0; i < N_TOK * HEAD_DIM; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "multi_row_multi_group: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 1.0e-2, msg);
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_multi_row_multi_group OK\n");
    return 0;
}

/* The RoPE tail (indices n_nope..head_dim-1) must be left byte-for-byte
 * untouched: pre-fill it with a sentinel pattern that is not the value 0.0f
 * a stub or a zero-initialised buffer would produce, and require exact
 * equality after the call. */
static int test_rope_half_untouched(void) {
    enum { N_TOK = 2, N_ROT = 6, N_NOPE = 64, HEAD_DIM = N_NOPE + N_ROT };
    float x[N_TOK * HEAD_DIM];
    float got[N_TOK * HEAD_DIM];

    for (int r = 0; r < N_TOK; r++) {
        for (int i = 0; i < N_NOPE; i++) {
            x[r * HEAD_DIM + i] = (float)((i % 9) - 4) * 0.2f;
        }
        for (int i = N_NOPE; i < HEAD_DIM; i++) {
            x[r * HEAD_DIM + i] = 12345.6789f + (float)(r * HEAD_DIM + i);
        }
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "rope_half_untouched: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "rope_half_untouched: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "rope_half_untouched: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "rope_half_untouched: read");

    for (int r = 0; r < N_TOK; r++) {
        for (int i = N_NOPE; i < HEAD_DIM; i++) {
            CHECK_CLOSE(got[r * HEAD_DIM + i], x[r * HEAD_DIM + i], 0.0,
                        "rope_half_untouched: RoPE tail element was modified");
        }
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_rope_half_untouched OK\n");
    return 0;
}

/* Two groups in the same row with amax values several orders of magnitude
 * apart: group 0 stays tiny (amax well under 0.01), group 1 is huge (amax
 * in the tens of thousands).  A kernel that reduced across the whole row
 * instead of per group would compute one scale tuned for group 1 and apply
 * it to group 0 too, flushing group 0 to zero.  This is verified directly
 * below by computing that wrong per-row result on the host and confirming
 * it disagrees with the correct per-group expectation. */
static int test_per_group_independent_scaling(void) {
    enum { N_TOK = 1, N_ROT = 0, N_NOPE = 128, HEAD_DIM = N_NOPE + N_ROT };
    float x[HEAD_DIM];
    float want[HEAD_DIM];
    float got[HEAD_DIM];

    for (int i = 0; i < 64; i++) x[i] = (float)((i % 7) - 3) * 0.003f;
    for (int i = 64; i < 128; i++) x[i] = (float)((i % 7) - 3) * 15000.0f;
    memcpy(want, x, sizeof(x));
    dsv4_fp8_kv_quantize_row_inplace_cpu(want, HEAD_DIM, N_ROT);

    /* Confirm this construction really does force per-group scales apart:
     * hand-compute the WRONG single-row scale (tuned by the row's overall
     * amax, i.e. group 1's amax) and what applying it to group 0 would
     * produce, then require that it disagrees with the correct per-group
     * expectation for at least one element of group 0. */
    float row_amax = 0.0f;
    for (int i = 0; i < HEAD_DIM; i++) {
        const float av = fabsf(x[i]);
        if (av > row_amax) row_amax = av;
    }
    const float wrong_row_scale = ldexpf(1.0f, (int)ceilf(log2f(row_amax / 448.0f)));
    int any_group0_disagrees = 0;
    for (int i = 0; i < 64; i++) {
        float v = x[i] / wrong_row_scale;
        if (v > 448.0f) v = 448.0f;
        if (v < -448.0f) v = -448.0f;
        const float wrong = dsv4_e4m3fn_dequant_cpu(v) * wrong_row_scale;
        if (fabsf(wrong - want[i]) > 1.0e-6f) any_group0_disagrees = 1;
    }
    CHECK(any_group0_disagrees,
          "per_group_independent_scaling: test data does not actually force per-group vs per-row apart");

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "per_group_independent_scaling: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "per_group_independent_scaling: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "per_group_independent_scaling: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "per_group_independent_scaling: read");

    /* Tolerance must be tight relative to group 0's magnitude (amax under
     * 0.01): a per-row-scale bug flushes group 0 to zero, a discrepancy of
     * about 0.009, which a coarser tolerance sized for group 1's
     * tens-of-thousands magnitude would silently swallow. */
    for (int i = 0; i < HEAD_DIM; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "per_group_independent_scaling: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 1.0e-6, msg);
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_per_group_independent_scaling OK\n");
    return 0;
}

/* An all-zero group: amax is 0, so the 1.0e-4f floor must be applied
 * (otherwise log2f(0.0f / 448.0f) is -inf and the scale becomes 0 or NaN).
 * The correct output is exactly zero.
 *
 * NOTE for a future reader who ablates the floor to check this test
 * catches it: it will not, and this is not a gap in the test.  The clamp
 * immediately below the scale computation is done via fmax/fmin, whose
 * IEEE-754 semantics return the non-NaN operand whenever exactly one side
 * is NaN.  For an all-zero group, removing the floor makes v/scale
 * evaluate to 0/0 = NaN, but fmax(-448, NaN) then yields -448 (not NaN),
 * so the value survives the clamp as a finite -448, dequantises to a
 * proper representable magnitude, and is finally multiplied by scale
 * (itself 0, not NaN, since exp2(ceil(log2(0))) = exp2(-inf) = 0) to give
 * exactly -0.0f: verified directly against the device with and without
 * the floor during development, both producing bit pattern 0x80000000.
 * A small-but-nonzero amax well under the floor was also tried as an
 * alternative probe and found equally unable to discriminate, for a
 * different, purely mathematical reason: E4M3 is itself a floating-point
 * format, so a power-of-two scale change shifts only the exponent and
 * leaves the mantissa's relative precision unaffected, meaning the
 * floored and unfloored reconstructions converge to nearly the same
 * value regardless (confirmed by direct device comparison during
 * development).  This test is kept because the all-zero-input scenario
 * (e.g. padding) is a real case worth locking down regardless of which
 * exact line is providing the guarantee. */
static int test_amax_floor(void) {
    enum { N_TOK = 1, N_ROT = 0, N_NOPE = 64, HEAD_DIM = N_NOPE + N_ROT };
    float x[HEAD_DIM];
    float got[HEAD_DIM];
    memset(x, 0, sizeof(x));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "amax_floor: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "amax_floor: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "amax_floor: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "amax_floor: read");

    for (int i = 0; i < HEAD_DIM; i++) {
        CHECK(!isnan(got[i]), "amax_floor: output is NaN");
        CHECK_CLOSE(got[i], 0.0f, 0.0, "amax_floor: output is not exactly zero");
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_amax_floor OK\n");
    return 0;
}

/* n_nope = 100 is not a multiple of 64: one full group (0..63) and one
 * partial final group of 36 elements (64..99).  The unbounded transcribed
 * oracle cannot compute the partial group correctly, so
 * oracle_fp8_kv_group_bounded is used for both groups instead.  Elements
 * at and beyond n_nope (the RoPE tail, 100..107) must be untouched. */
static int test_partial_final_group(void) {
    enum { N_TOK = 1, N_ROT = 8, N_NOPE = 100, HEAD_DIM = N_NOPE + N_ROT };
    float x[HEAD_DIM];
    float want[HEAD_DIM];
    float got[HEAD_DIM];

    for (int i = 0; i < 64; i++) x[i] = (float)((i % 9) - 4) * 0.7f;
    for (int i = 64; i < N_NOPE; i++) x[i] = (float)((i % 5) - 2) * 250.0f;
    for (int i = N_NOPE; i < HEAD_DIM; i++) x[i] = 777.777f + (float)i;
    memcpy(want, x, sizeof(x));
    oracle_fp8_kv_group_bounded(&want[0], 64);
    oracle_fp8_kv_group_bounded(&want[64], N_NOPE - 64);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "partial_final_group: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "partial_final_group: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "partial_final_group: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "partial_final_group: read");

    for (int i = 0; i < N_NOPE; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "partial_final_group: mismatch at %d", i);
        CHECK_CLOSE(got[i], want[i], 1.0e-2, msg);
    }
    for (int i = N_NOPE; i < HEAD_DIM; i++) {
        CHECK_CLOSE(got[i], x[i], 0.0, "partial_final_group: element at/beyond n_nope was modified");
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_partial_final_group OK\n");
    return 0;
}

/* Probes the spec 6b uninitialised-scratch hazard directly, rather than
 * relying on the tensor's own contents: a warm-up call first runs a FULL,
 * huge-magnitude single group (every lane in range), so on hardware/driver
 * combinations that reuse the same physical local memory across
 * back-to-back submissions of this kernel, the scratch slots are left
 * holding large nonzero values.  An immediately following probe call then
 * uses a PARTIAL group with small in-range values: a kernel that skips the
 * explicit zero write for out-of-range lanes would fold that stale large
 * value into the max reduction and compute a wildly oversized scale,
 * producing a result distinguishable from correct by construction, not by
 * coincidence.  Seeding the tensor itself cannot force this (the kernel's
 * read of xr[off+tid] is independently guarded by the same range check, so
 * tensor contents past n_nope are never read); only the physical memory
 * left behind by a prior submission can leak in.  Whether this hardware
 * and driver stack actually reuses that memory across submissions is
 * implementation-defined, so a PASS here is not proof the explicit zero
 * write is unnecessary -- only a controlled ablation of that write is
 * authoritative for that question. */
static int test_scratch_hazard_probe(void) {
    enum { N_TOK = 1, N_ROT = 0, HEAD_DIM = 64 };
    float warm[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) warm[i] = 300000.0f;

    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(warm));
    CHECK(tw != NULL, "scratch_hazard_probe: warm-up allocation failed");
    CHECK(ds4_gpu_tensor_write(tw, 0, warm, sizeof(warm)) != 0,
          "scratch_hazard_probe: warm-up write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tw, N_TOK, HEAD_DIM, N_ROT) != 0,
          "scratch_hazard_probe: warm-up call");
    ds4_gpu_tensor_free(tw);

    enum { N_ROT_PROBE = 32, N_NOPE_PROBE = HEAD_DIM - N_ROT_PROBE };
    float x[HEAD_DIM];
    float want[HEAD_DIM];
    float got[HEAD_DIM];
    for (int i = 0; i < N_NOPE_PROBE; i++) x[i] = (float)((i % 5) - 2) * 0.01f;
    for (int i = N_NOPE_PROBE; i < HEAD_DIM; i++) x[i] = 999.0f + (float)i;
    memcpy(want, x, sizeof(x));
    oracle_fp8_kv_group_bounded(want, N_NOPE_PROBE);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "scratch_hazard_probe: probe allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "scratch_hazard_probe: probe write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT_PROBE) != 0,
          "scratch_hazard_probe: probe call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "scratch_hazard_probe: probe read");

    for (int i = 0; i < N_NOPE_PROBE; i++) {
        char msg[80];
        snprintf(msg, sizeof(msg), "scratch_hazard_probe: mismatch at %d (possible scratch leak)", i);
        CHECK_CLOSE(got[i], want[i], 1.0e-2, msg);
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_scratch_hazard_probe OK\n");
    return 0;
}

/* The inner reduction loop's barrier (rocm/ds4_rocm_fp8_kv.cuh:20-23) is
 * OUTSIDE the `if`, so every lane must reach it on every iteration.
 * Missing it is a genuine cross-sub-group race once the first fold
 * (stride 32) narrows work to lanes 0..31, which on this hardware's
 * 32-wide sub-groups is where correctness would depend on the barrier: a
 * single small launch does not reliably expose the race (observed
 * passing during development on this hardware/driver even with the
 * barrier removed), but a launch large enough to run many work-groups
 * concurrently across the whole device does, reproducibly, in ad hoc
 * testing during development (roughly 2% of elements wrong across
 * repeated trials).  This shape (N_TOK * groups work-groups, all
 * independent rows/groups) is chosen to reach that concurrency
 * reliably. */
static int test_reduction_barrier_stress(void) {
    enum { N_TOK = 4000, N_ROT = 0, HEAD_DIM = 1024 };
    static float x[N_TOK * HEAD_DIM];
    static float want[N_TOK * HEAD_DIM];
    static float got[N_TOK * HEAD_DIM];

    unsigned seed = 12345u;
    for (long i = 0; i < (long)N_TOK * HEAD_DIM; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((float)((seed >> 8) % 20001u) - 10000.0f) * 0.001f;
    }
    memcpy(want, x, sizeof(x));
    for (int r = 0; r < N_TOK; r++) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(&want[(long)r * HEAD_DIM], HEAD_DIM, N_ROT);
    }

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "reduction_barrier_stress: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "reduction_barrier_stress: write");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, N_TOK, HEAD_DIM, N_ROT) != 0,
          "reduction_barrier_stress: call");
    CHECK(ds4_gpu_tensor_read(tx, 0, got, sizeof(got)) != 0, "reduction_barrier_stress: read");

    long mismatches = 0;
    for (long i = 0; i < (long)N_TOK * HEAD_DIM; i++) {
        if (fabs((double)got[i] - (double)want[i]) > 1.0e-2) mismatches++;
    }
    if (mismatches != 0) {
        fprintf(stderr, "FAIL: reduction_barrier_stress: %ld/%ld elements mismatched\n",
                mismatches, (long)N_TOK * HEAD_DIM);
        return 1;
    }

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_reduction_barrier_stress OK\n");
    return 0;
}

/* The three zero-work successes, each returning nonzero:
 *   - n_tok == 0
 *   - head_dim == 0 (together with n_rot == 0: see the validation-order
 *     rejection test below for why n_rot must also be 0 here)
 *   - n_nope == 0 (n_rot == head_dim, both nonzero) */
static int test_zero_work_successes(void) {
    enum { BUF = 64 };
    float x[BUF];
    memset(x, 0, sizeof(x));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "zero_work_successes: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "zero_work_successes: write");

    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, 0, 8, 2) != 0,
          "zero_work_successes: n_tok=0 must succeed");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, 3, 0, 0) != 0,
          "zero_work_successes: head_dim=0,n_rot=0 must succeed");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, 2, 16, 16) != 0,
          "zero_work_successes: n_nope=0 (n_rot==head_dim) must succeed");

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_zero_work_successes OK\n");
    return 0;
}

/* The rejections, each returning 0.  The head_dim=0,n_rot=1 case is the
 * validation-order subtlety: the launcher checks n_rot > head_dim BEFORE
 * the zero-work early-outs, so head_dim==0 together with n_rot>0 is a
 * rejection (1 > 0), not a zero-work success, even though head_dim==0
 * alone (with n_rot==0) succeeds in test_zero_work_successes above. */
static int test_rejections(void) {
    enum { N_TOK = 4, HEAD_DIM = 16 };
    float x[N_TOK * HEAD_DIM];
    memset(x, 0, sizeof(x));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    CHECK(tx != NULL, "rejections: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0, "rejections: write");

    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, 1, 4, 5) == 0,
          "rejections: n_rot > head_dim must be rejected");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tx, 2, 0, 1) == 0,
          "rejections: head_dim=0 with n_rot>0 must be rejected, not treated as zero-work");

    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(sizeof(x) / 2);
    CHECK(small != NULL, "rejections: undersized allocation failed");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(small, N_TOK, HEAD_DIM, 0) == 0,
          "rejections: x too small for n_tok * head_dim floats must be rejected");
    ds4_gpu_tensor_free(small);

    ds4_gpu_tensor_free(tx);
    fprintf(stderr, "  test_rejections OK\n");
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

    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    failures += test_multi_row_multi_group();
    failures += test_rope_half_untouched();
    failures += test_per_group_independent_scaling();
    failures += test_amax_floor();
    failures += test_partial_final_group();
    failures += test_scratch_hazard_probe();
    failures += test_reduction_barrier_stress();
    failures += test_zero_work_successes();
    failures += test_rejections();
    ds4_gpu_cleanup();

    if (failures == 0) {
        printf("test_sycl_fp8_kv: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_sycl_fp8_kv: %d test group(s) failed\n", failures);
    return 1;
}
