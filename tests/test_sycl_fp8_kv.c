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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Test-only hooks defined directly in sycl/ds4_sycl_fp8_kv.hpp.  They are
 * not part of the production ABI (not declared in ds4_gpu.h or
 * ds4_gpu_mgpu.h), so they are forward-declared here instead. */
extern float ds4_sycl_test_e4m3fn_value(int code);
extern float ds4_sycl_test_e4m3fn_dequant(float x);

/* ds4_sycl_test_hip_round_f16_bits calls the header's ported
 * sycl_f32_to_f16_bits_hip_round device function directly: it is pure
 * integer bit manipulation, so it needs no device queue and gives the
 * same result whether it runs on the host (as here) or on a device.
 *
 * ds4_sycl_test_sycl_half_encode_bits instead runs an actual one-element
 * device kernel doing `sycl::bit_cast<uint16_t>((sycl::half)f)`: this
 * exists only to empirically characterise the ACTUAL device hardware's
 * float-to-half rounding, which is what matters for deciding whether
 * production code may use sycl::half for this direction.  (A host-side
 * sycl::half cast was tried during development and found to behave
 * differently again, apparently truncating rather than rounding in the
 * subnormal range; that is irrelevant to production, which only ever
 * runs this conversion on the device, so only the device path is worth
 * testing here.)  Requires ds4_gpu_init() to have already selected a
 * device; returns 0 if it could not run. */
extern uint16_t ds4_sycl_test_hip_round_f16_bits(float f);
extern int ds4_sycl_test_sycl_half_encode_bits(float f, uint16_t *out_bits);

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

/* =========================================================================
 * ds4_gpu_store_raw_kv_batch_tensor: the raw KV ring-buffer batch store.
 * Ported from store_raw_kv_batch_kernel (rocm/ds4_rocm_fp8_kv.cuh:31-40)
 * and its launcher ds4_gpu_store_raw_kv_batch_tensor
 * (rocm/ds4_rocm_attention_launch.cuh:19-26).
 * ========================================================================= */

/* Host transcription, verbatim, of f32_to_f16_bits_hip_round
 * (rocm/ds4_rocm_common.cuh:372-390): the hand-rolled F32-to-F16 bit
 * encoder used by the store kernel.  Both its normal-exponent and
 * subnormal paths make their rounding decision from a SINGLE discarded
 * bit (e.g. `mant & 0x1000` in the normal path), with no check of
 * whether the remaining discarded bits are all zero.  For any input
 * whose discarded fraction is not exactly one half, that single bit is
 * enough to reproduce correct round-to-nearest: the bit is 1 exactly
 * when the discarded fraction is >= 0.5, and testing it alone cannot
 * tell an exact half apart from something slightly larger.  At an exact
 * tie, this rounds up unconditionally ("round half up"), which is a
 * different rule from IEEE round-to-nearest-even. */
static uint16_t f32_to_f16_bits_hip_round_oracle(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

/* Standard IEEE-754 half-to-float widening.  Unlike the encode oracle
 * above, this is not a transcription of a specific proprietary source:
 * rocm/ds4_rocm_common.cuh's f16_bits_to_f32 is a one-line call to the
 * __half2float/__ushort_as_half compiler intrinsics, with no exposed
 * bit-manipulation source to transcribe.  Every F16 value (subnormals
 * included) is exactly representable in F32, so there is no rounding
 * decision in this direction and any conformant widen implementation
 * must agree with any other; this one exists only to compute expected
 * stored values for the tests below. */
static float f16_bits_to_f32_oracle(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)((h >> 10) & 0x1fu);
    uint32_t mant = (uint32_t)(h & 0x3ffu);
    uint32_t bits;
    if (exp == 0u && mant == 0u) {
        bits = sign;
    } else if (exp == 0u) {
        /* Subnormal half: normalise by shifting the mantissa left until
         * its implicit leading bit reaches position 10, tracking how
         * many shifts that took. */
        int shifts = 0;
        while ((mant & 0x400u) == 0u) {
            mant <<= 1;
            shifts++;
        }
        mant &= 0x3ffu;
        bits = sign | ((113u - (uint32_t)shifts) << 23) | (mant << 13);
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Sanity-checks f16_bits_to_f32_oracle itself against a handful of
 * hand-known bit patterns, independent of any device: every branch
 * (zero, subnormal, normal, infinity) is exercised at least once. */
static int test_f16_decode_oracle_self_check(void) {
    CHECK_CLOSE(f16_bits_to_f32_oracle(0x0000u), 0.0f, 0.0, "decode oracle: +0 mismatch");
    CHECK(!signbit(f16_bits_to_f32_oracle(0x0000u)), "decode oracle: +0 has sign bit set");
    /* Checked via the raw bit pattern rather than signbit(): under this
     * build's -ffast-math (-fno-signed-zeros), the compiler is entitled
     * to canonicalise a compile-time-constant -0.0f into +0.0f, which
     * made signbit() on this particular case an optimiser artifact
     * rather than a real test of f16_bits_to_f32_oracle's logic. */
    {
        const float negzero = f16_bits_to_f32_oracle(0x8000u);
        uint32_t negzero_bits;
        memcpy(&negzero_bits, &negzero, sizeof(negzero_bits));
        CHECK(negzero_bits == 0x80000000u, "decode oracle: -0 missing sign bit");
    }
    CHECK_CLOSE(f16_bits_to_f32_oracle(0x3c00u), 1.0f, 0.0, "decode oracle: 1.0 mismatch");
    CHECK_CLOSE(f16_bits_to_f32_oracle(0x4248u), 3.140625f, 0.0, "decode oracle: normal value mismatch");
    CHECK_CLOSE(f16_bits_to_f32_oracle(0x0001u), 5.9604644775390625e-08f, 0.0, "decode oracle: min subnormal mismatch");
    CHECK(isinf(f16_bits_to_f32_oracle(0x7c00u)) && f16_bits_to_f32_oracle(0x7c00u) > 0.0f, "decode oracle: +inf mismatch");
    CHECK(isinf(f16_bits_to_f32_oracle(0xfc00u)) && f16_bits_to_f32_oracle(0xfc00u) < 0.0f, "decode oracle: -inf mismatch");
    fprintf(stderr, "  test_f16_decode_oracle_self_check OK\n");
    return 0;
}

/* Confirms the header's ported sycl_f32_to_f16_bits_hip_round function
 * (exposed here as ds4_sycl_test_hip_round_f16_bits) reproduces this
 * file's own oracle transcription exactly, across subnormal, near-max,
 * overflow and mantissa-carry cases.  Since both are the same algorithm,
 * any mismatch here means the port itself has a transcription bug, not a
 * hardware rounding difference. */
static int test_f16_ported_encoder_matches_oracle(void) {
    const float values[] = {
        6.0e-8f, 1.0e-7f, 65504.0f, 65496.0f, 65520.0f, 100000.0f,
        1.99951171875f, 3.14159f, -123.456f, 0.1f, 0.0f, -0.0f,
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        char msg[80];
        snprintf(msg, sizeof(msg), "ported encoder mismatch at values[%zu]=%.9g", i, (double)values[i]);
        CHECK(ds4_sycl_test_hip_round_f16_bits(values[i]) == f32_to_f16_bits_hip_round_oracle(values[i]), msg);
    }
    fprintf(stderr, "  test_f16_ported_encoder_matches_oracle OK\n");
    return 0;
}

/* The genuinely open question: does the ACTUAL DEVICE'S sycl::half
 * float-to-half conversion agree with f32_to_f16_bits_hip_round_oracle?
 * Answered empirically here, not assumed, via ds4_sycl_test_sycl_half_encode_bits
 * which runs the conversion in a real device kernel.  Covers every
 * category the brief asks for except the tie (handled separately below,
 * since it needs its own tie-construction verification first): subnormal
 * F16 results, values at/near the F16 maximum, overflow to F16 infinity,
 * and a value that rounds up into the next exponent. */
static int test_f16_sycl_half_matches_device_on_non_tie_values(void) {
    const float values[] = {
        6.0e-8f,           /* subnormal F16 result */
        1.0e-7f,           /* subnormal F16 result, two ULPs */
        65504.0f,          /* exactly the F16 maximum */
        65496.0f,          /* near the F16 maximum, still finite */
        65520.0f,          /* overflows to F16 infinity */
        100000.0f,         /* far beyond F16 range: infinity */
        1.99951171875f,    /* rounds up, carrying into the next exponent (-> 2.0) */
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        uint16_t got = 0;
        char msg[96];
        snprintf(msg, sizeof(msg), "sycl_half_encode_bits call failed for values[%zu]=%.9g", i, (double)values[i]);
        CHECK(ds4_sycl_test_sycl_half_encode_bits(values[i], &got) != 0, msg);
        uint16_t want = f32_to_f16_bits_hip_round_oracle(values[i]);
        snprintf(msg, sizeof(msg), "sycl::half (device) vs oracle mismatch at values[%zu]=%.9g", i, (double)values[i]);
        CHECK(got == want, msg);
    }
    fprintf(stderr, "  test_f16_sycl_half_matches_device_on_non_tie_values OK\n");
    return 0;
}

/* The exact-tie case, constructed by hand from the normal-exponent path's
 * bit layout: `half = sign | (exp<<10) | (mant>>13); if (mant & 0x1000)
 * half++;` discards the low 13 bits of a 23-bit F32 mantissa.  An exact
 * tie needs those 13 discarded bits to be exactly `1000000000000` (the
 * top discarded bit set, every lower discarded bit zero).  Taking F32
 * exponent field 127 (value in [1,2)) and mantissa 0x001000 (bit 12 set,
 * bits 0-11 and the kept bits 13-22 all zero) gives F32 bit pattern
 * 0x3F801000, i.e. f = 1.00048828125, an exact dyadic value representable
 * without rounding as a float literal.  Its two F16 neighbours are 0x3c00
 * (1.0) and 0x3c01 (1.0009765625), also exact dyadic values.
 *
 * Empirically confirmed on this hardware (Intel Arc A770, Level Zero,
 * oneAPI 2025.3): the oracle (round half up) rounds this tie up to
 * 0x3c01; the device's actual sycl::half conversion rounds it to 0x3c00
 * instead, i.e. to the EVEN neighbour, confirming round-to-nearest-even.
 * The same divergence was reproduced at ties in several other exponents
 * during development. This is why the production kernel below ports
 * f32_to_f16_bits_hip_round literally as a device function instead of
 * using sycl::half for the encode direction: sycl::half would silently
 * change ring-buffer store results at exact ties relative to the ROCm
 * backend. */
static int test_f16_exact_tie_sycl_half_diverges_from_oracle(void) {
    const float midpoint = 1.00048828125f;
    const float lo = 1.0f;
    const float hi = 1.0009765625f;
    CHECK(fabsf(midpoint - lo) == fabsf(midpoint - hi), "constructed F16 midpoint is not an exact tie");

    const uint16_t oracle_bits = f32_to_f16_bits_hip_round_oracle(midpoint);
    CHECK(oracle_bits == 0x3c01u, "oracle did not round the constructed tie up as expected");
    CHECK_CLOSE(f16_bits_to_f32_oracle(oracle_bits), hi, 0.0, "oracle tie result does not decode to the upper neighbour");

    uint16_t device_bits = 0;
    CHECK(ds4_sycl_test_sycl_half_encode_bits(midpoint, &device_bits) != 0, "sycl_half_encode_bits call failed for the tie value");
    CHECK(device_bits != oracle_bits,
          "sycl::half unexpectedly agreed with the round-half-up oracle at an exact tie "
          "(re-run and re-derive the tie construction before trusting sycl::half here)");
    CHECK(device_bits == 0x3c00u, "sycl::half tie result was neither the round-up nor the expected round-to-even value");
    CHECK_CLOSE(f16_bits_to_f32_oracle(device_bits), lo, 0.0, "sycl::half tie result does not decode to the lower (even) neighbour");

    fprintf(stderr, "  test_f16_exact_tie_sycl_half_diverges_from_oracle OK\n");
    return 0;
}

/* Computes the value that should end up stored in the ring buffer after
 * encode-then-decode through the ported hip-round F16 path, for use as
 * the "want" value in the entry-point tests below. */
static float expected_stored_value(float f) {
    return f16_bits_to_f32_oracle(f32_to_f16_bits_hip_round_oracle(f));
}

/* Wraparound with NO collisions: n_tokens <= raw_cap, and pos0 is chosen
 * so that some destination rows wrap from the end of the ring buffer back
 * to the start.  This exercises the modulo arithmetic itself without
 * depending on the order of two concurrent, non-atomic writes to the same
 * address, which is genuinely unordered on this hardware (see the
 * concerns in the accompanying report; deliberately not tested here).
 *
 * n_tokens (4) != head_dim (2), so a t/d index-decomposition swap is
 * distinguishable by this test. Values are not exactly representable in
 * F16, so a store that skipped the F16 round trip is also distinguishable. */
static int test_store_raw_kv_ring_wraparound(void) {
    enum { RAW_CAP = 5, POS0 = 3, N_TOKENS = 4, HEAD_DIM = 2 };
    float kv[N_TOKENS * HEAD_DIM] = {
        1.23456f, -7.891f, 12345.6789f, -0.00042f, 3.14159265f, 65504.0f, -65504.0f, 0.1f,
    };
    float raw_init[RAW_CAP * HEAD_DIM];
    for (int i = 0; i < RAW_CAP * HEAD_DIM; i++) raw_init[i] = -999.0f;

    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(sizeof(raw_init));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc(sizeof(kv));
    CHECK(traw != NULL && tkv != NULL, "ring_wraparound: allocation failed");
    CHECK(ds4_gpu_tensor_write(traw, 0, raw_init, sizeof(raw_init)) != 0, "ring_wraparound: raw write");
    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "ring_wraparound: kv write");

    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, tkv, RAW_CAP, POS0, N_TOKENS, HEAD_DIM) != 0,
          "ring_wraparound: call must succeed");

    float got[RAW_CAP * HEAD_DIM];
    CHECK(ds4_gpu_tensor_read(traw, 0, got, sizeof(got)) != 0, "ring_wraparound: read");

    /* t=0->row3, t=1->row4, t=2->row(5%5)=0, t=3->row(6%5)=1; row2 untouched. */
    const int row_for_t[N_TOKENS] = { 3, 4, 0, 1 };
    for (int t = 0; t < N_TOKENS; t++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            char msg[80];
            snprintf(msg, sizeof(msg), "ring_wraparound: t=%d d=%d mismatch", t, d);
            CHECK_CLOSE(got[row_for_t[t] * HEAD_DIM + d], expected_stored_value(kv[t * HEAD_DIM + d]), 0.0, msg);
        }
    }
    for (int d = 0; d < HEAD_DIM; d++) {
        CHECK_CLOSE(got[2 * HEAD_DIM + d], -999.0f, 0.0, "ring_wraparound: untouched row 2 was modified");
    }

    ds4_gpu_tensor_free(traw);
    ds4_gpu_tensor_free(tkv);
    fprintf(stderr, "  test_store_raw_kv_ring_wraparound OK\n");
    return 0;
}

/* pos0 >= raw_cap: the source has no separate check for this, the
 * kernel's `row = (pos0 + t) % raw_cap` simply wraps normally.  Confirms
 * that directly rather than assuming pos0 out of range is rejected. */
static int test_store_raw_kv_pos0_beyond_cap(void) {
    enum { RAW_CAP = 4, POS0 = 10, N_TOKENS = 3, HEAD_DIM = 1 };
    float kv[N_TOKENS] = { 11.5f, 22.75f, 33.125f };
    float raw_init[RAW_CAP];
    for (int i = 0; i < RAW_CAP; i++) raw_init[i] = -777.0f;

    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(sizeof(raw_init));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc(sizeof(kv));
    CHECK(traw != NULL && tkv != NULL, "pos0_beyond_cap: allocation failed");
    CHECK(ds4_gpu_tensor_write(traw, 0, raw_init, sizeof(raw_init)) != 0, "pos0_beyond_cap: raw write");
    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "pos0_beyond_cap: kv write");

    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, tkv, RAW_CAP, POS0, N_TOKENS, HEAD_DIM) != 0,
          "pos0_beyond_cap: pos0 >= raw_cap must not be rejected");

    float got[RAW_CAP];
    CHECK(ds4_gpu_tensor_read(traw, 0, got, sizeof(got)) != 0, "pos0_beyond_cap: read");

    /* (10+0)%4=2, (10+1)%4=3, (10+2)%4=0; row 1 untouched. */
    CHECK_CLOSE(got[2], expected_stored_value(kv[0]), 0.0, "pos0_beyond_cap: row 2 mismatch");
    CHECK_CLOSE(got[3], expected_stored_value(kv[1]), 0.0, "pos0_beyond_cap: row 3 mismatch");
    CHECK_CLOSE(got[0], expected_stored_value(kv[2]), 0.0, "pos0_beyond_cap: row 0 mismatch");
    CHECK_CLOSE(got[1], -777.0f, 0.0, "pos0_beyond_cap: untouched row 1 was modified");

    ds4_gpu_tensor_free(traw);
    ds4_gpu_tensor_free(tkv);
    fprintf(stderr, "  test_store_raw_kv_pos0_beyond_cap OK\n");
    return 0;
}

/* Straightforward, non-wrapping store: rows outside [pos0, pos0+n_tokens)
 * must be left byte-for-byte untouched.  Sentinel pre-fill is not a value
 * the store, nor a zero-initialised buffer, would ever produce. */
static int test_store_raw_kv_untouched_rows_outside_range(void) {
    enum { RAW_CAP = 6, POS0 = 0, N_TOKENS = 2, HEAD_DIM = 3 };
    float kv[N_TOKENS * HEAD_DIM] = { 1.0f, 2.5f, -3.25f, 400.5f, -0.125f, 6.75f };
    float raw_init[RAW_CAP * HEAD_DIM];
    for (int i = 0; i < RAW_CAP * HEAD_DIM; i++) raw_init[i] = 24680.1357f + (float)i;

    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(sizeof(raw_init));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc(sizeof(kv));
    CHECK(traw != NULL && tkv != NULL, "untouched_rows: allocation failed");
    CHECK(ds4_gpu_tensor_write(traw, 0, raw_init, sizeof(raw_init)) != 0, "untouched_rows: raw write");
    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "untouched_rows: kv write");

    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, tkv, RAW_CAP, POS0, N_TOKENS, HEAD_DIM) != 0,
          "untouched_rows: call");

    float got[RAW_CAP * HEAD_DIM];
    CHECK(ds4_gpu_tensor_read(traw, 0, got, sizeof(got)) != 0, "untouched_rows: read");

    for (int t = 0; t < N_TOKENS; t++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            char msg[80];
            snprintf(msg, sizeof(msg), "untouched_rows: written row t=%d d=%d mismatch", t, d);
            CHECK_CLOSE(got[t * HEAD_DIM + d], expected_stored_value(kv[t * HEAD_DIM + d]), 0.0, msg);
        }
    }
    for (int row = N_TOKENS; row < RAW_CAP; row++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            char msg[80];
            snprintf(msg, sizeof(msg), "untouched_rows: row %d d=%d was modified", row, d);
            CHECK_CLOSE(got[row * HEAD_DIM + d], raw_init[row * HEAD_DIM + d], 0.0, msg);
        }
    }

    ds4_gpu_tensor_free(traw);
    ds4_gpu_tensor_free(tkv);
    fprintf(stderr, "  test_store_raw_kv_untouched_rows_outside_range OK\n");
    return 0;
}

/* n_tokens == 0: the source has no dedicated early return for this (see
 * the production comment above ds4_gpu_store_raw_kv_batch_tensor for the
 * full reasoning); ported behaviour is success with zero rows touched. */
static int test_store_raw_kv_zero_tokens(void) {
    enum { RAW_CAP = 4, HEAD_DIM = 2 };
    float raw_init[RAW_CAP * HEAD_DIM];
    for (int i = 0; i < RAW_CAP * HEAD_DIM; i++) raw_init[i] = -55.5f;
    float kv[HEAD_DIM] = { 1.0f, 2.0f };

    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(sizeof(raw_init));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc(sizeof(kv));
    CHECK(traw != NULL && tkv != NULL, "zero_tokens: allocation failed");
    CHECK(ds4_gpu_tensor_write(traw, 0, raw_init, sizeof(raw_init)) != 0, "zero_tokens: raw write");
    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "zero_tokens: kv write");

    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, tkv, RAW_CAP, 0, 0, HEAD_DIM) != 0,
          "zero_tokens: n_tokens=0 must succeed");

    float got[RAW_CAP * HEAD_DIM];
    CHECK(ds4_gpu_tensor_read(traw, 0, got, sizeof(got)) != 0, "zero_tokens: read");
    for (int i = 0; i < RAW_CAP * HEAD_DIM; i++) {
        CHECK_CLOSE(got[i], raw_init[i], 0.0, "zero_tokens: buffer was modified despite n_tokens=0");
    }

    ds4_gpu_tensor_free(traw);
    ds4_gpu_tensor_free(tkv);
    fprintf(stderr, "  test_store_raw_kv_zero_tokens OK\n");
    return 0;
}

/* The five rejections, each returning 0: null raw_cache, null kv,
 * raw_cap == 0, undersized raw_cache, undersized kv. */
static int test_store_raw_kv_rejections(void) {
    enum { RAW_CAP = 4, HEAD_DIM = 2, N_TOKENS = 3 };
    float raw_buf[RAW_CAP * HEAD_DIM];
    float kv_buf[N_TOKENS * HEAD_DIM];
    memset(raw_buf, 0, sizeof(raw_buf));
    memset(kv_buf, 0, sizeof(kv_buf));

    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(sizeof(raw_buf));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc(sizeof(kv_buf));
    CHECK(traw != NULL && tkv != NULL, "rejections: allocation failed");
    CHECK(ds4_gpu_tensor_write(traw, 0, raw_buf, sizeof(raw_buf)) != 0, "rejections: raw write");
    CHECK(ds4_gpu_tensor_write(tkv, 0, kv_buf, sizeof(kv_buf)) != 0, "rejections: kv write");

    CHECK(ds4_gpu_store_raw_kv_batch_tensor(NULL, tkv, RAW_CAP, 0, N_TOKENS, HEAD_DIM) == 0,
          "rejections: null raw_cache must be rejected");
    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, NULL, RAW_CAP, 0, N_TOKENS, HEAD_DIM) == 0,
          "rejections: null kv must be rejected");
    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, tkv, 0, 0, N_TOKENS, HEAD_DIM) == 0,
          "rejections: raw_cap == 0 must be rejected");

    ds4_gpu_tensor *small_raw = ds4_gpu_tensor_alloc(sizeof(raw_buf) - sizeof(float));
    CHECK(small_raw != NULL, "rejections: undersized raw_cache allocation failed");
    CHECK(ds4_gpu_store_raw_kv_batch_tensor(small_raw, tkv, RAW_CAP, 0, N_TOKENS, HEAD_DIM) == 0,
          "rejections: undersized raw_cache must be rejected");
    ds4_gpu_tensor_free(small_raw);

    ds4_gpu_tensor *small_kv = ds4_gpu_tensor_alloc(sizeof(kv_buf) - sizeof(float));
    CHECK(small_kv != NULL, "rejections: undersized kv allocation failed");
    CHECK(ds4_gpu_store_raw_kv_batch_tensor(traw, small_kv, RAW_CAP, 0, N_TOKENS, HEAD_DIM) == 0,
          "rejections: undersized kv must be rejected");
    ds4_gpu_tensor_free(small_kv);

    ds4_gpu_tensor_free(traw);
    ds4_gpu_tensor_free(tkv);
    fprintf(stderr, "  test_store_raw_kv_rejections OK\n");
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

    failures += test_f16_decode_oracle_self_check();
    failures += test_f16_ported_encoder_matches_oracle();

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

    failures += test_f16_sycl_half_matches_device_on_non_tie_values();
    failures += test_f16_exact_tie_sycl_half_diverges_from_oracle();
    failures += test_store_raw_kv_ring_wraparound();
    failures += test_store_raw_kv_pos0_beyond_cap();
    failures += test_store_raw_kv_untouched_rows_outside_range();
    failures += test_store_raw_kv_zero_tokens();
    failures += test_store_raw_kv_rejections();
    ds4_gpu_cleanup();

    if (failures == 0) {
        printf("test_sycl_fp8_kv: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_sycl_fp8_kv: %d test group(s) failed\n", failures);
    return 1;
}
