/* Correctness tests for the routed-MoE SYCL kernels.  Self-contained (no
 * tests/test_sycl_harness.h on this branch's base commit): defines its own
 * CHECK/CHECK_CLOSE matching the semantics used in tests/test_sycl_kernels.c.
 * Scalar oracles here reimplement the documented ROCm/ds4.c formulas with
 * line numbers cited; ds4.c's static functions cannot be linked directly,
 * matching the existing precedent in test_sycl_kernels.c. Needs no model
 * file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
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

/* Must match the device-side poison pattern in
 * sycl_moe_build_sorted_pairs's poison_for_test path (0xAB every byte). */
enum { SENTINEL = 0xABABABABu };

/* Test-only side door into sycl/ds4_sycl_moe.hpp, matching the precedent
 * set by sycl/ds4_sycl_streaming.hpp's ds4_sycl_stream_test_* functions.
 * Runs the full counting sort (rocm/ds4_rocm_moe.cuh:1255-1316) plus the
 * expert-tile build (moe.cuh:1327-1350) and reads every intermediate
 * array back to the host. */
/* Q8_K activation quantiser, rocm/ds4_rocm_moe.cuh:750-798.  out_bytes
 * must hold n_rows*(in_dim/256) blocks of 292 bytes each (float d, int8
 * qs[256], int16 bsums[16], no padding between fields). */
extern int ds4_sycl_moe_test_q8_k_quantize(const float *x, uint32_t in_dim,
                                           uint32_t n_rows, uint8_t *out_bytes);

/* Sub-group sum reduction, rocm/ds4_rocm_moe.cuh:732-749 (half_warp_sum_f32
 * for width 16, quarter_warp_sum_f32 for width 8).  in holds
 * n_groups*width values; out[g] is the sum of in[g*width .. g*width+width). */
extern int ds4_sycl_moe_test_subgroup_sum(int width, const float *in,
                                          uint32_t n_groups, float *out);

/* Weighted-sum-across-experts combine, rocm/ds4_rocm_moe.cuh:3877-3918.
 * mode 0 = moe_sum_kernel (f32 down), 1 = moe_sum_f16_kernel (down is raw
 * f16 bit patterns), 2 = moe_sum_f16x2_kernel (same, vectorised path). */
extern int ds4_sycl_moe_test_sum(int mode, const void *down, uint32_t out_dim,
                                 uint32_t n_expert, uint32_t n_tokens, float *out);

/* Drives sycl_moe_q2k_down_direct (the untagged-family Q2_K down kernel,
 * rocm/ds4_rocm_moe.cuh:2909-2936, generalised over n_tokens) directly,
 * for the slot-order regression test below; every ABI-level test in this
 * file uses n_expert == 2, which cannot discriminate summation order. */
extern int ds4_sycl_moe_test_q2k_down_direct(
        const uint8_t *down_bytes, const uint8_t *midq_bytes, const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t n_total_expert,
        float *out);

/* MXFP4 decode/tiny-batch regime side doors, driving
 * sycl_moe_mxfp4_gate_up_mid_decode / sycl_moe_mxfp4_down_sum6 directly,
 * independent of the public dispatcher's format gate. */
extern int ds4_sycl_moe_test_mxfp4_gate_up_mid_decode(
        const void *gate_model, const void *up_model, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t n_total_expert, const float *x, uint32_t in_dim,
        uint32_t n_tokens, const int32_t *selected, const float *weights, uint32_t n_expert,
        uint32_t mid_dim, float clamp, float *mid_out);
extern int ds4_sycl_moe_test_mxfp4_down_sum6(
        const void *down_model, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t n_total_expert, const float *mid, uint32_t mid_dim, uint32_t n_tokens,
        const int32_t *selected, uint32_t n_expert, uint32_t out_dim, float *out);

/* Mirror of ds4_sycl_moe_test_q2k_down_direct above, for iq2_iq2_path's
 * down projection (sycl_moe_iq2_down_direct, down_type == 16). */
extern int ds4_sycl_moe_test_iq2_down_direct(
        const uint8_t *down_bytes, const uint8_t *midq_bytes, const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t n_total_expert,
        float *out);

/* Instrumentation: how many experts, and how many gate+up+down
 * bytes, the most recently completed dispatcher call actually staged
 * host-to-device. Compaction is invisible in the numeric output (spec
 * 6w), so these are what the distinguishing test below reads. */
extern uint32_t ds4_sycl_moe_test_last_staged_expert_count(void);
extern uint64_t ds4_sycl_moe_test_last_staged_bytes(void);

enum { Q8_K_BLOCK_BYTES = 292 };

/* d (float, offset 0), qs (int8[256], offset 4), bsums (int16[16], offset 260). */
static float q8k_read_d(const uint8_t *blk) {
    float v;
    memcpy(&v, blk, sizeof(v));
    return v;
}
static int8_t q8k_read_qs(const uint8_t *blk, int i) { return (int8_t)blk[4 + i]; }
static int16_t q8k_read_bsum(const uint8_t *blk, int i) {
    int16_t v;
    memcpy(&v, blk + 260 + i * 2, sizeof(v));
    return v;
}

/* Minimal round-to-nearest-even IEEE754 binary16 encoder, sufficient for
 * the finite, moderate-magnitude test values used below (no inf/NaN/
 * subnormal handling needed). */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign; /* flush to zero, not needed by tests */
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t rounded_mant = (uint16_t)(mant >> 13);
    if (mant & 0x1000u) rounded_mant++; /* round to nearest, ties up */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | rounded_mant);
}

/* Inverse of f32_to_f16_bits, matching the sycl::bit_cast<sycl::half>
 * decode every SYCL-side f16 field in this subsystem uses. */
static float f16_bits_to_f32(uint16_t bits) {
    const int sign = (bits & 0x8000) ? -1 : 1;
    const int exp = (bits >> 10) & 0x1f;
    const int mant = bits & 0x3ff;
    if (exp == 0) return (float)sign * (float)mant * (float)ldexp(1.0, -24);
    return (float)sign * (float)(1024 + mant) * (float)ldexp(1.0, exp - 25);
}

/* ---- MXFP4: independent format oracle, no GPU involved -----------------
 *
 * Every MXFP4 kernel this port will add ultimately reduces to one 17-byte
 * block layout (ds4.c:807-819, block_mxfp4: uint8_t e; uint8_t qs[16]) and
 * two conversions: the E8M0 shared exponent (ds4.c:3732-3737) and the
 * 16-value E2M1 element table (ds4.c:3739-3742).  ds4-sycl-mxfp4-
 * validation.md's central risk is that a test oracle and a kernel derive
 * both from the same misreading of that format and quietly agree while
 * both being wrong.  This section carries two INDEPENDENTLY derived
 * computations of the same sixteen dequantized values, cross-checked
 * against each other on random data instead of one being tested against
 * a copy of itself:
 *
 *   - oracle_mxfp4_code_value_ocp: the OCP MX E2M1 bit fields (sign, a
 *     2-bit exponent with bias 1, a 1-bit mantissa) combined via ldexp,
 *     derived from the format definition, not from ds4_mxfp4_values or
 *     any existing test file's table.
 *   - oracle_mxfp4_code_twice: a doubled-integer bit manipulation in the
 *     SHAPE of dev_mxfp4_unpack4's portable branch
 *     (rocm/ds4_rocm_moe.cuh:301-326), independently re-derived here
 *     rather than copied, used only as the cross-check.
 *
 * These format-level checks were written and passing before any MXFP4
 * kernel existed; that is deliberate, since what they test is the oracle
 * itself, the one place in this project where a test legitimately passes
 * before the device code it will later validate exists. */

enum { MXFP4_QK = 32 };

typedef struct {
    uint8_t e;
    uint8_t qs[MXFP4_QK / 2];
} oracle_mxfp4_block;

/* ds4_e8m0_to_f32, ds4.c:3732-3737: bits = e==0 ? 0x00400000 : (e<<23),
 * i.e. the raw byte placed directly into a float's 8-bit exponent field.
 * Derived here via ldexpf rather than the bit-cast ds4.c and the SYCL
 * kernel both use, so a wrong exponent bias would have to be wrong the
 * same way in two unrelated code shapes to go unnoticed. */
static float oracle_mxfp4_e8m0_scale(uint8_t e) {
    return e == 0u ? ldexpf(1.0f, -127) : ldexpf(1.0f, (int)e - 127);
}

/* Independent derivation 1: OCP MX E2M1 bit fields.  Bit 3 is sign, bits
 * 2:1 are a 2-bit exponent field with bias 1, bit 0 is a 1-bit mantissa.
 * Exponent field 0 is the subnormal case (leading significand bit 0
 * instead of 1).  Arithmetic on the field values, not a lookup table. */
static float oracle_mxfp4_code_value_ocp(uint8_t code) {
    const int sign = (code & 8u) ? -1 : 1;
    const int exp_field = (code >> 1) & 3;
    const int mantissa_bit = code & 1;
    const double leading = exp_field == 0 ? 0.0 : 1.0;
    const double significand = leading + (double)mantissa_bit * 0.5;
    const double scale = exp_field == 0 ? ldexp(1.0, 0) : ldexp(1.0, exp_field - 1);
    return (float)((double)sign * significand * scale);
}

/* Independent derivation 2: doubled-integer bit manipulation, matching
 * the shape of dev_mxfp4_unpack4's portable branch: base walks 0..7, the
 * two "missing mantissa step" codes (5 and 6) get bumped past the gap,
 * and code 7 gets a further +2 to reach 12 (6.0 doubled) instead of the
 * naive 7+3=10.  Every value is exact in integer doubled form. */
static int oracle_mxfp4_code_twice(uint8_t code) {
    const uint32_t base = code & 7u;
    int32_t value = (int32_t)(base + (base > 4u ? base - 4u : 0u) + (base == 7u ? 2u : 0u));
    if (code & 8u) value = -value;
    return value;
}

static float oracle_mxfp4_dot(const oracle_mxfp4_block *blocks, int n_blocks, const float *y) {
    float sum = 0.0f;
    for (int ib = 0; ib < n_blocks; ib++) {
        const float scale = oracle_mxfp4_e8m0_scale(blocks[ib].e);
        for (int j = 0; j < MXFP4_QK / 2; j++) {
            const uint8_t byte = blocks[ib].qs[j];
            sum += scale * oracle_mxfp4_code_value_ocp(byte & 0x0f) * y[ib * MXFP4_QK + j];
            sum += scale * oracle_mxfp4_code_value_ocp(byte >> 4) * y[ib * MXFP4_QK + j + MXFP4_QK / 2];
        }
    }
    return sum;
}

static int test_mxfp4_block_layout(void) {
    CHECK(sizeof(oracle_mxfp4_block) == 17, "mxfp4: block size must be 17 bytes");
    fprintf(stderr, "  test_mxfp4_block_layout OK\n");
    return 0;
}

static int test_mxfp4_e8m0_boundary(void) {
    /* e == 0 is 2^-127, a small subnormal-magnitude float, NOT zero: a
     * plausible-looking but wrong simplification is to treat a
     * zero-exponent block as an all-zero block. */
    CHECK(oracle_mxfp4_e8m0_scale(0) == ldexpf(1.0f, -127),
          "mxfp4: e=0 scale must be 2^-127, not 0");
    CHECK(oracle_mxfp4_e8m0_scale(0) != 0.0f, "mxfp4: e=0 scale must not be zero");
    CHECK(oracle_mxfp4_e8m0_scale(1) == ldexpf(1.0f, -126), "mxfp4: e=1 scale mismatch");
    CHECK(oracle_mxfp4_e8m0_scale(127) == 1.0f, "mxfp4: e=127 scale must be 1.0");
    fprintf(stderr, "  test_mxfp4_e8m0_boundary OK\n");
    return 0;
}

/* Pinned, not resolved: e==255 places an all-ones exponent field with a
 * zero mantissa into a float, which IEEE754 defines as +infinity, not a
 * finite 2^128 -- 2^(255-127) = 2^128 overflows any finite float either
 * way, so a bit-cast and an ldexpf-based scale necessarily agree here.
 * The OCP MX spec reserves e==255 for NaN; this project's format does not
 * implement that.  This test pins the actual (diverging) behaviour rather
 * than adjudicating whether the divergence matters. */
static int test_mxfp4_e255_pinned(void) {
    const float scale = oracle_mxfp4_e8m0_scale(255);
    CHECK(isinf(scale) && scale > 0.0f,
          "mxfp4: e=255 must be +infinity, not a finite value or NaN");
    fprintf(stderr, "  test_mxfp4_e255_pinned OK\n");
    return 0;
}

static int test_mxfp4_e2m1_codes(void) {
    static const float expect[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    for (int code = 0; code < 16; code++) {
        const float got_ocp = oracle_mxfp4_code_value_ocp((uint8_t)code);
        const float got_twice = 0.5f * (float)oracle_mxfp4_code_twice((uint8_t)code);
        CHECK(got_ocp == expect[code], "mxfp4: OCP-derived code value mismatch");
        CHECK(got_twice == expect[code], "mxfp4: twice-integer code value mismatch");
    }
    fprintf(stderr, "  test_mxfp4_e2m1_codes OK\n");
    return 0;
}

static int test_mxfp4_exponent_scaling(void) {
    /* Exact power-of-two ratio between adjacent nonzero exponents, bit
     * exact (every value here is a small power-of-two multiple, exactly
     * representable in float). */
    for (int e = 1; e < 250; e++) {
        const float lo = oracle_mxfp4_e8m0_scale((uint8_t)e);
        const float hi = oracle_mxfp4_e8m0_scale((uint8_t)(e + 1));
        CHECK(hi == 2.0f * lo, "mxfp4: adjacent exponents must scale by exactly 2x");
    }
    for (int code = 1; code < 8; code++) {
        const float v_e1 = oracle_mxfp4_code_value_ocp((uint8_t)code) * oracle_mxfp4_e8m0_scale(10);
        const float v_e2 = oracle_mxfp4_code_value_ocp((uint8_t)code) * oracle_mxfp4_e8m0_scale(11);
        CHECK(v_e2 == 2.0f * v_e1, "mxfp4: dequantized value must double with exponent+1");
    }
    fprintf(stderr, "  test_mxfp4_exponent_scaling OK\n");
    return 0;
}

static int test_mxfp4_monotonicity(void) {
    for (int code = 0; code < 7; code++) {
        CHECK(oracle_mxfp4_code_value_ocp((uint8_t)code) <
                  oracle_mxfp4_code_value_ocp((uint8_t)(code + 1)),
              "mxfp4: non-negative codes must be strictly increasing");
    }
    for (int code = 0; code < 8; code++) {
        CHECK(oracle_mxfp4_code_value_ocp((uint8_t)(code + 8)) ==
                  -oracle_mxfp4_code_value_ocp((uint8_t)code),
              "mxfp4: sign bit must exactly negate the magnitude code");
    }
    fprintf(stderr, "  test_mxfp4_monotonicity OK\n");
    return 0;
}

static int test_mxfp4_nibble_order(void) {
    /* y is zero on the block's second half, so only the low-nibble
     * contribution (elements 0..15) survives the dot.  A uniform y
     * (every element 1.0) sums both halves regardless of which nibble
     * feeds which half and cannot tell nibble order from a value-table
     * bug at all; that degenerate form of this test was tried first and
     * an ablation swapping the nibble order did not fail it, only
     * test_mxfp4_random_dot caught it (spec 6i: the test data, not just
     * the property under test, decides whether an ablation discriminates).
     * Low nibble 0x6 -> code 6 -> value 4.0; high nibble 0xf -> code 15
     * -> value -6.0. */
    oracle_mxfp4_block block;
    float y[MXFP4_QK];
    block.e = 127;
    memset(block.qs, 0xf6, sizeof(block.qs));
    for (int i = 0; i < MXFP4_QK / 2; i++) y[i] = 1.0f;
    for (int i = MXFP4_QK / 2; i < MXFP4_QK; i++) y[i] = 0.0f;
    const float got = oracle_mxfp4_dot(&block, 1, y);
    CHECK(got == 16.0f * 4.0f, "mxfp4: nibble order mismatch");
    fprintf(stderr, "  test_mxfp4_nibble_order OK\n");
    return 0;
}

static int test_mxfp4_random_dot(void) {
    enum { N_BLOCK = 5, N = N_BLOCK * MXFP4_QK };
    oracle_mxfp4_block blocks[N_BLOCK];
    float y[N];
    uint32_t state = 0xC0FFEEu;
    for (int trial = 0; trial < 200; trial++) {
        for (int ib = 0; ib < N_BLOCK; ib++) {
            state = state * 1664525u + 1013904223u;
            blocks[ib].e = (uint8_t)(100u + state % 55u);
            for (int j = 0; j < MXFP4_QK / 2; j++) {
                state = state * 1664525u + 1013904223u;
                blocks[ib].qs[j] = (uint8_t)(state >> 24);
            }
        }
        for (int i = 0; i < N; i++) {
            state = state * 1664525u + 1013904223u;
            y[i] = (float)(int16_t)(state >> 16) / 4096.0f;
        }

        const float got = oracle_mxfp4_dot(blocks, N_BLOCK, y);
        float expected = 0.0f;
        float abs_sum = 0.0f;
        for (int ib = 0; ib < N_BLOCK; ib++) {
            const float scale = oracle_mxfp4_e8m0_scale(blocks[ib].e);
            for (int j = 0; j < MXFP4_QK / 2; j++) {
                const uint8_t byte = blocks[ib].qs[j];
                const float t0 = scale * 0.5f * (float)oracle_mxfp4_code_twice(byte & 0x0f) *
                                 y[ib * MXFP4_QK + j];
                const float t1 = scale * 0.5f * (float)oracle_mxfp4_code_twice(byte >> 4) *
                                 y[ib * MXFP4_QK + j + MXFP4_QK / 2];
                expected += t0;
                expected += t1;
                abs_sum += fabsf(t0) + fabsf(t1);
            }
        }
        /* Tolerance floors on the sum of term magnitudes, not the (possibly
         * heavily cancelled) final expected value: random signed terms at
         * widely varying per-block scales can sum to something much
         * smaller than any individual term, and a relative tolerance on
         * that near-zero total would fail on ordinary float rounding, not
         * on a real defect (spec 6f's cancellation hazard, applied to plain
         * summation rather than a normalisation). */
        const float tolerance = 2.0e-5f * fmaxf(1.0f, abs_sum);
        CHECK(fabsf(got - expected) <= tolerance,
              "mxfp4: dot vs independently-derived reference mismatch");
    }
    fprintf(stderr, "  test_mxfp4_random_dot OK\n");
    return 0;
}

static int test_q8_k_quantize(void) {
    /* Two superblocks (in_dim = 512), two rows.  Row 0 has a wide dynamic
     * range (values spanning several orders of magnitude plus a mix of
     * signs); row 1 is entirely zero, exercising the amax==0 branch. */
    enum { IN_DIM = 512, N_ROWS = 2, N_BLOCKS = IN_DIM / 256 };
    float x[N_ROWS * IN_DIM];
    for (int i = 0; i < IN_DIM; i++) {
        float v = ((float)((i * 37) % 251) - 125.0f) * ((i % 5 == 0) ? 100.0f : 0.01f);
        if (i % 7 == 0) v = -v;
        x[i] = v;
    }
    for (int i = 0; i < IN_DIM; i++) x[IN_DIM + i] = 0.0f;

    uint8_t *out = malloc((size_t)N_ROWS * N_BLOCKS * Q8_K_BLOCK_BYTES);
    CHECK(ds4_sycl_moe_test_q8_k_quantize(x, IN_DIM, N_ROWS, out) != 0,
          "q8_k_quantize: call failed");

    for (int row = 0; row < N_ROWS; row++) {
        for (int b = 0; b < N_BLOCKS; b++) {
            const float *xr = x + row * IN_DIM + b * 256;
            const uint8_t *blk = out + ((size_t)row * N_BLOCKS + b) * Q8_K_BLOCK_BYTES;
            float amax = 0.0f, maxv = 0.0f;
            for (int i = 0; i < 256; i++) {
                if (fabsf(xr[i]) > amax) { amax = fabsf(xr[i]); maxv = xr[i]; }
            }
            char msg[96];
            if (amax == 0.0f) {
                CHECK(q8k_read_d(blk) == 0.0f, "q8_k_quantize: zero block d must be 0");
                for (int i = 0; i < 256; i++) {
                    CHECK(q8k_read_qs(blk, i) == 0, "q8_k_quantize: zero block qs must be 0");
                }
                for (int i = 0; i < 16; i++) {
                    CHECK(q8k_read_bsum(blk, i) == 0, "q8_k_quantize: zero block bsums must be 0");
                }
                continue;
            }
            const float iscale = -127.0f / maxv;
            const float d = q8k_read_d(blk);
            CHECK_CLOSE(d, 1.0f / iscale, fabsf(d) * 1e-5f + 1e-8f, "q8_k_quantize: d mismatch");
            for (int i = 0; i < 256; i++) {
                int qv = (int)lrintf(iscale * xr[i]);
                if (qv > 127) qv = 127;
                if (qv < -128) qv = -128;
                snprintf(msg, sizeof(msg), "q8_k_quantize: qs[%d] mismatch (row %d block %d)", i, row, b);
                CHECK(q8k_read_qs(blk, i) == (int8_t)qv, msg);
            }
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += q8k_read_qs(blk, g * 16 + i);
                snprintf(msg, sizeof(msg), "q8_k_quantize: bsums[%d] mismatch (row %d block %d)", g, row, b);
                CHECK(q8k_read_bsum(blk, g) == (int16_t)sum, msg);
            }
        }
    }
    free(out);
    fprintf(stderr, "  test_q8_k_quantize OK\n");
    return 0;
}

/* A row length that is not a multiple of 256 must be rejected (the
 * quantiser has no defined behaviour for a partial superblock; the
 * dispatcher's own validation, ported in a later task, rejects this
 * shape before ever reaching the kernel). */
static int test_q8_k_quantize_rejects_partial_row(void) {
    float x[300];
    for (int i = 0; i < 300; i++) x[i] = (float)i;
    uint8_t out[2 * Q8_K_BLOCK_BYTES];
    CHECK(ds4_sycl_moe_test_q8_k_quantize(x, 300, 1, out) == 0,
          "q8_k_quantize: non-multiple-of-256 in_dim must be rejected");
    fprintf(stderr, "  test_q8_k_quantize_rejects_partial_row OK\n");
    return 0;
}

/* Both reduction widths, with values chosen so a reduction over the wrong
 * lane count gives a different answer: group g's values are
 * (g*100 + lane + 1), so summing 8 of them differs from summing 16 by a
 * large, easily-distinguished amount (not just "some upper lanes happen
 * to be zero"). */
static int test_subgroup_sum(void) {
    const int widths[2] = {8, 16};
    for (int wi = 0; wi < 2; wi++) {
        const int width = widths[wi];
        enum { N_GROUPS = 6 };
        float in[6 * 16];
        float want[6];
        for (int g = 0; g < N_GROUPS; g++) {
            float sum = 0.0f;
            for (int lane = 0; lane < width; lane++) {
                float v = (float)(g * 100 + lane + 1);
                in[g * width + lane] = v;
                sum += v;
            }
            want[g] = sum;
        }
        float got[N_GROUPS];
        char msg[64];
        snprintf(msg, sizeof(msg), "subgroup_sum: width %d call failed", width);
        CHECK(ds4_sycl_moe_test_subgroup_sum(width, in, N_GROUPS, got) != 0, msg);
        for (int g = 0; g < N_GROUPS; g++) {
            snprintf(msg, sizeof(msg), "subgroup_sum: width %d group %d mismatch", width, g);
            CHECK_CLOSE(got[g], want[g], 1e-3, msg);
        }
    }
    fprintf(stderr, "  test_subgroup_sum OK\n");
    return 0;
}

/* moe_sum_kernel accumulates strictly in ascending slot order
 * (e = 0..n_expert-1 over the pair index tok*n_expert+e).  Floating point
 * addition is not associative, so a wrong accumulation order (as
 * layer_routed_moe_batch's expert-id ordering would produce against this
 * kernel's own down-tensor layout, see ds4-sycl-moe-reference.md section
 * 4(a)) is only actually distinguishable from the correct order when the
 * terms have a wide enough dynamic range for reordering to change
 * rounding.  Each row's four values are {+1e8, -1e8, v1, v2}: summed in
 * ascending order the two huge terms cancel to exactly 0.0 before the
 * small terms are added (result v1+v2); summed in any order that adds a
 * small term between the two huge ones, the small term is lost to
 * rounding against the ~1e8-magnitude partial sum instead. */
static int test_sum_slot_order(void) {
    enum { OUT_DIM = 6, N_EXPERT = 4, N_TOKENS = 3 };
    float down[N_TOKENS * N_EXPERT * OUT_DIM];
    float want[N_TOKENS * OUT_DIM];
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            const float v1 = (float)((t + 1) * 10 + row + 1);
            const float v2 = (float)((t + 1) * 7 + row + 3);
            const float vals[N_EXPERT] = {1.0e8f, -1.0e8f, v1, v2};
            float sum = 0.0f;
            for (int e = 0; e < N_EXPERT; e++) {
                down[(t * N_EXPERT + e) * OUT_DIM + row] = vals[e];
                sum += vals[e];
            }
            want[t * OUT_DIM + row] = sum;
        }
    }
    float got[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(0, down, OUT_DIM, N_EXPERT, N_TOKENS, got) != 0,
          "sum_slot_order: f32 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], 1e-2, "sum_slot_order: f32 value mismatch");
    }

    /* f16 and f16x2 variants: half precision cannot represent 1e8 (max
     * finite value ~65504), so these use their own moderate-magnitude
     * data instead of the cancellation values above; the order-sensitive
     * property was already exercised by the f32 case.  OUT_DIM is even,
     * so f16x2 is exercised too. */
    uint16_t down_h[N_TOKENS * N_EXPERT * OUT_DIM];
    float want_h[N_TOKENS * OUT_DIM];
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            for (int e = 0; e < N_EXPERT; e++) {
                float v = (float)((t + 1) * 12 - e * 5 + row);
                down_h[(t * N_EXPERT + e) * OUT_DIM + row] = f32_to_f16_bits(v);
            }
        }
    }
    /* want_h is derived from the actual half-rounded values (decoded via
     * f16_bits_to_f32) so the oracle matches what the kernel reads, not
     * full f32 precision. */
    for (int t = 0; t < N_TOKENS; t++) {
        for (int row = 0; row < OUT_DIM; row++) {
            float sum = 0.0f;
            for (int e = 0; e < N_EXPERT; e++) {
                uint16_t bits = down_h[(t * N_EXPERT + e) * OUT_DIM + row];
                sum += f16_bits_to_f32(bits);
            }
            want_h[t * OUT_DIM + row] = sum;
        }
    }
    float got_h[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(1, down_h, OUT_DIM, N_EXPERT, N_TOKENS, got_h) != 0,
          "sum_slot_order: f16 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got_h[i], want_h[i], 1e-2, "sum_slot_order: f16 value mismatch");
    }
    float got_h2[N_TOKENS * OUT_DIM];
    CHECK(ds4_sycl_moe_test_sum(2, down_h, OUT_DIM, N_EXPERT, N_TOKENS, got_h2) != 0,
          "sum_slot_order: f16x2 call failed");
    for (int i = 0; i < N_TOKENS * OUT_DIM; i++) {
        CHECK_CLOSE(got_h2[i], want_h[i], 1e-2, "sum_slot_order: f16x2 value mismatch");
    }
    fprintf(stderr, "  test_sum_slot_order OK\n");
    return 0;
}

extern int ds4_sycl_moe_test_sort(const int32_t *selected, uint32_t pair_count,
                                  uint32_t n_total_expert, uint32_t block_m,
                                  uint32_t *counts_out, uint32_t *offsets_out,
                                  uint32_t *sorted_pairs_out,
                                  uint32_t *tile_total_out,
                                  uint32_t *tile_experts_out,
                                  uint32_t *tile_starts_out,
                                  uint32_t tile_capacity_cap);

static uint32_t sort_tile_capacity(uint32_t pair_count, uint32_t n_total_expert,
                                   uint32_t block_m) {
    return (pair_count + block_m - 1u) / block_m + n_total_expert;
}

/* Scalar oracle for the counting sort, cited to rocm/ds4_rocm_moe.cuh:
 * 1255-1316 and to ds4.c:11126-11166 (layer_routed_moe_batch's own
 * scalar sort, which the GPU sort structurally mirrors). */
static void oracle_sort(const int32_t *selected, uint32_t pair_count,
                        uint32_t n_total_expert, uint32_t *counts,
                        uint32_t *offsets, uint32_t *sorted_pairs) {
    memset(counts, 0, (size_t)n_total_expert * sizeof(uint32_t));
    for (uint32_t p = 0; p < pair_count; p++) {
        int32_t e = selected[p];
        if (e < 0) e = 0;
        if ((uint32_t)e >= n_total_expert) continue;
        counts[(uint32_t)e]++;
    }
    uint32_t sum = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        offsets[e] = sum;
        sum += counts[e];
    }
    offsets[n_total_expert] = sum;
    uint32_t *cursor = malloc((size_t)n_total_expert * sizeof(uint32_t));
    memcpy(cursor, offsets, (size_t)n_total_expert * sizeof(uint32_t));
    for (uint32_t p = 0; p < pair_count; p++) {
        int32_t e = selected[p];
        if (e < 0) e = 0;
        if ((uint32_t)e >= n_total_expert) continue;
        sorted_pairs[cursor[(uint32_t)e]++] = p;
    }
    free(cursor);
}

static int test_sort_basic(void) {
    /* n_total_expert=6, expert 0 selected by no pair, expert 3 selected
     * by every one of the 5 tokens (n_expert slots = 2, so pair index =
     * tok*2+slot), and pair-index order within a bucket must come out
     * strictly ascending (the "_deterministic" guarantee: a scan over
     * pairs 0..pair_count-1 in order, never an atomic-append race). */
    enum { N_TOK = 5, N_SLOT = 2, N_TOTAL_EXPERT = 6 };
    const uint32_t pair_count = N_TOK * N_SLOT;
    int32_t selected[N_TOK * N_SLOT];
    /* Every token selects expert 3 in slot 0, and a rotating expert among
     * {1,2,4,5} in slot 1.  Expert 0 is never selected. */
    const int32_t rotating[N_TOK] = {1, 2, 4, 5, 1};
    for (int t = 0; t < N_TOK; t++) {
        selected[t * N_SLOT + 0] = 3;
        selected[t * N_SLOT + 1] = rotating[t];
    }

    uint32_t oc[N_TOTAL_EXPERT], oo[N_TOTAL_EXPERT + 1], osp[N_TOK * N_SLOT];
    oracle_sort(selected, pair_count, N_TOTAL_EXPERT, oc, oo, osp);
    CHECK(oc[0] == 0u, "sort_basic: expert 0 selected by no pair");
    CHECK(oc[3] == N_TOK, "sort_basic: expert 3 selected by every token");

    const uint32_t block_m = 8u;
    const uint32_t tile_cap = sort_tile_capacity(pair_count, N_TOTAL_EXPERT, block_m);
    uint32_t counts[N_TOTAL_EXPERT], offsets[N_TOTAL_EXPERT + 1];
    uint32_t sorted_pairs[N_TOK * N_SLOT];
    uint32_t tile_total = SENTINEL;
    uint32_t *tile_experts = malloc((size_t)tile_cap * sizeof(uint32_t));
    uint32_t *tile_starts = malloc((size_t)tile_cap * sizeof(uint32_t));
    for (uint32_t i = 0; i < tile_cap; i++) {
        tile_experts[i] = SENTINEL;
        tile_starts[i] = SENTINEL;
    }
    for (uint32_t i = 0; i < pair_count; i++) sorted_pairs[i] = SENTINEL;

    CHECK(ds4_sycl_moe_test_sort(selected, pair_count, N_TOTAL_EXPERT, block_m,
                                 counts, offsets, sorted_pairs, &tile_total,
                                 tile_experts, tile_starts, tile_cap) != 0,
          "sort_basic: ds4_sycl_moe_test_sort failed");

    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sort_basic: counts[%u] mismatch", e);
        CHECK(counts[e] == oc[e], msg);
        snprintf(msg, sizeof(msg), "sort_basic: offsets[%u] mismatch", e);
        CHECK(offsets[e] == oo[e], msg);
    }
    CHECK(offsets[N_TOTAL_EXPERT] == oo[N_TOTAL_EXPERT],
          "sort_basic: final offset mismatch");
    for (uint32_t p = 0; p < pair_count; p++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sort_basic: sorted_pairs[%u] mismatch", p);
        CHECK(sorted_pairs[p] == osp[p], msg);
    }

    /* Deterministic order: within expert 3's bucket (5 entries), the
     * scatter must produce pairs in strictly ascending pair-index order,
     * since a per-expert sequential scan (not a racing atomic append)
     * is what "_deterministic" means. */
    uint32_t prev = 0;
    for (uint32_t i = offsets[3]; i < offsets[3] + counts[3]; i++) {
        if (i > offsets[3]) {
            CHECK(sorted_pairs[i] > prev,
                  "sort_basic: expert 3 bucket not in ascending pair order");
        }
        prev = sorted_pairs[i];
    }

    /* Tile build: verify tile_total, and every filled tile_experts/
     * tile_starts entry against a hand oracle; everything from tile_total
     * up to tile_cap must remain the sentinel (unwritten), proving the
     * kernel does not spill past its own tile count. */
    uint32_t expect_tile_total = 0;
    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        expect_tile_total += (counts[e] + block_m - 1u) / block_m;
    }
    CHECK(tile_total == expect_tile_total, "sort_basic: tile_total mismatch");
    uint32_t tile_idx = 0;
    for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
        uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
        for (uint32_t t = 0; t < ntiles; t++) {
            char msg[80];
            snprintf(msg, sizeof(msg), "sort_basic: tile_experts[%u] mismatch", tile_idx);
            CHECK(tile_experts[tile_idx] == e, msg);
            snprintf(msg, sizeof(msg), "sort_basic: tile_starts[%u] mismatch", tile_idx);
            CHECK(tile_starts[tile_idx] == t * block_m, msg);
            tile_idx++;
        }
    }
    CHECK(tile_idx == tile_total, "sort_basic: tile walk did not reach tile_total");
    for (uint32_t i = tile_total; i < tile_cap; i++) {
        CHECK(tile_experts[i] == SENTINEL, "sort_basic: tile_experts spilled past tile_total");
        CHECK(tile_starts[i] == SENTINEL, "sort_basic: tile_starts spilled past tile_total");
    }

    free(tile_experts);
    free(tile_starts);
    fprintf(stderr, "  test_sort_basic OK\n");
    return 0;
}

/* A second block_m to prove the tile math is not hardcoded to 8 (Q4_K's
 * own expert_tile_m), and a wider pair_count so at least one expert's
 * bucket spans multiple tiles. */
static int test_sort_tile_sizes(void) {
    enum { N_TOTAL_EXPERT = 4 };
    const uint32_t pair_count = 40;
    int32_t selected[40];
    /* Expert 0 gets 17 pairs (spans 3 tiles at block_m=8, 6 at block_m=3),
     * expert 1 gets 0, expert 2 gets 12, expert 3 gets 11. */
    uint32_t idx = 0;
    for (uint32_t i = 0; i < 17; i++) selected[idx++] = 0;
    for (uint32_t i = 0; i < 12; i++) selected[idx++] = 2;
    for (uint32_t i = 0; i < 11; i++) selected[idx++] = 3;

    const uint32_t block_ms[2] = {8u, 3u};
    for (int bi = 0; bi < 2; bi++) {
        const uint32_t block_m = block_ms[bi];
        const uint32_t tile_cap = sort_tile_capacity(pair_count, N_TOTAL_EXPERT, block_m);
        uint32_t counts[N_TOTAL_EXPERT], offsets[N_TOTAL_EXPERT + 1];
        uint32_t *sorted_pairs = malloc((size_t)pair_count * sizeof(uint32_t));
        uint32_t tile_total = SENTINEL;
        uint32_t *tile_experts = malloc((size_t)tile_cap * sizeof(uint32_t));
        uint32_t *tile_starts = malloc((size_t)tile_cap * sizeof(uint32_t));

        CHECK(ds4_sycl_moe_test_sort(selected, pair_count, N_TOTAL_EXPERT, block_m,
                                     counts, offsets, sorted_pairs, &tile_total,
                                     tile_experts, tile_starts, tile_cap) != 0,
              "sort_tile_sizes: ds4_sycl_moe_test_sort failed");

        uint32_t expect_total = 0;
        for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) {
            expect_total += (counts[e] + block_m - 1u) / block_m;
        }
        CHECK(tile_total == expect_total, "sort_tile_sizes: tile_total mismatch");
        CHECK(counts[1] == 0u, "sort_tile_sizes: expert 1 must have zero count");

        free(sorted_pairs);
        free(tile_experts);
        free(tile_starts);
    }
    fprintf(stderr, "  test_sort_tile_sizes OK\n");
    return 0;
}

/* ---- Dispatcher skeleton and the two entry points --------------------
 *
 * IN_DIM/MID_DIM are two Q8_K superblocks wide (512), not one (256): a
 * width of exactly one block makes every per-chunk loop in the kernels
 * under test ("for b in 0..xq_blocks") execute a single iteration, which
 * cannot discriminate a kernel that dots every weight block against
 * activation chunk 0 instead of its own chunk b (see 35852c0, which fixed
 * exactly that for sycl_dev_dot_q4_k_q8_k_block8). OUT_DIM is widened to
 * 128 alongside them: the down tensor doubles as Q8_K quantisation
 * scratch for the activations (moe_launch.cuh:746's scratch-reuse
 * precondition, `down->bytes >= xq_bytes`), and at IN_DIM=512 (xq_blocks=2)
 * that needs n_expert*OUT_DIM*4 >= xq_blocks*292, i.e. OUT_DIM >= 73 for
 * RM_N_EXPERT=2; 128 is verified sufficient below and keeps headroom.
 * gate similarly doubles as midq scratch but that precondition
 * (expert_mid_dim*4 >= midq_blocks*292) holds for any multiple-of-256
 * MID_DIM, since 4*256 > 292 already at one block. No separate narrow
 * shape is kept: at these sizes the whole suite still runs in a few
 * seconds (see the report), so a second, narrower constant set would only
 * add maintenance cost for no measured speed benefit. */
enum {
    RM_EXPERT_IN_DIM  = 512,
    RM_EXPERT_MID_DIM = 512,
    RM_OUT_DIM        = 128,
    RM_N_EXPERT       = 2,
    RM_N_TOKENS       = 2,
    RM_N_TOTAL_EXPERT = 4,
    RM_Q4K_BLOCK_BYTES = 144, /* cuda_block_q4_K, ds4_rocm.cu:72-77 */
};

typedef struct {
    unsigned char *model;
    uint64_t       model_size;
    uint64_t       gate_offset, up_offset, down_offset;
    uint64_t       gate_expert_bytes, gate_row_bytes;
    uint64_t       down_expert_bytes, down_row_bytes;
} rm_model;

enum {
    RM_IQ2_BLOCK_BYTES = 66, /* cuda_block_iq2_xxs, ds4_rocm.cu:85-88 */
    RM_Q2K_BLOCK_BYTES = 84, /* cuda_block_q2_K, ds4_rocm.cu:65-70 */
};

/* Generalised model builder: gate/up always share one block size (both
 * paired formats implemented here use the same weight type for gate
 * and up), down may use a different one (iq2_path pairs IQ2_XXS gate/up
 * with Q2_K down), and the table holds n_total_expert experts rather than
 * always RM_N_TOTAL_EXPERT -- the compaction tests need a table
 * bigger than RM_N_TOTAL_EXPERT to make "staged only the selected
 * experts" distinguishable from "staged everything" at all. */
static rm_model rm_build_model_n(uint32_t n_total_expert, uint32_t gate_block_bytes,
                                 uint32_t down_block_bytes) {
    rm_model m;
    m.gate_row_bytes = (RM_EXPERT_IN_DIM / 256u) * gate_block_bytes;
    m.gate_expert_bytes = (uint64_t)RM_EXPERT_MID_DIM * m.gate_row_bytes;
    m.down_row_bytes = (RM_EXPERT_MID_DIM / 256u) * down_block_bytes;
    m.down_expert_bytes = (uint64_t)RM_OUT_DIM * m.down_row_bytes;
    m.gate_offset = 0;
    m.up_offset = m.gate_expert_bytes * n_total_expert;
    m.down_offset = m.up_offset + m.gate_expert_bytes * n_total_expert;
    m.model_size = m.down_offset + m.down_expert_bytes * n_total_expert;
    m.model = calloc(1, (size_t)m.model_size);
    return m;
}

static rm_model rm_build_model_ex(uint32_t gate_block_bytes, uint32_t down_block_bytes) {
    return rm_build_model_n(RM_N_TOTAL_EXPERT, gate_block_bytes, down_block_bytes);
}

/* type: 12 = Q4_K (used for gate_row_bytes sizing regardless of the
 * gate_type/down_type the caller then passes; only the byte layout
 * needs to be self-consistent for the model-range bounds checks to
 * pass, and every format the dispatcher can currently reach
 * (implemented or stubbed) uses 256-wide superblocks). */
static rm_model rm_build_model(void) {
    return rm_build_model_ex(RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
}

typedef struct {
    ds4_gpu_tensor *out, *gate, *up, *mid, *down, *selected, *weights, *x;
} rm_tensors;

static rm_tensors rm_build_tensors(void) {
    rm_tensors t;
    t.out = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_OUT_DIM * sizeof(float));
    t.gate = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.up = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.mid = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    t.down = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    t.selected = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    t.weights = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(float));
    t.x = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t sel[RM_N_TOKENS * RM_N_EXPERT];
    float w[RM_N_TOKENS * RM_N_EXPERT];
    for (int i = 0; i < RM_N_TOKENS * RM_N_EXPERT; i++) {
        sel[i] = i % RM_N_TOTAL_EXPERT;
        w[i] = 1.0f;
    }
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    float xv[RM_N_TOKENS * RM_EXPERT_IN_DIM];
    for (int i = 0; i < RM_N_TOKENS * RM_EXPERT_IN_DIM; i++) xv[i] = 0.01f * (float)i;
    ds4_gpu_tensor_write(t.x, 0, xv, sizeof(xv));
    return t;
}

static void rm_free_tensors(rm_tensors *t) {
    ds4_gpu_tensor_free(t->out);
    ds4_gpu_tensor_free(t->gate);
    ds4_gpu_tensor_free(t->up);
    ds4_gpu_tensor_free(t->mid);
    ds4_gpu_tensor_free(t->down);
    ds4_gpu_tensor_free(t->selected);
    ds4_gpu_tensor_free(t->weights);
    ds4_gpu_tensor_free(t->x);
}

static int rm_call_batch(rm_model *m, rm_tensors *t, uint32_t gate_type,
                         uint32_t down_type, uint32_t n_tokens, bool *mid_is_f16) {
    return ds4_gpu_routed_moe_batch_tensor(
            t->out, t->gate, t->up, t->mid, t->down, m->model, m->model_size,
            m->gate_offset, m->up_offset, m->down_offset, gate_type, down_type,
            m->gate_expert_bytes, m->gate_row_bytes, m->down_expert_bytes,
            m->down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
            t->selected, t->weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t->x,
            /*layer_index=*/0u, n_tokens, mid_is_f16, /*force_resident=*/false);
}

/* n_tokens == 0 is a FAILURE (routed_moe_build_plan, moe_launch.cuh:479),
 * unlike several other entries ported earlier in this project where
 * zero-work is a free success. */
static int test_dispatcher_zero_tokens_fails(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 12u, 12u, 0u, NULL) == 0,
          "dispatcher: n_tokens == 0 must fail, not succeed");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_zero_tokens_fails OK\n");
    return 0;
}

static int test_dispatcher_unrecognised_format_fails(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 99u, 99u, RM_N_TOKENS, NULL) == 0,
          "dispatcher: unrecognised gate/down type pair must fail");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_unrecognised_format_fails OK\n");
    return 0;
}

/* Both iq2_path and q2k_path formats are now implemented: they now
 * reach their own delimited dispatcher block and succeed on a
 * well-formed call, not fail as "not yet implemented". The mxfp4_path
 * sibling test below is unchanged; that format is implemented separately below. */
static int test_dispatcher_iq2_succeeds(void) {
    rm_model m = rm_build_model_ex(RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES);
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 16u, 10u, RM_N_TOKENS, NULL) != 0,
          "dispatcher: iq2_path must succeed now that it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_iq2_succeeds OK\n");
    return 0;
}

static int test_dispatcher_q2k_succeeds(void) {
    rm_model m = rm_build_model_ex(RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES);
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 10u, 10u, RM_N_TOKENS, NULL) != 0,
          "dispatcher: q2k_path must succeed now that it is implemented");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_q2k_succeeds OK\n");
    return 0;
}

/* Both MXFP4 regimes are implemented here: RM_N_TOKENS == 2 falls
 * in the tiny-batch decode regime (n_tokens <= 4), so the dispatcher now
 * takes this call structurally rather than failing before reaching its
 * mxfp4_path block.  rm_build_model's byte layout is Q4_K-shaped (144-byte
 * rows), not MXFP4-shaped (17-byte blocks), so this checks only that the
 * call completes and returns success; test_mxfp4_tiny_batch and
 * test_mxfp4_prefill (below) are what check the actual numeric output
 * against the format oracle, using a real MXFP4-shaped model. */
static int test_dispatcher_mxfp4_now_implemented(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    CHECK(rm_call_batch(&m, &t, 39u, 39u, RM_N_TOKENS, NULL) != 0,
          "dispatcher: mxfp4_path must now succeed on a well-formed tiny-batch call");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_mxfp4_now_implemented OK\n");
    return 0;
}

/* Uses a fully well-formed q4k_path call (the only format this test
 * model's byte layout can actually complete successfully) so the add_in check is proven to fire
 * before a call that would otherwise succeed, not merely alongside other
 * validation failures that would return 0 anyway. */
static int test_dispatcher_add_in_rejected(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    ds4_gpu_tensor *dummy = ds4_gpu_tensor_alloc(64);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              dummy, 0u, false) == 0,
          "dispatcher: non-null add_in must fail on the decode entry");
    ds4_gpu_tensor_free(dummy);
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_add_in_rejected OK\n");
    return 0;
}

static int test_dispatcher_mid_is_f16_written_false(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();
    bool mid_is_f16 = true;
    /* mid_is_f16 must be written unconditionally at entry, independent of
     * whatever this call goes on to return (it now succeeds, but this
     * test's own property does not depend on that outcome). */
    rm_call_batch(&m, &t, 39u, 39u, RM_N_TOKENS, &mid_is_f16);
    CHECK(mid_is_f16 == false,
          "dispatcher: mid_is_f16 must be written false regardless of outcome");
    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_mid_is_f16_written_false OK\n");
    return 0;
}

/* A representative sample of routed_moe_build_plan's own validation
 * rejections (moe_launch.cuh:479-509), each independent of format. */
static int test_dispatcher_build_plan_validation(void) {
    rm_model m = rm_build_model();
    rm_tensors t = rm_build_tensors();

    /* expert_in_dim not a multiple of 256: reuse the batch call but with
     * an ad hoc direct call so expert_in_dim can be perturbed. */
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, /*expert_in_dim=*/255u, RM_EXPERT_MID_DIM,
              RM_OUT_DIM, t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT,
              0.0f, t.x, 0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: expert_in_dim not a multiple of 256 must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, /*n_expert=*/9u, 0.0f,
              t.x, 0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: n_expert above DS4_ROCM_N_EXPERT_USED must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              NULL, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: null out tensor must fail");

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model,
              /*model_size=*/1u /* too small for gate/up/down ranges */,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              0u, RM_N_TOKENS, NULL, false) == 0,
          "build_plan: model range exceeding model_size must fail");

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_dispatcher_build_plan_validation OK\n");
    return 0;
}

/* ---- Q4_K: full decode/batch path against a scalar CPU oracle --------
 *
 * The oracle below reimplements ds4.c's layer_routed_moe_one_prealloc
 * (ds4.c:10984-11092) for the Q4_K case: matvec_q4_k_experts_mid_prequant
 * (ds4.c:8970-9014, via matvec_q4_k_mid_worker at :8946-8965) for gate/up,
 * ds4_quantize_row_q8_K (ds4.c:3336-3370) to requantize the mid
 * activations, and matvec_q4_k_experts_accum_prequant (ds4.c:9042-9078,
 * via matvec_q4_k_accum_worker at :9024-9036) for the down projection,
 * which sums over slots in ascending order inside one worker call -- the
 * same order ds4-sycl-moe-reference.md section 4(a) attributes to a
 * "for i in 0..n_expert_used" loop that this reading of ds4.c did not
 * find; the loop shape differs from the research document's description
 * but the property that matters (ascending slot-order summation,
 * matching moe_sum_kernel) is the same either way, re-confirmed directly
 * against ds4.c rather than trusted from the document.
 *
 * ds4.c's own scalar routines (ds4_vec_dot_q4_K_q8_K, ds4_quantize_row_q8_K)
 * cannot be linked (static, and ds4.c is not part of this build), so
 * every scalar helper below is an independent reimplementation, matching
 * the precedent in tests/test_mxfp4_rocm.c and every oracle earlier in
 * this file. */

/* q4_k_get_scale_min, ds4.c:3514-3521 (and dev_q4_K_get_scale_min,
 * moe.cuh:250-262): unpacks one of 8 6-bit (scale, min) pairs from a
 * Q4_K block's 12-byte scales array. */
static void oracle_q4k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Inverse packing: given 8 (scale, min) pairs (each 0-63), produce the
 * 12-byte scales array a real Q4_K block would carry.  Derived by
 * inverting oracle_q4k_get_scale_min's two branches; self-checked by
 * test_q4k_scale_pack_roundtrip below before being trusted for anything
 * else. */
static void oracle_q4k_pack_scales(uint8_t q[12], const uint8_t sc[8], const uint8_t m[8]) {
    for (int i = 0; i < 4; i++) {
        q[i]     = (uint8_t)((sc[i] & 0x3Fu) | ((uint32_t)(sc[i + 4] >> 4) << 6));
        q[i + 4] = (uint8_t)((m[i] & 0x3Fu) | ((uint32_t)(m[i + 4] >> 4) << 6));
        q[i + 8] = (uint8_t)((sc[i + 4] & 0x0Fu) | ((uint32_t)(m[i + 4] & 0x0Fu) << 4));
    }
}

static int test_q4k_scale_pack_roundtrip(void) {
    uint8_t sc[8], m[8], q[12];
    for (int seed = 0; seed < 5; seed++) {
        for (int j = 0; j < 8; j++) {
            sc[j] = (uint8_t)((seed * 17 + j * 23) % 64);
            m[j] = (uint8_t)((seed * 29 + j * 11) % 64);
        }
        oracle_q4k_pack_scales(q, sc, m);
        for (int j = 0; j < 8; j++) {
            uint8_t got_sc, got_m;
            oracle_q4k_get_scale_min(j, q, &got_sc, &got_m);
            char msg[80];
            snprintf(msg, sizeof(msg), "scale_pack_roundtrip: seed %d j %d sc mismatch", seed, j);
            CHECK(got_sc == sc[j], msg);
            snprintf(msg, sizeof(msg), "scale_pack_roundtrip: seed %d j %d m mismatch", seed, j);
            CHECK(got_m == m[j], msg);
        }
    }
    fprintf(stderr, "  test_q4k_scale_pack_roundtrip OK\n");
    return 0;
}

/* Packs one 144-byte cuda_block_q4_K (ds4_rocm.cu:72-77): d and dmin as
 * f16 bits, 12 bytes of packed scale/min, 128 bytes of packed 4-bit
 * codes (two sub-blocks share each 32-byte qs region, low nibble for the
 * even sub-block, high nibble for the odd one, per moe.cuh:274-287's
 * byte_off/shift math). */
static void oracle_q4k_pack_block(uint8_t out[144], float d, float dmin,
                                  const uint8_t sc[8], const uint8_t m[8],
                                  const uint8_t nib[256]) {
    uint16_t dh = f32_to_f16_bits(d);
    uint16_t dminh = f32_to_f16_bits(dmin);
    memcpy(out, &dh, 2);
    memcpy(out + 2, &dminh, 2);
    oracle_q4k_pack_scales(out + 4, sc, m);
    uint8_t qs[128];
    memset(qs, 0, sizeof(qs));
    for (int j = 0; j < 8; j++) {
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) ? 4 : 0;
        for (int k = 0; k < 32; k++) {
            uint8_t v = nib[j * 32 + k] & 0x0Fu;
            if (shift == 0) qs[byte_off + k] = (uint8_t)((qs[byte_off + k] & 0xF0u) | v);
            else qs[byte_off + k] = (uint8_t)((qs[byte_off + k] & 0x0Fu) | (v << 4));
        }
    }
    memcpy(out + 16, qs, 128);
}

typedef struct {
    int8_t  qs[256];
    int16_t bsums[16];
    float   d;
} oracle_q8k_block;

/* ds4_quantize_row_q8_K / q8_K_quantize_kernel, both cited above
 * test_q8_k_quantize: identical amax scan, iscale = -127/max, round to
 * nearest, clamp to [-128,127], and 16-wide block sums. */
static void oracle_q8k_quantize_block(const float *xr, oracle_q8k_block *out) {
    float amax = 0.0f, maxv = 0.0f;
    for (int i = 0; i < 256; i++) {
        float ax = fabsf(xr[i]);
        if (ax > amax) { amax = ax; maxv = xr[i]; }
    }
    if (amax == 0.0f) {
        out->d = 0.0f;
        memset(out->qs, 0, sizeof(out->qs));
        memset(out->bsums, 0, sizeof(out->bsums));
        return;
    }
    const float iscale = -127.0f / maxv;
    for (int i = 0; i < 256; i++) {
        int v = (int)lrintf(iscale * xr[i]);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out->qs[i] = (int8_t)v;
    }
    for (int g = 0; g < 16; g++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += out->qs[g * 16 + i];
        out->bsums[g] = (int16_t)sum;
    }
    out->d = 1.0f / iscale;
}

/* dev_dot_q4_K_q8_K_block / ds4_vec_dot_q4_K_q8_K, single 256-value block. */
static float oracle_q4k_dot_block(const uint8_t blk[144], const oracle_q8k_block *y) {
    uint16_t dbits, dminbits;
    memcpy(&dbits, blk, 2);
    memcpy(&dminbits, blk + 2, 2);
    const float xd = f16_bits_to_f32(dbits);
    const float xmin = f16_bits_to_f32(dminbits);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int32_t isum = 0, summs = 0;
    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        oracle_q4k_get_scale_min(j, scales, &sc, &m);
        summs += (int32_t)m * (int32_t)(y->bsums[2 * j] + y->bsums[2 * j + 1]);
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) ? 4 : 0;
        int32_t block_sum = 0;
        for (int i = 0; i < 32; i++) {
            block_sum += (int32_t)((qs[byte_off + i] >> shift) & 0x0F) * (int32_t)y->qs[j * 32 + i];
        }
        isum += (int32_t)sc * block_sum;
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

static float oracle_silu(float x) { return x / (1.0f + expf(-x)); }

/* Deterministic per-(expert,row,block) weight-block generator with a
 * non-linear interaction term (spec 6f: pure affine test data makes
 * every element mutually proportional and hides scale-only bugs).  `blk`
 * selects which 256-wide Q8_K chunk of the row this block covers: it is
 * salted into every one of sc/m/nib/d/dmin with its own multiplier so
 * chunk 1 is not a shifted or scaled copy of chunk 0 (spec 6i: adjacent
 * test data must not let a wrong-chunk dot accidentally reproduce the
 * right-chunk answer). */
static void q4k_fill_row(uint8_t out[144], uint32_t phase, uint32_t expert, uint32_t row,
                         uint32_t blk) {
    uint8_t sc[8], m[8], nib[256];
    for (int j = 0; j < 8; j++) {
        sc[j] = (uint8_t)(1u + (phase * 5u + expert * 7u + row * 3u + blk * 41u +
                                (uint32_t)j * 5u) % 40u);
        m[j] = (uint8_t)((phase * 3u + expert * 11u + row * 2u + blk * 31u +
                          (uint32_t)j * 13u) % 20u);
    }
    for (int k = 0; k < 256; k++) {
        nib[k] = (uint8_t)((phase * 19u + expert * 13u + row * 17u + blk * 53u + (uint32_t)k +
                            (expert * row) % 5u + ((uint32_t)k * row) % 7u +
                            ((uint32_t)k * blk) % 11u) % 16u);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row + blk * 3u) % 7u);
    const float dmin = 0.001f * (float)((phase + expert * 2u + row + blk * 2u) % 5u);
    oracle_q4k_pack_block(out, d, dmin, sc, m, nib);
}

/* Spec 6n: clean multiples of 0.01 land iscale*x[i] (Q8_K's quantisation
 * input) on an exact int8 rounding tie far more often than real
 * activations do, and this GPU and the host CPU do not always resolve
 * such a tie the same way (an ULP-level difference in -ffast-math's
 * division, not a logic error). Confirmed by direct measurement: the
 * plain affine formula below, before this dither was added, put 2-7
 * exact ties in every 256-element block across the first eight tokens at
 * IN_DIM=512, one of which (tok=3, block 1) produced a real mismatch once
 * the multi-chunk tests below started exercising block 1 at all. The
 * per-element hashed dither (same construction rm_batch_test_dithered
 * uses at a coarser 1e-2 scale for its stress shape) is small enough to
 * leave every block's magnitude and sign pattern intact but large enough
 * to move iscale*x[i] off the knife edge for both engines: re-measured at
 * zero exact ties (1e-5 tolerance) across 128 blocks spanning 64 tokens
 * after adding it. */
static void q4k_fill_x_row(float *x, uint32_t in_dim, uint32_t tok) {
    for (uint32_t k = 0; k < in_dim; k++) {
        const float base = 0.01f * (float)((int)((tok * 37u + k * 11u + (tok * k) % 13u + 5u) % 200u) - 100);
        uint32_t h = (tok * 2654435761u) ^ (k * 40503u);
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        const float frac = (float)(h % 10007u) / 10007.0f - 0.5f; /* [-0.5, 0.5) */
        x[k] = base + 1.0e-3f * frac;
    }
}

enum { Q4K_PHASE_GATE = 0, Q4K_PHASE_UP = 100, Q4K_PHASE_DOWN = 200 };

/* Fills an rm_model's gate/up/down sections with q4k_fill_row blocks,
 * xq_blocks = RM_EXPERT_IN_DIM/256 blocks per gate/up row and
 * midq_blocks = mid_dim/256 blocks per down row, each block distinct
 * (see q4k_fill_row). */
static void q4k_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim,
                           uint32_t out_dim) {
    const uint32_t xq_blocks = RM_EXPERT_IN_DIM / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            for (uint32_t b = 0; b < xq_blocks; b++) {
                uint8_t blk[144];
                const uint64_t row_off = (uint64_t)row * m->gate_row_bytes + (uint64_t)b * RM_Q4K_BLOCK_BYTES;
                q4k_fill_row(blk, Q4K_PHASE_GATE, e, row, b);
                memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
                q4k_fill_row(blk, Q4K_PHASE_UP, e, row, b);
                memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < midq_blocks; b++) {
                uint8_t blk[144];
                q4k_fill_row(blk, Q4K_PHASE_DOWN, e, row, b);
                memcpy(m->model + m->down_offset + e * m->down_expert_bytes +
                               (uint64_t)row * m->down_row_bytes + (uint64_t)b * RM_Q4K_BLOCK_BYTES,
                       blk, sizeof(blk));
            }
        }
    }
}

/* layer_routed_moe_one_prealloc for the Q4_K case, one token at a time.
 * Quantises x into in_dim/256 Q8_K chunks and mid into mid_dim/256
 * chunks, dotting weight block b against activation chunk b and summing
 * over chunks -- the property the chunk-index ablation in the report
 * exercises. */
static void oracle_q4k_one_token(const rm_model *m, uint32_t in_dim, uint32_t mid_dim,
                                 uint32_t out_dim, uint32_t n_expert,
                                 const int32_t *sel, const float *w, const float *x,
                                 float clamp, float *out) {
    const uint32_t xq_blocks = in_dim / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    oracle_q8k_block *xq = malloc((size_t)xq_blocks * sizeof(oracle_q8k_block));
    for (uint32_t b = 0; b < xq_blocks; b++) {
        oracle_q8k_quantize_block(x + (size_t)b * 256u, &xq[b]);
    }

    float *mid = malloc((size_t)n_expert * mid_dim * sizeof(float));
    for (uint32_t s = 0; s < n_expert; s++) {
        const uint32_t expert = (uint32_t)sel[s];
        for (uint32_t row = 0; row < mid_dim; row++) {
            const uint8_t *gate_row = m->model + m->gate_offset + (uint64_t)expert * m->gate_expert_bytes +
                                      (uint64_t)row * m->gate_row_bytes;
            const uint8_t *up_row = m->model + m->up_offset + (uint64_t)expert * m->gate_expert_bytes +
                                    (uint64_t)row * m->gate_row_bytes;
            float gate = 0.0f, up = 0.0f;
            for (uint32_t b = 0; b < xq_blocks; b++) {
                gate += oracle_q4k_dot_block(gate_row + (size_t)b * RM_Q4K_BLOCK_BYTES, &xq[b]);
                up += oracle_q4k_dot_block(up_row + (size_t)b * RM_Q4K_BLOCK_BYTES, &xq[b]);
            }
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid[(size_t)s * mid_dim + row] = oracle_silu(gate) * up * w[s];
        }
    }

    oracle_q8k_block *midq = malloc((size_t)n_expert * midq_blocks * sizeof(oracle_q8k_block));
    for (uint32_t s = 0; s < n_expert; s++) {
        for (uint32_t b = 0; b < midq_blocks; b++) {
            oracle_q8k_quantize_block(mid + (size_t)s * mid_dim + (size_t)b * 256u,
                                      &midq[(size_t)s * midq_blocks + b]);
        }
    }

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_expert; s++) { /* ascending slot order */
            const uint32_t expert = (uint32_t)sel[s];
            const uint8_t *down_row = m->model + m->down_offset + (uint64_t)expert * m->down_expert_bytes +
                                      (uint64_t)row * m->down_row_bytes;
            for (uint32_t b = 0; b < midq_blocks; b++) {
                acc += oracle_q4k_dot_block(down_row + (size_t)b * RM_Q4K_BLOCK_BYTES,
                                            &midq[(size_t)s * midq_blocks + b]);
            }
        }
        out[row] = acc;
    }
    free(xq);
    free(mid);
    free(midq);
}

static void q4k_fill_selected(int32_t *sel, float *w, uint32_t tok, uint32_t n_expert,
                              uint32_t n_total_expert) {
    for (uint32_t s = 0; s < n_expert; s++) {
        sel[s] = (int32_t)((tok * 3u + s * 2u + 1u) % n_total_expert);
        w[s] = 0.5f + 0.25f * (float)((tok + s) % 3u);
    }
}

/* n_tokens == 1 (decode) against the per-token oracle, at tight tolerance. */
static int test_q4k_decode(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "q4k_decode: call failed");

    float want[RM_OUT_DIM];
    oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                         sel, w, x, 0.0f, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4, "q4k_decode: value mismatch");
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode OK\n");
    return 0;
}

/* Batched path against N ordered per-token oracle calls, for an n_tokens
 * both below and at/above the sorted-pairs threshold (32), so both the
 * untiled and tiled routes are exercised. */
static int test_q4k_batch(uint32_t n_tokens) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t *sel = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    float *w = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(float));
    float *xv = malloc((size_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        q4k_fill_selected(sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT, t,
                          RM_N_EXPERT, RM_N_TOTAL_EXPERT);
        q4k_fill_x_row(xv + (size_t)t * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, t);
    }
    ds4_gpu_tensor_write(selected, 0, sel, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor_write(weights, 0, w, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    bool mid_is_f16 = true;
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, /*layer_index=*/0u, n_tokens,
              &mid_is_f16, /*force_resident=*/false) != 0,
          "q4k_batch: call failed");
    CHECK(mid_is_f16 == false, "q4k_batch: mid_is_f16 must be written false");

    float *got = malloc((size_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));

    float want_row[RM_OUT_DIM];
    for (uint32_t t = 0; t < n_tokens; t++) {
        oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                             sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT,
                             xv + (size_t)t * RM_EXPERT_IN_DIM, 0.0f, want_row);
        for (int i = 0; i < RM_OUT_DIM; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "q4k_batch(n=%u): token %u value mismatch", n_tokens, t);
            CHECK_CLOSE(got[(size_t)t * RM_OUT_DIM + i], want_row[i],
                       fabs(want_row[i]) * 1e-3 + 1e-4, msg);
        }
    }

    free(sel);
    free(w);
    free(xv);
    free(got);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_batch(n_tokens=%u) OK\n", n_tokens);
    return 0;
}

/* Differential check between the decode path (ds4_gpu_routed_moe_one_tensor,
 * always n_tokens == 1) and the batch path (ds4_gpu_routed_moe_batch_tensor
 * called with n_tokens == 1u): both dispatch to the same "decode" gate/up
 * kernel and the same sum6 down kernel, so they must produce identical
 * output on identical input.  This substitutes for a differential test
 * against the F32 fallback path: that fallback turned out to hardcode
 * IQ2_XXS/Q2_K weight-block casts, so it cannot validate
 * Q4_K at all; comparing decode against batch on identical input is the
 * cheapest available cross-check between two independently dispatched
 * real code paths instead. */
static int test_q4k_decode_matches_batch_of_one(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t1 = rm_build_tensors();
    rm_tensors t2 = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/2, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/2);
    ds4_gpu_tensor_write(t1.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t1.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t1.x, 0, x, sizeof(x));
    ds4_gpu_tensor_write(t2.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t2.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t2.x, 0, x, sizeof(x));

    CHECK(ds4_gpu_routed_moe_one_tensor(
              t1.out, t1.gate, t1.up, t1.mid, t1.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t1.selected, t1.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t1.x,
              NULL, 0u, false) != 0,
          "decode_matches_batch: decode call failed");
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t2.out, t2.gate, t2.up, t2.mid, t2.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t2.selected, t2.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t2.x,
              0u, /*n_tokens=*/1u, NULL, false) != 0,
          "decode_matches_batch: batch(n=1) call failed");

    float got1[RM_OUT_DIM], got2[RM_OUT_DIM];
    ds4_gpu_tensor_read(t1.out, 0, got1, sizeof(got1));
    ds4_gpu_tensor_read(t2.out, 0, got2, sizeof(got2));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got1[i], got2[i], fabs(got1[i]) * 1e-4 + 1e-5,
                   "decode_matches_batch: decode vs batch(n=1) mismatch");
    }

    rm_free_tensors(&t1);
    rm_free_tensors(&t2);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode_matches_batch_of_one OK\n");
    return 0;
}

/* The scratch-buffer-reuse precondition (moe_launch.cuh:746) failing must
 * be a clean failure for Q4_K: ROCm's own
 * fallback for this case cannot be reused (it hardcodes IQ2_XXS/Q2_K
 * weight-block casts, wrong for Q4_K bytes). */
static int test_q4k_scratch_precondition_failure(void) {
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    /* gate/down sized only for their literal F32 role, too small to also
     * hold the Q8_K quantisation scratch this path needs. */
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)RM_N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));
    int32_t sel[RM_N_TOKENS * RM_N_EXPERT] = {0, 1, 2, 3};
    float w[RM_N_TOKENS * RM_N_EXPERT] = {1, 1, 1, 1};
    ds4_gpu_tensor_write(selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(weights, 0, w, sizeof(w));

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, 0u, RM_N_TOKENS, NULL,
              false) == 0,
          "q4k_scratch_precondition: undersized gate/down must fail cleanly");

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_scratch_precondition_failure OK\n");
    return 0;
}

/* ---- Selected-expert staging ---------------------------------
 *
 * test_q4k_decode_stages_only_selected is the distinguishing test spec
 * 6w demands: the existing decode/batch tests above pass identically
 * whether or not compaction happened, because compaction changes what
 * gets copied, not what the copy computes. A table of 64 experts with
 * only RM_N_EXPERT (2) selected makes "staged 2" and "staged 64" two
 * different, directly assertable numbers.
 *
 * test_q4k_batch_wide_union_falls_back checks the other side of the same
 * mechanism: once the union of selected experts across a batch is large
 * enough, the dispatcher must fall back to staging the whole table rather
 * than compacting it.
 *
 * test_q4k_decode_identity_remap_ablation is the ablation spec 6n
 * requires: a clean baseline run first, then DS4_ROUTED_MOE_DEBUG_IDENTITY_
 * REMAP forcing compaction to keep the table small while breaking the id
 * remap, confirmed to produce a wrong answer against the same oracle the
 * baseline just matched. */

enum { RM_WIDE_N_TOTAL_EXPERT = 64u };

static int test_q4k_decode_stages_only_selected(void) {
    rm_model m = rm_build_model_n(RM_WIDE_N_TOTAL_EXPERT, RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
    q4k_fill_model(&m, RM_WIDE_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_WIDE_N_TOTAL_EXPERT);
    CHECK(sel[0] != sel[1], "compaction: test fixture must select distinct experts");
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_WIDE_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "compaction: decode call failed");

    const uint32_t staged_experts = ds4_sycl_moe_test_last_staged_expert_count();
    const uint64_t staged_bytes = ds4_sycl_moe_test_last_staged_bytes();
    const uint64_t want_bytes = 2ull * RM_N_EXPERT * m.gate_expert_bytes +
                               (uint64_t)RM_N_EXPERT * m.down_expert_bytes;
    const uint64_t full_table_bytes = 2ull * RM_WIDE_N_TOTAL_EXPERT * m.gate_expert_bytes +
                                      (uint64_t)RM_WIDE_N_TOTAL_EXPERT * m.down_expert_bytes;
    CHECK(staged_experts == RM_N_EXPERT,
          "compaction: decode must stage exactly n_expert_used experts, not the full table");
    CHECK(staged_bytes == want_bytes,
          "compaction: staged byte count must match the compacted table exactly");
    CHECK(staged_bytes < full_table_bytes,
          "compaction: staged bytes must be less than a full-table stage");

    float want[RM_OUT_DIM];
    oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                         sel, w, x, 0.0f, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4,
                    "compaction: value mismatch against the full-table oracle");
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode_stages_only_selected OK (staged %u/%u experts, "
                    "%llu/%llu bytes)\n",
            staged_experts, (unsigned)RM_WIDE_N_TOTAL_EXPERT,
            (unsigned long long)staged_bytes, (unsigned long long)full_table_bytes);
    return 0;
}

/* ---- Expert-parallel decode, owned + compaction composition ---
 *
 * ds4_gpu_routed_moe_one_owned_tensor splits one token's six selected
 * experts across two ranks by ownership (a contiguous half of the expert
 * table each) and must compose that split with the selected-expert
 * compaction: a
 * rank stages only the DISTINCT experts it owns among those six, never
 * its whole owned half. A staged-count-only test could pass with the
 * split silently disabled (falling back to full compaction with no
 * ownership filter); a final-output-only test could pass with staging
 * far too much (a 42x-style regression, invisible in the
 * numbers). This test checks both, plus the property that actually
 * matters operationally: splitting by ownership must not change the
 * answer, only how it is computed.
 *
 *   1. The home rank (owns [0,32)) stages exactly the number of DISTINCT
 *      selected experts below 32 -- not 32 (the whole owned half) and not
 *      6 (the whole selection).
 *   2. The partner rank (owns [32,64), packed into four slots) stages
 *      exactly the number of distinct selected experts at or above 32.
 *   3. Combining the home rank's six unpacked slots with the partner's
 *      four packed slots, via
 *      ds4_gpu_routed_moe_owned_packed_combine_tensor, reproduces the
 *      SAME token output a single unsplit ds4_gpu_routed_moe_one_tensor
 *      call already tested above (oracle_q4k_one_token) would produce. */

enum {
    RM_OWNED_N_EXPERT_USED  = 6u,
    RM_OWNED_N_TOTAL_EXPERT = 64u,
    RM_OWNED_EXPERT_SPLIT   = 32u,
};

static uint32_t rm_owned_count_distinct_on_side(const int32_t *sel, uint32_t n_expert,
                                                uint32_t split, int want_home) {
    int32_t seen[RM_OWNED_N_EXPERT_USED];
    uint32_t n_seen = 0;
    for (uint32_t s = 0; s < n_expert; s++) {
        const int on_home = sel[s] < (int32_t)split;
        if ((want_home && !on_home) || (!want_home && on_home)) continue;
        int dup = 0;
        for (uint32_t i = 0; i < n_seen; i++) {
            if (seen[i] == sel[s]) dup = 1;
        }
        if (!dup) seen[n_seen++] = sel[s];
    }
    return n_seen;
}

static int test_q4k_owned_decode_composes_ownership_with_compaction(void) {
    rm_model m = rm_build_model_n(RM_OWNED_N_TOTAL_EXPERT, RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
    q4k_fill_model(&m, RM_OWNED_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    /* Deliberately straddles the ownership boundary at 32: three experts
     * below (home-owned), three at or above (partner-owned), all six
     * distinct so each rank's staged count is directly the count of
     * slots on its side. */
    const int32_t sel[RM_OWNED_N_EXPERT_USED] = {2, 40, 9, 55, 17, 33};
    float w[RM_OWNED_N_EXPERT_USED];
    for (uint32_t s = 0; s < RM_OWNED_N_EXPERT_USED; s++) w[s] = 0.5f + 0.25f * (float)(s % 3u);
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);

    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(sizeof(sel));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(w));
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor_write(selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(xt, 0, x, sizeof(x));

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate =
            ds4_gpu_tensor_alloc((uint64_t)RM_OWNED_N_EXPERT_USED * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up =
            ds4_gpu_tensor_alloc((uint64_t)RM_OWNED_N_EXPERT_USED * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid =
            ds4_gpu_tensor_alloc((uint64_t)RM_OWNED_N_EXPERT_USED * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *home_down =
            ds4_gpu_tensor_alloc((uint64_t)RM_OWNED_N_EXPERT_USED * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *partner_down =
            ds4_gpu_tensor_alloc((uint64_t)RM_OWNED_N_EXPERT_USED * RM_OUT_DIM * sizeof(float));
    CHECK(out && gate && up && mid && home_down && partner_down && selected && weights && xt,
          "owned decode: tensor allocation");

    CHECK(ds4_gpu_routed_moe_one_owned_tensor(
              out, gate, up, mid, home_down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes, m.gate_row_bytes,
              m.down_expert_bytes, m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM,
              RM_OUT_DIM, selected, weights, RM_OWNED_N_TOTAL_EXPERT, RM_OWNED_N_EXPERT_USED,
              /*resident_expert_base=*/0u, /*resident_expert_count=*/RM_OWNED_EXPERT_SPLIT, 0.0f,
              xt, /*down_output=*/NULL, /*pack_fixed3=*/false, /*shared_prequant=*/NULL) != 0,
          "owned decode: home rank call failed");
    const uint32_t home_unique =
            rm_owned_count_distinct_on_side(sel, RM_OWNED_N_EXPERT_USED, RM_OWNED_EXPERT_SPLIT, 1);
    CHECK(ds4_sycl_moe_test_last_staged_expert_count() == home_unique,
          "owned decode: home rank must stage exactly its distinct owned-and-selected experts");
    CHECK(home_unique > 0 && home_unique < RM_OWNED_EXPERT_SPLIT,
          "owned decode: fixture must actually shrink the home stage");

    CHECK(ds4_gpu_routed_moe_one_owned_tensor(
              out, gate, up, mid, partner_down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes, m.gate_row_bytes,
              m.down_expert_bytes, m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM,
              RM_OUT_DIM, selected, weights, RM_OWNED_N_TOTAL_EXPERT, RM_OWNED_N_EXPERT_USED,
              /*resident_expert_base=*/RM_OWNED_EXPERT_SPLIT,
              /*resident_expert_count=*/RM_OWNED_N_TOTAL_EXPERT - RM_OWNED_EXPERT_SPLIT, 0.0f, xt,
              /*down_output=*/NULL, /*pack_fixed3=*/true, /*shared_prequant=*/NULL) != 0,
          "owned decode: partner rank call failed");
    const uint32_t partner_unique =
            rm_owned_count_distinct_on_side(sel, RM_OWNED_N_EXPERT_USED, RM_OWNED_EXPERT_SPLIT, 0);
    CHECK(ds4_sycl_moe_test_last_staged_expert_count() == partner_unique,
          "owned decode: partner rank must stage exactly its distinct owned-and-selected experts");
    CHECK(partner_unique > 0 && partner_unique < RM_OWNED_N_TOTAL_EXPERT - RM_OWNED_EXPERT_SPLIT,
          "owned decode: fixture must actually shrink the partner stage");

    CHECK(ds4_gpu_routed_moe_owned_packed_combine_tensor(out, home_down, partner_down, selected,
                                                         RM_OUT_DIM, RM_OWNED_EXPERT_SPLIT) != 0,
          "owned decode: packed combine failed");

    float want[RM_OUT_DIM];
    oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
                         RM_OWNED_N_EXPERT_USED, sel, w, x, 0.0f, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4,
                    "owned decode: combined output must match the unsplit oracle");
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(home_down);
    ds4_gpu_tensor_free(partner_down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(xt);
    free(m.model);
    fprintf(stderr,
            "  test_q4k_owned_decode_composes_ownership_with_compaction OK "
            "(home staged %u, partner staged %u, of %u total)\n",
            home_unique, partner_unique, (unsigned)RM_OWNED_N_TOTAL_EXPERT);
    return 0;
}

/* q4k_fill_selected(tok, RM_N_EXPERT=2, RM_N_TOTAL_EXPERT=4) for tok in
 * 0..3 gives {1,3}, {0,2}, {3,1}, {2,0}: the union across all four tokens
 * is every expert in the table (all 4), which exceeds the n_total_expert
 * / 2 fallback threshold (2), so this batch must fall back to staging the
 * full table exactly as a wide prefill batch would. */
static int test_q4k_batch_wide_union_falls_back(void) {
    enum { N_TOKENS = 4u };
    rm_model m = rm_build_model();
    q4k_fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t sel[N_TOKENS * RM_N_EXPERT];
    float w[N_TOKENS * RM_N_EXPERT];
    for (uint32_t tok = 0; tok < N_TOKENS; tok++) {
        q4k_fill_selected(sel + (size_t)tok * RM_N_EXPERT, w + (size_t)tok * RM_N_EXPERT, tok,
                          RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    }
    ds4_gpu_tensor_write(selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(weights, 0, w, sizeof(w));
    float xv[N_TOKENS * RM_EXPERT_IN_DIM];
    for (uint32_t tok = 0; tok < N_TOKENS; tok++) {
        q4k_fill_x_row(xv + (size_t)tok * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, tok);
    }
    ds4_gpu_tensor_write(x, 0, xv, sizeof(xv));

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, 0u, N_TOKENS, NULL,
              false) != 0,
          "compaction: wide-union batch call failed");

    CHECK(ds4_sycl_moe_test_last_staged_expert_count() == RM_N_TOTAL_EXPERT,
          "compaction: a union above the fallback threshold must stage the full table");

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_batch_wide_union_falls_back OK\n");
    return 0;
}

/* The sorted-pairs trap, tested directly: q4k_path only builds sorted
 * pairs (sycl_moe_build_sorted_pairs) once n_tokens >= 32, and that path
 * must see the remapped ids too, not the global ones -- addressing a
 * compacted buffer with an unremapped id "reads whatever happens to be
 * at that offset and returns plausible wrong numbers" per the plan.  A
 * table of 64 experts with every one of 40 tokens drawing from a fixed
 * 4-expert pool keeps the union at 4 (well under the fallback threshold
 * of 32), while n_tokens=40 forces q4k_path into the sorted-pairs regime,
 * so this is the one test in the suite that exercises compaction and
 * sorted pairs together. */
static int test_q4k_batch_compaction_with_sorted_pairs(void) {
    enum { N_TOKENS = 40u, POOL_SIZE = 4u };
    static const int32_t pool[POOL_SIZE] = {5, 19, 31, 47};
    rm_model m = rm_build_model_n(RM_WIDE_N_TOTAL_EXPERT, RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
    q4k_fill_model(&m, RM_WIDE_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t sel[N_TOKENS * RM_N_EXPERT];
    float w[N_TOKENS * RM_N_EXPERT];
    float xv[N_TOKENS * RM_EXPERT_IN_DIM];
    for (uint32_t tok = 0; tok < N_TOKENS; tok++) {
        sel[tok * RM_N_EXPERT + 0] = pool[tok % POOL_SIZE];
        sel[tok * RM_N_EXPERT + 1] = pool[(tok + 1u) % POOL_SIZE];
        w[tok * RM_N_EXPERT + 0] = 0.5f + 0.25f * (float)(tok % 3u);
        w[tok * RM_N_EXPERT + 1] = 0.5f + 0.25f * (float)((tok + 1u) % 3u);
        q4k_fill_x_row(xv + (size_t)tok * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, tok);
    }
    ds4_gpu_tensor_write(selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(x, 0, xv, sizeof(xv));

    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, 12u, 12u, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_WIDE_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, 0u, N_TOKENS, NULL,
              false) != 0,
          "compaction+sorted-pairs: batch call failed");

    CHECK(ds4_sycl_moe_test_last_staged_expert_count() == POOL_SIZE,
          "compaction+sorted-pairs: must stage exactly the 4-expert pool, not the full table");

    float *got = malloc((size_t)N_TOKENS * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor_read(out, 0, got, (uint64_t)N_TOKENS * RM_OUT_DIM * sizeof(float));
    float want_row[RM_OUT_DIM];
    for (uint32_t tok = 0; tok < N_TOKENS; tok++) {
        oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                             sel + (size_t)tok * RM_N_EXPERT, w + (size_t)tok * RM_N_EXPERT,
                             xv + (size_t)tok * RM_EXPERT_IN_DIM, 0.0f, want_row);
        for (int i = 0; i < RM_OUT_DIM; i++) {
            char msg[80];
            snprintf(msg, sizeof(msg),
                    "compaction+sorted-pairs: token %u value mismatch", tok);
            CHECK_CLOSE(got[(size_t)tok * RM_OUT_DIM + i], want_row[i],
                       fabs(want_row[i]) * 1e-3 + 1e-4, msg);
        }
    }

    free(got);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  test_q4k_batch_compaction_with_sorted_pairs OK\n");
    return 0;
}

/* Spec 6n: establish a clean baseline at the exact shape and data the
 * ablation will use before touching anything, so a pre-existing mismatch
 * is never mistaken for the ablation's effect. Reuses
 * test_q4k_decode_stages_only_selected's fixture (64 experts, 2 selected,
 * well inside the compaction threshold) because the ablation only has
 * something to corrupt when compaction is actually active. */
static int test_q4k_decode_identity_remap_ablation(void) {
    rm_model m = rm_build_model_n(RM_WIDE_N_TOTAL_EXPERT, RM_Q4K_BLOCK_BYTES, RM_Q4K_BLOCK_BYTES);
    q4k_fill_model(&m, RM_WIDE_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_WIDE_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    float want[RM_OUT_DIM];
    oracle_q4k_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, RM_N_EXPERT,
                         sel, w, x, 0.0f, want);

    /* Clean baseline, no ablation active. */
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_WIDE_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "ablation baseline: decode call failed");
    float got_baseline[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got_baseline, sizeof(got_baseline));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got_baseline[i], want[i], fabs(want[i]) * 1e-3 + 1e-4,
                    "ablation baseline: must match the oracle before the ablation runs");
    }

    /* Ablated: compaction still runs (same 2-of-64 shape), but the id
     * remap is replaced with (global id) mod (unique count), so the
     * kernels address the compacted table with the wrong slot for every
     * selected expert whose slot position differs from its id mod 2. */
    setenv("DS4_ROUTED_MOE_DEBUG_IDENTITY_REMAP", "1", 1);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_WIDE_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "ablation: decode call failed");
    unsetenv("DS4_ROUTED_MOE_DEBUG_IDENTITY_REMAP");

    float got_ablated[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got_ablated, sizeof(got_ablated));
    int mismatched = 0;
    for (int i = 0; i < RM_OUT_DIM; i++) {
        if (fabs((double)got_ablated[i] - (double)want[i]) > fabs(want[i]) * 1e-3 + 1e-4) {
            mismatched = 1;
            break;
        }
    }
    CHECK(mismatched,
          "ablation: identity-mod remap must diverge from the oracle, and did not");

    /* Restored: the same call after unsetenv must match the oracle again,
     * proving the ablation knob does not leave residual state behind. */
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, 12u, 12u,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_WIDE_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          "ablation restore: decode call failed");
    float got_restored[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got_restored, sizeof(got_restored));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        CHECK_CLOSE(got_restored[i], want[i], fabs(want[i]) * 1e-3 + 1e-4,
                    "ablation restore: must match the oracle again once unset");
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  test_q4k_decode_identity_remap_ablation OK\n");
    return 0;
}

/* ---- IQ2_XXS and Q2_K: full decode/batch paths against scalar oracles --
 *
 * Both oracles reimplement layer_routed_moe_one_prealloc's dispatch for
 * these formats: matvec_iq2_xxs_experts_mid_prequant / matvec_q2_k_experts_
 * mid_prequant (ds4.c:8079, 8377) for gate/up, matvec_iq2_xxs_experts_
 * accum_prequant / matvec_q2_k_experts_accum_prequant (ds4.c:8580, 8297)
 * for down, both dispatched from matvec_experts_mid_prequant /
 * matvec_experts_down_accum_prequant (ds4.c:9452, 9480).  ds4_vec_dot_
 * q2_K_q8_K (ds4.c:3376) and ds4_vec_dot_iq2_xxs_q8_K (ds4.c:3933) were
 * read directly (non-NEON branch, since this machine is x86) and verified
 * field-for-field against dev_dot_q2_K_q8_K_block / dev_dot_iq2_xxs_q8_K_
 * block (moe.cuh:622-645, 101-123): identical aux0/aux1/ls decode, scale
 * indexing and accumulation order in both cases.  As with Q4_K, ds4.c's
 * own scalar routines cannot be linked (static, ds4.c not part of this
 * build), so every helper below is an independent reimplementation. */

/* cuda_iq2xxs_grid / cuda_ksigns_iq2xs, ds4_iq2_tables_cuda.inc: the exact
 * same tables sycl/ds4_sycl_moe.hpp embeds for the kernels under test.
 * Duplicated here rather than shared because this is a plain C
 * translation unit and ds4.c's own IQ2_XXS tables (iq2xxs_signed_grid) are
 * precomputed in a different, though mathematically equivalent, shape
 * (grid and sign already combined) that this file has no way to link to
 * either. */
static const uint8_t g_iq2_signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t g_iq2_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

static uint32_t oracle_popcount_u32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0f0f0f0fu;
    return (v * 0x01010101u) >> 24;
}

static int32_t oracle_iq2_grid_dot8(uint8_t grid_idx, uint8_t sign_idx, const int8_t *q8) {
    uint64_t grid = g_iq2_grid[grid_idx];
    uint8_t raw = g_iq2_signs[sign_idx & 127u];
    uint32_t parity = oracle_popcount_u32((uint32_t)raw) & 1u;
    uint8_t s = (uint8_t)(raw ^ (uint8_t)(parity << 7));
    int32_t sum = 0;
    for (int b = 0; b < 8; b++) {
        int32_t v = (int32_t)(int8_t)((grid >> (b * 8)) & 0xffu);
        if ((s >> b) & 1u) v = -v;
        sum += v * (int32_t)q8[b];
    }
    return sum;
}

static int32_t oracle_iq2_pair_dot16(uint8_t g0, uint8_t s0, uint8_t g1, uint8_t s1, const int8_t *q8) {
    return oracle_iq2_grid_dot8(g0, s0, q8) + oracle_iq2_grid_dot8(g1, s1, q8 + 8);
}

/* Packs one ib32 group (32 elements: 4 grid indices, 4 sign indices, 1
 * scale nibble) into the 4 uint16 codes the real format uses, inverting
 * dev_dot_iq2_xxs_q8_K_block's aux0/aux1 decode exactly. */
static void oracle_iq2_pack_ib32(uint16_t out4[4], uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3,
                                 uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                                 uint32_t ls_nibble) {
    uint32_t aux0 = (uint32_t)a0 | ((uint32_t)a1 << 8) | ((uint32_t)a2 << 16) | ((uint32_t)a3 << 24);
    uint32_t aux1 = (s0 & 127u) | ((s1 & 127u) << 7) | ((s2 & 127u) << 14) | ((s3 & 127u) << 21) |
                    ((ls_nibble & 15u) << 28);
    out4[0] = (uint16_t)(aux0 & 0xffffu);
    out4[1] = (uint16_t)(aux0 >> 16);
    out4[2] = (uint16_t)(aux1 & 0xffffu);
    out4[3] = (uint16_t)(aux1 >> 16);
}

/* Packs one 66-byte cuda_block_iq2_xxs (ds4_rocm.cu:85-88): d (f16) plus
 * 32 packed u16 codes, 8 groups of 4. */
static void oracle_iq2_pack_block(uint8_t out[66], float d, const uint16_t qs[32]) {
    uint16_t dh = f32_to_f16_bits(d);
    memcpy(out, &dh, 2);
    memcpy(out + 2, qs, 64);
}

/* `blk` selects the 256-wide Q8_K chunk this block covers, salted
 * distinctly into every field (see q4k_fill_row's comment for why). */
static void iq2_fill_row(uint8_t out[66], uint32_t phase, uint32_t expert, uint32_t row, uint32_t blk) {
    uint16_t qs[32];
    for (uint32_t g = 0; g < 8u; g++) {
        uint8_t a0 = (uint8_t)((phase * 7u + expert * 13u + row * 5u + blk * 61u + g * 3u) % 256u);
        uint8_t a1 = (uint8_t)((phase * 11u + expert * 17u + row * 7u + blk * 67u + g * 5u +
                                (expert * row) % 17u) % 256u);
        uint8_t a2 = (uint8_t)((phase * 13u + expert * 19u + row * 11u + blk * 71u + g * 7u) % 256u);
        uint8_t a3 = (uint8_t)((phase * 17u + expert * 23u + row * 13u + blk * 73u + g * 11u +
                                (g * row) % 13u) % 256u);
        uint32_t s0 = (phase + expert * 3u + row * 2u + blk * 19u + g) % 128u;
        uint32_t s1 = (phase * 2u + expert + row * 5u + blk * 23u + g * 2u) % 128u;
        uint32_t s2 = (phase * 3u + expert * 5u + row + blk * 29u + g * 3u) % 128u;
        uint32_t s3 = (phase * 5u + expert * 2u + row * 3u + blk * 37u + g * 5u) % 128u;
        uint32_t ls_nibble = (phase + expert + row + blk * 5u + g) % 16u;
        oracle_iq2_pack_ib32(&qs[g * 4u], a0, a1, a2, a3, s0, s1, s2, s3, ls_nibble);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row + blk * 3u) % 7u);
    oracle_iq2_pack_block(out, d, qs);
}

static float oracle_iq2_dot_block(const uint8_t *blk, const oracle_q8k_block *y) {
    uint16_t dbits;
    memcpy(&dbits, blk, 2);
    const float xd = f16_bits_to_f32(dbits);
    const uint16_t *q2 = (const uint16_t *)(blk + 2);
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        uint32_t ls = 2u * (aux1 >> 28) + 1u;
        uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = oracle_iq2_pair_dot16(a0, (uint8_t)((aux1 >> 0) & 127u), a1,
                                             (uint8_t)((aux1 >> 7) & 127u), q8);
        q8 += 16;
        sumi += oracle_iq2_pair_dot16(a2, (uint8_t)((aux1 >> 14) & 127u), a3,
                                      (uint8_t)((aux1 >> 21) & 127u), q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

/* Packs one 84-byte cuda_block_q2_K (ds4_rocm.cu:65-70): 16 bytes of
 * packed (scale, min) nibble pairs, 64 bytes of 2-bit codes, d and dmin
 * as f16.  The per-element byte/shift placement is lifted directly from
 * ds4.c's own accessor q2_k_value_f32 (ds4.c:3494-3505: group = idx/16,
 * q_base = 32*(group/8) + 16*(group&1), shift = ((group/2)&3)*2), which
 * this test also uses as ground truth for the roundtrip check below --
 * not derived by hand from the dot-product loop nesting, to avoid
 * transcribing that nesting incorrectly. */
static void oracle_q2k_pack_block(uint8_t out[84], float d, float dmin,
                                  const uint8_t scales[16], const uint8_t nib[256]) {
    memset(out, 0, 84);
    memcpy(out, scales, 16);
    for (uint32_t idx = 0; idx < 256u; idx++) {
        uint32_t group = idx / 16u;
        uint32_t l = idx - group * 16u;
        uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        uint32_t shift = ((group / 2u) & 3u) * 2u;
        out[16u + q_base + l] = (uint8_t)(out[16u + q_base + l] | ((nib[idx] & 3u) << shift));
    }
    uint16_t dh = f32_to_f16_bits(d), dminh = f32_to_f16_bits(dmin);
    memcpy(out + 80, &dh, 2);
    memcpy(out + 82, &dminh, 2);
}

/* Read back one packed element the same way q2_k_value_f32 would (minus
 * the dequant scale), to self-check oracle_q2k_pack_block before trusting
 * it for anything else. */
static uint8_t oracle_q2k_read_nib(const uint8_t blk[84], uint32_t idx) {
    uint32_t group = idx / 16u;
    uint32_t l = idx - group * 16u;
    uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    uint32_t shift = ((group / 2u) & 3u) * 2u;
    return (uint8_t)((blk[16u + q_base + l] >> shift) & 3u);
}

static int test_q2k_pack_roundtrip(void) {
    uint8_t scales[16], nib[256], blk[84];
    for (int i = 0; i < 16; i++) scales[i] = (uint8_t)((i * 13 + 7) & 0xff);
    for (int i = 0; i < 256; i++) nib[i] = (uint8_t)((i * 3 + i / 7) & 3);
    oracle_q2k_pack_block(blk, 0.5f, 0.25f, scales, nib);
    for (uint32_t idx = 0; idx < 256u; idx++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "q2k_pack_roundtrip: idx %u mismatch", idx);
        CHECK(oracle_q2k_read_nib(blk, idx) == nib[idx], msg);
    }
    fprintf(stderr, "  test_q2k_pack_roundtrip OK\n");
    return 0;
}

/* `blk` selects the 256-wide Q8_K chunk this block covers, salted
 * distinctly into every field (see q4k_fill_row's comment for why). */
static void q2k_fill_row(uint8_t out[84], uint32_t phase, uint32_t expert, uint32_t row, uint32_t blk) {
    uint8_t scales[16], nib[256];
    for (int g = 0; g < 16; g++) {
        uint8_t sc = (uint8_t)(1u + (phase * 5u + expert * 7u + row * 3u + blk * 43u +
                                     (uint32_t)g * 5u) % 15u);
        uint8_t mn = (uint8_t)((phase * 3u + expert * 11u + row * 2u + blk * 47u +
                               (uint32_t)g * 13u) % 15u);
        scales[g] = (uint8_t)((sc & 0x0fu) | (uint8_t)(mn << 4));
    }
    for (int k = 0; k < 256; k++) {
        nib[k] = (uint8_t)((phase * 19u + expert * 13u + row * 17u + blk * 59u + (uint32_t)k +
                            (expert * row) % 5u + ((uint32_t)k * row) % 7u +
                            ((uint32_t)k * blk) % 11u) % 4u);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row + blk * 3u) % 7u);
    const float dmin = 0.001f * (float)((phase + expert * 2u + row + blk * 2u) % 5u);
    oracle_q2k_pack_block(out, d, dmin, scales, nib);
}

static float oracle_q2k_dot_block16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += (int32_t)((q2[i] >> shift) & 3) * (int32_t)q8[i];
    return (float)sum;
}

static float oracle_q2k_dot_block(const uint8_t *blk, const oracle_q8k_block *y) {
    const uint8_t *sc = blk;
    const uint8_t *q2 = blk + 16;
    uint16_t dbits, dminbits;
    memcpy(&dbits, blk + 80, 2);
    memcpy(&dminbits, blk + 82, 2);
    const float dall = y->d * f16_bits_to_f32(dbits);
    const float dmin = y->d * f16_bits_to_f32(dminbits);
    int32_t summs = 0;
    for (int j = 0; j < 16; j++) summs += (int32_t)y->bsums[j] * (int32_t)(sc[j] >> 4);
    int32_t isum = 0;
    int is = 0;
    const int8_t *q8 = y->qs;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * (int32_t)oracle_q2k_dot_block16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * (int32_t)oracle_q2k_dot_block16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

typedef float (*oracle_dot_fn)(const uint8_t *blk, const oracle_q8k_block *y);

/* Generic one-token oracle: layer_routed_moe_one_prealloc's shape (gate/up
 * against Q8_K-quantised x, SwiGLU, requantise mid, down against
 * Q8_K-quantised mid, ascending slot-order sum), parameterised over which
 * dot product gate/up and down each use.  Covers Q4_K (gate_dot==down_dot
 * == oracle_q4k_dot_block would work too, though the existing per-format
 * test above predates this and is left as-is), IQ2_XXS/Q2_K (iq2_path),
 * IQ2_XXS/IQ2_XXS (iq2_iq2_path) and Q2_K/Q2_K (q2k_path).
 *
 * Generalised over chunk count: x is quantised into in_dim/256 Q8_K
 * blocks and mid into mid_dim/256, weight block b dotted against
 * activation chunk b and summed over chunks, matching the shape every
 * tiled kernel under test computes (see oracle_q4k_one_token's twin
 * comment; kept as two functions rather than merged, since this one
 * additionally threads block-byte-size and dot-function parameters the
 * Q4_K-only oracle does not need). */
static void oracle_moe_one_token(const rm_model *m, uint32_t in_dim, uint32_t mid_dim,
                                 uint32_t out_dim, uint32_t gate_block_bytes,
                                 uint32_t down_block_bytes, uint32_t n_expert,
                                 const int32_t *sel, const float *w, const float *x,
                                 float clamp, oracle_dot_fn gate_dot,
                                 oracle_dot_fn down_dot, float *out) {
    const uint32_t xq_blocks = in_dim / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    oracle_q8k_block *xq = malloc((size_t)xq_blocks * sizeof(oracle_q8k_block));
    for (uint32_t b = 0; b < xq_blocks; b++) {
        oracle_q8k_quantize_block(x + (size_t)b * 256u, &xq[b]);
    }

    float *mid = malloc((size_t)n_expert * mid_dim * sizeof(float));
    for (uint32_t s = 0; s < n_expert; s++) {
        const uint32_t expert = (uint32_t)sel[s];
        for (uint32_t row = 0; row < mid_dim; row++) {
            const uint8_t *gate_row = m->model + m->gate_offset + (uint64_t)expert * m->gate_expert_bytes +
                                      (uint64_t)row * m->gate_row_bytes;
            const uint8_t *up_row = m->model + m->up_offset + (uint64_t)expert * m->gate_expert_bytes +
                                    (uint64_t)row * m->gate_row_bytes;
            float gate = 0.0f, up = 0.0f;
            for (uint32_t b = 0; b < xq_blocks; b++) {
                gate += gate_dot(gate_row + (size_t)b * gate_block_bytes, &xq[b]);
                up += gate_dot(up_row + (size_t)b * gate_block_bytes, &xq[b]);
            }
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid[(size_t)s * mid_dim + row] = oracle_silu(gate) * up * w[s];
        }
    }

    oracle_q8k_block *midq = malloc((size_t)n_expert * midq_blocks * sizeof(oracle_q8k_block));
    for (uint32_t s = 0; s < n_expert; s++) {
        for (uint32_t b = 0; b < midq_blocks; b++) {
            oracle_q8k_quantize_block(mid + (size_t)s * mid_dim + (size_t)b * 256u,
                                      &midq[(size_t)s * midq_blocks + b]);
        }
    }

    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_expert; s++) {
            const uint32_t expert = (uint32_t)sel[s];
            const uint8_t *down_row = m->model + m->down_offset + (uint64_t)expert * m->down_expert_bytes +
                                      (uint64_t)row * m->down_row_bytes;
            for (uint32_t b = 0; b < midq_blocks; b++) {
                acc += down_dot(down_row + (size_t)b * down_block_bytes,
                                &midq[(size_t)s * midq_blocks + b]);
            }
        }
        out[row] = acc;
    }
    free(xq);
    free(mid);
    free(midq);
}

/* xq_blocks = RM_EXPERT_IN_DIM/256 blocks per gate/up row, midq_blocks =
 * mid_dim/256 per down row, each distinct (see iq2_fill_row/q2k_fill_row). */
static void iq2_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    const uint32_t xq_blocks = RM_EXPERT_IN_DIM / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            for (uint32_t b = 0; b < xq_blocks; b++) {
                uint8_t blk[RM_IQ2_BLOCK_BYTES];
                const uint64_t row_off = (uint64_t)row * m->gate_row_bytes + (uint64_t)b * RM_IQ2_BLOCK_BYTES;
                iq2_fill_row(blk, Q4K_PHASE_GATE, e, row, b);
                memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
                iq2_fill_row(blk, Q4K_PHASE_UP, e, row, b);
                memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < midq_blocks; b++) {
                uint8_t blk[RM_Q2K_BLOCK_BYTES];
                q2k_fill_row(blk, Q4K_PHASE_DOWN, e, row, b);
                memcpy(m->model + m->down_offset + e * m->down_expert_bytes +
                               (uint64_t)row * m->down_row_bytes + (uint64_t)b * RM_Q2K_BLOCK_BYTES,
                       blk, sizeof(blk));
            }
        }
    }
}

static void iq2iq2_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    const uint32_t xq_blocks = RM_EXPERT_IN_DIM / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            for (uint32_t b = 0; b < xq_blocks; b++) {
                uint8_t blk[RM_IQ2_BLOCK_BYTES];
                const uint64_t row_off = (uint64_t)row * m->gate_row_bytes + (uint64_t)b * RM_IQ2_BLOCK_BYTES;
                iq2_fill_row(blk, Q4K_PHASE_GATE, e, row, b);
                memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
                iq2_fill_row(blk, Q4K_PHASE_UP, e, row, b);
                memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < midq_blocks; b++) {
                uint8_t blk[RM_IQ2_BLOCK_BYTES];
                iq2_fill_row(blk, Q4K_PHASE_DOWN, e, row, b);
                memcpy(m->model + m->down_offset + e * m->down_expert_bytes +
                               (uint64_t)row * m->down_row_bytes + (uint64_t)b * RM_IQ2_BLOCK_BYTES,
                       blk, sizeof(blk));
            }
        }
    }
}

static void q2k_fill_model(rm_model *m, uint32_t n_total_expert, uint32_t mid_dim, uint32_t out_dim) {
    const uint32_t xq_blocks = RM_EXPERT_IN_DIM / 256u;
    const uint32_t midq_blocks = mid_dim / 256u;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            for (uint32_t b = 0; b < xq_blocks; b++) {
                uint8_t blk[RM_Q2K_BLOCK_BYTES];
                const uint64_t row_off = (uint64_t)row * m->gate_row_bytes + (uint64_t)b * RM_Q2K_BLOCK_BYTES;
                q2k_fill_row(blk, Q4K_PHASE_GATE, e, row, b);
                memcpy(m->model + m->gate_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
                q2k_fill_row(blk, Q4K_PHASE_UP, e, row, b);
                memcpy(m->model + m->up_offset + e * m->gate_expert_bytes + row_off, blk, sizeof(blk));
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < midq_blocks; b++) {
                uint8_t blk[RM_Q2K_BLOCK_BYTES];
                q2k_fill_row(blk, Q4K_PHASE_DOWN, e, row, b);
                memcpy(m->model + m->down_offset + e * m->down_expert_bytes +
                               (uint64_t)row * m->down_row_bytes + (uint64_t)b * RM_Q2K_BLOCK_BYTES,
                       blk, sizeof(blk));
            }
        }
    }
}

/* rm_call_one/batch wrappers per format, mirroring the Q4_K precedent
 * (test_q4k_decode/test_q4k_batch) but parameterised over gate_type/
 * down_type and the pair of oracle dot functions, to avoid tripling the
 * decode/batch/decode-matches-batch boilerplate across the three
 * additional formats. */
static int rm_decode_test(const char *name, uint32_t gate_type, uint32_t down_type,
                          uint32_t gate_block_bytes, uint32_t down_block_bytes,
                          void (*fill_model)(rm_model *, uint32_t, uint32_t, uint32_t),
                          oracle_dot_fn gate_dot, oracle_dot_fn down_dot) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/0, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/0);
    ds4_gpu_tensor_write(t.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t.x, 0, x, sizeof(x));

    char msg[96];
    snprintf(msg, sizeof(msg), "%s: call failed", name);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t.out, t.gate, t.up, t.mid, t.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t.selected, t.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t.x,
              /*add_in=*/NULL, /*layer_index=*/0u, /*force_resident=*/false) != 0,
          msg);

    float want[RM_OUT_DIM];
    oracle_moe_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, gate_block_bytes,
                         down_block_bytes, RM_N_EXPERT, sel, w, x, 0.0f, gate_dot, down_dot, want);
    float got[RM_OUT_DIM];
    ds4_gpu_tensor_read(t.out, 0, got, sizeof(got));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        snprintf(msg, sizeof(msg), "%s: value mismatch", name);
        CHECK_CLOSE(got[i], want[i], fabs(want[i]) * 1e-3 + 1e-4, msg);
    }

    rm_free_tensors(&t);
    free(m.model);
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

/* x_dither: added to every element of q4k_fill_x_row's output before
 * quantisation. Zero for every ordinary correctness test. q4k_fill_x_row
 * produces clean multiples of 0.01, so iscale*x[i] (Q8_K's quantisation
 * input, sycl/ds4_sycl_moe.hpp's sycl_moe_q8_k_quantize) lands on an exact
 * half-integer tie far more often than real model activations ever would.
 * At an exact tie, this GPU and this host CPU do not always agree on
 * which side of the tie iscale*x[i] falls, because -ffast-math's division
 * codegen differs enough between them to be off by roughly one ULP at
 * that specific product -- both correctly round-to-nearest-even
 * afterwards (sycl::rint on device, lrintf on host, spec 6k's already-
 * fixed rint-vs-round distinction), but they can be rounding two
 * different inputs. A nonzero x_dither exists so a large stress batch
 * (see test_iq2_batch_stress) can be sized to expose a genuine race
 * without also tripping over this unrelated, expected cross-device
 * floating-point boundary case. */
static int rm_batch_test_dithered(const char *name, uint32_t gate_type, uint32_t down_type,
                                  uint32_t gate_block_bytes, uint32_t down_block_bytes,
                                  uint32_t n_tokens,
                                  void (*fill_model)(rm_model *, uint32_t, uint32_t, uint32_t),
                                  oracle_dot_fn gate_dot, oracle_dot_fn down_dot,
                                  float x_dither) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_EXPERT_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    int32_t *sel = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    float *w = malloc((size_t)n_tokens * RM_N_EXPERT * sizeof(float));
    float *xv = malloc((size_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        q4k_fill_selected(sel + (size_t)t * RM_N_EXPERT, w + (size_t)t * RM_N_EXPERT, t,
                          RM_N_EXPERT, RM_N_TOTAL_EXPERT);
        q4k_fill_x_row(xv + (size_t)t * RM_EXPERT_IN_DIM, RM_EXPERT_IN_DIM, t);
        if (x_dither != 0.0f) {
            for (uint32_t k = 0; k < RM_EXPERT_IN_DIM; k++) {
                /* A cheap integer hash, not a small-modulus formula: a
                 * small-period dither can realign with q4k_fill_x_row's
                 * own mod-200/mod-13/mod-31 structure and reproduce the
                 * exact tie it was meant to break, for a different
                 * (t, k). */
                uint32_t h = (t * 2654435761u) ^ (k * 40503u);
                h ^= h >> 15;
                h *= 2246822519u;
                h ^= h >> 13;
                const float frac = (float)(h % 10007u) / 10007.0f - 0.5f; /* [-0.5, 0.5) */
                xv[(size_t)t * RM_EXPERT_IN_DIM + k] += x_dither * frac;
            }
        }
    }
    ds4_gpu_tensor_write(selected, 0, sel, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(int32_t));
    ds4_gpu_tensor_write(weights, 0, w, (uint64_t)n_tokens * RM_N_EXPERT * sizeof(float));
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * RM_EXPERT_IN_DIM * sizeof(float));

    bool mid_is_f16 = true;
    char msg[96];
    snprintf(msg, sizeof(msg), "%s(n=%u): call failed", name, n_tokens);
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              out, gate, up, mid, down, m.model, m.model_size, m.gate_offset,
              m.up_offset, m.down_offset, gate_type, down_type, m.gate_expert_bytes,
              m.gate_row_bytes, m.down_expert_bytes, m.down_row_bytes,
              RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, selected, weights,
              RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, x, /*layer_index=*/0u, n_tokens,
              &mid_is_f16, /*force_resident=*/false) != 0,
          msg);
    snprintf(msg, sizeof(msg), "%s(n=%u): mid_is_f16 must be written false", name, n_tokens);
    CHECK(mid_is_f16 == false, msg);

    float *got = malloc((size_t)n_tokens * RM_OUT_DIM * sizeof(float));
    ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tokens * RM_OUT_DIM * sizeof(float));

    float want_row[RM_OUT_DIM];
    if (x_dither == 0.0f) {
        /* Ordinary correctness call: every value must match tightly. */
        for (uint32_t t = 0; t < n_tokens; t++) {
            oracle_moe_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, gate_block_bytes,
                                 down_block_bytes, RM_N_EXPERT, sel + (size_t)t * RM_N_EXPERT,
                                 w + (size_t)t * RM_N_EXPERT, xv + (size_t)t * RM_EXPERT_IN_DIM, 0.0f,
                                 gate_dot, down_dot, want_row);
            for (int i = 0; i < RM_OUT_DIM; i++) {
                snprintf(msg, sizeof(msg), "%s(n=%u): token %u value mismatch", name, n_tokens, t);
                CHECK_CLOSE(got[(size_t)t * RM_OUT_DIM + i], want_row[i],
                           fabs(want_row[i]) * 1e-3 + 1e-4, msg);
            }
        }
    } else {
        /* A dithered run is a large-batch concurrency stress shape (see
         * test_iq2_batch_stress), not a bit-parity check: at this many
         * independent Q8_K quantisation decisions (n_tokens *
         * RM_EXPERT_IN_DIM for gate/up, plus one more per down
         * projection, over a million total for this shape), a handful
         * will land within one ULP of an exact rounding tie no matter how
         * the input is dithered, and this GPU and this host CPU do not
         * always resolve the same tie the same way -- both are
         * internally consistent round-to-nearest-even
         * (sycl_moe_q8_k_quantize's sycl::rint versus oracle_q8k_
         * quantize_block's lrintf, spec 6k's already-fixed rint-vs-round
         * distinction), they can merely be rounding inputs that differ by
         * one ULP because -ffast-math's division codegen is not required
         * to be bit-identical across host and device. A handful of such
         * one-code-out-of-256 divergences is expected and unrelated to
         * the barrier this test exists to guard; a missing barrier reads
         * uninitialised or another sub-group's not-yet-written local
         * memory, which corrupts a large, obvious fraction of the output
         * (empirically 1.2% of all values, essentially all of them, when
         * this barrier was actually removed to confirm), not one value in
         * a quarter of a million. So this shape counts mismatches at a 1%
         * relative tolerance (loose enough to absorb a real tie, tight
         * enough that a wrong value is still wrong) and requires the
         * mismatch RATE to stay far below what a real corruption
         * produces, rather than requiring every single value to match. */
        uint64_t total = 0, mismatches = 0;
        for (uint32_t t = 0; t < n_tokens; t++) {
            oracle_moe_one_token(&m, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM, gate_block_bytes,
                                 down_block_bytes, RM_N_EXPERT, sel + (size_t)t * RM_N_EXPERT,
                                 w + (size_t)t * RM_N_EXPERT, xv + (size_t)t * RM_EXPERT_IN_DIM, 0.0f,
                                 gate_dot, down_dot, want_row);
            for (int i = 0; i < RM_OUT_DIM; i++) {
                float g = got[(size_t)t * RM_OUT_DIM + i];
                float wv = want_row[i];
                total++;
                if (fabs((double)g - (double)wv) > fabs((double)wv) * 1.0e-2 + 1e-3) mismatches++;
            }
        }
        snprintf(msg, sizeof(msg), "%s(n=%u): mismatch rate %llu/%llu too high for tie noise",
                name, n_tokens, (unsigned long long)mismatches, (unsigned long long)total);
        CHECK(mismatches * 1000ull < total, msg); /* < 0.1% mismatched at 1% relative tolerance */
    }

    free(sel);
    free(w);
    free(xv);
    free(got);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(m.model);
    fprintf(stderr, "  %s(n_tokens=%u) OK\n", name, n_tokens);
    return 0;
}

static int rm_batch_test(const char *name, uint32_t gate_type, uint32_t down_type,
                         uint32_t gate_block_bytes, uint32_t down_block_bytes,
                         uint32_t n_tokens,
                         void (*fill_model)(rm_model *, uint32_t, uint32_t, uint32_t),
                         oracle_dot_fn gate_dot, oracle_dot_fn down_dot) {
    return rm_batch_test_dithered(name, gate_type, down_type, gate_block_bytes,
                                  down_block_bytes, n_tokens, fill_model, gate_dot, down_dot,
                                  0.0f);
}

/* Decode versus batch(n=1) cross-check, mirroring test_q4k_decode_matches_
 * batch_of_one: both dispatch to the same decode gate/up kernel and the
 * same direct-to-out down kernel, so they must agree exactly (well within
 * float-reordering tolerance) on identical input. */
static int rm_decode_matches_batch_of_one(const char *name, uint32_t gate_type,
                                          uint32_t down_type, uint32_t gate_block_bytes,
                                          uint32_t down_block_bytes,
                                          void (*fill_model)(rm_model *, uint32_t, uint32_t,
                                                             uint32_t)) {
    rm_model m = rm_build_model_ex(gate_block_bytes, down_block_bytes);
    fill_model(&m, RM_N_TOTAL_EXPERT, RM_EXPERT_MID_DIM, RM_OUT_DIM);
    rm_tensors t1 = rm_build_tensors();
    rm_tensors t2 = rm_build_tensors();

    int32_t sel[RM_N_EXPERT];
    float w[RM_N_EXPERT];
    float x[RM_EXPERT_IN_DIM];
    q4k_fill_selected(sel, w, /*tok=*/2, RM_N_EXPERT, RM_N_TOTAL_EXPERT);
    q4k_fill_x_row(x, RM_EXPERT_IN_DIM, /*tok=*/2);
    ds4_gpu_tensor_write(t1.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t1.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t1.x, 0, x, sizeof(x));
    ds4_gpu_tensor_write(t2.selected, 0, sel, sizeof(sel));
    ds4_gpu_tensor_write(t2.weights, 0, w, sizeof(w));
    ds4_gpu_tensor_write(t2.x, 0, x, sizeof(x));

    char msg[96];
    snprintf(msg, sizeof(msg), "%s: decode call failed", name);
    CHECK(ds4_gpu_routed_moe_one_tensor(
              t1.out, t1.gate, t1.up, t1.mid, t1.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t1.selected, t1.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t1.x,
              NULL, 0u, false) != 0,
          msg);
    snprintf(msg, sizeof(msg), "%s: batch(n=1) call failed", name);
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              t2.out, t2.gate, t2.up, t2.mid, t2.down, m.model, m.model_size,
              m.gate_offset, m.up_offset, m.down_offset, gate_type, down_type,
              m.gate_expert_bytes, m.gate_row_bytes, m.down_expert_bytes,
              m.down_row_bytes, RM_EXPERT_IN_DIM, RM_EXPERT_MID_DIM, RM_OUT_DIM,
              t2.selected, t2.weights, RM_N_TOTAL_EXPERT, RM_N_EXPERT, 0.0f, t2.x,
              0u, /*n_tokens=*/1u, NULL, false) != 0,
          msg);

    float got1[RM_OUT_DIM], got2[RM_OUT_DIM];
    ds4_gpu_tensor_read(t1.out, 0, got1, sizeof(got1));
    ds4_gpu_tensor_read(t2.out, 0, got2, sizeof(got2));
    for (int i = 0; i < RM_OUT_DIM; i++) {
        snprintf(msg, sizeof(msg), "%s: decode vs batch(n=1) mismatch", name);
        CHECK_CLOSE(got1[i], got2[i], fabs(got1[i]) * 1e-4 + 1e-5, msg);
    }

    rm_free_tensors(&t1);
    rm_free_tensors(&t2);
    free(m.model);
    fprintf(stderr, "  %s OK\n", name);
    return 0;
}

/* ---- iq2_path: gate_type==16 (IQ2_XXS), down_type==10 (Q2_K) ---------- */

static int test_iq2_decode(void) {
    return rm_decode_test("test_iq2_decode", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES,
                          iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
static int test_iq2_batch_small(void) {
    return rm_batch_test("test_iq2_batch", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 3u,
                         iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
static int test_iq2_batch_large(void) {
    return rm_batch_test("test_iq2_batch", 16u, 10u, RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 40u,
                         iq2_fill_model, oracle_iq2_dot_block, oracle_q2k_dot_block);
}
/* Spec 6j: a barrier ablation on sycl_moe_iq2_gate_up_mid_tile8's local-
 * memory staging did not fail at the shapes above (n_tokens up to 40,
 * tile_capacity in the single digits) -- consistent with the general
 * finding that a launch too small to create real cross-sub-group contention
 * cannot expose a missing barrier at all. At n_tokens=4096 (tile_capacity
 * in the hundreds), dropping the barrier reproduced a wrong value at
 * token 341 reliably across repeated runs, so this shape is kept as a
 * permanent regression test rather than a one-off ablation probe. */
static int test_iq2_batch_stress(void) {
    return rm_batch_test_dithered("test_iq2_batch_stress", 16u, 10u, RM_IQ2_BLOCK_BYTES,
                                  RM_Q2K_BLOCK_BYTES, 4096u, iq2_fill_model,
                                  oracle_iq2_dot_block, oracle_q2k_dot_block, 1.0e-2f);
}

static int test_iq2_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_iq2_decode_matches_batch_of_one", 16u, 10u,
                                          RM_IQ2_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, iq2_fill_model);
}

/* ---- iq2_iq2_path: gate_type==16, down_type==16 (both IQ2_XXS) -------- */

static int test_iq2iq2_decode(void) {
    return rm_decode_test("test_iq2iq2_decode", 16u, 16u, RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES,
                          iq2iq2_fill_model, oracle_iq2_dot_block, oracle_iq2_dot_block);
}
static int test_iq2iq2_batch(void) {
    return rm_batch_test("test_iq2iq2_batch", 16u, 16u, RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES, 12u,
                         iq2iq2_fill_model, oracle_iq2_dot_block, oracle_iq2_dot_block);
}
static int test_iq2iq2_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_iq2iq2_decode_matches_batch_of_one", 16u, 16u,
                                          RM_IQ2_BLOCK_BYTES, RM_IQ2_BLOCK_BYTES,
                                          iq2iq2_fill_model);
}

/* ---- q2k_path: gate_type==10, down_type==10 (both Q2_K) --------------- */

static int test_q2k_decode(void) {
    return rm_decode_test("test_q2k_decode", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES,
                          q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_batch_small(void) {
    return rm_batch_test("test_q2k_batch", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 5u,
                         q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_batch_large(void) {
    return rm_batch_test("test_q2k_batch", 10u, 10u, RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, 40u,
                         q2k_fill_model, oracle_q2k_dot_block, oracle_q2k_dot_block);
}
static int test_q2k_decode_matches_batch_of_one(void) {
    return rm_decode_matches_batch_of_one("test_q2k_decode_matches_batch_of_one", 10u, 10u,
                                          RM_Q2K_BLOCK_BYTES, RM_Q2K_BLOCK_BYTES, q2k_fill_model);
}

/* An ablation ("route q2k_path inputs through the
 * untagged down kernels and confirm the test catches the difference") is
 * the second, explicitly anticipated outcome rather than the first: in
 * this port, q2k_path's down projection and iq2_path's down projection
 * are the same function, sycl_moe_q2k_down_direct
 * (sycl/ds4_sycl_moe.hpp), because both are validated against the same
 * Q8_K-prequantised-mid CPU oracle (matvec_q2_k_experts_accum_prequant,
 * ds4.c:8297) rather than against ROCm's own kernel selection. There is
 * no separate "route through the other kernel" test to write: both
 * test_q2k_decode/test_q2k_batch above and test_iq2_decode/test_iq2_batch
 * already exercise this identical function (confirmed by reading the
 * dispatch call sites in sycl/ds4_sycl_moe_launch.hpp, not inferred), so
 * running one format's test already is running the other's down kernel.
 * ROCm's own q2k_path and iq2_path down kernels genuinely differ
 * (moe_down_q2K_expert_batch_sharedmid_kernel's raw-float mid versus
 * moe_down_sum6_qwarp32_kernel's Q8_K-quantised mid); this port merges
 * them deliberately, so they are alike by design here, not by an
 * ablation that failed to discriminate. */

/* The down projection's central property, tested directly rather than through the ABI:
 * the down projection sums in ascending SLOT order, matching
 * moe_sum_kernel and layer_routed_moe_one_prealloc, not ascending
 * expert-id order (ds4-sycl-moe-reference.md section 4(a)). Every ABI
 * test above uses RM_N_EXPERT == 2, and IEEE754 addition of exactly two
 * values is exactly commutative (a+b bit-equals b+a always), so no
 * 2-term test can ever catch an order bug. This test uses 3 slots
 * engineered so two extreme, exactly-opposite-sign values cancel to
 * precisely 0.0 only when summed adjacently, and a third, ~6e8 times
 * smaller value survives only when it is not the first term combined
 * with either extreme -- summing in slot order [big+, big-, small]
 * gives exactly `small`; any other order that separates the two extremes
 * gives exactly 0.0 (verified by hand: float addition of two bit-exact
 * negations is exact regardless of what else is in flight, and 6e8 is
 * far past float32's ~1.19e-7 relative precision, so adding `small` to
 * either extreme first always rounds it away). */
static int test_q2k_down_slot_order(void) {
    enum { N_EXPERT = 3, N_TOTAL_EXPERT = 6 };
    uint8_t scales[16], nib[256];
    for (int g = 0; g < 16; g++) scales[g] = 15u; /* scale=15, min=0 */
    /* Constant code (not alternating): x below is also constant, so the
     * per-element products all share one sign and accumulate rather than
     * partially cancelling, which a symmetric-around-zero x would do. */
    for (int k = 0; k < 256; k++) nib[k] = 1u;

    const uint16_t d_pos_bits = f32_to_f16_bits(60000.0f);
    const uint16_t d_neg_bits = (uint16_t)(d_pos_bits ^ 0x8000u); /* exact negation */
    const uint16_t d_small_bits = f32_to_f16_bits(0.001f);
    const uint16_t zero16 = 0;

    uint8_t down[N_TOTAL_EXPERT][84];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) {
        memset(down[e], 0, sizeof(down[e]));
        memcpy(down[e], scales, 16);
        for (uint32_t idx = 0; idx < 256u; idx++) {
            uint32_t group = idx / 16u;
            uint32_t l = idx - group * 16u;
            uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
            uint32_t shift = ((group / 2u) & 3u) * 2u;
            down[e][16u + q_base + l] =
                (uint8_t)(down[e][16u + q_base + l] | ((nib[idx] & 3u) << shift));
        }
        memcpy(down[e] + 82, &zero16, 2); /* dmin = 0 for every expert */
    }
    memcpy(down[0] + 80, &d_pos_bits, 2);   /* expert 0: +big */
    memcpy(down[5] + 80, &d_neg_bits, 2);   /* expert 5: -big, exact negation */
    memcpy(down[2] + 80, &d_small_bits, 2); /* expert 2: small */

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 10.0f; /* constant: no cross-element cancellation */
    oracle_q8k_block xq;
    oracle_q8k_quantize_block(x, &xq);

    uint8_t midq_bytes[3 * Q8_K_BLOCK_BYTES];
    for (int s = 0; s < N_EXPERT; s++) {
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES, &xq.d, 4);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 4, xq.qs, 256);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 260, xq.bsums, 32);
    }

    uint8_t down_flat[N_TOTAL_EXPERT * 84];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) memcpy(down_flat + e * 84, down[e], 84);

    /* Slot order: expert 0 (+big), expert 5 (-big), expert 2 (small). */
    int32_t sel[N_EXPERT] = {0, 5, 2};
    float got = 0.0f;
    CHECK(ds4_sycl_moe_test_q2k_down_direct(down_flat, midq_bytes, sel,
                                            /*down_expert_bytes=*/84u,
                                            /*down_row_bytes=*/84u, /*midq_blocks=*/1u,
                                            /*out_dim=*/1u, N_EXPERT, /*n_tokens=*/1u,
                                            N_TOTAL_EXPERT, &got) != 0,
          "q2k_down_slot_order: call failed");

    float want = oracle_q2k_dot_block(down[2], &xq);
    CHECK(fabsf(want) > 1.0f, "q2k_down_slot_order: test data too weak (small term near zero)");
    CHECK_CLOSE(got, want, fabs(want) * 1e-3 + 1e-4,
               "q2k_down_slot_order: slot-order sum must equal the small term exactly, "
               "not be swallowed by the two cancelling extremes");
    fprintf(stderr, "  test_q2k_down_slot_order OK (got %.6g, want %.6g)\n", (double)got,
            (double)want);
    return 0;
}

/* ---- MXFP4: decode/tiny-batch regime (n_tokens <= 4) against the format
 * oracle above --------------------------------------------------------
 *
 * Driven through the component-level side-door hooks
 * (ds4_sycl_moe_test_mxfp4_gate_up_mid_decode / _down_sum6) rather than the
 * public ds4_gpu_routed_moe_{one,batch}_tensor entries, so a gate/up-mid
 * failure and a down failure are never conflated into one combined
 * mismatch.  test_mxfp4_prefill below exercises the same oracle through
 * the real public entries instead, covering both regimes end to end. */

enum {
    MX_IN_DIM         = 512, /* 2 Q8_K chunks per row */
    MX_MID_DIM        = 512, /* 2 Q8_K chunks per row */
    MX_OUT_DIM        = 512,
    MX_N_EXPERT       = 6,
    MX_N_TOTAL_EXPERT = 32,
    MX_XQ_BLOCKS      = MX_IN_DIM / 256,    /* Q8_K chunks per gate/up row */
    MX_MIDQ_BLOCKS    = MX_MID_DIM / 256,   /* Q8_K chunks per down row */
    MX_GATE_MXFP4_BLOCKS = MX_IN_DIM / MXFP4_QK,  /* mxfp4 blocks per gate/up row */
    MX_DOWN_MXFP4_BLOCKS = MX_MID_DIM / MXFP4_QK, /* mxfp4 blocks per down row */
    MX_N_PATTERN      = 4,
};
/* A finite clamp near the natural magnitude of these dot products makes a
 * legitimate, tolerance-level rounding difference between the oracle and
 * the kernel land on opposite sides of the clamp boundary for some rows,
 * amplifying a tiny pre-clamp difference into a large post-clamp one: a
 * clamp-boundary-sensitivity artefact of the test data, not a kernel bug
 * (diagnosed by an initial FAIL at mid[3065] whose magnitude matched
 * exactly what an up-side clamp flip would produce). Effectively disabled
 * here so the oracle comparison tests the dot product and SwiGLU, not
 * clamp-boundary placement; SwiGLU clamping itself is exercised by the
 * shared expert and Q4_K test suites already. */
static const float MX_CLAMP = 1.0e6f;

enum { MXFP4_PHASE_GATE = 0, MXFP4_PHASE_UP = 1000, MXFP4_PHASE_DOWN = 2000 };

/* Deterministic per-(expert,row,block) weight-row generator with a
 * non-linear interaction term (spec 6f: pure affine test data makes every
 * element mutually proportional and hides scale-only bugs). */
static void mxfp4_fill_row(oracle_mxfp4_block *blocks, uint32_t n_blocks, uint32_t phase,
                           uint32_t expert, uint32_t row) {
    for (uint32_t b = 0; b < n_blocks; b++) {
        uint32_t key = phase * 977u + expert * 131u + row * 17u + b * 7u +
                       (expert * row) % 11u + ((row + b) * expert) % 13u;
        key = key * 2654435761u;
        /* A narrow exponent band (matching tests/test_mxfp4_rocm.c's
         * fill_matrix: 120 + key%5): a wide spread here would let two
         * blocks in the same dot product differ in scale by many powers
         * of two, which is not representative of a real quantised weight
         * row and inflates rounding error well past any tolerance a real
         * checkpoint would need. */
        blocks[b].e = (uint8_t)(120u + ((key >> 24) % 5u));
        for (int i = 0; i < 16; i++) {
            const uint32_t h = (key + (uint32_t)i * 0x9e3779b9u) * 2654435761u;
            blocks[b].qs[i] = (uint8_t)(h >> 24);
        }
    }
}

/* dot_mxfp4_q8_K, tests/test_mxfp4_rocm.c:121-140, re-expressed over this
 * file's own independently derived and ablated oracle_mxfp4_e8m0_scale /
 * oracle_mxfp4_code_value_ocp rather than that file's mxfp4_values table. */
static float oracle_mxfp4_dot_q8k(const oracle_mxfp4_block *row, const oracle_q8k_block *x,
                                  uint32_t input_dim) {
    float sum = 0.0f;
    const uint32_t mxfp4_per_q8 = 256u / MXFP4_QK;
    for (uint32_t block = 0; block < input_dim / MXFP4_QK; block++) {
        const oracle_mxfp4_block *b = row + block;
        const oracle_q8k_block *xb = x + block / mxfp4_per_q8;
        const uint32_t q8_offset = (block % mxfp4_per_q8) * MXFP4_QK;
        const float scale = oracle_mxfp4_e8m0_scale(b->e) * xb->d;
        for (uint32_t i = 0; i < MXFP4_QK / 2u; i++) {
            const uint8_t byte = b->qs[i];
            sum += scale * oracle_mxfp4_code_value_ocp(byte & 0x0f) * (float)xb->qs[q8_offset + i];
            sum += scale * oracle_mxfp4_code_value_ocp(byte >> 4) *
                   (float)xb->qs[q8_offset + i + MXFP4_QK / 2u];
        }
    }
    return sum;
}

/* One combined host buffer plus offsets, matching rm_model's shape: lets
 * the same weight table back both the component-level side-door hooks
 * (which take separate gate/up/down pointers, so t.model+t.gate_offset
 * etc. are passed directly) and the public
 * ds4_gpu_routed_moe_{one,batch}_tensor entries (which need one
 * model_map plus offsets). */
typedef struct {
    unsigned char *model;
    uint64_t model_size;
    uint64_t gate_offset, up_offset, down_offset;
    uint64_t gate_expert_bytes, gate_row_bytes;
    uint64_t down_expert_bytes, down_row_bytes;
} mx_model;

static mx_model mx_build_model(void) {
    mx_model m;
    /* Row byte size is the number of MXFP4 BLOCKS per row (in_dim/32), not
     * the number of Q8_K CHUNKS per row (in_dim/256, MX_XQ_BLOCKS/
     * MX_MIDQ_BLOCKS): those are 8x smaller, and an earlier version of
     * this function used them here, understating every row's stride by 8x
     * and aliasing every 8th row's data across the rows in between.
     * Caught by test_mxfp4_tiny_batch's oracle check (a hand-recompute of
     * expert 31 row 505 turned up a corrupted block with e=76, far
     * outside the generator's intended [120,125) band). */
    m.gate_row_bytes = (uint64_t)MX_GATE_MXFP4_BLOCKS * sizeof(oracle_mxfp4_block);
    m.gate_expert_bytes = (uint64_t)MX_MID_DIM * m.gate_row_bytes;
    m.down_row_bytes = (uint64_t)MX_DOWN_MXFP4_BLOCKS * sizeof(oracle_mxfp4_block);
    m.down_expert_bytes = (uint64_t)MX_OUT_DIM * m.down_row_bytes;
    m.gate_offset = 0;
    m.up_offset = m.gate_expert_bytes * MX_N_TOTAL_EXPERT;
    m.down_offset = m.up_offset + m.gate_expert_bytes * MX_N_TOTAL_EXPERT;
    m.model_size = m.down_offset + m.down_expert_bytes * MX_N_TOTAL_EXPERT;
    m.model = malloc((size_t)m.model_size);
    for (uint32_t e = 0; e < MX_N_TOTAL_EXPERT; e++) {
        for (uint32_t row = 0; row < MX_MID_DIM; row++) {
            mxfp4_fill_row((oracle_mxfp4_block *)(m.model + m.gate_offset +
                                                  (uint64_t)e * m.gate_expert_bytes +
                                                  (uint64_t)row * m.gate_row_bytes),
                           MX_GATE_MXFP4_BLOCKS, MXFP4_PHASE_GATE, e, row);
            mxfp4_fill_row((oracle_mxfp4_block *)(m.model + m.up_offset +
                                                  (uint64_t)e * m.gate_expert_bytes +
                                                  (uint64_t)row * m.gate_row_bytes),
                           MX_GATE_MXFP4_BLOCKS, MXFP4_PHASE_UP, e, row);
        }
        for (uint32_t row = 0; row < MX_OUT_DIM; row++) {
            mxfp4_fill_row((oracle_mxfp4_block *)(m.model + m.down_offset +
                                                  (uint64_t)e * m.down_expert_bytes +
                                                  (uint64_t)row * m.down_row_bytes),
                           MX_DOWN_MXFP4_BLOCKS, MXFP4_PHASE_DOWN, e, row);
        }
    }
    return m;
}

static void mx_free_model(mx_model *m) { free(m->model); }

/* Four routing patterns, cycled by token index, matching
 * tests/test_mxfp4_rocm.c's init_patterns shape: expert ids at both ends
 * of the [0, MX_N_TOTAL_EXPERT) range (0 and 31 both appear), uneven
 * occupancy (expert 31 appears in three of the four patterns, most
 * experts in none), and non-uniform per-slot weights (spec 6i: expert ids
 * and occupancy are test data, not neutral labels). */
static const int32_t mx_pattern_selected[MX_N_PATTERN][MX_N_EXPERT] = {
    { 0, 1, 2, 3, 4, 31 },
    { 5, 17, 20, 25, 30, 31 },
    { 31, 0, 16, 8, 3, 24 },
    { 2, 11, 19, 27, 5, 29 },
};

static void mx_build_pattern(uint32_t p, int32_t sel[MX_N_EXPERT], float w[MX_N_EXPERT],
                             float *x) {
    memcpy(sel, mx_pattern_selected[p], sizeof(mx_pattern_selected[p]));
    for (uint32_t s = 0; s < MX_N_EXPERT; s++) w[s] = 0.5f + 0.25f * (float)((p + s) % 3u);
    q4k_fill_x_row(x, MX_IN_DIM, p * 97u + 5u); /* generic non-linear generator, spec 6f */
}

/* layer_routed_moe_one_prealloc's shape, MXFP4 case: gate/up dot against
 * Q8_K-quantised x, SwiGLU with the router weight folded in at the mid
 * stage (confirmed against rocm/ds4_rocm_moe.cuh:2108-2229's epilogue,
 * not assumed). */
static void oracle_mxfp4_gate_up_mid(const mx_model *m, const int32_t *sel, const float *w,
                                     const float *x, float clamp, float *mid_out) {
    oracle_q8k_block xq[MX_XQ_BLOCKS];
    for (uint32_t b = 0; b < MX_XQ_BLOCKS; b++) {
        oracle_q8k_quantize_block(x + (uint64_t)b * 256u, &xq[b]);
    }
    for (uint32_t s = 0; s < MX_N_EXPERT; s++) {
        const uint32_t expert = (uint32_t)sel[s];
        for (uint32_t row = 0; row < MX_MID_DIM; row++) {
            const oracle_mxfp4_block *gate_row = (const oracle_mxfp4_block *)
                    (m->model + m->gate_offset + (uint64_t)expert * m->gate_expert_bytes +
                     (uint64_t)row * m->gate_row_bytes);
            const oracle_mxfp4_block *up_row = (const oracle_mxfp4_block *)
                    (m->model + m->up_offset + (uint64_t)expert * m->gate_expert_bytes +
                     (uint64_t)row * m->gate_row_bytes);
            float gate = oracle_mxfp4_dot_q8k(gate_row, xq, MX_IN_DIM);
            float up = oracle_mxfp4_dot_q8k(up_row, xq, MX_IN_DIM);
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            mid_out[(size_t)s * MX_MID_DIM + row] = oracle_silu(gate) * up * w[s];
        }
    }
}

/* Stage-isolated down half, tests/test_mxfp4_rocm.c's check_out_from_gpu_mid
 * pattern: takes an already-computed mid (the GPU kernel's own, when
 * called from a test) rather than re-deriving it from the oracle, so a
 * down-kernel bug is distinguishable from a gate/up bug. Ascending
 * slot order, matching sycl_moe_sum / moe_sum_kernel. */
static void oracle_mxfp4_down_from_mid(const mx_model *m, const int32_t *sel, const float *mid,
                                       float *out) {
    oracle_q8k_block midq[MX_N_EXPERT][MX_MIDQ_BLOCKS];
    for (uint32_t s = 0; s < MX_N_EXPERT; s++) {
        for (uint32_t b = 0; b < MX_MIDQ_BLOCKS; b++) {
            oracle_q8k_quantize_block(mid + (uint64_t)s * MX_MID_DIM + (uint64_t)b * 256u,
                                      &midq[s][b]);
        }
    }
    for (uint32_t row = 0; row < MX_OUT_DIM; row++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < MX_N_EXPERT; s++) {
            const uint32_t expert = (uint32_t)sel[s];
            const oracle_mxfp4_block *down_row = (const oracle_mxfp4_block *)
                    (m->model + m->down_offset + (uint64_t)expert * m->down_expert_bytes +
                     (uint64_t)row * m->down_row_bytes);
            acc += oracle_mxfp4_dot_q8k(down_row, midq[s], MX_MID_DIM);
        }
        out[row] = acc;
    }
}

/* Covers n_tokens in {1,2,3,4}, the decode/tiny-batch regime.  Checks
 * gate/up-mid against the full oracle, then feeds the SYCL kernel's own
 * mid into the down oracle half and checks out against that: a
 * stage-isolated check, so a down bug is distinguishable from a gate/up
 * bug rather than compounding into one end-to-end mismatch. */
static int test_mxfp4_tiny_batch(const mx_model *t, uint32_t n_tokens) {
    int32_t sel_pat[MX_N_PATTERN][MX_N_EXPERT];
    float w_pat[MX_N_PATTERN][MX_N_EXPERT];
    float x_pat[MX_N_PATTERN][MX_IN_DIM];
    for (uint32_t p = 0; p < MX_N_PATTERN; p++) mx_build_pattern(p, sel_pat[p], w_pat[p], x_pat[p]);

    int32_t *sel_all = malloc((size_t)n_tokens * MX_N_EXPERT * sizeof(int32_t));
    float *w_all = malloc((size_t)n_tokens * MX_N_EXPERT * sizeof(float));
    float *x_all = malloc((size_t)n_tokens * MX_IN_DIM * sizeof(float));
    float *mid_oracle = malloc((size_t)n_tokens * MX_N_EXPERT * MX_MID_DIM * sizeof(float));
    float *mid_gpu = malloc((size_t)n_tokens * MX_N_EXPERT * MX_MID_DIM * sizeof(float));
    float *out_oracle = malloc((size_t)n_tokens * MX_OUT_DIM * sizeof(float));
    float *out_gpu = malloc((size_t)n_tokens * MX_OUT_DIM * sizeof(float));
    int rc = 1;

    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t p = tok % MX_N_PATTERN;
        memcpy(sel_all + (size_t)tok * MX_N_EXPERT, sel_pat[p], sizeof(sel_pat[p]));
        memcpy(w_all + (size_t)tok * MX_N_EXPERT, w_pat[p], sizeof(w_pat[p]));
        memcpy(x_all + (size_t)tok * MX_IN_DIM, x_pat[p], sizeof(x_pat[p]));
        oracle_mxfp4_gate_up_mid(t, sel_pat[p], w_pat[p], x_pat[p], MX_CLAMP,
                                 mid_oracle + (size_t)tok * MX_N_EXPERT * MX_MID_DIM);
    }

    if (ds4_sycl_moe_test_mxfp4_gate_up_mid_decode(
            t->model + t->gate_offset, t->model + t->up_offset, t->gate_expert_bytes,
            t->gate_row_bytes, MX_N_TOTAL_EXPERT, x_all, MX_IN_DIM, n_tokens, sel_all, w_all,
            MX_N_EXPERT, MX_MID_DIM, MX_CLAMP, mid_gpu) == 0) {
        fprintf(stderr, "FAIL: mxfp4_tiny_batch(%u): gate_up_mid_decode call failed\n", n_tokens);
        rc = 1;
        goto out;
    }
    {
        uint64_t nmis = 0, first_bad = (uint64_t)-1;
        for (uint64_t i = 0; i < (uint64_t)n_tokens * MX_N_EXPERT * MX_MID_DIM; i++) {
            const float tol = fabsf(mid_oracle[i]) * 1.0e-4f + 1.0e-4f;
            if (fabsf(mid_gpu[i] - mid_oracle[i]) > tol) {
                if (first_bad == (uint64_t)-1) first_bad = i;
                nmis++;
            }
        }
        if (nmis) {
            const uint64_t i = first_bad;
            const uint32_t dbg_pair = (uint32_t)(i / MX_MID_DIM);
            const uint32_t dbg_row = (uint32_t)(i % MX_MID_DIM);
            fprintf(stderr,
                    "FAIL: mxfp4_tiny_batch(%u): %llu/%llu mid mismatches, first mid[%llu] "
                    "(pair=%u row=%u expert=%d weight=%.6f) got %.9g want %.9g\n",
                    n_tokens, (unsigned long long)nmis,
                    (unsigned long long)((uint64_t)n_tokens * MX_N_EXPERT * MX_MID_DIM),
                    (unsigned long long)i, dbg_pair, dbg_row, sel_all[dbg_pair],
                    (double)w_all[dbg_pair], (double)mid_gpu[i], (double)mid_oracle[i]);
            rc = 1;
            goto out;
        }
    }

    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t p = tok % MX_N_PATTERN;
        oracle_mxfp4_down_from_mid(t, sel_pat[p], mid_gpu + (size_t)tok * MX_N_EXPERT * MX_MID_DIM,
                                   out_oracle + (size_t)tok * MX_OUT_DIM);
    }
    if (ds4_sycl_moe_test_mxfp4_down_sum6(t->model + t->down_offset, t->down_expert_bytes,
                                          t->down_row_bytes, MX_N_TOTAL_EXPERT, mid_gpu,
                                          MX_MID_DIM, n_tokens, sel_all, MX_N_EXPERT, MX_OUT_DIM,
                                          out_gpu) == 0) {
        fprintf(stderr, "FAIL: mxfp4_tiny_batch(%u): down_sum6 call failed\n", n_tokens);
        rc = 1;
        goto out;
    }
    for (uint64_t i = 0; i < (uint64_t)n_tokens * MX_OUT_DIM; i++) {
        const float tol = fabsf(out_oracle[i]) * 2.0e-4f + 2.0e-4f;
        if (fabsf(out_gpu[i] - out_oracle[i]) > tol) {
            fprintf(stderr,
                    "FAIL: mxfp4_tiny_batch(%u): out[%llu] got %.9g want %.9g (from GPU's own mid)\n",
                    n_tokens, (unsigned long long)i, (double)out_gpu[i], (double)out_oracle[i]);
            rc = 1;
            goto out;
        }
    }

    fprintf(stderr, "  test_mxfp4_tiny_batch(n_tokens=%u) OK\n", n_tokens);
    rc = 0;
out:
    free(sel_all);
    free(w_all);
    free(x_all);
    free(mid_oracle);
    free(mid_gpu);
    free(out_oracle);
    free(out_gpu);
    return rc;
}

/* Runs one routed_moe_batch_tensor call with the four-pattern-cycled
 * routing already established above and returns the `out` tensor read
 * back into a caller-supplied buffer (n_tokens*MX_OUT_DIM floats).
 * Shared by test_mxfp4_batch's own oracle check and
 * test_mxfp4_down_rgroup_byte_exact's A/B comparison, which needs the
 * identical routing run twice under different environment settings. */
static int mx_run_batch_out(const mx_model *m, uint32_t n_tokens, float *out_buf) {
    int32_t sel_pat[MX_N_PATTERN][MX_N_EXPERT];
    float w_pat[MX_N_PATTERN][MX_N_EXPERT];
    float x_pat[MX_N_PATTERN][MX_IN_DIM];
    for (uint32_t p = 0; p < MX_N_PATTERN; p++) mx_build_pattern(p, sel_pat[p], w_pat[p], x_pat[p]);

    const uint64_t route_count = (uint64_t)n_tokens * MX_N_EXPERT;
    int32_t *sel_all = malloc((size_t)(route_count * sizeof(int32_t)));
    float *w_all = malloc((size_t)(route_count * sizeof(float)));
    float *x_all = malloc((size_t)((uint64_t)n_tokens * MX_IN_DIM * sizeof(float)));
    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t p = tok % MX_N_PATTERN;
        memcpy(sel_all + (size_t)tok * MX_N_EXPERT, sel_pat[p], sizeof(sel_pat[p]));
        memcpy(w_all + (size_t)tok * MX_N_EXPERT, w_pat[p], sizeof(w_pat[p]));
        memcpy(x_all + (size_t)tok * MX_IN_DIM, x_pat[p], sizeof(x_pat[p]));
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MX_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(route_count * MX_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(route_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MX_IN_DIM * sizeof(float));
    int ok = sel_all && w_all && x_all && out && gate && up && mid && down && selected &&
             weights && x;

    ok = ok && ds4_gpu_tensor_write(selected, 0, sel_all, route_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(weights, 0, w_all, route_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(x, 0, x_all, (uint64_t)n_tokens * MX_IN_DIM * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
                       out, gate, up, mid, down, m->model, m->model_size, m->gate_offset,
                       m->up_offset, m->down_offset, 39u, 39u, m->gate_expert_bytes,
                       m->gate_row_bytes, m->down_expert_bytes, m->down_row_bytes, MX_IN_DIM,
                       MX_MID_DIM, MX_OUT_DIM, selected, weights, MX_N_TOTAL_EXPERT, MX_N_EXPERT,
                       MX_CLAMP, x, 0u, n_tokens, NULL, false) != 0;
    ok = ok && ds4_gpu_tensor_read(out, 0, out_buf,
                                   (uint64_t)n_tokens * MX_OUT_DIM * sizeof(float)) != 0;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(sel_all);
    free(w_all);
    free(x_all);
    return ok;
}

/* DS4_ROCM_MXFP4_DOWN_RGROUP parameterises the canonical down tile8
 * kernel's row grouping; it is not a distinct kernel body, so per the
 * plan it is checked byte-exact against the default (row_groups=1)
 * rather than needing its own oracle.  Token counts and the shared
 * four-pattern routing already exercise uneven occupancy, an empty
 * expert (most of the 32 experts are never selected by any pattern) and
 * a tail tile (pair counts per expert are not multiples of 8) without
 * further setup. */
static int test_mxfp4_down_rgroup_byte_exact(const mx_model *m) {
    static const uint32_t token_cases[] = {5u, 32u, 128u, 512u};
    for (size_t i = 0; i < sizeof(token_cases) / sizeof(token_cases[0]); i++) {
        const uint32_t n_tokens = token_cases[i];
        float *out_default = malloc((size_t)((uint64_t)n_tokens * MX_OUT_DIM * sizeof(float)));
        float *out_rgroup4 = malloc((size_t)((uint64_t)n_tokens * MX_OUT_DIM * sizeof(float)));
        int ok = out_default && out_rgroup4;

        unsetenv("DS4_ROCM_MXFP4_DOWN_RGROUP");
        ok = ok && mx_run_batch_out(m, n_tokens, out_default);
        setenv("DS4_ROCM_MXFP4_DOWN_RGROUP", "4", 1);
        ok = ok && mx_run_batch_out(m, n_tokens, out_rgroup4);
        unsetenv("DS4_ROCM_MXFP4_DOWN_RGROUP");

        if (!ok) {
            fprintf(stderr, "FAIL: mxfp4_down_rgroup_byte_exact(%u): a batch call failed\n",
                    n_tokens);
            free(out_default);
            free(out_rgroup4);
            return 1;
        }
        if (memcmp(out_default, out_rgroup4, (size_t)((uint64_t)n_tokens * MX_OUT_DIM *
                                                       sizeof(float))) != 0) {
            fprintf(stderr,
                    "FAIL: mxfp4_down_rgroup_byte_exact(%u): DOWN_RGROUP=4 not bitwise "
                    "identical to the default path\n",
                    n_tokens);
            free(out_default);
            free(out_rgroup4);
            return 1;
        }
        free(out_default);
        free(out_rgroup4);
    }
    fprintf(stderr, "  test_mxfp4_down_rgroup_byte_exact OK\n");
    return 0;
}

/* Covers every n_tokens through the real public entries
 * (ds4_gpu_routed_moe_batch_tensor), straddling the launcher's own
 * mxfp4_path threshold (n_tokens >= 5 takes the sorted-pairs tile8 path;
 * n_tokens <= 4 takes the same decode/tiny-batch kernels
 * test_mxfp4_tiny_batch already checked through the component-level side
 * doors, this time end to end through the dispatcher).  Four routing
 * patterns cycle by token index (tests/test_mxfp4_rocm.c's own shape),
 * so the oracle cost stays flat regardless of n_tokens: mid is checked
 * against the full oracle per pattern, and out is checked against the
 * down oracle fed the GPU's own mid for that pattern, re-derived only
 * when a pattern's GPU mid output changes (it should not, since the
 * computation is deterministic; a memcmp confirms rather than assumes
 * this). */
static int test_mxfp4_batch(const mx_model *m, uint32_t n_tokens) {
    int32_t sel_pat[MX_N_PATTERN][MX_N_EXPERT];
    float w_pat[MX_N_PATTERN][MX_N_EXPERT];
    float x_pat[MX_N_PATTERN][MX_IN_DIM];
    float mid_oracle_pat[MX_N_PATTERN][MX_N_EXPERT * MX_MID_DIM];
    for (uint32_t p = 0; p < MX_N_PATTERN; p++) {
        mx_build_pattern(p, sel_pat[p], w_pat[p], x_pat[p]);
        oracle_mxfp4_gate_up_mid(m, sel_pat[p], w_pat[p], x_pat[p], MX_CLAMP, mid_oracle_pat[p]);
    }

    const uint64_t route_count = (uint64_t)n_tokens * MX_N_EXPERT;
    int32_t *sel_all = malloc((size_t)(route_count * sizeof(int32_t)));
    float *w_all = malloc((size_t)(route_count * sizeof(float)));
    float *x_all = malloc((size_t)((uint64_t)n_tokens * MX_IN_DIM * sizeof(float)));
    float *mid_gpu = malloc((size_t)(route_count * MX_MID_DIM * sizeof(float)));
    float *out_gpu = malloc((size_t)((uint64_t)n_tokens * MX_OUT_DIM * sizeof(float)));
    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t p = tok % MX_N_PATTERN;
        memcpy(sel_all + (size_t)tok * MX_N_EXPERT, sel_pat[p], sizeof(sel_pat[p]));
        memcpy(w_all + (size_t)tok * MX_N_EXPERT, w_pat[p], sizeof(w_pat[p]));
        memcpy(x_all + (size_t)tok * MX_IN_DIM, x_pat[p], sizeof(x_pat[p]));
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MX_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(route_count * MX_MID_DIM * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(route_count * MX_OUT_DIM * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(route_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MX_IN_DIM * sizeof(float));
    int ok = sel_all && w_all && x_all && mid_gpu && out_gpu && out && gate && up && mid &&
             down && selected && weights && x;
    int rc = 1;

    ok = ok && ds4_gpu_tensor_write(selected, 0, sel_all, route_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(weights, 0, w_all, route_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(x, 0, x_all, (uint64_t)n_tokens * MX_IN_DIM * sizeof(float));

    bool mid_is_f16 = true;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
                       out, gate, up, mid, down, m->model, m->model_size, m->gate_offset,
                       m->up_offset, m->down_offset, 39u, 39u, m->gate_expert_bytes,
                       m->gate_row_bytes, m->down_expert_bytes, m->down_row_bytes, MX_IN_DIM,
                       MX_MID_DIM, MX_OUT_DIM, selected, weights, MX_N_TOTAL_EXPERT, MX_N_EXPERT,
                       MX_CLAMP, x, 0u, n_tokens, &mid_is_f16, false) != 0;
    if (!ok) {
        fprintf(stderr, "FAIL: mxfp4_batch(%u): routed_moe_batch_tensor call failed\n", n_tokens);
        goto out;
    }
    if (mid_is_f16) {
        fprintf(stderr, "FAIL: mxfp4_batch(%u): mid_is_f16 unexpectedly true\n", n_tokens);
        goto out;
    }
    ok = ds4_gpu_tensor_read(mid, 0, mid_gpu, route_count * MX_MID_DIM * sizeof(float)) != 0 &&
         ds4_gpu_tensor_read(out, 0, out_gpu, (uint64_t)n_tokens * MX_OUT_DIM * sizeof(float)) != 0;
    if (!ok) {
        fprintf(stderr, "FAIL: mxfp4_batch(%u): tensor readback failed\n", n_tokens);
        goto out;
    }

    for (uint32_t tok = 0; tok < n_tokens; tok++) {
        const uint32_t p = tok % MX_N_PATTERN;
        const float *got = mid_gpu + (uint64_t)tok * MX_N_EXPERT * MX_MID_DIM;
        const float *want = mid_oracle_pat[p];
        for (uint32_t i = 0; i < MX_N_EXPERT * MX_MID_DIM; i++) {
            const float tol = fabsf(want[i]) * 1.0e-4f + 1.0e-4f;
            if (fabsf(got[i] - want[i]) > tol) {
                fprintf(stderr,
                        "FAIL: mxfp4_batch(%u): mid mismatch at tok=%u i=%u got %.9g want %.9g\n",
                        n_tokens, tok, i, (double)got[i], (double)want[i]);
                goto out;
            }
        }
    }

    {
        float down_oracle_cache[MX_N_PATTERN][MX_OUT_DIM];
        int32_t cached_tok[MX_N_PATTERN] = {-1, -1, -1, -1};
        for (uint32_t tok = 0; tok < n_tokens; tok++) {
            const uint32_t p = tok % MX_N_PATTERN;
            const float *mid_tok = mid_gpu + (uint64_t)tok * MX_N_EXPERT * MX_MID_DIM;
            const uint64_t mid_bytes = (uint64_t)MX_N_EXPERT * MX_MID_DIM * sizeof(float);
            if (cached_tok[p] < 0 ||
                memcmp(mid_gpu + (uint64_t)cached_tok[p] * MX_N_EXPERT * MX_MID_DIM, mid_tok,
                       (size_t)mid_bytes) != 0) {
                oracle_mxfp4_down_from_mid(m, sel_pat[p], mid_tok, down_oracle_cache[p]);
                cached_tok[p] = (int32_t)tok;
            }
            const float *got = out_gpu + (uint64_t)tok * MX_OUT_DIM;
            const float *want = down_oracle_cache[p];
            for (uint32_t i = 0; i < MX_OUT_DIM; i++) {
                const float tol = fabsf(want[i]) * 2.0e-4f + 2.0e-4f;
                if (fabsf(got[i] - want[i]) > tol) {
                    fprintf(stderr,
                            "FAIL: mxfp4_batch(%u): out mismatch at tok=%u i=%u got %.9g "
                            "want %.9g (from GPU's own mid)\n",
                            n_tokens, tok, i, (double)got[i], (double)want[i]);
                    goto out;
                }
            }
        }
    }

    fprintf(stderr, "  test_mxfp4_batch(n_tokens=%u) OK\n", n_tokens);
    rc = 0;
out:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    free(sel_all);
    free(w_all);
    free(x_all);
    free(mid_gpu);
    free(out_gpu);
    return rc;
}

/* Mirror of test_q2k_down_slot_order above, for sycl_moe_iq2_down_direct
 * (down_type == 16, iq2_iq2_path's down projection): the identical gap a
 * review found -- every ABI-level test that exercises this function
 * (test_iq2iq2_decode, test_iq2iq2_batch) uses RM_N_EXPERT == 2, which is
 * exactly commutative under float addition and so can never discriminate
 * slot-order summation from any other order, the same blind spot
 * test_q2k_down_slot_order exists to close for its sibling function.
 *
 * oracle_iq2_dot_block's output is exactly linear in the weight block's d
 * (0.125f * xd * y->d * bsum, with no dmin term the way Q2_K's dot has),
 * so the same three-slot construction works with no extra zeroing: every
 * expert's down block shares one grid/sign/ls-nibble template (grid index
 * 0, no sign flips, the maximum ls nibble, so every term is the same sign
 * and magnitude rather than partially cancelling), and only d differs
 * across the three slots that matter -- +60000, its exact bit-negation,
 * and 0.001, the same ratio test_q2k_down_slot_order uses to guarantee
 * the two extremes round away completely under any order that does not
 * place them adjacently. */
static int test_iq2_down_slot_order(void) {
    enum { N_EXPERT = 3, N_TOTAL_EXPERT = 6 };
    uint16_t qs[32];
    for (uint32_t g = 0; g < 8u; g++) {
        oracle_iq2_pack_ib32(&qs[g * 4u], 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 15u);
    }

    const uint16_t d_pos_bits = f32_to_f16_bits(60000.0f);
    const uint16_t d_neg_bits = (uint16_t)(d_pos_bits ^ 0x8000u); /* exact negation */
    const uint16_t d_small_bits = f32_to_f16_bits(0.001f);

    uint8_t down[N_TOTAL_EXPERT][RM_IQ2_BLOCK_BYTES];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) oracle_iq2_pack_block(down[e], 1.0f, qs);
    memcpy(down[0], &d_pos_bits, 2);   /* expert 0: +big */
    memcpy(down[5], &d_neg_bits, 2);   /* expert 5: -big, exact negation */
    memcpy(down[2], &d_small_bits, 2); /* expert 2: small */

    float x[256];
    for (int i = 0; i < 256; i++) x[i] = 10.0f; /* constant: no cross-element cancellation */
    oracle_q8k_block xq;
    oracle_q8k_quantize_block(x, &xq);

    uint8_t midq_bytes[3 * Q8_K_BLOCK_BYTES];
    for (int s = 0; s < N_EXPERT; s++) {
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES, &xq.d, 4);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 4, xq.qs, 256);
        memcpy(midq_bytes + s * Q8_K_BLOCK_BYTES + 260, xq.bsums, 32);
    }

    uint8_t down_flat[N_TOTAL_EXPERT * RM_IQ2_BLOCK_BYTES];
    for (int e = 0; e < N_TOTAL_EXPERT; e++) {
        memcpy(down_flat + e * RM_IQ2_BLOCK_BYTES, down[e], RM_IQ2_BLOCK_BYTES);
    }

    /* Slot order: expert 0 (+big), expert 5 (-big), expert 2 (small). */
    int32_t sel[N_EXPERT] = {0, 5, 2};
    float got = 0.0f;
    CHECK(ds4_sycl_moe_test_iq2_down_direct(down_flat, midq_bytes, sel,
                                            /*down_expert_bytes=*/RM_IQ2_BLOCK_BYTES,
                                            /*down_row_bytes=*/RM_IQ2_BLOCK_BYTES, /*midq_blocks=*/1u,
                                            /*out_dim=*/1u, N_EXPERT, /*n_tokens=*/1u,
                                            N_TOTAL_EXPERT, &got) != 0,
          "iq2_down_slot_order: call failed");

    float want = oracle_iq2_dot_block(down[2], &xq);
    CHECK(fabsf(want) > 1.0f, "iq2_down_slot_order: test data too weak (small term near zero)");
    CHECK_CLOSE(got, want, fabs(want) * 1e-3 + 1e-4,
               "iq2_down_slot_order: slot-order sum must equal the small term exactly, "
               "not be swallowed by the two cancelling extremes");
    fprintf(stderr, "  test_iq2_down_slot_order OK (got %.6g, want %.6g)\n", (double)got,
            (double)want);
    return 0;
}

int main(void) {
    /* MXFP4 format-oracle tests need no GPU; run them before ds4_gpu_init
     * so a failure here is never confused with a device-side problem. */
    if (test_mxfp4_block_layout() != 0) return 1;
    if (test_mxfp4_e8m0_boundary() != 0) return 1;
    if (test_mxfp4_e255_pinned() != 0) return 1;
    if (test_mxfp4_e2m1_codes() != 0) return 1;
    if (test_mxfp4_exponent_scaling() != 0) return 1;
    if (test_mxfp4_monotonicity() != 0) return 1;
    if (test_mxfp4_nibble_order() != 0) return 1;
    if (test_mxfp4_random_dot() != 0) return 1;

    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_q8_k_quantize() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q8_k_quantize_rejects_partial_row() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_subgroup_sum() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sum_slot_order() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sort_basic() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_sort_tile_sizes() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_zero_tokens_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_unrecognised_format_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_iq2_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_q2k_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mxfp4_now_implemented() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_add_in_rejected() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_mid_is_f16_written_false() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_dispatcher_build_plan_validation() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_scale_pack_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch(8u) != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch(40u) != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_scratch_precondition_failure() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode_stages_only_selected() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_owned_decode_composes_ownership_with_compaction() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch_wide_union_falls_back() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_batch_compaction_with_sorted_pairs() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q4k_decode_identity_remap_ablation() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_pack_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_batch_small() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_batch_large() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_batch_stress() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_batch() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_iq2iq2_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_decode() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_batch_small() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_batch_large() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_decode_matches_batch_of_one() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_q2k_down_slot_order() != 0) { ds4_gpu_cleanup(); return 1; }

    {
        mx_model mxm = mx_build_model();
        int mx_ok = 1;
        mx_ok = mx_ok && test_mxfp4_tiny_batch(&mxm, 1u) == 0;
        mx_ok = mx_ok && test_mxfp4_tiny_batch(&mxm, 2u) == 0;
        mx_ok = mx_ok && test_mxfp4_tiny_batch(&mxm, 3u) == 0;
        mx_ok = mx_ok && test_mxfp4_tiny_batch(&mxm, 4u) == 0;
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 1u) == 0;
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 5u) == 0;
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 32u) == 0;
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 128u) == 0;
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 512u) == 0;
        /* Permanent stress regression (spec 6j): a dropped tile8
         * staging-barrier ablation did not fail through n_tokens=4096
         * (thousands of tiles across 32 experts), only at 16384, matching
         * the earlier finding that an equivalent race needed a launch
         * two orders of magnitude bigger than the functional test suite
         * before real work-group contention exposed it.  Runs in under
         * two seconds end to end, cheap enough to keep permanently. */
        mx_ok = mx_ok && test_mxfp4_batch(&mxm, 16384u) == 0;
        mx_ok = mx_ok && test_mxfp4_down_rgroup_byte_exact(&mxm) == 0;
        mx_free_model(&mxm);
        if (!mx_ok) { ds4_gpu_cleanup(); return 1; }
    }

    if (test_iq2_down_slot_order() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_moe OK\n");
    return 0;
}
