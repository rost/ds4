/* Correctness tests for the SYCL shared expert: a Q8_0 SwiGLU MLP run for
 * every token on every layer.  Oracles reimplement the documented formula
 * with rocm/ds4_rocm_shared_expert.cuh and rocm/ds4_rocm_q8.cuh line
 * numbers cited (ROCm, not ds4.c, is the port source; ds4.c is cited only
 * for which computation is the right oracle, see below).  Needs no model
 * file.
 *
 * Oracle choice: ds4.c's layer_shared_ffn_one (ds4.c:10569-10598)
 * quantises the activation to Q8_0 before the matvec.  Every GPU kernel
 * this file exercises instead reads the F32 activation directly with no
 * quantisation step (rocm/ds4_rocm_q8.cuh:948-1046 and :524-671 both take
 * `const float *x`).  So the right oracle here is the UNQUANTISED
 * variant, matching ds4.c's layer_glm_shared_ffn_one_f32_ref
 * (ds4.c:14600-14620: matvec_q8_0_f32_ref against a raw f32 x), not the
 * quantised layer_shared_ffn_one.  Reimplemented here as
 * oracle_shared_gate_up_swiglu rather than linked, since ds4.c's
 * functions are static. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <stdio.h>
#include <string.h>

/* Half-precision encode, plain truncation (not round-to-nearest); safe
 * here because the decode side (sycl_q8_0_dequant, via the kernel under
 * test) reads back the identical stored bit pattern, so both sides agree
 * regardless of which rounding produced those bits.  Mirrors
 * tests/test_sycl_matmul.c's test_float_to_half. */
static uint16_t test_float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Q8_0 row layout, rocm/ds4_rocm_common.cuh:19-63: blocks of 32 values, 34
 * bytes per block, a little-endian F16 scale in bytes 0-1 followed by 32
 * signed int8 values.  `salt` lets gate and up weights differ so a kernel
 * that aliased the two tables would produce a visibly wrong result. The
 * int8 payload carries an (o, k) product interaction term, not a plain
 * a*i + b, for the same reason. */
static void test_encode_q8_0_row(unsigned char *row, uint32_t in_dim,
                                 uint32_t o, uint32_t salt) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t braw =
                test_float_to_half(0.05f * (float)(o + blk + salt + 1u));
        bp[0] = (unsigned char)(braw & 0xFFu);
        bp[1] = (unsigned char)((braw >> 8) & 0xFFu);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const int qv = (k < in_dim)
                    ? (int)(((o + salt + 1u) * (k + 3u)) % 13u) - 6
                    : 0;
            bp[2 + idx] = (unsigned char)(signed char)qv;
        }
    }
}

static float oracle_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 32u;
    const uint32_t idx = col % 32u;
    const unsigned char *bp = row + (size_t)blk * 34u;
    const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
    const float scale = oracle_half_to_float(raw);
    const signed char qv = (signed char)bp[2 + idx];
    return scale * (float)qv;
}

static float oracle_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* One token's gate = W_gate . x, up = W_up . x, mid = silu(gate) * up,
 * with clamp applied ASYMMETRICALLY before the SwiGLU: gate is clamped
 * above only, up is clamped both above and below.  Ported from
 * rocm/ds4_rocm_q8.cuh:948-1046 (shared_gate_up_swiglu_q8_0_rows_w32_kernel)
 * and its clamp block, and cross-checked against swiglu(), ds4.c:10556-10567,
 * which carries the identical asymmetry. */
static void oracle_shared_gate_up_swiglu(float *gate, float *up, float *mid,
                                         const float *x,
                                         const unsigned char *wg,
                                         const unsigned char *wu,
                                         uint32_t in_dim, uint32_t out_dim,
                                         uint64_t row_bytes, float clamp) {
    for (uint32_t o = 0; o < out_dim; o++) {
        const unsigned char *rg = wg + (size_t)o * row_bytes;
        const unsigned char *ru = wu + (size_t)o * row_bytes;
        double sg = 0.0, su = 0.0;
        for (uint32_t k = 0; k < in_dim; k++) {
            sg += (double)x[k] * (double)oracle_dequant(rg, k);
            su += (double)x[k] * (double)oracle_dequant(ru, k);
        }
        float g = (float)sg;
        float u = (float)su;
        gate[o] = g;
        up[o] = u;
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        mid[o] = oracle_silu(g) * u;
    }
}

/* in_dim deliberately not 4096 and not a multiple of 32, so remainder
 * handling in the general path is exercised (the fused fast path added in
 * a later task requires in_dim == 4096, so this shape always takes the
 * general path).  Gate and up weights differ (salt), so aliasing them
 * would be visible. */
static int test_shared_gate_up_swiglu_general(void) {
    enum { IN_DIM = 37, OUT_DIM = 9 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const float clamp = 1e9f; /* effectively no clamp for this test */

    unsigned char wg[OUT_DIM * 68]; /* row_bytes <= 2*34 = 68 here */
    unsigned char wu[OUT_DIM * 68];
    float x[IN_DIM];
    float want_gate[OUT_DIM], want_up[OUT_DIM], want_mid[OUT_DIM];
    float got_gate[OUT_DIM], got_up[OUT_DIM], got_mid[OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 0);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 100);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        x[k] = (float)(((k + 1) * 7) % 11) - 5.0f + 0.1f * (float)(k % 3);
    }
    oracle_shared_gate_up_swiglu(want_gate, want_up, want_mid, x, wg, wu,
                                 IN_DIM, OUT_DIM, row_bytes, clamp);

    unsigned char combined[sizeof(wg) + sizeof(wu)];
    memcpy(combined, wg, sizeof(wg));
    memcpy(combined + sizeof(wg), wu, sizeof(wu));
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = sizeof(wg);
    const uint64_t model_size = sizeof(combined);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tgate = ds4_gpu_tensor_alloc(sizeof(got_gate));
    ds4_gpu_tensor *tup = ds4_gpu_tensor_alloc(sizeof(got_up));
    ds4_gpu_tensor *tmid = ds4_gpu_tensor_alloc(sizeof(got_mid));
    CHECK(tx && tgate && tup && tmid,
          "shared_gate_up_swiglu_general: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_general: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, clamp) != 0,
          "shared_gate_up_swiglu_general: call");
    CHECK(ds4_gpu_tensor_read(tgate, 0, got_gate, sizeof(got_gate)) != 0,
          "shared_gate_up_swiglu_general: read gate");
    CHECK(ds4_gpu_tensor_read(tup, 0, got_up, sizeof(got_up)) != 0,
          "shared_gate_up_swiglu_general: read up");
    CHECK(ds4_gpu_tensor_read(tmid, 0, got_mid, sizeof(got_mid)) != 0,
          "shared_gate_up_swiglu_general: read mid");

    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(got_gate[i], want_gate[i], 1e-2,
                    "shared_gate_up_swiglu_general: gate mismatch");
        CHECK_CLOSE(got_up[i], want_up[i], 1e-2,
                    "shared_gate_up_swiglu_general: up mismatch");
        CHECK_CLOSE(got_mid[i], want_mid[i], 1e-2,
                    "shared_gate_up_swiglu_general: mid mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    fprintf(stderr, "  test_shared_gate_up_swiglu_general OK\n");
    return 0;
}

/* Exercises the clamp asymmetry: gate clamped above only, up clamped both
 * ways.  Weights and x are chosen so at least one row's gate exceeds
 * +clamp and at least one row's up goes below -clamp.  A naive symmetric
 * clamp on gate (clamping negative gate values to -clamp too) would change
 * mid for the row whose gate is very negative; this test's tolerance is
 * tight enough to catch that. */
static int test_shared_gate_up_swiglu_clamp_asymmetry(void) {
    enum { IN_DIM = 32, OUT_DIM = 4 };
    const uint64_t row_bytes = 34; /* one block, IN_DIM == 32 */
    const float clamp = 2.0f;

    unsigned char wg[OUT_DIM * 34];
    unsigned char wu[OUT_DIM * 34];
    float x[IN_DIM];
    float want_gate[OUT_DIM], want_up[OUT_DIM], want_mid[OUT_DIM];
    float got_gate[OUT_DIM], got_up[OUT_DIM], got_mid[OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 0);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 50);
    }
    /* Large-magnitude x so |gate| and |up| land well past +-clamp for
     * every row: this specific weight/x combination drives gate strongly
     * negative for at least one row (so the asymmetry, not just the
     * clamp's existence, is exercised) and up both above and below its
     * band. */
    for (uint32_t k = 0; k < IN_DIM; k++) {
        x[k] = (float)((k % 2 == 0) ? 30 : -30);
    }
    oracle_shared_gate_up_swiglu(want_gate, want_up, want_mid, x, wg, wu,
                                 IN_DIM, OUT_DIM, row_bytes, clamp);

    /* Sanity: the oracle itself must actually exercise both the
     * "gate very negative, left unclamped" and the "up clamped on both
     * sides" cases, or this test would not discriminate the asymmetry. */
    int saw_negative_gate_unclamped = 0;
    int saw_up_clamped_low = 0;
    for (int i = 0; i < OUT_DIM; i++) {
        if (want_gate[i] < -clamp) saw_negative_gate_unclamped = 1;
        if (want_up[i] < -clamp) saw_up_clamped_low = 1;
    }
    CHECK(saw_negative_gate_unclamped,
          "shared_gate_up_swiglu_clamp_asymmetry: test data does not drive "
          "gate below -clamp");
    CHECK(saw_up_clamped_low,
          "shared_gate_up_swiglu_clamp_asymmetry: test data does not drive "
          "up below -clamp");

    unsigned char combined[sizeof(wg) + sizeof(wu)];
    memcpy(combined, wg, sizeof(wg));
    memcpy(combined + sizeof(wg), wu, sizeof(wu));
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = sizeof(wg);
    const uint64_t model_size = sizeof(combined);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tgate = ds4_gpu_tensor_alloc(sizeof(got_gate));
    ds4_gpu_tensor *tup = ds4_gpu_tensor_alloc(sizeof(got_up));
    ds4_gpu_tensor *tmid = ds4_gpu_tensor_alloc(sizeof(got_mid));
    CHECK(tx && tgate && tup && tmid,
          "shared_gate_up_swiglu_clamp_asymmetry: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_clamp_asymmetry: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, clamp) != 0,
          "shared_gate_up_swiglu_clamp_asymmetry: call");
    CHECK(ds4_gpu_tensor_read(tmid, 0, got_mid, sizeof(got_mid)) != 0,
          "shared_gate_up_swiglu_clamp_asymmetry: read mid");
    CHECK(ds4_gpu_tensor_read(tgate, 0, got_gate, sizeof(got_gate)) != 0,
          "shared_gate_up_swiglu_clamp_asymmetry: read gate");
    CHECK(ds4_gpu_tensor_read(tup, 0, got_up, sizeof(got_up)) != 0,
          "shared_gate_up_swiglu_clamp_asymmetry: read up");

    for (int i = 0; i < OUT_DIM; i++) {
        /* gate/up outputs are the UNCLAMPED raw values (clamp only applies
         * ahead of the SwiGLU multiply), matching
         * rocm/ds4_rocm_q8.cuh:1017-1024, which stores g/u before clamping
         * sg/su into locals for the mid computation. */
        CHECK_CLOSE(got_gate[i], want_gate[i], 1e-2,
                    "shared_gate_up_swiglu_clamp_asymmetry: gate mismatch");
        CHECK_CLOSE(got_up[i], want_up[i], 1e-2,
                    "shared_gate_up_swiglu_clamp_asymmetry: up mismatch");
        CHECK_CLOSE(got_mid[i], want_mid[i], 1e-2,
                    "shared_gate_up_swiglu_clamp_asymmetry: mid mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    fprintf(stderr, "  test_shared_gate_up_swiglu_clamp_asymmetry OK\n");
    return 0;
}

/* Null pointers, zero/oversized dimensions, and undersized tensors must
 * all be rejected (return 0), matching
 * rocm/ds4_rocm_shared_expert.cuh:19-33. */
static int test_shared_gate_up_swiglu_rejections(void) {
    enum { IN_DIM = 16, OUT_DIM = 3 };
    const uint64_t row_bytes = 34;
    unsigned char wg[OUT_DIM * 34];
    unsigned char wu[OUT_DIM * 34];
    float x[IN_DIM];
    float gate[OUT_DIM], up[OUT_DIM], mid[OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 0);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 20);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) x[k] = (float)k;

    unsigned char combined[sizeof(wg) + sizeof(wu)];
    memcpy(combined, wg, sizeof(wg));
    memcpy(combined + sizeof(wg), wu, sizeof(wu));
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = sizeof(wg);
    const uint64_t model_size = sizeof(combined);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *tgate = ds4_gpu_tensor_alloc(sizeof(gate));
    ds4_gpu_tensor *tup = ds4_gpu_tensor_alloc(sizeof(up));
    ds4_gpu_tensor *tmid = ds4_gpu_tensor_alloc(sizeof(mid));
    ds4_gpu_tensor *tsmall = ds4_gpu_tensor_alloc(sizeof(float));
    CHECK(tx && tgate && tup && tmid && tsmall,
          "shared_gate_up_swiglu_rejections: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_rejections: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  NULL, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: null gate must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, NULL, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: null up must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, NULL, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: null mid must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, NULL, model_size, gate_offset, up_offset,
                  IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: null model_map must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, NULL, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: null x must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, 0, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: zero in_dim must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, 0, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: zero out_dim must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, (uint64_t)UINT32_MAX + 1, OUT_DIM, tx,
                  0.0f) == 0,
          "shared_gate_up_swiglu_rejections: oversized in_dim must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, (uint64_t)UINT32_MAX + 1, tx,
                  0.0f) == 0,
          "shared_gate_up_swiglu_rejections: oversized out_dim must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tsmall, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: undersized gate must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tsmall, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: undersized up must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tsmall, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: undersized mid must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tsmall, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: undersized x must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, model_size,
                  up_offset, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: out-of-range gate_offset must "
          "be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  model_size, IN_DIM, OUT_DIM, tx, 0.0f) == 0,
          "shared_gate_up_swiglu_rejections: out-of-range up_offset must "
          "be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    ds4_gpu_tensor_free(tsmall);
    fprintf(stderr, "  test_shared_gate_up_swiglu_rejections OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_shared_gate_up_swiglu_general() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_clamp_asymmetry() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_rejections() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_shared_expert OK\n");
    return 0;
}
