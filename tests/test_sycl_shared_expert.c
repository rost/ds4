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
#include <stdlib.h>
#include <string.h>

/* Test-only diagnostic (sycl/ds4_sycl_shared_expert.hpp), the same
 * hit-counter technique used by ds4_sycl_stream_test_hit_miss: a
 * correctness assertion on gate/up/mid cannot tell "the fused kernel ran"
 * apart from "the general path computed the same dot product", so tests
 * that must confirm which branch actually executed read this counter. */
extern uint64_t ds4_sycl_shared_expert_test_fast_path_hits(void);

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

    /* IN_DIM != 4096, so this must take the general path: the fast-path
     * hit counter must not move.  This is the test that would catch a
     * port that takes the fused kernel unconditionally regardless of
     * shape, since IN_DIM is also not a multiple of 32, so the fused
     * kernel's unguarded `x[b*32+lane]` read would run past the end of
     * the IN_DIM-sized x buffer for the remainder block, corrupting the
     * result (see the fast_path_correctness test's own header comment for
     * why a multiple-of-32-but-not-4096 shape could not catch this). */
    const uint64_t fast_hits_before = ds4_sycl_shared_expert_test_fast_path_hits();
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, clamp) != 0,
          "shared_gate_up_swiglu_general: call");
    CHECK(ds4_sycl_shared_expert_test_fast_path_hits() == fast_hits_before,
          "shared_gate_up_swiglu_general: must not take the fused fast "
          "path for IN_DIM != 4096");
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

/* Fused fast path: in_dim == 4096 triggers
 * shared_gate_up_swiglu_q8_0_rows_w32_kernel (rocm/ds4_rocm_q8.cuh:948-1046)
 * instead of the general path.  out_dim deliberately not a multiple of 32
 * (the fast kernel's row-block size), so the final work-group's partial
 * row-block guard (`if (row >= out_dim) return;`) is exercised: with
 * rows_per_block == 32 and OUT_DIM == 50, the second work-group covers
 * rows 32..63 but only 32..49 are valid.  gate/up/mid are allocated with
 * PADDED_DIM == 64 capacity (one full row-block beyond the last one that
 * matters) and pre-filled with a canary pattern the entry is never told
 * about (out_dim is passed as 50, not 64): dropping the guard would write
 * computed values into rows 50..63 too, which the OUT_DIM-only read in an
 * earlier version of this test could not see, since it only read back the
 * 50 rows it asked for. Reading and checking the full padded range is
 * what makes that ablation observable; a first version of this test
 * allocated exactly OUT_DIM-sized buffers and the guard-drop ablation
 * passed anyway, corrupting memory just past the tensors' ends without
 * moving any value this test inspected.  Heap-allocated: IN_DIM == 4096
 * makes the per-row weight tables and activation too large for a
 * comfortable stack frame. */
static int test_shared_gate_up_swiglu_fast_path_correctness(void) {
    enum { IN_DIM = 4096, OUT_DIM = 50, PADDED_DIM = 64 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const float clamp = 4.0f;
    const float canary = -123456.0f;

    unsigned char *wg = (unsigned char *)malloc((size_t)OUT_DIM * row_bytes);
    unsigned char *wu = (unsigned char *)malloc((size_t)OUT_DIM * row_bytes);
    float *x = (float *)malloc((size_t)IN_DIM * sizeof(float));
    float *want_gate = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *want_up = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *want_mid = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *got_gate = (float *)malloc((size_t)PADDED_DIM * sizeof(float));
    float *got_up = (float *)malloc((size_t)PADDED_DIM * sizeof(float));
    float *got_mid = (float *)malloc((size_t)PADDED_DIM * sizeof(float));
    float *padded_canary = (float *)malloc((size_t)PADDED_DIM * sizeof(float));
    unsigned char *combined =
            (unsigned char *)malloc((size_t)2 * OUT_DIM * row_bytes);
    CHECK(wg && wu && x && want_gate && want_up && want_mid && got_gate &&
          got_up && got_mid && padded_canary && combined,
          "shared_gate_up_swiglu_fast_path: allocation failed");

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 0);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 100);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        x[k] = (float)(((k + 1) * 11) % 17) - 8.0f + 0.01f * (float)(k % 7);
    }
    for (int i = 0; i < PADDED_DIM; i++) padded_canary[i] = canary;
    oracle_shared_gate_up_swiglu(want_gate, want_up, want_mid, x, wg, wu,
                                 IN_DIM, OUT_DIM, row_bytes, clamp);

    memcpy(combined, wg, (size_t)OUT_DIM * row_bytes);
    memcpy(combined + (size_t)OUT_DIM * row_bytes, wu,
           (size_t)OUT_DIM * row_bytes);
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = (uint64_t)OUT_DIM * row_bytes;
    const uint64_t model_size = 2ull * OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((size_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *tgate =
            ds4_gpu_tensor_alloc((size_t)PADDED_DIM * sizeof(float));
    ds4_gpu_tensor *tup =
            ds4_gpu_tensor_alloc((size_t)PADDED_DIM * sizeof(float));
    ds4_gpu_tensor *tmid =
            ds4_gpu_tensor_alloc((size_t)PADDED_DIM * sizeof(float));
    CHECK(tx && tgate && tup && tmid,
          "shared_gate_up_swiglu_fast_path: tensor allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, (size_t)IN_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_path: write x");
    CHECK(ds4_gpu_tensor_write(tgate, 0, padded_canary,
                               (size_t)PADDED_DIM * sizeof(float)) != 0 &&
          ds4_gpu_tensor_write(tup, 0, padded_canary,
                               (size_t)PADDED_DIM * sizeof(float)) != 0 &&
          ds4_gpu_tensor_write(tmid, 0, padded_canary,
                               (size_t)PADDED_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_path: write canary");

    const uint64_t fast_hits_before = ds4_sycl_shared_expert_test_fast_path_hits();
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, clamp) != 0,
          "shared_gate_up_swiglu_fast_path: call");
    CHECK(ds4_sycl_shared_expert_test_fast_path_hits() == fast_hits_before + 1,
          "shared_gate_up_swiglu_fast_path: must take the fused fast path "
          "for IN_DIM == 4096");
    CHECK(ds4_gpu_tensor_read(tgate, 0, got_gate,
                              (size_t)PADDED_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_path: read gate");
    CHECK(ds4_gpu_tensor_read(tup, 0, got_up,
                              (size_t)PADDED_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_path: read up");
    CHECK(ds4_gpu_tensor_read(tmid, 0, got_mid,
                              (size_t)PADDED_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_path: read mid");

    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(got_gate[i], want_gate[i], 1e-1,
                    "shared_gate_up_swiglu_fast_path: gate mismatch");
        CHECK_CLOSE(got_up[i], want_up[i], 1e-1,
                    "shared_gate_up_swiglu_fast_path: up mismatch");
        CHECK_CLOSE(got_mid[i], want_mid[i], 1e-1,
                    "shared_gate_up_swiglu_fast_path: mid mismatch");
    }
    for (int i = OUT_DIM; i < PADDED_DIM; i++) {
        CHECK(got_gate[i] == canary,
              "shared_gate_up_swiglu_fast_path: gate canary past out_dim "
              "was overwritten");
        CHECK(got_up[i] == canary,
              "shared_gate_up_swiglu_fast_path: up canary past out_dim was "
              "overwritten");
        CHECK(got_mid[i] == canary,
              "shared_gate_up_swiglu_fast_path: mid canary past out_dim "
              "was overwritten");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    free(wg);
    free(wu);
    free(x);
    free(padded_canary);
    free(want_gate);
    free(want_up);
    free(want_mid);
    free(got_gate);
    free(got_up);
    free(got_mid);
    free(combined);
    fprintf(stderr, "  test_shared_gate_up_swiglu_fast_path_correctness OK\n");
    return 0;
}

/* Differential test: for the SAME in_dim == 4096 shape, the public entry
 * always takes the fused fast path (there is no way to force it through
 * the general path via the public ABI once the shape qualifies), so the
 * "general path" side of this comparison is built by calling the two
 * primitives the general path itself calls,
 * ds4_gpu_matmul_q8_0_pair_tensor and ds4_gpu_swiglu_tensor, directly.
 * gate, up AND mid must all agree: gate/up equality catches a fused
 * kernel that has drifted from the pair matmul's dequant/accumulate, and
 * mid equality additionally exercises the fused kernel's own inline SwiGLU
 * against the already-validated ds4_gpu_swiglu_tensor. */
static int test_shared_gate_up_swiglu_fast_vs_general_differential(void) {
    enum { IN_DIM = 4096, OUT_DIM = 20 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const float clamp = 2.5f;

    unsigned char *wg = (unsigned char *)malloc((size_t)OUT_DIM * row_bytes);
    unsigned char *wu = (unsigned char *)malloc((size_t)OUT_DIM * row_bytes);
    float *x = (float *)malloc((size_t)IN_DIM * sizeof(float));
    unsigned char *combined =
            (unsigned char *)malloc((size_t)2 * OUT_DIM * row_bytes);
    CHECK(wg && wu && x && combined,
          "shared_gate_up_swiglu_fast_vs_general: allocation failed");

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 3);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 200);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        x[k] = (float)(((k + 2) * 13) % 19) - 9.0f + 0.02f * (float)(k % 5);
    }
    memcpy(combined, wg, (size_t)OUT_DIM * row_bytes);
    memcpy(combined + (size_t)OUT_DIM * row_bytes, wu,
           (size_t)OUT_DIM * row_bytes);
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = (uint64_t)OUT_DIM * row_bytes;
    const uint64_t model_size = 2ull * OUT_DIM * row_bytes;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc((size_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *tgen_gate =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tgen_up =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tgen_mid =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tfused_gate =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tfused_up =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *tfused_mid =
            ds4_gpu_tensor_alloc((size_t)OUT_DIM * sizeof(float));
    CHECK(tx && tgen_gate && tgen_up && tgen_mid && tfused_gate &&
          tfused_up && tfused_mid,
          "shared_gate_up_swiglu_fast_vs_general: tensor allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, (size_t)IN_DIM * sizeof(float)) != 0,
          "shared_gate_up_swiglu_fast_vs_general: write x");

    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(tgen_gate, tgen_up, combined,
                                          model_size, gate_offset, up_offset,
                                          IN_DIM, OUT_DIM, OUT_DIM, tx,
                                          1) != 0,
          "shared_gate_up_swiglu_fast_vs_general: general pair matmul call");
    CHECK(ds4_gpu_swiglu_tensor(tgen_mid, tgen_gate, tgen_up, OUT_DIM, clamp,
                                1.0f) != 0,
          "shared_gate_up_swiglu_fast_vs_general: general swiglu call");

    /* Confirms the "general" side above genuinely bypassed the fused
     * kernel (it calls the primitives directly, not the dispatching
     * entry, so it must not move the counter at all), and that the fused
     * call below is what actually exercises the fast path being compared
     * against. */
    const uint64_t fast_hits_before = ds4_sycl_shared_expert_test_fast_path_hits();
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  tfused_gate, tfused_up, tfused_mid, combined, model_size,
                  gate_offset, up_offset, IN_DIM, OUT_DIM, tx, clamp) != 0,
          "shared_gate_up_swiglu_fast_vs_general: fused call");
    CHECK(ds4_sycl_shared_expert_test_fast_path_hits() == fast_hits_before + 1,
          "shared_gate_up_swiglu_fast_vs_general: fused call must take the "
          "fast path exactly once");

    float gen_g[OUT_DIM], gen_u[OUT_DIM], gen_m[OUT_DIM];
    float fus_g[OUT_DIM], fus_u[OUT_DIM], fus_m[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(tgen_gate, 0, gen_g, sizeof(gen_g)) != 0 &&
          ds4_gpu_tensor_read(tgen_up, 0, gen_u, sizeof(gen_u)) != 0 &&
          ds4_gpu_tensor_read(tgen_mid, 0, gen_m, sizeof(gen_m)) != 0 &&
          ds4_gpu_tensor_read(tfused_gate, 0, fus_g, sizeof(fus_g)) != 0 &&
          ds4_gpu_tensor_read(tfused_up, 0, fus_u, sizeof(fus_u)) != 0 &&
          ds4_gpu_tensor_read(tfused_mid, 0, fus_m, sizeof(fus_m)) != 0,
          "shared_gate_up_swiglu_fast_vs_general: readback failed");

    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(fus_g[i], gen_g[i], 1e-1,
                    "shared_gate_up_swiglu_fast_vs_general: gate mismatch");
        CHECK_CLOSE(fus_u[i], gen_u[i], 1e-1,
                    "shared_gate_up_swiglu_fast_vs_general: up mismatch");
        CHECK_CLOSE(fus_m[i], gen_m[i], 1e-1,
                    "shared_gate_up_swiglu_fast_vs_general: mid mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgen_gate);
    ds4_gpu_tensor_free(tgen_up);
    ds4_gpu_tensor_free(tgen_mid);
    ds4_gpu_tensor_free(tfused_gate);
    ds4_gpu_tensor_free(tfused_up);
    ds4_gpu_tensor_free(tfused_mid);
    free(wg);
    free(wu);
    free(x);
    free(combined);
    fprintf(stderr,
            "  test_shared_gate_up_swiglu_fast_vs_general_differential OK\n");
    return 0;
}

/* ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor.  ROCm's own
 * definition (rocm/ds4_rocm_shared_expert.cuh:69-86) voids every argument
 * and unconditionally returns 0: a real stub, not an oversight, in that
 * reference.  This backend deliberately diverges from that stub: Metal
 * implements this entry for real (ds4_metal.m:19368, kernel
 * "kernel_dsv4_shared_gate_up_swiglu_q8_0"), it is declared in ds4_gpu.h
 * as an ordinary primitive with no "optional" framing, and it has a
 * Flash-reachable call site at ds4.c:62246
 * (metal_graph_encode_native_session_batch_shared, the generic non-GLM
 * batched-session path; the other call site, ds4.c:43369, is
 * glm_graph_-prefixed and out of scope).  ROCm being the structural
 * reference exists to keep semantics aligned, not to reproduce ROCm's own
 * incomplete backend coverage on an entry Metal already implements; the
 * SYCL port of ds4_gpu_matmul_q8_0_rows_scalar_tensor
 * (sycl/ds4_sycl_matmul.hpp) made the identical judgement call for the
 * structurally analogous dense-matmul entry.
 *
 * n_tok == 1 delegates to the core entry, matching Metal's own
 * ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor.  For n_tok > 1,
 * unlike ROCm's fused batch kernel (rocm/ds4_rocm_q8.cuh:524-586,
 * reachable here through ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor),
 * which has no clamp parameter at all and is why _rows_tensor refuses
 * clamp > 1e-6f rather than route through it, Metal's real
 * kernel_dsv4_shared_gate_up_swiglu_q8_0 DOES accept and apply clamp for
 * n_tok > 1.  This SYCL implementation matches Metal's capability rather
 * than ROCm's limitation: it generalises the already-implemented,
 * already-ablated general path
 * (ds4_gpu_matmul_q8_0_pair_tensor + ds4_gpu_swiglu_tensor) over n_tok
 * rather than refusing a nonzero clamp, since both primitives already
 * accept an arbitrary token/element count in one launch and swiglu's
 * clamp is applied elementwise regardless of how the flat buffer is
 * divided into per-token rows.  Test data below deliberately drives clamp
 * so at least one token actually clamps, proving this multi-token clamp
 * support rather than merely not crashing with clamp == 0. */
static int test_shared_gate_up_swiglu_rows_scalar_n_tok_1(void) {
    enum { IN_DIM = 37, OUT_DIM = 9 };
    const uint64_t blocks = (IN_DIM + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const float clamp = 3.0f;

    unsigned char wg[OUT_DIM * 68];
    unsigned char wu[OUT_DIM * 68];
    float x[IN_DIM];
    float want_gate[OUT_DIM], want_up[OUT_DIM], want_mid[OUT_DIM];
    float got_gate[OUT_DIM], got_up[OUT_DIM], got_mid[OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 7);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 77);
    }
    for (uint32_t k = 0; k < IN_DIM; k++) {
        x[k] = (float)(((k + 3) * 5) % 13) - 6.0f;
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
          "shared_gate_up_swiglu_rows_scalar_n_tok_1: allocation failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_rows_scalar_n_tok_1: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 1, clamp) != 0,
          "shared_gate_up_swiglu_rows_scalar_n_tok_1: call");
    CHECK(ds4_gpu_tensor_read(tgate, 0, got_gate, sizeof(got_gate)) != 0 &&
          ds4_gpu_tensor_read(tup, 0, got_up, sizeof(got_up)) != 0 &&
          ds4_gpu_tensor_read(tmid, 0, got_mid, sizeof(got_mid)) != 0,
          "shared_gate_up_swiglu_rows_scalar_n_tok_1: readback failed");

    for (int i = 0; i < OUT_DIM; i++) {
        CHECK_CLOSE(got_gate[i], want_gate[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_n_tok_1: gate mismatch");
        CHECK_CLOSE(got_up[i], want_up[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_n_tok_1: up mismatch");
        CHECK_CLOSE(got_mid[i], want_mid[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_n_tok_1: mid mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    fprintf(stderr, "  test_shared_gate_up_swiglu_rows_scalar_n_tok_1 OK\n");
    return 0;
}

/* n_tok > 1, checked against a PER-TOKEN oracle (every token, not only the
 * first), with a clamp that actually bites for at least one token so this
 * entry's multi-token clamp support (see the block comment above) is
 * genuinely exercised rather than merely tolerated. */
static int test_shared_gate_up_swiglu_rows_scalar_multi_token(void) {
    enum { IN_DIM = 32, OUT_DIM = 6, N_TOK = 4 };
    const uint64_t row_bytes = 34; /* one block, IN_DIM == 32 */
    const float clamp = 1.5f;

    unsigned char wg[OUT_DIM * 34];
    unsigned char wu[OUT_DIM * 34];
    float x[N_TOK * IN_DIM];
    float want_gate[N_TOK * OUT_DIM], want_up[N_TOK * OUT_DIM],
            want_mid[N_TOK * OUT_DIM];
    float got_gate[N_TOK * OUT_DIM], got_up[N_TOK * OUT_DIM],
            got_mid[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 1);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 51);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            /* (t * k) interaction term, per spec 6f: a purely affine
             * a*i + b pattern would make every token's row proportional
             * to every other, hiding a bug that mixes up which token's
             * slice a row belongs to. */
            x[t * IN_DIM + k] =
                    (float)(((t + 1) * (k + 2)) % 11) - 5.0f +
                    0.3f * (float)((t * k) % 7);
        }
        oracle_shared_gate_up_swiglu(want_gate + t * OUT_DIM,
                                     want_up + t * OUT_DIM,
                                     want_mid + t * OUT_DIM, x + t * IN_DIM,
                                     wg, wu, IN_DIM, OUT_DIM, row_bytes,
                                     clamp);
    }

    /* Sanity: confirm the oracle itself actually drives at least one
     * token's clamp, or this test could not tell "clamp applied" from
     * "clamp ignored". */
    int saw_clamped = 0;
    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        if (want_gate[i] > clamp || want_up[i] > clamp || want_up[i] < -clamp) {
            saw_clamped = 1;
        }
    }
    CHECK(saw_clamped,
          "shared_gate_up_swiglu_rows_scalar_multi_token: test data does "
          "not drive clamp for any token");

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
          "shared_gate_up_swiglu_rows_scalar_multi_token: allocation "
          "failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_rows_scalar_multi_token: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, N_TOK, clamp) != 0,
          "shared_gate_up_swiglu_rows_scalar_multi_token: call");
    CHECK(ds4_gpu_tensor_read(tgate, 0, got_gate, sizeof(got_gate)) != 0 &&
          ds4_gpu_tensor_read(tup, 0, got_up, sizeof(got_up)) != 0 &&
          ds4_gpu_tensor_read(tmid, 0, got_mid, sizeof(got_mid)) != 0,
          "shared_gate_up_swiglu_rows_scalar_multi_token: readback failed");

    for (int i = 0; i < N_TOK * OUT_DIM; i++) {
        CHECK_CLOSE(got_gate[i], want_gate[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_multi_token: gate "
                    "mismatch");
        CHECK_CLOSE(got_up[i], want_up[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_multi_token: up "
                    "mismatch");
        CHECK_CLOSE(got_mid[i], want_mid[i], 1e-2,
                    "shared_gate_up_swiglu_rows_scalar_multi_token: mid "
                    "mismatch");
    }

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    fprintf(stderr,
            "  test_shared_gate_up_swiglu_rows_scalar_multi_token OK\n");
    return 0;
}

static int test_shared_gate_up_swiglu_rows_scalar_rejections(void) {
    enum { IN_DIM = 32, OUT_DIM = 4, N_TOK = 3 };
    const uint64_t row_bytes = 34;
    unsigned char wg[OUT_DIM * 34];
    unsigned char wu[OUT_DIM * 34];
    float x[N_TOK * IN_DIM];
    float gate[N_TOK * OUT_DIM], up[N_TOK * OUT_DIM], mid[N_TOK * OUT_DIM];

    for (uint32_t o = 0; o < OUT_DIM; o++) {
        test_encode_q8_0_row(wg + (size_t)o * row_bytes, IN_DIM, o, 2);
        test_encode_q8_0_row(wu + (size_t)o * row_bytes, IN_DIM, o, 42);
    }
    for (uint32_t k = 0; k < N_TOK * IN_DIM; k++) x[k] = (float)k;

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
          "shared_gate_up_swiglu_rows_scalar_rejections: allocation "
          "failed");
    CHECK(ds4_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: write x");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  NULL, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: null gate must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, NULL, model_size, gate_offset, up_offset,
                  IN_DIM, OUT_DIM, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: null model_map "
          "must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, NULL, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: null x must be "
          "rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, 0, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: zero n_tok must "
          "be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, 0, OUT_DIM, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: zero in_dim must "
          "be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, 0, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: zero out_dim must "
          "be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tsmall, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: undersized gate "
          "must be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, gate_offset,
                  up_offset, IN_DIM, OUT_DIM, tsmall, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: undersized x must "
          "be rejected");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                  tgate, tup, tmid, combined, model_size, model_size,
                  up_offset, IN_DIM, OUT_DIM, tx, N_TOK, 0.0f) == 0,
          "shared_gate_up_swiglu_rows_scalar_rejections: out-of-range "
          "gate_offset must be rejected");

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tup);
    ds4_gpu_tensor_free(tmid);
    ds4_gpu_tensor_free(tsmall);
    fprintf(stderr,
            "  test_shared_gate_up_swiglu_rows_scalar_rejections OK\n");
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
    if (test_shared_gate_up_swiglu_fast_path_correctness() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_fast_vs_general_differential() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_rows_scalar_n_tok_1() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_rows_scalar_multi_token() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    if (test_shared_gate_up_swiglu_rows_scalar_rejections() != 0) {
        ds4_gpu_cleanup();
        return 1;
    }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_shared_expert OK\n");
    return 0;
}
