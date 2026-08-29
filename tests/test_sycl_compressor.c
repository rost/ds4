/* Correctness tests for SYCL compute kernels, validated against scalar
 * oracles implemented here.  The ds4.c CPU references are static and
 * cannot be linked, so each oracle reimplements the documented formula
 * with the ds4.c line number cited.  Needs no model file.
 *
 * This file covers the KV compressor kernels: APE-based store, pooled
 * readback, in-place state update, and prefill (both the general path and
 * the ratio-4 state/replay specialisations). */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "test_sycl_harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* oracle_ape_value(model, ape_type, width, row, col): the APE accessor
 * later compressor tasks reuse.  `model` points directly at the start of
 * the APE table (offset already applied), matching how the GPU side reads
 * model_ape_value_dev(..., width, row, col) as row * width + col
 * (rocm/ds4_rocm_norm_rope.cuh:366-378); ds4's CPU reads the same element
 * via tensor_2d_value(model, ape, j, pos_mod) (ds4.c:7882-7887), which is
 * tensor_1d_value(y * dim[0] + x).  These address the same element.
 * ape_type is restricted to 0 = F32, 1 = F16, 8 = Q8_0
 * (rocm/ds4_rocm_compressor.cuh:181-183); the Q8_0 branch uses the same
 * 32-value, 34-byte block layout (2-byte little-endian F16 scale then 32
 * int8) as test_embed_q8_0 in tests/test_sycl_embedding.c. */
static float oracle_ape_value(const unsigned char *model, uint32_t ape_type,
                              uint32_t width, uint32_t row, uint32_t col) {
    if (ape_type == 1u) {
        const uint16_t *p = (const uint16_t *)model;
        return oracle_half_to_float(p[(uint64_t)row * width + col]);
    }
    if (ape_type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *blk =
                model + (uint64_t)row * row_bytes + (uint64_t)(col / 32u) * 34u;
        const uint16_t raw = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const float scale = oracle_half_to_float(raw);
        const signed char q = (signed char)blk[2 + (col % 32u)];
        return scale * (float)q;
    }
    const float *p = (const float *)model;
    return p[(uint64_t)row * width + col];
}

/* Fills an ape_type-encoded table of `rows` rows by `width` columns with a
 * value distinguishable per (row, col), so a transposed or misindexed read
 * produces a visibly wrong value rather than a coincidental match.  The F16
 * branch is restricted to the 8 exactly-representable small integers used
 * by test_embed_f16 in tests/test_sycl_embedding.c, for the same reason:
 * no float-to-half encoder exists in this C test, so the oracle must stay
 * exact by construction rather than by rounding agreement with the
 * kernel. */
static void oracle_fill_ape(unsigned char *dst, uint32_t ape_type,
                            uint32_t width, uint32_t rows) {
    static const uint16_t kHalfSmallInt[8] = {
        0x0000, /* 0.0 */ 0x3C00, /* 1.0 */ 0x4000, /* 2.0 */ 0x4200, /* 3.0 */
        0x4400, /* 4.0 */ 0x4500, /* 5.0 */ 0x4600, /* 6.0 */ 0x4700  /* 7.0 */
    };
    if (ape_type == 0u) {
        float *p = (float *)dst;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < width; c++) {
                p[(uint64_t)r * width + c] = (float)((int)(r * 13u + c * 7u) - 20);
            }
        }
        return;
    }
    if (ape_type == 1u) {
        uint16_t *p = (uint16_t *)dst;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t c = 0; c < width; c++) {
                p[(uint64_t)r * width + c] = kHalfSmallInt[(r * 3u + c * 5u) % 8u];
            }
        }
        return;
    }
    /* Q8_0: `rows` independent row blocks, each ceil(width/32) blocks of
     * 32 values / 34 bytes.  Scale is fixed at exactly 1.0 (0x3C00) so the
     * dequantised value equals the stored int8 exactly. */
    const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < width; c += 32u) {
            unsigned char *blk = dst + (uint64_t)r * row_bytes + (uint64_t)(c / 32u) * 34u;
            blk[0] = 0x00; blk[1] = 0x3C;
            for (uint32_t i = 0; i < 32u && c + i < width; i++) {
                blk[2 + i] = (unsigned char)(signed char)((r * width + c + i) % 15u - 7);
            }
        }
    }
}

/* The store oracle: the store half of compressor_decode_one
 * (ds4.c:12549-12554), whose ring math is at ds4.c:12521-12524, applied
 * per token in a batch rather than one token at a time. */
static void oracle_compressor_store(float *state_kv, float *state_score,
                                    const float *kv, const float *sc,
                                    const unsigned char *ape, uint32_t ape_type,
                                    uint32_t width, uint32_t ratio,
                                    uint32_t pos0, uint32_t n_tokens) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos_mod = (pos0 + t) % ratio;
        const uint32_t dst_row = (ratio == 4u) ? ratio + pos_mod : pos_mod;
        for (uint32_t j = 0; j < width; j++) {
            state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
            state_score[(uint64_t)dst_row * width + j] =
                    sc[(uint64_t)t * width + j] +
                    oracle_ape_value(ape, ape_type, width, pos_mod, j);
        }
    }
}

/* Runs one (ape_type, ratio) combination of test_compressor_store below:
 * builds kv/sc/ape inputs, seeds the state tensors with a sentinel so any
 * row the kernel does not touch is provably untouched, calls the entry
 * under test, and compares every state row against oracle_compressor_store.
 * `head_dim` and `ratio` are caller-chosen so this same body covers both
 * the general-ratio and the ratio == 4 destination-row path. */
static int oracle_compressor_store_case(uint32_t ape_type, uint32_t head_dim,
                                        uint32_t ratio, uint32_t pos0,
                                        uint32_t n_tokens, uint64_t ape_offset) {
    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t ape_bytes =
            ape_type == 0u ? (uint64_t)width * ratio * sizeof(float) :
            ape_type == 1u ? (uint64_t)width * ratio * sizeof(uint16_t) :
                             (uint64_t)ratio * (((uint64_t)width + 31u) / 32u * 34u);
    const uint64_t model_size = ape_offset + ape_bytes;

    float *kv    = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *sc    = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *want_kv    = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *want_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_kv     = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_score  = (float *)malloc((size_t)state_rows * width * sizeof(float));
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    CHECK(kv && sc && want_kv && want_score && got_kv && got_score && model,
          "compressor_store: host allocation failed");

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < width; j++) {
            kv[(uint64_t)t * width + j] = (float)(t * 1000u + j * 10u + 3u);
            sc[(uint64_t)t * width + j] = (float)((int)(t * 100u) - (int)(j * 2u) - 5);
        }
    }
    /* Sentinel padding before ape_offset: an implementation that ignores
     * ape_offset and reads from the start of the model buffer would decode
     * this instead of the real table and produce a visibly wrong score. */
    memset(model, 0xAA, (size_t)ape_offset);
    oracle_fill_ape(model + ape_offset, ape_type, width, ratio);

    /* Sentinel state rows: for ratio == 4 only the high half (rows
     * ratio..2*ratio-1) should ever be written.  Seeding the whole buffer
     * with a value no oracle row produces lets the post-call comparison
     * prove the low half was left alone, not just that the high half is
     * correct. */
    for (uint32_t i = 0; i < state_rows * width; i++) {
        want_kv[i] = want_score[i] = -999.0f;
    }
    oracle_compressor_store(want_kv, want_score, kv, sc, model + ape_offset,
                            ape_type, width, ratio, pos0, n_tokens);

    ds4_gpu_tensor *tkv    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tsc    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tskv   = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tssc   = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    CHECK(tkv && tsc && tskv && tssc, "compressor_store: device allocation failed");

    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_store: write kv");
    CHECK(ds4_gpu_tensor_write(tsc, 0, sc, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_store: write sc");
    CHECK(ds4_gpu_tensor_write(tskv, 0, want_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: seed state_kv sentinel");
    CHECK(ds4_gpu_tensor_write(tssc, 0, want_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: seed state_score sentinel");

    CHECK(ds4_gpu_compressor_store_batch_tensor(
              tkv, tsc, tskv, tssc, model, model_size, ape_offset, ape_type,
              head_dim, ratio, pos0, n_tokens) != 0,
          "compressor_store: call");

    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_store: read state_score");

    for (uint32_t i = 0; i < state_rows * width; i++) {
        CHECK(got_kv[i] == want_kv[i], "compressor_store: state_kv mismatch");
        CHECK_CLOSE(got_score[i], want_score[i], 1e-3,
                    "compressor_store: state_score mismatch");
    }

    ds4_gpu_tensor_free(tkv);
    ds4_gpu_tensor_free(tsc);
    ds4_gpu_tensor_free(tskv);
    ds4_gpu_tensor_free(tssc);
    free(kv); free(sc); free(want_kv); free(want_score);
    free(got_kv); free(got_score); free(model);
    return 0;
}

/* ds4_gpu_compressor_store_batch_tensor has no ds4.c call site: it is
 * invoked backend-internally by ds4_gpu_compressor_update_tensor with
 * n_tokens = 1 (rocm/ds4_rocm_compressor.cuh:292-298), never by ds4.c
 * directly.  This entry point is tested here directly against
 * oracle_compressor_store for all three permitted ape_type values, and
 * for both the general-ratio and the ratio == 4 destination-row path
 * (they compute dst_row differently; covering only one would prove only
 * half the entry).  pos0 is non-zero and not a multiple of ratio in both
 * cases, so the modular ring wrap is genuinely exercised rather than
 * hidden by tokens landing at rows 0..n_tokens-1 in order. */
static int test_compressor_store(void) {
    const uint32_t ape_types[] = {0u, 1u, 8u};

    for (size_t i = 0; i < sizeof(ape_types) / sizeof(ape_types[0]); i++) {
        const uint32_t ape_type = ape_types[i];

        /* General ratio: coff == 1, dst_row == pos_mod.  ratio = 3,
         * pos0 = 8 (8 % 3 == 2, non-zero and not a multiple of 3), and
         * n_tokens == ratio so the three tokens sweep the whole ring
         * exactly once: pos_mod goes 2, 0, 1, wrapping from the last row
         * back to the first partway through the batch. */
        if (oracle_compressor_store_case(ape_type, /*head_dim=*/5u, /*ratio=*/3u,
                                         /*pos0=*/8u, /*n_tokens=*/3u,
                                         /*ape_offset=*/16u) != 0) {
            return 1;
        }

        /* ratio == 4: coff == 2, dst_row == ratio + pos_mod, landing in the
         * HIGH half (rows 4..7) of the 8-row ring; the low half (rows
         * 0..3) must be left untouched.  pos0 = 6 (6 % 4 == 2, non-zero
         * and not a multiple of 4); pos_mod goes 2, 3, 0, 1, again
         * wrapping mid-batch. */
        if (oracle_compressor_store_case(ape_type, /*head_dim=*/5u, /*ratio=*/4u,
                                         /*pos0=*/6u, /*n_tokens=*/4u,
                                         /*ape_offset=*/16u) != 0) {
            return 1;
        }
    }

    /* An unsupported ape_type must be rejected (rocm/ds4_rocm_compressor.cuh:
     * 181-183 permits only 0, 1 and 8). */
    {
        enum { HEAD_DIM = 5, RATIO = 3, WIDTH = HEAD_DIM, N_TOKENS = 3 };
        unsigned char model[WIDTH * RATIO * sizeof(float)];
        memset(model, 0, sizeof(model));
        ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)RATIO * WIDTH * sizeof(float));
        ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)RATIO * WIDTH * sizeof(float));
        CHECK(tkv && tsc && tskv && tssc,
              "compressor_store: allocation failed (unsupported ape_type case)");
        CHECK(ds4_gpu_compressor_store_batch_tensor(
                  tkv, tsc, tskv, tssc, model, sizeof(model), 0, /*ape_type=*/2u,
                  HEAD_DIM, RATIO, 8u, N_TOKENS) == 0,
              "compressor_store: unsupported ape_type must be rejected");
        /* n_tokens == 0 must also be rejected: unlike most entries in this
         * backend, zero-sized work is not success here, matching the
         * validation list at rocm/ds4_rocm_compressor.cuh:204-208. */
        CHECK(ds4_gpu_compressor_store_batch_tensor(
                  tkv, tsc, tskv, tssc, model, sizeof(model), 0, 0u,
                  HEAD_DIM, RATIO, 8u, 0u) == 0,
              "compressor_store: n_tokens == 0 must be rejected");
        ds4_gpu_tensor_free(tkv);
        ds4_gpu_tensor_free(tsc);
        ds4_gpu_tensor_free(tskv);
        ds4_gpu_tensor_free(tssc);
    }

    fprintf(stderr, "  test_compressor_store OK\n");
    return 0;
}

/* The pool oracle: compressor_update_pool_kernel
 * (rocm/ds4_rocm_compressor.cuh:121-160), which matches
 * compressor_pool_decode_state (ds4.c:12452-12503) but expressed as an
 * unconditional softmax rather than the CPU's early-out on the sentinel
 * DS4_NEG_INF.  The `max_s > -INFINITY` guard below is NOT part of a
 * byte-for-byte port of the CUDA loop: without it, an all -inf candidate
 * set (the empty-softmax case this file tests) hits the IEEE `inf - inf`
 * indeterminate form and resolves to NaN instead of 0. */
static void oracle_compressor_pool(float *out, const float *state_kv,
                                   const float *state_score,
                                   uint32_t head_dim, uint32_t ratio) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;

    for (uint32_t d = 0; d < head_dim; d++) {
        float vals[16], scores[16];
        uint32_t n = 0;
        float max_s = -INFINITY;

        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n] = state_kv[(uint64_t)r * width + d];
                scores[n] = state_score[(uint64_t)r * width + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
                scores[n] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                vals[n] = state_kv[(uint64_t)r * width + d];
                scores[n] = state_score[(uint64_t)r * width + d];
                if (scores[n] > max_s) max_s = scores[n];
                n++;
            }
        }

        float den = 0.0f, acc = 0.0f;
        if (max_s > -INFINITY) {
            for (uint32_t i = 0; i < n; i++) {
                const float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
        }
        out[d] = den != 0.0f ? acc / den : 0.0f;
    }
}

/* Step 1: the pool kernel driven in isolation through
 * ds4_gpu_compressor_update_tensor with state_already_stored = true and
 * n_rot = 0, so the store delegation and RoPE are both bypassed and only
 * pooling plus the RMS weight normalisation run.  The reference calls this
 * "the single most complex kernel in the four subsystems" and recommends
 * proving it before trusting any orchestration around it. */
static int test_compressor_pool(void) {
    const float RMS_EPS = 1e-5f;

    /* A1: general ratio (coff == 1), ratio = 3, head_dim = 4. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float norm_w[HEAD_DIM], pool_want[HEAD_DIM], want[HEAD_DIM];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 200.0f + (float)r * 20.0f + (float)c * 3.0f;
                state_score[r * WIDTH + c] = 10.0f + (float)r * 4.0f + (float)c * 0.5f;
            }
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d % 3 + 1) * 0.4f;

        oracle_compressor_pool(pool_want, state_kv, state_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(want, pool_want, norm_w, HEAD_DIM, RMS_EPS);

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A1)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A1)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A1)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A1)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A1)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A1)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A1)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/5u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A1)");

        float got[HEAD_DIM], got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A1)");
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_pool: read state_kv (A1)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_pool: read state_score (A1)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got[d], want[d], 1e-4, "compressor_pool: emitted value mismatch (A1)");
        }
        /* General ratio never shifts: the ring must be untouched. */
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == state_kv[i], "compressor_pool: ring kv mutated (A1)");
            CHECK(got_score[i] == state_score[i], "compressor_pool: ring score mutated (A1)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* A2: ratio == 4, the asymmetric-lane gather.  The low ring rows carry
     * one value range at column d (the low lane) and a POISONED,
     * unmistakably different value range at column head_dim + d (a lane a
     * correct low-row read never touches).  The high ring rows are the
     * mirror image: genuine data at head_dim + d, poison at d.  A port
     * that reads column d for both halves -- the sharpest trap in this
     * task -- would pick up the high rows' poison instead of their
     * genuine high-lane data, producing a visibly different, wrong
     * result. */
    {
        enum { HEAD_DIM = 3, RATIO = 4, WIDTH = HEAD_DIM * 2, STATE_ROWS = RATIO * 2 };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];

        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 100.0f + (float)r * 10.0f + (float)c;
                state_score[r * WIDTH + c] = 5.0f + (float)r + 0.1f * (float)c;
            }
        }
        for (uint32_t r = 0; r < 4u; r++) {
            const uint32_t row = 4u + r;
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[row * WIDTH + c] = -500.0f - (float)r - (float)c;
                    state_score[row * WIDTH + c] = -777.0f - (float)r - (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[row * WIDTH + c] = 700.0f + (float)r * 10.0f + (float)d;
                    state_score[row * WIDTH + c] = 20.0f + (float)r + 0.1f * (float)d;
                }
            }
        }

        float norm_w[HEAD_DIM], pool_want[HEAD_DIM], want[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 1) * 0.5f;
        oracle_compressor_pool(pool_want, state_kv, state_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(want, pool_want, norm_w, HEAD_DIM, RMS_EPS);

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A2)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A2)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A2)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A2)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A2)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A2)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A2)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/7u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A2)");

        float got[HEAD_DIM], got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A2)");
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_pool: read state_kv (A2)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_pool: read state_score (A2)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got[d], want[d], 1e-4, "compressor_pool: emitted value mismatch (A2, ratio4 lanes)");
        }

        /* ratio == 4 always shifts on emit, even when the store step was
         * skipped: rows 0..3 must become the OLD rows 4..7 verbatim (full
         * width, both lanes, poison columns included), and rows 4..7 must
         * be unchanged (the shift copies the high half onto itself too). */
        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                const float want_kv_v = state_kv[(4u + r) * WIDTH + c];
                const float want_sc_v = state_score[(4u + r) * WIDTH + c];
                CHECK(got_kv[r * WIDTH + c] == want_kv_v, "compressor_pool: shifted low kv mismatch (A2)");
                CHECK(got_score[r * WIDTH + c] == want_sc_v, "compressor_pool: shifted low score mismatch (A2)");
                CHECK(got_kv[(4u + r) * WIDTH + c] == want_kv_v, "compressor_pool: high kv changed by shift (A2)");
                CHECK(got_score[(4u + r) * WIDTH + c] == want_sc_v, "compressor_pool: high score changed by shift (A2)");
            }
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* A3: every candidate score is -inf.  den must resolve to exactly 0,
     * not NaN; a dropped guard here only manifests at sequence starts,
     * where the ring genuinely holds nothing yet. */
    {
        enum { HEAD_DIM = 2, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            state_kv[i] = 42.0f;
            state_score[i] = -INFINITY;
        }
        float norm_w[HEAD_DIM] = {1.0f, 1.0f};

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_pool: model allocation failed (A3)");
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0};
        float comp_sentinel[HEAD_DIM] = {-999.0f, -999.0f};

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_pool: device allocation failed (A3)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_pool: write kv_cur (A3)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_pool: write sc_cur (A3)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_pool: write state_kv (A3)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_pool: write state_score (A3)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_pool: write comp sentinel (A3)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/0u,
                  /*n_ctx_orig=*/4096u, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                  /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                  /*beta_slow=*/1.0f, RMS_EPS, /*state_already_stored=*/true,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_pool: call (A3)");

        float got[HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got, sizeof(got)) != 0, "compressor_pool: read comp (A3)");
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK(got[d] == 0.0f, "compressor_pool: empty softmax must be exactly 0, not NaN (A3)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_pool OK\n");
    return 0;
}

/* Step 2: ds4_gpu_compressor_update_tensor is itself single-token, so it
 * compares 1:1 against compressor_decode_one (ds4.c:12507-12602) with no
 * loop needed.  Covers a non-emit position (store only), an emit boundary
 * (store, pool, RMS weight norm and RoPE tail together), and for ratio ==
 * 4 the post-emit shift that copies the high half of the ring into BOTH
 * halves (rocm/ds4_rocm_compressor.cuh:162-172).  Every case asserts the
 * full state ring after the call, not only the emitted row: a correct
 * emission with a corrupted ring would pass otherwise and break the next
 * token. */
static int test_compressor_update(void) {
    const float RMS_EPS = 1e-5f;
    const uint32_t N_CTX_ORIG = 4096u;
    const float FREQ_BASE = 10000.0f, FREQ_SCALE = 1.0f, EXT_FACTOR = 0.0f;
    const float ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

    /* B1: a non-emit position stores the current token into the ring and
     * returns success without touching comp_cache. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        const uint32_t POS = 3u; /* (3+1) % 3 == 1: not an emit boundary */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 300.0f + (float)r * 20.0f + (float)c * 2.0f;
                state_score[r * WIDTH + c] = 8.0f + (float)r * 3.0f + (float)c * 0.2f;
            }
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = 777.0f + (float)c;
            sc_cur[c] = 333.0f + (float)c * 2.0f;
        }

        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B1)");
        oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 1) * 0.25f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float comp_sentinel[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) comp_sentinel[d] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B1)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B1)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B1)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B1)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B1)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B1)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, /*comp_row=*/0u, /*n_rot=*/0u,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B1, non-emit)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH], got_comp[HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B1)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B1)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B1)");

        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_kv[i], "compressor_update: stored kv mismatch (B1)");
            CHECK_CLOSE(got_score[i], want_score[i], 1e-3, "compressor_update: stored score mismatch (B1)");
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK(got_comp[d] == comp_sentinel[d], "compressor_update: comp_cache must be untouched on non-emit (B1)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* B2: an emit boundary, general ratio.  Store, pool, RMS weight norm
     * and RoPE tail all run and are checked against one combined
     * oracle. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO,
               N_ROT = 2, COMP_ROW = 1, COMP_ROWS = COMP_ROW + 1 };
        const uint32_t POS = 2u; /* (2+1) % 3 == 0: emit */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        for (uint32_t r = 0; r < STATE_ROWS; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                state_kv[r * WIDTH + c] = 50.0f + (float)r * 15.0f + (float)c;
                state_score[r * WIDTH + c] = 3.0f + (float)r * 2.0f + (float)c * 0.3f;
            }
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = 900.0f + (float)c * 5.0f;
            sc_cur[c] = 60.0f - (float)c;
        }

        const uint64_t ape_offset = 24;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B2)");
        oracle_fill_ape(model + ape_offset, 0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d % 4 + 1) * 0.35f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float pool_want[HEAD_DIM], normed_want[HEAD_DIM];
        oracle_compressor_pool(pool_want, want_kv, want_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, HEAD_DIM, RMS_EPS);
        const uint32_t comp_pos = POS + 1u - RATIO;
        oracle_rope_tail_row(normed_want, HEAD_DIM, N_ROT, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);

        float comp_sentinel[COMP_ROWS * HEAD_DIM];
        for (uint32_t i = 0; i < COMP_ROWS * HEAD_DIM; i++) comp_sentinel[i] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B2)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B2)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B2)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B2)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B2)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B2)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, COMP_ROW, N_ROT,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B2, emit)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        float got_comp[COMP_ROWS * HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B2)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B2)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B2)");

        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_kv[i], "compressor_update: post-emit ring kv mismatch (B2)");
            CHECK_CLOSE(got_score[i], want_score[i], 1e-3, "compressor_update: post-emit ring score mismatch (B2)");
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got_comp[COMP_ROW * HEAD_DIM + d], normed_want[d], 1e-3,
                       "compressor_update: emitted row mismatch (B2)");
        }
        for (uint32_t i = 0; i < COMP_ROW * HEAD_DIM; i++) {
            CHECK(got_comp[i] == comp_sentinel[i], "compressor_update: rows before comp_row must be untouched (B2)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* B3: ratio == 4, exercising the full store -> pool -> norm -> RoPE
     * chain together with the post-emit shift.  The low ring rows and the
     * high ring rows (and, within the high rows, the low vs. high lane
     * columns) all carry visibly different values so a lane or row swap
     * anywhere in the pipeline produces a detectably wrong answer, and the
     * ring is checked in full afterward, not just the emitted row. */
    {
        enum { HEAD_DIM = 4, RATIO = 4, WIDTH = HEAD_DIM * 2, STATE_ROWS = RATIO * 2,
               N_ROT = 2, COMP_ROW = 0, COMP_ROWS = COMP_ROW + 1 };
        const uint32_t POS = 7u; /* pos % 4 == 3: last slot of the window; (7+1)%4==0: emit */
        float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
        float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
        float kv_cur[WIDTH], sc_cur[WIDTH];

        /* Low ring (rows 0..3): previous window.  Genuine data at column
         * d < HEAD_DIM; poison at head_dim + d, which a correct low-row
         * read never touches. */
        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[r * WIDTH + c] = 100.0f + (float)r * 10.0f + (float)c;
                    state_score[r * WIDTH + c] = 5.0f + (float)r + 0.1f * (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[r * WIDTH + c] = -300.0f - (float)r - (float)d;
                    state_score[r * WIDTH + c] = -666.0f - (float)r - (float)d;
                }
            }
        }
        /* High ring (rows 4..6): the first three tokens of the current
         * window, already stored.  Poison at column d < HEAD_DIM, genuine
         * data at head_dim + d. */
        for (uint32_t r = 0; r < 3u; r++) {
            const uint32_t row = 4u + r;
            for (uint32_t c = 0; c < WIDTH; c++) {
                if (c < HEAD_DIM) {
                    state_kv[row * WIDTH + c] = -500.0f - (float)r - (float)c;
                    state_score[row * WIDTH + c] = -777.0f - (float)r - (float)c;
                } else {
                    const uint32_t d = c - HEAD_DIM;
                    state_kv[row * WIDTH + c] = 700.0f + (float)r * 10.0f + (float)d;
                    state_score[row * WIDTH + c] = 20.0f + (float)r + 0.1f * (float)d;
                }
            }
        }
        /* Row 7 (the fourth slot of the window) is not pre-seeded: this
         * call's store fills it from kv_cur/sc_cur. */
        for (uint32_t c = 0; c < WIDTH; c++) {
            state_kv[7 * WIDTH + c] = 0.0f;
            state_score[7 * WIDTH + c] = 0.0f;
        }
        memcpy(want_kv, state_kv, sizeof(state_kv));
        memcpy(want_score, state_score, sizeof(state_score));
        for (uint32_t c = 0; c < WIDTH; c++) {
            kv_cur[c] = c < HEAD_DIM ? (-800.0f - (float)c) : (750.0f + (float)(c - HEAD_DIM));
            sc_cur[c] = c < HEAD_DIM ? (-888.0f - (float)c) : (25.0f + (float)(c - HEAD_DIM));
        }

        const uint64_t ape_offset = 32;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (B3)");
        oracle_fill_ape(model + ape_offset, 0u, WIDTH, RATIO);
        float norm_w[HEAD_DIM];
        for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d + 2) * 0.3f;
        memcpy(model + norm_offset, norm_w, sizeof(norm_w));

        oracle_compressor_store(want_kv, want_score, kv_cur, sc_cur,
                                model + ape_offset, 0u, WIDTH, RATIO, POS, 1u);

        float pool_want[HEAD_DIM], normed_want[HEAD_DIM];
        oracle_compressor_pool(pool_want, want_kv, want_score, HEAD_DIM, RATIO);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, HEAD_DIM, RMS_EPS);
        const uint32_t comp_pos = POS + 1u - RATIO;
        oracle_rope_tail_row(normed_want, HEAD_DIM, N_ROT, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);

        /* The expected post-shift ring: rows 0..3 become the OLD rows
         * 4..7 (i.e. want_kv/want_score after the store above, before any
         * shift); rows 4..7 stay the same. */
        float want_shifted_kv[STATE_ROWS * WIDTH], want_shifted_score[STATE_ROWS * WIDTH];
        memcpy(want_shifted_kv, want_kv, sizeof(want_kv));
        memcpy(want_shifted_score, want_score, sizeof(want_score));
        for (uint32_t r = 0; r < 4u; r++) {
            memcpy(&want_shifted_kv[r * WIDTH], &want_kv[(4u + r) * WIDTH], WIDTH * sizeof(float));
            memcpy(&want_shifted_score[r * WIDTH], &want_score[(4u + r) * WIDTH], WIDTH * sizeof(float));
        }

        float comp_sentinel[COMP_ROWS * HEAD_DIM];
        for (uint32_t i = 0; i < COMP_ROWS * HEAD_DIM; i++) comp_sentinel[i] = -999.0f;

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (B3)");

        CHECK(ds4_gpu_tensor_write(tkv_cur, 0, kv_cur, sizeof(kv_cur)) != 0, "compressor_update: write kv_cur (B3)");
        CHECK(ds4_gpu_tensor_write(tsc_cur, 0, sc_cur, sizeof(sc_cur)) != 0, "compressor_update: write sc_cur (B3)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (B3)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (B3)");
        CHECK(ds4_gpu_tensor_write(tcomp, 0, comp_sentinel, sizeof(comp_sentinel)) != 0, "compressor_update: write comp sentinel (B3)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp,
                  model, model_size, ape_offset, /*ape_type=*/0u,
                  norm_offset, /*norm_type=*/0u, HEAD_DIM, RATIO,
                  POS, COMP_ROW, N_ROT,
                  N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                  BETA_FAST, BETA_SLOW, RMS_EPS, /*state_already_stored=*/false,
                  /*decode_one_token=*/false, /*defer_finalize=*/false) != 0,
              "compressor_update: call (B3, ratio4 emit + shift)");

        float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
        float got_comp[COMP_ROWS * HEAD_DIM];
        CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0, "compressor_update: read state_kv (B3)");
        CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0, "compressor_update: read state_score (B3)");
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0, "compressor_update: read comp (B3)");

        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK_CLOSE(got_comp[COMP_ROW * HEAD_DIM + d], normed_want[d], 1e-3,
                       "compressor_update: emitted row mismatch (B3, ratio4)");
        }
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
            CHECK(got_kv[i] == want_shifted_kv[i], "compressor_update: post-shift ring kv mismatch (B3)");
            CHECK_CLOSE(got_score[i], want_shifted_score[i], 1e-3, "compressor_update: post-shift ring score mismatch (B3)");
        }

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    /* Validation: an odd n_rot, n_rot > head_dim, and a non-zero norm_type
     * must all be rejected before any kernel launches
     * (rocm/ds4_rocm_compressor.cuh:272-279). */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO };
        float state_kv[STATE_ROWS * WIDTH] = {0}, state_score[STATE_ROWS * WIDTH] = {0};
        float kv_cur[WIDTH] = {0}, sc_cur[WIDTH] = {0}, comp[HEAD_DIM] = {0};
        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_update: model allocation failed (validation)");

        ds4_gpu_tensor *tkv_cur = ds4_gpu_tensor_alloc(sizeof(kv_cur));
        ds4_gpu_tensor *tsc_cur = ds4_gpu_tensor_alloc(sizeof(sc_cur));
        ds4_gpu_tensor *tskv    = ds4_gpu_tensor_alloc(sizeof(state_kv));
        ds4_gpu_tensor *tssc    = ds4_gpu_tensor_alloc(sizeof(state_score));
        ds4_gpu_tensor *tcomp   = ds4_gpu_tensor_alloc(sizeof(comp));
        CHECK(tkv_cur && tsc_cur && tskv && tssc && tcomp,
              "compressor_update: device allocation failed (validation)");
        CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv, sizeof(state_kv)) != 0, "compressor_update: write state_kv (validation)");
        CHECK(ds4_gpu_tensor_write(tssc, 0, state_score, sizeof(state_score)) != 0, "compressor_update: write state_score (validation)");

        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/1u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: odd n_rot must be rejected");
        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/HEAD_DIM + 2u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: n_rot > head_dim must be rejected");
        CHECK(ds4_gpu_compressor_update_tensor(
                  tkv_cur, tsc_cur, tskv, tssc, tcomp, model, model_size,
                  ape_offset, 0u, norm_offset, /*norm_type=*/1u, HEAD_DIM, RATIO,
                  /*pos=*/2u, /*comp_row=*/0u, /*n_rot=*/0u, N_CTX_ORIG,
                  FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                  BETA_SLOW, RMS_EPS, true, false, false) == 0,
              "compressor_update: non-zero norm_type must be rejected");

        ds4_gpu_tensor_free(tkv_cur); ds4_gpu_tensor_free(tsc_cur);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_update OK\n");
    return 0;
}

/* Mirrors compressor_finish_prefill_state_cpu, ds4.c:12407-12427: after a
 * from-scratch prefill, clear whichever ring rows the trailing partial
 * window's per-token stores never touched, so decode resumes from the same
 * partial-window state the streaming path would have produced.  Rows the
 * tail actually wrote are left alone.  ds4.c's function has no pos0
 * parameter: it is only ever reached with pos0 == 0 (ds4.c:28119's
 * zero_prefix gate is a precondition for calling the batched GPU prefill
 * entry this test drives), so this oracle assumes pos0 == 0 too. */
static void oracle_compressor_finish_prefill(float *state_kv, float *state_score,
                                             uint32_t head_dim, uint32_t ratio,
                                             uint32_t n_tokens) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rem = n_tokens % ratio;
    const uint32_t clear_start = ratio == 4u ? ratio + rem : rem;
    const uint32_t clear_end = ratio == 4u ? 2u * ratio : ratio;
    for (uint32_t row = clear_start; row < clear_end; row++) {
        for (uint32_t j = 0; j < width; j++) {
            state_kv[(uint64_t)row * width + j] = 0.0f;
            state_score[(uint64_t)row * width + j] = -INFINITY;
        }
    }
}

/* The central deliverable tested here: ds4_gpu_compressor_prefill_tensor has no
 * CPU counterpart with a matching signature.  ds4's own CPU prefill never
 * batches the compressor either: it always loops per token
 * (prefill_layer_major_cpu, ds4.c:13757, reaching compressor_decode_one via
 * ds4.c:13107, :13125, :13342, :13360).  So the oracle here IS that
 * per-token loop, built entirely from the compressor-store oracles already
 * validated above in this file (oracle_compressor_store,
 * oracle_compressor_pool, oracle_rms_norm_weight, oracle_rope_tail_row), run
 * once per position from pos0 == 0 to n_tokens - 1 with a fresh state ring
 * (state_kv zeroed, state_score -inf, matching
 * rocm/ds4_rocm_compressor.cuh:384-387), taking
 * the emitted row at every ratio boundary and applying the ratio == 4 shift
 * exactly as oracle_compressor_pool's caller does above.  pos0 == 0 matches
 * the only configuration ds4.c ever calls this entry with (see the finish
 * oracle's comment above), which is what lets the final ring be compared
 * against oracle_compressor_finish_prefill directly.
 *
 * ape_type is fixed at F32 (0): the batched entry's ape/norm handling is
 * identical to the already-covered store and update entries regardless of
 * ape_type, so re-sweeping all three here would not exercise anything new. */
static int run_compressor_prefill_case(uint32_t head_dim, uint32_t ratio,
                                       uint32_t n_tokens, uint32_t n_rot,
                                       const char *tag) {
    const float RMS_EPS = 1e-5f;
    const uint32_t N_CTX_ORIG = 4096u;
    const float FREQ_BASE = 10000.0f, FREQ_SCALE = 1.0f, EXT_FACTOR = 0.0f;
    const float ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp     = n_tokens / ratio;

    float *kv = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *sc = (float *)malloc((size_t)n_tokens * width * sizeof(float));
    float *norm_w = (float *)malloc((size_t)head_dim * sizeof(float));
    float *state_kv = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *state_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *comp_want = n_comp ? (float *)malloc((size_t)n_comp * head_dim * sizeof(float)) : NULL;
    CHECK(kv && sc && norm_w && state_kv && state_score && (n_comp == 0 || comp_want),
          "compressor_prefill: host allocation failed");

    /* kv includes a (t * j) % 7 interaction term, unlike a plain affine
     * ramp: a purely affine kv makes every candidate row differ from every
     * other only by a per-row additive constant, which RMS-normalisation
     * almost entirely divides back out regardless of which candidate the
     * softmax or the previous-window gather picks.  test_compressor_prefill
     * _ratio4_replay hit exactly this degeneracy first; see its comment for
     * the numeric detail. Without the term, an off-by-one in the ratio-4
     * previous-window gather (base = c * ratio instead of (c - 1) * ratio)
     * moves the post-RMS-norm output by about 6e-10, far under this test's
     * 1e-2 tolerance, so this data must not be affine either.
     *
     * sc needed a second, independent fix, verified numerically before
     * writing it here: the original `t * 19` term grows without bound
     * across windows, so for ratio == 4 the current window (larger t)
     * always outscores the previous window by dozens of units, and after
     * the softmax's exp() the previous window's weight is indistinguishable
     * from zero.  With that bias in place the previous-window off-by-one is
     * invisible no matter what kv contains, because the previous window
     * never contributes to the pooled result either way.  Keying the
     * affine term off (t % ratio) instead of raw t removes that unbounded
     * growth, and weighting j more heavily than t (* 20 here, versus the
     * * 19 on the bounded t term) instead makes the previous window's own
     * best candidate reliably outscore the current window's, since it reads
     * the smaller j half of the row: column d rather than head_dim + d.  So
     * the previous window's candidate wins the softmax in both the correct
     * and the ablated build, and what changes is only WHICH raw token that
     * winning candidate's value comes from, which is exactly the property
     * the previous-window base computation is responsible for getting
     * right.  A numeric simulation of this exact configuration (head_dim
     * = 4, ratio = 4, n_tokens = 11) confirmed the four pre-RMS-norm pooled
     * values for comp index 1 move by roughly 25 to 37 under the ablation,
     * comfortably above this test's 1e-2 tolerance. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < width; j++) {
            kv[(uint64_t)t * width + j] = (float)(t * 37u + j * 11u + 5u) * 0.25f +
                                          (float)((t * j) % 7u) * 2.0f;
            sc[(uint64_t)t * width + j] =
                    (float)((int)((t % ratio) * 19u) - (int)(j * 20u) - 7) +
                    (float)((t * j) % 11u);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) norm_w[d] = (float)(d % 5u + 1u) * 0.2f;

    const uint64_t ape_offset = 16;
    const uint64_t norm_offset = ape_offset + (uint64_t)width * ratio * sizeof(float);
    const uint64_t model_size = norm_offset + (uint64_t)head_dim * sizeof(float);
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    CHECK(model != NULL, "compressor_prefill: model allocation failed");
    oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, width, ratio);
    memcpy(model + norm_offset, norm_w, (size_t)head_dim * sizeof(float));

    /* Step 1: fresh ring, asymmetric init. */
    for (uint32_t i = 0; i < state_rows * width; i++) {
        state_kv[i] = 0.0f;
        state_score[i] = -INFINITY;
    }

    uint32_t emitted = 0;
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        oracle_compressor_store(state_kv, state_score, kv + (uint64_t)pos * width,
                                sc + (uint64_t)pos * width, model + ape_offset,
                                /*ape_type=*/0u, width, ratio, pos, 1u);
        if (((pos + 1u) % ratio) != 0u) continue;

        float pool_want[256], normed_want[256];
        oracle_compressor_pool(pool_want, state_kv, state_score, head_dim, ratio);
        oracle_rms_norm_weight(normed_want, pool_want, norm_w, (int)head_dim, RMS_EPS);
        const uint32_t comp_pos = pos + 1u - ratio;
        oracle_rope_tail_row(normed_want, head_dim, n_rot, comp_pos, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                             BETA_FAST, BETA_SLOW, /*inverse=*/0);
        memcpy(comp_want + (uint64_t)emitted * head_dim, normed_want,
               (size_t)head_dim * sizeof(float));
        emitted++;

        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)r * width], &state_kv[(uint64_t)(4u + r) * width],
                       (size_t)width * sizeof(float));
                memcpy(&state_score[(uint64_t)r * width], &state_score[(uint64_t)(4u + r) * width],
                       (size_t)width * sizeof(float));
            }
            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)(4u + r) * width], &state_kv[(uint64_t)r * width],
                       (size_t)width * sizeof(float));
                memcpy(&state_score[(uint64_t)(4u + r) * width], &state_score[(uint64_t)r * width],
                       (size_t)width * sizeof(float));
            }
        }
    }
    CHECK(emitted == n_comp, "compressor_prefill: oracle emit count mismatch");

    /* Post-prefill state cleanup: clear whatever the tail never wrote. */
    oracle_compressor_finish_prefill(state_kv, state_score, head_dim, ratio, n_tokens);

    ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc((uint64_t)(n_comp ? n_comp : 1u) * head_dim * sizeof(float));
    CHECK(tkv && tsc && tskv && tssc && tcomp, "compressor_prefill: device allocation failed");

    CHECK(ds4_gpu_tensor_write(tkv, 0, kv, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_prefill: write kv");
    CHECK(ds4_gpu_tensor_write(tsc, 0, sc, (uint64_t)n_tokens * width * sizeof(float)) != 0,
          "compressor_prefill: write sc");

    /* Sentinel-seed the state and comp_cache tensors: the entry must
     * reinitialise state itself (step 1), and must leave comp_cache alone
     * beyond row n_comp when n_comp == 0. */
    {
        float *sentinel_state = (float *)malloc((size_t)state_rows * width * sizeof(float));
        CHECK(sentinel_state != NULL, "compressor_prefill: sentinel allocation failed");
        for (uint32_t i = 0; i < state_rows * width; i++) sentinel_state[i] = 123456.0f;
        CHECK(ds4_gpu_tensor_write(tskv, 0, sentinel_state, (uint64_t)state_rows * width * sizeof(float)) != 0,
              "compressor_prefill: seed state_kv sentinel");
        CHECK(ds4_gpu_tensor_write(tssc, 0, sentinel_state, (uint64_t)state_rows * width * sizeof(float)) != 0,
              "compressor_prefill: seed state_score sentinel");
        free(sentinel_state);
    }
    {
        const uint32_t comp_alloc_rows = n_comp ? n_comp : 1u;
        float *sentinel_comp = (float *)malloc((size_t)comp_alloc_rows * head_dim * sizeof(float));
        CHECK(sentinel_comp != NULL, "compressor_prefill: comp sentinel allocation failed");
        for (uint32_t i = 0; i < comp_alloc_rows * head_dim; i++) sentinel_comp[i] = -999.0f;
        CHECK(ds4_gpu_tensor_write(tcomp, 0, sentinel_comp, (uint64_t)comp_alloc_rows * head_dim * sizeof(float)) != 0,
              "compressor_prefill: seed comp sentinel");
        free(sentinel_comp);
    }

    CHECK(ds4_gpu_compressor_prefill_tensor(
              tcomp, tskv, tssc, tkv, tsc,
              model, model_size, ape_offset, /*ape_type=*/0u,
              norm_offset, /*norm_type=*/0u, head_dim, ratio,
              /*pos0=*/0u, n_tokens, n_rot, N_CTX_ORIG,
              /*quantize_fp8=*/false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) != 0,
          "compressor_prefill: call");

    float *got_kv = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_score = (float *)malloc((size_t)state_rows * width * sizeof(float));
    float *got_comp = n_comp ? (float *)malloc((size_t)n_comp * head_dim * sizeof(float)) : NULL;
    CHECK(got_kv && got_score && (n_comp == 0 || got_comp), "compressor_prefill: readback allocation failed");
    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_prefill: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, (uint64_t)state_rows * width * sizeof(float)) != 0,
          "compressor_prefill: read state_score");
    if (n_comp) {
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, (uint64_t)n_comp * head_dim * sizeof(float)) != 0,
              "compressor_prefill: read comp");
    }

    for (uint32_t i = 0; i < n_comp * head_dim; i++) {
        CHECK_CLOSE(got_comp[i], comp_want[i], 1e-2, "compressor_prefill: emitted row mismatch");
    }
    for (uint32_t i = 0; i < state_rows * width; i++) {
        if (state_score[i] == -INFINITY) {
            CHECK(got_score[i] == -INFINITY, "compressor_prefill: cleared score row mismatch");
            CHECK(got_kv[i] == 0.0f, "compressor_prefill: cleared kv row mismatch");
        } else {
            CHECK(got_kv[i] == state_kv[i], "compressor_prefill: final ring kv mismatch");
            CHECK_CLOSE(got_score[i], state_score[i], 1e-3, "compressor_prefill: final ring score mismatch");
        }
    }

    fprintf(stderr, "  run_compressor_prefill_case(%s) OK\n", tag);

    ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc);
    ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
    free(kv); free(sc); free(norm_w); free(state_kv); free(state_score);
    free(comp_want); free(got_kv); free(got_score); free(got_comp);
    free(model);
    return 0;
}

/* Step 4's central deliverable: the batched prefill entry, checked against
 * the per-token driver above.  head_dim = 4 keeps N_ROT = 2 valid (n_rot
 * must be even and <= head_dim) while staying small enough for the 256-wide
 * scratch buffers in run_compressor_prefill_case. */
static int test_compressor_prefill(void) {
    /* General ratio: n_tokens = 7, ratio = 3 -> n_comp = 2, rem = 1 != 0,
     * so the trailing partial window's tail-seed branch runs. */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/3u,
                                    /*n_tokens=*/7u, /*n_rot=*/2u,
                                    "general ratio, rem != 0") != 0) {
        return 1;
    }

    /* ratio == 4: n_tokens = 11 -> n_comp = 2, cutoff = 8 >= ratio (the
     * PREVIOUS-window seed branch runs, pulling window 1's raw tokens
     * rather than window 0's), and rem = 3 != 0 (the CURRENT-window tail
     * seed branch also runs, into rows 4..6, leaving row 7 to be cleared). */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/4u,
                                    /*n_tokens=*/11u, /*n_rot=*/2u,
                                    "ratio == 4, cutoff >= ratio and rem != 0") != 0) {
        return 1;
    }

    /* n_comp == 0: n_tokens = 2 < ratio = 3.  The entry must seed state and
     * return success without ever touching comp_cache. */
    if (run_compressor_prefill_case(/*head_dim=*/4u, /*ratio=*/3u,
                                    /*n_tokens=*/2u, /*n_rot=*/2u,
                                    "n_comp == 0") != 0) {
        return 1;
    }

    /* Validation and failure-propagation checks. */
    {
        enum { HEAD_DIM = 4, RATIO = 3, WIDTH = HEAD_DIM, STATE_ROWS = RATIO, N_TOKENS = 6 };
        float kv[N_TOKENS * WIDTH] = {0}, sc[N_TOKENS * WIDTH] = {0};
        const uint64_t ape_offset = 16;
        const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
        const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
        unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
        CHECK(model != NULL, "compressor_prefill: model allocation failed (validation)");

        ds4_gpu_tensor *tkv  = ds4_gpu_tensor_alloc(sizeof(kv));
        ds4_gpu_tensor *tsc  = ds4_gpu_tensor_alloc(sizeof(sc));
        ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
        ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc((uint64_t)(N_TOKENS / RATIO) * HEAD_DIM * sizeof(float));
        CHECK(tkv && tsc && tskv && tssc && tcomp, "compressor_prefill: device allocation failed (validation)");
        CHECK(ds4_gpu_tensor_write(tkv, 0, kv, sizeof(kv)) != 0, "compressor_prefill: write kv (validation)");
        CHECK(ds4_gpu_tensor_write(tsc, 0, sc, sizeof(sc)) != 0, "compressor_prefill: write sc (validation)");

        /* n_tokens == 0 must be rejected, matching the validation list at
         * rocm/ds4_rocm_compressor.cuh:357. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, /*n_tokens=*/0u, /*n_rot=*/0u, 4096u,
                  false, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0,
              "compressor_prefill: n_tokens == 0 must be rejected");

        /* A non-zero norm_type must be rejected before any kernel launches. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, /*norm_type=*/1u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, N_TOKENS, /*n_rot=*/0u, 4096u,
                  false, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0,
              "compressor_prefill: non-zero norm_type must be rejected");

        /* quantize_fp8 = true reaches ds4_gpu_dsv4_fp8_kv_quantize_tensor.
         * That entry used to be a stub, so this assertion used to require
         * failure; now that it is implemented the call must succeed.  The
         * attention compressor's only production call sites pass true
         * unconditionally (ds4.c:28705, :28797, :28824), so this is the path
         * that actually runs. */
        CHECK(ds4_gpu_compressor_prefill_tensor(
                  tcomp, tskv, tssc, tkv, tsc, model, model_size,
                  ape_offset, 0u, norm_offset, 0u, HEAD_DIM, RATIO,
                  /*pos0=*/0u, N_TOKENS, /*n_rot=*/0u, 4096u,
                  /*quantize_fp8=*/true, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) != 0,
              "compressor_prefill: quantize_fp8 must succeed now the FP8 entry is real");

        ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc);
        ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
        free(model);
    }

    fprintf(stderr, "  test_compressor_prefill OK\n");
    return 0;
}

/* ds4_gpu_compressor_prefill_state_ratio4_tensor is the store-only half of
 * ds4_gpu_compressor_prefill_ratio4_replay_tensor's own tail-reseed step,
 * applied standalone to a tail of exactly four raw tokens, so it reuses
 * the compressor-store oracle (oracle_compressor_store) defined above
 * directly rather than needing a new one.
 *
 * oracle_compressor_store's own ratio == 4 formula always writes a token
 * into the HIGH half of an 8-row ring (dst_row = ratio + pos_mod), matching
 * the live decode/store path.  This entry instead writes into the LOW half
 * directly (rows 0..3), the shape ROCm's compressor_set_rows_kernel uses
 * with dst0 == 0.  Choosing POS0 as a multiple of 4 makes phase(r) == r for
 * r in 0..3, so oracle_compressor_store's high-half write for source row r
 * lands at oracle row 4 + r with no permutation to reason about: the
 * oracle's high half becomes exactly the device's expected low half, and
 * the oracle's untouched low half (still at the 0 / -inf reset baseline)
 * becomes exactly the device's expected high half. */
static int test_compressor_prefill_state_ratio4(void) {
    enum { HEAD_DIM = 5, RATIO = 4, WIDTH = 2 * HEAD_DIM, STATE_ROWS = 2 * RATIO, POS0 = 4 };

    float kv_tail[RATIO * WIDTH], sc_tail[RATIO * WIDTH];
    for (uint32_t t = 0; t < RATIO; t++) {
        for (uint32_t j = 0; j < WIDTH; j++) {
            kv_tail[t * WIDTH + j] = (float)(t * 50u + j * 3u + 1u);
            sc_tail[t * WIDTH + j] = (float)((int)(t * 17u) - (int)(j * 2u) - 4);
        }
    }

    const uint64_t ape_offset = 24;
    const uint64_t ape_bytes = (uint64_t)WIDTH * RATIO * sizeof(float);
    const uint64_t model_size = ape_offset + ape_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    CHECK(model != NULL, "compressor_state_ratio4: model allocation failed");
    memset(model, 0xAA, (size_t)ape_offset);
    oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, WIDTH, RATIO);

    float oracle_kv[STATE_ROWS * WIDTH], oracle_score[STATE_ROWS * WIDTH];
    for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
        oracle_kv[i] = 0.0f;
        oracle_score[i] = -INFINITY;
    }
    oracle_compressor_store(oracle_kv, oracle_score, kv_tail, sc_tail,
                            model + ape_offset, /*ape_type=*/0u, WIDTH, RATIO,
                            POS0, /*n_tokens=*/RATIO);

    float want_kv[STATE_ROWS * WIDTH], want_score[STATE_ROWS * WIDTH];
    for (uint32_t r = 0; r < STATE_ROWS; r++) {
        const uint32_t src_row = (r + RATIO) % STATE_ROWS;
        for (uint32_t j = 0; j < WIDTH; j++) {
            want_kv[r * WIDTH + j] = oracle_kv[src_row * WIDTH + j];
            want_score[r * WIDTH + j] = oracle_score[src_row * WIDTH + j];
        }
    }

    ds4_gpu_tensor *tkv_tail = ds4_gpu_tensor_alloc(sizeof(kv_tail));
    ds4_gpu_tensor *tsc_tail = ds4_gpu_tensor_alloc(sizeof(sc_tail));
    ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
    CHECK(tkv_tail && tsc_tail && tskv && tssc,
          "compressor_state_ratio4: device allocation failed");
    CHECK(ds4_gpu_tensor_write(tkv_tail, 0, kv_tail, sizeof(kv_tail)) != 0,
          "compressor_state_ratio4: write kv_tail");
    CHECK(ds4_gpu_tensor_write(tsc_tail, 0, sc_tail, sizeof(sc_tail)) != 0,
          "compressor_state_ratio4: write sc_tail");

    {
        float sentinel[STATE_ROWS * WIDTH];
        for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) sentinel[i] = 123456.0f;
        CHECK(ds4_gpu_tensor_write(tskv, 0, sentinel, sizeof(sentinel)) != 0,
              "compressor_state_ratio4: seed state_kv sentinel");
        CHECK(ds4_gpu_tensor_write(tssc, 0, sentinel, sizeof(sentinel)) != 0,
              "compressor_state_ratio4: seed state_score sentinel");
    }

    CHECK(ds4_gpu_compressor_prefill_state_ratio4_tensor(
              tskv, tssc, tkv_tail, tsc_tail, model, model_size,
              ape_offset, /*ape_type=*/0u, HEAD_DIM, POS0) != 0,
          "compressor_state_ratio4: call");

    float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0,
          "compressor_state_ratio4: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0,
          "compressor_state_ratio4: read state_score");

    for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
        CHECK(got_kv[i] == want_kv[i], "compressor_state_ratio4: state_kv mismatch");
        if (want_score[i] == -INFINITY) {
            CHECK(got_score[i] == -INFINITY, "compressor_state_ratio4: cleared score mismatch");
        } else {
            CHECK_CLOSE(got_score[i], want_score[i], 1e-3,
                        "compressor_state_ratio4: state_score mismatch");
        }
    }

    /* Validation: head_dim == 0 and an unsupported ape_type must both be
     * rejected before any kernel launches. */
    CHECK(ds4_gpu_compressor_prefill_state_ratio4_tensor(
              tskv, tssc, tkv_tail, tsc_tail, model, model_size,
              ape_offset, /*ape_type=*/0u, /*head_dim=*/0u, POS0) == 0,
          "compressor_state_ratio4: head_dim == 0 must be rejected");
    CHECK(ds4_gpu_compressor_prefill_state_ratio4_tensor(
              tskv, tssc, tkv_tail, tsc_tail, model, model_size,
              ape_offset, /*ape_type=*/2u, HEAD_DIM, POS0) == 0,
          "compressor_state_ratio4: unsupported ape_type must be rejected");

    ds4_gpu_tensor_free(tkv_tail); ds4_gpu_tensor_free(tsc_tail);
    ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc);
    free(model);

    fprintf(stderr, "  test_compressor_prefill_state_ratio4 OK\n");
    return 0;
}

/* ds4_gpu_compressor_prefill_ratio4_replay_tensor has NO CPU reference:
 * ds4.c has no "replay" function at all, so unlike every other entry point
 * in this file there is no oracle with matching semantics to port.  This
 * test is validated against a state-continuation scenario it constructs
 * itself instead: it runs the SAME per-token store/pool/shift driver used
 * by run_compressor_prefill_case's oracle above, continuously, across the
 * boundary the replay call is supposed to pick up at.  This is weaker
 * evidence than the other four entries here get, and a future
 * reader should treat it that way rather than assume CPU/GPU parity has
 * been shown here the way it has for store, pool, update, and prefill.
 *
 * Positions 0 .. POS0 - 1 (two ratio-4 windows) are run first with a fresh
 * ring, exactly as a real caller's prior prefill/replay call would have
 * left it; the ring is snapshotted right after that (i.e. after the store,
 * pool, and post-emission shift for position POS0 - 1 all complete) and
 * that snapshot becomes the INCOMING state the replay call under test is
 * given.  The driver then keeps running for positions POS0 ..
 * POS0 + N_TOKENS - 1 without resetting anything, which is what makes
 * window c == 0's candidates come from the ring (matching the replay
 * flag's effect on compressor_prefill_pool_kernel,
 * rocm/ds4_rocm_compressor.cuh:54-119) while later windows' candidates are
 * mathematically identical whether read from the ring (as the driver does)
 * or straight from kv/sc (as the real batched kernel does for c > 0):
 * oracle_compressor_store copies values verbatim and adds APE with the same
 * arithmetic either way.  Finally oracle_compressor_finish_prefill clears
 * rows 4..7 exactly as replay's own reset-then-reseed step does, since
 * n_tokens here is a multiple of 4 (rem == 0) regardless of pos0.
 *
 * N_TOKENS = 12 gives n_comp = 3 (n_comp == 1 would only exercise the
 * c == 0 branch), and POS0 = 8 is a non-zero multiple of 4, both chosen
 * deliberately. */
static int test_compressor_prefill_ratio4_replay(void) {
    enum {
        HEAD_DIM = 4, RATIO = 4, WIDTH = 2 * HEAD_DIM, STATE_ROWS = 2 * RATIO,
        POS0 = 8, N_TOKENS = 12, N_COMP = N_TOKENS / RATIO, N_ROT = 2,
        TOTAL = POS0 + N_TOKENS
    };
    const float RMS_EPS = 1e-5f;
    const uint32_t N_CTX_ORIG = 4096u;
    const float FREQ_BASE = 10000.0f, FREQ_SCALE = 1.0f, EXT_FACTOR = 0.0f;
    const float ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

    /* kv includes a (t * j) % 7 interaction term, unlike the plain affine
     * ramp run_compressor_prefill_case above uses: a purely affine kv makes
     * every candidate row differ from every other only by a per-row
     * additive constant, which RMS-normalisation almost entirely divides
     * back out regardless of which candidate the softmax picks.  That
     * degeneracy was verified empirically -- with the plain affine ramp,
     * temporarily inverting this test's step order (see the master
     * verification for this task) still passed, because the wrong
     * candidate normalised to nearly the same vector as the right one. The
     * interaction term breaks that: different dominant candidates now
     * normalise to visibly different shapes. */
    float kv[TOTAL * WIDTH], sc[TOTAL * WIDTH], norm_w[HEAD_DIM];
    for (uint32_t t = 0; t < TOTAL; t++) {
        for (uint32_t j = 0; j < WIDTH; j++) {
            kv[t * WIDTH + j] = (float)(t * 37u + j * 11u + 5u) * 0.25f +
                                 (float)((t * j) % 7u) * 2.0f;
            sc[t * WIDTH + j] = (float)((int)(t * 19u) - (int)(j * 3u) - 7);
        }
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) norm_w[d] = (float)(d % 5u + 1u) * 0.2f;

    const uint64_t ape_offset = 16;
    const uint64_t norm_offset = ape_offset + (uint64_t)WIDTH * RATIO * sizeof(float);
    const uint64_t model_size = norm_offset + (uint64_t)HEAD_DIM * sizeof(float);
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    CHECK(model != NULL, "compressor_replay: model allocation failed");
    oracle_fill_ape(model + ape_offset, /*ape_type=*/0u, WIDTH, RATIO);
    memcpy(model + norm_offset, norm_w, (size_t)HEAD_DIM * sizeof(float));

    float state_kv[STATE_ROWS * WIDTH], state_score[STATE_ROWS * WIDTH];
    for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
        state_kv[i] = 0.0f;
        state_score[i] = -INFINITY;
    }

    float state_kv_in[STATE_ROWS * WIDTH], state_score_in[STATE_ROWS * WIDTH];
    float comp_want[N_COMP * HEAD_DIM];
    uint32_t emitted = 0;

    for (uint32_t pos = 0; pos < TOTAL; pos++) {
        oracle_compressor_store(state_kv, state_score, kv + (uint64_t)pos * WIDTH,
                                sc + (uint64_t)pos * WIDTH, model + ape_offset,
                                /*ape_type=*/0u, WIDTH, RATIO, pos, 1u);
        if (((pos + 1u) % RATIO) == 0u) {
            float pool_want[HEAD_DIM], normed_want[HEAD_DIM];
            oracle_compressor_pool(pool_want, state_kv, state_score, HEAD_DIM, RATIO);
            oracle_rms_norm_weight(normed_want, pool_want, norm_w, HEAD_DIM, RMS_EPS);
            const uint32_t comp_pos = pos + 1u - RATIO;
            oracle_rope_tail_row(normed_want, HEAD_DIM, N_ROT, comp_pos, N_CTX_ORIG,
                                 FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                                 BETA_FAST, BETA_SLOW, /*inverse=*/0);
            if (pos >= POS0) {
                memcpy(comp_want + (uint64_t)emitted * HEAD_DIM, normed_want,
                       (size_t)HEAD_DIM * sizeof(float));
                emitted++;
            }

            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)r * WIDTH], &state_kv[(uint64_t)(4u + r) * WIDTH],
                       (size_t)WIDTH * sizeof(float));
                memcpy(&state_score[(uint64_t)r * WIDTH], &state_score[(uint64_t)(4u + r) * WIDTH],
                       (size_t)WIDTH * sizeof(float));
            }
            for (uint32_t r = 0; r < 4u; r++) {
                memcpy(&state_kv[(uint64_t)(4u + r) * WIDTH], &state_kv[(uint64_t)r * WIDTH],
                       (size_t)WIDTH * sizeof(float));
                memcpy(&state_score[(uint64_t)(4u + r) * WIDTH], &state_score[(uint64_t)r * WIDTH],
                       (size_t)WIDTH * sizeof(float));
            }
        }
        if (pos == POS0 - 1u) {
            memcpy(state_kv_in, state_kv, sizeof(state_kv));
            memcpy(state_score_in, state_score, sizeof(state_score));
        }
    }
    CHECK(emitted == N_COMP, "compressor_replay: oracle emit count mismatch");

    /* Replay's own reset-then-reseed step clears whatever its tail window
     * never wrote, exactly as the batched prefill entry's does; n_tokens
     * here is this call's own N_TOKENS, not the driver's TOTAL. */
    oracle_compressor_finish_prefill(state_kv, state_score, HEAD_DIM, RATIO, N_TOKENS);

    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
    ds4_gpu_tensor *tsc = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * WIDTH * sizeof(float));
    ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *tssc = ds4_gpu_tensor_alloc((uint64_t)STATE_ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc((uint64_t)N_COMP * HEAD_DIM * sizeof(float));
    CHECK(tkv && tsc && tskv && tssc && tcomp, "compressor_replay: device allocation failed");

    CHECK(ds4_gpu_tensor_write(tkv, 0, kv + (uint64_t)POS0 * WIDTH,
                               (uint64_t)N_TOKENS * WIDTH * sizeof(float)) != 0,
          "compressor_replay: write kv");
    CHECK(ds4_gpu_tensor_write(tsc, 0, sc + (uint64_t)POS0 * WIDTH,
                               (uint64_t)N_TOKENS * WIDTH * sizeof(float)) != 0,
          "compressor_replay: write sc");
    CHECK(ds4_gpu_tensor_write(tskv, 0, state_kv_in, sizeof(state_kv_in)) != 0,
          "compressor_replay: write incoming state_kv");
    CHECK(ds4_gpu_tensor_write(tssc, 0, state_score_in, sizeof(state_score_in)) != 0,
          "compressor_replay: write incoming state_score");
    {
        float sentinel_comp[N_COMP * HEAD_DIM];
        for (uint32_t i = 0; i < N_COMP * HEAD_DIM; i++) sentinel_comp[i] = -999.0f;
        CHECK(ds4_gpu_tensor_write(tcomp, 0, sentinel_comp, sizeof(sentinel_comp)) != 0,
              "compressor_replay: seed comp sentinel");
    }

    CHECK(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
              tcomp, tskv, tssc, tkv, tsc,
              model, model_size, ape_offset, /*ape_type=*/0u,
              norm_offset, /*norm_type=*/0u, HEAD_DIM,
              POS0, N_TOKENS, N_ROT, N_CTX_ORIG,
              /*quantize_fp8=*/false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) != 0,
          "compressor_replay: call");

    float got_comp[N_COMP * HEAD_DIM];
    float got_kv[STATE_ROWS * WIDTH], got_score[STATE_ROWS * WIDTH];
    CHECK(ds4_gpu_tensor_read(tcomp, 0, got_comp, sizeof(got_comp)) != 0,
          "compressor_replay: read comp");
    CHECK(ds4_gpu_tensor_read(tskv, 0, got_kv, sizeof(got_kv)) != 0,
          "compressor_replay: read state_kv");
    CHECK(ds4_gpu_tensor_read(tssc, 0, got_score, sizeof(got_score)) != 0,
          "compressor_replay: read state_score");

    for (uint32_t i = 0; i < N_COMP * HEAD_DIM; i++) {
        CHECK_CLOSE(got_comp[i], comp_want[i], 1e-2, "compressor_replay: emitted row mismatch");
    }
    /* The state ring matters as much as the emission here: replay's whole
     * purpose is leaving it correctly seeded for the next call, so a
     * correct emission with a wrongly seeded ring must still fail. */
    for (uint32_t i = 0; i < STATE_ROWS * WIDTH; i++) {
        if (state_score[i] == -INFINITY) {
            CHECK(got_score[i] == -INFINITY, "compressor_replay: cleared score row mismatch");
            CHECK(got_kv[i] == 0.0f, "compressor_replay: cleared kv row mismatch");
        } else {
            CHECK(got_kv[i] == state_kv[i], "compressor_replay: final ring kv mismatch");
            CHECK_CLOSE(got_score[i], state_score[i], 1e-3, "compressor_replay: final ring score mismatch");
        }
    }

    /* Validation: both n_tokens and pos0 must be multiples of 4, and
     * n_tokens must be non-zero, unlike the general-ratio prefill entry. */
    CHECK(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
              tcomp, tskv, tssc, tkv, tsc, model, model_size,
              ape_offset, 0u, norm_offset, 0u, HEAD_DIM,
              POS0, /*n_tokens=*/0u, N_ROT, N_CTX_ORIG,
              false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) == 0,
          "compressor_replay: n_tokens == 0 must be rejected");
    CHECK(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
              tcomp, tskv, tssc, tkv, tsc, model, model_size,
              ape_offset, 0u, norm_offset, 0u, HEAD_DIM,
              POS0, /*n_tokens=*/N_TOKENS + 1u, N_ROT, N_CTX_ORIG,
              false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) == 0,
          "compressor_replay: n_tokens not a multiple of 4 must be rejected");
    CHECK(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
              tcomp, tskv, tssc, tkv, tsc, model, model_size,
              ape_offset, 0u, norm_offset, 0u, HEAD_DIM,
              /*pos0=*/POS0 + 1u, N_TOKENS, N_ROT, N_CTX_ORIG,
              false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
              ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) == 0,
          "compressor_replay: pos0 not a multiple of 4 must be rejected");

    ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc);
    ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
    free(model);

    fprintf(stderr, "  test_compressor_prefill_ratio4_replay OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_compressor_store() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_pool() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_update() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_prefill() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_prefill_state_ratio4() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_compressor_prefill_ratio4_replay() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_compressor OK\n");
    return 0;
}
