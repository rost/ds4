#pragma once

/* DeepSeek streaming KV/score compressor: store step and the single-token
 * decode update built on it.
 *
 * Ported from rocm/ds4_rocm_compressor.cuh, which is the authority for all
 * semantics here.  Every `ratio` consecutive raw KV/score rows are pooled
 * into one compressed row by a per-dimension softmax, with an additive
 * positional embedding (APE) added to the scores first.  This file
 * implements the STORE step (filing raw rows into a ring buffer with their
 * APE applied) and the decode UPDATE step (store one token, then on a
 * ratio boundary pool the ring, RMS-normalise and RoPE-rotate the result,
 * and for ratio == 4 shift the ring).  The prefill pooling kernel and its
 * ratio-4 replay/state variants are future work.
 *
 * compressor_store_kernel (rocm/ds4_rocm_compressor.cuh:1-25) and
 * compressor_update_pool_kernel (rocm/ds4_rocm_compressor.cuh:121-160) are
 * both flat 1D grids (over n_tokens * width and over head_dim
 * respectively) with no reduction and no shared/local memory: a pure
 * scatter and a per-work-item private-array softmax.  This subsystem does
 * not use ds4_sycl_norm_rope.hpp's tree-reduction shape; do not import it here.
 *
 * Entry is NONZERO-means-success, verified at the launchers
 * (rocm/ds4_rocm_compressor.cuh:191-237 for store, 239-327 for update):
 * validation failure returns 0, and n_tokens == 0 is itself a validation
 * failure for the store entry (unlike the zero-length-is-success
 * convention used by most other entries in this backend), so there is no
 * separate "n == 0 returns 1" branch there.  The update entry has no
 * n_tokens parameter (it is always exactly one token); its own
 * zero-work-shaped case is a non-emit position, which returns 1 after the
 * store completes, without touching comp_cache at all. */

#include "ds4_sycl_common.hpp"

#include <cmath>

/* Forward declaration of sycl/ds4_sycl_norm_rope.hpp's stride-aware RoPE
 * launcher.  It keeps internal linkage (`static`), so this declaration must
 * repeat that, but C++ permits declare-then-define for an internal-linkage
 * function within one translation unit: ds4_sycl.cpp includes
 * ds4_sycl_norm_rope.hpp before this header today, which would make the
 * declaration merely redundant, but a forward declaration removes the
 * dependency on that order rather than documenting it, and a missing
 * definition at link time fails loudly instead of silently compiling
 * against the wrong overload.  Deliberately at FILE scope, matching the
 * real definition: putting it inside the anonymous namespace below would
 * declare a second, distinct entity with the same name and signature,
 * which is ambiguous at the call site once both are visible. */
static int sycl_rope_tail_stride_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t pos_stride,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow);

namespace {

/* Matches rocm/ds4_rocm_runtime.cuh:34 (DS4_ROCM_COMPRESSOR_MAX_RATIO).
 * Sizes the private vals[]/scores[] arrays compressor_update_pool_kernel's
 * SYCL port gathers into below, and the shape validator shared by every
 * compressor entry. */
constexpr uint32_t kCompressorMaxRatio = 128u;

static inline int sycl_ape_type_supported(uint32_t type) {
    return type == 0u || type == 1u || type == 8u;
}

static inline int sycl_compressor_shape_supported(uint32_t head_dim, uint32_t ratio) {
    if (head_dim == 0u || ratio == 0u || ratio > kCompressorMaxRatio) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    return head_dim <= UINT32_MAX / coff;
}

/* Matches cuda_tensor_2d_bytes (rocm/ds4_rocm_compressor.cuh:174-179): the
 * byte size of a `rows`-row APE table `width` elements wide, under the
 * element type given by `type`. */
static inline uint64_t sycl_ape_2d_bytes(uint32_t type, uint64_t width, uint64_t rows) {
    if (type == 0u) return width * rows * sizeof(float);
    if (type == 1u) return width * rows * sizeof(uint16_t);
    if (type == 8u) return rows * (((width + 31u) / 32u) * 34u);
    return 0;
}

/* Device-side APE lookup, matching model_ape_value_dev
 * (rocm/ds4_rocm_norm_rope.cuh:366-378).  `base` is staged device scratch
 * holding exactly the ape_type's encoding of a `width`-wide, `ratio`-row
 * table; `row`/`col` address one element as row * width + col, same as
 * ds4's CPU-side tensor_2d_value(model, ape, col, row) (ds4.c:7882-7887).
 * The Q8_0 branch defers to the shared sycl_q8_0_row_bytes_checked and
 * sycl_q8_0_dequant helpers in ds4_sycl_common.hpp, the same ones the
 * embedding kernels in sycl/ds4_sycl_embedding.hpp use. */
static inline float sycl_ape_value_dev(const unsigned char *base, uint32_t type,
                                       uint32_t width, uint32_t row, uint32_t col) {
    if (type == 1u) {
        const uint16_t *p = (const uint16_t *)base;
        return (float)sycl::bit_cast<sycl::half>(p[(uint64_t)row * width + col]);
    }
    if (type == 8u) {
        /* width is head_dim, already bounded by
         * sycl_compressor_shape_supported above, so this cannot overflow;
         * the same reasoning sycl_ape_2d_bytes relies on. */
        uint64_t row_bytes = 0;
        (void)sycl_q8_0_row_bytes_checked(width, &row_bytes);
        return sycl_q8_0_dequant(base + (uint64_t)row * row_bytes, col);
    }
    const float *p = (const float *)base;
    return p[(uint64_t)row * width + col];
}

}  // namespace

/* Store step of the streaming compressor.  For token t and column j:
 *   pos_mod  = (pos0 + t) % ratio
 *   dst_row  = (ratio == 4) ? ratio + pos_mod : pos_mod
 *   state_kv   [dst_row][j] = kv[t][j]
 *   state_score[dst_row][j] = sc[t][j] + ape(width, pos_mod, j)
 *
 * For ratio == 4 the state ring has 2*ratio = 8 rows; the write lands in
 * the HIGH half (rows 4..7), the "current" window, while the low half
 * holds the previous one.  For every other ratio the ring has exactly
 * `ratio` rows and dst_row == pos_mod.
 *
 * This entry is not called from ds4.c: it is invoked backend-internally
 * by ds4_gpu_compressor_update_tensor with n_tokens = 1, matching
 * rocm/ds4_rocm_compressor.cuh:292-298. */
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        !sycl_compressor_shape_supported(head_dim, ratio) || n_tokens == 0u ||
        !sycl_ape_type_supported(ape_type)) {
        return 0;
    }

    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;

    uint64_t kv_bytes = 0, state_bytes = 0;
    const uint64_t ape_bytes = sycl_ape_2d_bytes(ape_type, width, ratio);
    if (!sycl_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !sycl_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !sycl_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !sycl_tensor_has_bytes(kv, kv_bytes) || !sycl_tensor_has_bytes(sc, kv_bytes) ||
        !sycl_tensor_has_bytes(state_kv, state_bytes) ||
        !sycl_tensor_has_bytes(state_score, state_bytes)) {
        return 0;
    }

    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes,
                                           model_size, "compressor_ape");
    if (!ape) return 0;
    if (g_devices.empty()) return 0;

    const uint64_t n = (uint64_t)n_tokens * width;

    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        /* The APE table lives in the host mmap.  Stage it to device
         * scratch under a guard, as ds4_gpu_rms_norm_weight_rows_tensor
         * (ds4_sycl_norm_rope.hpp) does; never a bare malloc_device/free
         * pair. */
        sycl_device_scratch_guard dape_guard = sycl_stage_host_bytes(q, ape, ape_bytes);
        if (!dape_guard.p) return 0;
        unsigned char *dape = (unsigned char *)dape_guard.p;

        const float *pkv = (const float *)kv->ptr;
        const float *psc = (const float *)sc->ptr;
        float       *skv = (float *)state_kv->ptr;
        float       *ssc = (float *)state_score->ptr;
        const uint32_t w    = width;
        const uint32_t r    = ratio;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;

        sycl::event _ds4_prof_ev19 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
            const uint32_t t = (uint32_t)(gid / w);
            const uint32_t j = (uint32_t)(gid - (uint64_t)t * w);
            const uint32_t pos_mod = (p0 + t) % r;
            const uint32_t dst_row = (r == 4u) ? r + pos_mod : pos_mod;
            skv[(uint64_t)dst_row * w + j] = pkv[(uint64_t)t * w + j];
            ssc[(uint64_t)dst_row * w + j] =
                    psc[(uint64_t)t * w + j] +
                    sycl_ape_value_dev(dape, type, w, pos_mod, j);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev19);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor_store failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Decode-time update: store the current token unless it was already
 * stored, then on a ratio boundary pool the ring, RMS-normalise the
 * pooled row with the layer's compressor norm weight, RoPE-rotate its
 * tail, and for ratio == 4 shift the ring.  Matches
 * ds4_gpu_compressor_update_tensor (rocm/ds4_rocm_compressor.cuh:239-327).
 *
 * decode_one_token and defer_finalize are intentionally unused here: CUDA
 * discards them identically (ds4_cuda.cu:16488 has the same pair of
 * (void) casts), and only Metal's implementation actually reads them (14
 * references in ds4_metal.m). Two of the three non-Metal backends no-op
 * these parameters, so SYCL follows suit rather than reverse-engineering
 * behaviour from their names. */
extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize) {
    (void)decode_one_token;
    (void)defer_finalize;

    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || !sycl_compressor_shape_supported(head_dim, ratio) ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        !sycl_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit       = ((pos + 1u) % ratio) == 0u ? 1u : 0u;

    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = sycl_ape_2d_bytes(ape_type, width, ratio);
    if (!sycl_u64_mul_checked(width, sizeof(float), &kv_bytes) ||
        !sycl_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !sycl_u64_mul3_checked((uint64_t)comp_row + (emit ? 1u : 0u), head_dim, sizeof(float), &comp_bytes) ||
        !sycl_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !sycl_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !sycl_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !sycl_tensor_has_bytes(kv_cur, kv_bytes) || !sycl_tensor_has_bytes(sc_cur, kv_bytes) ||
        !sycl_tensor_has_bytes(state_kv, state_bytes) || !sycl_tensor_has_bytes(state_score, state_bytes) ||
        (emit && !sycl_tensor_has_bytes(comp_cache, comp_bytes))) {
        return 0;
    }

    /* Store delegates to ds4_gpu_compressor_store_batch_tensor with
     * n_tokens = 1; it is NONZERO-means-success like every entry here, so
     * a 0 return fails this call too. */
    if (!state_already_stored) {
        if (ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                   model_map, model_size, ape_offset, ape_type,
                                                   head_dim, ratio, pos, 1u) == 0) {
            return 0;
        }
    }
    if (!emit) return 1;
    if (g_devices.empty()) return 0;

    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;

    /* `ok` is tracked rather than returning early on the first failed
     * sub-call, because comp_row_view is a heap-allocated non-owning view
     * that must be freed on every path; an early return here (matching
     * the general "if (sub_call(...) == 0) return 0" propagation shape
     * used above and by ds4_gpu_compressor_store_batch_tensor) would leak
     * it.  Matches ROCm's own
     * ok-tracking shape at rocm/ds4_rocm_compressor.cuh:311-322. */
    int ok = 1;
    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        float       *out = (float *)comp_row_view->ptr;
        const float *skv = (const float *)state_kv->ptr;
        const float *ssc = (const float *)state_score->ptr;
        const uint32_t hd = head_dim;
        const uint32_t r  = ratio;
        const uint32_t w  = width;

        sycl::event _ds4_prof_ev20 = q.parallel_for(sycl::range<1>((size_t)hd), [=](sycl::id<1> id) {
            const uint32_t d = (uint32_t)id[0];

            /* Private per-work-item arrays, sized to the max-ratio
             * constant: no local_accessor, no barrier, no tree reduction.
             * This subsystem's kernels are flat gathers, not
             * ds4_sycl_norm_rope.hpp's work-group-per-row shape. */
            float vals[kCompressorMaxRatio];
            float scores[kCompressorMaxRatio];
            float max_s = -INFINITY;
            uint32_t n = 0;

            if (r == 4u) {
                /* Asymmetric in BOTH row and column: the low half (ring
                 * rows 0..3) reads the LOW lane at column d, and the high
                 * half (rows 4..7) reads the HIGH lane at column
                 * head_dim + d.  This is NOT eight rows all read at
                 * column d -- that still averages eight real values and
                 * produces a smooth, plausible, wrong result. */
                for (uint32_t rr = 0; rr < 4u; rr++) {
                    vals[n] = skv[(uint64_t)rr * w + d];
                    scores[n] = ssc[(uint64_t)rr * w + d];
                    max_s = sycl::fmax(max_s, scores[n]);
                    n++;
                }
                for (uint32_t rr = 0; rr < 4u; rr++) {
                    vals[n] = skv[(uint64_t)(r + rr) * w + hd + d];
                    scores[n] = ssc[(uint64_t)(r + rr) * w + hd + d];
                    max_s = sycl::fmax(max_s, scores[n]);
                    n++;
                }
            } else {
                /* General path: `ratio` rows at r * width + d, with
                 * width == head_dim since coff is 1 here. */
                for (uint32_t rr = 0; rr < r; rr++) {
                    vals[n] = skv[(uint64_t)rr * w + d];
                    scores[n] = ssc[(uint64_t)rr * w + d];
                    max_s = sycl::fmax(max_s, scores[n]);
                    n++;
                }
            }

            float den = 0.0f, acc = 0.0f;
            /* max_s > -INFINITY guards the empty-softmax case.  Without
             * it, a ring whose candidates are all -inf (nothing stored
             * yet, at a sequence start) hits the IEEE `inf - inf`
             * indeterminate form inside the loop below and resolves to
             * NaN; with it, den stays exactly 0 and the ternary below
             * yields exactly 0, matching the required empty-softmax
             * result. */
            if (max_s > -INFINITY) {
                for (uint32_t i = 0; i < n; i++) {
                    const float wgt = sycl::exp(scores[i] - max_s);
                    den += wgt;
                    acc += vals[i] * wgt;
                }
            }
            out[d] = den != 0.0f ? acc / den : 0.0f;
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev20);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor_update pool launch failed: %s\n", e.what());
        ok = 0;
    }

    if (ok && ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, 1u, rms_eps) == 0) {
        ok = 0;
    }
    if (ok && ds4_gpu_rope_tail_tensor(comp_row_view, 1u, 1u, head_dim, n_rot,
                                       pos + 1u - ratio, n_ctx_orig, false,
                                       freq_base, freq_scale, ext_factor, attn_factor,
                                       beta_fast, beta_slow) == 0) {
        ok = 0;
    }
    ds4_gpu_tensor_free(comp_row_view);

    if (ok && ratio == 4u) {
        try {
            sycl::queue &q = ds4_sycl_queue(state_kv->device_id);
            float *skv = (float *)state_kv->ptr;
            float *ssc = (float *)state_score->ptr;
            const uint64_t half = 4ull * width;

            sycl::event _ds4_prof_ev21 = q.parallel_for(sycl::range<1>((size_t)half), [=](sycl::id<1> id) {
                const uint64_t i = id[0];
                const float v = skv[half + i];
                const float s = ssc[half + i];
                skv[i] = v;
                ssc[i] = s;
                skv[half + i] = v;
                ssc[half + i] = s;
            });
            q.wait_and_throw();
            ds4_sycl_profile_record(_ds4_prof_ev21);
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor ratio4 shift launch failed: %s\n", e.what());
            ok = 0;
        }
    }

    return ok;
}

/* Batched prefill entry: pools every complete `ratio`-token window of a
 * whole prefill batch in one call, instead of decode's one-token-at-a-time
 * ring walk above.  Matches ds4_gpu_compressor_prefill_tensor
 * (rocm/ds4_rocm_compressor.cuh:328-439); the orchestration order there is
 * load-bearing and is followed step for step:
 *
 *   1. Reset the ring: state_kv zeroed, state_score filled with -inf.  This
 *      is ASYMMETRIC by design and is what the NaN guard in the pool
 *      gather below exists to handle -- an all -inf candidate window must
 *      be excluded from the softmax, not treated as a real zero-score
 *      candidate (rocm/ds4_rocm_compressor.cuh:384-387).
 *   2. Seed the tail rows so decode resumes from the same partial-window
 *      state the streaming path would produce.  A three-way branch
 *      (rocm/ds4_rocm_compressor.cuh:389-417):
 *        - ratio == 4 and a previous window exists (cutoff >= ratio): copy
 *          that whole previous window (source cutoff - ratio, `ratio` rows)
 *          into the ring's low half.
 *        - ratio == 4 and a trailing partial window exists (rem != 0):
 *          copy it (source cutoff, `rem` rows) into the ring's high half.
 *        - otherwise, a trailing partial window (rem != 0): copy it
 *          (source cutoff, `rem` rows) into the ring from row 0.
 *   3. When there is at least one complete window (n_comp != 0), pool each
 *      one with a per-dimension softmax, then RMS-normalise and RoPE-rotate
 *      the pooled rows, then optionally FP8-quantise them.
 *
 * When n_comp == 0 the entry seeds state and returns success without
 * pooling: there is nothing yet to emit. */
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        !sycl_compressor_shape_supported(head_dim, ratio) || n_tokens == 0u ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        !sycl_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff       = ratio == 4u ? 2u : 1u;
    const uint32_t width      = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp     = n_tokens / ratio;
    const uint32_t cutoff     = n_comp * ratio;
    const uint32_t rem        = n_tokens - cutoff;

    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = sycl_ape_2d_bytes(ape_type, width, ratio);
    if (!sycl_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !sycl_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !sycl_u64_mul3_checked(n_comp, head_dim, sizeof(float), &comp_bytes) ||
        !sycl_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !sycl_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !sycl_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !sycl_tensor_has_bytes(kv, kv_bytes) || !sycl_tensor_has_bytes(sc, kv_bytes) ||
        !sycl_tensor_has_bytes(state_kv, state_bytes) || !sycl_tensor_has_bytes(state_score, state_bytes) ||
        (n_comp != 0u && !sycl_tensor_has_bytes(comp_cache, comp_bytes))) {
        return 0;
    }

    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes,
                                           model_size, "compressor_ape");
    if (!ape) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        /* The APE table lives in the host mmap; stage it to device scratch
         * under a guard, as every other entry in this file does.  Its
         * lifetime spans both the tail seeding below and the pool gather
         * further down, both of which read it. */
        sycl_device_scratch_guard dape_guard = sycl_stage_host_bytes(q, ape, ape_bytes);
        if (!dape_guard.p) return 0;
        unsigned char *dape = (unsigned char *)dape_guard.p;

        float       *skv  = (float *)state_kv->ptr;
        float       *ssc  = (float *)state_score->ptr;
        const float *pkv  = (const float *)kv->ptr;
        const float *psc  = (const float *)sc->ptr;
        const uint32_t w    = width;
        const uint32_t r    = ratio;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;

        /* Step 1: reset the ring.  Asymmetric on purpose -- see the
         * function comment above. */
        const uint64_t state_n = (uint64_t)state_rows * width;
        sycl::event _ds4_prof_ev22 = q.memset(skv, 0, (size_t)(state_n * sizeof(float)));
        _ds4_prof_ev22.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev22);
        sycl::event _ds4_prof_ev23 = q.parallel_for(sycl::range<1>((size_t)state_n), [=](sycl::id<1> gid) {
            ssc[gid[0]] = -INFINITY;
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev23);

        /* Step 2: seed the tail rows over an arbitrary contiguous source
         * range, matching compressor_set_rows_kernel
         * (rocm/ds4_rocm_compressor.cuh:27-52).  `phase` is the APE row
         * index -- the row's true position modulo ratio -- which can differ
         * from the destination row `dst0 + rr` the caller chooses. */
        auto seed_rows = [&](uint32_t src0, uint32_t dst0, uint32_t rows) {
            const uint64_t n = (uint64_t)rows * w;
            sycl::event _ds4_prof_ev24 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
                const uint32_t rr  = (uint32_t)(gid / w);
                const uint32_t j   = (uint32_t)(gid - (uint64_t)rr * w);
                const uint32_t src = src0 + rr;
                const uint32_t dst = dst0 + rr;
                const uint32_t phase = (p0 + src) % r;
                skv[(uint64_t)dst * w + j] = pkv[(uint64_t)src * w + j];
                ssc[(uint64_t)dst * w + j] =
                        psc[(uint64_t)src * w + j] +
                        sycl_ape_value_dev(dape, type, w, phase, j);
            });
            q.wait_and_throw();
            ds4_sycl_profile_record(_ds4_prof_ev24);
        };

        if (ratio == 4u) {
            if (cutoff >= ratio) {
                seed_rows(cutoff - ratio, 0u, ratio);
            }
            if (rem != 0u) {
                seed_rows(cutoff, ratio, rem);
            }
        } else if (rem != 0u) {
            seed_rows(cutoff, 0u, rem);
        }

        /* Step 3: pool every complete window, then normalise and rotate. */
        if (n_comp != 0u) {
            float *comp = (float *)comp_cache->ptr;
            const uint32_t hd = head_dim;
            const uint32_t nc = n_comp;
            const uint64_t n  = (uint64_t)hd * nc;

            sycl::event _ds4_prof_ev25 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
                const uint32_t d = (uint32_t)(gid % hd);
                const uint32_t c = (uint32_t)(gid / hd);

                /* Private per-work-item arrays, sized to the max-ratio
                 * constant: no local_accessor, no barrier, no tree
                 * reduction, matching compressor_prefill_pool_kernel
                 * (rocm/ds4_rocm_compressor.cuh:54-119). */
                float vals[kCompressorMaxRatio];
                float scores[kCompressorMaxRatio];
                float max_s = -INFINITY;
                uint32_t n_cand = 0;

                if (r == 4u) {
                    /* Previous window (low lane, column d), only once one
                     * exists.  With no previous window (c == 0) these
                     * candidates are simply absent rather than gathered as
                     * -inf: mathematically identical, since an all -inf
                     * half would contribute exactly zero softmax weight
                     * anyway, but it avoids reading state_kv/state_score
                     * here at all -- this kernel pools straight from the
                     * raw kv/sc batch, not from the ring. */
                    if (c > 0u) {
                        const uint32_t base = (c - 1u) * r;
                        for (uint32_t rr = 0; rr < 4u; rr++) {
                            const uint32_t t = base + rr;
                            const uint32_t phase = (p0 + t) % r;
                            const float ape_v = sycl_ape_value_dev(dape, type, w, phase, d);
                            vals[n_cand] = pkv[(uint64_t)t * w + d];
                            scores[n_cand] = psc[(uint64_t)t * w + d] + ape_v;
                            max_s = sycl::fmax(max_s, scores[n_cand]);
                            n_cand++;
                        }
                    }
                    /* Current window (high lane, column head_dim + d). */
                    const uint32_t base = c * r;
                    for (uint32_t rr = 0; rr < 4u; rr++) {
                        const uint32_t t = base + rr;
                        const uint32_t phase = (p0 + t) % r;
                        const float ape_v = sycl_ape_value_dev(dape, type, w, phase, hd + d);
                        vals[n_cand] = pkv[(uint64_t)t * w + hd + d];
                        scores[n_cand] = psc[(uint64_t)t * w + hd + d] + ape_v;
                        max_s = sycl::fmax(max_s, scores[n_cand]);
                        n_cand++;
                    }
                } else {
                    const uint32_t base = c * r;
                    for (uint32_t rr = 0; rr < r; rr++) {
                        const uint32_t t = base + rr;
                        const uint32_t phase = (p0 + t) % r;
                        const float ape_v = sycl_ape_value_dev(dape, type, w, phase, d);
                        vals[n_cand] = pkv[(uint64_t)t * w + d];
                        scores[n_cand] = psc[(uint64_t)t * w + d] + ape_v;
                        max_s = sycl::fmax(max_s, scores[n_cand]);
                        n_cand++;
                    }
                }

                float den = 0.0f, acc = 0.0f;
                /* Same NaN guard as the decode-time pool above: without it,
                 * an all -inf candidate set hits `inf - inf` and resolves
                 * to NaN instead of the required exact 0. */
                if (max_s > -INFINITY) {
                    for (uint32_t i = 0; i < n_cand; i++) {
                        const float wgt = sycl::exp(scores[i] - max_s);
                        den += wgt;
                        acc += vals[i] * wgt;
                    }
                }
                comp[(uint64_t)c * hd + d] = den != 0.0f ? acc / den : 0.0f;
            });
            q.wait_and_throw();
            ds4_sycl_profile_record(_ds4_prof_ev25);
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor_prefill failed: %s\n", e.what());
        return 0;
    }

    if (n_comp == 0u) return 1;

    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map,
                                             model_size, norm_offset, head_dim,
                                             n_comp, rms_eps)) {
        return 0;
    }
    /* pos_stride = ratio, NOT 1: compressed rows sit `ratio` tokens apart in
     * position space, so row c's angle must use pos0 + c * ratio.  This is
     * exactly why the stride variant stays separate from the public
     * ds4_gpu_rope_tail_tensor wrapper (pos_stride == 1) -- decode's
     * one-row-at-a-time update above never needed a stride other than 1,
     * but this batched entry does. */
    if (n_rot != 0u && !sycl_rope_tail_stride_tensor(comp_cache, n_comp, 1u, head_dim,
                                                      n_rot, pos0, ratio, n_ctx_orig, false,
                                                      freq_base, freq_scale, ext_factor,
                                                      attn_factor, beta_fast, beta_slow)) {
        return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) {
        return 0;
    }
    return 1;
}

/* Ratio-4 replay: pools a batch of complete ratio-4 windows against a state
 * ring the caller already primed, then resets and reseeds that ring for the
 * next call.  Matches ds4_gpu_compressor_prefill_ratio4_replay_tensor
 * (rocm/ds4_rocm_compressor.cuh:440-522).  Ratio is fixed at 4 -- there is
 * no ratio parameter -- and both n_tokens and pos0 must be multiples of 4:
 * every window is complete, so n_comp = n_tokens / 4 exactly, with no
 * partial-window handling at all.
 *
 * THE ORCHESTRATION ORDER HERE IS INVERTED RELATIVE TO
 * ds4_gpu_compressor_prefill_tensor ABOVE:
 *
 *   1. Pool every window FIRST, reading the incoming state ring as it
 *      stands.  The replay flag on the pool gather (rocm/ds4_rocm_compressor.
 *      cuh:54-119) selects, for window c == 0 only, lane 0's first four
 *      candidates from the caller's state_kv/state_score rather than from
 *      freshly computed kv/sc; every other window's candidates, and c == 0's
 *      other four candidates, always come from kv/sc.  Then RMS-normalise,
 *      RoPE-rotate, and optionally FP8-quantise the pooled rows.
 *   2. Only THEN reset the ring (state_kv zeroed, state_score -inf) and
 *      reseed it from this call's own trailing ratio-4 window
 *      (prev_start = n_tokens - ratio), so the ring is ready for the next
 *      replay call.
 *
 * Seeding first, mirroring the prefill entry's order, would destroy the
 * very state the pool step reads for c == 0: the result would still look
 * numerically plausible, because the freshly seeded rows hold real data,
 * which is exactly why this must not be reordered to match prefill. */
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0u || n_tokens == 0u || (n_tokens & 3u) != 0u || (pos0 & 3u) != 0u ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        !sycl_ape_type_supported(ape_type) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio      = 4u;
    const uint32_t width      = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp     = n_tokens / ratio;

    uint64_t kv_bytes = 0, state_bytes = 0, comp_bytes = 0, norm_bytes = 0;
    const uint64_t ape_bytes = sycl_ape_2d_bytes(ape_type, width, ratio);
    if (!sycl_u64_mul3_checked(n_tokens, width, sizeof(float), &kv_bytes) ||
        !sycl_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !sycl_u64_mul3_checked(n_comp, head_dim, sizeof(float), &comp_bytes) ||
        !sycl_u64_mul_checked(head_dim, sizeof(float), &norm_bytes) ||
        !sycl_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !sycl_model_range_fits(model_size, norm_offset, norm_bytes) ||
        !sycl_tensor_has_bytes(kv, kv_bytes) || !sycl_tensor_has_bytes(sc, kv_bytes) ||
        !sycl_tensor_has_bytes(state_kv, state_bytes) || !sycl_tensor_has_bytes(state_score, state_bytes) ||
        !sycl_tensor_has_bytes(comp_cache, comp_bytes)) {
        return 0;
    }

    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes,
                                           model_size, "compressor_ape");
    if (!ape) return 0;
    if (g_devices.empty()) return 0;

    /* Step 1: pool every window against the ring as the caller passed it
     * in, before anything here touches that ring. */
    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        sycl_device_scratch_guard dape_guard = sycl_stage_host_bytes(q, ape, ape_bytes);
        if (!dape_guard.p) return 0;
        unsigned char *dape = (unsigned char *)dape_guard.p;

        const float *skv  = (const float *)state_kv->ptr;
        const float *ssc  = (const float *)state_score->ptr;
        const float *pkv  = (const float *)kv->ptr;
        const float *psc  = (const float *)sc->ptr;
        float       *comp = (float *)comp_cache->ptr;
        const uint32_t hd   = head_dim;
        const uint32_t w    = width;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;
        const uint64_t n    = (uint64_t)hd * n_comp;

        sycl::event _ds4_prof_ev26 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
            const uint32_t d = (uint32_t)(gid % hd);
            const uint32_t c = (uint32_t)(gid / hd);

            /* Private per-work-item arrays, sized to the max-ratio
             * constant: no local_accessor, no barrier, no tree reduction,
             * matching compressor_prefill_pool_kernel
             * (rocm/ds4_rocm_compressor.cuh:54-119). */
            float vals[kCompressorMaxRatio];
            float scores[kCompressorMaxRatio];
            float max_s = -INFINITY;
            uint32_t n_cand = 0;

            if (c == 0u) {
                /* Replay's defining read: the incoming ring, low lane,
                 * column d -- NOT kv/sc, and NOT the high lane. */
                for (uint32_t rr = 0; rr < 4u; rr++) {
                    vals[n_cand] = skv[(uint64_t)rr * w + d];
                    scores[n_cand] = ssc[(uint64_t)rr * w + d];
                    max_s = sycl::fmax(max_s, scores[n_cand]);
                    n_cand++;
                }
            } else {
                const uint32_t base = (c - 1u) * 4u;
                for (uint32_t rr = 0; rr < 4u; rr++) {
                    const uint32_t t = base + rr;
                    const uint32_t phase = (p0 + t) % 4u;
                    const float ape_v = sycl_ape_value_dev(dape, type, w, phase, d);
                    vals[n_cand] = pkv[(uint64_t)t * w + d];
                    scores[n_cand] = psc[(uint64_t)t * w + d] + ape_v;
                    max_s = sycl::fmax(max_s, scores[n_cand]);
                    n_cand++;
                }
            }
            /* Current window, high lane, column head_dim + d: always from
             * kv/sc, regardless of c. */
            const uint32_t base = c * 4u;
            for (uint32_t rr = 0; rr < 4u; rr++) {
                const uint32_t t = base + rr;
                const uint32_t phase = (p0 + t) % 4u;
                const float ape_v = sycl_ape_value_dev(dape, type, w, phase, hd + d);
                vals[n_cand] = pkv[(uint64_t)t * w + hd + d];
                scores[n_cand] = psc[(uint64_t)t * w + hd + d] + ape_v;
                max_s = sycl::fmax(max_s, scores[n_cand]);
                n_cand++;
            }

            float den = 0.0f, acc = 0.0f;
            /* Same NaN guard as every other pool gather in this file: an
             * all -inf candidate set must resolve to exactly 0, not NaN. */
            if (max_s > -INFINITY) {
                for (uint32_t i = 0; i < n_cand; i++) {
                    const float wgt = sycl::exp(scores[i] - max_s);
                    den += wgt;
                    acc += vals[i] * wgt;
                }
            }
            comp[(uint64_t)c * hd + d] = den != 0.0f ? acc / den : 0.0f;
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev26);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor_replay pool launch failed: %s\n", e.what());
        return 0;
    }

    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map,
                                             model_size, norm_offset, head_dim,
                                             n_comp, rms_eps)) {
        return 0;
    }
    /* pos_stride = ratio, matching the batched prefill entry above: row c's
     * angle is pos0 + c * ratio. */
    if (n_rot != 0u && !sycl_rope_tail_stride_tensor(comp_cache, n_comp, 1u, head_dim,
                                                      n_rot, pos0, ratio, n_ctx_orig, false,
                                                      freq_base, freq_scale, ext_factor,
                                                      attn_factor, beta_fast, beta_slow)) {
        return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) {
        return 0;
    }

    /* Step 2: only now reset the ring and reseed it from this batch's own
     * trailing ratio-4 window, readying it for the next replay call. */
    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        sycl_device_scratch_guard dape_guard = sycl_stage_host_bytes(q, ape, ape_bytes);
        if (!dape_guard.p) return 0;
        unsigned char *dape = (unsigned char *)dape_guard.p;

        float       *skv = (float *)state_kv->ptr;
        float       *ssc = (float *)state_score->ptr;
        const float *pkv = (const float *)kv->ptr;
        const float *psc = (const float *)sc->ptr;
        const uint32_t w    = width;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;

        const uint64_t state_n = (uint64_t)state_rows * width;
        sycl::event _ds4_prof_ev27 = q.memset(skv, 0, (size_t)(state_n * sizeof(float)));
        _ds4_prof_ev27.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev27);
        sycl::event _ds4_prof_ev28 = q.parallel_for(sycl::range<1>((size_t)state_n), [=](sycl::id<1> gid) {
            ssc[gid[0]] = -INFINITY;
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev28);

        const uint32_t prev_start = n_tokens - ratio;
        const uint64_t n = (uint64_t)ratio * w;
        sycl::event _ds4_prof_ev29 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
            const uint32_t rr  = (uint32_t)(gid / w);
            const uint32_t j   = (uint32_t)(gid - (uint64_t)rr * w);
            const uint32_t src = prev_start + rr;
            const uint32_t phase = (p0 + src) % 4u;
            skv[(uint64_t)rr * w + j] = pkv[(uint64_t)src * w + j];
            ssc[(uint64_t)rr * w + j] =
                    psc[(uint64_t)src * w + j] +
                    sycl_ape_value_dev(dape, type, w, phase, j);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev29);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor_replay state seed failed: %s\n", e.what());
        return 0;
    }

    return 1;
}

/* Ratio-4 state seed: the tail-seeding half of step 2 above, exposed
 * standalone so a caller can prime a fresh ring from the last four raw
 * tokens of a window without a full replay call.  Matches
 * ds4_gpu_compressor_prefill_state_ratio4_tensor
 * (rocm/ds4_rocm_compressor.cuh:523 onward): reset the whole 8-row ring,
 * then seed rows 0..3 from kv_tail/sc_tail (always exactly ratio = 4 rows,
 * hence no n_tokens parameter here), with each row's true position modulo
 * 4 as its APE phase.  There is no partial-window handling and no pos0
 * multiple-of-4 requirement: the tail is always a complete window by
 * construction. */
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0u || !sycl_ape_type_supported(ape_type)) {
        return 0;
    }

    const uint32_t ratio      = 4u;
    const uint32_t width      = 2u * head_dim;
    const uint32_t state_rows = 8u;

    uint64_t tail_bytes = 0, state_bytes = 0;
    const uint64_t ape_bytes = sycl_ape_2d_bytes(ape_type, width, ratio);
    if (!sycl_u64_mul3_checked(ratio, width, sizeof(float), &tail_bytes) ||
        !sycl_u64_mul3_checked(state_rows, width, sizeof(float), &state_bytes) ||
        !sycl_model_range_fits(model_size, ape_offset, ape_bytes) ||
        !sycl_tensor_has_bytes(kv_tail, tail_bytes) || !sycl_tensor_has_bytes(sc_tail, tail_bytes) ||
        !sycl_tensor_has_bytes(state_kv, state_bytes) || !sycl_tensor_has_bytes(state_score, state_bytes)) {
        return 0;
    }

    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes,
                                           model_size, "compressor_ape");
    if (!ape) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(state_kv->device_id);

        sycl_device_scratch_guard dape_guard = sycl_stage_host_bytes(q, ape, ape_bytes);
        if (!dape_guard.p) return 0;
        unsigned char *dape = (unsigned char *)dape_guard.p;

        float       *skv = (float *)state_kv->ptr;
        float       *ssc = (float *)state_score->ptr;
        const float *pkv = (const float *)kv_tail->ptr;
        const float *psc = (const float *)sc_tail->ptr;
        const uint32_t w    = width;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;

        const uint64_t state_n = (uint64_t)state_rows * width;
        sycl::event _ds4_prof_ev30 = q.memset(skv, 0, (size_t)(state_n * sizeof(float)));
        _ds4_prof_ev30.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev30);
        sycl::event _ds4_prof_ev31 = q.parallel_for(sycl::range<1>((size_t)state_n), [=](sycl::id<1> gid) {
            ssc[gid[0]] = -INFINITY;
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev31);

        const uint64_t n = (uint64_t)ratio * w;
        sycl::event _ds4_prof_ev32 = q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
            const uint32_t rr = (uint32_t)(gid / w);
            const uint32_t j  = (uint32_t)(gid - (uint64_t)rr * w);
            const uint32_t phase = (p0 + rr) % 4u;
            skv[(uint64_t)rr * w + j] = pkv[(uint64_t)rr * w + j];
            ssc[(uint64_t)rr * w + j] =
                    psc[(uint64_t)rr * w + j] +
                    sycl_ape_value_dev(dape, type, w, phase, j);
        });
        q.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev32);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor state seed failed: %s\n", e.what());
        return 0;
    }

    return 1;
}
