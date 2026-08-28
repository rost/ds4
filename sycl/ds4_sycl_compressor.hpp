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
 * The Q8_0 branch uses the same 32-value, 34-byte block layout (2-byte
 * little-endian F16 scale then 32 int8) used by the embedding kernels in
 * sycl/ds4_sycl_embedding.hpp. */
static inline float sycl_ape_value_dev(const unsigned char *base, uint32_t type,
                                       uint32_t width, uint32_t row, uint32_t col) {
    if (type == 1u) {
        const uint16_t *p = (const uint16_t *)base;
        return (float)sycl::bit_cast<sycl::half>(p[(uint64_t)row * width + col]);
    }
    if (type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *blk =
                base + (uint64_t)row * row_bytes + (uint64_t)(col >> 5) * 34u;
        const uint16_t raw = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const float scale = (float)sycl::bit_cast<sycl::half>(raw);
        const signed char q = (signed char)blk[2u + (col & 31u)];
        return scale * (float)q;
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
        unsigned char *dape = sycl::malloc_device<unsigned char>((size_t)ape_bytes, q);
        if (!dape) return 0;
        sycl_device_scratch_guard dape_guard(q, dape);
        q.memcpy(dape, ape, (size_t)ape_bytes).wait_and_throw();

        const float *pkv = (const float *)kv->ptr;
        const float *psc = (const float *)sc->ptr;
        float       *skv = (float *)state_kv->ptr;
        float       *ssc = (float *)state_score->ptr;
        const uint32_t w    = width;
        const uint32_t r    = ratio;
        const uint32_t p0   = pos0;
        const uint32_t type = ape_type;

        q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid) {
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

        q.parallel_for(sycl::range<1>((size_t)hd), [=](sycl::id<1> id) {
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

            q.parallel_for(sycl::range<1>((size_t)half), [=](sycl::id<1> id) {
                const uint64_t i = id[0];
                const float v = skv[half + i];
                const float s = ssc[half + i];
                skv[i] = v;
                ssc[i] = s;
                skv[half + i] = v;
                ssc[half + i] = s;
            });
            q.wait_and_throw();
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "compressor ratio4 shift launch failed: %s\n", e.what());
            ok = 0;
        }
    }

    return ok;
}
