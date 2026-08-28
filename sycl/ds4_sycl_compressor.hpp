#pragma once

/* DeepSeek streaming KV/score compressor: store step.
 *
 * Ported from rocm/ds4_rocm_compressor.cuh, which is the authority for all
 * semantics here.  Every `ratio` consecutive raw KV/score rows are pooled
 * into one compressed row by a per-dimension softmax, with an additive
 * positional embedding (APE) added to the scores first.  This file
 * implements only the STORE step: filing raw rows into a ring buffer with
 * their APE applied.  The pooling softmax is a later kernel.
 *
 * compressor_store_kernel (rocm/ds4_rocm_compressor.cuh:1-25) is a flat 1D
 * grid over n_tokens * width with no reduction and no shared/local memory:
 * a pure scatter.  This subsystem does not use a tree-reduction
 * shape; do not import it here.
 *
 * Entry is NONZERO-means-success, verified at the launcher
 * (rocm/ds4_rocm_compressor.cuh:191-237): validation failure returns 0,
 * and n_tokens == 0 is itself a validation failure here (unlike the
 * zero-length-is-success convention used by most other entries in this
 * backend), so there is no separate "n == 0 returns 1" branch. */

#include "ds4_sycl_common.hpp"

namespace {

/* Matches rocm/ds4_rocm_runtime.cuh:34 (DS4_ROCM_COMPRESSOR_MAX_RATIO).
 * The store kernel does not itself need a ratio-sized private array,
 * but the shape validator it shares with later compressor entries does. */
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
