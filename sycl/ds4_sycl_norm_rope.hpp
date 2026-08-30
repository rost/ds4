#pragma once

/* RMS normalisation and DeepSeek partial RoPE.
 *
 * Ported from rocm/ds4_rocm_norm_rope.cuh, which is the authority for all
 * semantics here.  The entry points in this file share one reduction
 * shape: one work-group per row, a FIXED 256 work-items, a strided
 * accumulate into local memory, a power-of-two tree reduction, then a
 * strided scale pass.  The QKV entry runs that shape twice per row, at two
 * independently selected widths, in a single launch.
 *
 * Every entry is NONZERO-means-success: validation failure returns 0,
 * zero-length work returns 1, a completed launch returns 1.  Verified at
 * rocm/ds4_rocm_norm_rope.cuh:412-449. */

#include "ds4_sycl_common.hpp"

/* Not reliably defined by <cmath> under -fsycl; ds4.c:369-371 carries the
 * same guard for the CPU build. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

/* The work-group size is FIXED at 256 to match the CUDA launch geometry
 * (<<<rows, 256>>>).  This is NOT the variable-sizing case used by
 * ds4_gpu_directional_steering_project_tensor; do not carry that loop
 * over here.  The tree reduction below depends on this being a power of
 * two. */
constexpr size_t kRmsNormGroup = 256;

/* YaRN ramp helper, from rocm/ds4_rocm_norm_rope.cuh:246-249
 * (rope_yarn_ramp_dev).  Indexed by the CHANNEL index i0 = pair * 2, not
 * by the pair index: the division by two below undoes that before
 * comparing against the correction dims. */
static inline float sycl_rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}

}  // namespace

extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out,
                                                  const ds4_gpu_tensor *x,
                                                  uint32_t n, uint32_t rows,
                                                  float eps) {
    if (!sycl_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(x, n, rows, sizeof(float))) {
        return 0;
    }
    if (n == 0u || rows == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o = (float *)out->ptr;
        const float *px = (const float *)x->ptr;
        const uint32_t width = n;

        sycl::event _ds4_prof_ev130 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)rows * kRmsNormGroup),
                                  sycl::range<1>(kRmsNormGroup)),
                [=](sycl::nd_item<1> it) {
                    const size_t row = it.get_group(0);
                    const size_t lid = it.get_local_id(0);
                    const size_t lsz = it.get_local_range(0);

                    const float *xr   = px + row * width;
                    float       *orow = o  + row * width;

                    float sum = 0.0f;
                    for (size_t i = lid; i < width; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }

                    /* sycl_block_row_reduce (ds4_sycl_common.hpp) carries
                     * the full explanation of why every lane must call it
                     * unconditionally, including one whose loop above ran
                     * zero times. Matching rocm/ds4_rocm_norm_rope.cuh:11-12. */
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());

                    const float scale =
                            sycl::rsqrt(total / (float)width + eps);
                    for (size_t i = lid; i < width; i += lsz) {
                        orow[i] = xr[i] * scale;
                    }
                });
        });
        /* No trailing wait. out and x are both device tensors
         * with no host read and no staged scratch to free here, so the
         * in_order queue (ds4_sycl.cpp) is enough to order this kernel
         * against whatever reads `out` next on this same tier's queue; a
         * real completion signal is only needed at ds4_gpu_synchronize /
         * ds4_gpu_end_commands or a host readback, neither of which this
         * entry is. */
        ds4_sycl_profile_record_named("rms_norm_plain_rows", _ds4_prof_ev130);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "rms_norm_plain failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Single-row form.  ROCm launches the same kernel with rows=1
 * (rocm/ds4_rocm_norm_rope.cuh:415), so forward rather than duplicating. */
extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out,
                                             const ds4_gpu_tensor *x,
                                             uint32_t n, float eps) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(x, n)) return 0;
    if (n == 0u) return 1;
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1u, eps);
}

extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows,
        float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map ||
        !sycl_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(x, n, rows, sizeof(float))) {
        return 0;
    }
    if (n == 0u || rows == 0u) return 1;

    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "rms_weight");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        /* ROCm hands the host pointer straight to the kernel because CUDA
         * has the registered model device-accessible.  SYCL cannot
         * dereference the host mmap from a kernel, so stage the weight row
         * into device scratch (or pass through a pointer the model-range
         * cache already made device-resident, sycl_stage_host_bytes,
         * ds4_sycl_common.hpp).  The guard frees an owned allocation on
         * every path including the exception path, and frees nothing for
         * a passed-through pointer. */
        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        float *dw = (float *)dw_guard.p;
        if (!dw) return 0;

        float       *o  = (float *)out->ptr;
        const float *px = (const float *)x->ptr;
        const uint32_t width = n;

        sycl::event _ds4_prof_ev131 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)rows * kRmsNormGroup),
                                  sycl::range<1>(kRmsNormGroup)),
                [=](sycl::nd_item<1> it) {
                    const size_t row = it.get_group(0);
                    const size_t lid = it.get_local_id(0);
                    const size_t lsz = it.get_local_range(0);

                    const float *xr   = px + row * width;
                    float       *orow = o  + row * width;

                    float sum = 0.0f;
                    for (size_t i = lid; i < width; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());

                    const float scale =
                            sycl::rsqrt(total / (float)width + eps);
                    for (size_t i = lid; i < width; i += lsz) {
                        orow[i] = xr[i] * scale * dw[i];
                    }
                });
        });
        /* Wait kept. dw_guard may own a freshly staged scratch
         * allocation (sycl_stage_host_bytes, not a cached device-resident
         * pass-through), which its destructor frees on the host the
         * moment this function returns; in_order only orders queue
         * commands against each other, never a host-side sycl::free
         * against a kernel still in flight, so removing this wait would
         * be spec 6g's use-after-free on every uncached call. */
        q.wait_and_throw();
        ds4_sycl_profile_record_named("rms_norm_weight_rows", _ds4_prof_ev131);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "rms_norm_weight failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Single-row form.  ROCm launches the same kernel with rows=1
 * (rocm/ds4_rocm_norm_rope.cuh:425-436), so forward rather than
 * duplicating. */
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out,
                                              const ds4_gpu_tensor *x,
                                              const void *model_map,
                                              uint64_t model_size,
                                              uint64_t weight_offset,
                                              uint32_t n, float eps) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(x, n)) return 0;
    if (n == 0u) return 1;
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size,
                                               weight_offset, n, 1u, eps);
}

/* No kernel of its own: a literal composition over ds4_gpu_add_tensor (plan
 * 2) and ds4_gpu_rms_norm_weight_tensor above, matching ROCm's
 * ds4_gpu_add_rms_norm_weight_tensor at rocm/ds4_rocm_norm_rope.cuh:451-469.
 * Both sub-calls are NONZERO-means-success, so either returning 0 fails the
 * whole call. */
extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor *norm_out, ds4_gpu_tensor *sum_out,
        const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n, float eps) {
    if (ds4_gpu_add_tensor(sum_out, a, b, n) == 0) return 0;
    if (ds4_gpu_rms_norm_weight_tensor(norm_out, sum_out, model_map,
                                       model_size, weight_offset, n,
                                       eps) == 0) {
        return 0;
    }
    return 1;
}

/* Runs rms_norm_weight_kernel twice per row in one launch: Q with width
 * q_n, KV with width kv_n, matching ROCm's dsv4_qkv_rms_norm_rows_kernel
 * (rocm/ds4_rocm_norm_rope.cuh:47-81) and its launcher
 * (rocm/ds4_rocm_norm_rope.cuh:471-509).  The CUDA grid is (rows, 2, 1)
 * with blockIdx.y selecting Q versus KV; the natural SYCL equivalent is an
 * nd_range<2> whose second dimension has extent 2, so it.get_group(1) is
 * the selector and it.get_group(0) is the row.  Each side strides by ITS
 * OWN selected width: upstream computes the row base only after choosing
 * q_n or kv_n, so Q rows stride by q_n and KV rows stride by kv_n. */
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        uint32_t              rows,
        float                 eps) {
    uint64_t q_weight_bytes = 0, kv_weight_bytes = 0;
    if (!model_map ||
        !sycl_u64_mul_checked(q_n, sizeof(float), &q_weight_bytes) ||
        !sycl_u64_mul_checked(kv_n, sizeof(float), &kv_weight_bytes) ||
        !sycl_model_range_fits(model_size, q_weight_offset, q_weight_bytes) ||
        !sycl_model_range_fits(model_size, kv_weight_offset, kv_weight_bytes) ||
        !sycl_tensor_has_elems2(q_out, q_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(q, q_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(kv_out, kv_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(kv, kv_n, rows, sizeof(float))) {
        return 0;
    }
    if ((q_n == 0u && kv_n == 0u) || rows == 0u) return 1;

    const char *q_wptr = sycl_model_range_ptr(model_map, q_weight_offset,
                                              q_weight_bytes, model_size,
                                              "q_rms_weight");
    const char *kv_wptr = sycl_model_range_ptr(model_map, kv_weight_offset,
                                               kv_weight_bytes, model_size,
                                               "kv_rms_weight");
    if (!q_wptr || !kv_wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &sq = ds4_sycl_queue(q_out->device_id);

        /* Two independent weight vectors, each staged into its own device
         * scratch buffer under its own guard (or passed through if the
         * model-range cache already made it device-resident), matching
         * the staging pattern above.  A device allocation of width
         * 0 is permitted to return either a null or a valid pointer (SYCL
         * spec); only treat null as failure when the corresponding width
         * is nonzero, since one side's width can be 0 while the other's
         * is not. */
        sycl_device_scratch_guard dqw_guard = sycl_stage_host_bytes(sq, q_wptr, q_weight_bytes);
        float *dqw = (float *)dqw_guard.p;
        if (!dqw && q_n != 0u) return 0;

        sycl_device_scratch_guard dkvw_guard = sycl_stage_host_bytes(sq, kv_wptr, kv_weight_bytes);
        float *dkvw = (float *)dkvw_guard.p;
        if (!dkvw && kv_n != 0u) return 0;

        float       *qo   = (float *)q_out->ptr;
        const float *qx   = (const float *)q->ptr;
        float       *kvo  = (float *)kv_out->ptr;
        const float *kvx  = (const float *)kv->ptr;
        const uint32_t q_width  = q_n;
        const uint32_t kv_width = kv_n;

        sycl::event _ds4_prof_ev132 = sq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)rows * kRmsNormGroup, 2),
                        sycl::range<2>(kRmsNormGroup, 1)),
                [=](sycl::nd_item<2> it) {
                    const size_t row   = it.get_group(0);
                    const size_t which = it.get_group(1);
                    const size_t lid   = it.get_local_id(0);
                    const size_t lsz   = it.get_local_range(0);

                    const uint32_t width = which == 0 ? q_width : kv_width;
                    const float *xr = (which == 0 ? qx : kvx) + row * width;
                    float       *orow = (which == 0 ? qo : kvo) + row * width;
                    const float *w = which == 0 ? dqw : dkvw;

                    float sum = 0.0f;
                    for (size_t i = lid; i < width; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());

                    const float scale =
                            sycl::rsqrt(total / (float)width + eps);
                    for (size_t i = lid; i < width; i += lsz) {
                        orow[i] = xr[i] * scale * w[i];
                    }
                });
        });
        /* Wait kept, same reasoning as
         * ds4_gpu_rms_norm_weight_rows_tensor above -- dqw_guard and
         * dkvw_guard can each own scratch this function frees on return. */
        sq.wait_and_throw();
        ds4_sycl_profile_record_named("dsv4_qkv_rms_norm_rows", _ds4_prof_ev132);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "dsv4_qkv_rms_norm_rows failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Fuses the QKV RMS norm above with the KV rotary tail, in one launch keyed
 * by (row, which).  Ported from CUDA's dsv4_qkv_rms_norm_rows_kv_rope_kernel
 * (ds4_cuda.cu:6565) and its launcher (ds4_cuda.cu:16208): CUDA is the
 * only reference for this entry, the first in this port where ROCm has no
 * implementation at all to fall back on.  Only the kernel's SEMANTICS are
 * ported, not its stream assumptions (spec 6t): the single `parallel_for`
 * below is followed by one `wait_and_throw`, the same self-contained
 * shape every other entry in this file already uses, so there is no
 * cross-submit ordering to establish here.
 *
 * which == 0 (q): plain RMS-norm-with-weight over the full q_n row,
 * identical to the q branch above.  No RoPE: Q's rotary tail is applied
 * later, after the q_b projection, by
 * ds4_gpu_head_rms_norm_rope_tail_tensor.
 *
 * which == 1 (kv): the RMS-norm SCALE is ONE reduction over the WHOLE
 * kv_n row (every kv_n_head head concatenated), not per head -- this is
 * NOT ds4_gpu_head_rms_norm_rope_tail_tensor's per-(token,head)
 * reduction.  That single scale is then reused for every head: each
 * head's leading n_nope = kv_head_dim - n_rot channels are scaled and
 * weighted in place, and each head's trailing n_rot channels are scaled,
 * weighted AND rotated together, the same "scale-then-rotate" order as
 * ds4_gpu_head_rms_norm_rope_tail_tensor's tail loop but with kv_n_head
 * heads folded into one grid row instead of one head per row. */
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        uint32_t              rows,
        uint32_t              kv_n_head,
        uint32_t              kv_head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps) {
    uint64_t q_weight_bytes = 0, kv_weight_bytes = 0, kv_n_check = 0;
    if (kv_n_head == 0u || kv_head_dim == 0u ||
        n_rot > kv_head_dim || (n_rot & 1u) ||
        !sycl_u64_mul_checked(kv_n_head, kv_head_dim, &kv_n_check) ||
        kv_n_check != (uint64_t)kv_n ||
        !model_map ||
        !sycl_u64_mul_checked(q_n, sizeof(float), &q_weight_bytes) ||
        !sycl_u64_mul_checked(kv_n, sizeof(float), &kv_weight_bytes) ||
        !sycl_model_range_fits(model_size, q_weight_offset, q_weight_bytes) ||
        !sycl_model_range_fits(model_size, kv_weight_offset, kv_weight_bytes) ||
        !sycl_tensor_has_elems2(q_out, q_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(q, q_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(kv_out, kv_n, rows, sizeof(float)) ||
        !sycl_tensor_has_elems2(kv, kv_n, rows, sizeof(float))) {
        return 0;
    }
    if (rows == 0u) return 1;

    const char *q_wptr = sycl_model_range_ptr(model_map, q_weight_offset,
                                              q_weight_bytes, model_size,
                                              "q_rms_weight");
    const char *kv_wptr = sycl_model_range_ptr(model_map, kv_weight_offset,
                                               kv_weight_bytes, model_size,
                                               "kv_rms_weight");
    if (!q_wptr || !kv_wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &sq = ds4_sycl_queue(q_out->device_id);

        /* Zero-width staging can return a null pointer under the SYCL
         * spec; only treat that as failure when the corresponding side
         * actually has work to do, matching the unfused entry above. */
        sycl_device_scratch_guard dqw_guard =
                sycl_stage_host_bytes(sq, q_wptr, q_weight_bytes);
        if (!dqw_guard.p && q_n != 0u) return 0;
        sycl_device_scratch_guard dkvw_guard =
                sycl_stage_host_bytes(sq, kv_wptr, kv_weight_bytes);
        if (!dkvw_guard.p && kv_n != 0u) return 0;

        const float *dqw  = (const float *)dqw_guard.p;
        const float *dkvw = (const float *)dkvw_guard.p;

        float       *qo   = (float *)q_out->ptr;
        const float *qx   = (const float *)q->ptr;
        float       *kvo  = (float *)kv_out->ptr;
        const float *kvx  = (const float *)kv->ptr;
        const uint32_t q_width     = q_n;
        const uint32_t kv_width    = kv_n;
        const uint32_t n_nope      = kv_head_dim - n_rot;
        const uint32_t pairs_head  = n_rot / 2u;
        const uint32_t total_pairs = kv_n_head * pairs_head;

        sycl::event _ds4_prof_ev133 = sq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)rows * kRmsNormGroup, 2),
                        sycl::range<2>(kRmsNormGroup, 1)),
                [=](sycl::nd_item<2> it) {
                    const size_t row   = it.get_group(0);
                    const size_t which = it.get_group(1);
                    const size_t lid   = it.get_local_id(0);
                    const size_t lsz   = it.get_local_range(0);

                    const uint32_t width = which == 0 ? q_width : kv_width;
                    const float *xr   = (which == 0 ? qx : kvx) + row * width;
                    float       *orow = (which == 0 ? qo : kvo) + row * width;
                    const float *w    = which == 0 ? dqw : dkvw;

                    float sum = 0.0f;
                    for (size_t i = lid; i < width; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());
                    const float scale =
                            sycl::rsqrt(total / (float)width + eps);

                    if (which == 0) {
                        for (size_t i = lid; i < width; i += lsz) {
                            orow[i] = xr[i] * scale * w[i];
                        }
                        return;
                    }

                    for (uint32_t hh = 0; hh < kv_n_head; hh++) {
                        const uint32_t head_base = hh * kv_head_dim;
                        for (size_t d = lid; d < n_nope; d += lsz) {
                            const uint32_t i = head_base + (uint32_t)d;
                            orow[i] = xr[i] * scale * w[i];
                        }
                    }

                    /* Correction dims recomputed by every work-item,
                     * matching the CUDA source and the fused per-head
                     * kernel above. */
                    float corr0 = 0.0f, corr1 = 0.0f;
                    if (ext_factor != 0.0f) {
                        const float denom = 2.0f * sycl::log(freq_base);
                        corr0 = sycl::floor((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_fast * 2.0f * (float)M_PI)) / denom);
                        corr1 = sycl::ceil((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_slow * 2.0f * (float)M_PI)) / denom);
                        corr0 = sycl::fmax(0.0f, corr0);
                        corr1 = sycl::fmin((float)(n_rot - 1), corr1);
                    }
                    const float theta_scale =
                            sycl::pow(freq_base, -2.0f / (float)n_rot);

                    for (size_t p = lid; p < total_pairs; p += lsz) {
                        const uint32_t hh   = (uint32_t)p / pairs_head;
                        const uint32_t pair = (uint32_t)p - hh * pairs_head;
                        const uint32_t d    = n_nope + pair * 2u;
                        const uint32_t i0   = hh * kv_head_dim + d;
                        const uint32_t i    = pair * 2u;

                        /* Direct power, not iterative accumulation; see
                         * the matching comment on the per-head fused
                         * kernel above. */
                        const float theta_extrap =
                                (float)(pos0 + row) *
                                sycl::pow(theta_scale, (float)pair);
                        const float theta_interp = freq_scale * theta_extrap;
                        float theta  = theta_interp;
                        float mscale = attn_factor;
                        if (ext_factor != 0.0f) {
                            const float ramp_mix =
                                    sycl_rope_yarn_ramp(corr0, corr1, (int)i) *
                                    ext_factor;
                            theta = theta_interp * (1.0f - ramp_mix) +
                                    theta_extrap * ramp_mix;
                            mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                        }
                        float c = sycl::cos(theta) * mscale;
                        float s = sycl::sin(theta) * mscale;
                        if (inverse) s = -s;

                        const float x0 = xr[i0]     * scale * w[i0];
                        const float x1 = xr[i0 + 1u] * scale * w[i0 + 1u];
                        orow[i0]      = x0 * c - x1 * s;
                        orow[i0 + 1u] = x0 * s + x1 * c;
                    }
                });
        });
        /* Wait kept, same reasoning as
         * ds4_gpu_rms_norm_weight_rows_tensor above -- dqw_guard and
         * dkvw_guard can each own scratch this function frees on return. */
        sq.wait_and_throw();
        ds4_sycl_profile_record_named("dsv4_qkv_rms_norm_rows_kv_rope", _ds4_prof_ev133);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "dsv4_qkv_rms_norm_rows_kv_rope failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Normalises each (token, head) slice of head_dim IN PLACE, once per
 * flattened row over n_tok * n_head, matching ROCm's head_rms_norm_kernel
 * (rocm/ds4_rocm_norm_rope.cuh:83-101) and its launcher
 * (rocm/ds4_rocm_norm_rope.cuh:511-518).  There is no separate output
 * tensor: x is read and rewritten in the same pass. */
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok,
                                            uint32_t n_head, uint32_t head_dim,
                                            float eps) {
    uint64_t rows64 = 0;
    if (!sycl_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !sycl_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    if (rows64 == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float         *px    = (float *)x->ptr;
        const uint32_t width = head_dim;
        const size_t   rows  = (size_t)rows64;

        sycl::event _ds4_prof_ev134 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * kRmsNormGroup),
                                  sycl::range<1>(kRmsNormGroup)),
                [=](sycl::nd_item<1> it) {
                    const size_t row = it.get_group(0);
                    const size_t lid = it.get_local_id(0);
                    const size_t lsz = it.get_local_range(0);

                    float *xr = px + row * width;

                    float sum = 0.0f;
                    for (size_t i = lid; i < width; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());

                    const float scale =
                            sycl::rsqrt(total / (float)width + eps);
                    for (size_t i = lid; i < width; i += lsz) {
                        xr[i] *= scale;
                    }
                });
        });
        /* No trailing wait, same reasoning as
         * ds4_gpu_rms_norm_plain_rows_tensor above -- in-place on a device
         * tensor, no staged scratch, no host read. */
        ds4_sycl_profile_record_named("head_rms_norm", _ds4_prof_ev134);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "head_rms_norm failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Fused head RMS-norm plus DeepSeek partial RoPE, one work-group per
 * (token, head) row, matching ROCm's head_rms_norm_rope_tail_kernel
 * (rocm/ds4_rocm_norm_rope.cuh:105-172) and its launcher
 * (rocm/ds4_rocm_norm_rope.cuh:523-530).  Shares the reduction shape used
 * throughout this file: fixed 256 work-group, strided accumulate, tree
 * reduction, then a strided pass over the row.
 *
 * PARTIAL ROPE, READ CAREFULLY: the leading n_nope = head_dim - n_rot
 * channels are scaled ONLY, in the loop below.  The trailing n_rot
 * channels are NOT touched by that loop; their scale is applied inside
 * the rotation loop, at the point each pair is read.  Scaling the whole
 * row here and then rotating would DOUBLE-SCALE the tail -- a smooth
 * magnitude error that looks entirely plausible on inspection. */
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    uint64_t rows64 = 0;
    if (n_rot > head_dim || (n_rot & 1u) ||
        !sycl_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !sycl_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    if (rows64 == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float         *px   = (float *)x->ptr;
        const size_t   rows = (size_t)rows64;

        sycl::event _ds4_prof_ev135 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(
                    sycl::range<1>(kRmsNormGroup), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * kRmsNormGroup),
                                  sycl::range<1>(kRmsNormGroup)),
                [=](sycl::nd_item<1> it) {
                    const size_t row = it.get_group(0);
                    const size_t lid = it.get_local_id(0);
                    const size_t lsz = it.get_local_range(0);
                    const uint32_t t = (uint32_t)(row / n_head);

                    float *xr = px + row * head_dim;

                    float sum = 0.0f;
                    for (size_t i = lid; i < head_dim; i += lsz) {
                        const float v = xr[i];
                        sum += v * v;
                    }
                    const float total = sycl_block_row_reduce(
                            it, partial, sum, sycl::plus<float>());
                    const float scale =
                            sycl::rsqrt(total / (float)head_dim + eps);

                    /* See the function comment above: this loop scales
                     * ONLY the leading n_nope channels.  The tail is
                     * scaled inside the rotation loop below, not here. */
                    const uint32_t n_nope = head_dim - n_rot;
                    for (size_t i = lid; i < n_nope; i += lsz) {
                        xr[i] *= scale;
                    }

                    /* Correction dims are recomputed by every work-item,
                     * matching the CUDA source, which does not gate this
                     * behind threadIdx.x == 0.  Keep it redundant. */
                    float corr0 = 0.0f, corr1 = 0.0f;
                    if (ext_factor != 0.0f) {
                        const float denom = 2.0f * sycl::log(freq_base);
                        corr0 = sycl::floor((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_fast * 2.0f * (float)M_PI)) / denom);
                        corr1 = sycl::ceil((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_slow * 2.0f * (float)M_PI)) / denom);
                        corr0 = sycl::fmax(0.0f, corr0);
                        corr1 = sycl::fmin((float)(n_rot - 1), corr1);
                    }

                    const float theta_scale =
                            sycl::pow(freq_base, -2.0f / (float)n_rot);

                    for (size_t pair = lid; pair < n_rot / 2; pair += lsz) {
                        const uint32_t i = (uint32_t)pair * 2u;
                        /* Direct power, NOT iterative accumulation.  ds4's
                         * CPU reference accumulates (ds4.c:10273); every GPU
                         * backend uses this form.  Mathematically identical,
                         * not bit-identical.  Do not "fix" this to match the
                         * CPU: it would diverge from Metal, CUDA and ROCm at
                         * once. */
                        const float theta_extrap =
                                (float)(pos0 + t) *
                                sycl::pow(theta_scale, (float)pair);
                        const float theta_interp = freq_scale * theta_extrap;
                        float theta  = theta_interp;
                        float mscale = attn_factor;
                        if (ext_factor != 0.0f) {
                            const float ramp_mix =
                                    sycl_rope_yarn_ramp(corr0, corr1, (int)i) *
                                    ext_factor;
                            theta = theta_interp * (1.0f - ramp_mix) +
                                    theta_extrap * ramp_mix;
                            mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                        }
                        float c = sycl::cos(theta) * mscale;
                        float s = sycl::sin(theta) * mscale;
                        if (inverse) s = -s;

                        float *tail = xr + n_nope;
                        const float x0 = tail[i]     * scale;
                        const float x1 = tail[i + 1] * scale;
                        tail[i]     = x0 * c - x1 * s;
                        tail[i + 1] = x0 * s + x1 * c;
                    }
                });
        });
        /* No trailing wait, same reasoning as
         * ds4_gpu_rms_norm_plain_rows_tensor above -- in-place on a device
         * tensor, no staged scratch, no host read. */
        ds4_sycl_profile_record_named("head_rms_norm_rope_tail", _ds4_prof_ev135);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "head_rms_norm_rope_tail failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Flat pair-parallel DeepSeek partial RoPE with no reduction, matching
 * ROCm's rope_tail_kernel (rocm/ds4_rocm_norm_rope.cuh:251-300) and its
 * stride-aware launcher cuda_rope_tail_stride_tensor
 * (rocm/ds4_rocm_norm_rope.cuh:603-609).  STRUCTURALLY DIFFERENT from
 * ds4_gpu_head_rms_norm_rope_tail_tensor above, not the same kernel with
 * the norm stripped out: there is no work-group per row, no local
 * accessor, no barrier and no `scale` (nothing was normalised).  Instead
 * one work-item is launched per rotation pair, and the flat id is
 * decomposed into (pair, head, token) inside the kernel.
 *
 * pos_stride multiplies the token index in the angle
 * (pos0 + t * pos_stride), not just pos0 itself.  This function stays
 * static and keeps pos_stride as an explicit parameter, separate from the
 * public ds4_gpu_rope_tail_tensor wrapper below, because the
 * compressor calls this stride variant directly with pos_stride != 1. */
static int sycl_rope_tail_stride_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t pos_stride,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    if (n_rot > head_dim || (n_rot & 1u) ||
        !sycl_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    const uint32_t pairs = n_tok * n_head * (n_rot / 2u);
    if (pairs == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float       *px    = (float *)x->ptr;
        const size_t group = kRmsNormGroup;
        const size_t grid  = ((size_t)pairs + group - 1) / group * group;

        sycl::event _ds4_prof_ev136 = q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(grid),
                                  sycl::range<1>(group)),
                [=](sycl::nd_item<1> it) {
                    const size_t gid = it.get_global_id(0);
                    if (gid >= (size_t)pairs) return;

                    const uint32_t half   = n_rot / 2u;
                    const uint32_t pair   = (uint32_t)gid % half;
                    const uint32_t tmp    = (uint32_t)gid / half;
                    const uint32_t h      = tmp % n_head;
                    const uint32_t t      = tmp / n_head;
                    const uint32_t n_nope = head_dim - n_rot;
                    const uint32_t i      = pair * 2u;

                    float corr0 = 0.0f, corr1 = 0.0f;
                    if (ext_factor != 0.0f) {
                        const float denom = 2.0f * sycl::log(freq_base);
                        corr0 = sycl::floor((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_fast * 2.0f * (float)M_PI)) / denom);
                        corr1 = sycl::ceil((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_slow * 2.0f * (float)M_PI)) / denom);
                        corr0 = sycl::fmax(0.0f, corr0);
                        corr1 = sycl::fmin((float)(n_rot - 1), corr1);
                    }

                    const float theta_scale =
                            sycl::pow(freq_base, -2.0f / (float)n_rot);
                    /* Direct power, NOT iterative accumulation; see the
                     * matching comment in the fused kernel above. */
                    const float theta_extrap =
                            (float)(pos0 + t * pos_stride) *
                            sycl::pow(theta_scale, (float)pair);
                    const float theta_interp = freq_scale * theta_extrap;
                    float theta  = theta_interp;
                    float mscale = attn_factor;
                    if (ext_factor != 0.0f) {
                        const float ramp_mix =
                                sycl_rope_yarn_ramp(corr0, corr1, (int)i) *
                                ext_factor;
                        theta = theta_interp * (1.0f - ramp_mix) +
                                theta_extrap * ramp_mix;
                        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                    }
                    float c = sycl::cos(theta) * mscale;
                    float s = sycl::sin(theta) * mscale;
                    if (inverse) s = -s;

                    /* No `scale`: nothing was normalised on this path, so
                     * the pair is read and rotated as-is. */
                    float *tail = px +
                            ((uint64_t)t * n_head + h) * head_dim + n_nope;
                    const float x0 = tail[i];
                    const float x1 = tail[i + 1];
                    tail[i]     = x0 * c - x1 * s;
                    tail[i + 1] = x0 * s + x1 * c;
                });
        });
        /* No trailing wait, same reasoning as
         * ds4_gpu_rms_norm_plain_rows_tensor above -- in-place on a device
         * tensor, no staged scratch, no host read. */
        ds4_sycl_profile_record_named("rope_tail_stride", _ds4_prof_ev136);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "rope_tail failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Thin wrapper over the stride variant above with pos_stride = 1, matching
 * ROCm's ds4_gpu_rope_tail_tensor (rocm/ds4_rocm_norm_rope.cuh:610-612). */
extern "C" int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    return sycl_rope_tail_stride_tensor(x, n_tok, n_head, head_dim, n_rot,
                                        pos0, 1u, n_ctx_orig, inverse,
                                        freq_base, freq_scale, ext_factor,
                                        attn_factor, beta_fast, beta_slow);
}

/* ds4_gpu_rope_tail_decode_rows_tensor, ds4_cuda.cu:16311-16349
 * (rope_tail_decode_rows_kernel, ds4_cuda.cu:6787-6844): the row-batched
 * form of ds4_gpu_rope_tail_tensor above, generalising
 * sycl_rope_tail_stride_tensor's per-pair kernel over an explicit row
 * table instead of a single pos0 + t * pos_stride derivation. None of
 * this exists in rocm/; CUDA is the only reference (a CUDA-only entry).
 *
 * Called at ds4.c:64836 (on the KV row, n_head = DS4_N_HEAD_KV == 1 for
 * Flash) and ds4.c:64855 (on the Q row, n_head = DS4_N_HEAD), both inside
 * metal_graph_encode_qkv_session_batch, which projects, RMS-normalises
 * and RoPE-rotates several concurrent sessions' rows in one launch when
 * ds4_sessions_eval_batch is called with count >= 3 (the QKV grouping
 * path's own threshold, metal_graph_session_batch_qkv_supported).
 *
 * Only rows[i].pos is read; the row struct's KV-cache fields (raw_kv,
 * comp_kv, topk, ...) are irrelevant to this entry, exactly as CUDA's own
 * launcher copies just the row array's .pos field into its row table
 * here. The row table is a host array (ds4_gpu_attention_decode_row has
 * no device-resident form anywhere in this ABI), so per design spec 6l it
 * cannot be dereferenced by a kernel directly: the per-row positions are
 * staged to device scratch first, the same discipline as every other
 * host-side table this backend stages before a launch. */
extern "C" int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                     *x,
        const ds4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow) {
    if (!x || !rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0u ||
        n_rot == 0u || n_rot > head_dim || (n_rot & 1u) != 0u ||
        !sycl_tensor_has_elems3(x, n_rows, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    if (pairs == 0u) return 1;
    if (g_devices.empty()) return 0;

    std::vector<uint32_t> row_pos((size_t)n_rows);
    for (uint32_t i = 0; i < n_rows; i++) row_pos[i] = rows[i].pos;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        sycl_device_scratch_guard pos_guard = sycl_stage_host_bytes(
                q, row_pos.data(), (uint64_t)n_rows * sizeof(uint32_t));
        if (!pos_guard.p) return 0;
        const uint32_t *ppos  = (const uint32_t *)pos_guard.p;
        float           *px   = (float *)x->ptr;
        const size_t     group = kRmsNormGroup;
        const size_t     grid  = ((size_t)pairs + group - 1) / group * group;

        sycl::event _ds4_prof_ev137 = q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(grid),
                                  sycl::range<1>(group)),
                [=](sycl::nd_item<1> it) {
                    const size_t gid = it.get_global_id(0);
                    if (gid >= (size_t)pairs) return;

                    const uint32_t half   = n_rot / 2u;
                    const uint32_t pair   = (uint32_t)gid % half;
                    const uint32_t tmp    = (uint32_t)gid / half;
                    const uint32_t h      = tmp % n_head;
                    const uint32_t row    = tmp / n_head;
                    const uint32_t n_nope = head_dim - n_rot;
                    const uint32_t i      = pair * 2u;

                    float corr0 = 0.0f, corr1 = 0.0f;
                    if (ext_factor != 0.0f) {
                        const float denom = 2.0f * sycl::log(freq_base);
                        corr0 = sycl::floor((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_fast * 2.0f * (float)M_PI)) / denom);
                        corr1 = sycl::ceil((float)n_rot *
                                sycl::log((float)n_ctx_orig /
                                          (beta_slow * 2.0f * (float)M_PI)) / denom);
                        corr0 = sycl::fmax(0.0f, corr0);
                        corr1 = sycl::fmin((float)(n_rot - 1), corr1);
                    }

                    const float theta_scale =
                            sycl::pow(freq_base, -2.0f / (float)n_rot);
                    /* Direct power, NOT iterative accumulation; see the
                     * matching comment on the fused kernel above. Each
                     * row's own absolute position, not a shared pos0. */
                    const float theta_extrap =
                            (float)ppos[row] * sycl::pow(theta_scale, (float)pair);
                    const float theta_interp = freq_scale * theta_extrap;
                    float theta  = theta_interp;
                    float mscale = attn_factor;
                    if (ext_factor != 0.0f) {
                        const float ramp_mix =
                                sycl_rope_yarn_ramp(corr0, corr1, (int)i) *
                                ext_factor;
                        theta = theta_interp * (1.0f - ramp_mix) +
                                theta_extrap * ramp_mix;
                        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                    }
                    float c = sycl::cos(theta) * mscale;
                    float s = sycl::sin(theta) * mscale;
                    if (inverse) s = -s;

                    float *tail = px +
                            ((uint64_t)row * n_head + h) * head_dim + n_nope;
                    const float x0 = tail[i];
                    const float x1 = tail[i + 1];
                    tail[i]     = x0 * c - x1 * s;
                    tail[i + 1] = x0 * s + x1 * c;
                });
        });
        /* Wait kept, same reasoning as
         * ds4_gpu_rms_norm_weight_rows_tensor above -- pos_guard can own
         * scratch this function frees on return. */
        q.wait_and_throw();
        ds4_sycl_profile_record_named("rope_tail_decode_rows", _ds4_prof_ev137);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "rope_tail_decode_rows failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}
