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

        q.submit([&](sycl::handler &h) {
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
        q.wait_and_throw();
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
         * into device scratch.  The guard frees it on every path including
         * the exception path. */
        float *dw = sycl::malloc_device<float>(n, q);
        if (!dw) return 0;
        sycl_device_scratch_guard dw_guard(q, dw);
        q.memcpy(dw, wptr, (size_t)weight_bytes).wait_and_throw();

        float       *o  = (float *)out->ptr;
        const float *px = (const float *)x->ptr;
        const uint32_t width = n;

        q.submit([&](sycl::handler &h) {
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
        q.wait_and_throw();
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
         * scratch buffer under its own guard, following the same staging
         * pattern used above.  A device allocation of width 0 is permitted to
         * return either a null or a valid pointer (SYCL spec); only treat
         * null as failure when the corresponding width is nonzero, since
         * one side's width can be 0 while the other's is not. */
        float *dqw = sycl::malloc_device<float>(q_n, sq);
        if (!dqw && q_n != 0u) return 0;
        sycl_device_scratch_guard dqw_guard(sq, dqw);
        if (q_n != 0u) sq.memcpy(dqw, q_wptr, (size_t)q_weight_bytes).wait_and_throw();

        float *dkvw = sycl::malloc_device<float>(kv_n, sq);
        if (!dkvw && kv_n != 0u) return 0;
        sycl_device_scratch_guard dkvw_guard(sq, dkvw);
        if (kv_n != 0u) sq.memcpy(dkvw, kv_wptr, (size_t)kv_weight_bytes).wait_and_throw();

        float       *qo   = (float *)q_out->ptr;
        const float *qx   = (const float *)q->ptr;
        float       *kvo  = (float *)kv_out->ptr;
        const float *kvx  = (const float *)kv->ptr;
        const uint32_t q_width  = q_n;
        const uint32_t kv_width = kv_n;

        sq.submit([&](sycl::handler &h) {
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
        sq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "dsv4_qkv_rms_norm_rows failed: %s\n",
                e.what());
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

        q.submit([&](sycl::handler &h) {
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
        q.wait_and_throw();
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

        q.submit([&](sycl::handler &h) {
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
        q.wait_and_throw();
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

        q.submit([&](sycl::handler &h) {
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
        q.wait_and_throw();
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
