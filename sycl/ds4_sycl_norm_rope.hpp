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

namespace {

/* The work-group size is FIXED at 256 to match the CUDA launch geometry
 * (<<<rows, 256>>>).  This is NOT the variable-sizing case used by
 * ds4_gpu_directional_steering_project_tensor; do not carry that loop
 * over here.  The tree reduction below depends on this being a power of
 * two. */
constexpr size_t kRmsNormGroup = 256;

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

                    /* EVERY work-item writes its slot, including those
                     * whose loop above ran zero times and whose sum is
                     * therefore 0.  Matching
                     * rocm/ds4_rocm_norm_rope.cuh:11-12.  Writing only the
                     * slots that had work would tree-reduce uninitialised
                     * local memory whenever width < 256.
                     *
                     * This repo's SYCL tests cannot catch removal of this
                     * write on the current hardware and driver stack,
                     * where uninitialised local memory has been observed
                     * to read as zero (Arc A770, Level Zero, oneAPI
                     * 2025.3). Do not treat a passing test suite as
                     * licence to remove it. The SYCL specification
                     * guarantees no zero-initialisation, so the behaviour
                     * may differ on other hardware, including the
                     * Battlemage devices this backend targets. */
                    partial[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) partial[lid] += partial[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    const float scale =
                            sycl::rsqrt(partial[0] / (float)width + eps);
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
                    /* Every work-item writes its slot unconditionally; see
                     * the long comment on the equivalent line in
                     * ds4_gpu_rms_norm_plain_rows_tensor above. */
                    partial[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) partial[lid] += partial[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    const float scale =
                            sycl::rsqrt(partial[0] / (float)width + eps);
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
                    /* Every work-item writes its slot unconditionally; see
                     * the long comment on the equivalent line in
                     * ds4_gpu_rms_norm_plain_rows_tensor above. */
                    partial[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) partial[lid] += partial[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    const float scale =
                            sycl::rsqrt(partial[0] / (float)width + eps);
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
                    /* Every work-item writes its slot unconditionally; see
                     * the long comment on the equivalent line in
                     * ds4_gpu_rms_norm_plain_rows_tensor above. */
                    partial[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) partial[lid] += partial[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    const float scale =
                            sycl::rsqrt(partial[0] / (float)width + eps);
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
