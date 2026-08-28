#pragma once

/* RMS normalisation and DeepSeek partial RoPE.
 *
 * Ported from rocm/ds4_rocm_norm_rope.cuh, which is the authority for all
 * semantics here.  Four entry points in this file share one reduction
 * shape: one work-group per row, a FIXED 256 work-items, a strided
 * accumulate into local memory, a power-of-two tree reduction, then a
 * strided scale pass.
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
