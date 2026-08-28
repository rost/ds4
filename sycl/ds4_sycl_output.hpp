#pragma once

/* Elementwise and reduction kernels backing the output-side entry points.
 * ROCm splits these across ds4_rocm_output.cuh (kernels) and
 * ds4_rocm_misc_launch.cuh / _shared_expert.cuh / _hc_output_launch.cuh
 * (launchers) because CUDA needs __global__ functions at namespace scope.
 * SYCL has no such constraint, so each entry point is kept whole here.
 *
 * Every entry in this file is NONZERO-means-success: validation failure
 * returns 0, zero-length work returns 1, a completed launch returns 1.
 * Verified against rocm/ds4_rocm_misc_launch.cuh:1-6. */

#include "ds4_sycl_common.hpp"

extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                                  const ds4_gpu_tensor *b, uint32_t n) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(a, n) ||
        !sycl_tensor_has_f32(b, n)) {
        return 0;
    }
    if (n == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o = (float *)out->ptr;
        const float *pa = (const float *)a->ptr;
        const float *pb = (const float *)b->ptr;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            o[i] = pa[i] + pb[i];
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "add failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_add3_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                                   const ds4_gpu_tensor *b,
                                   const ds4_gpu_tensor *c, uint32_t n) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(a, n) ||
        !sycl_tensor_has_f32(b, n) || !sycl_tensor_has_f32(c, n)) {
        return 0;
    }
    if (n == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o = (float *)out->ptr;
        const float *pa = (const float *)a->ptr;
        const float *pb = (const float *)b->ptr;
        const float *pc = (const float *)c->ptr;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            o[i] = pa[i] + pb[i] + pc[i];
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "add3 failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
