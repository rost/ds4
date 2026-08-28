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

extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out,
                                     const ds4_gpu_tensor *gate,
                                     const ds4_gpu_tensor *up, uint32_t n,
                                     float clamp, float weight) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(gate, n) ||
        !sycl_tensor_has_f32(up, n)) {
        return 0;
    }
    if (n == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o = (float *)out->ptr;
        const float *pg = (const float *)gate->ptr;
        const float *pu = (const float *)up->ptr;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float g = pg[i];
            float u = pu[i];
            /* Clamp is symmetric on up and one-sided on gate, matching
             * swiglu_kernel at rocm/ds4_rocm_output.cuh:23-34. */
            if (clamp > 1e-6f) {
                u = sycl::clamp(u, -clamp, clamp);
                g = sycl::min(g, clamp);
            }
            float s = g / (1.0f + sycl::exp(-g));
            o[i] = s * u * weight;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "swiglu failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *pre, const void *model_map,
        uint64_t model_size, uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_hc, float eps) {
    if (!out || !pre || !model_map || n_hc == 0u) return 0;

    /* Element count matches ds4_gpu_output_hc_weights_tensor at
     * rocm/ds4_rocm_hc_output_launch.cuh:203-233, the authority: out must
     * hold a whole number of n_hc-wide rows (one row per token), and pre
     * must be at least as large as out.  This is stricter than deriving n
     * from out's raw byte capacity alone, which would silently accept an
     * output tensor that is not an exact multiple of n_hc floats. */
    uint64_t row_bytes = 0;
    if (!sycl_u64_mul_checked(n_hc, sizeof(float), &row_bytes)) return 0;
    if (!sycl_tensor_has_bytes(out, row_bytes) || out->bytes % row_bytes != 0 ||
        !sycl_tensor_has_bytes(pre, out->bytes)) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const uint64_t n = n_tokens * n_hc;

    const char *scale_p = sycl_model_range_ptr(model_map, scale_offset,
                                               sizeof(float), model_size, "hc scale");
    const char *base_p  = sycl_model_range_ptr(model_map, base_offset,
                                               row_bytes, model_size, "hc base");
    if (!scale_p || !base_p) return 0;
    if (g_devices.empty()) return 0;

    /* scale and base live in the host mmap; copy the few bytes we need to
     * the device rather than dereferencing host memory from a kernel. */
    const float scale = *(const float *)scale_p;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float *dbase = sycl::malloc_device<float>(n_hc, q);
        if (!dbase) return 0;
        sycl_device_scratch_guard dbase_guard(q, dbase);
        q.memcpy(dbase, base_p, (size_t)row_bytes).wait_and_throw();

        float       *o  = (float *)out->ptr;
        const float *pp = (const float *)pre->ptr;
        const uint32_t hc = n_hc;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            float z = pp[i] * scale + dbase[i % hc];
            /* Unstable sigmoid, matching output_hc_weights_kernel at
             * rocm/ds4_rocm_output.cuh:6-20.  ds4's CPU path uses the
             * branch-stable form; the two differ only for large negative
             * z, which bounded activations do not reach. */
            o[i] = 1.0f / (1.0f + sycl::exp(-z)) + eps;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "output_hc_weights failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *directions, uint32_t layer,
        uint32_t width, uint32_t rows, float scale) {
    if (!x || !directions || width == 0u || rows == 0u) return 0;
    if (!sycl_tensor_has_elems2(x, rows, width, sizeof(float))) return 0;

    uint64_t dir_elems = 0;
    if (!sycl_u64_mul_checked((uint64_t)layer + 1, width, &dir_elems)) return 0;
    if (!sycl_tensor_has_f32(directions, dir_elems)) return 0;

    /* A zero scale is a no-op that returns SUCCESS without launching,
     * matching rocm/ds4_rocm_misc_launch.cuh:70 `if (scale == 0.0f) return 1;`.
     * Do not remove this: launching a kernel that subtracts zero would give
     * the same numbers but diverge behaviourally from every other backend. */
    if (scale == 0.0f) return 1;
    if (g_devices.empty()) return 0;

    /* Largest power of two that is both <= 256 and <= width.  The local
     * accessor and the reduction loop below must both use THIS value, not a
     * hardcoded 256.
     *
     * ROCm writes the same computation as a downward halve from 256
     * (rocm/ds4_rocm_misc_launch.cuh:72-73):
     *     uint32_t nth = 256u; while (nth > width && nth > 1u) nth >>= 1;
     * The upward form below is equivalent for every width (verified at
     * width 300, 100 and 1) and is kept because it reads more directly.
     * Either is correct; do not "fix" one into the other. */
    size_t wg = 1;
    while (wg * 2 <= 256 && wg * 2 <= (size_t)width) wg *= 2;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float       *px = (float *)x->ptr;
        const float *pd = (const float *)directions->ptr + (size_t)layer * width;
        const uint32_t w = width;

        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(sycl::range<1>(wg), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)rows * wg),
                                  sycl::range<1>(wg)),
                [=](sycl::nd_item<1> it) {
                    const size_t row = it.get_group(0);
                    const size_t lid = it.get_local_id(0);
                    const size_t lsz = it.get_local_range(0);

                    float acc = 0.0f;
                    for (size_t i = lid; i < w; i += lsz) {
                        acc += px[row * w + i] * pd[i];
                    }
                    partial[lid] = acc;
                    it.barrier(sycl::access::fence_space::local_space);

                    for (size_t s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) partial[lid] += partial[lid + s];
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    const float k = scale * partial[0];
                    for (size_t i = lid; i < w; i += lsz) {
                        px[row * w + i] -= k * pd[i];
                    }
                });
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "directional steering failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}
