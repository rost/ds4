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

namespace {

/* Frees a sycl::malloc_device allocation when it goes out of scope, on
 * every exit path including an exception unwinding through the try block
 * that owns it.  Without this, a throw between allocation and the
 * matching sycl::free (a memcpy or kernel wait_and_throw failing) leaks
 * device memory: the catch block returns before reaching the free.  This
 * pattern recurs whenever an entry point stages small host-side data
 * (scale/base vectors, quant tables, and similar) into a scratch device
 * buffer ahead of a kernel launch, so later kernels doing the same thing
 * should reuse this guard rather than a bare malloc_device/free pair. */
struct sycl_device_scratch_guard {
    sycl::queue &q;
    void        *p;
    sycl_device_scratch_guard(sycl::queue &queue, void *ptr) : q(queue), p(ptr) {}
    ~sycl_device_scratch_guard() {
        /* Destructors are implicitly noexcept: if sycl::free throws while
         * we are already unwinding from another exception (e.g. the
         * memcpy's wait_and_throw), an escaping exception here would call
         * std::terminate instead of surfacing a clean failure.  There is
         * no meaningful recovery from a failed free during unwinding, so
         * log and swallow rather than let it propagate. */
        if (!p) return;
        try {
            sycl::free(p, q);
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "device scratch free failed: %s\n", e.what());
        }
    }
    sycl_device_scratch_guard(const sycl_device_scratch_guard &) = delete;
    sycl_device_scratch_guard &operator=(const sycl_device_scratch_guard &) = delete;
};

}  // namespace

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
