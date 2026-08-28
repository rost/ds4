#pragma once

/* Argument-validation helpers shared by every SYCL kernel entry point.
 * These mirror the cuda_* family in rocm/ds4_rocm_runtime.cuh:376-530.
 * Every one is overflow-safe by construction: ds4 entry points receive
 * sizes from model metadata and must never compute a product or a range
 * that wraps. */

#include "ds4_gpu_mgpu.h"

#include <cstdint>
#include <cstdio>

static inline int sycl_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static inline int sycl_u64_mul3_checked(uint64_t a, uint64_t b, uint64_t c,
                                        uint64_t *out) {
    uint64_t tmp = 0;
    return sycl_u64_mul_checked(a, b, &tmp) && sycl_u64_mul_checked(tmp, c, out);
}

static inline int sycl_tensor_has_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static inline int sycl_tensor_has_elems(const ds4_gpu_tensor *t, uint64_t elems,
                                        uint64_t elem_size) {
    uint64_t bytes = 0;
    return sycl_u64_mul_checked(elems, elem_size, &bytes) &&
           sycl_tensor_has_bytes(t, bytes);
}

static inline int sycl_tensor_has_elems2(const ds4_gpu_tensor *t, uint64_t a,
                                         uint64_t b, uint64_t elem_size) {
    uint64_t bytes = 0;
    return sycl_u64_mul3_checked(a, b, elem_size, &bytes) &&
           sycl_tensor_has_bytes(t, bytes);
}

static inline int sycl_tensor_has_f32(const ds4_gpu_tensor *t, uint64_t elems) {
    return sycl_tensor_has_elems(t, elems, sizeof(float));
}

static inline int sycl_tensor_has_i32(const ds4_gpu_tensor *t, uint64_t elems) {
    return sycl_tensor_has_elems(t, elems, sizeof(int32_t));
}

/* Overflow-safe range check against the mmapped model.  Same form as
 * cuda_model_range_fits, rocm/ds4_rocm_runtime.cuh:491. */
static inline int sycl_model_range_fits(uint64_t model_size, uint64_t offset,
                                        uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static inline const char *sycl_model_range_ptr(const void *model_map,
                                               uint64_t offset, uint64_t bytes,
                                               uint64_t model_size,
                                               const char *what) {
    if (!model_map || !sycl_model_range_fits(model_size, offset, bytes)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s range %llu+%llu outside model\n",
                what ? what : "tensor", (unsigned long long)offset,
                (unsigned long long)bytes);
        return nullptr;
    }
    return (const char *)model_map + offset;
}

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
