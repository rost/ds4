#pragma once

/* Argument-validation helpers shared by every SYCL kernel entry point.
 * This is a subset of the cuda_* family in rocm/ds4_rocm_runtime.cuh:376-530,
 * ported as each helper is needed rather than all at once.  Known absent:
 * cuda_tensor_has_f16 and cuda_tensor_has_u16.  No current entry needs
 * them; add the SYCL equivalent of one only when a later kernel actually
 * requires it.  Every one is overflow-safe by
 * construction: ds4 entry points receive sizes from model metadata and
 * must never compute a product or a range that wraps. */

#include "ds4_gpu_mgpu.h"

#include <cstdint>
#include <cstdio>

static inline int sycl_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

/* Maps onto rocm/ds4_rocm_moe_launch.cuh:1-4 (routed_moe_u64_add_checked).
 * Used by the MoE scratch-layout arithmetic and by the streaming cache's
 * per-expert offset validation. */
static inline int sycl_u64_add_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

/* Maps onto rocm/ds4_rocm_moe_launch.cuh:6-9 (routed_moe_align256_checked). */
static inline int sycl_align256_checked(uint64_t v, uint64_t *out) {
    if (!out || v > UINT64_MAX - 255ull) return 0;
    *out = (v + 255ull) & ~(uint64_t)255ull;
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

static inline int sycl_tensor_has_elems3(const ds4_gpu_tensor *t, uint64_t a,
                                         uint64_t b, uint64_t c,
                                         uint64_t elem_size) {
    uint64_t ab = 0, elems = 0, bytes = 0;
    return sycl_u64_mul_checked(a, b, &ab) &&
           sycl_u64_mul_checked(ab, c, &elems) &&
           sycl_u64_mul_checked(elems, elem_size, &bytes) &&
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

/* Q8_0 row layout: blocks of 32 values, 34 bytes per block, a little-endian
 * F16 scale in bytes 0-1 followed by 32 signed int8 values.  See
 * rocm/ds4_rocm_common.cuh:19-63.  This is the shared Q8_0 block-layout
 * helper used by the Q8_0 matmul family in sycl/ds4_sycl_matmul.hpp; it
 * lives here rather than duplicated per-entry so those entries do not each
 * have to touch this file. */
static inline int sycl_q8_0_row_bytes_checked(uint64_t in_dim, uint64_t *row_bytes) {
    if (!row_bytes) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    return sycl_u64_mul_checked(blocks, 34u, row_bytes);
}

/* Dequantises one column of a Q8_0 row.  Device-callable: mirrors the
 * inline decode already used in sycl/ds4_sycl_embedding.hpp, factored out
 * so later Q8_0 matmul kernels can call it instead of repeating it. */
static inline float sycl_q8_0_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 32u;
    const uint32_t idx = col % 32u;
    const unsigned char *bp = row + (size_t)blk * 34u;
    const uint16_t raw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
    const float scale = (float)sycl::bit_cast<sycl::half>(raw);
    const signed char qv = (signed char)bp[2 + idx];
    return scale * (float)qv;
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
