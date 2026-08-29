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

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <level_zero/zes_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
    /* Transfers ownership rather than copying it, nulling the source so
     * only one side ever frees.  Needed so sycl_stage_host_bytes below can
     * return a guard by value: the guard must already own the allocation
     * before the staging memcpy runs, so the factory cannot build it from
     * a prvalue at the return statement, and returning a named local
     * requires an accessible move constructor even where the compiler
     * elides the move at runtime. */
    sycl_device_scratch_guard(sycl_device_scratch_guard &&other) noexcept
            : q(other.q), p(other.p) {
        other.p = nullptr;
    }
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

/* Stages `bytes` bytes of host memory at `host_ptr` (typically a range
 * inside the read-only model mmap) into fresh device scratch on `q`,
 * waiting for the copy to finish before returning a guard that owns the
 * allocation.  On allocation failure the returned guard wraps a null
 * pointer; callers must check `.p` before using it.  Callers must keep
 * the returned guard alive for as long as its pointer is used by kernels
 * dispatched on `q`.
 *
 * This is the single-buffer host-to-device staging shape used wherever a
 * kernel needs bytes that live in the host mmap.  Per design-spec section
 * 6l, a SYCL kernel cannot dereference that mmap pointer directly: it
 * silently reads zeros and reports success rather than faulting, so this
 * staging is load-bearing correctness code, not an optimisation. */
static inline sycl_device_scratch_guard sycl_stage_host_bytes(
        sycl::queue &q, const void *host_ptr, uint64_t bytes) {
    unsigned char *dptr = sycl::malloc_device<unsigned char>((size_t)bytes, q);
    if (!dptr) return sycl_device_scratch_guard(q, nullptr);
    sycl_device_scratch_guard guard(q, dptr);
    q.memcpy(dptr, host_ptr, (size_t)bytes).wait_and_throw();
    return guard;
}

/* Live free-VRAM query via Level Zero Sysman. Shared by
 * ds4_gpu_args_probe_auto_cuda (sycl/ds4_sycl_mgpu.hpp, the --gpu-vram
 * auto CLI path) and, as future work, sycl_stream_vram_ceiling's live
 * upgrade (sycl/ds4_sycl_streaming.hpp): that function today substitutes a
 * static ceiling (total device memory minus a fixed reserve) because this
 * capability did not exist yet, per its own header comment.  Not rewired
 * here: only the CLI path uses the helper here; wiring the streaming
 * cache to it is future work.
 *
 * Requires ZES_ENABLE_SYSMAN to have been armed before the Level Zero
 * loader's first platform/device enumeration (see the setenv calls in
 * ds4_gpu_init and ds4_gpu_args_probe_auto_cuda); if Sysman was never
 * armed, or the device is not on the Level Zero backend, every zes* call
 * below fails and this returns false -- "no data available", not a
 * fabricated zero, which a caller could otherwise mistake for a genuinely
 * full device.  Sums every memory module Sysman reports rather than
 * assuming exactly one, since the API is written for zero-or-more. */
static inline bool sycl_zes_free_bytes(const sycl::device &d, uint64_t *out_free_bytes) {
    if (!out_free_bytes) return false;
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) return false;
    try {
        const ze_device_handle_t ze_dev =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
        const zes_device_handle_t zes_dev = (zes_device_handle_t)ze_dev;

        uint32_t count = 0;
        if (zesDeviceEnumMemoryModules(zes_dev, &count, nullptr) != ZE_RESULT_SUCCESS ||
            count == 0) {
            return false;
        }
        std::vector<zes_mem_handle_t> mods(count);
        if (zesDeviceEnumMemoryModules(zes_dev, &count, mods.data()) != ZE_RESULT_SUCCESS) {
            return false;
        }

        uint64_t total_free = 0;
        bool any = false;
        for (uint32_t i = 0; i < count; i++) {
            zes_mem_state_t st;
            memset(&st, 0, sizeof(st));
            st.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
            if (zesMemoryGetState(mods[i], &st) == ZE_RESULT_SUCCESS) {
                total_free += st.free;
                any = true;
            }
        }
        if (!any) return false;
        *out_free_bytes = total_free;
        return true;
    } catch (const sycl::exception &) {
        return false;
    }
}

/* Ported literally from f32_to_f16_bits_hip_round
 * (rocm/ds4_rocm_common.cuh:372-390): a hand-rolled F32-to-F16 bit encoder
 * that rounds up whenever the single highest discarded bit is set, with
 * no sticky-bit check of the remaining discarded bits.  This means it
 * rounds an EXACT tie up unconditionally ("round half up"), never to the
 * even neighbour.
 *
 * sycl::half's own device-side float-to-half conversion was measured
 * directly against this oracle on an Intel Arc A770 (Level Zero, oneAPI
 * 2025.3), via tests/test_sycl_fp8_kv.c's
 * test_f16_sycl_half_matches_device_on_non_tie_values and
 * test_f16_exact_tie_sycl_half_diverges_from_oracle: the two agree at
 * every non-tie input tried (subnormal, near-maximum, overflow-to-infinity,
 * mantissa-carry-into-next-exponent), and disagree at a deliberately
 * constructed exact tie, where sycl::half rounds to the even neighbour and
 * this function rounds up.  Using sycl::half for this direction would
 * silently change ring-buffer store results at exact ties relative to the
 * ROCm backend, so this function is ported bit-for-bit instead.
 *
 * Shared by ds4_sycl_fp8_kv.hpp (the KV ring-buffer store) and
 * ds4_sycl_attention_output.hpp (the F16 output-projection entry, spec
 * 6k): both need the encode direction's exact rounding rule, so this lives
 * here rather than being ported a second time. */
static inline uint16_t sycl_f32_to_f16_bits_hip_round(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

/* Work-group tree reduction over one row: every lane in the group passes
 * its own already-accumulated `value`, and every lane gets back the same
 * fully-reduced result.  `combine` must be associative and commutative
 * (`[](float a, float b) { return a + b; }` for a sum, `sycl::fmax` for a
 * max); `local` must be sized to at least the work-group's local range.
 * `it` is accepted as a template parameter rather than a fixed nd_item<N>
 * so this serves both a flat 1D row-per-group launch and a 2D launch that
 * adds a second grid dimension (norm_rope's q/kv selector, fp8_kv's
 * per-row group-of-groups split): both only ever address dimension 0 for
 * the work-group's local id and range, which is all this needs.
 *
 * Every lane must call this, including a lane whose own per-lane loop
 * accumulated zero terms, passing that lane's identity value (0.0f for a
 * sum, 0.0f or -INFINITY for a max) rather than skipping the call: this
 * is what the design spec's section 6b requires and it doubles as the
 * enforcement, since there is no shorter path through this function that
 * leaves a lane's slot unwritten.  On this hardware (Arc A770, Level
 * Zero, oneAPI 2025.3), uninitialised local memory has been observed to
 * read as zero, so no test on this machine can catch a lane skipping its
 * write; the SYCL specification guarantees no such zero-initialisation,
 * and behaviour may differ on other hardware, including the Battlemage
 * devices this backend targets. Do not treat a passing test suite as
 * licence to bypass this. */
template <typename Item, typename Combine>
static inline float sycl_block_row_reduce(Item it, const sycl::local_accessor<float, 1> &local,
                                          float value, Combine combine) {
    const size_t lid = it.get_local_id(0);
    const size_t lsz = it.get_local_range(0);
    local[lid] = value;
    it.barrier(sycl::access::fence_space::local_space);
    for (size_t s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) local[lid] = combine(local[lid], local[lid + s]);
        it.barrier(sycl::access::fence_space::local_space);
    }
    return local[0];
}

}  // namespace
