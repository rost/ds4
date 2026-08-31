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

/* The per-tier timeline recorder. Included here rather than from
 * ds4_sycl.cpp's trailing include block because its record points are
 * spread across ds4_sycl_mgpu.hpp, ds4_sycl_moe_owned.hpp and
 * ds4_sycl_commands.hpp, and every one of those already reaches this
 * header. It depends on nothing defined in ds4_sycl.cpp: the two
 * entry points that need a queue or the device list take them as
 * parameters for exactly that reason. */
#include "ds4_sycl_timeline.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <level_zero/zes_api.h>

#include <oneapi/mkl/blas.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
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

/* Every [[sycl::reqd_sub_group_size(N)]] width this backend's kernels
 * require, enumerated directly from the sycl headers (2026-08-31): width
 * 16 appears 25 times (ds4_sycl_moe.hpp x14, ds4_sycl_moe_owned.hpp x6,
 * ds4_sycl_attention.hpp x2, plus one each behind a named constant in
 * ds4_sycl_attention_output.hpp via kGroupedQ8ASubgroupWidth,
 * ds4_sycl_hc.hpp via kHcExpandTileSubgroup and ds4_sycl_matmul.hpp via
 * kQ8_0TileSubgroup); width 32 appears 12 times
 * (ds4_sycl_attention.hpp x4, ds4_sycl_moe.hpp x2,
 * ds4_sycl_shared_expert.hpp x2, ds4_sycl_attention_output.hpp x2,
 * ds4_sycl_indexer.hpp x1, ds4_sycl_router.hpp x1).  Count them by name
 * as well as by literal: three of the width-16 sites are spelled as a
 * constant and a numeric grep alone will miss them.
 *
 * WIDTH 8 IS DELIBERATELY ABSENT AND MUST STAY ABSENT. Xe2 (Battlemage,
 * Arc Pro B60) reports sub-group widths {16, 32} only; Xe1 (Alchemist,
 * A770) also offered 8. Requiring 8 refuses every Xe2 card outright, which
 * is what this backend did until B60 silicon was first available to test
 * against. Kernels that want an 8-lane cooperative group get one by
 * running in a 16-wide sub-group and treating it as two independent
 * 8-lane halves (sycl_moe_lane8, ds4_sycl_moe.hpp), which is the same
 * masked-partition scheme CUDA uses inside its own 32-wide warps.
 *
 * Per spec 6m, [[sycl::reqd_sub_group_size(N)]] is NOT a hard requirement
 * on this stack: a device that cannot honour N is not refused by the
 * driver, which silently runs a hardware-feasible width instead. Every
 * kernel carrying this annotation then performs a shuffle reduction whose
 * tree depth is compiled in for width N (see sycl_attn_subgroup_sum<N> in
 * ds4_sycl_attention.hpp and the matching shape in the MoE kernels), so a
 * silent width mismatch reduces over the wrong set of lanes and returns a
 * plausible, wrong, answer with no error anywhere. The device-discovery
 * check in ds4_gpu_init (ds4_sycl.cpp) is this backend's only guard against
 * that failure mode.
 *
 * IF YOU ADD A NEW reqd_sub_group_size(N) ANNOTATION AT A WIDTH NOT ALREADY
 * LISTED HERE, ADD THE WIDTH HERE TOO, or ds4_gpu_init will not check for
 * it and a device unable to honour it will silently corrupt that kernel's
 * output. */
static constexpr uint32_t kRequiredSubGroupWidths[] = {16u, 32u};

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

/* Defined for real in sycl/ds4_sycl_model_cache.hpp, included well after
 * this header (which is included before g_devices and g_current_tier
 * exist -- see ds4_sycl.cpp's own comments on why several headers are
 * included at the bottom of that file).  A forward-declared static
 * function defined later in the same translation unit is ordinary C++;
 * every sycl header file is #include-d directly into ds4_sycl.cpp, so
 * this and its later definition are one translation unit, the same
 * pattern ds4_sycl_streaming.hpp's own forward-declared teardown hooks in
 * ds4_sycl.cpp already use.
 *
 * Returns a device pointer valid for at least `bytes` bytes from the
 * cache maintained for the CURRENT tier when [offset, offset+bytes) is
 * contained in some range previously cached (via ds4_gpu_cache_model_range)
 * for this exact `model_map`, sub-ranges included (CUDA's
 * cuda_model_range_ptr containment logic, ds4_cuda.cu:703); nullptr on a
 * cache miss, which the caller falls back to today's host-mmap-plus-
 * per-call-staging behaviour for. */
static const char *sycl_model_cache_resolve(const void *model_map,
                                            uint64_t offset, uint64_t bytes);

/* The multi-GPU placement path installs its per-tier weights in a
 * SEPARATE cache from ds4_gpu_cache_model_range's (see
 * sycl/ds4_sycl_placement.hpp): ds4.c's multi-tier startup calls
 * ds4_gpu_device_cache_tensors, which fills g_placement_tier, while
 * ds4_gpu_cache_model_range fills g_sycl_model_cache_tier.  Resolution
 * has to consult both or every multi-GPU run re-stages, per call, weights
 * that are already resident -- which is exactly what it did: a profiled
 * 14-GPU run reported model-cache hits=0 misses=1434 per forward pass,
 * with 86% of all device time going to the redundant staging copies.
 * Defined in ds4_sycl_placement.hpp, which owns the memory it hands back
 * and frees it itself; nothing here takes ownership. */
static const char *sycl_placement_cache_resolve(const void *model_map,
                                                uint64_t offset, uint64_t bytes);

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
    const char *cached = sycl_model_cache_resolve(model_map, offset, bytes);
    if (cached) return cached;
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

/* Dot-products one whole Q8_0 block against the activations at
 * xs[base .. base+n), returning scale * sum.  n is 32 for every block but
 * the last one of an in_dim that is not a multiple of 32.
 *
 * This exists because sycl_q8_0_dequant above, called once per element
 * from a matmul inner loop, re-reads the block's 2-byte F16 scale and
 * re-converts it to F32 for every one of the block's 32 values, and gives
 * the compiler no wide load to form for the int8 payload: consecutive
 * columns belong to consecutive lanes there, so each lane issues one
 * single-byte load per multiply-add.  Reading a whole block per lane
 * instead converts the scale once and lets the 32 int8 codes arrive as
 * one 32-byte copy.  Measured on an A770 (sycl_q8_0_matmul_*, n_tok 1,
 * out_dim 4096, ms per call, per-element against per-block): in_dim 4096
 * 0.0907 against 0.0829, in_dim 7168 0.1792 against 0.1279.
 *
 * The payload copy is a __builtin_memcpy and NOT a uint32_t reinterpret
 * cast: blocks start at 34*b, so a block is only 2-byte aligned and half
 * of them are 2 mod 4.  memcpy states the real alignment and lets the
 * compiler pick loads the device can actually issue; a reinterpret cast
 * would assert an alignment the layout does not have.
 *
 * XAcc is anything indexable by a uint32_t -- a raw activation pointer or
 * a local_accessor -- so the same helper serves kernels that read the
 * activation row from global memory and kernels that stage it in local
 * memory. */
static inline float sycl_q8_0_block_scale(const unsigned char *bp) {
    uint16_t raw;
    __builtin_memcpy(&raw, bp, sizeof(raw));
    return (float)sycl::bit_cast<sycl::half>(raw);
}

template <typename XAcc>
static inline float sycl_q8_0_block_dot(const unsigned char *bp, const XAcc &xs,
                                        uint32_t base, uint32_t n) {
    const float scale = sycl_q8_0_block_scale(bp);
    float acc = 0.0f;
    if (n == 32u) {
        signed char qv[32];
        __builtin_memcpy(qv, bp + 2, sizeof(qv));
        for (uint32_t i = 0; i < 32u; i++) acc += xs[base + i] * (float)qv[i];
    } else {
        for (uint32_t i = 0; i < n; i++) {
            acc += xs[base + i] * (float)(signed char)bp[2 + i];
        }
    }
    return scale * acc;
}

/* The all-integer counterpart of sycl_q8_0_block_dot, for the callers
 * whose activations are already quantised to int8 against a per-block
 * scale of their own: the dot product of one Q8_0 block's codes against
 * `n` of those, left unscaled so the caller applies both scales once.
 * Same reason for the wide copies as sycl_q8_0_block_dot above -- the
 * loop is otherwise 64 single-byte loads per block -- and same
 * __builtin_memcpy rather than a reinterpret cast, since a Q8_0 block
 * starts at 34*b and is only 2-byte aligned.  Widening the loads does not
 * move a single result: the products and their sum are exact in int, and
 * the order is unchanged. */
static inline int sycl_q8_0_block_dot_i8(const unsigned char *bp,
                                         const int8_t *xq, uint32_t n) {
    int dot = 0;
    if (n == 32u) {
        int8_t qv[32], xv[32];
        __builtin_memcpy(qv, bp + 2, sizeof(qv));
        __builtin_memcpy(xv, xq, sizeof(xv));
        for (uint32_t i = 0; i < 32u; i++) dot += (int)qv[i] * (int)xv[i];
    } else {
        for (uint32_t i = 0; i < n; i++) {
            dot += (int)(signed char)bp[2 + i] * (int)xq[i];
        }
    }
    return dot;
}

/* Q4_K row layout: cuda_block_q4_K, ds4_rocm.cu:72-77 (also the routed-MoE's
 * own sycl_block_q4_K in ds4_sycl_moe.hpp before this move): 256-value
 * superblock, F16 scale `d` and F16 min `dmin`, 12 bytes of packed 6-bit
 * (scale, min) codes for 8 sub-blocks of 32, 128 bytes of packed 4-bit
 * codes.  Moved here from ds4_sycl_moe.hpp (the routed-MoE Q4_K path)
 * because the dense matmul and attention-output entries need the
 * exact same block layout and scale/min decode; ds4_sycl_moe.hpp now
 * references these definitions instead of keeping its own copy, per this
 * project's standing rule against a second Q4_K dequantiser. */
static inline int sycl_q4_k_row_bytes_checked(uint64_t in_dim, uint64_t *row_bytes) {
    if (!row_bytes) return 0;
    const uint64_t blocks = (in_dim + 255u) / 256u;
    return sycl_u64_mul_checked(blocks, 144u, row_bytes);
}

struct sycl_block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

/* dev_q4_K_get_scale_min, moe.cuh:250-262 (and ds4.c's own CPU oracle,
 * q4_k_get_scale_min, ds4.c:3524-3532, which computes the identical two
 * 6-bit codes from the same 12-byte layout): sub-block j in [0,8) packs its
 * scale and min across the 12-byte `scales` array two different ways
 * depending on whether j is in the first or second half. */
static inline void sycl_dev_q4_k_get_scale_min(uint32_t j, const uint8_t *scales,
                                               uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

/* Dequantises one column of a Q4_K row: value = d*sc*nibble - dmin*m, the
 * same per-element formula ds4.c's own ds4_vec_dot_q4_K_f32 (ds4.c:3709-
 * 3736) accumulates over a whole row, and the same decomposition the
 * routed-MoE's device dot product (sycl_dev_dot_q4_k_q8_k_block,
 * ds4_sycl_moe.hpp) uses per sub-block.  Sub-block j = col/32 within the
 * 256-wide superblock picks
 * its (scale, min) pair via sycl_dev_q4_k_get_scale_min; sub-blocks pair up
 * two-to-a-byte-range (j=2m holds the low nibble of qs[m*32 .. m*32+31],
 * j=2m+1 the high nibble of the SAME bytes), matching moe.cuh's
 * dev_dot_q4_32 byte_off = (j>>1)*32 / shift = (j&1)?4:0. */
static inline float sycl_q4_k_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 256u;
    const uint32_t idx = col % 256u;
    const sycl_block_q4_K *bp =
            (const sycl_block_q4_K *)(row + (size_t)blk * 144u);
    const uint32_t j = idx / 32u;
    const uint32_t pos = idx % 32u;
    uint8_t sc = 0, m = 0;
    sycl_dev_q4_k_get_scale_min(j, bp->scales, &sc, &m);
    const uint32_t byte_off = (j >> 1u) * 32u + pos;
    const uint8_t raw = bp->qs[byte_off];
    const uint8_t nib = (j & 1u) ? (uint8_t)(raw >> 4u) : (uint8_t)(raw & 0x0fu);
    const float d = (float)sycl::bit_cast<sycl::half>(bp->d);
    const float dmin = (float)sycl::bit_cast<sycl::half>(bp->dmin);
    return d * (float)sc * (float)nib - dmin * (float)m;
}

/* Q4_0 row layout: the standard GGUF q4_0 block (gguf_types[2] in ds4.c,
 * block_elems=32, block_bytes=18).  Unlike Q4_K/Q8_0, nothing in ds4.c,
 * ds4_cuda.cu or rocm/ has ever decoded this format before now: it is
 * declared as a legal dense-quant type (tensor_type_is_dense_quant, ds4.c)
 * but ds4_cuda.cu's own ds4_gpu_matmul_quant_tensor rejects every type but
 * Q8_0/F16, and ds4_metal.m's implementation is a bespoke, non-portable
 * Metal compute kernel with no scalar reference to transcribe.  This is
 * therefore the standard llama.cpp/GGUF q4_0 layout, not a port: an F16
 * scale `d` followed by 16 bytes of packed 4-bit codes for 32 values, byte
 * i holding value i in its low nibble and value i+16 in its high nibble,
 * decoded as a symmetric signed range via a zero-point of 8 (v = d *
 * (nibble - 8)). */
static inline int sycl_q4_0_row_bytes_checked(uint64_t in_dim, uint64_t *row_bytes) {
    if (!row_bytes) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    return sycl_u64_mul_checked(blocks, 18u, row_bytes);
}

static inline float sycl_q4_0_dequant(const unsigned char *row, uint32_t col) {
    const uint32_t blk = col / 32u;
    const uint32_t idx = col % 32u;
    const unsigned char *bp = row + (size_t)blk * 18u;
    const uint16_t draw = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
    const float d = (float)sycl::bit_cast<sycl::half>(draw);
    const unsigned char *qs = bp + 2;
    const uint8_t byte = (idx < 16u) ? qs[idx] : qs[idx - 16u];
    const uint8_t nib = (idx < 16u) ? (uint8_t)(byte & 0x0fu) : (uint8_t)(byte >> 4u);
    return d * ((float)nib - 8.0f);
}

/* Command-graph capture hooks, defined in sycl/ds4_sycl_graph.hpp.
 * Declared here because the two pieces of shared infrastructure that must
 * behave differently while a batch is recording -- the scratch guard just
 * below and sycl_batch_wait further down -- both live in this header,
 * while how a batch is built and submitted is that header's business
 * alone. Capture is off unless DS4_SYCL_GRAPH is set, and with it off
 * sycl_graph_batch_recording is constantly false, so every use below
 * collapses to exactly the un-captured behaviour it had before. */
static bool sycl_graph_batch_recording(void);
static void sycl_graph_batch_defer_free(sycl::queue &q, void *p);
static bool sycl_graph_batch_flush(void);
static bool sycl_graph_batch_suspend(void);
static bool sycl_graph_batch_resume(void);

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
    /* False when `p` is a pointer this guard does not own -- a weight
     * range the model-range cache (sycl/ds4_sycl_model_cache.hpp) already
     * made device-resident, passed through by sycl_stage_host_bytes below
     * instead of staged. Freeing such a pointer here would be exactly the
     * use-after-free shape spec 6g warns about: the cache, not this call,
     * owns that memory and frees it exactly once at teardown. Defaults to
     * true so every pre-existing call site (a real allocation) is
     * unaffected. */
    bool         owns;
    sycl_device_scratch_guard(sycl::queue &queue, void *ptr, bool owns_ptr = true)
            : q(queue), p(ptr), owns(owns_ptr) {}
    /* Transfers ownership rather than copying it, nulling the source so
     * only one side ever frees.  Needed so sycl_stage_host_bytes below can
     * return a guard by value: the guard must already own the allocation
     * before the staging memcpy runs, so the factory cannot build it from
     * a prvalue at the return statement, and returning a named local
     * requires an accessible move constructor even where the compiler
     * elides the move at runtime. */
    sycl_device_scratch_guard(sycl_device_scratch_guard &&other) noexcept
            : q(other.q), p(other.p), owns(other.owns) {
        other.p = nullptr;
        other.owns = false;
    }
    /* Whether going out of scope will actually release device memory.
     * The kernel-entry wait that keeps scratch alive until its reader
     * finishes is needed exactly when this is true, so both the
     * destructor below and sycl_any_scratch_frees ask the same question
     * of the same predicate rather than each spelling it out. */
    bool frees() const { return p != nullptr && owns; }
    ~sycl_device_scratch_guard() {
        /* Destructors are implicitly noexcept: if sycl::free throws while
         * we are already unwinding from another exception (e.g. the
         * memcpy's wait_and_throw), an escaping exception here would call
         * std::terminate instead of surfacing a clean failure.  There is
         * no meaningful recovery from a failed free during unwinding, so
         * log and swallow rather than let it propagate. */
        if (!frees()) return;
        /* Freeing here is safe only because the kernels reading this
         * scratch have already completed by the time the guard goes out of
         * scope -- every entry point that stages scratch waits before
         * returning whenever a guard of its own reaches this branch
         * (sycl_any_scratch_frees below). A recording batch breaks that
         * assumption: its
         * commands have been recorded but not yet run, so the batch takes
         * the pointer and frees it once the submission that reads it has
         * completed (sycl/ds4_sycl_graph.hpp). */
        if (sycl_graph_batch_recording()) {
            sycl_graph_batch_defer_free(q, p);
            return;
        }
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

/* True when at least one of `guards` will release device memory when the
 * entry point holding them returns.
 *
 * A kernel entry that stages host bytes into device scratch must drain
 * its queue before returning, or the guard frees that scratch out from
 * under a kernel still reading it. That drain is the entry's only reason
 * to wait: it computes nothing the host reads, and the in_order queue
 * already orders it against whatever the caller submits next. Since
 * sycl_stage_host_bytes started handing back a non-owning guard for any
 * weight range the model-range cache had already made device-resident,
 * the common decode case frees nothing at all, and the drain buys nothing
 * while costing the dispatch latency of the next kernel, which the queue
 * would otherwise have overlapped with this one. Guard the wait with this
 * and a fully cache-resident decode stops draining after every kernel. */
template <typename... Guards>
static inline bool sycl_any_scratch_frees(const Guards &...guards) {
    return (... || guards.frees());
}

/* The drain a kernel entry performs purely so its own scratch guards can
 * free safely when it returns.
 *
 * Distinct from sycl_batch_wait below: that one stands in for a wait the
 * caller genuinely needs, and delivers it under capture by flushing the
 * batch. This one carries no such meaning. Its only job is to keep staged
 * scratch alive until the kernels reading it have finished, and a
 * recording batch already guarantees exactly that by deferring the free
 * past the submission that reads it (sycl/ds4_sycl_graph.hpp), so under
 * capture there is nothing left to wait for.
 *
 * That distinction is what makes capture worth anything on a
 * cache-resident decode, where these are most of the waits that remain:
 * flushing at every one of them chops a token into dozens of graphs a
 * handful of commands long, and a graph that short costs more to finalize
 * than the submissions it saves. With capture off this is exactly the
 * guarded drain it replaces. */
template <typename... Guards>
static inline void sycl_scratch_release_wait(sycl::queue &q, const Guards &...guards) {
    if (sycl_graph_batch_recording()) return;
    if (!sycl_any_scratch_frees(guards...)) return;
    q.wait_and_throw();
}

/* The wait a kernel entry performs before scratch guards free under it.
 *
 * With capture off this is exactly the wait_and_throw it replaces. With a
 * batch recording it flushes instead: a queue in the recording state
 * throws on any wait, so the commands recorded so far are submitted and
 * waited on, which is what the caller actually asked for. Because every
 * tier's queue is in_order, waiting for the whole batch is strictly
 * stronger than waiting for one event inside it, so the substitution can
 * only over-synchronize.
 *
 * Call sites keep naming their own event even though the batch path
 * ignores it: the event still documents which kernel the wait belongs to,
 * and it is what the un-captured path waits on. */
static inline void sycl_batch_wait(sycl::queue &q) {
    if (sycl_graph_batch_recording()) {
        sycl_graph_batch_flush();
        return;
    }
    q.wait_and_throw();
}

static inline void sycl_batch_wait(const sycl::event &ev) {
    if (sycl_graph_batch_recording()) {
        sycl_graph_batch_flush();
        return;
    }
    const_cast<sycl::event &>(ev).wait_and_throw();
}

/* True when `ptr` is already a sycl::usm::alloc::device allocation on
 * `q`'s context -- e.g. a weight range the model-range cache
 * (sycl/ds4_sycl_model_cache.hpp) resolved to a cached device pointer,
 * rather than an ordinary host pointer into the model mmap or a
 * host-backed test fixture.  get_pointer_type returns
 * sycl::usm::alloc::unknown for any pointer not allocated as USM in this
 * context (per the SYCL specification), never device, so this is an exact
 * test, not a heuristic, and is safe to call on an arbitrary host
 * pointer. Used by the staging helpers below to skip a redundant
 * host-to-device copy: the three helpers pass through
 * when handed a pointer that is already device-resident. */
static inline bool sycl_ptr_is_device_resident(sycl::queue &q, const void *ptr) {
    if (!ptr) return false;
    return sycl::get_pointer_type(ptr, q.get_context()) == sycl::usm::alloc::device;
}

/* Test/report-only instrumentation: per-kernel device
 * profiling used to measure where a layer-eval's wall-clock time actually
 * goes, before deciding whether draining the queue after every ABI entry
 * is worth deferring. Disabled (one bool check per call site) unless
 * DS4_SYCL_PROFILE is set before ds4_gpu_init runs, since
 * property::queue::enable_profiling() is a queue-construction-time
 * property (ds4_sycl.cpp, ds4_sycl_build_devices) and cannot be turned on
 * after the fact. g_sycl_profile_enabled is a separate runtime switch so a
 * test can reset and re-enable accumulation around just the loop it wants
 * measured, without the setup phases before it polluting the totals.
 *
 * Every wait_and_throw call site instrumented for this measurement now
 * captures its own kernel's sycl::event explicitly (either one that was
 * already a named local, or a newly introduced one) rather than reaching
 * for queue::ext_oneapi_get_last_event(). Written when these queues were
 * still out-of-order, where that DPC++ extension throws; every tier's
 * queue is now property::queue::in_order() (ds4_sycl.cpp,
 * ds4_sycl_queue_properties), so the extension would no longer throw, but
 * explicit per-kernel capture is kept anyway: it names exactly which
 * event this measurement is timing at the call site, which reads better
 * than a queue-global "whatever was last submitted" even where both are
 * legal.
 *
 * Deliberately does NOT compute an inter-kernel "gap" by subtracting one
 * event's command_submit from a previous event's command_end. A first
 * attempt at that produced per-layer gap sums many times larger than the
 * wall-clock total for the whole loop they came from, which is impossible
 * if the two events are as close together as "consecutive kernels in one
 * layer-eval" implies: the likely cause is Level Zero's device timestamp
 * counter, which is narrow enough to wrap within the loop's runtime, so a
 * naive subtraction between two events that are not actually adjacent
 * (this measurement's coverage is real but partial: some call sites launch
 * their kernel through a helper this instrumentation does not reach, so
 * the "previous" recorded event can be from several real kernels earlier)
 * aliases to a huge, meaningless value. Reporting a number here would be
 * worse than not measuring it. Summing one event's own (command_end -
 * command_start) does not have this problem: both timestamps come from the
 * same command's own short window and cannot span a wrap in practice. */
static bool     g_sycl_profile_enabled = false;
static uint64_t g_sycl_profile_kernel_ns = 0;
static uint64_t g_sycl_profile_kernel_count = 0;

static inline void ds4_sycl_profile_record(const sycl::event &ev) {
    if (!g_sycl_profile_enabled) return;
    try {
        const uint64_t start_ns = ev.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        const uint64_t end_ns = ev.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        if (end_ns > start_ns) g_sycl_profile_kernel_ns += (end_ns - start_ns);
        g_sycl_profile_kernel_count++;
    } catch (const sycl::exception &) {
        /* Profiling info unavailable for this event (queue built without
         * enable_profiling, or the event has no device command): skip
         * rather than let a measurement-only probe fail a real call. */
    }
}

extern "C" void ds4_sycl_test_profile_enable(int enable) {
    g_sycl_profile_enabled = enable != 0;
}

/* Per-kernel-family device time, layered on top of the
 * aggregate counters above. A fixed small table of named buckets keyed by
 * a call site's own string literal (compared by content, not pointer
 * identity, so two call sites can legitimately share one bucket): this is
 * test/report-only instrumentation used to RANK the dense-matmul kernel
 * family before deciding which one is worth rewriting first, not a
 * runtime hot path, so an O(buckets) linear scan per call is fine. Only
 * the call sites being ranked pass a name; every other
 * call site keeps calling the unnamed ds4_sycl_profile_record above,
 * which still counts toward the aggregate total the existing gguf-load
 * measurement already reports.
 *
 * A later pass raised the cap from 16 to 160: naming every remaining
 * kernel in MoE routing and expert compute, the router, the indexer, hc,
 * norm/rope, the compressor, attention decode/output, the shared expert
 * and fp8 KV added on the order of 100 more distinct bucket names on top
 * of the matmul family named above, comfortably exceeding the old cap.
 * Past the cap a bucket is silently dropped (see the check below), which
 * would quietly re-introduce exactly the "counted in the total but not
 * named" gap this instrumentation exists to close, so the cap must stay
 * above the real distinct-name count rather than be raised just enough to
 * fit today's count exactly. */
struct sycl_profile_bucket {
    const char *name;
    uint64_t    ns;
    uint64_t    count;
};
static constexpr int kSyclProfileMaxBuckets = 160;
static sycl_profile_bucket g_sycl_profile_buckets[kSyclProfileMaxBuckets];
static int g_sycl_profile_bucket_count = 0;

static inline void ds4_sycl_profile_record_named(const char *name, const sycl::event &ev) {
    if (!g_sycl_profile_enabled) return;
    ds4_sycl_profile_record(ev);
    try {
        const uint64_t start_ns = ev.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        const uint64_t end_ns = ev.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        if (end_ns <= start_ns) return;
        const uint64_t dur = end_ns - start_ns;
        for (int i = 0; i < g_sycl_profile_bucket_count; i++) {
            if (strcmp(g_sycl_profile_buckets[i].name, name) == 0) {
                g_sycl_profile_buckets[i].ns += dur;
                g_sycl_profile_buckets[i].count++;
                return;
            }
        }
        if (g_sycl_profile_bucket_count < kSyclProfileMaxBuckets) {
            g_sycl_profile_buckets[g_sycl_profile_bucket_count] = {name, dur, 1};
            g_sycl_profile_bucket_count++;
        }
    } catch (const sycl::exception &) {
        /* Same reasoning as ds4_sycl_profile_record above: profiling info
         * unavailable for this event should not fail a real call. */
    }
}

extern "C" void ds4_sycl_test_profile_reset(void) {
    g_sycl_profile_kernel_ns = 0;
    g_sycl_profile_kernel_count = 0;
    g_sycl_profile_bucket_count = 0;
}
extern "C" uint64_t ds4_sycl_test_profile_kernel_ns(void) { return g_sycl_profile_kernel_ns; }
extern "C" uint64_t ds4_sycl_test_profile_kernel_count(void) { return g_sycl_profile_kernel_count; }
extern "C" uint64_t ds4_sycl_test_profile_bucket_count(void) {
    return (uint64_t)g_sycl_profile_bucket_count;
}
extern "C" const char *ds4_sycl_test_profile_bucket_name(uint64_t i) {
    return i < (uint64_t)g_sycl_profile_bucket_count ? g_sycl_profile_buckets[i].name : "";
}
extern "C" uint64_t ds4_sycl_test_profile_bucket_ns(uint64_t i) {
    return i < (uint64_t)g_sycl_profile_bucket_count ? g_sycl_profile_buckets[i].ns : 0;
}
extern "C" uint64_t ds4_sycl_test_profile_bucket_calls(uint64_t i) {
    return i < (uint64_t)g_sycl_profile_bucket_count ? g_sycl_profile_buckets[i].count : 0;
}

/* N-wide sub-group tree reduction via shuffle: every lane passes its own
 * partial sum and gets back the fully-reduced total. Used by the tiled
 * dense-matmul kernels in ds4_sycl_matmul.hpp for the sub-group
 * cooperative reduction over one output row's dot product. Mirrors the
 * shuffle shape of ds4_sycl_moe.hpp's own reqd_sub_group_size(N) row
 * reductions (sycl_moe_subgroup_sum); kept as a separate small copy here,
 * rather than shared with that file, so callers of this header do not
 * take on a dependency on ds4_sycl_moe.hpp's own include position in the
 * single translation unit that assembles every sycl header file. */
template <int N>
static inline float sycl_subgroup_sum(sycl::sub_group sg, float v) {
    for (int offset = N >> 1; offset > 0; offset >>= 1) {
        v += sycl::shift_group_left(sg, v, (uint32_t)offset);
    }
    return v;
}

/* ---- Batched launches that would exceed DPC++'s fits-in-int range ------
 *
 * DPC++ refuses a launch whose global range does not fit in an int:
 * "Provided range and/or offset does not fit in int. Pass
 * `-fno-sycl-id-queries-fit-in-int' to remove this limit."  The check
 * (sycl/detail/id_queries_fit_in_int.hpp) is on the LINEARISED global
 * size, not on each dimension: a 1D range's single value, and for 2D and
 * 3D the PRODUCT of every dimension, with the local range not checked at
 * all.  So a launch whose every dimension looks modest still throws once
 * they multiply past INT32_MAX.
 *
 * Two entries did exactly that on a real 16665-token prompt.  The Q8_0
 * dense matmul (ds4_sycl_matmul.hpp) crosses the limit at the vocabulary
 * head, 129280/16 work-groups of 256 against a 4096-token prefill chunk.
 * ds4_gpu_indexer_scores_prefill_tensor (ds4_sycl_indexer.hpp) launches
 * (n_comp * 256) x n_tokens, which at 4096 tokens crosses once n_comp
 * passes 2048, i.e. once the prompt passes 8192 positions at compress
 * ratio 4; that is why the failure moved two layers further along after
 * the matmul was fixed, layers 0 and 1 being Flash's two uncompressed
 * ones.
 *
 * Building the whole translation unit with
 * -fno-sycl-id-queries-fit-in-int would remove the check, but the check is
 * a loud, clean failure and is not the only thing standing in the way: it
 * also widens every id query in every kernel here, so any kernel still
 * computing an index in 32 bits would wrap SILENTLY instead of failing.
 * Splitting the launch keeps both properties, and leaves the kernels
 * untouched.
 */

/* How many work-groups one launch may carry along the dimension being
 * split, so that the whole global range stays inside the limit.
 * `items_per_group` is what one such group contributes to the linearised
 * size: its own local size along that dimension times the global size of
 * every other dimension.
 *
 * Never returns zero.  Zero would make the caller's loop make no progress
 * and spin forever; a single group that alone exceeds the limit is emitted
 * anyway and throws, which is the same clean failure as before and far
 * better than a hang. */
static inline uint32_t sycl_range_split_max_groups(uint64_t items_per_group) {
    if (items_per_group == 0u) return 1u;
    const uint64_t limit = (uint64_t)INT32_MAX / items_per_group;
    if (limit == 0u) return 1u;
    return limit > UINT32_MAX ? UINT32_MAX : (uint32_t)limit;
}

/* Test-only override of the split above: nonzero forces that many
 * work-groups per launch regardless of the real limit.  The shapes that
 * genuinely overflow cannot be allocated on any development machine, so
 * without this the split loop only ever runs one iteration and the seam
 * between two chunks is never executed.  Same precedent, and the same
 * reason, as ds4_sycl_attention.hpp's `forced_tile_tokens`: force the
 * split directly at a small, fast shape rather than leave the loop
 * unexercised.  Zero (the default) is the production path. */
static uint32_t g_sycl_range_split_forced_groups = 0;

/* How many launches the most recent split emitted.  A seam test that
 * forces a small chunk is worthless if the forcing silently stops working
 * -- it would then run one launch, see the right answers and pass while
 * testing nothing, which is the exact failure mode SYCL.md's ablation
 * section catalogues.  Asserting this counter is what makes the seam test
 * notice. */
static uint64_t g_sycl_range_split_last_chunks = 0;

extern "C" void ds4_sycl_test_range_split_force_groups(uint32_t groups) {
    g_sycl_range_split_forced_groups = groups;
}

extern "C" uint64_t ds4_sycl_test_range_split_last_chunks(void) {
    return g_sycl_range_split_last_chunks;
}

extern "C" uint32_t ds4_sycl_test_range_split_max_groups(uint64_t items_per_group) {
    return sycl_range_split_max_groups(items_per_group);
}

/* Submits `full` as one launch when its global range already fits in an
 * int, and as consecutive slices along `split_dim` when it does not.
 *
 * `submit(chunk, group_base, group_count)` receives one slice's own
 * nd_range plus which work-groups along `split_dim` it covers.  Every
 * other dimension, and the local range, are exactly the caller's own, so
 * a kernel's [[sycl::reqd_sub_group_size]] and its local-memory shape
 * carry across a split unchanged.  Callers express a slice by offsetting
 * their base pointers by `group_base`, which is the first token for every
 * caller here since each token owns one work-group along the split
 * dimension; the kernel bodies need no change at all.
 *
 * Returns the last slice's event.  Every tier's queue is in_order
 * (ds4_sycl.cpp), so that one event completing means all the slices have,
 * which is what lets callers keep their single trailing wait. */
template <int Dims, typename Submit>
static inline sycl::event sycl_launch_range_split(const sycl::nd_range<Dims> &full,
                                                  int split_dim, Submit submit) {
    const sycl::range<Dims> global = full.get_global_range();
    const sycl::range<Dims> local = full.get_local_range();
    const uint64_t group_size = local[split_dim];
    uint64_t items_per_group = group_size;
    for (int d = 0; d < Dims; d++) {
        if (d != split_dim) items_per_group *= global[d];
    }
    const uint64_t n_groups = group_size ? global[split_dim] / group_size : 0u;
    const uint64_t max_groups = g_sycl_range_split_forced_groups != 0u
                                        ? (uint64_t)g_sycl_range_split_forced_groups
                                        : (uint64_t)sycl_range_split_max_groups(items_per_group);

    sycl::event ev;
    uint64_t chunks = 0;
    for (uint64_t base = 0; base < n_groups; base += max_groups) {
        const uint64_t take = n_groups - base < max_groups ? n_groups - base : max_groups;
        sycl::range<Dims> chunk = global;
        chunk[split_dim] = (size_t)(take * group_size);
        ev = submit(sycl::nd_range<Dims>(chunk, local), base, take);
        chunks++;
    }
    g_sycl_range_split_last_chunks = chunks;
    return ev;
}

/* Test/report-only instrumentation: cumulative bytes actually
 * copied host-to-device by sycl_stage_host_bytes below (never incremented
 * on its pass-through branch). Since an audit confirmed every
 * weight-staging call site in this backend funnels through this helper or
 * through sycl_moe_stage_weights / sycl_moe_stage_selected_experts (both
 * of which call this helper internally for their own device crossings),
 * this one counter captures essentially all host-to-device weight
 * traffic in the backend, letting a test measure real bytes-per-token
 * with the model-range cache on versus off. Not part of the ABI. */
static uint64_t g_sycl_stage_host_bytes_total = 0;

extern "C" uint64_t ds4_sycl_test_stage_host_bytes_total(void) {
    return g_sycl_stage_host_bytes_total;
}
extern "C" void ds4_sycl_test_stage_host_bytes_reset(void) {
    g_sycl_stage_host_bytes_total = 0;
}

/* Copies `bytes` bytes from `host_src` to the existing device allocation
 * `dev_dst`, chunked through a reused, bounded, always-resident heap
 * buffer rather than handed straight to `q.memcpy` as `host_src`.
 *
 * `host_src` is typically a pointer into the read-only model mmap: a
 * file-backed mapping whose pages are faulted in lazily, by the CPU, on
 * first touch. A queue.memcpy() reading directly from such a pointer asks
 * the GPU's DMA engine to read host memory that may not be resident yet,
 * and on this Level Zero stack that read does not block on the page
 * fault the way an ordinary CPU load would: it silently completes
 * anyway, with whatever was in the not-yet-faulted page (observed as all
 * zeros) rather than the file's real contents. Found via real-hardware
 * testing against a genuine 1.83 GiB GGUF file that no earlier test on
 * this backend was large enough to reach; reproduced with as little as
 * ~25 MiB copied from an offset the CPU had never previously touched,
 * and never reproduced from anonymous (malloc/calloc) host memory, which
 * has no lazy backing store to race against. This is spec 6l's exact
 * symptom -- a kernel reading unstaged host memory reads zeros with no
 * error -- one layer lower: here the "kernel" is the DMA engine driving
 * this very copy, and the "unstaged" memory is a host page the CPU has
 * not faulted in. Routing every byte through `stage` first forces each
 * page through a synchronous CPU memcpy, which faults it in; the GPU
 * then only ever reads `stage`, always anonymous and always resident.
 * Chunked so a very large single range (a whole per-layer MoE table, or
 * a multi-hundred-MiB merged model-cache span) does not double the
 * caller's peak host memory use by staging it all at once.
 *
 * Every direct `queue.memcpy` in this backend whose source can be a
 * pointer into the model mmap must go through this helper instead: that
 * is sycl_stage_host_bytes below, ds4_gpu_cache_model_range
 * (sycl/ds4_sycl_model_cache.hpp) and ds4_gpu_device_cache_tensors
 * (sycl/ds4_sycl_placement.hpp, the per-device selective cache,
 * which has the identical shape and was carrying the identical bug,
 * simply never exercised by a test large enough to reach it before now).
 * Throws sycl::exception on a failed device copy, exactly like a bare
 * `queue.memcpy(...).wait_and_throw()` would, so callers keep their
 * existing try/catch. */
static inline void sycl_copy_host_to_device_paged_safe(
        sycl::queue &q, void *dev_dst, const void *host_src, uint64_t bytes) {
    constexpr uint64_t kChunk = 64ull * 1024ull * 1024ull;
    std::vector<unsigned char> stage((size_t)(bytes < kChunk ? bytes : kChunk));
    const unsigned char *src = (const unsigned char *)host_src;
    unsigned char       *dst = (unsigned char *)dev_dst;
    for (uint64_t done = 0; done < bytes; done += kChunk) {
        const uint64_t n = bytes - done < kChunk ? bytes - done : kChunk;
        memcpy(stage.data(), src + done, (size_t)n);
        sycl::event _ds4_prof_ev18 = q.memcpy(dst + done, stage.data(), (size_t)n);
        /* Genuinely load-bearing, not a scratch-lifetime wait: `stage` is
         * one host buffer reused by every chunk, so the copy out of it
         * must have completed before the next iteration overwrites it.
         * Under command-graph capture this is the one wait shape that must
         * still force a real flush rather than be deferred. */
        sycl_batch_wait(_ds4_prof_ev18);
        ds4_sycl_profile_record(_ds4_prof_ev18);
    }
}

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
 * staging is load-bearing correctness code, not an optimisation.
 *
 * When `host_ptr` is already device-resident (the model-range
 * cache resolved it to a cached device pointer, sycl_model_range_ptr
 * above), pass through instead: no allocation, no copy, no wait, and the
 * returned guard does not own the pointer, so it is never freed here --
 * the cache owns it and frees it exactly once at teardown (spec 6g). This
 * one check covers every call site that stages a weight range through
 * this helper without any change at the call site itself.
 *
 * When not resident, the copy goes through sycl_copy_host_to_device_
 * paged_safe above rather than a bare queue.memcpy: see that function's
 * comment for the mmap-page-fault defect it guards against. */
static inline sycl_device_scratch_guard sycl_stage_host_bytes(
        sycl::queue &q, const void *host_ptr, uint64_t bytes) {
    if (sycl_ptr_is_device_resident(q, host_ptr)) {
        return sycl_device_scratch_guard(q, const_cast<void *>(host_ptr),
                                         /*owns_ptr=*/false);
    }
    unsigned char *dptr = sycl::malloc_device<unsigned char>((size_t)bytes, q);
    if (!dptr) return sycl_device_scratch_guard(q, nullptr);
    sycl_device_scratch_guard guard(q, dptr);
    sycl_copy_host_to_device_paged_safe(q, dptr, host_ptr, bytes);
    g_sycl_stage_host_bytes_total += bytes;
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

/* Strided-batched F32 GEMM via oneMKL's oneapi::mkl::blas::gemm_batch (USM,
 * strided form), the direct analogue of cublasSgemmStridedBatched used
 * throughout rocm/ds4_rocm_attention_launch.cuh and
 * rocm/ds4_rocm_matmul.cuh's cuBLAS fast paths. oneMKL's `blas::` namespace
 * is column-major by default (blas::column_major is an inline namespace),
 * matching cuBLAS: port each caller's transpose flags and leading
 * dimensions directly from the ROCm source's cublasSgemmStridedBatched
 * arguments rather than re-deriving the maths.
 *
 * `stride_a == 0` broadcasts one A matrix across every batch element, which
 * is what ROCm's raw-KV prefill path uses to share one KV table across every
 * head (a real batch dimension). The C operand's stride has no such
 * broadcast case: an actual per-batch output stride is required even at
 * batch_size == 1, confirmed empirically (a stride_c of 0 raises
 * oneapi::mkl::invalid_argument at runtime) -- this is a real difference
 * from cuBLAS's own tolerance and every caller here must pass a genuine
 * stride_c.
 *
 * oneMKL returns a sycl::event rather than ordering work on a stream the way
 * cuBLAS does. Per spec 6t this backend's queues are out-of-order raw-USM
 * queues, so this returned event is the caller's only handle on ordering: it
 * must be passed as a dependency to whatever reads `c` next (kernel or
 * another GEMM), or waited on directly, or both this call and the
 * consumer's own dependency chain race on `c`. */
static inline sycl::event sycl_gemm_batch_f32(
        sycl::queue &q,
        oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        int64_t m, int64_t n, int64_t k,
        float alpha,
        const float *a, int64_t lda, int64_t stride_a,
        const float *b, int64_t ldb, int64_t stride_b,
        float beta,
        float *c, int64_t ldc, int64_t stride_c,
        int64_t batch_size,
        const std::vector<sycl::event> &deps = {}) {
    return oneapi::mkl::blas::gemm_batch(q, transa, transb, m, n, k, alpha, a,
                                         lda, stride_a, b, ldb, stride_b, beta,
                                         c, ldc, stride_c, batch_size, deps);
}

/* Single (non-batched) F16 GEMM via oneMKL's oneapi::mkl::blas::gemm, the
 * analogue of cublasGemmEx(..., CUDA_R_16F, ..., CUBLAS_COMPUTE_32F, ...)
 * used by ROCm's cuda_matmul_q8_0_tensor_f16_gemm_out_half
 * (rocm/ds4_rocm_matmul.cuh:216-291), the one matmul entry with no non-GEMM
 * fallback in ROCm at all. oneMKL's (sycl::half, sycl::half, sycl::half,
 * sycl::half) instantiation is the closest available match: cuBLAS's alpha
 * and beta are float pointers under CUBLAS_COMPUTE_32F, but both this
 * caller's alpha (1.0) and beta (0.0) are exactly representable in half,
 * so passing them as sycl::half loses no precision; the alpha/beta type
 * has no bearing on the GEMM's own internal accumulation precision, which
 * is implementation-defined on both platforms regardless of Ts. Same
 * ordering contract as sycl_gemm_batch_f32 above: the returned event is
 * the caller's only handle on ordering against whatever reads `c` next. */
static inline sycl::event sycl_gemm_f16(
        sycl::queue &q,
        oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        int64_t m, int64_t n, int64_t k,
        sycl::half alpha,
        const sycl::half *a, int64_t lda,
        const sycl::half *b, int64_t ldb,
        sycl::half beta,
        sycl::half *c, int64_t ldc,
        const std::vector<sycl::event> &deps = {}) {
    return oneapi::mkl::blas::gemm(q, transa, transb, m, n, k, alpha, a, lda,
                                   b, ldb, beta, c, ldc, deps);
}

}  // namespace
