#pragma once

/* E4M3-style FP8 value table and nearest-rounding dequantiser for the
 * DeepSeek V4 compressed KV cache, ported from the CPU reference
 * dsv4_e4m3fn_value_cpu/dsv4_e4m3fn_dequant_cpu (ds4.c:3196-3235) and
 * cross-checked against the CUDA/ROCm device versions
 * dsv4_e4m3fn_value_dev/dsv4_e4m3fn_dequant_dev
 * (rocm/ds4_rocm_norm_rope.cuh:307-330).  The two device references agree
 * with the CPU lookup table at every exponent: for the non-subnormal
 * branch (exp 1..15) exp_scale[exp] == exp2f((float)exp - 7.0f) exactly,
 * since every table entry is an exact power of two; the subnormal branch
 * (exp == 0) is an explicit special case in both, not part of the table.
 * sycl_e4m3fn_value below follows the exp2f-style device formula rather
 * than reproducing the CPU's 16-entry table, since that agreement holds.
 *
 * The magnitude code is 7 bits: a 4-bit exponent field and a 3-bit
 * mantissa field, packed as (exp << 3) | mant, ranging 0..126 (127 is
 * unused: 448 is the largest representable magnitude, at code 126).
 * sycl_e4m3fn_dequant finds the nearest representable magnitude via
 * binary search, then breaks an exact tie to the even neighbour: ported
 * literally from ds4.c:3228-3232, including the direction check on which
 * of the two tied codes is even, since collapsing it to a plain `<=` or
 * dropping it changes the result for exactly half of all ties.
 *
 * Kept in an anonymous namespace: these are internal helpers for this
 * header's own kernel code below, not part of any public API.  The two
 * extern "C" functions immediately after the namespace exist only so
 * tests/test_sycl_fp8_kv.c (plain C) can exercise them from the host
 * directly, separately from the production ABI entry point further down. */

#include "ds4_sycl_common.hpp"

namespace {

/* The FP8 quantiser kernel's work-group size, matching the CUDA launch's
 * fixed 64-thread block (see the comment above
 * ds4_gpu_dsv4_fp8_kv_quantize_tensor below). The reduction's initial
 * stride is derived from this rather than a separate literal, so the
 * coupling between the group size and where the tree reduction starts is
 * explicit in the code. */
constexpr uint32_t kFp8KvGroup = 64u;

static inline float sycl_e4m3fn_value(int code) {
    const int exp = (code >> 3) & 0x0f;
    const int mant = code & 0x07;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * sycl::exp2((float)exp - 7.0f);
}

static inline float sycl_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = sycl::fmin(sycl::fabs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (sycl_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = sycl::fabs(ax - sycl_e4m3fn_value(best));
        const float next_diff = sycl::fabs(ax - sycl_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * sycl_e4m3fn_value(best);
}

} // namespace

/* Both of these call sycl::exp2/sycl::fmin/sycl::fabs, where host and
 * device code paths could in principle differ, especially under
 * -ffast-math. This is covered empirically rather than assumed: in
 * tests/test_sycl_fp8_kv.c, test_per_group_independent_scaling exercises
 * the real device kernel path (not these host-callable test hooks) at a
 * tight tolerance (1.0e-6 absolute against values of magnitude ~15000,
 * i.e. roughly 7e-11 relative) that no meaningfully imprecise device-side
 * exp2/fmin/fabs implementation could pass, so host/device numerical
 * parity for these functions is exercised indirectly through that test
 * even though these two particular hooks themselves only run on the
 * host. */
extern "C" float ds4_sycl_test_e4m3fn_value(int code) {
    return sycl_e4m3fn_value(code);
}

extern "C" float ds4_sycl_test_e4m3fn_dequant(float x) {
    return sycl_e4m3fn_dequant(x);
}

/* DeepSeek V4 compressed-KV FP8 quantiser, in place on the non-RoPE part of
 * each row.  Ported from ROCm's fp8_kv_quantize_kernel
 * (rocm/ds4_rocm_fp8_kv.cuh:7-29) and its launcher
 * ds4_gpu_dsv4_fp8_kv_quantize_tensor (rocm/ds4_rocm_fp8_kv_launch.cuh).
 *
 * The CUDA launch is <<<dim3(n_tok, groups), 64>>>: a 2D grid (row, group)
 * with a fixed 64-thread block (lane).  The natural SYCL equivalent, per
 * the same pattern used by ds4_gpu_dsv4_qkv_rms_norm_rows_tensor in
 * sycl/ds4_sycl_norm_rope.hpp, is an nd_range<2> whose second dimension has
 * extent `groups`, so it.get_group(1) selects the 64-wide group of
 * non-RoPE elements this work-group covers and it.get_group(0) is the row.
 *
 * The validation order below is load-bearing and matches the launcher
 * exactly: n_rot > head_dim is rejected BEFORE the zero-work early-outs,
 * so head_dim == 0 together with n_rot > 0 is a rejection (n_rot > head_dim
 * catches it), not a zero-work success.  Only head_dim == 0 together with
 * n_rot == 0 reaches the zero-work success path. */
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x,
                                                   uint32_t n_tok,
                                                   uint32_t head_dim,
                                                   uint32_t n_rot) {
    if (n_rot > head_dim ||
        !sycl_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) {
        return 0;
    }
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    if (g_devices.empty()) return 0;

    const uint32_t groups = (n_nope + kFp8KvGroup - 1u) / kFp8KvGroup;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float         *px    = (float *)x->ptr;
        const uint32_t width = head_dim;

        sycl::event _ds4_prof_ev37 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(kFp8KvGroup), h);
            h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)n_tok * kFp8KvGroup, groups),
                        sycl::range<2>(kFp8KvGroup, 1)),
                [=](sycl::nd_item<2> it) {
                    const size_t   row = it.get_group(0);
                    const uint32_t grp = (uint32_t)it.get_group(1);
                    const uint32_t tid = (uint32_t)it.get_local_id(0);

                    const uint32_t off = grp * kFp8KvGroup;
                    float *xr = px + row * width;

                    const bool  in_range = off + tid < n_nope;
                    const float v = in_range ? xr[off + tid] : 0.0f;

                    /* Every one of the 64 lanes calls sycl_block_row_reduce,
                     * including lanes past n_nope, which pass an explicit
                     * 0.0f: see that function's comment in
                     * ds4_sycl_common.hpp for why, matching
                     * rocm/ds4_rocm_fp8_kv.cuh:20-23. */
                    const float lane_abs = in_range ? sycl::fabs(v) : 0.0f;
                    const float row_max = sycl_block_row_reduce(
                            it, scratch, lane_abs,
                            [](float a, float b) { return sycl::fmax(a, b); });

                    /* Every lane computes the scale, not just lane 0:
                     * local memory is visible to the whole group after the
                     * reduction above. */
                    const float scale = sycl::exp2(sycl::ceil(
                            sycl::log2(sycl::fmax(row_max, 1.0e-4f) / 448.0f)));
                    if (in_range) {
                        const float clamped =
                                sycl::fmin(448.0f, sycl::fmax(-448.0f, v / scale));
                        xr[off + tid] = sycl_e4m3fn_dequant(clamped) * scale;
                    }
                });
        });
        q.wait_and_throw();
        ds4_sycl_profile_record_named("dsv4_fp8_kv_quantize", _ds4_prof_ev37);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "dsv4_fp8_kv_quantize failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

namespace {

/* sycl_f32_to_f16_bits_hip_round now lives in ds4_sycl_common.hpp: this
 * file's F16 store direction needs the same exact-tie rounding rule as
 * ds4_sycl_attention_output.hpp's F16 output entry (spec 6k), so it is
 * shared rather than ported a second time. See that header for the full
 * derivation comment (measured against sycl::half on an Arc A770 in
 * tests/test_sycl_fp8_kv.c).
 *
 * store_raw_kv_batch_kernel, ported from rocm/ds4_rocm_fp8_kv.cuh:31-40:
 * flattens (token, dim) into one linear range and, for each element,
 * encodes the KV value to F16 with the rounding rule above then
 * immediately widens it back to F32 before storing it in the ring
 * buffer, matching f16_bits_to_f32 (rocm/ds4_rocm_common.cuh, just below
 * f32_to_f16_bits_hip_round) exactly: that direction is a pure,
 * lossless bit-reinterpret-then-widen with no rounding decision, so
 * sycl::half is used directly here (the same pattern already established
 * by sycl_ape_value_dev in sycl/ds4_sycl_compressor.hpp), unlike the
 * encode direction above. */
static void sycl_store_raw_kv_batch_kernel(sycl::id<1> gid_id, float *raw,
                                           const float *kv, uint32_t raw_cap,
                                           uint32_t pos0, uint32_t head_dim) {
    const uint64_t gid = gid_id[0];
    const uint32_t d = (uint32_t)(gid % head_dim);
    const uint32_t t = (uint32_t)(gid / head_dim);
    const uint32_t row = (pos0 + t) % raw_cap;
    const uint16_t hb = sycl_f32_to_f16_bits_hip_round(kv[(uint64_t)t * head_dim + d]);
    raw[(uint64_t)row * head_dim + d] = (float)sycl::bit_cast<sycl::half>(hb);
}

} // namespace

extern "C" uint16_t ds4_sycl_test_hip_round_f16_bits(float f) {
    return sycl_f32_to_f16_bits_hip_round(f);
}

/* Test-only hook: runs sycl::half's own float-to-half conversion in an
 * actual one-element device kernel, so tests/test_sycl_fp8_kv.c (plain C,
 * unable to name sycl::half itself) can empirically compare the real
 * hardware's rounding against the ported oracle above.  Not part of the
 * production ABI. Returns 0 (leaving *out_bits untouched) if no device is
 * available or the kernel failed; nonzero on success. */
extern "C" int ds4_sycl_test_sycl_half_encode_bits(float f, uint16_t *out_bits) {
    if (!out_bits || g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(0);
        uint16_t *dout = sycl::malloc_shared<uint16_t>(1, q);
        sycl_device_scratch_guard guard(q, dout);
        if (!dout) return 0;
        sycl::event _ds4_prof_ev38 = q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                dout[0] = sycl::bit_cast<uint16_t>((sycl::half)f);
            });
        });
        q.wait_and_throw();
        ds4_sycl_profile_record_named("fp8_kv_test_half_encode_bits", _ds4_prof_ev38);
        *out_bits = dout[0];
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "sycl_half_encode_bits probe failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Raw KV ring-buffer batch store.  Ported from store_raw_kv_batch_kernel
 * (rocm/ds4_rocm_fp8_kv.cuh:31-40) and its launcher
 * ds4_gpu_store_raw_kv_batch_tensor (rocm/ds4_rocm_attention_launch.cuh:
 * 19-26); only that nine-line launcher is ported here, not the rest of
 * that file (which belongs to a later plan).
 *
 * Validation uses the overflow-safe helpers from ds4_sycl_common.hpp
 * (sycl_tensor_has_elems2) in place of the source launcher's raw,
 * unchecked multiplications `raw_cap * head_dim * sizeof(float)` and
 * `n_tokens * head_dim * sizeof(float)`, either of which can overflow
 * uint64_t for large enough inputs and wrongly validate an undersized
 * buffer as sufficient: the same class of defect already fixed at five
 * other entry points in this backend. sycl_tensor_has_elems2 also
 * rejects a tensor with a null `ptr`, which the source launcher's checks
 * do not do explicitly; raw_cap == 0 still needs its own check since,
 * unlike an undersized buffer, it is not caught by any byte comparison
 * (any raw_cache passes a `>= 0` bytes requirement) and would otherwise
 * reach a modulo-by-zero in the kernel.
 *
 * n_tokens == 0 has no dedicated zero-work branch in the source launcher,
 * unlike the sibling quantiser entry ds4_gpu_dsv4_fp8_kv_quantize_tensor
 * above, which does have one. Whether the CUDA/HIP zero-grid-dimension
 * launch this would otherwise produce is actually a legal no-op was not
 * verified against real CUDA/HIP hardware in this environment (none was
 * available), so no claim is made here about what the source actually
 * does on this input. Instead, this SYCL port deliberately chooses to
 * return success for n_tokens == 0 regardless of the source's actual
 * behaviour, because that matches this backend's established zero-work
 * convention (other entries, such as ds4_gpu_rms_norm_plain_rows_tensor,
 * also return success for zero-length work) and matches what every real
 * ds4.c call site expects: every call site passes a live batch count, so
 * this path is not reachable in practice, but returning success rather
 * than failure is the safer choice if it ever were. The actual kernel
 * submission is skipped directly below when there is no work (a
 * sycl::range<1>(0) kernel submission is not a well-defined SYCL-safe
 * equivalent of whatever the CUDA launch does), while still falling
 * through to the same `return 1`. */
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache,
                                                 const ds4_gpu_tensor *kv,
                                                 uint32_t raw_cap, uint32_t pos0,
                                                 uint32_t n_tokens,
                                                 uint32_t head_dim) {
    if (raw_cap == 0 ||
        !sycl_tensor_has_elems2(raw_cache, raw_cap, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(kv, n_tokens, head_dim, sizeof(float))) {
        return 0;
    }

    /* Safe: the elems2 check above already proved n_tokens * head_dim *
     * sizeof(float) fits in uint64_t without overflow, so the smaller
     * product n_tokens * head_dim fits too. */
    const uint64_t n = (uint64_t)n_tokens * head_dim;
    if (n == 0) return 1;

    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(raw_cache->device_id);
        float         *praw = (float *)raw_cache->ptr;
        const float   *pkv  = (const float *)kv->ptr;

        sycl::event _ds4_prof_ev39 = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid_id) {
                sycl_store_raw_kv_batch_kernel(gid_id, praw, pkv, raw_cap, pos0, head_dim);
            });
        });
        q.wait_and_throw();
        ds4_sycl_profile_record_named("store_raw_kv_batch", _ds4_prof_ev39);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "store_raw_kv_batch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* ds4_gpu_kv_fp8_store_raw_decode_rows_tensor, ds4_cuda.cu:16366-16397
 * (fp8_kv_quantize_store_rows_kernel, ds4_cuda.cu:6975-6992): the row-
 * batched form of ds4_gpu_kv_fp8_store_raw_tensor (sycl/ds4_sycl_
 * attention.hpp), called at ds4.c:64872 inside metal_graph_encode_qkv_
 * session_batch to quantise and store several concurrent sessions' KV
 * rows, each into its OWN private raw_cache, in one call. None of this
 * exists in rocm/; CUDA is the only reference.
 *
 * Composed from two already-tested steps rather than one new fused
 * kernel, the same choice ds4_gpu_kv_fp8_store_raw_tensor's own SYCL port
 * already made for the single-row form: quantise every row in place
 * first, by calling the already-landed ds4_gpu_dsv4_fp8_kv_quantize_
 * tensor treating the n_rows rows as its n_tok, then store each now-
 * quantised row into its own session's raw_cache below.
 *
 * CUDA instead fuses both into one <<<n_rows, 64>>> launch whose inner
 * quantiser loop (fp8_kv_quantize_row) handles n_nope > 64 by looping
 * chunk-by-chunk within that SAME 64-wide work-group, calling the shared
 * max-reduction a second time against the same local scratch buffer
 * before every lane has necessarily finished reading the first call's
 * result: sycl_block_row_reduce's own trailing barrier guarantees every
 * lane has ARRIVED at it, not that every lane's post-barrier read of
 * local[0] has completed before another lane starts the next call's
 * local[lid] = value write into the same slot. Reusing it twice per
 * group with no intervening barrier is exactly the shape spec 6t warns
 * about, one level down: ordering within a work-group is this code's job
 * just as ordering between submits is. The two-step composition
 * sidesteps it entirely: ds4_gpu_dsv4_fp8_kv_quantize_tensor's own
 * multi-group grid already handles any n_nope correctly and is already
 * tested on its own, and the per-row store step below has no reduction
 * and needs no barrier at all, so there is nothing to reuse unsafely.
 * The quantiser call's own trailing q.wait_and_throw() orders it ahead of
 * the store submit below by construction (spec 6t), with no separate
 * fence needed.
 *
 * Every raw_cache tensor must share kv's device: this backend already
 * checks that on the single-row sibling (ds4_gpu_kv_fp8_store_raw_
 * tensor's raw_cache->device_id != kv->device_id guard via ds4_gpu_
 * store_raw_kv_tensor), a check CUDA's own unified-addressing launcher
 * has no need for but this backend's Level Zero devices do (spec 6a).
 *
 * The per-row store step's target pointers are a host array of already
 * device-resident raw_cache addresses (each ds4_gpu_tensor->ptr). Per
 * spec 6l a kernel cannot read this table from the host, so it is staged
 * to device scratch, alongside the per-row cap and start-row arrays,
 * before the launch. */
extern "C" int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot) {
    if (!kv || !raw_caches || !raw_caps || !raw_rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_rot > head_dim ||
        !sycl_tensor_has_elems2(kv, n_rows, head_dim, sizeof(float))) {
        return 0;
    }

    std::vector<float *>  row_raw_ptr((size_t)n_rows);
    std::vector<uint32_t> row_raw_cap((size_t)n_rows);
    std::vector<uint32_t> row_raw_start((size_t)n_rows);
    for (uint32_t i = 0; i < n_rows; i++) {
        ds4_gpu_tensor *raw = raw_caches[i];
        if (!raw || raw_caps[i] == 0u || raw_rows[i] >= raw_caps[i] ||
            raw->device_id != kv->device_id ||
            !sycl_tensor_has_elems2(raw, raw_caps[i], head_dim, sizeof(float))) {
            return 0;
        }
        row_raw_ptr[i]   = (float *)raw->ptr;
        row_raw_cap[i]   = raw_caps[i];
        row_raw_start[i] = raw_rows[i];
    }
    if (g_devices.empty()) return 0;

    if (ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, n_rows, head_dim, n_rot) == 0) {
        return 0;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(kv->device_id);
        sycl_device_scratch_guard ptr_guard = sycl_stage_host_bytes(
                q, row_raw_ptr.data(), (uint64_t)n_rows * sizeof(float *));
        sycl_device_scratch_guard cap_guard = sycl_stage_host_bytes(
                q, row_raw_cap.data(), (uint64_t)n_rows * sizeof(uint32_t));
        sycl_device_scratch_guard start_guard = sycl_stage_host_bytes(
                q, row_raw_start.data(), (uint64_t)n_rows * sizeof(uint32_t));
        if (!ptr_guard.p || !cap_guard.p || !start_guard.p) return 0;

        float *const   *pptrs  = (float *const *)ptr_guard.p;
        const uint32_t *pcaps  = (const uint32_t *)cap_guard.p;
        const uint32_t *pstart = (const uint32_t *)start_guard.p;
        const float     *pkv   = (const float *)kv->ptr;
        const uint32_t   width = head_dim;
        const uint64_t   n     = (uint64_t)n_rows * width;

        sycl::event _ds4_prof_ev40 = q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid_id) {
                const uint64_t  gid = gid_id[0];
                const uint32_t  d   = (uint32_t)(gid % width);
                const uint32_t  row = (uint32_t)(gid / width);
                float          *raw = pptrs[row];
                const uint32_t  raw_row = pstart[row] % pcaps[row];
                const uint16_t hb = sycl_f32_to_f16_bits_hip_round(
                        pkv[(uint64_t)row * width + d]);
                raw[(uint64_t)raw_row * width + d] =
                        (float)sycl::bit_cast<sycl::half>(hb);
            });
        });
        q.wait_and_throw();
        ds4_sycl_profile_record_named("kv_fp8_store_raw_decode_rows", _ds4_prof_ev40);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "kv_fp8_store_raw_decode_rows failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
