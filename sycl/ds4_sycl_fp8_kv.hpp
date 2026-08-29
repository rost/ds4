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

    const uint32_t groups = (n_nope + 63u) / 64u;

    try {
        sycl::queue &q = ds4_sycl_queue(x->device_id);
        float         *px    = (float *)x->ptr;
        const uint32_t width = head_dim;

        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scratch(sycl::range<1>(64), h);
            h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)n_tok * 64, groups),
                        sycl::range<2>(64, 1)),
                [=](sycl::nd_item<2> it) {
                    const size_t   row = it.get_group(0);
                    const uint32_t grp = (uint32_t)it.get_group(1);
                    const uint32_t tid = (uint32_t)it.get_local_id(0);

                    const uint32_t off = grp * 64u;
                    float *xr = px + row * width;

                    const bool  in_range = off + tid < n_nope;
                    const float v = in_range ? xr[off + tid] : 0.0f;

                    /* EVERY one of the 64 lanes writes its scratch slot,
                     * including lanes past n_nope (which write an explicit
                     * 0.0f), before the barrier below.  On this hardware
                     * (Arc A770, Level Zero, oneAPI 2025.3), uninitialised
                     * local memory has been observed to read as zero, so a
                     * kernel that skipped this write for out-of-range
                     * lanes would still pass here, but the SYCL
                     * specification guarantees no zero-initialisation and
                     * behaviour may differ on other hardware, including
                     * the Battlemage devices this backend targets. See the
                     * equivalent comment in
                     * ds4_gpu_rms_norm_plain_rows_tensor in
                     * sycl/ds4_sycl_norm_rope.hpp. */
                    scratch[tid] = in_range ? sycl::fabs(v) : 0.0f;
                    it.barrier(sycl::access::fence_space::local_space);

                    /* The barrier is OUTSIDE the `if`: every one of the 64
                     * lanes in the work-group must reach it on every
                     * iteration, matching rocm/ds4_rocm_fp8_kv.cuh:20-23. */
                    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
                        if (tid < stride) {
                            scratch[tid] = sycl::fmax(scratch[tid], scratch[tid + stride]);
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    /* Every lane computes the scale, not just lane 0:
                     * local memory is visible to the whole group after the
                     * barrier above. */
                    const float scale = sycl::exp2(sycl::ceil(
                            sycl::log2(sycl::fmax(scratch[0], 1.0e-4f) / 448.0f)));
                    if (in_range) {
                        const float clamped =
                                sycl::fmin(448.0f, sycl::fmax(-448.0f, v / scale));
                        xr[off + tid] = sycl_e4m3fn_dequant(clamped) * scale;
                    }
                });
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "dsv4_fp8_kv_quantize failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

namespace {

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
 * ROCm backend, so this function is ported bit-for-bit instead. */
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

/* store_raw_kv_batch_kernel, ported from rocm/ds4_rocm_fp8_kv.cuh:31-40:
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
        q.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                dout[0] = sycl::bit_cast<uint16_t>((sycl::half)f);
            });
        });
        q.wait_and_throw();
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
 * n_tokens == 0 has no dedicated early return in the source launcher:
 * validation passes trivially (kv needs zero bytes), the CUDA launch
 * becomes a legal zero-block no-op, and cuda_ok() on a clean launch
 * reports success purely as a side effect of the normal path. A
 * sycl::range<1>(0) kernel submission is not a well-defined SYCL-safe
 * equivalent of a zero-block CUDA launch, so the actual kernel submission
 * is skipped directly below when there is no work, while still falling
 * through to the same `return 1`: this preserves the launcher's
 * observable success-on-empty-input behaviour through a different,
 * SYCL-safe mechanism, since there is no explicit zero-work branch in the
 * source to port that mechanism from. */
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

        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> gid_id) {
                sycl_store_raw_kv_batch_kernel(gid_id, praw, pkv, raw_cap, pos0, head_dim);
            });
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "store_raw_kv_batch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}
