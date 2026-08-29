#pragma once

/* Dense matmul entries.  Ported from rocm/ds4_rocm_matmul.cuh, which is
 * the authority for all semantics here.  This header implements 7 of the 10
 * entries in that file; the remaining 3 depend on oneMKL and stay stubbed
 * in ds4_sycl_unavailable.cpp until that is wired in. */

#include "ds4_sycl_common.hpp"

/* Submits (without waiting) out[t][o] = sum_k x[t][k] * f16_to_f32(w[o][k])
 * for one weight table, row-major out_dim rows of in_dim half values each.
 * Shared by both weight offsets of ds4_gpu_matmul_f16_pair_tensor below.
 * ROCm additionally special-cases n_tok == 1 with a perf-tuned kernel
 * (rocm/ds4_rocm_matmul.cuh:873-931); one general kernel covering every
 * n_tok is enough for correctness. */
static void sycl_matmul_f16_launch(sycl::queue &q, float *out,
                                   const uint16_t *w, const float *x,
                                   uint32_t in_dim, uint32_t out_dim,
                                   uint32_t n_tok) {
    const uint64_t n = (uint64_t)n_tok * out_dim;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t o  = (uint32_t)(gid % out_dim);
        const uint32_t t  = (uint32_t)(gid / out_dim);
        const float    *xr = x + (uint64_t)t * in_dim;
        const uint16_t *wr = w + (uint64_t)o * in_dim;
        float sum = 0.0f;
        for (uint32_t k = 0; k < in_dim; k++) {
            sum += xr[k] * (float)sycl::bit_cast<sycl::half>(wr[k]);
        }
        out[gid] = sum;
    });
}

/* Paired dense F16 projection: two independent weight tables applied to
 * the same activations in one call.  Validation ported from
 * rocm/ds4_rocm_matmul.cuh:873-931 (the ROCm entry's own bounds/overflow
 * checks), generalised to every n_tok the way ds4_gpu_matmul_f16_tensor's
 * own validation is (rocm/ds4_rocm_matmul.cuh:804-814), rather than only
 * n_tok == 1 the way ROCm's perf-tuned kernel-selection branch is.  For
 * n_tok != 1 ROCm instead calls ds4_gpu_matmul_f16_tensor twice and ANDs
 * the results; that entry is one of the three matmul entries this backend
 * defers pending oneMKL and stays permanently stubbed, so it is not called
 * here.  This entry owns one general kernel for every n_tok instead. */
extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b, const void *model_map,
        uint64_t model_size, uint64_t weight_a_offset, uint64_t weight_b_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out_a || !out_b || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t weight_bytes = 0;
    if (!sycl_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_a_offset, weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_b_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out_a, n_tok, out_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out_b, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wa_ptr = sycl_model_range_ptr(model_map, weight_a_offset,
                                              weight_bytes, model_size,
                                              "f16_pair_a");
    const char *wb_ptr = sycl_model_range_ptr(model_map, weight_b_offset,
                                              weight_bytes, model_size,
                                              "f16_pair_b");
    if (!wa_ptr || !wb_ptr) return 0;
    if (g_devices.empty()) return 0;

    const size_t weight_elems = (size_t)(weight_bytes / sizeof(uint16_t));

    try {
        sycl::queue &q = ds4_sycl_queue(out_a->device_id);

        uint16_t *dwa = sycl::malloc_device<uint16_t>(weight_elems, q);
        if (!dwa) return 0;
        sycl_device_scratch_guard dwa_guard(q, dwa);
        uint16_t *dwb = sycl::malloc_device<uint16_t>(weight_elems, q);
        if (!dwb) return 0;
        sycl_device_scratch_guard dwb_guard(q, dwb);

        q.memcpy(dwa, wa_ptr, (size_t)weight_bytes).wait_and_throw();
        q.memcpy(dwb, wb_ptr, (size_t)weight_bytes).wait_and_throw();

        sycl_matmul_f16_launch(q, (float *)out_a->ptr, dwa, (const float *)x->ptr,
                               (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
        sycl_matmul_f16_launch(q, (float *)out_b->ptr, dwb, (const float *)x->ptr,
                               (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_f16_pair failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Submits (without waiting) out[t][o] = sum_k x[t][k] * dequant(w[o][k])
 * for one Q8_0-quantised weight table, row-major out_dim rows of row_bytes
 * bytes each (blocks of 32 values, 34 bytes per block; see
 * sycl_q8_0_dequant in ds4_sycl_common.hpp).  One work-item per
 * (token, out_row) pair, covering every n_tok with no shape gating. */
static void sycl_q8_0_matmul_launch(sycl::queue &q, float *out,
                                    const unsigned char *w, const float *x,
                                    uint32_t in_dim, uint32_t out_dim,
                                    uint32_t n_tok, uint64_t row_bytes) {
    const uint64_t n = (uint64_t)n_tok * out_dim;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t o  = (uint32_t)(gid % out_dim);
        const uint32_t t  = (uint32_t)(gid / out_dim);
        const float         *xr = x + (uint64_t)t * in_dim;
        const unsigned char *wr = w + (uint64_t)o * row_bytes;
        float sum = 0.0f;
        for (uint32_t k = 0; k < in_dim; k++) {
            sum += xr[k] * sycl_q8_0_dequant(wr, k);
        }
        out[gid] = sum;
    });
}

/* Core Q8_0 dense matmul: out[t][o] = sum_k x[t][k] * dequant(w[o][k]) for
 * one weight table.  Validation and shared kernel selection ported from
 * ROCm's cuda_matmul_q8_0_tensor_labeled (rocm/ds4_rocm_matmul.cuh:311-524),
 * which every entry taking a plain Q8_0 weight table delegates to; this
 * helper plays the same role here for ds4_gpu_matmul_q8_0_tensor below and
 * for the later Q8_0 matmul entries in this file.
 *
 * Two things ROCm does are intentionally not ported:
 *
 * - The cuBLAS/GEMM fast path (rocm/ds4_rocm_matmul.cuh:325-330 and the
 *   f16-expansion-cache retry further down) is a perf path requiring a
 *   GEMM library; this backend has no oneMKL wiring yet, so it is skipped
 *   entirely and every call goes through the native kernel below.
 * - ROCm then branches on n_tok and on in_dim's shape to pick among
 *   several tiled/vendor kernels (the sharedx warp-row kernels gated on
 *   `in_dim & 31 == 0`, an AMD-only wmma kernel, and a prequantized-to-int8
 *   dp4a path) before falling through to a shape-ungated kernel
 *   (matmul_q8_0_f32_warp8_kernel for n_tok == 1, :380-387;
 *   matmul_q8_0_f32_batch_warp8_kernel for n_tok > 1, :425-434) that does
 *   the same per-(token, out_row) dequant-and-accumulate as every other
 *   branch, just with different tiling.  Those branches are ROCm/CUDA
 *   vendor performance tuning with no correctness difference from the
 *   general path; the dp4a branch additionally assumes an NVIDIA/AMD
 *   dot-product-of-4-int8 instruction with no direct SYCL equivalent worth
 *   chasing for a first port.  This helper always uses the shape-ungated
 *   kernel form (sycl_q8_0_matmul_launch above), for every n_tok. */
static int sycl_q8_0_matmul_general(ds4_gpu_tensor *out, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size, "q8_0");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        unsigned char *dw = sycl::malloc_device<unsigned char>(
                (size_t)weight_bytes, q);
        if (!dw) return 0;
        sycl_device_scratch_guard dw_guard(q, dw);
        q.memcpy(dw, wptr, (size_t)weight_bytes).wait_and_throw();

        sycl_q8_0_matmul_launch(q, (float *)out->ptr, dw, (const float *)x->ptr,
                                (uint32_t)in_dim, (uint32_t)out_dim,
                                (uint32_t)n_tok, row_bytes);
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_q8_0 failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Core Q8_0 dense matmul entry point.  ROCm's own entry
 * (rocm/ds4_rocm_matmul.cuh:526-529) is a two-line wrapper over
 * cuda_matmul_q8_0_tensor_labeled with label "q8_0"; the label is
 * diagnostic-only here (it only otherwise selects the omitted GEMM cache
 * path), so sycl_q8_0_matmul_general above needs no label parameter. */
extern "C" int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return sycl_q8_0_matmul_general(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok);
}

/* Single-token decode paths for the Q8_0 dense matmul.  ROCm's own
 * entries (rocm/ds4_rocm_matmul.cuh:531-549 and :551-569) are, like
 * ds4_gpu_matmul_q8_0_tensor above, two-line wrappers over
 * cuda_matmul_q8_0_tensor_labeled with model_map, weight_offset and every
 * other argument identical to the plain entry's call; the only difference
 * between all three ROCm entries is the diagnostic label string passed
 * ("q8_0" / "q8_0_decode" / "q8_0_decode_model_view"), which only
 * otherwise selects the omitted GEMM cache path. All three are therefore
 * functionally identical in ROCm: do not "fix" this by inventing a
 * behavioural difference between the two entries below, or between them
 * and ds4_gpu_matmul_q8_0_tensor. */
extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return sycl_q8_0_matmul_general(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return sycl_q8_0_matmul_general(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok);
}

/* ROCm's own entry (rocm/ds4_rocm_matmul.cuh:571-583) is dead code: every
 * one of its 8 parameters is cast to (void) and it unconditionally
 * `return 0;`.  This entry is not a permanently non-functional no-op on
 * every backend, though: ds4_metal.m implements it for real (delegating to
 * ds4_gpu_matmul_q8_0_tensor for n_tok == 1), so it is a live ABI entry with
 * a working implementation elsewhere, and ds4_gpu.h documents it in the
 * "Dense Projections" section alongside the other required dense-matmul
 * entries with no "optional"/fusion framing distinguishing it from a
 * required primitive (contrast ds4_gpu_matmul_f16_pair_compressor_store_tensor,
 * which ds4_gpu.h explicitly documents as "Optional Metal decode fusion").
 * Rather than port ROCm's dead no-op, this entry is a real, correct Q8_0
 * dense matmul with the same validation, return polarity, and numerical
 * contract as ds4_gpu_matmul_q8_0_tensor, delegating to the same
 * sycl_q8_0_matmul_general helper.  This is a deliberate divergence from
 * the ROCm reference, not a literal port. */
extern "C" int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return sycl_q8_0_matmul_general(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok);
}

/* Paired dense Q8_0 projection: two independent Q8_0 weight tables applied
 * to the same activations in one call, ported from
 * rocm/ds4_rocm_matmul.cuh:585-688.  Own validation ported from that
 * entry's null/zero/oversized checks (:597-601) and weight-range checks
 * (:608-622), generalised to every n_tok rather than only n_tok == 1 the
 * way ROCm's weight-range checks are (ROCm only runs them ahead of its
 * dedicated single-token kernel; for n_tok != 1 it instead calls the
 * shared labeled function twice, which does its own validation), mirroring
 * ds4_gpu_matmul_f16_pair_tensor above.  For the actual compute this entry
 * calls sycl_q8_0_matmul_general (the helper built for
 * ds4_gpu_matmul_q8_0_tensor) twice, once per weight offset, for every
 * n_tok including n_tok == 1: ROCm's dedicated n_tok == 1 pair kernel
 * (:628-656) and its prequantized-to-int8 variant (:658-687) are
 * shape-specific performance specialisations with no correctness
 * difference from calling the general path twice, the same class of
 * ROCm/CUDA vendor tuning sycl_q8_0_matmul_general's own comment already
 * declines. */
extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0u || out0_dim == 0u ||
        out1_dim == 0u || n_tok == 0u || in_dim > UINT32_MAX ||
        out0_dim > UINT32_MAX || out1_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0, weight0_bytes = 0, weight1_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out0_dim, row_bytes, &weight0_bytes) ||
        !sycl_u64_mul_checked(out1_dim, row_bytes, &weight1_bytes) ||
        !sycl_model_range_fits(model_size, weight0_offset, weight0_bytes) ||
        !sycl_model_range_fits(model_size, weight1_offset, weight1_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out0, n_tok, out0_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out1, n_tok, out1_dim, sizeof(float))) {
        return 0;
    }

    return sycl_q8_0_matmul_general(out0, model_map, model_size,
                                    weight0_offset, in_dim, out0_dim, x,
                                    n_tok) &&
           sycl_q8_0_matmul_general(out1, model_map, model_size,
                                    weight1_offset, in_dim, out1_dim, x,
                                    n_tok);
}

/* Optional Metal decode fusion (paired projection plus recurrent
 * compressor-state store).  This backend has no such fused path; ROCm's
 * own implementation (rocm/ds4_rocm_matmul.cuh:933-965) is `(void)` on
 * every one of its 15 parameters then `return 0;`, a complete
 * implementation of the tri-state contract's "optimized path unavailable"
 * branch (ds4_gpu.h), not a stub for something unfinished.  Ported
 * verbatim: touch nothing, decline. */
extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const void *model_map, uint64_t model_size, uint64_t weight_kv_offset,
        uint64_t weight_score_offset, uint64_t ape_offset, uint32_t ape_type,
        uint64_t in_dim, uint32_t width, const ds4_gpu_tensor *x,
        uint32_t ratio, uint32_t pos) {
    (void)out_kv;
    (void)out_score;
    (void)state_kv;
    (void)state_score;
    (void)model_map;
    (void)model_size;
    (void)weight_kv_offset;
    (void)weight_score_offset;
    (void)ape_offset;
    (void)ape_type;
    (void)in_dim;
    (void)width;
    (void)x;
    (void)ratio;
    (void)pos;
    return 0;
}
