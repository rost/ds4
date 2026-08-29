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

/* Optional Metal decode fusion (paired projection plus recurrent
 * compressor-state store).  This backend has no such fused path; ROCm's
 * own implementation (rocm/ds4_rocm_matmul.cuh:933-965) is `(void)` on
 * every one of its 14 parameters then `return 0;`, a complete
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
