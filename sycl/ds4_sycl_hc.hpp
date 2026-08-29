#pragma once

/* DS4 hyper-connection (HC) subsystem: Sinkhorn-normalised splitting,
 * weighted summation across HC branches, and expansion back out, plus the
 * fused variants ds4 actually dispatches.
 *
 * Ported from rocm/ds4_rocm_hc.cuh (kernels) and rocm/ds4_rocm_hc_output_
 * launch.cuh (launchers), which are the authority for all semantics here.
 * The CPU oracle lives in ds4.c: hc_split_sinkhorn_one (:9718),
 * hc_weighted_sum_one (:9816) and hc_post_one (around :9920), plus the
 * hc_pre_from_state_one* family that composes them.
 *
 * Every entry in this file is NONZERO-means-success, verified against both
 * rocm/ds4_rocm_hc_output_launch.cuh's `cuda_ok(...)`/`return 0` shape and
 * every ds4.c call site (e.g. ds4.c:19823, `ok = ds4_gpu_hc_split_sinkhorn_
 * tensor(...) && ...`, used as a C bool). Zero-dimension arguments are a
 * validation FAILURE for every entry in this file (return 0), never a free
 * success: each helper below requires a nonzero row/token count to return
 * true, matching cuda_hc_flat_token_count / cuda_hc_hc_token_count in the
 * ROCm source.
 *
 * ds4_gpu_output_hc_weights_tensor (hc_output_launch.cuh:203-234) is
 * already implemented in sycl/ds4_sycl_output.hpp; it is the worked
 * example for this file's mmap staging.  ds4_gpu_shared_down_hc_expand_
 * q8_0_tensor and ds4_gpu_matmul_q8_0_hc_expand_tensor (:394-443) delegate
 * to a fused Q8_0-matmul-plus-HC-expand kernel family that lives in
 * rocm/ds4_rocm_matmul.cuh, outside this file's scope, and remain stubbed. */

#include "ds4_sycl_common.hpp"

namespace {

/* Fixed work-group width used by every row-per-work-group kernel in this
 * file (the Sinkhorn split's fused variants), matching ROCm's own literal
 * launch geometry ("hc_split_weighted_sum_fused_kernel<<<n_rows, 256>>>",
 * rocm/ds4_rocm_hc_output_launch.cuh:135 and :198). */
constexpr uint32_t kSyclHcRowGroup = 256u;

/* Token-count helpers, ported from the identically-named static helpers in
 * rocm/ds4_rocm_hc_output_launch.cuh:45-71.  A flat (n_embd-wide) tensor
 * derives its token count from raw byte capacity; an HC-shaped
 * (n_hc * n_embd-wide) tensor derives it the same way with the wider row.
 * Both require the byte count to be an exact multiple of one row and the
 * resulting token count to be nonzero: a zero-token tensor is a validation
 * failure in this file, never a free success. */
static int sycl_hc_flat_token_count(const ds4_gpu_tensor *out, uint32_t n_embd,
                                    uint64_t *n_tokens) {
    if (!out || n_embd == 0u || !n_tokens) return 0;
    uint64_t row_bytes = 0;
    if (!sycl_u64_mul3_checked(n_embd, 1u, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out->bytes < row_bytes || (out->bytes % row_bytes) != 0u) {
        return 0;
    }
    *n_tokens = out->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

static int sycl_hc_hc_token_count(const ds4_gpu_tensor *out_hc, uint32_t n_embd,
                                  uint32_t n_hc, uint64_t *n_tokens) {
    if (!out_hc || n_embd == 0u || n_hc == 0u || !n_tokens) return 0;
    uint64_t row_elems = 0, row_bytes = 0;
    if (!sycl_u64_mul_checked(n_hc, n_embd, &row_elems) ||
        !sycl_u64_mul_checked(row_elems, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out_hc->bytes < row_bytes || (out_hc->bytes % row_bytes) != 0u) {
        return 0;
    }
    *n_tokens = out_hc->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

/* Width of one packed HC "mix"/"split" row: n_hc pre weights, n_hc post
 * gates, n_hc * n_hc combine entries.  Matches cuda_hc_mix_width,
 * rocm/ds4_rocm_hc_output_launch.cuh:64-71. */
static int sycl_hc_mix_width(uint32_t n_hc, uint64_t *mix_hc) {
    if (n_hc == 0u || !mix_hc) return 0;
    const uint64_t h = (uint64_t)n_hc;
    const uint64_t mix = 2ull * h + h * h;
    if (mix > UINT32_MAX) return 0;
    *mix_hc = mix;
    return 1;
}

/* Per-row Sinkhorn-normalised HC split.  Ported literally from
 * hc4_split_one, rocm/ds4_rocm_hc.cuh:6-52: it hardcodes n_hc == 4
 * throughout rather than taking n_hc as a parameter, so this port does
 * the same.  Every caller of this helper enforces n_hc == 4 itself before
 * calling in (see the entry points below), so the hardcoding is never a
 * silent narrowing.
 *
 * out layout: [0, n_hc) pre weights, [n_hc, 2*n_hc) post gates,
 * [2*n_hc, 2*n_hc + n_hc*n_hc) row-then-column doubly-normalised combine
 * matrix, addressed as comb[dst + src * n_hc] (matching hc_post_one,
 * ds4.c, and hc_expand_kernel's `comb[dst_hc + src_hc * n_hc]`). */
static void sycl_hc4_split_one(float *out, const float *mix, const float *scale,
                               const float *base, uint32_t sinkhorn_iters,
                               float epsv) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + sycl::exp(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + sycl::exp(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = sycl::fmax(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = sycl::exp(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

}  // namespace

/* Splits the HC control projection into pre weights, post gates and a
 * doubly-normalised combine matrix, one row per token.  Ported from
 * ds4_gpu_hc_split_sinkhorn_tensor, rocm/ds4_rocm_hc_output_launch.cuh:
 * 27-44.  n_hc != 4 is rejected up front: hc4_split_one hardcodes n_hc == 4
 * and never receives it as a parameter, so this is the only place that
 * guard can be enforced. */
extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map,
        uint64_t model_size, uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4u) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (!sycl_model_range_fits(model_size, scale_offset, 3ull * sizeof(float)) ||
        !sycl_model_range_fits(model_size, base_offset, mix_bytes) ||
        !sycl_tensor_has_bytes(mix, mix_bytes) || !sycl_tensor_has_bytes(out, mix_bytes)) {
        return 0;
    }
    const char *scale_p = sycl_model_range_ptr(model_map, scale_offset,
                                                3ull * sizeof(float), model_size, "hc_scale");
    const char *base_p = sycl_model_range_ptr(model_map, base_offset, mix_bytes,
                                               model_size, "hc_base");
    if (!scale_p || !base_p) return 0;
    if (g_devices.empty()) return 0;

    /* Matches rocm/ds4_rocm_hc_output_launch.cuh:35-36: n_rows is the
     * smaller of mix's and out's row capacity, both already checked >= 1
     * row above, so n_rows is always >= 1 here. */
    uint64_t n_rows = mix->bytes / mix_bytes;
    const uint64_t out_rows = out->bytes / mix_bytes;
    if (out_rows < n_rows) n_rows = out_rows;
    if (n_rows == 0u || n_rows > UINT32_MAX) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        /* scale and base live in the host mmap; stage both to device
         * scratch per design-spec section 6l rather than dereferencing the
         * host pointer from a kernel. */
        sycl_device_scratch_guard scale_guard =
                sycl_stage_host_bytes(q, scale_p, 3ull * sizeof(float));
        if (!scale_guard.p) return 0;
        sycl_device_scratch_guard base_guard = sycl_stage_host_bytes(q, base_p, mix_bytes);
        if (!base_guard.p) return 0;

        float       *o     = (float *)out->ptr;
        const float *m     = (const float *)mix->ptr;
        const float *scale = (const float *)scale_guard.p;
        const float *base  = (const float *)base_guard.p;
        const uint32_t rows  = (uint32_t)n_rows;
        const uint32_t iters = sinkhorn_iters;
        const float    epsv  = eps;

        q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
            const uint32_t row = (uint32_t)id[0];
            sycl_hc4_split_one(o + (uint64_t)row * 24, m + (uint64_t)row * 24,
                               scale, base, iters, epsv);
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_split_sinkhorn failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Reduces the n_hc HC streams into the plain n_embd-wide sublayer input,
 * one work-item per (token, dim) pair.  Ported from
 * ds4_gpu_hc_weighted_sum_tensor, rocm/ds4_rocm_hc_output_launch.cuh:73-85.
 * weights is one n_hc-wide row per token (no post/comb section). */
extern "C" int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, residual_bytes = 0, weights_bytes = 0;
    if (!out || !residual_hc || !weights || n_hc == 0u ||
        !sycl_hc_flat_token_count(out, n_embd, &n_tokens64) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &residual_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &weights_bytes) ||
        residual_hc->bytes < residual_bytes || weights->bytes < weights_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o  = (float *)out->ptr;
        const float *x  = (const float *)residual_hc->ptr;
        const float *w  = (const float *)weights->ptr;
        const uint32_t hc = n_hc;
        const uint32_t embd = n_embd;
        const uint32_t stride = n_hc;
        const uint64_t n = (uint64_t)embd * n_tokens;

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            const uint32_t d = (uint32_t)(gid % embd);
            const uint32_t t = (uint32_t)(gid / embd);
            float acc = 0.0f;
            for (uint32_t h = 0; h < hc; h++) {
                acc += x[(uint64_t)t * hc * embd + (uint64_t)h * embd + d] *
                       w[(uint64_t)t * stride + h];
            }
            o[(uint64_t)t * embd + d] = acc;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_weighted_sum failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* As above, but the weights come from the first n_hc columns of a packed
 * split row (pre | post | comb) rather than a standalone n_hc-wide tensor.
 * Ported from ds4_gpu_hc_weighted_sum_split_tensor, rocm/ds4_rocm_hc_
 * output_launch.cuh:86-100. */
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, residual_bytes = 0, split_bytes = 0, mix_hc = 0;
    if (!out || !residual_hc || !split ||
        !sycl_hc_flat_token_count(out, n_embd, &n_tokens64) ||
        !sycl_hc_mix_width(n_hc, &mix_hc) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &residual_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, mix_hc, sizeof(float), &split_bytes) ||
        residual_hc->bytes < residual_bytes || split->bytes < split_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint32_t stride = (uint32_t)mix_hc;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o  = (float *)out->ptr;
        const float *x  = (const float *)residual_hc->ptr;
        const float *w  = (const float *)split->ptr;
        const uint32_t hc = n_hc;
        const uint32_t embd = n_embd;
        const uint64_t n = (uint64_t)embd * n_tokens;

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            const uint32_t d = (uint32_t)(gid % embd);
            const uint32_t t = (uint32_t)(gid / embd);
            float acc = 0.0f;
            for (uint32_t h = 0; h < hc; h++) {
                acc += x[(uint64_t)t * hc * embd + (uint64_t)h * embd + d] *
                       w[(uint64_t)t * stride + h];
            }
            o[(uint64_t)t * embd + d] = acc;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_weighted_sum_split failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Repeats one n_embd-wide row into n_hc identical HC streams.  Ported from
 * ds4_gpu_repeat_hc_tensor, rocm/ds4_rocm_hc_output_launch.cuh:1-10
 * (kernel: repeat_hc_kernel, rocm/ds4_rocm_common.cuh:316-321). */
extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row,
                                        uint32_t n_embd, uint32_t n_hc) {
    uint64_t n = 0;
    if (n_embd == 0u || n_hc == 0u || !out || !row ||
        !sycl_u64_mul_checked(n_embd, n_hc, &n) ||
        !sycl_tensor_has_f32(row, n_embd) || !sycl_tensor_has_f32(out, n)) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o = (float *)out->ptr;
        const float *r = (const float *)row->ptr;
        const uint32_t embd = n_embd;

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            o[gid] = r[gid % embd];
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "repeat_hc failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Repeats n_tokens rows, each n_embd wide, into n_hc identical HC streams
 * per token.  Ported from ds4_gpu_repeat_hc_rows_tensor, rocm/ds4_rocm_
 * hc_output_launch.cuh:12-25 (kernel: repeat_hc_rows_kernel,
 * rocm/ds4_rocm_common.cuh:323-330). */
extern "C" int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *rows,
                                             uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t rows_elems = 0;
    uint64_t n = 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || !out || !rows ||
        !sycl_u64_mul_checked(n_tokens, n_embd, &rows_elems) ||
        !sycl_u64_mul_checked(rows_elems, n_hc, &n) ||
        !sycl_tensor_has_f32(rows, rows_elems) || !sycl_tensor_has_f32(out, n)) {
        return 0;
    }
    if (n > UINT32_MAX) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float       *o  = (float *)out->ptr;
        const float *rw = (const float *)rows->ptr;
        const uint32_t embd = n_embd;
        const uint64_t hc_row = (uint64_t)n_hc * n_embd;

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            const uint64_t tok = gid / hc_row;
            const uint32_t d = (uint32_t)(gid % embd);
            o[gid] = rw[tok * embd + d];
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "repeat_hc_rows failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
