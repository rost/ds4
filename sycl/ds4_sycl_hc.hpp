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
 * to a fused Q8_0-matmul-plus-HC-expand kernel family that ROCm defines in
 * rocm/ds4_rocm_matmul.cuh, outside this file's own kernel set; both
 * entries are implemented here via sycl_matmul_q8_0_hc_expand_labeled,
 * which reuses this file's HC-expand combine and sycl_q8_0_dequant from
 * ds4_sycl_common.hpp rather than duplicating either. */

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

namespace {

/* Shared per-output-element HC expand combine for the "general" (runtime
 * n_hc, not compile-time-unrolled) kernels: covers hc_expand_kernel,
 * hc_expand_half_kernel and hc_expand_add_half_kernel in ROCm terms
 * (rocm/ds4_rocm_hc.cuh:74-164). The caller resolves block_v -- decoding
 * F16 and/or adding block_add -- before calling in, since that is the
 * only thing that differs between the three ROCm kernels; the combine
 * math itself is identical across all of them. */
static inline float sycl_hc_expand_general(float block_v, const float *post, const float *comb,
                                           const float *residual_hc, uint32_t n_embd, uint32_t n_hc,
                                           uint32_t post_stride, uint32_t comb_stride, uint64_t t,
                                           uint32_t dst_hc, uint32_t d) {
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        const float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        const float res_v =
                residual_hc[t * (uint64_t)n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    return acc;
}

/* Shared n_hc == 4 unrolled combine for hc_expand4_kernel and its add/half
 * siblings (rocm/ds4_rocm_hc.cuh:166-298): same math as
 * sycl_hc_expand_general, specialised to a compile-time-unrolled 4-wide
 * loop over one already-fetched 24-wide split row (pre | post | comb),
 * matching ROCm's `sp + 4` (post) and `sp + 8` (comb) pointer offsets.
 * The caller resolves bv the same way as sycl_hc_expand_general's block_v. */
static inline void sycl_hc_expand4_write(float bv, const float *sp, float r0, float r1, float r2,
                                         float r3, uint32_t n_embd, uint64_t hc_base, float *out_hc) {
    const float *post = sp + 4;
    const float *comb = sp + 8;
    for (uint32_t dst = 0; dst < 4u; dst++) {
        float acc = bv * post[dst];
        acc += comb[0u * 4u + dst] * r0;
        acc += comb[1u * 4u + dst] * r1;
        acc += comb[2u * 4u + dst] * r2;
        acc += comb[3u * 4u + dst] * r3;
        out_hc[hc_base + (uint64_t)dst * n_embd] = acc;
    }
}

}  // namespace

/* Expands the sublayer output back into n_hc HC streams using separate
 * post/comb tensors (not a packed split row).  Always uses the general
 * kernel regardless of n_hc: ROCm has no n_hc == 4 specialisation for this
 * entry (rocm/ds4_rocm_hc_output_launch.cuh:235-257 has no such branch).
 * Ported with has_add == 0 (block_add unused), matching the ROCm launcher's
 * own call, which passes block_out for both pointers. */
extern "C" int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, post_bytes = 0, comb_bytes = 0, comb_stride = 0;
    if (!out_hc || !block_out || !residual_hc || !post || !comb ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &post_bytes) ||
        !sycl_u64_mul_checked(n_hc, n_hc, &comb_stride) || comb_stride > UINT32_MAX ||
        !sycl_u64_mul3_checked(n_tokens64, comb_stride, sizeof(float), &comb_bytes) ||
        block_out->bytes < flat_bytes || residual_hc->bytes < hc_bytes ||
        post->bytes < post_bytes || comb->bytes < comb_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint32_t hc = n_hc, embd = n_embd, cstride = (uint32_t)comb_stride;
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float       *ohc = (float *)out_hc->ptr;
        const float *bo  = (const float *)block_out->ptr;
        const float *res = (const float *)residual_hc->ptr;
        const float *pp  = (const float *)post->ptr;
        const float *cp  = (const float *)comb->ptr;

        q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            const uint32_t d = (uint32_t)(gid % embd);
            const uint64_t tmp = gid / embd;
            const uint32_t dst_hc = (uint32_t)(tmp % hc);
            const uint64_t t = tmp / hc;
            const float block_v = bo[t * embd + d];
            const float acc =
                    sycl_hc_expand_general(block_v, pp, cp, res, embd, hc, hc, cstride, t, dst_hc, d);
            ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Tensor-parallel ATTN-gate combine: as ds4_gpu_hc_expand_tensor
 * above, but block_v is block_out[t][d] + block_add[t][d] before the
 * combine, matching ds4_cuda.cu's ds4_gpu_hc_expand_add_tensor exactly (it
 * calls the identical hc_expand_kernel with is_add=1, block_add taking the
 * kernel's second data argument instead of a repeated block_out). ds4.c:
 * 23625/23629 calls this to fold the two ranks' partial attention outputs
 * straight into the HC expand without materialising their sum separately
 * first (see ds4_gpu_add_xdev_tensor in ds4_sycl_mgpu.hpp for the sibling
 * entry that DOES materialise a cross-rank sum, used on the directional-
 * steering branch of the same call site). Not gated on tp_world in any
 * way: the combine math has no TP-specific step, it is only ever called
 * under TP. Same validation shape and NONZERO-success polarity as
 * ds4_gpu_hc_expand_tensor. */
extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, post_bytes = 0, comb_bytes = 0, comb_stride = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !post || !comb ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &post_bytes) ||
        !sycl_u64_mul_checked(n_hc, n_hc, &comb_stride) || comb_stride > UINT32_MAX ||
        !sycl_u64_mul3_checked(n_tokens64, comb_stride, sizeof(float), &comb_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || post->bytes < post_bytes || comb->bytes < comb_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint32_t hc = n_hc, embd = n_embd, cstride = (uint32_t)comb_stride;
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float       *ohc = (float *)out_hc->ptr;
        const float *bo  = (const float *)block_out->ptr;
        const float *ba  = (const float *)block_add->ptr;
        const float *res = (const float *)residual_hc->ptr;
        const float *pp  = (const float *)post->ptr;
        const float *cp  = (const float *)comb->ptr;

        q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
            const uint64_t gid = gid_id[0];
            const uint32_t d = (uint32_t)(gid % embd);
            const uint64_t tmp = gid / embd;
            const uint32_t dst_hc = (uint32_t)(tmp % hc);
            const uint64_t t = tmp / hc;
            const float block_v = bo[t * embd + d] + ba[t * embd + d];
            const float acc =
                    sycl_hc_expand_general(block_v, pp, cp, res, embd, hc, hc, cstride, t, dst_hc, d);
            ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand_add failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Expands using a packed split row (pre | post | comb) rather than
 * separate post/comb tensors.  n_hc == 4 dispatches to the unrolled
 * specialisation (hc_expand4_kernel); any other n_hc falls back to the
 * general kernel with post/comb addressed by offsetting the split
 * pointer, matching rocm/ds4_rocm_hc_output_launch.cuh:258-290 exactly:
 * `base + n_hc` and `base + 2 * n_hc`, strided by mix_hc per row. */
extern "C" int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !residual_hc || !split ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_hc_mix_width(n_hc, &mix_hc64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || residual_hc->bytes < hc_bytes || split->bytes < split_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float       *ohc = (float *)out_hc->ptr;
        const float *bo  = (const float *)block_out->ptr;
        const float *res = (const float *)residual_hc->ptr;
        const float *sp0 = (const float *)split->ptr;
        const uint32_t embd = n_embd;

        if (n_hc == 4u) {
            const uint64_t n = (uint64_t)n_tokens * n_embd;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t t = gid / embd;
                const uint64_t td = t * embd + d;
                const uint64_t hc_base = t * 4u * embd + d;
                const float bv = bo[td];
                const float r0 = res[hc_base + 0u * embd];
                const float r1 = res[hc_base + 1u * embd];
                const float r2 = res[hc_base + 2u * embd];
                const float r3 = res[hc_base + 3u * embd];
                sycl_hc_expand4_write(bv, sp0 + t * 24, r0, r1, r2, r3, embd, hc_base, ohc);
            });
        } else {
            const uint32_t hc = n_hc;
            const uint32_t mix_hc = (uint32_t)mix_hc64;
            const float   *post = sp0 + hc;
            const float   *comb = sp0 + 2u * hc;
            const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
            q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t tmp = gid / embd;
                const uint32_t dst_hc = (uint32_t)(tmp % hc);
                const uint64_t t = tmp / hc;
                const float block_v = bo[t * embd + d];
                const float acc = sycl_hc_expand_general(block_v, post, comb, res, embd, hc, mix_hc,
                                                          mix_hc, t, dst_hc, d);
                ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
            });
        }
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand_split failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* As ds4_gpu_hc_expand_split_tensor, but block_out is F16.  Decode is
 * lossless (design-spec section 6k), so sycl::bit_cast<sycl::half> is
 * used directly rather than a ported bit-manipulation routine, which is
 * only needed for the opposite (encode) direction -- a direction this
 * file never performs (grep of rocm/ds4_rocm_hc.cuh finds __half2float at
 * every F16 use site here and no __float2half anywhere). Ported from
 * rocm/ds4_rocm_hc_output_launch.cuh:291-322. */
extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_half_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out_h || !residual_hc || !split ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_hc_mix_width(n_hc, &mix_hc64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(uint16_t), &flat_half_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out_h->bytes < flat_half_bytes || residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float          *ohc = (float *)out_hc->ptr;
        const uint16_t *bo  = (const uint16_t *)block_out_h->ptr;
        const float    *res = (const float *)residual_hc->ptr;
        const float    *sp0 = (const float *)split->ptr;
        const uint32_t  embd = n_embd;

        if (n_hc == 4u) {
            const uint64_t n = (uint64_t)n_tokens * n_embd;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t t = gid / embd;
                const uint64_t td = t * embd + d;
                const uint64_t hc_base = t * 4u * embd + d;
                const float bv = (float)sycl::bit_cast<sycl::half>(bo[td]);
                const float r0 = res[hc_base + 0u * embd];
                const float r1 = res[hc_base + 1u * embd];
                const float r2 = res[hc_base + 2u * embd];
                const float r3 = res[hc_base + 3u * embd];
                sycl_hc_expand4_write(bv, sp0 + t * 24, r0, r1, r2, r3, embd, hc_base, ohc);
            });
        } else {
            const uint32_t hc = n_hc;
            const uint32_t mix_hc = (uint32_t)mix_hc64;
            const float   *post = sp0 + hc;
            const float   *comb = sp0 + 2u * hc;
            const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
            q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t tmp = gid / embd;
                const uint32_t dst_hc = (uint32_t)(tmp % hc);
                const uint64_t t = tmp / hc;
                const float block_v = (float)sycl::bit_cast<sycl::half>(bo[t * embd + d]);
                const float acc = sycl_hc_expand_general(block_v, post, comb, res, embd, hc, mix_hc,
                                                          mix_hc, t, dst_hc, d);
                ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
            });
        }
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand_split_half failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* As ds4_gpu_hc_expand_split_tensor, but adds block_add (F32) to block_out
 * before applying post/comb.  Ported from rocm/ds4_rocm_hc_output_launch.
 * cuh:323-357. */
extern "C" int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_hc_mix_width(n_hc, &mix_hc64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || split->bytes < split_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float       *ohc = (float *)out_hc->ptr;
        const float *bo  = (const float *)block_out->ptr;
        const float *ba  = (const float *)block_add->ptr;
        const float *res = (const float *)residual_hc->ptr;
        const float *sp0 = (const float *)split->ptr;
        const uint32_t embd = n_embd;

        if (n_hc == 4u) {
            const uint64_t n = (uint64_t)n_tokens * n_embd;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t t = gid / embd;
                const uint64_t td = t * embd + d;
                const uint64_t hc_base = t * 4u * embd + d;
                const float bv = bo[td] + ba[td];
                const float r0 = res[hc_base + 0u * embd];
                const float r1 = res[hc_base + 1u * embd];
                const float r2 = res[hc_base + 2u * embd];
                const float r3 = res[hc_base + 3u * embd];
                sycl_hc_expand4_write(bv, sp0 + t * 24, r0, r1, r2, r3, embd, hc_base, ohc);
            });
        } else {
            const uint32_t hc = n_hc;
            const uint32_t mix_hc = (uint32_t)mix_hc64;
            const float   *post = sp0 + hc;
            const float   *comb = sp0 + 2u * hc;
            const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
            q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t tmp = gid / embd;
                const uint32_t dst_hc = (uint32_t)(tmp % hc);
                const uint64_t t = tmp / hc;
                const float block_v = bo[t * embd + d] + ba[t * embd + d];
                const float acc = sycl_hc_expand_general(block_v, post, comb, res, embd, hc, mix_hc,
                                                          mix_hc, t, dst_hc, d);
                ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
            });
        }
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand_add_split failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Tensor parallelism: fused split-matmul + HC-expand-add. ds4_
 * cuda.cu's real kernel pair (quantize_q8_0_f32_kernel +
 * matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel, ds4_cuda.cu:14924-
 * 14996) does the k-slice matmul into block_out and the HC combine of
 * (block_out + block_add) in one launch; this port composes the two
 * pieces already built above instead of writing a third fused
 * kernel, the same "compose rather than duplicate" call this file's own
 * header comment documents for ds4_gpu_shared_down_hc_expand_q8_0_tensor
 * below. Mathematically identical: block_out is fully written by the
 * first call (and, per this file's own trailing-wait discipline, is
 * complete before the second call reads it) before
 * ds4_gpu_hc_expand_add_split_tensor folds it with block_add.
 *
 * Validation ported from ds4_cuda.cu:14924-14943, which additionally
 * requires out_dim == n_embd (the k-slice matmul's output width must
 * match the HC expand's per-token width for the composition to be
 * dimensionally sound; the two composed calls below do not check this
 * cross-condition on their own, so it is checked explicitly here). The
 * `n_tok > 65535` grid-dimension artefact is skipped for the same reason
 * ds4_gpu_matmul_q8_0_kslice_rows_tensor's own port skips it. NONZERO
 * means success, matching both composed calls and ds4_cuda.cu's own
 * `cuda_ok(...)` return. */
extern "C" int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint64_t                in_start,
        uint64_t                in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !model_map || !x || !block_add ||
        !residual_hc || !split || in_dim == 0u || out_dim == 0u ||
        in_count == 0u || n_embd == 0u || n_hc == 0u ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if (!ds4_gpu_matmul_q8_0_kslice_rows_tensor(block_out, model_map, model_size,
                                                weight_offset, in_dim, out_dim,
                                                in_start, in_count, x, 1u)) {
        return 0;
    }
    return ds4_gpu_hc_expand_add_split_tensor(out_hc, block_out, block_add,
                                              residual_hc, split, n_embd, n_hc);
}

/* As ds4_gpu_hc_expand_add_split_tensor, but block_add is F16: decode then
 * add.  Decode is lossless (design-spec section 6k), same reasoning as
 * ds4_gpu_hc_expand_split_half_tensor above.  Ported from rocm/ds4_rocm_
 * hc_output_launch.cuh:358-393. */
extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, flat_half_bytes = 0, hc_bytes = 0, split_bytes = 0,
             mix_hc64 = 0;
    if (!out_hc || !block_out || !block_add_h || !residual_hc || !split ||
        !sycl_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !sycl_hc_mix_width(n_hc, &mix_hc64) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, n_embd, sizeof(uint16_t), &flat_half_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || block_add_h->bytes < flat_half_bytes ||
        residual_hc->bytes < hc_bytes || split->bytes < split_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);
        float          *ohc = (float *)out_hc->ptr;
        const float    *bo  = (const float *)block_out->ptr;
        const uint16_t *ba  = (const uint16_t *)block_add_h->ptr;
        const float    *res = (const float *)residual_hc->ptr;
        const float    *sp0 = (const float *)split->ptr;
        const uint32_t  embd = n_embd;

        if (n_hc == 4u) {
            const uint64_t n = (uint64_t)n_tokens * n_embd;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t t = gid / embd;
                const uint64_t td = t * embd + d;
                const uint64_t hc_base = t * 4u * embd + d;
                const float bv = bo[td] + (float)sycl::bit_cast<sycl::half>(ba[td]);
                const float r0 = res[hc_base + 0u * embd];
                const float r1 = res[hc_base + 1u * embd];
                const float r2 = res[hc_base + 2u * embd];
                const float r3 = res[hc_base + 3u * embd];
                sycl_hc_expand4_write(bv, sp0 + t * 24, r0, r1, r2, r3, embd, hc_base, ohc);
            });
        } else {
            const uint32_t hc = n_hc;
            const uint32_t mix_hc = (uint32_t)mix_hc64;
            const float   *post = sp0 + hc;
            const float   *comb = sp0 + 2u * hc;
            const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
            q.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> gid_id) {
                const uint64_t gid = gid_id[0];
                const uint32_t d = (uint32_t)(gid % embd);
                const uint64_t tmp = gid / embd;
                const uint32_t dst_hc = (uint32_t)(tmp % hc);
                const uint64_t t = tmp / hc;
                const float block_v =
                        bo[t * embd + d] + (float)sycl::bit_cast<sycl::half>(ba[t * embd + d]);
                const float acc = sycl_hc_expand_general(block_v, post, comb, res, embd, hc, mix_hc,
                                                          mix_hc, t, dst_hc, d);
                ohc[t * hc * embd + (uint64_t)dst_hc * embd + d] = acc;
            });
        }
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_expand_add_split_half_add failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Fuses a per-row Q8_0 dense matmul with the HC-expand combine above.
 * Ported from ROCm's cuda_matmul_q8_0_hc_expand_tensor_labeled
 * (rocm/ds4_rocm_matmul.cuh:690), which both
 * ds4_gpu_matmul_q8_0_hc_expand_tensor and
 * ds4_gpu_shared_down_hc_expand_q8_0_tensor below delegate to with
 * `block_add` NULL or non-NULL respectively.  Only the dp4a/prequantised-
 * activation fast path and the sharedx/warp tiling are not ported, the
 * same declined-for-now perf tier as sycl_q8_0_matmul_general in
 * sycl/ds4_sycl_matmul.hpp: this uses the identical shape-ungated scalar
 * dot product, one work-item per output row.
 *
 * `block_add`, when non-NULL, is added into the raw matmul result BEFORE
 * the HC-expand combine, matching ROCm's `has_add` branch; `block_out`
 * always receives the RAW (pre-add) matmul result either way, also
 * matching ROCm (`block_out[d] = acc;` runs before `block_v` picks up the
 * addend). out_dim must equal n_embd: this fused family always produces
 * exactly one HC-token-width row, the same requirement ROCm's own
 * validation enforces. Single-token only (t == 0 throughout): every
 * caller in ds4.c uses this for a decode-time projection, never a batch,
 * and ROCm's own kernel carries no token loop or stride either. */
static int sycl_matmul_q8_0_hc_expand_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    uint64_t row_bytes = 0, weight_bytes = 0, hc_bytes = 0, mix_hc64 = 0, split_bytes = 0;
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0u || out_dim == 0u || n_embd == 0u || n_hc == 0u ||
        out_dim != (uint64_t)n_embd ||
        !sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_hc_mix_width(n_hc, &mix_hc64) ||
        !sycl_u64_mul3_checked(n_hc, n_embd, sizeof(float), &hc_bytes) ||
        !sycl_u64_mul_checked(mix_hc64, sizeof(float), &split_bytes) ||
        !sycl_tensor_has_elems(x, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems(block_out, out_dim, sizeof(float)) ||
        !sycl_tensor_has_bytes(residual_hc, hc_bytes) ||
        !sycl_tensor_has_bytes(split, split_bytes) ||
        !sycl_tensor_has_bytes(out_hc, hc_bytes) ||
        (block_add && !sycl_tensor_has_elems(block_add, out_dim, sizeof(float)))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, weight_bytes,
                                            model_size, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out_hc->device_id);

        unsigned char *dw = sycl::malloc_device<unsigned char>((size_t)weight_bytes, q);
        if (!dw) return 0;
        sycl_device_scratch_guard dw_guard(q, dw);
        q.memcpy(dw, wptr, (size_t)weight_bytes).wait_and_throw();

        float         *ohc  = (float *)out_hc->ptr;
        float         *bo   = (float *)block_out->ptr;
        const float   *xp   = (const float *)x->ptr;
        const float   *ba   = block_add ? (const float *)block_add->ptr : nullptr;
        const float   *res  = (const float *)residual_hc->ptr;
        const float   *sp0  = (const float *)split->ptr;
        const uint32_t embd = n_embd;
        const uint32_t hc   = n_hc;
        const uint32_t mix_hc = (uint32_t)mix_hc64;
        const float   *post = sp0 + hc;
        const float   *comb = sp0 + 2u * hc;
        const uint32_t out_d = (uint32_t)out_dim;
        const uint32_t in_d  = (uint32_t)in_dim;
        const uint64_t rbytes = row_bytes;
        const bool has_add = block_add != nullptr;

        q.parallel_for(sycl::range<1>(out_d), [=](sycl::id<1> gid_id) {
            const uint32_t d = (uint32_t)gid_id[0];
            const unsigned char *wr = dw + (uint64_t)d * rbytes;
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_d; k++) {
                acc += xp[k] * sycl_q8_0_dequant(wr, k);
            }
            bo[d] = acc;
            const float block_v = has_add ? acc + ba[d] : acc;
            for (uint32_t dst_hc = 0; dst_hc < hc; dst_hc++) {
                const float v = sycl_hc_expand_general(block_v, post, comb, res,
                                                       embd, hc, mix_hc, mix_hc,
                                                       0u, dst_hc, d);
                ohc[(uint64_t)dst_hc * embd + d] = v;
            }
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n",
                label ? label : "matmul_q8_0_hc_expand", e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return sycl_matmul_q8_0_hc_expand_labeled(out_hc, block_out, model_map,
                                              model_size, weight_offset,
                                              in_dim, out_dim, x, nullptr,
                                              residual_hc, split, n_embd, n_hc,
                                              "q8_hc_expand");
}

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return sycl_matmul_q8_0_hc_expand_labeled(out_hc, shared_out, model_map,
                                              model_size, weight_offset,
                                              in_dim, out_dim, shared_mid,
                                              routed_out, residual_hc, split,
                                              n_embd, n_hc,
                                              "shared_down_hc_expand");
}

/* Fuses hc_split_sinkhorn_kernel and hc_weighted_sum_kernel into one launch,
 * one work group per token: lane 0 computes the Sinkhorn split into the
 * `split` output tensor (a genuine output, not just internal scratch), a
 * barrier makes that write visible to the rest of the group, then every
 * lane strides across n_embd computing the weighted sum.  Ported from
 * ds4_gpu_hc_split_weighted_sum_tensor, rocm/ds4_rocm_hc_output_launch.cuh:
 * 101-145 (kernel: hc_split_weighted_sum_fused_kernel, rocm/ds4_rocm_hc.cuh:
 * 300-326).
 *
 * The kernel-level `n_hc != 4` guard is preserved even though the entry
 * above it already rejects n_hc != 4 before ever launching: ROCm's own
 * kernel carries this same redundant check (hc.cuh:304), and a guard that
 * can never fire in practice is exactly the kind a later reader deletes
 * as dead unless it is ported deliberately. */
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc, const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset, uint32_t n_embd, uint32_t n_hc,
        uint32_t sinkhorn_iters, float eps) {
    if (!out || !split || !mix || !residual_hc || !model_map || n_embd == 0u || n_hc != 4u) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    uint64_t mix_bytes = 0, out_row_bytes = 0, residual_row_bytes = 0;
    if (!sycl_u64_mul_checked(mix_hc, sizeof(float), &mix_bytes) ||
        !sycl_u64_mul_checked(n_embd, sizeof(float), &out_row_bytes) ||
        !sycl_u64_mul3_checked(n_hc, n_embd, sizeof(float), &residual_row_bytes) ||
        out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0u ||
        !sycl_model_range_fits(model_size, scale_offset, 3ull * sizeof(float)) ||
        !sycl_model_range_fits(model_size, base_offset, mix_bytes)) {
        return 0;
    }
    const uint64_t n_rows = out->bytes / out_row_bytes;
    uint64_t mix_needed = 0, residual_needed = 0;
    if (n_rows == 0u || n_rows > UINT32_MAX ||
        !sycl_u64_mul_checked(n_rows, mix_bytes, &mix_needed) ||
        !sycl_u64_mul_checked(n_rows, residual_row_bytes, &residual_needed) ||
        mix->bytes < mix_needed || split->bytes < mix_needed ||
        residual_hc->bytes < residual_needed) {
        return 0;
    }
    const char *scale_p = sycl_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float),
                                                model_size, "hc_scale");
    const char *base_p = sycl_model_range_ptr(model_map, base_offset, mix_bytes, model_size, "hc_base");
    if (!scale_p || !base_p) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        sycl_device_scratch_guard scale_guard =
                sycl_stage_host_bytes(q, scale_p, 3ull * sizeof(float));
        if (!scale_guard.p) return 0;
        sycl_device_scratch_guard base_guard = sycl_stage_host_bytes(q, base_p, mix_bytes);
        if (!base_guard.p) return 0;

        float       *o     = (float *)out->ptr;
        float       *sp0   = (float *)split->ptr;
        const float *m     = (const float *)mix->ptr;
        const float *res   = (const float *)residual_hc->ptr;
        const float *scale = (const float *)scale_guard.p;
        const float *base  = (const float *)base_guard.p;
        const uint32_t rows  = (uint32_t)n_rows;
        const uint32_t embd  = n_embd;
        const uint32_t hc    = n_hc;
        const uint32_t iters = sinkhorn_iters;
        const float    epsv  = eps;

        q.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)rows * kSyclHcRowGroup),
                                  sycl::range<1>(kSyclHcRowGroup)),
                [=](sycl::nd_item<1> it) {
                    const uint32_t t = (uint32_t)it.get_group(0);
                    const uint32_t d = (uint32_t)it.get_local_id(0);
                    /* t depends only on the group id, so this early return
                     * is group-uniform: every lane in a given group agrees
                     * on it, and the barrier below is reached by either
                     * every lane in the group or none of them. */
                    if (t >= rows || hc != 4u) return;
                    float *sp = sp0 + (uint64_t)t * 24;
                    if (d == 0u) sycl_hc4_split_one(sp, m + (uint64_t)t * 24, scale, base, iters, epsv);
                    /* global_and_local: sp is a pointer into device global
                     * memory (the split output tensor), written by lane 0
                     * and read by every lane below, so the fence must cover
                     * global memory, not just the (unused here) local
                     * accessor space. */
                    it.barrier(sycl::access::fence_space::global_and_local);
                    for (uint32_t col = d; col < embd; col += kSyclHcRowGroup) {
                        float acc = 0.0f;
                        for (uint32_t h = 0; h < 4u; h++) {
                            acc += res[(uint64_t)t * 4u * embd + (uint64_t)h * embd + col] * sp[h];
                        }
                        o[(uint64_t)t * embd + col] = acc;
                    }
                });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_split_weighted_sum failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* As ds4_gpu_hc_split_weighted_sum_tensor, plus a fused RMS norm over the
 * weighted-sum output using the shared sycl_block_row_reduce helper
 * (ds4_sycl_common.hpp): every lane's per-lane sum-of-squares accumulates
 * across its strided columns (0.0f for a lane whose loop ran zero times,
 * satisfying design-spec section 6b), then the whole group reduces those
 * partial sums to one total before computing the norm scale.  Ported from
 * ds4_gpu_hc_split_weighted_sum_norm_tensor, rocm/ds4_rocm_hc_output_
 * launch.cuh:146-202 (kernel: hc_split_weighted_sum_norm_fused_kernel,
 * rocm/ds4_rocm_hc.cuh:328-373). */
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *norm_out, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const void *model_map,
        uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint64_t norm_weight_offset,
        uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps, float norm_eps) {
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map || n_embd == 0u ||
        n_hc != 4u) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    uint64_t mix_bytes = 0, out_row_bytes = 0, residual_row_bytes = 0, norm_w_bytes = 0;
    if (!sycl_u64_mul_checked(mix_hc, sizeof(float), &mix_bytes) ||
        !sycl_u64_mul_checked(n_embd, sizeof(float), &out_row_bytes) ||
        !sycl_u64_mul3_checked(n_hc, n_embd, sizeof(float), &residual_row_bytes) ||
        !sycl_u64_mul_checked(n_embd, sizeof(float), &norm_w_bytes) ||
        out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0u ||
        norm_out->bytes < out->bytes ||
        !sycl_model_range_fits(model_size, scale_offset, 3ull * sizeof(float)) ||
        !sycl_model_range_fits(model_size, base_offset, mix_bytes) ||
        !sycl_model_range_fits(model_size, norm_weight_offset, norm_w_bytes)) {
        return 0;
    }
    const uint64_t n_rows = out->bytes / out_row_bytes;
    uint64_t mix_needed = 0, residual_needed = 0;
    if (n_rows == 0u || n_rows > UINT32_MAX ||
        !sycl_u64_mul_checked(n_rows, mix_bytes, &mix_needed) ||
        !sycl_u64_mul_checked(n_rows, residual_row_bytes, &residual_needed) ||
        mix->bytes < mix_needed || split->bytes < mix_needed ||
        residual_hc->bytes < residual_needed) {
        return 0;
    }
    const char *scale_p = sycl_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float),
                                                model_size, "hc_scale");
    const char *base_p = sycl_model_range_ptr(model_map, base_offset, mix_bytes, model_size, "hc_base");
    const char *norm_w_p = sycl_model_range_ptr(model_map, norm_weight_offset, norm_w_bytes,
                                                model_size, "hc_norm_weight");
    if (!scale_p || !base_p || !norm_w_p) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        sycl_device_scratch_guard scale_guard =
                sycl_stage_host_bytes(q, scale_p, 3ull * sizeof(float));
        if (!scale_guard.p) return 0;
        sycl_device_scratch_guard base_guard = sycl_stage_host_bytes(q, base_p, mix_bytes);
        if (!base_guard.p) return 0;
        sycl_device_scratch_guard norm_w_guard = sycl_stage_host_bytes(q, norm_w_p, norm_w_bytes);
        if (!norm_w_guard.p) return 0;

        float       *o      = (float *)out->ptr;
        float       *no     = (float *)norm_out->ptr;
        float       *sp0    = (float *)split->ptr;
        const float *m      = (const float *)mix->ptr;
        const float *res    = (const float *)residual_hc->ptr;
        const float *scale  = (const float *)scale_guard.p;
        const float *base   = (const float *)base_guard.p;
        const float *norm_w = (const float *)norm_w_guard.p;
        const uint32_t rows      = (uint32_t)n_rows;
        const uint32_t embd      = n_embd;
        const uint32_t hc        = n_hc;
        const uint32_t iters     = sinkhorn_iters;
        const float    epsv      = eps;
        const float    norm_epsv = norm_eps;

        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> partial(sycl::range<1>(kSyclHcRowGroup), h);
            h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)rows * kSyclHcRowGroup),
                                      sycl::range<1>(kSyclHcRowGroup)),
                    [=](sycl::nd_item<1> it) {
                        const uint32_t t = (uint32_t)it.get_group(0);
                        const uint32_t d = (uint32_t)it.get_local_id(0);
                        if (t >= rows || hc != 4u) return;
                        float *sp = sp0 + (uint64_t)t * 24;
                        if (d == 0u) {
                            sycl_hc4_split_one(sp, m + (uint64_t)t * 24, scale, base, iters, epsv);
                        }
                        it.barrier(sycl::access::fence_space::global_and_local);

                        float sum = 0.0f;
                        for (uint32_t col = d; col < embd; col += kSyclHcRowGroup) {
                            float acc = 0.0f;
                            for (uint32_t h = 0; h < 4u; h++) {
                                acc += res[(uint64_t)t * 4u * embd + (uint64_t)h * embd + col] * sp[h];
                            }
                            o[(uint64_t)t * embd + col] = acc;
                            sum += acc * acc;
                        }

                        const float total = sycl_block_row_reduce(
                                it, partial, sum, [](float a, float b) { return a + b; });
                        const float norm_scale = sycl::rsqrt(total / (float)embd + norm_epsv);
                        for (uint32_t col = d; col < embd; col += kSyclHcRowGroup) {
                            const float v = o[(uint64_t)t * embd + col];
                            no[(uint64_t)t * embd + col] = v * norm_scale * norm_w[col];
                        }
                    });
        });
        q.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "hc_split_weighted_sum_norm failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
