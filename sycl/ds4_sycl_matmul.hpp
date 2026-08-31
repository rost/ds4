#pragma once

/* Dense matmul entries.  Ported from rocm/ds4_rocm_matmul.cuh, which is
 * the authority for all semantics here.  This header implements 7 of the 10
 * entries in that file; the remaining 3 depend on oneMKL and stay stubbed
 * in ds4_sycl_unavailable.cpp until that is wired in. */

#include "ds4_sycl_common.hpp"

/* ds4_sycl_test_gemm_batch_smoke: proves oneMKL's strided-batched gemm_batch
 * is wired correctly (build flags, column-major argument order, the
 * stride_a == 0 KV-broadcast case) against a hand-computed result,
 * independent of any production matmul or attention entry, so the tasks
 * that follow debug their own kernels rather than the BLAS wiring
 * underneath them. Not part of the ABI: a backend-internal test hook in the
 * same spirit as ds4_sycl_device_count (ds4_sycl.cpp), called directly by
 * tests/test_sycl_gemm_batch_smoke.c.
 *
 * Shape mirrors the real raw-prefill score GEMM
 * (rocm/ds4_rocm_attention_launch.cuh:261-278, scores = K^T Q): head_dim
 * (k) = 3, n_keys (m) = 2, n_tokens (n) = 2, batch = 2 "heads". A = the KV
 * table, broadcast across every head via stride_a == 0, transa = trans (A
 * is stored column-major head_dim x n_keys, matching a row-major
 * n_keys x head_dim buffer exactly the way raw_kv->ptr is laid out in the
 * real launch); B = per-head Q, transb = nontrans. B carries one padding
 * row per column (ldb = head_dim + 1, distinct from lda = head_dim) so an
 * lda/ldb swap is a genuine ablation rather than a no-op: the real launch's
 * B operand is itself a strided sub-view whose leading dimension
 * (n_head * head_dim) differs from A's (head_dim), and lda == ldb here
 * would let that class of argument-order bug go undetected.
 *
 * Hand-computed: raw_kv rows (key0=[1,2,3], key1=[4,5,6]); head 0's Q
 * columns (token0=[1,0,0], token1=[0,1,0]) pick out raw_kv's first and
 * second entries, so C0 = [[1,4],[2,5]] (column-major, key-major); head 1's
 * Q columns (token0=[0,0,1], token1=[1,1,1]) pick out the third entry and
 * the row sum, so C1 = [[3,6],[6,15]]. */
extern "C" int ds4_sycl_test_gemm_batch_smoke(void) {
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const int64_t head_dim = 3, n_keys = 2, n_tokens = 2, n_head = 2;
        const int64_t ldb = head_dim + 1;

        const float h_kv[6] = {1, 2, 3, 4, 5, 6};
        const float h_q[16] = {
                1, 0, 0, -9,  0, 1, 0, -9,   /* head 0, one pad row per column */
                0, 0, 1, -9,  1, 1, 1, -9};  /* head 1 */
        const float expected[8] = {
                1, 4, 2, 5,    /* head 0 */
                3, 6, 6, 15};  /* head 1 */

        float *a = sycl::malloc_device<float>((size_t)(head_dim * n_keys), q);
        float *b = sycl::malloc_device<float>((size_t)(ldb * n_tokens * n_head), q);
        float *c = sycl::malloc_device<float>((size_t)(n_keys * n_tokens * n_head), q);
        if (!a || !b || !c) {
            if (a) sycl::free(a, q);
            if (b) sycl::free(b, q);
            if (c) sycl::free(c, q);
            return 0;
        }
        sycl_device_scratch_guard a_guard(q, a);
        sycl_device_scratch_guard b_guard(q, b);
        sycl_device_scratch_guard c_guard(q, c);

        sycl::event _ds4_prof_ev65 = q.memcpy(a, h_kv, sizeof(h_kv));
        sycl_batch_wait(_ds4_prof_ev65);
        ds4_sycl_profile_record(_ds4_prof_ev65);
        sycl::event _ds4_prof_ev66 = q.memcpy(b, h_q, sizeof(h_q));
        sycl_batch_wait(_ds4_prof_ev66);
        ds4_sycl_profile_record(_ds4_prof_ev66);

        sycl::event ev = sycl_gemm_batch_f32(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                n_keys, n_tokens, head_dim,
                1.0f,
                a, head_dim, 0,
                b, ldb, ldb * n_tokens,
                0.0f,
                c, n_keys, n_keys * n_tokens,
                n_head);
        sycl_batch_wait(ev);
        ds4_sycl_profile_record(ev);

        float got[8];
        sycl::event _ds4_prof_ev67 = q.memcpy(got, c, sizeof(got));
        sycl_batch_wait(_ds4_prof_ev67);
        ds4_sycl_profile_record(_ds4_prof_ev67);

        for (int i = 0; i < 8; i++) {
            const float diff = got[i] - expected[i];
            if (diff < -1e-4f || diff > 1e-4f) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "gemm_batch smoke mismatch at %d: got %f want %f\n",
                        i, (double)got[i], (double)expected[i]);
                return 0;
            }
        }
    } catch (const std::exception &e) {
        /* std::exception, not sycl::exception: oneapi::mkl::invalid_argument
         * is on a different branch of the hierarchy and a sycl::exception
         * handler does not catch it, so an uncaught one terminates the
         * process rather than failing this call (spec 6v). */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "gemm_batch smoke failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Converts an F16 weight table (bit-pattern uint16_t, row-major out_dim
 * rows of in_dim half values each) to F32, one work-item per element.
 * Shared by both weight offsets of ds4_gpu_matmul_f16_pair_tensor below.
 *
 * This used to be a per-element dequant fused directly
 * into a one-work-item-per-output-element dot-product kernel
 * (sycl_matmul_f16_launch, ranked the most expensive dense-matmul kernel
 * in measurement, 5.530 ms/layer-eval). Splitting the dequant
 * out into its own embarrassingly-parallel elementwise kernel, then
 * handing the actual dot products to oneMKL's F32 GEMM
 * (sycl_gemm_batch_f32, ds4_sycl_common.hpp, the same helper the
 * attention prefill path already uses), is the F32/F16 half of the two
 * different problems being solved: a GEMM library can tile and
 * cooperate across a whole row in a way a single serial work-item
 * cannot.
 *
 * F32 output, not a native half-precision GEMM, is a deliberate choice:
 * oneMKL's (half, half, half, half) GEMM instantiation this backend uses
 * elsewhere (sycl_gemm_f16) only produces a half output, but this entry's
 * ABI contract is an F32 output tensor, so a native half GEMM would still
 * need a second unpack kernel afterward -- no cheaper than this approach
 * -- while also computing in lower precision than the naive kernel it
 * replaces did. Expanding the weights to F32 once and letting the GEMM's
 * own (already float) internal accumulation do the rest loses no more
 * precision than the kernel being replaced. */
static sycl::event sycl_f16_to_f32_launch(sycl::queue &q, float *out,
                                   const uint16_t *w, uint64_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        out[gid] = (float)sycl::bit_cast<sycl::half>(w[gid]);
    });
}

/* Core dense F16 matmul entry: out[t][o] = sum_k x[t][k] * f16_to_f32(w[o][k]).
 * ROCm's own implementation (rocm/ds4_rocm_matmul.cuh:804-869) has a
 * genuine non-cuBLAS kernel fallback, gated the same way the Q8_0 GEMM
 * fast path is (g_cublas_ready && n_tok > 1); this port instead always
 * takes the oneMKL GEMM path (see sycl_f16_to_f32_launch's own comment for
 * why), now that measurement shows the previous
 * hand-written kernel was the single most expensive dense-matmul kernel on
 * a real layer-eval. */
extern "C" int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t weight_bytes = 0;
    if (!sycl_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size, "f16");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        uint16_t *dw = (uint16_t *)dw_guard.p;
        if (!dw) return 0;

        float *w_f32 = sycl::malloc_device<float>((size_t)(out_dim * in_dim), q);
        if (!w_f32) return 0;
        sycl_device_scratch_guard wf32_guard(q, w_f32);
        sycl::event ev_cvt = sycl_f16_to_f32_launch(q, w_f32, dw, out_dim * in_dim);
        /* No wait here. q is in_order (ds4_sycl.cpp), so the GEMM
         * submitted just below starts only once this convert kernel has
         * finished; ev.wait_and_throw() below is the one wait this
         * function needs, since it runs before dw_guard/wf32_guard free
         * their scratch at scope exit and it also covers everything
         * submitted to q before it, ev_cvt included. */
        ds4_sycl_profile_record_named("matmul_f16_convert", ev_cvt);

        /* out (n_tok x out_dim row-major) as column-major is (out_dim x
         * n_tok); m=out_dim, n=n_tok, k=in_dim, matching sycl_gemm_batch_f32's
         * own comment and the raw-KV prefill GEMM's argument shape
         * (ds4_sycl_attention.hpp:2152): the weight-like operand (row-major
         * out_dim x in_dim, so column-major in_dim x out_dim) is transa=trans
         * to get out_dim x in_dim; the activation-like operand (row-major
         * n_tok x in_dim, so column-major in_dim x n_tok already) is
         * transb=nontrans. batch_size=1 since this is a single weight table,
         * not a batch. */
        sycl::event ev = sycl_gemm_batch_f32(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                1.0f,
                w_f32, (int64_t)in_dim, 0,
                (const float *)x->ptr, (int64_t)in_dim, 0,
                0.0f,
                /* stride_c must be a real (nonzero) per-batch stride even
                 * at batch_size == 1: unlike stride_a, oneMKL has no
                 * broadcast case for C and raises invalid_argument on 0
                 * (sycl_gemm_batch_f32's own comment, ds4_sycl_common.hpp;
                 * confirmed empirically here too). */
                (float *)out->ptr, (int64_t)out_dim, (int64_t)out_dim * (int64_t)n_tok,
                1);
        sycl_batch_wait(ev);
        ds4_sycl_profile_record_named("matmul_f16", ev);
    } catch (const std::exception &e) {
        /* std::exception, not sycl::exception: oneapi::mkl::invalid_argument
         * is on a different branch of the hierarchy (spec 6v; see the
         * gemm_batch smoke test's own comment above for the exact wording),
         * and a sycl::exception-only handler here would let a bad GEMM
         * argument terminate the whole process instead of this entry
         * returning its failure value. */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_f16 failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Paired dense F16 projection: two independent weight tables applied to
 * the same activations in one call.  Validation ported from
 * rocm/ds4_rocm_matmul.cuh:873-931 (the ROCm entry's own bounds/overflow
 * checks), generalised to every n_tok the way ds4_gpu_matmul_f16_tensor's
 * own validation is (rocm/ds4_rocm_matmul.cuh:804-814), rather than only
 * n_tok == 1 the way ROCm's perf-tuned kernel-selection branch is.  ROCm
 * calls ds4_gpu_matmul_f16_tensor twice for n_tok != 1 and ANDs the
 * results; this entry keeps its own pre-existing one-general-kernel-for-
 * every-n_tok design instead (mathematically identical, a call-structure
 * difference only), now that ds4_gpu_matmul_f16_tensor above is real
 * rather than stubbed. */
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

    try {
        sycl::queue &q = ds4_sycl_queue(out_a->device_id);

        sycl_device_scratch_guard dwa_guard = sycl_stage_host_bytes(q, wa_ptr, weight_bytes);
        uint16_t *dwa = (uint16_t *)dwa_guard.p;
        if (!dwa) return 0;
        sycl_device_scratch_guard dwb_guard = sycl_stage_host_bytes(q, wb_ptr, weight_bytes);
        uint16_t *dwb = (uint16_t *)dwb_guard.p;
        if (!dwb) return 0;

        float *wa_f32 = sycl::malloc_device<float>((size_t)(out_dim * in_dim), q);
        if (!wa_f32) return 0;
        sycl_device_scratch_guard wa32_guard(q, wa_f32);
        float *wb_f32 = sycl::malloc_device<float>((size_t)(out_dim * in_dim), q);
        if (!wb_f32) return 0;
        sycl_device_scratch_guard wb32_guard(q, wb_f32);

        sycl::event ev_cvt_a = sycl_f16_to_f32_launch(q, wa_f32, dwa, out_dim * in_dim);
        sycl::event ev_cvt_b = sycl_f16_to_f32_launch(q, wb_f32, dwb, out_dim * in_dim);
        /* No wait here, same reasoning as
         * ds4_gpu_matmul_f16_tensor above -- q is in_order, so both GEMMs
         * below start only after both converts finish, and ev_b's wait
         * further down is the one wait this function needs. */
        ds4_sycl_profile_record_named("matmul_f16_convert", ev_cvt_a);
        ds4_sycl_profile_record_named("matmul_f16_convert", ev_cvt_b);

        sycl::event ev_a = sycl_gemm_batch_f32(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                1.0f,
                wa_f32, (int64_t)in_dim, 0,
                (const float *)x->ptr, (int64_t)in_dim, 0,
                0.0f,
                /* stride_c must be nonzero even at batch_size == 1; see
                 * ds4_gpu_matmul_f16_tensor's own comment above. */
                (float *)out_a->ptr, (int64_t)out_dim, (int64_t)out_dim * (int64_t)n_tok,
                1);
        sycl::event ev_b = sycl_gemm_batch_f32(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                1.0f,
                wb_f32, (int64_t)in_dim, 0,
                (const float *)x->ptr, (int64_t)in_dim, 0,
                0.0f,
                (float *)out_b->ptr, (int64_t)out_dim, (int64_t)out_dim * (int64_t)n_tok,
                1);
        /* ev_a's own wait is redundant and dropped -- q is
         * in_order, so ev_b (submitted right after ev_a on the same
         * queue) cannot complete before ev_a does, and ev_b's wait below
         * is the one wait this function needs: it runs before
         * dwa_guard/dwb_guard/wa32_guard/wb32_guard free their scratch at
         * scope exit, and it covers everything submitted to q before it,
         * ev_a and both converts included. */
        sycl_batch_wait(ev_b);
        ds4_sycl_profile_record_named("matmul_f16", ev_a);
        ds4_sycl_profile_record_named("matmul_f16", ev_b);
    } catch (const std::exception &e) {
        /* std::exception, not sycl::exception: see
         * ds4_gpu_matmul_f16_tensor's own comment above for why. */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_f16_pair failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Submits (without waiting) out[t][o] = sum_k x[t][k] * dequant(w[o][k])
 * for one Q8_0-quantised weight table, row-major out_dim rows of row_bytes
 * bytes each (blocks of 32 values, 34 bytes per block; see
 * sycl_q8_0_dequant in ds4_sycl_common.hpp).  One work-item per
 * (token, out_row) pair, covering every n_tok with no shape gating.
 *
 * This scalar form is kept only as the fallback
 * sycl_q8_0_matmul_general uses when a row is too wide for the tiled
 * kernel's local-memory activation stage below (see that kernel's own
 * comment); every call this backend's dense weight shapes actually reach
 * takes the tiled path instead, which measured faster on the real
 * layer-eval. */
static sycl::event sycl_q8_0_matmul_launch(sycl::queue &q, float *out,
                                    const unsigned char *w, const float *x,
                                    uint32_t in_dim, uint32_t out_dim,
                                    uint32_t n_tok, uint64_t row_bytes) {
    const uint64_t n = (uint64_t)n_tok * out_dim;
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
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

/* Tiled Q8_0 dense matmul: replaces
 * sycl_q8_0_matmul_launch's one-work-item-per-output-element shape, which
 * profiling showed dominating this backend's real per-layer
 * kernel time (matmul_q8_0 ranked #1 in that measurement).
 * That kernel has a single work-item serially walk the whole in_dim for
 * one output element; this one instead gives every output row a
 * kSubgroup-wide sub-group to split the row across, reducing with a
 * shuffle (sycl_subgroup_sum, ds4_sycl_common.hpp), and stages the
 * token's activation row into local memory once per work-group so every
 * output row the group computes reads it from local memory instead of
 * global. Mirrors the shape of ds4_sycl_moe.hpp's Q4_K tile kernels
 * (e.g. sycl_moe_q4k_gate_up_mid_tile8): local-memory activation staging
 * plus a sub-group cooperative reduction, one work-group per output tile,
 * matching the established pattern for quantised-weight tile kernels.
 *
 * kSubgroup is 16, already in kRequiredSubGroupWidths
 * (ds4_sycl_common.hpp), so ds4_gpu_init's device-capability guard already
 * checks it; no change needed there.
 *
 * The activation row is staged in full (in_dim floats) rather than in a
 * bounded sub-tile: every dense Q8_0 weight table in the DeepSeek V4
 * Flash/Pro shapes has in_dim <= 4096 (16 KiB), comfortably inside Arc's
 * local-memory budget, so this is not a partial-tile scheme. The caller
 * (sycl_q8_0_matmul_general) checks the row fits the device's actual
 * reported local-memory size before choosing this kernel over the scalar
 * fallback above, so an unexpectedly wide in_dim degrades to a working
 * (if slower) kernel instead of an oversized local_accessor allocation
 * failing at submission time. */
static constexpr int      kQ8_0TileSubgroup     = 16;
static constexpr uint32_t kQ8_0TileGroupSize    = 256u;
static constexpr uint32_t kQ8_0TileRowsPerGroup =
        kQ8_0TileGroupSize / (uint32_t)kQ8_0TileSubgroup;

static sycl::event sycl_q8_0_matmul_tiled_launch(sycl::queue &q, float *out,
                                    const unsigned char *w, const float *x,
                                    uint32_t in_dim, uint32_t out_dim,
                                    uint32_t n_tok, uint64_t row_bytes) {
    const uint32_t row_blocks =
            (out_dim + kQ8_0TileRowsPerGroup - 1u) / kQ8_0TileRowsPerGroup;
    return q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> x_local(sycl::range<1>(in_dim), h);
        h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)row_blocks * kQ8_0TileGroupSize, n_tok),
                        sycl::range<2>(kQ8_0TileGroupSize, 1u)),
                [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kQ8_0TileSubgroup)]] {
                    const uint32_t tok = (uint32_t)it.get_group(1);
                    const uint32_t lid = (uint32_t)it.get_local_id(0);
                    const uint32_t group_size = (uint32_t)it.get_local_range(0);

                    /* Stage this token's activation row once; every lane
                     * in the work-group participates in this strided
                     * fill regardless of which output row it will later
                     * compute, so every element up to in_dim is written
                     * by some lane before the barrier (spec 6b: an idle
                     * lane must never be the reason a local-memory slot
                     * goes unwritten). */
                    const float *xr = x + (uint64_t)tok * in_dim;
                    for (uint32_t i = lid; i < in_dim; i += group_size) {
                        x_local[i] = xr[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    const sycl::sub_group sg = it.get_sub_group();
                    const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                    const uint32_t row_in_group = lid / (uint32_t)kQ8_0TileSubgroup;
                    const uint32_t row = (uint32_t)it.get_group(0) * kQ8_0TileRowsPerGroup
                                        + row_in_group;
                    if (row >= out_dim) return;

                    const unsigned char *wr = w + (uint64_t)row * row_bytes;
                    float sum = 0.0f;
                    for (uint32_t k = lane; k < in_dim; k += (uint32_t)kQ8_0TileSubgroup) {
                        sum += x_local[k] * sycl_q8_0_dequant(wr, k);
                    }
                    sum = sycl_subgroup_sum<kQ8_0TileSubgroup>(sg, sum);
                    if (lane == 0u) {
                        out[(uint64_t)tok * out_dim + row] = sum;
                    }
                });
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
 * - The cuBLAS/GEMM fast path (rocm/ds4_rocm_matmul.cuh:325-330,
 *   `in_dim == 2048 && out_dim == 4096`, calling
 *   cuda_matmul_q8_0_tensor_f16_gemm) is still not taken, even though
 *   oneMKL is now wired in (unlike ds4_gpu_matmul_q8_0_f16_out_tensor
 *   below, which genuinely needs it). What ROCm's fast path actually needs
 *   is cuda_q8_f16_ptr's cross-call F16-expansion CACHE (keyed by model
 *   offset, with cuda_q8_f16_cache_disable_after_failure fallback logic on
 *   a GEMM failure): without that cache, taking this path would redo the
 *   Q8_0-to-F16 dequant on every single call at this shape, which is not
 *   "cheap" the way a fast path needs to be, and this backend has
 *   no weight-expansion cache of any kind yet (the per-call staging
 *   throughout this file is a documented placeholder, not a cache). Every
 *   call at this shape goes through the native kernel below, unchanged.
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

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        unsigned char *dw = (unsigned char *)dw_guard.p;
        if (!dw) return 0;

        /* The tiled kernel stages one in_dim-wide activation row into
         * local memory per work-group; fall back to the scalar kernel
         * (no local memory needed) rather than risk an oversized
         * local_accessor allocation on a device or shape this margin does
         * not anticipate. 4096 bytes of headroom for whatever else the
         * runtime reserves per work-group, checked against this specific
         * device rather than assumed, since local memory size is not
         * uniform across every SYCL device this backend might run on. */
        const uint64_t x_local_bytes = (uint64_t)in_dim * sizeof(float);
        const uint64_t local_mem_size =
                (uint64_t)q.get_device().get_info<sycl::info::device::local_mem_size>();
        const bool use_tiled = x_local_bytes + 4096u <= local_mem_size;

        sycl::event ev = use_tiled
                ? sycl_q8_0_matmul_tiled_launch(
                        q, (float *)out->ptr, dw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes)
                : sycl_q8_0_matmul_launch(
                        q, (float *)out->ptr, dw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes);
        /* Wait kept -- this function's only kernel, and dw_guard
         * may own scratch this function frees on return. */
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named("matmul_q8_0", ev);
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

/* Tensor parallelism: the split-matmul family. Each TP rank owns
 * a contiguous, block-aligned K range of a Q8_0 weight table; this entry
 * computes that rank's partial projection, which the caller sums with the
 * peer's partial result (ds4_gpu_add_xdev_tensor, ds4_sycl_mgpu.hpp) or
 * folds directly into an HC combine (ds4_gpu_hc_expand_add_tensor,
 * ds4_sycl_hc.hpp) to reconstruct the full-width result.
 *
 * "Rows" here are input rows: each of the n_tok rows of x holds only the
 * owned in_count-wide slice (contiguous, pre-sliced by the caller), while
 * every output row spans the full out_dim width -- ds4_gpu.h's own comment
 * on the docstring for the singular ds4_gpu_matmul_q8_0_kslice_tensor
 * below. Ported from ds4_cuda.cu's real kernel pair
 * (quantize_q8_0_f32_kernel + matmul_q8_0_kslice_preq_warp8_kernel):
 * this port skips CUDA's activation-side int8 prequantisation and dp4a
 * fast path, for the same reason sycl_q8_0_matmul_general above already
 * declines it (no direct SYCL equivalent worth chasing for a first port;
 * a plain per-column dequant-and-accumulate is mathematically identical,
 * only slower). Validation ported from ds4_cuda.cu:14859-14884 exactly,
 * MINUS its `n_tok > 65535` check, which is a CUDA grid-dimension limit
 * (blockIdx.y), not a semantic contract requirement, and has no SYCL
 * analogue worth inventing.
 *
 * Weight staging copies the FULL row (out_dim * full_blocks * 34 bytes,
 * same as ds4_cuda.cu's cuda_resolve_weight_ptr call): the model file
 * itself is not physically sharded by rank, only the K-range each rank
 * reads from it is, so this matches the CUDA reference's own behaviour,
 * not a missed optimisation. NONZERO means success, matching ds4_gpu.h
 * and ds4_cuda.cu's own `cuda_ok(...)` return. */
extern "C" int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                out_dim,
        uint64_t                in_start,
        uint64_t                in_count,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out || !x || !model_map || full_in_dim == 0u || out_dim == 0u ||
        in_count == 0u || n_tok == 0u || full_in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX) {
        return 0;
    }
    if ((in_start % 32u) != 0u || (in_count % 32u) != 0u ||
        in_start > full_in_dim || in_count > full_in_dim - in_start) {
        return 0;
    }

    uint64_t full_row_bytes = 0, weight_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(full_in_dim, &full_row_bytes) ||
        !sycl_u64_mul_checked(out_dim, full_row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_count, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "q8_0_kslice_rows");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        unsigned char *dw = (unsigned char *)dw_guard.p;
        if (!dw) return 0;

        float       *out_ptr = (float *)out->ptr;
        const float *x_ptr   = (const float *)x->ptr;
        const uint32_t out_dim32   = (uint32_t)out_dim;
        const uint32_t n_tok32     = (uint32_t)n_tok;
        const uint32_t in_start32  = (uint32_t)in_start;
        const uint32_t in_count32  = (uint32_t)in_count;
        const uint64_t n = (uint64_t)n_tok32 * out_dim32;
        sycl::event _ds4_prof_ev70 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            const uint32_t o = (uint32_t)(gid % out_dim32);
            const uint32_t t = (uint32_t)(gid / out_dim32);
            const float         *xr = x_ptr + (uint64_t)t * in_count32;
            const unsigned char *wr = dw + (uint64_t)o * full_row_bytes;
            float sum = 0.0f;
            for (uint32_t k = 0; k < in_count32; k++) {
                sum += xr[k] * sycl_q8_0_dequant(wr, in_start32 + k);
            }
            out_ptr[gid] = sum;
        });
        /* Wait kept -- this function's only kernel, and dw_guard
         * may own scratch this function frees on return. */
        sycl_batch_wait(q);
        ds4_sycl_profile_record(_ds4_prof_ev70);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_q8_0_kslice_rows failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Single-token convenience wrapper: builds a k_cnt-wide view of x starting
 * at x_elem_off and delegates to the rows variant above with n_tok = 1,
 * matching ds4_cuda.cu:30649-30671 exactly (it does the same x-slice
 * construction and the same delegation, to ds4_gpu_matmul_q8_0_kslice_rows_
 * tensor there too). Checked directly against every call site in ds4.c:
 * this entry currently has ZERO call sites anywhere in the repo (confirmed
 * by grep, matching stub-reachability-v2.md's own finding), so unlike the
 * rows variant above this one is unreachable dead code today. Kept as a
 * live, correct wrapper anyway (cheap given the rows variant already
 * exists, and it is part of the authoritative 26-entry ABI list), not
 * "Metal-only, out of scope" the way this file used to say: that claim was
 * wrong for the row-batched sibling, which DOES have real call sites, so
 * it was corrected rather than copied forward. */
extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    if (!x) return 0;
    uint64_t x_elems = x->bytes / sizeof(float);
    if (x_elem_off > x_elems || k_cnt > x_elems - x_elem_off) return 0;

    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset, full_in_dim, out_dim,
            k_off, k_cnt, &x_slice, 1u);
}

namespace {

/* Dequantises an entire Q8_0 weight table to F16, row-major out_dim rows of
 * in_dim half values each, one work-item per (out_row, column). Mirrors
 * cuda_q8_f16_ptr's expansion (rocm/ds4_rocm_matmul.cuh, referenced from
 * cuda_matmul_q8_0_tensor_f16_gemm_out_half) minus its cross-call cache:
 * this backend has no weight-expansion cache yet (see the comment on
 * sycl_q8_0_matmul_general above), so every call redoes the expansion, a
 * documented placeholder like every other per-call weight staging in this
 * file. */
static sycl::event sycl_q8_0_to_f16_launch(sycl::queue &q, sycl::half *out,
                                    const unsigned char *w, uint32_t in_dim,
                                    uint32_t out_dim, uint64_t row_bytes) {
    const uint64_t n = (uint64_t)out_dim * in_dim;
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t k = (uint32_t)(gid % in_dim);
        const uint32_t o = (uint32_t)(gid / in_dim);
        const unsigned char *row = w + (uint64_t)o * row_bytes;
        out[gid] = sycl::half(sycl_q8_0_dequant(row, k));
    });
}

/* Converts n F32 activations to F16, mirroring f32_to_f16_kernel
 * (rocm/ds4_rocm_common.cuh:345-347, `__float2half`): the hardware's own
 * native round-to-nearest-even conversion, NOT
 * sycl_f32_to_f16_bits_hip_round's ported round-half-up bit trick, which
 * exists for a different call site per spec 6k (this backend's own
 * explicit F16 encoder, used where ds4 stores F16 state directly rather
 * than handing it to a GEMM library's own internal cast). */
static sycl::event sycl_f32_to_f16_launch(sycl::queue &q, sycl::half *out,
                                   const float *x, uint64_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        out[gid] = sycl::half(x[gid]);
    });
}

}  // namespace

/* ds4_gpu_matmul_q8_0_f16_out_tensor, rocm/ds4_rocm_matmul.cuh:216-291
 * (cuda_matmul_q8_0_tensor_f16_gemm_out_half): unlike every other entry in
 * this file, ROCm has NO non-GEMM fallback here at all (unconditional
 * `if (!g_cublas_ready ...) return 0;`), so oneMKL is genuinely load-
 * bearing correctness here, not a performance choice the way it is for
 * ds4_gpu_matmul_f32_tensor and ds4_gpu_matmul_f16_tensor above.
 *
 * Two independent device-side conversions (the Q8_0-to-F16 weight dequant
 * and the F32-to-F16 activation convert) each get their own
 * wait_and_throw() before the GEMM reads them, the same trailing-wait
 * discipline used throughout this file, rather than passing them as an
 * event-dependency vector: the two conversions do not depend on each
 * other, only the GEMM depends on both having finished. The GEMM writes
 * directly into out_h->ptr, reinterpreted as sycl::half*: that is exactly
 * the two-byte-per-element binary16 layout every F16 ABI tensor already
 * uses in this backend, so no separate output scratch or unpack kernel is
 * needed.  alpha/beta are passed as sycl::half (1.0 and 0.0, both exactly
 * representable) since oneMKL's closest available USM gemm instantiation
 * to cuBLAS's CUDA_R_16F-in/out, CUBLAS_COMPUTE_32F call is
 * (half, half, half, half); see sycl_gemm_f16's own comment
 * (ds4_sycl_common.hpp) for why that type mismatch on alpha/beta carries
 * no numerical consequence here. */
extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out_h || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out_h, n_tok, out_dim, sizeof(uint16_t))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size,
                                            "q8_0_f16_out");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out_h->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        unsigned char *dw = (unsigned char *)dw_guard.p;
        if (!dw) return 0;

        sycl::half *w_h = sycl::malloc_device<sycl::half>(
                (size_t)(out_dim * in_dim), q);
        if (!w_h) return 0;
        sycl_device_scratch_guard wh_guard(q, w_h);
        sycl::event _ds4_prof_ev71 = sycl_q8_0_to_f16_launch(
                q, w_h, dw, (uint32_t)in_dim, (uint32_t)out_dim, row_bytes);
        /* No wait here or after the activation convert below.
         * The function comment above (written before in_order queues)
         * still correctly describes the two converts as independent of
         * each other; what changes is how that independence is honoured.
         * q is in_order, so both still run in submission order regardless
         * -- in_order queue semantics never permitted the runtime to skip
         * ahead to the GEMM early, waited or not -- and ev.wait_and_throw()
         * below, which runs before dw_guard/wh_guard/xh_guard free their
         * scratch, is the one wait this function needs; it covers both
         * converts too. */
        ds4_sycl_profile_record(_ds4_prof_ev71);

        sycl::half *x_h = sycl::malloc_device<sycl::half>(
                (size_t)(n_tok * in_dim), q);
        if (!x_h) return 0;
        sycl_device_scratch_guard xh_guard(q, x_h);
        sycl::event _ds4_prof_ev71b =
                sycl_f32_to_f16_launch(q, x_h, (const float *)x->ptr, n_tok * in_dim);
        ds4_sycl_profile_record(_ds4_prof_ev71b);

        sycl::event ev = sycl_gemm_f16(
                q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                sycl::half(1.0f),
                w_h, (int64_t)in_dim,
                x_h, (int64_t)in_dim,
                sycl::half(0.0f),
                (sycl::half *)out_h->ptr, (int64_t)out_dim);
        /* Wait kept -- covers both converts above plus this GEMM,
         * and dw_guard/wh_guard/xh_guard may each own scratch this
         * function frees on return. */
        sycl_batch_wait(ev);
        ds4_sycl_profile_record(ev);
    } catch (const std::exception &e) {
        /* std::exception, not sycl::exception: a bad argument to
         * sycl_gemm_f16 throws oneapi::mkl::invalid_argument synchronously
         * (oneapi::mkl::exception -> std::exception, a different hierarchy
         * from sycl::exception, which also derives from std::exception, so
         * this one clause still catches both). Found by an ablation in the
         * raw-prefill entry that terminated the whole process instead of
         * this entry returning its failure value, when the catch clause
         * here still only caught sycl::exception. */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_q8_0_f16_out failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
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

namespace {

/* Mirrors matmul_f32_kernel exactly (rocm/ds4_rocm_common.cuh:288-306): one
 * work-item per (token, out_row) output element, a plain accumulate loop
 * over in_dim. */
static sycl::event sycl_matmul_f32_launch(sycl::queue &q, float *out, const float *w,
                                   const float *x, uint32_t in_dim,
                                   uint32_t out_dim, uint32_t n_tok) {
    const uint64_t n = (uint64_t)n_tok * out_dim;
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t o  = (uint32_t)(gid % out_dim);
        const uint32_t t  = (uint32_t)(gid / out_dim);
        const float *xr = x + (uint64_t)t * in_dim;
        const float *wr = w + (uint64_t)o * in_dim;
        float sum = 0.0f;
        for (uint32_t k = 0; k < in_dim; k++) sum += xr[k] * wr[k];
        out[gid] = sum;
    });
}

}  // namespace

/* Core dense F32 matmul entry.  ROCm's own implementation
 * (rocm/ds4_rocm_matmul.cuh:967-991) has a genuine non-cuBLAS kernel
 * fallback (matmul_f32_kernel), gated the same way the Q8_0 GEMM fast path
 * is (g_cublas_ready && n_tok > 1). Kept as this hand-written kernel
 * rather than converted to a GEMM the way ds4_gpu_matmul_f16_tensor and
 * ds4_gpu_matmul_q8_0_tensor above were: real-layer-eval measurement
 * shows zero calls to this entry at
 * all in the DeepSeek V4 Flash checkpoint this backend targets (every
 * dense weight tensor in that shape is Q8_0 or F16, never plain F32), so
 * rewriting it cannot be measured against the real workload and would be
 * exactly the kind of unranked, unmeasured work worth avoiding. Revisit
 * if a checkpoint that actually uses F32 dense weights shows up in a
 * future profile. */
extern "C" int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t weight_bytes = 0;
    if (!sycl_u64_mul3_checked(out_dim, in_dim, sizeof(float), &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size, "f32");
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        float *dw = (float *)dw_guard.p;
        if (!dw) return 0;

        sycl::event _ds4_prof_ev72 = sycl_matmul_f32_launch(
                q, (float *)out->ptr, dw, (const float *)x->ptr,
                (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
        /* Wait kept -- this function's only kernel, and dw_guard
         * may own scratch this function frees on return. */
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named("matmul_f32", _ds4_prof_ev72);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_f32 failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* ---- Dense Q4_K / Q4_0 matmul, plus the type-dispatching quant entries --
 *
 * ds4_gpu_matmul_quant_tensor and ds4_gpu_matmul_quant_kslice_tensor fire
 * whenever attn_q_a, attn_kv, attn_output_a/b or ffn_gate/up/down_shexp is
 * quantised Q4_K or Q4_0 rather than Q8_0: tensor_expect_dense_quant_layout
 * (ds4.c) accepts all three for those tensors, so a legitimate Flash
 * checkpoint can choose either, and weights_validate_layout does not pin
 * Q8_0 (contrast mtp_weights_validate_layout, which does, for the unrelated
 * MTP draft weights).
 *
 * CUDA is only a partial reference here: ds4_cuda.cu's own
 * ds4_gpu_matmul_quant_tensor has real dispatch logic, but only for its
 * Q8_0 and F16 cases -- its `default` branch rejects Q4_K/Q4_0 exactly like
 * this file's own rejects an unsupported type, so CUDA itself has never
 * run a Q4_K/Q4_0 dense-quant checkpoint through this entry. Only
 * ds4_metal.m genuinely implements Q4_K/Q4_0 here, through bespoke,
 * non-portable Metal compute shaders (kernel_mul_mv_q4_K_dense_f32 and
 * friends) with no scalar formula to transcribe line-for-line. What is
 * ported instead is the standard block-quantisation MATH those shaders (and
 * ds4.c's own ds4_vec_dot_q4_K_f32, :3709-3736, used by the GLM CPU
 * reference path) all compute: for Q4_K, value = d*sc*nibble - dmin*m per
 * column, via ds4_sycl_common.hpp's sycl_q4_k_dequant, itself sharing its
 * block struct and scale/min decode with the routed-MoE Q4_K path
 * rather than a second copy; for Q4_0, the standard GGUF layout (never
 * decoded anywhere in this codebase before now; see
 * sycl_q4_0_dequant's own comment). ds4_gpu_matmul_quant_kslice_tensor's
 * own ds4_cuda.cu definition is a plain unconditional stub even for Q8_0,
 * so it has no CUDA reference behaviour at all; see its own comment below
 * for why it is implemented anyway and what its real reachability turned
 * out to be. */

/* One work-item per (token, out_row) output element, generic over the
 * per-column dequantiser: `Dequant` is a stateless functor
 * `float(const unsigned char *row, uint32_t col)`, the same signature
 * sycl_q8_0_dequant/sycl_q4_k_dequant/sycl_q4_0_dequant already share.
 * Passed as a template parameter (a distinct functor type per call site,
 * always a capture-less lambda below) rather than a runtime function
 * pointer so the SYCL device compiler sees a direct, statically resolved
 * call inside the kernel body rather than an indirect one. One level more
 * general than sycl_q8_0_matmul_launch/sycl_matmul_f32_launch above, which
 * this could replace but does not, to keep this change scoped to the new
 * entries it actually needs. */
template <typename Dequant>
static sycl::event sycl_dense_quant_matmul_launch(sycl::queue &q, float *out,
                                           const unsigned char *w, const float *x,
                                           uint32_t in_dim, uint32_t out_dim,
                                           uint32_t n_tok, uint64_t row_bytes,
                                           Dequant dequant) {
    const uint64_t n = (uint64_t)n_tok * out_dim;
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t o  = (uint32_t)(gid % out_dim);
        const uint32_t t  = (uint32_t)(gid / out_dim);
        const float         *xr = x + (uint64_t)t * in_dim;
        const unsigned char *wr = w + (uint64_t)o * row_bytes;
        float sum = 0.0f;
        for (uint32_t k = 0; k < in_dim; k++) {
            sum += xr[k] * dequant(wr, k);
        }
        out[gid] = sum;
    });
}

/* Validation, staging and launch shared by the Q4_K and Q4_0 dense matmul
 * paths: the same shape as sycl_q8_0_matmul_general above, generalised over
 * the dequantiser and its row-bytes computation. */
template <typename Dequant>
static int sycl_dense_quant_matmul_general(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, const char *what,
        int (*row_bytes_checked)(uint64_t, uint64_t *), Dequant dequant) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_tok == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size, what);
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        unsigned char *dw = (unsigned char *)dw_guard.p;
        if (!dw) return 0;

        sycl::event _ds4_prof_ev73 = sycl_dense_quant_matmul_launch(
                q, (float *)out->ptr, dw,
                (const float *)x->ptr, (uint32_t)in_dim,
                (uint32_t)out_dim, (uint32_t)n_tok,
                row_bytes, dequant);
        /* Wait kept -- this function's only kernel, and dw_guard
         * may own scratch this function frees on return. */
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named(what, _ds4_prof_ev73);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s matmul failed: %s\n", what, e.what());
        return 0;
    }
    return 1;
}

static int sycl_q4_k_matmul_general(ds4_gpu_tensor *out, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const ds4_gpu_tensor *x, uint64_t n_tok) {
    return sycl_dense_quant_matmul_general(
            out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok,
            "q4_K", sycl_q4_k_row_bytes_checked,
            [](const unsigned char *row, uint32_t col) { return sycl_q4_k_dequant(row, col); });
}

static int sycl_q4_0_matmul_general(ds4_gpu_tensor *out, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const ds4_gpu_tensor *x, uint64_t n_tok) {
    return sycl_dense_quant_matmul_general(
            out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok,
            "q4_0", sycl_q4_0_row_bytes_checked,
            [](const unsigned char *row, uint32_t col) { return sycl_q4_0_dequant(row, col); });
}

/* Type-dispatching dense matmul entry. Return polarity and dispatch shape
 * both verified against ds4_cuda.cu's own ds4_gpu_matmul_quant_tensor
 * (switch on weight_type, nonzero-success, `default` rejects with a
 * diagnostic and returns 0) and against ds4.c's two call sites
 * (metal_graph_matmul_plain_tensor:25768-25778 and
 * metal_graph_matmul_dense_quant_abs:25802-25810, both `... != 0`).
 * weight_type is ds4's DS4_TENSOR_* enum value, passed numerically per this
 * backend's existing convention (no DS4_TENSOR_* constants are exposed to
 * backend code; see ds4_sycl_moe_launch.hpp's own comment on gate_type/
 * down_type). */
extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    switch (weight_type) {
    case 8u:   /* Q8_0 */
        return sycl_q8_0_matmul_general(out, model_map, model_size, weight_offset,
                                        in_dim, out_dim, x, n_tok);
    case 1u:   /* F16 */
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok);
    case 12u:  /* Q4_K */
        return sycl_q4_k_matmul_general(out, model_map, model_size, weight_offset,
                                        in_dim, out_dim, x, n_tok);
    case 2u:   /* Q4_0 */
        return sycl_q4_0_matmul_general(out, model_map, model_size, weight_offset,
                                        in_dim, out_dim, x, n_tok);
    default:
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
}

/* ds4_gpu_matmul_quant_kslice_tensor: out[out_dim] = W[:, k_off:k_off+k_cnt]
 * @ x[x_elem_off:+k_cnt], single token, generalising ds4_gpu.h's doc
 * comment on ds4_gpu_matmul_q8_0_kslice_tensor ("Tensor-parallel sliced
 * projections (Metal decode path only)") to every dense-quant type.
 *
 * Reachability, checked directly rather than inferred from the design
 * docs: both of ds4.c's call sites that reach this entry
 * (metal_graph_matmul_dense_quant_kslice at :24323's tp_split_shared branch,
 * :25009, and metal_graph_attention_output_dense_quant_tp, :25921, itself
 * only called from the g->tp_world==2 branch at :23542-23558) require
 * g->tp_world == 2, i.e. tensor parallelism -- permanently out of scope for
 * this single-GPU port (SYCL.md; stub-reachability.md's TP section). Unlike
 * the plain entry above, this one is NOT reached by single-GPU Flash decode
 * or prefill, contradicting the design table, which lists it alongside
 * the plain entry as firing under the same "any non-Q8_0/F16/F32
 * dense-quant weight" condition. ds4_cuda.cu's own definition is also a
 * plain unconditional `return 0` for every weight_type including Q8_0, so
 * there is no CUDA reference behaviour to check this against either.
 *
 * Implemented anyway, since it is listed as an interface to build, and
 * kept simple rather than staging only the addressed column range: the
 * whole weight table is staged and the kernel indexes the absolute column
 * k_off+k into it, rather than computing a per-row byte-range slice. That
 * tight-slice staging is what a genuinely TP-reachable version would want
 * for bandwidth, but this path is unreachable code today and not worth the
 * complexity until TP is in scope. */
template <typename Dequant>
static int sycl_dense_quant_matmul_kslice_general(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off, const char *what,
        int (*row_bytes_checked)(uint64_t, uint64_t *), Dequant dequant) {
    if (!out || !x || !model_map || full_in_dim == 0u || k_cnt == 0u ||
        out_dim == 0u || full_in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        k_off > full_in_dim || k_cnt > full_in_dim - k_off) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0, x_end = 0;
    if (!row_bytes_checked(full_in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !sycl_u64_add_checked(x_elem_off, k_cnt, &x_end) ||
        !sycl_tensor_has_f32(x, x_end) ||
        !sycl_tensor_has_f32(out, out_dim)) {
        return 0;
    }
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, model_size, what);
    if (!wptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);

        sycl_device_scratch_guard dw_guard = sycl_stage_host_bytes(q, wptr, weight_bytes);
        unsigned char *dw = (unsigned char *)dw_guard.p;
        if (!dw) return 0;

        float       *out_ptr = (float *)out->ptr;
        const float *x_ptr   = (const float *)x->ptr + x_elem_off;
        const uint32_t k_off32   = (uint32_t)k_off;
        const uint32_t k_cnt32   = (uint32_t)k_cnt;
        const uint32_t out_dim32 = (uint32_t)out_dim;
        sycl::event _ds4_prof_ev74 = q.parallel_for(sycl::range<1>(out_dim32), [=](sycl::id<1> gid) {
             const uint32_t o = (uint32_t)gid[0];
             const unsigned char *wr = dw + (uint64_t)o * row_bytes;
             float sum = 0.0f;
             for (uint32_t k = 0; k < k_cnt32; k++) {
                 sum += x_ptr[k] * dequant(wr, k_off32 + k);
             }
             out_ptr[o] = sum;
         });
         /* Wait kept -- this function's only kernel, and dw_guard
          * may own scratch this function frees on return. */
         sycl_batch_wait(_ds4_prof_ev74);
         ds4_sycl_profile_record(_ds4_prof_ev74);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s matmul_kslice failed: %s\n", what, e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    switch (weight_type) {
    case 8u:   /* Q8_0 */
        return sycl_dense_quant_matmul_kslice_general(
                out, model_map, model_size, weight_offset, full_in_dim, k_off,
                k_cnt, out_dim, x, x_elem_off, "q8_0_kslice",
                sycl_q8_0_row_bytes_checked,
                [](const unsigned char *row, uint32_t col) { return sycl_q8_0_dequant(row, col); });
    case 12u:  /* Q4_K */
        return sycl_dense_quant_matmul_kslice_general(
                out, model_map, model_size, weight_offset, full_in_dim, k_off,
                k_cnt, out_dim, x, x_elem_off, "q4_K_kslice",
                sycl_q4_k_row_bytes_checked,
                [](const unsigned char *row, uint32_t col) { return sycl_q4_k_dequant(row, col); });
    case 2u:   /* Q4_0 */
        return sycl_dense_quant_matmul_kslice_general(
                out, model_map, model_size, weight_offset, full_in_dim, k_off,
                k_cnt, out_dim, x, x_elem_off, "q4_0_kslice",
                sycl_q4_0_row_bytes_checked,
                [](const unsigned char *row, uint32_t col) { return sycl_q4_0_dequant(row, col); });
    default:
        fprintf(stderr, DS4_GPU_LOG_PREFIX "matmul_quant_kslice: unsupported type %u\n",
                weight_type);
        return 0;
    }
}
