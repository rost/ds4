#pragma once

/* Shared expert: a Q8_0 SwiGLU MLP that runs for every token on every
 * layer.  Ported from rocm/ds4_rocm_shared_expert.cuh, which is the
 * authority for all semantics here, plus two kernels from
 * rocm/ds4_rocm_q8.cuh.
 *
 * ROCm defines nine extern "C" symbols in that file; only six reach
 * ds4_gpu.h and only four of those need a real implementation:
 *   - ds4_gpu_shared_gate_up_swiglu_q8_0_tensor: the real work, fused fast
 *     path plus a general fallback (this file).
 *   - ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor: pure
 *     delegation to the above with identical arguments.
 *   - ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor: n_tok == 1 delegates
 *     to the above; n_tok > 1 uses the batched path below.
 *   - ds4_gpu_shared_mid_swiglu_q8_0_tensor: allocates gate/up scratch,
 *     then delegates to the core entry.
 *
 * ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor stays stubbed in
 * ds4_sycl_unavailable.cpp: ROCm's own implementation
 * (rocm/ds4_rocm_shared_expert.cuh:69-86) voids every argument and
 * unconditionally returns 0, so the SYCL stub is a faithful port of a real
 * stub in the reference implementation, not an omission.
 *
 * ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor: implemented
 * below by delegating to ds4_gpu_shared_mid_swiglu_q8_0_tensor, plus the
 * expert-parallel load-balance predicate and cross-device write CUDA's own
 * bespoke kernel adds. See that entry's own comment for why delegation is
 * safe here even though CUDA's version is a distinct kernel.
 *

 * Not ported at all: ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor is a
 * ROCm-internal helper reached only from the rows entry, ported below as
 * an internal static function rather than an extern "C" entry point.  The
 * async pair (ds4_gpu_shared_gate_up_swiglu_q8_0_async_tensor and
 * ds4_gpu_shared_gate_up_async_wait) is dead code in ROCm itself (nothing
 * calls the async entry; only its cleanup helper is wired into teardown)
 * and neither symbol is declared in ds4_gpu.h, so ds4.c cannot reach them
 * either way.  ds4_gpu_router_shared_gate_up_q8_0_tensor is Metal-only.
 *
 * Every entry in this file is NONZERO-means-success: validation failure
 * returns 0, a completed launch returns 1.  Verified against
 * rocm/ds4_rocm_shared_expert.cuh's own launchers and the ds4.c call
 * sites (ds4.c:41919, :42353, :42834, :42847, :48345, :54751, :55029).
 * There is no zero-work case in this surface: in_dim == 0 or out_dim == 0
 * is a hard rejection in every entry (rocm/ds4_rocm_shared_expert.cuh:19-21),
 * not a success, unlike ds4_gpu_swiglu_tensor's own n == 0 short-circuit.
 *
 * ROCm gates its fused fast path additionally on
 * !cuda_runtime_config()->disable_shared_gate_up_fused_w32, and its
 * store_gate_up flag is 1 only in quality mode or under a graph dump.
 * This backend has no runtime config object, and none is introduced here:
 * the fast path is taken whenever the shape qualifies, and gate/up are
 * ALWAYS stored.  Always storing is a strict superset of ROCm's
 * behaviour (no caller can observe fewer writes than it already
 * tolerates), it costs bandwidth rather than correctness, and it lets the
 * fast and general paths be differentially tested against each other on
 * gate and up, not only on mid. */

#include "ds4_sycl_common.hpp"

namespace {

/* Counts how many times the fused fast path below actually launches its
 * kernel, exposed to tests via ds4_sycl_shared_expert_test_fast_path_hits.
 * A correctness assertion on gate/up/mid cannot distinguish "the fast
 * kernel ran" from "the general path happened to compute the same
 * numbers", since both are the same dot product; this counter is the
 * observable side channel that lets a test confirm the shape gate is
 * actually selecting the branch it claims to, the same technique used by
 * g_sycl_stream_hits/g_sycl_stream_misses in ds4_sycl_streaming.hpp. */
uint64_t g_sycl_shared_fast_path_hits = 0;

}  // namespace

extern "C" uint64_t ds4_sycl_shared_expert_test_fast_path_hits(void) {
    return g_sycl_shared_fast_path_hits;
}

/* Fused fast path: shared_gate_up_swiglu_q8_0_rows_w32_kernel,
 * rocm/ds4_rocm_q8.cuh:948-1046.  One work-group of rows_per_block == 32
 * rows, one sub-group of 32 lanes per row, each lane owning one Q8_0
 * column of every block; the sub-group reduce below plays the role of
 * ROCm's warp_sum_f32.  Every block of in_dim is exactly 32 wide here
 * (the caller only takes this path when in_dim == 4096, a multiple of 32
 * with no remainder), so there is no partial-block guard on the K
 * dimension, matching the ROCm kernel exactly; the only partial-tile case
 * is the final row work-group when out_dim is not a multiple of 32,
 * guarded by `if (row >= out_dim) return;`.
 *
 * store_gate_up is not a parameter here: gate and up are ALWAYS written
 * (see the file header comment for why), unlike ROCm's kernel which takes
 * it as a runtime flag. */
static sycl::event sycl_shared_gate_up_swiglu_q8_0_rows_w32(
        sycl::queue &q, float *gate, float *up, float *mid,
        const unsigned char *wg, const unsigned char *wu, const float *x,
        uint32_t n_blocks, uint32_t out_dim, uint64_t row_bytes,
        float clamp) {
    constexpr uint32_t kRowsPerBlock = 32u;
    const uint32_t n_groups = (out_dim + kRowsPerBlock - 1u) / kRowsPerBlock;
    const size_t local_size = (size_t)kRowsPerBlock * 32u;

    return q.submit([&](sycl::handler &h) {
        h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)n_groups * local_size),
                                  sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const uint32_t tid = (uint32_t)it.get_local_id(0);
                    const uint32_t lane = tid & 31u;
                    const uint32_t wave = tid >> 5u;
                    const uint64_t row =
                            (uint64_t)it.get_group(0) * kRowsPerBlock + wave;
                    if (row >= out_dim) return;

                    const unsigned char *row_g = wg + row * row_bytes;
                    const unsigned char *row_u = wu + row * row_bytes;
                    float acc_g = 0.0f, acc_u = 0.0f;
                    for (uint32_t b = 0; b < n_blocks; b++) {
                        const uint32_t col = b * 32u + lane;
                        acc_g += x[col] * sycl_q8_0_dequant(row_g, col);
                        acc_u += x[col] * sycl_q8_0_dequant(row_u, col);
                    }

                    sycl::sub_group sg = it.get_sub_group();
                    const float g =
                            sycl::reduce_over_group(sg, acc_g, sycl::plus<float>());
                    const float u =
                            sycl::reduce_over_group(sg, acc_u, sycl::plus<float>());
                    if (sg.get_local_id()[0] == 0u) {
                        gate[row] = g;
                        up[row] = u;
                        float sg_val = g;
                        float su_val = u;
                        if (clamp > 1.0e-6f) {
                            sg_val = sycl::min(sg_val, clamp);
                            su_val = sycl::clamp(su_val, -clamp, clamp);
                        }
                        mid[row] = (sg_val / (1.0f + sycl::exp(-sg_val))) * su_val;
                    }
                });
    });
}

/* Core entry: the real work, a fused fast path plus a general fallback.
 * Ported from rocm/ds4_rocm_shared_expert.cuh:7-66.
 *
 * Validation mirrors rocm/ds4_rocm_shared_expert.cuh:19-33: null/shape
 * checks first, then overflow-safe row/weight/x/out byte counts, then a
 * capacity check on every tensor.  Note this level does NOT bounds-check
 * gate_offset/up_offset against model_size; ROCm only does that ahead of
 * its fast-path kernel launch (since it forms raw pointers into the
 * model map there), and the general path's own callee,
 * ds4_gpu_matmul_q8_0_pair_tensor, already performs that check itself
 * before forming its own pointers.
 *
 * Fast-path shape gate, rocm/ds4_rocm_shared_expert.cuh:35-38: in_dim ==
 * 4096 (the `(in_dim & 31u) == 0u` ROCm also checks is implied by
 * in_dim == 4096 but kept for fidelity to the source) AND both weight
 * ranges fit the mapped model.  ROCm's cuda_model_range_fits check is
 * ANDed INTO the fast-path condition, not a hard rejection ahead of it:
 * a weight range that does not fit falls through to the general path,
 * which performs its own (identical) range check and fails there
 * instead.  Ported exactly: a range that does not fit here also falls
 * through, it does not return 0 directly. ROCm additionally requires
 * !cuda_runtime_config()->disable_shared_gate_up_fused_w32; this backend
 * has no runtime config object, so the shape gate alone decides here (see
 * file header comment). */
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_u64_mul_checked(in_dim, sizeof(float), &x_bytes) ||
        !sycl_u64_mul_checked(out_dim, sizeof(float), &out_bytes) ||
        !sycl_tensor_has_bytes(x, x_bytes) ||
        !sycl_tensor_has_bytes(gate, out_bytes) ||
        !sycl_tensor_has_bytes(up, out_bytes) ||
        !sycl_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }

    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        sycl_model_range_fits(model_size, gate_offset, weight_bytes) &&
        sycl_model_range_fits(model_size, up_offset, weight_bytes)) {
        const char *wg = sycl_model_range_ptr(model_map, gate_offset,
                                              weight_bytes, model_size,
                                              "shared_gate_q8");
        const char *wu = sycl_model_range_ptr(model_map, up_offset,
                                              weight_bytes, model_size,
                                              "shared_up_q8");
        if (!wg || !wu) return 0;
        if (g_devices.empty()) return 0;

        try {
            sycl::queue &q = ds4_sycl_queue(gate->device_id);

            sycl_device_scratch_guard dwg_guard = sycl_stage_host_bytes(q, wg, weight_bytes);
            unsigned char *dwg = (unsigned char *)dwg_guard.p;
            if (!dwg) return 0;
            sycl_device_scratch_guard dwu_guard = sycl_stage_host_bytes(q, wu, weight_bytes);
            unsigned char *dwu = (unsigned char *)dwu_guard.p;
            if (!dwu) return 0;

            sycl::event ev = sycl_shared_gate_up_swiglu_q8_0_rows_w32(
                    q, (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    dwg, dwu, (const float *)x->ptr, (uint32_t)(in_dim >> 5u),
                    (uint32_t)out_dim, row_bytes, clamp);
            sycl_batch_wait(q);
            ds4_sycl_profile_record_named("shared_expert_gate_up_swiglu_q8_0_w32", ev);
            g_sycl_shared_fast_path_hits++;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "shared gate/up fused q8 launch failed: %s\n", e.what());
            return 0;
        }
        return 1;
    }

    /* General path, matching rocm/ds4_rocm_shared_expert.cuh:60-65.  The
     * short-circuit && matters: a failed pair matmul must skip the SwiGLU
     * and report failure, not run SwiGLU over stale/undefined gate and up
     * buffers and report success. */
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                           gate_offset, up_offset, in_dim,
                                           out_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp,
                                 1.0f);
}

/* ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor.  ROCm's own
 * definition (rocm/ds4_rocm_shared_expert.cuh:69-86) voids every argument
 * and unconditionally returns 0: a real stub in that reference, not an
 * oversight.  This backend deliberately DOES NOT port that stub: Metal
 * implements this entry for real (ds4_metal.m:19368, kernel
 * "kernel_dsv4_shared_gate_up_swiglu_q8_0"), ds4_gpu.h declares it as an
 * ordinary primitive with no "optional" framing, and it has a
 * Flash-reachable call site (ds4.c:62246,
 * metal_graph_encode_native_session_batch_shared, the generic non-GLM
 * batched-session path). ROCm being the structural reference exists to
 * keep semantics aligned, not to reproduce ROCm's own incomplete backend
 * coverage on an entry another real backend implements; the SYCL port of
 * ds4_gpu_matmul_q8_0_rows_scalar_tensor (sycl/ds4_sycl_matmul.hpp) made
 * the identical judgement call for the structurally analogous dense-matmul
 * entry. Do not "fix" this back to a stub for ROCm-consistency.
 *
 * n_tok == 1 delegates to the core entry above, matching Metal's own
 * implementation.  For n_tok > 1: unlike ROCm's fused batch kernel
 * (rocm/ds4_rocm_q8.cuh:524-586, reached from
 * ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor below), which takes no
 * clamp parameter at all and is why that entry refuses clamp > 1e-6f
 * rather than route through it, Metal's real
 * kernel_dsv4_shared_gate_up_swiglu_q8_0 DOES accept and apply clamp for
 * n_tok > 1.  This implementation matches Metal's capability, not ROCm's
 * limitation: ds4_gpu_matmul_q8_0_pair_tensor and ds4_gpu_swiglu_tensor
 * already accept an arbitrary token/element count in one launch, and
 * swiglu's clamp is applied elementwise regardless of how the flat buffer
 * divides into per-token rows, so generalising the general path above
 * over n_tok needs no new kernel and no clamp restriction. */
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1u) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                gate, up, mid, model_map, model_size, gate_offset, up_offset,
                in_dim, out_dim, x, clamp);
    }
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0u ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    uint64_t out_elems = 0;
    if (!sycl_u64_mul_checked(out_dim, n_tok, &out_elems) ||
        out_elems > UINT32_MAX) {
        return 0;
    }

    /* The short-circuit && matters here for the same reason as the core
     * entry above: a failed pair matmul must skip the SwiGLU rather than
     * run it over stale gate/up and report success. */
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                           gate_offset, up_offset, in_dim,
                                           out_dim, out_dim, x, n_tok) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_elems, clamp,
                                 1.0f);
}

/* Pure delegation to the core entry with identical arguments,
 * rocm/ds4_rocm_shared_expert.cuh:133-156. */
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid, model_map,
                                                     model_size, gate_offset,
                                                     up_offset, in_dim,
                                                     out_dim, x, clamp);
}

/* Allocates a 2 * out_dim scratch block, carves gate and up out of it, and
 * delegates to the core entry, rocm/ds4_rocm_shared_expert.cuh:101-131.
 * Validates out_dim but NOT in_dim before allocating, matching ROCm
 * exactly; an in_dim == 0 (or otherwise invalid) call still fails, just
 * one level down, inside the delegated core entry, after this wrapper's
 * scratch allocation has already succeeded.
 *
 * ROCm's own carved tensors use a three-element brace initialiser against
 * the four-field struct (ds4_gpu_mgpu.h:39: ptr, bytes, owner, device_id),
 * leaving device_id at its default of 0.  0 means "device 0" here, not
 * "untagged" (only -1 is legacy/untagged, per that struct's own comment),
 * so this carries forward the exact same cross-device gap spec section 6a
 * already documents for every multi-tensor entry point in this backend:
 * harmless while every tensor lives on tier 0, which is true for all
 * current work, and not a gap to close here ahead of the multi-GPU plan.
 * owner 0 is also deliberate: this wrapper does not own the scratch
 * allocation the way a real tensor would, since sycl_device_scratch_guard
 * below already owns and frees it.
 *
 * device_id is `mid->device_id`, NOT the struct-default 0 an earlier
 * version of this wrapper left it at: `tmp` is allocated on
 * ds4_sycl_queue(mid->device_id), so gate_tmp/up_tmp must report that same
 * tier, or the delegated call below (which reads its own queue from
 * gate_tmp->device_id) would compute on tier 0 regardless of where mid,
 * and the memory it actually allocated, live. Harmless previously, since
 * every tensor this backend touched lived on tier 0; ds4_gpu_shared_mid_
 * swiglu_q8_0_decode_exact_tensor below delegates here from a genuinely
 * non-zero tier, so the bug had to be fixed here, not just noted. */
extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!mid || out_dim == 0u || out_dim > UINT32_MAX) return 0;

    uint64_t tmp_elems = 0, tmp_bytes = 0;
    if (!sycl_u64_mul_checked(2u, out_dim, &tmp_elems) ||
        !sycl_u64_mul_checked(tmp_elems, sizeof(float), &tmp_bytes)) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(mid->device_id);

        float *tmp = sycl::malloc_device<float>((size_t)tmp_elems, q);
        if (!tmp) return 0;
        sycl_device_scratch_guard tmp_guard(q, tmp);

        ds4_gpu_tensor gate_tmp = { tmp, out_dim * sizeof(float), 0,
                                    mid->device_id };
        ds4_gpu_tensor up_tmp = { tmp + out_dim, out_dim * sizeof(float), 0,
                                  mid->device_id };

        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                &gate_tmp, &up_tmp, mid, model_map, model_size, gate_offset,
                up_offset, in_dim, out_dim, x, clamp);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "shared gate/up mid wrapper failed: %s\n", e.what());
        return 0;
    }
}

/* Expert-parallel decode: the shared expert's gate/up/SwiGLU,
 * load-balanced across the two EP ranks by whichever has fewer OWNED
 * routed experts among this token's six selected slots, and written
 * cross-device when the computing rank differs from `mid`'s home tier.
 *
 * Ported from ds4_cuda.cu:18552-18637 and its device kernel
 * shared_mid_q8_0_preq_warp8_exact_kernel (:5264-5328). CUDA computes on
 * whichever device is currently active (`cuda_decode_stream()`) and writes
 * `mid` directly, relying on a validated peer-access pair to make that a
 * legal cross-device pointer write. This backend has no established
 * pattern for a kernel writing another device's USM allocation directly
 * (every cross-device transfer elsewhere in this backend, including this
 * backend's own MoE combine entries, is an explicit synchronous copy -- spec
 * 6g/6j's cross-device discipline), so this port computes into local
 * scratch on the ACTIVATION's device (`x->device_id`, the device the real
 * kernel would run on) and then crosses to `mid`'s home tier via
 * ds4_gpu_tensor_copy_xdev when the two differ, rather than assume a
 * cross-device write would work.
 *
 * `selected`/`expert_split`/`home_rank` reproduce CUDA's per-call
 * assignment predicate exactly (home_count <= peer_count keeps ties on the
 * home rank), computed here from a synchronous 6-int32 readback rather
 * than per-thread on device, since the predicate is identical for every
 * row: either this whole call is assigned or none of it is.
 *
 * `prequant`: a CUDA-only cached-quantisation optimisation. This backend's
 * shared-expert kernels never quantise the activation at all (dequantise
 * WEIGHTS to float and dot against the plain float `x` instead, the design
 * substitution documented on ds4_gpu_shared_gate_up_swiglu_q8_0_tensor's
 * fused fast path above and shared again by ds4_sycl_hc.hpp's fused
 * hc-expand), so this parameter has no correctness content on this
 * backend and is deliberately unread. */
extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank) {
    (void)prequant;
    if (!mid || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        out_dim > UINT32_MAX ||
        (selected && (selected->bytes < 6u * sizeof(int32_t) || expert_split == 0u))) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    try {
        if (selected) {
            const int sel_dev = selected->device_id >= 0 ? selected->device_id
                                                          : g_current_tier;
            int32_t sel_host[6];
            sycl_batch_wait(ds4_sycl_queue(sel_dev)
                    .memcpy(sel_host, selected->ptr, sizeof(sel_host)));
            uint32_t home_count = 0u, peer_count = 0u;
            for (uint32_t i = 0; i < 6u; i++) {
                const int32_t expert = sel_host[i];
                if (expert >= 0 && (uint32_t)expert < expert_split) {
                    home_count++;
                } else if (expert >= 0 && (uint32_t)expert < 2u * expert_split) {
                    peer_count++;
                }
            }
            const bool assigned =
                    home_rank ? home_count <= peer_count : peer_count < home_count;
            if (!assigned) return 1;
        }

        const int compute_dev = x->device_id >= 0 ? x->device_id : g_current_tier;
        if (mid->device_id == compute_dev) {
            return ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                    mid, model_map, model_size, gate_offset, up_offset, in_dim,
                    out_dim, x, clamp);
        }

        sycl::queue &cq = ds4_sycl_queue(compute_dev);
        float *scratch = sycl::malloc_device<float>((size_t)out_dim, cq);
        if (!scratch) return 0;
        sycl_device_scratch_guard scratch_guard(cq, scratch);
        ds4_gpu_tensor scratch_tensor = { scratch, out_dim * sizeof(float), 0,
                                          compute_dev };
        if (!ds4_gpu_shared_mid_swiglu_q8_0_tensor(&scratch_tensor, model_map,
                                                   model_size, gate_offset,
                                                   up_offset, in_dim, out_dim, x,
                                                   clamp)) {
            return 0;
        }
        return ds4_gpu_tensor_copy_xdev(mid, &scratch_tensor,
                                        out_dim * sizeof(float));
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "shared mid swiglu decode exact failed: %s\n", e.what());
        return 0;
    }
}

/* Batched rows path.  Internal helper, NOT an extern "C" entry point,
 * matching the file header's treatment of ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor
 * as a ROCm-internal helper reached only from the rows entry below. Ports
 * rocm/ds4_rocm_shared_expert.cuh:390-439 (validation) and
 * shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel
 * (rocm/ds4_rocm_q8.cuh:524-671, instantiated there only with
 * TOK_TILE == BLOCKS_TILE == 16, so those are plain constants here rather
 * than a template).
 *
 * The staged activation tile in local memory is exactly the hazard spec
 * section 6b describes: uninitialised local memory reads as zero on this
 * A770, so a missing barrier would not show up as garbage here. Every
 * element of shx is written unconditionally (a real value or 0.0f for a
 * position past n_blocks/n_tok) before the barrier that follows it, with
 * a second barrier after the row/token accumulation loop before the next
 * tile's staging pass reuses the same local memory.
 *
 * An early ablation at small scale (IN_DIM == 64, one staging/consume
 * iteration, 4 work-groups total) dropped either barrier, or both, with no
 * test failure across five repeated runs, even after the test data was
 * rebuilt so every staged value is bounded away from zero (ruling out
 * spec 6b's "torn read returns zero" explanation). That was recorded as a
 * fourth ablation that fails to fail, per spec section 6j the wrong
 * conclusion to stop at: too few work-groups were resident at once for the
 * cross-wave race either barrier guards to have contention to act on.
 * tests/test_sycl_shared_expert.c's
 * test_shared_gate_up_swiglu_rows_batch_barrier_stress scales the same
 * kernel to IN_DIM == 1024 (two full staging/consume iterations, so the
 * second barrier's reuse-across-iterations hazard has an iteration
 * boundary to expose) and OUT_DIM == 512 by N_TOK == 4096 (4096
 * work-groups, about 4.19 million work-items, launched at once). At that
 * scale, dropping the first barrier alone corrupted roughly 13-15 percent
 * of output elements, and dropping both corrupted roughly 60 percent,
 * reliably across five repeated runs each, while the small-scale
 * correctness test in the same binary kept passing throughout. Both
 * barriers are correct by construction, by direct comparison with the
 * ROCm source's __syncthreads placement, and now by an ablation that
 * actually discriminates; do not remove either. */
static int sycl_shared_gate_up_swiglu_q8_0_batch(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint64_t up_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0u ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t x_bytes = 0, out_bytes = 0;
    if (!sycl_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !sycl_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        !sycl_tensor_has_bytes(x, x_bytes) ||
        !sycl_tensor_has_bytes(gate, out_bytes) ||
        !sycl_tensor_has_bytes(up, out_bytes) ||
        !sycl_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }

    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!sycl_q8_0_row_bytes_checked(in_dim, &row_bytes) ||
        !sycl_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !sycl_model_range_fits(model_size, gate_offset, weight_bytes) ||
        !sycl_model_range_fits(model_size, up_offset, weight_bytes)) {
        return 0;
    }
    const char *wg = sycl_model_range_ptr(model_map, gate_offset,
                                          weight_bytes, model_size,
                                          "shared_gate_q8_batch");
    const char *wu = sycl_model_range_ptr(model_map, up_offset, weight_bytes,
                                          model_size, "shared_up_q8_batch");
    if (!wg || !wu) return 0;
    if (g_devices.empty()) return 0;

    constexpr uint32_t kTokTile = 16u;
    constexpr uint32_t kBlocksTile = 16u;
    constexpr uint32_t kRowsPerBlock = 32u;
    const uint32_t n_blocks = (uint32_t)(in_dim >> 5u);
    const uint32_t out_dim32 = (uint32_t)out_dim;
    const uint32_t n_tok32 = (uint32_t)n_tok;
    const uint32_t grid_x =
            (out_dim32 + kRowsPerBlock - 1u) / kRowsPerBlock;
    const uint32_t grid_y = (n_tok32 + kTokTile - 1u) / kTokTile;
    const size_t local_x = (size_t)kRowsPerBlock * 32u;
    const size_t shmem_floats = (size_t)kTokTile * kBlocksTile * 32u;

    try {
        sycl::queue &q = ds4_sycl_queue(gate->device_id);

        sycl_device_scratch_guard dwg_guard = sycl_stage_host_bytes(q, wg, weight_bytes);
        unsigned char *dwg = (unsigned char *)dwg_guard.p;
        if (!dwg) return 0;
        sycl_device_scratch_guard dwu_guard = sycl_stage_host_bytes(q, wu, weight_bytes);
        unsigned char *dwu = (unsigned char *)dwu_guard.p;
        if (!dwu) return 0;

        float *pgate = (float *)gate->ptr;
        float *pup = (float *)up->ptr;
        float *pmid = (float *)mid->ptr;
        const float *px = (const float *)x->ptr;

        sycl::event _ds4_prof_ev147 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> shx(sycl::range<1>(shmem_floats),
                                               h);
            h.parallel_for(
                    sycl::nd_range<2>(
                            sycl::range<2>((size_t)grid_x * local_x, grid_y),
                            sycl::range<2>(local_x, 1)),
                    [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                        const uint32_t tid = (uint32_t)it.get_local_id(0);
                        const uint32_t lane = tid & 31u;
                        const uint32_t wave = tid >> 5u;
                        const uint32_t row =
                                (uint32_t)it.get_group(0) * kRowsPerBlock +
                                wave;
                        const uint32_t t0 =
                                (uint32_t)it.get_group(1) * kTokTile;
                        const bool row_valid = row < out_dim32;
                        const unsigned char *wgr =
                                dwg + (uint64_t)(row_valid ? row : 0u) *
                                              row_bytes;
                        const unsigned char *wur =
                                dwu + (uint64_t)(row_valid ? row : 0u) *
                                              row_bytes;

                        float accg[kTokTile];
                        float accu[kTokTile];
                        for (uint32_t u = 0; u < kTokTile; u++) {
                            accg[u] = 0.0f;
                            accu[u] = 0.0f;
                        }

                        for (uint32_t b0 = 0; b0 < n_blocks;
                             b0 += kBlocksTile) {
                            const uint32_t b_count =
                                    ((b0 + kBlocksTile) <= n_blocks)
                                            ? kBlocksTile
                                            : (n_blocks - b0);
                            /* Every position of shx is written here,
                             * including out-of-range (t, bb) pairs, which
                             * get an explicit 0.0f rather than being left
                             * unwritten; see the section 6b note above. */
                            for (uint32_t j = tid;
                                 j < kTokTile * kBlocksTile * 32u;
                                 j += (uint32_t)local_x) {
                                const uint32_t u = j / (kBlocksTile * 32u);
                                const uint32_t r = j - u * (kBlocksTile * 32u);
                                const uint32_t bb = r >> 5u;
                                const uint32_t k = r & 31u;
                                const uint32_t t = t0 + u;
                                shx[j] = (t < n_tok32 && bb < b_count)
                                        ? px[(uint64_t)t * in_dim +
                                             ((uint64_t)(b0 + bb) << 5u) + k]
                                        : 0.0f;
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            if (row_valid) {
                                for (uint32_t bb = 0; bb < b_count; bb++) {
                                    const unsigned char *bg =
                                            wgr + (uint64_t)(b0 + bb) * 34u;
                                    const unsigned char *bu =
                                            wur + (uint64_t)(b0 + bb) * 34u;
                                    const float wvg = sycl_q8_0_dequant(bg, lane);
                                    const float wvu = sycl_q8_0_dequant(bu, lane);
                                    for (uint32_t u = 0; u < kTokTile; u++) {
                                        const float xv =
                                                shx[(u * kBlocksTile + bb) *
                                                            32u +
                                                    lane];
                                        accg[u] += wvg * xv;
                                        accu[u] += wvu * xv;
                                    }
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }

                        sycl::sub_group sg = it.get_sub_group();
                        for (uint32_t u = 0; u < kTokTile; u++) {
                            accg[u] = sycl::reduce_over_group(
                                    sg, accg[u], sycl::plus<float>());
                            accu[u] = sycl::reduce_over_group(
                                    sg, accu[u], sycl::plus<float>());
                        }
                        if (sg.get_local_id()[0] == 0u && row_valid) {
                            for (uint32_t u = 0; u < kTokTile; u++) {
                                const uint32_t t = t0 + u;
                                if (t < n_tok32) {
                                    const uint64_t off =
                                            (uint64_t)t * out_dim32 + row;
                                    const float g = accg[u];
                                    const float uv = accu[u];
                                    pgate[off] = g;
                                    pup[off] = uv;
                                    pmid[off] =
                                            (g / (1.0f + sycl::exp(-g))) * uv;
                                }
                            }
                        }
                    });
        });
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named("shared_gate_up_swiglu_q8_0_batch", _ds4_prof_ev147);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "shared gate/up fused q8 batch launch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* n_tok == 1 delegates to the core entry; n_tok > 1 uses the batched rows
 * path above (sycl_shared_gate_up_swiglu_q8_0_batch).  ROCm refuses
 * (returns 0, not a fallback to the general path) rather than route a
 * nonzero clamp through the batch kernel, which has no clamp parameter at
 * all (rocm/ds4_rocm_shared_expert.cuh:184); ported exactly. */
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1u) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                gate, up, mid, model_map, model_size, gate_offset, up_offset,
                in_dim, out_dim, x, clamp);
    }
    if (clamp > 1.0e-6f) return 0;
    return sycl_shared_gate_up_swiglu_q8_0_batch(gate, up, mid, model_map,
                                                 model_size, gate_offset,
                                                 up_offset, in_dim, out_dim,
                                                 x, n_tok);
}
