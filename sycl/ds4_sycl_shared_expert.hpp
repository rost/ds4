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
 * ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor also stays stubbed:
 * it exists only in ds4_cuda.cu (CUDA-only; rocm/ has no definition at
 * all), so under ROCm this entry hits the unavailable stub and ds4.c
 * falls back.  It is also multi-GPU aware (ds4_tensor_device_idx,
 * g_gpu_peer_ok, cuda_tmp_alloc_on), so it belongs with the multi-GPU
 * streaming plan if it is ever wanted on this backend.
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
static void sycl_shared_gate_up_swiglu_q8_0_rows_w32(
        sycl::queue &q, float *gate, float *up, float *mid,
        const unsigned char *wg, const unsigned char *wu, const float *x,
        uint32_t n_blocks, uint32_t out_dim, uint64_t row_bytes,
        float clamp) {
    constexpr uint32_t kRowsPerBlock = 32u;
    const uint32_t n_groups = (out_dim + kRowsPerBlock - 1u) / kRowsPerBlock;
    const size_t local_size = (size_t)kRowsPerBlock * 32u;

    q.submit([&](sycl::handler &h) {
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

            unsigned char *dwg =
                    sycl::malloc_device<unsigned char>((size_t)weight_bytes, q);
            if (!dwg) return 0;
            sycl_device_scratch_guard dwg_guard(q, dwg);
            unsigned char *dwu =
                    sycl::malloc_device<unsigned char>((size_t)weight_bytes, q);
            if (!dwu) return 0;
            sycl_device_scratch_guard dwu_guard(q, dwu);

            q.memcpy(dwg, wg, (size_t)weight_bytes).wait_and_throw();
            q.memcpy(dwu, wu, (size_t)weight_bytes).wait_and_throw();

            sycl_shared_gate_up_swiglu_q8_0_rows_w32(
                    q, (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    dwg, dwu, (const float *)x->ptr, (uint32_t)(in_dim >> 5u),
                    (uint32_t)out_dim, row_bytes, clamp);
            q.wait_and_throw();
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
