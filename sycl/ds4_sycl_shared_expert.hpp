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

/* Core entry.  General path only for now: the fused fast path
 * (in_dim == 4096) is added alongside this validation block once it
 * exists; until then every shape falls through to
 * ds4_gpu_matmul_q8_0_pair_tensor + ds4_gpu_swiglu_tensor, both already
 * implemented by the dense matmul port.
 *
 * Validation mirrors rocm/ds4_rocm_shared_expert.cuh:19-33: null/shape
 * checks first, then overflow-safe row/weight/x/out byte counts, then a
 * capacity check on every tensor.  Note this level does NOT bounds-check
 * gate_offset/up_offset against model_size; ROCm only does that ahead of
 * its fast-path kernel launch (since it forms raw pointers into the
 * model map there), and the general path's own callee,
 * ds4_gpu_matmul_q8_0_pair_tensor, already performs that check itself
 * before forming its own pointers. */
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
