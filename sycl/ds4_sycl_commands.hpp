#pragma once

/* Command-encoder-shaped lifecycle: begin_commands / commands_active /
 * end_commands / flush_commands / flush_encoder / synchronize.
 *
 * This surface is modelled on Metal's real command-encoder API (open a
 * command buffer, encode into it, commit, wait). Neither CUDA nor ROCm has
 * an encoder object: both dispatch each kernel straight to a stream and
 * treat begin/commands_active as trivial constants, reserving the real work
 * for end_commands/flush_commands/synchronize, which all resolve to
 * cudaDeviceSynchronize() (rocm/ds4_rocm_runtime.cuh:6050-6093,
 * ds4_cuda.cu:3683-3691). This backend has no encoder either, so it follows
 * that same shape rather than inventing a Metal-style buffer object.
 *
 * ds4.c gates real decode and prefill work behind these, unconditionally,
 * on every backend it builds: metal_graph_eval_token_raw_swa (ds4.c:30765)
 * -- the real per-token decode path reached from ds4_session_eval_internal
 * -- runs "bool ok = ds4_gpu_begin_commands() != 0; if (ok) ok =
 * <encode>...; if (ok) ok = ds4_gpu_end_commands() != 0;" with no
 * DS4_SYCL_BUILD or __APPLE__ gate anywhere in that function. A stub that
 * returns its own failure value here fails every decode token before a
 * single kernel is ever dispatched. */

#include "ds4_sycl_common.hpp"
#include "ds4_sycl_graph.hpp"

/* Opens the batch. With command-graph capture off (the default) there is
 * still no encoder object and this is the same trivial, unconditional
 * success as both structural references (rocm/ds4_rocm_runtime.cuh:6050,
 * ds4_cuda.cu:3683: `return 1;`).
 *
 * With capture armed (DS4_SYCL_GRAPH), this is where recording starts:
 * ds4.c brackets a whole decode token's encode in this entry and
 * ds4_gpu_end_commands below, which is exactly the run of commands a
 * graph wants to capture. Failing to start recording is not an error --
 * the batch simply runs un-captured, which is what every caller already
 * expects -- so this still returns success either way. */
extern "C" int ds4_gpu_begin_commands(void) {
    if (!g_devices.empty()) (void)g_sycl_graph_batch.begin(ds4_sycl_current_queue());
    return 1;
}

/* Never reports an outstanding batch: with no encoder object there is
 * nothing to be active. Matches both structural references exactly
 * (rocm/ds4_rocm_runtime.cuh:6053 `return 0;`,
 * ds4_cuda.cu:26353 `return false;`). Callers that gate a begin/end pair on
 * this (glm_graph_begin_commands_if_needed / _end_commands_if_active,
 * ds4.c:47224-47230) always take the "not active, do it yourself" branch,
 * which is correct here: every real op below already orders itself. */
extern "C" int ds4_gpu_commands_active(void) { return 0; }

/* Real work: wait for everything already submitted to the current tier's
 * queue to finish, and report the result through this entry's own
 * nonzero-means-success convention (ds4_gpu.h:111, confirmed against
 * cuda_ok's polarity at rocm/ds4_rocm_runtime.cuh:6093).
 *
 * Spec 6t is why this cannot be a no-op. Every tier's queue is
 * property::queue::in_order() (ds4_sycl.cpp, ds4_sycl_queue_properties),
 * which orders commands within that queue against each other, but that is
 * an ordering between device-side commands, not a completion signal to
 * the host: every allocation is raw malloc_device USM, so nothing tracks
 * completion automatically between the caller and whatever this tier's
 * queue has outstanding, in_order or not. A caller relying on this entry
 * to mean "the GPU is done" needs an actual wait, not a trivial success
 * return.
 *
 * g_devices.empty() is checked directly rather than routed through
 * ds4_sycl_queue/ds4_sycl_current_queue: that helper aborts the process on
 * an empty device list by design (ds4_sycl.cpp:196-200, "a caller that
 * reaches this branch anyway is a bug in this file"), which is correct for
 * an internal alloc/free/transfer call driven by a live tensor, but wrong
 * here -- these three entries are reachable from ds4.c error-handling paths
 * with no tensor in hand at all (e.g. `if (!ok) (void)ds4_gpu_synchronize();`
 * at ds4.c:31426 and many more), so an uninitialised or torn-down backend
 * must fail gracefully here instead of crashing the process. */
static int ds4_sycl_wait_current_tier(const char *label) {
    if (g_devices.empty()) return 0;
    /* Ends any recording first: the commands a captured batch holds have
     * not run yet, so draining the queue without submitting them would
     * report completion of work that never happened. This also covers the
     * error paths that reach ds4_gpu_synchronize without knowing whether a
     * batch was ever opened, and the mid-layer ds4_gpu_flush_commands
     * splits, both of which must see every command issued so far actually
     * finished. */
    if (!g_sycl_graph_batch.end()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: command-graph batch\n",
                label);
        return 0;
    }
    try {
        ds4_sycl_current_queue().wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n", label, e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_end_commands(void) {
    return ds4_sycl_wait_current_tier("end commands");
}

extern "C" int ds4_gpu_flush_commands(void) {
    return ds4_sycl_wait_current_tier("flush");
}

/* ROCm delegates to flush_commands (rocm/ds4_rocm_runtime.cuh:6052:
 * `return ds4_gpu_flush_commands();`); CUDA instead returns a bare 1 with
 * the comment "CUDA kernels are already queued in stream order, nothing to
 * split" (ds4_cuda.cu:26462-26465). That CUDA-specific reasoning is exactly
 * what spec 6t says does not hold here: this backend's queues are
 * out-of-order and carry no automatic dependency tracking between separate
 * submit() calls, so "already in order" is not a safe assumption to copy.
 * Follow ROCm's shape instead (spec 6e: ROCm is the structural reference,
 * not CUDA, except where multi-GPU is the explicit documented exception),
 * which is also the conservative reading of an entry named "flush an
 * encoder" when this backend has no encoder to flush: wait for real. */
extern "C" int ds4_gpu_flush_encoder(void) {
    return ds4_gpu_flush_commands();
}

/* Same real wait as end_commands/flush_commands. Kept as its own function
 * (rather than a one-line delegate) because it is the entry the spec calls
 * out by name (6t) as the one most likely to be misread as "just report
 * success": callers use it standalone as a final drain, e.g.
 * metal_graph_eval_token_raw_swa's failure path (ds4.c:30792) synchronizing
 * to let any in-flight GPU work finish before giving up on the token. */
extern "C" int ds4_gpu_synchronize(void) {
    return ds4_sycl_wait_current_tier("synchronize");
}
