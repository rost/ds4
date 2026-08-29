#pragma once

/* DS4 SYCL selected-expert readback: the four-entry ABI Flash's hash-routed
 * MoE layers use to read back, on the host, which experts a layer selected
 * (via layer_hash_selected_experts, a CPU function) without stalling the
 * whole command queue on that readback.
 *
 * Structural reference: ROCm implements all four
 * (rocm/ds4_rocm_runtime.cuh:6054-6088, plus
 * ds4_gpu_routed_moe_set_selected_override in
 * rocm/ds4_rocm_current_api_compat.cuh:337-345). ROCm's shape:
 *
 *   - A single, lazily created cudaEvent is reused across every call to
 *     ds4_gpu_signal_selected_readback_ready: each call re-records into the
 *     SAME event object rather than creating a new one.
 *   - event_value is NOT a handle to that event. It is a plain incrementing
 *     counter (rocm/ds4_rocm_runtime.cuh:17,6077), handed back to the
 *     caller as a nonzero sentinel meaning "something was signalled."
 *     Neither ds4_gpu_commit_and_wait_selected_readback nor
 *     ds4_gpu_wait_selected_readback_ready inspects its own event_value
 *     argument beyond checking it is nonzero: both wait on THE CURRENT
 *     global event, whichever call to signal_selected_readback_ready most
 *     recently recorded it, not on "the event matching this specific
 *     value." That is ROCm's actual behaviour, not a simplification made
 *     here; ds4.c's own call pattern (signal once, do independent work,
 *     wait once, all per layer, never two outstanding signals at once)
 *     never exercises the difference.
 *
 * The SYCL analogue of "record an event capturing everything queued to the
 * current device's queue so far" is queue::ext_oneapi_submit_barrier(),
 * already this backend's own precedent for a genuine cross-submission SYCL
 * event fence (sycl/ds4_sycl_mgpu.hpp, the mechanism design spec 6t names
 * as the right tool for cross-queue ordering). A sycl::event is copyable
 * and reference-counted, so "re-record into the same event object" has no
 * direct SYCL equivalent; the natural port instead REPLACES the stored
 * event on every signal call, which has the identical observable effect
 * for this caller pattern: whichever event is currently stored is the one
 * both wait entries wait on, and there is no way to name an older one once
 * a newer signal has landed -- exactly ROCm's own single-slot behaviour.
 *
 * What happens on a double wait: sycl::event::wait_and_throw() on an
 * already-completed event returns immediately (SYCL 2020 defines an event
 * wait as idempotent), so calling either wait entry twice on the same
 * event_value is safe and cheap, matching ROCm's cudaEventSynchronize being
 * safe to call more than once on the same cudaEvent.
 *
 * What happens with no wait at all: the next
 * ds4_gpu_signal_selected_readback_ready call overwrites the stored
 * sycl::event with a new one. The old event's destructor does not block
 * and does not cancel the work it referred to; that work keeps running on
 * the device and completes on its own schedule, the same guarantee any
 * unwaited SYCL submission already has. It is simply no longer nameable
 * through this ABI. Nothing is leaked: a sycl::event owns no device
 * allocation, only a reference-counted runtime handle, so dropping the
 * last reference to an unwaited event is exactly as safe as dropping the
 * last reference to a waited one. */

#include "ds4_sycl_common.hpp"

#include <optional>

namespace {

/* ROCm's own cap for the routed-MoE selected-expert override list
 * (rocm/ds4_rocm_runtime.cuh:29, DS4_ROCM_N_EXPERT_USED), matching ds4.c's
 * own DS4_MAX_EXPERT_USED (ds4.c:480). Flash only ever selects
 * DS4_N_EXPERT_USED == 6 (per the reachability audit), but the ABI itself
 * carries no compile-time bound beyond what the caller passes, so this
 * mirrors ROCm's storage capacity rather than Flash's current value. */
constexpr uint32_t kSyclMaxExpertUsed = 8u;

int32_t  g_sycl_routed_moe_selected_override[kSyclMaxExpertUsed];
uint32_t g_sycl_routed_moe_selected_override_n = 0u;

std::optional<sycl::event> g_sycl_selected_readback_event;
uint64_t                   g_sycl_selected_readback_value = 0u;

}  // namespace

/* Drops any stored selected-readback event before g_devices.clear()
 * destroys the queue it was recorded against. Not load-bearing for
 * correctness (see this header's own top comment: a sycl::event's
 * destructor never blocks and owns no device allocation), only hygiene
 * against a stale handle surviving into a later re-init within the same
 * process, which the test binaries here do exercise. Forward-
 * declared in ds4_sycl.cpp so ds4_gpu_cleanup can call it. */
static void sycl_readback_teardown(void) {
    g_sycl_selected_readback_event.reset();
    g_sycl_selected_readback_value = 0u;
    g_sycl_routed_moe_selected_override_n = 0u;
}

/* Test-only accessor: lets tests/test_sycl_readback.c confirm what was
 * stored without needing a routed-MoE call site of its own. This
 * backend's routed MoE dispatch does not currently consume this override
 * at all -- see the comment on ds4_gpu_routed_moe_set_selected_override
 * below for why that is the honest state of the port rather than a
 * shortcut taken here. Not part of the production ABI. */
extern "C" uint32_t ds4_sycl_test_routed_moe_selected_override_snapshot(
        int32_t *out, uint32_t cap) {
    const uint32_t n = g_sycl_routed_moe_selected_override_n;
    if (out && cap >= n) {
        for (uint32_t i = 0; i < n; i++) out[i] = g_sycl_routed_moe_selected_override[i];
    }
    return n;
}

/* ds4_gpu_routed_moe_set_selected_override, ds4_rocm_current_api_compat.cuh:
 * 337-345: a plain storage setter for the hash-routed layer's currently
 * selected experts, called at ds4.c:20948 (metal_graph_decode_set_hash_
 * selected_override, Flash's hash-routed layers, layer->ffn_gate_tid2eid
 * != NULL) and ds4.c:21012 (metal_graph_decode_cpu_router, the CPU-router
 * fallback for the same hash-routed layers, DS4_METAL_PRO_Q4_CPU_ROUTER_
 * PROFILE / DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE). Both are
 * metal_graph_* Flash code, not glm_graph_*: reachable for Flash, not
 * GLM-only.
 *
 * ROCm's only consumer of this stored value is
 * cuda_stream_selected_pending_matches, which correlates it against a
 * PENDING streaming-cache load to decide whether that load is still valid
 * for the selection actually used. This backend's streaming expert cache
 * has that correlation machinery nowhere: its three lookup hooks
 * (sycl/ds4_sycl_moe_launch.hpp:118-120) are hardcoded to report a cache
 * miss, so MoE always stages fresh from the host mmap regardless of what
 * this override holds (ds4-sycl-STATE.md, "Streaming is implemented but
 * not wired in"). Wiring that correlation up is explicitly out of scope
 * here ("Explicitly not done"). So this entry, faithfully ported,
 * stores the selection exactly as ROCm does; nothing in this backend reads
 * it back for anything but the test hook above, which is the honest state
 * of the port, not a shortcut taken here. */
extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected,
                                                         uint32_t n_selected) {
    if (n_selected > kSyclMaxExpertUsed || (!selected && n_selected != 0u)) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_sycl_routed_moe_selected_override[i] = selected[i];
    }
    g_sycl_routed_moe_selected_override_n = n_selected;
    return 1;
}

/* ds4_gpu_signal_selected_readback_ready, ds4_rocm_runtime.cuh:6054-6079,
 * called at ds4.c:24519 inside metal_graph_encode_decode_layer_phase's
 * shared/routed overlap path (Flash, reachable whenever any of the
 * q4/iq2/mxfp4/cuda "selected slots" conditions hold). ds4.c:30099's call
 * site the plan and reachability audit both cite is NOT reachable for
 * SYCL: it sits inside a `#ifdef DS4_ROCM_BUILD ... #endif` block
 * (ds4.c:30084-30115) that never compiles for this backend, contradicting
 * the audit's own table, which listed it as a second reachable call site
 * without checking the surrounding ifdef.
 *
 * See this header's top comment for the event/counter shape. */
extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!event_value) return 0;
    *event_value = 0;
    if (g_devices.empty()) return 0;
    try {
        g_sycl_selected_readback_event =
                ds4_sycl_queue(g_current_tier).ext_oneapi_submit_barrier();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "selected readback signal failed: %s\n",
                e.what());
        return 0;
    }
    *event_value = ++g_sycl_selected_readback_value;
    return 1;
}

/* ds4_gpu_commit_and_wait_selected_readback, ds4_rocm_runtime.cuh:
 * 6080-6084, called at ds4.c:24605, the same overlap path as the signal
 * call above (Flash, reachable). Waits on whichever event is currently
 * stored; see this header's top comment for why event_value itself is
 * only checked for nonzero rather than matched against a specific
 * recorded event. */
extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value,
                                                          const char *label) {
    if (event_value == 0u || !g_sycl_selected_readback_event.has_value()) return 0;
    try {
        g_sycl_selected_readback_event->wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n",
                label ? label : "selected readback", e.what());
        return 0;
    }
    return 1;
}

/* ds4_gpu_wait_selected_readback_ready, ds4_rocm_runtime.cuh:6085-6088: an
 * identical body to ds4_gpu_commit_and_wait_selected_readback above under a
 * different diagnostic label, matching ROCm exactly (both entries call
 * cudaEventSynchronize on the same single global event there too).
 * Called at ds4.c:21351, inside metal_graph_selected_async_load_run's
 * `#else` branch of `#ifdef DS4_ROCM_BUILD` (ds4.c:21339-21362): the
 * non-ROCm branch is exactly the one SYCL compiles, so this call site is
 * reachable for Flash. The other call site the plan cites, ds4.c:42120,
 * sits inside GLM's streaming async-load path (g_glm_streaming_async_
 * profile, glm_graph_* naming throughout the surrounding function) and is
 * confirmed GLM-only, out of scope. */
extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value,
                                                     const char *label) {
    return ds4_gpu_commit_and_wait_selected_readback(
            event_value, label ? label : "selected readback wait");
}
