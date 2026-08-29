/* Selected-expert readback tests: ds4_gpu_signal_selected_readback_ready,
 * ds4_gpu_commit_and_wait_selected_readback, ds4_gpu_wait_selected_readback_ready,
 * ds4_gpu_routed_moe_set_selected_override.
 *
 * Deliberately self-contained: no shared harness header, matching
 * tests/test_sycl_fp8_kv.c, tests/test_sycl_router.c and
 * tests/test_sycl_mgpu.c. Needs no model file. */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Test-only accessor into sycl/ds4_sycl_readback.hpp's stored override,
 * declared extern here the same way tests/test_sycl_mgpu.c forward-declares
 * its own SYCL-side hooks (ds4_sycl.h is C++-only and cannot be included
 * from this C test). */
extern uint32_t ds4_sycl_test_routed_moe_selected_override_snapshot(int32_t *out, uint32_t cap);

/* Second ablation, the genuine case: before any signal has ever happened
 * in this process, no event is stored at all, so waiting on ANY nonzero
 * value -- one this process never received back from
 * ds4_gpu_signal_selected_readback_ready -- must fail rather than hang or
 * silently succeed. Must run before any test below calls signal, which is
 * why it is first in main(). */
static int test_wait_before_any_signal_fails(void) {
    CHECK(ds4_gpu_commit_and_wait_selected_readback(12345u, "never signalled") == 0,
          "commit_and_wait: a nonzero value must fail before any signal ever ran");
    CHECK(ds4_gpu_wait_selected_readback_ready(12345u, "never signalled") == 0,
          "wait_selected_readback_ready: a nonzero value must fail before any signal ever ran");
    fprintf(stderr, "  test_wait_before_any_signal_fails OK\n");
    return 0;
}

/* Signal then wait, with nothing queued between the two calls: an
 * ablation of "signal without recording anything, so the wait
 * returns immediately". ext_oneapi_submit_barrier over an otherwise idle
 * queue completes essentially at once, so the wait must not hang. */
static int test_signal_wait_roundtrip(void) {
    uint64_t event_value = 0;
    CHECK(ds4_gpu_signal_selected_readback_ready(&event_value) != 0,
          "signal: must succeed");
    CHECK(event_value != 0, "signal: must return a nonzero event value");
    CHECK(ds4_gpu_commit_and_wait_selected_readback(event_value, "test roundtrip") != 0,
          "commit_and_wait: must succeed after a real signal");
    fprintf(stderr, "  test_signal_wait_roundtrip OK\n");
    return 0;
}

/* ds4_gpu_wait_selected_readback_ready is the sibling entry ds4.c's async
 * expert-load worker thread uses instead of commit_and_wait; ROCm gives it
 * an identical body under a different diagnostic label, so it must succeed
 * on the same event too. */
static int test_wait_selected_readback_ready_after_signal(void) {
    uint64_t event_value = 0;
    CHECK(ds4_gpu_signal_selected_readback_ready(&event_value) != 0,
          "signal: must succeed");
    CHECK(ds4_gpu_wait_selected_readback_ready(event_value, "test wait ready") != 0,
          "wait_selected_readback_ready: must succeed after a real signal");
    fprintf(stderr, "  test_wait_selected_readback_ready_after_signal OK\n");
    return 0;
}

/* Second ablation: waiting on a value that was never signalled must fail,
 * not silently succeed or crash. A fresh nonzero value that this process
 * never received from signal_selected_readback_ready stands in for "never
 * signalled" together with covering event_value == 0 explicitly, since
 * ds4.c itself uses 0 as its own "nothing pending" sentinel at every call
 * site. */
static int test_wait_never_signalled_fails(void) {
    CHECK(ds4_gpu_commit_and_wait_selected_readback(0, "test zero value") == 0,
          "commit_and_wait: event_value 0 must fail");
    CHECK(ds4_gpu_wait_selected_readback_ready(0, "test zero value") == 0,
          "wait_selected_readback_ready: event_value 0 must fail");
    fprintf(stderr, "  test_wait_never_signalled_fails OK\n");
    return 0;
}

/* Double wait on the same event_value: sycl::event::wait_and_throw() on an
 * already-completed event is defined to return immediately, so both calls
 * must succeed, matching ROCm's cudaEventSynchronize being safe to call
 * more than once on the same cudaEvent. */
static int test_double_wait_succeeds(void) {
    uint64_t event_value = 0;
    CHECK(ds4_gpu_signal_selected_readback_ready(&event_value) != 0,
          "signal: must succeed");
    CHECK(ds4_gpu_commit_and_wait_selected_readback(event_value, "first wait") != 0,
          "commit_and_wait: first wait must succeed");
    CHECK(ds4_gpu_commit_and_wait_selected_readback(event_value, "second wait") != 0,
          "commit_and_wait: second wait on the same value must also succeed");
    fprintf(stderr, "  test_double_wait_succeeds OK\n");
    return 0;
}

/* A later signal replaces the stored event outright: waiting on the OLD
 * event_value after a newer signal still succeeds, because (matching
 * ROCm's own single-slot behaviour, see sycl/ds4_sycl_readback.hpp) neither
 * wait entry checks that its argument names the CURRENTLY stored event, only
 * that it is nonzero and that some event is stored. This is deliberately
 * ported, not an oversight: ds4.c's own call pattern never has two
 * outstanding signals at once. */
static int test_wait_on_stale_value_still_succeeds(void) {
    uint64_t first = 0, second = 0;
    CHECK(ds4_gpu_signal_selected_readback_ready(&first) != 0, "first signal must succeed");
    CHECK(ds4_gpu_signal_selected_readback_ready(&second) != 0, "second signal must succeed");
    CHECK(first != second, "two signals must return different counter values");
    CHECK(ds4_gpu_commit_and_wait_selected_readback(first, "stale value") != 0,
          "commit_and_wait: a stale but nonzero value must still succeed");
    fprintf(stderr, "  test_wait_on_stale_value_still_succeeds OK\n");
    return 0;
}

/* ds4_gpu_routed_moe_set_selected_override: a plain storage setter, per
 * ds4_rocm_current_api_compat.cuh:337-345. Verified by round-tripping
 * through the test-only snapshot hook, since (see
 * sycl/ds4_sycl_readback.hpp's own top comment on this entry) nothing in
 * this backend's routed-MoE compute path reads this value back: its only
 * ROCm consumer correlates it against the streaming expert cache's pending
 * load, and that cache's lookup hooks are hardcoded to report a miss here
 * (ds4_sycl_moe_launch.hpp), so there is no routed-MoE call site this test
 * could observe the override through even if the ABI worked perfectly. */
static int test_selected_override_round_trip(void) {
    const int32_t selected[6] = {3, 300, 7, 1, 900, 42};
    CHECK(ds4_gpu_routed_moe_set_selected_override(selected, 6u) != 0,
          "set_selected_override: must succeed for a 6-element list");

    int32_t got[8];
    uint32_t n = ds4_sycl_test_routed_moe_selected_override_snapshot(got, 8u);
    CHECK(n == 6u, "snapshot: count must match what was stored");
    for (uint32_t i = 0; i < 6u; i++) {
        CHECK(got[i] == selected[i], "snapshot: stored value must match");
    }

    /* A later call with n_selected == 0 overwrites the count outright
     * (ROCm's own shape: g_routed_moe_selected_override_n = 0), not an
     * append or a no-op. */
    CHECK(ds4_gpu_routed_moe_set_selected_override(NULL, 0u) != 0,
          "set_selected_override: n_selected == 0 with a null pointer must succeed");
    n = ds4_sycl_test_routed_moe_selected_override_snapshot(got, 8u);
    CHECK(n == 0u, "snapshot: count must be cleared by the zero-length call");

    fprintf(stderr, "  test_selected_override_round_trip OK\n");
    return 0;
}

static int test_selected_override_rejections(void) {
    const int32_t selected[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(ds4_gpu_routed_moe_set_selected_override(selected, 9u) == 0,
          "set_selected_override: an oversized list must be rejected");
    CHECK(ds4_gpu_routed_moe_set_selected_override(NULL, 3u) == 0,
          "set_selected_override: a null pointer with n_selected != 0 must be rejected");
    fprintf(stderr, "  test_selected_override_rejections OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_wait_before_any_signal_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_signal_wait_roundtrip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_wait_selected_readback_ready_after_signal() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_wait_never_signalled_fails() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_double_wait_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_wait_on_stale_value_still_succeeds() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_selected_override_round_trip() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_selected_override_rejections() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_readback OK\n");
    return 0;
}
