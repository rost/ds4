/* Tests for the three ds4.c engine edits made here: narrowing the
 * ssd_streaming && multi_tier refusal for SYCL GPU-only placements,
 * adding SYCL to ds4_backend_supports_streaming_auto_cache, and looping
 * ds4_engine_configure_streaming_auto_cache over every tier.
 *
 * Deliberately self-contained, matching tests/test_sycl_mgpu.c and
 * tests/test_sycl_streaming.c: no shared harness header, no model file.
 * The refusal predicate and the backend-support query are pure host-side
 * logic exercised through DS4_TEST_HOOKS entries defined in ds4.c, needing
 * no GPU at all. The auto-cache loop test does need a real device (it
 * calls ds4_gpu_recommended_working_set_size_on), so it uses the same
 * "two logical tiers on one physical device" technique as
 * tests/test_sycl_streaming.c's per-tier isolation tests. */

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_layer_pack.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* DS4_TEST_HOOKS entries defined in ds4.c; not declared in a shipped
 * header, following the pattern tests/test_sycl_session_smoke.c and
 * tests/test_engine_mgpu_runtime.c already use for this backend's own
 * test-only entry points. */
int ds4_test_ssd_streaming_multi_tier_blocked(const int *placement,
                                              int        n_placement_entries);
int ds4_test_backend_supports_streaming_auto_cache(int backend);
int ds4_test_configure_streaming_auto_cache_multi_tier(int n_gpus);

/* Test-only side door into sycl/ds4_sycl_streaming.hpp (also forward-
 * declared this way in tests/test_sycl_streaming.c): resets every
 * tracked tier's cache state, including its configured budget.
 * ds4_gpu_cleanup() frees each tier's resident/selected device memory but
 * does not reset the configured budget itself, so without this call
 * test_auto_cache_configures_every_tier's budget would still be visible
 * to the next test in this binary. */
extern void ds4_sycl_stream_test_reset(void);

/* ---- engine_ssd_streaming_multi_tier_blocked --------------------------- */

static int test_gpu_only_placement_admitted_on_sycl(void) {
    /* Two tiers, no CPU-spill entry: this build defines DS4_SYCL_BUILD
     * (Makefile:522), so the predicate must NOT block it. */
    const int placement[] = {0, 1, 0, 1};
    CHECK(ds4_test_ssd_streaming_multi_tier_blocked(
                  placement, (int)(sizeof(placement) / sizeof(placement[0]))) == 0,
          "GPU-only multi-tier placement must be admitted on a SYCL build");
    fprintf(stderr, "  test_gpu_only_placement_admitted_on_sycl OK\n");
    return 0;
}

static int test_cpu_spill_placement_still_blocked(void) {
    /* Same two tiers, but one entry spills to the CPU tier: this must
     * stay refused even on SYCL, since streaming addresses GPU-resident
     * experts only. */
    const int placement[] = {0, DS4_LAYER_PACK_CPU, 1};
    CHECK(ds4_test_ssd_streaming_multi_tier_blocked(
                  placement, (int)(sizeof(placement) / sizeof(placement[0]))) == 1,
          "CPU-spill multi-tier placement must stay blocked, even on SYCL");
    fprintf(stderr, "  test_cpu_spill_placement_still_blocked OK\n");
    return 0;
}

static int test_single_tier_placement_not_blocked(void) {
    /* Every entry on the same tier: not genuinely multi-tier, and the
     * predicate must not need e->multi_tier to already be false to say
     * so -- it only looks for a CPU entry. */
    const int placement[] = {0, 0, 0};
    CHECK(ds4_test_ssd_streaming_multi_tier_blocked(
                  placement, (int)(sizeof(placement) / sizeof(placement[0]))) == 0,
          "an all-GPU placement with no CPU entry must not be blocked");
    fprintf(stderr, "  test_single_tier_placement_not_blocked OK\n");
    return 0;
}

/* ---- ds4_backend_supports_streaming_auto_cache ------------------------- */

static int test_sycl_supports_streaming_auto_cache(void) {
    CHECK(ds4_test_backend_supports_streaming_auto_cache(DS4_BACKEND_CUDA) != 0,
          "SYCL (backend==DS4_BACKEND_CUDA on this build) must support "
          "streaming auto cache");
    fprintf(stderr, "  test_sycl_supports_streaming_auto_cache OK\n");
    return 0;
}

/* ---- ds4_engine_configure_streaming_auto_cache's per-tier loop --------- */

static int test_auto_cache_configures_every_tier(void) {
    ds4_gpu_cleanup();
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0,
          "init_multi for two logical tiers (same physical device)");
    ds4_gpu_set_ssd_streaming(true);

    CHECK(ds4_test_configure_streaming_auto_cache_multi_tier(2) != 0,
          "configure_streaming_auto_cache must succeed for a synthetic "
          "2-tier engine with one routed-expert layer");

    const uint32_t tier0 = ds4_gpu_stream_expert_cache_configured_count_on(0);
    const uint32_t tier1 = ds4_gpu_stream_expert_cache_configured_count_on(1);
    CHECK(tier0 != 0, "tier 0's expert cache must be configured (nonzero budget)");
    CHECK(tier1 != 0,
          "tier 1's expert cache must ALSO be configured: before this "
          "plan, the multi-tier engine-creation branch never ran the "
          "streaming setup at all, so tier 1 would stay at budget 0");
    /* Both logical tiers share the same physical device and the same
     * synthetic weights, so their independently-computed plans must
     * agree exactly. */
    CHECK(tier0 == tier1,
          "two tiers on identical hardware with identical weights must "
          "get the same computed budget");

    ds4_gpu_cleanup();
    fprintf(stderr, "  test_auto_cache_configures_every_tier OK\n");
    return 0;
}

static int test_auto_cache_single_tier_unaffected(void) {
    /* gpu_cfg.n_gpus == 0 is the single-tier legacy shape (no gpu_cfg
     * passed to engine_classify_multi_tier at all): the per-tier loop
     * inside ds4_engine_configure_streaming_auto_cache must not run for
     * it, since tier 0's own application still belongs to the existing
     * single-tier call site in ds4_engine_create (unreachable from this
     * synthetic test, but this at least proves the loop's own "n_gpus > 1"
     * guard keeps single-tier callers untouched). */
    ds4_gpu_cleanup();
    CHECK(ds4_gpu_init() != 0, "init for single-tier regression check");
    ds4_sycl_stream_test_reset();
    ds4_gpu_set_ssd_streaming(true);

    CHECK(ds4_test_configure_streaming_auto_cache_multi_tier(0) != 0,
          "configure_streaming_auto_cache must still succeed with "
          "gpu_cfg.n_gpus == 0 (single-tier shape)");
    CHECK(ds4_gpu_stream_expert_cache_configured_count_on(0) == 0,
          "single-tier shape must not apply any budget itself: the real "
          "single-tier call site (unreached here) owns that");

    ds4_gpu_cleanup();
    fprintf(stderr, "  test_auto_cache_single_tier_unaffected OK\n");
    return 0;
}

int main(void) {
    if (test_gpu_only_placement_admitted_on_sycl() != 0) return 1;
    if (test_cpu_spill_placement_still_blocked() != 0) return 1;
    if (test_single_tier_placement_not_blocked() != 0) return 1;
    if (test_sycl_supports_streaming_auto_cache() != 0) return 1;
    if (test_auto_cache_configures_every_tier() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_auto_cache_single_tier_unaffected() != 0) { ds4_gpu_cleanup(); return 1; }
    fprintf(stderr, "  test_sycl_engine_streaming_mgpu OK\n");
    return 0;
}
