/* Full single-token decode test for the SYCL backend.
 *
 * tests/test_sycl_full_layer.c proved one DeepSeek V4 Flash decoder
 * layer encodes through the engine's real per-token path
 * (metal_graph_encode_decode_layer). A real decode token needs
 * metal_graph_encode_token_raw_swa (ds4.c:27485), which runs ALL layers
 * plus the output head and then reads logits back
 * (metal_graph_eval_token_raw_swa, ds4.c:30748). Everything in that
 * remaining stretch is implemented and unit-tested and had never executed
 * before this test.
 *
 * This test drives ds4_test_graph_full_token_encode (ds4.c, DS4_TEST_HOOKS),
 * which calls metal_graph_eval_token_raw_swa directly -- the exact function
 * every real decode token calls -- on a synthetic DS4_TW_N_LAYER-layer
 * (4-layer) Flash shape built by tests/test_sycl_layer_weights.h, with no
 * model file. DS4_TW_N_HASH_LAYER (3) of those layers route by a hash
 * lookup table (ffn_gate_tid2eid) instead of top-k, so this is also the
 * first test on this backend to exercise hash routing and the first to
 * exercise the layer-to-layer hyper-connection carry (a single layer, as
 * test_sycl_full_layer.c drives, cannot thread state between iterations of
 * a loop it never runs).
 *
 * Four assertions, in order of value:
 *   1. The token encode returns success at all.
 *   2. The logits read back are finite and not all zero. This is a
 *      different claim from (1): an unstaged mmap read can return zeros
 *      and still report success, and an output tensor that is allocated
 *      but never read back proves nothing.
 *   3. The logits change when the input token changes.
 *   4. Determinism: encoding the same token twice produces bit-identical
 *      logits. ds4_sycl.cpp builds queues without the in_order property,
 *      and a queue-ordering race has already been found and fixed once on
 *      this backend; running the same input twice and requiring identical
 *      output is the cheapest race detector available. This is checked
 *      with several independent repeats, not a single pair, and any
 *      mismatch is reported as a flaky result rather than absorbed into a
 *      tolerance. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "test_sycl_layer_weights.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* DS4_TEST_HOOKS entry points defined in ds4.c; not declared in a shipped
 * header, following the same pattern as ds4_test_graph_full_layer_encode
 * in tests/test_sycl_full_layer.c. ds4_test_graph_full_token_encode_cached
 * is the cached sibling: identical except it hands the synthetic model to
 * ds4_gpu_cache_model_range before running, so the whole decode below
 * reads weights from the device-resident cache instead of staging them
 * per call. */
int ds4_test_graph_full_token_encode(int token, float *out_logits, uint64_t out_logits_floats);
int ds4_test_graph_full_token_encode_cached(int token, float *out_logits, uint64_t out_logits_floats);

/* sycl/ds4_sycl_model_cache.hpp test-only hooks; not part of the
 * ABI. Used below to prove the cached run actually took the cached path,
 * not merely that its output happens to match -- per design-spec 6w, a
 * numbers-only comparison cannot distinguish the two by construction. */
extern uint64_t ds4_sycl_test_model_cache_hit_count(void);
extern uint64_t ds4_sycl_test_model_cache_bytes(int tier);

/* sycl/ds4_sycl_moe_launch.hpp test-only hook; not part of the
 * ABI. This synthetic Flash shape's hash-routed and top-k-routed layers
 * both exercise sycl_routed_moe_launch, so this confirms the MoE-specific
 * bypass (skip compaction and staging entirely, read the cached table at
 * its real expert id) actually ran, not just the simpler dense/attention
 * weight paths through sycl_stage_host_bytes. */
extern int ds4_sycl_moe_test_last_used_resident_weights(void);

/* sycl/ds4_sycl_common.hpp test/report-only hook; not part of
 * the ABI. Cumulative bytes actually copied host-to-device by
 * sycl_stage_host_bytes, which measurement confirms is the chokepoint
 * (directly or via the two MoE staging helpers) essentially all
 * host-to-device weight traffic in this backend passes through. */
extern uint64_t ds4_sycl_test_stage_host_bytes_total(void);
extern void     ds4_sycl_test_stage_host_bytes_reset(void);

static double now_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

/* This test needs measured host-to-device bytes and wall-clock per
 * token, caching on versus off. This synthetic DS4_TW_N_LAYER-layer shape
 * is far smaller than a real Flash model, so the ABSOLUTE numbers here are
 * not representative of a real decode; what is representative is the
 * SHAPE of the difference (bytes essentially disappear, wall-clock drops)
 * and the report says so explicitly rather than presenting a toy number
 * as if it were a production one. Each cached repeat still pays a fresh
 * ds4_gpu_cache_model_range call (this harness builds a new synthetic
 * model, and therefore a new cache entry, every repeat -- there is no
 * persistent-model/many-tokens harness on this backend to measure a
 * steady-state decode separately from the one-time cache install), so the
 * cached numbers below include that one-time cost, not just decode.
 *
 * ds4_gpu_cleanup() before each phase is load-bearing, not cleanliness:
 * this harness frees each call's synthetic host buffer
 * (ds4_test_free_flash_layer_weights) and the next call's calloc of the
 * identical size routinely reuses that exact freed address. The
 * model-range cache keys purely on (host_base, offset, bytes) with no
 * content check, so a cache entry installed by an EARLIER call (assertion
 * 5 above, before this function ever runs) can still match a LATER call's
 * reused address even after that earlier host buffer was freed -- an
 * uncached call would then silently resolve through the stale cached
 * device pointer instead of staging anything, reporting 0 bytes staged
 * for a call that never asked to be cached at all. Observed directly: an
 * earlier draft of this measurement, run right after assertion 5 without
 * this cleanup, reported 0 bytes staged for BOTH the cached and
 * "uncached" loops, because both were secretly served by assertion 5's
 * leftover cache entry. In production this cannot happen: ds4_engine_
 * close always calls ds4_gpu_cleanup (ds4.c) before a closed engine's
 * model.map could ever be replaced by a new one, which is exactly what
 * ds4_gpu_cleanup here restores between phases -- a clean cache with no
 * address it could accidentally still recognise. */
static int measure_bytes_and_time(int repeats) {
    ds4_gpu_cleanup();
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for the uncached measurement phase");
    ds4_sycl_test_stage_host_bytes_reset();
    const double t0 = now_secs();
    for (int r = 0; r < repeats; r++) {
        float logits[DS4_TW_N_VOCAB];
        (void)ds4_test_graph_full_token_encode(/*token=*/3, logits, DS4_TW_N_VOCAB);
    }
    const double t1 = now_secs();
    const uint64_t bytes_uncached = ds4_sycl_test_stage_host_bytes_total();

    ds4_gpu_cleanup();
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for the cached measurement phase");
    ds4_sycl_test_stage_host_bytes_reset();
    const double t2 = now_secs();
    for (int r = 0; r < repeats; r++) {
        float logits[DS4_TW_N_VOCAB];
        (void)ds4_test_graph_full_token_encode_cached(/*token=*/3, logits, DS4_TW_N_VOCAB);
    }
    const double t3 = now_secs();
    const uint64_t bytes_cached = ds4_sycl_test_stage_host_bytes_total();

    fprintf(stderr,
            "  measurement (synthetic %u-layer shape, %d repeats, "
            "cache-install cost included in the cached numbers):\n"
            "    uncached: %.3f ms/token, %llu bytes staged/token\n"
            "    cached:   %.3f ms/token, %llu bytes staged/token\n",
            (unsigned)DS4_TW_N_LAYER, repeats,
            (t1 - t0) * 1000.0 / repeats,
            (unsigned long long)(bytes_uncached / (uint64_t)repeats),
            (t3 - t2) * 1000.0 / repeats,
            (unsigned long long)(bytes_cached / (uint64_t)repeats));
    return 0;
}

/* Repeats a full-token encode of the same token DS4_TW_N_VOCAB times over
 * (kRepeats independent encodes, each with a freshly built graph and
 * synthetic weights), returning 1 if every repeat succeeded and produced
 * bit-identical logits to the first, 0 on the first failure or mismatch. A
 * mismatch prints which repeat first diverged and at which vocab index, so
 * a flaky run is reported as a flaky run rather than silently retried. */
#define DS4_FULL_TOKEN_DETERMINISM_REPEATS 5

static int check_determinism(int token, const float *baseline) {
    for (int r = 0; r < DS4_FULL_TOKEN_DETERMINISM_REPEATS; r++) {
        float logits[DS4_TW_N_VOCAB];
        if (!ds4_test_graph_full_token_encode(token, logits, DS4_TW_N_VOCAB)) {
            fprintf(stderr,
                    "FAIL: determinism repeat %d of token %d failed to encode\n",
                    r, token);
            return 0;
        }
        for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
            if (!isfinite(logits[i])) {
                fprintf(stderr,
                        "FAIL: determinism repeat %d produced a non-finite logit at index %llu\n",
                        r, (unsigned long long)i);
                return 0;
            }
        }
        if (memcmp(logits, baseline, sizeof(logits)) != 0) {
            uint64_t first_diff = 0;
            for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
                if (logits[i] != baseline[i]) { first_diff = i; break; }
            }
            fprintf(stderr,
                    "FAIL: determinism repeat %d of token %d diverged from the "
                    "baseline at vocab index %llu (baseline=%.9g, repeat=%.9g) -- "
                    "this is a FLAKY result, not a clean pass or a clean fail\n",
                    r, token, (unsigned long long)first_diff,
                    (double)baseline[first_diff], (double)logits[first_diff]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    float logits_a[DS4_TW_N_VOCAB];
    float logits_b[DS4_TW_N_VOCAB];
    memset(logits_a, 0, sizeof(logits_a));
    memset(logits_b, 0, sizeof(logits_b));

    /* Assertion 1: a full token -- all layers, both hash-routed and
     * top-k-routed, plus the output head -- encodes and reads logits back
     * successfully. Nothing on this backend has ever established this
     * before. */
    CHECK(ds4_test_graph_full_token_encode(/*token=*/3, logits_a, DS4_TW_N_VOCAB) != 0,
          "a full DeepSeek V4 Flash token failed to encode end to end "
          "(alloc/begin_commands/embed/layers/output-head/end_commands/"
          "readback)");

    /* Assertion 2: the values actually read back are finite and not all
     * zero, not just that the call reported success. */
    int any_nonzero = 0;
    for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
        CHECK(isfinite(logits_a[i]), "token logits contain a non-finite value");
        if (logits_a[i] != 0.0f) any_nonzero = 1;
    }
    CHECK(any_nonzero, "token logits are all zero");

    /* Assertion 3: encoding a different token produces different logits. */
    CHECK(ds4_test_graph_full_token_encode(/*token=*/97, logits_b, DS4_TW_N_VOCAB) != 0,
          "a full-token encode with a different input token failed");

    int any_diff = 0;
    for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
        CHECK(isfinite(logits_b[i]),
              "token logits for the second token contain a non-finite value");
        if (logits_a[i] != logits_b[i]) any_diff = 1;
    }
    CHECK(any_diff, "token logits did not change when the input token changed");

    /* Assertion 4: the same token, encoded independently several times
     * (fresh graph and fresh synthetic weights each time), produces
     * bit-identical logits every time. ds4_sycl.cpp builds queues without
     * the in_order property; this is the cheapest available detector for
     * a queue-ordering race across a graph with enough concurrent work
     * (four layers of routed MoE) to expose one. */
    CHECK(check_determinism(/*token=*/3, logits_a) != 0,
          "the same token did not produce bit-identical logits across "
          "repeated encodes -- see the diagnostic above for whether this "
          "was a clean failure or a flaky one");

    /* Assertion 5: the device-resident model-range cache. logits_a
     * above is already a clean baseline of the UNCACHED path (spec 6n):
     * ds4_test_graph_full_token_encode never calls ds4_gpu_cache_model_
     * range, and check_determinism just reconfirmed it is stable. Running
     * the cached sibling and requiring bit-identical output, PLUS proof
     * the cached path was actually taken, is what distinguishes "caching
     * is correct" from "caching was silently skipped and got lucky": per
     * design-spec 6w, both paths produce identical numbers by
     * construction, so the hit counter and the cache's own byte count,
     * not the logits, are what tell the two apart. */
    const uint64_t hits_before = ds4_sycl_test_model_cache_hit_count();
    const uint64_t cache_bytes_before = ds4_sycl_test_model_cache_bytes(0);

    float logits_cached[DS4_TW_N_VOCAB];
    memset(logits_cached, 0, sizeof(logits_cached));
    CHECK(ds4_test_graph_full_token_encode_cached(/*token=*/3, logits_cached,
                                                  DS4_TW_N_VOCAB) != 0,
          "the cached full-token encode failed");

    CHECK(ds4_sycl_test_model_cache_bytes(0) > cache_bytes_before,
          "the cached run did not grow the model-range cache at all -- "
          "ds4_gpu_cache_model_range declined instead of caching a model "
          "this small");
    CHECK(ds4_sycl_test_model_cache_hit_count() > hits_before,
          "the cached run's weight lookups never hit the cache: the cache "
          "was installed but every kernel still read through the staging "
          "path, which is not what this assertion is checking");
    CHECK(memcmp(logits_cached, logits_a, sizeof(logits_a)) == 0,
          "caching the model changed the logits: caching on and off must "
          "produce EXACTLY the same output, not merely close output");
    CHECK(ds4_sycl_moe_test_last_used_resident_weights() != 0,
          "the routed-MoE dispatcher did not take the device-resident "
          "bypass on the last layer's MoE call even though the whole "
          "synthetic model was cached -- MoE weights are still being "
          "staged or compacted instead of read directly");

    fprintf(stderr,
            "  test_sycl_full_token OK (determinism stable across %d repeats, "
            "cached vs uncached bit-identical)\n",
            DS4_FULL_TOKEN_DETERMINISM_REPEATS);

    if (measure_bytes_and_time(10)) return 1;

    ds4_gpu_cleanup();
    return 0;
}
