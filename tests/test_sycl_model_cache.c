/* Device-resident model-weight cache tests: ds4_gpu_cache_model_range
 * and the sycl_model_range_ptr lookup it feeds (sycl/ds4_sycl_model_cache.hpp,
 * sycl/ds4_sycl_common.hpp).
 *
 * Per design-spec section 6w, a numbers-only test cannot distinguish the
 * cached path from the staged path: both read the same bytes and produce
 * identical output by construction. Every test below that claims caching
 * happened checks the cache's own instrumentation (byte count, hit/miss
 * counters, or a direct is-cached query), never logits or output tensors.
 * tests/test_sycl_full_token.c separately proves bit-identical output
 * on versus off through a real decode.
 *
 * This machine has one A770, so every test below runs against the one
 * real tier available; nothing here can exercise a genuine cross-tier
 * cache. Deliberately self-contained: no shared harness header, matching
 * tests/test_sycl_placement.c and tests/test_sycl_mgpu.c. Needs no model
 * file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                      \
        }                                                                   \
    } while (0)

/* sycl/ds4_sycl_model_cache.hpp test-only hooks; not part of the ABI. */
extern uint64_t ds4_sycl_test_model_cache_hit_count(void);
extern uint64_t ds4_sycl_test_model_cache_miss_count(void);
extern uint64_t ds4_sycl_test_model_cache_bytes(int tier);
extern void     ds4_sycl_test_model_cache_set_wrong_delta(uint64_t delta);
extern int      ds4_sycl_test_model_range_is_cached(const void *model_map, uint64_t model_size,
                                                     uint64_t offset, uint64_t bytes);
extern int      ds4_sycl_test_model_cache_read_back(const void *model_map, uint64_t offset,
                                                     uint64_t bytes, void *out);

/* ---- Return-polarity and argument validation ---------------------------
 *
 * Spec 3a: read this entry's polarity from its own contract rather than
 * assuming a blanket convention. ds4_gpu_cache_model_range's own comment
 * (sycl/ds4_sycl_model_cache.hpp) records the trap: the caller
 * (accelerator_cache_model_tensors, ds4.c) calls this cache "optional" but
 * ABORTS ds4_engine_open if it returns 0, so every path except a
 * genuinely invalid (offset, bytes) against model_size must return 1. */

static int test_cache_model_range_null_and_zero_is_success(void) {
    unsigned char host[16];
    CHECK(ds4_gpu_cache_model_range(NULL, 4096, 0, 16, "null-map") == 1,
          "a NULL model_map must report success (nothing to cache, nothing wrong)");
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, 0, "zero-bytes") == 1,
          "a zero-byte range must report success");
    return 0;
}

static int test_cache_model_range_out_of_bounds_fails(void) {
    unsigned char host[64];
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 100, 16, "past-end") == 0,
          "an offset past model_size must fail");
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 60, 16, "overruns-end") == 0,
          "a range extending past model_size must fail");
    /* Overflow-safe: offset + bytes must not wrap and appear in-bounds. */
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 8, UINT64_MAX - 4, "overflow") == 0,
          "an overflowing offset+bytes must fail, not wrap into a false pass");
    return 0;
}

/* One of the "four things that will bite": declining to cache
 * must still report success. No device at all (init never ran) is the
 * most extreme decline case. */
static int test_cache_model_range_no_device_declines_gracefully(void) {
    unsigned char host[4096];
    memset(host, 0x42, sizeof(host));
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, sizeof(host), "no-device") == 1,
          "caching with no device initialised must decline as success, not fail");
    return 0;
}

/* ---- Real allocation, lookup and sub-range containment ------------------ */

static int test_cache_range_allocates_and_resolves(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for allocation test");

    unsigned char host[8192];
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 31u + 7u);

    CHECK(ds4_sycl_test_model_cache_bytes(0) == 0,
          "cache starts empty in a fresh process");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 0, 4096) == 0,
          "nothing is cached before the first ds4_gpu_cache_model_range call");

    const uint64_t hits_before = ds4_sycl_test_model_cache_hit_count();
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, 4096, "range-a") == 1,
          "a valid, budget-fitting range must cache successfully");
    CHECK(ds4_sycl_test_model_cache_bytes(0) == 4096,
          "cache_bytes reflects exactly the bytes just cached");

    /* Exact-range and sub-range lookups both resolve to a device pointer
     * (CUDA's cuda_model_range_ptr containment logic, sub-ranges
     * included, not just exact-offset hits). */
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 0, 4096) == 1,
          "the exact cached range must resolve to a device pointer");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 1024, 512) == 1,
          "a sub-range fully inside the cached range must resolve to a device pointer");
    CHECK(ds4_sycl_test_model_cache_hit_count() > hits_before,
          "resolving a cached range must increment the hit counter");

    /* A range only partially inside the cached span, or entirely outside
     * it, must NOT resolve -- the fallback (host pointer, per-call
     * staging) must still be reachable for genuinely uncached bytes. */
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 3800, 512) == 0,
          "a range spanning past the end of the cached span must miss, not "
          "silently truncate to a wrong device pointer");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 6000, 100) == 0,
          "a range entirely outside the cached span must miss");

    /* Read-back proves the copied bytes are correct, not just that a
     * pointer was returned. */
    unsigned char readback[512];
    memset(readback, 0, sizeof(readback));
    CHECK(ds4_sycl_test_model_cache_read_back(host, 1024, 512, readback) == 1,
          "read-back of a cached sub-range must hit");
    CHECK(memcmp(readback, host + 1024, 512) == 0,
          "read-back bytes must match the original host bytes exactly");

    /* A second call with a range already fully covered by an existing
     * cached range is a no-op success and must not double-count bytes. */
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, 4096, "range-a-again") == 1,
          "re-caching an already-covered range must still report success");
    CHECK(ds4_sycl_test_model_cache_bytes(0) == 4096,
          "re-caching an already-covered range must not double-count bytes");

    ds4_gpu_cleanup();
    return 0;
}

/* Two disjoint cached spans on the same host buffer, with a genuine gap
 * between them (mirrors accelerator_prepare_model_tensor_spans merging
 * tensors into several large spans rather than one contiguous one): each
 * span resolves on its own, the gap does not, and a request straddling a
 * span boundary is a clean miss rather than a wrong partial read. */
static int test_cache_disjoint_spans_and_gap(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for disjoint-span test");

    unsigned char host[16384];
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 17u + 3u);

    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, 4096, "span-a") == 1,
          "first span caches");
    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 8192, 4096, "span-b") == 1,
          "second span caches");
    CHECK(ds4_sycl_test_model_cache_bytes(0) == 8192,
          "cache_bytes reflects both disjoint spans");

    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 100, 200) == 1,
          "a range inside the first span resolves");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 8200, 200) == 1,
          "a range inside the second span resolves");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 5000, 200) == 0,
          "a range inside the uncached gap between the two spans misses");
    CHECK(ds4_sycl_test_model_range_is_cached(host, sizeof(host), 4000, 200) == 0,
          "a range straddling the end of the first span and the gap must "
          "miss cleanly rather than resolve to a truncated or wrong pointer");

    unsigned char readback[200];
    CHECK(ds4_sycl_test_model_cache_read_back(host, 8200, 200, readback) == 1,
          "read-back inside the second span hits");
    CHECK(memcmp(readback, host + 8200, 200) == 0,
          "second span's read-back bytes match the original host bytes");

    ds4_gpu_cleanup();
    return 0;
}

/* A second, distinct host_base must not be confused with the first: the
 * cache is keyed by host_base as well as offset, exactly as CUDA's own
 * g_model_ranges entries are (r.host_base == model_map). */
static int test_cache_distinguishes_host_base(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for host_base test");

    unsigned char host_a[2048];
    unsigned char host_b[2048];
    memset(host_a, 0xAA, sizeof(host_a));
    memset(host_b, 0xBB, sizeof(host_b));

    CHECK(ds4_gpu_cache_model_range(host_a, sizeof(host_a), 0, 1024, "host-a") == 1,
          "caching host_a's range succeeds");
    CHECK(ds4_sycl_test_model_range_is_cached(host_a, sizeof(host_a), 0, 1024) == 1,
          "host_a's own range resolves");
    CHECK(ds4_sycl_test_model_range_is_cached(host_b, sizeof(host_b), 0, 1024) == 0,
          "the identical (offset, bytes) against a DIFFERENT host_base must "
          "miss, not accidentally resolve to host_a's device copy");

    ds4_gpu_cleanup();
    return 0;
}

/* ---- Bounded cache, shared VRAM ledger, and fallback survival -----------
 *
 * One of the "four things that will bite": the cache must be
 * bounded by ds4_gpu_tier_free_vram's budget, and declining a range that
 * does not fit must still report success and leave the fallback (per-call
 * staging) fully intact. */

/* A real admission through this cache must show up on the SAME shared
 * ledger ds4_gpu_device_cache_tensors uses (ds4_gpu_tier_free_vram,
 * sycl/ds4_sycl_placement.hpp): otherwise the two caches could each
 * believe they have the tier's full headroom and together over-commit
 * it. Mirrors tests/test_sycl_placement.c's own
 * test_tier_free_vram_decreases_with_real_admission for the other
 * cache. */
static int test_cache_committed_bytes_reduce_tier_free_vram(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for ledger test");

    unsigned char host[4096];
    memset(host, 0x77, sizeof(host));

    const uint64_t free_before = ds4_gpu_tier_free_vram(0);
    CHECK(free_before > sizeof(host), "tier has room for a 4 KiB admission");

    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, sizeof(host), "ledger") == 1,
          "a small, budget-fitting range caches");

    const uint64_t free_after = ds4_gpu_tier_free_vram(0);
    CHECK(free_before - free_after == sizeof(host),
          "tier_free_vram drops by exactly the admitted byte count, the "
          "same ledger ds4_gpu_device_cache_tensors shares");

    ds4_gpu_cleanup();
    return 0;
}

/* A range that fits the DECLARED model_size but exceeds the tier's real
 * headroom must be refused before any device allocation or copy is
 * attempted. The declared model_size here is far larger than the real
 * backing allocation, matching tests/test_sycl_placement.c's own
 * headroom-refusal test: the refusal happens purely from the byte-count
 * comparison against ds4_gpu_tier_free_vram, before any byte of the
 * (mostly unbacked) range would be read. */
static int test_cache_declines_when_budget_exceeded_and_fallback_survives(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for budget test");

    const uint64_t free_before = ds4_gpu_tier_free_vram(0);
    CHECK(free_before > 0, "tier reports nonzero free VRAM on a real device");

    unsigned char *host = malloc(4096);
    CHECK(host != NULL, "host buffer alloc for budget test");
    const uint64_t huge_model_size = free_before + (4ull * 1024 * 1024 * 1024);
    const uint64_t too_big = huge_model_size - 4096; /* within the declared mapping */

    CHECK(ds4_gpu_cache_model_range(host, huge_model_size, 0, too_big, "too-big") == 1,
          "a range exceeding real remaining headroom must decline as "
          "success, not fail and abort startup");
    CHECK(ds4_sycl_test_model_cache_bytes(0) == 0,
          "a declined admission must commit nothing to this cache");
    CHECK(ds4_sycl_test_model_range_is_cached(host, huge_model_size, 0, 4096) == 0,
          "a declined range must not resolve to a device pointer");
    CHECK(ds4_gpu_tier_free_vram(0) == free_before,
          "declining must not itself consume any of the tier's headroom");

    /* The fallback -- sycl_model_range_ptr returning the host pointer
     * unchanged when nothing is cached -- must still be exactly today's
     * behaviour: bounded caching must not have broken anything for the
     * bytes it declined. tests/test_sycl_matmul.c and friends already
     * exercise that path exhaustively through real kernels; this test's
     * job is only to prove the decline itself is clean. */

    free(host);
    ds4_gpu_cleanup();
    return 0;
}

/* ---- Ablation: a lookup that resolves to the WRONG range --------------
 *
 * Spec 6n: establish a clean baseline before touching anything. Spec 6w:
 * both a correct and a subtly-wrong lookup produce a real, plausible
 * device pointer, so the mismatch has to be checked by reading the bytes
 * back and comparing, not by whether the call "succeeded". */
static int test_ablation_wrong_range_lookup_is_detected(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for ablation test");

    /* Per spec 6n: a clean arithmetic progression (i & 0xFF, i % N, ...)
     * repeats with a period that can divide evenly into the ablation's
     * own shift, making two genuinely different byte ranges compare equal
     * by coincidence and the ablation fail to fail for the wrong reason.
     * This exact fill DID that on the first draft of this test: i & 0xFF
     * repeats every 256 bytes, and the delta below (2048) is a multiple
     * of 256, so host[0..128) and host[2048..2176) were bytewise
     * identical and the ablation below could not discriminate. A
     * multiplicative hash has no such small period. */
    unsigned char host[4096];
    for (unsigned i = 0; i < sizeof(host); i++) {
        host[i] = (unsigned char)(((i * 2654435761u) >> 13) ^ (i * 40503u));
    }

    CHECK(ds4_gpu_cache_model_range(host, sizeof(host), 0, sizeof(host), "ablation-range") == 1,
          "caching the full 4096-byte range for the ablation test");

    /* Clean baseline, established before any ablation lever is touched
     * (spec 6n): reading the first 128 bytes back must match the host
     * bytes exactly. */
    unsigned char baseline[128];
    CHECK(ds4_sycl_test_model_cache_read_back(host, 0, sizeof(baseline), baseline) == 1,
          "clean baseline read-back must hit");
    CHECK(memcmp(baseline, host, sizeof(baseline)) == 0,
          "clean baseline read-back must match the original host bytes exactly "
          "-- if this fails, the ablation below proves nothing (spec 6n)");

    /* Ablate: force every resolved pointer 2048 bytes further into the
     * SAME cached allocation than it should be (still fully in-bounds:
     * the cached range is 4096 bytes, the request only needs 128 plus
     * the 2048 delta, well under the allocation's size). This is exactly
     * "the lookup returns a cached pointer for the wrong range". */
    ds4_sycl_test_model_cache_set_wrong_delta(2048);

    unsigned char ablated[128];
    CHECK(ds4_sycl_test_model_cache_read_back(host, 0, sizeof(ablated), ablated) == 1,
          "the ablated lookup still resolves (a real, in-bounds, wrong pointer)");
    CHECK(memcmp(ablated, host, sizeof(ablated)) != 0,
          "FAIL TO FAIL: the ablated lookup produced the same bytes as the "
          "correct range -- this test does not discriminate (spec 6w)");
    CHECK(memcmp(ablated, host + 2048, sizeof(ablated)) == 0,
          "the ablated read-back should read exactly the bytes 2048 further "
          "into the same cached buffer, proving precisely what went wrong");

    /* Revert and confirm the correct behaviour returns. */
    ds4_sycl_test_model_cache_set_wrong_delta(0);
    unsigned char restored[128];
    CHECK(ds4_sycl_test_model_cache_read_back(host, 0, sizeof(restored), restored) == 1,
          "read-back after reverting the ablation must still hit");
    CHECK(memcmp(restored, host, sizeof(restored)) == 0,
          "read-back after reverting the ablation must match the original "
          "host bytes exactly again");

    ds4_gpu_cleanup();
    return 0;
}

int main(void) {
    if (test_cache_model_range_null_and_zero_is_success()) return 1;
    if (test_cache_model_range_out_of_bounds_fails()) return 1;
    if (test_cache_model_range_no_device_declines_gracefully()) return 1;
    if (test_cache_range_allocates_and_resolves()) return 1;
    if (test_cache_disjoint_spans_and_gap()) return 1;
    if (test_cache_distinguishes_host_base()) return 1;
    if (test_cache_committed_bytes_reduce_tier_free_vram()) return 1;
    if (test_cache_declines_when_budget_exceeded_and_fallback_survives()) return 1;
    if (test_ablation_wrong_range_lookup_is_detected()) return 1;
    fprintf(stderr, "test_sycl_model_cache: all tests passed\n");
    return 0;
}
