#pragma once

/* Device-resident model-weight cache: the real implementation of
 * ds4_gpu_cache_model_range and of sycl_model_cache_resolve, forward
 * declared in ds4_sycl_common.hpp and called from sycl_model_range_ptr.
 *
 * Structural reference is CUDA's cuda_model_range_ptr (ds4_cuda.cu:703):
 * a range is copied to device memory once and every later lookup for a
 * contained sub-range resolves to `device_ptr + (offset - r.offset)`
 * instead of the host mmap pointer. CUDA's own g_model_ranges /
 * g_model_range_by_offset are a single flat table (CUDA has no separate
 * per-tier addressing for this cache; ds4_gpu_device_cache_tensors /
 * g_dev_cache, sycl/ds4_sycl_placement.hpp's structural reference, is the
 * separate multi-tier mechanism). This cache follows
 * ds4_sycl_streaming.hpp's per-tier convention instead (one table per
 * logical tier, held as an array parallel to g_devices, resolved from
 * g_current_tier at call time with no ABI change) because the SYCL
 * backend's single ds4_gpu_cache_model_range call has no tier argument
 * either, exactly the shape ds4_sycl_streaming.hpp's own header comment
 * documents for its resident expert cache and ds4_gpu_stream_expert_
 * cache_* entries.
 *
 * Multi-GPU execution is not yet implemented in this backend (see
 * ds4_gpu_init's own log message in ds4_sycl.cpp), so g_current_tier is
 * always 0 in every configuration this cache can currently be exercised
 * in; the per-tier structure is sized correctly for when that lands
 * rather than hard-coding tier 0.
 *
 * Included after g_devices, g_current_tier, ds4_sycl_current_tier and
 * ds4_sycl_queue are declared (ds4_sycl.cpp), and before
 * ds4_sycl_placement.hpp, whose ds4_gpu_tier_free_vram must also
 * subtract this cache's committed bytes (sycl_model_cache_committed_bytes
 * below) from its budget -- the two caches share one VRAM ledger, exactly
 * as ds4_gpu_tier_free_vram's own header comment already does for the
 * placement cache and the streaming resident-expert cache. */

#include "ds4_sycl_common.hpp"

#include <cstdlib>
#include <vector>

struct sycl_model_cache_range {
    const void    *host_base;
    uint64_t       offset;
    uint64_t       bytes;
    unsigned char *device_ptr;
};

struct sycl_model_cache_tier {
    std::vector<sycl_model_cache_range> ranges;
    uint64_t                            bytes = 0;
};

/* Indexed by logical tier, parallel to g_devices -- see this file's
 * header comment. */
static std::vector<sycl_model_cache_tier> g_sycl_model_cache_tier;

static sycl_model_cache_tier &sycl_model_cache_tier_for(int tier) {
    if ((size_t)tier >= g_sycl_model_cache_tier.size()) {
        g_sycl_model_cache_tier.resize((size_t)tier + 1);
    }
    return g_sycl_model_cache_tier[(size_t)tier];
}

/* Read by ds4_gpu_tier_free_vram (sycl/ds4_sycl_placement.hpp) so the
 * placement cache and this cache draw from one shared VRAM ledger instead
 * of each independently believing it has the whole budget. */
static uint64_t sycl_model_cache_committed_bytes(int tier) {
    if (tier < 0 || (size_t)tier >= g_sycl_model_cache_tier.size()) return 0;
    return g_sycl_model_cache_tier[(size_t)tier].bytes;
}

/* ---- Test-only instrumentation ------------------------------------------
 *
 * Per design-spec section 6w, a numbers-only test cannot distinguish the
 * cached path from the staged path: both produce identical logits by
 * construction. These hooks let a test prove which path actually ran,
 * the same shape used for ds4_sycl_moe_test_last_staged_expert_
 * count. Not part of the ABI. */

static uint64_t g_sycl_model_cache_hit_count  = 0;
static uint64_t g_sycl_model_cache_miss_count = 0;

extern "C" uint64_t ds4_sycl_test_model_cache_hit_count(void) {
    return g_sycl_model_cache_hit_count;
}
extern "C" uint64_t ds4_sycl_test_model_cache_miss_count(void) {
    return g_sycl_model_cache_miss_count;
}

extern "C" uint64_t ds4_sycl_test_model_cache_bytes(int tier) {
    return sycl_model_cache_committed_bytes(tier);
}

/* Ablation lever (spec 6n: establish a clean baseline before touching
 * this). 0 (the default) is the correct lookup. A nonzero delta is added
 * to every resolved device pointer, simulating "the lookup resolved to a
 * pointer for the wrong range" -- the exact defect class the
 * device-resident cache's correctness requirement is about -- while staying in-bounds: a test
 * using this caches a range with at least `delta` bytes of slack beyond
 * what it reads, so the corrupted pointer still lands inside the same
 * device allocation and reads different, verifiably wrong, bytes rather
 * than reading out of bounds. */
static uint64_t g_sycl_model_cache_test_wrong_delta = 0;

extern "C" void ds4_sycl_test_model_cache_set_wrong_delta(uint64_t delta) {
    g_sycl_model_cache_test_wrong_delta = delta;
}

/* Real definition of the function forward-declared in
 * ds4_sycl_common.hpp; see that declaration's comment for why a
 * forward-declared static function defined later in the same
 * translation unit is the right shape here. Linear scan: this cache
 * holds the handful of large merged spans accelerator_prepare_model_
 * tensor_spans (ds4.c) builds at startup (tens of entries, not
 * thousands), so CUDA's exact-offset unordered_map fast path is not
 * worth porting here. */
static const char *sycl_model_cache_resolve(const void *model_map,
                                            uint64_t offset, uint64_t bytes) {
    if (!model_map || bytes == 0) return nullptr;
    const int tier = ds4_sycl_current_tier();
    if (tier < 0) {
        g_sycl_model_cache_miss_count++;
        return nullptr;
    }
    const uint64_t end = offset + bytes; /* caller already overflow-checked */

    /* An empty tier vector here is not "no cache": the multi-GPU path
     * fills the placement cache instead, checked below.  Bounds-check
     * rather than returning early, or that path is never reached. */
    if ((size_t)tier < g_sycl_model_cache_tier.size()) {
        const sycl_model_cache_tier &c = g_sycl_model_cache_tier[(size_t)tier];
        for (const sycl_model_cache_range &r : c.ranges) {
            if (r.host_base == model_map && offset >= r.offset && end >= offset &&
                end <= r.offset + r.bytes) {
                g_sycl_model_cache_hit_count++;
                return (const char *)(r.device_ptr + (offset - r.offset) +
                                      g_sycl_model_cache_test_wrong_delta);
            }
        }
    }

    /* Second store, same question: ds4_gpu_device_cache_tensors installed
     * these (see sycl_placement_cache_resolve's own comment for why there
     * are two).  Counted as a hit because that is what it is -- the range
     * is device-resident and the caller must not stage it again. */
    if (const char *placed = sycl_placement_cache_resolve(model_map, offset, bytes)) {
        g_sycl_model_cache_hit_count++;
        return placed + g_sycl_model_cache_test_wrong_delta;
    }

    g_sycl_model_cache_miss_count++;
    return nullptr;
}

/* Test-only: true when sycl_model_range_ptr would resolve this exact
 * (model_map, offset, bytes) to a cached device pointer rather than the
 * host mmap. A direct, semantic answer to "was this specific range
 * cached", complementing the hit/miss counters above (which prove a
 * cache was consulted across a whole run, not which range). Not part of
 * the ABI. */
extern "C" int ds4_sycl_test_model_range_is_cached(const void *model_map, uint64_t model_size,
                                                    uint64_t offset, uint64_t bytes) {
    if (g_devices.empty()) return 0;
    const char *p = sycl_model_range_ptr(model_map, offset, bytes, model_size, "test-lookup");
    if (!p) return 0;
    return sycl_ptr_is_device_resident(ds4_sycl_current_queue(), p) ? 1 : 0;
}

/* Test-only: resolves [offset, offset+bytes) against the cache for
 * `model_map` on the current tier (the ablation lever above included) and
 * copies the result back to `out`. Returns 1 on a covering hit, 0 on a
 * miss or bad argument. Mirrors ds4_sycl_test_placement_read_back
 * (sycl/ds4_sycl_placement.hpp) for the same reason: proving this cache's
 * correctness, and its ablation, needs a real device round trip, not just
 * an in-bounds pointer check. Not part of the ABI. */
extern "C" int ds4_sycl_test_model_cache_read_back(const void *model_map, uint64_t offset,
                                                    uint64_t bytes, void *out) {
    if (!out || bytes == 0 || g_devices.empty()) return 0;
    const char *p = sycl_model_cache_resolve(model_map, offset, bytes);
    if (!p) return 0;
    try {
        ds4_sycl_current_queue().memcpy(out, p, (size_t)bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "test model cache read-back failed: %s\n", e.what());
        return 0;
    }
}

/* Forward-declared in ds4_sycl.cpp and called from ds4_gpu_cleanup while
 * every tier's queue is still alive, matching sycl_stream_teardown_all
 * and sycl_placement_teardown_all's identical ordering contract: freeing
 * this cache's device buffers AFTER g_devices.clear() destroys their
 * queues would be a use-after-free of the allocation context (spec 6g),
 * so this must run BEFORE that clear. */
static void sycl_model_cache_teardown_all(void) {
    for (size_t t = 0; t < g_sycl_model_cache_tier.size(); t++) {
        sycl_model_cache_tier &c = g_sycl_model_cache_tier[t];
        if (t < g_devices.size()) {
            for (sycl_model_cache_range &r : c.ranges) {
                if (!r.device_ptr) continue;
                try {
                    ds4_sycl_queue((int)t).wait_and_throw();
                    sycl::free(r.device_ptr, ds4_sycl_queue((int)t));
                } catch (const sycl::exception &e) {
                    fprintf(stderr, DS4_GPU_LOG_PREFIX
                            "model range cache teardown failed: %s\n", e.what());
                }
            }
        }
        c.ranges.clear();
        c.bytes = 0;
    }
}

/* ds4_gpu_cache_model_range: 1 on success (including "validated but
 * declined to cache"), 0 only for a genuinely invalid range. Read this
 * comment fully before changing the return value on any new path.
 *
 * ds4.c's caller (accelerator_cache_model_tensors ->
 * accelerator_prepare_model_tensor_spans, ds4.c:2978) treats this cache
 * as an optional performance layer -- its own comment there calls it
 * "optional model cache" -- but ABORTS ds4_engine_open
 * (ds4.c:59504-59512, ds4.c:59521-59528) if this returns 0. Every
 * SYCL kernel that reads a weight range through sycl_model_range_ptr
 * already has a fully correct fallback (the per-call host-to-device
 * staging that predates this cache), so a failure to cache -- budget too
 * small, an allocation failure, a copy that threw -- is never a reason
 * to fail startup: only a genuinely out-of-bounds (offset, bytes)
 * against model_size, which indicates a caller bug rather than a VRAM
 * condition, returns 0. This mirrors the stub this function replaces,
 * which recorded exactly this trap in its own comment, and CUDA's own
 * ds4_gpu_cache_model_range (ds4_cuda.cu:4494), which only ever fails
 * this same way. */
extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                                         uint64_t offset, uint64_t bytes,
                                         const char *label) {
    if (model_map == nullptr || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;

    /* No device to cache onto (init never ran, or already torn down):
     * decline, per this function's own "optional" contract above. */
    if (g_devices.empty()) return 1;
    const int tier = ds4_sycl_current_tier();
    if (tier < 0 || (size_t)tier >= g_devices.size()) return 1;

    /* Already covered by an earlier call (accelerator_prepare_model_
     * tensor_spans can be invoked more than once per engine, e.g. once
     * for the main model and once for a support model with a distinct
     * host_base, or twice for the same host_base if the caller ever
     * merges spans differently across calls): avoid double-caching and
     * double-counting committed bytes. */
    if (sycl_model_cache_resolve(model_map, offset, bytes) != nullptr) return 1;

    sycl::queue &q = ds4_sycl_queue(tier);

    /* Bounded by the same shared VRAM ledger ds4_gpu_device_cache_tensors
     * uses (sycl/ds4_sycl_placement.hpp): decline gracefully rather than
     * risk starving the tier of room for the graph scratch, cuBLAS
     * workspace and every other allocation session_create makes after
     * this runs. An 80 GiB model against one A770 or one 24 GiB B60
     * cannot fit here in full; declining the remainder and falling back
     * to per-call staging for it is the documented behaviour
     * ("the fallback must survive"), not a bug. */
    const uint64_t free_before = ds4_gpu_tier_free_vram(tier);
    if (bytes > free_before) {
        if (getenv("DS4_SYCL_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "declining to cache %s (%.2f MiB): only %.2f MiB free on "
                    "tier %d; falling back to per-call staging for this range\n",
                    label ? label : "weights", (double)bytes / 1048576.0,
                    (double)free_before / 1048576.0, tier);
        }
        return 1;
    }

    unsigned char *dev = sycl::malloc_device<unsigned char>((size_t)bytes, q);
    if (!dev) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "model range cache allocation failed for %s (%.2f MiB); "
                "falling back to per-call staging for this range\n",
                label ? label : "weights", (double)bytes / 1048576.0);
        return 1;
    }
    try {
        /* Real-hardware testing against a genuine 1.83 GiB memory-mapped
         * GGUF file (tests/test_sycl_gguf_load.c) -- large enough to
         * reach an offset the CPU had never previously touched, which no
         * earlier test on this backend was -- found that copying straight
         * from `model_map + offset` here reproduced spec 6l's "unstaged
         * host read returns zero" defect one layer lower, in the DMA
         * engine driving this copy rather than in a compute kernel.
         * sycl_copy_host_to_device_paged_safe (ds4_sycl_common.hpp) has
         * the full account and the fix; every host-to-device copy of
         * model-mapped bytes in this backend must go through it. */
        sycl_copy_host_to_device_paged_safe(q, dev, (const unsigned char *)model_map + offset, bytes);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "model range cache copy failed for %s: %s; falling back to "
                "per-call staging for this range\n",
                label ? label : "weights", e.what());
        try {
            sycl::free(dev, q);
        } catch (const sycl::exception &free_err) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "model range cache cleanup free failed: %s\n", free_err.what());
        }
        return 1;
    }

    sycl_model_cache_tier &c = sycl_model_cache_tier_for(tier);
    c.ranges.push_back({model_map, offset, bytes, dev});
    c.bytes += bytes;

    if (getenv("DS4_SYCL_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "cached %s %.2f MiB on tier %d (total %.2f GiB)\n",
                label ? label : "weights", (double)bytes / 1048576.0, tier,
                (double)c.bytes / 1073741824.0);
    }
    return 1;
}
