#pragma once

/* Multi-tier placement: the per-device free-VRAM query, the no-copy model
 * map registration, and the permanent per-device selective weight cache.
 * Implements the three ds4_gpu_mgpu.h entries this backend needs to place
 * DeepSeek V4 Flash (roughly 80 GiB) across multiple B60 tiers (24 GiB
 * each, 336 GiB aggregate): ds4_gpu_tier_free_vram,_register_model_map_
 * no_copy and _device_cache_tensors. ds4_gpu_device_cache_support_tensors
 * and ds4_gpu_register_support_map stay stubbed in ds4_sycl_unavailable.cpp:
 * they are DSpark speculative-decoding support-model entries, out of scope
 * for baseline Flash.
 *
 * Structural reference is ds4_cuda.cu's g_dev_cache / g_cache_ranges /
 * ds4_gpu_device_cache_tensors (ds4_cuda.cu:3928-4095) and its
 * ds4_gpu_register_model_map_no_copy (ds4_cuda.cu:3797-3850): same
 * validate-before-allocate shape, same grow-by-reallocate-and-copy slab,
 * same sorted range table. Two points genuinely differ, not by oversight:
 *
 *   1. CUDA's ds4_gpu_lookup_cache / _strict resolve a cached range back
 *      to a device pointer for its "_tensor"-suffixed kernel-dispatch
 *      wrappers (ds4_cuda.cu:4200-4300). SYCL has no such wrappers: every
 *      "_tensor" entry is permanently stubbed here (see
 *      ds4_sycl_unavailable.cpp's own comment on
 *      ds4_gpu_hc_expand_add_tensor and neighbours), because this
 *      backend's kernels are dispatched directly, not through that
 *      generic ABI. So nothing in this codebase ever calls
 *      ds4_gpu_lookup_cache/_strict/_device for SYCL, confirmed by grep
 *      (ds4.c never calls them directly either), and they are correctly
 *      absent from both ds4_sycl_unavailable.cpp's stub list and this
 *      file: implementing them would add dead code. The cache built here
 *      is real (weights genuinely move to and stay on device), but readers
 *      are future work: ds4_sycl_streaming.hpp's own header
 *      comment documents the identical gap for the resident expert cache.
 *      A test-only accessor below (ds4_sycl_test_placement_read_back)
 *      exists so this cache's correctness is provable now.
 *
 *   2. CUDA's admission check calls cudaMemGetInfo for a live headroom
 *      reading. Per spec 6p, Level Zero Sysman's free-memory reading does
 *      not move under real allocation pressure on this driver and is
 *      degenerate (free == total) on an idle device, so a live query here
 *      would look dynamic while lying. ds4_gpu_tier_free_vram instead
 *      starts from the same static ceiling ds4_sycl_streaming.hpp already
 *      uses (total device memory minus a fixed reserve, sycl_stream_vram_
 *      ceiling) and subtracts bytes THIS backend has itself committed on
 *      that tier: the selective cache built here, plus the streaming
 *      resident expert cache. That number is honest by construction --
 *      it only ever reports what we did not already spend -- and it
 *      strictly decreases as either cache admits more, which a number
 *      that never moves cannot do. A conservative constant that never
 *      decreases would be worse: the placement planner would believe it
 *      even after this cache is nearly full. See spec 6p before
 *      considering a live Sysman upgrade here. */

#include "ds4_sycl_common.hpp"

#include <algorithm>
#include <vector>

struct sycl_placement_cache_range {
    uint64_t       source_offset;
    uint64_t       bytes;
    unsigned char *device_ptr;
};

struct sycl_placement_tier_cache {
    unsigned char                          *base  = nullptr;
    uint64_t                                bytes = 0;
    std::vector<sycl_placement_cache_range> ranges;
};

/* Indexed by LOGICAL tier, parallel to g_devices, like every other
 * per-tier table in this backend (g_sycl_stream_tier,
 * g_sycl_model_cache_tier) and like the "_on(tier)" entries that feed
 * them (ds4_gpu_tensor_alloc_on, sycl/ds4_sycl_mgpu.hpp). It is NOT
 * indexed by the physical device id the operator passed in --gpu-devices:
 * ds4_gpu_device_cache_tensors below receives one of those and translates
 * it with sycl_tier_for_device_id (sycl/ds4_sycl_mgpu.hpp) before
 * touching this table, so a non-contiguous ordering such as
 * --gpu-devices 0,2,4,6,1,3,5,7 resolves to the tier that actually owns
 * the card. CUDA can index its own slabs by the physical id directly
 * (g_dev_cache[device_id], ds4_cuda.cu:3973) because cudaSetDevice takes
 * one; SYCL has no such id space, only tiers. */
static std::vector<sycl_placement_tier_cache> g_placement_tier;

static sycl_placement_tier_cache &sycl_placement_cache_for(int tier) {
    if ((size_t)tier >= g_placement_tier.size()) {
        g_placement_tier.resize((size_t)tier + 1);
    }
    return g_placement_tier[(size_t)tier];
}

/* Forward-declared in ds4_sycl.cpp and called from ds4_gpu_cleanup while
 * every tier's queue is still alive, mirroring sycl_stream_teardown_all's
 * ordering contract exactly: freeing this cache's slabs AFTER
 * g_devices.clear() destroys their queues would be a use-after-free of
 * the allocation context (spec 6g), so this must run BEFORE that clear,
 * not after. */
static void sycl_placement_teardown_all(void) {
    for (size_t t = 0; t < g_placement_tier.size(); t++) {
        sycl_placement_tier_cache &c = g_placement_tier[t];
        if (c.base && t < g_devices.size()) {
            try {
                ds4_sycl_queue((int)t).wait_and_throw();
                sycl::free(c.base, ds4_sycl_queue((int)t));
            } catch (const sycl::exception &e) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "placement cache teardown failed: %s\n", e.what());
            }
        }
        c.base = nullptr;
        c.bytes = 0;
        c.ranges.clear();
    }
}

/* The registered host model map: set only by ds4_gpu_register_model_map_
 * no_copy below, read only by ds4_gpu_device_cache_tensors. */
static const void *g_placement_host_base = nullptr;
static uint64_t    g_placement_host_size = 0;

/* Resolves a model-mmap range to the device copy ds4_gpu_device_cache_
 * tensors already installed on the current tier, or nullptr.
 *
 * This cache and ds4_gpu_cache_model_range's (sycl/ds4_sycl_model_cache.hpp)
 * are separate stores filled by separate entry points, and only the other
 * one used to be consulted when the MoE path resolved a weight pointer.
 * On a multi-GPU run that meant every lookup missed and every weight was
 * re-staged from the host mmap per call, per layer, per token, with the
 * already-resident copy sitting unused in VRAM the whole time.
 *
 * Ranges are offsets into the registered model map, so a caller asking
 * about any other host pointer cannot match by construction.  The
 * returned pointer is owned by this cache and freed with the tier's slab;
 * callers must treat it as borrowed (sycl_stage_host_bytes does: it
 * passes it through with owns=false). */
static const char *sycl_placement_cache_resolve(const void *model_map,
                                                uint64_t offset, uint64_t bytes) {
    if (!model_map || bytes == 0 || model_map != g_placement_host_base) return nullptr;
    const int tier = ds4_sycl_current_tier();
    if (tier < 0 || (size_t)tier >= g_placement_tier.size()) return nullptr;
    const uint64_t end = offset + bytes; /* caller already overflow-checked */
    const sycl_placement_tier_cache &c = g_placement_tier[(size_t)tier];
    for (const sycl_placement_cache_range &r : c.ranges) {
        if (offset >= r.source_offset && end >= offset &&
            end <= r.source_offset + r.bytes) {
            return (const char *)(r.device_ptr + (offset - r.source_offset));
        }
    }
    return nullptr;
}

/* uint64_t return, no failure-value convention to match (there is no
 * "invalid" bit pattern in a byte count): 0 doubles as "unknown/tier out
 * of range" and "no headroom left", exactly as CUDA's own implementation
 * folds a cudaSetDevice failure and a genuinely full device into the same
 * 0 (ds4_cuda.cu:1245-1253). See this file's header comment for why the
 * number returned is a self-tracked accounting figure, not a live Sysman
 * reading. */
extern "C" uint64_t ds4_gpu_tier_free_vram(int logical_tier) {
    if (logical_tier < 0 || (size_t)logical_tier >= g_devices.size()) return 0;

    const uint64_t ceiling = sycl_stream_vram_ceiling(ds4_sycl_queue(logical_tier));

    uint64_t committed = 0;
    if ((size_t)logical_tier < g_placement_tier.size()) {
        committed += g_placement_tier[(size_t)logical_tier].bytes;
    }
    if ((size_t)logical_tier < g_sycl_stream_tier.size()) {
        committed += g_sycl_stream_tier[(size_t)logical_tier].resident_bytes;
    }
    /* The device-resident model-weight cache
     * (sycl/ds4_sycl_model_cache.hpp) draws from this same ledger: without
     * this, that cache and this one could each believe they have the
     * tier's full headroom and together over-commit it. */
    committed += sycl_model_cache_committed_bytes(logical_tier);
    return ceiling > committed ? ceiling - committed : 0;
}

/* 1 on success, 0 on error, matching ds4_gpu_mgpu.h:214's documented
 * polarity and confirmed against CUDA's own implementation
 * (ds4_cuda.cu:3797, "Returns 1 on success, 0 on error"). No device-side
 * copy: SYCL kernels never dereference the host mmap directly (spec 6l),
 * so unlike CUDA's cudaHostRegister this entry has nothing to pin --
 * ds4_gpu_device_cache_tensors below stages every byte through an
 * explicit device-side slab instead. Re-registering a DIFFERENT model map
 * (a different pointer or size) releases every tier's existing selective
 * cache first: its entries are keyed by source_offset into the OLD host
 * mapping and would silently resolve to the wrong bytes against a new
 * one, mirroring CUDA's own teardown-on-re-register
 * (ds4_cuda.cu:3711-3715). */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map,
                                                   uint64_t    model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_placement_host_base == model_map && g_placement_host_size == model_size) {
        return 1;
    }

    /* Same release path ds4_gpu_cleanup uses (queue-drained free, before
     * any queue could be torn down), reused here rather than duplicated:
     * a re-register is exactly a targeted teardown of every tier's stale
     * cache, not a process-wide one. */
    sycl_placement_teardown_all();

    g_placement_host_base = model_map;
    g_placement_host_size = model_size;
    return 1;
}

/* 0 on success, a distinct positive value on error (see this backend's own
 * comment in ds4_sycl_unavailable.cpp, cross-checked against
 * ds4_cuda.cu:3944-4095 and ds4.c:56888/57069's "if (rc != 0) ... failed").
 * Error codes mirror CUDA's numbering where the same failure exists here:
 *   1 bad device_id/tier, 2 bad ranges argument, 3 no model map
 *   registered, 5 allocation or headroom failure, 7 a staging copy threw,
 *   8/9 a range falls outside the registered host mapping, 10 byte-count
 *   overflow. CUDA's 4 (cudaSetDevice failure) and 6 (growth d2d copy
 *   failure reported separately from a general copy failure) have no
 *   distinct SYCL equivalent: tier validity is folded into code 1, and a
 *   failed growth copy is folded into code 7 along with every other
 *   queue-op failure, since both are try/catch around the same submitted
 *   operations here. */
extern "C" int ds4_gpu_device_cache_tensors(int                    device_id,
                                            const ds4_tensor_range *ranges,
                                            int                     n_ranges) {
    /* device_id, and every ranges[i].target_device, are PHYSICAL device
     * ids -- the values the operator passed in --gpu-devices -- not
     * logical tiers. ds4.c:57024 passes g_gpu[d].device_id and
     * ds4.c:57197 stamps that same value into target_device, and CUDA
     * reads both the same way: it validates device_id against
     * DS4_MAX_GPUS rather than g_n_gpus, indexes g_dev_cache by it and
     * hands it straight to cudaSetDevice (ds4_cuda.cu:3947, :3973,
     * :3977).
     *
     * Nothing inside this backend is addressable by a physical id, so
     * translate once here and keep every index below a tier. Treating the
     * two as interchangeable was only ever correct for a device list
     * contiguous from zero; with --gpu-devices 0,2,4,6,1,3,5,7 -- the
     * ordering --cuda-tensor-parallel wants, since it pairs tier i with
     * tier i + n_gpus/2 -- it selected the wrong queue and the wrong
     * cache slab, staging each tier's weights onto another tier's card. */
    const int tier = sycl_tier_for_device_id(device_id);
    if (tier < 0 || (size_t)tier >= g_devices.size()) return 1;
    if (n_ranges < 0 || (!ranges && n_ranges > 0)) return 2;
    if (n_ranges == 0) return 0;
    if (!g_placement_host_base || g_placement_host_size == 0) return 3;

    /* Validate every range before touching the allocator, so a bad input
     * cannot partially grow the slab (same ordering as ds4_cuda.cu). */
    uint64_t want_bytes = 0;
    for (int i = 0; i < n_ranges; i++) {
        /* Physical against physical, matching ds4_cuda.cu:3959. */
        if (ranges[i].target_device != device_id) continue;
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb  = ranges[i].bytes;
        if (nb == 0) continue;
        if (off > g_placement_host_size) return 8;
        if (nb > g_placement_host_size - off) return 9;
        if (!sycl_u64_add_checked(want_bytes, nb, &want_bytes)) return 10;
    }
    if (want_bytes == 0) return 0;

    const uint64_t free_before = ds4_gpu_tier_free_vram(tier);
    if (want_bytes > free_before) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "device cache slab needs %.2f GiB on tier %d (device %d) but "
                "only %.2f GiB free; lower --gpu-vram or use --gpu-vram auto "
                "on a host with more headroom\n",
                (double)want_bytes / 1073741824.0, tier, device_id,
                (double)free_before / 1073741824.0);
        return 5;
    }

    sycl_placement_tier_cache &c = sycl_placement_cache_for(tier);
    sycl::queue               &q = ds4_sycl_queue(tier);
    const uint64_t new_bytes = c.bytes + want_bytes;

    unsigned char *new_base = sycl::malloc_device<unsigned char>((size_t)new_bytes, q);
    if (!new_base) return 5;

    try {
        const uint64_t old_bytes = c.bytes;
        if (c.base && old_bytes > 0) {
            sycl::event _ds4_prof_ev144 = q.memcpy(new_base, c.base, (size_t)old_bytes);
            _ds4_prof_ev144.wait_and_throw();
            ds4_sycl_profile_record(_ds4_prof_ev144);
            for (sycl_placement_cache_range &r : c.ranges) {
                r.device_ptr = new_base + (r.device_ptr - c.base);
            }
            sycl::free(c.base, q);
        }
        c.base = new_base;

        const unsigned char *host_base = (const unsigned char *)g_placement_host_base;
        uint64_t write_off = old_bytes;
        for (int i = 0; i < n_ranges; i++) {
            if (ranges[i].target_device != device_id || ranges[i].bytes == 0) continue;
            unsigned char *dev_ptr = new_base + write_off;
            /* host_base is the registered model mmap (ds4_gpu_register_
             * model_map_no_copy): see sycl_copy_host_to_device_paged_safe's
             * comment (ds4_sycl_common.hpp) for why a bare queue.memcpy
             * from it is unsafe on this stack and must not be reused here
             * even though this cache predates the model-range cache. */
            sycl_copy_host_to_device_paged_safe(q, dev_ptr,
                                                host_base + ranges[i].source_offset,
                                                ranges[i].bytes);
            c.ranges.push_back({ranges[i].source_offset, ranges[i].bytes, dev_ptr});
            write_off += ranges[i].bytes;
        }
        c.bytes = new_bytes;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "device cache install failed: %s\n", e.what());
        return 7;
    }

    std::sort(c.ranges.begin(), c.ranges.end(),
              [](const sycl_placement_cache_range &a,
                 const sycl_placement_cache_range &b) {
                  return a.source_offset < b.source_offset;
              });
    return 0;
}

/* Test-only: total bytes committed to tier's selective cache. Not part of
 * the ABI. */
extern "C" uint64_t ds4_sycl_test_placement_cache_bytes(int tier) {
    if (tier < 0 || (size_t)tier >= g_placement_tier.size()) return 0;
    return g_placement_tier[(size_t)tier].bytes;
}

/* Test-only: copies `bytes` back from the cached entry covering
 * [source_offset, source_offset + bytes) on `tier` into `out`, so a real
 * device round trip can be verified byte-for-byte. Returns 1 on a
 * covering hit, 0 on a miss or bad argument. Not part of the ABI --
 * mirrors ds4_sycl_test_zes_free_bytes (sycl/ds4_sycl_mgpu.hpp) and the
 * streaming module's own test read hooks. */
extern "C" int ds4_sycl_test_placement_read_back(int tier, uint64_t source_offset,
                                                 uint64_t bytes, void *out) {
    if (!out || bytes == 0) return 0;
    if (tier < 0 || (size_t)tier >= g_placement_tier.size()) return 0;
    const sycl_placement_tier_cache &c = g_placement_tier[(size_t)tier];
    for (const sycl_placement_cache_range &r : c.ranges) {
        if (source_offset < r.source_offset) continue;
        const uint64_t into = source_offset - r.source_offset;
        if (into > r.bytes || bytes > r.bytes - into) continue;
        try {
            ds4_sycl_queue(tier)
                .memcpy(out, r.device_ptr + into, (size_t)bytes)
                .wait_and_throw();
            return 1;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "test placement read-back failed: %s\n", e.what());
            return 0;
        }
    }
    return 0;
}

/* Prefill-pipeline pair: a plain engine-wide flag, matching CUDA's own
 * g_q8_cache_suppressed / ds4_gpu_q8_cache_suppressed / _set_
 * (ds4_cuda.cu:425, :3638-3644) with no int-return-polarity ambiguity --
 * ds4_gpu_q8_cache_suppressed returns the flag's own value, not a
 * success/failure code, and ds4_gpu_set_q8_cache_suppressed is void.
 * Reachability (ds4-sycl-stub-reachability-v2.md, "Multi-tier placement
 * only") is gated on ds4.c:34212's metal_graph_build_prefill_stages(...)
 * && n_stages >= 2, which a single GPU never produces, so this flag has
 * no observable effect on this backend yet; it becomes load-bearing only
 * once multi-tier decode pipelining is wired up. */
static int g_sycl_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_sycl_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_sycl_q8_cache_suppressed = suppressed ? 1 : 0;
}
