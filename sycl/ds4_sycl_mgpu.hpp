#pragma once

/* Multi-GPU plumbing: shared-context device/queue construction (built in
 * ds4_sycl.cpp; see ds4_sycl_enumerate_gpus / ds4_sycl_build_devices
 * there), per-device budget bookkeeping, the peer-access validation
 * matrix, the real tier switch, per-tier allocation, and the
 * ds4_gpu_tensor_copy_xdev family. Implements ds4_gpu_mgpu.h.
 *
 * No ROCm structural reference exists for the shared-context or
 * peer-validation design: ROCm refuses any n_gpus != 1
 * (ds4_rocm_compat.cu:26-34) and never drives a second device, so
 * ds4_cuda.cu:2578-2760 is the only implementation of real multi-GPU
 * behaviour anywhere in this project, cited here as the documented
 * exception to "ROCm, not CUDA" (spec 6e). ROCm's own single-device shims
 * for the surface this header covers (rocm_tier_valid gating each entry to
 * tier 0 with g_n_gpus == 1) remain the model for return-value polarity
 * and argument validation, which do generalise. */

#include "ds4_gpu_args.h"
#include "ds4_sycl_common.hpp"

/* ds4_gpu_config, ds4_gpu_ctx, DS4_MAX_GPUS, g_gpu, g_n_gpus and
 * g_gpu_peer_ok all come from ds4_gpu_mgpu.h, included by ds4_sycl.cpp
 * before this header. ds4_gpu_args.h is included so the compiler checks
 * ds4_gpu_args_probe_auto_cuda's definition below against its one
 * canonical declaration rather than relying on extern "C" name matching
 * alone. */

/* The peer-access validation protocol, ported whole from
 * ds4_cuda.cu:2630-2760, not just its capability-query steps: the comment
 * there (:2629-2645) records that on RTX 6000 Ada, cudaDeviceCanAccessPeer
 * and cudaDeviceEnablePeerAccess both reported success while
 * cudaMemcpyPeer silently delivered wrong data, non-deterministically, at
 * realistic sizes. The failure being guarded against is a driver/hardware
 * property, not a CUDA-specific one, and Intel's own documentation
 * describes cross-card SYCL access as "slow... through host memory" even
 * where the extension reports support (spec finding 7's B60 Dual: two
 * dies behind a PCIe switch is exactly the suspect topology), so the same
 * byte-exact round trip is required here, not a translation of the
 * capability query alone. */

static const size_t SYCL_PEER_BYTECHECK_SIZES[] = {
    4u * 1024u,
    256u * 1024u,
    1u * 1024u * 1024u,
    16u * 1024u * 1024u,
};
static const int    SYCL_PEER_BYTECHECK_N_SIZES = 4;
static const int    SYCL_PEER_BYTECHECK_ITERS    = 4;
static const size_t SYCL_PEER_BYTECHECK_MAX_BYTES = 16u * 1024u * 1024u;

/* The byte-exact round trip itself: write a distinguishable per-iteration
 * pattern to a host buffer, copy host to source device, source device to
 * destination device (the actual peer-copy attempt, submitted on the
 * source's queue against a pointer that lives on the destination's
 * device), destination device back to host, and compare. Every one of
 * `SYCL_PEER_BYTECHECK_N_SIZES * SYCL_PEER_BYTECHECK_ITERS` rounds must be
 * byte-perfect.
 *
 * `corrupt`, `inject_compare_bug` and `vary_pattern` are test-only knobs
 * (see ds4_sycl_test_peer_bytecheck below); the one production call site
 * (sycl_validate_peer_pair) always passes (false, false, true).
 *
 * With `corrupt` set, one byte of the destination-side readback is
 * clobbered on the very first round before the comparison runs, so the
 * comparison MUST report a mismatch; if it does not, the comparison is not
 * wired to the data it claims to check.
 *
 * With `inject_compare_bug` set, a round is compared against the PREVIOUS
 * iteration's expected pattern instead of its own once one exists,
 * simulating a stale-comparison-window defect: real pattern variation
 * (vary_pattern=true) makes that defect visible as a mismatch, while a
 * constant pattern across iterations (vary_pattern=false) makes the
 * previous iteration's expected value equal the current one, hiding it.
 * This is the mechanism spec 6i and the "same pattern for every
 * iteration" ablation are both about: a comparison bug that only a varying
 * pattern can expose. */
static bool sycl_peer_bytecheck(sycl::queue &qi, void *dev_src, sycl::queue &qj,
                                 void *dev_dst, size_t *out_fail_bytes,
                                 int *out_fail_iter, bool corrupt,
                                 bool inject_compare_bug, bool vary_pattern) {
    std::vector<unsigned char> host_src(SYCL_PEER_BYTECHECK_MAX_BYTES);
    std::vector<unsigned char> host_dst(SYCL_PEER_BYTECHECK_MAX_BYTES);
    std::vector<unsigned char> prev_expected(SYCL_PEER_BYTECHECK_MAX_BYTES);
    try {
        for (int s_idx = 0; s_idx < SYCL_PEER_BYTECHECK_N_SIZES; s_idx++) {
            const size_t n = SYCL_PEER_BYTECHECK_SIZES[s_idx];
            /* Scoped to one size: comparing against a previous round's
             * buffer only makes sense within a fixed byte count. Reset at
             * every new size so a smaller earlier round's leftover bytes
             * never leak into a larger round's comparison. */
            bool have_prev = false;
            for (int it = 0; it < SYCL_PEER_BYTECHECK_ITERS; it++) {
                const size_t it_term = vary_pattern ? (size_t)it * 17u : 0u;
                const size_t s_term  = vary_pattern ? (size_t)s_idx * 53u : 0u;
                for (size_t k = 0; k < n; k++) {
                    host_src[k] =
                        (unsigned char)((k * 31u + it_term + s_term + 11u) & 0xffu);
                }
                qi.memcpy(dev_src, host_src.data(), n).wait_and_throw();
                qi.memcpy(dev_dst, dev_src, n).wait_and_throw();
                qj.memcpy(host_dst.data(), dev_dst, n).wait_and_throw();
                if (corrupt && s_idx == 0 && it == 0) host_dst[0] ^= 0xFFu;

                const unsigned char *expected = host_src.data();
                if (inject_compare_bug && have_prev) expected = prev_expected.data();
                if (memcmp(expected, host_dst.data(), n) != 0) {
                    if (out_fail_bytes) *out_fail_bytes = n;
                    if (out_fail_iter) *out_fail_iter = it;
                    return false;
                }
                memcpy(prev_expected.data(), host_src.data(), n);
                have_prev = true;
            }
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "peer bytecheck threw: %s\n", e.what());
        return false;
    }
    return true;
}

/* Test-only: exercises the shared bytecheck loop directly against a single
 * real device used as both legs of the "pair", since this box has no
 * second device to validate real cross-die behaviour against. This proves
 * the probe's own mechanics (pattern generation, the multi-size/iteration
 * loop, and the comparison) are sound; it cannot and does not claim
 * anything about real PCIe peer correctness, which is what
 * sycl_validate_peer_pair below exists to determine and which requires a
 * second physical device to ever run for real. Not part of the ABI (not
 * declared in ds4_gpu.h or ds4_gpu_mgpu.h). */
extern "C" int ds4_sycl_test_peer_bytecheck(int tier, int corrupt,
                                             int inject_compare_bug, int vary_pattern) {
    if (tier < 0 || (size_t)tier >= g_devices.size()) return 0;
    sycl::queue &q = g_devices[(size_t)tier].queue;
    void *dev_src = sycl::malloc_device(SYCL_PEER_BYTECHECK_MAX_BYTES, q);
    void *dev_dst = sycl::malloc_device(SYCL_PEER_BYTECHECK_MAX_BYTES, q);
    if (!dev_src || !dev_dst) {
        if (dev_src) sycl::free(dev_src, q);
        if (dev_dst) sycl::free(dev_dst, q);
        return 0;
    }
    size_t fail_bytes = 0;
    int fail_iter = -1;
    bool ok = sycl_peer_bytecheck(q, dev_src, q, dev_dst, &fail_bytes, &fail_iter,
                                   corrupt != 0, inject_compare_bug != 0,
                                   vary_pattern != 0);
    sycl::free(dev_src, q);
    sycl::free(dev_dst, q);
    return ok ? 1 : 0;
}

/* Runs the full protocol for one ordered device pair: capability query,
 * enable, then (regardless of what either reported) the byte-exact round
 * trip. Only an all-perfect round trip marks the pair usable; any
 * exception or any single mismatch anywhere disables it and every
 * cross-device copy for that pair keeps using the host bounce
 * automatically (ds4_gpu_tensor_copy_xdev above already checks
 * g_gpu_peer_ok). `ext_oneapi_enable_peer_access` returns void per the
 * extension's declared signature, so unlike CUDA's checked return value an
 * exception is the only failure signal here; it is wrapped accordingly. */
static bool sycl_validate_peer_pair(int i, int j) {
    sycl::device &dev_i = g_devices[(size_t)i].dev;
    sycl::device &dev_j = g_devices[(size_t)j].dev;

    bool can = false;
    try {
        can = dev_i.ext_oneapi_can_access_peer(
            dev_j, sycl::ext::oneapi::peer_access::access_supported);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "peer access %d->%d: can_access_peer threw: %s\n", i, j, e.what());
        return false;
    }
    if (!can) return false;

    try {
        dev_i.ext_oneapi_enable_peer_access(dev_j);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "peer access %d->%d: enable_peer_access threw: %s\n", i, j, e.what());
        return false;
    }

    sycl::queue &qi = g_devices[(size_t)i].queue;
    sycl::queue &qj = g_devices[(size_t)j].queue;
    void *dev_src = nullptr;
    void *dev_dst = nullptr;
    bool   validated  = false;
    size_t fail_bytes = 0;
    int    fail_iter  = -1;
    try {
        dev_src = sycl::malloc_device(SYCL_PEER_BYTECHECK_MAX_BYTES, qi);
        dev_dst = sycl::malloc_device(SYCL_PEER_BYTECHECK_MAX_BYTES, qj);
        if (!dev_src || !dev_dst) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "peer access %d->%d: probe buffer allocation failed\n", i, j);
        } else {
            validated = sycl_peer_bytecheck(qi, dev_src, qj, dev_dst, &fail_bytes,
                                             &fail_iter, false, false, true);
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "peer access %d->%d probe threw: %s\n",
                i, j, e.what());
        validated = false;
    }
    if (dev_src) { try { sycl::free(dev_src, qi); } catch (const sycl::exception &) {} }
    if (dev_dst) { try { sycl::free(dev_dst, qj); } catch (const sycl::exception &) {} }

    if (validated) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "peer access %d->%d validated across %d sizes x %d iterations "
                "(max %zu MiB)\n",
                i, j, SYCL_PEER_BYTECHECK_N_SIZES, SYCL_PEER_BYTECHECK_ITERS,
                SYCL_PEER_BYTECHECK_MAX_BYTES / (1024u * 1024u));
    } else {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "peer access %d->%d FAILED validation at size=%zu iter=%d; "
                "falling back to pinned-host bounce\n",
                i, j, fail_bytes, fail_iter);
    }
    return validated;
}

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) return 0;
    if (g_initialised) ds4_gpu_cleanup();

    /* Must run before the first Level Zero enumeration, which
     * ds4_sycl_enumerate_gpus immediately below performs: Sysman cannot be
     * armed retroactively on an already-initialised loader. Overwrite is 0
     * so an operator's own exported value is respected. This is the single
     * init path now (ds4_gpu_init is a shim over it), but
     * ds4_gpu_args_probe_auto_cuda below can still reach Sysman first when
     * CLI parsing runs before any init, and sets the same variable. */
    setenv("ZES_ENABLE_SYSMAN", "1", 0);

    try {
        const std::vector<sycl::device> all = ds4_sycl_enumerate_gpus();
        std::vector<sycl::device> wanted;
        for (int i = 0; i < cfg->n_gpus; i++) {
            const int idx = cfg->device_indices[i];
            if (idx < 0 || (size_t)idx >= all.size()) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "init_multi: device index %d out of range (%zu "
                        "enumerated)\n",
                        idx, all.size());
                return 0;
            }
            wanted.push_back(all[(size_t)idx]);
        }

        /* ds4_sycl_build_devices always produces one queue per requested
         * device (it groups by platform rather than dropping devices), so
         * g_devices.size() == wanted.size() on return here. */
        ds4_sycl_build_devices(wanted, &g_devices);

        /* Before anything else touches these devices, and before the peer
         * probes below spend time on them: refuse any card that cannot
         * honour a required sub-group width.  Defined in ds4_sycl.cpp,
         * which includes this header after it. */
        if (!sycl_verify_subgroup_widths()) return 0;

        g_n_gpus = cfg->n_gpus;
        for (int i = 0; i < g_n_gpus; i++) {
            g_gpu[i].device_id    = cfg->device_indices[i];
            g_gpu[i].budget_bytes = cfg->vram_bytes[i];
            g_gpu[i].used_bytes   = 0;
        }

        /* NxN peer-access matrix: the diagonal is trivially true (a device
         * always reaches its own memory), every other ordered pair runs
         * the full validation protocol (sycl_validate_peer_pair, defined
         * above this function) and defaults to 0 (bounce-only) on any
         * failure.  The i != j branch needs a second physical device to
         * run for real, and now gets one: 0->1 and 1->0 both validate on a
         * 14-device Arc Pro B60 host. */
        for (int i = 0; i < g_n_gpus; i++) {
            for (int j = 0; j < g_n_gpus; j++) {
                g_gpu_peer_ok[i][j] = (i == j) ? 1 : (sycl_validate_peer_pair(i, j) ? 1 : 0);
            }
        }

        g_current_tier = 0;
        g_initialised  = true;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "%d device(s) on a shared context per SYCL platform, "
                "using %s\n",
                g_n_gpus,
                g_devices[0].dev.get_info<sycl::info::device::name>().c_str());
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "init_multi failed: %s\n", e.what());
        g_devices.clear();
        g_n_gpus       = 0;
        g_current_tier = 0;
        g_initialised  = false;
        return 0;
    }
}

/* Maps a PHYSICAL device id -- the value the operator passed in
 * --gpu-devices, which ds4_gpu_init_multi above stores in
 * g_gpu[tier].device_id -- to the LOGICAL tier that owns it, or -1 when
 * no configured tier does. The two coincide only when the device list is
 * contiguous from zero, so every ABI entry that CUDA defines in terms of
 * a physical id must come through here before touching anything
 * tier-indexed. ds4_gpu_device_cache_tensors (ds4_cuda.cu:3944) is the
 * only such entry this backend implements; ds4_gpu_lookup_cache_strict's
 * expected_device (ds4_gpu_mgpu.h:216-228) is the same convention and is
 * cited so a future implementation of it starts here rather than
 * rediscovering the mismatch.
 *
 * CUDA needs no such helper because a physical id is directly usable
 * there: it hands device_id straight to cudaSetDevice and indexes its
 * per-device slabs by it (g_dev_cache[device_id], ds4_cuda.cu:3973,
 * :3977). SYCL has no physical device-id space at all -- ds4_sycl_queue
 * takes a tier -- so the translation has to happen somewhere, and doing
 * it once at each boundary keeps every index inside this backend a tier.
 *
 * Pure lookup over the configured tier table: it answers "which tier was
 * this card assigned to", never "is that tier usable". Callers bounds-
 * check the returned tier against g_devices themselves, the same
 * separation ds4_gpu_set_current_device_fenced's own reverse scan uses
 * on CUDA (ds4_cuda.cu:3897-3899). */
static int sycl_tier_for_device_id(int device_id) {
    if (device_id < 0) return -1;
    const int n_tiers = g_n_gpus < DS4_MAX_GPUS ? g_n_gpus : DS4_MAX_GPUS;
    for (int t = 0; t < n_tiers; t++) {
        if (g_gpu[t].device_id == device_id) return t;
    }
    return -1;
}

/* Test-only: exposes the physical-to-logical translation directly so a
 * non-contiguous --gpu-devices ordering can be exercised on a box with
 * fewer devices than the ordering names. Not part of the ABI (not
 * declared in ds4_gpu.h or ds4_gpu_mgpu.h). */
extern "C" int ds4_sycl_test_tier_for_device_id(int device_id) {
    return sycl_tier_for_device_id(device_id);
}

/* Real tier switch: 0 means success (the minority convention; see spec 3a
 * and ds4_gpu_mgpu.h:199-200). An out-of-range tier is rejected and
 * g_current_tier is left untouched, matching CUDA's own
 * ds4_gpu_set_current_device (ds4_cuda.cu:3870-3879), which never mutates
 * its cached device on failure either. */
extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    if (logical_tier < 0 || (size_t)logical_tier >= g_devices.size()) return 1;
    g_current_tier = logical_tier;
    return 0;
}

/* Fenced switch: orders every future submission on the destination tier's
 * queue after everything already submitted to the previous tier's queue,
 * the SYCL analogue of CUDA's fenced switch
 * (ds4_cuda.cu:3888-3927, event record + stream-wait-event). A no-op
 * switch (same tier) still reports success without submitting a barrier,
 * matching CUDA's own early-return shape. */
extern "C" int ds4_gpu_set_current_device_fenced(int logical_tier) {
    if (logical_tier < 0 || (size_t)logical_tier >= g_devices.size()) return 1;
    if (logical_tier == g_current_tier) return 0;
    try {
        sycl::event fence = ds4_sycl_queue(g_current_tier).ext_oneapi_submit_barrier();
        ds4_sycl_queue(logical_tier).ext_oneapi_submit_barrier({fence});
        g_current_tier = logical_tier;
        return 0;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "set_current_device_fenced failed: %s\n",
                e.what());
        return 1;
    }
}

/* Caller-supplied struct alloc on a specific logical tier: 0 means
 * success (documented at ds4_gpu_mgpu.h:107-108 and confirmed against both
 * ROCm and CUDA by the research pass), 1/2/3 distinct errors, matching
 * ROCm's own error-code shape (ds4_rocm_compat.cu:45-56). */
extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes) {
    if (!t) return 1;
    if (device_id < 0 || (size_t)device_id >= g_devices.size()) return 2;
    if (bytes == 0) bytes = 1;
    try {
        sycl::queue &q = ds4_sycl_queue(device_id);
        void *ptr = sycl::malloc_device(bytes, q);
        if (ptr == nullptr) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tensor_alloc_on: malloc_device of %llu bytes failed\n",
                    (unsigned long long)bytes);
            return 3;
        }
        t->ptr       = ptr;
        t->bytes     = bytes;
        t->owner     = 1;
        t->device_id = device_id;
        g_gpu[device_id].used_bytes += bytes;
        return 0;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_on failed: %s\n", e.what());
        return 3;
    }
}

/* Pairs with ds4_gpu_tensor_alloc_on. void return, matching both ROCm
 * (ds4_rocm_compat.cu:69-72, zeroes the whole struct after freeing) and
 * CUDA (ds4_cuda.cu:2943-2952). Follows the same after-cleanup ordering
 * contract as ds4_gpu_tensor_free (ds4_sycl.cpp): a tensor freed once
 * g_devices is empty has no live queue to free through, so this logs and
 * leaks the device pointer instead of indexing an empty vector. */
extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->owner != 0 && t->ptr != nullptr) {
        if (g_devices.empty()) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tensor_free_in_place after cleanup; leaking device pointer\n");
        } else {
            try {
                sycl::free(t->ptr,
                           ds4_sycl_queue(t->device_id >= 0 ? t->device_id
                                                             : g_current_tier));
            } catch (const sycl::exception &e) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_free_in_place failed: %s\n",
                        e.what());
            }
        }
    }
    memset(t, 0, sizeof(*t));
}

/* Cross-device copy: host-bounce only. The direct peer path (guarded by a
 * validated g_gpu_peer_ok[sd][dd]) arrives with the byte-validation
 * protocol; see sycl_validate_peer_pair further down. Nonzero means
 * success, the majority convention (ds4_gpu_mgpu.h:129-137, confirmed at
 * ds4.c:15601 in metal_graph_set_active_tier_decode:
 * `if (!ds4_gpu_tensor_copy_xdev(dst, src, hc_bytes)) return false;`).
 *
 * g_sycl_xdev_bounce_calls is test-only instrumentation (see
 * ds4_sycl_test_xdev_bounce_calls below): this machine has one GPU, so
 * every copy_xdev call in the test suite has sd == dd and this function is
 * never reached; the counter lets the test prove that rather than assert
 * it. */
static int g_sycl_xdev_bounce_calls = 0;

static int sycl_xdev_host_bounce(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                                  uint64_t bytes, int sd, int dd) {
    g_sycl_xdev_bounce_calls++;
    sycl::queue &sq = ds4_sycl_queue(sd);
    void *host = sycl::malloc_host(bytes, sq);
    if (host == nullptr) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "tensor_copy_xdev: host bounce malloc_host of %llu bytes failed\n",
                (unsigned long long)bytes);
        return 0;
    }
    int ok = 1;
    try {
        sycl::event _ds4_prof_ev75 = sq.memcpy(host, src->ptr, bytes);
        sycl_batch_wait(_ds4_prof_ev75);
        ds4_sycl_profile_record(_ds4_prof_ev75);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_copy_xdev: bounce d2h failed: %s\n",
                e.what());
        ok = 0;
    }
    if (ok) {
        try {
            sycl_batch_wait(ds4_sycl_queue(dd).memcpy(dst->ptr, host, bytes));
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_copy_xdev: bounce h2d failed: %s\n",
                    e.what());
            ok = 0;
        }
    }
    sycl::free(host, sq);
    return ok;
}

/* Test-only: lets the test binary confirm the bounce path never ran on a
 * single-device box, rather than merely assert it. Not part of the ABI
 * (not declared in ds4_gpu.h or ds4_gpu_mgpu.h). */
extern "C" int ds4_sycl_test_xdev_bounce_calls(void) { return g_sycl_xdev_bounce_calls; }

/* A validated pair shares one sycl::context (ds4_sycl_build_devices groups
 * queues by platform, and ext_oneapi_can_access_peer only ever reports
 * true within one backend per the peer-access extension, so a validated
 * pair is always same-platform), so a plain queue::memcpy between their
 * USM pointers is exactly the SYCL analogue of CUDA's cudaMemcpyPeerAsync:
 * legal because both pointers share a context, submitted on the
 * destination's queue for destination-side ordering. Honors
 * DS4_FORCE_HOST_BOUNCE=1 per ds4_gpu_mgpu.h:131. Any exception here falls
 * back to the host bounce rather than propagating: a pair that validated
 * at init can still be worth a fallback rather than a hard failure.
 *
 * The peer copy does not block the host, and that is the whole point of
 * this entry rather than an optimisation on top of it. Expert parallelism
 * exchanges a partial block output with the paired tier on every routed
 * layer, so a decode token performs on the order of two of these per
 * layer; blocking the host on each one cost more than splitting the
 * experts saved, and measured slower than not splitting them at all.
 *
 * Correctness comes from two barriers rather than a wait, which is what
 * CUDA's cudaMemcpyPeerAsync plus its per-tier boundary event does
 * (ds4_cuda.cu:3350-3367). The first orders the copy after everything the
 * SOURCE tier has already submitted, since the copy runs on the
 * destination queue and in_order says nothing across queues. The second
 * puts the source queue behind the copy, so a later write to src cannot
 * overtake the read this copy has not performed yet. Destination-side
 * ordering needs nothing extra: the copy is submitted on the destination
 * queue, which is in_order. DS4_SYCL_SYNC_XDEV=1 restores the blocking
 * form, for isolating this entry when a multi-GPU run misbehaves. */
extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                                        uint64_t bytes) {
    if (!dst || !src) return 0;
    if (bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    const int sd = src->device_id >= 0 ? src->device_id : g_current_tier;
    const int dd = dst->device_id >= 0 ? dst->device_id : g_current_tier;
    if (sd == dd) return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);

    const bool peer_ok = sd >= 0 && dd >= 0 && sd < DS4_MAX_GPUS && dd < DS4_MAX_GPUS &&
                         g_gpu_peer_ok[sd][dd] != 0;
    if (peer_ok && getenv("DS4_FORCE_HOST_BOUNCE") == nullptr) {
        try {
            sycl::queue &sq = ds4_sycl_queue(sd);
            sycl::queue &dq = ds4_sycl_queue(dd);
            if (getenv("DS4_SYCL_SYNC_XDEV") != nullptr) {
                sycl_batch_wait(dq.memcpy(dst->ptr, src->ptr, bytes));
                return 1;
            }
            sycl::event src_ready = sq.ext_oneapi_submit_barrier();
            sycl::event copied = dq.submit([&](sycl::handler &h) {
                h.depends_on(src_ready);
                h.memcpy(dst->ptr, src->ptr, bytes);
            });
            sq.ext_oneapi_submit_barrier({copied});
            return 1;
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tensor_copy_xdev: validated peer copy %d->%d failed, falling "
                    "back to host bounce: %s\n",
                    sd, dd, e.what());
        }
    }
    return sycl_xdev_host_bounce(dst, src, bytes, sd, dd);
}

/* SYCL has no default-stream concept distinct from a device's own queue,
 * so there is no separate "default stream" ordering domain for this entry
 * to distinguish from ds4_gpu_tensor_copy_xdev. Delegates, matching
 * ROCm's own shape (ds4_rocm_compat.cu:93-97), the structural reference
 * for the whole copy_xdev family. */
extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

/* The copy this delegates to is submitted on the destination tier's own
 * queue, which is in_order, so it is already ordered after everything
 * previously submitted there and this entry has nothing to add. That
 * remains true now that the copy no longer blocks the host: what orders
 * it was never the wait. Delegates for the same reason ROCm does
 * (ds4_rocm_compat.cu:99-103). */
extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

/* Per-pair delegation, matching ROCm's boolean-AND shape
 * (ds4_rocm_compat.cu:105-110). CUDA additionally batches these three
 * transfers into shared event plumbing when all three share one
 * source/destination tier pair (ds4_cuda.cu:3489-3583); that is a
 * throughput optimisation, real only once cross-device transfer overhead
 * exists to amortise, and out of scope for this host-bounce-first phase. */
extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

/* No copy: orders the destination tier's queue after the source tier's
 * already-submitted work, for peer-read-without-copy patterns. Genuinely
 * meaningful only once a validated peer pair lets the destination
 * dereference the source's memory directly; the ordering
 * primitive itself does not depend on that validation, so it is
 * implemented here rather than deferred. Nonzero means success, matching
 * both ROCm's boolean shape (ds4_rocm_compat.cu:124-128) and CUDA's
 * event-based one (ds4_cuda.cu:3590-3608). */
extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    if (!src) return 0;
    if (dst_tier < 0 || (size_t)dst_tier >= g_devices.size()) return 0;
    const int sd = src->device_id >= 0 ? src->device_id : g_current_tier;
    if (sd == dst_tier) return 1;
    try {
        sycl::event ready = ds4_sycl_queue(sd).ext_oneapi_submit_barrier();
        ds4_sycl_queue(dst_tier).ext_oneapi_submit_barrier({ready});
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_wait_xdev failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src, int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

/* Tensor parallelism: cross-rank combine, out = local + remote.
 * This is THE cross-rank exchange primitive: ds4.c's TP paths call it to
 * fold a peer's partial block output into this rank's own, reconstructing
 * the full-width result the two ranks jointly computed. Ported from
 * ds4_cuda.cu:18647-18683 exactly, including its device-ownership shape
 * (out and local must already live on the same device; remote may live on
 * either device, and a remote_tmp scratch tensor on out's device is
 * required only when it does not).
 *
 * ORDERING ACROSS DEVICES, spec 6g/6j: an in_order queue (ds4_sycl.cpp)
 * only orders commands within that one queue; it says
 * nothing about ordering between a command on tier A's queue and a
 * command on tier B's queue, which is the case that matters here (out
 * and remote_tmp can live on different tiers). What orders the two is
 * that ds4_gpu_tensor_copy_xdev submits its copy on the DESTINATION
 * tier's queue, and remote_tmp is required above to live on out's tier,
 * so the copy and the add kernel below are two commands on one in_order
 * queue and need no event, fence or wait between them.
 *
 * That is the ordering argument, and it is deliberately not "the copy
 * blocked before returning". It used to be: every branch of
 * ds4_gpu_tensor_copy_xdev ended in wait_and_throw, and this entry ended
 * in one of its own. Expert parallelism calls this primitive on every
 * routed layer of every token, so those host round trips were measured
 * costing more than splitting the experts saved. The waits are gone and
 * the ordering is unchanged, because the waits were never what provided
 * it.
 *
 * The reverse discipline failure -- reading remote_tmp before the copy
 * lands, or freeing it before the add kernel that reads it has finished --
 * is exactly spec 6g's use-after-free and 6j's too-small-to-race launch:
 * neither is exercisable on one GPU (there is no second device for
 * remote_tmp to ever be genuinely cross-device), so this entry's
 * same-device path (od == ld == rd) is the only one a single A770 can
 * verify; the cross-device branch is verified by inspection only. See
 * the cross-device verification notes for what to check first on the B60 machine.
 *
 * NONZERO means success, matching ds4_gpu.h and ds4_cuda.cu's own return
 * shape. n == 0 is a free success (`return 1`), matching ds4_cuda.cu:
 * 18656 exactly, unlike this file's own ds4_gpu_tensor_copy_xdev above
 * (bytes == 0 is also free success there) but UNLIKE ds4_gpu_add_tensor
 * in ds4_sycl_output.hpp (n == 0 there still runs the empty kernel and
 * falls through to `return 1` the same way, so behaviourally identical,
 * just structured as an explicit early return here to match ds4_cuda.cu's
 * own early return at :18656 line-for-line). */
extern "C" int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *local,
                                       const ds4_gpu_tensor *remote,
                                       ds4_gpu_tensor *remote_tmp, uint32_t n) {
    if (!sycl_tensor_has_f32(out, n) || !sycl_tensor_has_f32(local, n) ||
        !sycl_tensor_has_f32(remote, n)) {
        return 0;
    }
    if (n == 0u) return 1;
    if (g_devices.empty()) return 0;

    const int od = out->device_id >= 0 ? out->device_id : g_current_tier;
    const int ld = local->device_id >= 0 ? local->device_id : g_current_tier;
    const int rd = remote->device_id >= 0 ? remote->device_id : g_current_tier;
    if (od != ld) return 0;

    const ds4_gpu_tensor *rhs = remote;
    if (rd != od) {
        const int rtd = remote_tmp && remote_tmp->device_id >= 0
                                 ? remote_tmp->device_id
                                 : g_current_tier;
        if (!sycl_tensor_has_f32(remote_tmp, n) || rtd != od) return 0;
        if (!ds4_gpu_tensor_copy_xdev(remote_tmp, remote, (uint64_t)n * sizeof(float))) {
            return 0;
        }
        rhs = remote_tmp;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(od);
        float       *o  = (float *)out->ptr;
        const float *pl = (const float *)local->ptr;
        const float *pr = (const float *)rhs->ptr;
        sycl::event _ds4_prof_ev76 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            o[i] = pl[i] + pr[i];
        });
        /* No wait: q is in_order, so whatever reads `out` next is already
         * ordered behind this kernel, nothing here is staged scratch to
         * keep alive, and the host reads nothing back. */
        ds4_sycl_profile_record(_ds4_prof_ev76);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "add_xdev failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Tensor parallelism: the ds4_gpu_tp_* per-layer gate family.
 *
 * ds4_gpu.h documents this whole family as "Tensor-parallel per-layer
 * gates (Metal only)": the encoder closes the current Metal command
 * buffer, signals a shared event, queues the exchange on a service
 * thread, and waits for a CPU-signalled release before the combine
 * kernel runs. That mechanism is Metal command-buffer machinery with no
 * SYCL analogue, and it turns out CUDA never built one either: checked
 * directly against ds4_cuda.cu (not inferred from the Metal-only
 * docstring), every one of these six entries is ALREADY a stub there --
 * ds4_gpu_tp_gate_encode and ds4_gpu_tp_big_gate_encode literally print
 * "ds4: CUDA stub called" and return 0 (ds4_cuda.cu:30461-30471);
 * ds4_gpu_tp_big_gate_kick, ds4_gpu_tp_big_gate_wait,
 * ds4_gpu_tp_batch_gate_encode and ds4_gpu_tp_suspend_expert_sharding are
 * unconditional `(void)every_argument; return 0;` / no-op void
 * (ds4_cuda.cu:30804-30829). CUDA and ROCm multi-GPU machines never
 * exercise tensor parallelism at all: ds4_engine_tp_bind (ds4.c) refuses
 * to bind on any backend other than DS4_BACKEND_METAL
 * ("tensor parallelism requires the Metal backend", checked directly),
 * which is the ONLY place g->tp_world is ever set to 2
 * (ds4.c: `if (e->tp.active) { s->graph.tp_world = 2; ... }`). That gate
 * is unconditional on backend identity, not on GPU count, so g->tp_world
 * cannot reach 2 on this backend even with two real Intel GPUs present --
 * a backend-level gate, not a hardware limit, so it would hold even on a
 * machine with more GPUs than the one A770 this was tested on.
 *
 * Given that, e->backend == DS4_BACKEND_CUDA is true for a SYCL build
 * (spec's own standing rule), so the CUDA branch is authoritative, and
 * the CUDA branch's authoritative behaviour for all six of these entries
 * is "always fail / no-op". Faithfully mirroring that is the CORRECT
 * port, not a placeholder stub: every signature below is the real, fixed
 * ds4_gpu.h signature (not the unavailable file's variadic `...`), and
 * every return value is checked against ds4_cuda.cu's actual line rather
 * than picked from the macro default. NONZERO means success for every
 * int-returning entry here (ds4_gpu.h), so `return 0` reports failure
 * faithfully; ds4_gpu_tp_big_gate_kick returns a uint64_t sequence number
 * with 0 documented as failure (ds4_gpu.h), matching CUDA's `return 0`
 * exactly. */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    return 0;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t, uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(uint32_t layer, uint32_t rows,
                                             const ds4_gpu_tensor *out_t,
                                             ds4_gpu_tensor *in_t, uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_big_gate_wait(uint64_t seq) {
    (void)seq;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    return 0;
}

/* The coordinator-only DSpark support model does not participate in TP
 * (ds4_gpu.h's own comment on this entry); ds4.c saves and restores
 * g->tp_world to 0 around the encode this brackets
 * (ds4.c:32655-32698/33746-33842) rather than querying this flag back,
 * so there is no backend-side consumer of the suspended state for this
 * entry to track even in principle. Matches ds4_cuda.cu:30804-30806
 * exactly. */
extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

/* --gpu-vram auto CLI probe. Zero means success, nonzero on failure with
 * errbuf populated (ds4_gpu_args.h:63, confirmed against ROCm's real
 * implementation, ds4_rocm_compat.cu:144-186, which returns 1 on every one
 * of its seven failure paths and 0 only at the very end). CUDA's own
 * implementation (ds4_cuda.cu:25923-26018) is the structural reference for
 * this entry specifically rather than ROCm's, because CUDA genuinely
 * probes every visible device (or an explicit filter) while ROCm refuses
 * more than one; this backend's init_multi now supports real multi-device
 * use (above), so the CLI probe should not artificially restrict
 * `auto` to one device the way ROCm's single-GPU backend must. */
extern "C" int ds4_gpu_args_probe_auto_cuda(const int *device_filter, int filter_len,
                                            ds4_gpu_config *out,
                                            size_t safety_margin_bytes, char *errbuf,
                                            size_t errbuflen) {
    /* See the identical call and comment in ds4_gpu_init (ds4_sycl.cpp):
     * must run before ds4_sycl_enumerate_gpus below performs the first
     * Level Zero platform/device enumeration. CLI argument parsing runs
     * before engine setup, so this is very often the actual first call
     * into this backend, and ds4_gpu_init's own call is too late for it. */
    setenv("ZES_ENABLE_SYSMAN", "1", 0);

    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    const std::vector<sycl::device> all = ds4_sycl_enumerate_gpus();
    if (all.empty()) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "no SYCL GPU device found");
        return 1;
    }

    int devs[DS4_MAX_GPUS];
    int n_dev = 0;
    if (device_filter && filter_len > 0) {
        if (filter_len > DS4_MAX_GPUS) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen, "--gpu-devices filter has %d entries (max %d)",
                         filter_len, DS4_MAX_GPUS);
            }
            return 1;
        }
        for (int i = 0; i < filter_len; i++) {
            const int d = device_filter[i];
            if (d < 0 || (size_t)d >= all.size()) {
                if (errbuf && errbuflen) {
                    snprintf(errbuf, errbuflen, "--gpu-devices: device %d not in 0..%zu",
                             d, all.size() - 1);
                }
                return 1;
            }
            devs[n_dev++] = d;
        }
    } else {
        const int cap = (int)(all.size() < (size_t)DS4_MAX_GPUS ? all.size()
                                                                : (size_t)DS4_MAX_GPUS);
        for (int i = 0; i < cap; i++) devs[n_dev++] = i;
    }

    memset(out, 0, sizeof(*out));
    out->n_gpus = n_dev;
    out->safety_margin_bytes = safety_margin_bytes;
    for (int i = 0; i < n_dev; i++) {
        uint64_t free_bytes = 0;
        if (!sycl_zes_free_bytes(all[(size_t)devs[i]], &free_bytes)) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "Level Zero Sysman free-memory query failed on device %d; "
                         "pass explicit --gpu-vram budgets instead",
                         devs[i]);
            }
            return 1;
        }
        /* Reserve = max(2 GiB, 5% of free), matching CUDA's own auto-probe
         * reserve exactly (ds4_cuda.cu:26009-26012): a fixed floor for
         * small cards where 5% is under a GiB, scaling up on larger ones.
         * Explicit --gpu-vram budgets do not go through this probe and are
         * unaffected. */
        const uint64_t reserve_floor = 2ull * 1024 * 1024 * 1024;
        const uint64_t reserve_pct   = free_bytes / 20u;
        const uint64_t reserve       = reserve_floor > reserve_pct ? reserve_floor
                                                                    : reserve_pct;
        out->device_indices[i] = devs[i];
        out->vram_bytes[i]     = free_bytes > reserve ? (size_t)(free_bytes - reserve) : 0;
    }
    return 0;
}

/* Test-only: exposes the exact same Sysman free-memory reading the probe
 * itself uses, so a test can verify the reserve subtraction (probe's
 * `free_bytes - reserve`) actually ran, on the same accounting basis the
 * probe uses. Sysman's own total (zes_mem_state_t::size) was found not to
 * agree with sycl::info::device::global_mem_size on this hardware (see the
 * report), so comparing against the SYCL device query is the wrong ground
 * truth; this hook avoids that mismatch entirely by reusing the identical
 * query. Not part of the ABI. */
extern "C" uint64_t ds4_sycl_test_zes_free_bytes(int tier) {
    if (tier < 0 || (size_t)tier >= g_devices.size()) return 0;
    uint64_t free_bytes = 0;
    if (!sycl_zes_free_bytes(g_devices[(size_t)tier].dev, &free_bytes)) return 0;
    return free_bytes;
}
