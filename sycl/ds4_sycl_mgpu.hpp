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

    /* See the identical call in ds4_gpu_init (ds4_sycl.cpp): must run
     * before the first Level Zero enumeration, which ds4_sycl_enumerate_gpus
     * immediately below performs. This entry point is a distinct process
     * startup path from ds4_gpu_init, so it needs its own call rather than
     * relying on that one having already run. */
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
         * failure. On this single-device box only the diagonal ever
         * executes; the i != j branch requires a second physical device to
         * run for real (see the report on testability). */
        for (int i = 0; i < g_n_gpus; i++) {
            for (int j = 0; j < g_n_gpus; j++) {
                g_gpu_peer_ok[i][j] = (i == j) ? 1 : (sycl_validate_peer_pair(i, j) ? 1 : 0);
            }
        }

        g_current_tier = 0;
        g_initialised  = true;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "init_multi: %d device(s) on a shared context per SYCL "
                "platform\n",
                g_n_gpus);
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
        sq.memcpy(host, src->ptr, bytes).wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_copy_xdev: bounce d2h failed: %s\n",
                e.what());
        ok = 0;
    }
    if (ok) {
        try {
            ds4_sycl_queue(dd).memcpy(dst->ptr, host, bytes).wait_and_throw();
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
 * at init can still be worth a fallback rather than a hard failure. */
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
            ds4_sycl_queue(dd).memcpy(dst->ptr, src->ptr, bytes).wait_and_throw();
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

/* SYCL has no default-stream concept distinct from a device's own queue:
 * every queue in this backend is already used synchronously (every
 * operation here ends in wait_and_throw() before returning), so there is
 * no separate "default stream" ordering domain for this entry to
 * distinguish from ds4_gpu_tensor_copy_xdev. Delegates, matching ROCm's
 * own shape (ds4_rocm_compat.cu:93-97), the structural reference for the
 * whole copy_xdev family. */
extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

/* Every copy in this backend already blocks until complete before
 * returning, so a call reaching this entry is always ordered after
 * everything previously submitted to the destination queue: there is no
 * outstanding work left to order against. Delegates for the same reason
 * ROCm does (ds4_rocm_compat.cu:99-103). */
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
