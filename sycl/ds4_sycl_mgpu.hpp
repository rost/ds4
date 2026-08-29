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

#include "ds4_sycl_common.hpp"

/* ds4_gpu_config, ds4_gpu_ctx, DS4_MAX_GPUS, g_gpu, g_n_gpus and
 * g_gpu_peer_ok all come from ds4_gpu_mgpu.h, included by ds4_sycl.cpp
 * before this header. */

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) return 0;
    if (g_initialised) ds4_gpu_cleanup();

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

        /* NxN peer-access matrix. Seeds only the trivial diagonal
         * (a device always reaches its own memory); the real i != j
         * byte-validation protocol is sycl_validate_peer_pair below. Every
         * pair defaults to 0 (bounce-only) until validated. */
        for (int i = 0; i < g_n_gpus; i++) {
            for (int j = 0; j < g_n_gpus; j++) {
                g_gpu_peer_ok[i][j] = (i == j) ? 1 : 0;
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

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                                        uint64_t bytes) {
    if (!dst || !src) return 0;
    if (bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    const int sd = src->device_id >= 0 ? src->device_id : g_current_tier;
    const int dd = dst->device_id >= 0 ? dst->device_id : g_current_tier;
    if (sd == dd) return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
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
