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
