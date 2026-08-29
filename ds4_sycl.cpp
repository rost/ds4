#include "ds4_sycl.h"

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

std::vector<ds4_sycl_device> g_devices;
int                          g_current_tier = 0;
bool                         g_initialised  = false;

/* Set for the duration of ds4_gpu_cleanup.  ~sycl::queue invokes this
 * translation unit's async handler for any error still pending when the
 * queue is destroyed; g_devices.clear() in ds4_gpu_cleanup destroys every
 * queue, and a handler that rethrows during that destruction would escape
 * a noexcept destructor and call std::terminate.  While this flag is set
 * the handler logs instead of rethrowing (see ds4_sycl_async_handler); it
 * also covers the namespace-scope g_devices destructor run at process
 * exit, since that also invokes the handler. */
bool g_tearing_down = false;

/* Installed on every queue so asynchronous errors are not silently
 * discarded.  Rethrowing here (rather than logging and swallowing) means
 * a caller synchronising with wait_and_throw() sees the error as a C++
 * exception at the call site, where the surrounding try/catch (see the
 * tensor alloc/free/transfer functions below) logs it and returns the
 * correct failure value for that entry's convention.  Every exception in
 * the list is logged before the first is rethrown so a caller who never
 * synchronises again does not silently lose the others.  While
 * g_tearing_down is set (see ds4_gpu_cleanup) the handler only logs: a
 * queue destructor is noexcept, and rethrowing out of it terminates the
 * process. */
void ds4_sycl_async_handler(sycl::exception_list exceptions) {
    std::exception_ptr first = nullptr;
    for (const std::exception_ptr &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (const sycl::exception &ex) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "async error: %s\n", ex.what());
        }
        if (!first) first = e;
    }
    if (first && !g_tearing_down) std::rethrow_exception(first);
}

/* Identifies a physical device independent of which backend or runtime
 * exposed it.  The device UUID is authoritative when the platform reports
 * it; the device name is the fallback, which is sufficient because two
 * runtimes fronting the same card report the same name. */
std::string ds4_sycl_device_identity(const sycl::device &d) {
    if (d.has(sycl::aspect::ext_intel_device_info_uuid)) {
        auto uuid = d.get_info<sycl::ext::intel::info::device::uuid>();
        return std::string(reinterpret_cast<const char *>(uuid.data()),
                            uuid.size());
    }
    return d.get_info<sycl::info::device::name>();
}

/* Backend preference (Level Zero over OpenCL) is not sufficient on its own
 * to guarantee one entry per physical card: this host has two OpenCL
 * runtimes installed concurrently, and nothing prevents either backend
 * from exposing the same GPU more than once under different driver state.
 * Dedupe by physical identity so a doubled card never ends up with two
 * independent queues, which would silently corrupt tier routing once tiers
 * map to real devices. */
std::vector<sycl::device> ds4_sycl_dedup_devices(
    const std::vector<sycl::device> &devices) {
    std::vector<sycl::device> unique;
    std::vector<std::string>  seen;
    for (const sycl::device &d : devices) {
        std::string id = ds4_sycl_device_identity(d);
        if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
        seen.push_back(id);
        unique.push_back(d);
    }
    return unique;
}

} /* namespace */

/* Defined in sycl/ds4_sycl_streaming.hpp, included at the end of this
 * file; forward-declared here so ds4_gpu_cleanup (below) can call it
 * while the queue every streaming cache slab and buffer was allocated
 * through is still alive, before g_devices.clear() destroys it. */
static void sycl_stream_teardown_all(sycl::queue &q);

/* Multi-GPU plumbing globals declared extern by ds4_gpu_mgpu.h and read
 * directly by ds4.c.  Before ds4_gpu_init runs (or if it never runs) this
 * exposes a single logical tier with no peers, matching the default
 * ds4_rocm_compat.cu uses before ROCm's real device enumeration runs;
 * ds4_gpu_init overwrites g_n_gpus and each g_gpu[].device_id with the
 * real enumeration once it succeeds.  The stream/cublas/scratch/budget
 * fields stay zeroed until the multi-GPU plan backs them. */
ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int         g_n_gpus            = 1;
int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

extern "C" int ds4_sycl_device_count(void) {
    return (int)g_devices.size();
}

extern "C" int ds4_sycl_current_tier(void) {
    return g_current_tier;
}

sycl::queue &ds4_sycl_queue(int tier) {
    /* g_devices can legitimately be empty (init never ran, init failed, or
     * cleanup already ran), and this function returns a reference so it has
     * no value it can use to signal that.  Every real caller (alloc, free,
     * transfer) checks g_devices.empty() itself before reaching here and
     * returns its own failure value instead of calling in; a caller that
     * reaches this branch anyway is a bug in this file, not a recoverable
     * runtime condition, so fail loudly here instead of indexing an empty
     * vector. */
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "ds4_sycl_queue called with no device initialised\n");
        abort();
    }
    /* Out-of-range tiers indicate an engine bug rather than a recoverable
     * condition, but returning tier 0 keeps the skeleton from indexing past
     * the end of the vector while later plans build out real tier routing. */
    if (tier < 0 || (size_t)tier >= g_devices.size()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tier %d out of range, using 0\n",
                tier);
        return g_devices[0].queue;
    }
    return g_devices[(size_t)tier].queue;
}

sycl::queue &ds4_sycl_current_queue(void) {
    return ds4_sycl_queue(g_current_tier);
}

extern "C" int ds4_gpu_init(void) {
    if (g_initialised) return 1;

    try {
        std::vector<sycl::device> gpus;
        for (const sycl::platform &p : sycl::platform::get_platforms()) {
            for (const sycl::device &d : p.get_devices()) {
                if (d.is_gpu()) gpus.push_back(d);
            }
        }

        if (gpus.empty()) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "no SYCL GPU device found\n");
            return 0;
        }

        /* Prefer the Level Zero backend when a device is exposed by more
         * than one platform, which is the normal case on Intel: the same
         * physical GPU appears under both OpenCL and Level Zero. */
        std::vector<sycl::device> preferred;
        for (const sycl::device &d : gpus) {
            if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                preferred.push_back(d);
            }
        }
        const std::vector<sycl::device> &chosen = preferred.empty() ? gpus
                                                                    : preferred;

        for (const sycl::device &d : ds4_sycl_dedup_devices(chosen)) {
            g_devices.push_back(
                ds4_sycl_device{d, sycl::queue(d, ds4_sycl_async_handler)});
        }

        /* Keep g_n_gpus and g_gpu[].device_id in step with what was just
         * enumerated: ds4.c reads these directly (see ds4.c's engine setup
         * and multi-tier dispatch) and would otherwise still see the
         * single-tier default declared below even when g_devices holds
         * every physical GPU on the box.  Only device_id is meaningful
         * here; stream, cublas, scratch and budget belong to the
         * multi-GPU plan and stay zeroed. */
        size_t n = g_devices.size();
        if (n > (size_t)DS4_MAX_GPUS) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "%zu devices found, clamping to DS4_MAX_GPUS=%d\n", n,
                    DS4_MAX_GPUS);
            n = (size_t)DS4_MAX_GPUS;
        }
        g_n_gpus = (int)n;
        for (int i = 0; i < g_n_gpus; i++) {
            g_gpu[i].device_id = i;
        }

        /* g_n_gpus > 1 arms ds4.c's multi-tier dispatch paths (gated on
         * g_n_gpus <= 1), but this backend's tensor alloc and current-device
         * entries are still failing stubs, so those paths cannot work yet.
         * One clear message at init time beats letting the caller discover
         * this through a mysterious nullptr or set_current_device failure. */
        if (g_n_gpus > 1) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "enumerated %d devices, but multi-GPU execution is not "
                    "yet implemented in the SYCL backend; single-device use "
                    "is expected until the multi-GPU work lands\n",
                    g_n_gpus);
        }

        g_current_tier = 0;
        g_initialised  = true;

        fprintf(stderr, DS4_GPU_LOG_PREFIX "%zu device(s), using %s\n",
                g_devices.size(),
                g_devices[0].dev.get_info<sycl::info::device::name>()
                    .c_str());
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "device init failed: %s\n",
                e.what());
        g_devices.clear();
        g_n_gpus       = 0;
        g_current_tier = 0;
        g_initialised  = false;
        return 0;
    }
}

/* Ordering contract: every ds4_gpu_tensor allocated through this backend
 * must be freed with ds4_gpu_tensor_free before ds4_gpu_cleanup runs.
 * Cleanup does not track or free live USM allocations itself, matching
 * ROCm's ds4_gpu_cleanup, which releases its own internal caches but never
 * frees caller-owned tensors either; tensor lifetime is the caller's
 * responsibility on every backend.  A tensor freed after cleanup does not
 * corrupt memory: ds4_gpu_tensor_free (like every other entry point here)
 * checks g_devices.empty() first and logs-and-leaks the device pointer
 * instead of calling into a torn-down queue.
 *
 * Teardown safety: g_tearing_down is set for the duration of this function
 * so ds4_sycl_async_handler logs rather than rethrows, then each queue is
 * drained with wait_and_throw() (not the plain wait() used elsewhere) so
 * any pending async error surfaces here, inside a try/catch that logs and
 * swallows it, rather than escaping ~sycl::queue during g_devices.clear()
 * and calling std::terminate out of that noexcept destructor. */
extern "C" void ds4_gpu_cleanup(void) {
    g_tearing_down = true;
    for (ds4_sycl_device &d : g_devices) {
        try {
            d.queue.wait_and_throw();
        } catch (const sycl::exception &e) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "cleanup: async error during "
                    "teardown: %s\n", e.what());
        }
    }
    /* Frees every streaming cache slab and scratch buffer while their
     * queue is still alive, matching ROCm's cuda_stream_selected_cache_release
     * fan-out from ds4_gpu_cleanup (rocm/ds4_rocm_runtime.cuh:5873). Must
     * run before g_devices.clear() below destroys the queue. */
    if (!g_devices.empty()) sycl_stream_teardown_all(ds4_sycl_current_queue());
    g_devices.clear();
    g_n_gpus       = 0;
    g_current_tier = 0;
    g_initialised  = false;
    g_tearing_down = false;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    /* Match ROCm (rocm/ds4_rocm_runtime.cuh): a zero-size request still
     * returns a real, freeable one-byte allocation rather than NULL, which
     * would otherwise read as allocation failure. */
    if (bytes == 0) bytes = 1;
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc with no device initialised\n");
        return nullptr;
    }

    try {
        sycl::queue &q = ds4_sycl_current_queue();
        void *ptr = sycl::malloc_device(bytes, q);
        if (ptr == nullptr) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "malloc_device of %llu bytes failed\n",
                    (unsigned long long)bytes);
            return nullptr;
        }

        /* owner 1 means this tensor must free ptr; device_id is the logical
         * tier the allocation lives on. */
        ds4_gpu_tensor *t = new ds4_gpu_tensor{ptr, bytes, 1, g_current_tier};
        return t;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_managed with no device initialised\n");
        return nullptr;
    }

    try {
        sycl::queue &q = ds4_sycl_current_queue();
        void *ptr = sycl::malloc_shared(bytes, q);
        if (ptr == nullptr) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "malloc_shared of %llu bytes failed\n",
                    (unsigned long long)bytes);
            return nullptr;
        }

        ds4_gpu_tensor *t = new ds4_gpu_tensor{ptr, bytes, 1, g_current_tier};
        return t;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_managed failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base,
                                               uint64_t offset,
                                               uint64_t bytes) {
    if (base == nullptr || base->ptr == nullptr) return nullptr;
    /* Overflow-safe: offset + bytes can wrap past UINT64_MAX. */
    if (offset > base->bytes || bytes > base->bytes - offset) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "view %llu+%llu exceeds %llu bytes\n",
                (unsigned long long)offset, (unsigned long long)bytes,
                (unsigned long long)base->bytes);
        return nullptr;
    }

    /* owner 0 marks a non-owning view: free must not release the pointer.
     * The view inherits the base tensor's tier. */
    ds4_gpu_tensor *v = new ds4_gpu_tensor{(char *)base->ptr + offset, bytes,
                                           0, base->device_id};
    return v;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (tensor == nullptr) return;
    if (tensor->owner != 0 && tensor->ptr != nullptr) {
        /* See the ordering contract on ds4_gpu_cleanup: a tensor freed
         * after cleanup has no live queue to free through.  Log and leak
         * rather than indexing the empty device vector. */
        if (g_devices.empty()) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tensor_free after cleanup; leaking device pointer\n");
        } else {
            try {
                sycl::free(tensor->ptr, ds4_sycl_queue(tensor->device_id));
            } catch (const sycl::exception &e) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_free failed: %s\n",
                        e.what());
            }
        }
    }
    delete tensor;
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    return tensor ? tensor->ptr : nullptr;
}

/* Declared in ds4_gpu_mgpu.h.  Returns the logical tier the allocation lives
 * on, or -1 when untagged, which the engine treats as tier 0. */
extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                                    const void *data, uint64_t bytes) {
    if (tensor == nullptr || tensor->ptr == nullptr || data == nullptr) return 0;
    /* Overflow-safe: offset + bytes can wrap past UINT64_MAX. */
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_write with no device initialised\n");
        return 0;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                              : g_current_tier);
        q.memcpy((char *)tensor->ptr + offset, data, bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_write failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                                   void *data, uint64_t bytes) {
    if (tensor == nullptr || tensor->ptr == nullptr || data == nullptr) return 0;
    /* Overflow-safe: offset + bytes can wrap past UINT64_MAX. */
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_read with no device initialised\n");
        return 0;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                              : g_current_tier);
        q.memcpy(data, (const char *)tensor->ptr + offset, bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_read failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t bytes) {
    if (dst == nullptr || src == nullptr) return 0;
    if (dst->ptr == nullptr || src->ptr == nullptr) return 0;
    /* Overflow-safe on both sides. */
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 0;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;

    /* Same-device copy only.  A cross-tier pair is a caller bug: raw USM
     * memcpy across contexts is undefined behaviour, not merely slow.
     * Cross-device transfer arrives with the mgpu plan and routes through
     * ds4_gpu_tensor_copy_xdev instead. */
    const int dst_tier = dst->device_id >= 0 ? dst->device_id : g_current_tier;
    const int src_tier = src->device_id >= 0 ? src->device_id : g_current_tier;
    if (dst_tier != src_tier) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "tensor_copy: cross-device copy (dst tier %d, src tier %d) "
                "not supported; use ds4_gpu_tensor_copy_xdev\n",
                dst_tier, src_tier);
        return 0;
    }
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_copy with no device initialised\n");
        return 0;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(dst_tier);
        q.memcpy((char *)dst->ptr + dst_offset,
                 (const char *)src->ptr + src_offset, bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_copy failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                                       uint64_t count) {
    if (tensor == nullptr || tensor->ptr == nullptr) return 0;
    /* Overflow-safe: count * sizeof(float) can wrap.  Divide instead. */
    if (count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    if (g_devices.empty()) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_fill_f32 with no device initialised\n");
        return 0;
    }

    try {
        sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                              : g_current_tier);
        q.fill((float *)tensor->ptr, value, (size_t)count).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_fill_f32 failed: %s\n", e.what());
        return 0;
    }
}

/* Included at the end of this file, not with the includes at the top:
 * kernel entry points in ds4_sycl_output.hpp reference g_devices (declared
 * in the anonymous namespace opened above at the top of this file) and
 * ds4_sycl_queue (defined above), so this header must come after both are
 * in scope.  Do not move it up to join the other includes. */
#include "sycl/ds4_sycl_output.hpp"

/* Same scoping reason as ds4_sycl_output.hpp above: kernel entry points
 * here reference g_devices and ds4_sycl_queue.  This header does not
 * depend on ds4_sycl_output.hpp; its own sycl_device_scratch_guard use
 * comes from ds4_sycl_common.hpp, which it includes directly. */
#include "sycl/ds4_sycl_embedding.hpp"

/* Same scoping reason as the two includes above. */
#include "sycl/ds4_sycl_norm_rope.hpp"

/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_compressor.hpp"

/* Same scoping reason as the includes above: the resident expert cache
 * entry points reference g_devices and ds4_sycl_current_queue. */
#include "sycl/ds4_sycl_streaming.hpp"
/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_matmul.hpp"
/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_router.hpp"
/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_fp8_kv.hpp"

/* Same scoping reason as the includes above: the shared expert's fused
 * fast-path kernel references g_devices and ds4_sycl_queue, and its
 * general path calls ds4_gpu_matmul_q8_0_pair_tensor and
 * ds4_gpu_swiglu_tensor above, both of which must already be defined. */
#include "sycl/ds4_sycl_shared_expert.hpp"

/* Same scoping reason as the includes above.  Kernels before launcher:
 * ds4_sycl_moe.hpp defines the routed-MoE kernels and internal helpers,
 * ds4_sycl_moe_launch.hpp the ABI entry points that dispatch to them. */
#include "sycl/ds4_sycl_moe.hpp"
#include "sycl/ds4_sycl_moe_launch.hpp"
