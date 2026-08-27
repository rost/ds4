#include "ds4_sycl.h"

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

std::vector<ds4_sycl_device> g_devices;
int                          g_current_tier = 0;
bool                         g_initialised  = false;

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

/* Multi-GPU plumbing globals declared extern by ds4_gpu_mgpu.h and read
 * directly by ds4.c.  Not yet backed by real per-device state, so this
 * skeleton exposes a single logical tier with no peers, matching the
 * default ds4_rocm_compat.cu uses before ROCm's real device enumeration
 * runs. */
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
    if (g_initialised) return 0;

    try {
        std::vector<sycl::device> gpus;
        for (const sycl::platform &p : sycl::platform::get_platforms()) {
            for (const sycl::device &d : p.get_devices()) {
                if (d.is_gpu()) gpus.push_back(d);
            }
        }

        if (gpus.empty()) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "no SYCL GPU device found\n");
            return 1;
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
            g_devices.push_back(ds4_sycl_device{d, sycl::queue(d)});
        }

        g_current_tier = 0;
        g_initialised  = true;

        fprintf(stderr, DS4_GPU_LOG_PREFIX "%zu device(s), using %s\n",
                g_devices.size(),
                g_devices[0].dev.get_info<sycl::info::device::name>()
                    .c_str());
        return 0;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "device init failed: %s\n",
                e.what());
        g_devices.clear();
        g_current_tier = 0;
        g_initialised  = false;
        return 1;
    }
}

extern "C" void ds4_gpu_cleanup(void) {
    for (ds4_sycl_device &d : g_devices) d.queue.wait();
    g_devices.clear();
    g_current_tier = 0;
    g_initialised  = false;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) return nullptr;

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
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) return nullptr;

    sycl::queue &q = ds4_sycl_current_queue();
    void *ptr = sycl::malloc_shared(bytes, q);
    if (ptr == nullptr) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "malloc_shared of %llu bytes failed\n",
                (unsigned long long)bytes);
        return nullptr;
    }

    ds4_gpu_tensor *t = new ds4_gpu_tensor{ptr, bytes, 1, g_current_tier};
    return t;
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
        sycl::free(tensor->ptr, ds4_sycl_queue(tensor->device_id));
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
    if (tensor == nullptr || tensor->ptr == nullptr || data == nullptr) return 1;
    /* Overflow-safe: offset + bytes can wrap past UINT64_MAX. */
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 1;
    if (bytes == 0) return 0;

    sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                          : g_current_tier);
    q.memcpy((char *)tensor->ptr + offset, data, bytes).wait();
    return 0;
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                                   void *data, uint64_t bytes) {
    if (tensor == nullptr || tensor->ptr == nullptr || data == nullptr) return 1;
    /* Overflow-safe: offset + bytes can wrap past UINT64_MAX. */
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 1;
    if (bytes == 0) return 0;

    sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                          : g_current_tier);
    q.memcpy(data, (const char *)tensor->ptr + offset, bytes).wait();
    return 0;
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t bytes) {
    if (dst == nullptr || src == nullptr) return 1;
    if (dst->ptr == nullptr || src->ptr == nullptr) return 1;
    /* Overflow-safe on both sides. */
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 1;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 1;
    if (bytes == 0) return 0;

    /* Same-device copy only.  Cross-device transfer arrives with the mgpu
     * plan and routes through ds4_gpu_tensor_copy_xdev instead. */
    sycl::queue &q = ds4_sycl_queue(dst->device_id >= 0 ? dst->device_id
                                                        : g_current_tier);
    q.memcpy((char *)dst->ptr + dst_offset,
             (const char *)src->ptr + src_offset, bytes).wait();
    return 0;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                                       uint64_t count) {
    if (tensor == nullptr || tensor->ptr == nullptr) return 1;
    /* Overflow-safe: count * sizeof(float) can wrap.  Divide instead. */
    if (count > tensor->bytes / sizeof(float)) return 1;
    if (count == 0) return 0;

    sycl::queue &q = ds4_sycl_queue(tensor->device_id >= 0 ? tensor->device_id
                                                          : g_current_tier);
    q.fill((float *)tensor->ptr, value, (size_t)count).wait();
    return 0;
}
