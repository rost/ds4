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
