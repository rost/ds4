#include "ds4_sycl.h"

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cstdio>
#include <cstring>

namespace {

std::vector<ds4_sycl_device> g_devices;
int                          g_current_tier = 0;
bool                         g_initialised  = false;

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
    (void)g_initialised;
    return 0;
}

extern "C" void ds4_gpu_cleanup(void) {
}
