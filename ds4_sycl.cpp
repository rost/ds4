#include "ds4_sycl.h"

#include "ds4_gpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

/* Mirrors ds4_gpu_ctx from ds4_gpu_mgpu.h field-for-field.  That header
 * cannot be included here: its struct ds4_gpu_tensor definition (four
 * fields, with device_id) conflicts with the three-field definition in
 * ds4_sycl.h that this backend uses, the same reason ds4_rocm.cu keeps its
 * own local ds4_gpu_tensor instead of including ds4_gpu_mgpu.h.  Keep this
 * layout in sync with ds4_gpu_ctx if that changes. */
struct ds4_sycl_gpu_ctx {
    int      device_id;
    void    *stream;
    void    *cublas;
    int      cublas_ready;
    void    *scratch;
    uint64_t scratch_bytes;
    uint64_t budget_bytes;
    uint64_t used_bytes;
    void    *boundary_event;
};

constexpr int kMaxGpus = 16; /* mirrors DS4_MAX_GPUS from ds4_gpu_mgpu.h */

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
ds4_sycl_gpu_ctx g_gpu[kMaxGpus] = {};
int              g_n_gpus       = 1;
int              g_gpu_peer_ok[kMaxGpus][kMaxGpus] = {{1}};

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
