#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

#define DS4_GPU_BACKEND_NAME "SYCL"
#define DS4_GPU_LOG_PREFIX   "ds4: SYCL "
#define DS4_GPU_BLAS_NAME    "oneMKL"

/* One logical tier.  ds4 addresses devices by tier index and always calls
 * ds4_gpu_set_current_device(tier) before per-tier work, so per-tier state
 * lives here from the start rather than being retrofitted later. */
struct ds4_sycl_device {
    sycl::device dev;
    sycl::queue  queue;
};

/* Layout must match the ROCm and CUDA definition exactly.  owner is the
 * logical tier that owns ptr, or -1 for a non-owning view. */
struct ds4_gpu_tensor {
    void    *ptr;
    uint64_t bytes;
    int      owner;
};

/* These two are called from C test code, so they need C linkage. */
extern "C" int ds4_sycl_device_count(void);
extern "C" int ds4_sycl_current_tier(void);

/* C++ only: these return SYCL types and are never called from C. */
sycl::queue  &ds4_sycl_queue(int tier);
sycl::queue  &ds4_sycl_current_queue(void);
