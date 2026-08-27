#pragma once

/* Never create a local sycl/sycl.hpp in this tree.  Many Makefile rules
 * compile with -I. (see rules for ds4_sycl.o, tests/test_sycl_smoke.o,
 * and friends), so a project-local sycl/sycl.hpp would shadow the real
 * SYCL header below and silently break every build that includes it. */
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

/* ds4_gpu_tensor itself is not defined here: it comes from the single
 * authoritative definition in ds4_gpu_mgpu.h (four fields: ptr, bytes,
 * owner, device_id), guarded by DS4_GPU_TENSOR_DEFINED.  ds4_sycl.cpp
 * includes that header directly. */

/* These two are called from C test code, so they need C linkage. */
extern "C" int ds4_sycl_device_count(void);
extern "C" int ds4_sycl_current_tier(void);

/* C++ only: these return SYCL types and are never called from C. */
sycl::queue  &ds4_sycl_queue(int tier);
sycl::queue  &ds4_sycl_current_queue(void);
