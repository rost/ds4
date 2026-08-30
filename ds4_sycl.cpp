#include "ds4_sycl.h"

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "sycl/ds4_sycl_common.hpp"

#include <algorithm>
#include <cstdint>
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

/* Every physical GPU on the box, backend-preferred (Level Zero over
 * OpenCL) and deduplicated.  Shared by ds4_gpu_init (which uses every
 * enumerated device) and ds4_gpu_init_multi/ds4_gpu_args_probe_auto_cuda
 * (sycl/ds4_sycl_mgpu.hpp; both index into this same list by physical
 * position), so the physical index space one entry point validates
 * against is exactly the index space another one enumerates. */
std::vector<sycl::device> ds4_sycl_enumerate_gpus() {
    std::vector<sycl::device> gpus;
    for (const sycl::platform &p : sycl::platform::get_platforms()) {
        for (const sycl::device &d : p.get_devices()) {
            if (d.is_gpu()) gpus.push_back(d);
        }
    }

    std::vector<sycl::device> preferred;
    for (const sycl::device &d : gpus) {
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            preferred.push_back(d);
        }
    }
    const std::vector<sycl::device> &chosen = preferred.empty() ? gpus : preferred;
    return ds4_sycl_dedup_devices(chosen);
}

/* Builds one sycl::queue per device in `wanted`, and appends them (paired
 * with their device) to *out, in `wanted`'s order.  USM allocations are
 * bound to a context, not a device (SYCL 2020), and any queue that
 * dereferences one must be built from that same context; a context in
 * turn can only span devices of one SYCL platform.  The original
 * per-device `sycl::queue(d, handler)` construction relied on whatever
 * default context the implementation happened to attach, which SYCL does
 * not guarantee is shared across devices -- not something multi-device
 * USM sharing can be built on.  This builds one explicit sycl::context per
 * distinct platform present in `wanted` (normally just one: Intel's own
 * multi-card guidance builds a single context spanning every root device
 * on the box, which is exactly what happens here when every device shares
 * a platform), and every device joins the context for its own platform, so
 * devices that DO share a platform also share USM visibility, while a
 * device on a different platform still gets a valid queue of its own
 * rather than being dropped.  Tier order always follows `wanted`'s order,
 * independent of how devices group into contexts. */

/* Measurement-only: property::queue::enable_profiling()
 * makes event::get_profiling_info usable on every event this queue
 * produces, at a per-submission cost real enough not to pay by default.
 * Gated on DS4_SYCL_PROFILE, read once at queue construction, so an
 * ordinary run or test pays nothing; a measurement run sets it before
 * ds4_gpu_init so every tier's queue is built profiling-capable from the
 * start. Deliberately does not add property::queue::in_order: spec-level
 * guidance elsewhere in this backend is that in_order would serialise the
 * very kernel overlap this backend is trying to make possible, and a
 * measurement tool must not change the thing it is trying to measure. */
static sycl::property_list ds4_sycl_queue_properties(void) {
    if (getenv("DS4_SYCL_PROFILE")) {
        return sycl::property_list{sycl::property::queue::enable_profiling()};
    }
    return sycl::property_list{};
}

void ds4_sycl_build_devices(const std::vector<sycl::device> &wanted,
                             std::vector<ds4_sycl_device> *out) {
    std::vector<sycl::platform> platforms;
    std::vector<sycl::context>  contexts;
    for (const sycl::device &d : wanted) {
        const sycl::platform p = d.get_platform();
        bool have_ctx = false;
        for (const sycl::platform &seen : platforms) {
            if (seen == p) { have_ctx = true; break; }
        }
        if (have_ctx) continue;
        std::vector<sycl::device> group;
        for (const sycl::device &d2 : wanted) {
            if (d2.get_platform() == p) group.push_back(d2);
        }
        platforms.push_back(p);
        contexts.push_back(sycl::context(group));
    }

    for (const sycl::device &d : wanted) {
        const sycl::platform p = d.get_platform();
        for (size_t k = 0; k < platforms.size(); k++) {
            if (platforms[k] == p) {
                out->push_back(ds4_sycl_device{
                    d, sycl::queue(contexts[k], d, ds4_sycl_async_handler,
                                   ds4_sycl_queue_properties())});
                break;
            }
        }
    }
}

} /* namespace */

/* Returns the first width in kRequiredSubGroupWidths that is absent from
 * `supported` (an array of `n_supported` widths a device reports via
 * sycl::info::device::sub_group_sizes), or 0 if every required width is
 * present. Pure host-side comparison, no device access, so it is directly
 * testable with a synthetic supported-widths array. 0 is never itself a
 * valid sub-group width, so it unambiguously means "nothing missing" here;
 * this is a local convention for this one helper, not a claim about the
 * ABI-wide return-polarity convention documented in spec 3a, which does
 * not apply to this internal (non-ABI) helper. */
extern "C" uint32_t ds4_sycl_missing_required_subgroup_width(
        const uint32_t *supported, int n_supported) {
    for (uint32_t required : kRequiredSubGroupWidths) {
        bool found = false;
        for (int i = 0; i < n_supported; i++) {
            if (supported[i] == required) { found = true; break; }
        }
        if (!found) return required;
    }
    return 0;
}

/* Defined in sycl/ds4_sycl_streaming.hpp, included at the end of this
 * file; forward-declared here so ds4_gpu_cleanup (below) can call it
 * while every tier's queue is still alive, before g_devices.clear()
 * destroys them.  Takes no argument: it loops over every live tier and
 * tears each one down through its own queue, not just whichever queue
 * the caller happens to pass in. */
static void sycl_stream_teardown_all(void);

/* Defined in sycl/ds4_sycl_readback.hpp, included at the end of this
 * file; forward-declared here so ds4_gpu_cleanup (below) can drop any
 * stored selected-readback event before g_devices.clear() destroys the
 * queue it was recorded against. */
static void sycl_readback_teardown(void);

/* Defined in sycl/ds4_sycl_placement.hpp, included at the end of this
 * file; forward-declared here for the same reason as
 * sycl_stream_teardown_all above: the per-device selective weight cache's
 * slabs must be freed through their own still-live queue before
 * g_devices.clear() destroys it, not after (spec 6g: a use-after-free
 * here would be exactly the shape that ablation could not make fail on
 * this hardware, so correctness has to come from construction order, not
 * from a test proving it). */
static void sycl_placement_teardown_all(void);

/* Defined in sycl/ds4_sycl_model_cache.hpp, included at the end of this
 * file; forward-declared here for the same reason as
 * sycl_placement_teardown_all above: the device-resident weight
 * cache's buffers must be freed through their own still-live queue before
 * g_devices.clear() destroys it, not after (spec 6g). */
static void sycl_model_cache_teardown_all(void);

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

    /* Must run before the first Level Zero platform/device enumeration
     * (ds4_sycl_enumerate_gpus, immediately below): Sysman cannot be armed
     * retroactively on an already-initialised Level Zero loader. Overwrite
     * is 0 so an operator's own exported value is respected; harmless when
     * Sysman is never queried (sycl_zes_free_bytes, ds4_sycl_common.hpp).
     * ds4_gpu_args_probe_auto_cuda (sycl/ds4_sycl_mgpu.hpp) sets the same
     * variable for the case where CLI argument parsing runs first and
     * reaches Sysman before this function ever does. */
    setenv("ZES_ENABLE_SYSMAN", "1", 0);

    try {
        std::vector<sycl::device> chosen = ds4_sycl_enumerate_gpus();

        if (chosen.empty()) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "no SYCL GPU device found\n");
            return 0;
        }

        ds4_sycl_build_devices(chosen, &g_devices);

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

        /* Per spec 6m, [[sycl::reqd_sub_group_size(N)]] is not enforced by
         * the driver on this stack: a device that cannot honour N silently
         * runs a different width instead of failing the kernel launch, and
         * every kernel using that annotation then reduces over the wrong
         * number of lanes for a plausible, wrong, answer with no error
         * anywhere (see kRequiredSubGroupWidths, ds4_sycl_common.hpp).
         * Checked against every device, not just tier 0, since the target
         * hardware is many independent cards and any one of them can be the
         * one missing a width. */
        for (const ds4_sycl_device &d : g_devices) {
            std::vector<size_t> raw = d.dev.get_info<sycl::info::device::sub_group_sizes>();
            std::vector<uint32_t> widths;
            widths.reserve(raw.size());
            for (size_t w : raw) widths.push_back((uint32_t)w);

            uint32_t missing = ds4_sycl_missing_required_subgroup_width(
                    widths.data(), (int)widths.size());
            if (missing != 0) {
                std::string reported;
                for (size_t i = 0; i < widths.size(); i++) {
                    if (i) reported += ", ";
                    reported += std::to_string(widths[i]);
                }
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "device \"%s\" does not report required sub-group width "
                        "%u (reports: %s); refusing to start. Per spec 6m, "
                        "[[sycl::reqd_sub_group_size(%u)]] is not enforced by the "
                        "driver on this stack, so continuing would not fail the "
                        "kernel launch -- it would silently run a different "
                        "hardware sub-group width and return wrong numbers with "
                        "no error\n",
                        d.dev.get_info<sycl::info::device::name>().c_str(), missing,
                        reported.empty() ? "(none)" : reported.c_str(), missing);
                g_devices.clear();
                g_n_gpus       = 0;
                g_current_tier = 0;
                g_initialised  = false;
                return 0;
            }
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
    /* Frees every tier's streaming cache slabs and scratch buffers while
     * their own queue is still alive, matching ROCm's
     * cuda_stream_selected_cache_release fan-out from ds4_gpu_cleanup
     * (rocm/ds4_rocm_runtime.cuh:5873). Must run before g_devices.clear()
     * below destroys the queues. */
    if (!g_devices.empty()) sycl_stream_teardown_all();
    sycl_readback_teardown();
    if (!g_devices.empty()) sycl_placement_teardown_all();
    if (!g_devices.empty()) sycl_model_cache_teardown_all();
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

/* Tier-aware heap allocators.  ds4.c calls these unconditionally for tier 0
 * on every session, single-GPU included (e.g. ds4.c:17073), and the ABI
 * contract in ds4_gpu_mgpu.h:111-118 says tier 0 is equivalent to the
 * legacy one-argument allocator.  The Apple fallback at ds4.c:179-192
 * spells out the failure when that contract is not honoured: every
 * multi-tier-aware allocation returns NULL, the validation chain fails,
 * and session_create silently reports failure.  These were stubbed to
 * nullptr until now, which meant the SYCL backend could not create a
 * session at all; no test caught it because every test calls ABI entries
 * directly rather than going through the engine's graph-creation path.
 *
 * Tiers above 0 are rejected rather than silently redirected: multi-device
 * allocation is future work, and a wrong-device tensor is far worse
 * than a clean failure under Level Zero, which has no unified virtual
 * addressing to paper over it (spec 6a). */
static int sycl_tier_is_valid(int tier, const char *what) {
    if (tier >= 0 && (size_t)tier < g_devices.size()) return 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "%s: bad tier %d (%zu device(s))\n",
            what, tier, g_devices.size());
    return 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    if (g_devices.empty() || !sycl_tier_is_valid(tier, "tensor_alloc_ptr_on")) {
        return nullptr;
    }
    if (tier == g_current_tier) return ds4_gpu_tensor_alloc(bytes);
    if (bytes == 0) bytes = 1;
    try {
        sycl::queue &q = ds4_sycl_queue(tier);
        void *ptr = sycl::malloc_device(bytes, q);
        if (ptr == nullptr) return nullptr;
        return new ds4_gpu_tensor{ptr, bytes, 1, tier};
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_ptr_on failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (g_devices.empty() || !sycl_tier_is_valid(tier, "tensor_alloc_managed_on")) {
        return nullptr;
    }
    if (tier == g_current_tier) return ds4_gpu_tensor_alloc_managed(bytes);
    if (bytes == 0) bytes = 1;
    try {
        sycl::queue &q = ds4_sycl_queue(tier);
        void *ptr = sycl::malloc_shared(bytes, q);
        if (ptr == nullptr) return nullptr;
        return new ds4_gpu_tensor{ptr, bytes, 1, tier};
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_managed_on failed: %s\n", e.what());
        return nullptr;
    }
}

/* Model-map registration.  ds4_engine_open calls one of these two on every
 * startup and ABORTS if it reports failure (ds4.c:58352 for the range form,
 * :58254 and :58308 for the spans form, both checked at :58358).  While they
 * were stubbed to their failure value, ds4_engine_open could not succeed for
 * any real model on this backend, which no test could see because the whole
 * suite calls ABI entries directly rather than opening an engine.
 *
 * This backend keeps no device-resident copy of the model: every kernel
 * stages the weight range it needs per call (spec 6l), because a SYCL kernel
 * cannot dereference the host mmap at all.  So there is nothing to register
 * and the honest implementation is a range validation.  ROCm does more here
 * (rocm/ds4_rocm_runtime.cuh:6133) only because its streaming mode maintains
 * a device-side model cache with a size limit; when that lands for SYCL, this
 * is where the limit check belongs. */
extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                           uint64_t map_offset, uint64_t map_size,
                                           uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (model_map == nullptr || model_size == 0) return 0;
    /* Overflow-safe: map_offset + map_size can wrap past UINT64_MAX. */
    if (map_offset > model_size || map_size > model_size - map_offset) return 0;
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size,
                                           const uint64_t *offsets, const uint64_t *sizes,
                                           uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (model_map == nullptr || model_size == 0) return 0;
    if (count != 0 && (offsets == nullptr || sizes == nullptr)) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] > model_size - offsets[i]) return 0;
    }
    return 1;
}

/* An OPTIONAL F16 preload of a Q8_0 weight range.  ds4.c:3020 checks the
 * result and aborts startup on failure (ds4.c:58405), so "not supported" must
 * be reported as success, not failure.  ROCm agrees: its implementation
 * (rocm/ds4_rocm_runtime.cuh:6314) returns 1 for a null map, zero bytes, a
 * disabled preload or a label its policy declines, and returns 0 only for a
 * genuinely out-of-range request.  This backend builds no such cache, so it
 * validates and reports success. */
extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                                          uint64_t offset, uint64_t bytes,
                                          uint64_t in_dim, uint64_t out_dim,
                                          const char *label) {
    (void)in_dim; (void)out_dim; (void)label;
    if (model_map == nullptr || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    return 1;
}

/* ds4_gpu_cache_model_range is implemented for real in
 * sycl/ds4_sycl_model_cache.hpp (included near the bottom of this file,
 * after g_devices and ds4_gpu_tier_free_vram exist): this replaced the
 * range-validation-only stub that used to live here with an actual
 * device-resident weight cache, once read by nobody
 * (sycl_model_range_ptr, ds4_sycl_common.hpp, returned a host pointer
 * unconditionally) and now consulted by it. See that header's comment for
 * the "optional cache, but a 0 return aborts ds4_engine_open" trap this
 * entry's return value must keep honouring. */

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
        sycl::event _ds4_prof_ev154 = q.memcpy((char *)tensor->ptr + offset, data, bytes);
        _ds4_prof_ev154.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev154);
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
        sycl::event _ds4_prof_ev155 = q.memcpy(data, (const char *)tensor->ptr + offset, bytes);
        _ds4_prof_ev155.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev155);
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
        sycl::event _ds4_prof_ev159 = q.memcpy((char *)dst->ptr + dst_offset,
                 (const char *)src->ptr + src_offset, bytes);
        _ds4_prof_ev159.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev159);
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
        sycl::event _ds4_prof_ev156 = q.fill((float *)tensor->ptr, value, (size_t)count);
        _ds4_prof_ev156.wait_and_throw();
        ds4_sycl_profile_record(_ds4_prof_ev156);
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_fill_f32 failed: %s\n", e.what());
        return 0;
    }
}

/* Same scoping reason as every include below: entry points here reference
 * g_devices, g_initialised, g_current_tier, ds4_sycl_queue,
 * ds4_gpu_cleanup and the enumeration/context helpers defined above, all
 * of which must already be in scope. */
#include "sycl/ds4_sycl_mgpu.hpp"

/* Same scoping reason as ds4_sycl_mgpu.hpp above: the command-lifecycle
 * entry points reference g_devices and ds4_sycl_current_queue. */
#include "sycl/ds4_sycl_commands.hpp"

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
/* Same scoping reason as the includes above: defines
 * sycl_model_cache_resolve (forward-declared in ds4_sycl_common.hpp and
 * called from sycl_model_range_ptr) and ds4_gpu_cache_model_range for
 * real. Must come before ds4_sycl_placement.hpp: ds4_gpu_tier_free_vram
 * there also subtracts this cache's committed bytes
 * (sycl_model_cache_committed_bytes) from its shared VRAM ledger. */
#include "sycl/ds4_sycl_model_cache.hpp"
/* Same scoping reason as the includes above, plus a direct dependency:
 * ds4_gpu_tier_free_vram reads ds4_sycl_streaming.hpp's g_sycl_stream_tier
 * and sycl_stream_vram_ceiling, and ds4_sycl_model_cache.hpp's
 * sycl_model_cache_committed_bytes, so this include must come after both. */
#include "sycl/ds4_sycl_placement.hpp"
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

/* Expert-parallel decode: owned-routed-MoE entries, reusing
 * sycl_routed_moe_launch and sycl_moe_stage_selected_experts above. Must
 * come before ds4_sycl_hc.hpp, which reuses its
 * sycl_moe_owned_packed_combine_row for the owned hc-expand fusion. */
#include "sycl/ds4_sycl_moe_owned.hpp"

/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_hc.hpp"

/* Same scoping reason as the includes above. */
#include "sycl/ds4_sycl_indexer.hpp"

/* Same scoping reason as the includes above. Must come after
 * ds4_sycl_fp8_kv.hpp (reuses its raw-KV batch store and FP8 quantiser) and
 * ds4_sycl_indexer.hpp (reuses its indexed_topk_sort_512_asc_kernel port). */
#include "sycl/ds4_sycl_attention.hpp"
/* Same scoping reason as the includes above: calls
 * ds4_gpu_matmul_q8_0_tensor, defined in ds4_sycl_matmul.hpp above. */
#include "sycl/ds4_sycl_attention_output.hpp"

/* Same scoping reason as the includes above: uses g_current_tier,
 * ds4_sycl_queue and g_devices directly, and defines sycl_readback_
 * teardown, forward-declared above for ds4_gpu_cleanup to call. */
#include "sycl/ds4_sycl_readback.hpp"
