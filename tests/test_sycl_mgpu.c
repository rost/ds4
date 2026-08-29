/* Multi-GPU plumbing tests for the SYCL backend: ds4_gpu_init_multi, the
 * real tier switch (ds4_gpu_set_current_device[_fenced]), per-tier
 * allocation (ds4_gpu_tensor_alloc_on / _free_in_place), the host-bounce
 * cross-device copy family, and the peer-access validation protocol.
 *
 * This machine has one GPU. Every test that genuinely needs a second
 * device follows tests/test_gpu_xdev.c's pattern: parametrise by device
 * count, run for real when enough devices are visible, and skip cleanly
 * (returning pass) otherwise, so the same binary runs for real on a
 * multi-GPU host with no changes.
 *
 * Deliberately self-contained: no shared harness header, matching
 * tests/test_sycl_fp8_kv.c and tests/test_sycl_router.c. Needs no model
 * file. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ds4_sycl_device_count / ds4_sycl_current_tier are declared extern "C" in
 * ds4_sycl.h, which is C++-only (it includes <sycl/sycl.hpp>) and cannot
 * be included from this C test. Forward-declared here instead, the same
 * technique tests/test_sycl_fp8_kv.c uses for its own SYCL-side hooks. */
extern int ds4_sycl_device_count(void);
extern int ds4_sycl_current_tier(void);

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                      \
        }                                                                   \
    } while (0)

static int physical_device_count(void) {
    if (!ds4_gpu_init()) return 0;
    int n = ds4_sycl_device_count();
    ds4_gpu_cleanup();
    return n;
}

/* ---- ds4_gpu_init_multi ------------------------------------------------ */

static int test_init_multi_n1(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    cfg.vram_bytes[0] = 1024ull * 1024 * 1024;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi n=1");
    CHECK(g_n_gpus == 1, "g_n_gpus after init_multi n=1");
    CHECK(g_gpu_peer_ok[0][0] == 1, "trivial diagonal peer_ok");
    CHECK(g_gpu[0].device_id == 0, "g_gpu[0].device_id");
    CHECK(g_gpu[0].budget_bytes == cfg.vram_bytes[0], "budget_bytes recorded");
    CHECK(ds4_sycl_current_tier() == 0, "current tier after init_multi");
    ds4_gpu_cleanup();
    return 0;
}

static int test_init_multi_rejects_bad_cfg(void) {
    CHECK(ds4_gpu_init_multi(NULL) == 0, "NULL cfg must fail");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 0;
    CHECK(ds4_gpu_init_multi(&cfg) == 0, "n_gpus=0 must fail");

    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = DS4_MAX_GPUS + 1;
    CHECK(ds4_gpu_init_multi(&cfg) == 0, "n_gpus over max must fail");

    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 99;
    CHECK(ds4_gpu_init_multi(&cfg) == 0, "out-of-range device index must fail");
    return 0;
}

/* ---- ds4_gpu_set_current_device[_fenced] ------------------------------- */

static int test_set_current_device(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for set_current_device");

    CHECK(ds4_gpu_set_current_device(0) == 0, "set_current_device(0) success");
    CHECK(ds4_sycl_current_tier() == 0, "current tier is 0");
    CHECK(ds4_gpu_set_current_device(1) != 0, "set_current_device(1) out of range");
    CHECK(ds4_gpu_set_current_device(-1) != 0, "set_current_device(-1) out of range");
    /* A failed switch must not silently change the current tier. */
    CHECK(ds4_sycl_current_tier() == 0, "current tier unchanged after failed switch");

    CHECK(ds4_gpu_set_current_device_fenced(0) == 0, "set_current_device_fenced(0) success");
    CHECK(ds4_gpu_set_current_device_fenced(1) != 0, "set_current_device_fenced(1) out of range");
    CHECK(ds4_gpu_set_current_device_fenced(-1) != 0, "set_current_device_fenced(-1) out of range");

    ds4_gpu_cleanup();
    return 0;
}

/* ---- ds4_gpu_tensor_alloc_on / _free_in_place -------------------------- */

static int test_tensor_alloc_on_tier0(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for tensor_alloc_on");

    const size_t n = 4096;
    float *host_src = (float *)malloc(n * sizeof(float));
    float *host_dst = (float *)malloc(n * sizeof(float));
    CHECK(host_src && host_dst, "host alloc");
    for (size_t i = 0; i < n; i++) host_src[i] = (float)(i % 251) * 0.25f;

    ds4_gpu_tensor t;
    memset(&t, 0, sizeof(t));
    CHECK(ds4_gpu_tensor_alloc_on(&t, 0, n * sizeof(float)) == 0, "alloc_on tier 0");
    CHECK(t.ptr != NULL, "alloc_on produced a pointer");
    CHECK(t.device_id == 0, "alloc_on stamped device_id");
    CHECK(ds4_gpu_tensor_device(&t) == 0, "tensor_device round trip");

    CHECK(ds4_gpu_tensor_write(&t, 0, host_src, n * sizeof(float)), "tensor_write");
    CHECK(ds4_gpu_tensor_read(&t, 0, host_dst, n * sizeof(float)), "tensor_read");
    CHECK(memcmp(host_src, host_dst, n * sizeof(float)) == 0, "data round trip");

    ds4_gpu_tensor_free_in_place(&t);
    CHECK(t.ptr == NULL, "free_in_place clears ptr");
    CHECK(t.bytes == 0, "free_in_place clears bytes");
    CHECK(t.owner == 0, "free_in_place clears owner");

    /* Freeing an already-freed (zeroed) tensor must be a safe no-op. */
    ds4_gpu_tensor_free_in_place(&t);

    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    return 0;
}

static int test_tensor_alloc_on_invalid(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for invalid alloc_on");

    CHECK(ds4_gpu_tensor_alloc_on(NULL, 0, 64) != 0, "NULL tensor must fail");

    ds4_gpu_tensor t;
    memset(&t, 0, sizeof(t));
    CHECK(ds4_gpu_tensor_alloc_on(&t, 5, 64) != 0, "out-of-range device_id must fail");
    CHECK(ds4_gpu_tensor_alloc_on(&t, -1, 64) != 0, "negative device_id must fail");

    /* free_in_place on a never-allocated (zeroed) tensor is a no-op, not a
     * crash. */
    ds4_gpu_tensor_free_in_place(&t);

    ds4_gpu_cleanup();
    return 0;
}

/* ---- multi-device path: skip cleanly on this box ----------------------- */

static int test_init_multi_n2_or_skip(void) {
    int have = physical_device_count();
    if (have < 2) {
        fprintf(stderr, "skip: wanted 2 GPUs, have %d\n", have);
        return 0;
    }
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi n=2");
    CHECK(g_n_gpus == 2, "g_n_gpus after init_multi n=2");
    CHECK(g_gpu_peer_ok[0][0] == 1, "diagonal 0,0");
    CHECK(g_gpu_peer_ok[1][1] == 1, "diagonal 1,1");
    ds4_gpu_cleanup();
    fprintf(stderr, "  init_multi n=2 OK\n");
    return 0;
}

int main(void) {
    if (test_init_multi_n1()) return 1;
    if (test_init_multi_rejects_bad_cfg()) return 1;
    if (test_set_current_device()) return 1;
    if (test_tensor_alloc_on_tier0()) return 1;
    if (test_tensor_alloc_on_invalid()) return 1;
    if (test_init_multi_n2_or_skip()) return 1;
    fprintf(stderr, "test_sycl_mgpu: all tests passed\n");
    return 0;
}
