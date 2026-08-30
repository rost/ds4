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
#include "ds4_gpu_args.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ds4_sycl_device_count / ds4_sycl_current_tier are declared extern "C" in
 * ds4_sycl.h, which is C++-only (it includes <sycl/sycl.hpp>) and cannot
 * be included from this C test. Forward-declared here instead, the same
 * technique tests/test_sycl_fp8_kv.c uses for its own SYCL-side hooks. */
extern int ds4_sycl_device_count(void);
extern int ds4_sycl_current_tier(void);

/* Test-only instrumentation on the host-bounce helper behind
 * ds4_gpu_tensor_copy_xdev, defined in sycl/ds4_sycl_mgpu.hpp: lets the
 * cross-device-bounce tests below prove the bounce path is unreached on
 * this single-device box rather than merely assert it. */
extern int ds4_sycl_test_xdev_bounce_calls(void);

/* Test-only: exercises the peer-access byte-validation protocol's shared
 * inner loop against a single real device used as both legs, since this
 * box has no second device to validate real cross-die PCIe behaviour
 * against. See sycl/ds4_sycl_mgpu.hpp for exactly what this can and
 * cannot prove. */
extern int ds4_sycl_test_peer_bytecheck(int tier, int corrupt,
                                         int inject_compare_bug, int vary_pattern);

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

/* ---- ds4_gpu_tensor_copy_xdev family: same-device fast path ----------- */

static int test_copy_xdev_same_device(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for copy_xdev");

    const size_t n = 8192;
    float *host_src = (float *)malloc(n * sizeof(float));
    float *host_dst = (float *)malloc(n * sizeof(float));
    CHECK(host_src && host_dst, "host alloc");
    for (size_t i = 0; i < n; i++) host_src[i] = (float)((i * 7) % 991) * 0.125f;

    ds4_gpu_tensor src, dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    CHECK(ds4_gpu_tensor_alloc_on(&src, 0, n * sizeof(float)) == 0, "alloc src");
    CHECK(ds4_gpu_tensor_alloc_on(&dst, 0, n * sizeof(float)) == 0, "alloc dst");
    CHECK(ds4_gpu_tensor_write(&src, 0, host_src, n * sizeof(float)), "write src");

    const int calls_before = ds4_sycl_test_xdev_bounce_calls();
    CHECK(ds4_gpu_tensor_copy_xdev(&dst, &src, n * sizeof(float)) != 0,
          "copy_xdev same-device");
    CHECK(ds4_sycl_test_xdev_bounce_calls() == calls_before,
          "same-device copy_xdev must not take the host-bounce path");

    CHECK(ds4_gpu_tensor_read(&dst, 0, host_dst, n * sizeof(float)), "read dst");
    CHECK(memcmp(host_src, host_dst, n * sizeof(float)) == 0, "copy_xdev data match");

    /* A corrupted expectation must make this exact comparison fail: proves
     * the memcmp above is discriminating, not tautological. */
    unsigned char *corrupted = (unsigned char *)malloc(n * sizeof(float));
    CHECK(corrupted != NULL, "corrupted host alloc");
    memcpy(corrupted, host_dst, n * sizeof(float));
    corrupted[0] ^= 0xFFu;
    CHECK(memcmp(host_src, corrupted, n * sizeof(float)) != 0,
          "corrupted comparison must be caught");
    free(corrupted);

    /* bytes == 0 is a trivial success with no work performed. */
    CHECK(ds4_gpu_tensor_copy_xdev(&dst, &src, 0) != 0, "copy_xdev zero bytes");
    /* NULL and oversized-bytes must fail. */
    CHECK(ds4_gpu_tensor_copy_xdev(NULL, &src, 64) == 0, "copy_xdev NULL dst");
    CHECK(ds4_gpu_tensor_copy_xdev(&dst, NULL, 64) == 0, "copy_xdev NULL src");
    CHECK(ds4_gpu_tensor_copy_xdev(&dst, &src, dst.bytes + 1) == 0,
          "copy_xdev oversized bytes");

    ds4_gpu_tensor_free_in_place(&src);
    ds4_gpu_tensor_free_in_place(&dst);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    return 0;
}

static int test_copy_xdev_default_and_ordered_same_device(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for copy_xdev variants");

    const size_t n = 1024;
    float *host_src = (float *)malloc(n * sizeof(float));
    float *host_dst = (float *)malloc(n * sizeof(float));
    CHECK(host_src && host_dst, "host alloc");
    for (size_t i = 0; i < n; i++) host_src[i] = (float)i * 0.5f - 10.0f;

    ds4_gpu_tensor src, dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    CHECK(ds4_gpu_tensor_alloc_on(&src, 0, n * sizeof(float)) == 0, "alloc src");
    CHECK(ds4_gpu_tensor_alloc_on(&dst, 0, n * sizeof(float)) == 0, "alloc dst");
    CHECK(ds4_gpu_tensor_write(&src, 0, host_src, n * sizeof(float)), "write src");

    CHECK(ds4_gpu_tensor_copy_xdev_default(&dst, &src, n * sizeof(float)) != 0,
          "copy_xdev_default");
    CHECK(ds4_gpu_tensor_read(&dst, 0, host_dst, n * sizeof(float)), "read dst 1");
    CHECK(memcmp(host_src, host_dst, n * sizeof(float)) == 0, "copy_xdev_default match");

    memset(host_dst, 0, n * sizeof(float));
    CHECK(ds4_gpu_tensor_copy_xdev_ordered(&dst, &src, n * sizeof(float)) != 0,
          "copy_xdev_ordered");
    CHECK(ds4_gpu_tensor_read(&dst, 0, host_dst, n * sizeof(float)), "read dst 2");
    CHECK(memcmp(host_src, host_dst, n * sizeof(float)) == 0, "copy_xdev_ordered match");

    ds4_gpu_tensor_free_in_place(&src);
    ds4_gpu_tensor_free_in_place(&dst);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    return 0;
}

static int test_copy_xdev3_same_device(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for copy_xdev3");

    const uint32_t n_a = 64, n_b = 3, n_c = 17;
    float host_a[64], got_a[64];
    int32_t host_b[3] = {5, -2, 1000};
    int32_t got_b[3] = {0, 0, 0};
    float host_c[17], got_c[17];
    for (uint32_t i = 0; i < n_a; i++) host_a[i] = (float)i * 0.1f;
    for (uint32_t i = 0; i < n_c; i++) host_c[i] = (float)i - 8.0f;

    ds4_gpu_tensor sa, da, sb, db, sc, dc;
    memset(&sa, 0, sizeof(sa)); memset(&da, 0, sizeof(da));
    memset(&sb, 0, sizeof(sb)); memset(&db, 0, sizeof(db));
    memset(&sc, 0, sizeof(sc)); memset(&dc, 0, sizeof(dc));
    CHECK(ds4_gpu_tensor_alloc_on(&sa, 0, sizeof(host_a)) == 0, "alloc sa");
    CHECK(ds4_gpu_tensor_alloc_on(&da, 0, sizeof(host_a)) == 0, "alloc da");
    CHECK(ds4_gpu_tensor_alloc_on(&sb, 0, sizeof(host_b)) == 0, "alloc sb");
    CHECK(ds4_gpu_tensor_alloc_on(&db, 0, sizeof(host_b)) == 0, "alloc db");
    CHECK(ds4_gpu_tensor_alloc_on(&sc, 0, sizeof(host_c)) == 0, "alloc sc");
    CHECK(ds4_gpu_tensor_alloc_on(&dc, 0, sizeof(host_c)) == 0, "alloc dc");

    int ok = ds4_gpu_tensor_write(&sa, 0, host_a, sizeof(host_a)) &&
             ds4_gpu_tensor_write(&sb, 0, host_b, sizeof(host_b)) &&
             ds4_gpu_tensor_write(&sc, 0, host_c, sizeof(host_c)) &&
             ds4_gpu_tensor_copy_xdev3(&da, &sa, sizeof(host_a),
                                       &db, &sb, sizeof(host_b),
                                       &dc, &sc, sizeof(host_c)) &&
             ds4_gpu_tensor_read(&da, 0, got_a, sizeof(host_a)) &&
             ds4_gpu_tensor_read(&db, 0, got_b, sizeof(host_b)) &&
             ds4_gpu_tensor_read(&dc, 0, got_c, sizeof(host_c));
    CHECK(ok, "copy_xdev3 IO");
    CHECK(memcmp(host_a, got_a, sizeof(host_a)) == 0, "copy_xdev3 a");
    CHECK(memcmp(host_b, got_b, sizeof(host_b)) == 0, "copy_xdev3 b");
    CHECK(memcmp(host_c, got_c, sizeof(host_c)) == 0, "copy_xdev3 c");
    (void)n_b;

    /* One zero-byte leg among the three must be skipped, not treated as a
     * failure. */
    CHECK(ds4_gpu_tensor_copy_xdev3(&da, &sa, sizeof(host_a),
                                    &db, &sb, 0,
                                    &dc, &sc, sizeof(host_c)) != 0,
          "copy_xdev3 with one zero-byte leg");

    CHECK(ds4_gpu_tensor_copy_xdev3_default_dst(&da, &sa, sizeof(host_a),
                                                &db, &sb, sizeof(host_b),
                                                &dc, &sc, sizeof(host_c)) != 0,
          "copy_xdev3_default_dst");

    ds4_gpu_tensor_free_in_place(&sa); ds4_gpu_tensor_free_in_place(&da);
    ds4_gpu_tensor_free_in_place(&sb); ds4_gpu_tensor_free_in_place(&db);
    ds4_gpu_tensor_free_in_place(&sc); ds4_gpu_tensor_free_in_place(&dc);
    ds4_gpu_cleanup();
    return 0;
}

/* ---- Tensor parallelism: ds4_gpu_add_xdev_tensor ------------- */

static int test_add_xdev_same_device(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for add_xdev");

    enum { N = 4096 };
    float host_local[N], host_remote[N], want[N], got[N];
    for (int i = 0; i < N; i++) {
        host_local[i] = (float)((i * 7) % 991) * 0.1f;
        host_remote[i] = (float)((i * 13) % 773) * 0.05f - 3.0f;
        want[i] = host_local[i] + host_remote[i];
    }

    ds4_gpu_tensor out, local, remote;
    memset(&out, 0, sizeof(out));
    memset(&local, 0, sizeof(local));
    memset(&remote, 0, sizeof(remote));
    CHECK(ds4_gpu_tensor_alloc_on(&out, 0, sizeof(want)) == 0, "add_xdev: alloc out");
    CHECK(ds4_gpu_tensor_alloc_on(&local, 0, sizeof(host_local)) == 0, "add_xdev: alloc local");
    CHECK(ds4_gpu_tensor_alloc_on(&remote, 0, sizeof(host_remote)) == 0, "add_xdev: alloc remote");
    CHECK(ds4_gpu_tensor_write(&local, 0, host_local, sizeof(host_local)) != 0,
          "add_xdev: write local");
    CHECK(ds4_gpu_tensor_write(&remote, 0, host_remote, sizeof(host_remote)) != 0,
          "add_xdev: write remote");

    /* Same device end to end (od == ld == rd): the only path a single GPU
     * can exercise. remote_tmp is NULL here on purpose -- the same-device
     * branch must never touch it, matching ds4_cuda.cu's own `if (rd !=
     * od)` guard around the only place remote_tmp is read. */
    CHECK(ds4_gpu_add_xdev_tensor(&out, &local, &remote, NULL, N) != 0,
          "add_xdev: same-device call");
    CHECK(ds4_gpu_tensor_read(&out, 0, got, sizeof(got)) != 0, "add_xdev: read out");
    for (int i = 0; i < N; i++) {
        CHECK(fabsf(got[i] - want[i]) <= 1e-4f, "add_xdev: value mismatch");
    }

    /* n == 0 is a free success, matching ds4_cuda.cu:18656 exactly. */
    CHECK(ds4_gpu_add_xdev_tensor(&out, &local, &remote, NULL, 0) != 0,
          "add_xdev: n=0 must succeed");

    /* Validation: undersized tensors and NULLs must be rejected. */
    ds4_gpu_tensor small;
    memset(&small, 0, sizeof(small));
    CHECK(ds4_gpu_tensor_alloc_on(&small, 0, sizeof(float) * 2) == 0, "add_xdev: alloc small");
    CHECK(ds4_gpu_add_xdev_tensor(&small, &local, &remote, NULL, N) == 0,
          "add_xdev: undersized out must be rejected");
    CHECK(ds4_gpu_add_xdev_tensor(&out, &small, &remote, NULL, N) == 0,
          "add_xdev: undersized local must be rejected");
    CHECK(ds4_gpu_add_xdev_tensor(&out, &local, &small, NULL, N) == 0,
          "add_xdev: undersized remote must be rejected");
    CHECK(ds4_gpu_add_xdev_tensor(NULL, &local, &remote, NULL, N) == 0,
          "add_xdev: null out must be rejected");

    ds4_gpu_tensor_free_in_place(&out);
    ds4_gpu_tensor_free_in_place(&local);
    ds4_gpu_tensor_free_in_place(&remote);
    ds4_gpu_tensor_free_in_place(&small);
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_add_xdev_same_device OK\n");
    return 0;
}

/* Cross-device branch (rd != od, the actual cross-rank exchange path):
 * requires a second physical GPU. Honest skip on this A770, per the
 * standing testing-honesty rule -- do not fake coverage of a path that
 * cannot run here. First thing to check on the B60 machine: this exact
 * test with have >= 2, confirming out == local + remote after a genuine
 * device-to-device copy_xdev, not just that the call returns nonzero. */
static int test_add_xdev_cross_device_or_skip(void) {
    int have = physical_device_count();
    if (have < 2) {
        fprintf(stderr, "skip: add_xdev cross-device path wanted 2 GPUs, have %d "
                        "(verify this case on the B60 machine)\n", have);
        return 0;
    }
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for add_xdev cross-device");

    enum { N = 4096 };
    float host_local[N], host_remote[N], want[N], got[N];
    for (int i = 0; i < N; i++) {
        host_local[i] = (float)((i * 7) % 991) * 0.1f;
        host_remote[i] = (float)((i * 13) % 773) * 0.05f - 3.0f;
        want[i] = host_local[i] + host_remote[i];
    }

    ds4_gpu_tensor out, local, remote, remote_tmp;
    memset(&out, 0, sizeof(out));
    memset(&local, 0, sizeof(local));
    memset(&remote, 0, sizeof(remote));
    memset(&remote_tmp, 0, sizeof(remote_tmp));
    CHECK(ds4_gpu_tensor_alloc_on(&out, 0, sizeof(want)) == 0, "add_xdev: alloc out");
    CHECK(ds4_gpu_tensor_alloc_on(&local, 0, sizeof(host_local)) == 0, "add_xdev: alloc local");
    CHECK(ds4_gpu_tensor_alloc_on(&remote, 1, sizeof(host_remote)) == 0,
          "add_xdev: alloc remote on tier 1");
    CHECK(ds4_gpu_tensor_alloc_on(&remote_tmp, 0, sizeof(host_remote)) == 0,
          "add_xdev: alloc remote_tmp on tier 0");
    CHECK(ds4_gpu_tensor_write(&local, 0, host_local, sizeof(host_local)) != 0,
          "add_xdev: write local");
    CHECK(ds4_gpu_tensor_write(&remote, 0, host_remote, sizeof(host_remote)) != 0,
          "add_xdev: write remote");

    CHECK(ds4_gpu_add_xdev_tensor(&out, &local, &remote, &remote_tmp, N) != 0,
          "add_xdev: cross-device call");
    CHECK(ds4_gpu_tensor_read(&out, 0, got, sizeof(got)) != 0, "add_xdev: read out");
    for (int i = 0; i < N; i++) {
        CHECK(fabsf(got[i] - want[i]) <= 1e-4f, "add_xdev: cross-device value mismatch");
    }
    /* Missing/undersized remote_tmp on a genuine cross-device call must be
     * rejected, not silently fall back to the same-device path. */
    CHECK(ds4_gpu_add_xdev_tensor(&out, &local, &remote, NULL, N) == 0,
          "add_xdev: cross-device without remote_tmp must be rejected");

    ds4_gpu_tensor_free_in_place(&out);
    ds4_gpu_tensor_free_in_place(&local);
    ds4_gpu_tensor_free_in_place(&remote);
    ds4_gpu_tensor_free_in_place(&remote_tmp);
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_add_xdev_cross_device OK\n");
    return 0;
}

/* ---- ds4_gpu_tensor_wait_xdev[_default] -------------------------------- */

static int test_wait_xdev(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for wait_xdev");

    ds4_gpu_tensor t;
    memset(&t, 0, sizeof(t));
    CHECK(ds4_gpu_tensor_alloc_on(&t, 0, 256) == 0, "alloc t");

    CHECK(ds4_gpu_tensor_wait_xdev(&t, 0) != 0, "wait_xdev same tier");
    CHECK(ds4_gpu_tensor_wait_xdev(&t, 1) == 0, "wait_xdev out-of-range dst_tier");
    CHECK(ds4_gpu_tensor_wait_xdev(NULL, 0) == 0, "wait_xdev NULL src");

    CHECK(ds4_gpu_tensor_wait_xdev_default(&t, 0) != 0, "wait_xdev_default same tier");
    CHECK(ds4_gpu_tensor_wait_xdev_default(&t, 1) == 0,
          "wait_xdev_default out-of-range dst_tier");

    ds4_gpu_tensor_free_in_place(&t);
    ds4_gpu_cleanup();
    return 0;
}

/* ---- peer-access byte-validation protocol ------------------------------
 *
 * sycl_validate_peer_pair itself (the capability query, enable, and
 * dispatch into the shared bytecheck loop between two DISTINCT devices)
 * cannot run for real on this box: g_gpu_peer_ok's off-diagonal entries
 * are only ever populated by ds4_gpu_init_multi's i != j loop, which never
 * executes with one device. What follows exercises the shared bytecheck
 * loop's own mechanics directly through ds4_sycl_test_peer_bytecheck. */

static int test_peer_bytecheck_baseline(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for peer bytecheck");

    /* Clean baseline (spec 6n): establish that the unablated probe passes
     * before any ablation is attempted below. */
    CHECK(ds4_sycl_test_peer_bytecheck(0, /*corrupt=*/0, /*inject_compare_bug=*/0,
                                        /*vary_pattern=*/1) == 1,
          "peer bytecheck clean baseline");

    ds4_gpu_cleanup();
    return 0;
}

static int test_peer_bytecheck_corrupt_is_caught(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for corrupt bytecheck");

    /* A genuinely corrupted round MUST be reported as a failure: proves
     * the comparison is wired to the data it claims to check. */
    CHECK(ds4_sycl_test_peer_bytecheck(0, /*corrupt=*/1, /*inject_compare_bug=*/0,
                                        /*vary_pattern=*/1) == 0,
          "corrupted round must be caught");

    ds4_gpu_cleanup();
    return 0;
}

static int test_peer_bytecheck_pattern_variation_matters(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi for pattern-variation bytecheck");

    /* A stale-comparison-window defect (compare against the previous
     * iteration's expected pattern) must be exposed when the pattern
     * varies per iteration... */
    CHECK(ds4_sycl_test_peer_bytecheck(0, /*corrupt=*/0, /*inject_compare_bug=*/1,
                                        /*vary_pattern=*/1) == 0,
          "varying pattern must expose the stale-comparison defect");
    /* ...and hidden when it does not: this is the "same pattern for every
     * iteration" ablation from the plan, run directly against the shared
     * loop since sycl_validate_peer_pair itself cannot execute on one
     * device. A constant pattern across iterations makes the previous
     * iteration's expected value equal the current one, so the injected
     * defect produces no observable mismatch. */
    CHECK(ds4_sycl_test_peer_bytecheck(0, /*corrupt=*/0, /*inject_compare_bug=*/1,
                                        /*vary_pattern=*/0) == 1,
          "constant pattern hides the same stale-comparison defect");

    ds4_gpu_cleanup();
    return 0;
}

/* ---- ds4_gpu_args_probe_auto_cuda (--gpu-vram auto) -------------------- */

static int test_probe_auto_success(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    char errbuf[256];
    errbuf[0] = '\0';

    CHECK(ds4_gpu_args_probe_auto_cuda(NULL, 0, &cfg, 12345, errbuf,
                                       sizeof(errbuf)) == 0,
          "probe_auto success (0 = success)");
    CHECK(cfg.n_gpus == 1, "probe_auto n_gpus on a one-device box");
    CHECK(cfg.device_indices[0] == 0, "probe_auto device_indices[0]");
    CHECK(cfg.vram_bytes[0] > 0, "probe_auto vram_bytes[0] positive");
    CHECK(cfg.safety_margin_bytes == 12345, "probe_auto safety_margin_bytes passthrough");
    return 0;
}

static int test_probe_auto_null_out(void) {
    char errbuf[256];
    errbuf[0] = '\0';
    CHECK(ds4_gpu_args_probe_auto_cuda(NULL, 0, NULL, 0, errbuf, sizeof(errbuf)) != 0,
          "probe_auto NULL out must fail");
    CHECK(errbuf[0] != '\0', "probe_auto NULL out must populate errbuf");
    return 0;
}

static int test_probe_auto_bad_filter(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    char errbuf[256];

    int too_many[DS4_MAX_GPUS + 1];
    memset(too_many, 0, sizeof(too_many));
    CHECK(ds4_gpu_args_probe_auto_cuda(too_many, DS4_MAX_GPUS + 1, &cfg, 0, errbuf,
                                       sizeof(errbuf)) != 0,
          "probe_auto filter longer than DS4_MAX_GPUS must fail");

    int out_of_range[1] = {99};
    CHECK(ds4_gpu_args_probe_auto_cuda(out_of_range, 1, &cfg, 0, errbuf,
                                       sizeof(errbuf)) != 0,
          "probe_auto out-of-range filtered device must fail");

    int valid[1] = {0};
    CHECK(ds4_gpu_args_probe_auto_cuda(valid, 1, &cfg, 0, errbuf, sizeof(errbuf)) == 0,
          "probe_auto valid single-device filter must succeed");
    CHECK(cfg.n_gpus == 1 && cfg.device_indices[0] == 0,
          "probe_auto valid filter populates the requested device");
    return 0;
}

/* ds4_sycl_test_zes_free_bytes is a test-only hook (see
 * sycl/ds4_sycl_mgpu.hpp) that runs the identical Sysman free-memory query
 * the probe itself uses, so this test can confirm the reserve subtraction
 * actually ran, compared on the SAME accounting basis the probe uses.
 * Comparing against sycl::info::device::global_mem_size instead was tried
 * and found unreliable: Sysman's own total (zes_mem_state_t::size, 17.08
 * GB) does not agree with the SYCL device query (16.23 GB) on this
 * hardware, an environment-specific discrepancy reported separately.
 *
 * Separately, a direct experiment (see the report) found Sysman's `free`
 * reading does not respond to real allocation pressure on this A770 and
 * driver combination at all: a held 2 GiB USM allocation left it
 * completely unchanged. A before/after-allocation comparison (the more
 * direct test that the probe reads free rather than total) would not
 * discriminate anything on this box and is not attempted here. */
extern uint64_t ds4_sycl_test_zes_free_bytes(int tier);

static int test_probe_auto_reserve_applied(void) {
    /* ds4_sycl_test_zes_free_bytes reads g_devices, which needs an init
     * call first; ds4_gpu_args_probe_auto_cuda itself does not, since it
     * enumerates independently. */
    CHECK(ds4_gpu_init(), "ds4_gpu_init for reserve check");

    const uint64_t free_bytes = ds4_sycl_test_zes_free_bytes(0);
    CHECK(free_bytes > 0, "Sysman free-memory query must answer");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    char errbuf[256];
    CHECK(ds4_gpu_args_probe_auto_cuda(NULL, 0, &cfg, 0, errbuf, sizeof(errbuf)) == 0,
          "probe_auto for reserve check");

    CHECK(cfg.vram_bytes[0] < free_bytes,
          "probe_auto budget must be below free device memory (a reserve was applied)");
    const uint64_t reserve = free_bytes - cfg.vram_bytes[0];
    const uint64_t two_gib = 2ull * 1024 * 1024 * 1024;
    CHECK(reserve >= two_gib - (two_gib / 100),
          "probe_auto reserve must be at least approximately the 2 GiB floor");

    ds4_gpu_cleanup();
    return 0;
}

int main(void) {
    if (test_init_multi_n1()) return 1;
    if (test_init_multi_rejects_bad_cfg()) return 1;
    if (test_set_current_device()) return 1;
    if (test_tensor_alloc_on_tier0()) return 1;
    if (test_tensor_alloc_on_invalid()) return 1;
    if (test_init_multi_n2_or_skip()) return 1;
    if (test_copy_xdev_same_device()) return 1;
    if (test_copy_xdev_default_and_ordered_same_device()) return 1;
    if (test_copy_xdev3_same_device()) return 1;
    if (test_add_xdev_same_device()) return 1;
    if (test_add_xdev_cross_device_or_skip()) return 1;
    if (test_wait_xdev()) return 1;
    if (test_peer_bytecheck_baseline()) return 1;
    if (test_peer_bytecheck_corrupt_is_caught()) return 1;
    if (test_peer_bytecheck_pattern_variation_matters()) return 1;
    if (test_probe_auto_success()) return 1;
    if (test_probe_auto_null_out()) return 1;
    if (test_probe_auto_bad_filter()) return 1;
    if (test_probe_auto_reserve_applied()) return 1;
    fprintf(stderr, "test_sycl_mgpu: all tests passed\n");
    return 0;
}
