/* Multi-tier placement tests for the SYCL backend: ds4_gpu_tier_free_vram,
 * ds4_gpu_register_model_map_no_copy, ds4_gpu_device_cache_tensors and the
 * ds4_gpu_q8_cache_suppressed pair (sycl/ds4_sycl_placement.hpp).
 *
 * This machine has one A770, so nothing here can exercise a placement that
 * genuinely spans two devices, cross-device caching, or a real CPU-spill
 * decision made from a real second tier's free-VRAM reading. Every test
 * below either runs for real against the one available tier (tier_free_vram,
 * the selective cache's admission/growth/eviction, the classify-acceptance
 * proof, and the physical-device-id to logical-tier translation, which one
 * tier holding a non-zero physical id is enough to exercise) or validates
 * argument handling and bounds independent of device count. See the report
 * for the exact list of what remains untestable until a second real device
 * exists.
 *
 * Deliberately self-contained: no shared harness header, matching
 * tests/test_sycl_mgpu.c and tests/test_sycl_fp8_kv.c. Needs no model
 * file. Links ds4_sycl_test_hooks.o (ds4.c under DS4_TEST_HOOKS) so it can
 * drive ds4_test_classify_multi_tier directly, the same hook
 * tests/test_engine_mgpu_placement.c uses. */

#include "ds4_gpu.h"
#include "ds4_gpu_args.h"
#include "ds4_gpu_mgpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                      \
        }                                                                   \
    } while (0)

/* sycl/ds4_sycl_placement.hpp test-only hooks; not part of the ABI. */
extern int      ds4_sycl_device_count(void);
extern int      ds4_sycl_test_model_range_is_cached(const void *model_map,
                                                   uint64_t model_size,
                                                   uint64_t offset, uint64_t bytes);
extern uint64_t ds4_sycl_test_placement_cache_bytes(int tier);
extern int      ds4_sycl_test_placement_read_back(int tier, uint64_t source_offset,
                                                   uint64_t bytes, void *out);
extern int      ds4_sycl_test_tier_for_device_id(int device_id);

/* How many GPUs the machine actually has: ds4_gpu_init is a single-device
 * shim, so ds4_sycl_device_count after it always reports 1. Same probe
 * tests/test_sycl_mgpu.c uses, and the same enumeration --gpu-vram auto
 * itself performs, so it sees what a real multi-GPU startup would. */
static int physical_device_count(void) {
    ds4_gpu_config cfg;
    char errbuf[256];
    memset(&cfg, 0, sizeof(cfg));
    errbuf[0] = '\0';
    if (ds4_gpu_args_probe_auto_cuda(NULL, 0, &cfg, 0, errbuf, sizeof(errbuf)) != 0) {
        return 0;
    }
    return cfg.n_gpus;
}

/* DS4_TEST_HOOKS entry points defined in ds4.c, matching the declarations
 * in tests/test_engine_mgpu_placement.c. */
typedef struct {
    const char *name;
    uint64_t    bytes;
} ds4_test_fake_tensor;

int ds4_test_classify_multi_tier(const ds4_test_fake_tensor *tensors,
                                  int n_tensors,
                                  const ds4_gpu_config *cfg,
                                  int placement_out[],
                                  int *out_multi_tier,
                                  int *out_n_entries);

/* Matches DS4_N_LAYER's default (g_ds4_shape = DS4_SHAPE_FLASH, ds4.c:551)
 * before any model is loaded, same constant test_engine_mgpu_placement.c
 * uses. */
#define DS4_N_LAYER_LOCAL 43
#define DS4_N_ENTRIES (DS4_N_LAYER_LOCAL + 2)

/* ---- ds4_gpu_tier_free_vram --------------------------------------------- */

static int test_tier_free_vram_bounds(void) {
    CHECK(ds4_gpu_tier_free_vram(0) == 0,
          "tier_free_vram before init (no devices) is 0, not garbage");
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for tier_free_vram bounds");

    CHECK(ds4_gpu_tier_free_vram(-1) == 0, "negative tier is 0");
    /* The first out-of-range tier is the device count, not a hardcoded 1:
     * on a multi-GPU host tier 1 is a real device with real free VRAM.
     * ds4_gpu_init() ran just above, so the count is live here. */
    CHECK(ds4_gpu_tier_free_vram(ds4_sycl_device_count()) == 0,
          "the first out-of-range tier is 0");

    const uint64_t free0 = ds4_gpu_tier_free_vram(0);
    CHECK(free0 > 0, "tier 0 free VRAM is nonzero on a real device");

    ds4_gpu_cleanup();
    return 0;
}

/* The central claim, proven for real: a number that only ever
 * reflects what THIS backend itself committed must decrease by exactly
 * what was just admitted, on the one real tier available. This is the
 * spec-6p-driven design (static ceiling minus self-tracked committed
 * bytes, not a live Sysman reading) actually holding up against real
 * ds4_gpu_device_cache_tensors admissions, not just against arithmetic. */
static int test_tier_free_vram_decreases_with_real_admission(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for admission test");

    unsigned char *host = malloc(4096);
    CHECK(host != NULL, "host buffer alloc");
    memset(host, 0xAB, 4096);

    CHECK(ds4_gpu_register_model_map_no_copy(host, 4096) != 0,
          "register_model_map_no_copy for admission test");

    const uint64_t free_before = ds4_gpu_tier_free_vram(0);
    CHECK(free_before > 4096, "tier has room for a 4 KiB admission");

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = 4096;
    r.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 1) == 0,
          "device_cache_tensors admits a valid 4 KiB range");

    const uint64_t free_after = ds4_gpu_tier_free_vram(0);
    CHECK(free_before - free_after == 4096,
          "tier_free_vram drops by exactly the admitted byte count");

    free(host);
    ds4_gpu_cleanup();
    return 0;
}

/* Weight resolution must find a range installed by
 * ds4_gpu_device_cache_tensors, not only one installed by
 * ds4_gpu_cache_model_range.  Those are two separate per-tier caches, and
 * resolution used to consult only the second: on the multi-GPU path,
 * which fills the first, every lookup therefore missed and every weight
 * was re-staged from the host mmap per call, per layer, per token, while
 * the resident copy sat unused in VRAM.  A profiled 14-GPU run measured
 * hits=0 misses=1434 per forward pass with 86% of all device time going
 * to those redundant copies. */
static int test_device_cached_range_resolves(void) {
    ds4_gpu_cleanup();
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for placement resolve test");

    unsigned char *host = malloc(4096);
    CHECK(host != NULL, "host buffer alloc");
    memset(host, 0xCD, 4096);
    CHECK(ds4_gpu_register_model_map_no_copy(host, 4096) != 0,
          "register_model_map_no_copy for placement resolve test");

    CHECK(ds4_sycl_test_model_range_is_cached(host, 4096, 0, 4096) == 0,
          "range must not resolve before it is installed");

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = 4096;
    r.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 1) == 0,
          "device_cache_tensors admits the range");

    CHECK(ds4_sycl_test_model_range_is_cached(host, 4096, 0, 4096) == 1,
          "an installed range must resolve as device-resident");
    CHECK(ds4_sycl_test_model_range_is_cached(host, 4096, 1024, 512) == 1,
          "a sub-range wholly inside it must resolve too");
    CHECK(ds4_sycl_test_model_range_is_cached(host, 4096, 2048, 4096) == 0,
          "a range running past the installed end must not resolve");

    free(host);
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_device_cached_range_resolves OK\n");
    return 0;
}

/* ---- ds4_gpu_register_model_map_no_copy --------------------------------- */

static int test_register_model_map_no_copy_args(void) {
    CHECK(ds4_gpu_register_model_map_no_copy(NULL, 4096) == 0,
          "NULL model_map fails");
    unsigned char buf[16];
    CHECK(ds4_gpu_register_model_map_no_copy(buf, 0) == 0,
          "zero model_size fails");
    CHECK(ds4_gpu_register_model_map_no_copy(buf, sizeof(buf)) != 0,
          "valid args succeed");
    CHECK(ds4_gpu_register_model_map_no_copy(buf, sizeof(buf)) != 0,
          "re-registering the identical map+size is idempotent success");
    return 0;
}

/* ---- ds4_gpu_device_cache_tensors: argument validation ------------------ */

static int test_device_cache_tensors_arg_validation(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for arg validation");

    unsigned char host[64];
    memset(host, 0, sizeof(host));
    CHECK(ds4_gpu_register_model_map_no_copy(host, sizeof(host)) != 0,
          "register for arg validation");

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = 16;
    r.target_device = 0;

    CHECK(ds4_gpu_device_cache_tensors(-1, &r, 1) != 0,
          "negative device_id fails");
    CHECK(ds4_gpu_device_cache_tensors(ds4_sycl_device_count(), &r, 1) != 0,
          "device_id one past the last device fails");
    CHECK(ds4_gpu_device_cache_tensors(0, NULL, -1) != 0,
          "n_ranges < 0 with NULL ranges fails");
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 0) == 0,
          "n_ranges == 0 is a no-op success");

    ds4_tensor_range bad_off = r;
    bad_off.source_offset = sizeof(host) + 1;
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_off, 1) != 0,
          "source_offset past the registered mapping fails");

    ds4_tensor_range bad_bytes = r;
    bad_bytes.source_offset = sizeof(host) - 4;
    bad_bytes.bytes = 8; /* extends 4 bytes past the mapping */
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_bytes, 1) != 0,
          "range extending past the registered mapping fails");

    ds4_gpu_cleanup();
    return 0;
}

/* ---- ds4_gpu_device_cache_tensors: real admission, growth and read-back  */

static int test_device_cache_tensors_admission_and_readback(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for admission/readback");

    unsigned char host[8192];
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 37u + 5u);
    CHECK(ds4_gpu_register_model_map_no_copy(host, sizeof(host)) != 0,
          "register for admission/readback");

    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 0,
          "cache starts empty on a freshly registered map");

    ds4_tensor_range first;
    first.source_offset = 0;
    first.bytes = 1024;
    first.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &first, 1) == 0,
          "first range admitted");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 1024,
          "cache_bytes reflects the first admission");

    unsigned char readback1[1024];
    memset(readback1, 0, sizeof(readback1));
    CHECK(ds4_sycl_test_placement_read_back(0, 0, 1024, readback1) == 1,
          "read-back hits the first admitted range");
    CHECK(memcmp(readback1, host, 1024) == 0,
          "read-back matches the original host bytes exactly");

    /* Second admission forces the slab to grow-and-rebase: exercises the
     * d2d copy of the OLD range's bytes and the pointer-rebase loop that
     * keeps the first range's device_ptr correct after the slab moves.
     * This can only be proven this thoroughly with a real device, which
     * this box has. */
    ds4_tensor_range second;
    second.source_offset = 4096;
    second.bytes = 2048;
    second.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &second, 1) == 0,
          "second range admitted (forces slab growth)");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 1024 + 2048,
          "cache_bytes reflects both admissions after growth");

    unsigned char readback1_again[1024];
    memset(readback1_again, 0, sizeof(readback1_again));
    CHECK(ds4_sycl_test_placement_read_back(0, 0, 1024, readback1_again) == 1,
          "first range still resolves correctly after growth (rebase)");
    CHECK(memcmp(readback1_again, host, 1024) == 0,
          "first range's bytes survive the grow-and-rebase unchanged");

    unsigned char readback2[2048];
    memset(readback2, 0, sizeof(readback2));
    CHECK(ds4_sycl_test_placement_read_back(0, 4096, 2048, readback2) == 1,
          "second range resolves correctly");
    CHECK(memcmp(readback2, host + 4096, 2048) == 0,
          "second range's bytes match the original host bytes");

    unsigned char miss[16];
    CHECK(ds4_sycl_test_placement_read_back(0, 2048, 16, miss) == 0,
          "an uncached offset is a genuine miss, not a false hit");

    ds4_gpu_cleanup();
    return 0;
}

/* Re-registering a DIFFERENT model map must release every existing
 * per-device admission: entries are keyed by offset into the OLD mapping
 * and would silently resolve to the wrong bytes under a new one. */
static int test_register_model_map_no_copy_evicts_old_cache(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for eviction test");

    unsigned char host_a[256];
    memset(host_a, 0x11, sizeof(host_a));
    CHECK(ds4_gpu_register_model_map_no_copy(host_a, sizeof(host_a)) != 0,
          "register host_a");

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = 128;
    r.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 1) == 0, "admit into host_a's cache");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 128,
          "cache_bytes nonzero before re-registration");

    unsigned char host_b[256];
    memset(host_b, 0x22, sizeof(host_b));
    CHECK(ds4_gpu_register_model_map_no_copy(host_b, sizeof(host_b)) != 0,
          "register a DIFFERENT map (host_b)");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 0,
          "re-registration releases the previous cache");

    ds4_gpu_cleanup();
    return 0;
}

/* Headroom refusal: a range that fits the registered host mapping but
 * exceeds the tier's own tier_free_vram must be refused before any
 * device allocation or copy is attempted (never dereferences the
 * oversized fake host pointer, since the refusal is upfront). */
static int test_device_cache_tensors_headroom_refusal(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for headroom refusal");

    /* A real, small backing allocation stands in for the host mmap: the
     * declared model_size below is far larger than this allocation, but
     * the refusal this test targets happens before any byte of it would
     * be read. */
    unsigned char *host = malloc(4096);
    CHECK(host != NULL, "host buffer alloc for headroom refusal");
    const uint64_t huge_model_size = 200ull * 1024 * 1024 * 1024;
    CHECK(ds4_gpu_register_model_map_no_copy(host, huge_model_size) != 0,
          "register an oversized (but unread) model size");

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = huge_model_size - 4096; /* within the declared mapping */
    r.target_device = 0;
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 1) != 0,
          "a range exceeding real tier headroom is refused");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 0,
          "a refused admission commits nothing");

    free(host);
    ds4_gpu_cleanup();
    return 0;
}

/* ---- non-contiguous --gpu-devices --------------------------------------- */

/* ds4_gpu_device_cache_tensors takes a PHYSICAL device id, the value the
 * operator passed in --gpu-devices, and so does every ranges[i].
 * target_device (ds4.c:57024, ds4.c:57197; CUDA reads both the same way,
 * ds4_cuda.cu:3947-3977). Everything inside the SYCL backend is addressed
 * by LOGICAL tier instead, and the entry used to treat the two as the
 * same integer -- correct only for a device list contiguous from zero.
 *
 * This box has one GPU, so a genuine multi-card ordering cannot be
 * initialised here. What CAN be exercised for real is the case that
 * actually broke: a configured tier whose physical id is not its tier
 * index. Reassigning g_gpu[0].device_id after init is exactly the state
 * `--gpu-devices 6` on a 14-card host leaves behind -- one tier, holding
 * a card whose physical id is 6 -- and nothing else in this backend reads
 * g_gpu[].device_id, so the one real device still backs tier 0.
 *
 * Before the fix this test failed twice over: device 6 was rejected as
 * out of range, and had it been accepted the slab would have been
 * installed in g_placement_tier[6] behind ds4_sycl_queue(6). */
static int test_device_cache_tensors_physical_id_is_not_tier(void) {
    ds4_gpu_cleanup();
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for physical-id translation");

    const int saved_device_id = g_gpu[0].device_id;
    g_gpu[0].device_id = 6;

    int rc = 0;
    unsigned char host[2048];
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 53u + 7u);

    if (ds4_gpu_register_model_map_no_copy(host, sizeof(host)) == 0) {
        fprintf(stderr, "FAIL: register_model_map_no_copy for physical-id test\n");
        rc = 1;
    }

    ds4_tensor_range r;
    r.source_offset = 0;
    r.bytes = 1024;
    r.target_device = 6;

    /* Tier 0's index is no longer a device this run configured. */
    if (rc == 0 && ds4_gpu_device_cache_tensors(0, &r, 1) == 0) {
        fprintf(stderr, "FAIL: an unconfigured physical device id must be refused\n");
        rc = 1;
    }
    if (rc == 0 && ds4_sycl_test_tier_for_device_id(6) != 0) {
        fprintf(stderr, "FAIL: physical device 6 must resolve to tier 0\n");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_device_cache_tensors(6, &r, 1) != 0) {
        fprintf(stderr, "FAIL: the configured physical device id must be accepted\n");
        rc = 1;
    }
    if (rc == 0 && ds4_sycl_test_placement_cache_bytes(0) != 1024) {
        fprintf(stderr, "FAIL: the slab must be installed on tier 0, not slot 6\n");
        rc = 1;
    }
    if (rc == 0 && ds4_sycl_test_placement_cache_bytes(6) != 0) {
        fprintf(stderr, "FAIL: nothing may be installed in the physical id's slot\n");
        rc = 1;
    }

    /* The bytes really are on tier 0's device, not merely accounted there. */
    unsigned char readback[1024];
    memset(readback, 0, sizeof(readback));
    if (rc == 0 && ds4_sycl_test_placement_read_back(0, 0, 1024, readback) != 1) {
        fprintf(stderr, "FAIL: read-back from tier 0 must hit the installed range\n");
        rc = 1;
    }
    if (rc == 0 && memcmp(readback, host, sizeof(readback)) != 0) {
        fprintf(stderr, "FAIL: tier 0's copy must match the host bytes exactly\n");
        rc = 1;
    }
    /* And weight resolution, which keys off the CURRENT tier, finds them. */
    if (rc == 0 && ds4_sycl_test_model_range_is_cached(host, sizeof(host), 0, 1024) != 1) {
        fprintf(stderr, "FAIL: the installed range must resolve as device-resident\n");
        rc = 1;
    }

    g_gpu[0].device_id = saved_device_id;
    ds4_gpu_cleanup();
    if (rc == 0) {
        fprintf(stderr, "  test_device_cache_tensors_physical_id_is_not_tier OK\n");
    }
    return rc;
}

/* The same claim against a genuinely multi-card, genuinely non-contiguous
 * ordering: --gpu-devices 1,0 makes tier 0 hold physical device 1 and
 * tier 1 hold physical device 0, so a range stamped target_device=0 must
 * be installed on tier 1. Needs two real devices; skips cleanly on this
 * A770 rather than faking the coverage. First thing to check on the
 * 14-card B60 host, ideally widened there to the full
 * --gpu-devices 0,2,4,6,1,3,5,7 topology --cuda-tensor-parallel wants. */
static int test_device_cache_tensors_non_contiguous_devices_or_skip(void) {
    const int have = physical_device_count();
    if (have < 2) {
        fprintf(stderr, "skip: non-contiguous device list wanted 2 GPUs, have %d "
                        "(verify this case on the B60 machine)\n", have);
        return 0;
    }
    ds4_gpu_cleanup();

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 1;
    cfg.device_indices[1] = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "init_multi with a reversed device list");
    CHECK(g_gpu[0].device_id == 1, "tier 0 holds physical device 1");
    CHECK(g_gpu[1].device_id == 0, "tier 1 holds physical device 0");
    CHECK(ds4_sycl_test_tier_for_device_id(0) == 1, "physical device 0 is tier 1");
    CHECK(ds4_sycl_test_tier_for_device_id(1) == 0, "physical device 1 is tier 0");

    unsigned char host[2048];
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 29u + 3u);
    CHECK(ds4_gpu_register_model_map_no_copy(host, sizeof(host)) != 0,
          "register for the reversed device list");

    ds4_tensor_range r;
    r.source_offset = 512;
    r.bytes = 1024;
    r.target_device = 0; /* physical device 0, which is tier 1 */
    CHECK(ds4_gpu_device_cache_tensors(0, &r, 1) == 0,
          "physical device 0's ranges are admitted");
    CHECK(ds4_sycl_test_placement_cache_bytes(1) == 1024,
          "they are installed on tier 1, the tier that owns physical device 0");
    CHECK(ds4_sycl_test_placement_cache_bytes(0) == 0,
          "tier 0 must not have received another tier's weights");

    unsigned char readback[1024];
    memset(readback, 0, sizeof(readback));
    CHECK(ds4_sycl_test_placement_read_back(1, 512, 1024, readback) == 1,
          "read-back from tier 1 hits the installed range");
    CHECK(memcmp(readback, host + 512, sizeof(readback)) == 0,
          "tier 1's copy matches the host bytes exactly");

    ds4_gpu_cleanup();
    fprintf(stderr, "  test_device_cache_tensors_non_contiguous_devices OK\n");
    return 0;
}

/* ---- ds4_gpu_q8_cache_suppressed pair ------------------------------------ */

static int test_q8_cache_suppressed(void) {
    CHECK(ds4_gpu_q8_cache_suppressed() == 0, "starts unsuppressed");
    ds4_gpu_set_q8_cache_suppressed(1);
    CHECK(ds4_gpu_q8_cache_suppressed() != 0, "set(1) suppresses");
    ds4_gpu_set_q8_cache_suppressed(0);
    CHECK(ds4_gpu_q8_cache_suppressed() == 0, "set(0) unsuppresses");
    ds4_gpu_set_q8_cache_suppressed(5);
    CHECK(ds4_gpu_q8_cache_suppressed() == 1, "any nonzero clamps to 1");
    ds4_gpu_set_q8_cache_suppressed(0);
    return 0;
}

/* ---- The linchpin claim: engine_classify_multi_tier accepts a
 * config built from ds4_gpu_tier_free_vram rather than refusing it for
 * an all-zero budget. Only one real tier exists here, so this cannot
 * prove a genuine cross-tier split; it proves the
 * chain CLI-shaped config -> tier_free_vram -> classify does not refuse
 * on real hardware, which a zero or wrong tier_free_vram would break. */
static int build_tiny_synthetic_model(ds4_test_fake_tensor *out, int cap) {
    int n = 0;
    static char names[DS4_N_LAYER_LOCAL * 2 + 3][32];
    if (cap < DS4_N_LAYER_LOCAL * 2 + 3) return -1;

    snprintf(names[n], 32, "token_embd.weight");
    out[n].name = names[n];
    out[n].bytes = 8ull * 1024 * 1024;
    n++;

    for (int il = 0; il < DS4_N_LAYER_LOCAL; il++) {
        snprintf(names[n], 32, "blk.%d.attn_q.weight", il);
        out[n].name = names[n];
        out[n].bytes = 1ull * 1024 * 1024;
        n++;
        snprintf(names[n], 32, "blk.%d.ffn_down.weight", il);
        out[n].name = names[n];
        out[n].bytes = 2ull * 1024 * 1024;
        n++;
    }

    snprintf(names[n], 32, "output.weight");
    out[n].name = names[n];
    out[n].bytes = 4ull * 1024 * 1024;
    n++;
    snprintf(names[n], 32, "output_norm.weight");
    out[n].name = names[n];
    out[n].bytes = 64ull * 1024;
    n++;
    return n;
}

static int test_classify_accepts_tier_free_vram_config(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for classify-acceptance");

    const uint64_t free0 = ds4_gpu_tier_free_vram(0);
    CHECK(free0 > 0, "tier_free_vram must be nonzero to build a real config");

    ds4_test_fake_tensor tensors[DS4_N_LAYER_LOCAL * 2 + 3];
    int n = build_tiny_synthetic_model(tensors, DS4_N_LAYER_LOCAL * 2 + 3);
    CHECK(n > 0, "tiny synthetic model built");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    cfg.vram_bytes[0] = (size_t)free0;
    cfg.safety_margin_bytes = 0;

    int placement[DS4_N_ENTRIES];
    int multi_tier = 99;
    int n_entries = 0;
    int rc = ds4_test_classify_multi_tier(tensors, n, &cfg, placement,
                                          &multi_tier, &n_entries);
    CHECK(rc == 0,
          "engine_classify_multi_tier accepts a config built from "
          "ds4_gpu_tier_free_vram instead of refusing it for a zero budget");
    CHECK(n_entries == DS4_N_ENTRIES, "n_entries == DS4_N_LAYER + 2");

    ds4_gpu_cleanup();
    return 0;
}

int main(void) {
    if (test_tier_free_vram_bounds()) return 1;
    if (test_tier_free_vram_decreases_with_real_admission()) return 1;
    if (test_device_cached_range_resolves()) return 1;
    if (test_register_model_map_no_copy_args()) return 1;
    if (test_device_cache_tensors_arg_validation()) return 1;
    if (test_device_cache_tensors_admission_and_readback()) return 1;
    if (test_register_model_map_no_copy_evicts_old_cache()) return 1;
    if (test_device_cache_tensors_headroom_refusal()) return 1;
    if (test_device_cache_tensors_physical_id_is_not_tier()) return 1;
    if (test_device_cache_tensors_non_contiguous_devices_or_skip()) return 1;
    if (test_q8_cache_suppressed()) return 1;
    if (test_classify_accepts_tier_free_vram_config()) return 1;
    fprintf(stderr, "test_sycl_placement: all tests passed\n");
    return 0;
}
