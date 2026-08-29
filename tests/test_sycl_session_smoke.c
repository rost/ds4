/* Session-creation smoke test for the SYCL backend.
 *
 * Every other SYCL test in this suite calls a ds4_gpu_* ABI entry directly
 * with a hand-built buffer. None of them ever runs the engine's own graph
 * allocator in ds4.c, so "the engine calls a ds4_gpu_* entry this backend
 * left stubbed, or implemented with the wrong contract" was a whole class
 * of defect invisible to 162 passing tests. That is not hypothetical: it is
 * exactly how ds4_gpu_tensor_alloc_ptr_on/_managed_on shipped stubbed to an
 * unconditional NULL for tier 0, which meant this backend could not create
 * a session at all, on any GPU count, the first time a real session was
 * attempted.
 *
 * This test drives ds4_test_graph_alloc_smoke (ds4.c, DS4_TEST_HOOKS),
 * which calls metal_graph_alloc_raw_cap directly: the exact function
 * ds4_session_create calls to build a session's GPU graph, using a
 * synthetic single-layer shape and hand-built tensor descriptors so it
 * needs no model weights and no network access. metal_graph_alloc_raw_cap
 * only reads tensor dims to size its allocations, never tensor data, which
 * is what makes this reachable without a loaded model.
 *
 * A single call here exercises ds4_gpu_tensor_alloc_ptr_on dozens of times
 * across the per-tier scratch, per-tier decode/prefill state, and output
 * head allocation classes -- the exact call sites that were broken. If any
 * of them regress to a failing stub, this test fails. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* DS4_TEST_HOOKS entry point defined in ds4.c; not declared in a shipped
 * header, following the same pattern as ds4_test_session_read_logits and
 * ds4_test_engine_placement in tests/test_engine_mgpu_runtime.c. */
int ds4_test_graph_alloc_smoke(void);

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    CHECK(ds4_test_graph_alloc_smoke() != 0,
          "metal_graph_alloc_raw_cap failed on a synthetic single-tier "
          "shape; the engine's real session-creation path cannot allocate "
          "a graph on this backend");

    fprintf(stderr, "  test_sycl_session_smoke OK\n");
    /* ds4_engine_open calls these on every startup and ABORTS if any
     * reports failure (ds4.c:58352 and :58358 for the map registration,
     * ds4.c:3020 and :58405 for the optional Q8 preload).  All three were
     * stubbed to their failure value, so no real model could be opened on
     * this backend.  Nothing above catches that: the graph allocator runs
     * much later, and every other test calls an ABI entry directly. */
    unsigned char fake_model[4096];
    memset(fake_model, 0, sizeof(fake_model));

    CHECK(ds4_gpu_set_model_map_range(fake_model, sizeof(fake_model), 0,
                                      sizeof(fake_model), 1024) != 0,
          "set_model_map_range rejected a valid whole-file range");
    CHECK(ds4_gpu_set_model_map_range(NULL, sizeof(fake_model), 0, 0, 0) == 0,
          "set_model_map_range accepted a null map");
    CHECK(ds4_gpu_set_model_map_range(fake_model, sizeof(fake_model),
                                      sizeof(fake_model), 1, 0) == 0,
          "set_model_map_range accepted a range past the end");
    CHECK(ds4_gpu_set_model_map_range(fake_model, sizeof(fake_model),
                                      0xFFFFFFFFFFFFFF00ULL, 0x200ULL, 0) == 0,
          "set_model_map_range accepted an offset+size that overflows");

    const uint64_t span_off[2] = {0, 2048};
    const uint64_t span_len[2] = {2048, 2048};
    CHECK(ds4_gpu_set_model_map_spans(fake_model, sizeof(fake_model),
                                      span_off, span_len, 2, 1024) != 0,
          "set_model_map_spans rejected two valid spans");
    const uint64_t bad_len[2] = {2048, 4096};
    CHECK(ds4_gpu_set_model_map_spans(fake_model, sizeof(fake_model),
                                      span_off, bad_len, 2, 1024) == 0,
          "set_model_map_spans accepted a span past the end");

    /* The Q8 preload is optional: "this backend has no such cache" must be
     * reported as success, or startup aborts on a missing optimisation. */
    CHECK(ds4_gpu_cache_q8_f16_range(fake_model, sizeof(fake_model), 0, 1024,
                                     32, 32, "test") != 0,
          "cache_q8_f16_range reported failure for an unsupported preload");
    CHECK(ds4_gpu_cache_q8_f16_range(fake_model, sizeof(fake_model),
                                     sizeof(fake_model), 1, 32, 32, "test") == 0,
          "cache_q8_f16_range accepted a range past the end");

    ds4_gpu_cleanup();
    return 0;
}
