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
    ds4_gpu_cleanup();
    return 0;
}
