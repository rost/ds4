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
 * of them regress to a failing stub, this test fails.
 *
 * ds4_test_graph_encode_smoke goes one level deeper still: allocation alone
 * does not prove the engine can ENCODE anything into the graph it just
 * built. That gap was real, not hypothetical, in the same way as the
 * allocator gap above: ds4_gpu_begin_commands, _commands_active,
 * _end_commands, _flush_commands, _flush_encoder and _synchronize were all
 * stubbed to their own failure value, and metal_graph_eval_token_raw_swa --
 * the real per-token decode path -- gates its entire encode step behind
 * begin_commands succeeding and its result behind end_commands succeeding,
 * with no backend gate. That stub combination failed every decode token on
 * this backend before a single kernel ever ran. */

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

/* DS4_TEST_HOOKS entry points defined in ds4.c; not declared in a shipped
 * header, following the same pattern as ds4_test_session_read_logits and
 * ds4_test_engine_placement in tests/test_engine_mgpu_runtime.c. */
int ds4_test_graph_alloc_smoke(void);
int ds4_test_graph_encode_smoke(void);

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    CHECK(ds4_test_graph_alloc_smoke() != 0,
          "metal_graph_alloc_raw_cap failed on a synthetic single-tier "
          "shape; the engine's real session-creation path cannot allocate "
          "a graph on this backend");

    /* One level deeper than the allocation-only check above: drives
     * ds4_gpu_begin_commands -> a real kernel dispatch
     * (ds4_gpu_embed_token_hc_tensor) -> ds4_gpu_end_commands ->
     * ds4_gpu_synchronize, the exact sequence metal_graph_eval_token_raw_swa
     * runs for every real decode token (ds4.c:30765-30792). All six
     * ds4_gpu_begin_commands/_commands_active/_end_commands/_flush_commands/
     * _flush_encoder/_synchronize entries were previously stubbed to their
     * own failure value, which meant this exact call sequence -- and every
     * real decode token -- failed before a single kernel ran. */
    CHECK(ds4_test_graph_encode_smoke() != 0,
          "begin_commands/embed_token_hc/end_commands/synchronize failed on "
          "a synthetic single-layer graph; the engine's real decode path "
          "cannot encode a token on this backend");

    /* Direct ABI contract checks for the command-lifecycle entries
     * themselves, independent of the engine call above: begin_commands and
     * commands_active are trivial constants (this backend has no encoder
     * object to track), matching both ROCm and CUDA exactly. */
    CHECK(ds4_gpu_begin_commands() != 0, "begin_commands reported failure");
    CHECK(ds4_gpu_commands_active() == 0,
          "commands_active reported an outstanding batch this backend never "
          "tracks");
    CHECK(ds4_gpu_end_commands() != 0, "end_commands reported failure");
    CHECK(ds4_gpu_flush_commands() != 0, "flush_commands reported failure");
    CHECK(ds4_gpu_flush_encoder() != 0, "flush_encoder reported failure");
    CHECK(ds4_gpu_synchronize() != 0, "synchronize reported failure");

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

    /* ds4_gpu_cache_model_range (ds4.c:2978) is reached unconditionally at
     * engine open on the CUDA branch, which SYCL takes, unless
     * DS4_CUDA_DIRECT_MODEL is set. The caller's own error text calls this
     * cache "optional" (ds4.c:58749, "failed to prepare optional model
     * cache"), but a stub returning failure still aborts ds4_engine_open at
     * :58746-58756. This backend keeps no device-resident model copy, so
     * "not cached" is a legitimate answer and must be reported as success,
     * the same shape as cache_q8_f16_range above. */
    CHECK(ds4_gpu_cache_model_range(fake_model, sizeof(fake_model), 0,
                                    sizeof(fake_model), "test") != 0,
          "cache_model_range rejected a valid whole-file range");
    CHECK(ds4_gpu_cache_model_range(NULL, sizeof(fake_model), 0, 0,
                                    "test") != 0,
          "cache_model_range rejected a null map with zero bytes");
    CHECK(ds4_gpu_cache_model_range(fake_model, sizeof(fake_model),
                                    sizeof(fake_model), 1, "test") == 0,
          "cache_model_range accepted a range past the end");
    CHECK(ds4_gpu_cache_model_range(fake_model, sizeof(fake_model),
                                    0xFFFFFFFFFFFFFF00ULL, 0x200ULL,
                                    "test") == 0,
          "cache_model_range accepted an offset+size that overflows");

    ds4_gpu_cleanup();
    return 0;
}
