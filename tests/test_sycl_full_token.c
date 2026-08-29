/* Full single-token decode test for the SYCL backend.
 *
 * tests/test_sycl_full_layer.c proved one DeepSeek V4 Flash decoder
 * layer encodes through the engine's real per-token path
 * (metal_graph_encode_decode_layer). A real decode token needs
 * metal_graph_encode_token_raw_swa (ds4.c:27485), which runs ALL layers
 * plus the output head and then reads logits back
 * (metal_graph_eval_token_raw_swa, ds4.c:30748). Everything in that
 * remaining stretch is implemented and unit-tested and had never executed
 * before this test.
 *
 * This test drives ds4_test_graph_full_token_encode (ds4.c, DS4_TEST_HOOKS),
 * which calls metal_graph_eval_token_raw_swa directly -- the exact function
 * every real decode token calls -- on a synthetic DS4_TW_N_LAYER-layer
 * (4-layer) Flash shape built by tests/test_sycl_layer_weights.h, with no
 * model file. DS4_TW_N_HASH_LAYER (3) of those layers route by a hash
 * lookup table (ffn_gate_tid2eid) instead of top-k, so this is also the
 * first test on this backend to exercise hash routing and the first to
 * exercise the layer-to-layer hyper-connection carry (a single layer, as
 * test_sycl_full_layer.c drives, cannot thread state between iterations of
 * a loop it never runs).
 *
 * Four assertions, in order of value:
 *   1. The token encode returns success at all.
 *   2. The logits read back are finite and not all zero. This is a
 *      different claim from (1): an unstaged mmap read can return zeros
 *      and still report success, and an output tensor that is allocated
 *      but never read back proves nothing.
 *   3. The logits change when the input token changes.
 *   4. Determinism: encoding the same token twice produces bit-identical
 *      logits. ds4_sycl.cpp builds queues without the in_order property,
 *      and a queue-ordering race has already been found and fixed once on
 *      this backend; running the same input twice and requiring identical
 *      output is the cheapest race detector available. This is checked
 *      with several independent repeats, not a single pair, and any
 *      mismatch is reported as a flaky result rather than absorbed into a
 *      tolerance. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "test_sycl_layer_weights.h"

#include <math.h>
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
 * header, following the same pattern as ds4_test_graph_full_layer_encode
 * in tests/test_sycl_full_layer.c. */
int ds4_test_graph_full_token_encode(int token, float *out_logits, uint64_t out_logits_floats);

/* Repeats a full-token encode of the same token DS4_TW_N_VOCAB times over
 * (kRepeats independent encodes, each with a freshly built graph and
 * synthetic weights), returning 1 if every repeat succeeded and produced
 * bit-identical logits to the first, 0 on the first failure or mismatch. A
 * mismatch prints which repeat first diverged and at which vocab index, so
 * a flaky run is reported as a flaky run rather than silently retried. */
#define DS4_FULL_TOKEN_DETERMINISM_REPEATS 5

static int check_determinism(int token, const float *baseline) {
    for (int r = 0; r < DS4_FULL_TOKEN_DETERMINISM_REPEATS; r++) {
        float logits[DS4_TW_N_VOCAB];
        if (!ds4_test_graph_full_token_encode(token, logits, DS4_TW_N_VOCAB)) {
            fprintf(stderr,
                    "FAIL: determinism repeat %d of token %d failed to encode\n",
                    r, token);
            return 0;
        }
        for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
            if (!isfinite(logits[i])) {
                fprintf(stderr,
                        "FAIL: determinism repeat %d produced a non-finite logit at index %llu\n",
                        r, (unsigned long long)i);
                return 0;
            }
        }
        if (memcmp(logits, baseline, sizeof(logits)) != 0) {
            uint64_t first_diff = 0;
            for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
                if (logits[i] != baseline[i]) { first_diff = i; break; }
            }
            fprintf(stderr,
                    "FAIL: determinism repeat %d of token %d diverged from the "
                    "baseline at vocab index %llu (baseline=%.9g, repeat=%.9g) -- "
                    "this is a FLAKY result, not a clean pass or a clean fail\n",
                    r, token, (unsigned long long)first_diff,
                    (double)baseline[first_diff], (double)logits[first_diff]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    float logits_a[DS4_TW_N_VOCAB];
    float logits_b[DS4_TW_N_VOCAB];
    memset(logits_a, 0, sizeof(logits_a));
    memset(logits_b, 0, sizeof(logits_b));

    /* Assertion 1: a full token -- all layers, both hash-routed and
     * top-k-routed, plus the output head -- encodes and reads logits back
     * successfully. Nothing on this backend has ever established this
     * before. */
    CHECK(ds4_test_graph_full_token_encode(/*token=*/3, logits_a, DS4_TW_N_VOCAB) != 0,
          "a full DeepSeek V4 Flash token failed to encode end to end "
          "(alloc/begin_commands/embed/layers/output-head/end_commands/"
          "readback)");

    /* Assertion 2: the values actually read back are finite and not all
     * zero, not just that the call reported success. */
    int any_nonzero = 0;
    for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
        CHECK(isfinite(logits_a[i]), "token logits contain a non-finite value");
        if (logits_a[i] != 0.0f) any_nonzero = 1;
    }
    CHECK(any_nonzero, "token logits are all zero");

    /* Assertion 3: encoding a different token produces different logits. */
    CHECK(ds4_test_graph_full_token_encode(/*token=*/97, logits_b, DS4_TW_N_VOCAB) != 0,
          "a full-token encode with a different input token failed");

    int any_diff = 0;
    for (uint64_t i = 0; i < DS4_TW_N_VOCAB; i++) {
        CHECK(isfinite(logits_b[i]),
              "token logits for the second token contain a non-finite value");
        if (logits_a[i] != logits_b[i]) any_diff = 1;
    }
    CHECK(any_diff, "token logits did not change when the input token changed");

    /* Assertion 4: the same token, encoded independently several times
     * (fresh graph and fresh synthetic weights each time), produces
     * bit-identical logits every time. ds4_sycl.cpp builds queues without
     * the in_order property; this is the cheapest available detector for
     * a queue-ordering race across a graph with enough concurrent work
     * (four layers of routed MoE) to expose one. */
    CHECK(check_determinism(/*token=*/3, logits_a) != 0,
          "the same token did not produce bit-identical logits across "
          "repeated encodes -- see the diagnostic above for whether this "
          "was a clean failure or a flaky one");

    fprintf(stderr,
            "  test_sycl_full_token OK (determinism stable across %d repeats)\n",
            DS4_FULL_TOKEN_DETERMINISM_REPEATS);
    ds4_gpu_cleanup();
    return 0;
}
