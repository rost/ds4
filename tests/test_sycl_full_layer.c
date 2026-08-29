/* Full decoder layer encode test for the SYCL backend.
 *
 * Every other SYCL test in this suite either calls one ds4_gpu_* ABI entry
 * directly, or (test_sycl_session_smoke) drives the engine's graph
 * allocator and one embedding kernel. Nothing has ever driven a full
 * DeepSeek V4 Flash decoder layer -- attention, hyper-connection, router,
 * routed MoE, shared expert -- through the engine's own real per-token
 * decode path on this backend. That is the single gap between "it builds"
 * and "it plausibly runs" this test closes.
 *
 * This test drives ds4_test_graph_full_layer_encode (ds4.c, DS4_TEST_HOOKS),
 * which calls metal_graph_encode_decode_layer (ds4.c:25120) -- the exact
 * wrapper metal_graph_encode_token_raw_swa calls for every layer of every
 * real decode token -- on a synthetic single-layer DeepSeek V4 Flash shape
 * built by tests/test_sycl_layer_weights.h, with no model file. The
 * quantisation mix matches Flash as shipped: Q8_0 dense weights, F16
 * router, IQ2_XXS gate/up and Q2_K down for the routed experts, F32 norms
 * and hyper-connection state.
 *
 * Three assertions, in order of value:
 *   1. The layer encode returns success at all.
 *   2. The readback of metal_graph_after_ffn_hc (this layer's real output
 *      hidden state) is finite and not all zero. This is a different claim
 *      from (1): an unstaged mmap read can return zeros and still report
 *      success, and an output tensor that is allocated but never read back
 *      proves nothing.
 *   3. The output changes when the input token changes. A layer that
 *      returns the same output regardless of its input has succeeded at
 *      nothing. */

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
 * header, following the same pattern as ds4_test_graph_alloc_smoke and
 * ds4_test_graph_encode_smoke in tests/test_sycl_session_smoke.c. */
int ds4_test_graph_full_layer_encode(int token, float *out_hc, uint64_t out_hc_floats);

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    float hc_token_a[DS4_TW_HC_DIM];
    float hc_token_b[DS4_TW_HC_DIM];
    memset(hc_token_a, 0, sizeof(hc_token_a));
    memset(hc_token_b, 0, sizeof(hc_token_b));

    /* Assertion 1: a full layer encode returns success. Nothing on this
     * backend has ever established this before. */
    CHECK(ds4_test_graph_full_layer_encode(/*token=*/3, hc_token_a,
                                           DS4_TW_HC_DIM) != 0,
          "a full DeepSeek V4 Flash decoder layer failed to encode "
          "end to end (alloc/begin_commands/embed/full-layer-encode/"
          "end_commands/synchronize/readback)");

    /* Assertion 2: the values actually read back are finite and not all
     * zero, not just that the call reported success. */
    int any_nonzero = 0;
    for (uint64_t i = 0; i < DS4_TW_HC_DIM; i++) {
        CHECK(isfinite(hc_token_a[i]),
              "full-layer output hyper-connection state contains a "
              "non-finite value");
        if (hc_token_a[i] != 0.0f) any_nonzero = 1;
    }
    CHECK(any_nonzero, "full-layer output hyper-connection state is all zero");

    /* Assertion 3: encoding a different token produces a different output.
     * Re-run with a different token and compare against the first result. */
    CHECK(ds4_test_graph_full_layer_encode(/*token=*/97, hc_token_b,
                                           DS4_TW_HC_DIM) != 0,
          "a full-layer encode with a different input token failed");

    int any_diff = 0;
    for (uint64_t i = 0; i < DS4_TW_HC_DIM; i++) {
        CHECK(isfinite(hc_token_b[i]),
              "full-layer output for the second token contains a "
              "non-finite value");
        if (hc_token_a[i] != hc_token_b[i]) any_diff = 1;
    }
    CHECK(any_diff,
          "full-layer output did not change when the input token changed");

    fprintf(stderr, "  test_sycl_full_layer OK\n");
    ds4_gpu_cleanup();
    return 0;
}
