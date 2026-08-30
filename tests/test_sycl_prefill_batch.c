/* Prefill-batch encode test for the SYCL backend.
 *
 * Every other SYCL engine-level test drives decode: one token, n_tokens ==
 * 1. Prefill is a different code path with different kernels, different
 * dispatch thresholds and a different attention implementation, and none of
 * it has ever run through the engine before this. This test drives
 * ds4_test_graph_prefill_layer_batch_encode (ds4.c, DS4_TEST_HOOKS), which
 * calls metal_graph_encode_layer_batch (ds4.c:30533) -- the plain,
 * non-pipelined batch path a single GPU always takes, since
 * metal_graph_build_prefill_stages (ds4.c:34132) always groups every layer
 * into one stage for a single-tier placement, and
 * metal_graph_prefill_pipeline_stage_major (ds4.c:34189) refuses whenever
 * n_stages < 2 (ds4.c:34212). Same synthetic one-layer DeepSeek V4 Flash
 * shape as test_sycl_full_layer.c, built by tests/test_sycl_layer_weights.h.
 *
 * Batch sizes and the dispatch regime each targets, traced from
 * sycl/ds4_sycl_moe_launch.hpp: this fixture's routed experts are IQ2_XXS
 * gate/up with a Q2_K down projection (Flash's real combination, "iq2_path"
 * in that file's own terms), whose dispatch is only two-way --
 * `use_sorted_pairs = n_tokens > 1u` (ds4_sycl_moe_launch.hpp:260), with no
 * ">= 32" or ">= 5" threshold the way Q4_K and MXFP4 have. Those two
 * thresholds are real, but for a *different* format this fixture does not
 * use. The batch sizes below still
 * straddle every threshold that exists for a format this fixture actually
 * exercises, plus the sorted-pairs tile width (block_m == 8,
 * sycl_moe_build_sorted_pairs's tile parameter):
 *   n_tokens = 2:  just above the n_tokens == 1 decode-kernel boundary.
 *   n_tokens = 8:  exactly one sorted-pairs tile.
 *   n_tokens = 9:  one tile plus one row, crossing a tile boundary.
 *   n_tokens = 32: Q4_K's own threshold; not a regime change for this
 *                  fixture's IQ2_XXS/Q2_K format, included to confirm
 *                  nothing regresses at that size.
 *   n_tokens = 64: a larger prefill chunk, further tile-count scaling.
 * Attention's own gate (sycl/ds4_sycl_attention.hpp:2586) is head_dim ==
 * 512; this fixture's head_dim is 32 (test_sycl_layer_weights.h, matching
 * decode's existing shape), so every size below takes the scalar fallback
 * (sycl_attention_prefill_raw_kernel), never oneMKL's gemm_batch. The
 * fixture is deliberately not changed to head_dim ==
 * 512 to chase that path: a decode-side reachability trace
 * shows it is not reachable from this entry regardless of head_dim, for the
 * window value a single-GPU raw-attention layer ever supplies.
 *
 * A compressed-prefill block is added at the end of main: every batch
 * above runs layer 0 at ratio 0, so none of them reach
 * ds4_gpu_attention_prefill_static_mixed_heads_tensor (ds4.c:29532),
 * dereferenced only when ds4_layer_compress_ratio(il) != 0. That entry is
 * where long-context prefill over compressed KV lives, and it had never
 * executed through the engine before now. */

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

/* DS4_TEST_HOOKS entries defined in ds4.c; not declared in a shipped
 * header, following test_sycl_full_layer.c's own precedent. */
int ds4_test_graph_prefill_layer_batch_encode(const int *tokens, uint32_t n_tokens,
                                              float *out_hc, uint64_t out_hc_floats);
int ds4_test_graph_decode_layer_sequence_encode(const int *tokens, uint32_t n_tokens,
                                                float *out_hc, uint64_t out_hc_floats);

/* Layer 3 (ds4_tw_compress_ratio(3) == 128) through the same real
 * batch path, at ratio 128 and at ratio 0 from the identical fixture
 * weights -- see both functions' own ds4.c comments. */
int ds4_test_graph_prefill_layer_batch_encode_compressed(const int *tokens, uint32_t n_tokens,
                                                         float *out_hc, uint64_t out_hc_floats);
int ds4_test_graph_prefill_layer_batch_encode_compressed_baseline(
        const int *tokens, uint32_t n_tokens, float *out_hc, uint64_t out_hc_floats);

/* sycl/ds4_sycl_attention.hpp test-only hooks; not part of the
 * ABI. Count how many times ds4_gpu_attention_prefill_static_mixed_heads_
 * tensor and its sibling ds4_gpu_attention_indexed_mixed_batch_heads_
 * tensor have actually reached a kernel launch, so this test can show
 * WHICH of the two mixed-attention entries the compressed prefill run
 * below took, not just that some path produced plausible-looking output
 * (design-spec 6w). */
extern uint64_t ds4_sycl_test_attn_prefill_static_mixed_calls(void);
extern void     ds4_sycl_test_attn_prefill_static_mixed_calls_reset(void);
extern uint64_t ds4_sycl_test_attn_indexed_mixed_calls(void);
extern void     ds4_sycl_test_attn_indexed_mixed_calls_reset(void);

#define DS4_TEST_PREFILL_MAX_TOKENS 64u

/* The smallest batch that can reach n_comp != 0 for layer 3's
 * ratio (128) in a single zero_prefix prefill chunk (n_comp = n_tokens /
 * ratio, ds4.c:28676-28679) -- see
 * ds4_test_graph_prefill_layer_batch_encode_compressed's own ds4.c
 * comment for why n_comp != 0 is required to reach the mixed-attention
 * branches at all. */
#define DS4_TEST_PREFILL_COMPRESSED_TOKENS 128u

/* Deterministic, non-affine, bounded token ids (spec 6f/6n: an affine
 * progression launders bugs a real, varied prompt would not). `salt`
 * distinguishes one token stream from another for assertion 3. */
static void fill_tokens(int *tokens, uint32_t n_tokens, uint32_t salt) {
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t h = (i + salt) * 2654435761u;
        h ^= h >> 13;
        tokens[i] = (int)(h % DS4_TW_N_VOCAB);
    }
}

static int check_finite_and_nonzero(const float *hc, uint64_t n, const char *what) {
    int any_nonzero = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(hc[i])) {
            fprintf(stderr, "FAIL: %s contains a non-finite value at index %llu\n",
                    what, (unsigned long long)i);
            return 0;
        }
        if (hc[i] != 0.0f) any_nonzero = 1;
    }
    if (!any_nonzero) {
        fprintf(stderr, "FAIL: %s is all zero\n", what);
        return 0;
    }
    return 1;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init returned zero");

    const uint32_t hc_dim = DS4_TW_HC_DIM;
    const uint32_t batch_sizes[] = {2u, 8u, 9u, 32u, 64u};
    const int n_batch_sizes = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

    static float prefill_out[DS4_TEST_PREFILL_MAX_TOKENS * DS4_TW_HC_DIM];
    static float decode_out[DS4_TEST_PREFILL_MAX_TOKENS * DS4_TW_HC_DIM];
    static float prefill_out_alt[8u * DS4_TW_HC_DIM];
    int tokens[DS4_TEST_PREFILL_MAX_TOKENS];

    for (int b = 0; b < n_batch_sizes; b++) {
        const uint32_t n_tokens = batch_sizes[b];
        fill_tokens(tokens, n_tokens, /*salt=*/1u);

        /* Assertion 1: the prefill batch encodes successfully at this size. */
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "prefill batch encode failed at n_tokens=%u", n_tokens);
        CHECK(ds4_test_graph_prefill_layer_batch_encode(
                      tokens, n_tokens, prefill_out, (uint64_t)n_tokens * hc_dim) != 0,
              msg);

        /* Assertion 2: read back, finite, not all zero -- a separate claim
         * from (1) per spec 6l/6w. */
        snprintf(msg, sizeof(msg), "prefill output at n_tokens=%u", n_tokens);
        CHECK(check_finite_and_nonzero(prefill_out, (uint64_t)n_tokens * hc_dim, msg), msg);

        fprintf(stderr, "  prefill batch n_tokens=%u: encoded, finite, non-zero\n", n_tokens);
    }

    /* Assertion 3: output changes with input. Re-run the smallest
     * multi-token batch with a different token stream and compare. */
    {
        const uint32_t n_tokens = 8u;
        fill_tokens(tokens, n_tokens, /*salt=*/1u);
        CHECK(ds4_test_graph_prefill_layer_batch_encode(
                      tokens, n_tokens, prefill_out, (uint64_t)n_tokens * hc_dim) != 0,
              "prefill batch re-encode (salt 1) failed");
        fill_tokens(tokens, n_tokens, /*salt=*/99u);
        CHECK(ds4_test_graph_prefill_layer_batch_encode(
                      tokens, n_tokens, prefill_out_alt, (uint64_t)n_tokens * hc_dim) != 0,
              "prefill batch re-encode (salt 99) failed");
        int any_diff = 0;
        for (uint64_t i = 0; i < (uint64_t)n_tokens * hc_dim; i++) {
            CHECK(isfinite(prefill_out_alt[i]), "prefill output (salt 99) has a non-finite value");
            if (prefill_out[i] != prefill_out_alt[i]) any_diff = 1;
        }
        CHECK(any_diff, "prefill output did not change when the input tokens changed");
        fprintf(stderr, "  prefill batch output changes with input: OK\n");
    }

    /* Assertion 4: prefilling a batch and decoding those same positions one
     * at a time should produce closely matching state. This is the only
     * assertion that can catch a prefill path which runs cleanly and
     * computes something subtly different from decode. Tolerance is
     * ABS_TOL + REL_TOL * |decode value|.
     *
     * Established empirically, per spec 6n, from a clean unablated run
     * before choosing any tolerance: the observed max relative difference
     * ranged from 5.9e-5 (n_tokens=2) to 1.19e-3 (n_tokens=64), growing
     * with n_tokens as expected -- batched prefill and per-token decode sum
     * the same attention and MoE-expert contributions in different orders
     * and through different kernels (attention: sycl_attention_prefill_raw_kernel
     * batched over tokens vs. its single-token decode counterpart; MoE:
     * sorted-pairs tile8 vs. the decode kernel), which is exactly spec 6n's
     * summation-order noise, not a defect. REL_TOL below is set to roughly
     * 8x the largest of those observed values, not tuned down to just pass;
     * ABS_TOL is a floor for near-zero decode values, which this fixture's
     * large-magnitude, un-renormalized hyper-connection state never
     * approaches, so it never binds here. */
    {
        const float ABS_TOL = 1.0e-2f;
        const float REL_TOL = 1.0e-2f;
        for (int b = 0; b < n_batch_sizes; b++) {
            const uint32_t n_tokens = batch_sizes[b];
            fill_tokens(tokens, n_tokens, /*salt=*/1u);

            CHECK(ds4_test_graph_prefill_layer_batch_encode(
                          tokens, n_tokens, prefill_out, (uint64_t)n_tokens * hc_dim) != 0,
                  "assertion-4 prefill batch encode failed");
            CHECK(ds4_test_graph_decode_layer_sequence_encode(
                          tokens, n_tokens, decode_out, (uint64_t)n_tokens * hc_dim) != 0,
                  "assertion-4 decode sequence encode failed");

            float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
            uint64_t max_diff_idx = 0;
            for (uint64_t i = 0; i < (uint64_t)n_tokens * hc_dim; i++) {
                CHECK(isfinite(decode_out[i]), "assertion-4 decode output has a non-finite value");
                const float diff = fabsf(prefill_out[i] - decode_out[i]);
                const float rel = diff / (fabsf(decode_out[i]) + 1.0f);
                if (diff > max_abs_diff) { max_abs_diff = diff; max_diff_idx = i; }
                if (rel > max_rel_diff) max_rel_diff = rel;
                if (diff > ABS_TOL + REL_TOL * fabsf(decode_out[i])) {
                    fprintf(stderr,
                            "FAIL: n_tokens=%u index %llu prefill=%.6f decode=%.6f "
                            "diff=%.6f exceeds tolerance\n",
                            n_tokens, (unsigned long long)i, prefill_out[i], decode_out[i], diff);
                    return 1;
                }
            }
            fprintf(stderr,
                    "  prefill/decode agreement n_tokens=%u: max_abs_diff=%.6f "
                    "(at index %llu) max_rel_diff=%.6f\n",
                    n_tokens, max_abs_diff, (unsigned long long)max_diff_idx, max_rel_diff);
        }
    }

    /* Prefill through the compressed mixed-attention path. Every
     * batch above ran at ratio 0 (layer 0); this drives layer 3, whose
     * ds4_tw_compress_ratio is 128, at a batch size large enough for
     * n_comp != 0. Three claims, in order of value:
     *   1. Output differs from the ratio-0 baseline for the identical
     *      input and weight buffer (spec 6w) -- proves the compressor
     *      actually ran, not just that the call returned success.
     *   2. ds4_gpu_attention_prefill_static_mixed_heads_tensor's own call
     *      counter went from 0 to non-zero, and its sibling
     *      ds4_gpu_attention_indexed_mixed_batch_heads_tensor's counter
     *      stayed at 0 -- proves WHICH mixed-attention entry executed,
     *      not merely that some kernel did.
     *   3. Determinism holds with compression on. */
    {
        static int compressed_tokens[DS4_TEST_PREFILL_COMPRESSED_TOKENS];
        static float compressed_out[DS4_TEST_PREFILL_COMPRESSED_TOKENS * DS4_TW_HC_DIM];
        static float baseline_out[DS4_TEST_PREFILL_COMPRESSED_TOKENS * DS4_TW_HC_DIM];
        static float compressed_out_repeat[DS4_TEST_PREFILL_COMPRESSED_TOKENS * DS4_TW_HC_DIM];
        const uint64_t n_floats = (uint64_t)DS4_TEST_PREFILL_COMPRESSED_TOKENS * hc_dim;

        fill_tokens(compressed_tokens, DS4_TEST_PREFILL_COMPRESSED_TOKENS, /*salt=*/7u);

        CHECK(ds4_test_graph_prefill_layer_batch_encode_compressed_baseline(
                      compressed_tokens, DS4_TEST_PREFILL_COMPRESSED_TOKENS,
                      baseline_out, n_floats) != 0,
              "compressed-prefill ratio-0 baseline (layer 3) failed to encode");
        CHECK(check_finite_and_nonzero(baseline_out, n_floats,
                                       "compressed-prefill ratio-0 baseline output"),
              "compressed-prefill ratio-0 baseline output");

        ds4_sycl_test_attn_prefill_static_mixed_calls_reset();
        ds4_sycl_test_attn_indexed_mixed_calls_reset();
        CHECK(ds4_test_graph_prefill_layer_batch_encode_compressed(
                      compressed_tokens, DS4_TEST_PREFILL_COMPRESSED_TOKENS,
                      compressed_out, n_floats) != 0,
              "compressed prefill batch encode (layer 3, ratio 128) failed");
        CHECK(check_finite_and_nonzero(compressed_out, n_floats, "compressed prefill output"),
              "compressed prefill output");

        int compressed_any_diff = 0;
        for (uint64_t i = 0; i < n_floats; i++) {
            if (compressed_out[i] != baseline_out[i]) { compressed_any_diff = 1; break; }
        }
        CHECK(compressed_any_diff,
              "enabling compression did not change the prefill output at all -- "
              "the compressed path did not actually run");

        const uint64_t static_mixed_calls = ds4_sycl_test_attn_prefill_static_mixed_calls();
        const uint64_t indexed_mixed_calls = ds4_sycl_test_attn_indexed_mixed_calls();
        CHECK(static_mixed_calls > 0,
              "ds4_gpu_attention_prefill_static_mixed_heads_tensor never reached its "
              "kernel launch during the compressed prefill run");
        CHECK(indexed_mixed_calls == 0,
              "ds4_gpu_attention_indexed_mixed_batch_heads_tensor ran during a ratio-128 "
              "prefill batch, which should only ever take the static-mixed path");

        CHECK(ds4_test_graph_prefill_layer_batch_encode_compressed(
                      compressed_tokens, DS4_TEST_PREFILL_COMPRESSED_TOKENS,
                      compressed_out_repeat, n_floats) != 0,
              "compressed prefill batch encode repeat failed");
        CHECK(memcmp(compressed_out, compressed_out_repeat, (size_t)n_floats * sizeof(float)) == 0,
              "compressed prefill output is not deterministic across repeats");

        fprintf(stderr,
                "  compressed prefill (layer 3, ratio 128, n_tokens=%u): differs from "
                "ratio-0 baseline, ds4_gpu_attention_prefill_static_mixed_heads_tensor "
                "called %llu time(s), ds4_gpu_attention_indexed_mixed_batch_heads_tensor "
                "called %llu time(s), deterministic across repeat\n",
                DS4_TEST_PREFILL_COMPRESSED_TOKENS,
                (unsigned long long)static_mixed_calls,
                (unsigned long long)indexed_mixed_calls);
    }

    fprintf(stderr, "  test_sycl_prefill_batch OK\n");
    ds4_gpu_cleanup();
    return 0;
}
