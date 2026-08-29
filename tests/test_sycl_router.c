/* Correctness tests for the SYCL router's single-token entry,
 * ds4_gpu_router_select_tensor, validated against a scalar CPU oracle
 * implemented here.  The ds4.c CPU references are static and cannot be
 * linked, so this oracle reimplements the documented formula with the
 * ds4.c line numbers cited. Needs no model file.
 *
 * This is a self-contained file, following this suite's established convention: it defines its own
 * CHECK/CHECK_CLOSE macros (copied in shape from tests/test_sycl_kernels.c
 * lines 18-40) rather than sharing a harness, and does not touch
 * tests/test_sycl_kernels.c or tests/test_sycl_harness.h. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK_CLOSE(got, want, tol, msg)                                    \
    do {                                                                    \
        double d_ = fabs((double)(got) - (double)(want));                   \
        if (!(d_ <= (tol))) {                                               \
            fprintf(stderr, "FAIL: %s (got %.9g want %.9g delta %.3g)\n",   \
                    (msg), (double)(got), (double)(want), d_);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum { MAX_EXPERT = 384, MAX_EXPERT_USED = 8 };

/* Matches the GPU kernel's softplus_dev (rocm/ds4_rocm_router.cuh:1-5), NOT
 * ds4.c's own softplus_stable (ds4.c:10550-10554).  The two thresholds
 * differ (-10.0f here versus -20.0f there): this oracle validates the GPU
 * kernel, so it must mirror the GPU kernel's own numerics rather than the
 * CPU engine's, or a naive log1p(exp(x)) bug at, say, x = -15 would go
 * uncaught. */
static float oracle_softplus_gpu(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return expf(x);
    return log1pf(expf(x));
}

/* Matches layer_router_probs_one, ds4.c:10712-10725. */
static void oracle_router_probs(float *probs_out, const float *logits, uint32_t n_expert) {
    for (uint32_t i = 0; i < n_expert; i++) {
        probs_out[i] = sqrtf(oracle_softplus_gpu(logits[i]));
    }
}

/* Matches topk_desc, ds4.c:10745-10767: ties resolve toward the lower
 * index because a candidate only displaces an existing slot on a strictly
 * greater comparison, and candidates are scanned in ascending index
 * order. */
static void oracle_topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

/* Matches layer_topk_selected_experts_from_probs, ds4.c:10769-10790:
 * selection uses the biased score (prob + bias), but the emitted weight is
 * the UNBIASED prob of the selected expert, normalised by the sum of those
 * unbiased probs with the 6.103515625e-5f floor, then scaled. */
static void oracle_topk_select(
        int32_t *selected_out, float *weight_out,
        const float *probs, const float *bias, int has_bias,
        uint32_t n_expert, uint32_t n_expert_used, float scale) {
    float selection[MAX_EXPERT];
    memcpy(selection, probs, (size_t)n_expert * sizeof(float));
    if (has_bias) {
        for (uint32_t i = 0; i < n_expert; i++) selection[i] += bias[i];
    }

    int idx[MAX_EXPERT_USED];
    oracle_topk_desc(selection, (int)n_expert, (int)n_expert_used, idx);

    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        selected_out[j] = idx[j];
        weight_out[j] = probs[idx[j]];
        sum += weight_out[j];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        weight_out[j] = weight_out[j] / sum * scale;
    }
}

/* Runs ds4_gpu_router_select_tensor for one (logits, bias) case and
 * compares every output against the oracle above.  `model` must be at
 * least n_expert floats large when has_bias is set; bias is read from
 * offset 0. */
static int run_router_case(
        const char *label,
        uint32_t n_expert, uint32_t n_expert_used, float scale,
        const float *logits, int has_bias, const float *bias) {
    ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(int32_t));
    ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(float));
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");

    /* Sentinel-fill the probs output so an unwritten element is
     * detectable: a bug that leaves any of the n_expert entries untouched
     * must surface as a mismatch against the sentinel, not pass silently. */
    float sentinel[MAX_EXPERT];
    for (uint32_t i = 0; i < n_expert; i++) sentinel[i] = -12345.0f;
    CHECK(ds4_gpu_tensor_write(tprobs, 0, sentinel, (uint64_t)n_expert * sizeof(float)) != 0,
          "router: probs sentinel write failed");
    CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, (uint64_t)n_expert * sizeof(float)) != 0,
          "router: logits write failed");

    unsigned char model[MAX_EXPERT * sizeof(float)];
    memset(model, 0, sizeof(model));
    if (has_bias) memcpy(model, bias, (size_t)n_expert * sizeof(float));

    int ok = ds4_gpu_router_select_tensor(
            tselected, tweights, tprobs, model, sizeof(model),
            /*bias_offset=*/0, /*hash_offset=*/0, /*hash_rows=*/0, /*token=*/0,
            n_expert, n_expert_used, scale, /*n_expert_groups=*/0, /*n_group_used=*/0,
            has_bias != 0, /*hash_mode=*/false, tlogits);
    CHECK(ok != 0, label);

    float got_probs[MAX_EXPERT];
    int32_t got_selected[MAX_EXPERT_USED];
    float got_weights[MAX_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tprobs, 0, got_probs, (uint64_t)n_expert * sizeof(float)) != 0,
          "router: probs read failed");
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, (uint64_t)n_expert_used * sizeof(int32_t)) != 0,
          "router: selected read failed");
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, (uint64_t)n_expert_used * sizeof(float)) != 0,
          "router: weights read failed");

    float want_probs[MAX_EXPERT];
    oracle_router_probs(want_probs, logits, n_expert);
    for (uint32_t i = 0; i < n_expert; i++) {
        CHECK_CLOSE(got_probs[i], want_probs[i], 1e-5, "router: probs mismatch");
    }

    int32_t want_selected[MAX_EXPERT_USED];
    float want_weights[MAX_EXPERT_USED];
    oracle_topk_select(want_selected, want_weights, want_probs, bias, has_bias,
                        n_expert, n_expert_used, scale);
    for (uint32_t j = 0; j < n_expert_used; j++) {
        CHECK(got_selected[j] == want_selected[j], "router: selected expert mismatch");
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5, "router: weight mismatch");
    }

    ds4_gpu_tensor_free(tlogits);
    ds4_gpu_tensor_free(tselected);
    ds4_gpu_tensor_free(tweights);
    ds4_gpu_tensor_free(tprobs);
    return 0;
}

/* Covers all three softplus branches: a logit above 20 (identity), one
 * below -10 (exp), and several in between (log1p(exp)).  This is the
 * cheapest place to catch a naive log1p(exp(x)) port that ignores the
 * piecewise thresholds. */
static int test_router_softplus_branches(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = ((float)i - 128.0f) * 0.05f;
    logits[0] = 25.0f;    /* x > 20 branch */
    logits[1] = -15.0f;   /* x < -10 branch */
    logits[2] = 0.0f;     /* middle branch */
    logits[3] = -9.9f;    /* just inside the middle branch */
    logits[4] = -10.1f;   /* just inside the exp branch */

    if (run_router_case("router: softplus branches", N_EXPERT, N_EXPERT_USED, 1.5f,
                         logits, 0, NULL) != 0) {
        return 1;
    }
    fprintf(stderr, "  test_router_softplus_branches OK\n");
    return 0;
}

/* Bias changes the SELECTED set but the emitted weight must still be the
 * unbiased probability of whichever expert is selected, normalised over the
 * selected set.  Uses n_expert_used == 3 deliberately, not 1: with exactly
 * one selected expert, weight/sum*scale collapses to exactly `scale`
 * regardless of what the numerator was, so a corruption that feeds the
 * BIASED score into the weight instead of the unbiased prob would be
 * invisible at n_expert_used == 1.  Unbiased top-3 is {0, 1, 2}; a large
 * negative bias on 2 and a large positive bias on 5 flips the biased top-3
 * to {5, 0, 1}, and 5's bias (+100) is applied to only one of the three
 * selected experts, so a bias-into-weight corruption produces wildly
 * different relative weights than the correct unbiased-prob weighting. */
static int test_router_bias_changes_selection(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 3 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = -5.0f;
    logits[0] = 10.0f;  /* unbiased top-1 */
    logits[1] = 9.0f;   /* unbiased top-2 */
    logits[2] = 8.0f;   /* unbiased top-3, weakest of the three */
    logits[5] = 7.0f;   /* unbiased 4th, just outside the top-3 */

    float bias[N_EXPERT];
    memset(bias, 0, sizeof(bias));
    bias[2] = -100.0f;  /* kicks expert 2 out of the biased top-3 */
    bias[5] = 100.0f;   /* pulls expert 5 into the biased top-3 */

    /* Confirm the unbiased top-3 really is {0, 1, 2} in that order, so the
     * bias case below is a genuine flip, not an accident of the data. */
    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    int unbiased_top3[N_EXPERT_USED];
    oracle_topk_desc(want_probs, N_EXPERT, N_EXPERT_USED, unbiased_top3);
    CHECK(unbiased_top3[0] == 0 && unbiased_top3[1] == 1 && unbiased_top3[2] == 2,
          "router: test data does not make {0,1,2} the unbiased top-3");

    if (run_router_case("router: bias changes selection", N_EXPERT, N_EXPERT_USED, 1.5f,
                         logits, 1, bias) != 0) {
        return 1;
    }

    /* Directly assert the flip: with bias, {5, 0, 1} must be chosen, not
     * {0, 1, 2}, and in that score-descending order. */
    {
        ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
        ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
        ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
        ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
        CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");
        CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, sizeof(logits)) != 0, "router: logits write failed");

        unsigned char model[N_EXPERT * sizeof(float)];
        memcpy(model, bias, sizeof(bias));

        CHECK(ds4_gpu_router_select_tensor(
                  tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
                  N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, true, false, tlogits) != 0,
              "router: bias-flip case failed");

        int32_t got_selected[N_EXPERT_USED];
        float got_weights[N_EXPERT_USED];
        CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0,
              "router: selected read failed");
        CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0,
              "router: weight read failed");
        CHECK(got_selected[0] == 5 && got_selected[1] == 0 && got_selected[2] == 1,
              "router: bias must flip the selected set to {5, 0, 1}");

        /* Unbiased-prob weighting, normalised over exactly the selected
         * set {5, 0, 1}. */
        const float p5 = want_probs[5], p0 = want_probs[0], p1 = want_probs[1];
        float sum = p5 + p0 + p1;
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        CHECK_CLOSE(got_weights[0], p5 / sum * 1.5f, 1e-5, "router: weight[0] (expert 5) mismatch");
        CHECK_CLOSE(got_weights[1], p0 / sum * 1.5f, 1e-5, "router: weight[1] (expert 0) mismatch");
        CHECK_CLOSE(got_weights[2], p1 / sum * 1.5f, 1e-5, "router: weight[2] (expert 1) mismatch");

        ds4_gpu_tensor_free(tlogits);
        ds4_gpu_tensor_free(tselected);
        ds4_gpu_tensor_free(tweights);
        ds4_gpu_tensor_free(tprobs);
    }

    fprintf(stderr, "  test_router_bias_changes_selection OK\n");
    return 0;
}

/* Two experts with bit-identical biased scores: the lower index must win. */
static int test_router_ties(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 1 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = -20.0f;
    /* Experts 1 and 2 get bit-identical logits (hence bit-identical probs,
     * hence a real float tie on biased score with no bias applied).
     *
     * The pair matters, not just the tie: expert indices that differ in
     * only a single bit (e.g. 3 and 7, which differ only at bit 2) let the
     * cross-lane XOR-shuffle reduction's own topology coincidentally
     * recover the lower index even with the tie-break entirely removed,
     * because lane 0 (which reports the final answer) always shares a
     * zero at that single differing bit with the lower of the two indices.
     * That was verified empirically while writing this test: it produced
     * a false pass under the "remove the tie-break" ablation. Expert
     * indices 1 and 2 differ in two bits (01 vs 10), which was verified
     * empirically to make the reduction's outcome, without a tie-break,
     * depend on which lane wins an internal tie rather than on index
     * order -- lane 0 ends up reporting 2 (the higher index) when the
     * tie-break is removed, so this pair actually discriminates. */
    logits[1] = 4.0f;
    logits[2] = 4.0f;

    ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");
    CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, sizeof(logits)) != 0, "router: logits write failed");

    unsigned char model[16];
    memset(model, 0, sizeof(model));

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) != 0,
          "router: tie case failed");

    int32_t got_selected;
    CHECK(ds4_gpu_tensor_read(tselected, 0, &got_selected, sizeof(got_selected)) != 0,
          "router: selected read failed");
    CHECK(got_selected == 1, "router: tie must resolve to the lower index (1, not 2)");

    ds4_gpu_tensor_free(tlogits);
    ds4_gpu_tensor_free(tselected);
    ds4_gpu_tensor_free(tweights);
    ds4_gpu_tensor_free(tprobs);
    fprintf(stderr, "  test_router_ties OK\n");
    return 0;
}

/* probs must be fully written for every n_expert entry, not just the
 * selected ones: covered by run_router_case's sentinel-fill-then-compare,
 * exercised here at both supported expert counts to also satisfy "both
 * expert counts" from the brief. */
static int test_router_probs_fully_written_both_counts(void) {
    {
        enum { N_EXPERT = 256, N_EXPERT_USED = 8 };
        float logits[N_EXPERT];
        for (uint32_t i = 0; i < N_EXPERT; i++) {
            /* Affine-plus-interaction term (spec 6f): a purely affine
             * a*i+b would make every logit proportional to its index,
             * which is not itself a normalisation-hiding risk here since
             * there is no downstream normalisation of probs, but a varied,
             * non-monotonic input is still what actually exercises
             * sentinel detection at every position rather than a single
             * pattern. */
            logits[i] = 0.01f * (float)i - 3.0f + 0.37f * (float)((i * 13) % 7);
        }
        if (run_router_case("router: probs fully written (256)", N_EXPERT, N_EXPERT_USED, 1.5f,
                             logits, 0, NULL) != 0) {
            return 1;
        }
    }
    {
        enum { N_EXPERT = 384, N_EXPERT_USED = 8 };
        float logits[N_EXPERT];
        for (uint32_t i = 0; i < N_EXPERT; i++) {
            logits[i] = 0.01f * (float)i - 2.0f + 0.29f * (float)((i * 17) % 11);
        }
        if (run_router_case("router: probs fully written (384)", N_EXPERT, N_EXPERT_USED, 1.5f,
                             logits, 0, NULL) != 0) {
            return 1;
        }
    }
    fprintf(stderr, "  test_router_probs_fully_written_both_counts OK\n");
    return 0;
}

/* n_expert == 0 must resolve to the default (256), matching ROCm's launcher
 * (active_n_expert = n_expert != 0 ? n_expert : DS4_ROCM_N_EXPERT). Tensors
 * are sized for the actual resolved width (256), the call passes
 * n_expert == 0, and every output is checked against the same oracle used
 * elsewhere with n_expert == 256 explicitly: if the zero were mistakenly
 * left unresolved, validation would reject it (0 is neither 256 nor 384)
 * and the very first CHECK below would fail. */
static int test_router_n_expert_zero_default(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 8 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) {
        logits[i] = 0.01f * (float)i - 3.0f + 0.31f * (float)((i * 11) % 5);
    }

    ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");
    CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, sizeof(logits)) != 0, "router: logits write failed");

    unsigned char model[16];
    memset(model, 0, sizeof(model));

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              /*n_expert=*/0, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) != 0,
          "router: n_expert == 0 must resolve to the default (256), not be rejected");

    float got_probs[N_EXPERT];
    int32_t got_selected[N_EXPERT_USED];
    float got_weights[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tprobs, 0, got_probs, sizeof(got_probs)) != 0, "router: probs read failed");
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0, "router: selected read failed");
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0, "router: weights read failed");

    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    for (uint32_t i = 0; i < N_EXPERT; i++) {
        CHECK_CLOSE(got_probs[i], want_probs[i], 1e-5, "router: n_expert==0 default probs mismatch");
    }

    int32_t want_selected[N_EXPERT_USED];
    float want_weights[N_EXPERT_USED];
    oracle_topk_select(want_selected, want_weights, want_probs, NULL, 0, N_EXPERT, N_EXPERT_USED, 1.5f);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK(got_selected[j] == want_selected[j], "router: n_expert==0 default selected mismatch");
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5, "router: n_expert==0 default weight mismatch");
    }

    ds4_gpu_tensor_free(tlogits);
    ds4_gpu_tensor_free(tselected);
    ds4_gpu_tensor_free(tweights);
    ds4_gpu_tensor_free(tprobs);
    fprintf(stderr, "  test_router_n_expert_zero_default OK\n");
    return 0;
}

/* n_expert_used == 0 must resolve to the default (8), matching ROCm's
 * launcher. Tensors are sized for the actual resolved width (8), the call
 * passes n_expert_used == 0, and every selected/weight output is checked
 * against the oracle with n_expert_used == 8 explicitly: if the zero were
 * mistakenly left unresolved, the topk loop would run zero iterations and
 * every comparison below would fail against the real 8-expert oracle. */
static int test_router_n_expert_used_zero_default(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 8 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) {
        logits[i] = 0.01f * (float)i - 2.5f + 0.23f * (float)((i * 7) % 9);
    }

    ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");
    CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, sizeof(logits)) != 0, "router: logits write failed");

    unsigned char model[16];
    memset(model, 0, sizeof(model));

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, /*n_expert_used=*/0, 1.5f, 0, 0, false, false, tlogits) != 0,
          "router: n_expert_used == 0 must resolve to the default (8), not be rejected");

    int32_t got_selected[N_EXPERT_USED];
    float got_weights[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0, "router: selected read failed");
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0, "router: weights read failed");

    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    int32_t want_selected[N_EXPERT_USED];
    float want_weights[N_EXPERT_USED];
    oracle_topk_select(want_selected, want_weights, want_probs, NULL, 0, N_EXPERT, N_EXPERT_USED, 1.5f);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK(got_selected[j] == want_selected[j], "router: n_expert_used==0 default selected mismatch");
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5, "router: n_expert_used==0 default weight mismatch");
    }

    ds4_gpu_tensor_free(tlogits);
    ds4_gpu_tensor_free(tselected);
    ds4_gpu_tensor_free(tweights);
    ds4_gpu_tensor_free(tprobs);
    fprintf(stderr, "  test_router_n_expert_used_zero_default OK\n");
    return 0;
}

/* Drives the selected probabilities small enough that their sum falls
 * below the 6.103515625e-5f floor, and asserts the floor is applied: very
 * negative logits push every prob toward 0. */
static int test_router_sum_floor(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = -60.0f;

    if (run_router_case("router: sum floor", N_EXPERT, N_EXPERT_USED, 1.5f,
                         logits, 0, NULL) != 0) {
        return 1;
    }

    /* Confirm the floor actually engaged for this data, i.e. the oracle
     * itself exercises the floored branch and this is not an accidentally
     * vacuous check. */
    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    float raw_sum = 0.0f;
    int idx[N_EXPERT_USED];
    oracle_topk_desc(want_probs, N_EXPERT, N_EXPERT_USED, idx);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) raw_sum += want_probs[idx[j]];
    CHECK(raw_sum < 6.103515625e-5f, "router: test data does not actually trigger the sum floor");

    fprintf(stderr, "  test_router_sum_floor OK\n");
    return 0;
}

/* Rejections: null pointers, unsupported n_expert, n_expert_used > 8,
 * n_expert_groups > 1, n_group_used > 0, non-positive expert_weight_scale,
 * undersized tensors, and a bias_offset whose range exceeds model_size. */
static int test_router_rejections(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 8 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = 0.1f * (float)i - 10.0f;

    ds4_gpu_tensor *tlogits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    ds4_gpu_tensor *tselected = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *tweights = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    CHECK(tlogits && tselected && tweights && tprobs, "router: allocation failed");
    CHECK(ds4_gpu_tensor_write(tlogits, 0, logits, sizeof(logits)) != 0, "router: logits write failed");

    unsigned char model[N_EXPERT * sizeof(float)];
    memset(model, 0, sizeof(model));

    CHECK(ds4_gpu_router_select_tensor(NULL, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: null selected must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, NULL, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: null weights must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, NULL, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: null probs must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, NULL, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: null model_map must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, NULL) == 0,
          "router: null logits must be rejected");

    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              255u, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: n_expert not in {256,384} must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, 9u, 1.5f, 0, 0, false, false, tlogits) == 0,
          "router: n_expert_used > 8 must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 2u, 0, false, false, tlogits) == 0,
          "router: n_expert_groups > 1 must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 1u, false, false, tlogits) == 0,
          "router: n_group_used > 0 must be rejected");
    /* expert_weight_scale == 0.0f is NOT a rejection: per the zero-means-
     * default convention (matching n_expert and n_expert_used), it resolves
     * to kSyclExpertWeightScale and the call succeeds. Only an explicitly
     * non-positive (negative) scale is rejected. */
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 0.0f, 0, 0, false, false, tlogits) != 0,
          "router: expert_weight_scale == 0 must resolve to the default, not be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, -1.5f, 0, 0, false, false, tlogits) == 0,
          "router: negative expert_weight_scale must be rejected");

    {
        ds4_gpu_tensor *small_logits = ds4_gpu_tensor_alloc((uint64_t)(N_EXPERT - 1) * sizeof(float));
        CHECK(small_logits, "router: allocation failed");
        CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
                  N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, small_logits) == 0,
              "router: undersized logits must be rejected");
        ds4_gpu_tensor_free(small_logits);
    }
    {
        ds4_gpu_tensor *small_probs = ds4_gpu_tensor_alloc((uint64_t)(N_EXPERT - 1) * sizeof(float));
        CHECK(small_probs, "router: allocation failed");
        CHECK(ds4_gpu_router_select_tensor(tselected, tweights, small_probs, model, sizeof(model), 0, 0, 0, 0,
                  N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
              "router: undersized probs must be rejected");
        ds4_gpu_tensor_free(small_probs);
    }
    {
        ds4_gpu_tensor *small_selected = ds4_gpu_tensor_alloc((uint64_t)(N_EXPERT_USED - 1) * sizeof(int32_t));
        CHECK(small_selected, "router: allocation failed");
        CHECK(ds4_gpu_router_select_tensor(small_selected, tweights, tprobs, model, sizeof(model), 0, 0, 0, 0,
                  N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
              "router: undersized selected must be rejected");
        ds4_gpu_tensor_free(small_selected);
    }
    {
        ds4_gpu_tensor *small_weights = ds4_gpu_tensor_alloc((uint64_t)(N_EXPERT_USED - 1) * sizeof(float));
        CHECK(small_weights, "router: allocation failed");
        CHECK(ds4_gpu_router_select_tensor(tselected, small_weights, tprobs, model, sizeof(model), 0, 0, 0, 0,
                  N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, false, tlogits) == 0,
              "router: undersized weights must be rejected");
        ds4_gpu_tensor_free(small_weights);
    }

    /* bias_offset whose range exceeds model_size. */
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model),
              /*bias_offset=*/sizeof(model), 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, /*has_bias=*/true, false, tlogits) == 0,
          "router: bias_offset at model_size (zero remaining room) must be rejected");
    CHECK(ds4_gpu_router_select_tensor(tselected, tweights, tprobs, model, sizeof(model),
              /*bias_offset=*/sizeof(model) - sizeof(float), 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, /*has_bias=*/true, false, tlogits) == 0,
          "router: bias_offset leaving less than n_expert floats must be rejected");

    ds4_gpu_tensor_free(tlogits);
    ds4_gpu_tensor_free(tselected);
    ds4_gpu_tensor_free(tweights);
    ds4_gpu_tensor_free(tprobs);
    fprintf(stderr, "  test_router_rejections OK\n");
    return 0;
}

/* Matches rocm/ds4_rocm_router.cuh:51-67: hash mode ignores scores
 * entirely and looks up the expert set from row `token` of a
 * hash_rows x n_expert_used int32 table, clamping an out-of-range token to
 * row 0 and any out-of-range expert id to weight 0.0f (the id itself is
 * still stored in `selected`, unconditionally, matching `sel[j] = e;`
 * running before the range check). Weights normalise the UNBIASED prob of
 * each selected id: bias is never applied in hash mode, since both
 * launchers pass has_bias && !hash_mode into the kernel
 * (rocm/ds4_rocm_router.cuh:157/:162 and :213/:229). */
static void oracle_hash_select(
        int32_t *selected_out, float *weight_out, const float *probs,
        const int32_t *hash_row, uint32_t n_expert, uint32_t n_expert_used,
        float scale) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        int32_t e = hash_row[j];
        selected_out[j] = e;
        float v = (e >= 0 && (uint32_t)e < n_expert) ? probs[(uint32_t)e] : 0.0f;
        weight_out[j] = v;
        sum += v;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t j = 0; j < n_expert_used; j++) weight_out[j] = weight_out[j] / sum * scale;
}

/* Shared allocation for the hash-mode tests below: four tensors sized for
 * (n_expert, n_expert_used), with `logits` already written. */
static int alloc_router_tensors(
        uint32_t n_expert, uint32_t n_expert_used, const float *logits,
        ds4_gpu_tensor **out_logits, ds4_gpu_tensor **out_selected,
        ds4_gpu_tensor **out_weights, ds4_gpu_tensor **out_probs) {
    *out_logits = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    *out_selected = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(int32_t));
    *out_weights = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(float));
    *out_probs = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    if (!*out_logits || !*out_selected || !*out_weights || !*out_probs) return 1;
    if (ds4_gpu_tensor_write(*out_logits, 0, logits, (uint64_t)n_expert * sizeof(float)) == 0) return 1;
    return 0;
}

static void free_router_tensors(ds4_gpu_tensor *l, ds4_gpu_tensor *s, ds4_gpu_tensor *w, ds4_gpu_tensor *p) {
    ds4_gpu_tensor_free(l);
    ds4_gpu_tensor_free(s);
    ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(p);
}

/* Hash mode selects from a fixed per-token table and ignores scores
 * entirely.  Uses logits whose unbiased top-4 is {0,1,2,3} so a run in
 * score mode on the identical logits provably differs from the hash-mode
 * result: a port that silently fell through to top-k could not pass this
 * test.  The hash table is placed at a nonzero offset to also exercise
 * hash_offset, and probs must still be fully written in hash mode. */
static int test_router_hash_mode_basic(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4, HASH_ROWS = 3, HASH_OFFSET = 64 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = -5.0f;
    logits[0] = 10.0f;
    logits[1] = 9.0f;
    logits[2] = 8.0f;
    logits[3] = 7.0f;
    /* Row 1's selected experts (100,110,120,130) deliberately get distinct
     * logits, not the uniform -5.0f every other non-top-4 expert shares:
     * discovered while ablating this test (a uniform-corruption ablation
     * of sprob, e.g. adding a constant offset to every entry, cancels out
     * exactly under normalisation when all four selected probs are
     * identical, per spec 6f's "normalisation erases scale/shift-only
     * corruption" class of bug). With distinct logits the four selected
     * probs differ, so any corruption that touches sprob uniformly still
     * changes their relative weights after normalisation. */
    logits[100] = -4.0f;
    logits[110] = -4.5f;
    logits[120] = -3.0f;
    logits[130] = -5.5f;

    const int32_t hash_table[HASH_ROWS][N_EXPERT_USED] = {
            {50, 60, 70, 80},
            {100, 110, 120, 130},
            {5, 6, 7, 8},
    };
    unsigned char model[HASH_OFFSET + sizeof(hash_table)];
    memset(model, 0, sizeof(model));
    memcpy(model + HASH_OFFSET, hash_table, sizeof(hash_table));

    ds4_gpu_tensor *tlogits, *tselected, *tweights, *tprobs;
    CHECK(alloc_router_tensors(N_EXPERT, N_EXPERT_USED, logits, &tlogits, &tselected, &tweights, &tprobs) == 0,
          "router: hash basic allocation failed");

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model),
              /*bias_offset=*/0, HASH_OFFSET, HASH_ROWS, /*token=*/1,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, /*has_bias=*/false, /*hash_mode=*/true, tlogits) != 0,
          "router: hash mode call failed");

    float got_probs[N_EXPERT];
    int32_t got_selected[N_EXPERT_USED];
    float got_weights[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tprobs, 0, got_probs, sizeof(got_probs)) != 0, "router: hash probs read failed");
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0, "router: hash selected read failed");
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0, "router: hash weights read failed");

    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    for (uint32_t i = 0; i < N_EXPERT; i++) {
        CHECK_CLOSE(got_probs[i], want_probs[i], 1e-5, "router: hash mode probs mismatch (not fully written)");
    }

    int32_t want_selected[N_EXPERT_USED];
    float want_weights[N_EXPERT_USED];
    oracle_hash_select(want_selected, want_weights, want_probs, hash_table[1], N_EXPERT, N_EXPERT_USED, 1.5f);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK(got_selected[j] == want_selected[j], "router: hash mode selected mismatch");
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5, "router: hash mode weight mismatch");
    }

    /* Score mode on the identical logits must pick a different set: the
     * unbiased top-4 is {0,1,2,3}, nothing like hash row 1's {100,110,120,130}. */
    ds4_gpu_tensor *tselected2 = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(int32_t));
    ds4_gpu_tensor *tweights2 = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT_USED * sizeof(float));
    ds4_gpu_tensor *tprobs2 = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(float));
    CHECK(tselected2 && tweights2 && tprobs2, "router: hash basic score-mode allocation failed");
    CHECK(ds4_gpu_router_select_tensor(
              tselected2, tweights2, tprobs2, model, sizeof(model), 0, 0, 0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, /*hash_mode=*/false, tlogits) != 0,
          "router: score mode call failed");
    int32_t score_selected[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tselected2, 0, score_selected, sizeof(score_selected)) != 0,
          "router: score mode selected read failed");
    int differs = 0;
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        if (score_selected[j] != got_selected[j]) differs = 1;
    }
    CHECK(differs, "router: hash mode and score mode must select differently on identical logits");
    ds4_gpu_tensor_free(tselected2);
    ds4_gpu_tensor_free(tweights2);
    ds4_gpu_tensor_free(tprobs2);

    free_router_tensors(tlogits, tselected, tweights, tprobs);
    fprintf(stderr, "  test_router_hash_mode_basic OK\n");
    return 0;
}

/* Bias must have zero effect in hash mode, even when has_bias is true and
 * bias_offset is itself out of range: since has_bias && !hash_mode gates
 * all bias staging and validation, an invalid bias_offset must not be
 * rejected either, which is itself evidence bias is never touched. */
static int test_router_hash_mode_bias_ignored(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4, HASH_ROWS = 1 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = 0.02f * (float)i - 2.0f;

    const int32_t hash_row[N_EXPERT_USED] = {10, 20, 30, 40};
    unsigned char model[sizeof(hash_row)];
    memcpy(model, hash_row, sizeof(hash_row));

    ds4_gpu_tensor *tlogits, *tselected, *tweights, *tprobs;
    CHECK(alloc_router_tensors(N_EXPERT, N_EXPERT_USED, logits, &tlogits, &tselected, &tweights, &tprobs) == 0,
          "router: hash bias allocation failed");

    /* bias_offset == sizeof(model) leaves zero room for a bias table, which
     * would be rejected if has_bias && !hash_mode staging ever ran. */
    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model),
              /*bias_offset=*/sizeof(model), /*hash_offset=*/0, HASH_ROWS, /*token=*/0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, /*has_bias=*/true, /*hash_mode=*/true, tlogits) != 0,
          "router: hash mode with has_bias=true and an out-of-range bias_offset must still succeed");

    float got_weights[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0,
          "router: hash bias weights read failed");

    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    int32_t want_selected[N_EXPERT_USED];
    float want_weights[N_EXPERT_USED];
    oracle_hash_select(want_selected, want_weights, want_probs, hash_row, N_EXPERT, N_EXPERT_USED, 1.5f);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5,
                    "router: hash mode weight must equal the unbiased-prob oracle regardless of has_bias");
    }

    free_router_tensors(tlogits, tselected, tweights, tprobs);
    fprintf(stderr, "  test_router_hash_mode_bias_ignored OK\n");
    return 0;
}

/* An out-of-range expert id in the hash row (negative, or >= n_expert)
 * must contribute weight 0.0f and must not crash; the id itself is still
 * copied into `selected` unconditionally, matching ROCm's `sel[j] = e;`
 * running before the range check. */
static int test_router_hash_mode_out_of_range_expert(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4, HASH_ROWS = 1 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = 0.01f * (float)i - 1.5f;

    const int32_t hash_row[N_EXPERT_USED] = {5, -3, N_EXPERT, 10};
    unsigned char model[sizeof(hash_row)];
    memcpy(model, hash_row, sizeof(hash_row));

    ds4_gpu_tensor *tlogits, *tselected, *tweights, *tprobs;
    CHECK(alloc_router_tensors(N_EXPERT, N_EXPERT_USED, logits, &tlogits, &tselected, &tweights, &tprobs) == 0,
          "router: hash out-of-range allocation failed");

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, HASH_ROWS, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, true, tlogits) != 0,
          "router: hash mode with out-of-range ids must still succeed");

    int32_t got_selected[N_EXPERT_USED];
    float got_weights[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0,
          "router: hash out-of-range selected read failed");
    CHECK(ds4_gpu_tensor_read(tweights, 0, got_weights, sizeof(got_weights)) != 0,
          "router: hash out-of-range weights read failed");

    CHECK(got_selected[1] == -3 && got_selected[2] == N_EXPERT,
          "router: out-of-range ids must still be copied into selected verbatim");
    CHECK(got_weights[1] == 0.0f, "router: negative expert id must contribute weight 0.0");
    CHECK(got_weights[2] == 0.0f, "router: expert id == n_expert must contribute weight 0.0");

    float want_probs[N_EXPERT];
    oracle_router_probs(want_probs, logits, N_EXPERT);
    int32_t want_selected[N_EXPERT_USED];
    float want_weights[N_EXPERT_USED];
    oracle_hash_select(want_selected, want_weights, want_probs, hash_row, N_EXPERT, N_EXPERT_USED, 1.5f);
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK(got_selected[j] == want_selected[j], "router: hash out-of-range selected mismatch");
        CHECK_CLOSE(got_weights[j], want_weights[j], 1e-5, "router: hash out-of-range weight mismatch");
    }

    free_router_tensors(tlogits, tselected, tweights, tprobs);
    fprintf(stderr, "  test_router_hash_mode_out_of_range_expert OK\n");
    return 0;
}

/* A token id at or beyond hash_rows must clamp to row 0, matching
 * `if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;`. */
static int test_router_hash_mode_token_clamp(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4, HASH_ROWS = 3 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = 0.01f * (float)i - 1.0f;

    const int32_t hash_table[HASH_ROWS][N_EXPERT_USED] = {
            {1, 2, 3, 4},
            {11, 12, 13, 14},
            {21, 22, 23, 24},
    };
    unsigned char model[sizeof(hash_table)];
    memcpy(model, hash_table, sizeof(hash_table));

    ds4_gpu_tensor *tlogits, *tselected, *tweights, *tprobs;
    CHECK(alloc_router_tensors(N_EXPERT, N_EXPERT_USED, logits, &tlogits, &tselected, &tweights, &tprobs) == 0,
          "router: hash clamp allocation failed");

    /* token == hash_rows + 50 is well out of range; it must clamp to 0, not
     * be rejected. */
    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, HASH_ROWS,
              /*token=*/HASH_ROWS + 50u,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, true, tlogits) != 0,
          "router: out-of-range token must clamp, not be rejected");

    int32_t got_selected[N_EXPERT_USED];
    CHECK(ds4_gpu_tensor_read(tselected, 0, got_selected, sizeof(got_selected)) != 0,
          "router: hash clamp selected read failed");
    for (uint32_t j = 0; j < N_EXPERT_USED; j++) {
        CHECK(got_selected[j] == hash_table[0][j], "router: out-of-range token must clamp to row 0");
    }

    free_router_tensors(tlogits, tselected, tweights, tprobs);
    fprintf(stderr, "  test_router_hash_mode_token_clamp OK\n");
    return 0;
}

/* hash_rows == 0 and a hash_offset whose range exceeds model_size must
 * both be rejected, matching rocm/ds4_rocm_router.cuh:142-150. */
static int test_router_hash_mode_rejections(void) {
    enum { N_EXPERT = 256, N_EXPERT_USED = 4, HASH_ROWS = 2 };
    float logits[N_EXPERT];
    for (uint32_t i = 0; i < N_EXPERT; i++) logits[i] = 0.01f * (float)i - 1.0f;

    const int32_t hash_table[HASH_ROWS][N_EXPERT_USED] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    unsigned char model[sizeof(hash_table)];
    memcpy(model, hash_table, sizeof(hash_table));

    ds4_gpu_tensor *tlogits, *tselected, *tweights, *tprobs;
    CHECK(alloc_router_tensors(N_EXPERT, N_EXPERT_USED, logits, &tlogits, &tselected, &tweights, &tprobs) == 0,
          "router: hash rejections allocation failed");

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0, 0, /*hash_rows=*/0, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, true, tlogits) == 0,
          "router: hash_rows == 0 must be rejected");

    CHECK(ds4_gpu_router_select_tensor(
              tselected, tweights, tprobs, model, sizeof(model), 0,
              /*hash_offset=*/sizeof(model), HASH_ROWS, 0,
              N_EXPERT, N_EXPERT_USED, 1.5f, 0, 0, false, true, tlogits) == 0,
          "router: hash_offset leaving no room for the table must be rejected");

    free_router_tensors(tlogits, tselected, tweights, tprobs);
    fprintf(stderr, "  test_router_hash_mode_rejections OK\n");
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_router_softplus_branches() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_bias_changes_selection() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_ties() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_probs_fully_written_both_counts() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_n_expert_zero_default() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_n_expert_used_zero_default() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_sum_floor() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_rejections() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_hash_mode_basic() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_hash_mode_bias_ignored() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_hash_mode_out_of_range_expert() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_hash_mode_token_clamp() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_hash_mode_rejections() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_router OK\n");
    return 0;
}
