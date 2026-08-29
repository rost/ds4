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

int main(void) {
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init failed");
    if (test_router_softplus_branches() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_bias_changes_selection() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_ties() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_probs_fully_written_both_counts() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_sum_floor() != 0) { ds4_gpu_cleanup(); return 1; }
    if (test_router_rejections() != 0) { ds4_gpu_cleanup(); return 1; }
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_sycl_router OK\n");
    return 0;
}
