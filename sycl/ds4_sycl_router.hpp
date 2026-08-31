#pragma once

/* DeepSeek MoE router: score computation and top-k expert selection for a
 * single token.
 *
 * Ported from rocm/ds4_rocm_router.cuh, which is the authority for all
 * semantics here.  The CPU oracle is layer_router_probs_one (ds4.c:10712-
 * 10725) plus layer_topk_selected_experts_from_probs (ds4.c:10769-10790),
 * with topk_desc (ds4.c:10745-10767) implementing the tie-break rule.
 *
 * router_select_warp_topk (this file's kernel) is one work-group of 32
 * lanes by 4 rows, matching the ROCm launch of dim3 block(32, 4, 1): the
 * lane dimension is a sub-group doing warp-style top-k via cross-lane
 * shuffles, and each of the 4 rows is an independent sub-group handling one
 * token.  ds4_gpu_router_select_tensor below launches exactly one such
 * work-group with n_tokens == 1 (only row 0 is used); ds4_gpu_router_select_
 * batch_tensor launches as many work-groups as n_tokens requires over real
 * per-token rows, sharing this same kernel body.
 *
 * Hash-mode routing (selecting experts from a fixed per-token lookup table
 * instead of top-k, ignoring scores and bias entirely) is fully ported
 * from ROCm alongside the top-k path: the kernel's hash-mode branch, and
 * ds4_gpu_router_select_tensor's hash validation, model-range checking and
 * device staging of the hash table, are all real, not stubs.  See
 * rocm/ds4_rocm_router.cuh:49-67 for the kernel side and :141-150 for the
 * launcher's hash validation.
 *
 * Entry is NONZERO-means-success, per the return-value convention note in
 * the design spec section 3a: verified directly against the ds4.c call
 * site (ds4.c:23851, `ok = ds4_gpu_router_select_tensor(...)` used as a
 * truthy bool; the sibling batch entry's call sites are ds4.c:30037 and
 * ds4.c:65036) and against ROCm's own validation-failure `return 0` shape.
 *
 * Barrier scope and early-exit shape (binding on any future addition to
 * this kernel body): the ROCm kernel's __syncwarp() between writing sprob
 * and reading it back is a WARP-scoped barrier, so it becomes
 * sycl::group_barrier(sg) over the sub-group specifically, never a
 * work-group barrier -- only certain lanes read sprob back (hash mode's
 * lane 0), and other sub-groups in the work-group may not be doing live
 * work in the batch case.  ROCm's guard `if (t >= n_tokens || lane >= 32u)
 * return;` is NOT ported as a bare early return: a work item that returns
 * before a barrier other work items in its own execution reach is
 * undefined behaviour in SYCL, unlike CUDA's warp-scoped __syncwarp, which
 * only requires the still-live lanes named in its mask to reach it.  This
 * kernel instead computes `active = t < n_tokens` once, gates every read,
 * write and the two per-row output blocks behind `if (active) { ... }`,
 * and reaches sycl::group_barrier(sg) unconditionally in the source for
 * every lane.  This is sound because t depends only on row_in_block, which
 * is constant across a whole sub-group (one sub-group is exactly one
 * row_in_block's 32 lanes): every lane in a given sub-group agrees on
 * `active`, so the group either all takes the pre-barrier write path or
 * none of it does, and every lane always reaches the barrier call itself.
 * The `lane >= 32u` half of ROCm's guard has no SYCL equivalent to port:
 * the work-group's lane dimension is sized to exactly 32, so lane can
 * never exceed it. */

#include "ds4_sycl_common.hpp"

namespace {

/* Matches rocm/ds4_rocm_runtime.cuh:27-36.  No shared SYCL constants file
 * exists yet (see the kCompressorMaxRatio precedent in
 * ds4_sycl_compressor.hpp), so these live here rather than in
 * ds4_sycl_common.hpp. */
constexpr uint32_t kSyclNExpert            = 256u;
constexpr uint32_t kSyclMaxNExpert         = 384u;
constexpr uint32_t kSyclNExpertUsed        = 8u;
constexpr float    kSyclExpertWeightScale  = 1.5f;

/* The ROCm launch geometry, dim3 block(32, 4, 1): 32 lanes forming one
 * sub-group/warp, 4 rows (tokens) per work-group. */
constexpr uint32_t kSyclRouterLanes        = 32u;
constexpr uint32_t kSyclRouterRowsPerGroup = 4u;

/* Matches softplus_dev, rocm/ds4_rocm_router.cuh:1-5.  NOTE this threshold
 * (-10.0f) deliberately differs from ds4.c's own softplus_stable
 * (ds4.c:10550-10554), which uses -20.0f: this is the GPU kernel and must
 * match the GPU kernel's own numerics, not the CPU's.  Any CPU-side oracle
 * validating this kernel must mirror THIS threshold, not ds4.c's. */
static inline float sycl_router_softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return sycl::exp(x);
    return sycl::log1p(sycl::exp(x));
}

/* Matches router_score_better, rocm/ds4_rocm_router.cuh:7-9.  Ties break
 * toward the lower index: verified against the CPU's topk_desc
 * (ds4.c:10745-10767), which scans candidates in ascending index order and
 * inserts a new candidate into slot j only on a strictly-greater
 * comparison, so an equal score never displaces the earlier (lower-index)
 * incumbent. */
static inline bool sycl_router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

/* Shared top-k / hash-select kernel body, templated on N_EXPERT exactly as
 * ROCm's router_select_warp_topk_kernel<N_EXPERT> is: N_EXPERT / 32 must be
 * an integer (8 for 256, 12 for 384) so lane j*32+lane covers every expert
 * exactly once per lane's private strided slice.
 *
 * `tokens` is the batch entry's per-token index tensor, always NULL from
 * the single-token launcher below (token_scalar plays that role). */
template <uint32_t N_EXPERT>
static void sycl_router_select_launch(
        sycl::queue &q,
        int32_t *selected, float *weights, float *probs,
        const float *bias, const int32_t *hash,
        const float *logits, const int32_t *tokens,
        int32_t token_scalar, uint32_t hash_rows, uint32_t n_tokens,
        uint32_t n_expert_used, float expert_weight_scale,
        int has_bias, int hash_mode) {
    constexpr uint32_t kPerLane = N_EXPERT / kSyclRouterLanes;
    const uint32_t n_groups = (n_tokens + kSyclRouterRowsPerGroup - 1u) / kSyclRouterRowsPerGroup;

    sycl::event _ds4_prof_ev145 = q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 2> sprob(
                sycl::range<2>(kSyclRouterRowsPerGroup, N_EXPERT), h);

        /* Local range (rows, lanes) = (4, 32): lanes are dimension 1, the
         * fastest-varying SYCL dimension, so a sub-group of 32 (requested
         * below) lines up exactly with one row's 32 lanes, matching one
         * CUDA warp. */
        h.parallel_for(
                sycl::nd_range<2>(
                        sycl::range<2>((size_t)n_groups * kSyclRouterRowsPerGroup, kSyclRouterLanes),
                        sycl::range<2>(kSyclRouterRowsPerGroup, kSyclRouterLanes)),
                [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                    const uint32_t row_in_block = (uint32_t)it.get_local_id(0);
                    const uint32_t lane         = (uint32_t)it.get_local_id(1);
                    const uint32_t group_id     = (uint32_t)it.get_group(0);
                    const uint32_t t            = group_id * kSyclRouterRowsPerGroup + row_in_block;
                    const bool active           = t < n_tokens;
                    sycl::sub_group sg          = it.get_sub_group();

                    float local_prob[kPerLane];
                    float local_score[kPerLane];

                    if (active) {
                        const float *log_row = logits + (uint64_t)t * N_EXPERT;
                        float       *prob_row = probs + (uint64_t)t * N_EXPERT;

                        #pragma unroll
                        for (uint32_t j = 0; j < kPerLane; j++) {
                            const uint32_t e = lane + j * kSyclRouterLanes;
                            const float p = sycl::sqrt(sycl_router_softplus_dev(log_row[e]));
                            local_prob[j] = p;
                            local_score[j] = p + (has_bias ? bias[e] : 0.0f);
                            /* Every (row_in_block, e) slot is written here
                             * unconditionally by whichever lane owns it,
                             * before the barrier below that a hash-mode
                             * lane 0 later reads back through.  Per spec
                             * 6b, an omitted write reads as zero on this
                             * hardware (Arc A770, Level Zero, oneAPI
                             * 2025.3) rather than failing loudly, so no
                             * test in this repo can catch a dropped write
                             * here; this comment is the only thing
                             * protecting the invariant. */
                            sprob[row_in_block][e] = p;
                            prob_row[e] = p;
                        }
                    }

                    /* Reached by every lane regardless of `active`: see the
                     * file-level comment on barrier scope and early-exit
                     * shape above. */
                    sycl::group_barrier(sg);

                    if (active) {
                        int32_t *sel_row = selected + (uint64_t)t * n_expert_used;
                        float   *w_row   = weights + (uint64_t)t * n_expert_used;

                        if (hash_mode) {
                            if (lane == 0) {
                                int32_t tok = tokens ? tokens[t] : token_scalar;
                                if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
                                const int32_t *row = hash + (uint64_t)tok * n_expert_used;
                                float sum = 0.0f;
                                for (uint32_t j = 0; j < n_expert_used; j++) {
                                    const int32_t e = row[j];
                                    sel_row[j] = e;
                                    /* Out-of-range hash ids must contribute weight 0.0, matching
                                     * rocm/ds4_rocm_router.cuh:60.  This guard's upper bound could
                                     * not be proven necessary by ablation on this hardware: with
                                     * only one active row (the single-token entry always launches
                                     * with n_tokens == 1), removing the upper bound makes an
                                     * out-of-range id spill into an adjacent row of sprob that no
                                     * lane in this launch ever wrote, and per spec 6b uninitialised
                                     * local memory reads as zero on this Arc A770 / Level Zero /
                                     * oneAPI 2025.3 stack, so the spillover reads back as 0.0
                                     * anyway and the removal is invisible here.  The lower bound
                                     * (e >= 0) was not even attempted: casting a negative id to
                                     * uint32_t and indexing with it reads far outside sprob's
                                     * backing allocation, which is not safe to run even as a
                                     * throwaway ablation.  Both halves of the guard stay, backed by
                                     * the ROCm source and this reasoning rather than by a test that
                                     * can discriminate on this hardware. */
                                    const float v = (e >= 0 && (uint32_t)e < N_EXPERT)
                                            ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                                    w_row[j] = v;
                                    sum += v;
                                }
                                sum = sycl::fmax(sum, 6.103515625e-5f);
                                for (uint32_t j = 0; j < n_expert_used; j++) {
                                    w_row[j] = w_row[j] / sum * expert_weight_scale;
                                }
                            }
                        } else {
                            float out_prob[kSyclNExpertUsed] = {0.0f};
                            uint32_t out_idx[kSyclNExpertUsed] = {0u};

                            for (uint32_t k = 0; k < n_expert_used; k++) {
                                float best_score = -INFINITY;
                                float best_prob = 0.0f;
                                uint32_t best_idx = UINT32_MAX;

                                #pragma unroll
                                for (uint32_t j = 0; j < kPerLane; j++) {
                                    const uint32_t e = lane + j * kSyclRouterLanes;
                                    const float s = local_score[j];
                                    if (sycl_router_score_better(s, e, best_score, best_idx)) {
                                        best_score = s;
                                        best_prob = local_prob[j];
                                        best_idx = e;
                                    }
                                }

                                #pragma unroll
                                for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
                                    const float other_score = sycl::permute_group_by_xor(sg, best_score, mask);
                                    const float other_prob = sycl::permute_group_by_xor(sg, best_prob, mask);
                                    const uint32_t other_idx = sycl::permute_group_by_xor(sg, best_idx, mask);
                                    if (sycl_router_score_better(other_score, other_idx, best_score, best_idx)) {
                                        best_score = other_score;
                                        best_prob = other_prob;
                                        best_idx = other_idx;
                                    }
                                }

                                #pragma unroll
                                for (uint32_t j = 0; j < kPerLane; j++) {
                                    const uint32_t e = lane + j * kSyclRouterLanes;
                                    if (e == best_idx) local_score[j] = -INFINITY;
                                }

                                if (lane == 0) {
                                    out_idx[k] = best_idx;
                                    out_prob[k] = best_prob;
                                }
                            }

                            if (lane == 0) {
                                float sum = 0.0f;
                                for (uint32_t j = 0; j < n_expert_used; j++) {
                                    sel_row[j] = (int32_t)out_idx[j];
                                    w_row[j] = out_prob[j];
                                    sum += out_prob[j];
                                }
                                sum = sycl::fmax(sum, 6.103515625e-5f);
                                for (uint32_t j = 0; j < n_expert_used; j++) {
                                    w_row[j] = w_row[j] / sum * expert_weight_scale;
                                }
                            }
                        }
                    }
                });
    });
    /* No wait: q is in_order (ds4_sycl.cpp), so the MoE dispatch that
     * reads `selected`/`weights` is already ordered behind this kernel.
     * The bias/hash table this kernel reads is staged by the caller, not
     * here, so it is the caller that decides whether its guard has
     * anything to keep alive (sycl_any_scratch_frees). */
    ds4_sycl_profile_record_named("router_select", _ds4_prof_ev145);
}

/* Validates the bias/hash table shared by both router entries and resolves
 * where its bytes come from in the host mmap, without touching the device.
 * At most one of bias_bytes/hash_bytes is ever nonzero: ROCm only loads
 * bias when has_bias && !hash_mode (rocm/ds4_rocm_router.cuh:130), and only
 * loads hash when hash_mode, so the two entries below share a single
 * scratch allocation sized by whichever one applies rather than needing
 * one each. On success writes *out_scratch_bytes and *out_scratch_src
 * (both zero/null if neither table is needed) and returns 1; returns 0 on
 * any validation failure, in which case the caller must return 0 without
 * touching either output. */
static int sycl_router_resolve_bias_hash(
        const void *model_map, uint64_t model_size, uint64_t bias_offset,
        uint64_t hash_offset, uint32_t hash_rows, uint32_t active_n_expert,
        uint32_t active_n_expert_used, bool has_bias, bool hash_mode,
        uint64_t *out_scratch_bytes, const void **out_scratch_src) {
    uint64_t bias_bytes = 0, hash_bytes = 0;
    if (has_bias && !hash_mode) {
        if (!sycl_u64_mul_checked(active_n_expert, sizeof(float), &bias_bytes) ||
            !sycl_model_range_fits(model_size, bias_offset, bias_bytes)) {
            return 0;
        }
    }
    if (hash_mode) {
        if (hash_rows == 0u ||
            !sycl_u64_mul3_checked(hash_rows, active_n_expert_used, sizeof(int32_t), &hash_bytes) ||
            !sycl_model_range_fits(model_size, hash_offset, hash_bytes)) {
            return 0;
        }
    }

    const char *bias_src = (has_bias && !hash_mode)
            ? sycl_model_range_ptr(model_map, bias_offset, bias_bytes, model_size, "router_bias")
            : nullptr;
    if (has_bias && !hash_mode && !bias_src) return 0;

    const char *hash_src = hash_mode
            ? sycl_model_range_ptr(model_map, hash_offset, hash_bytes, model_size, "router_hash")
            : nullptr;
    if (hash_mode && !hash_src) return 0;

    *out_scratch_bytes = (has_bias && !hash_mode) ? bias_bytes : (hash_mode ? hash_bytes : 0u);
    *out_scratch_src = (has_bias && !hash_mode) ? (const void *)bias_src
                     : (hash_mode ? (const void *)hash_src : nullptr);
    return 1;
}

}  // namespace

/* Single-token router selection: computes probs[n_expert] from logits, then
 * either top-k selects n_expert_used experts by biased score (weighting the
 * result by the UNBIASED probability, per ds4.c:10769's comment and
 * rocm/ds4_rocm_router.cuh:45 versus :82/:104) or, in hash mode, looks the
 * selection up from a fixed per-token table.  Matches
 * ds4_gpu_router_select_tensor, rocm/ds4_rocm_router.cuh:120-166. */
extern "C" int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : kSyclNExpert;
    const uint32_t active_n_expert_used = n_expert_used != 0u ? n_expert_used : kSyclNExpertUsed;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : kSyclExpertWeightScale;

    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != kSyclNExpert && active_n_expert != kSyclMaxNExpert) ||
        active_n_expert_used > kSyclNExpertUsed ||
        !(active_scale > 0.0f) ||
        !sycl_tensor_has_f32(logits, active_n_expert) ||
        !sycl_tensor_has_f32(probs, active_n_expert) ||
        !sycl_tensor_has_i32(selected, active_n_expert_used) ||
        !sycl_tensor_has_f32(weights, active_n_expert_used)) {
        return 0;
    }

    uint64_t scratch_bytes = 0;
    const void *scratch_src = nullptr;
    if (!sycl_router_resolve_bias_hash(
                model_map, model_size, bias_offset, hash_offset, hash_rows,
                active_n_expert, active_n_expert_used, has_bias, hash_mode,
                &scratch_bytes, &scratch_src)) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    const int32_t tok = (int32_t)token;

    try {
        sycl::queue &q = ds4_sycl_queue(selected->device_id);

        /* The bias/hash table lives in the host mmap (or is already
         * device-resident, per the model-range cache); stage it to device
         * scratch under a guard, same as ds4_sycl_compressor.hpp's APE
         * staging -- never a bare malloc_device/free pair. Skip the call
         * entirely for scratch_bytes == 0: neither has_bias nor hash_mode
         * needs a table then, and scratch_src may be null. */
        sycl_device_scratch_guard scratch_guard = scratch_bytes != 0u
                ? sycl_stage_host_bytes(q, scratch_src, scratch_bytes)
                : sycl_device_scratch_guard(q, nullptr);
        unsigned char *dscratch = (unsigned char *)scratch_guard.p;
        if (scratch_bytes != 0u && !dscratch) return 0;

        const float   *dbias = (has_bias && !hash_mode) ? (const float *)dscratch : nullptr;
        const int32_t *dhash = hash_mode ? (const int32_t *)dscratch : nullptr;

        if (active_n_expert == kSyclMaxNExpert) {
            sycl_router_select_launch<kSyclMaxNExpert>(
                    q, (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    dbias, dhash, (const float *)logits->ptr, nullptr, tok, hash_rows, 1u,
                    active_n_expert_used, active_scale,
                    (has_bias && !hash_mode) ? 1 : 0, hash_mode ? 1 : 0);
        } else {
            sycl_router_select_launch<kSyclNExpert>(
                    q, (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    dbias, dhash, (const float *)logits->ptr, nullptr, tok, hash_rows, 1u,
                    active_n_expert_used, active_scale,
                    (has_bias && !hash_mode) ? 1 : 0, hash_mode ? 1 : 0);
        }
        if (sycl_any_scratch_frees(scratch_guard)) sycl_batch_wait(q);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "router_select launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Batch router selection: the same kernel as ds4_gpu_router_select_tensor
 * above, over n_tokens real rows instead of one, four tokens per
 * work-group.  Matches ds4_gpu_router_select_batch_tensor,
 * rocm/ds4_rocm_router.cuh:167-233.  Differences from the single-token
 * entry: a `tokens` tensor replaces the scalar token id (required, and
 * validated as such, whenever hash_mode is set -- the kernel's own
 * `tokens ? tokens[t] : token_scalar` ternary exists to serve the
 * single-token entry above, which always passes a null tokens pointer;
 * this entry never does), tensor shape validation is 2D
 * (n_tokens x active_n_expert or active_n_expert_used) via
 * sycl_tensor_has_elems2 rather than a flat element count, and
 * n_tokens == 0 is itself a rejection (rocm/ds4_rocm_router.cuh:172), not
 * a zero-work success. */
extern "C" int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : kSyclNExpert;
    const uint32_t active_n_expert_used = n_expert_used != 0u ? n_expert_used : kSyclNExpertUsed;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : kSyclExpertWeightScale;

    if (!selected || !weights || !probs || !logits || !model_map || n_tokens == 0u ||
        n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != kSyclNExpert && active_n_expert != kSyclMaxNExpert) ||
        active_n_expert_used > kSyclNExpertUsed ||
        !(active_scale > 0.0f) ||
        (hash_mode && !sycl_tensor_has_i32(tokens, n_tokens)) ||
        !sycl_tensor_has_elems2(logits, n_tokens, active_n_expert, sizeof(float)) ||
        !sycl_tensor_has_elems2(probs, n_tokens, active_n_expert, sizeof(float)) ||
        !sycl_tensor_has_elems2(selected, n_tokens, active_n_expert_used, sizeof(int32_t)) ||
        !sycl_tensor_has_elems2(weights, n_tokens, active_n_expert_used, sizeof(float))) {
        return 0;
    }

    uint64_t scratch_bytes = 0;
    const void *scratch_src = nullptr;
    if (!sycl_router_resolve_bias_hash(
                model_map, model_size, bias_offset, hash_offset, hash_rows,
                active_n_expert, active_n_expert_used, has_bias, hash_mode,
                &scratch_bytes, &scratch_src)) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(selected->device_id);

        /* Same staging pattern as the single-token entry above: never a
         * bare malloc_device/free pair. */
        sycl_device_scratch_guard scratch_guard = scratch_bytes != 0u
                ? sycl_stage_host_bytes(q, scratch_src, scratch_bytes)
                : sycl_device_scratch_guard(q, nullptr);
        unsigned char *dscratch = (unsigned char *)scratch_guard.p;
        if (scratch_bytes != 0u && !dscratch) return 0;

        const float   *dbias = (has_bias && !hash_mode) ? (const float *)dscratch : nullptr;
        const int32_t *dhash = hash_mode ? (const int32_t *)dscratch : nullptr;
        const int32_t *dtokens = tokens ? (const int32_t *)tokens->ptr : nullptr;

        if (active_n_expert == kSyclMaxNExpert) {
            sycl_router_select_launch<kSyclMaxNExpert>(
                    q, (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    dbias, dhash, (const float *)logits->ptr, dtokens, 0, hash_rows, n_tokens,
                    active_n_expert_used, active_scale,
                    (has_bias && !hash_mode) ? 1 : 0, hash_mode ? 1 : 0);
        } else {
            sycl_router_select_launch<kSyclNExpert>(
                    q, (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    dbias, dhash, (const float *)logits->ptr, dtokens, 0, hash_rows, n_tokens,
                    active_n_expert_used, active_scale,
                    (has_bias && !hash_mode) ? 1 : 0, hash_mode ? 1 : 0);
        }
        if (sycl_any_scratch_frees(scratch_guard)) sycl_batch_wait(q);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "router_select_batch launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
