#pragma once

/* DeepSeek V4 Flash indexer: compressed-row scoring, top-k selection over
 * those scores, vocabulary argmax, the top-k-to-mask conversion, and the
 * Hadamard+FP4 QAT round trip applied to indexer Q and indexer compressor
 * KV rows.
 *
 * Ported from rocm/ds4_rocm_indexer.cuh (1,259 lines, 27 kernel/device
 * definitions across four groups), which is the structural reference, not
 * the CUDA tree. The oracle for the scoring and top-k entries together is
 * indexer_allowed_decode_one (ds4.c:12936-12993): despite its name it runs
 * once per token inside BOTH the single-token decode path
 * (ds4.c:13141) and the token-by-token sliding-window prefill batch
 * (ds4.c:13378, inside layer_attention_raw_swa_batch), so it covers
 * prefill and decode alike, not decode only. Its top-k selection scans
 * candidates in ascending c order and only replaces the current pick on a
 * STRICT `>` (ds4.c:12979: `scores[c] > best_score`), so an exact tie
 * always keeps the earlier, lower-indexed candidate -- the same
 * ascending-index tie-break topk_score_better below encodes explicitly.
 * dsv4_indexer_qat_row_inplace_cpu/_rows_inplace_cpu (ds4.c:3322,3328) are
 * the direct oracles for the QAT entry.
 *
 * Two decisions carried over from earlier research, not re-litigated
 * here:
 *
 * - indexer_topk_8192_cub_kernel (indexer.cuh:392-430) is real CUB usage,
 *   but every one of its four call sites is a defensive probe of opt-in
 *   shared memory that falls straight through to the bitonic
 *   indexer_topk_pow2_u16_kernel<8192> on any failure, computing the
 *   identical result by construction. This port always takes the bitonic
 *   path and never ports the CUB kernel or its two key-packing helpers
 *   (topk_float_ordered_key, topk_pack_key), which exist only to serve it.
 *
 * - indexer_scores_wmma128_kernel (indexer.cuh:128-252) is DS4 Flash's
 *   real production prefill-scoring path (head_dim==128, n_head==64,
 *   g_quality_mode always false on this backend), but indexer_scores_
 *   launch (indexer.cuh:832-881) is a plain three-way dispatch with no
 *   shared exclusion state between its branches, so always falling to the
 *   scalar indexer_scores_kernel instead is fully correct, merely slower
 *   and marginally MORE precise (the scalar path stays F32 throughout;
 *   WMMA narrows Q and index_comp to __half first). THIS IS A DOCUMENTED
 *   PERFORMANCE-ONLY DIVERGENCE FROM ROCM'S DEFAULT DISPATCH, the same
 *   shape as the routed-MoE Q2_K activation divergence -- not an omission.
 *   The decode-only direct kernel, indexer_score_one_direct_kernel, is a
 *   different, non-tensor-core kernel and IS ported: it is what
 *   indexer_scores_launch already reaches for n_tokens==1 with this shape,
 *   ahead of the WMMA check.
 *
 * ds4_gpu_dspark_markov_argmax_tensor and its dspark_markov_argmax_kernel
 * are DSpark-only (a separate, optional speculative-decoding support-model
 * path, ds4.c:2532 DS4_SUPPORT_DSPARK), not part of baseline DeepSeek V4
 * Flash, and stay stubbed; see the stub site in ds4_sycl_unavailable.cpp.
 *
 * None of the seven in-scope entries here read the host model mmap: every
 * argument is already a device ds4_gpu_tensor, so spec 6l's staging
 * requirement does not apply to this file (the DSpark entry is the only
 * one that touches model_map, and it is exactly the one left stubbed).
 *
 * Early-return-before-barrier safety: every early `return` in every kernel
 * below is gated only on values uniform across the WHOLE work-group (a
 * grid/group index such as the row or chunk being processed, or a kernel
 * argument such as head_dim or causal, never a per-lane local id compared
 * against something that varies by lane), so a work-group either takes
 * the return unanimously or none of its work-items do; no code path
 * returns some lanes before a barrier while others in the same group
 * still need to reach it. This is the same reasoning ds4_sycl_router.hpp
 * documents for its own early exits. */

#include "ds4_sycl_common.hpp"

namespace {

/* ---- FP4 E2M1FN value table and nearest-rounding dequantiser --------
 * Ported from dsv4_e2m1fn_value_dev/dsv4_e2m1fn_dequant_dev
 * (rocm/ds4_rocm_norm_rope.cuh:332-357), which match the CPU reference
 * dsv4_e2m1fn_value_cpu/dsv4_e2m1fn_dequant_cpu (ds4.c:3260-3279)
 * exactly. Only 8 codes exist, so this is ported as the literal linear
 * scan ROCm itself uses (unlike the 128-entry E4M3FN table in
 * ds4_sycl_fp8_kv.hpp, which is large enough to justify a binary search):
 * a table small enough that a scan needs no optimising, and a literal
 * port removes any risk of transcribing the tie-break test wrong. The
 * tie-break prefers the EVEN code between two equally-close candidates,
 * ported bit-for-bit including that direction check, since collapsing it
 * to a plain `<=` changes the result for exactly half of all ties. */
static inline float sycl_e2m1fn_value(int code) {
    switch (code & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

static inline float sycl_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = sycl::fmin(sycl::fabs(x), 6.0f);
    int best = 0;
    float best_diff = sycl::fabs(ax - sycl_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = sycl::fabs(ax - sycl_e2m1fn_value(i));
        if (diff < best_diff ||
            (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * sycl_e2m1fn_value(best);
}

/* Shared by every top-k-shaped kernel in this file (bitonic sort networks
 * and the plain vocabulary argmax alike): value wins outright, and an
 * exact tie is broken by ascending index. Matches topk_score_better,
 * rocm/ds4_rocm_indexer.cuh:308-310, and, independently, the reduction
 * shape of argmax_kernel's own tie-break (:254-...) and ds4.c's
 * sample_argmax (ds4.c:38282-38292, `if (v > best_v)` scanning ascending,
 * which keeps the first/lowest-index winner on a tie by construction). */
static inline bool sycl_topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

/* Hadamard-then-FP4-QAT row transform, ported from indexer_hadamard_fp4_
 * kernel (rocm/ds4_rocm_indexer.cuh:1-38). One work-group per row, exactly
 * 128 work-items (head_dim is fixed at 128 by both the launcher's
 * validation and the kernel's own `head_dim != 128u` guard).
 *
 * Stage 1: an in-place 128-wide Hadamard butterfly over `vals`, identical
 * in shape to dsv4_hadamard128_inplace_cpu (ds4.c:3282-3294) but built
 * from shared memory and a barrier per stride instead of a host loop.
 * Stage 2: scale by 1/sqrt(128), then per-32-lane-group max-of-abs
 * reduction (four independent groups within the one 128-wide work-group,
 * matching the FP4 quantiser's 32-wide block granularity), then the FP4
 * round trip via sycl_e2m1fn_dequant, rescaled back up. */
static void sycl_indexer_hadamard_fp4_kernel(
        sycl::nd_item<1> it,
        sycl::local_accessor<float, 1> vals,
        sycl::local_accessor<float, 1> absbuf,
        float *x, uint32_t n_rows, uint32_t head_dim) {
    const uint32_t row = (uint32_t)it.get_group(0);
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    it.barrier(sycl::access::fence_space::local_space);

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    const float v = vals[tid] * 0.08838834764831845f;
    const uint32_t fp4_block = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = sycl::fabs(v);
    it.barrier(sycl::access::fence_space::local_space);

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] =
                    sycl::fmax(absbuf[block_base + lane], absbuf[block_base + lane + stride]);
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    const float amax = sycl::fmax(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = sycl::exp2(sycl::ceil(sycl::log2(amax / 6.0f)));
    xr[tid] = sycl_e2m1fn_dequant(sycl::fmin(6.0f, sycl::fmax(-6.0f, v / scale))) * scale;
}

/* General scalar prefill/batch-decode scoring kernel, ported from
 * indexer_scores_kernel (rocm/ds4_rocm_indexer.cuh:41-77). One work-group
 * per (comp row c, token t) pair, 256 work-items, reducing the per-head
 * dot product across the work-group in shared memory (`partial`) and
 * accumulating the ReLU'd, per-head-weighted sum in `total` across all
 * n_head heads before a single lane writes the result. This is the
 * generic path indexer_scores_launch always falls to whenever the WMMA
 * fast path would otherwise trigger (n_tokens > 1) and whenever the
 * decode-direct kernel's shape does not apply. */
static void sycl_indexer_scores_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> partial,
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_head, uint32_t head_dim, uint32_t ratio,
        float scale, int causal) {
    const uint32_t c = (uint32_t)it.get_group(0);
    const uint32_t t = (uint32_t)it.get_group(1);
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block_dim = (uint32_t)it.get_local_range(0);
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        const uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (tid == 0u) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += block_dim) dot += qh[d] * kh[d];
        partial[tid] = dot;
        it.barrier(sycl::access::fence_space::local_space);
        for (uint32_t s = block_dim >> 1; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            it.barrier(sycl::access::fence_space::local_space);
        }
        total += sycl::fmax(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (tid == 0u) scores[(uint64_t)t * n_comp + c] = total * scale;
}

/* Decode fast path, ported from indexer_score_one_direct_kernel
 * (rocm/ds4_rocm_indexer.cuh:80-127). One work-group per comp row c, 128
 * work-items as four 32-wide sub-groups, each sub-group handling one head
 * at a time (warp = sub-group index 0..3, head = h0 + warp). Each lane
 * owns a contiguous 4-wide slice of the 128-wide head_dim, matching
 * ROCm's float4 vector load exactly via four scalar multiplies in the
 * same operand order, so this is bit-identical without needing a vector
 * type. sycl::reduce_over_group over the sub-group plays the role of
 * ROCm's warp_sum_f32 (rocm/ds4_rocm_common.cuh:350-357), the same
 * substitution already established in ds4_sycl_shared_expert.hpp. */
static void sycl_indexer_score_one_direct_kernel(
        sycl::nd_item<1> it,
        sycl::local_accessor<float, 1> krow,
        sycl::local_accessor<float, 1> partial4,
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t pos0,
        uint32_t ratio, float scale, int causal) {
    const uint32_t c = (uint32_t)it.get_group(0);
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0u) scores[c] = -INFINITY;
            return;
        }
    }

    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    it.barrier(sycl::access::fence_space::local_space);

    sycl::sub_group sg = it.get_sub_group();
    float total = 0.0f;
    const uint32_t base = lane * 4u;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float *qh = q + (uint64_t)h * 128u;
        float dot = qh[base] * krow[base] + qh[base + 1] * krow[base + 1] +
                    qh[base + 2] * krow[base + 2] + qh[base + 3] * krow[base + 3];
        dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        if (lane == 0u) partial4[warp] = sycl::fmax(dot, 0.0f) * weights[h] * scale;
        it.barrier(sycl::access::fence_space::local_space);
        if (tid == 0u) total += partial4[0] + partial4[1] + partial4[2] + partial4[3];
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (tid == 0u) scores[c] = total;
}

/* Ported from indexer_scores_launch (rocm/ds4_rocm_indexer.cuh:832-881),
 * with the WMMA branch removed per the file header comment: the
 * decode-direct kernel is still taken whenever ROCm would take it, and
 * every other case falls to the general scalar kernel. Return polarity is
 * nonzero-success, matching cuda_ok's convention here and every ds4.c
 * call site (e.g. ds4.c:22996, 29269, 29390: `ok = ds4_gpu_indexer_..._
 * tensor(...)` used as a truthy result). */
static int sycl_indexer_scores_launch(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale, uint32_t causal) {
    if (!sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(weights, n_tokens, n_head, sizeof(float)) ||
        !sycl_tensor_has_elems2(index_comp, n_comp, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(scores, n_tokens, n_comp, sizeof(float)) ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(scores->device_id);
        float *pscores = (float *)scores->ptr;
        const float *pq = (const float *)q->ptr;
        const float *pweights = (const float *)weights->ptr;
        const float *pindex_comp = (const float *)index_comp->ptr;
        const int causal_i = causal ? 1 : 0;

        if (n_tokens == 1u && head_dim == 128u && n_head == 64u) {
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> krow(sycl::range<1>(128), h);
                sycl::local_accessor<float, 1> partial4(sycl::range<1>(4), h);
                h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>((size_t)n_comp * 128u),
                                          sycl::range<1>(128)),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                            sycl_indexer_score_one_direct_kernel(
                                    it, krow, partial4, pscores, pq, pweights,
                                    pindex_comp, n_comp, pos0, ratio, scale, causal_i);
                        });
            });
        } else {
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> partial(sycl::range<1>(256), h);
                h.parallel_for(
                        sycl::nd_range<2>(
                                sycl::range<2>((size_t)n_comp * 256u, n_tokens),
                                sycl::range<2>(256, 1)),
                        [=](sycl::nd_item<2> it) {
                            sycl_indexer_scores_kernel(
                                    it, partial, pscores, pq, pweights,
                                    pindex_comp, n_comp, n_tokens, pos0, n_head,
                                    head_dim, ratio, scale, causal_i);
                        });
            });
        }
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "indexer scores launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

}  // namespace

extern "C" float ds4_sycl_test_e2m1fn_value(int code) {
    return sycl_e2m1fn_value(code);
}

extern "C" float ds4_sycl_test_e2m1fn_dequant(float x) {
    return sycl_e2m1fn_dequant(x);
}

/* Decode, n_tokens == 1. Matches rocm/ds4_rocm_indexer.cuh:883-894. */
extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale) {
    return sycl_indexer_scores_launch(scores, q, weights, index_comp, n_comp,
                                      1, 0, n_head, head_dim, 1, scale, 0);
}

/* Causal prefill. Matches rocm/ds4_rocm_indexer.cuh:896-909. */
extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t ratio, float scale) {
    return sycl_indexer_scores_launch(scores, q, weights, index_comp, n_comp,
                                      n_tokens, 0, n_head, head_dim, ratio,
                                      scale, 1);
}

/* Batched decode. Matches rocm/ds4_rocm_indexer.cuh:911-925. */
extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale) {
    return sycl_indexer_scores_launch(scores, q, weights, index_comp, n_comp,
                                      n_tokens, pos0, n_head, head_dim, ratio,
                                      scale, 1);
}

/* Vocabulary argmax, ported from argmax_kernel (rocm/ds4_rocm_indexer.cuh:
 * 254-280). One work-group of 1024 work-items: each lane strides over
 * n_vocab keeping its own best-so-far (a per-lane STRICT `>` scan, so
 * within one lane the lowest-index value among equal maxima already
 * wins), then a power-of-two tree reduction using sycl_topk_score_better
 * to resolve cross-lane ties toward the lower index. Matches ds4_gpu.h's
 * own doc comment on this entry ("Tie-break: lower index wins") and
 * ds4.c's sample_argmax (ds4.c:38282-38292). */
extern "C" int ds4_gpu_argmax_tensor(ds4_gpu_tensor *out_idx,
                                     const ds4_gpu_tensor *logits,
                                     uint32_t n_vocab) {
    uint64_t logits_bytes = 0;
    if (!out_idx || !logits || n_vocab == 0u ||
        out_idx->bytes < sizeof(int32_t) ||
        !sycl_u64_mul3_checked(n_vocab, 1u, sizeof(float), &logits_bytes) ||
        logits->bytes < logits_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    constexpr uint32_t kThreads = 1024u;
    try {
        sycl::queue &dq = ds4_sycl_queue(out_idx->device_id);
        int32_t *pout = (int32_t *)out_idx->ptr;
        const float *plogits = (const float *)logits->ptr;

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> sm_val(sycl::range<1>(kThreads), h);
            sycl::local_accessor<uint32_t, 1> sm_idx(sycl::range<1>(kThreads), h);
            h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(kThreads), sycl::range<1>(kThreads)),
                    [=](sycl::nd_item<1> it) {
                        const uint32_t tid = (uint32_t)it.get_local_id(0);
                        float local_v = -INFINITY;
                        uint32_t local_i = 0;
                        for (uint32_t i = tid; i < n_vocab; i += kThreads) {
                            const float v = plogits[i];
                            if (v > local_v) {
                                local_v = v;
                                local_i = i;
                            }
                        }
                        sm_val[tid] = local_v;
                        sm_idx[tid] = local_i;
                        it.barrier(sycl::access::fence_space::local_space);
                        for (uint32_t s = kThreads / 2u; s > 0u; s >>= 1u) {
                            if (tid < s) {
                                const float vr = sm_val[tid + s];
                                const uint32_t ir = sm_idx[tid + s];
                                if (sycl_topk_score_better(vr, ir, sm_val[tid], sm_idx[tid])) {
                                    sm_val[tid] = vr;
                                    sm_idx[tid] = ir;
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        if (tid == 0u) *pout = (int32_t)sm_idx[0];
                    });
        });
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "argmax launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* DSpark Markov-corrected argmax: a separate, optional speculative-decoding
 * support-model path (DS4_SUPPORT_DSPARK, ds4.c:2532), not part of
 * baseline DeepSeek V4 Flash's own forward pass. Left stubbed in
 * ds4_sycl_unavailable.cpp; see this file's header comment. */

/* Float mask from a top-k index list, ported from topk_mask_kernel
 * (rocm/ds4_rocm_indexer.cuh:816-830). A flat 1D launch over
 * max(n_tokens*n_comp, n_tokens*top_k) work-items, decomposed into (t, c)
 * exactly as ROCm decomposes (gid / n_comp, gid % n_comp); an out-of-range
 * topk entry (e.g. a padding sentinel like UINT32_MAX from an underfull
 * upstream sort) never equals any valid c in [0, n_comp), so it safely
 * contributes nothing rather than reading out of bounds or matching by
 * accident -- the kernel only ever compares values, never indexes memory
 * by a topk entry. */
extern "C" int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *mask,
                                             const ds4_gpu_tensor *topk,
                                             uint32_t n_comp, uint32_t n_tokens,
                                             uint32_t top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        !sycl_tensor_has_elems2(mask, n_tokens, n_comp, sizeof(float)) ||
        !sycl_tensor_has_elems2(topk, n_tokens, top_k, sizeof(uint32_t))) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    const uint64_t n = (uint64_t)n_tokens * n_comp;
    const uint64_t nk = (uint64_t)n_tokens * top_k;
    const uint64_t total = n > nk ? n : nk;
    constexpr uint32_t kBlock = 256u;
    const uint64_t groups = (total + kBlock - 1u) / kBlock;

    try {
        sycl::queue &dq = ds4_sycl_queue(mask->device_id);
        float *pmask = (float *)mask->ptr;
        const uint32_t *ptopk = (const uint32_t *)topk->ptr;

        dq.submit([&](sycl::handler &h) {
            h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)(groups * kBlock)),
                                      sycl::range<1>(kBlock)),
                    [=](sycl::nd_item<1> it) {
                        const uint64_t gid = it.get_global_id(0);
                        if (gid >= n) return;
                        const uint32_t t = (uint32_t)(gid / n_comp);
                        const uint32_t c = (uint32_t)(gid - (uint64_t)t * n_comp);
                        float v = -INFINITY;
                        for (uint32_t k = 0; k < top_k; k++) {
                            if (ptopk[(uint64_t)t * top_k + k] == c) {
                                v = 0.0f;
                                break;
                            }
                        }
                        pmask[gid] = v;
                    });
        });
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "topk mask launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* Hadamard+FP4 QAT round trip, applied in place to n_rows independent
 * 128-wide rows. Matches rocm/ds4_rocm_indexer.cuh:1252-1259. */
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x,
                                               uint32_t n_rows,
                                               uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        !sycl_tensor_has_elems2(x, n_rows, head_dim, sizeof(float))) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(x->device_id);
        float *px = (float *)x->ptr;

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> vals(sycl::range<1>(128), h);
            sycl::local_accessor<float, 1> absbuf(sycl::range<1>(128), h);
            h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)n_rows * 128u),
                                      sycl::range<1>(128)),
                    [=](sycl::nd_item<1> it) {
                        sycl_indexer_hadamard_fp4_kernel(it, vals, absbuf, px,
                                                         n_rows, head_dim);
                    });
        });
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "indexer qat launch failed: %s\n", e.what());
        return 0;
    }
    return 1;
}
