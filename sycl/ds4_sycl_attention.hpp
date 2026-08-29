#pragma once

/* DS4 SYCL attention: decode, batched decode, indexed decode against the
 * indexer's selection, and prefill (small-window static, raw-only GEMM and
 * scalar, and mixed). Ported from rocm/ds4_rocm_attention.cuh (kernels)
 * and rocm/ds4_rocm_attention_launch.cuh (entry points).
 *
 * Scope, and why it stops here: attention_launch.cuh has fourteen entries.
 * This header covers the eight that are decode-reachable and testable
 * today, or that oneMKL now unblocks: ds4_gpu_store_raw_kv_tensor,
 * ds4_gpu_kv_fp8_store_raw_tensor, ds4_gpu_attention_decode_heads_tensor,
 * ds4_gpu_attention_decode_raw_batch_heads_tensor,
 * ds4_gpu_attention_decode_mixed_batch_heads_tensor,
 * ds4_gpu_attention_indexed_mixed_batch_heads_tensor,
 * ds4_gpu_attention_prefill_raw_heads_tensor and
 * ds4_gpu_attention_prefill_static_mixed_heads_tensor.
 *
 * Deliberately NOT here:
 *   - The three output-projection entries
 *     (ds4_gpu_attention_output_q8_batch_f16_tensor,
 *     _q8_batch_tensor, _low_q8_tensor): they need the six-kernel
 *     grouped_q8_0_a_* family plus quantize_q8_0_f32_kernel and
 *     q8_partial_sum8_kernel from ds4_rocm_q8.cuh, used exclusively by
 *     those entries and not ported by the dense matmul work.
 *   - ds4_gpu_attention_noncausal_raw_batch_heads_tensor: DSpark-only,
 *     out of scope for DS4 Flash.
 *   - ds4_gpu_attention_prefill_masked_mixed_heads_tensor
 *     (attention_launch.cuh:943-963): dead for this port.
 *     `grep -n "masked_mixed_heads" ds4.c` returns nothing; its only
 *     caller anywhere is a test_metal_*-prefixed, __APPLE__/DS4_TEST_MODEL
 *     -gated function in tests/ds4_test.c, which is linked into no
 *     test-sycl* Makefile target. Correctly absent from
 *     ds4_sycl_unavailable.cpp: nothing in a SYCL build ever references it.
 *
 * "oldhip" is the default decode path, not a legacy fallback:
 * attention_decode_mixed_one_fast_oldhip_kernel and
 * attention_decode_indexed_mixed_one_fast_oldhip_kernel are selected via
 * cfg->oldhip_attention_decode, set to `!g_quality_mode` at ROCm runtime
 * init. g_quality_mode can never become true on this backend because
 * ds4_gpu_set_quality is a no-op stub (ds4_sycl_unavailable.cpp), the same
 * conclusion reached elsewhere in this backend for other
 * g_quality_mode-gated branches. So the oldhip kernels are treated as the
 * only reachable path for the single-token entries below, not a
 * compatibility shim.
 *
 * attention_indexed_mixed_scalar_kernel (attention.cuh:860-945) is dead:
 * `grep -rn attention_indexed_mixed_scalar_kernel .` from the repo root
 * finds only its own definition, no launch site anywhere. It carries no
 * DS4_ROCM_UNUSED annotation because none of attention.cuh, attention_
 * launch.cuh, indexer.cuh, hc.cuh or hc_output_launch.cuh use that
 * convention (confirmed: `grep -n DS4_ROCM_UNUSED` across all five returns
 * nothing). Its live successor, attention_indexed_mixed_kernel, is what
 * this header ports instead. Not ported here, deliberately.
 *
 * attention_indexed_mixed_heads8_online_kernel<8,32> is HIP-guarded at
 * attention_launch.cuh:547-569 (`#if defined(__HIP_PLATFORM_AMD__) ||
 * defined(__HIPCC__)`). Reading the kernel body (attention.cuh:1103-1265)
 * confirms the HEADS_PER_GROUP template parameter only controls grid shape
 * (how many warps-per-block each own a distinct head, gated by a per-head
 * bounds check with no other coupling to the group width): a pure
 * occupancy choice, fully general for any n_head. This SYCL port always
 * takes the non-HIP fallback shape, <8,16>, unconditionally: correctness-
 * safe, launching more (smaller) grid-y groups than ROCm would, a
 * performance divergence only, not a correctness gap.
 */

#include "ds4_sycl_common.hpp"

namespace {

/* attention_dot_f32_vec4_oldhip, attention.cuh:505-521. `use_vec4` mirrors
 * the ROCm launchers' `(head_dim & 3u) == 0u`, computed on the host and
 * passed down rather than re-derived per lane. The four-way partial-sum
 * grouping (s0+s1)+(s2+s3) is kept exactly as ROCm sums it, since summation
 * order affects which floating-point value comes out; plain sycl::vec loads
 * are not needed for that, only the same grouping of scalar accesses. */
static inline float sycl_attn_dot_vec4(const float *a, const float *b, uint32_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const uint32_t n4 = n >> 2u;
    for (uint32_t i = 0; i < n4; i++) {
        const uint32_t base = i << 2u;
        s0 += a[base + 0u] * b[base + 0u];
        s1 += a[base + 1u] * b[base + 1u];
        s2 += a[base + 2u] * b[base + 2u];
        s3 += a[base + 3u] * b[base + 3u];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (uint32_t i = n4 << 2u; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float sycl_attn_dot(const float *a, const float *b, uint32_t n, uint32_t use_vec4) {
    if (use_vec4) return sycl_attn_dot_vec4(a, b, n);
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* attention_decode_mixed_one_fast_oldhip_kernel, attention.cuh:523-595: the
 * default single-token decode kernel, one work-group per head. The CUDA
 * source reduces `local_max`/`local_sum` with attention_block_max_oldhip_w32
 * / _sum_oldhip_w32 (a warp-shuffle-then-shared-memory two-level scheme,
 * attention.cuh:462-503). Substituted here with sycl_block_row_reduce
 * (ds4_sycl_common.hpp): both compute the identical whole-work-group
 * max/sum over the same per-lane values, so the substitution changes only
 * the reduction's internal mechanism, not its result, and follows the
 * explicit guidance to reuse sycl_block_row_reduce rather than
 * reimplementing a reduction primitive that already exists. As in the CUDA
 * source, no explicit barrier separates the score-computation loops from
 * the block-wide max reduction: sycl_block_row_reduce's own internal
 * barriers (like attention_block_max_oldhip_w32's) fence ALL local memory
 * for the whole work-group, including the `scores` accessor written just
 * above, not only the reduction's own scratch buffer. */
static void sycl_attention_decode_mixed_one_fast_oldhip_kernel(
        sycl::nd_item<1> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        const float *sinks,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t use_mask,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t use_vec4) {
    const uint32_t h = (uint32_t)it.get_group(0);
    if (h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const uint32_t n_rows = n_raw + n_comp;
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = sycl::rsqrt((float)head_dim);

    float local_max = sinks[h];
    for (uint32_t r = tid; r < n_raw; r += block) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = sycl_attn_dot(qh, kv, head_dim, use_vec4) * scale;
        scores[r] = s;
        local_max = sycl::fmax(local_max, s);
    }
    for (uint32_t c = tid; c < n_comp; c += block) {
        float s = -3.4e38f;
        if (!(use_mask && comp_mask && comp_mask[c] <= -5.0e29f)) {
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float dot = sycl_attn_dot(qh, kv, head_dim, use_vec4);
            s = dot * scale;
            if (use_mask && comp_mask) s += comp_mask[c];
        }
        scores[n_raw + c] = s;
        local_max = sycl::fmax(local_max, s);
    }
    const float max_score = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < n_rows; r += block) {
        const float w = sycl::exp(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_score);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = tid; d < head_dim; d += block) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
    }
}

}  // namespace

/* ds4_gpu_store_raw_kv_tensor, attention_launch.cuh:12-18: single-token raw
 * KV store. ROCm's own implementation launches store_raw_kv_batch_kernel
 * with n_tokens fixed at 1; this delegates to the already-ported
 * ds4_gpu_store_raw_kv_batch_tensor (sycl/ds4_sycl_fp8_kv.hpp) the
 * same way, rather than reimplementing the kernel launch. */
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache,
                                           const ds4_gpu_tensor *kv,
                                           uint32_t raw_cap,
                                           uint32_t row,
                                           uint32_t head_dim) {
    return ds4_gpu_store_raw_kv_batch_tensor(raw_cache, kv, raw_cap, row, 1u, head_dim);
}

/* ds4_gpu_kv_fp8_store_raw_tensor, attention_launch.cuh:2-11: FP8 variant.
 * ROCm's own implementation is exactly this composition: quantise the row
 * in place with the FP8 KV quantiser, then store it. Both callees
 * already validate their own arguments; the ROCm source performs no extra
 * validation at this level either. */
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *kv,
                                               ds4_gpu_tensor *raw_cache,
                                               uint32_t raw_cap,
                                               uint32_t raw_row,
                                               uint32_t head_dim,
                                               uint32_t n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1u, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}

/* ds4_gpu_attention_decode_heads_tensor, attention_launch.cuh:146-233:
 * single-token decode across all heads. Only the oldhip fast path is
 * ported: cfg->oldhip_attention_decode is unconditionally true on this
 * backend (see this header's top comment), so the score-buffer-fits /
 * heads8-online / general-scalar branches below the oldhip check in the
 * ROCm source are unreachable here and are not ported by this entry (the
 * general scalar batch kernel and the heads8-online kernel are ported
 * separately, for the batch entries that reach them via
 * attention_decode_batch_launch, which never checks oldhip_attention_decode
 * at all).
 *
 * attn_sinks staging site 1 of 6 (attention_launch.cuh:175): the model
 * mmap's per-head sink-logit bias. Staged via sycl_stage_host_bytes before
 * the kernel launch, per spec 6l: a SYCL kernel handed the raw model_map
 * pointer would read zeros and report success. */
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        uint32_t              n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t              use_mask,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (comp_kv_f16) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems2(heads, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(q, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float)) ||
        (n_comp && !sycl_tensor_has_elems2(comp_kv, n_comp, head_dim, sizeof(float))) ||
        (use_mask && !sycl_tensor_has_f32(comp_mask, n_comp))) {
        return 0;
    }
    /* Zero-work: matches this backend's established convention (see
     * ds4_gpu_store_raw_kv_batch_tensor's comment in ds4_sycl_fp8_kv.hpp)
     * of returning success rather than attempting a kernel launch with a
     * zero grid dimension, which is not a well-defined SYCL-safe
     * equivalent of the CUDA launch's legal zero-block no-op. Not reachable
     * from any real ds4.c call site, which always passes a live n_head. */
    if (n_head == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float       *pheads = (float *)heads->ptr;
        const float *pq     = (const float *)q->ptr;
        const float *praw   = (const float *)raw_kv->ptr;
        const float *pcomp  = n_comp ? (const float *)comp_kv->ptr : nullptr;
        const float *pmask  = use_mask ? (const float *)comp_mask->ptr : nullptr;
        const uint32_t rows = n_raw + n_comp;
        const uint32_t use_vec4 = (head_dim & 3u) == 0u ? 1u : 0u;
        constexpr uint32_t kBlock = 256u;

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scores(sycl::range<1>(rows ? rows : 1u), h);
            sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(kBlock), h);
            h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)n_head * kBlock),
                                      sycl::range<1>(kBlock)),
                    [=](sycl::nd_item<1> it) {
                        sycl_attention_decode_mixed_one_fast_oldhip_kernel(
                                it, scores, reduce_scratch, pheads, pq, praw, pcomp,
                                pmask, psinks, n_raw, raw_cap, raw_start, n_comp,
                                use_mask, n_head, head_dim, use_vec4);
                    });
        });
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention decode oldhip fast launch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

namespace {

/* dot4_f32, rocm/ds4_rocm_common.cuh:396-398: componentwise dot of two
 * float4 vectors. */
static inline float sycl_attn_dot4(sycl::float4 a, sycl::float4 b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/* warp_sum_f32 / attention_warp_sum_oldhip_w32 shape, generalised over the
 * sub-group width: a shuffle-down reduction tree. Ported as a genuine
 * sycl::sub_group reduction rather than a masked partition of a wider warp,
 * per the sub-group-width guidance (spec 6m): a
 * [[sycl::reqd_sub_group_size(N)]] kernel gets a real N-wide hardware
 * sub-group on this stack, so no mask is needed the way CUDA needs one to
 * fence a shuffle to a sub-range of its 32-wide warp. As with CUDA's
 * shfl_down-based reduction, only lane 0 holds the fully-reduced total when
 * this returns; callers that need every lane to see it broadcast lane 0's
 * value afterwards (sycl::group_broadcast), exactly mirroring the
 * `__shfl_sync(mask, score, 0)` immediately following every warp_sum_f32
 * call at its CUDA call sites. */
template <int N>
static inline float sycl_attn_subgroup_sum(sycl::sub_group sg, float v) {
    for (int offset = N >> 1; offset > 0; offset >>= 1) {
        v += sycl::shift_group_left(sg, v, (uint32_t)offset);
    }
    return v;
}

/* attention_decode_mixed_kernel, attention.cuh:692-858: the general scalar
 * decode-batch kernel, one work-group per (token, head). Two score-scoring
 * shapes depending on whether any compressed row is visible and whether
 * this is a true batch (n_tokens > 1): a plain per-lane strided loop when
 * there is nothing to gain from parallelising a single row's dot product
 * (visible_comp == 0 or n_tokens == 1, matching the CUDA source exactly),
 * and an 8-lane-per-row grouping, genuinely 8-wide via reqd_sub_group_size,
 * otherwise. The block-wide max/sum reductions (CUDA's own plain shared-
 * memory `partial[]` tree, not the oldhip warp-shuffle scheme) are
 * substituted with sycl_block_row_reduce for the same reason as the oldhip
 * kernel above: identical whole-work-group reduction, reused rather than
 * reimplemented. `raw_meta` packs the CUDA source's two `__shared__
 * uint32_t raw_count`/`raw_first_idx` singletons into a two-element local
 * buffer written by lane 0 and read by every lane after the barrier that
 * follows. */
static void sycl_attention_decode_mixed_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<uint32_t, 1> raw_rows,
        sycl::local_accessor<float, 1> reduce_scratch,
        sycl::local_accessor<uint32_t, 1> raw_meta,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = sycl::rsqrt((float)head_dim);

    if (tid == 0u) {
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_meta[0] = raw_count;
        raw_meta[1] = raw_first_idx;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t raw_count = raw_meta[0];
    const uint32_t raw_first_idx = raw_meta[1];
    for (uint32_t r = tid; r < raw_count; r += block) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    it.barrier(sycl::access::fence_space::local_space);

    const uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0u || n_tokens == 1u) {
        for (uint32_t r = tid; r < raw_count; r += block) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = sycl::fmax(local_max, scores[r]);
        }
        for (uint32_t c = tid; c < visible_comp; c += block) {
            const float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = sycl::fmax(local_max, s);
        }
    } else {
        sycl::sub_group sg = it.get_sub_group();
        const uint32_t qlane = tid & 7u;
        const uint32_t qgroup = tid >> 3u;
        for (uint32_t row0 = 0u; row0 < n_score; row0 += 32u) {
            const uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = nullptr;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    const uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    dot = sycl_attn_subgroup_sum<8>(sg, dot);
                    s = dot * scale + add;
                }
                if (qlane == 0u) scores[row] = s;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (uint32_t i = tid; i < n_score; i += block) {
            local_max = sycl::fmax(local_max, scores[i]);
        }
    }

    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < n_score; i += block) {
        scores[i] = sycl::exp(scores[i] - max_s);
        local_sum += scores[i];
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && block == 256u) {
        const uint32_t d0 = tid, d1 = d0 + 256u;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = tid; d < head_dim; d += block) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

/* attention_decode_mixed_heads8_online_kernel, attention.cuh:1390-1561: the
 * head_dim == 512 fast path, one 32-wide sub-group (warp) per head, up to
 * eight heads per work-group. Despite the "online" name this is a two-pass
 * algorithm (a max-only pass, then a second pass that recomputes every
 * score and accumulates the weighted KV sum against the now-final max), not
 * a single-pass online-softmax recurrence with incremental rescaling; that
 * genuine online form is attention_indexed_mixed_heads8_online_kernel
 * (ported separately). `max_s` is still a running maximum ACROSS TILES
 * within pass one (`max_s = fmax(max_s, score)` accumulated tile by tile as
 * `row0` advances), which is what an ablation of this tracking would
 * target. KV rows are staged into `kv_shared` cooperatively by the WHOLE
 * work-group once per 4-row tile and read by every warp, since the staged
 * rows do not depend on which head a warp owns. */
static void sycl_attention_decode_mixed_heads8_online_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<uint32_t, 1> raw_rows,
        sycl::local_accessor<uint32_t, 1> raw_meta,
        sycl::local_accessor<sycl::float4, 1> kv_shared,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t head_group = (uint32_t)it.get_group(1);
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_id()[0];
    const uint32_t warp = tid >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0u;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (tid == 0u) {
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_meta[0] = raw_count;
        raw_meta[1] = raw_first_idx;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t raw_count = raw_meta[0];
    const uint32_t raw_first_idx = raw_meta[1];
    for (uint32_t r = tid; r < raw_count; r += block) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    it.barrier(sycl::access::fence_space::local_space);

    const uint32_t n_score = raw_count + comp_count;
    const float scale = sycl::rsqrt((float)head_dim);
    sycl::float4 q0(0.0f), q1(0.0f), q2(0.0f), q3(0.0f);
    if (valid_head) {
        const sycl::float4 *q4 =
                (const sycl::float4 *)(q + ((uint64_t)t * n_head + head) * head_dim);
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    float max_s = valid_head ? sinks[head] : -INFINITY;

    for (uint32_t row0 = 0u; row0 < n_score; row0 += 4u) {
        const uint32_t nr = (n_score - row0 < 4u) ? (n_score - row0) : 4u;
        for (uint32_t off = tid; off < nr * 128u; off += block) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const sycl::float4 *src = sr < raw_count
                    ? (const sycl::float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                    : (const sycl::float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const sycl::float4 k0 = kv_shared[rr * 128u + lane + 0u];
                const sycl::float4 k1 = kv_shared[rr * 128u + lane + 32u];
                const sycl::float4 k2 = kv_shared[rr * 128u + lane + 64u];
                const sycl::float4 k3 = kv_shared[rr * 128u + lane + 96u];
                float score = sycl_attn_dot4(q0, k0) + sycl_attn_dot4(q1, k1) +
                              sycl_attn_dot4(q2, k2) + sycl_attn_dot4(q3, k3);
                score = sycl_attn_subgroup_sum<32>(sg, score) * scale;
                score = sycl::group_broadcast(sg, score, 0u);
                max_s = sycl::fmax(max_s, score);
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    float sum_s = valid_head ? sycl::exp(sinks[head] - max_s) : 0.0f;
    sycl::float4 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f);

    for (uint32_t row0 = 0u; row0 < n_score; row0 += 4u) {
        const uint32_t nr = (n_score - row0 < 4u) ? (n_score - row0) : 4u;
        for (uint32_t off = tid; off < nr * 128u; off += block) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const sycl::float4 *src = sr < raw_count
                    ? (const sycl::float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                    : (const sycl::float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const sycl::float4 k0 = kv_shared[rr * 128u + lane + 0u];
                const sycl::float4 k1 = kv_shared[rr * 128u + lane + 32u];
                const sycl::float4 k2 = kv_shared[rr * 128u + lane + 64u];
                const sycl::float4 k3 = kv_shared[rr * 128u + lane + 96u];
                float score = sycl_attn_dot4(q0, k0) + sycl_attn_dot4(q1, k1) +
                              sycl_attn_dot4(q2, k2) + sycl_attn_dot4(q3, k3);
                score = sycl_attn_subgroup_sum<32>(sg, score) * scale;
                score = sycl::group_broadcast(sg, score, 0u);
                const float row_scale = sycl::exp(score - max_s);
                sum_s += row_scale;
                o0 += k0 * row_scale;
                o1 += k1 * row_scale;
                o2 += k2 * row_scale;
                o3 += k3 * row_scale;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0 *= inv_s;
        o1 *= inv_s;
        o2 *= inv_s;
        o3 *= inv_s;
        sycl::float4 *out4 = (sycl::float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* attention_decode_batch_launch, attention_launch.cuh:333-423: the shared
 * static helper both batch entries below delegate to. Ported once here.
 *
 * attn_sinks staging site 2 of 6 (attention_launch.cuh:366).
 *
 * Branch order matches the ROCm source exactly: an oversized compressed-row
 * count first tries the head_dim == 512 online path and otherwise refuses
 * (a real limit, not exercised by these tests: DS4_ROCM_ATTENTION_
 * SCORE_CAP - DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 8192 - 256 = 7936
 * compressed rows); then, for a normal-sized compressed count, an unmasked
 * batch at head_dim == 512 also takes the online path
 * (fast_window_attention is `!g_quality_mode`, unconditionally true on this
 * backend per this header's top comment); everything else takes the
 * general scalar kernel. */
static int sycl_attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t              use_comp_mask,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0u || n_raw == 0u ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0u && !comp_kv) || (use_comp_mask && !comp_mask) ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float)) ||
        (n_comp && !sycl_tensor_has_elems2(comp_kv, n_comp, head_dim, sizeof(float))) ||
        (use_comp_mask &&
         !sycl_tensor_has_elems2(comp_mask, n_tokens, n_comp, sizeof(float)))) {
        return 0;
    }
    if (n_comp != 0u && ratio == 0u) return 0;
    if (n_head == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    constexpr uint32_t kScoreCap = 8192u;
    constexpr uint32_t kRawScoreCap = 256u;
    const bool score_fits = n_comp <= (kScoreCap - kRawScoreCap);
    if (!score_fits && (use_comp_mask || head_dim != 512u)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    const bool use_online =
            !score_fits || (!use_comp_mask && n_tokens > 1u && head_dim == 512u);

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float       *pheads = (float *)heads->ptr;
        const float *pq     = (const float *)q->ptr;
        const float *praw   = (const float *)raw_kv->ptr;
        const float *pcomp  = n_comp ? (const float *)comp_kv->ptr : nullptr;
        const float *pmask  = use_comp_mask ? (const float *)comp_mask->ptr : nullptr;
        constexpr uint32_t kBlock = 256u;

        if (use_online) {
            const uint32_t grid_y = (n_head + 7u) / 8u;
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> raw_rows(sycl::range<1>(kRawScoreCap), h);
                sycl::local_accessor<uint32_t, 1> raw_meta(sycl::range<1>(2), h);
                sycl::local_accessor<sycl::float4, 1> kv_shared(sycl::range<1>(4u * 128u), h);
                h.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * kBlock, grid_y),
                                          sycl::range<2>(kBlock, 1)),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                            sycl_attention_decode_mixed_heads8_online_kernel(
                                    it, raw_rows, raw_meta, kv_shared, pheads, psinks, pq,
                                    praw, pcomp, n_tokens, pos0, n_raw, raw_cap, raw_start,
                                    n_comp, window, ratio, n_head, head_dim);
                        });
            });
        } else {
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> scores(sycl::range<1>(kScoreCap), h);
                sycl::local_accessor<uint32_t, 1> raw_rows(sycl::range<1>(kRawScoreCap), h);
                sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(kBlock), h);
                sycl::local_accessor<uint32_t, 1> raw_meta(sycl::range<1>(2), h);
                h.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * kBlock, n_head),
                                          sycl::range<2>(kBlock, 1)),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                            sycl_attention_decode_mixed_kernel(
                                    it, scores, raw_rows, reduce_scratch, raw_meta, pheads,
                                    psinks, pq, praw, pcomp, pmask, use_comp_mask, n_tokens,
                                    pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                                    n_head, head_dim);
                        });
            });
        }
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention decode batch launch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

}  // namespace

/* ds4_gpu_attention_decode_raw_batch_heads_tensor, attention_launch.cuh:
 * 425-444: raw-only thin wrapper over attention_decode_batch_launch, ratio
 * fixed at 1 (no compressed rows) matching the ROCm source's own call. */
extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              window,
        uint32_t              n_head,
        uint32_t              head_dim) {
    return sycl_attention_decode_batch_launch(
            heads, model_map, model_size, sinks_offset, q, raw_kv, nullptr, nullptr, 0u,
            n_tokens, pos0, n_raw, raw_cap, raw_start, 0u, window, 1u, n_head, head_dim);
}

/* ds4_gpu_attention_decode_mixed_batch_heads_tensor, attention_launch.cuh:
 * 446-472: raw plus compressed thin wrapper over
 * attention_decode_batch_launch. */
extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t              use_comp_mask,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (comp_kv_f16) return 0;
    return sycl_attention_decode_batch_launch(
            heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv, comp_mask,
            use_comp_mask, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
            n_head, head_dim);
}

namespace {

/* warp_max_f32 shape, generalised over sub-group width, matching
 * sycl_attn_subgroup_sum's own generalisation. */
template <int N>
static inline float sycl_attn_subgroup_max(sycl::sub_group sg, float v) {
    for (int offset = N >> 1; offset > 0; offset >>= 1) {
        v = sycl::fmax(v, sycl::shift_group_left(sg, v, (uint32_t)offset));
    }
    return v;
}

/* attention_decode_indexed_mixed_one_fast_oldhip_kernel, attention.cuh:
 * 597-691: the default single-token indexed decode kernel, one work-group
 * per head. Structurally the indexed sibling of
 * sycl_attention_decode_mixed_one_fast_oldhip_kernel above: the same
 * oldhip block-reduction substitution applies for the same reason. The
 * topk list is compacted into `comp_rows` by lane 0 once, up to
 * DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP (1024) entries, before any lane
 * scores a compressed row. */
static void sycl_attention_decode_indexed_mixed_one_fast_oldhip_kernel(
        sycl::nd_item<1> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<float, 1> reduce_scratch,
        sycl::local_accessor<uint32_t, 1> comp_rows,
        sycl::local_accessor<uint32_t, 1> comp_meta,
        float *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        const float *sinks,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t pos0,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t use_vec4) {
    const uint32_t h = (uint32_t)it.get_group(0);
    if (h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = sycl::rsqrt((float)head_dim);
    constexpr uint32_t kTopkCap = 1024u;

    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (pos0 + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    if (tid == 0u) {
        uint32_t comp_count = 0u;
        for (uint32_t i = 0; i < top_k && comp_count < kTopkCap; i++) {
            const int32_t ci = topk[i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) comp_rows[comp_count++] = c;
        }
        comp_meta[0] = comp_count;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t comp_count = comp_meta[0];
    const uint32_t n_rows = n_raw + comp_count;

    float local_max = sinks[h];
    for (uint32_t r = tid; r < n_raw; r += block) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = sycl_attn_dot(qh, kv, head_dim, use_vec4) * scale;
        scores[r] = s;
        local_max = sycl::fmax(local_max, s);
    }
    for (uint32_t c = tid; c < comp_count; c += block) {
        const uint32_t row = comp_rows[c];
        const float *kv = comp_kv + (uint64_t)row * head_dim;
        float dot = sycl_attn_dot(qh, kv, head_dim, use_vec4);
        const float s = dot * scale;
        scores[n_raw + c] = s;
        local_max = sycl::fmax(local_max, s);
    }
    const float max_score = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < n_rows; r += block) {
        const float w = sycl::exp(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_score);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = tid; d < head_dim; d += block) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            const uint32_t row = comp_rows[c];
            acc += scores[n_raw + c] * comp_kv[(uint64_t)row * head_dim + d];
        }
        heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
    }
}

/* attention_indexed_mixed_kernel, attention.cuh:946-1102: the general
 * scalar indexed decode-batch kernel, one work-group per (token, head).
 * Structurally the indexed sibling of sycl_attention_decode_mixed_kernel:
 * same causal raw-window derivation (this kernel's CUDA source has no
 * single_all special case, unlike the non-indexed sibling, so none is
 * ported here either), same 8-wide-vs-plain scoring split, same
 * sycl_block_row_reduce substitution for the block-wide reductions. The
 * per-token topk list is compacted into `comp_rows` by lane 0, capped at
 * DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP (1024) entries, exactly mirroring the
 * oldhip kernel above. */
static void sycl_attention_indexed_mixed_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<uint32_t, 1> raw_rows,
        sycl::local_accessor<uint32_t, 1> comp_rows,
        sycl::local_accessor<float, 1> reduce_scratch,
        sycl::local_accessor<uint32_t, 1> raw_meta,
        sycl::local_accessor<uint32_t, 1> comp_meta,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = sycl::rsqrt((float)head_dim);
    constexpr uint32_t kTopkCap = 1024u;

    if (tid == 0u) {
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_meta[0] = raw_count;
        raw_meta[1] = raw_first_idx;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t raw_count = raw_meta[0];
    const uint32_t raw_first_idx = raw_meta[1];
    for (uint32_t r = tid; r < raw_count; r += block) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (tid == 0u) {
        uint32_t comp_count = 0u;
        for (uint32_t i = 0; i < top_k && comp_count < kTopkCap; i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci >= 0 && (uint32_t)ci < visible_comp) comp_rows[comp_count++] = (uint32_t)ci;
        }
        comp_meta[0] = comp_count;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t comp_count = comp_meta[0];
    const uint32_t n_score = raw_count + comp_count;

    float local_max = sinks[h];
    if (comp_count == 0u) {
        for (uint32_t r = tid; r < raw_count; r += block) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = sycl::fmax(local_max, scores[r]);
        }
    } else {
        sycl::sub_group sg = it.get_sub_group();
        const uint32_t qlane = tid & 7u;
        const uint32_t qgroup = tid >> 3u;
        for (uint32_t row0 = 0u; row0 < n_score; row0 += 32u) {
            const uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                        ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                        : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                dot = sycl_attn_subgroup_sum<8>(sg, dot);
                if (qlane == 0u) scores[row] = dot * scale;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (uint32_t i = tid; i < n_score; i += block) {
            local_max = sycl::fmax(local_max, scores[i]);
        }
    }

    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < n_score; i += block) {
        scores[i] = sycl::exp(scores[i] - max_s);
        local_sum += scores[i];
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && block == 256u) {
        const uint32_t d0 = tid, d1 = d0 + 256u;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = tid; d < head_dim; d += block) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                acc += comp_kv[(uint64_t)comp_rows[c] * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

/* attention_indexed_mixed_heads8_online_kernel<8,16>, attention.cuh:
 * 1103-1265: the head_dim == 512, top_k <= 1024 fast path, one 32-wide
 * sub-group per head, 16 heads per work-group (512 threads). Unlike
 * attention_decode_mixed_heads8_online_kernel above, this one is a genuine
 * single-pass online softmax: `max_s`/`sum_s`/the four float4 accumulators
 * are all rescaled together every time a new row raises the running
 * maximum (`old_scale = exp(max_s - new_m)`), so the whole running state
 * only ever reflects the correct normalisation for the rows seen so far,
 * with no second pass needed. `sum_s` starts at 1.0f, representing the
 * sink's own contribution exp(sinks[head] - max_s) evaluated at max_s ==
 * sinks[head] (its initial value) -- exactly 1 -- and is rescaled by every
 * subsequent `old_scale` alongside every real row's contribution, so the
 * sink never needs a separate additive term the way the two-pass kernels
 * need one.
 *
 * The <8,32> HIP-only instantiation (attention_launch.cuh:547-569,
 * `#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)`) is not ported:
 * reading the kernel body confirms HEADS_PER_GROUP only controls grid
 * shape (`head = head_group * HEADS_PER_GROUP + warp`, gated by a per-head
 * `valid_head` bounds check with no other coupling to the group width), so
 * this SYCL port always takes the shape the non-HIP fallback already uses,
 * <8,16>, launching more (smaller) grid-y groups than ROCm's HIP path
 * would -- a performance divergence, not a correctness gap. */
static void sycl_attention_indexed_mixed_heads8_online_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<uint32_t, 1> raw_rows,
        sycl::local_accessor<uint32_t, 1> comp_rows,
        sycl::local_accessor<uint32_t, 1> raw_meta,
        sycl::local_accessor<uint32_t, 1> comp_meta,
        sycl::local_accessor<sycl::float4, 1> kv_shared,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t head_group = (uint32_t)it.get_group(1);
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_id()[0];
    const uint32_t warp = tid >> 5u;
    constexpr uint32_t kHeadsPerGroup = 16u;
    constexpr uint32_t kRowsPerStage = 8u;
    constexpr uint32_t kTopkCap = 1024u;
    const uint32_t head = head_group * kHeadsPerGroup + warp;
    const bool valid_head = head < n_head;

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    if (tid == 0u) {
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_meta[0] = raw_count;
        raw_meta[1] = raw_first_idx;
        uint32_t comp_count = 0u;
        for (uint32_t i = 0; i < top_k && comp_count < kTopkCap; i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) comp_rows[comp_count++] = c;
        }
        comp_meta[0] = comp_count;
    }
    it.barrier(sycl::access::fence_space::local_space);
    const uint32_t raw_count = raw_meta[0];
    const uint32_t raw_first_idx = raw_meta[1];
    for (uint32_t r = tid; r < raw_count; r += block) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    it.barrier(sycl::access::fence_space::local_space);

    const uint32_t comp_count = comp_meta[0];
    const uint32_t n_score = raw_count + comp_count;
    const float scale = sycl::rsqrt((float)head_dim);
    sycl::float4 q0(0.0f), q1(0.0f), q2(0.0f), q3(0.0f);
    if (valid_head) {
        const sycl::float4 *q4 =
                (const sycl::float4 *)(q + ((uint64_t)t * n_head + head) * head_dim);
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    float sum_s = valid_head ? 1.0f : 0.0f;
    sycl::float4 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f);

    for (uint32_t row0 = 0u; row0 < n_score; row0 += kRowsPerStage) {
        const uint32_t nr = (n_score - row0 < kRowsPerStage) ? (n_score - row0) : kRowsPerStage;
        for (uint32_t off = tid; off < nr * 128u; off += block) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const sycl::float4 *src = sr < raw_count
                    ? (const sycl::float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                    : (const sycl::float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const sycl::float4 k0 = kv_shared[rr * 128u + lane + 0u];
                const sycl::float4 k1 = kv_shared[rr * 128u + lane + 32u];
                const sycl::float4 k2 = kv_shared[rr * 128u + lane + 64u];
                const sycl::float4 k3 = kv_shared[rr * 128u + lane + 96u];
                float score = sycl_attn_dot4(q0, k0) + sycl_attn_dot4(q1, k1) +
                              sycl_attn_dot4(q2, k2) + sycl_attn_dot4(q3, k3);
                score = sycl_attn_subgroup_sum<32>(sg, score) * scale;
                score = sycl::group_broadcast(sg, score, 0u);

                const float new_m = sycl::fmax(max_s, score);
                const float old_scale = sycl::exp(max_s - new_m);
                const float row_scale = sycl::exp(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0 = o0 * old_scale + k0 * row_scale;
                o1 = o1 * old_scale + k1 * row_scale;
                o2 = o2 * old_scale + k2 * row_scale;
                o3 = o3 * old_scale + k3 * row_scale;
                max_s = new_m;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0 *= inv_s;
        o1 *= inv_s;
        o2 *= inv_s;
        o3 *= inv_s;
        sycl::float4 *out4 = (sycl::float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* attention_static_mixed_heads8_online_kernel, attention.cuh:1266-1389: the
 * small-window static prefill fast path, one 32-wide sub-group per head,
 * 8 heads per work-group. Despite the "online" name this is a two-pass
 * algorithm sharing a per-warp `scores` staging buffer (max discovery, then
 * normalise-and-accumulate) rather than a single-pass recurrence -- the
 * comment on ROCm's own source explains why: an earlier online recurrence
 * here "crossed greedy near-ties on long prompts", so this kernel
 * deliberately uses the same two-pass score/softmax shape as the oldhip
 * warp-rows path instead. Per-lane dot products read `q`/`kv_shared`
 * directly as flat arrays with a plain strided scalar loop, matching the
 * CUDA source exactly (no float4 vectorised loads here, unlike the other
 * heads8 kernels), then a real 32-wide sub-group sum. `n_score > 768`
 * returns without writing anything, mirroring the CUDA source's own
 * `__shared__ float scores[8 * 768]; if (n_score > 768u) return;` bound. */
static void sycl_attention_static_mixed_heads8_online_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<sycl::float4, 1> kv_shared,
        sycl::local_accessor<float, 1> scores,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t head_group = (uint32_t)it.get_group(1);
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_id()[0];
    const uint32_t warp = tid >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    const uint32_t raw_count = (window != 0u && t + 1u > window) ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0u;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = sycl::rsqrt((float)head_dim);
    if (n_score > 768u) return;

    for (uint32_t row0 = 0u; row0 < n_score; row0 += 4u) {
        const uint32_t nr = (n_score - row0 < 4u) ? (n_score - row0) : 4u;
        for (uint32_t off = tid; off < nr * 128u; off += block) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const sycl::float4 *src = sr < raw_count
                    ? (const sycl::float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                    : (const sycl::float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (valid_head) {
            const float *qh = q + ((uint64_t)t * n_head + head) * head_dim;
            for (uint32_t rr = 0; rr < nr; rr++) {
                float dot = 0.0f;
                for (uint32_t d = lane; d < 512u; d += 32u) {
                    dot += qh[d] * ((const float *)&kv_shared[rr * 128u])[d];
                }
                dot = sycl_attn_subgroup_sum<32>(sg, dot);
                if (lane == 0u) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        for (uint32_t i = lane; i < n_score; i += 32u) {
            max_s = sycl::fmax(max_s, scores[warp * 768u + i]);
        }
        max_s = sycl_attn_subgroup_max<32>(sg, max_s);
        max_s = sycl::group_broadcast(sg, max_s, 0u);
    }
    float den = 0.0f;
    if (valid_head) {
        for (uint32_t i = lane; i < n_score; i += 32u) {
            const float p = sycl::exp(scores[warp * 768u + i] - max_s);
            scores[warp * 768u + i] = p;
            den += p;
        }
        den = sycl_attn_subgroup_sum<32>(sg, den);
        den += sycl::exp(sinks[head] - max_s);
        den = sycl::group_broadcast(sg, den, 0u);
    }

    sycl::float4 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f);
    for (uint32_t row0 = 0u; row0 < n_score; row0 += 4u) {
        const uint32_t nr = (n_score - row0 < 4u) ? (n_score - row0) : 4u;
        for (uint32_t off = tid; off < nr * 128u; off += block) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const sycl::float4 *src = sr < raw_count
                    ? (const sycl::float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                    : (const sycl::float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : scores[warp * 768u + row0 + rr] / den;
                const sycl::float4 k0 = kv_shared[rr * 128u + lane + 0u];
                const sycl::float4 k1 = kv_shared[rr * 128u + lane + 32u];
                const sycl::float4 k2 = kv_shared[rr * 128u + lane + 64u];
                const sycl::float4 k3 = kv_shared[rr * 128u + lane + 96u];
                o0 += k0 * p;
                o1 += k1 * p;
                o2 += k2 * p;
                o3 += k3 * p;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (valid_head) {
        sycl::float4 *out4 = (sycl::float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

}  // namespace

/* ds4_gpu_attention_indexed_mixed_batch_heads_tensor, attention_launch.cuh:
 * 474-921: the largest entry ported in this header. Branch order matches
 * the ROCm source exactly: n_tokens == 1 always takes the oldhip fast
 * kernel (per this header's top comment, oldhip_attention_decode is
 * unconditionally true here); n_tokens > 1 with top_k == 512 first sorts
 * the per-token topk list ascending via indexed_topk_sort_512_asc_kernel
 * (this is that kernel's first consumer: its contract -- sort each row of top_k
 * int32 values, including negative sentinels, into ascending order in
 * place -- matches exactly what this entry needs, since every downstream
 * kernel here already treats a negative topk entry as "skip" regardless of
 * where it sorts to); n_tokens > 1 with head_dim == 512 and top_k within
 * the cap takes the online heads8 kernel (always the <8,16> shape, per
 * this header's top comment); everything else falls through to the
 * general scalar kernel.
 *
 * attn_sinks staging site 3 of 6 (attention_launch.cuh:509). */
extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              top_k,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (comp_kv_f16) return 0;
    constexpr uint32_t kTopkCap = 1024u;
    if (!heads || !q || !raw_kv || !comp_kv || !topk || !model_map || n_tokens == 0u ||
        n_raw == 0u || raw_cap < n_raw || raw_start >= raw_cap || n_comp == 0u ||
        top_k == 0u ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(comp_kv, n_comp, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(topk, n_tokens, top_k, sizeof(int32_t))) {
        return 0;
    }
    if (top_k > kTopkCap) return 0;
    if (n_head == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float         *pheads = (float *)heads->ptr;
        const float   *pq     = (const float *)q->ptr;
        const float   *praw   = (const float *)raw_kv->ptr;
        const float   *pcomp  = (const float *)comp_kv->ptr;
        const int32_t *ptopk  = (const int32_t *)topk->ptr;
        constexpr uint32_t kBlock = 256u;

        if (n_tokens == 1u) {
            const uint32_t rows = n_raw + (top_k < n_comp ? top_k : n_comp);
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> scores(sycl::range<1>(rows ? rows : 1u), h);
                sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(kBlock), h);
                sycl::local_accessor<uint32_t, 1> comp_rows(sycl::range<1>(kTopkCap), h);
                sycl::local_accessor<uint32_t, 1> comp_meta(sycl::range<1>(1), h);
                const uint32_t use_vec4 = (head_dim & 3u) == 0u ? 1u : 0u;
                h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>((size_t)n_head * kBlock),
                                          sycl::range<1>(kBlock)),
                        [=](sycl::nd_item<1> it) {
                            sycl_attention_decode_indexed_mixed_one_fast_oldhip_kernel(
                                    it, scores, reduce_scratch, comp_rows, comp_meta, pheads,
                                    pq, praw, pcomp, ptopk, psinks, n_raw, raw_cap, raw_start,
                                    n_comp, top_k, pos0, ratio, n_head, head_dim, use_vec4);
                        });
            });
            dq.wait_and_throw();
            return 1;
        }

        const int32_t *psorted = ptopk;
        int32_t *sorted_buf = nullptr;
        if (top_k == 512u) {
            sorted_buf = sycl::malloc_device<int32_t>((size_t)n_tokens * 512u, dq);
            if (!sorted_buf) return 0;
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<int32_t, 1> rows(sycl::range<1>(512), h);
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * 512u),
                                                 sycl::range<1>(512u)),
                               [=](sycl::nd_item<1> it) {
                                   sycl_indexed_topk_sort_512_asc_kernel(it, rows, sorted_buf,
                                                                         ptopk, n_tokens);
                               });
            });
            dq.wait_and_throw();
            psorted = sorted_buf;
        }
        sycl_device_scratch_guard sorted_guard(dq, sorted_buf);

        if (head_dim == 512u && top_k <= kTopkCap) {
            const uint32_t grid_y = (n_head + 15u) / 16u;
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<uint32_t, 1> raw_rows(sycl::range<1>(256), h);
                sycl::local_accessor<uint32_t, 1> comp_rows(sycl::range<1>(kTopkCap), h);
                sycl::local_accessor<uint32_t, 1> raw_meta(sycl::range<1>(2), h);
                sycl::local_accessor<uint32_t, 1> comp_meta(sycl::range<1>(1), h);
                sycl::local_accessor<sycl::float4, 1> kv_shared(sycl::range<1>(8u * 128u), h);
                h.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 512u, grid_y),
                                          sycl::range<2>(512u, 1u)),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                            sycl_attention_indexed_mixed_heads8_online_kernel(
                                    it, raw_rows, comp_rows, raw_meta, comp_meta, kv_shared,
                                    pheads, psinks, pq, praw, pcomp, psorted, n_tokens, pos0,
                                    n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio,
                                    n_head, head_dim);
                        });
            });
            dq.wait_and_throw();
            return 1;
        }

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scores(sycl::range<1>(256u + kTopkCap), h);
            sycl::local_accessor<uint32_t, 1> raw_rows(sycl::range<1>(256), h);
            sycl::local_accessor<uint32_t, 1> comp_rows(sycl::range<1>(kTopkCap), h);
            sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(kBlock), h);
            sycl::local_accessor<uint32_t, 1> raw_meta(sycl::range<1>(2), h);
            sycl::local_accessor<uint32_t, 1> comp_meta(sycl::range<1>(1), h);
            h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * kBlock, n_head),
                                      sycl::range<2>(kBlock, 1)),
                    [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                        sycl_attention_indexed_mixed_kernel(
                                it, scores, raw_rows, comp_rows, reduce_scratch, raw_meta,
                                comp_meta, pheads, psinks, pq, praw, pcomp, psorted, n_tokens,
                                pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio,
                                n_head, head_dim);
                    });
        });
        dq.wait_and_throw();
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention indexed mixed launch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

namespace {

/* attention_prefill_raw_softmax_kernel, attention.cuh:222-266: normalises
 * one (token, head) row of the score GEMM's output in place, applying the
 * causal/window mask and the sink logit bias. Consumes the score GEMM's
 * output, so the caller must order this launch after that GEMM's returned
 * event (see the ABI entry below). Reuses sycl_block_row_reduce for the
 * block-wide max/sum, the same substitution used by every other reduction
 * in this file. */
static void sycl_attention_prefill_raw_softmax_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;

    float local_max = sinks[h];
    for (uint32_t k = tid; k < n_keys; k += block) {
        const bool valid = k <= t && (window == 0u || t - k < window);
        const float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = sycl::fmax(local_max, s);
    }
    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t k = tid; k < n_keys; k += block) {
        const float p = sycl::isfinite(row[k]) ? sycl::exp(row[k] - max_s) : 0.0f;
        row[k] = p;
        local_sum += p;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    for (uint32_t k = tid; k < n_keys; k += block) row[k] /= denom;
}

/* attention_prefill_unpack_heads_kernel, attention.cuh:402-417: converts
 * the value GEMM's per-head-major (head, token, dim) output layout back to
 * ds4's (token, head, dim) heads layout, one work-item per output
 * element. Consumes the value GEMM's output, so the caller must order this
 * launch after that GEMM's returned event. Submits without waiting, like
 * every other "_launch" helper in this file; the caller decides ordering. */
static void sycl_attention_prefill_unpack_heads_launch(
        sycl::queue &q, float *heads, const float *tmp,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim) {
    const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t d  = (uint32_t)(gid % head_dim);
        const uint64_t qi = gid / head_dim;
        const uint32_t h  = (uint32_t)(qi % n_head);
        const uint32_t t  = (uint32_t)(qi / n_head);
        heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
    });
}

/* attention_prefill_raw_kernel, attention.cuh:86-141: the scalar fallback,
 * one work-group per (token, head), reached whenever head_dim != 512 (the
 * small-window static and GEMM paths below both require head_dim == 512).
 * raw_count can never exceed 256 by the time this runs: the ABI entry's
 * own validation rejects window > 256 outright, and separately refuses
 * window == 0 with n_tokens > 256 before ever reaching this launch, so
 * `scores`'s local accessor is sized to a fixed 256, exactly matching
 * ROCm's own `__shared__ float scores[256]`, rather than dynamically to
 * raw_count. */
static void sycl_attention_prefill_raw_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const uint32_t raw_count = (window != 0u && t + 1u > window) ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = sycl::rsqrt((float)head_dim);

    float local_max = sinks[h];
    for (uint32_t r = tid; r < raw_count; r += block) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = sycl::fmax(local_max, scores[r]);
    }
    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < raw_count; r += block) {
        const float w = sycl::exp(scores[r] - max_s);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += block) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

}  // namespace

/* ds4_gpu_attention_prefill_raw_heads_tensor, attention_launch.cuh:234-332:
 * raw-only prefill for a fresh, zero-prefix, ratio == 0 (no compression)
 * sub-batch. Oracle and call-site tracing: tests/test_sycl_attention.c's
 * header comment for the oracle-selection method this whole file follows;
 * this entry's own oracle is layer_attention_prefix_batch_worker
 * (ds4.c:12843-12903) with comp_counts == NULL, confirmed by tracing both
 * of this entry's real ds4.c call sites (:28575, :29576), both of which
 * pass a zero-prefix, ratio == 0 sub-batch with g->raw_window as window.
 *
 * Three code paths, gated exactly as ROCm's launcher gates them:
 *
 * 1. Small-window static (n_tokens > 1 && head_dim == 512 &&
 *    (window != 0 ? window : n_tokens) <= 768): reuses
 *    sycl_attention_static_mixed_heads8_online_kernel, already ported for
 *    ds4_gpu_attention_prefill_static_mixed_heads_tensor below, with
 *    n_comp = 0 and ratio = 1 exactly as ROCm's own raw-entry branch calls
 *    it (attention_launch.cuh:245-260). Because this entry's own
 *    validation rejects window > 256 outright, window != 0 always
 *    satisfies the <= 768 bound, so this path is taken for every window != 0
 *    call at head_dim == 512; only window == 0 can fall through past it.
 * 2. The GEMM path (head_dim == 512, window == 0, n_tokens above the
 *    static path's bound): oneMKL's gemm_batch, strided, broadcasting
 *    raw_kv across every head via stride_a == 0 (the same KV table serves
 *    every head; only Q differs per head). Two GEMMs and two dependent
 *    kernels, in this exact order:
 *      (a) score GEMM: scores = (1/sqrt(head_dim)) * K^T Q, batched over
 *          n_head.
 *      (b) softmax kernel, ordered after (a) by an explicit
 *          wait_and_throw() on the GEMM's returned event before this
 *          launch submits -- chosen over passing the event as a
 *          dependency because every other multi-stage entry in this
 *          backend already uses a trailing host wait between stages
 *          (spec 6t's "self-waiting helpers" discipline), and this
 *          entry's stages are strictly sequential with no independent work
 *          to overlap in the meantime.
 *      (c) value GEMM: out = V * scores, batched over n_head, again with
 *          stride_a == 0 broadcasting raw_kv.
 *      (d) unpack kernel, ordered after (c) the same way as (b) after (a).
 *    This is the path that makes long-context prefill correct at all: the
 *    scalar fallback below refuses whenever window == 0 && n_tokens > 256,
 *    so before oneMKL there was no path here above 256 combined keys, let
 *    alone 2048.
 * 3. The scalar fallback (every other head_dim): attention_prefill_raw_kernel,
 *    refusing window == 0 && n_tokens > 256 exactly as ROCm does, since
 *    raw_count would otherwise exceed its fixed 256-wide local staging.
 *
 * attn_sinks staging site 5 of 6 (attention_launch.cuh:241, this entry's
 * own sinks read). */
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_tokens,
        uint32_t              window,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (!heads || !q || !raw_kv || !model_map ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, n_tokens, head_dim, sizeof(float)) ||
        window > 256u) {
        return 0;
    }
    if (n_tokens == 0u || n_head == 0u || head_dim == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float       *pheads = (float *)heads->ptr;
        const float *pq     = (const float *)q->ptr;
        const float *praw   = (const float *)raw_kv->ptr;

        const uint32_t window_bound = window != 0u ? window : n_tokens;
        const bool small_window_static =
                n_tokens > 1u && head_dim == 512u && (uint64_t)window_bound <= 768u;

        if (small_window_static) {
            const uint32_t grid_y = (n_head + 7u) / 8u;
            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<sycl::float4, 1> kv_shared(sycl::range<1>(4u * 128u), h);
                sycl::local_accessor<float, 1> scores(sycl::range<1>(8u * 768u), h);
                h.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 256u, grid_y),
                                          sycl::range<2>(256u, 1u)),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                            sycl_attention_static_mixed_heads8_online_kernel(
                                    it, kv_shared, scores, pheads, psinks, pq, praw, praw,
                                    n_tokens, 0u, window, 1u, n_head, head_dim);
                        });
            });
            dq.wait_and_throw();
            return 1;
        }

        if (head_dim == 512u && n_tokens > 1u) {
            const uint32_t n_keys = n_tokens;
            uint64_t score_count = 0, out_count = 0, score_bytes = 0, out_bytes = 0;
            if (!sycl_u64_mul3_checked(n_head, n_tokens, n_keys, &score_count) ||
                !sycl_u64_mul3_checked(n_head, n_tokens, head_dim, &out_count) ||
                !sycl_u64_mul_checked(score_count, sizeof(float), &score_bytes) ||
                !sycl_u64_mul_checked(out_count, sizeof(float), &out_bytes)) {
                return 0;
            }

            float *scores = sycl::malloc_device<float>((size_t)score_count, dq);
            if (!scores) return 0;
            sycl_device_scratch_guard scores_guard(dq, scores);
            float *out_tmp = sycl::malloc_device<float>((size_t)out_count, dq);
            if (!out_tmp) return 0;
            sycl_device_scratch_guard out_guard(dq, out_tmp);

            const float scale = 1.0f / sycl::sqrt((float)head_dim);
            sycl::event ev1 = sycl_gemm_batch_f32(
                    dq, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                    (int64_t)n_keys, (int64_t)n_tokens, (int64_t)head_dim,
                    scale,
                    praw, (int64_t)head_dim, 0,
                    pq, (int64_t)((uint64_t)n_head * head_dim), (int64_t)head_dim,
                    0.0f,
                    scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * n_tokens),
                    (int64_t)n_head);
            ev1.wait_and_throw();

            dq.submit([&](sycl::handler &h) {
                sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(256u), h);
                h.parallel_for(
                        sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 256u, n_head),
                                          sycl::range<2>(256u, 1u)),
                        [=](sycl::nd_item<2> it) {
                            sycl_attention_prefill_raw_softmax_kernel(
                                    it, reduce_scratch, scores, psinks, n_tokens, window,
                                    n_keys);
                        });
            });
            dq.wait_and_throw();

            sycl::event ev2 = sycl_gemm_batch_f32(
                    dq, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                    (int64_t)head_dim, (int64_t)n_tokens, (int64_t)n_keys,
                    1.0f,
                    praw, (int64_t)head_dim, 0,
                    scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * n_tokens),
                    0.0f,
                    out_tmp, (int64_t)head_dim, (int64_t)((uint64_t)head_dim * n_tokens),
                    (int64_t)n_head);
            ev2.wait_and_throw();

            sycl_attention_prefill_unpack_heads_launch(dq, pheads, out_tmp, n_tokens, n_head,
                                                       head_dim);
            dq.wait_and_throw();
            return 1;
        }

        if (window == 0u && n_tokens > 256u) return 0;

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scores(sycl::range<1>(256u), h);
            sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(128u), h);
            h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 128u, n_head),
                                      sycl::range<2>(128u, 1u)),
                    [=](sycl::nd_item<2> it) {
                        sycl_attention_prefill_raw_kernel(
                                it, scores, reduce_scratch, pheads, psinks, pq, praw,
                                n_tokens, window, n_head, head_dim);
                    });
        });
        dq.wait_and_throw();
    } catch (const std::exception &e) {
        /* std::exception, not sycl::exception: oneMKL's own exceptions
         * (oneapi::mkl::invalid_argument and siblings, thrown synchronously
         * by gemm_batch itself for a bad argument, not only from a later
         * wait) derive from oneapi::mkl::exception -> std::exception, a
         * different hierarchy from sycl::exception (which also derives from
         * std::exception, so this one clause still catches both). Caught
         * here, not found by inspection: an earlier version of this catch
         * clause only caught sycl::exception, and an ablation that gave
         * oneMKL a bad lda terminated the whole process instead of this
         * entry returning its failure value. */
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention prefill raw launch failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

namespace {

/* attention_mixed_cublas_tmp_bytes, attention_launch.cuh:611-623: the
 * scratch size one non-tiled mixed-prefill call needs, packed KV plus
 * scores plus the unpack staging buffer. ROCm sums these into one
 * combined allocation with 256-byte-aligned offsets, since its
 * cuda_tmp_alloc is an arena allocator; this port uses three separate
 * malloc_device allocations instead (no manual offset arithmetic needed,
 * matching every other multi-buffer entry in this file), so this helper
 * returns the unpadded sum, which differs from ROCm's own padded value by
 * at most a few hundred bytes -- utterly negligible against the 4 GiB
 * threshold it is compared to below. Overflow-safe throughout. */
static uint64_t sycl_attention_mixed_gemm_tmp_bytes(uint32_t n_keys, uint32_t n_tokens,
                                                    uint32_t n_head, uint32_t head_dim) {
    uint64_t kv_bytes = 0, score_count = 0, score_bytes = 0, out_count = 0, out_bytes = 0;
    uint64_t total = 0;
    if (!sycl_u64_mul3_checked(n_keys, head_dim, sizeof(float), &kv_bytes) ||
        !sycl_u64_mul3_checked(n_head, n_tokens, n_keys, &score_count) ||
        !sycl_u64_mul_checked(score_count, sizeof(float), &score_bytes) ||
        !sycl_u64_mul3_checked(n_head, n_tokens, head_dim, &out_count) ||
        !sycl_u64_mul_checked(out_count, sizeof(float), &out_bytes) ||
        !sycl_u64_add_checked(kv_bytes, score_bytes, &total) ||
        !sycl_u64_add_checked(total, out_bytes, &total)) {
        return UINT64_MAX;
    }
    return total;
}

/* attention_prefill_pack_mixed_kv_kernel, attention.cuh:386-401: packs the
 * raw and compressed KV tables into one contiguous (n_tokens + n_comp,
 * head_dim) buffer so both GEMMs below can address the whole key range
 * through a single pointer with a single leading dimension. One work-item
 * per output element. */
static void sycl_attention_prefill_pack_mixed_kv_launch(
        sycl::queue &q, float *dst, const float *raw_kv, const float *comp_kv,
        uint32_t n_tokens, uint32_t n_comp, uint32_t head_dim) {
    const uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
        const uint32_t d = (uint32_t)(gid % head_dim);
        const uint64_t r = gid / head_dim;
        dst[gid] = r < n_tokens ? raw_kv[r * head_dim + d]
                                : comp_kv[(r - n_tokens) * head_dim + d];
    });
}

/* attention_prefill_mixed_softmax_kernel, attention.cuh:267-324: normalises
 * one (token, head) row of the non-tiled score GEMM's output in place.
 * use_comp_mask is always false for every entry reachable in this port
 * (the one masked entry, ds4_gpu_attention_prefill_masked_mixed_heads_tensor,
 * is dead: see this header's top-of-file scope note), so that dimension is
 * dropped entirely rather than ported with a permanently-false flag
 * threaded through every call. */
static void sycl_attention_prefill_mixed_softmax_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens || ratio == 0u) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    const uint32_t visible_comp = (t + 1u) / ratio;

    float local_max = sinks[h];
    for (uint32_t k = tid; k < n_keys; k += block) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0u || t - k < window)) s = row[k];
        } else {
            const uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) s = row[k];
        }
        row[k] = s;
        local_max = sycl::fmax(local_max, s);
    }
    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t k = tid; k < n_keys; k += block) {
        const float p = sycl::isfinite(row[k]) ? sycl::exp(row[k] - max_s) : 0.0f;
        row[k] = p;
        local_sum += p;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    for (uint32_t k = tid; k < n_keys; k += block) row[k] /= denom;
}

/* attention_prefill_mixed_softmax_tile_kernel, attention.cuh:325-385: the
 * tiled sibling of the kernel above, masking against the GLOBAL token
 * position (tile_start + t) rather than the tile-local one, since the
 * causal and ratio bounds are defined over the whole sequence, not the
 * tile. */
static void sycl_attention_prefill_mixed_softmax_tile_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *scores,
        const float *sinks,
        uint32_t raw_tokens,
        uint32_t tile_start,
        uint32_t tile_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= tile_tokens || ratio == 0u) return;
    const uint32_t global_t = tile_start + t;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    float *row = scores + ((uint64_t)h * tile_tokens + t) * n_keys;
    const uint32_t visible_comp = (global_t + 1u) / ratio;

    float local_max = sinks[h];
    for (uint32_t k = tid; k < n_keys; k += block) {
        float s = -INFINITY;
        if (k < raw_tokens) {
            if (k <= global_t && (window == 0u || global_t - k < window)) s = row[k];
        } else {
            const uint32_t c = k - raw_tokens;
            if (c < n_comp && c < visible_comp) s = row[k];
        }
        row[k] = s;
        local_max = sycl::fmax(local_max, s);
    }
    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t k = tid; k < n_keys; k += block) {
        const float p = sycl::isfinite(row[k]) ? sycl::exp(row[k] - max_s) : 0.0f;
        row[k] = p;
        local_sum += p;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    for (uint32_t k = tid; k < n_keys; k += block) row[k] /= denom;
}

/* attention_prefill_mixed_kernel, attention.cuh:142-221: the scalar
 * fallback, one work-group per (token, head). Reached whenever head_dim !=
 * 512, refusing outright above DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP
 * (2048) combined raw+visible-compressed keys, matching ROCm's own
 * `__shared__ float scores[2048]` fixed staging, so `scores`'s local
 * accessor here is sized to that same fixed 2048 rather than dynamically. */
static void sycl_attention_prefill_mixed_kernel(
        sycl::nd_item<2> it,
        sycl::local_accessor<float, 1> scores,
        sycl::local_accessor<float, 1> reduce_scratch,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = (uint32_t)it.get_group(0);
    const uint32_t h = (uint32_t)it.get_group(1);
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = (uint32_t)it.get_local_id(0);
    const uint32_t block = (uint32_t)it.get_local_range(0);
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const uint32_t raw_start = (window != 0u && t + 1u > window) ? t + 1u - window : 0u;
    const uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float scale = sycl::rsqrt((float)head_dim);
    const uint32_t n_score = raw_count + visible_comp;

    float local_max = sinks[h];
    for (uint32_t r = tid; r < raw_count; r += block) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = sycl::fmax(local_max, scores[r]);
    }
    for (uint32_t c = tid; c < visible_comp; c += block) {
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[raw_count + c] = dot * scale;
        local_max = sycl::fmax(local_max, scores[raw_count + c]);
    }
    const float max_s = sycl_block_row_reduce(
            it, reduce_scratch, local_max,
            [](float a, float b) { return sycl::fmax(a, b); });

    float local_sum = 0.0f;
    for (uint32_t i = tid; i < n_score; i += block) {
        const float w = sycl::exp(scores[i] - max_s);
        scores[i] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += sycl::exp(sinks[h] - max_s);
    const float denom = sycl_block_row_reduce(
            it, reduce_scratch, local_sum,
            [](float a, float b) { return a + b; });

    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += block) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        }
        oh[d] = acc / denom;
    }
}

/* attention_prefill_mixed_cublas_tiled, attention_launch.cuh:625-736: the
 * tiled GEMM path, taken when the non-tiled path's scratch would exceed
 * the 4 GiB cap sycl_attention_mixed_gemm_tmp_bytes measures against, or
 * when that scratch's allocation genuinely fails. The KV table is packed
 * ONCE, since it does not change per tile; only the per-tile score and
 * unpack staging are re-used across iterations, sized once for the
 * (possibly already-halved) tile_tokens rather than reallocated per tile.
 * Ordering within one tile matches the non-tiled path exactly (score GEMM,
 * wait, softmax, wait, value GEMM, wait, unpack, wait); across tiles there
 * is no ordering hazard to manage since each tile's GEMMs and kernels only
 * ever touch that tile's own slice of `heads`, never a later tile's.
 *
 * `forced_tile_tokens`, not present in ROCm: every production call passes
 * 0, meaning "compute tile_tokens from the 4 GiB cap exactly as ROCm
 * does". Forcing a genuine multi-gigabyte scratch shape on this 16 GiB
 * development card to reach that computed path is impractical (the CPU
 * oracle's O(n_tokens * n_keys * head_dim) cost alone makes a
 * multi-gigabyte scores buffer intractable to verify against), so a test
 * may pass a nonzero value to force a specific tile size directly at a
 * small, fast shape, exercising the tiling LOOP and the tile-boundary
 * softmax masking (attention_prefill_mixed_softmax_tile_kernel's
 * tile_start arithmetic) for real, while the 4 GiB size threshold itself
 * is verified by inspection only, per spec 6g's precedent for a property
 * this hardware cannot exercise directly. */
static int sycl_attention_prefill_mixed_cublas_tiled(
        sycl::queue &dq,
        float *pheads,
        const float *psinks,
        const float *pq,
        const float *praw,
        const float *pcomp,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t forced_tile_tokens = 0u) {
    const uint32_t n_keys = n_tokens + n_comp;
    uint32_t tile_tokens = n_tokens;
    if (forced_tile_tokens != 0u) {
        tile_tokens = forced_tile_tokens < n_tokens ? forced_tile_tokens : n_tokens;
    } else {
        constexpr uint64_t kTileCap = 4ull * 1024ull * 1024ull * 1024ull;
        while (tile_tokens > 1u &&
               sycl_attention_mixed_gemm_tmp_bytes(n_keys, tile_tokens, n_head, head_dim) >
                       kTileCap) {
            tile_tokens = (tile_tokens + 1u) >> 1u;
        }
    }

    uint64_t kv_count = 0, score_count = 0, out_count = 0;
    if (!sycl_u64_mul_checked(n_keys, head_dim, &kv_count) ||
        !sycl_u64_mul3_checked(n_head, tile_tokens, n_keys, &score_count) ||
        !sycl_u64_mul3_checked(n_head, tile_tokens, head_dim, &out_count)) {
        return 0;
    }

    float *kv = sycl::malloc_device<float>((size_t)kv_count, dq);
    if (!kv) return 0;
    sycl_device_scratch_guard kv_guard(dq, kv);
    float *scores = sycl::malloc_device<float>((size_t)score_count, dq);
    if (!scores) return 0;
    sycl_device_scratch_guard scores_guard(dq, scores);
    float *out_tmp = sycl::malloc_device<float>((size_t)out_count, dq);
    if (!out_tmp) return 0;
    sycl_device_scratch_guard out_guard(dq, out_tmp);

    sycl_attention_prefill_pack_mixed_kv_launch(dq, kv, praw, pcomp, n_tokens, n_comp, head_dim);
    dq.wait_and_throw();

    const float scale = 1.0f / sycl::sqrt((float)head_dim);
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += tile_tokens) {
        const uint32_t nt = (t0 + tile_tokens <= n_tokens) ? tile_tokens : (n_tokens - t0);
        const float *q_tile = pq + (uint64_t)t0 * n_head * head_dim;

        sycl::event ev1 = sycl_gemm_batch_f32(
                dq, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)n_keys, (int64_t)nt, (int64_t)head_dim,
                scale,
                kv, (int64_t)head_dim, 0,
                q_tile, (int64_t)((uint64_t)n_head * head_dim), (int64_t)head_dim,
                0.0f,
                scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * nt),
                (int64_t)n_head);
        ev1.wait_and_throw();

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(256u), h);
            h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)nt * 256u, n_head),
                                      sycl::range<2>(256u, 1u)),
                    [=](sycl::nd_item<2> it) {
                        sycl_attention_prefill_mixed_softmax_tile_kernel(
                                it, reduce_scratch, scores, psinks, n_tokens, t0, nt, n_comp,
                                window, ratio, n_keys);
                    });
        });
        dq.wait_and_throw();

        sycl::event ev2 = sycl_gemm_batch_f32(
                dq, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                (int64_t)head_dim, (int64_t)nt, (int64_t)n_keys,
                1.0f,
                kv, (int64_t)head_dim, 0,
                scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * nt),
                0.0f,
                out_tmp, (int64_t)head_dim, (int64_t)((uint64_t)head_dim * nt),
                (int64_t)n_head);
        ev2.wait_and_throw();

        sycl_attention_prefill_unpack_heads_launch(
                dq, pheads + (uint64_t)t0 * n_head * head_dim, out_tmp, nt, n_head, head_dim);
        dq.wait_and_throw();
    }
    return 1;
}

/* attention_prefill_mixed_launch, attention_launch.cuh:738-920: the shared
 * helper both ds4_gpu_attention_prefill_static_mixed_heads_tensor (below)
 * and the dead masked entry delegate to. use_comp_mask/comp_mask are
 * dropped, per this header's top-of-file scope note and the softmax
 * kernels' own comment above: no reachable caller in this port ever sets
 * them. Four paths, gated in ROCm's own order:
 *
 * 1. Small-window static (already ported): reuses
 *    sycl_attention_static_mixed_heads8_online_kernel unchanged.
 * 2. The GEMM path (head_dim == 512, n_tokens > 1): oneMKL gemm_batch,
 *    packed KV, broadcasting the packed table across every head via
 *    stride_a == 0, same four-stage shape as the raw-only entry's GEMM
 *    path (score GEMM, softmax, value GEMM, unpack), each stage ordered
 *    against the next by an explicit wait_and_throw() on the GEMM's
 *    returned event, for the same reason given there.
 * 3. The tiled variant of 2, taken when the non-tiled scratch would
 *    exceed 4 GiB or when its allocation fails outright. ROCm gates the
 *    size check on g_quality_mode, which is permanently false on this
 *    backend and can never become true (ds4_gpu_set_quality is a no-op
 *    stub here, the same conclusion this header's own top comment already
 *    reaches for every other g_quality_mode-gated branch); rather than
 *    port that check as dead code, this applies the same 4 GiB proactive
 *    check unconditionally. This is a deliberate, documented divergence
 *    from a literal port, not a correctness change: tiled and non-tiled
 *    compute identical results, so the only effect is which of two
 *    always-correct paths a given shape takes, and the allocation-failure
 *    trigger (ROCm's own actual production behaviour when g_quality_mode
 *    is false) still applies regardless as a second, independent trigger.
 * 4. The scalar fallback (head_dim != 512), refusing above 2048 combined
 *    keys exactly as ROCm's own launcher does.
 *
 * attn_sinks staging site 4 of 6 (attention_launch.cuh:765, this entry's
 * own sinks read). */
static int sycl_attention_prefill_mixed_launch(
        sycl::queue &dq,
        float *pheads,
        const float *psinks,
        const float *pq,
        const float *praw,
        const float *pcomp,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t window_bound = window != 0u ? window : n_tokens;
    const bool small_window_static =
            n_tokens > 1u && head_dim == 512u && (uint64_t)window_bound + n_comp <= 768u;

    if (small_window_static) {
        const uint32_t grid_y = (n_head + 7u) / 8u;
        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<sycl::float4, 1> kv_shared(sycl::range<1>(4u * 128u), h);
            sycl::local_accessor<float, 1> scores(sycl::range<1>(8u * 768u), h);
            h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 256u, grid_y),
                                      sycl::range<2>(256u, 1u)),
                    [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                        sycl_attention_static_mixed_heads8_online_kernel(
                                it, kv_shared, scores, pheads, psinks, pq, praw, pcomp,
                                n_tokens, n_comp, window, ratio, n_head, head_dim);
                    });
        });
        dq.wait_and_throw();
        return 1;
    }

    if (head_dim == 512u && n_tokens > 1u) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t tmp_bytes = sycl_attention_mixed_gemm_tmp_bytes(
                n_keys, n_tokens, n_head, head_dim);
        constexpr uint64_t kTileCap = 4ull * 1024ull * 1024ull * 1024ull;
        if (tmp_bytes > kTileCap) {
            return sycl_attention_prefill_mixed_cublas_tiled(
                    dq, pheads, psinks, pq, praw, pcomp, n_tokens, n_comp, window, ratio,
                    n_head, head_dim);
        }

        uint64_t kv_count = 0, score_count = 0, out_count = 0;
        if (!sycl_u64_mul_checked(n_keys, head_dim, &kv_count) ||
            !sycl_u64_mul3_checked(n_head, n_tokens, n_keys, &score_count) ||
            !sycl_u64_mul3_checked(n_head, n_tokens, head_dim, &out_count)) {
            return 0;
        }
        float *kv = sycl::malloc_device<float>((size_t)kv_count, dq);
        float *scores = kv ? sycl::malloc_device<float>((size_t)score_count, dq) : nullptr;
        float *out_tmp =
                scores ? sycl::malloc_device<float>((size_t)out_count, dq) : nullptr;
        if (!kv || !scores || !out_tmp) {
            if (kv) sycl::free(kv, dq);
            if (scores) sycl::free(scores, dq);
            if (out_tmp) sycl::free(out_tmp, dq);
            return sycl_attention_prefill_mixed_cublas_tiled(
                    dq, pheads, psinks, pq, praw, pcomp, n_tokens, n_comp, window, ratio,
                    n_head, head_dim);
        }
        sycl_device_scratch_guard kv_guard(dq, kv);
        sycl_device_scratch_guard scores_guard(dq, scores);
        sycl_device_scratch_guard out_guard(dq, out_tmp);

        sycl_attention_prefill_pack_mixed_kv_launch(dq, kv, praw, pcomp, n_tokens, n_comp,
                                                    head_dim);
        dq.wait_and_throw();

        const float scale = 1.0f / sycl::sqrt((float)head_dim);
        sycl::event ev1 = sycl_gemm_batch_f32(
                dq, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                (int64_t)n_keys, (int64_t)n_tokens, (int64_t)head_dim,
                scale,
                kv, (int64_t)head_dim, 0,
                pq, (int64_t)((uint64_t)n_head * head_dim), (int64_t)head_dim,
                0.0f,
                scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * n_tokens),
                (int64_t)n_head);
        ev1.wait_and_throw();

        dq.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(256u), h);
            h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 256u, n_head),
                                      sycl::range<2>(256u, 1u)),
                    [=](sycl::nd_item<2> it) {
                        sycl_attention_prefill_mixed_softmax_kernel(
                                it, reduce_scratch, scores, psinks, n_tokens, n_comp, window,
                                ratio, n_keys);
                    });
        });
        dq.wait_and_throw();

        sycl::event ev2 = sycl_gemm_batch_f32(
                dq, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                (int64_t)head_dim, (int64_t)n_tokens, (int64_t)n_keys,
                1.0f,
                kv, (int64_t)head_dim, 0,
                scores, (int64_t)n_keys, (int64_t)((uint64_t)n_keys * n_tokens),
                0.0f,
                out_tmp, (int64_t)head_dim, (int64_t)((uint64_t)head_dim * n_tokens),
                (int64_t)n_head);
        ev2.wait_and_throw();

        sycl_attention_prefill_unpack_heads_launch(dq, pheads, out_tmp, n_tokens, n_head,
                                                   head_dim);
        dq.wait_and_throw();
        return 1;
    }

    const uint32_t max_raw = (window != 0u && window < n_tokens) ? window : n_tokens;
    constexpr uint32_t kMixedScoreCap = 2048u;
    if ((uint64_t)max_raw + n_comp > kMixedScoreCap) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "attention mixed scalar fallback unsupported for %llu scores "
                "(cap=%u, tokens=%u, comp=%u, window=%u)\n",
                (unsigned long long)((uint64_t)max_raw + n_comp), kMixedScoreCap, n_tokens,
                n_comp, window);
        return 0;
    }
    dq.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> scores(sycl::range<1>(kMixedScoreCap), h);
        sycl::local_accessor<float, 1> reduce_scratch(sycl::range<1>(256u), h);
        h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>((size_t)n_tokens * 256u, n_head),
                                  sycl::range<2>(256u, 1u)),
                [=](sycl::nd_item<2> it) {
                    sycl_attention_prefill_mixed_kernel(
                            it, scores, reduce_scratch, pheads, psinks, pq, praw, pcomp,
                            n_tokens, n_comp, window, ratio, n_head, head_dim);
                });
    });
    dq.wait_and_throw();
    return 1;
}

}  // namespace

/* ds4_gpu_attention_prefill_static_mixed_heads_tensor, attention_launch.cuh:
 * 922-941: delegates to attention_prefill_mixed_launch with comp_mask =
 * NULL, use_comp_mask = 0, now extended beyond the small-window static
 * shape to the GEMM, tiled and scalar paths sycl_attention_prefill_mixed_launch
 * covers above.
 *
 * attn_sinks staging site 4 of 6 (attention_launch.cuh:765). */
extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (comp_kv_f16) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0u || ratio == 0u ||
        (n_comp != 0u && !comp_kv) ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, n_tokens, head_dim, sizeof(float)) ||
        (n_comp && !sycl_tensor_has_elems2(comp_kv, n_comp, head_dim, sizeof(float)))) {
        return 0;
    }
    if (n_head == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float       *pheads = (float *)heads->ptr;
        const float *pq     = (const float *)q->ptr;
        const float *praw   = (const float *)raw_kv->ptr;
        const float *pcomp  = n_comp ? (const float *)comp_kv->ptr : praw;

        return sycl_attention_prefill_mixed_launch(dq, pheads, psinks, pq, praw, pcomp,
                                                   n_tokens, n_comp, window, ratio, n_head,
                                                   head_dim);
    } catch (const std::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention prefill mixed launch failed: %s\n",
                e.what());
        return 0;
    }
}

/* Test-only hook, not part of the ABI: calls
 * sycl_attention_prefill_mixed_cublas_tiled directly with a caller-chosen
 * tile size, bypassing the small-window-static/GEMM/tiled dispatch in
 * ds4_gpu_attention_prefill_static_mixed_heads_tensor above entirely. See
 * that helper's own comment on forced_tile_tokens for why: reaching the
 * tiled path through the real 4 GiB size threshold needs a scratch shape
 * whose CPU oracle cost is intractable on this development machine, so
 * this hook lets a test exercise the tiling loop and the tile-boundary
 * softmax masking at a small, fast shape instead, forcing tile_tokens
 * directly. Same validation, staging and argument shape as the real entry;
 * only the dispatch is different. In the same spirit as
 * ds4_sycl_test_gemm_batch_smoke (sycl/ds4_sycl_matmul.hpp). */
extern "C" int ds4_sycl_test_attention_prefill_mixed_tiled_forced(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              forced_tile_tokens) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0u || ratio == 0u ||
        forced_tile_tokens == 0u || (n_comp != 0u && !comp_kv) ||
        !sycl_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float)) ||
        !sycl_tensor_has_elems3(heads, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(raw_kv, n_tokens, head_dim, sizeof(float)) ||
        (n_comp && !sycl_tensor_has_elems2(comp_kv, n_comp, head_dim, sizeof(float)))) {
        return 0;
    }
    if (n_head == 0u) return 1;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &dq = ds4_sycl_queue(heads->device_id);
        const char *sinks_host = sycl_model_range_ptr(
                model_map, sinks_offset, (uint64_t)n_head * sizeof(float), model_size,
                "attn_sinks");
        if (!sinks_host) return 0;
        sycl_device_scratch_guard sinks_guard = sycl_stage_host_bytes(
                dq, sinks_host, (uint64_t)n_head * sizeof(float));
        if (!sinks_guard.p) return 0;
        const float *psinks = (const float *)sinks_guard.p;

        float       *pheads = (float *)heads->ptr;
        const float *pq     = (const float *)q->ptr;
        const float *praw   = (const float *)raw_kv->ptr;
        const float *pcomp  = n_comp ? (const float *)comp_kv->ptr : praw;

        return sycl_attention_prefill_mixed_cublas_tiled(
                dq, pheads, psinks, pq, praw, pcomp, n_tokens, n_comp, window, ratio, n_head,
                head_dim, forced_tile_tokens);
    } catch (const std::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention prefill mixed tiled (forced) failed: %s\n",
                e.what());
        return 0;
    }
}
