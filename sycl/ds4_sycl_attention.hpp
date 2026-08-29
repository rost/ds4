#pragma once

/* DS4 SYCL attention: decode, batched decode, indexed decode against the
 * indexer's selection, and small-window static prefill. Ported from
 * rocm/ds4_rocm_attention.cuh (kernels) and
 * rocm/ds4_rocm_attention_launch.cuh (entry points).
 *
 * Scope, and why it stops here: attention_launch.cuh has fourteen entries.
 * This header covers the seven that are decode-reachable and testable today
 * against a CPU oracle with no BLAS: ds4_gpu_store_raw_kv_tensor,
 * ds4_gpu_kv_fp8_store_raw_tensor, ds4_gpu_attention_decode_heads_tensor,
 * ds4_gpu_attention_decode_raw_batch_heads_tensor,
 * ds4_gpu_attention_decode_mixed_batch_heads_tensor,
 * ds4_gpu_attention_indexed_mixed_batch_heads_tensor and
 * ds4_gpu_attention_prefill_static_mixed_heads_tensor.
 *
 * Deliberately NOT here:
 *   - The three prefill entries whose real fast path is
 *     cublasSgemmStridedBatched (attention_prefill_mixed_launch,
 *     ds4_gpu_attention_prefill_raw_heads_tensor): oneMKL is not installed,
 *     and the scalar fallback refuses above 2048 combined keys, so there is
 *     no correct path for long-context prefill yet.
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
