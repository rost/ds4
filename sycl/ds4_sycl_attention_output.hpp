#pragma once

/* Attention output projection: low-rank Q8_0 grouped A-stage into `low`,
 * then a plain Q8_0 matvec B-stage into the embedding dimension. Ported
 * from three entries in rocm/ds4_rocm_attention_launch.cuh (964-1499) plus
 * the grouped_q8_0_a_* kernel family they depend on in rocm/ds4_rocm_q8.cuh
 * (1345-1831), which nothing else in this backend uses: the dense
 * matmul is a completely different kernel set in matmul.cuh proper.
 *
 * The oracle is ds4.c:10482-10498 (layer_grouped_out_one, single token) and
 * ds4.c:10517-10535 (layer_grouped_out_batch): both quantise the activation
 * to Q8_0 (quantize_q8_0_activation / _batch) before the grouped dot
 * product, then call the plain Q8_0 matvec/matmul for the B-stage. This is
 * why the A-stage kernel ported here is the PREQUANTISED family
 * (grouped_q8_0_a_preq_warp8_kernel), not the raw-float family the batch
 * entry's own non-cublas ROCm branch happens to reach at Flash's shape --
 * see the reachability note above ds4_gpu_attention_output_low_q8_tensor
 * below for the full trace.
 *
 * Kernel family reachability (nine grouped_q8_0_a_* kernels plus
 * quantize_q8_0_f32_kernel, from rocm/ds4_rocm_q8.cuh):
 *
 *  PORTED:
 *   - quantize_q8_0_f32_kernel (:81-112) -> sycl_quantize_q8_0_rows_kernel.
 *     NOT the same port as the E4M3 FP8 KV quantiser
 *     (sycl_e4m3fn_dequant in ds4_sycl_fp8_kv.hpp): that quantises to a
 *     7-bit E4M3-style float code for the KV cache, a completely different
 *     numeric format from this Q8_0 int8-with-per-block-F16-scale
 *     encoding. An earlier note asserting the FP8 KV quantiser had already
 *     ported quantize_q8_0_f32_kernel does not hold up against the source:
 *     grep confirms no existing sycl header defines it before this file.
 *   - grouped_q8_0_a_preq_warp8_kernel (:1794-1831) ->
 *     sycl_grouped_q8_0_a_preq_kernel. Used, with n_tokens generalised to 1
 *     or N, for the A-stage of all three entries below.
 *
 *  TRACED UNREACHABLE, given this backend's fixed defaults (g_quality_mode
 *  is always false here, matching the precedent already documented in
 *  ds4_sycl_indexer.hpp and ds4_sycl_moe*.hpp; there is no oneMKL/cuBLAS
 *  equivalent, so g_cublas_ready is always false too):
 *   - grouped_q8_0_a_f32_warp8_kernel (:1345) and
 *     grouped_q8_0_a_f32_sharedx_rows_w32_2row_kernel (:1375): only reached
 *     from ds4_gpu_attention_output_low_q8_tensor's
 *     `!cuda_q8_prequant_decode_enabled()` branch. That function returns
 *     `!g_glm_model && cfg->q8_prequant_decode`, and
 *     `q8_prequant_decode = !g_quality_mode && (env == NULL || ...)`
 *     (rocm/ds4_rocm_runtime.cuh:4782-4785). For Flash (not GLM) with
 *     g_quality_mode fixed false and no env override, this is always true,
 *     so the branch containing these two kernels never runs.
 *   - grouped_q8_0_a_partial16_w32_kernel (:1428) and
 *     q8_partial_sum8_kernel (:1467): gated by
 *     `!cfg->disable_splitk_attn_out_low`, and
 *     `disable_splitk_attn_out_low = !g_quality_mode`
 *     (rocm/ds4_rocm_runtime.cuh:4786): true whenever quality mode is off,
 *     which disables this branch. With g_quality_mode fixed false, this
 *     branch never runs regardless of the shape match it also requires.
 *   - grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel (:1577):
 *     its own header comment (:1571-1575) says it exists for GLM's strided
 *     QK-low projection, and its only call sites are in ds4_rocm_glm.cuh.
 *     GLM is out of scope for this backend (SYCL.md: "DeepSeek V4 Flash
 *     only. No PRO variant, no GLM").
 *   - grouped_q8_0_a_f32_batch_wmma_onthefly_kernel (:1658-1735): grep
 *     across every ROCm source file for its name finds only this
 *     definition, no `<<<...>>>` launch site anywhere. It is dead code in
 *     ROCm itself, on every platform including HIP (the
 *     `#if defined(__HIP_PLATFORM_AMD__)` guard has no exclusion logic to
 *     replicate: it wraps a kernel nothing calls, not an
 *     always-vs-sometimes dispatch choice). Skipping it is not a
 *     performance divergence, since ROCm itself never runs it either.
 *   - grouped_q8_0_a_f32_batch_warp8_kernel (:1476) and
 *     grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel (:1509): these
 *     ARE reached by ROCm's non-cublas branch of
 *     ds4_gpu_attention_output_q8_batch_tensor (the `if (!attn_output_cublas)`
 *     block, taken on ROCm itself whenever SSD streaming is active or
 *     cublas is otherwise unavailable) -- this is the one pair in the
 *     family that is genuinely live on ROCm under real conditions, not
 *     merely gated off by a fixed backend default. This port declines them
 *     anyway, and that is a deliberate design substitution, not a
 *     data-flow dead-code finding: they compute the SAME grouped Q8_0
 *     projection against a raw (non-activation-quantised) float32
 *     activation, which is a different, less accurate approximation of the
 *     projection than the CPU oracle's activation-quantised algorithm and
 *     than grouped_q8_0_a_preq_warp8_kernel which this port already needs
 *     for the low-rank entry. Reusing one kernel for the A-stage of all
 *     three entries (rather than porting a second, less accurate,
 *     shape-tiled kernel purely to shave one activation-quantisation pass
 *     off the batch path) follows the precedent already set by
 *     ds4_sycl_matmul.hpp's sycl_q8_0_matmul_general, whose own comment
 *     declines ROCm's shape-tiled Q8_0 matmul variants as "vendor
 *     performance tuning with no correctness difference from the general
 *     path". At Flash's fixed shape (group_dim=4096, 32-aligned) ROCm
 *     itself always takes the sharedx_chunked kernel over the plain
 *     warp8 one, so between this pair specifically, only the tiled one is
 *     ever live -- moot here since this port uses neither.
 *
 * Staging (spec 6l): out_a (attn_output_a, the grouped Q8_0 A weight) is
 * staged by this file directly, once per entry (three call sites: the
 * low-rank entry, the F32 batch entry, the F16 batch entry -- each
 * validates and stages its own out_a range since each is an independent
 * ABI entry point). out_b (attn_output_b, the plain Q8_0 B weight) is
 * staged inside the delegated ds4_gpu_matmul_q8_0_tensor call (already
 * landed in ds4_sycl_matmul.hpp), reached by the two batch
 * entries only -- the low-rank entry has no B-stage.
 */

#include "ds4_sycl_common.hpp"

namespace {

/* Ported from quantize_q8_0_f32_kernel, rocm/ds4_rocm_q8.cuh:81-112.
 * One work-group of 32 lanes per (row, block) pair, row meaning any
 * logically-independent activation vector of length in_dim (a flattened
 * (token, group) index for the A-stage callers below): computes the
 * per-block int8 quantisation ROCm's kernel computes, via a genuine 32-wide
 * sub-group amax reduction (sycl::reduce_over_group already returns the
 * combined value to every lane, so no separate broadcast step is needed
 * the way ROCm's manual __shfl broadcast requires). The width is 32, not a
 * tuning choice: a Q8_0 block is exactly 32 elements, one per lane, the
 * same width ROCm's own kernel uses.
 *
 * Every lane participates in the reduction even when idle (lane >= bn),
 * passing the identity value for max (0.0f, safe here specifically because
 * every real contribution is a fabs() and therefore already >= 0.0f,
 * matching ROCm's own choice of 0.0f rather than -INFINITY for the same
 * reason): see design-spec section 6b before removing this on the
 * assumption it is redundant. */
static void sycl_quantize_q8_0_rows_kernel(sycl::nd_item<1> it, int8_t *xq,
                                           float *xscale, const float *x,
                                           uint32_t in_dim, uint32_t blocks) {
    const uint32_t lane = (uint32_t)it.get_local_id(0);
    const uint64_t rb = (uint64_t)it.get_group(0);
    const uint32_t b = (uint32_t)(rb % blocks);
    const uint64_t row = rb / blocks;
    const uint32_t i0 = b * 32u;
    const uint32_t bn = (in_dim - i0 < 32u) ? (in_dim - i0) : 32u;
    const float *xr = x + row * in_dim + i0;

    const float lane_val = (lane < bn) ? sycl::fabs(xr[lane]) : 0.0f;
    const sycl::sub_group sg = it.get_sub_group();
    const float amax = sycl::reduce_over_group(sg, lane_val, sycl::maximum<float>());
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) xscale[row * blocks + b] = d;

    int8_t *dst = xq + (row * blocks + b) * 32u;
    if (lane < bn) {
        int v = (int)sycl::rint(xr[lane] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[lane] = (int8_t)v;
    } else {
        dst[lane] = 0;
    }
}

/* n_rows independent activation vectors of length in_dim each, quantised
 * block-by-block: each work-group covers exactly one (row, block) pair,
 * 32 work-items wide, matching the kernel above. */
static sycl::event sycl_quantize_q8_0_rows_launch(sycl::queue &q, int8_t *xq,
                                           float *xscale, const float *x,
                                           uint32_t in_dim, uint32_t blocks,
                                           uint64_t n_rows) {
    if (n_rows == 0 || blocks == 0) return sycl::event();
    return q.submit([&](sycl::handler &h) {
        h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)(n_rows * blocks * 32u)),
                                  sycl::range<1>(32u)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    sycl_quantize_q8_0_rows_kernel(it, xq, xscale, x, in_dim, blocks);
                });
    });
}

/* Expert-parallel decode's session-batch attention-output split
 * (metal_graph_encode_attn_post_session_batch, ds4.c:65901): as
 * sycl_quantize_q8_0_rows_kernel above, but `heads` holds ALL
 * n_groups_total groups per row and only [group0, group0+group_cnt) is
 * quantised. That range is contiguous within one row (group indices are
 * consecutive) but NOT contiguous across rows when group_cnt <
 * n_groups_total, unlike the single-row TP entry
 * (ds4_gpu_attention_output_q8_tp_tensor) whose n_tokens==1 heads_slice
 * trick works only because a single row has no "across rows" case. Ported
 * from ds4_cuda.cu's quantize_q8_0_group_slice_rows_kernel (referenced at
 * :18420, not separately listed in this port's kernel inventory until this
 * entry needed it): `out_row` (the flattened destination index) is compact
 * (tok*group_cnt + local_group), `src_row` (where it reads from) strides
 * by the FULL n_groups_total. */
static void sycl_quantize_q8_0_group_slice_rows_kernel(sycl::nd_item<1> it, int8_t *xq,
                                                        float *xscale, const float *heads,
                                                        uint32_t group_dim, uint32_t blocks,
                                                        uint32_t n_groups_total, uint32_t group0,
                                                        uint32_t group_cnt) {
    const uint32_t lane = (uint32_t)it.get_local_id(0);
    const uint64_t rb = (uint64_t)it.get_group(0);
    const uint32_t b = (uint32_t)(rb % blocks);
    const uint64_t out_row = rb / blocks;
    const uint64_t tok = out_row / group_cnt;
    const uint64_t local_group = out_row - tok * group_cnt;
    const uint64_t src_row = tok * n_groups_total + (group0 + local_group);
    const uint32_t i0 = b * 32u;
    const uint32_t bn = (group_dim - i0 < 32u) ? (group_dim - i0) : 32u;
    const float *xr = heads + src_row * group_dim + i0;

    const float lane_val = (lane < bn) ? sycl::fabs(xr[lane]) : 0.0f;
    const sycl::sub_group sg = it.get_sub_group();
    const float amax = sycl::reduce_over_group(sg, lane_val, sycl::maximum<float>());
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) xscale[out_row * blocks + b] = d;

    int8_t *dst = xq + (out_row * blocks + b) * 32u;
    if (lane < bn) {
        int v = (int)sycl::rint(xr[lane] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[lane] = (int8_t)v;
    } else {
        dst[lane] = 0;
    }
}

static void sycl_quantize_q8_0_group_slice_rows_launch(sycl::queue &q, int8_t *xq,
                                                        float *xscale, const float *heads,
                                                        uint32_t group_dim, uint32_t blocks,
                                                        uint32_t n_groups_total, uint32_t group0,
                                                        uint32_t group_cnt, uint64_t n_rows) {
    if (n_rows == 0 || blocks == 0 || group_cnt == 0) return;
    const uint64_t x_rows = n_rows * (uint64_t)group_cnt;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)(x_rows * blocks * 32u)),
                                  sycl::range<1>(32u)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    sycl_quantize_q8_0_group_slice_rows_kernel(it, xq, xscale, heads, group_dim,
                                                               blocks, n_groups_total, group0,
                                                               group_cnt);
                });
    });
}

/* Ported from grouped_q8_0_a_preq_warp8_kernel, rocm/ds4_rocm_q8.cuh:1794-
 * 1831: one row of `low` (a single (token, group, row-in-group) triple) is
 * computed by one sub-group. Each lane owns a strided subset of the
 * group's Q8_0 blocks (lane, lane+WIDTH, lane+2*WIDTH, ...), computes a
 * full int8 dot product over each of its own blocks sequentially, and the
 * sub-group sums the per-lane partial sums at the end
 * (sycl::reduce_over_group with sycl::plus), matching ROCm's own
 * warp_sum_f32 reduction over the same per-lane partials.
 *
 * ROCm's own kernel always uses a 32-lane reduction here (`lane =
 * threadIdx.x & 31`, `warp_sum_f32`). The width is free: correctness is
 * identical either way, each lane simply owns more or fewer blocks before
 * the final sum, and Flash's real shape has blocks=128, divisible by 8, 16
 * and 32 alike with no remainder.
 *
 * 16 is the narrowest width every supported device actually has. This was
 * 8 until B60 silicon showed that Xe2 has no 8-wide sub-group at all (see
 * kRequiredSubGroupWidths, ds4_sycl_common.hpp). Unlike the MoE kernels,
 * which keep 8-lane halves inside a 16-wide sub-group so their
 * accumulation order survives unchanged, this kernel widens genuinely: it
 * is written entirely in terms of the constant below, reduce_over_group
 * included, so the only effect is that each lane owns half as many blocks.
 * That does shift float32 accumulation order, which this kernel's tests
 * already tolerate by construction. */
static constexpr uint32_t kGroupedQ8ASubgroupWidth = 16u;

static void sycl_grouped_q8_0_a_preq_kernel(sycl::nd_item<1> it, float *low,
                                            const unsigned char *w,
                                            const int8_t *xq, const float *xscale,
                                            uint32_t group_dim, uint64_t rank,
                                            uint32_t n_groups, uint64_t blocks,
                                            uint64_t low_dim) {
    const uint32_t lane = (uint32_t)it.get_local_id(0);
    const uint64_t rt = (uint64_t)it.get_group(0);
    const uint64_t row = rt % low_dim;
    const uint64_t tok = rt / low_dim;
    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;

    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
    const uint64_t xrow = tok * n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32u;
    const float *xsr = xscale + xrow * blocks;

    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += kGroupedQ8ASubgroupWidth) {
        const uint64_t i0 = b * 32u;
        const uint32_t bn = (uint32_t)((group_dim - i0 < 32u) ? (group_dim - i0) : 32u);
        const unsigned char *blk = wr + b * 34u;
        /* sycl_q8_0_block_dot_i8 (ds4_sycl_common.hpp) reads the block's
         * 32 codes and the 32 activations as two wide copies instead of
         * 64 single-byte loads.  Bit-identical to the byte loop it
         * replaced -- the int products and their sum are exact and the
         * order is unchanged -- and measured 0.0698 ms against 0.0524 ms
         * per call on an A770 at group_dim 4096, low_dim 2048, one
         * token. */
        acc += sycl_q8_0_block_scale(blk) * xsr[b] *
               (float)sycl_q8_0_block_dot_i8(blk, xqr + b * 32u, bn);
    }
    const sycl::sub_group sg = it.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0u) low[tok * low_dim + row] = acc;
}

/* n_tokens * n_groups * rank independent output rows, one
 * kGroupedQ8ASubgroupWidth-lane sub-group per row. `low_dim` is passed precomputed (n_groups * rank) since every
 * caller below has already overflow-checked it. */
static sycl::event sycl_grouped_q8_0_a_preq_launch(sycl::queue &q, float *low,
                                            const unsigned char *w,
                                            const int8_t *xq, const float *xscale,
                                            uint32_t group_dim, uint64_t rank,
                                            uint32_t n_groups, uint64_t blocks,
                                            uint64_t low_dim, uint64_t n_tokens) {
    if (low_dim == 0 || n_tokens == 0) return sycl::event();
    return q.submit([&](sycl::handler &h) {
        h.parallel_for(
                sycl::nd_range<1>(
                        sycl::range<1>((size_t)(low_dim * n_tokens * kGroupedQ8ASubgroupWidth)),
                        sycl::range<1>(kGroupedQ8ASubgroupWidth)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kGroupedQ8ASubgroupWidth)]] {
                    sycl_grouped_q8_0_a_preq_kernel(it, low, w, xq, xscale, group_dim,
                                                    rank, n_groups, blocks, low_dim);
                });
    });
}

/* Shared A-stage: quantises `heads` (n_tokens * n_groups independent
 * group_dim-wide rows) to Q8_0 into fresh device scratch, then runs the
 * grouped preq dot product into `low_ptr` (already device memory, either
 * caller-provided or scratch -- see the two ABI entries' own comments for
 * which). `w_ptr` must already be staged device memory for the out_a
 * weight range (spec 6l); staging happens once at each entry's own call
 * site so the entry's own error path can report the right context on
 * failure. Returns 0 (nothing launched) on scratch allocation failure,
 * matching this file's other internal helpers' own return convention
 * rather than throwing, so callers can check-and-return 0 like every other
 * allocation in this file instead of adding a new exception path. */
static int sycl_attention_output_a_stage(sycl::queue &q, float *low_ptr,
                                         const unsigned char *w_ptr,
                                         const float *heads_ptr,
                                         uint32_t group_dim, uint64_t rank,
                                         uint32_t n_groups, uint64_t blocks_a,
                                         uint64_t low_dim, uint64_t n_tokens) {
    const uint64_t x_rows = n_tokens * (uint64_t)n_groups;
    int8_t *xq = sycl::malloc_device<int8_t>((size_t)(x_rows * blocks_a * 32u), q);
    sycl_device_scratch_guard xq_guard(q, xq);
    float *xscale = sycl::malloc_device<float>((size_t)(x_rows * blocks_a), q);
    sycl_device_scratch_guard xscale_guard(q, xscale);
    if (!xq || !xscale) return 0;

    sycl::event ev_q = sycl_quantize_q8_0_rows_launch(q, xq, xscale, heads_ptr, group_dim,
                                   (uint32_t)blocks_a, x_rows);
    /* HISTORICAL NOTE (spec 6t): the preq kernel below reads xq/xscale,
     * which the quantize kernel just wrote, as two separate q.submit()
     * calls over raw USM (no buffer/accessor dependency tracking).
     * Previously, this queue was out-of-order, so nothing guaranteed the preq
     * kernel would not start, or even run concurrently, before the
     * quantize kernel's writes landed; an audit found this exact gap and
     * fixed it with a targeted wait here rather than by converting the
     * queue to in_order, since in_order was, at the time, a decision
     * nobody had taken for the backend as a whole.
     *
     * That decision was later taken (ds4_sycl.cpp, ds4_sycl_queue_
     * properties): q is now in_order, so ev_p below cannot start before
     * ev_q completes purely from submission order, with no host wait
     * required to make that true. The targeted wait that used to be here
     * is therefore redundant and removed; ev_p's own wait below (needed
     * regardless, since xq_guard/xscale_guard free their scratch when
     * this function returns) already covers everything submitted to q
     * before it, ev_q included. Keeping xq/xscale alive is that drain's
     * only purpose, so it goes through sycl_scratch_release_wait: a
     * recording batch defers those frees itself and needs no wait at
     * all. */
    ds4_sycl_profile_record_named("attn_output_quantize_q8_0_rows", ev_q);
    sycl::event ev_p = sycl_grouped_q8_0_a_preq_launch(q, low_ptr, w_ptr, xq, xscale, group_dim,
                                    rank, n_groups, blocks_a, low_dim, n_tokens);
    sycl_scratch_release_wait(q, xq_guard, xscale_guard);
    ds4_sycl_profile_record_named("attn_output_grouped_q8_0_a_preq", ev_p);
    return 1;
}

/* As sycl_attention_output_a_stage above, but for the group-sliced,
 * multi-row shape ds4_gpu_attention_output_low_q8_rows_exact_tensor needs
 * (see that entry's own comment for why the single-row TP entry's
 * contiguous-slice trick does not generalise to n_rows > 1). `w_ptr` is
 * already staged for the SLICED weight range only (group_cnt*rank rows,
 * not n_groups_total*rank), so sycl_grouped_q8_0_a_preq_launch is called
 * with n_groups=group_cnt exactly as the single-row entry calls it with
 * n_groups=group_cnt after its own heads_slice/weight-offset shift -- the
 * only difference here is the quantise step, which must know the
 * FULL n_groups_total stride to find each row's slice inside `heads`. */
static int sycl_attention_output_a_stage_rows_exact(
        sycl::queue &q, float *low_ptr, const unsigned char *w_ptr, const float *heads_ptr,
        uint32_t group_dim, uint64_t rank, uint32_t n_groups_total, uint32_t group0,
        uint32_t group_cnt, uint64_t blocks_a, uint64_t low_dim, uint64_t n_rows) {
    const uint64_t x_rows = n_rows * (uint64_t)group_cnt;
    int8_t *xq = sycl::malloc_device<int8_t>((size_t)(x_rows * blocks_a * 32u), q);
    sycl_device_scratch_guard xq_guard(q, xq);
    float *xscale = sycl::malloc_device<float>((size_t)(x_rows * blocks_a), q);
    sycl_device_scratch_guard xscale_guard(q, xscale);
    if (!xq || !xscale) return 0;

    sycl_quantize_q8_0_group_slice_rows_launch(q, xq, xscale, heads_ptr, group_dim,
                                               (uint32_t)blocks_a, n_groups_total, group0,
                                               group_cnt, n_rows);
    /* No wait here, same reasoning as sycl_attention_output_a_
     * stage above -- q is in_order, so the preq launch below cannot start
     * before this quantize kernel completes. */
    sycl_grouped_q8_0_a_preq_launch(q, low_ptr, w_ptr, xq, xscale, group_dim, rank, group_cnt,
                                    blocks_a, low_dim, n_rows);
    sycl_batch_wait(q);
    return 1;
}

}  // namespace

/* Test-only hooks, exercising the kernel family directly through the
 * device-tensor ABI (same pattern as ds4_sycl_test_indexed_topk_sort_512_asc
 * in ds4_sycl_indexer.hpp) before either ABI entry below existed, per
 * explicit instruction to test the kernels this way. Not part of ds4_gpu.h. */
extern "C" int ds4_sycl_test_quantize_q8_0_rows(ds4_gpu_tensor *xq,
                                                ds4_gpu_tensor *xscale,
                                                const ds4_gpu_tensor *x,
                                                uint32_t in_dim, uint32_t n_rows) {
    if (!xq || !xscale || !x || in_dim == 0u || n_rows == 0u) return 0;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    if (!sycl_tensor_has_elems2(xq, n_rows, (uint64_t)blocks * 32u, sizeof(int8_t)) ||
        !sycl_tensor_has_elems2(xscale, n_rows, blocks, sizeof(float)) ||
        !sycl_tensor_has_elems2(x, n_rows, in_dim, sizeof(float))) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(xq->device_id);
        sycl_quantize_q8_0_rows_launch(q, (int8_t *)xq->ptr, (float *)xscale->ptr,
                                       (const float *)x->ptr, in_dim, blocks, n_rows);
        sycl_batch_wait(q);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "test_quantize_q8_0_rows failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_sycl_test_grouped_q8_0_a_preq(ds4_gpu_tensor *low,
                                                 const ds4_gpu_tensor *w,
                                                 const ds4_gpu_tensor *xq,
                                                 const ds4_gpu_tensor *xscale,
                                                 uint32_t group_dim, uint32_t rank,
                                                 uint32_t n_groups, uint32_t n_tokens) {
    if (!low || !w || !xq || !xscale || group_dim == 0u || rank == 0u ||
        n_groups == 0u || n_tokens == 0u) {
        return 0;
    }
    uint64_t low_dim = 0;
    const uint32_t blocks = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(n_groups, rank, &low_dim) ||
        !sycl_tensor_has_elems2(low, n_tokens, low_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(w, (uint64_t)n_groups * rank, blocks, 34u, 1u) ||
        !sycl_tensor_has_elems2(xq, (uint64_t)n_tokens * n_groups,
                                (uint64_t)blocks * 32u, sizeof(int8_t)) ||
        !sycl_tensor_has_elems2(xscale, (uint64_t)n_tokens * n_groups, blocks,
                                sizeof(float))) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(low->device_id);
        sycl_grouped_q8_0_a_preq_launch(q, (float *)low->ptr,
                                        (const unsigned char *)w->ptr,
                                        (const int8_t *)xq->ptr,
                                        (const float *)xscale->ptr, group_dim, rank,
                                        n_groups, blocks, low_dim, n_tokens);
        sycl_batch_wait(q);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "test_grouped_q8_0_a_preq failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* The low-rank stage alone, matching ds4.c:10482-10498's first line
 * (matvec_q8_0_grouped_rows) exactly: no n_tokens parameter, one head
 * vector per group. Ported from
 * ds4_gpu_attention_output_low_q8_tensor, rocm/ds4_rocm_attention_launch.cuh
 * :1377-1499.
 *
 * Return polarity: verified against both the ROCm launcher (plain `return 0`
 * on every rejection, `cuda_ok(...)` -- nonzero on success -- on every
 * live path) and the two ds4.c call sites (ds4.c:23463 and :25866 both
 * read the result through `ok = ... != 0` / `return ...`), confirming
 * NONZERO-success, the majority convention (spec 3a). Zero-sized
 * dimensions are a rejection here (return 0), not a zero-work success:
 * ROCm's own guard is `group_dim == 0 || rank == 0 || n_groups == 0`,
 * unconditional, no early "size zero, nothing to do" success branch. */
extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups == 0u || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    uint64_t low_dim = 0, out_a_bytes = 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(n_groups, rank, &low_dim) ||
        !sycl_u64_mul3_checked((uint64_t)n_groups * rank, blocks_a, 34u, &out_a_bytes) ||
        !sycl_model_range_fits(model_size, out_a_offset, out_a_bytes) ||
        !sycl_tensor_has_elems2(heads, n_groups, group_dim, sizeof(float)) ||
        !sycl_tensor_has_f32(low, low_dim)) {
        return 0;
    }
    const char *out_a_ptr = sycl_model_range_ptr(model_map, out_a_offset, out_a_bytes,
                                                 model_size, "attn_out_a");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(low->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, out_a_bytes);
        if (!w_guard.p) return 0;

        if (!sycl_attention_output_a_stage(q, (float *)low->ptr,
                                          (const unsigned char *)w_guard.p,
                                          (const float *)heads->ptr, (uint32_t)group_dim,
                                          rank, n_groups, blocks_a, low_dim, 1u)) {
            return 0;
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_low_q8 failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* The group-sliced, exact-row-count sibling of
 * ds4_gpu_attention_output_low_q8_tensor above, used by the expert-parallel
 * session-batch attention-output split (metal_graph_encode_attn_post_
 * session_batch, ds4.c:65901-66020): a rank owns groups [group0,
 * group0+group_cnt) of the attention heads across `n_rows` tokens at once,
 * not just one. ds4_gpu_attention_output_q8_tp_tensor already covers the
 * single-row case by composition (a heads_slice pointer offset, legal only
 * because one row's slice is contiguous); this entry is why that
 * composition does not extend to n_rows > 1, and is ported here as a real
 * kernel pair instead (sycl_quantize_q8_0_group_slice_rows_kernel plus the
 * existing grouped preq kernel). Ported from ds4_cuda.cu:18373-18444.
 *
 * Return polarity: NONZERO means success, matching
 * ds4_gpu_attention_output_low_q8_tensor immediately above (same ROCm/CUDA
 * shape: `return 0` on every rejection, `cuda_ok(...)` on every live path)
 * and the CUDA source's own validation, which this entry's checks mirror. */
extern "C" int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows) {
    if (!low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups_total == 0u || group_cnt == 0u || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || n_rows == 0u ||
        group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    uint64_t low_dim = 0, row_a_bytes = 0, out_a_bytes = 0, a_off_delta = 0, a_offset = 0;
    uint64_t heads_row_elems = 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(group_cnt, rank, &low_dim) ||
        !sycl_u64_mul_checked(blocks_a, 34u, &row_a_bytes) ||
        !sycl_u64_mul_checked(low_dim, row_a_bytes, &out_a_bytes) ||
        !sycl_u64_mul_checked(group0, rank, &a_off_delta) ||
        !sycl_u64_mul_checked(a_off_delta, row_a_bytes, &a_off_delta) ||
        !sycl_u64_add_checked(out_a_offset, a_off_delta, &a_offset) ||
        !sycl_model_range_fits(model_size, a_offset, out_a_bytes) ||
        !sycl_u64_mul_checked(n_groups_total, group_dim, &heads_row_elems) ||
        !sycl_tensor_has_elems2(heads, n_rows, heads_row_elems, sizeof(float)) ||
        !sycl_tensor_has_elems2(low, n_rows, low_dim, sizeof(float))) {
        return 0;
    }
    const char *out_a_ptr = sycl_model_range_ptr(model_map, a_offset, out_a_bytes, model_size,
                                                 "attn_out_a_rows");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(low->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, out_a_bytes);
        if (!w_guard.p) return 0;

        if (!sycl_attention_output_a_stage_rows_exact(
                    q, (float *)low->ptr, (const unsigned char *)w_guard.p,
                    (const float *)heads->ptr, (uint32_t)group_dim, rank, n_groups_total,
                    group0, group_cnt, blocks_a, low_dim, n_rows)) {
            return 0;
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_low_q8_rows_exact failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Tensor parallelism: the group-sliced attention output pair for
 * a single TP rank, n_tokens == 1 only (ds4_gpu.h's own docstring on this
 * entry). A rank owns groups [group0, group0+group_cnt) of the attention
 * heads; this computes that rank's partial low-rank projection over just
 * those groups, then the matching k-slice of the expand (B) projection,
 * producing this rank's partial contribution to the full-width attention
 * block output (which the caller sums with the peer's contribution the
 * same way ds4_gpu_matmul_q8_0_kslice_rows_tensor's own callers do).
 *
 * ds4_cuda.cu's own implementation (ds4_cuda.cu:18460-18516) is itself
 * pure composition, not a fused kernel: it builds a heads_slice view
 * (legal as a flat pointer offset only because n_tokens == 1 makes the
 * group0..group0+group_cnt range of a single row contiguous; a
 * multi-row version needs a strided kernel instead, which is exactly why
 * ds4_gpu_attention_output_low_q8_rows_exact_tensor below -- the batched
 * n_rows sibling of this entry, used by the expert-parallel session-batch
 * attention-output split -- is a real kernel pair rather than composition;
 * see its own comment) and calls ds4_gpu_attention_output_low_q8_tensor then
 * ds4_gpu_matmul_q8_0_kslice_rows_tensor. This port is a literal,
 * line-for-line translation of that composition, both callees already
 * implemented in this backend. Validation ported from
 * ds4_cuda.cu:18474-18490. NONZERO means success, matching both composed
 * calls (`&&`, matching CUDA's own `return ... && ...`). */
extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups_total == 0u || group_cnt == 0u || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || out_dim == 0u) {
        return 0;
    }

    uint64_t row_a_bytes = 0, low_dim_total = 0, k_off = 0, k_cnt = 0, heads_min_bytes = 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(blocks_a, 34u, &row_a_bytes) ||
        !sycl_u64_mul_checked((uint64_t)n_groups_total, rank, &low_dim_total) ||
        !sycl_u64_mul_checked((uint64_t)group0, rank, &k_off) ||
        !sycl_u64_mul_checked((uint64_t)group_cnt, rank, &k_cnt) ||
        (k_off % 32u) != 0u || (k_cnt % 32u) != 0u ||
        !sycl_u64_mul_checked((uint64_t)group0 + group_cnt, group_dim, &heads_min_bytes) ||
        !sycl_u64_mul_checked(heads_min_bytes, sizeof(float), &heads_min_bytes) ||
        heads->bytes < heads_min_bytes ||
        !sycl_tensor_has_f32(low, k_cnt) ||
        !sycl_tensor_has_f32(out, out_dim)) {
        return 0;
    }

    ds4_gpu_tensor heads_slice = *heads;
    heads_slice.ptr = (char *)heads->ptr + (uint64_t)group0 * group_dim * sizeof(float);
    heads_slice.bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
    heads_slice.owner = 0;

    uint64_t a_off_delta = 0, a_off = 0;
    if (!sycl_u64_mul_checked((uint64_t)group0, rank, &a_off_delta) ||
        !sycl_u64_mul_checked(a_off_delta, row_a_bytes, &a_off_delta) ||
        !sycl_u64_add_checked(out_a_offset, a_off_delta, &a_off)) {
        return 0;
    }

    return ds4_gpu_attention_output_low_q8_tensor(low, model_map, model_size, a_off,
                                                  group_dim, rank, group_cnt,
                                                  &heads_slice) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(out, model_map, model_size, out_b_offset,
                                                  low_dim_total, out_dim, k_off, k_cnt,
                                                  low, 1u);
}

/* Full two-stage batched projection, F32 output. Matches
 * ds4.c:10517-10535 (layer_grouped_out_batch): the grouped A-stage above,
 * generalised to n_tokens, then a plain Q8_0 matvec/matmul B-stage. Ported
 * from ds4_gpu_attention_output_q8_batch_tensor,
 * rocm/ds4_rocm_attention_launch.cuh:1072-1376. `group_tmp` and `low_tmp`
 * are unused, matching ROCm's own `(void)group_tmp; (void)low_tmp;`: they
 * exist for a scratch shape this port does not need (this port allocates
 * its own A-stage scratch per sycl_attention_output_a_stage, same as every
 * other landed subsystem, rather than relying on caller-provided scratch
 * tensors ROCm itself does not use either).
 *
 * B-stage: delegates to ds4_gpu_matmul_q8_0_tensor (ds4_sycl_matmul.hpp),
 * which already stages attn_output_b's weight range itself and
 * already implements the exact ROCm-kernel-matching Q8_0 dense matmul the
 * oracle's second line (matvec_q8_0 / matmul_q8_0_batch) needs,
 * rather than porting cuda_matmul_q8_0_tensor_labeled's plain-kernel branch
 * a second time.
 *
 * Return polarity: verified against the ROCm launcher (plain `return 0` on
 * every rejection and staging failure; the final return is
 * cuda_matmul_q8_0_tensor_labeled's own result, itself nonzero-success) and
 * against ds4.c:25988 and :32534, both of which read the result as
 * `ok = ... != 0` / `if (ok) ok = ...`. Nonzero-success, matching the
 * low-rank entry above. */
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups == 0u || out_dim == 0u || n_tokens == 0u ||
        group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    uint64_t low_dim = 0, out_a_bytes = 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(n_groups, rank, &low_dim) ||
        !sycl_u64_mul3_checked((uint64_t)n_groups * rank, blocks_a, 34u, &out_a_bytes) ||
        !sycl_model_range_fits(model_size, out_a_offset, out_a_bytes) ||
        !sycl_tensor_has_elems2(heads, (uint64_t)n_tokens * n_groups, group_dim,
                                sizeof(float)) ||
        !sycl_tensor_has_elems2(low, n_tokens, low_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    const char *out_a_ptr = sycl_model_range_ptr(model_map, out_a_offset, out_a_bytes,
                                                 model_size, "attn_out_a");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, out_a_bytes);
        if (!w_guard.p) return 0;

        if (!sycl_attention_output_a_stage(q, (float *)low->ptr,
                                          (const unsigned char *)w_guard.p,
                                          (const float *)heads->ptr, (uint32_t)group_dim,
                                          rank, n_groups, blocks_a, low_dim, n_tokens)) {
            return 0;
        }
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_q8_batch A-stage failed: %s\n",
                e.what());
        return 0;
    }

    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, out_b_offset,
                                      low_dim, out_dim, low, n_tokens);
}

/* Same two-stage projection as above, F16 output (spec 6k). Ported from
 * ds4_gpu_attention_output_q8_batch_f16_tensor,
 * rocm/ds4_rocm_attention_launch.cuh:964-1071 -- but that ROCm function's
 * entire body is a cuBLAS pipeline gated on `!g_cublas_ready` returning 0
 * unconditionally when cuBLAS/rocBLAS is unavailable: it has NO kernel-only
 * fallback path to port literally. This backend has no oneMKL/cuBLAS
 * equivalent (design-spec section on the matmul tier), so a literal port
 * would make this entry permanently fail on every call, which is not the
 * intent here: this entry is meant to be implemented, not stubbed,
 * with its own F16-tie ablations. Instead this entry computes the exact
 * same two-stage projection as the F32 batch entry above (same A-stage
 * kernel, same B-stage delegation to ds4_gpu_matmul_q8_0_tensor into an F32
 * scratch buffer) and converts the F32 result to F16 as the last step,
 * using the ported bit-manipulation encoder (sycl_f32_to_f16_bits_hip_round
 * in ds4_sycl_common.hpp) rather than sycl::half, per spec 6k: sycl::half's
 * device-side conversion rounds exact ties to even, ds4's own converter
 * rounds every exact tie up, and they disagree only at exact ties.
 *
 * Unlike ROCm's cuBLAS pipeline (which ignores its own `low` parameter,
 * `(void)low;`, keeping its own internal packed-F16 scratch instead), this
 * entry writes the real A-stage result into the caller-provided `low`
 * tensor, the same convention as the F32 batch entry: there is no reason
 * to allocate a second scratch buffer for it when the ABI already has a
 * home for it. This means, unlike ROCm's null-check list for this entry
 * (`!out_h || !heads || !model_map`, `low` not required since ROCm never
 * touches it), this port's null/size check DOES require `low`.
 *
 * Return polarity: verified against the ROCm launcher (plain `return 0` on
 * every rejection, `st == CUBLAS_STATUS_SUCCESS` -- nonzero-success -- on
 * the one live path) and ds4.c:29718 (`attn_out_f16 = ... != 0`).
 * Nonzero-success, matching both entries above. */
extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!out_h || !low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups == 0u || out_dim == 0u || n_tokens == 0u ||
        group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    uint64_t low_dim = 0, out_a_bytes = 0, out_elems = 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (!sycl_u64_mul_checked(n_groups, rank, &low_dim) ||
        !sycl_u64_mul_checked(n_tokens, out_dim, &out_elems)) {
        return 0;
    }
    if (!sycl_u64_mul3_checked((uint64_t)n_groups * rank, blocks_a, 34u, &out_a_bytes) ||
        !sycl_model_range_fits(model_size, out_a_offset, out_a_bytes) ||
        !sycl_tensor_has_elems2(heads, (uint64_t)n_tokens * n_groups, group_dim,
                                sizeof(float)) ||
        !sycl_tensor_has_elems2(low, n_tokens, low_dim, sizeof(float)) ||
        !sycl_tensor_has_elems(out_h, out_elems, sizeof(uint16_t))) {
        return 0;
    }
    const char *out_a_ptr = sycl_model_range_ptr(model_map, out_a_offset, out_a_bytes,
                                                 model_size, "attn_out_a");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out_h->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, out_a_bytes);
        if (!w_guard.p) return 0;

        if (!sycl_attention_output_a_stage(q, (float *)low->ptr,
                                          (const unsigned char *)w_guard.p,
                                          (const float *)heads->ptr, (uint32_t)group_dim,
                                          rank, n_groups, blocks_a, low_dim, n_tokens)) {
            return 0;
        }

        float *out_f32 = sycl::malloc_device<float>((size_t)out_elems, q);
        sycl_device_scratch_guard out_f32_guard(q, out_f32);
        if (!out_f32) return 0;
        ds4_gpu_tensor out_f32_view = {out_f32, out_elems * sizeof(float), 0, out_h->device_id};

        if (!ds4_gpu_matmul_q8_0_tensor(&out_f32_view, model_map, model_size, out_b_offset,
                                        low_dim, out_dim, low, n_tokens)) {
            return 0;
        }

        uint16_t *out_bits = (uint16_t *)out_h->ptr;
        sycl::event _ds4_prof_ev16 = q.parallel_for(sycl::range<1>((size_t)out_elems), [=](sycl::id<1> gid) {
             out_bits[gid[0]] = sycl_f32_to_f16_bits_hip_round(out_f32[gid[0]]);
         });
         sycl_batch_wait(_ds4_prof_ev16);
         ds4_sycl_profile_record_named("attn_output_test_quantize_q8_0_rows", _ds4_prof_ev16);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_q8_batch_f16 failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* ---- Q4_K attention output projection -----------------------------------
 *
 * ds4_gpu_attention_output_low_q4_K_slice_tensor and
 * ds4_gpu_attention_output_q4_K_batch_tensor parallel the three Q8_0
 * entries above: same low-rank-then-expand structure (a per-group,
 * per-token A-stage projecting `heads` down to `low`, then a plain dense
 * matmul B-stage expanding `low` up to the embedding width), fired instead
 * of the Q8_0 entries whenever attn_output_a is Q4_K.
 *
 * Both ds4_cuda.cu's own definitions of these two entries are plain
 * unconditional `return 0` stubs (every parameter cast to void), so unlike
 * the plain dense matmul entries in ds4_sycl_matmul.hpp, there is no CUDA
 * reference behaviour here at all, not even a partial one. Only
 * ds4_metal.m implements them for real, through bespoke Metal compute
 * shaders (ds4_gpu_attention_output_q4_K_batch_tensor,
 * ds4_gpu_attention_output_low_q4_K_slice_tensor, both non-portable). The
 * math ported is the same standard Q4_K block decode the dense matmul
 * entries already use (sycl_q4_k_dequant, ds4_sycl_common.hpp): unlike the
 * Q8_0 A-stage above, no activation quantisation step is needed, since
 * ds4_metal.m's own Q4_K kernels (ds4_gpu_matmul_quant_impl_tensor) read
 * the raw F32 activation directly, the same contract the Q4_K dense
 * matmul already follows.
 *
 * Reachability, traced directly against ds4.c rather than assumed: the
 * low-rank entry fires on every decode token and every prefill
 * batch whenever attn_output_a is Q4_K (metal_graph_attention_output_
 * dense_quant_low, ds4.c:25856-25919, dispatches here unconditionally for
 * that type, with NO fallback if this entry fails -- unlike the generic
 * per-group loop the same function falls through to for every OTHER
 * dense-quant type). The batch entry only fires for n_tokens >= 32
 * (metal_graph_attention_output_dense_quant_batch, ds4.c:26009), and ITS
 * failure path (ds4.c:26029-26066) calls back into the very same low-rank
 * entry, once per token with group0=0/group_cnt=n_groups, followed by
 * ds4_gpu_matmul_quant_tensor for the B-stage -- confirming the audit's
 * claim that the batch entry's apparent "fallback" is not a route around a
 * missing implementation, only a slower path through the same low-rank
 * scalar kernel this file also builds. */

/* One work-item per (token, group-local row) output element: row = gid %
 * low_dim decomposes into (group, row_in_group), tok = gid / low_dim.
 * `w` holds only the addressed group_cnt*rank rows (already offset to the
 * right slice by the caller's staging), so `group` here always starts at 0
 * regardless of which slice of the checkpoint's real group range this
 * call addresses -- the caller's own group0 only selects WHICH bytes get
 * staged, never appears in this kernel's own indexing. Mirrors
 * sycl_grouped_q8_0_a_preq_kernel's row/tok/group decomposition exactly,
 * with a direct float dot against sycl_q4_k_dequant in place of that
 * kernel's quantise-then-int8-dot A-stage. */
static sycl::event sycl_grouped_q4_k_a_launch(sycl::queue &q, float *low,
                                       const unsigned char *w, const float *heads,
                                       uint32_t group_dim, uint64_t rank,
                                       uint32_t n_groups, uint64_t row_bytes,
                                       uint64_t low_dim, uint64_t n_tokens) {
    if (low_dim == 0 || n_tokens == 0) return sycl::event();
    return q.parallel_for(sycl::range<1>((size_t)(low_dim * n_tokens)), [=](sycl::id<1> gid) {
        const uint64_t row = gid[0] % low_dim;
        const uint64_t tok = gid[0] / low_dim;
        const uint64_t group = row / rank;
        const uint64_t row_in_group = row - group * rank;

        const unsigned char *wr = w + (group * rank + row_in_group) * row_bytes;
        const float *xr = heads + (tok * n_groups + group) * group_dim;

        float sum = 0.0f;
        for (uint32_t k = 0; k < group_dim; k++) {
            sum += xr[k] * sycl_q4_k_dequant(wr, k);
        }
        low[tok * low_dim + row] = sum;
    });
}

/* The low-rank stage alone, sliced to groups [group0, group0+group_cnt),
 * matching ds4.c's own per-group generic fallback's addressing
 * (out_a->abs_offset + (group0+i)*group_weight_bytes,
 * metal_graph_attention_output_dense_quant_low, ds4.c:25905-25918) rather
 * than the Q8_0 low-rank entry's own unsliced n_groups parameter: unlike
 * Q8_0, this entry is also the TP-sliced call target
 * (metal_graph_attention_output_dense_quant_tp, ds4.c:25921-25974), so its
 * ABI carries group0/group_cnt even though single-GPU Flash always calls it
 * with group0=0, group_cnt=n_groups (ds4.c:23559-23577).
 *
 * Return polarity: verified against ds4_metal.m's real implementation
 * (nonzero-success on every live path, plain `return 0` on every rejection)
 * and against both ds4.c call sites (:23562, :26046, both `ok = ... != 0`
 * or `ok = metal_graph_attention_output_dense_quant_low(...)` itself
 * returning a bool built the same way). Zero-sized dimensions are a
 * rejection, matching the Q8_0 sibling's own treatment of group_dim==0/
 * rank==0/n_groups==0. */
extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        group_cnt == 0u || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    /* out_a_offset is the FULL out_a tensor's own base offset (ds4.c passes
     * out_a->abs_offset unmodified, group0 as a separate argument): this
     * entry must fold group0 into the byte address itself, the same
     * arithmetic ds4.c's own per-group generic fallback does inline
     * (out_a->abs_offset + (group0+i)*group_weight_bytes,
     * metal_graph_attention_output_dense_quant_low). Getting this wrong
     * (e.g. treating out_a_offset as already-sliced) reads the wrong
     * groups' weights while still landing inside the mapped model, so the
     * bounds checks below would not catch it -- only a numeric mismatch
     * against the oracle would. */
    uint64_t low_dim = 0, row_bytes = 0, group_weight_bytes = 0, slice_bytes = 0;
    uint64_t group0_bytes = 0, slice_offset = 0;
    if (!sycl_u64_mul_checked(group_cnt, rank, &low_dim) ||
        !sycl_q4_k_row_bytes_checked(group_dim, &row_bytes) ||
        !sycl_u64_mul_checked(rank, row_bytes, &group_weight_bytes) ||
        !sycl_u64_mul_checked(group_cnt, group_weight_bytes, &slice_bytes) ||
        !sycl_u64_mul_checked((uint64_t)group0, group_weight_bytes, &group0_bytes) ||
        !sycl_u64_add_checked(out_a_offset, group0_bytes, &slice_offset) ||
        !sycl_tensor_has_elems2(heads, group_cnt, group_dim, sizeof(float)) ||
        !sycl_tensor_has_f32(low, low_dim)) {
        return 0;
    }
    if (!sycl_model_range_fits(model_size, slice_offset, slice_bytes)) return 0;
    const char *out_a_ptr = sycl_model_range_ptr(model_map, slice_offset, slice_bytes,
                                                 model_size, "attn_out_a_q4_K_slice");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(low->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, slice_bytes);
        if (!w_guard.p) return 0;

        sycl::event ev = sycl_grouped_q4_k_a_launch(q, (float *)low->ptr, (const unsigned char *)w_guard.p,
                                   (const float *)heads->ptr, (uint32_t)group_dim, rank,
                                   group_cnt, row_bytes, low_dim, 1u);
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named("attn_output_grouped_q4_k_a", ev);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_low_q4_K_slice failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* Full two-stage batched projection, the Q4_K analogue of
 * ds4_gpu_attention_output_q8_batch_tensor above: the grouped Q4_K A-stage
 * over the FULL [0, n_groups) range (unlike the slice entry above, this
 * entry's own ds4.c call site never slices it, ds4.c:26010-26024), then a
 * type-dispatching dense matmul B-stage.
 *
 * out_b_type exists because attn_output_a and attn_output_b are validated
 * (and therefore choosable) independently in weights_validate_layout
 * (ds4.c:5079-5080): a checkpoint can quantise attn_output_a as Q4_K while
 * attn_output_b is Q8_0, Q4_K or Q4_0. The B-stage therefore delegates to
 * the type-dispatching ds4_gpu_matmul_quant_tensor
 * (sycl/ds4_sycl_matmul.hpp) rather than assuming Q8_0 the way the Q8_0
 * batch entry above assumes its own B-stage weight is Q8_0 too.
 * `group_tmp`/`low_tmp` are unused, matching the Q8_0 batch entry's own
 * `(void)` treatment of ROCm's unused scratch-tensor parameters (ROCm does
 * not implement this entry, so there is nothing to mirror there beyond the
 * ABI signature itself).
 *
 * Return polarity: verified against ds4_metal.m's real implementation and
 * against ds4.c's one call site (:26010, `if (... != 0) return true;`). */
extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        n_groups == 0u || out_dim == 0u || n_tokens == 0u ||
        group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    uint64_t low_dim = 0, row_bytes = 0, out_a_bytes = 0, n_rows = 0;
    if (!sycl_u64_mul_checked(n_groups, rank, &low_dim) ||
        !sycl_q4_k_row_bytes_checked(group_dim, &row_bytes) ||
        !sycl_u64_mul_checked(n_groups, rank, &n_rows) ||
        !sycl_u64_mul_checked(n_rows, row_bytes, &out_a_bytes) ||
        !sycl_model_range_fits(model_size, out_a_offset, out_a_bytes) ||
        !sycl_tensor_has_elems2(heads, (uint64_t)n_tokens * n_groups, group_dim,
                                sizeof(float)) ||
        !sycl_tensor_has_elems2(low, n_tokens, low_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    const char *out_a_ptr = sycl_model_range_ptr(model_map, out_a_offset, out_a_bytes,
                                                 model_size, "attn_out_a_q4_K_batch");
    if (!out_a_ptr) return 0;
    if (g_devices.empty()) return 0;

    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        sycl_device_scratch_guard w_guard = sycl_stage_host_bytes(q, out_a_ptr, out_a_bytes);
        if (!w_guard.p) return 0;

        sycl::event ev = sycl_grouped_q4_k_a_launch(q, (float *)low->ptr, (const unsigned char *)w_guard.p,
                                   (const float *)heads->ptr, (uint32_t)group_dim, rank,
                                   n_groups, row_bytes, low_dim, n_tokens);
        sycl_batch_wait(q);
        ds4_sycl_profile_record_named("attn_output_grouped_q4_k_a", ev);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "attention_output_q4_K_batch A-stage failed: %s\n",
                e.what());
        return 0;
    }

    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size, out_b_offset, out_b_type,
                                       low_dim, out_dim, low, n_tokens);
}
