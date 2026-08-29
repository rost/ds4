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
static void sycl_quantize_q8_0_rows_launch(sycl::queue &q, int8_t *xq,
                                           float *xscale, const float *x,
                                           uint32_t in_dim, uint32_t blocks,
                                           uint64_t n_rows) {
    if (n_rows == 0 || blocks == 0) return;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)(n_rows * blocks * 32u)),
                                  sycl::range<1>(32u)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    sycl_quantize_q8_0_rows_kernel(it, xq, xscale, x, in_dim, blocks);
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
 * threadIdx.x & 31`, `warp_sum_f32`). This port deliberately uses an
 * 8-lane sub-group instead: correctness is identical either way (each lane
 * simply owns more or fewer blocks before the final sum; Flash's real
 * shape has blocks=128, divisible by both 8 and 32 with no remainder), and
 * using 8 here alongside the 32-wide quantiser above exercises both
 * sub-group widths the tech stack calls for. It also gives the
 * required ablation ("an 8-wide reduction done at 16 or 32 lanes, built as
 * a genuine sub-group spanning two logical groups") a real substrate: none
 * of the nine named kernels has a genuine 8-wide hardware reduction (their
 * "8" in "warp8" counts 8 independent 32-wide warps per thread block, not
 * an 8-wide reduction), so this port supplies one rather than fabricate the
 * ablation against a width that was never actually 8-wide anywhere in the
 * ROCm source. */
static constexpr uint32_t kGroupedQ8ASubgroupWidth = 8u;

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
        const uint16_t sraw = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        const float wscale = (float)sycl::bit_cast<sycl::half>(sraw);
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const int8_t *xb = xqr + b * 32u;
        int dot = 0;
        for (uint32_t i = 0; i < bn; i++) dot += (int)qs[i] * (int)xb[i];
        acc += wscale * xsr[b] * (float)dot;
    }
    const sycl::sub_group sg = it.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0u) low[tok * low_dim + row] = acc;
}

/* n_tokens * n_groups * rank independent output rows, one 8-lane sub-group
 * per row. `low_dim` is passed precomputed (n_groups * rank) since every
 * caller below has already overflow-checked it. */
static void sycl_grouped_q8_0_a_preq_launch(sycl::queue &q, float *low,
                                            const unsigned char *w,
                                            const int8_t *xq, const float *xscale,
                                            uint32_t group_dim, uint64_t rank,
                                            uint32_t n_groups, uint64_t blocks,
                                            uint64_t low_dim, uint64_t n_tokens) {
    if (low_dim == 0 || n_tokens == 0) return;
    q.submit([&](sycl::handler &h) {
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

    sycl_quantize_q8_0_rows_launch(q, xq, xscale, heads_ptr, group_dim,
                                   (uint32_t)blocks_a, x_rows);
    sycl_grouped_q8_0_a_preq_launch(q, low_ptr, w_ptr, xq, xscale, group_dim,
                                    rank, n_groups, blocks_a, low_dim, n_tokens);
    q.wait_and_throw();
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
        q.wait_and_throw();
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
        q.wait_and_throw();
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
