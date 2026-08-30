#pragma once

/* Routed-MoE dispatcher: the two ABI entries and the shared launcher they
 * both call.  Ported from rocm/ds4_rocm_moe_launch.cuh, which is
 * ~2,113 lines in `routed_moe_launch` alone.  This header implements the
 * dispatcher skeleton once, with one delimited block per quantised
 * format, and fills in only the q4k_path block; the other three blocks
 * return 0 (a clean failure) with a comment naming what fills each in,
 * per the "dispatcher structure" decision.
 *
 * Return convention for both ABI entries: NONZERO means success, 0 means
 * failure.  Verified against `routed_moe_launch` (moe_launch.cuh:733,
 * `int ok = 1;`, set to 0 on every failure, returned unchanged on
 * success) and the ds4.c call site (`ok = ds4_gpu_routed_moe_batch_tensor(
 * ...) != 0;`, ds4.c:30364).
 *
 * n_tokens == 0 is a FAILURE here, not a free success: routed_moe_build_plan
 * (moe_launch.cuh:479) rejects it explicitly.  This is the opposite of
 * several other entries ported earlier in this project; do not assume the
 * convention carries over.  An unrecognised gate/down type pairing is
 * also a failure (moe_launch.cuh:498-500). */

#include "ds4_sycl_moe.hpp"

#include <cstdlib>
#include <vector>

namespace {

/* Set at the end of every successful sycl_routed_moe_launch call to the
 * number of experts, and the total gate+up+down bytes, actually copied
 * host-to-device for that call -- whichever of the compacted or
 * full-table path ran. Test-only instrumentation, read back through
 * ds4_sycl_moe_test_last_staged_expert_count / _bytes below. The
 * whole point of staging n_expert_used experts instead of n_total_expert
 * is invisible in decode's numeric output -- the same weights end up at
 * the same addresses either way -- so per spec 6w, only a counter, not
 * an oracle comparison, can prove compaction actually happened.
 *
 * Also set by sycl_routed_moe_one_owned_q4k (ds4_sycl_moe_owned.hpp)
 * to the distinct owned-and-selected expert count its own compaction
 * stages, the same counter reused for the same reason: composing
 * ownership with compaction is invisible in the decode output too. */
static uint32_t g_sycl_moe_last_staged_expert_count = 0;
static uint64_t g_sycl_moe_last_staged_bytes = 0;

/* ---- routed_moe_build_plan, moe_launch.cuh:451-509 ------------------
 *
 * gate_type/down_type numeric values are ds4's DS4_TENSOR_* enum:
 * 12 = Q4_K, 16 = IQ2_XXS, 10 = Q2_K, 39 = MXFP4 (verified against the
 * literals routed_moe_build_plan itself compares against). */
struct sycl_routed_moe_plan {
    bool q4k_path      = false;
    bool iq2_path      = false;
    bool iq2_iq2_path  = false;
    bool q2k_path      = false;
    bool mxfp4_path    = false;
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
};

static bool sycl_routed_moe_build_plan(
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes, uint32_t expert_in_dim,
        uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert,
        const ds4_gpu_tensor *x, uint32_t n_tokens, sycl_routed_moe_plan *plan) {
    if (!plan) return false;
    *plan = sycl_routed_moe_plan{};
    if (!out || !gate || !up || !mid || !down || !model_map || !selected ||
        !weights || !x || n_tokens == 0u || n_total_expert == 0u ||
        n_expert == 0u || n_expert > 8u /* DS4_ROCM_N_EXPERT_USED */ ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % kMoeQK != 0u || expert_mid_dim % kMoeQK != 0u ||
        !sycl_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !sycl_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !sycl_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !sycl_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !sycl_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return false;
    }
    plan->q4k_path     = (gate_type == 12u && down_type == 12u);
    plan->iq2_path     = (gate_type == 16u && down_type == 10u);
    plan->iq2_iq2_path = (gate_type == 16u && down_type == 16u);
    plan->q2k_path     = (gate_type == 10u && down_type == 10u);
    plan->mxfp4_path   = (gate_type == 39u && down_type == 39u);
    if (!plan->q4k_path && !plan->iq2_path && !plan->iq2_iq2_path &&
        !plan->q2k_path && !plan->mxfp4_path) {
        return false;
    }
    if (!sycl_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !sycl_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !sycl_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !sycl_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !sycl_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return false;
    }
    return true;
}

/* ---- Streaming lookups, all stubbed to their "streaming disabled"
 * behaviour --------------------------------------------------------
 *
 * The research document (ds4-sycl-moe-reference.md) names two streaming
 * lookups reachable from routed_moe_launch's shared preamble
 * unconditionally: cuda_stream_layer_expert_cache_apply (moe_launch.cuh
 * :593 in that draft's line numbering) and cuda_stream_batch_selected_
 * apply_split.  Reading the ROCm source directly (moe_launch.cuh:523-745)
 * found three more reachable from a plain Q4_K decode call
 * (n_tokens == 1, no format gate): cuda_stream_selected_apply_split and
 * cuda_stream_selected_apply feed `split_selected`/`compact_selected`,
 * which gate on n_tokens == 1 alone, not on iq2_gate_path || q2k_path.
 * The batch_stream_selected/batch_stream_split_selected pair (backed by
 * cuda_stream_batch_selected_prepare and cuda_stream_batch_selected_
 * apply_split) IS gated on `iq2_gate_path || q2k_path`, so it is not
 * reachable by any format implemented or stubbed here (q4k_path takes
 * neither branch, and iq2/q2k's own blocks return 0 before any of this
 * preamble matters to them); those two are therefore NOT ported here at
 * all, deferred entirely to whichever future format implementation
 * first reaches them.
 *
 * Every lookup below returns exactly what its ROCm counterpart returns
 * when `g_ssd_streaming_mode` is false (the only state this backend is
 * ever in today; the resident streaming expert cache exists but nothing wires its
 * ds4_gpu_stream_expert_cache_* entries into this dispatcher, so treating
 * streaming as permanently off here is not a simplification of real
 * behaviour, it IS current behaviour). */

static bool sycl_stream_layer_expert_cache_apply_lookup() { return false; }
static bool sycl_stream_selected_apply_split_lookup() { return false; }
static bool sycl_stream_selected_apply_lookup() { return false; }

/* cuda_stream_batch_selected_apply_split and the batch-selected prepare it
 * feeds (cuda_stream_batch_selected_prepare), moe_launch.cuh:612-661:
 * gated on `iq2_gate_path || q2k_path`, so unreachable until a format
 * that reaches them is implemented.  Both are left unported for that
 * reason; both are pure lookups returning false whenever streaming is
 * disabled, which is the only state this backend is ever in (see the
 * block comment above).  The resident streaming expert cache is what would make
 * these meaningful; nothing wires it into this dispatcher yet. */
[[maybe_unused]] static bool sycl_stream_batch_selected_apply_split_lookup() { return false; }
[[maybe_unused]] static bool sycl_stream_batch_selected_prepare_lookup() { return false; }

/* routed_moe_full_table_is_cached, moe_launch.cuh:511-521: checks whether
 * the model's mmap pages for gate/up/down are already resident in ROCm's
 * own page-cache bookkeeping (cuda_model_range_is_cached).  This backend
 * has no equivalent page-residency tracker (SYCL/Level Zero has no
 * concept mirroring ROCm's unified-memory page cache here), so this is a
 * permanent `false`, not a placeholder: there is nothing to report as
 * cached. It only ever gates the batch-streaming branches above, which
 * are unreachable from any format implemented here regardless. */
static bool sycl_routed_moe_full_table_is_cached() { return false; }

/* ---- Q4_K format block -------------------------------------------
 *
 * Dispatch thresholds mirror moe_launch.cuh:750-777 restricted to the
 * q4k_path-reachable branches (`use_sorted_pairs = n_tokens > 1u &&
 * (!q4k_path || n_tokens >= 32u)`, which for q4k_path collapses to
 * `n_tokens >= 32u`; `use_direct_down_sum6 = n_tokens == 1u` once
 * `use_mxfp4_tiny_batch` is excluded and `n_expert <=
 * DS4_ROCM_N_EXPERT_USED` is already guaranteed by build_plan). Three
 * regimes:
 *   n_tokens == 1:        decode gate/up, sum6 direct-to-out down.
 *   1 < n_tokens < 32:    decode gate/up (same kernel, larger pair grid),
 *                         untiled down into scratch, then moe_sum combine.
 *   n_tokens >= 32:       sorted-pairs tile8 gate/up and down into
 *                         scratch, then moe_sum combine. */
static int sycl_routed_moe_q4k_dispatch(
        sycl::queue &q, ds4_gpu_tensor *out, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, sycl_block_q8_K *xq, sycl_block_q8_K *midq,
        const char *gate_w, const char *up_w, const char *down_w,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t xq_blocks,
        uint32_t midq_blocks, uint32_t expert_mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, uint32_t n_tokens,
        uint32_t pair_count, float clamp) {
    const int32_t *sel = (const int32_t *)selected->ptr;
    const float *w = (const float *)weights->ptr;
    float *out_ptr = (float *)out->ptr;
    float *mid_ptr = (float *)mid->ptr;
    float *down_ptr = (float *)down->ptr;

    /* Gate/up, writing SwiGLU intermediates into the real `mid` tensor
     * (moe_launch.cuh:1479-1495 for n_tokens < 32, :1256-1279 for
     * n_tokens >= 32): the "decode" kernel handles every n_tokens < 32
     * call, true decode included, since sorted_pairs (and therefore the
     * tile8 kernel) is only built once n_tokens >= 32 for q4k_path. */
    void *tile_scratch = nullptr;
    sycl_moe_sorted_pairs sp;
    const bool use_sorted_pairs = n_tokens >= 32u;
    if (use_sorted_pairs) {
        if (!sycl_moe_build_sorted_pairs(q, sel, pair_count, n_total_expert, 8u,
                                         &tile_scratch, &sp)) {
            return 0;
        }
    }
    sycl_device_scratch_guard tile_guard(q, tile_scratch);

    if (use_sorted_pairs) {
        sycl_moe_q4k_gate_up_mid_tile8(q, mid_ptr, gate_w, up_w, xq, sp.sorted_pairs,
                                       sp.offsets, sp.counts, sp.tile_total,
                                       sp.tile_experts, sp.tile_starts, w,
                                       gate_expert_bytes, gate_row_bytes, xq_blocks,
                                       expert_mid_dim, n_expert, sp.tile_capacity,
                                       clamp);
    } else {
        sycl_moe_q4k_gate_up_mid_decode(q, mid_ptr, gate_w, up_w, xq, sel, w,
                                        gate_expert_bytes, gate_row_bytes,
                                        xq_blocks, expert_mid_dim, n_expert,
                                        pair_count, clamp);
    }

    /* Quantise the freshly computed mid activations, moe_launch.cuh:1630-1634
     * (`q8_K_quantize_kernel<<<midq_grid,256>>>(midq, mid->ptr, expert_mid_dim,
     * pair_count)`), unconditionally, before any down kernel runs.  Common
     * to every n_tokens regime: gate/up always writes pair_count rows into
     * `mid` regardless of which kernel above produced them. */
    sycl_moe_q8_k_quantize(q, midq, mid_ptr, expert_mid_dim, pair_count);

    if (n_tokens == 1u) {
        /* Direct-to-out combine: no separate moe_sum_kernel pass. */
        sycl_moe_q4k_down_sum6(q, out_ptr, down_w, midq, sel, down_expert_bytes,
                               down_row_bytes, midq_blocks, out_dim, n_expert);
        return 1;
    }
    if (use_sorted_pairs) {
        sycl_moe_q4k_down_tile8(q, down_ptr, down_w, midq, sp.sorted_pairs,
                                sp.offsets, sp.counts, sp.tile_total,
                                sp.tile_experts, sp.tile_starts, down_expert_bytes,
                                down_row_bytes, midq_blocks, out_dim,
                                sp.tile_capacity);
    } else {
        sycl_moe_q4k_down_untiled(q, down_ptr, down_w, midq, sel, down_expert_bytes,
                                  down_row_bytes, midq_blocks, out_dim, n_expert,
                                  pair_count);
    }
    sycl_moe_sum(q, out_ptr, down_ptr, out_dim, n_expert, n_tokens);
    return 1;
}

/* ---- IQ2_XXS gate/up format block (iq2_path, iq2_iq2_path) -----------
 *
 * Dispatch mirrors moe_launch.cuh:750-759 restricted to iq2_gate_path:
 * `use_sorted_pairs = n_tokens > 1u && !disable_resident_iq2_sorted`
 * collapses, once q4k_path and mxfp4_path are excluded, to plain
 * `n_tokens > 1u` -- there is no ">= 32" threshold for this format the
 * way there is for q4k_path.  Two regimes only:
 *   n_tokens == 1: decode gate/up, direct-to-out down.
 *   n_tokens > 1:  sorted-pairs tile8 gate/up, direct-to-out down.
 * down_type distinguishes the two down projections: iq2_path
 * (down_type==10) uses the Q2_K-cast direct-sum kernel; iq2_iq2_path
 * (down_type==16) uses the IQ2_XXS-cast one. Both are direct-to-out
 * (no separate down-scratch-plus-combine pass is needed for either). */
static int sycl_routed_moe_iq2_dispatch(
        sycl::queue &q, ds4_gpu_tensor *out, ds4_gpu_tensor *mid, sycl_block_q8_K *xq,
        sycl_block_q8_K *midq, const char *gate_w, const char *up_w, const char *down_w,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t xq_blocks, uint32_t midq_blocks,
        uint32_t expert_mid_dim, uint32_t out_dim, uint32_t n_total_expert,
        uint32_t n_expert, uint32_t n_tokens, uint32_t pair_count, float clamp,
        bool iq2_iq2_path) {
    const int32_t *sel = (const int32_t *)selected->ptr;
    const float *w = (const float *)weights->ptr;
    float *out_ptr = (float *)out->ptr;
    float *mid_ptr = (float *)mid->ptr;

    void *tile_scratch = nullptr;
    sycl_moe_sorted_pairs sp;
    const bool use_sorted_pairs = n_tokens > 1u;
    if (use_sorted_pairs) {
        if (!sycl_moe_build_sorted_pairs(q, sel, pair_count, n_total_expert, 8u,
                                         &tile_scratch, &sp)) {
            return 0;
        }
    }
    sycl_device_scratch_guard tile_guard(q, tile_scratch);

    if (use_sorted_pairs) {
        sycl_moe_iq2_gate_up_mid_tile8(q, mid_ptr, gate_w, up_w, xq, sp.sorted_pairs,
                                       sp.offsets, sp.counts, sp.tile_total,
                                       sp.tile_experts, sp.tile_starts, w,
                                       gate_expert_bytes, gate_row_bytes, xq_blocks,
                                       expert_mid_dim, n_expert, sp.tile_capacity, clamp);
    } else {
        sycl_moe_iq2_gate_up_mid_decode(q, mid_ptr, gate_w, up_w, xq, sel, w,
                                        gate_expert_bytes, gate_row_bytes, xq_blocks,
                                        expert_mid_dim, n_expert, pair_count, clamp);
    }

    sycl_moe_q8_k_quantize(q, midq, mid_ptr, expert_mid_dim, pair_count);

    if (iq2_iq2_path) {
        sycl_moe_iq2_down_direct(q, out_ptr, down_w, midq, sel, down_expert_bytes,
                                 down_row_bytes, midq_blocks, out_dim, n_expert, n_tokens);
    } else {
        sycl_moe_q2k_down_direct(q, out_ptr, down_w, midq, sel, down_expert_bytes,
                                 down_row_bytes, midq_blocks, out_dim, n_expert, n_tokens);
    }
    return 1;
}

/* ---- Standalone Q2_K format block (q2k_path) --------------------------
 *
 * gate_type==10 && down_type==10: a genuinely separate implementation
 * from iq2_path's down side, not a reuse (moe_launch.cuh:1908 excludes
 * q2k_path from the whole shared block above; it owns its own 670-line
 * exclusive block at moe_launch.cuh:1909-2578).  This port deliberately
 * takes a different shape from ROCm's: it always quantises the
 * activation to Q8_K and uses the Q8_K-quantised gate/up and down
 * kernels (moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel and
 * moe_down_sum6_qwarp32_kernel, generalised over the token dimension) for
 * every n_tokens, never ROCm's raw-float dequant-and-dot kernels
 * (moe_gate_up_mid_q2K_rows_rpb1_w32/rows_w32/expert_batch_sharedx,
 * moe_down_q2K_sum_rows_w32/expert_batch_sharedmid).  ROCm itself only
 * takes the Q8_K path at n_tokens==1 with g_quality_mode off
 * (moe_launch.cuh:2255-2256, :2464-2465); this port takes it always,
 * because the mandated CPU oracle (matvec_q2_k_experts_mid_prequant /
 * matvec_q2_k_experts_accum_prequant) always operates on Q8_K-prequantised
 * activations, and the raw-float kernels compute a value that cannot
 * agree with that oracle at tight tolerance regardless of how faithfully
 * they are ported. See the plan report. */
static int sycl_routed_moe_q2k_dispatch(
        sycl::queue &q, ds4_gpu_tensor *out, ds4_gpu_tensor *mid, sycl_block_q8_K *xq,
        sycl_block_q8_K *midq, const char *gate_w, const char *up_w, const char *down_w,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t xq_blocks, uint32_t midq_blocks,
        uint32_t expert_mid_dim, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens,
        uint32_t pair_count, float clamp) {
    const int32_t *sel = (const int32_t *)selected->ptr;
    const float *w = (const float *)weights->ptr;
    float *out_ptr = (float *)out->ptr;
    float *mid_ptr = (float *)mid->ptr;

    sycl_moe_q2k_gate_up_mid_decode(q, mid_ptr, gate_w, up_w, xq, sel, w, gate_expert_bytes,
                                    gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                    pair_count, clamp);
    sycl_moe_q8_k_quantize(q, midq, mid_ptr, expert_mid_dim, pair_count);
    sycl_moe_q2k_down_direct(q, out_ptr, down_w, midq, sel, down_expert_bytes, down_row_bytes,
                             midq_blocks, out_dim, n_expert, n_tokens);
    return 1;
}

/* Spec 6l: a SYCL kernel cannot dereference the host mmap, so gate_w/up_w/
 * down_w (ordinary host memory, or a host-backed test fixture) must be
 * copied to device memory before any kernel reads them.  q4k_path needed
 * this first; iq2_path/iq2_iq2_path and q2k_path need the identical copy a
 * second and third time, so it is factored here
 * rather than written a fourth time.  Callers still own their own
 * sycl_device_scratch_guard triple for the returned pointers (this
 * function never partially succeeds without freeing what it allocated,
 * but ownership of a successful stage passes to the caller). */
static bool sycl_moe_stage_weights(sycl::queue &q, const char *gate_w, const char *up_w,
                                   const char *down_w, uint64_t gate_bytes, uint64_t down_bytes,
                                   void **gate_dev, void **up_dev, void **down_dev) {
    *gate_dev = sycl::malloc_device((size_t)gate_bytes, q);
    *up_dev = sycl::malloc_device((size_t)gate_bytes, q);
    *down_dev = sycl::malloc_device((size_t)down_bytes, q);
    if (!*gate_dev || !*up_dev || !*down_dev) {
        if (*gate_dev) sycl::free(*gate_dev, q);
        if (*up_dev) sycl::free(*up_dev, q);
        if (*down_dev) sycl::free(*down_dev, q);
        *gate_dev = *up_dev = *down_dev = nullptr;
        return false;
    }
    q.memcpy(*gate_dev, gate_w, (size_t)gate_bytes);
    q.memcpy(*up_dev, up_w, (size_t)gate_bytes);
    q.memcpy(*down_dev, down_w, (size_t)down_bytes);
    q.wait_and_throw();
    return true;
}

/* ---- Selected-expert compaction -----------------------------
 *
 * Every MoE kernel addresses a weight row as `base + expert *
 * expert_bytes + row * row_bytes`, with `expert` read straight out of the
 * `selected` array (verified against every gate/up/down kernel in
 * ds4_sycl_moe.hpp: the decode kernels at ds4_sycl_moe_launch.hpp's own
 * call sites and the tiled kernels via tile_experts, which
 * sycl_moe_build_sorted_pairs derives from the same `selected` array).
 * That arithmetic does not care whether `expert` is a global expert id or
 * a position in a smaller packed table, so staging only the experts a
 * call actually selected -- packed contiguously -- and rewriting
 * `selected` to hold each pair's position in that packed table instead of
 * its global id needs no kernel change. The `weights` array (combine-time
 * per-slot scale) is indexed by (token, slot), never by expert id, so it
 * is untouched either way.
 *
 * sycl_moe_build_expert_compaction below computes that packed table and
 * the rewritten ids; sycl_moe_stage_selected_experts stages the packed
 * table plus the rewritten ids to device memory. sycl_routed_moe_launch
 * decides, per call, whether the packed table is small enough for this to
 * be worth doing at all (see its fallback-threshold comment) and falls
 * back to sycl_moe_stage_weights, addressed with the untouched global
 * `selected`, otherwise. */

/* Reads `selected` back to host and builds two things: the unique set of
 * global expert ids this call actually needs, in first-appearance order
 * (deterministic: `selected` is read in a fixed pair order regardless of
 * how it was produced upstream), and, for every (token, slot) pair, that
 * expert's position within the unique set -- the id every kernel will use
 * once the caller stages only the unique set instead of the full table.
 *
 * Returns false if any id in `selected` is at or beyond n_total_expert.
 * routed_moe_build_plan's own checks do not bound individual `selected`
 * values (only its overall tensor shape), so this cannot assume the
 * upstream router always produced one; per the "or when anything
 * about the shape is unexpected, fall back to today's behaviour"
 * guidance, an out-of-range id is treated as unexpected shape, and the
 * caller falls back to the untouched full-table path rather than trusting
 * a compacted table built from it.
 *
 * DS4_ROUTED_MOE_DEBUG_IDENTITY_REMAP, set to any non-empty value: for
 * the compaction ablation only. Skips the slot remap and instead maps every
 * pair to (global id) mod (unique count) -- compaction still happens, but
 * ids are no longer the correct position in the packed table. This
 * reproduces, deliberately, the exact defect the sorted-pairs trap warns
 * against (a compacted buffer addressed with un-remapped ids) while
 * staying inside the compacted buffer's bounds, so it corrupts output
 * instead of reading past allocated device memory. */
static bool sycl_moe_build_expert_compaction(sycl::queue &q, const int32_t *sel_dev,
                                             uint32_t pair_count, uint32_t n_total_expert,
                                             std::vector<int32_t> *unique_ids,
                                             std::vector<int32_t> *remap) {
    std::vector<int32_t> sel_host((size_t)pair_count);
    q.memcpy(sel_host.data(), sel_dev, (size_t)pair_count * sizeof(int32_t)).wait_and_throw();

    std::vector<int32_t> id_to_slot((size_t)n_total_expert, -1);
    unique_ids->clear();
    remap->assign((size_t)pair_count, 0);
    for (uint32_t i = 0; i < pair_count; i++) {
        int32_t e = sel_host[i];
        if (e < 0) e = 0;
        if ((uint32_t)e >= n_total_expert) return false;
        if (id_to_slot[(size_t)e] < 0) {
            id_to_slot[(size_t)e] = (int32_t)unique_ids->size();
            unique_ids->push_back(e);
        }
        (*remap)[i] = id_to_slot[(size_t)e];
    }

    const char *identity_debug = getenv("DS4_ROUTED_MOE_DEBUG_IDENTITY_REMAP");
    if (identity_debug && identity_debug[0]) {
        const uint32_t unique_count = (uint32_t)unique_ids->size();
        for (uint32_t i = 0; i < pair_count; i++) {
            int32_t e = sel_host[i];
            if (e < 0) e = 0;
            (*remap)[i] = (int32_t)((uint32_t)e % unique_count);
        }
    }
    return true;
}

/* Gathers the host mmap rows for exactly the experts named in
 * `unique_ids` (packed contiguously, in that order) into three fresh
 * device buffers, plus the pair-indexed `remap` array into a fourth.
 *
 * The per-expert rows are gathered into contiguous host staging buffers
 * first (one std::memcpy per expert per matrix, at host RAM bandwidth,
 * not PCIe), then each matrix crosses to the device in a single
 * sycl_stage_host_bytes call. This keeps the number of enqueued SYCL
 * copy commands fixed at four regardless of unique_count, rather than
 * issuing 3*unique_count separate device-copy commands -- each with its
 * own queue-submission overhead -- the way copying straight from the
 * scattered mmap rows one-per-expert would. See sycl_routed_moe_launch's
 * fallback-threshold comment for the arithmetic this trades against.
 *
 * Callers own the four returned pointers via their own
 * sycl_device_scratch_guard, matching sycl_moe_stage_weights's
 * ownership contract exactly: on failure, every allocation this call
 * made is already freed (the local guards below own them until success
 * is confirmed) and every out pointer is left null. */
static bool sycl_moe_stage_selected_experts(
        sycl::queue &q, const char *gate_w, const char *up_w, const char *down_w,
        const int32_t *unique_ids, uint32_t unique_count, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes, const int32_t *remap_host, uint32_t pair_count,
        void **gate_dev, void **up_dev, void **down_dev, void **sel_dev) {
    *gate_dev = *up_dev = *down_dev = *sel_dev = nullptr;
    uint64_t gate_bytes = 0, down_bytes = 0, sel_bytes = 0;
    if (!sycl_u64_mul_checked(unique_count, gate_expert_bytes, &gate_bytes) ||
        !sycl_u64_mul_checked(unique_count, down_expert_bytes, &down_bytes) ||
        !sycl_u64_mul_checked(pair_count, (uint32_t)sizeof(int32_t), &sel_bytes)) {
        return false;
    }

    std::vector<char> gate_stage((size_t)gate_bytes);
    std::vector<char> up_stage((size_t)gate_bytes);
    std::vector<char> down_stage((size_t)down_bytes);
    for (uint32_t i = 0; i < unique_count; i++) {
        const uint64_t expert = (uint64_t)(uint32_t)unique_ids[i];
        memcpy(gate_stage.data() + (uint64_t)i * gate_expert_bytes,
               gate_w + expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(up_stage.data() + (uint64_t)i * gate_expert_bytes,
               up_w + expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(down_stage.data() + (uint64_t)i * down_expert_bytes,
               down_w + expert * down_expert_bytes, (size_t)down_expert_bytes);
    }

    sycl_device_scratch_guard gate_guard = sycl_stage_host_bytes(q, gate_stage.data(), gate_bytes);
    sycl_device_scratch_guard up_guard = sycl_stage_host_bytes(q, up_stage.data(), gate_bytes);
    sycl_device_scratch_guard down_guard = sycl_stage_host_bytes(q, down_stage.data(), down_bytes);
    sycl_device_scratch_guard sel_guard = sycl_stage_host_bytes(q, remap_host, sel_bytes);
    if (!gate_guard.p || !up_guard.p || !down_guard.p || !sel_guard.p) {
        return false;
    }
    /* Ownership transfers to the caller's own guards from here: null the
     * locals so their destructors, run when this function returns, do
     * not free what the caller now owns. */
    *gate_dev = gate_guard.p;
    *up_dev = up_guard.p;
    *down_dev = down_guard.p;
    *sel_dev = sel_guard.p;
    gate_guard.p = up_guard.p = down_guard.p = sel_guard.p = nullptr;
    return true;
}

/* ---- MXFP4 format block -------------------------------------------
 *
 * Dispatch threshold mirrors moe_launch.cuh:750-777 restricted to the
 * mxfp4_path-reachable branches: `use_mxfp4_tiny_batch = mxfp4_path &&
 * n_tokens <= 4u`, and `use_sorted_pairs = n_tokens > 1u &&
 * !use_mxfp4_tiny_batch && (!q4k_path || n_tokens >= 32u)`, which for
 * mxfp4_path (q4k_path false, so the q4k clause is unconditionally true)
 * collapses to `n_tokens > 4u`.  So MXFP4's tile threshold is n_tokens >=
 * 5, NOT Q4_K's n_tokens >= 32 -- confirmed directly against the launch
 * site rather than assumed from Q4_K's own threshold.  Two regimes:
 *   n_tokens <= 4:  decode gate/up (grid.y = pair_count, covers true
 *                   decode and the tiny-batch case identically), sum6
 *                   direct-to-out down (grid.y = n_tokens, no moe_sum
 *                   pass: moe_launch.cuh:1706-1707's comment, "the direct
 *                   decode kernel writes the final token row").
 *   n_tokens >= 5:  sorted-pairs tile8 gate/up and down into scratch,
 *                   then moe_sum combine, identical structure to q4k's
 *                   n_tokens >= 32 regime.
 *
 * Explicitly NOT ported: the four env-gated occupancy variants of the
 * gate/up tile8 kernel (TILE4, ROW64, LDSB, TILE32,
 * rocm/ds4_rocm_moe.cuh:2246-2229 and 2381-2637).  Each is documented in
 * ROCm as bit-exact against the canonical tile8 kernel (same dot helper,
 * same reduction, only tile/thread geometry differs), off by default, and
 * exists purely to trade staging footprint against resident-warp count on
 * AMDGCN hardware.  Porting them costs four more kernels' worth of A/B
 * tests and zero oracle-backed correctness surface (this trade is
 * explicitly optional); skipping them costs a tuning pass this
 * project defers until real occupancy data from the B60
 * target exists to tune against.  The row-group parameter below is
 * different in kind (a parameter on the one kernel actually shipped here,
 * not a fifth kernel body) and is ported for real. */
/* DS4_ROCM_MXFP4_DOWN_RGROUP, rocm/ds4_rocm_moe_launch.cuh:801-807: a
 * row-group count parameter on the canonical down tile8 kernel, not a
 * distinct kernel body, so it is A/B-tested byte-exact against the
 * default (1) rather than needing its own oracle (see
 * the MXFP4 port's scope note).  Same env var name and
 * validation range as ROCm (1..8), defaulting to 1 when unset, empty, or
 * out of range. */
static uint32_t sycl_mxfp4_down_row_groups_from_env() {
    const char *v = getenv("DS4_ROCM_MXFP4_DOWN_RGROUP");
    if (v && v[0]) {
        const long rv = strtol(v, nullptr, 10);
        if (rv >= 1 && rv <= 8) return (uint32_t)rv;
    }
    return 1u;
}

static int sycl_routed_moe_mxfp4_dispatch(
        sycl::queue &q, ds4_gpu_tensor *out, ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        sycl_block_q8_K *xq, sycl_block_q8_K *midq, const char *gate_w, const char *up_w,
        const char *down_w, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t xq_blocks, uint32_t midq_blocks,
        uint32_t expert_mid_dim, uint32_t out_dim, uint32_t n_total_expert, uint32_t n_expert,
        uint32_t n_tokens, uint32_t pair_count, float clamp, uint32_t down_row_groups) {
    const int32_t *sel = (const int32_t *)selected->ptr;
    const float *w = (const float *)weights->ptr;
    float *out_ptr = (float *)out->ptr;
    float *mid_ptr = (float *)mid->ptr;
    float *down_ptr = (float *)down->ptr;
    const bool tiny_batch = n_tokens <= 4u;

    void *tile_scratch = nullptr;
    sycl_moe_sorted_pairs sp;
    if (!tiny_batch) {
        if (!sycl_moe_build_sorted_pairs(q, sel, pair_count, n_total_expert, 8u, &tile_scratch,
                                         &sp)) {
            return 0;
        }
    }
    sycl_device_scratch_guard tile_guard(q, tile_scratch);

    if (tiny_batch) {
        sycl_moe_mxfp4_gate_up_mid_decode(q, mid_ptr, gate_w, up_w, xq, sel, w, gate_expert_bytes,
                                          gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                          pair_count, clamp);
    } else {
        sycl_moe_mxfp4_gate_up_mid_tile8(q, mid_ptr, gate_w, up_w, xq, sp.sorted_pairs,
                                         sp.offsets, sp.counts, sp.tile_total, sp.tile_experts,
                                         sp.tile_starts, w, gate_expert_bytes, gate_row_bytes,
                                         xq_blocks, expert_mid_dim, n_expert, sp.tile_capacity,
                                         clamp);
    }

    sycl_moe_q8_k_quantize(q, midq, mid_ptr, expert_mid_dim, pair_count);

    if (tiny_batch) {
        sycl_moe_mxfp4_down_sum6(q, out_ptr, down_w, midq, sel, down_expert_bytes, down_row_bytes,
                                 midq_blocks, out_dim, n_expert, n_tokens);
        return 1;
    }

    sycl_moe_mxfp4_down_tile8(q, down_ptr, down_w, midq, sp.sorted_pairs, sp.offsets, sp.counts,
                              sp.tile_total, sp.tile_experts, sp.tile_starts, down_expert_bytes,
                              down_row_bytes, midq_blocks, out_dim, sp.tile_capacity,
                              down_row_groups);
    sycl_moe_sum(q, out_ptr, down_ptr, out_dim, n_expert, n_tokens);
    return 1;
}

/* ---- The shared launcher ---------------------------------------------
 *
 * Mirrors routed_moe_launch (moe_launch.cuh:523-745 for the preamble
 * ported here; the not-yet-implemented formats do not need the rest).
 * Not ported: the cudaEvent-based decode-profiling harness
 * (moe_launch.cuh:27-195, ~170 lines, env-var gated, not required for
 * correctness). */
static int sycl_routed_moe_launch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        bool force_resident) {
    (void)layer_index;
    (void)force_resident;

    sycl_routed_moe_plan plan;
    if (!sycl_routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                                    gate_offset, up_offset, down_offset, gate_type,
                                    down_type, gate_expert_bytes, down_expert_bytes,
                                    expert_in_dim, expert_mid_dim, out_dim, selected,
                                    weights, n_total_expert, n_expert, x, n_tokens,
                                    &plan)) {
        return 0;
    }
    if (g_devices.empty()) return 0;

    uint64_t pair_count64 = 0;
    if (!sycl_u64_mul_checked(n_tokens, n_expert, &pair_count64) ||
        pair_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t pair_count = (uint32_t)pair_count64;

    try {
        sycl::queue &q = ds4_sycl_current_queue();

        /* Streaming preamble: every lookup below is permanently
         * streaming-disabled (see the block comment above), so
         * stream_full_layer/split_selected/compact_selected are always
         * false and every call always falls through to reading weights
         * straight from the model mmap, exactly as ROCm itself does
         * whenever g_ssd_streaming_mode is false. */
        const bool stream_full_layer = (n_tokens > 1u || force_resident) &&
                                       sycl_stream_layer_expert_cache_apply_lookup();
        const bool full_table_cached =
                !stream_full_layer && sycl_routed_moe_full_table_is_cached();
        const bool split_selected = !stream_full_layer && n_tokens == 1u &&
                                    sycl_stream_selected_apply_split_lookup();
        const bool compact_selected = split_selected ||
                (!stream_full_layer && n_tokens == 1u &&
                 sycl_stream_selected_apply_lookup());
        (void)full_table_cached;
        (void)compact_selected;

        const char *gate_w = sycl_model_range_ptr(model_map, gate_offset,
                                                  plan.gate_bytes, model_size,
                                                  "moe_gate");
        const char *up_w = sycl_model_range_ptr(model_map, up_offset,
                                                plan.gate_bytes, model_size,
                                                "moe_up");
        const char *down_w = sycl_model_range_ptr(model_map, down_offset,
                                                  plan.down_bytes, model_size,
                                                  "moe_down");
        if (!gate_w || !up_w || !down_w) return 0;

        /* Scratch-buffer-reuse precondition, moe_launch.cuh:746: gate and
         * down (the caller's own scratch tensors, sized for the F32
         * gate/up/down activations) double as Q8_K quantisation scratch
         * for the activations and mid-layer values respectively, since
         * the two never need to be live at once.  ROCm's fallback for
         * this precondition failing hardcodes IQ2_XXS (moe_gate_up_mid_
         * f32_kernel, moe.cuh:5344) and Q2_K (moe_down_f32_kernel,
         * moe.cuh:5464) block casts regardless of the caller's actual
         * gate_type/down_type -- despite its name, "F32 fallback" means
         * "activations stay F32-unquantised", not "weights are F32".
         * Reusing it for Q4_K, MXFP4, IQ2_XXS-paired-with-IQ2_XXS-down or
         * Q2_K-both-sides would silently misinterpret weight bytes. No
         * format this dispatcher implements can correctly take that
         * fallback; a precondition failure is therefore a clean failure
         * here rather than ROCm's fallback path. */
        const uint32_t xq_blocks = expert_in_dim / kMoeQK;
        const uint32_t midq_blocks = expert_mid_dim / kMoeQK;
        uint64_t xq_count = 0, midq_count = 0, xq_bytes = 0, midq_bytes = 0;
        if (!sycl_u64_mul_checked(n_tokens, xq_blocks, &xq_count) ||
            !sycl_u64_mul_checked(pair_count64, midq_blocks, &midq_count) ||
            !sycl_u64_mul_checked(xq_count, sizeof(sycl_block_q8_K), &xq_bytes) ||
            !sycl_u64_mul_checked(midq_count, sizeof(sycl_block_q8_K), &midq_bytes)) {
            return 0;
        }
        if (!(down->bytes >= xq_bytes && gate->bytes >= midq_bytes)) {
            return 0;
        }

        sycl_block_q8_K *xq = (sycl_block_q8_K *)down->ptr;
        sycl_block_q8_K *midq = (sycl_block_q8_K *)gate->ptr;
        sycl_moe_q8_k_quantize(q, xq, (const float *)x->ptr, expert_in_dim, n_tokens);

        /* gate_w/up_w/down_w point into model_map, ordinary host memory (or
         * a plain host-backed test fixture): a SYCL kernel cannot
         * dereference an arbitrary host pointer the way a CUDA/HIP kernel
         * can under unified virtual addressing (spec section 6a's "known
         * gap" is the general case of this; ROCm's own routed_moe_launch
         * reads gate_w/up_w/down_w directly inside its kernels and gets
         * away with it for exactly this reason).  Every other SYCL kernel
         * in this backend that reads a model-mapped weight range copies it
         * to device memory first; every format block below applies that
         * same pattern to the whole per-layer gate/up/down table, since
         * MoE does not know which experts `selected` names until the
         * device has already computed it.  Real zero-copy residency is
         * ds4_gpu_register_model_map_no_copy's job, still a stub in this
         * backend; this is a correctness-first placeholder that costs a
         * full-table copy on every call until that (or the resident
         * streaming expert cache) is wired in here.  Applies to every format path
         * without exception: an unstaged range does not fault, it reads
         * zeros and reports success, so a path that skips this is wrong in
         * a way no test on this hardware can see.
         *
         * Which table gets staged is decided here: the full
         * n_total_expert table, as above, or only the experts this call's
         * `selected` names. Compaction's guaranteed reduction factor is
         * n_total_expert / unique_count, so it is worth doing whenever
         * unique_count is small; the question is where it stops being
         * worth it. sycl_moe_stage_selected_experts gathers its packed
         * table on the host before crossing to device (see its own
         * comment), so, pessimistically, treat that host gather as
         * costing as much again as the device copy itself -- even though
         * host RAM bandwidth comfortably exceeds PCIe bandwidth, so this
         * assumption is conservative in compaction's favour. Under that
         * assumption compaction is a net win exactly while 2 * unique_count
         * <= n_total_expert, i.e. unique_count <= n_total_expert / 2:
         * below that point staging only the selected experts moves fewer
         * effective bytes than staging the full table even with the
         * gather taxed twice; at or above it, fall back to the plain
         * full-table copy, whose cost does not depend on how the union
         * came out. Decode's real union (n_expert_used, 6 of 256 for
         * Flash) sits far below this threshold; a wide prefill batch's
         * union approaches the full table and is
         * exactly what this fallback is for. */
        std::vector<int32_t> unique_ids, remap_ids;
        const bool compaction_shape_ok = sycl_moe_build_expert_compaction(
                q, (const int32_t *)selected->ptr, pair_count, n_total_expert,
                &unique_ids, &remap_ids);
        const uint32_t unique_count =
                compaction_shape_ok ? (uint32_t)unique_ids.size() : n_total_expert;
        const bool use_compaction =
                compaction_shape_ok && unique_count > 0u && unique_count <= n_total_expert / 2u;

        void *gate_dev = nullptr, *up_dev = nullptr, *down_dev = nullptr, *remap_dev = nullptr;
        ds4_gpu_tensor remapped_selected{};
        const ds4_gpu_tensor *effective_selected = selected;
        uint32_t effective_n_total_expert = n_total_expert;

        if (use_compaction) {
            if (!sycl_moe_stage_selected_experts(
                        q, gate_w, up_w, down_w, unique_ids.data(), unique_count,
                        gate_expert_bytes, down_expert_bytes, remap_ids.data(), pair_count,
                        &gate_dev, &up_dev, &down_dev, &remap_dev)) {
                return 0;
            }
            remapped_selected = ds4_gpu_tensor{remap_dev, (uint64_t)pair_count * sizeof(int32_t),
                                               /*owner=*/0, selected->device_id};
            effective_selected = &remapped_selected;
            effective_n_total_expert = unique_count;
        } else {
            if (!sycl_moe_stage_weights(q, gate_w, up_w, down_w, plan.gate_bytes, plan.down_bytes,
                                        &gate_dev, &up_dev, &down_dev)) {
                return 0;
            }
        }
        sycl_device_scratch_guard gate_guard(q, gate_dev);
        sycl_device_scratch_guard up_guard(q, up_dev);
        sycl_device_scratch_guard down_guard(q, down_dev);
        sycl_device_scratch_guard remap_guard(q, remap_dev);

        g_sycl_moe_last_staged_expert_count = use_compaction ? unique_count : n_total_expert;
        g_sycl_moe_last_staged_bytes =
                2ull * (uint64_t)g_sycl_moe_last_staged_expert_count * gate_expert_bytes +
                (uint64_t)g_sycl_moe_last_staged_expert_count * down_expert_bytes;

        if (plan.q4k_path) {
            return sycl_routed_moe_q4k_dispatch(
                    q, out, mid, down, xq, midq, (const char *)gate_dev,
                    (const char *)up_dev, (const char *)down_dev, weights,
                    effective_selected, gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                    down_row_bytes, xq_blocks, midq_blocks, expert_mid_dim, out_dim,
                    effective_n_total_expert, n_expert, n_tokens, pair_count, clamp);
        }
        if (plan.iq2_path || plan.iq2_iq2_path) {
            return sycl_routed_moe_iq2_dispatch(
                    q, out, mid, xq, midq, (const char *)gate_dev, (const char *)up_dev,
                    (const char *)down_dev, weights, effective_selected, gate_expert_bytes,
                    gate_row_bytes, down_expert_bytes, down_row_bytes, xq_blocks,
                    midq_blocks, expert_mid_dim, out_dim, effective_n_total_expert, n_expert,
                    n_tokens, pair_count, clamp, plan.iq2_iq2_path);
        }
        if (plan.q2k_path) {
            return sycl_routed_moe_q2k_dispatch(
                    q, out, mid, xq, midq, (const char *)gate_dev, (const char *)up_dev,
                    (const char *)down_dev, weights, effective_selected, gate_expert_bytes,
                    gate_row_bytes, down_expert_bytes, down_row_bytes, xq_blocks,
                    midq_blocks, expert_mid_dim, out_dim, n_expert, n_tokens, pair_count,
                    clamp);
        }
        if (plan.mxfp4_path) {
            return sycl_routed_moe_mxfp4_dispatch(
                    q, out, mid, down, xq, midq, (const char *)gate_dev, (const char *)up_dev,
                    (const char *)down_dev, weights, effective_selected, gate_expert_bytes,
                    gate_row_bytes, down_expert_bytes, down_row_bytes, xq_blocks, midq_blocks,
                    expert_mid_dim, out_dim, effective_n_total_expert, n_expert, n_tokens,
                    pair_count, clamp, sycl_mxfp4_down_row_groups_from_env());
        }
        return 0; /* unreachable: build_plan already required exactly one path. */
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_launch failed: %s\n", e.what());
        return 0;
    }
}

}  // namespace

/* Test-only instrumentation: reports how many experts, and how
 * many gate+up+down bytes, the most recently completed
 * sycl_routed_moe_launch call actually staged host-to-device, whichever
 * of the compacted or full-table path it took. See the comment on
 * g_sycl_moe_last_staged_expert_count above for why a counter, not an
 * oracle comparison, is what proves compaction happened. */
extern "C" uint32_t ds4_sycl_moe_test_last_staged_expert_count(void) {
    return g_sycl_moe_last_staged_expert_count;
}

extern "C" uint64_t ds4_sycl_moe_test_last_staged_bytes(void) {
    return g_sycl_moe_last_staged_bytes;
}

/* ---- ABI entries ------------------------------------------------------
 *
 * ds4_gpu_routed_moe_one_tensor / _batch_tensor, moe_launch.cuh:2638-2662.
 * The two entry points differ only in how they call the shared launcher:
 * the decode entry rejects a non-null add_in (the addend fold is
 * Metal-only) and always passes n_tokens=1; the batch entry always
 * writes false through mid_is_f16 (if given a non-null pointer) and
 * forwards the caller's n_tokens. */

extern "C" int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in,
        uint32_t layer_index, bool force_resident) {
    if (add_in) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed MoE addend fold is Metal-only\n");
        return 0;
    }
    return sycl_routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                                  gate_offset, up_offset, down_offset, gate_type,
                                  down_type, gate_expert_bytes, gate_row_bytes,
                                  down_expert_bytes, down_row_bytes, expert_in_dim,
                                  expert_mid_dim, out_dim, selected, weights,
                                  n_total_expert, n_expert, clamp, x, layer_index,
                                  1u, force_resident);
}

extern "C" int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        bool *mid_is_f16, bool force_resident) {
    if (mid_is_f16) *mid_is_f16 = false;
    return sycl_routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                                  gate_offset, up_offset, down_offset, gate_type,
                                  down_type, gate_expert_bytes, gate_row_bytes,
                                  down_expert_bytes, down_row_bytes, expert_in_dim,
                                  expert_mid_dim, out_dim, selected, weights,
                                  n_total_expert, n_expert, clamp, x, layer_index,
                                  n_tokens, force_resident);
}
