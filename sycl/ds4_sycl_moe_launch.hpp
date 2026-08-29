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

namespace {

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
         * Reusing it for Q4_K, MXFP4 or IQ2_XXS-paired-with-IQ2_XXS-down
         * would silently misinterpret weight bytes. This dispatcher does
         * not yet implement IQ2_XXS or Q2_K, so there is no format it
         * can correctly take that fallback for; a
         * precondition failure is therefore a clean failure here rather
         * than ROCm's fallback path. */
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

        if (plan.q4k_path) {
            /* gate_w/up_w/down_w point into model_map, ordinary host
             * memory (or a plain host-backed test fixture): a SYCL
             * kernel cannot dereference an arbitrary host pointer the
             * way a CUDA/HIP kernel can under unified virtual
             * addressing (spec section 6a's "known gap" is the general
             * case of this; ROCm's own routed_moe_launch reads gate_w/
             * up_w/down_w directly inside its kernels and gets away with
             * it for exactly this reason).  Every other SYCL kernel in
             * this backend that reads a model-mapped weight range copies
             * it to device memory first (ds4_sycl_embedding.hpp's
             * per-token row copy, ds4_sycl_compressor.hpp's APE-table
             * staging); this is that same pattern applied to the whole
             * per-layer gate/up/down table, since MoE does not know
             * which experts `selected` names until the device has
             * already computed it.  Real zero-copy residency is
             * ds4_gpu_register_model_map_no_copy's job, still a stub in
             * this backend; this is a correctness-first placeholder that
             * costs a full-table copy on every call until that (or the
             * resident streaming expert cache) is wired in here. */
            void *gate_dev = sycl::malloc_device((size_t)plan.gate_bytes, q);
            void *up_dev = sycl::malloc_device((size_t)plan.gate_bytes, q);
            void *down_dev = sycl::malloc_device((size_t)plan.down_bytes, q);
            if (!gate_dev || !up_dev || !down_dev) {
                if (gate_dev) sycl::free(gate_dev, q);
                if (up_dev) sycl::free(up_dev, q);
                if (down_dev) sycl::free(down_dev, q);
                return 0;
            }
            sycl_device_scratch_guard gate_guard(q, gate_dev);
            sycl_device_scratch_guard up_guard(q, up_dev);
            sycl_device_scratch_guard down_guard(q, down_dev);
            q.memcpy(gate_dev, gate_w, (size_t)plan.gate_bytes);
            q.memcpy(up_dev, up_w, (size_t)plan.gate_bytes);
            q.memcpy(down_dev, down_w, (size_t)plan.down_bytes);
            q.wait_and_throw();

            return sycl_routed_moe_q4k_dispatch(
                    q, out, mid, down, xq, midq, (const char *)gate_dev,
                    (const char *)up_dev, (const char *)down_dev, weights,
                    selected, gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                    down_row_bytes, xq_blocks, midq_blocks, expert_mid_dim, out_dim,
                    n_total_expert, n_expert, n_tokens, pair_count, clamp);
        }
        if (plan.iq2_path || plan.iq2_iq2_path) {
            /* IQ2_XXS gate/up (+ Q2_K down for iq2_path); not yet implemented. */
            return 0;
        }
        if (plan.q2k_path) {
            /* Standalone Q2_K both-sides path, including the WMMA
             * hotlist kernels and its own dedicated scratch launcher
             * (routed_moe_q2_float_down_launch); not yet implemented. */
            return 0;
        }
        if (plan.mxfp4_path) {
            /* Not yet implemented; no CPU oracle exists for this format at all. */
            return 0;
        }
        return 0; /* unreachable: build_plan already required exactly one path. */
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_launch failed: %s\n", e.what());
        return 0;
    }
}

}  // namespace

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
