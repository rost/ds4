#pragma once

/* Routed-MoE kernels: the format-agnostic machinery every quantised
 * format's dispatcher block depends on (sort infrastructure, the Q8_K
 * activation quantiser, sub-group reductions, the weighted-sum combine),
 * plus the Q4_K kernels this header proves the machinery with.
 *
 * Ported from rocm/ds4_rocm_moe.cuh.  Line numbers cited per section below
 * were verified directly against the ROCm source while writing this file,
 * not trusted from ds4-sycl-moe-reference.md, per the project standing
 * rule for load-bearing claims.
 *
 * This header defines internal kernels and helpers only.  No ABI entry
 * points live here; those are in ds4_sycl_moe_launch.hpp. */

#include "ds4_sycl_common.hpp"

#include <cstring>

namespace {

/* dev_f16_to_f32, rocm/ds4_rocm_moe.cuh:9-11.  Same bit-cast pattern
 * already used in ds4_sycl_embedding.hpp and ds4_sycl_compressor.hpp; not
 * consolidated into a shared helper yet (see ds4-sycl-STATE.md's deferred
 * cleanup list, blocked on a pending test-split refactor landing). */
static inline float sycl_moe_f16_to_f32(uint16_t v) {
    return (float)sycl::bit_cast<sycl::half>(v);
}

/* ---- Sort infrastructure ------------------------------------------
 *
 * rocm/ds4_rocm_moe.cuh:1255-1350: a counting sort of (token, slot) pairs
 * by selected expert.  Every format's expert-tiled batched path consumes
 * its output (sorted_pairs, offsets, counts, tile_experts, tile_starts)
 * with an identical memory layout, and the CPU oracle for the whole
 * subsystem (ds4.c:11126-11166, layer_routed_moe_batch) performs the same
 * sort in scalar form.
 *
 * moe_scatter_sorted_pairs_kernel (moe.cuh:1284-1300, no "_deterministic"
 * suffix) is NOT ported: it is unannotated but superseded by
 * moe_scatter_sorted_pairs_deterministic_kernel, which is the only scatter
 * kernel any launch site in moe_launch.cuh actually calls (verified by
 * grepping every "_kernel<<<" launch site in that file: moe_launch.cuh
 * only ever launches the "_deterministic" name). */

static void sycl_moe_count_sorted_pairs(sycl::queue &q, const int32_t *selected,
                                        uint32_t *counts, uint32_t pair_count,
                                        uint32_t n_total_expert) {
    if (pair_count == 0u) return;
    q.parallel_for(sycl::range<1>(pair_count), [=](sycl::id<1> id) {
         uint32_t pair = (uint32_t)id[0];
         int32_t expert_i = selected[pair];
         if (expert_i < 0) expert_i = 0;
         if ((uint32_t)expert_i >= n_total_expert) return;
         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
             ref(counts[(uint32_t)expert_i]);
         ref.fetch_add(1u);
     }).wait_and_throw();
}

/* moe_prefix_sorted_pairs_kernel, moe.cuh:1266-1279: a single-work-item
 * exclusive prefix sum over n_total_expert buckets.  cursors is written
 * for parity with the ROCm signature; this port's scatter kernel below
 * recomputes each bucket's write position with its own sequential scan
 * rather than an atomic cursor increment (see the determinism note), so
 * cursors is not read back, but keeping the parameter matches the ROCm
 * shape and leaves room for a future incremental scatter variant. */
static void sycl_moe_prefix_sorted_pairs(sycl::queue &q, uint32_t *offsets,
                                         uint32_t *cursors, const uint32_t *counts,
                                         uint32_t n_total_expert) {
    q.single_task([=]() {
         uint32_t sum = 0;
         for (uint32_t e = 0; e < n_total_expert; e++) {
             offsets[e] = sum;
             cursors[e] = sum;
             sum += counts[e];
         }
         offsets[n_total_expert] = sum;
     }).wait_and_throw();
}

/* moe_scatter_sorted_pairs_deterministic_kernel, moe.cuh:1301-1316: one
 * work-item per expert, each doing its own sequential scan over every
 * pair so bucket order is a stable function of pair index, never of
 * thread-scheduling order.  This determinism is load-bearing: the
 * expert-tile kernels this sort feeds are row-position sensitive enough
 * that a non-deterministic atomic-append order changes logits (comment
 * preserved from the ROCm source, moe.cuh:1298-1299). */
static void sycl_moe_scatter_sorted_pairs_deterministic(
        sycl::queue &q, uint32_t *sorted_pairs, const uint32_t *offsets,
        const int32_t *selected, uint32_t pair_count, uint32_t n_total_expert) {
    q.parallel_for(sycl::range<1>(n_total_expert), [=](sycl::id<1> id) {
         uint32_t expert = (uint32_t)id[0];
         uint32_t pos = offsets[expert];
         for (uint32_t pair = 0; pair < pair_count; pair++) {
             int32_t expert_i = selected[pair];
             if (expert_i < 0) expert_i = 0;
             if ((uint32_t)expert_i == expert) sorted_pairs[pos++] = pair;
         }
     }).wait_and_throw();
}

/* moe_build_expert_tile_offsets_kernel, moe.cuh:1327-1340: single-work-item
 * exclusive prefix sum of per-expert tile counts (ceil(count/block_m)). */
static void sycl_moe_build_expert_tile_offsets(
        sycl::queue &q, uint32_t *tile_offsets, uint32_t *tile_total,
        const uint32_t *counts, uint32_t block_m, uint32_t n_total_expert) {
    q.single_task([=]() {
         uint32_t sum = 0;
         for (uint32_t e = 0; e < n_total_expert; e++) {
             tile_offsets[e] = sum;
             sum += (counts[e] + block_m - 1u) / block_m;
         }
         tile_offsets[n_total_expert] = sum;
         *tile_total = sum;
     }).wait_and_throw();
}

/* moe_build_expert_tiles_kernel, moe.cuh:1342-1350: one work-item per
 * expert, filling in which expert and starting pair-index each of that
 * expert's tiles owns. */
static void sycl_moe_build_expert_tiles(
        sycl::queue &q, uint32_t *tile_experts, uint32_t *tile_starts,
        const uint32_t *tile_offsets, const uint32_t *counts, uint32_t block_m,
        uint32_t n_total_expert) {
    q.parallel_for(sycl::range<1>(n_total_expert), [=](sycl::id<1> id) {
         uint32_t e = (uint32_t)id[0];
         uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
         uint32_t off = tile_offsets[e];
         for (uint32_t t = 0; t < ntiles; t++) {
             tile_experts[off + t] = e;
             tile_starts[off + t] = t * block_m;
         }
     }).wait_and_throw();
}

/* Combined scratch layout for one sorted-pairs-plus-tiles build: counts,
 * offsets, sorted_pairs and the tile arrays all come from a single
 * allocation, mirroring ROCm's single cuda_tmp_alloc call
 * (moe_launch.cuh:1002-1004) so a partial-allocation-failure state (some
 * arrays present, others not) is structurally impossible, exactly as in
 * ROCm.  Every offset is uint32_t-aligned by construction (every array
 * element is 4 bytes). */
struct sycl_moe_sorted_pairs {
    uint32_t *counts        = nullptr;  /* n_total_expert */
    uint32_t *offsets       = nullptr;  /* n_total_expert + 1 */
    uint32_t *sorted_pairs  = nullptr;  /* pair_count */
    uint32_t *tile_total    = nullptr;  /* 1 */
    uint32_t *tile_experts  = nullptr;  /* tile_capacity */
    uint32_t *tile_starts   = nullptr;  /* tile_capacity */
    uint32_t  tile_capacity = 0;
};

/* Runs the full counting sort plus expert-tile build described above.
 * Returns false (and leaves *out untouched) on overflow or allocation
 * failure.  On success, *scratch_out is the single device allocation
 * backing every pointer in *out; the caller frees it (a
 * sycl_device_scratch_guard is the expected way). */
static bool sycl_moe_build_sorted_pairs(sycl::queue &q, const int32_t *d_selected,
                                        uint32_t pair_count, uint32_t n_total_expert,
                                        uint32_t block_m, void **scratch_out,
                                        sycl_moe_sorted_pairs *out,
                                        bool poison_for_test = false) {
    if (!out || !scratch_out || n_total_expert == 0u || block_m == 0u) return false;
    const uint32_t tile_capacity =
        (pair_count + block_m - 1u) / block_m + n_total_expert;
    uint64_t counts_bytes = 0, offsets_bytes = 0, sorted_bytes = 0,
             tile_offsets_bytes = 0, tile_arr_bytes = 0;
    if (!sycl_u64_mul_checked(n_total_expert, sizeof(uint32_t), &counts_bytes) ||
        !sycl_u64_mul_checked((uint64_t)n_total_expert + 1u, sizeof(uint32_t),
                              &offsets_bytes) ||
        !sycl_u64_mul_checked(pair_count, sizeof(uint32_t), &sorted_bytes) ||
        !sycl_u64_mul_checked((uint64_t)n_total_expert + 1u, sizeof(uint32_t),
                              &tile_offsets_bytes) ||
        !sycl_u64_mul_checked(tile_capacity, sizeof(uint32_t), &tile_arr_bytes)) {
        return false;
    }
    const uint64_t tile_total_bytes = sizeof(uint32_t);

    uint64_t off = 0;
    const uint64_t counts_off = off;
    if (!sycl_u64_add_checked(off, counts_bytes, &off)) return false;
    const uint64_t offsets_off = off;
    if (!sycl_u64_add_checked(off, offsets_bytes, &off)) return false;
    const uint64_t cursors_off = off;
    if (!sycl_u64_add_checked(off, counts_bytes, &off)) return false;
    const uint64_t sorted_off = off;
    if (!sycl_u64_add_checked(off, sorted_bytes, &off)) return false;
    const uint64_t tile_offsets_off = off;
    if (!sycl_u64_add_checked(off, tile_offsets_bytes, &off)) return false;
    const uint64_t tile_total_off = off;
    if (!sycl_u64_add_checked(off, tile_total_bytes, &off)) return false;
    const uint64_t tile_experts_off = off;
    if (!sycl_u64_add_checked(off, tile_arr_bytes, &off)) return false;
    const uint64_t tile_starts_off = off;
    if (!sycl_u64_add_checked(off, tile_arr_bytes, &off)) return false;
    const uint64_t scratch_bytes = off;

    void *scratch = sycl::malloc_device((size_t)scratch_bytes, q);
    if (!scratch) return false;
    *scratch_out = scratch;
    uint8_t *base = (uint8_t *)scratch;

    uint32_t *counts       = (uint32_t *)(base + counts_off);
    uint32_t *offsets      = (uint32_t *)(base + offsets_off);
    uint32_t *cursors      = (uint32_t *)(base + cursors_off);
    uint32_t *sorted_pairs = (uint32_t *)(base + sorted_off);
    uint32_t *tile_offsets = (uint32_t *)(base + tile_offsets_off);
    uint32_t *tile_total   = (uint32_t *)(base + tile_total_off);
    uint32_t *tile_experts = (uint32_t *)(base + tile_experts_off);
    uint32_t *tile_starts  = (uint32_t *)(base + tile_starts_off);

    /* Test-only: poison the whole scratch region with a fixed non-zero,
     * non-plausible-index byte pattern before anything runs, so a caller
     * verifying "unwritten output is detectable" (e.g. a tile array slot
     * past this build's own tile_total) is checking a known poison value
     * rather than the platform's uninitialised-device-memory contents,
     * which are not guaranteed to differ from a real index. malloc_device
     * memory is NOT zero-initialised by the SYCL spec, so skipping this
     * for the real (non-test) dispatcher path is correct: every byte the
     * dispatcher reads back is one of this function's own kernels wrote. */
    if (poison_for_test) {
        q.memset(scratch, 0xAB, (size_t)scratch_bytes).wait_and_throw();
    }
    q.memset(counts, 0, (size_t)counts_bytes).wait_and_throw();
    sycl_moe_count_sorted_pairs(q, d_selected, counts, pair_count, n_total_expert);
    sycl_moe_prefix_sorted_pairs(q, offsets, cursors, counts, n_total_expert);
    sycl_moe_scatter_sorted_pairs_deterministic(q, sorted_pairs, offsets,
                                                d_selected, pair_count,
                                                n_total_expert);
    sycl_moe_build_expert_tile_offsets(q, tile_offsets, tile_total, counts,
                                       block_m, n_total_expert);
    sycl_moe_build_expert_tiles(q, tile_experts, tile_starts, tile_offsets,
                                counts, block_m, n_total_expert);

    out->counts        = counts;
    out->offsets       = offsets;
    out->sorted_pairs  = sorted_pairs;
    out->tile_total    = tile_total;
    out->tile_experts  = tile_experts;
    out->tile_starts   = tile_starts;
    out->tile_capacity = tile_capacity;
    return true;
}

}  // namespace

/* ---- Test-only side doors ------------------------------------------
 *
 * None of the routed-MoE ABI entries expose the sort's intermediate arrays;
 * these extern "C" hooks exist purely so tests/test_sycl_moe.c (a plain C
 * file) can drive the sort in isolation, matching the precedent in
 * sycl/ds4_sycl_streaming.hpp's ds4_sycl_stream_test_* functions. */

extern "C" int ds4_sycl_moe_test_sort(const int32_t *selected, uint32_t pair_count,
                                      uint32_t n_total_expert, uint32_t block_m,
                                      uint32_t *counts_out, uint32_t *offsets_out,
                                      uint32_t *sorted_pairs_out,
                                      uint32_t *tile_total_out,
                                      uint32_t *tile_experts_out,
                                      uint32_t *tile_starts_out,
                                      uint32_t tile_capacity_cap) {
    if (g_devices.empty() || !selected || !counts_out || !offsets_out ||
        !sorted_pairs_out) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        int32_t *d_selected = (int32_t *)sycl::malloc_device(
                (size_t)pair_count * sizeof(int32_t), q);
        if (!d_selected && pair_count != 0u) return 0;
        sycl_device_scratch_guard sel_guard(q, d_selected);
        if (pair_count != 0u) {
            q.memcpy(d_selected, selected, (size_t)pair_count * sizeof(int32_t))
                    .wait_and_throw();
        }

        void *scratch = nullptr;
        sycl_moe_sorted_pairs res;
        if (!sycl_moe_build_sorted_pairs(q, d_selected, pair_count, n_total_expert,
                                         block_m, &scratch, &res,
                                         /*poison_for_test=*/true)) {
            return 0;
        }
        sycl_device_scratch_guard scratch_guard(q, scratch);
        if (res.tile_capacity > tile_capacity_cap) return 0;

        q.memcpy(counts_out, res.counts, (size_t)n_total_expert * sizeof(uint32_t));
        q.memcpy(offsets_out, res.offsets,
                 (size_t)(n_total_expert + 1u) * sizeof(uint32_t));
        q.memcpy(sorted_pairs_out, res.sorted_pairs,
                 (size_t)pair_count * sizeof(uint32_t));
        if (tile_total_out) {
            q.memcpy(tile_total_out, res.tile_total, sizeof(uint32_t));
        }
        if (tile_experts_out) {
            q.memcpy(tile_experts_out, res.tile_experts,
                     (size_t)res.tile_capacity * sizeof(uint32_t));
        }
        if (tile_starts_out) {
            q.memcpy(tile_starts_out, res.tile_starts,
                     (size_t)res.tile_capacity * sizeof(uint32_t));
        }
        q.wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_sort failed: %s\n", e.what());
        return 0;
    }
}
