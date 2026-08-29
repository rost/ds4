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

/* ---- Q8_K activation quantiser -------------------------------------
 *
 * cuda_block_q8_K, ds4_rocm.cu:79-83: 256-value superblock, float scale,
 * signed 8-bit codes, and 16 group sums of 16 codes each (bsums) used by
 * the Q4_K/Q2_K/IQ2_XXS dot-product helpers to fold the min-value term
 * without re-reading every code.  No half/f16 fields, unlike the weight
 * block types, so no bit-cast conversion is needed here. */
constexpr uint32_t kMoeQK = 256;  /* CUDA_QK_K, ds4_rocm.cu:47 */

struct sycl_block_q8_K {
    float   d;
    int8_t  qs[kMoeQK];
    int16_t bsums[kMoeQK / 16];
};

/* q8_K_quantize_kernel, moe.cuh:750-798.  Every launch site in
 * moe_launch.cuh uses block dim 256 (== kMoeQK) exactly, so the CUDA
 * kernel's "tid < CUDA_QK_K" guards are unconditionally true; this port
 * omits them for that reason, not by aiming for a smaller work-group.
 * Grid is (in_dim/256, n_rows): one work-group per (superblock, row).
 * Every work-item writes its abs_part/val_part slot unconditionally
 * before the barrier (spec section 6b): here that invariant holds
 * trivially, since the work-group width equals the block width exactly
 * and there is no strided remainder loop that could leave a work-item
 * idle, unlike the norm_rope reductions spec 6c discusses. */
static void sycl_moe_q8_k_quantize(sycl::queue &q, sycl_block_q8_K *out,
                                   const float *x, uint32_t in_dim,
                                   uint32_t n_rows) {
    const uint32_t xq_blocks = in_dim / kMoeQK;
    if (xq_blocks == 0u || n_rows == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<float, 1> abs_part(sycl::range<1>(kMoeQK), h);
         sycl::local_accessor<float, 1> val_part(sycl::range<1>(kMoeQK), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)xq_blocks * kMoeQK, n_rows),
                               sycl::range<2>(kMoeQK, 1)),
             [=](sycl::nd_item<2> it) {
                 const uint32_t b = (uint32_t)it.get_group(0);
                 const uint32_t row = (uint32_t)it.get_group(1);
                 const uint32_t tid = (uint32_t)it.get_local_id(0);
                 const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * kMoeQK;
                 sycl_block_q8_K *yb = out + (uint64_t)row * xq_blocks + b;

                 const float v = xr[tid];
                 abs_part[tid] = sycl::fabs(v);
                 val_part[tid] = v;
                 it.barrier(sycl::access::fence_space::local_space);

                 for (uint32_t stride = kMoeQK >> 1; stride > 0; stride >>= 1) {
                     if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
                         abs_part[tid] = abs_part[tid + stride];
                         val_part[tid] = val_part[tid + stride];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                 }

                 const float amax = abs_part[0];
                 if (amax == 0.0f) {
                     if (tid == 0) yb->d = 0.0f;
                     yb->qs[tid] = 0;
                     if (tid < kMoeQK / 16u) yb->bsums[tid] = 0;
                     return;
                 }

                 const float maxv = val_part[0];
                 const float iscale = -127.0f / maxv;
                 /* rint, not round: CUDA's lrintf (moe.cuh:783) rounds per
                  * the current FPU mode (round-to-nearest-even by
                  * default), which sycl::round's round-half-away-from-zero
                  * does not match at exact .5 ties. */
                 int qv = (int)sycl::rint(iscale * xr[tid]);
                 if (qv > 127) qv = 127;
                 if (qv < -128) qv = -128;
                 yb->qs[tid] = (int8_t)qv;
                 it.barrier(sycl::access::fence_space::global_and_local);

                 if (tid < kMoeQK / 16u) {
                     int sum = 0;
                     for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
                     yb->bsums[tid] = (int16_t)sum;
                 }
                 if (tid == 0) yb->d = 1.0f / iscale;
             });
     }).wait_and_throw();
}

/* ---- Sub-group reductions -------------------------------------------
 *
 * half_warp_sum_f32 (16-lane) and quarter_warp_sum_f32 (8-lane),
 * moe.cuh:732-749.  CUDA implements these as a MASKED partition of one
 * 32-wide warp (a shuffle with an explicit width parameter); Intel Arc
 * supports real 8/16/32-wide sub-groups, so this port uses an actual
 * reqd_sub_group_size(N) sub-group rather than emulating CUDA's masking
 * scheme, per the plan.  Within a genuine N-wide sub-group, no mask is
 * needed: sycl::sub_group's shuffle already scopes to the sub-group. */

template <int N>
static float sycl_moe_subgroup_sum(sycl::sub_group sg, float v) {
    for (int offset = N >> 1; offset > 0; offset >>= 1) {
        v += sycl::shift_group_left(sg, v, (uint32_t)offset);
    }
    return v;
}

/* ---- Sum / combine ---------------------------------------------------
 *
 * moe_sum_kernel / moe_sum_f16_kernel / moe_sum_f16x2_kernel,
 * moe.cuh:3877-3918.  Weighted-sum-across-selected-experts into the
 * final `out` tensor.  Accumulation order is `for e in 0..n_expert`,
 * i.e. router-SLOT order, matching ds4.c's layer_routed_moe_one_prealloc
 * (ds4.c:11075-11077) and NOT layer_routed_moe_batch's expert-id order
 * (ds4.c:11156-11160); see ds4-sycl-moe-reference.md section 4(a). */
static void sycl_moe_sum(sycl::queue &q, float *out, const float *down,
                         uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (n == 0u) return;
    q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> id) {
         uint64_t gid = id[0];
         uint32_t tok = (uint32_t)(gid / out_dim);
         uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
         float acc = 0.0f;
         for (uint32_t e = 0; e < n_expert; e++) {
             acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
         }
         out[gid] = acc;
     }).wait_and_throw();
}

static void sycl_moe_sum_f16(sycl::queue &q, float *out, const uint16_t *down_h,
                             uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (n == 0u) return;
    q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> id) {
         uint64_t gid = id[0];
         uint32_t tok = (uint32_t)(gid / out_dim);
         uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
         float acc = 0.0f;
         for (uint32_t e = 0; e < n_expert; e++) {
             acc += sycl_moe_f16_to_f32(down_h[((uint64_t)tok * n_expert + e) * out_dim + row]);
         }
         out[gid] = acc;
     }).wait_and_throw();
}

static void sycl_moe_sum_f16x2(sycl::queue &q, float *out, const uint16_t *down_h,
                               uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint32_t out_dim2 = out_dim >> 1u;
    const uint64_t n2 = (uint64_t)n_tokens * out_dim2;
    if (n2 == 0u) return;
    q.parallel_for(sycl::range<1>((size_t)n2), [=](sycl::id<1> id) {
         uint64_t gid = id[0];
         uint32_t tok = (uint32_t)(gid / out_dim2);
         uint32_t row = (uint32_t)((gid - (uint64_t)tok * out_dim2) << 1u);
         float acc0 = 0.0f, acc1 = 0.0f;
         for (uint32_t e = 0; e < n_expert; e++) {
             const uint64_t off = ((uint64_t)tok * n_expert + e) * out_dim + row;
             acc0 += sycl_moe_f16_to_f32(down_h[off]);
             acc1 += sycl_moe_f16_to_f32(down_h[off + 1u]);
         }
         const uint64_t out_off = (uint64_t)tok * out_dim + row;
         out[out_off] = acc0;
         out[out_off + 1u] = acc1;
     }).wait_and_throw();
}

/* Test-only: exercises sycl_moe_subgroup_sum<N> as a real N-wide
 * sub-group reduction (one sub-group per group of N input values, lane 0
 * writes the result), so a test can drive the 8-lane and 16-lane shapes
 * directly without needing a full Q4_K kernel launch.  Two separate
 * kernels because reqd_sub_group_size must be a compile-time constant. */
static void sycl_moe_test_subgroup_sum8(sycl::queue &q, const float *in,
                                        uint32_t n_groups, float *out) {
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<1>(sycl::range<1>((size_t)n_groups * 8u),
                               sycl::range<1>(8u)),
             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t gid = (uint32_t)it.get_group(0);
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 float v = in[gid * 8u + lane];
                 v = sycl_moe_subgroup_sum<8>(sg, v);
                 if (lane == 0u) out[gid] = v;
             });
     }).wait_and_throw();
}

static void sycl_moe_test_subgroup_sum16(sycl::queue &q, const float *in,
                                         uint32_t n_groups, float *out) {
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<1>(sycl::range<1>((size_t)n_groups * 16u),
                               sycl::range<1>(16u)),
             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t gid = (uint32_t)it.get_group(0);
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 float v = in[gid * 16u + lane];
                 v = sycl_moe_subgroup_sum<16>(sg, v);
                 if (lane == 0u) out[gid] = v;
             });
     }).wait_and_throw();
}

/* ---- Q4_K ------------------------------------------------------------
 *
 * cuda_block_q4_K, ds4_rocm.cu:72-77: 256-value superblock, half scale
 * and min, 12 bytes of sub-block scale/min codes, 128 bytes of packed
 * 4-bit codes.  Of the 9 Q4_K kernels and 4 device helpers inventoried
 * by ds4-sycl-moe-reference.md, this header ports 5 kernels and 3 helpers;
 * the other 4 are dead code
 * (two kernels gated by a compile-time-constant expert_tile_m that is
 * always 8, never taking their "else" branch; two more whose only
 * possible caller condition -- a sorted-pairs allocation that succeeds
 * while its co-allocated tile arrays fail -- is unreachable because both
 * come from the same single allocation). */

struct sycl_block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

/* dev_q4_K_get_scale_min, moe.cuh:250-262. */
static inline void sycl_dev_q4_k_get_scale_min(uint32_t j, const uint8_t *scales,
                                               uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

/* dev_dot_q4_32, moe.cuh:264-271: dots 32 packed 4-bit codes (shifted by
 * `shift` out of their byte) against 32 signed Q8_K codes.  ROCm's form
 * packs 4 bytes at a time through __dp4a; this is the unpacked
 * equivalent, which is exact-integer arithmetic throughout so there is
 * no associativity concern in grouping it differently. */
static inline int32_t sycl_dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 32u; i++) {
        sum += (int32_t)((qs[i] >> shift) & 0x0f) * (int32_t)q8[i];
    }
    return sum;
}

/* dev_dot_q4_K_q8_K_block, moe.cuh:274-287: one Q4_K weight block dotted
 * against one Q8_K activation block. */
static inline float sycl_dev_dot_q4_k_q8_k_block(const sycl_block_q4_K *x,
                                                 const sycl_block_q8_K *y) {
    const float xd = sycl_moe_f16_to_f32(x->d);
    const float xmin = sycl_moe_f16_to_f32(x->dmin);
    int32_t isum = 0, summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        sycl_dev_q4_k_get_scale_min(j, x->scales, &sc, &m);
        summs += (int32_t)m * (int32_t)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int32_t)sc * sycl_dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

/* dev_dot_q4_K_q8_K_block8, moe.cuh:588-617: the same dot product against
 * up to 8 activation blocks at once, accumulating into acc[8], used by
 * the expert-tile kernels which process 8 (token,slot) pairs per tile. */
/* yb selects which Q8_K chunk of each token's activation to dot against,
 * matching the weight block x.  The single-token helper above takes an
 * already-offset pointer; this one takes the base plus an index because
 * its callers stage the whole activation into local memory and then reuse
 * the same base for every block. */
static inline void sycl_dev_dot_q4_k_q8_k_block8(
        const sycl_block_q4_K *x, const sycl_block_q8_K *const ys[8],
        uint32_t yb, uint32_t n, float acc[8]) {
    const float xd = sycl_moe_f16_to_f32(x->d);
    const float xmin = sycl_moe_f16_to_f32(x->dmin);
    int32_t isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        sycl_dev_q4_k_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            const sycl_block_q8_K *y = ys[p] + yb;
            summs[p] += (int32_t)m * (int32_t)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
            isum[p] += (int32_t)sc * sycl_dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (!ys[p]) continue;
        const sycl_block_q8_K *y = ys[p] + yb;
        acc[p] += y->d * xd * (float)isum[p] - y->d * xmin * (float)summs[p];
    }
}

/* moe_gate_up_mid_decode_q4K_qwarp32_kernel, moe.cuh:2713-2764.  Despite
 * its "decode" name, ROCm reaches this for every n_tokens < 32 call
 * (both true single-token decode and small untiled batches), not just
 * n_tokens == 1: it is the "else" of "sorted_pairs" in the gate/up
 * dispatch (moe_launch.cuh:1470-1488), and sorted_pairs is only built
 * when n_tokens >= 32 for q4k_path (moe_launch.cuh:757).  8-lane
 * sub-groups: 32 groups per 256-work-item work-group, 4 row-groups of 32
 * rows each per pair. */
static void sycl_moe_q4k_gate_up_mid_decode(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const int32_t *selected, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert, uint32_t pair_count, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 127u) / 128u;
    if (row_blocks == 0u || pair_count == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, pair_count),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row_lane = (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t row_block = (uint32_t)it.get_group(0);
                 const uint32_t pair = (uint32_t)it.get_group(1);
                 const uint32_t tok = pair / n_expert;
                 const uint32_t slot = pair - tok * n_expert;
                 int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                 if (expert_i < 0) expert_i = 0;
                 const uint32_t expert = (uint32_t)expert_i;
                 const sycl_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
                 for (uint32_t rr = 0; rr < 4u; rr++) {
                     const uint32_t row = row_block * 128u + row_lane + rr * 32u;
                     if (row >= expert_mid_dim) continue;
                     const sycl_block_q4_K *gr = (const sycl_block_q4_K *)
                             (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     const sycl_block_q4_K *ur = (const sycl_block_q4_K *)
                             (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     float gate = 0.0f, up = 0.0f;
                     for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                         gate += sycl_dev_dot_q4_k_q8_k_block(gr + b, xqb + b);
                         up += sycl_dev_dot_q4_k_q8_k_block(ur + b, xqb + b);
                     }
                     gate = sycl_moe_subgroup_sum<8>(sg, gate);
                     up = sycl_moe_subgroup_sum<8>(sg, up);
                     if (lane == 0u) {
                         if (clamp > 1.0e-6f) {
                             if (gate > clamp) gate = clamp;
                             if (up > clamp) up = clamp;
                             if (up < -clamp) up = -clamp;
                         }
                         const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                         mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up *
                                        weights[(uint64_t)tok * n_expert + slot];
                     }
                 }
             });
     }).wait_and_throw();
}

/* moe_gate_up_mid_q4K_expert_tile8_row32_kernel, moe.cuh:2021-2098:
 * n_tokens >= 32 tiled batched path, block_m = 8 pairs per tile, 8-lane
 * sub-groups.  Staging Q8_K activation blocks into local memory when
 * xq_blocks <= 16 (matching ROCm's __shared__ sxq[8][16] cap) is what
 * every kernel using this cluster's sort output does; this is the only
 * one of the two tile kernels ported here (see the file header's
 * dead-code note for why the tile4 sibling is not). */
static void sycl_moe_q4k_gate_up_mid_tile8(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total, const uint32_t *tile_experts,
        const uint32_t *tile_starts, const float *weights, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t n_expert, uint32_t tile_capacity, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 31u) / 32u;
    if (row_blocks == 0u || tile_capacity == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 2> sxq(sycl::range<2>(8, 16), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, tile_capacity),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 const uint32_t tile = (uint32_t)it.get_group(1);
                 if (tile >= *tile_total) return;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t expert = tile_experts[tile];
                 const uint32_t count = counts[expert];
                 const uint32_t local_start = tile_starts[tile];

                 uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 const sycl_block_q8_K *xqb[8] = {nullptr, nullptr, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr};
                 uint32_t np = 0;
                 for (; np < 8u; np++) {
                     const uint32_t local_pair = local_start + np;
                     if (local_pair >= count) break;
                     pair[np] = sorted_pairs[offsets[expert] + local_pair];
                     tok[np] = pair[np] / n_expert;
                     slot[np] = pair[np] - tok[np] * n_expert;
                     xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
                 }
                 if (xq_blocks <= 16u) {
                     const uint32_t lid = (uint32_t)it.get_local_id(0);
                     for (uint32_t i = lid; i < np * xq_blocks; i += 256u) {
                         const uint32_t p = i / xq_blocks;
                         const uint32_t b = i - p * xq_blocks;
                         sxq[p][b] = xqb[p][b];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                     for (uint32_t p = 0; p < np; p++) xqb[p] = &sxq[p][0];
                 }
                 if (row >= expert_mid_dim) return;
                 const sycl_block_q4_K *gr = (const sycl_block_q4_K *)
                         (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 const sycl_block_q4_K *ur = (const sycl_block_q4_K *)
                         (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 float gate[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 float up[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                     sycl_dev_dot_q4_k_q8_k_block8(gr + b, xqb, b, np, gate);
                     sycl_dev_dot_q4_k_q8_k_block8(ur + b, xqb, b, np, up);
                 }
                 for (uint32_t p = 0; p < np; p++) {
                     gate[p] = sycl_moe_subgroup_sum<8>(sg, gate[p]);
                     up[p] = sycl_moe_subgroup_sum<8>(sg, up[p]);
                     if (lane == 0u) {
                         float g = gate[p], u = up[p];
                         if (clamp > 1.0e-6f) {
                             if (g > clamp) g = clamp;
                             if (u > clamp) u = clamp;
                             if (u < -clamp) u = -clamp;
                         }
                         const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                         mid_out[off] = (g / (1.0f + sycl::exp(-g))) * u *
                                        weights[(uint64_t)tok[p] * n_expert + slot[p]];
                     }
                 }
             });
     }).wait_and_throw();
}

/* moe_down_q4K_sum6_qwarp32_kernel, moe.cuh:3075-3099: n_tokens == 1
 * only, writes the final combined output row directly (no separate
 * moe_sum_kernel combine pass; this is the "direct_down_sum6" fast path,
 * so called because it sums over up to DS4_ROCM_N_EXPERT_USED=6 slots in
 * one kernel instead of writing per-slot rows for a later reduction). */
static void sycl_moe_q4k_down_sum6(
        sycl::queue &q, float *out, const char *down_base, const sycl_block_q8_K *midq,
        const int32_t *selected, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<1>(sycl::range<1>((size_t)row_blocks * 256u),
                               sycl::range<1>(256u)),
             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 if (row >= out_dim) return;
                 float total = 0.0f;
                 for (uint32_t slot = 0; slot < 8u /* DS4_ROCM_N_EXPERT_USED */; slot++) {
                     if (slot >= n_expert) continue;
                     int32_t expert_i = selected[slot];
                     if (expert_i < 0) expert_i = 0;
                     const sycl_block_q4_K *wr = (const sycl_block_q4_K *)
                             (down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
                     const sycl_block_q8_K *xqb = midq + (uint64_t)slot * midq_blocks;
                     float acc = 0.0f;
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         acc += sycl_dev_dot_q4_k_q8_k_block(wr + b, xqb + b);
                     }
                     acc = sycl_moe_subgroup_sum<8>(sg, acc);
                     if (lane == 0u) total += acc;
                 }
                 if (lane == 0u) out[row] = total;
             });
     }).wait_and_throw();
}

/* moe_down_q4K_qwarp32_kernel, moe.cuh:3162-3187: 1 < n_tokens < 32,
 * untiled, one (token,slot) pair per blockIdx.y; writes per-pair rows
 * into the down scratch tensor for a later moe_sum_kernel combine. */
static void sycl_moe_q4k_down_untiled(
        sycl::queue &q, float *down_out, const char *down_base, const sycl_block_q8_K *midq,
        const int32_t *selected, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert, uint32_t pair_count) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u || pair_count == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, pair_count),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t pair = (uint32_t)it.get_group(1);
                 if (row >= out_dim) return;
                 const uint32_t tok = pair / n_expert;
                 const uint32_t slot = pair - tok * n_expert;
                 int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                 if (expert_i < 0) expert_i = 0;
                 const sycl_block_q4_K *wr = (const sycl_block_q4_K *)
                         (down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
                 const sycl_block_q8_K *xqb = midq + (uint64_t)pair * midq_blocks;
                 float acc = 0.0f;
                 for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                     acc += sycl_dev_dot_q4_k_q8_k_block(wr + b, xqb + b);
                 }
                 acc = sycl_moe_subgroup_sum<8>(sg, acc);
                 if (lane == 0u) down_out[(uint64_t)pair * out_dim + row] = acc;
             });
     }).wait_and_throw();
}

/* moe_down_q4K_expert_tile8_row32_kernel, moe.cuh:3276-3330: n_tokens >=
 * 32 tiled path.  ROCm's atomic_out flag (accumulate straight into `out`
 * for very large batches, n_tokens >= 128) is deliberately not ported:
 * it is a bitwise-nondeterministic performance optimisation (concurrent
 * atomic float adds have no fixed order), not a distinct algorithm, and
 * this port always writes per-pair rows into the down scratch tensor
 * for a moe_sum_kernel combine instead -- a strict correctness superset
 * at the cost of one extra kernel launch on batches this large. */
static void sycl_moe_q4k_down_tile8(
        sycl::queue &q, float *down_out, const char *down_base, const sycl_block_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t tile_capacity) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u || tile_capacity == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 2> sxq(sycl::range<2>(8, 8), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, tile_capacity),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 const uint32_t tile = (uint32_t)it.get_group(1);
                 if (tile >= *tile_total) return;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t expert = tile_experts[tile];
                 const uint32_t local_start = tile_starts[tile];

                 uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 const sycl_block_q8_K *xqb[8] = {nullptr, nullptr, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr};
                 uint32_t np = 0;
                 for (; np < 8u; np++) {
                     const uint32_t local_pair = local_start + np;
                     if (local_pair >= counts[expert]) break;
                     pair[np] = sorted_pairs[offsets[expert] + local_pair];
                     xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
                 }
                 if (midq_blocks <= 8u) {
                     const uint32_t lid = (uint32_t)it.get_local_id(0);
                     for (uint32_t i = lid; i < np * midq_blocks; i += 256u) {
                         const uint32_t p = i / midq_blocks;
                         const uint32_t b = i - p * midq_blocks;
                         sxq[p][b] = xqb[p][b];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                     for (uint32_t p = 0; p < np; p++) xqb[p] = &sxq[p][0];
                 }
                 if (row >= out_dim) return;
                 const sycl_block_q4_K *wr = (const sycl_block_q4_K *)
                         (down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
                 float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                     sycl_dev_dot_q4_k_q8_k_block8(wr + b, xqb, b, np, acc);
                 }
                 for (uint32_t p = 0; p < np; p++) {
                     acc[p] = sycl_moe_subgroup_sum<8>(sg, acc[p]);
                     if (lane == 0u) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                 }
             });
     }).wait_and_throw();
}

/* ---- IQ2_XXS ----------------------------------------------------------
 *
 * cuda_block_iq2_xxs, ds4_rocm.cu:85-88: 256-value superblock, half scale,
 * 32 packed u16 "grid" codes.  This is DS4 Flash's shipped gate/up format
 * (spec section 6e); the kernels that consume it, moe_gate_up_mid_kernel
 * and its siblings, carry NO format suffix in their names despite
 * hardcoding this cast (verified at moe.cuh:984, inside
 * moe_gate_up_mid_qwarp32_kernel) -- a trap for anyone assuming kernel
 * names describe their formats.  The grid/sign tables below are copied verbatim from
 * ds4_iq2_tables_cuda.inc, which rocm/ds4_rocm.cu:104 includes unchanged;
 * ROCm and CUDA share one physical table, so there is only one table to
 * port. */

struct sycl_block_iq2_xxs {
    uint16_t d;
    uint16_t qs[32];
};

/* cuda_ksigns_iq2xs, ds4_iq2_tables_cuda.inc:1-9: 128 entries, each an
 * 8-bit sign pattern with bit 7 not yet parity-corrected (see
 * sycl_iq2_fix_sign_byte below, mirroring dev_unpack_iq2_signs,
 * moe.cuh:19-23). */
static constexpr uint8_t kIq2Signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/* cuda_iq2xxs_grid, ds4_iq2_tables_cuda.inc:12-77: 256 entries, each 8
 * packed magnitude bytes (values in {0x08, 0x19, 0x2b}) for one group of 8
 * elements. */
static constexpr uint64_t kIq2Grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static inline uint32_t sycl_popcount_u32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0f0f0f0fu;
    return (v * 0x01010101u) >> 24;
}

/* dev_unpack_iq2_signs, moe.cuh:19-23: the ksigns table stores 7 sign bits
 * plus a not-yet-corrected bit 7; this recovers the true 8-bit pattern by
 * forcing overall byte parity, then applies it as one sign per grid byte.
 * Written as a direct scalar loop rather than porting the dp4a/vsub4/
 * vcmpne4 SIMD-in-a-register trick literally: verified by hand that
 * "(byte XOR 0xFF) - 0xFF" is exactly two's-complement negation, which is
 * what CUDA's per-byte vsub4 does when the sign mask byte is 0xFF, so this
 * scalar form and the ROCm SIMD form compute the identical integer per
 * byte. */
static inline int32_t sycl_iq2_grid_dot8(uint8_t grid_idx, uint8_t sign_idx, const int8_t *q8) {
    const uint64_t grid = kIq2Grid[grid_idx];
    const uint8_t raw = kIq2Signs[sign_idx & 127u];
    const uint32_t parity = sycl_popcount_u32((uint32_t)raw) & 1u;
    const uint8_t s = (uint8_t)(raw ^ (uint8_t)(parity << 7));
    int32_t sum = 0;
    for (uint32_t b = 0; b < 8u; b++) {
        int32_t v = (int32_t)(int8_t)((grid >> (b * 8u)) & 0xffu);
        if ((s >> b) & 1u) v = -v;
        sum += v * (int32_t)q8[b];
    }
    return sum;
}

static inline int32_t sycl_iq2_pair_dot16(uint8_t grid0, uint8_t sign0, uint8_t grid1,
                                          uint8_t sign1, const int8_t *q8) {
    return sycl_iq2_grid_dot8(grid0, sign0, q8) + sycl_iq2_grid_dot8(grid1, sign1, q8 + 8);
}

/* dev_dot_iq2_xxs_q8_K_block, moe.cuh:101-123: one IQ2_XXS weight block
 * dotted against one Q8_K activation block. */
static inline float sycl_dev_dot_iq2_xxs_q8_k_block(const sycl_block_iq2_xxs *x,
                                                    const sycl_block_q8_K *y) {
    const float d = sycl_moe_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < (int)(kMoeQK / 32u); ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = sycl_iq2_pair_dot16(a0, (uint8_t)((aux1 >> 0) & 127u), a1,
                                           (uint8_t)((aux1 >> 7) & 127u), q8);
        q8 += 16;
        sumi += sycl_iq2_pair_dot16(a2, (uint8_t)((aux1 >> 14) & 127u), a3,
                                    (uint8_t)((aux1 >> 21) & 127u), q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

/* Scalar re-derivation of dev_dot_iq2_xxs_q8_K_block8 (moe.cuh:211-248,
 * DS4_ROCM_UNUSED there because CUDA's own tile kernel uses a LUT-staged
 * variant instead): same aux0/aux1/ls decode as the single-block form
 * above, batched across up to 8 activation blocks at once for the
 * expert-tile kernels.  Not a port of dead code, since nothing here is
 * reused verbatim from the unused CUDA kernel; only the arithmetic shape
 * is shared.
 *
 * yb selects which Q8_K chunk of each token's activation to dot against,
 * matching the weight block x, mirroring sycl_dev_dot_q4_k_q8_k_block8's
 * yb parameter (fixed in 35852c0 for exactly this reason). Before this
 * fix, every ys[p] read here ignored which chunk the caller's loop was
 * on and always read chunk 0's block, so at any expert_in_dim wider than
 * 256 (DS4 Flash's real model_dim=4096 gives 16 chunks) this kernel
 * dotted every weight block after the first against the wrong activation
 * data -- a live bug, caught by widening tests/test_sycl_moe.c's
 * RM_EXPERT_IN_DIM past one Q8_K block (see the report). */
static inline void sycl_dev_dot_iq2_xxs_q8_k_block8(
        const sycl_block_iq2_xxs *x, const sycl_block_q8_K *const ys[8],
        uint32_t yb, uint32_t n, float acc[8]) {
    const float xd = sycl_moe_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int ib32 = 0; ib32 < (int)(kMoeQK / 32u); ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        const uint8_t s0i = (uint8_t)((aux1 >> 0) & 127u);
        const uint8_t s1i = (uint8_t)((aux1 >> 7) & 127u);
        const uint8_t s2i = (uint8_t)((aux1 >> 14) & 127u);
        const uint8_t s3i = (uint8_t)((aux1 >> 21) & 127u);
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            const int8_t *q8 = (ys[p] + yb)->qs + ib32 * 32;
            int32_t sumi = sycl_iq2_pair_dot16(a0, s0i, a1, s1i, q8) +
                           sycl_iq2_pair_dot16(a2, s2i, a3, s3i, q8 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (!ys[p]) continue;
        acc[p] += 0.125f * xd * (ys[p] + yb)->d * (float)bsum[p];
    }
}

/* moe_gate_up_mid_qwarp32_kernel, moe.cuh:955-1006 (the untagged family;
 * hardcodes cuda_block_iq2_xxs, verified at moe.cuh:984).  ROCm's decode
 * dispatch prefers moe_gate_up_mid_decode_lut_qwarp32_kernel
 * (moe.cuh:1121-1185) whenever xq_blocks <= 16, which is DS4 Flash's real
 * shape (MODEL_DIM=4096 => xq_blocks=16).  Read side by side, the LUT
 * kernel's dev_dot_iq2_xxs_q8_K_block_lut and this kernel's
 * dev_dot_iq2_xxs_q8_K_block decode the identical aux0/aux1/ls fields,
 * apply the identical sign unpack, and differ only in whether the
 * grid/signs tables are staged into a work-group's local memory first --
 * a shared-memory bandwidth optimisation, not a different computation.
 * This port implements only the plain (non-LUT) form and reads the tables
 * directly, which is mathematically identical and needs no local-memory
 * staging of the tables themselves.  write_gate_up is always 0 at every
 * live call site (moe_launch.cuh:761's write_gate_up constant), so
 * gate_out/up_out are never written here either, matching the Q4_K decode
 * kernel's precedent above. */
static void sycl_moe_iq2_gate_up_mid_decode(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const int32_t *selected, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert, uint32_t pair_count, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 127u) / 128u;
    if (row_blocks == 0u || pair_count == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, pair_count),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row_lane = (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t row_block = (uint32_t)it.get_group(0);
                 const uint32_t pair = (uint32_t)it.get_group(1);
                 const uint32_t tok = pair / n_expert;
                 const uint32_t slot = pair - tok * n_expert;
                 int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                 if (expert_i < 0) expert_i = 0;
                 const uint32_t expert = (uint32_t)expert_i;
                 const sycl_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
                 for (uint32_t rr = 0; rr < 4u; rr++) {
                     const uint32_t row = row_block * 128u + row_lane + rr * 32u;
                     if (row >= expert_mid_dim) continue;
                     const sycl_block_iq2_xxs *gr = (const sycl_block_iq2_xxs *)
                             (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     const sycl_block_iq2_xxs *ur = (const sycl_block_iq2_xxs *)
                             (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     float gate = 0.0f, up = 0.0f;
                     for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                         gate += sycl_dev_dot_iq2_xxs_q8_k_block(gr + b, xqb + b);
                         up += sycl_dev_dot_iq2_xxs_q8_k_block(ur + b, xqb + b);
                     }
                     gate = sycl_moe_subgroup_sum<8>(sg, gate);
                     up = sycl_moe_subgroup_sum<8>(sg, up);
                     if (lane == 0u) {
                         if (clamp > 1.0e-6f) {
                             if (gate > clamp) gate = clamp;
                             if (up > clamp) up = clamp;
                             if (up < -clamp) up = -clamp;
                         }
                         const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                         mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up *
                                        weights[(uint64_t)tok * n_expert + slot];
                     }
                 }
             });
     }).wait_and_throw();
}

/* moe_gate_up_mid_expert_tile8_row32_kernel, moe.cuh:1545-1638: n_tokens >
 * 1 tiled batched path (iq2_path has NO ">= 32" threshold the way q4k_path
 * does -- verified directly against moe_launch.cuh:754-758:
 * use_sorted_pairs collapses to plain "n_tokens > 1u" once q4k_path is
 * false, so ROCm takes this tiled path for every batched call, not only
 * large ones).
 *
 * Mandatory correctness fix, not a port: ROCm's kernel takes a max_count
 * parameter (moe.cuh:1564,1573) fed from iq2_gate_scalar_max
 * (moe_launch.cuh:1139), which skips ("excludes") any tile whose expert
 * has count >= max_count on the assumption that
 * moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel will compute those experts
 * instead.  That WMMA launch is `#if HIP`-guarded (moe_launch.cuh:1552),
 * but use_iq2_gate_wmma itself, and the loop that populates
 * iq2_gate_hot_count from it, are NOT guarded (moe_launch.cuh:1113-1139) --
 * verified directly, not inferred from the launch site alone. On this SYCL
 * backend, which never defines the HIP macros, adopting max_count as
 * written would silently skip every "hot" (>=8-pair) expert with nothing
 * ever computing it. This port does not implement the WMMA kernels and
 * does NOT implement the exclusion either: every tile is always
 * unconditionally computed, matching what ROCm's own non-HIP builds do at
 * the two down-side guard sites (moe_launch.cuh:240-246, :2023-2030),
 * which force their own equivalent scalar_max to 0. */
static void sycl_moe_iq2_gate_up_mid_tile8(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total, const uint32_t *tile_experts,
        const uint32_t *tile_starts, const float *weights, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t n_expert, uint32_t tile_capacity, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 31u) / 32u;
    if (row_blocks == 0u || tile_capacity == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 2> sxq(sycl::range<2>(8, 16), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, tile_capacity),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 const uint32_t tile = (uint32_t)it.get_group(1);
                 if (tile >= *tile_total) return;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t expert = tile_experts[tile];
                 const uint32_t count = counts[expert];
                 const uint32_t local_start = tile_starts[tile];

                 uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 const sycl_block_q8_K *xqb[8] = {nullptr, nullptr, nullptr, nullptr,
                                                  nullptr, nullptr, nullptr, nullptr};
                 uint32_t np = 0;
                 for (; np < 8u; np++) {
                     const uint32_t local_pair = local_start + np;
                     if (local_pair >= count) break;
                     pair[np] = sorted_pairs[offsets[expert] + local_pair];
                     tok[np] = pair[np] / n_expert;
                     slot[np] = pair[np] - tok[np] * n_expert;
                     xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
                 }
                 if (xq_blocks <= 16u) {
                     const uint32_t lid = (uint32_t)it.get_local_id(0);
                     for (uint32_t i = lid; i < np * xq_blocks; i += 256u) {
                         const uint32_t p = i / xq_blocks;
                         const uint32_t b = i - p * xq_blocks;
                         sxq[p][b] = xqb[p][b];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                     for (uint32_t p = 0; p < np; p++) xqb[p] = &sxq[p][0];
                 }
                 if (row >= expert_mid_dim) return;
                 const sycl_block_iq2_xxs *gr = (const sycl_block_iq2_xxs *)
                         (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 const sycl_block_iq2_xxs *ur = (const sycl_block_iq2_xxs *)
                         (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 float gate[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 float up[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                     sycl_dev_dot_iq2_xxs_q8_k_block8(gr + b, xqb, b, np, gate);
                     sycl_dev_dot_iq2_xxs_q8_k_block8(ur + b, xqb, b, np, up);
                 }
                 for (uint32_t p = 0; p < np; p++) {
                     gate[p] = sycl_moe_subgroup_sum<8>(sg, gate[p]);
                     up[p] = sycl_moe_subgroup_sum<8>(sg, up[p]);
                     if (lane == 0u) {
                         float g = gate[p], u = up[p];
                         if (clamp > 1.0e-6f) {
                             if (g > clamp) g = clamp;
                             if (u > clamp) u = clamp;
                             if (u < -clamp) u = -clamp;
                         }
                         const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                         mid_out[off] = (g / (1.0f + sycl::exp(-g))) * u *
                                        weights[(uint64_t)tok[p] * n_expert + slot[p]];
                     }
                 }
             });
     }).wait_and_throw();
}

/* ---- Q2_K --------------------------------------------------------------
 *
 * cuda_block_q2_K, ds4_rocm.cu:65-70: 256-value superblock, 16 bytes of
 * packed (scale, min) nibble pairs (one byte per 16-value sub-block), 64
 * bytes of 2-bit codes, half scale and min.  Two independent uses in this
 * plan: the down-projection half of iq2_path's production path (the
 * untagged "down" family, moe.cuh:2603-3876, hardcodes this cast,
 * verified at moe.cuh:2705) and the standalone q2k_path
 * (gate_type==10 && down_type==10).
 *
 * Not ported, and why, consolidated here rather than scattered:
 *
 * - The untagged down family's other kernels (moe_down_kernel/_warp8/
 *   _hwarp16, all DS4_ROCM_UNUSED; moe_down_qwarp32_kernel; moe_down_
 *   sorted_qwarp32_kernel; moe_down_expert_tile4/8/16_row32/row2048/
 *   rowspan; moe_down_sorted_p2_qwarp32_kernel) are dead by data-flow
 *   trace, not by choice. moe_launch.cuh:746 already excludes q2k_path
 *   from ever reaching them. For every path that does reach that shared
 *   block (q4k_path, iq2_path, iq2_iq2_path, mxfp4_path):
 *   iq2_path's decode always takes use_direct_down_sum6 (moe_launch.cuh
 *   :775-777, n_tokens==1 with n_expert<=8 always true post build_plan);
 *   iq2_path's batch always takes use_iq2_q2_float_down (moe_launch.cuh
 *   :1627-1630, true whenever n_tokens>1 since sorted_pairs is always
 *   built for this path -- see the iq2 dispatcher comment above); and
 *   iq2_iq2_path always takes direct_iq2_down_done (moe_launch.cuh
 *   :1637). None of the three ever falls through to the generic
 *   sorted/tiled cascade at moe_launch.cuh:1691-1896 that would call
 *   these kernels, and the earlier-established "two *_sorted_qwarp32_
 *   kernels are unreachable by construction" finding (use_expert_tiles
 *   == use_sorted_pairs always, moe_launch.cuh:759) applies here too.
 * - The four WMMA kernels (moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel,
 *   moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel, moe_down_q2K_hotlist_
 *   wmma_kernel, moe_down_q2K_hotlist_wmma_n2_kernel, moe.cuh:4762-5284)
 *   are not ported at all: they are `#if HIP`-guarded and provably dead
 *   on any non-HIP build, and joint_matrix translation is deferred to a
 *   later tuning pass per the plan. The non-`_n2` moe_down_q2K_hotlist_
 *   wmma_kernel is additionally dead even on HIP builds
 *   (routed_moe_q2_float_down_launch's `const int no_n2 = 0;` at
 *   moe_launch.cuh:303 makes its branch unreachable). What is NOT
 *   skipped along with the WMMA bodies is the "hot expert" exclusion
 *   logic that feeds them -- see sycl_moe_iq2_gate_up_mid_tile8's
 *   comment above for the one site (moe_launch.cuh:1113-1139) where that
 *   exclusion is not `#if HIP`-guarded and adopting it would silently
 *   drop high-traffic experts.
 * - q2k_path's own raw-float kernels (moe_gate_up_mid_q2K_rows_rpb1_w32/
 *   rows_w32/expert_batch_sharedx, moe_down_q2K_sum_rows_w32/
 *   expert_batch_sharedmid, and by extension routed_moe_q2_float_down_
 *   launch's reuse of the sharedmid kernel for iq2_path's batch down) are
 *   a deliberate divergence, not a trace finding: see the q2k dispatcher
 *   comment below for why the Q8_K-quantised kernels are used instead. */

struct sycl_block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

/* dev_dot_q2_16, moe.cuh:36-44: dots 16 packed 2-bit codes (shifted by
 * `shift`) against 16 signed Q8_K codes. */
static inline int32_t sycl_dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16u; i++) {
        sum += (int32_t)((q2[i] >> shift) & 0x03) * (int32_t)q8[i];
    }
    return sum;
}

/* dev_dot_q2_K_q8_K_block, moe.cuh:622-645. */
static inline float sycl_dev_dot_q2_k_q8_k_block(const sycl_block_q2_K *x, const sycl_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int32_t summs = 0;
    for (int j = 0; j < 16; j++) summs += (int32_t)y->bsums[j] * (int32_t)(sc[j] >> 4);
    const float dall = y->d * sycl_moe_f16_to_f32(x->d);
    const float dmin = y->d * sycl_moe_f16_to_f32(x->dmin);
    int32_t isum = 0;
    int is = 0;
    for (int k = 0; k < (int)(kMoeQK / 128u); k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * sycl_dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * sycl_dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

/* dev_dot_q2_K_q8_K_block8, moe.cuh:688-730: the same dot product against
 * up to 8 activation blocks at once.  Unreachable by any format this
 * plan implements or stubs (see the Q2_K "not ported, and why" note
 * above sycl_block_q2_K's definition: every down path that could reach
 * a block8-style tiled Q2_K kernel goes through sycl_moe_q2k_down_direct
 * instead, which is untiled and indexes chunks with a plain `+ b`), so
 * this has no live caller.  It carried the identical chunk-index defect
 * sycl_dev_dot_iq2_xxs_q8_k_block8 had until this same change (see that
 * function's comment): every ys[p] read here ignored which chunk the
 * caller was on. Fixed with the same yb parameter for consistency and so
 * it is not a landmine if a future tiled Q2_K kernel calls it. */
static inline void sycl_dev_dot_q2_k_q8_k_block8(
        const sycl_block_q2_K *x, const sycl_block_q8_K *const ys[8],
        uint32_t yb, uint32_t n, float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = sycl_moe_f16_to_f32(x->d);
    const float xmin = sycl_moe_f16_to_f32(x->dmin);
    int32_t isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        if (!ys[p]) continue;
        for (int j = 0; j < 16; j++) summs[p] += (int32_t)(ys[p] + yb)->bsums[j] * (int32_t)(sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        if (!ys[p]) continue;
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = (ys[p] + yb)->qs;
        int is = 0;
        for (int k = 0; k < (int)(kMoeQK / 128u); k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * sycl_dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * sycl_dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (!ys[p]) continue;
        const float yd = (ys[p] + yb)->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

/* moe_down_sum6_qwarp32_kernel, moe.cuh:2909-2936, generalised from
 * single-token (selected[slot], out[row]) to n_tokens tokens
 * (selected[tok*n_expert+slot], out[tok*out_dim+row]).  For n_tokens == 1
 * this computes exactly what the ROCm kernel computes (tok is always 0,
 * so selected[0*n_expert+slot] == selected[slot]).  Traced by data flow
 * (moe_launch.cuh:746 excludes q2k_path from the whole shared block that
 * would otherwise reach the untagged down family's other kernels;
 * moe_launch.cuh:1627-1898 shows iq2_path's batched down always takes
 * use_iq2_q2_float_down instead, so every OTHER kernel in the untagged
 * down family besides this one is unreachable for any format implemented
 * here) -- this one
 * kernel, generalised over the token dimension, is what the decode path,
 * iq2_path's batched down, and q2k_path (both regimes)
 * all need, so it is written once here rather than three times. Reusing
 * it for iq2_path's batch is a deliberate divergence from ROCm's own
 * choice there (routed_moe_q2_float_down_launch's raw-float
 * moe_down_q2K_expert_batch_sharedmid_kernel, which skips Q8_K-quantising
 * mid) -- see the plan report for why: the mandated CPU oracle
 * (matvec_q2_k_experts_accum_prequant, ds4.c:8297) always dots against a
 * Q8_K-prequantised activation, and matching that at tight tolerance
 * takes priority over matching ROCm's specific batched-kernel choice. */
static void sycl_moe_q2k_down_direct(
        sycl::queue &q, float *out, const char *down_base, const sycl_block_q8_K *midq,
        const int32_t *selected, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u || n_tokens == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, n_tokens),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t tok = (uint32_t)it.get_group(1);
                 if (row >= out_dim) return;
                 float total = 0.0f;
                 for (uint32_t slot = 0; slot < 8u /* DS4_ROCM_N_EXPERT_USED */; slot++) {
                     if (slot >= n_expert) continue;
                     int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                     if (expert_i < 0) expert_i = 0;
                     const sycl_block_q2_K *wr = (const sycl_block_q2_K *)
                             (down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
                     const sycl_block_q8_K *xqb = midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
                     float acc = 0.0f;
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         acc += sycl_dev_dot_q2_k_q8_k_block(wr + b, xqb + b);
                     }
                     acc = sycl_moe_subgroup_sum<8>(sg, acc);
                     if (lane == 0u) total += acc;
                 }
                 if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
             });
     }).wait_and_throw();
}

/* moe_down_iq2_sum_qwarp32_batch_kernel, moe.cuh:3001-3036: iq2_iq2_path's
 * down (down_type == 16, so this is genuinely IQ2_XXS-cast, unlike every
 * other down kernel in this file).  Already per-token in ROCm (blockIdx.y
 * = tok), so this is a faithful port, not a generalisation like the
 * kernel above. */
static void sycl_moe_iq2_down_direct(
        sycl::queue &q, float *out, const char *down_base, const sycl_block_q8_K *midq,
        const int32_t *selected, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u || n_tokens == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, n_tokens),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t tok = (uint32_t)it.get_group(1);
                 if (row >= out_dim) return;
                 float total = 0.0f;
                 for (uint32_t slot = 0; slot < 8u /* DS4_ROCM_N_EXPERT_USED */; slot++) {
                     if (slot >= n_expert) continue;
                     int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                     if (expert_i < 0) expert_i = 0;
                     const sycl_block_iq2_xxs *wr = (const sycl_block_iq2_xxs *)
                             (down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
                     const sycl_block_q8_K *xqb = midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
                     float acc = 0.0f;
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         acc += sycl_dev_dot_iq2_xxs_q8_k_block(wr + b, xqb + b);
                     }
                     acc = sycl_moe_subgroup_sum<8>(sg, acc);
                     if (lane == 0u) total += acc;
                 }
                 if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
             });
     }).wait_and_throw();
}

/* moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel, moe.cuh:2849-2907:
 * q2k_path's Q8_K-quantised-activation gate/up kernel, 16-lane sub-groups.
 * ROCm reaches this only via q8k_gateup (moe_launch.cuh:2255-2256,
 * "!g_quality_mode && n_tokens==1u"), and reaches a raw-float
 * dequant-and-dot kernel (moe_gate_up_mid_q2K_rows_rpb1_w32/rows_w32,
 * moe_gate_up_mid_q2K_expert_batch_sharedx) for every other n_tokens.
 * This port takes the opposite choice deliberately: it uses THIS kernel,
 * generalised over no code change at all (pair = blockIdx.y already
 * covers any pair_count), for every n_tokens, because the mandated CPU
 * oracle (matvec_q2_k_experts_mid_prequant, ds4.c:8377, dispatched from
 * layer_routed_moe_one_prealloc via matvec_experts_mid_prequant) always
 * dots against a Q8_K-prequantised x -- the raw-float kernels compute a
 * genuinely different (unquantised-activation) value that cannot agree at
 * tight tolerance regardless of how faithfully they are ported.
 *
 * Three consequences a future reader needs before revisiting this, because
 * they are easy to miss from the reasoning above:
 *
 * 1. This is pervasive, not a corner case.  ROCm reaches a raw-float down
 *    kernel for essentially EVERY batched call on both iq2_path and
 *    q2k_path: iq2_path via use_iq2_q2_float_down
 *    (moe_launch.cuh:1627-1631), and q2k_path because its own Q8_K down
 *    path is gated on "n_tokens == 1u && !g_quality_mode"
 *    (moe_launch.cuh:2247-2576), so any n_tokens > 1 falls through to
 *    moe_down_q2K_sum_rows_w32_kernel.  This backend's production-path
 *    numerics therefore differ from ROCm's for all batched Q2_K and
 *    IQ2_XXS MoE work, permanently.
 * 2. "Matches the oracle" and "most accurate" point in DIFFERENT
 *    directions here.  Raw-float activations carry no quantisation loss;
 *    the Q8_K-prequantised route chosen here does.  Oracle fidelity was
 *    chosen deliberately over accuracy, because an un-oracled numerical
 *    path cannot be tested on this hardware at all.
 * 3. Revisiting it means either changing the CPU oracle in ds4.c, which
 *    is shared with every other backend, or accepting a kernel with no
 *    test that can discriminate it.  Neither is cheap.  If output quality
 *    ever looks wrong on batched prefill, start here. */
static void sycl_moe_q2k_gate_up_mid_decode(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const int32_t *selected, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert, uint32_t pair_count, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 255u) / 256u;
    if (row_blocks == 0u || pair_count == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 1> sxq(sycl::range<1>(16), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, pair_count),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row_lane = (uint32_t)(it.get_local_id(0) >> 4);
                 const uint32_t row_block = (uint32_t)it.get_group(0);
                 const uint32_t pair = (uint32_t)it.get_group(1);
                 const uint32_t tok = pair / n_expert;
                 const uint32_t slot = pair - tok * n_expert;
                 int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                 if (expert_i < 0) expert_i = 0;
                 const uint32_t expert = (uint32_t)expert_i;
                 const sycl_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
                 const uint32_t lid = (uint32_t)it.get_local_id(0);
                 if (xq_blocks <= 16u) {
                     for (uint32_t i = lid; i < xq_blocks; i += 256u) sxq[i] = xqb[i];
                     it.barrier(sycl::access::fence_space::local_space);
                     xqb = &sxq[0];
                 }
                 for (uint32_t rr = 0; rr < 16u; rr++) {
                     const uint32_t row = row_block * 256u + row_lane + rr * 16u;
                     if (row >= expert_mid_dim) continue;
                     const sycl_block_q2_K *gr = (const sycl_block_q2_K *)
                             (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     const sycl_block_q2_K *ur = (const sycl_block_q2_K *)
                             (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                     float gate = 0.0f, up = 0.0f;
                     for (uint32_t b = lane; b < xq_blocks; b += 16u) {
                         gate += sycl_dev_dot_q2_k_q8_k_block(gr + b, xqb + b);
                         up += sycl_dev_dot_q2_k_q8_k_block(ur + b, xqb + b);
                     }
                     gate = sycl_moe_subgroup_sum<16>(sg, gate);
                     up = sycl_moe_subgroup_sum<16>(sg, up);
                     if (lane == 0u) {
                         if (clamp > 1.0e-6f) {
                             if (gate > clamp) gate = clamp;
                             if (up > clamp) up = clamp;
                             if (up < -clamp) up = -clamp;
                         }
                         const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                         mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up *
                                        weights[(uint64_t)tok * n_expert + slot];
                     }
                 }
             });
     }).wait_and_throw();
}

/* ---- MXFP4 -------------------------------------------------------------
 *
 * cuda_block_mxfp4, ds4_rocm.cu (17 bytes: uint8_t e, uint8_t qs[16]),
 * 32 elements per block, low nibble is element j, high nibble is element
 * j+16 (rocm/ds4_rocm_moe.cuh:340-342's comment on
 * dev_dot_mxfp4_q8_K_block, cross-checked independently by
 * tests/test_sycl_moe.c's own oracle rather than trusted from that comment alone).
 *
 * The value-table and scale computations below are deliberately NOT the
 * float-table-plus-bit-cast shape tests/test_sycl_moe.c's oracle uses
 * (ds4-sycl-mxfp4-validation.md section 3a's symmetry-breaking argument):
 * sycl_dev_mxfp4_unpack1 is a bit-manipulation reconstruction of a doubled
 * signed magnitude, matching the SHAPE of dev_mxfp4_unpack4's portable
 * (non-AMDGCN) branch (rocm/ds4_rocm_moe.cuh:301-326), and
 * sycl_dev_e8m0_to_f32 bit-casts the exponent directly rather than calling
 * ldexpf.  A scalar per-element reformulation of ROCm's dp4a-packed
 * integer dot is used throughout (dp4a sums four exact int8-times-int8
 * products into one int32; summing the same 16 products in any order is
 * exact, so this is not an approximation of the packed form). */

struct sycl_block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

static inline float sycl_dev_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0u ? 0x00400000u : (uint32_t)e << 23u;
    return sycl::bit_cast<float>(bits);
}

/* dev_mxfp4_unpack4, rocm/ds4_rocm_moe.cuh:301-326, portable branch only
 * (the AMDGCN v_perm branch is Instinct-only and not applicable here):
 * reconstructs the doubled signed E2M1 magnitude {0,1,2,3,4,6,8,12} from
 * the 3 magnitude bits by bumping past the two missing 2^-1-mantissa
 * steps (codes 5 and 6) and adding 2 more at code 7 to reach 12, then
 * negates on the sign bit. */
static inline int32_t sycl_dev_mxfp4_unpack1(uint8_t nibble) {
    const uint32_t base = nibble & 7u;
    int32_t value = (int32_t)(base + (base > 4u ? base - 4u : 0u) + (base == 7u ? 2u : 0u));
    if (nibble & 8u) value = -value;
    return value;
}

/* dev_dot_mxfp4_q8_K_half_block, rocm/ds4_rocm_moe.cuh:365-388: dots half
 * of one 32-value MXFP4 block (16 elements) against the matching half of
 * one Q8_K sub-block's 32 activation codes.  half==0 covers weight bytes
 * 0..7 (elements 0..7 low-nibble, 16..23 high-nibble); half==1 covers
 * weight bytes 8..15 (elements 8..15 low-nibble, 24..31 high-nibble). */
static inline float sycl_dev_dot_mxfp4_q8_k_half_block(
        const sycl_block_mxfp4 *x, const sycl_block_q8_K *y, uint32_t subblock,
        uint32_t half) {
    const uint32_t weight_offset = half * 8u;
    const uint32_t activation_offset = subblock * 32u + weight_offset;
    const int8_t *q8_lo = y->qs + activation_offset;
    const int8_t *q8_hi = y->qs + activation_offset + 16u;
    int32_t bsum = 0;
    for (uint32_t k = 0; k < 8u; k++) {
        const uint8_t byte = x->qs[weight_offset + k];
        bsum += sycl_dev_mxfp4_unpack1(byte & 0x0fu) * (int32_t)q8_lo[k];
        bsum += sycl_dev_mxfp4_unpack1(byte >> 4u) * (int32_t)q8_hi[k];
    }
    return 0.5f * y->d * sycl_dev_e8m0_to_f32(x->e) * (float)bsum;
}

/* moe_gate_up_mid_decode_mxfp4_qwarp32_kernel, rocm/ds4_rocm_moe.cuh:
 * 2767-2843.  Handles every MXFP4 call with n_tokens <= 4
 * (rocm/ds4_rocm_moe_launch.cuh:753-754's use_mxfp4_tiny_batch, decode
 * included), a genuinely different algorithm from the tile8 kernel rather
 * than a smaller tiling of it: one 32-wide sub-group computes one
 * (pair,row) output, splitting the 8 q8_K sub-blocks per chunk 16 ways
 * across lane pairs (block_lane) and each MXFP4 block's 32 values in half
 * between the two lanes of a pair (half). */
static void sycl_moe_mxfp4_gate_up_mid_decode(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const int32_t *selected, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert, uint32_t pair_count, float clamp) {
    const uint32_t row_groups = (expert_mid_dim + 7u) / 8u;
    if (row_groups == 0u || pair_count == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_groups * 256u, pair_count),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t wave = (uint32_t)(it.get_local_id(0) >> 5);
                 const uint32_t block_lane = lane >> 1u;
                 const uint32_t half = lane & 1u;
                 const uint32_t row = (uint32_t)it.get_group(0) * 8u + wave;
                 if (row >= expert_mid_dim) return;
                 const uint32_t pair = (uint32_t)it.get_group(1);
                 const uint32_t tok = pair / n_expert;
                 const uint32_t slot = pair - tok * n_expert;
                 int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
                 if (expert_i < 0) expert_i = 0;
                 const uint32_t expert = (uint32_t)expert_i;
                 const sycl_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
                 const sycl_block_mxfp4 *gate_blocks = (const sycl_block_mxfp4 *)
                         (gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 const sycl_block_mxfp4 *up_blocks = (const sycl_block_mxfp4 *)
                         (up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                 const uint32_t mxfp4_blocks = xq_blocks * 8u;
                 float gate = 0.0f, up = 0.0f;
                 for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
                     const sycl_block_q8_K *yb = xqb + (mb >> 3u);
                     const uint32_t subblock = mb & 7u;
                     gate += sycl_dev_dot_mxfp4_q8_k_half_block(gate_blocks + mb, yb, subblock, half);
                     up += sycl_dev_dot_mxfp4_q8_k_half_block(up_blocks + mb, yb, subblock, half);
                 }
                 gate = sycl_moe_subgroup_sum<32>(sg, gate);
                 up = sycl_moe_subgroup_sum<32>(sg, up);
                 if (lane == 0u) {
                     if (clamp > 1.0e-6f) {
                         if (gate > clamp) gate = clamp;
                         if (up > clamp) up = clamp;
                         if (up < -clamp) up = -clamp;
                     }
                     const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                     mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up *
                                    weights[(uint64_t)tok * n_expert + slot];
                 }
             });
     }).wait_and_throw();
}

/* moe_down_mxfp4_sum6_qwarp32_kernel<Batch>, rocm/ds4_rocm_moe.cuh:
 * 3104-3157.  Covers every MXFP4 call with n_tokens <= 4 (the same
 * use_mxfp4_tiny_batch regime as the decode kernel above), writing the
 * final combined output row directly with no separate moe_sum combine
 * pass.  ROCm's template Batch flag only changes whether the token index
 * comes from blockIdx.y or is fixed at 0; since n_tokens == 1 launches
 * with grid.y == 1, blockIdx.y == 0 either way, so this port uses a
 * single kernel with tok = group(1) unconditionally instead of two
 * template instantiations. */
static void sycl_moe_mxfp4_down_sum6(
        sycl::queue &q, float *out, const char *down_base, const sycl_block_q8_K *midq,
        const int32_t *selected, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint32_t row_groups = (out_dim + 7u) / 8u;
    if (row_groups == 0u || n_tokens == 0u) return;
    q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_groups * 256u, n_tokens),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t wave = (uint32_t)(it.get_local_id(0) >> 5);
                 const uint32_t block_lane = lane >> 1u;
                 const uint32_t half = lane & 1u;
                 const uint32_t row = (uint32_t)it.get_group(0) * 8u + wave;
                 const uint32_t tok = (uint32_t)it.get_group(1);
                 if (row >= out_dim) return;
                 const int32_t *token_selected = selected + (uint64_t)tok * n_expert;
                 const sycl_block_q8_K *token_midq = midq + (uint64_t)tok * n_expert * midq_blocks;
                 float *token_out = out + (uint64_t)tok * out_dim;
                 const uint32_t mxfp4_blocks = midq_blocks * 8u;
                 float total = 0.0f;
                 for (uint32_t slot = 0; slot < 8u /* DS4_ROCM_N_EXPERT_USED */; slot++) {
                     if (slot >= n_expert) continue;
                     int32_t expert_i = token_selected[slot];
                     if (expert_i < 0) expert_i = 0;
                     const sycl_block_q8_K *xqb = token_midq + (uint64_t)slot * midq_blocks;
                     for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
                         const sycl_block_q8_K *yb = xqb + (mb >> 3u);
                         const sycl_block_mxfp4 *down_blocks = (const sycl_block_mxfp4 *)
                                 (down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                                  (uint64_t)row * down_row_bytes);
                         total += sycl_dev_dot_mxfp4_q8_k_half_block(down_blocks + mb, yb, mb & 7u, half);
                     }
                 }
                 total = sycl_moe_subgroup_sum<32>(sg, total);
                 if (lane == 0u) token_out[row] = total;
             });
     }).wait_and_throw();
}

/* dev_dot_mxfp4_q8_K_block8, rocm/ds4_rocm_moe.cuh:524-546: the same
 * per-Q8_K-chunk dot as sycl_dev_dot_mxfp4_q8_k_half_block's two halves
 * combined, against up to 8 activation chunks (one per token/slot pair in
 * a tile) at once.  bsum is reset per (sub-block, pair): ROCm's dp4a-packed
 * form resets one int32 accumulator per sub-block and folds four terms at
 * a time; this scalar reformulation sums the same 16 terms per sub-block
 * in a different grouping, which is exact for integers, so the two are
 * bit-identical despite the different shape. */
static inline void sycl_dev_dot_mxfp4_q8_k_block8(
        const sycl_block_mxfp4 *x8, const sycl_block_q8_K *const ys[8], uint32_t n,
        float acc[8]) {
    float chunk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t sb = 0; sb < 8u; sb++) {
        const sycl_block_mxfp4 *x = x8 + sb;
        const float d = sycl_dev_e8m0_to_f32(x->e);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q8 = ys[p]->qs + sb * 32u;
            int32_t bsum = 0;
            for (uint32_t k = 0; k < 16u; k++) {
                const uint8_t byte = x->qs[k];
                bsum += sycl_dev_mxfp4_unpack1(byte & 0x0fu) * (int32_t)q8[k];
                bsum += sycl_dev_mxfp4_unpack1(byte >> 4u) * (int32_t)q8[k + 16u];
            }
            chunk[p] += d * (float)bsum;
        }
    }
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.5f * ys[p]->d * chunk[p];
}

/* moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel, rocm/ds4_rocm_moe.cuh:
 * 2108-2229 (unstaged dot path only: the b128-aligned LDS staging
 * optimisation active when xq_blocks <= 28 is a performance path this
 * port does not need for correctness, following the precedent Q4_K's own
 * tile8 kernel already set in this file).  Canonical prefill gate/up+mid,
 * n_tokens >= 5 for MXFP4 (rocm/ds4_rocm_moe_launch.cuh:750-757:
 * mxfp4_path's use_mxfp4_tiny_batch is n_tokens <= 4, which collapses
 * use_sorted_pairs to n_tokens > 4 for this format, unlike Q4_K's
 * n_tokens >= 32 threshold -- confirmed by reading the launch site, not
 * assumed from Q4_K's own threshold). */
static void sycl_moe_mxfp4_gate_up_mid_tile8(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total, const uint32_t *tile_experts,
        const uint32_t *tile_starts, const float *weights, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t n_expert, uint32_t tile_capacity, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 31u) / 32u;
    if (row_blocks == 0u || tile_capacity == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 2> sxq(sycl::range<2>(8, 16), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, tile_capacity),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 const uint32_t tile = (uint32_t)it.get_group(1);
                 if (tile >= *tile_total) return;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t row = (uint32_t)it.get_group(0) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t expert = tile_experts[tile];
                 const uint32_t count = counts[expert];
                 const uint32_t local_start = tile_starts[tile];

                 uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 const sycl_block_q8_K *xqb[8] = {xq, xq, xq, xq, xq, xq, xq, xq};
                 uint32_t np = 0;
                 for (; np < 8u; np++) {
                     const uint32_t local_pair = local_start + np;
                     if (local_pair >= count) break;
                     pair[np] = sorted_pairs[offsets[expert] + local_pair];
                     const uint32_t tok = pair[np] / n_expert;
                     xqb[np] = xq + (uint64_t)tok * xq_blocks;
                 }
                 if (xq_blocks <= 16u) {
                     const uint32_t lid = (uint32_t)it.get_local_id(0);
                     for (uint32_t i = lid; i < np * xq_blocks; i += 256u) {
                         const uint32_t p = i / xq_blocks;
                         const uint32_t b = i - p * xq_blocks;
                         sxq[p][b] = xqb[p][b];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                     for (uint32_t p = 0; p < np; p++) xqb[p] = &sxq[p][0];
                 }
                 if (row >= expert_mid_dim) return;
                 const char *gate_row = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
                 const char *up_row = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
                 const uint64_t gate_chunk_bytes = gate_row_bytes / xq_blocks;
                 float gate[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 float up[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                     const sycl_block_mxfp4 *gb = (const sycl_block_mxfp4 *)(gate_row + (uint64_t)b * gate_chunk_bytes);
                     const sycl_block_mxfp4 *ub = (const sycl_block_mxfp4 *)(up_row + (uint64_t)b * gate_chunk_bytes);
                     const sycl_block_q8_K *ys[8];
                     for (uint32_t p = 0; p < 8u; p++) ys[p] = xqb[p] + b;
                     sycl_dev_dot_mxfp4_q8_k_block8(gb, ys, np, gate);
                     sycl_dev_dot_mxfp4_q8_k_block8(ub, ys, np, up);
                 }
                 for (uint32_t p = 0; p < np; p++) {
                     gate[p] = sycl_moe_subgroup_sum<8>(sg, gate[p]);
                     up[p] = sycl_moe_subgroup_sum<8>(sg, up[p]);
                     if (lane == 0u) {
                         float g = gate[p], u = up[p];
                         if (clamp > 1.0e-6f) {
                             if (g > clamp) g = clamp;
                             if (u > clamp) u = clamp;
                             if (u < -clamp) u = -clamp;
                         }
                         mid_out[(uint64_t)pair[p] * expert_mid_dim + row] =
                                 (g / (1.0f + sycl::exp(-g))) * u * weights[pair[p]];
                     }
                 }
             });
     }).wait_and_throw();
}

/* moe_down_mxfp4_expert_tile8_row32_kernel, rocm/ds4_rocm_moe.cuh:
 * 3339-3436 (unstaged dot path only, same rationale as the gate/up tile8
 * kernel above).  Canonical prefill down, n_tokens >= 5 for MXFP4.
 * row_groups is DS4_ROCM_MXFP4_DOWN_RGROUP (rocm/ds4_rocm_moe_launch.cuh:
 * 801-807): parameterises how many 32-row groups one work-group covers
 * against one staged copy of the tile's activations, wired to the
 * dispatcher's environment-variable read below.  The arithmetic per (token,row) is
 * untouched by row_groups (same helper, same lane-strided block loop,
 * same reduction), so results are bit-identical for any valid value. */
static void sycl_moe_mxfp4_down_tile8(
        sycl::queue &q, float *down_out, const char *down_base, const sycl_block_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t tile_capacity, uint32_t row_groups) {
    if (row_groups == 0u) row_groups = 1u;
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    const uint32_t grid_x = (row_blocks + row_groups - 1u) / row_groups;
    if (grid_x == 0u || tile_capacity == 0u) return;
    q.submit([&](sycl::handler &h) {
         sycl::local_accessor<sycl_block_q8_K, 2> sxq(sycl::range<2>(8, 8), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)grid_x * 256u, tile_capacity),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                 const uint32_t tile = (uint32_t)it.get_group(1);
                 if (tile >= *tile_total) return;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = (uint32_t)sg.get_local_id()[0];
                 const uint32_t expert = tile_experts[tile];
                 const uint32_t count = counts[expert];
                 const uint32_t local_start = tile_starts[tile];

                 uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                 const sycl_block_q8_K *xqb[8] = {midq, midq, midq, midq, midq, midq, midq, midq};
                 uint32_t np = 0;
                 for (; np < 8u; np++) {
                     const uint32_t local_pair = local_start + np;
                     if (local_pair >= count) break;
                     pair[np] = sorted_pairs[offsets[expert] + local_pair];
                     xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
                 }
                 if (midq_blocks <= 8u) {
                     const uint32_t lid = (uint32_t)it.get_local_id(0);
                     for (uint32_t i = lid; i < np * midq_blocks; i += 256u) {
                         const uint32_t p = i / midq_blocks;
                         const uint32_t b = i - p * midq_blocks;
                         sxq[p][b] = xqb[p][b];
                     }
                     it.barrier(sycl::access::fence_space::local_space);
                     for (uint32_t p = 0; p < np; p++) xqb[p] = &sxq[p][0];
                 }
                 const uint64_t down_chunk_bytes = down_row_bytes / midq_blocks;
                 for (uint32_t rg = 0; rg < row_groups; rg++) {
                     const uint32_t row = (it.get_group(0) * row_groups + rg) * 32u + (uint32_t)(it.get_local_id(0) >> 3);
                     if (row >= out_dim) continue;
                     const char *down_row = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes;
                     float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         const sycl_block_mxfp4 *wb = (const sycl_block_mxfp4 *)(down_row + (uint64_t)b * down_chunk_bytes);
                         const sycl_block_q8_K *ys[8];
                         for (uint32_t p = 0; p < 8u; p++) ys[p] = xqb[p] + b;
                         sycl_dev_dot_mxfp4_q8_k_block8(wb, ys, np, acc);
                     }
                     for (uint32_t p = 0; p < np; p++) {
                         acc[p] = sycl_moe_subgroup_sum<8>(sg, acc[p]);
                         if (lane == 0u) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                     }
                 }
             });
     }).wait_and_throw();
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

/* Runs sycl_moe_q8_k_quantize and copies every output block back as raw
 * bytes: 4 (d, float) + 256 (qs, int8) + 32 (bsums, int16) = 292 bytes
 * per block, n_rows * (in_dim/256) blocks, matching sycl_block_q8_K's
 * layout exactly (verified padding-free: qs starts at offset 4, ends at
 * offset 260, which already satisfies bsums' 2-byte alignment). */
extern "C" int ds4_sycl_moe_test_q8_k_quantize(const float *x, uint32_t in_dim,
                                               uint32_t n_rows,
                                               uint8_t *out_bytes) {
    if (g_devices.empty() || !x || !out_bytes || in_dim == 0u || n_rows == 0u ||
        in_dim % kMoeQK != 0u) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint32_t xq_blocks = in_dim / kMoeQK;
        const uint64_t n_blocks = (uint64_t)n_rows * xq_blocks;
        uint64_t x_bytes = 0, out_elem_bytes = 0;
        if (!sycl_u64_mul3_checked(n_rows, in_dim, sizeof(float), &x_bytes) ||
            !sycl_u64_mul_checked(n_blocks, sizeof(sycl_block_q8_K), &out_elem_bytes)) {
            return 0;
        }

        float *d_x = (float *)sycl::malloc_device((size_t)x_bytes, q);
        sycl_block_q8_K *d_out =
                (sycl_block_q8_K *)sycl::malloc_device((size_t)out_elem_bytes, q);
        if (!d_x || !d_out) {
            if (d_x) sycl::free(d_x, q);
            if (d_out) sycl::free(d_out, q);
            return 0;
        }
        sycl_device_scratch_guard x_guard(q, d_x);
        sycl_device_scratch_guard out_guard(q, d_out);

        q.memcpy(d_x, x, (size_t)x_bytes).wait_and_throw();
        sycl_moe_q8_k_quantize(q, d_out, d_x, in_dim, n_rows);
        q.memcpy(out_bytes, d_out, (size_t)out_elem_bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_q8_k_quantize failed: %s\n",
                e.what());
        return 0;
    }
}

extern "C" int ds4_sycl_moe_test_subgroup_sum(int width, const float *in,
                                              uint32_t n_groups, float *out) {
    if (g_devices.empty() || !in || !out || (width != 8 && width != 16)) return 0;
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint64_t n_in = (uint64_t)n_groups * (uint32_t)width;
        float *d_in = (float *)sycl::malloc_device(n_in * sizeof(float), q);
        float *d_out = (float *)sycl::malloc_device((size_t)n_groups * sizeof(float), q);
        if (!d_in || !d_out) {
            if (d_in) sycl::free(d_in, q);
            if (d_out) sycl::free(d_out, q);
            return 0;
        }
        sycl_device_scratch_guard in_guard(q, d_in);
        sycl_device_scratch_guard out_guard(q, d_out);
        q.memcpy(d_in, in, n_in * sizeof(float)).wait_and_throw();
        if (width == 8) {
            sycl_moe_test_subgroup_sum8(q, d_in, n_groups, d_out);
        } else {
            sycl_moe_test_subgroup_sum16(q, d_in, n_groups, d_out);
        }
        q.memcpy(out, d_out, (size_t)n_groups * sizeof(float)).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_subgroup_sum failed: %s\n",
                e.what());
        return 0;
    }
}

/* mode: 0 = f32 (moe_sum_kernel), 1 = f16 (moe_sum_f16_kernel),
 * 2 = f16x2 (moe_sum_f16x2_kernel).  down/down_h is n_tokens*n_expert*out_dim
 * elements; out is n_tokens*out_dim floats. */
extern "C" int ds4_sycl_moe_test_sum(int mode, const void *down, uint32_t out_dim,
                                     uint32_t n_expert, uint32_t n_tokens,
                                     float *out) {
    if (g_devices.empty() || !down || !out || out_dim == 0u || n_expert == 0u ||
        n_tokens == 0u) {
        return 0;
    }
    if (mode == 2 && (out_dim & 1u)) return 0;
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint64_t n_down = (uint64_t)n_tokens * n_expert * out_dim;
        const uint64_t n_out = (uint64_t)n_tokens * out_dim;
        const size_t elem_size = mode == 0 ? sizeof(float) : sizeof(uint16_t);
        void *d_down = sycl::malloc_device(n_down * elem_size, q);
        float *d_out = (float *)sycl::malloc_device(n_out * sizeof(float), q);
        if (!d_down || !d_out) {
            if (d_down) sycl::free(d_down, q);
            if (d_out) sycl::free(d_out, q);
            return 0;
        }
        sycl_device_scratch_guard down_guard(q, d_down);
        sycl_device_scratch_guard out_guard(q, d_out);
        q.memcpy(d_down, down, n_down * elem_size).wait_and_throw();
        if (mode == 0) {
            sycl_moe_sum(q, d_out, (const float *)d_down, out_dim, n_expert, n_tokens);
        } else if (mode == 1) {
            sycl_moe_sum_f16(q, d_out, (const uint16_t *)d_down, out_dim, n_expert,
                             n_tokens);
        } else {
            sycl_moe_sum_f16x2(q, d_out, (const uint16_t *)d_down, out_dim, n_expert,
                               n_tokens);
        }
        q.memcpy(out, d_out, n_out * sizeof(float)).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_sum failed: %s\n", e.what());
        return 0;
    }
}

/* Drives sycl_moe_q2k_down_direct directly, bypassing gate/up and the full
 * ABI entry.  Every ABI-level test in tests/test_sycl_moe.c uses
 * RM_N_EXPERT == 2, and IEEE754 addition of exactly two values is exactly
 * commutative (a+b bit-equals b+a always), so no ABI-level test can ever
 * discriminate slot-order summation from expert-id-order summation. This
 * hook exists so a dedicated test can drive n_expert >= 3 without
 * threading a third weight table through every RM_* helper, which all
 * assume RM_N_EXPERT == 2 throughout. down_bytes holds n_total_expert
 * rows of down_expert_bytes; midq_bytes holds n_tokens*n_expert*
 * midq_blocks sycl_block_q8_K blocks, matching sycl_moe_q2k_down_direct's
 * own addressing. */
extern "C" int ds4_sycl_moe_test_q2k_down_direct(
        const uint8_t *down_bytes, const uint8_t *midq_bytes, const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t n_total_expert,
        float *out) {
    if (g_devices.empty() || !down_bytes || !midq_bytes || !selected || !out ||
        out_dim == 0u || n_expert == 0u || n_tokens == 0u || midq_blocks == 0u ||
        n_total_expert == 0u) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint64_t down_bytes_total = (uint64_t)n_total_expert * down_expert_bytes;
        const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
        const uint64_t midq_bytes_total = midq_count * sizeof(sycl_block_q8_K);
        const uint64_t sel_bytes = (uint64_t)n_tokens * n_expert * sizeof(int32_t);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);

        void *d_down = sycl::malloc_device((size_t)down_bytes_total, q);
        void *d_midq = sycl::malloc_device((size_t)midq_bytes_total, q);
        int32_t *d_sel = (int32_t *)sycl::malloc_device((size_t)sel_bytes, q);
        float *d_out = (float *)sycl::malloc_device((size_t)out_bytes, q);
        if (!d_down || !d_midq || !d_sel || !d_out) {
            if (d_down) sycl::free(d_down, q);
            if (d_midq) sycl::free(d_midq, q);
            if (d_sel) sycl::free(d_sel, q);
            if (d_out) sycl::free(d_out, q);
            return 0;
        }
        sycl_device_scratch_guard down_guard(q, d_down);
        sycl_device_scratch_guard midq_guard(q, d_midq);
        sycl_device_scratch_guard sel_guard(q, d_sel);
        sycl_device_scratch_guard out_guard(q, d_out);
        q.memcpy(d_down, down_bytes, (size_t)down_bytes_total);
        q.memcpy(d_midq, midq_bytes, (size_t)midq_bytes_total);
        q.memcpy(d_sel, selected, (size_t)sel_bytes);
        q.wait_and_throw();

        sycl_moe_q2k_down_direct(q, d_out, (const char *)d_down, (const sycl_block_q8_K *)d_midq,
                                 d_sel, down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                                 n_expert, n_tokens);
        q.memcpy(out, d_out, (size_t)out_bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_q2k_down_direct failed: %s\n", e.what());
        return 0;
    }
}

/* Test-only side doors for the MXFP4 decode-regime kernels (n_tokens <= 4):
 * exercise sycl_moe_mxfp4_gate_up_mid_decode / sycl_moe_mxfp4_down_sum6
 * directly rather than only through the public dispatcher, matching the
 * precedent set by this file's sort/quantise/sum test hooks above.  Host
 * buffers are laid out exactly like the real ABI's model_map: gate_model
 * and up_model are n_total_expert rows of gate_expert_bytes, each an
 * expert_mid_dim x gate_row_bytes table of packed MXFP4 blocks. */
extern "C" int ds4_sycl_moe_test_mxfp4_gate_up_mid_decode(
        const void *gate_model, const void *up_model, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t n_total_expert, const float *x, uint32_t in_dim,
        uint32_t n_tokens, const int32_t *selected, const float *weights, uint32_t n_expert,
        uint32_t mid_dim, float clamp, float *mid_out) {
    if (g_devices.empty() || !gate_model || !up_model || !x || !selected || !weights ||
        !mid_out || in_dim == 0u || in_dim % kMoeQK != 0u || mid_dim == 0u ||
        n_tokens == 0u || n_expert == 0u || n_total_expert == 0u) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint32_t xq_blocks = in_dim / kMoeQK;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
        const uint64_t table_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t x_bytes = (uint64_t)n_tokens * in_dim * sizeof(float);
        const uint64_t xq_bytes = (uint64_t)n_tokens * xq_blocks * sizeof(sycl_block_q8_K);
        const uint64_t sel_bytes = pair_count * sizeof(int32_t);
        const uint64_t w_bytes = pair_count * sizeof(float);
        const uint64_t mid_bytes = pair_count * (uint64_t)mid_dim * sizeof(float);

        void *gate_dev = sycl::malloc_device((size_t)table_bytes, q);
        void *up_dev = sycl::malloc_device((size_t)table_bytes, q);
        float *x_dev = (float *)sycl::malloc_device((size_t)x_bytes, q);
        sycl_block_q8_K *xq_dev = (sycl_block_q8_K *)sycl::malloc_device((size_t)xq_bytes, q);
        int32_t *sel_dev = (int32_t *)sycl::malloc_device((size_t)sel_bytes, q);
        float *w_dev = (float *)sycl::malloc_device((size_t)w_bytes, q);
        float *mid_dev = (float *)sycl::malloc_device((size_t)mid_bytes, q);
        if (!gate_dev || !up_dev || !x_dev || !xq_dev || !sel_dev || !w_dev || !mid_dev) {
            if (gate_dev) sycl::free(gate_dev, q);
            if (up_dev) sycl::free(up_dev, q);
            if (x_dev) sycl::free(x_dev, q);
            if (xq_dev) sycl::free(xq_dev, q);
            if (sel_dev) sycl::free(sel_dev, q);
            if (w_dev) sycl::free(w_dev, q);
            if (mid_dev) sycl::free(mid_dev, q);
            return 0;
        }
        sycl_device_scratch_guard gate_guard(q, gate_dev);
        sycl_device_scratch_guard up_guard(q, up_dev);
        sycl_device_scratch_guard x_guard(q, x_dev);
        sycl_device_scratch_guard xq_guard(q, xq_dev);
        sycl_device_scratch_guard sel_guard(q, sel_dev);
        sycl_device_scratch_guard w_guard(q, w_dev);
        sycl_device_scratch_guard mid_guard(q, mid_dev);

        q.memcpy(gate_dev, gate_model, (size_t)table_bytes);
        q.memcpy(up_dev, up_model, (size_t)table_bytes);
        q.memcpy(x_dev, x, (size_t)x_bytes);
        q.memcpy(sel_dev, selected, (size_t)sel_bytes);
        q.memcpy(w_dev, weights, (size_t)w_bytes);
        q.wait_and_throw();

        sycl_moe_q8_k_quantize(q, xq_dev, x_dev, in_dim, n_tokens);
        sycl_moe_mxfp4_gate_up_mid_decode(q, mid_dev, (const char *)gate_dev, (const char *)up_dev,
                                          xq_dev, sel_dev, w_dev, gate_expert_bytes, gate_row_bytes,
                                          xq_blocks, mid_dim, n_expert, (uint32_t)pair_count, clamp);
        q.memcpy(mid_out, mid_dev, (size_t)mid_bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_mxfp4_gate_up_mid_decode failed: %s\n",
                e.what());
        return 0;
    }
}

/* Stage-isolated MXFP4 down check, mirroring
 * tests/test_mxfp4_rocm.c's check_out_from_gpu_mid: the caller feeds
 * this hook the SYCL gate/up kernel's own mid output rather than a CPU
 * oracle's mid, so a down-kernel bug is distinguishable from a gate/up
 * bug rather than compounding into one end-to-end mismatch. */
extern "C" int ds4_sycl_moe_test_mxfp4_down_sum6(
        const void *down_model, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t n_total_expert, const float *mid, uint32_t mid_dim, uint32_t n_tokens,
        const int32_t *selected, uint32_t n_expert, uint32_t out_dim, float *out) {
    if (g_devices.empty() || !down_model || !mid || !selected || !out || mid_dim == 0u ||
        mid_dim % kMoeQK != 0u || out_dim == 0u || n_tokens == 0u || n_expert == 0u ||
        n_total_expert == 0u) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint32_t midq_blocks = mid_dim / kMoeQK;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
        const uint64_t table_bytes = (uint64_t)n_total_expert * down_expert_bytes;
        const uint64_t mid_bytes = pair_count * (uint64_t)mid_dim * sizeof(float);
        const uint64_t midq_bytes = pair_count * midq_blocks * sizeof(sycl_block_q8_K);
        const uint64_t sel_bytes = pair_count * sizeof(int32_t);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);

        void *down_dev = sycl::malloc_device((size_t)table_bytes, q);
        float *mid_dev = (float *)sycl::malloc_device((size_t)mid_bytes, q);
        sycl_block_q8_K *midq_dev = (sycl_block_q8_K *)sycl::malloc_device((size_t)midq_bytes, q);
        int32_t *sel_dev = (int32_t *)sycl::malloc_device((size_t)sel_bytes, q);
        float *out_dev = (float *)sycl::malloc_device((size_t)out_bytes, q);
        if (!down_dev || !mid_dev || !midq_dev || !sel_dev || !out_dev) {
            if (down_dev) sycl::free(down_dev, q);
            if (mid_dev) sycl::free(mid_dev, q);
            if (midq_dev) sycl::free(midq_dev, q);
            if (sel_dev) sycl::free(sel_dev, q);
            if (out_dev) sycl::free(out_dev, q);
            return 0;
        }
        sycl_device_scratch_guard down_guard(q, down_dev);
        sycl_device_scratch_guard mid_guard(q, mid_dev);
        sycl_device_scratch_guard midq_guard(q, midq_dev);
        sycl_device_scratch_guard sel_guard(q, sel_dev);
        sycl_device_scratch_guard out_guard(q, out_dev);

        q.memcpy(down_dev, down_model, (size_t)table_bytes);
        q.memcpy(mid_dev, mid, (size_t)mid_bytes);
        q.memcpy(sel_dev, selected, (size_t)sel_bytes);
        q.wait_and_throw();

        sycl_moe_q8_k_quantize(q, midq_dev, mid_dev, mid_dim, (uint32_t)pair_count);
        sycl_moe_mxfp4_down_sum6(q, out_dev, (const char *)down_dev, midq_dev, sel_dev,
                                 down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                                 n_expert, n_tokens);
        q.memcpy(out, out_dev, (size_t)out_bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_mxfp4_down_sum6 failed: %s\n", e.what());
        return 0;
    }
}

/* Mirror of ds4_sycl_moe_test_q2k_down_direct above, for iq2_iq2_path's
 * down projection (down_type == 16, sycl_moe_iq2_down_direct): the same
 * "n_expert >= 2 in every ABI test cannot discriminate slot order"
 * argument applies identically here, since sycl_moe_iq2_down_direct has
 * the same `for slot in 0..n_expert: total += acc` accumulation shape.
 * down_bytes holds n_total_expert rows of down_expert_bytes of
 * sycl_block_iq2_xxs data (not Q2_K); everything else matches the Q2_K
 * hook's addressing exactly. */
extern "C" int ds4_sycl_moe_test_iq2_down_direct(
        const uint8_t *down_bytes, const uint8_t *midq_bytes, const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens, uint32_t n_total_expert,
        float *out) {
    if (g_devices.empty() || !down_bytes || !midq_bytes || !selected || !out ||
        out_dim == 0u || n_expert == 0u || n_tokens == 0u || midq_blocks == 0u ||
        n_total_expert == 0u) {
        return 0;
    }
    try {
        sycl::queue &q = ds4_sycl_current_queue();
        const uint64_t down_bytes_total = (uint64_t)n_total_expert * down_expert_bytes;
        const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
        const uint64_t midq_bytes_total = midq_count * sizeof(sycl_block_q8_K);
        const uint64_t sel_bytes = (uint64_t)n_tokens * n_expert * sizeof(int32_t);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);

        void *d_down = sycl::malloc_device((size_t)down_bytes_total, q);
        void *d_midq = sycl::malloc_device((size_t)midq_bytes_total, q);
        int32_t *d_sel = (int32_t *)sycl::malloc_device((size_t)sel_bytes, q);
        float *d_out = (float *)sycl::malloc_device((size_t)out_bytes, q);
        if (!d_down || !d_midq || !d_sel || !d_out) {
            if (d_down) sycl::free(d_down, q);
            if (d_midq) sycl::free(d_midq, q);
            if (d_sel) sycl::free(d_sel, q);
            if (d_out) sycl::free(d_out, q);
            return 0;
        }
        sycl_device_scratch_guard down_guard(q, d_down);
        sycl_device_scratch_guard midq_guard(q, d_midq);
        sycl_device_scratch_guard sel_guard(q, d_sel);
        sycl_device_scratch_guard out_guard(q, d_out);
        q.memcpy(d_down, down_bytes, (size_t)down_bytes_total);
        q.memcpy(d_midq, midq_bytes, (size_t)midq_bytes_total);
        q.memcpy(d_sel, selected, (size_t)sel_bytes);
        q.wait_and_throw();

        sycl_moe_iq2_down_direct(q, d_out, (const char *)d_down, (const sycl_block_q8_K *)d_midq,
                                 d_sel, down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                                 n_expert, n_tokens);
        q.memcpy(out, d_out, (size_t)out_bytes).wait_and_throw();
        return 1;
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_test_iq2_down_direct failed: %s\n", e.what());
        return 0;
    }
}
