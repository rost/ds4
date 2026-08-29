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
static inline void sycl_dev_dot_q4_k_q8_k_block8(
        const sycl_block_q4_K *x, const sycl_block_q8_K *const ys[8],
        uint32_t n, float acc[8]) {
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
            summs[p] += (int32_t)m * (int32_t)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int32_t)sc * sycl_dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
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
                     sycl_dev_dot_q4_k_q8_k_block8(gr + b, xqb, np, gate);
                     sycl_dev_dot_q4_k_q8_k_block8(ur + b, xqb, np, up);
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
                     sycl_dev_dot_q4_k_q8_k_block8(wr + b, xqb, np, acc);
                 }
                 for (uint32_t p = 0; p < np; p++) {
                     acc[p] = sycl_moe_subgroup_sum<8>(sg, acc[p]);
                     if (lane == 0u) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
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
