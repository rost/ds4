#pragma once

/* Expert-parallel (owned) routed-MoE decode.
 *
 * CUDA's expert-parallel decode splits the DS4_N_EXPERT (256) routed
 * experts into two contiguous halves, one per rank, and gives each rank
 * the SAME six selected slots (the same token, the same router output) but
 * restricted to the experts it owns. This composes with the
 * selected-expert compaction rather than replacing it: ownership decides
 * WHICH of the six slots this rank computes at all, and among the owned
 * slots, compaction (reusing sycl_moe_build_expert_compaction's shape,
 * ds4_sycl_moe_launch.hpp) decides that only the DISTINCT owned experts are
 * staged to device, never the whole owned half. A rank typically owns
 * roughly half of six slots (about three), so this is staging three
 * experts, not 128 -- the same order of magnitude reduction achieved by
 * staging six instead of 256, just composed with the ownership split
 * instead of applied to the unsplit table.
 *
 * Ported from ds4_cuda.cu:19445-19553 (ownership helpers), :19892-19954,
 * :21062-21123, :21485-21924, :24987-25493 (the owned kernels and ABI
 * entries). The decode entry implements two formats: Q4_K
 * (gate_type==12, down_type==12) and the IQ2_XXS/Q2_K "LUT" pair
 * CUDA uses for decode (gate_type==16, down_type==10) -- Flash's
 * actual routed-expert format. MXFP4 owned decode is not ported; Flash
 * does not use it either. The combine entries and the batch entry are
 * format-agnostic or delegate to the already-multi-format base dispatcher
 * (sycl_routed_moe_launch, ds4_sycl_moe_launch.hpp) and so were never
 * restricted to Q4_K.
 *
 * Every entry below is NONZERO-means-success (spec 3a), verified against
 * ds4_gpu.h's declarations and ds4_cuda.cu's own `return 0` / `cuda_ok(...)`
 * shape for each entry ported here. */

#include "ds4_sycl_moe_launch.hpp"

namespace {

/* Set at the end of every successful owned decode dispatch to how many
 * device allocations that call still owns as it returns.
 *
 * Test-only instrumentation in the same shape as
 * g_sycl_moe_last_staged_expert_count (ds4_sycl_moe_launch.hpp), and for
 * the same reason: the property it pins is invisible in the numeric
 * output. Owning nothing is exactly the condition under which
 * sycl_scratch_release_wait does not drain, so a zero here is what proves
 * the entry returned with its kernels still in flight and the paired rank
 * therefore free to overlap it. Re-introducing an owned scratch buffer --
 * the 24-byte device remap this entry used to allocate, say -- would
 * silently re-serialise the two ranks while every oracle comparison kept
 * passing, so it is a counter that has to catch it. */
static uint32_t g_sycl_moe_owned_decode_owned_scratch = 0;

/* Set on every owned decode dispatch to how many device-to-host readbacks
 * that call performed.
 *
 * It must be 0 on the resident path. A readback there is not a small cost
 * to be traded off: the queue is in_order, so a device-to-host memcpy
 * drains everything already submitted to that tier, the previous layer's
 * kernels included. At two ranks times 43 routed layers that is 86 full
 * pipeline drains per token which the unsplit baseline never performs,
 * because sycl_routed_moe_launch skips its own compaction readback
 * entirely when the weights are resident. On a 14-GPU B60 box across 8
 * GPUs on real DeepSeek V4 Flash that asymmetry was the whole
 * expert-parallel decode regression, 4.35 t/s unsplit against 2.79 t/s
 * split. Nothing about it shows up in an oracle comparison, so a counter
 * is the only thing that can stop it coming back. */
static uint32_t g_sycl_moe_owned_decode_readbacks = 0;

/* moe_owned_local_expert, ds4_cuda.cu:19445-19455: true iff the GLOBAL
 * expert id `expert` falls in [expert_base, expert_base+expert_count),
 * writing the LOCAL (expert_base-relative) id when so. The single
 * definition of what this rank owns, deliberately shared by all three
 * places that need it rather than restated: the packing helper and the
 * packed combine below, the host loop that builds a compacted table in
 * sycl_routed_moe_one_owned_dispatch, and sycl_moe_owned_remap's own
 * device-side resolution, which runs this predicate inside every owned
 * decode kernel. Being callable from device code is why it stays a plain
 * inline function over scalars. */
static inline bool sycl_moe_owned_local_expert(int32_t expert, uint32_t expert_base,
                                               uint32_t expert_count,
                                               uint32_t *local_expert) {
    if (expert < 0) return false;
    const uint32_t e = (uint32_t)expert;
    if (e < expert_base || e - expert_base >= expert_count) return false;
    if (local_expert) *local_expert = e - expert_base;
    return true;
}

/* moe_owned_packed_component, ds4_cuda.cu:21521-21552: within a group of
 * three consecutive slots (group 0: slots 0-2, group 1: slots 3-5), decides
 * which original slot (or, when exactly two of the three are owned AND the
 * first two happen to be the owned pair, which PAIR of slots summed
 * together) a given packed component holds. This is what lets six slots
 * pack into four: whenever two owned slots share a group, their
 * contributions are pre-summed into one packed value, since the final
 * combine only ever needs the total. `*prefix_pair` reports that case. */
static inline int sycl_moe_owned_packed_component(const int32_t *selected, uint32_t group,
                                                   uint32_t component, uint32_t expert_base,
                                                   uint32_t expert_count, bool *prefix_pair) {
    const uint32_t slot0 = group * 3u;
    uint32_t mask = 0u;
    for (uint32_t i = 0; i < 3u; i++) {
        if (sycl_moe_owned_local_expert(selected[slot0 + i], expert_base, expert_count,
                                        nullptr)) {
            mask |= 1u << i;
        }
    }
    *prefix_pair = false;
    if ((mask & 3u) == 3u) {
        if (component == 0u) {
            *prefix_pair = true;
            return (int)slot0;
        }
        return (mask & 4u) != 0u ? (int)(slot0 + 2u) : -1;
    }
    uint32_t ordinal = 0u;
    for (uint32_t i = 0; i < 3u; i++) {
        if ((mask & (1u << i)) == 0u) continue;
        if (ordinal++ == component) return (int)(slot0 + i);
    }
    return -1;
}

/* moe_owned_packed_combine_row, ds4_cuda.cu:21852-21909: combines this
 * rank's own six (unpacked) slots with the peer's four packed slots into
 * one output row. Each of the six selected-expert contributions is read
 * from whichever side actually computed it -- home_slots directly for a
 * slot this rank owns, peer_packed (unpacked back out via the same pairing
 * sycl_moe_owned_packed_component describes) for a slot the peer owns --
 * then all six are summed. Shared by the packed combine ABI entry below
 * and by shared_down_hc_expand_owned_q8_0_tensor's fused block_v term
 * (ds4_sycl_hc.hpp). */
static inline float sycl_moe_owned_packed_combine_row(const float *home_slots,
                                                       const float *peer_packed,
                                                       const int32_t *selected, uint32_t row,
                                                       uint32_t out_dim,
                                                       uint32_t expert_split) {
    float groups[2];
    for (uint32_t group = 0; group < 2u; group++) {
        const uint32_t slot0 = group * 3u;
        uint32_t peer_mask = 0u, valid_mask = 0u;
        for (uint32_t i = 0; i < 3u; i++) {
            const int32_t expert = selected[slot0 + i];
            if (expert >= 0 && (uint32_t)expert < 2u * expert_split) valid_mask |= 1u << i;
            if (expert >= 0 && (uint32_t)expert >= expert_split &&
                (uint32_t)expert < 2u * expert_split) {
                peer_mask |= 1u << i;
            }
        }
        const float *packed = peer_packed + (uint64_t)group * 2u * out_dim + row;
        float acc;
        if ((peer_mask & 3u) == 3u) {
            acc = packed[0];
            float slot2 = 0.0f;
            if ((peer_mask & 4u) != 0u) {
                slot2 = packed[out_dim];
            } else if ((valid_mask & 4u) != 0u) {
                slot2 = home_slots[(uint64_t)(slot0 + 2u) * out_dim + row];
            }
            acc += slot2;
        } else {
            acc = 0.0f;
            uint32_t peer_operand = 0u;
            for (uint32_t i = 0; i < 3u; i++) {
                float value;
                if ((peer_mask & (1u << i)) != 0u) {
                    value = packed[(uint64_t)peer_operand * out_dim];
                    peer_operand++;
                } else if ((valid_mask & (1u << i)) != 0u) {
                    value = home_slots[(uint64_t)(slot0 + i) * out_dim + row];
                } else {
                    value = 0.0f;
                }
                acc += value;
            }
        }
        groups[group] = acc;
    }
    return groups[0] + groups[1];
}

/* ---- Decode owned Q4_K kernels ----------------------------------------
 *
 * All three kernels below take `remap`, NOT expert_base/expert_count
 * directly: it is `sycl_moe_owned_remap` below, which answers the one
 * question every owned kernel asks per slot, "which row of the weight
 * table this launch was handed does slot i read, or is slot i not mine",
 * without the kernel needing to know which of the two ways that table was
 * built. It travels BY VALUE as a kernel argument, never through a device
 * buffer, so the dispatch owns no scratch on its account. */

/* Slot-to-weight-table-row resolution for the owned decode kernels.
 *
 * Two addressings exist because there are two ways the weight table this
 * rank reads can come to be, and they are deliberately kept apart rather
 * than collapsed into one array of six ints:
 *
 *   Resident. The placement cache has already made this rank's whole
 *   owned half device-resident, so the table holds every expert of that
 *   half at its expert_base-relative id. The row is then a pure function
 *   of data the device already has: a range test on `selected`. Nothing
 *   about it needs the host, which is the entire point -- see
 *   sycl_moe_owned_remap_resident below.
 *
 *   Compacted. The weights had to cross to the device, so only the
 *   DISTINCT owned experts were staged, packed contiguously. The row is a
 *   position in that packed table, which only the host that built the
 *   table knows, so those six positions ride along by value.
 *
 * `selected` non-null selects the first, and is the discriminant purely
 * because the resident path is the one that has a device pointer to
 * derive from. Both answers mean the same thing to a kernel, so the
 * decision lives in local_of_slot() alone and no kernel repeats it. */
struct sycl_moe_owned_remap {
    /* Device pointer to the six ORIGINAL global selected ids, or null on
     * the compacted path. */
    const int32_t *selected;
    uint32_t expert_base;
    uint32_t expert_count;
    /* Packed-table positions, read only when `selected` is null. */
    int32_t packed[6];

    /* This rank's row for `slot`, or -1 when the peer owns that slot.
     * Shares sycl_moe_owned_local_expert with the host and with
     * sycl_moe_owned_packed_component, so the resident path's device-side
     * ownership test and the packed kernels' own test cannot drift apart:
     * both are that one predicate. */
    int32_t local_of_slot(uint32_t slot) const {
        if (!selected) return packed[slot];
        uint32_t local = 0;
        if (!sycl_moe_owned_local_expert(selected[slot], expert_base, expert_count, &local)) {
            return -1;
        }
        return (int32_t)local;
    }
};

/* The resident carrier: no host involvement at all, which is what lets
 * sycl_routed_moe_one_owned_dispatch return without a single host round
 * trip on the path a real --cuda-tensor-parallel decode takes.
 *
 * The readback this replaces was NOT a cheap six-int fetch. On an
 * in_order queue a device-to-host memcpy drains everything already
 * submitted to that tier, including the previous layer's kernels, so it
 * cost a full pipeline drain per rank per routed layer: 2 ranks times 43
 * routed layers is 86 drains per token that the unsplit baseline never
 * performs, since sycl_routed_moe_launch skips its own compaction
 * readback outright when the weights are resident. Measured on a 14-GPU
 * B60 box across 8 GPUs on real DeepSeek V4 Flash, that gap was the whole
 * regression: 4.35 t/s unsplit against 2.79 t/s expert-parallel, about
 * 128 ms per token over 86 drains, near enough 1.5 ms each. */
static inline sycl_moe_owned_remap sycl_moe_owned_remap_resident(const int32_t *selected,
                                                                 uint32_t expert_base,
                                                                 uint32_t expert_count) {
    sycl_moe_owned_remap r{};
    r.selected = selected;
    r.expert_base = expert_base;
    r.expert_count = expert_count;
    return r;
}

/* The compacted carrier: `packed[i]` is slot i's position in the packed
 * table the host just staged, or -1 when the peer owns slot i. */
static inline sycl_moe_owned_remap sycl_moe_owned_remap_packed(const int32_t (&packed)[6]) {
    sycl_moe_owned_remap r{};
    r.selected = nullptr;
    for (uint32_t slot = 0; slot < 6u; slot++) r.packed[slot] = packed[slot];
    return r;
}

static void sycl_moe_q4k_gate_up_mid_decode_owned(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, sycl_moe_owned_remap remap, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 127u) / 128u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev118 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 6u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t slot = (uint32_t)it.get_group(1);
                 const int32_t local = remap.local_of_slot(slot);
                 if (local < 0) return;
                 const uint32_t expert = (uint32_t)local;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_lane = (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t row_block = (uint32_t)it.get_group(0);
                 for (uint32_t rr = 0; rr < 4u; rr++) {
                     const uint32_t row_want = row_block * 128u + row_lane + rr * 32u;
                     const bool row_ok = row_want < expert_mid_dim;
                     const uint32_t row = row_ok ? row_want : 0u;
                     const sycl_block_q4_K *gr = (const sycl_block_q4_K *)
                             (gate_base + (uint64_t)expert * gate_expert_bytes +
                              (uint64_t)row * gate_row_bytes);
                     const sycl_block_q4_K *ur = (const sycl_block_q4_K *)
                             (up_base + (uint64_t)expert * gate_expert_bytes +
                              (uint64_t)row * gate_row_bytes);
                     float gate = 0.0f, up = 0.0f;
                     for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                         gate += sycl_dev_dot_q4_k_q8_k_block(gr + b, xq + b);
                         up += sycl_dev_dot_q4_k_q8_k_block(ur + b, xq + b);
                     }
                     gate = sycl_moe_subgroup_sum<8>(sg, gate);
                     up = sycl_moe_subgroup_sum<8>(sg, up);
                     if (row_ok && lane == 0u) {
                         if (clamp > 1.0e-6f) {
                             if (gate > clamp) gate = clamp;
                             if (up > clamp) up = clamp;
                             if (up < -clamp) up = -clamp;
                         }
                         const uint64_t off = (uint64_t)slot * expert_mid_dim + row;
                         mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up * weights[slot];
                     }
                 }
             });
     });
     /* No wait: q is in_order (ds4_sycl.cpp), so whatever the caller
      * submits next is already ordered behind this kernel, and nothing
      * this kernel reads is scratch this launcher has to keep alive.
      * Draining here would also hold up the OTHER expert-parallel rank,
      * whose kernels ds4.c submits to a different tier's queue right
      * after this one returns. Worth being exact about how much that
      * costs, since it is easy to overstate: removing these drains alone
      * moved 8-GPU decode from 2.78 to 2.79 t/s, which is nothing. The
      * drains were real but they were never the regression. The
      * device-to-host readback this entry used to perform was, and it is
      * gone too -- see sycl_moe_owned_remap_resident. */
     ds4_sycl_profile_record(_ds4_prof_ev118);
}

/* moe_gate_up_mid_decode_lut_owned_qwarp32_kernel, ds4_cuda.cu:19892-19954:
 * the IQ2_XXS/Q2_K ("LUT") owned gate/up, remap-based per the comment
 * above. CUDA stages the IQ2_XXS grid/signs tables into shared memory when
 * xq_blocks <= 16 before dotting; this port, like the non-owned decode
 * dispatcher's sycl_moe_iq2_gate_up_mid_decode (ds4_sycl_moe.hpp), reads
 * those tables directly through sycl_dev_dot_iq2_xxs_q8_k_block instead,
 * which is mathematically identical (see that function's own comment for
 * why this is a bandwidth optimisation CUDA makes, not a different
 * computation). write_aux (CUDA's optional gate/up debug output, gated on
 * DS4_CUDA_MOE_WRITE_GATE_UP) has no live caller on this backend, matching
 * every other decode kernel here that never writes it. */
static void sycl_moe_lut_gate_up_mid_decode_owned(
        sycl::queue &q, float *mid_out, const char *gate_base, const char *up_base,
        const sycl_block_q8_K *xq, sycl_moe_owned_remap remap, const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint32_t xq_blocks,
        uint32_t expert_mid_dim, float clamp) {
    const uint32_t row_blocks = (expert_mid_dim + 127u) / 128u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev119 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 6u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t slot = (uint32_t)it.get_group(1);
                 const int32_t local = remap.local_of_slot(slot);
                 if (local < 0) return;
                 const uint32_t expert = (uint32_t)local;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_lane = (uint32_t)(it.get_local_id(0) >> 3);
                 const uint32_t row_block = (uint32_t)it.get_group(0);
                 for (uint32_t rr = 0; rr < 4u; rr++) {
                     const uint32_t row_want = row_block * 128u + row_lane + rr * 32u;
                     const bool row_ok = row_want < expert_mid_dim;
                     const uint32_t row = row_ok ? row_want : 0u;
                     const sycl_block_iq2_xxs *gr = (const sycl_block_iq2_xxs *)
                             (gate_base + (uint64_t)expert * gate_expert_bytes +
                              (uint64_t)row * gate_row_bytes);
                     const sycl_block_iq2_xxs *ur = (const sycl_block_iq2_xxs *)
                             (up_base + (uint64_t)expert * gate_expert_bytes +
                              (uint64_t)row * gate_row_bytes);
                     float gate = 0.0f, up = 0.0f;
                     for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                         gate += sycl_dev_dot_iq2_xxs_q8_k_block(gr + b, xq + b);
                         up += sycl_dev_dot_iq2_xxs_q8_k_block(ur + b, xq + b);
                     }
                     gate = sycl_moe_subgroup_sum<8>(sg, gate);
                     up = sycl_moe_subgroup_sum<8>(sg, up);
                     if (row_ok && lane == 0u) {
                         if (clamp > 1.0e-6f) {
                             if (gate > clamp) gate = clamp;
                             if (up > clamp) up = clamp;
                             if (up < -clamp) up = -clamp;
                         }
                         const uint64_t off = (uint64_t)slot * expert_mid_dim + row;
                         mid_out[off] = (gate / (1.0f + sycl::exp(-gate))) * up * weights[slot];
                     }
                 }
             });
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev119);
}

/* q8_K_quantize_owned_kernel, ds4_cuda.cu:19476-19532, remap-based per the
 * comment above: only quantises the six activation rows this rank owns,
 * leaving the rest of `out` untouched (never read by the down kernels
 * below, which apply the identical remap check before reading). */
static void sycl_moe_q8_k_quantize_owned(sycl::queue &q, sycl_block_q8_K *out, const float *x,
                                         sycl_moe_owned_remap remap, uint32_t in_dim) {
    const uint32_t xq_blocks = in_dim / kMoeQK;
    if (xq_blocks == 0u) return;
    sycl::event _ds4_prof_ev120 = q.submit([&](sycl::handler &h) {
         sycl::local_accessor<float, 1> abs_part(sycl::range<1>(kMoeQK), h);
         sycl::local_accessor<float, 1> val_part(sycl::range<1>(kMoeQK), h);
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)xq_blocks * kMoeQK, 6u),
                               sycl::range<2>(kMoeQK, 1)),
             [=](sycl::nd_item<2> it) {
                 const uint32_t row = (uint32_t)it.get_group(1);
                 if (remap.local_of_slot(row) < 0) return;
                 const uint32_t b = (uint32_t)it.get_group(0);
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
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev120);
}

/* moe_down_q4K_owned_slots_qwarp32_kernel, ds4_cuda.cu:21715-21753: writes
 * the down projection for each OWNED slot into its own row of a six-row
 * output; a non-owned slot's row is left untouched (never read by the
 * cross-device combine, which independently knows which rank owns which
 * slot -- see sycl_moe_owned_slots_combine_rows below). */
static void sycl_moe_q4k_down_owned_slots(
        sycl::queue &q, float *down_out, const char *down_base, const sycl_block_q8_K *midq,
        sycl_moe_owned_remap remap, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev121 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 6u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t slot = (uint32_t)it.get_group(1);
                 const int32_t local = remap.local_of_slot(slot);
                 if (local < 0) return;
                 const uint32_t expert = (uint32_t)local;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_want = (uint32_t)it.get_group(0) * 32u +
                                      (uint32_t)(it.get_local_id(0) >> 3);
                 const bool row_ok = row_want < out_dim;
                 const uint32_t row = row_ok ? row_want : 0u;
                 const sycl_block_q4_K *wr = (const sycl_block_q4_K *)
                         (down_base + (uint64_t)expert * down_expert_bytes +
                          (uint64_t)row * down_row_bytes);
                 const sycl_block_q8_K *xqb = midq + (uint64_t)slot * midq_blocks;
                 float acc = 0.0f;
                 for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                     acc += sycl_dev_dot_q4_k_q8_k_block(wr + b, xqb + b);
                 }
                 acc = sycl_moe_subgroup_sum<8>(sg, acc);
                 if (row_ok && lane == 0u) down_out[(uint64_t)slot * out_dim + row] = acc;
             });
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev121);
}

/* moe_down_owned_slots_qwarp32_kernel, ds4_cuda.cu:21485-21515: the Q2_K
 * ("LUT" pair's down) owned per-slot kernel, remap-based like its Q4_K
 * sibling above. */
static void sycl_moe_lut_down_owned_slots(
        sycl::queue &q, float *down_out, const char *down_base, const sycl_block_q8_K *midq,
        sycl_moe_owned_remap remap, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev122 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 6u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t slot = (uint32_t)it.get_group(1);
                 const int32_t local = remap.local_of_slot(slot);
                 if (local < 0) return;
                 const uint32_t expert = (uint32_t)local;
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_want = (uint32_t)it.get_group(0) * 32u +
                                      (uint32_t)(it.get_local_id(0) >> 3);
                 const bool row_ok = row_want < out_dim;
                 const uint32_t row = row_ok ? row_want : 0u;
                 const sycl_block_q2_K *wr = (const sycl_block_q2_K *)
                         (down_base + (uint64_t)expert * down_expert_bytes +
                          (uint64_t)row * down_row_bytes);
                 const sycl_block_q8_K *xqb = midq + (uint64_t)slot * midq_blocks;
                 float acc = 0.0f;
                 for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                     acc += sycl_dev_dot_q2_k_q8_k_block(wr + b, xqb + b);
                 }
                 acc = sycl_moe_subgroup_sum<8>(sg, acc);
                 if (row_ok && lane == 0u) down_out[(uint64_t)slot * out_dim + row] = acc;
             });
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev122);
}

/* moe_down_q4K_owned_packed_qwarp32_kernel, ds4_cuda.cu:21755-21813: as
 * above, but packs the (up to) six owned contributions into four slots via
 * sycl_moe_owned_packed_component, pre-summing a group's pair when both its
 * first two slots are owned. `orig_selected`/`expert_base`/`expert_count`
 * are the ORIGINAL (global-id) ownership inputs -- the same ones the
 * packed combine on the other rank will use -- kept separate from `remap`,
 * which only addresses the locally staged/compacted weight table. */
static void sycl_moe_q4k_down_owned_packed(
        sycl::queue &q, float *packed_out, const char *down_base, const sycl_block_q8_K *midq,
        sycl_moe_owned_remap remap, const int32_t *orig_selected, uint32_t expert_base,
        uint32_t expert_count, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev123 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 4u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t packed_slot = (uint32_t)it.get_group(1);
                 bool prefix_pair = false;
                 const int first_slot = sycl_moe_owned_packed_component(
                         orig_selected, packed_slot / 2u, packed_slot & 1u, expert_base,
                         expert_count, &prefix_pair);
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_want = (uint32_t)it.get_group(0) * 32u +
                                      (uint32_t)(it.get_local_id(0) >> 3);
                 const bool row_ok = row_want < out_dim;
                 const uint32_t row = row_ok ? row_want : 0u;
                 if (first_slot < 0) {
                     if (row_ok && lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
                     return;
                 }
                 const uint32_t n_slots = prefix_pair ? 2u : 1u;
                 float packed = 0.0f;
                 for (uint32_t i = 0; i < n_slots; i++) {
                     const uint32_t slot = (uint32_t)first_slot + i;
                     const int32_t local = remap.local_of_slot(slot);
                     if (local < 0) continue;
                     const uint32_t expert = (uint32_t)local;
                     const sycl_block_q4_K *wr = (const sycl_block_q4_K *)
                             (down_base + (uint64_t)expert * down_expert_bytes +
                              (uint64_t)row * down_row_bytes);
                     const sycl_block_q8_K *xqb = midq + (uint64_t)slot * midq_blocks;
                     float acc = 0.0f;
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         acc += sycl_dev_dot_q4_k_q8_k_block(wr + b, xqb + b);
                     }
                     acc = sycl_moe_subgroup_sum<8>(sg, acc);
                     if (row_ok && lane == 0u) packed = prefix_pair ? packed + acc : acc;
                 }
                 if (row_ok && lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
             });
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev123);
}

/* moe_down_owned_packed_qwarp32_kernel, ds4_cuda.cu:21604-21654: the Q2_K
 * ("LUT" pair's down) owned packed kernel, mirroring
 * sycl_moe_q4k_down_owned_packed above with a Q2_K weight block and dot. */
static void sycl_moe_lut_down_owned_packed(
        sycl::queue &q, float *packed_out, const char *down_base, const sycl_block_q8_K *midq,
        sycl_moe_owned_remap remap, const int32_t *orig_selected, uint32_t expert_base,
        uint32_t expert_count, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t row_blocks = (out_dim + 31u) / 32u;
    if (row_blocks == 0u) return;
    sycl::event _ds4_prof_ev124 = q.submit([&](sycl::handler &h) {
         h.parallel_for(
             sycl::nd_range<2>(sycl::range<2>((size_t)row_blocks * 256u, 4u),
                               sycl::range<2>(256u, 1u)),
             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                 const uint32_t packed_slot = (uint32_t)it.get_group(1);
                 bool prefix_pair = false;
                 const int first_slot = sycl_moe_owned_packed_component(
                         orig_selected, packed_slot / 2u, packed_slot & 1u, expert_base,
                         expert_count, &prefix_pair);
                 sycl::sub_group sg = it.get_sub_group();
                 const uint32_t lane = sycl_moe_lane8(sg);
                 const uint32_t row_want = (uint32_t)it.get_group(0) * 32u +
                                      (uint32_t)(it.get_local_id(0) >> 3);
                 const bool row_ok = row_want < out_dim;
                 const uint32_t row = row_ok ? row_want : 0u;
                 if (first_slot < 0) {
                     if (row_ok && lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
                     return;
                 }
                 const uint32_t n_slots = prefix_pair ? 2u : 1u;
                 float packed = 0.0f;
                 for (uint32_t i = 0; i < n_slots; i++) {
                     const uint32_t slot = (uint32_t)first_slot + i;
                     const int32_t local = remap.local_of_slot(slot);
                     if (local < 0) continue;
                     const uint32_t expert = (uint32_t)local;
                     const sycl_block_q2_K *wr = (const sycl_block_q2_K *)
                             (down_base + (uint64_t)expert * down_expert_bytes +
                              (uint64_t)row * down_row_bytes);
                     const sycl_block_q8_K *xqb = midq + (uint64_t)slot * midq_blocks;
                     float acc = 0.0f;
                     for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                         acc += sycl_dev_dot_q2_k_q8_k_block(wr + b, xqb + b);
                     }
                     acc = sycl_moe_subgroup_sum<8>(sg, acc);
                     if (row_ok && lane == 0u) packed = prefix_pair ? packed + acc : acc;
                 }
                 if (row_ok && lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
             });
     });
     /* No wait, same reasoning as sycl_moe_q4k_gate_up_mid_decode_owned. */
     ds4_sycl_profile_record(_ds4_prof_ev124);
}

/* Function-pointer bundle selecting one format's owned-decode kernel
 * triple (gate/up/mid, per-slot down, packed down). Both formats this
 * backend implements (Q4_K and the IQ2_XXS/Q2_K "LUT" pair) share every
 * other line of ds4_gpu_routed_moe_one_owned_tensor's decode path --
 * validation, staging, remap construction, the two scratch-buffer reuses
 * and the staged-count instrumentation -- so only the three kernels that
 * actually touch a quantised weight block vary by format. This mirrors
 * how ds4_cuda.cu keeps ONE ABI entry and branches per kernel launch
 * (its q4k_path/mxfp4_path bools) rather than duplicating the whole
 * function per format. */
struct sycl_moe_owned_decode_kernels {
    void (*gate_up_mid)(sycl::queue &, float *, const char *, const char *,
                        const sycl_block_q8_K *, sycl_moe_owned_remap, const float *, uint64_t,
                        uint64_t, uint32_t, uint32_t, float);
    void (*down_slots)(sycl::queue &, float *, const char *, const sycl_block_q8_K *,
                       sycl_moe_owned_remap, uint64_t, uint64_t, uint32_t, uint32_t);
    void (*down_packed)(sycl::queue &, float *, const char *, const sycl_block_q8_K *,
                        sycl_moe_owned_remap, const int32_t *, uint32_t, uint32_t, uint64_t,
                        uint64_t, uint32_t, uint32_t);
};

static const sycl_moe_owned_decode_kernels sycl_moe_owned_decode_q4k = {
    sycl_moe_q4k_gate_up_mid_decode_owned,
    sycl_moe_q4k_down_owned_slots,
    sycl_moe_q4k_down_owned_packed,
};

static const sycl_moe_owned_decode_kernels sycl_moe_owned_decode_lut = {
    sycl_moe_lut_gate_up_mid_decode_owned,
    sycl_moe_lut_down_owned_slots,
    sycl_moe_lut_down_owned_packed,
};

/* ds4_gpu_routed_moe_one_owned_tensor, ds4_cuda.cu:24987-25312. Implements
 * two formats: Q4_K (gate_type==12 && down_type==12) and the IQ2_XXS/Q2_K
 * "LUT" pair CUDA uses for decode (gate_type==16 && down_type==10).
 * MXFP4 owned decode is NOT ported (Flash does not use it). All three fail
 * cleanly (return 0) rather than risk staging the wrong weights, per spec
 * 3a's "never infer a return convention" and the "getting it
 * wrong ... reads absent weights and returns plausible wrong numbers"
 * warning. */
static int sycl_routed_moe_one_owned_dispatch(
        const sycl_moe_owned_decode_kernels *k, ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim,
        uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t resident_expert_base, uint32_t resident_expert_count,
        float clamp, const ds4_gpu_tensor *x, ds4_gpu_tensor *down_output, bool pack_fixed3) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_total_expert == 0u || resident_expert_count == 0u || gate_expert_bytes == 0u ||
        gate_row_bytes == 0u || down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base ||
        expert_in_dim % kMoeQK != 0u || expert_mid_dim % kMoeQK != 0u ||
        selected->bytes < 6u * sizeof(int32_t) || weights->bytes < 6u * sizeof(float) ||
        x->bytes < (uint64_t)expert_in_dim * sizeof(float) ||
        mid->bytes < 6ull * expert_mid_dim * sizeof(float) ||
        out->bytes < (uint64_t)out_dim * sizeof(float)) {
        return 0;
    }
    if (pack_fixed3 && resident_expert_base == 0u) return 0;

    uint64_t gate_shift = 0, down_shift = 0, gate_bytes = 0, down_bytes = 0;
    if (!sycl_u64_mul_checked(resident_expert_base, gate_expert_bytes, &gate_shift) ||
        !sycl_u64_mul_checked(resident_expert_base, down_expert_bytes, &down_shift) ||
        !sycl_u64_mul_checked(resident_expert_count, gate_expert_bytes, &gate_bytes) ||
        !sycl_u64_mul_checked(resident_expert_count, down_expert_bytes, &down_bytes) ||
        !sycl_model_range_fits(model_size, gate_offset, gate_shift + gate_bytes) ||
        !sycl_model_range_fits(model_size, up_offset, gate_shift + gate_bytes) ||
        !sycl_model_range_fits(model_size, down_offset, down_shift + down_bytes)) {
        return 0;
    }

    const uint64_t down_output_bytes =
            (uint64_t)(pack_fixed3 ? 4u : 6u) * out_dim * sizeof(float);
    const uint32_t xq_blocks = expert_in_dim / kMoeQK;
    const uint32_t midq_blocks = expert_mid_dim / kMoeQK;
    const uint64_t xq_bytes = (uint64_t)xq_blocks * sizeof(sycl_block_q8_K);
    const uint64_t midq_bytes = 6ull * midq_blocks * sizeof(sycl_block_q8_K);
    if (down->bytes < xq_bytes || down->bytes < down_output_bytes ||
        (down_output && down_output->bytes < down_output_bytes) ||
        gate->bytes < midq_bytes) {
        return 0;
    }

    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        g_sycl_moe_owned_decode_readbacks = 0;

        const char *gate_w = sycl_model_range_ptr(model_map, gate_offset + gate_shift,
                                                  gate_bytes, model_size, "moe_owned_gate");
        const char *up_w = sycl_model_range_ptr(model_map, up_offset + gate_shift, gate_bytes,
                                                model_size, "moe_owned_up");
        const char *down_w = sycl_model_range_ptr(model_map, down_offset + down_shift,
                                                  down_bytes, model_size, "moe_owned_down");
        if (!gate_w || !up_w || !down_w) return 0;

        /* Which of the two addressings the kernels below will use. Both
         * reach a row as `base + id * expert_bytes + row * row_bytes`; the
         * only question is what `id` counts in.
         *
         * Compaction packs just the distinct owned experts and `id` is a
         * position in that packed table. It is the right choice when the
         * weights must cross to the device anyway, since it stages about
         * three experts instead of this rank's whole owned half.
         *
         * It is the wrong choice, and in fact impossible, once the
         * placement cache has already made this rank's owned range
         * device-resident: sycl_moe_stage_selected_experts gathers rows
         * with a host std::memcpy and refuses a device pointer outright
         * rather than read device memory from the host. That refusal is
         * not hypothetical here. Expert parallelism exists precisely to
         * keep each rank's owned experts resident, so on a real
         * --cuda-tensor-parallel decode all three pointers are resident
         * and this was the path taken; the refusal fired on the first
         * token. The resident table already holds every expert of this
         * rank's half at its expert_base-relative id, so `id` is simply
         * the local id and nothing needs staging or packing at all.
         *
         * A partial residency (one or two of the three cached) takes the
         * same local-id addressing and lets sycl_stage_host_bytes stage
         * only the pointers that are not resident yet, mirroring
         * sycl_routed_moe_launch's own three-way dispatch.
         *
         * This decision is made BEFORE anything looks at `selected`, and
         * that ordering is the point rather than a tidiness preference:
         * only the compacted path may read the selected ids back to the
         * host, so the resident path must never be downstream of a
         * readback it does not need. */
        const bool any_resident = sycl_ptr_is_device_resident(q, gate_w) ||
                                  sycl_ptr_is_device_resident(q, up_w) ||
                                  sycl_ptr_is_device_resident(q, down_w);

        void *gate_dev = nullptr, *up_dev = nullptr, *down_dev = nullptr;
        bool gate_owned = true, up_owned = true, down_owned = true;
        sycl_moe_owned_remap remap;
        uint32_t unique_count = 0;
        if (any_resident) {
            /* No readback: every kernel derives its own slot's local id
             * from `selected` on the device (sycl_moe_owned_remap_resident).
             * There is deliberately no "does this rank own any of the six"
             * early-out here either, because answering it is exactly the
             * host round trip being removed. Launching regardless is not a
             * wasted correctness risk: each kernel's own per-slot check
             * leaves an unowned slot's output untouched, which is the same
             * state the early-out produced, and the packed down kernel
             * writes an explicit 0.0f for a packed component with no owned
             * source rather than relying on the launch being skipped. */
            remap = sycl_moe_owned_remap_resident((const int32_t *)selected->ptr,
                                                  resident_expert_base, resident_expert_count);
            if (!sycl_moe_stage_weights(q, gate_w, up_w, down_w, gate_bytes, down_bytes,
                                        &gate_dev, &up_dev, &down_dev,
                                        &gate_owned, &up_owned, &down_owned)) {
                return 0;
            }
        } else {
            /* Compaction genuinely needs the ids on the host: it has to
             * know the DISTINCT owned experts before it can gather and
             * stage a packed table of exactly those. This readback pays
             * for itself against the host memcpy of whole experts that
             * follows it, and unlike the resident path it is not on any
             * steady-state decode. */
            int32_t sel_host[6];
            sycl::event _ds4_prof_ev125 = q.memcpy(sel_host, selected->ptr, sizeof(sel_host));
            sycl_batch_wait(_ds4_prof_ev125);
            ds4_sycl_profile_record(_ds4_prof_ev125);
            g_sycl_moe_owned_decode_readbacks++;

            /* `local_of_slot` is this rank's expert_base-relative id for
             * each of the six slots, or -1 for a slot the peer owns;
             * `unique_ids` is the distinct set of those locals, which is
             * what gets staged. */
            std::vector<int32_t> unique_ids;
            int32_t local_of_slot[6];
            for (uint32_t slot = 0; slot < 6u; slot++) {
                uint32_t local = 0;
                if (!sycl_moe_owned_local_expert(sel_host[slot], resident_expert_base,
                                                 resident_expert_count, &local)) {
                    local_of_slot[slot] = -1;
                    continue;
                }
                local_of_slot[slot] = (int32_t)local;
                bool seen = false;
                for (size_t i = 0; i < unique_ids.size(); i++) {
                    if (unique_ids[i] == (int32_t)local) { seen = true; break; }
                }
                if (!seen) unique_ids.push_back((int32_t)local);
            }
            if (unique_ids.empty()) {
                /* Nothing this rank owns among the six selected experts,
                 * and here the host already knows it: there is no packed
                 * table to build, so skipping the launches is free rather
                 * than something to pay a round trip for. */
                g_sycl_moe_owned_decode_owned_scratch = 0;
                return 1;
            }
            unique_count = (uint32_t)unique_ids.size();

            int32_t packed_of_slot[6];
            for (uint32_t slot = 0; slot < 6u; slot++) {
                packed_of_slot[slot] = -1;
                if (local_of_slot[slot] < 0) continue;
                for (size_t i = 0; i < unique_ids.size(); i++) {
                    if (unique_ids[i] == local_of_slot[slot]) {
                        packed_of_slot[slot] = (int32_t)i;
                        break;
                    }
                }
            }
            remap = sycl_moe_owned_remap_packed(packed_of_slot);
            /* No device remap out-parameter: this entry's kernels take the
             * remap by value (sycl_moe_owned_remap), so the only thing it
             * needs staged is the packed weight table. */
            if (!sycl_moe_stage_selected_experts(q, gate_w, up_w, down_w, unique_ids.data(),
                                                 unique_count, gate_expert_bytes,
                                                 down_expert_bytes, nullptr, 0u, &gate_dev,
                                                 &up_dev, &down_dev, nullptr)) {
                return 0;
            }
        }
        sycl_device_scratch_guard gate_guard(q, gate_dev, gate_owned);
        sycl_device_scratch_guard up_guard(q, up_dev, up_owned);
        sycl_device_scratch_guard down_guard(q, down_dev, down_owned);

        /* Same test-only counters sycl_routed_moe_launch updates
         * (ds4_sycl_moe_launch.hpp), read back by
         * ds4_sycl_moe_test_last_staged_expert_count/_bytes. This is the
         * number that matters: how many experts this call staged, which
         * is the distinct owned set under compaction and this rank's
         * whole owned half for whichever of the three tables was not
         * already resident. It is zero when all three were, which is the
         * steady state expert parallelism is built to reach. */
        const uint32_t staged_experts =
                any_resident ? ((gate_owned || up_owned || down_owned) ? resident_expert_count
                                                                       : 0u)
                             : unique_count;
        g_sycl_moe_last_staged_expert_count = staged_experts;
        g_sycl_moe_last_staged_bytes =
                any_resident
                        ? (((gate_owned ? 1ull : 0ull) + (up_owned ? 1ull : 0ull)) *
                                   (uint64_t)resident_expert_count * gate_expert_bytes +
                           (down_owned ? 1ull : 0ull) * (uint64_t)resident_expert_count *
                                   down_expert_bytes)
                        : (2ull * (uint64_t)unique_count * gate_expert_bytes +
                           (uint64_t)unique_count * down_expert_bytes);

        /* down->ptr doubles as the Q8_K activation scratch, then as the
         * final per-slot output once xq is no longer needed; gate->ptr
         * doubles as the mid quantisation scratch. Same reuse ds4_cuda.cu
         * relies on (moe_owned_gate/moe_owned_down comments there), kept
         * here so callers can size these tensors identically. */
        sycl_block_q8_K *xq = (sycl_block_q8_K *)down->ptr;
        sycl_block_q8_K *midq = (sycl_block_q8_K *)gate->ptr;
        sycl_moe_q8_k_quantize(q, xq, (const float *)x->ptr, expert_in_dim, 1u);

        k->gate_up_mid(q, (float *)mid->ptr, (const char *)gate_dev, (const char *)up_dev, xq,
                      remap, (const float *)weights->ptr, gate_expert_bytes, gate_row_bytes,
                      xq_blocks, expert_mid_dim, clamp);
        sycl_moe_q8_k_quantize_owned(q, midq, (const float *)mid->ptr, remap, expert_mid_dim);

        float *down_dst = (float *)(down_output ? down_output->ptr : down->ptr);
        if (pack_fixed3) {
            k->down_packed(q, down_dst, (const char *)down_dev, midq, remap,
                          (const int32_t *)selected->ptr, resident_expert_base,
                          resident_expert_count, down_expert_bytes, down_row_bytes, midq_blocks,
                          out_dim);
        } else {
            k->down_slots(q, down_dst, (const char *)down_dev, midq, remap, down_expert_bytes,
                         down_row_bytes, midq_blocks, out_dim);
        }
        /* The only reason left to drain: the compaction path's packed
         * weight tables are this function's own allocations, and the
         * guards above free them on return. The resident path -- the one a
         * real --cuda-tensor-parallel decode takes on every routed layer --
         * gets all three back as non-owning pass-throughs and so returns
         * with the kernels still in flight, which is what lets the partner
         * tier's half of this layer overlap the home tier's. */
        g_sycl_moe_owned_decode_owned_scratch = (uint32_t)gate_guard.frees() +
                                                (uint32_t)up_guard.frees() +
                                                (uint32_t)down_guard.frees();
        sycl_scratch_release_wait(q, gate_guard, up_guard, down_guard);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_one_owned failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

static int sycl_routed_moe_one_owned_q4k(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t resident_expert_base, uint32_t resident_expert_count,
        float clamp, const ds4_gpu_tensor *x, ds4_gpu_tensor *down_output, bool pack_fixed3) {
    return sycl_routed_moe_one_owned_dispatch(
            &sycl_moe_owned_decode_q4k, out, gate, up, mid, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes, expert_in_dim, expert_mid_dim, out_dim, selected,
            weights, n_total_expert, resident_expert_base, resident_expert_count, clamp, x,
            down_output, pack_fixed3);
}

static int sycl_routed_moe_one_owned_lut(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t resident_expert_base, uint32_t resident_expert_count,
        float clamp, const ds4_gpu_tensor *x, ds4_gpu_tensor *down_output, bool pack_fixed3) {
    return sycl_routed_moe_one_owned_dispatch(
            &sycl_moe_owned_decode_lut, out, gate, up, mid, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes, expert_in_dim, expert_mid_dim, out_dim, selected,
            weights, n_total_expert, resident_expert_base, resident_expert_count, clamp, x,
            down_output, pack_fixed3);
}

extern "C" int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type,
        uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim,
        uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert,
        uint32_t resident_expert_base, uint32_t resident_expert_count, float clamp,
        const ds4_gpu_tensor *x, ds4_gpu_tensor *down_output, bool pack_fixed3,
        ds4_gpu_tensor *shared_prequant) {
    /* shared_prequant is a CUDA-only perf optimisation (a cached Q8_0
     * quantisation of the activation, shared with
     * ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor's own `prequant`
     * parameter). Neither entry in this backend does activation
     * quantisation at all -- both dequantise WEIGHTS to float and dot
     * against the plain float activation instead, the same design
     * substitution ds4_sycl_hc.hpp's sycl_matmul_q8_0_hc_expand_labeled
     * already makes -- so this parameter changes performance, never
     * correctness, on this backend. Deliberately unread. */
    (void)shared_prequant;
    if (n_expert != 6u) return 0;
    if (gate_type == 12u && down_type == 12u) {
        return sycl_routed_moe_one_owned_q4k(
                out, gate, up, mid, experts, model_map, model_size, gate_offset, up_offset,
                down_offset, gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                expert_in_dim, expert_mid_dim, out_dim, selected, weights, n_total_expert,
                resident_expert_base, resident_expert_count, clamp, x, down_output, pack_fixed3);
    }
    if (gate_type == 16u && down_type == 10u) {
        return sycl_routed_moe_one_owned_lut(
                out, gate, up, mid, experts, model_map, model_size, gate_offset, up_offset,
                down_offset, gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                expert_in_dim, expert_mid_dim, out_dim, selected, weights, n_total_expert,
                resident_expert_base, resident_expert_count, clamp, x, down_output, pack_fixed3);
    }
    return 0;
}

/* ds4_gpu_routed_moe_owned_slots_combine_rows_tensor /
 * _combine_tensor, ds4_cuda.cu:21314-21357 (moe_owned_slots_combine_fixed3_
 * kernel): per selected slot, picks whichever rank's per-slot buffer
 * actually owns that slot's expert (compared against `expert_split`, NOT
 * against a remap: this runs on the ORIGINAL global selected ids, the
 * same array both ranks used to decide ownership independently) and sums
 * all six slots' picks into the final row. Generalised over `rows` (used
 * by the multi-session batch path, ds4.c's metal_graph_encode_routed_
 * session_batch); the single-row ABI entry delegates with rows=1. */
extern "C" int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *home_slots, const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected, uint32_t out_dim, uint32_t expert_split,
        uint32_t rows) {
    if (!out || !home_slots || !peer_slots || !selected || out_dim == 0u || rows == 0u ||
        rows > 65535u) {
        return 0;
    }
    uint64_t row_elems = 0, out_bytes = 0, slots_bytes = 0, selected_bytes = 0;
    if (!sycl_u64_mul_checked(rows, out_dim, &row_elems) ||
        !sycl_u64_mul_checked(row_elems, sizeof(float), &out_bytes) ||
        !sycl_u64_mul_checked(row_elems, 6u * sizeof(float), &slots_bytes) ||
        !sycl_u64_mul_checked(rows, 6u * sizeof(int32_t), &selected_bytes) ||
        out->bytes < out_bytes || home_slots->bytes < slots_bytes ||
        peer_slots->bytes < slots_bytes || selected->bytes < selected_bytes) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float *out_ptr = (float *)out->ptr;
        const float *hs = (const float *)home_slots->ptr;
        const float *ps = (const float *)peer_slots->ptr;
        const int32_t *sel = (const int32_t *)selected->ptr;
        const uint32_t od = out_dim;
        const uint32_t split = expert_split;
        sycl::event _ds4_prof_ev126 = q.parallel_for(sycl::range<1>((size_t)row_elems), [=](sycl::id<1> id) {
             const uint64_t gid = id[0];
             const uint32_t row = (uint32_t)(gid / od);
             const uint32_t col = (uint32_t)(gid - (uint64_t)row * od);
             const float *rhs = hs + (uint64_t)row * 6u * od;
             const float *rps = ps + (uint64_t)row * 6u * od;
             const int32_t *rsel = sel + (uint64_t)row * 6u;
             float acc = 0.0f;
             for (uint32_t slot = 0; slot < 6u; slot++) {
                 const int32_t expert = rsel[slot];
                 if (expert < 0 || (uint32_t)expert >= 2u * split) continue;
                 const float *src = (uint32_t)expert < split ? rhs : rps;
                 acc += src[(uint64_t)slot * od + col];
             }
             out_ptr[gid] = acc;
         });
         /* No wait: the caller's next command on this in_order queue is
          * already ordered behind this combine, and the peer half of
          * `peer_slots` was delivered by ds4_gpu_tensor_copy_xdev, which
          * submits its copy on THIS queue and so is ordered ahead of it. */
         ds4_sycl_profile_record(_ds4_prof_ev126);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_owned_slots_combine_rows failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *home_slots, const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected, uint32_t out_dim, uint32_t expert_split) {
    return ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(out, home_slots, peer_slots,
                                                              selected, out_dim, expert_split,
                                                              1u);
}

/* ds4_gpu_routed_moe_owned_packed_combine_tensor, ds4_cuda.cu:21359-21384:
 * decode-only (no row count -- the packed variant is only ever used for
 * n_tok==1, see ds4.c's cuda_tp_ep decode call sequence). */
extern "C" int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *home_slots, const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected, uint32_t out_dim, uint32_t expert_split) {
    const uint64_t home_bytes = 6ull * out_dim * sizeof(float);
    const uint64_t peer_bytes = 4ull * out_dim * sizeof(float);
    if (!out || !home_slots || !peer_packed || !selected || out_dim == 0u ||
        out->bytes < (uint64_t)out_dim * sizeof(float) || home_slots->bytes < home_bytes ||
        peer_packed->bytes < peer_bytes || selected->bytes < 6u * sizeof(int32_t)) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(out->device_id);
        float *out_ptr = (float *)out->ptr;
        const float *hs = (const float *)home_slots->ptr;
        const float *pp = (const float *)peer_packed->ptr;
        const int32_t *sel = (const int32_t *)selected->ptr;
        const uint32_t od = out_dim;
        const uint32_t split = expert_split;
        sycl::event _ds4_prof_ev127 = q.parallel_for(sycl::range<1>((size_t)out_dim), [=](sycl::id<1> id) {
             const uint32_t row = (uint32_t)id[0];
             out_ptr[row] = sycl_moe_owned_packed_combine_row(hs, pp, sel, row, od, split);
         });
         /* No wait, same reasoning as the per-slot combine above. */
         ds4_sycl_profile_record(_ds4_prof_ev127);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_owned_packed_combine failed: %s\n",
                e.what());
        return 0;
    }
    return 1;
}

/* moe_handoff_pack_kernel, ds4_cuda.cu:3177-3194: packs ffn_norm,
 * selected and weights into one contiguous buffer for a single
 * cross-device copy, matching the byte layout ds4.c's cuda_tp_moe_pack_
 * handoff branch reads back (norm_bytes, then selected, then weights). */
extern "C" int ds4_gpu_moe_handoff_pack_tensor(ds4_gpu_tensor *packed,
                                               const ds4_gpu_tensor *ffn_norm,
                                               const ds4_gpu_tensor *selected,
                                               const ds4_gpu_tensor *weights, uint32_t n_embd,
                                               uint32_t n_expert) {
    if (!packed || !ffn_norm || !selected || !weights || n_embd == 0u || n_expert == 0u) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)n_embd * sizeof(float) +
                           (uint64_t)n_expert * sizeof(int32_t) +
                           (uint64_t)n_expert * sizeof(float);
    if (packed->bytes < bytes || ffn_norm->bytes < (uint64_t)n_embd * sizeof(float) ||
        selected->bytes < (uint64_t)n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_expert * sizeof(float)) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(packed->device_id);
        unsigned char *dst = (unsigned char *)packed->ptr;
        const float *norm = (const float *)ffn_norm->ptr;
        const int32_t *sel = (const int32_t *)selected->ptr;
        const float *w = (const float *)weights->ptr;
        const uint32_t ne = n_embd, nx = n_expert;
        const uint64_t sel_off = (uint64_t)ne * sizeof(float);
        const uint64_t w_off = sel_off + (uint64_t)nx * sizeof(int32_t);
        const uint32_t n = ne > nx ? ne : nx;
        sycl::event _ds4_prof_ev128 = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
             const uint32_t i = (uint32_t)id[0];
             if (i < ne) {
                 *(float *)(dst + (uint64_t)i * sizeof(float)) = norm[i];
             }
             if (i < nx) {
                 *(int32_t *)(dst + sel_off + (uint64_t)i * sizeof(int32_t)) = sel[i];
                 *(float *)(dst + w_off + (uint64_t)i * sizeof(float)) = w[i];
             }
         });
         /* No wait: the cross-device copy that reads `packed` next is
          * ds4_gpu_tensor_copy_xdev, which opens with a barrier on the
          * SOURCE queue (this one) precisely so a still-pending producer
          * like this kernel is ordered ahead of the copy. */
         ds4_sycl_profile_record(_ds4_prof_ev128);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "moe_handoff_pack failed: %s\n", e.what());
        return 0;
    }
    return 1;
}

/* moe_filter_owned_pairs_kernel, ds4_cuda.cu:19534-19552: rewrites a
 * (token,slot) pair's selected id to be LOCAL to the owned half when
 * owned, or to -1 with its weight zeroed when not. This is what lets
 * ds4_gpu_routed_moe_batch_owned_tensor below delegate to the exact same
 * multi-format sycl_routed_moe_launch the non-owned batch entry uses
 * (ds4_sycl_moe_launch.hpp): a filtered pair's weight is 0, so whatever
 * (wrong, since the id was reset to a real-but-arbitrary local expert 0)
 * value its gate/up/down projection computes is multiplied by zero before
 * it ever reaches the final per-token sum, and the selected-expert
 * compaction (already inside sycl_routed_moe_launch) then further restricts staging
 * to the distinct owned-and-selected experts, not the whole owned half. */
static void sycl_moe_filter_owned_pairs(sycl::queue &q, int32_t *selected, float *weights,
                                        uint64_t pair_count, uint32_t n_total_expert,
                                        uint32_t expert_base, uint32_t expert_count) {
    if (pair_count == 0u) return;
    sycl::event _ds4_prof_ev129 = q.parallel_for(sycl::range<1>((size_t)pair_count), [=](sycl::id<1> id) {
         const uint64_t pair = id[0];
         const int32_t expert_i = selected[pair];
         if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert &&
             (uint32_t)expert_i >= expert_base &&
             (uint32_t)expert_i - expert_base < expert_count) {
             selected[pair] = expert_i - (int32_t)expert_base;
         } else {
             selected[pair] = -1;
             weights[pair] = 0.0f;
         }
     });
     /* No wait: sycl_routed_moe_launch below reads `selected` back on this
      * same in_order queue, so its readback is ordered behind this
      * rewrite. */
     ds4_sycl_profile_record(_ds4_prof_ev129);
}

/* ds4_gpu_routed_moe_batch_owned_tensor, ds4_cuda.cu:25416-25493: unlike
 * the decode entry above, CUDA's own implementation is a thin wrapper
 * around the ordinary (non-owned) routed_moe_launch, not a bespoke owned
 * kernel family -- see the block comment on sycl_moe_filter_owned_pairs
 * for why that composes correctly. Porting it the same way means every
 * quantised format sycl_routed_moe_launch already supports (Q4_K,
 * IQ2_XXS/Q2_K, standalone Q2_K, MXFP4) works here too, not just Q4_K. */
extern "C" int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type,
        uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim,
        uint32_t expert_mid_dim, uint32_t out_dim, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert,
        uint32_t resident_expert_base, uint32_t resident_expert_count, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!selected || !weights || n_tokens == 0u || n_expert == 0u || n_total_expert == 0u ||
        resident_expert_count == 0u || gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base) {
        return 0;
    }
    uint64_t pair_count = 0;
    if (!sycl_u64_mul_checked(n_tokens, n_expert, &pair_count) || pair_count > UINT32_MAX ||
        selected->bytes < pair_count * sizeof(int32_t) ||
        weights->bytes < pair_count * sizeof(float)) {
        return 0;
    }
    uint64_t gate_shift = 0, down_shift = 0, gate_offset_shifted = 0, up_offset_shifted = 0,
             down_offset_shifted = 0;
    if (!sycl_u64_mul_checked(resident_expert_base, gate_expert_bytes, &gate_shift) ||
        !sycl_u64_mul_checked(resident_expert_base, down_expert_bytes, &down_shift) ||
        !sycl_u64_add_checked(gate_offset, gate_shift, &gate_offset_shifted) ||
        !sycl_u64_add_checked(up_offset, gate_shift, &up_offset_shifted) ||
        !sycl_u64_add_checked(down_offset, down_shift, &down_offset_shifted)) {
        return 0;
    }
    if (g_devices.empty()) return 0;
    try {
        sycl::queue &q = ds4_sycl_queue(selected->device_id);
        sycl_moe_filter_owned_pairs(q, (int32_t *)selected->ptr, (float *)weights->ptr,
                                    pair_count, n_total_expert, resident_expert_base,
                                    resident_expert_count);
    } catch (const sycl::exception &e) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "owned routed_moe pair filter failed: %s\n",
                e.what());
        return 0;
    }
    return sycl_routed_moe_launch(out, gate, up, mid, experts, model_map, model_size,
                                  gate_offset_shifted, up_offset_shifted, down_offset_shifted,
                                  gate_type, down_type, gate_expert_bytes, gate_row_bytes,
                                  down_expert_bytes, down_row_bytes, expert_in_dim,
                                  expert_mid_dim, out_dim, selected, weights,
                                  resident_expert_count, n_expert, clamp, x, layer_index,
                                  n_tokens, /*force_resident=*/false);
}

}  // namespace

/* Test-only instrumentation: how many device allocations the most
 * recently completed owned decode dispatch still owned as it returned,
 * which is 0 exactly when it returned without draining its queue. See
 * g_sycl_moe_owned_decode_owned_scratch above. */
extern "C" uint32_t ds4_sycl_moe_test_owned_decode_owned_scratch(void) {
    return g_sycl_moe_owned_decode_owned_scratch;
}

/* Test-only instrumentation: how many device-to-host readbacks the most
 * recently completed owned decode dispatch performed, which must be 0 on
 * the resident path. See g_sycl_moe_owned_decode_readbacks above. */
extern "C" uint32_t ds4_sycl_moe_test_owned_decode_readbacks(void) {
    return g_sycl_moe_owned_decode_readbacks;
}
