/* Synthetic ds4_weights / ds4_layer_weights scaffolding for a multi-layer
 * DeepSeek V4 Flash decoder, at a small shape sized to drive
 * metal_graph_encode_decode_layer_phase (ds4.c:21856) and the output head
 * (metal_graph_encode_output_head, ds4.c:25147), the real per-token decode
 * path, with no model file.  Pure C, no SYCL or C++-only includes: this is
 * included both from plain C test files and, under DS4_TEST_HOOKS, from
 * ds4.c itself, which every GPU backend compiles with a plain C compiler.
 *
 * Two DS4_TEST_HOOKS hooks build a ds4_weights and ds4_layer_weights[] by
 * pointing ds4_tensor fields at the abs_offset/dim/type values this header
 * fills in, exactly as ds4_test_graph_alloc_smoke and
 * ds4_test_graph_encode_smoke (ds4.c:57544, ds4.c:57671) already do by hand
 * for their own smaller synthetic tensors: ds4_test_graph_full_layer_encode
 * drives one layer directly, ds4_test_graph_full_token_encode drives every
 * layer plus the output head through metal_graph_eval_token_raw_swa.
 *
 * The quantisation combination matches Flash as shipped: dense weights
 * (attn_q_a, attn_q_b, attn_kv, attn_output_a, attn_output_b,
 * ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, output) are Q8_0; the
 * router (ffn_gate_inp) is F16; routed experts are IQ2_XXS gate/up and
 * Q2_K down; the hash routing table (ffn_gate_tid2eid, layers below
 * DS4_TW_N_HASH_LAYER only) is I32; norms, hyper-connection (HC)
 * scale/base, attn_sinks and the output head's HC/norm tensors are F32
 * (output_hc_fn is F16, matching a real model's output_hc_fn.weight).
 *
 * The IQ2_XXS block packer, the Q2_K block packer, the Q8_0 row encoder,
 * and the non-affine float filler below are ports of already-existing,
 * already-exercised test helpers, not new byte layouts or a new fill
 * strategy: they reproduce, line for line, tests/test_sycl_moe.c's
 * f32_to_f16_bits (:113), oracle_iq2_pack_ib32 (:1693),
 * oracle_iq2_pack_block (:1707), iq2_fill_row (:1715),
 * oracle_q2k_pack_block (:1770), q2k_fill_row (:1813);
 * tests/test_sycl_matmul.c's test_encode_q8_0_row (:220, adapted to call
 * this header's f32_to_f16_bits instead of that file's test_float_to_half
 * -- the two are not the same conversion, see the note above
 * f32_to_f16_bits below); and tests/test_sycl_hc.c's fill_val (:147),
 * whose non-affine shape (spec 6f/6n) is exactly what this header's own
 * float filler needs and so replaces the header's original ad hoc hash.
 * They are reproduced here rather than shared by extraction because
 * test_sycl_moe.c documents itself as deliberately self-contained ("no
 * tests/test_sycl_harness.h on this branch's base commit") and every SYCL
 * test file in this suite follows the same precedent of a cited, local
 * reimplementation rather than a cross-file header dependency
 * (test_sycl_moe.c's own top comment cites test_sycl_kernels.c for this),
 * so this header follows that established convention instead of
 * retrofitting test_sycl_moe.c, test_sycl_matmul.c, or test_sycl_hc.c.
 * A real extraction (this header included by all three, their local
 * copies deleted) was tried and reverted: it would edit three already-
 * landed, already-ablated test files to serve a fourth, which risks the
 * tested code to tidy the testing code, and a later, deliberate
 * consolidation pass across the whole suite is the right place for it,
 * not here. */

#ifndef DS4_TEST_SYCL_LAYER_WEIGHTS_H
#define DS4_TEST_SYCL_LAYER_WEIGHTS_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- The synthetic shape ------------------------------------------------
 *
 * Every layer this header builds carries attention compressor tensors,
 * and layers where ds4_tw_compress_ratio(il) == 4 additionally carry
 * indexer tensors (see that function's own comment below): Flash's own
 * ratio pattern is "0, 0, 4, 128, 4, 128, ..." and metal_graph_encode_
 * decode_layer_phase only dereferences layer->attn_compressor_... /
 * layer->indexer_... when ds4_layer_compress_ratio(il) != 0, so a caller
 * that leaves g_ds4_compress_ratios at its all-zero DS4_TEST_HOOKS default
 * exercises the exact same ratio-0 configuration this header built before
 * these tensors existed, from the same weight buffer a caller that sets
 * per-layer ratios uses to exercise the compressed path -- one buffer, two
 * configurations, so a difference in engine output between them can only
 * come from the ratio flag, never from different underlying weights.
 * n_embd == n_ff_exp == 256 so every routed-expert row needs
 * exactly one 256-wide quant block (kMoeQK, sycl/ds4_sycl_moe.hpp:255;
 * build_plan rejects expert_in_dim/expert_mid_dim not a multiple of it,
 * sycl/ds4_sycl_moe_launch.hpp:57-58).  n_lora_q and n_lora_o are
 * multiples of 32 so the Q8_0 dense rows they become (attn_q_b via
 * DS4_N_LORA_Q, attn_output_b via DS4_N_OUT_GROUP*DS4_N_LORA_O) are
 * block-aligned (Q8_0 blocks are 32 elements, ds4.c:2028).
 *
 * A caller building the matching ds4_shape (only ds4.c can, ds4_shape is
 * private to that translation unit) should use these same values for
 * every field below, plus n_ff_dense=32, n_hash_layer=DS4_TW_N_HASH_LAYER,
 * n_swa=8, small n_indexer_* values, and the DS4_DEFAULT_* constants for
 * everything else -- the same pattern ds4_test_graph_alloc_smoke
 * (ds4.c:57544) uses -- so the shape fed to the engine and the tensors this
 * header builds never drift apart.
 *
 * DS4_TW_N_LAYER is 4 and DS4_TW_N_HASH_LAYER is 3, matching Flash's own
 * n_hash_layer (ds4.c:566): layers 0-2 route by hash and require an
 * ffn_gate_tid2eid table (ds4.c:4910, ds4.c:5948), layer 3 routes by top-k.
 * A single layer can never exercise the layer-to-layer hyper-connection
 * carry or the hash-routing path at all, so a full-token test needs both. */
#define DS4_TW_N_LAYER              4u
#define DS4_TW_N_HASH_LAYER         3u
#define DS4_TW_N_EMBD                256u
#define DS4_TW_N_VOCAB               128u
#define DS4_TW_N_HEAD                2u
#define DS4_TW_N_HEAD_KV             1u
#define DS4_TW_N_HEAD_DIM            32u
#define DS4_TW_N_VALUE_DIM           32u
#define DS4_TW_N_ROT                 16u
#define DS4_TW_N_OUT_GROUP           1u
#define DS4_TW_N_LORA_Q              64u
#define DS4_TW_N_LORA_O              32u
#define DS4_TW_N_EXPERT            256u
#define DS4_TW_N_EXPERT_USED         6u
#define DS4_TW_N_EXPERT_SHARED       1u
#define DS4_TW_N_FF_EXP              256u
#define DS4_TW_N_HC                  4u
#define DS4_TW_N_HC_SINKHORN_ITER    1u
#define DS4_TW_N_SWA                 8u
/* Matches the n_indexer_head / n_indexer_head_dim / n_indexer_top_k
 * literals every DS4_TEST_HOOKS entry point in ds4.c already sets on its
 * own g_ds4_shape (ds4.c:57614-57616 and its four siblings), so this
 * header's own compressor/indexer tensor shapes below and the shape the
 * engine is actually configured with cannot drift apart. */
#define DS4_TW_N_INDEXER_HEAD        2u
/* Fixed at 128, not model-configurable: every real Flash/GLM-DSA GGUF
 * preset in this file sets n_indexer_head_dim to exactly 128 (ds4.c:569,
 * 607, 646, 689), and sycl/ds4_sycl_indexer.hpp's Hadamard+FP4 QAT round
 * trip (ds4_gpu_dsv4_indexer_qat_tensor, its own kernel at :144) hard-
 * rejects any other value with `head_dim != 128u`: the transform is a
 * fixed 128-wide hardware trick, not a parameterised kernel. This is also
 * comfortably >= DS4_TW_N_ROT (16): sycl/ds4_sycl_compressor.hpp's
 * ds4_gpu_compressor_update_tensor rejects n_rot > head_dim, since the
 * indexer's RoPE tail rotates n_indexer_head_dim by n_rot exactly as the
 * main attention path rotates n_head_dim. The smaller value (8, later 16)
 * this header used before the indexer weights existed was never
 * validated against either invariant, because the compressor/indexer code
 * that checks them had never executed. */
#define DS4_TW_N_INDEXER_HEAD_DIM    128u
#define DS4_TW_N_INDEXER_TOP_K       4u

/* Derived exactly as weights_validate_layout derives them (ds4.c:5027-
 * 5030): hc_dim = n_embd*n_hc, hc_mix_dim = 2*n_hc + n_hc*n_hc, q_dim =
 * n_head*n_head_dim, out_low_dim = n_out_group*n_lora_o. */
#define DS4_TW_HC_DIM       (DS4_TW_N_EMBD * DS4_TW_N_HC)
#define DS4_TW_HC_MIX_DIM   (2u * DS4_TW_N_HC + DS4_TW_N_HC * DS4_TW_N_HC)
#define DS4_TW_Q_DIM        (DS4_TW_N_HEAD * DS4_TW_N_HEAD_DIM)
#define DS4_TW_OUT_LOW_DIM  (DS4_TW_N_OUT_GROUP * DS4_TW_N_LORA_O)

/* Flash's own per-layer compression ratio (ds4_expected_layer_compress_
 * ratio, ds4.c:1086-1096, DS4_VARIANT_FLASH case): layers 0 and 1 are
 * uncompressed, even layers from 2 up use ratio 4 (compressor + indexer),
 * odd layers from 3 up use ratio 128 (compressor only, no indexer --
 * weights_layer_has_required, ds4.c:4888-4907, only requires the six
 * indexer_* fields when ratio == 4, never for ratio == 128). Reproduced
 * here, not included from ds4.c, because ds4_expected_layer_compress_ratio
 * is a static function private to that translation unit; this is the same
 * literal switch, just addressable from a plain-C test file. With
 * DS4_TW_N_LAYER == 4 this gives the exact "0, 0, 4, 128" pattern
 * required: both ratio-0 layers, plus one of each compressed regime. */
static inline uint32_t ds4_tw_compress_ratio(uint32_t il) {
    if (il < 2u) return 0u;
    return (il & 1u) == 0u ? 4u : 128u;
}

/* A decode position whose (pos + 1) is divisible by both compressed
 * ratios this shape uses (4 and 128; lcm(4, 128) - 1 == 127), so a SINGLE
 * one-shot decode call (ds4_test_graph_full_token_encode_impl builds a
 * fresh graph and calls metal_graph_eval_token_raw_swa exactly once, at
 * whatever pos it is given -- there is no persistent multi-step decode
 * loop here) reaches "emit" (ds4.c:22613's `((pos + 1u) % ratio) == 0u`)
 * for layer 2 (ratio 4) and layer 3 (ratio 128) at the same time, in the
 * same call. At pos 0 -- every other decode hook's fixed position --
 * emit is false for every ratio > 1, so the attention compressor's own
 * output cache stays permanently empty and the compressed and ratio-0
 * paths compute identically by construction: pos 0 cannot distinguish
 * them, per design-spec 6w. This position can, and the compressed
 * decode assertion uses it for exactly that reason. */
#define DS4_TW_COMPRESSED_DECODE_POS 127u

/* ---- Tensor type codes this layer uses -----------------------------------
 *
 * Raw GGUF/ds4 type ids (ds4.c:2061-2073), not an enum, so this header can
 * be included from ds4.c -- where DS4_TENSOR_F32 etc. are already an enum
 * -- without a name collision; the numeric values are identical either
 * way. */
#define DS4_TW_TYPE_F32      0u
#define DS4_TW_TYPE_F16      1u
#define DS4_TW_TYPE_Q8_0     8u
#define DS4_TW_TYPE_Q2_K     10u
#define DS4_TW_TYPE_IQ2_XXS  16u
#define DS4_TW_TYPE_I32      26u

/* elements-per-block/bytes-per-block for exactly the six types above,
 * mirroring the relevant entries of ds4.c's gguf_types table (ds4.c:2028-
 * 2054) and its tensor_nbytes formula (ds4.c:2182-2188): bytes =
 * ceil(elements/block_elems) * block_bytes.  Every tensor built below has
 * dim[0] itself divisible by the type's block_elems, so this whole-tensor
 * formula and a per-row-block-aligned one agree exactly. */
static inline uint64_t ds4_tw_tensor_bytes(uint32_t type, uint64_t elements) {
    uint32_t block_elems = 0, block_bytes = 0;
    switch (type) {
        case DS4_TW_TYPE_F32:     block_elems = 1;   block_bytes = 4;  break;
        case DS4_TW_TYPE_F16:     block_elems = 1;   block_bytes = 2;  break;
        case DS4_TW_TYPE_Q8_0:    block_elems = 32;  block_bytes = 34; break;
        case DS4_TW_TYPE_Q2_K:    block_elems = 256; block_bytes = 84; break;
        case DS4_TW_TYPE_IQ2_XXS: block_elems = 256; block_bytes = 66; break;
        case DS4_TW_TYPE_I32:     block_elems = 1;   block_bytes = 4;  break;
        default: return 0;
    }
    const uint64_t blocks = (elements + block_elems - 1u) / block_elems;
    return blocks * (uint64_t)block_bytes;
}

/* ---- Block encoders, ported not shared -------------------------------------------------
 *
 * See the file header comment above for why these are ported (cited,
 * line-for-line copies) rather than shared by extraction. */

/* Minimal round-to-nearest-even IEEE754 binary16 encoder, sufficient for
 * the finite, moderate-magnitude values used below (no inf/NaN/subnormal
 * handling needed).  Ported from tests/test_sycl_moe.c:113
 * (f32_to_f16_bits).  Not the same conversion as
 * tests/test_sycl_matmul.c's test_float_to_half (:30), which truncates
 * the mantissa instead of rounding it; this header only needs one of the
 * two, so it takes the rounding one and does not also port the
 * truncating one. */
static inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign; /* flush to zero, not needed here */
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t rounded_mant = (uint16_t)(mant >> 13);
    if (mant & 0x1000u) rounded_mant++; /* round to nearest, ties up */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | rounded_mant);
}

/* Packs one ib32 group (32 elements: 4 grid indices, 4 sign indices, 1
 * scale nibble) into the 4 uint16 codes the real IQ2_XXS format uses.
 * Ported from tests/test_sycl_moe.c:1693 (oracle_iq2_pack_ib32). */
static inline void oracle_iq2_pack_ib32(uint16_t out4[4], uint8_t a0, uint8_t a1,
                                        uint8_t a2, uint8_t a3, uint32_t s0,
                                        uint32_t s1, uint32_t s2, uint32_t s3,
                                        uint32_t ls_nibble) {
    uint32_t aux0 = (uint32_t)a0 | ((uint32_t)a1 << 8) | ((uint32_t)a2 << 16) | ((uint32_t)a3 << 24);
    uint32_t aux1 = (s0 & 127u) | ((s1 & 127u) << 7) | ((s2 & 127u) << 14) | ((s3 & 127u) << 21) |
                    ((ls_nibble & 15u) << 28);
    out4[0] = (uint16_t)(aux0 & 0xffffu);
    out4[1] = (uint16_t)(aux0 >> 16);
    out4[2] = (uint16_t)(aux1 & 0xffffu);
    out4[3] = (uint16_t)(aux1 >> 16);
}

/* Packs one 66-byte block_iq2_xxs (ds4.c: gguf_types[16], iq2_xxs, 256
 * elements per block, 66 bytes): d (f16) plus 32 packed u16 codes, 8
 * groups of 4.  Ported from tests/test_sycl_moe.c:1707
 * (oracle_iq2_pack_block). */
static inline void oracle_iq2_pack_block(uint8_t out[66], float d, const uint16_t qs[32]) {
    uint16_t dh = f32_to_f16_bits(d);
    memcpy(out, &dh, 2);
    memcpy(out + 2, qs, 64);
}

/* Fills one real, non-zero, hashed 66-byte IQ2_XXS block for the 256-wide
 * chunk selected by `blk`, salted distinctly by phase/expert/row/blk so
 * different rows and experts do not coincide.  Ported from
 * tests/test_sycl_moe.c:1715 (iq2_fill_row). */
static inline void iq2_fill_row(uint8_t out[66], uint32_t phase, uint32_t expert,
                                uint32_t row, uint32_t blk) {
    uint16_t qs[32];
    for (uint32_t g = 0; g < 8u; g++) {
        uint8_t a0 = (uint8_t)((phase * 7u + expert * 13u + row * 5u + blk * 61u + g * 3u) % 256u);
        uint8_t a1 = (uint8_t)((phase * 11u + expert * 17u + row * 7u + blk * 67u + g * 5u +
                                (expert * row) % 17u) % 256u);
        uint8_t a2 = (uint8_t)((phase * 13u + expert * 19u + row * 11u + blk * 71u + g * 7u) % 256u);
        uint8_t a3 = (uint8_t)((phase * 17u + expert * 23u + row * 13u + blk * 73u + g * 11u +
                                (g * row) % 13u) % 256u);
        uint32_t s0 = (phase + expert * 3u + row * 2u + blk * 19u + g) % 128u;
        uint32_t s1 = (phase * 2u + expert + row * 5u + blk * 23u + g * 2u) % 128u;
        uint32_t s2 = (phase * 3u + expert * 5u + row + blk * 29u + g * 3u) % 128u;
        uint32_t s3 = (phase * 5u + expert * 2u + row * 3u + blk * 37u + g * 5u) % 128u;
        uint32_t ls_nibble = (phase + expert + row + blk * 5u + g) % 16u;
        oracle_iq2_pack_ib32(&qs[g * 4u], a0, a1, a2, a3, s0, s1, s2, s3, ls_nibble);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row + blk * 3u) % 7u);
    oracle_iq2_pack_block(out, d, qs);
}

/* Packs one 84-byte block_q2_K (ds4.c: gguf_types[10], q2_k, 256 elements
 * per block, 84 bytes): 16 bytes of packed (scale, min) nibble pairs, 64
 * bytes of 2-bit codes, d and dmin as f16.  The per-element byte/shift
 * placement matches ds4.c's own accessor q2_k_value_f32 (ds4.c:3494-3505:
 * group = idx/16, q_base = 32*(group/8) + 16*(group&1), shift =
 * ((group/2)&3)*2).  Ported from tests/test_sycl_moe.c:1770
 * (oracle_q2k_pack_block). */
static inline void oracle_q2k_pack_block(uint8_t out[84], float d, float dmin,
                                         const uint8_t scales[16], const uint8_t nib[256]) {
    memset(out, 0, 84);
    memcpy(out, scales, 16);
    for (uint32_t idx = 0; idx < 256u; idx++) {
        uint32_t group = idx / 16u;
        uint32_t l = idx - group * 16u;
        uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        uint32_t shift = ((group / 2u) & 3u) * 2u;
        out[16u + q_base + l] = (uint8_t)(out[16u + q_base + l] | ((nib[idx] & 3u) << shift));
    }
    uint16_t dh = f32_to_f16_bits(d), dminh = f32_to_f16_bits(dmin);
    memcpy(out + 80, &dh, 2);
    memcpy(out + 82, &dminh, 2);
}

/* Fills one real, non-zero, hashed 84-byte Q2_K block for the 256-wide
 * chunk selected by `blk`, salted distinctly by phase/expert/row/blk.
 * Ported from tests/test_sycl_moe.c:1813 (q2k_fill_row). */
static inline void q2k_fill_row(uint8_t out[84], uint32_t phase, uint32_t expert,
                                uint32_t row, uint32_t blk) {
    uint8_t scales[16], nib[256];
    for (uint32_t g = 0; g < 16u; g++) {
        uint8_t sc = (uint8_t)(1u + (phase * 5u + expert * 7u + row * 3u + blk * 43u +
                                     g * 5u) % 15u);
        uint8_t mn = (uint8_t)((phase * 3u + expert * 11u + row * 2u + blk * 47u +
                               g * 13u) % 15u);
        scales[g] = (uint8_t)((sc & 0x0fu) | (uint8_t)(mn << 4));
    }
    for (uint32_t k = 0; k < 256u; k++) {
        nib[k] = (uint8_t)((phase * 19u + expert * 13u + row * 17u + blk * 59u + k +
                            (expert * row) % 5u + (k * row) % 7u +
                            (k * blk) % 11u) % 4u);
    }
    const float d = 0.01f + 0.001f * (float)((phase + expert + row + blk * 3u) % 7u);
    const float dmin = 0.001f * (float)((phase + expert * 2u + row + blk * 2u) % 5u);
    oracle_q2k_pack_block(out, d, dmin, scales, nib);
}

/* Fills one real, non-zero, hashed Q8_0 row: blocks of 32 values, 34 bytes
 * per block, a little-endian F16 scale in bytes 0-1 followed by 32 signed
 * int8 values, matching sycl_q8_0_dequant (sycl/ds4_sycl_common.hpp).
 * Ported from tests/test_sycl_matmul.c:220 (test_encode_q8_0_row), with
 * its internal scale encode changed from that file's local
 * test_float_to_half (truncating) to this header's f32_to_f16_bits
 * (rounding) so this header does not also need to port the truncating
 * conversion; either is a valid Q8_0 scale, the row is still a real,
 * self-consistent, non-zero block. */
static inline void test_encode_q8_0_row(unsigned char *row, uint32_t in_dim, uint32_t o) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        unsigned char *bp = row + (size_t)blk * 34u;
        const uint16_t braw = f32_to_f16_bits(0.05f * (float)(o + blk + 1u));
        bp[0] = (unsigned char)(braw & 0xFFu);
        bp[1] = (unsigned char)((braw >> 8) & 0xFFu);
        for (uint32_t idx = 0; idx < 32u; idx++) {
            const uint32_t k = blk * 32u + idx;
            const int qv = (k < in_dim)
                    ? (int)(((o + 1u) * (k + 3u)) % 13u) - 6
                    : 0;
            bp[2 + idx] = (unsigned char)(signed char)qv;
        }
    }
}

/* Deterministic pseudo-random floats in a small range, with a non-affine
 * per-element term so no two rows are proportional to each other: per
 * design-spec section 6f, an ablation downstream of a normalisation
 * (Sinkhorn row/column normalisation here) can be laundered invisible by
 * purely affine test data.  Ported from tests/test_sycl_hc.c:147
 * (fill_val). */
static inline float fill_val(uint32_t i, uint32_t j) {
    return sinf((float)(i * 7 + j * 13 + 1)) * 0.7f + 0.05f * (float)((i * j) % 11);
}

/* Deterministic, non-zero, small-amplitude float fill for the plain F32/
 * F16 tensors (norms, HC scale/base, attn_sinks, the router, the HC mix
 * matrices, token_embd) that have no pre-existing block encoder to reuse.
 * `salt` distinguishes one tensor from another so no two tensors end up
 * with the same values by coincidence, the same technique
 * tests/test_sycl_hc.c itself uses at its own fill_val call sites (an
 * offset folded into the second argument, e.g. fill_val(t, h + 50));
 * this header ports fill_val itself (see above) rather than inventing
 * its own hash, per the survey this task requires.
 * scale/center rescale fill_val's own non-affine output uniformly, which
 * does not reintroduce the affine-ablation-hiding problem fill_val
 * itself avoids. */
static inline void ds4_tw_fill_f32(float *dst, uint64_t n, uint32_t salt, float center, float scale) {
    for (uint64_t i = 0; i < n; i++) dst[i] = center + scale * fill_val((uint32_t)i, salt);
}

static inline void ds4_tw_fill_f16(uint16_t *dst, uint64_t n, uint32_t salt, float center, float scale) {
    for (uint64_t i = 0; i < n; i++) dst[i] = f32_to_f16_bits(center + scale * fill_val((uint32_t)i, salt));
}

/* Fills a hash routing table (ffn_gate_tid2eid): int32, dim = [n_expert_used,
 * n_vocab], physically laid out token-major (n_vocab rows of n_expert_used
 * ids each) per layer_hash_selected_experts (ds4.c:10699, "table +
 * token * DS4_N_EXPERT_USED"). Every id is in [0, n_expert) and, within one
 * token's row, all n_expert_used ids are pairwise distinct: 41 is odd and
 * therefore coprime with any power of two, so `i * 41 mod n_expert` visits
 * n_expert_used distinct residues for the small n_expert_used this header
 * uses. `layer_salt` keeps different hash-routed layers from picking
 * identical experts for the same token. */
static inline void ds4_tw_fill_hash_tid2eid(int32_t *dst, uint32_t n_vocab,
                                            uint32_t n_expert_used, uint32_t n_expert,
                                            uint32_t layer_salt) {
    for (uint32_t tok = 0; tok < n_vocab; tok++) {
        const uint32_t base = (layer_salt * 29u + tok * 97u) % n_expert;
        for (uint32_t i = 0; i < n_expert_used; i++) {
            dst[(uint64_t)tok * n_expert_used + i] = (int32_t)((base + i * 41u) % n_expert);
        }
    }
}

/* ---- Tensor descriptors ---------------------------------------------------
 *
 * Carries exactly the ds4_tensor fields metal_graph_encode_decode_layer_phase
 * (and the token-embedding lookup that precedes it) reads: abs_offset,
 * type, ndim, dim[]; plus elements/bytes, since the phase function also
 * reads ->bytes directly for attn_sinks and the routed-expert tensors
 * (ds4.c:21856 region, e.g. "layer->attn_sinks->bytes / DS4_N_HEAD" and
 * "layer->ffn_gate_exps->bytes").  A caller populates a real ds4_tensor by
 * copying these five fields across (ds4_tensor.dim is sized DS4_MAX_DIMS
 * (8) in ds4.c; every tensor here has ndim <= 3, so only dim[0..2] are
 * ever meaningful). */
typedef struct {
    uint64_t abs_offset;
    uint64_t bytes;
    uint64_t elements;
    uint32_t type;
    uint32_t ndim;
    uint64_t dim[3];
} ds4_tw_tensor;

typedef struct {
    ds4_tw_tensor hc_attn_fn;
    ds4_tw_tensor hc_attn_scale;
    ds4_tw_tensor hc_attn_base;
    ds4_tw_tensor attn_norm;
    ds4_tw_tensor attn_q_a;
    ds4_tw_tensor attn_q_a_norm;
    ds4_tw_tensor attn_q_b;
    ds4_tw_tensor attn_kv;
    ds4_tw_tensor attn_kv_a_norm;
    ds4_tw_tensor attn_sinks;
    ds4_tw_tensor attn_output_a;
    ds4_tw_tensor attn_output_b;
    ds4_tw_tensor hc_ffn_fn;
    ds4_tw_tensor hc_ffn_scale;
    ds4_tw_tensor hc_ffn_base;
    ds4_tw_tensor ffn_norm;
    ds4_tw_tensor ffn_gate_inp;
    ds4_tw_tensor ffn_gate_exps;
    ds4_tw_tensor ffn_up_exps;
    ds4_tw_tensor ffn_down_exps;
    ds4_tw_tensor ffn_gate_shexp;
    ds4_tw_tensor ffn_up_shexp;
    ds4_tw_tensor ffn_down_shexp;
    /* Only reserved and filled for layers below DS4_TW_N_HASH_LAYER; left
     * zeroed (abs_offset/bytes/elements all 0) for top-k layers, exactly
     * like a real ds4_layer_weights leaves the pointer NULL. */
    ds4_tw_tensor ffn_gate_tid2eid;
    /* Attention compressor and indexer, reserved and filled only for
     * layers where ds4_tw_compress_ratio(il) says they apply, left zeroed
     * otherwise. ds4_layer_weights (ds4.c:4126-4138) also declares
     * indexer_attn_k, indexer_k_norm and indexer_k_norm_b, but those three
     * are read only by weights_glm_dsa_layer_has_required and the GLM-DSA
     * graph path (ds4.c:4809-4823, 40404-40406, 44599-47569); Flash is
     * DS4_MODEL_FAMILY_DEEPSEEK4, whose weights_layer_has_required
     * (ds4.c:4888-4907) and weights_validate_layout (ds4.c:5058-5075)
     * never reference them, so this header does not build them -- adding
     * three tensors nothing on Flash's path ever reads would not add
     * coverage, only dead scaffolding. */
    ds4_tw_tensor attn_compressor_ape;
    ds4_tw_tensor attn_compressor_kv;
    ds4_tw_tensor attn_compressor_gate;
    ds4_tw_tensor attn_compressor_norm;
    ds4_tw_tensor indexer_attn_q_b;
    ds4_tw_tensor indexer_proj;
    ds4_tw_tensor indexer_compressor_ape;
    ds4_tw_tensor indexer_compressor_kv;
    ds4_tw_tensor indexer_compressor_gate;
    ds4_tw_tensor indexer_compressor_norm;
} ds4_tw_layer;

typedef struct {
    uint8_t     *buf;
    uint64_t     buf_size;
    ds4_tw_tensor token_embd;
    ds4_tw_tensor output_hc_base;
    ds4_tw_tensor output_hc_fn;
    ds4_tw_tensor output_hc_scale;
    ds4_tw_tensor output_norm;
    ds4_tw_tensor output;
    ds4_tw_layer  layer[DS4_TW_N_LAYER];
} ds4_tw_flash_weights;

/* Rounds x up to the next multiple of align (align a power of two). */
static inline uint64_t ds4_tw_align_up(uint64_t x, uint64_t align) {
    return (x + align - 1u) & ~(align - 1u);
}

/* Reserves `elements(type, ndim, d0, d1, d2)` bytes at *cursor in the
 * eventual buffer, filling every ds4_tw_tensor field except the byte
 * content itself (a second pass writes that once the buffer exists).
 * Every tensor starts 8-byte aligned, so the fill pass below can cast
 * buf + abs_offset to a float pointer or a uint16_t pointer without an
 * unaligned access, no matter what the neighbouring tensor's block size
 * or byte count is. */
static inline void ds4_tw_reserve(ds4_tw_tensor *t, uint64_t *cursor, uint32_t type,
                                  uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    memset(t, 0, sizeof(*t));
    t->type = type;
    t->ndim = ndim;
    t->dim[0] = d0;
    t->dim[1] = d1;
    t->dim[2] = d2;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < ndim; i++) elements *= t->dim[i];
    t->elements = elements;
    t->bytes = ds4_tw_tensor_bytes(type, elements);
    t->abs_offset = ds4_tw_align_up(*cursor, 8u);
    *cursor = t->abs_offset + t->bytes;
}

/* Fills a Q8_0 dense tensor's rows in place: dim[0] is the row length (the
 * contraction dimension), dim[1] is the row count.  `salt_base` keeps
 * different dense tensors from producing identical rows when their row
 * indices coincide. */
static inline void ds4_tw_fill_q8_0_dense(uint8_t *buf, const ds4_tw_tensor *t, uint32_t salt_base) {
    const uint32_t in_dim = (uint32_t)t->dim[0];
    const uint32_t out_dim = (uint32_t)t->dim[1];
    const uint64_t row_bytes = t->bytes / out_dim;
    for (uint32_t row = 0; row < out_dim; row++) {
        test_encode_q8_0_row(buf + t->abs_offset + (uint64_t)row * row_bytes, in_dim, salt_base + row);
    }
}

/* Fills a routed-expert tensor (ndim=3, dim=[in_dim, mid_dim, n_expert])
 * whose rows are IQ2_XXS blocks, one 256-wide block per row (guaranteed by
 * the DS4_TW_N_EMBD == DS4_TW_N_FF_EXP == 256 shape above). */
static inline void ds4_tw_fill_iq2_routed(uint8_t *buf, const ds4_tw_tensor *t, uint32_t phase) {
    const uint32_t in_dim = (uint32_t)t->dim[0];
    const uint32_t mid_dim = (uint32_t)t->dim[1];
    const uint32_t n_expert = (uint32_t)t->dim[2];
    const uint32_t blocks_per_row = in_dim / 256u;
    const uint64_t row_bytes = (uint64_t)blocks_per_row * 66u;
    const uint64_t expert_bytes = (uint64_t)mid_dim * row_bytes;
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            for (uint32_t b = 0; b < blocks_per_row; b++) {
                uint8_t blk[66];
                iq2_fill_row(blk, phase, e, row, b);
                memcpy(buf + t->abs_offset + (uint64_t)e * expert_bytes + (uint64_t)row * row_bytes +
                               (uint64_t)b * 66u,
                       blk, sizeof(blk));
            }
        }
    }
}

/* Fills a routed-expert tensor (ndim=3, dim=[in_dim, out_dim, n_expert])
 * whose rows are Q2_K blocks, one 256-wide block per row. */
static inline void ds4_tw_fill_q2k_routed(uint8_t *buf, const ds4_tw_tensor *t, uint32_t phase) {
    const uint32_t in_dim = (uint32_t)t->dim[0];
    const uint32_t out_dim = (uint32_t)t->dim[1];
    const uint32_t n_expert = (uint32_t)t->dim[2];
    const uint32_t blocks_per_row = in_dim / 256u;
    const uint64_t row_bytes = (uint64_t)blocks_per_row * 84u;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_bytes;
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < blocks_per_row; b++) {
                uint8_t blk[84];
                q2k_fill_row(blk, phase, e, row, b);
                memcpy(buf + t->abs_offset + (uint64_t)e * expert_bytes + (uint64_t)row * row_bytes +
                               (uint64_t)b * 84u,
                       blk, sizeof(blk));
            }
        }
    }
}

enum {
    DS4_TW_PHASE_GATE = 0,
    DS4_TW_PHASE_UP   = 1,
    DS4_TW_PHASE_DOWN = 2,
};

/* Allocates and fills a host byte buffer standing in for model.map, holding
 * every tensor a compress-ratio-0 Flash decoder's full per-token phase
 * (metal_graph_encode_decode_layer_phase, ds4.c:21856), the token-embedding
 * lookup that precedes it, and the output head (metal_graph_encode_output_head,
 * ds4.c:25147) read, at the shape defined above, for all DS4_TW_N_LAYER
 * layers: layers below DS4_TW_N_HASH_LAYER additionally get a real
 * ffn_gate_tid2eid hash routing table.  Every layer's tensors are filled
 * with distinct salts (offset by the layer index) so no two layers end up
 * with identical weights by coincidence.  All tensors are filled with real,
 * non-zero, deterministic values (not zeroed), so a caller checking "the
 * output is not all zero" is checking something meaningful.
 *
 * Returns 1 on success (out->buf allocated and every field populated), 0
 * on failure (out of memory); on failure *out is left zeroed and there is
 * nothing to free. */
static inline int ds4_test_build_flash_layer_weights(ds4_tw_flash_weights *out) {
    memset(out, 0, sizeof(*out));
    uint64_t cursor = 0;

    ds4_tw_reserve(&out->token_embd, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, DS4_TW_N_VOCAB, 0);
    ds4_tw_reserve(&out->output_hc_base, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_HC, 0, 0);
    ds4_tw_reserve(&out->output_hc_fn, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_HC_DIM, DS4_TW_N_HC, 0);
    ds4_tw_reserve(&out->output_hc_scale, &cursor, DS4_TW_TYPE_F32, 1, 1, 0, 0);
    ds4_tw_reserve(&out->output_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_EMBD, 0, 0);
    ds4_tw_reserve(&out->output, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_EMBD, DS4_TW_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_TW_N_LAYER; il++) {
        ds4_tw_layer *l = &out->layer[il];
        ds4_tw_reserve(&l->hc_attn_fn, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_HC_DIM, DS4_TW_HC_MIX_DIM, 0);
        ds4_tw_reserve(&l->hc_attn_scale, &cursor, DS4_TW_TYPE_F32, 1, 3, 0, 0);
        ds4_tw_reserve(&l->hc_attn_base, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_HC_MIX_DIM, 0, 0);
        ds4_tw_reserve(&l->attn_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_EMBD, 0, 0);
        ds4_tw_reserve(&l->attn_q_a, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_EMBD, DS4_TW_N_LORA_Q, 0);
        ds4_tw_reserve(&l->attn_q_a_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_LORA_Q, 0, 0);
        ds4_tw_reserve(&l->attn_q_b, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_LORA_Q, DS4_TW_Q_DIM, 0);
        ds4_tw_reserve(&l->attn_kv, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_EMBD, DS4_TW_N_HEAD_DIM, 0);
        ds4_tw_reserve(&l->attn_kv_a_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_HEAD_DIM, 0, 0);
        ds4_tw_reserve(&l->attn_sinks, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_HEAD, 0, 0);
        ds4_tw_reserve(&l->attn_output_a, &cursor, DS4_TW_TYPE_Q8_0, 2,
                       DS4_TW_N_HEAD_DIM * (DS4_TW_N_HEAD / DS4_TW_N_OUT_GROUP), DS4_TW_OUT_LOW_DIM, 0);
        ds4_tw_reserve(&l->attn_output_b, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_OUT_LOW_DIM, DS4_TW_N_EMBD, 0);

        /* Attention compressor and indexer, shapes taken from
         * weights_validate_layout (ds4.c:5058-5075): coff is 2 for ratio 4
         * (the compressor also carries the indexer's rope-neighbour pair)
         * and 1 for ratio 128; comp_width = coff * n_head_dim;
         * index_width = 2 * n_indexer_head_dim (always coff==2, ratio 4 is
         * the only case that reserves indexer tensors at all).  Left
         * zeroed for a layer whose ratio is 0, exactly like
         * ffn_gate_tid2eid is left zeroed for a top-k-routed layer above. */
        {
            const uint32_t ratio = ds4_tw_compress_ratio(il);
            if (ratio != 0u) {
                const uint32_t coff = ratio == 4u ? 2u : 1u;
                const uint64_t comp_width = (uint64_t)coff * DS4_TW_N_HEAD_DIM;
                ds4_tw_reserve(&l->attn_compressor_ape, &cursor, DS4_TW_TYPE_F16, 2, comp_width, ratio, 0);
                ds4_tw_reserve(&l->attn_compressor_kv, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, comp_width, 0);
                ds4_tw_reserve(&l->attn_compressor_gate, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, comp_width, 0);
                ds4_tw_reserve(&l->attn_compressor_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_HEAD_DIM, 0, 0);
            }
            if (ratio == 4u) {
                const uint64_t index_q_dim = (uint64_t)DS4_TW_N_INDEXER_HEAD * DS4_TW_N_INDEXER_HEAD_DIM;
                const uint64_t index_width = 2u * DS4_TW_N_INDEXER_HEAD_DIM;
                ds4_tw_reserve(&l->indexer_attn_q_b, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_LORA_Q, index_q_dim, 0);
                ds4_tw_reserve(&l->indexer_proj, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, DS4_TW_N_INDEXER_HEAD, 0);
                ds4_tw_reserve(&l->indexer_compressor_ape, &cursor, DS4_TW_TYPE_F16, 2, index_width, ratio, 0);
                ds4_tw_reserve(&l->indexer_compressor_kv, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, index_width, 0);
                ds4_tw_reserve(&l->indexer_compressor_gate, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, index_width, 0);
                ds4_tw_reserve(&l->indexer_compressor_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_INDEXER_HEAD_DIM, 0, 0);
            }
        }

        ds4_tw_reserve(&l->hc_ffn_fn, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_HC_DIM, DS4_TW_HC_MIX_DIM, 0);
        ds4_tw_reserve(&l->hc_ffn_scale, &cursor, DS4_TW_TYPE_F32, 1, 3, 0, 0);
        ds4_tw_reserve(&l->hc_ffn_base, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_HC_MIX_DIM, 0, 0);
        ds4_tw_reserve(&l->ffn_norm, &cursor, DS4_TW_TYPE_F32, 1, DS4_TW_N_EMBD, 0, 0);
        ds4_tw_reserve(&l->ffn_gate_inp, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, DS4_TW_N_EXPERT, 0);
        ds4_tw_reserve(&l->ffn_gate_exps, &cursor, DS4_TW_TYPE_IQ2_XXS, 3,
                       DS4_TW_N_EMBD, DS4_TW_N_FF_EXP, DS4_TW_N_EXPERT);
        ds4_tw_reserve(&l->ffn_up_exps, &cursor, DS4_TW_TYPE_IQ2_XXS, 3,
                       DS4_TW_N_EMBD, DS4_TW_N_FF_EXP, DS4_TW_N_EXPERT);
        ds4_tw_reserve(&l->ffn_down_exps, &cursor, DS4_TW_TYPE_Q2_K, 3,
                       DS4_TW_N_FF_EXP, DS4_TW_N_EMBD, DS4_TW_N_EXPERT);
        ds4_tw_reserve(&l->ffn_gate_shexp, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_EMBD, DS4_TW_N_FF_EXP, 0);
        ds4_tw_reserve(&l->ffn_up_shexp, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_EMBD, DS4_TW_N_FF_EXP, 0);
        ds4_tw_reserve(&l->ffn_down_shexp, &cursor, DS4_TW_TYPE_Q8_0, 2, DS4_TW_N_FF_EXP, DS4_TW_N_EMBD, 0);
        if (il < DS4_TW_N_HASH_LAYER) {
            ds4_tw_reserve(&l->ffn_gate_tid2eid, &cursor, DS4_TW_TYPE_I32, 2,
                           DS4_TW_N_EXPERT_USED, DS4_TW_N_VOCAB, 0);
        }
    }

    out->buf_size = cursor;
    out->buf = (uint8_t *)calloc(1, (size_t)cursor);
    if (!out->buf) {
        memset(out, 0, sizeof(*out));
        return 0;
    }

    ds4_tw_fill_f16((uint16_t *)(out->buf + out->token_embd.abs_offset), out->token_embd.elements,
                    /*salt=*/1u, 0.0f, 0.2f);
    ds4_tw_fill_f32((float *)(out->buf + out->output_hc_base.abs_offset), out->output_hc_base.elements,
                    /*salt=*/14u, 0.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(out->buf + out->output_hc_fn.abs_offset), out->output_hc_fn.elements,
                    /*salt=*/15u, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + out->output_hc_scale.abs_offset), out->output_hc_scale.elements,
                    /*salt=*/16u, 0.5f, 0.2f);
    ds4_tw_fill_f32((float *)(out->buf + out->output_norm.abs_offset), out->output_norm.elements,
                    /*salt=*/17u, 1.0f, 0.1f);
    ds4_tw_fill_q8_0_dense(out->buf, &out->output, /*salt_base=*/8000u);

    for (uint32_t il = 0; il < DS4_TW_N_LAYER; il++) {
        ds4_tw_layer *l = &out->layer[il];
        const uint32_t s = il * 100u;
        const uint32_t sb = il * 10000u;

        ds4_tw_fill_f16((uint16_t *)(out->buf + l->hc_attn_fn.abs_offset), l->hc_attn_fn.elements,
                        /*salt=*/2u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->hc_attn_scale.abs_offset), l->hc_attn_scale.elements,
                        /*salt=*/3u + s, 0.5f, 0.2f);
        ds4_tw_fill_f32((float *)(out->buf + l->hc_attn_base.abs_offset), l->hc_attn_base.elements,
                        /*salt=*/4u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->attn_norm.abs_offset), l->attn_norm.elements,
                        /*salt=*/5u + s, 1.0f, 0.1f);
        ds4_tw_fill_q8_0_dense(out->buf, &l->attn_q_a, /*salt_base=*/0u + sb);
        ds4_tw_fill_f32((float *)(out->buf + l->attn_q_a_norm.abs_offset), l->attn_q_a_norm.elements,
                        /*salt=*/6u + s, 1.0f, 0.1f);
        ds4_tw_fill_q8_0_dense(out->buf, &l->attn_q_b, /*salt_base=*/1000u + sb);
        ds4_tw_fill_q8_0_dense(out->buf, &l->attn_kv, /*salt_base=*/2000u + sb);
        ds4_tw_fill_f32((float *)(out->buf + l->attn_kv_a_norm.abs_offset), l->attn_kv_a_norm.elements,
                        /*salt=*/7u + s, 1.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->attn_sinks.abs_offset), l->attn_sinks.elements,
                        /*salt=*/8u + s, 0.0f, 0.2f);
        ds4_tw_fill_q8_0_dense(out->buf, &l->attn_output_a, /*salt_base=*/3000u + sb);
        ds4_tw_fill_q8_0_dense(out->buf, &l->attn_output_b, /*salt_base=*/4000u + sb);

        /* Only the tensors ds4_tw_compress_ratio(il) actually reserved
         * above have a non-zero abs_offset/elements to fill; a ratio-0
         * layer's compressor/indexer ds4_tw_tensor fields are still
         * zeroed from the reserve pass and ds4_tw_fill_f16/f32 on zero
         * elements is a no-op, so no extra branch is needed here beyond
         * the one already in the reserve pass. Salts 18-27 are unused by
         * every other fill call in this loop (2-13 above). */
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->attn_compressor_ape.abs_offset),
                        l->attn_compressor_ape.elements, /*salt=*/18u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->attn_compressor_kv.abs_offset),
                        l->attn_compressor_kv.elements, /*salt=*/19u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->attn_compressor_gate.abs_offset),
                        l->attn_compressor_gate.elements, /*salt=*/20u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->attn_compressor_norm.abs_offset),
                        l->attn_compressor_norm.elements, /*salt=*/21u + s, 1.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->indexer_attn_q_b.abs_offset),
                        l->indexer_attn_q_b.elements, /*salt=*/22u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->indexer_proj.abs_offset),
                        l->indexer_proj.elements, /*salt=*/23u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->indexer_compressor_ape.abs_offset),
                        l->indexer_compressor_ape.elements, /*salt=*/24u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->indexer_compressor_kv.abs_offset),
                        l->indexer_compressor_kv.elements, /*salt=*/25u + s, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->indexer_compressor_gate.abs_offset),
                        l->indexer_compressor_gate.elements, /*salt=*/26u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->indexer_compressor_norm.abs_offset),
                        l->indexer_compressor_norm.elements, /*salt=*/27u + s, 1.0f, 0.1f);

        ds4_tw_fill_f16((uint16_t *)(out->buf + l->hc_ffn_fn.abs_offset), l->hc_ffn_fn.elements,
                        /*salt=*/9u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->hc_ffn_scale.abs_offset), l->hc_ffn_scale.elements,
                        /*salt=*/10u + s, 0.5f, 0.2f);
        ds4_tw_fill_f32((float *)(out->buf + l->hc_ffn_base.abs_offset), l->hc_ffn_base.elements,
                        /*salt=*/11u + s, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(out->buf + l->ffn_norm.abs_offset), l->ffn_norm.elements,
                        /*salt=*/12u + s, 1.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(out->buf + l->ffn_gate_inp.abs_offset), l->ffn_gate_inp.elements,
                        /*salt=*/13u + s, 0.0f, 0.2f);
        ds4_tw_fill_iq2_routed(out->buf, &l->ffn_gate_exps, DS4_TW_PHASE_GATE + il * 3u);
        ds4_tw_fill_iq2_routed(out->buf, &l->ffn_up_exps, DS4_TW_PHASE_UP + il * 3u);
        ds4_tw_fill_q2k_routed(out->buf, &l->ffn_down_exps, DS4_TW_PHASE_DOWN + il * 3u);
        ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_gate_shexp, /*salt_base=*/5000u + sb);
        ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_up_shexp, /*salt_base=*/6000u + sb);
        ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_down_shexp, /*salt_base=*/7000u + sb);
        if (il < DS4_TW_N_HASH_LAYER) {
            ds4_tw_fill_hash_tid2eid((int32_t *)(out->buf + l->ffn_gate_tid2eid.abs_offset),
                                     DS4_TW_N_VOCAB, DS4_TW_N_EXPERT_USED, DS4_TW_N_EXPERT,
                                     /*layer_salt=*/il + 1u);
        }
    }

    return 1;
}

/* Frees the buffer a prior call to ds4_test_build_flash_layer_weights
 * allocated.  Safe to call on a zeroed or failed *out (buf == NULL). */
static inline void ds4_test_free_flash_layer_weights(ds4_tw_flash_weights *out) {
    if (!out) return;
    free(out->buf);
    out->buf = NULL;
    out->buf_size = 0;
}

#endif /* DS4_TEST_SYCL_LAYER_WEIGHTS_H */
