/* Synthetic ds4_weights / ds4_layer_weights scaffolding for one DeepSeek V4
 * Flash decoder layer, at a small shape sized to drive
 * metal_graph_encode_decode_layer_phase (ds4.c:21856), the real per-token
 * decode path, with no model file.  Pure C, no SYCL or C++-only includes:
 * this is included both from a plain C test file and, under
 * DS4_TEST_HOOKS, from ds4.c itself, which every GPU backend compiles with
 * a plain C compiler.
 *
 * Nothing calls this header yet.  A later DS4_TEST_HOOKS hook builds a
 * ds4_weights and ds4_layer_weights by pointing ds4_tensor fields at the
 * abs_offset/dim/type values this header fills in, exactly as
 * ds4_test_graph_alloc_smoke and ds4_test_graph_encode_smoke (ds4.c:57544,
 * ds4.c:57671) already do by hand for their own smaller synthetic tensors.
 *
 * The quantisation combination matches Flash as shipped: dense weights
 * (attn_q_a, attn_q_b, attn_kv, attn_output_a, attn_output_b,
 * ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp) are Q8_0; the router
 * (ffn_gate_inp) is F16; routed experts are IQ2_XXS gate/up and Q2_K down;
 * norms, hyper-connection (HC) scale/base, and attn_sinks are F32.
 *
 * The IQ2_XXS block packer, the Q2_K block packer, and the Q8_0 row
 * encoder below are ports of already-existing, already-exercised block
 * encoders, not new byte layouts: they reproduce, line for line,
 * tests/test_sycl_moe.c's f32_to_f16_bits (:113), oracle_iq2_pack_ib32
 * (:1693), oracle_iq2_pack_block (:1707), iq2_fill_row (:1715),
 * oracle_q2k_pack_block (:1770), q2k_fill_row (:1813), and
 * tests/test_sycl_matmul.c's test_encode_q8_0_row (:220, adapted to call
 * this header's f32_to_f16_bits instead of that file's test_float_to_half
 * -- the two are not the same conversion, see the note above
 * f32_to_f16_bits below).  They are reproduced here rather than shared by
 * extraction because test_sycl_moe.c documents itself as deliberately
 * self-contained ("no tests/test_sycl_harness.h on this branch's base
 * commit") and every SYCL test file in this suite follows the same
 * precedent of a cited, local reimplementation rather than a cross-file
 * header dependency (test_sycl_moe.c's own top comment cites
 * test_sycl_kernels.c for this), so this header follows that established
 * convention instead of retrofitting either file. */

#ifndef DS4_TEST_SYCL_LAYER_WEIGHTS_H
#define DS4_TEST_SYCL_LAYER_WEIGHTS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- The synthetic shape ------------------------------------------------
 *
 * One layer, compress ratio 0 (the attention compressor and indexer are
 * out of scope: metal_graph_encode_decode_layer_phase only dereferences
 * layer->attn_compressor_... / layer->indexer_... when ds4_layer_compress_ratio
 * (il) != 0).  n_embd == n_ff_exp == 256 so every routed-expert row needs
 * exactly one 256-wide quant block (kMoeQK, sycl/ds4_sycl_moe.hpp:255;
 * build_plan rejects expert_in_dim/expert_mid_dim not a multiple of it,
 * sycl/ds4_sycl_moe_launch.hpp:57-58).  n_lora_q and n_lora_o are
 * multiples of 32 so the Q8_0 dense rows they become (attn_q_b via
 * DS4_N_LORA_Q, attn_output_b via DS4_N_OUT_GROUP*DS4_N_LORA_O) are
 * block-aligned (Q8_0 blocks are 32 elements, ds4.c:2028).
 *
 * A caller building the matching ds4_shape (only ds4.c can, ds4_shape is
 * private to that translation unit) should use these same values for
 * every field below, plus n_ff_dense=32, n_hash_layer=0, n_swa=8, small
 * n_indexer_* values, and the DS4_DEFAULT_* constants for everything else
 * -- the same pattern ds4_test_graph_alloc_smoke (ds4.c:57544) uses -- so
 * the shape fed to the engine and the tensors this header builds never
 * drift apart. */
#define DS4_TW_N_LAYER              1u
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
#define DS4_TW_N_EXPERT              4u
#define DS4_TW_N_EXPERT_USED         2u
#define DS4_TW_N_EXPERT_SHARED       1u
#define DS4_TW_N_FF_EXP              256u
#define DS4_TW_N_HC                  4u
#define DS4_TW_N_HC_SINKHORN_ITER    1u
#define DS4_TW_N_SWA                 8u

/* Derived exactly as weights_validate_layout derives them (ds4.c:5027-
 * 5030): hc_dim = n_embd*n_hc, hc_mix_dim = 2*n_hc + n_hc*n_hc, q_dim =
 * n_head*n_head_dim, out_low_dim = n_out_group*n_lora_o. */
#define DS4_TW_HC_DIM       (DS4_TW_N_EMBD * DS4_TW_N_HC)
#define DS4_TW_HC_MIX_DIM   (2u * DS4_TW_N_HC + DS4_TW_N_HC * DS4_TW_N_HC)
#define DS4_TW_Q_DIM        (DS4_TW_N_HEAD * DS4_TW_N_HEAD_DIM)
#define DS4_TW_OUT_LOW_DIM  (DS4_TW_N_OUT_GROUP * DS4_TW_N_LORA_O)

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

/* elements-per-block/bytes-per-block for exactly the five types above,
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
        default: return 0;
    }
    const uint64_t blocks = (elements + block_elems - 1u) / block_elems;
    return blocks * (uint64_t)block_bytes;
}

/* ---- Ported block encoders ------------------------------------------------
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

/* Deterministic, non-zero, small-amplitude float fill for the plain F32/
 * F16 tensors (norms, HC scale/base, attn_sinks, the router, the HC mix
 * matrices, token_embd) that have no pre-existing block encoder to reuse.
 * `salt` distinguishes one tensor from another so no two tensors end up
 * with the same values by coincidence. center +/- scale/2. */
static inline float ds4_tw_hashed_float(uint64_t i, uint32_t salt, float center, float scale) {
    const uint32_t h = (uint32_t)((i * 2654435761ull) ^ ((uint64_t)salt * 2246822519ull));
    const float frac = (float)((h >> 16) % 1000u) / 1000.0f - 0.5f; /* [-0.5, 0.5) */
    return center + scale * frac;
}

static inline void ds4_tw_fill_f32(float *dst, uint64_t n, uint32_t salt, float center, float scale) {
    for (uint64_t i = 0; i < n; i++) dst[i] = ds4_tw_hashed_float(i, salt, center, scale);
}

static inline void ds4_tw_fill_f16(uint16_t *dst, uint64_t n, uint32_t salt, float center, float scale) {
    for (uint64_t i = 0; i < n; i++) dst[i] = f32_to_f16_bits(ds4_tw_hashed_float(i, salt, center, scale));
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
} ds4_tw_layer;

typedef struct {
    uint8_t     *buf;
    uint64_t     buf_size;
    ds4_tw_tensor token_embd;
    ds4_tw_layer  layer;
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
 * every tensor a compress-ratio-0 Flash decoder layer's full per-token
 * phase (metal_graph_encode_decode_layer_phase, ds4.c:21856) and the
 * token-embedding lookup that precedes it read, at the shape defined
 * above.  All tensors are filled with real, non-zero, deterministic
 * values (not zeroed), so a caller checking "the layer's output is not
 * all zero" is checking something meaningful.
 *
 * Returns 1 on success (out->buf allocated and every field populated), 0
 * on failure (out of memory); on failure *out is left zeroed and there is
 * nothing to free. */
static inline int ds4_test_build_flash_layer_weights(ds4_tw_flash_weights *out) {
    memset(out, 0, sizeof(*out));
    uint64_t cursor = 0;

    ds4_tw_reserve(&out->token_embd, &cursor, DS4_TW_TYPE_F16, 2, DS4_TW_N_EMBD, DS4_TW_N_VOCAB, 0);

    ds4_tw_layer *l = &out->layer;
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

    out->buf_size = cursor;
    out->buf = (uint8_t *)calloc(1, (size_t)cursor);
    if (!out->buf) {
        memset(out, 0, sizeof(*out));
        return 0;
    }

    ds4_tw_fill_f16((uint16_t *)(out->buf + out->token_embd.abs_offset), out->token_embd.elements,
                    /*salt=*/1u, 0.0f, 0.2f);

    ds4_tw_fill_f16((uint16_t *)(out->buf + l->hc_attn_fn.abs_offset), l->hc_attn_fn.elements,
                    /*salt=*/2u, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + l->hc_attn_scale.abs_offset), l->hc_attn_scale.elements,
                    /*salt=*/3u, 0.5f, 0.2f);
    ds4_tw_fill_f32((float *)(out->buf + l->hc_attn_base.abs_offset), l->hc_attn_base.elements,
                    /*salt=*/4u, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + l->attn_norm.abs_offset), l->attn_norm.elements,
                    /*salt=*/5u, 1.0f, 0.1f);
    ds4_tw_fill_q8_0_dense(out->buf, &l->attn_q_a, /*salt_base=*/0u);
    ds4_tw_fill_f32((float *)(out->buf + l->attn_q_a_norm.abs_offset), l->attn_q_a_norm.elements,
                    /*salt=*/6u, 1.0f, 0.1f);
    ds4_tw_fill_q8_0_dense(out->buf, &l->attn_q_b, /*salt_base=*/1000u);
    ds4_tw_fill_q8_0_dense(out->buf, &l->attn_kv, /*salt_base=*/2000u);
    ds4_tw_fill_f32((float *)(out->buf + l->attn_kv_a_norm.abs_offset), l->attn_kv_a_norm.elements,
                    /*salt=*/7u, 1.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + l->attn_sinks.abs_offset), l->attn_sinks.elements,
                    /*salt=*/8u, 0.0f, 0.2f);
    ds4_tw_fill_q8_0_dense(out->buf, &l->attn_output_a, /*salt_base=*/3000u);
    ds4_tw_fill_q8_0_dense(out->buf, &l->attn_output_b, /*salt_base=*/4000u);
    ds4_tw_fill_f16((uint16_t *)(out->buf + l->hc_ffn_fn.abs_offset), l->hc_ffn_fn.elements,
                    /*salt=*/9u, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + l->hc_ffn_scale.abs_offset), l->hc_ffn_scale.elements,
                    /*salt=*/10u, 0.5f, 0.2f);
    ds4_tw_fill_f32((float *)(out->buf + l->hc_ffn_base.abs_offset), l->hc_ffn_base.elements,
                    /*salt=*/11u, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(out->buf + l->ffn_norm.abs_offset), l->ffn_norm.elements,
                    /*salt=*/12u, 1.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(out->buf + l->ffn_gate_inp.abs_offset), l->ffn_gate_inp.elements,
                    /*salt=*/13u, 0.0f, 0.2f);
    ds4_tw_fill_iq2_routed(out->buf, &l->ffn_gate_exps, DS4_TW_PHASE_GATE);
    ds4_tw_fill_iq2_routed(out->buf, &l->ffn_up_exps, DS4_TW_PHASE_UP);
    ds4_tw_fill_q2k_routed(out->buf, &l->ffn_down_exps, DS4_TW_PHASE_DOWN);
    ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_gate_shexp, /*salt_base=*/5000u);
    ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_up_shexp, /*salt_base=*/6000u);
    ds4_tw_fill_q8_0_dense(out->buf, &l->ffn_down_shexp, /*salt_base=*/7000u);

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
