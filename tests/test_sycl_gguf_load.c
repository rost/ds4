/* Real-GGUF model-load test for the SYCL backend.
 *
 * Every other SYCL test in this suite either calls a ds4_gpu_* ABI entry
 * directly with a hand-built buffer, or (test_sycl_session_smoke,
 * test_sycl_full_layer) drives the engine's graph allocator and decode path
 * on synthetic weights structs built in memory, with no model file. Nothing
 * has ever driven GGUF parsing and weight binding -- which live in ds4.c,
 * are backend-agnostic, and Metal/CUDA/ROCm exercise constantly -- into the
 * SYCL side of model loading with real offsets read from a real file.
 * ds4_gpu_set_model_map_range, ds4_gpu_set_model_map_spans,
 * ds4_gpu_cache_model_range, ds4_gpu_cache_q8_f16_range and
 * ds4_gpu_set_model_fd are only ever called by ds4_engine_open, and three of
 * them were once stubbed to their failure value, making it impossible to
 * open any model at all on this backend; they were found by reading a
 * header rather than by any test, and have never since been run against an
 * actual file. That is the gap this test closes.
 *
 * A real GGUF file cannot be small for this model family: config_validate_
 * deepseek4_model (ds4.c:5635) selects the runtime shape by requiring an
 * EXACT match against one of two hardcoded full-scale shapes, DS4_SHAPE_
 * FLASH or DS4_SHAPE_PRO (ds4.c:547, 585) -- every dimension, not just
 * n_expert. There is no reduced-size DeepSeek4 shape the loader will
 * accept, so this file uses the real Flash numbers throughout and instead
 * shrinks the file by loading only one transformer layer via the engine's
 * own distributed-pipeline slice mechanism (opt.load_slice, ds4.h): layer 3
 * (compress ratio 128, so no compressor/indexer complexity), with no token
 * embedding and no output head, since ds4_session_create only requires that
 * weights_first_bound_layer find one bound layer (ds4.c:59577) and
 * ds4_session_eval_layer_slice only requires token_embd when layer_start is
 * 0 and only requires the output head when output_logits is requested
 * (ds4.c:60166). Even minimised this way the file is on the order of two
 * gigabytes, dominated by 256 real routed experts per tensor -- that count
 * is load-bearing for router kernels, which require n_expert in
 * {256, 384}, and cannot be shrunk without failing shape selection.
 *
 * The required metadata keys and tensor set are read from ds4.c, not from
 * an external GGUF spec: config_validate_deepseek4_model's config_expect_u32
 * / required_f32 / required_bool calls (ds4.c:5635-5745) name every scalar
 * key, validate_compress_ratio_metadata and validate_swiglu_clamp_metadata
 * (ds4.c:5536, 5574) name the two required per-layer arrays, and
 * weights_bind_layer (ds4.c:5901) names every "blk.%u.*.weight" tensor a
 * compress-ratio-128 layer needs. vocab_load (ds4.c:37700) additionally
 * requires tokenizer.ggml.tokens/merges string arrays unconditionally,
 * independent of load_slice.
 *
 * The block encoders (Q8_0 row, IQ2_XXS block, Q2_K block, the F16 rounding
 * conversion, and the non-affine float filler) are the exact ones
 * tests/test_sycl_layer_weights.h already ports from test_sycl_moe.c,
 * test_sycl_matmul.c and test_sycl_hc.c; this file reuses them unchanged,
 * just called with real Flash dimensions instead of that header's small
 * synthetic shape, since ds4_tw_reserve and the ds4_tw_fill_* helpers take
 * dimensions as arguments rather than hardcoding them.
 *
 * Four assertions:
 *   1. ds4_engine_open succeeds on the generated file.
 *   2. ds4_session_create succeeds, exercising the real graph allocator
 *      against real tensor offsets for a real quantisation mix.
 *   3. ds4_session_eval_layer_slice, run over the one loaded layer, returns
 *      success and reads back a finite, non-all-zero, non-input-identical
 *      hidden state -- the real per-token decode kernels for this layer,
 *      each staging its own weight range to the device per spec 6l (this
 *      backend keeps no device-resident model copy, so there is no single
 *      "the weights are on the device" moment to poll; the only available
 *      proof is that the computation which depends on those staged bytes
 *      produced a real, non-zero, input-dependent result).
 *   4. An ablation: corrupting one tensor's rel_offset so its absolute
 *      offset falls outside the file makes ds4_engine_open reject the file
 *      instead of reading past the mapping. ds4.c's GGUF loader treats this
 *      as fatal (ds4_die calls exit(1), ds4.c:1072), so the corrupted
 *      attempt runs in a freshly exec'd child and the parent checks its
 *      exit status; the baseline open above already established the clean
 *      pre-ablation case per spec 6n before this corruption is applied, and
 *      the corruption is reverted afterward so the same file can be reused. */

#include "ds4.h"
#include "ds4_gpu.h"

#include "test_sycl_layer_weights.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s\n", (msg));                           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* Real DeepSeek V4 Flash shape (ds4.c:547-579, DS4_SHAPE_FLASH). Every
 * value here must match that struct exactly, or config_validate_
 * deepseek4_model's shape selection (ds4.c:5476) rejects the file before
 * any tensor is even looked at. */
#define FLASH_N_LAYER            43u
#define FLASH_N_EMBD           4096u
#define FLASH_N_VOCAB        129280u
#define FLASH_N_HEAD             64u
#define FLASH_N_HEAD_KV           1u
#define FLASH_N_HEAD_DIM        512u
#define FLASH_N_VALUE_DIM       512u
#define FLASH_N_ROT              64u
#define FLASH_N_OUT_GROUP         8u
#define FLASH_N_LORA_Q         1024u
#define FLASH_N_LORA_O         1024u
#define FLASH_N_EXPERT          256u
#define FLASH_N_EXPERT_USED       6u
#define FLASH_N_EXPERT_SHARED     1u
#define FLASH_N_FF_EXP         2048u
#define FLASH_N_HASH_LAYER        3u
#define FLASH_N_SWA             128u
#define FLASH_N_INDEXER_HEAD     64u
#define FLASH_N_INDEXER_HEAD_DIM 128u
#define FLASH_N_INDEXER_TOP_K   512u
#define FLASH_N_HC                4u
#define FLASH_N_HC_SINKHORN_ITER 20u
#define FLASH_RMS_EPS            1.0e-6f
#define FLASH_HC_EPS             1.0e-6f
#define FLASH_SWIGLU_CLAMP_EXP  10.0f
#define FLASH_ROPE_FREQ_BASE 10000.0f
#define FLASH_COMPRESS_ROPE_FREQ_BASE 160000.0f
#define FLASH_EXPERT_WEIGHT_SCALE 1.5f

#define FLASH_HC_DIM     (FLASH_N_EMBD * FLASH_N_HC)
#define FLASH_HC_MIX_DIM (2u * FLASH_N_HC + FLASH_N_HC * FLASH_N_HC)
#define FLASH_Q_DIM      (FLASH_N_HEAD * FLASH_N_HEAD_DIM)
#define FLASH_OUT_LOW_DIM (FLASH_N_OUT_GROUP * FLASH_N_LORA_O)

/* The one layer this file loads. Ratio 128 (ds4_expected_layer_compress_
 * ratio, ds4.c:1085: even layers >= 2 get 4, odd get 128) needs the
 * attn_compressor_* tensors but not the indexer ones, and is not a hash
 * layer (il >= DS4_N_HASH_LAYER == 3), so no ffn_gate_tid2eid either --
 * the smallest non-token_embd-requiring, non-trivial real Flash layer. */
#define LOAD_LAYER   3u
#define LOAD_RATIO 128u
#define COMP_WIDTH (FLASH_N_HEAD_DIM) /* coff=1 for ratio != 4 */

#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3u

/* Mirrors ds4.c's private GGUF_VALUE_* enum (ds4.c:2007-2019) so this file
 * can write the same on-disk value tags without ds4.c exposing them. */
enum {
    GV_UINT32  = 4,
    GV_FLOAT32 = 6,
    GV_BOOL    = 7,
    GV_STRING  = 8,
    GV_ARRAY   = 9,
};

typedef struct {
    FILE *f;
} writer;

static void w_bytes(writer *w, const void *p, size_t n) {
    if (fwrite(p, 1, n, w->f) != n) { perror("fwrite"); exit(1); }
}
static void w_u32(writer *w, uint32_t v) { w_bytes(w, &v, sizeof(v)); }
static void w_u64(writer *w, uint64_t v) { w_bytes(w, &v, sizeof(v)); }
static void w_f32(writer *w, float v) { w_bytes(w, &v, sizeof(v)); }
static void w_str(writer *w, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    w_u64(w, len);
    w_bytes(w, s, (size_t)len);
}
static void w_key(writer *w, const char *key, uint32_t type) {
    w_str(w, key);
    w_u32(w, type);
}
static void w_kv_u32(writer *w, const char *key, uint32_t v, uint64_t *n_kv) {
    w_key(w, key, GV_UINT32);
    w_u32(w, v);
    (*n_kv)++;
}
static void w_kv_f32(writer *w, const char *key, float v, uint64_t *n_kv) {
    w_key(w, key, GV_FLOAT32);
    w_f32(w, v);
    (*n_kv)++;
}
static void w_kv_bool(writer *w, const char *key, bool v, uint64_t *n_kv) {
    w_key(w, key, GV_BOOL);
    uint8_t b = v ? 1 : 0;
    w_bytes(w, &b, 1);
    (*n_kv)++;
}
static void w_kv_u32_array(writer *w, const char *key, const uint32_t *vals,
                           uint64_t n, uint64_t *n_kv) {
    w_key(w, key, GV_ARRAY);
    w_u32(w, GV_UINT32);
    w_u64(w, n);
    for (uint64_t i = 0; i < n; i++) w_u32(w, vals[i]);
    (*n_kv)++;
}
static void w_kv_f32_array(writer *w, const char *key, const float *vals,
                           uint64_t n, uint64_t *n_kv) {
    w_key(w, key, GV_ARRAY);
    w_u32(w, GV_FLOAT32);
    w_u64(w, n);
    for (uint64_t i = 0; i < n; i++) w_f32(w, vals[i]);
    (*n_kv)++;
}
static void w_kv_string_array(writer *w, const char *key,
                              const char *const *vals, uint64_t n,
                              uint64_t *n_kv) {
    w_key(w, key, GV_ARRAY);
    w_u32(w, GV_STRING);
    w_u64(w, n);
    for (uint64_t i = 0; i < n; i++) w_str(w, vals[i]);
    (*n_kv)++;
}

/* One entry in the tensor directory, tracked alongside the ds4_tw_tensor
 * that was reserved for it in the data blob so the directory's rel_offset
 * exactly matches where the tensor's bytes actually live. */
typedef struct {
    const char *name;
    ds4_tw_tensor t;
} dir_entry;

#define MAX_TENSORS 32
static dir_entry g_dir[MAX_TENSORS];
static uint32_t g_n_dir = 0;

static ds4_tw_tensor *reserve(uint64_t *cursor, const char *name, uint32_t type,
                              uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    assert(g_n_dir < MAX_TENSORS);
    dir_entry *e = &g_dir[g_n_dir++];
    e->name = name;
    ds4_tw_reserve(&e->t, cursor, type, ndim, d0, d1, d2);
    return &e->t;
}

/* Builds the GGUF file at `path`: real-Flash-shape metadata for layer
 * LOAD_LAYER, a minimal tokenizer, and every tensor weights_bind_layer
 * requires for that layer's compress ratio. Returns the absolute file
 * offset of the first tensor's rel_offset field (for the ablation) via
 * *corrupt_offset_out, and the correct 8 bytes originally written there
 * via *orig_rel_offset_out. */
static int build_gguf(const char *path, uint64_t *corrupt_offset_out,
                      uint64_t *orig_rel_offset_out) {
    /* ---- Reserve every layer-3 tensor into a host blob, exactly the way
     * ds4_test_build_flash_layer_weights does, but with real Flash
     * dimensions instead of the small synthetic shape's macros. */
    uint64_t cursor = 0;
    ds4_tw_tensor *hc_attn_fn = reserve(&cursor, "hc_attn_fn", DS4_TW_TYPE_F16, 2,
                                        FLASH_HC_DIM, FLASH_HC_MIX_DIM, 0);
    ds4_tw_tensor *hc_attn_scale = reserve(&cursor, "hc_attn_scale", DS4_TW_TYPE_F32, 1, 3, 0, 0);
    ds4_tw_tensor *hc_attn_base = reserve(&cursor, "hc_attn_base", DS4_TW_TYPE_F32, 1,
                                          FLASH_HC_MIX_DIM, 0, 0);
    ds4_tw_tensor *attn_norm = reserve(&cursor, "attn_norm", DS4_TW_TYPE_F32, 1, FLASH_N_EMBD, 0, 0);
    ds4_tw_tensor *attn_q_a = reserve(&cursor, "attn_q_a", DS4_TW_TYPE_Q8_0, 2,
                                      FLASH_N_EMBD, FLASH_N_LORA_Q, 0);
    ds4_tw_tensor *attn_q_a_norm = reserve(&cursor, "attn_q_a_norm", DS4_TW_TYPE_F32, 1,
                                           FLASH_N_LORA_Q, 0, 0);
    ds4_tw_tensor *attn_q_b = reserve(&cursor, "attn_q_b", DS4_TW_TYPE_Q8_0, 2,
                                      FLASH_N_LORA_Q, FLASH_Q_DIM, 0);
    ds4_tw_tensor *attn_kv = reserve(&cursor, "attn_kv", DS4_TW_TYPE_Q8_0, 2,
                                     FLASH_N_EMBD, FLASH_N_HEAD_DIM, 0);
    ds4_tw_tensor *attn_kv_a_norm = reserve(&cursor, "attn_kv_a_norm", DS4_TW_TYPE_F32, 1,
                                            FLASH_N_HEAD_DIM, 0, 0);
    ds4_tw_tensor *attn_sinks = reserve(&cursor, "attn_sinks", DS4_TW_TYPE_F32, 1, FLASH_N_HEAD, 0, 0);
    ds4_tw_tensor *attn_output_a = reserve(&cursor, "attn_output_a", DS4_TW_TYPE_Q8_0, 2,
                                           FLASH_N_HEAD_DIM * (FLASH_N_HEAD / FLASH_N_OUT_GROUP),
                                           FLASH_OUT_LOW_DIM, 0);
    ds4_tw_tensor *attn_output_b = reserve(&cursor, "attn_output_b", DS4_TW_TYPE_Q8_0, 2,
                                           FLASH_OUT_LOW_DIM, FLASH_N_EMBD, 0);
    ds4_tw_tensor *attn_compressor_ape = reserve(&cursor, "attn_compressor_ape", DS4_TW_TYPE_F16, 2,
                                                 COMP_WIDTH, LOAD_RATIO, 0);
    ds4_tw_tensor *attn_compressor_kv = reserve(&cursor, "attn_compressor_kv", DS4_TW_TYPE_F16, 2,
                                                FLASH_N_EMBD, COMP_WIDTH, 0);
    ds4_tw_tensor *attn_compressor_gate = reserve(&cursor, "attn_compressor_gate", DS4_TW_TYPE_F16, 2,
                                                  FLASH_N_EMBD, COMP_WIDTH, 0);
    ds4_tw_tensor *attn_compressor_norm = reserve(&cursor, "attn_compressor_norm", DS4_TW_TYPE_F32, 1,
                                                  FLASH_N_HEAD_DIM, 0, 0);
    ds4_tw_tensor *hc_ffn_fn = reserve(&cursor, "hc_ffn_fn", DS4_TW_TYPE_F16, 2,
                                       FLASH_HC_DIM, FLASH_HC_MIX_DIM, 0);
    ds4_tw_tensor *hc_ffn_scale = reserve(&cursor, "hc_ffn_scale", DS4_TW_TYPE_F32, 1, 3, 0, 0);
    ds4_tw_tensor *hc_ffn_base = reserve(&cursor, "hc_ffn_base", DS4_TW_TYPE_F32, 1,
                                         FLASH_HC_MIX_DIM, 0, 0);
    ds4_tw_tensor *ffn_norm = reserve(&cursor, "ffn_norm", DS4_TW_TYPE_F32, 1, FLASH_N_EMBD, 0, 0);
    ds4_tw_tensor *ffn_gate_inp = reserve(&cursor, "ffn_gate_inp", DS4_TW_TYPE_F16, 2,
                                          FLASH_N_EMBD, FLASH_N_EXPERT, 0);
    ds4_tw_tensor *ffn_gate_exps = reserve(&cursor, "ffn_gate_exps", DS4_TW_TYPE_IQ2_XXS, 3,
                                           FLASH_N_EMBD, FLASH_N_FF_EXP, FLASH_N_EXPERT);
    ds4_tw_tensor *ffn_up_exps = reserve(&cursor, "ffn_up_exps", DS4_TW_TYPE_IQ2_XXS, 3,
                                         FLASH_N_EMBD, FLASH_N_FF_EXP, FLASH_N_EXPERT);
    ds4_tw_tensor *ffn_down_exps = reserve(&cursor, "ffn_down_exps", DS4_TW_TYPE_Q2_K, 3,
                                           FLASH_N_FF_EXP, FLASH_N_EMBD, FLASH_N_EXPERT);
    ds4_tw_tensor *ffn_gate_shexp = reserve(&cursor, "ffn_gate_shexp", DS4_TW_TYPE_Q8_0, 2,
                                            FLASH_N_EMBD, FLASH_N_FF_EXP, 0);
    ds4_tw_tensor *ffn_up_shexp = reserve(&cursor, "ffn_up_shexp", DS4_TW_TYPE_Q8_0, 2,
                                          FLASH_N_EMBD, FLASH_N_FF_EXP, 0);
    ds4_tw_tensor *ffn_down_shexp = reserve(&cursor, "ffn_down_shexp", DS4_TW_TYPE_Q8_0, 2,
                                            FLASH_N_FF_EXP, FLASH_N_EMBD, 0);

    const uint64_t blob_size = cursor;
    fprintf(stderr, "ds4: generating layer-%u synthetic Flash GGUF, %.2f GiB tensor payload\n",
            LOAD_LAYER, (double)blob_size / 1073741824.0);
    uint8_t *buf = (uint8_t *)calloc(1, (size_t)blob_size);
    CHECK(buf != NULL, "out of memory allocating GGUF tensor blob");

    ds4_tw_fill_f16((uint16_t *)(buf + hc_attn_fn->abs_offset), hc_attn_fn->elements, 2, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + hc_attn_scale->abs_offset), hc_attn_scale->elements, 3, 0.5f, 0.2f);
    ds4_tw_fill_f32((float *)(buf + hc_attn_base->abs_offset), hc_attn_base->elements, 4, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + attn_norm->abs_offset), attn_norm->elements, 5, 1.0f, 0.1f);
    ds4_tw_fill_q8_0_dense(buf, attn_q_a, 0u);
    ds4_tw_fill_f32((float *)(buf + attn_q_a_norm->abs_offset), attn_q_a_norm->elements, 6, 1.0f, 0.1f);
    ds4_tw_fill_q8_0_dense(buf, attn_q_b, 1000u);
    ds4_tw_fill_q8_0_dense(buf, attn_kv, 2000u);
    ds4_tw_fill_f32((float *)(buf + attn_kv_a_norm->abs_offset), attn_kv_a_norm->elements, 7, 1.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + attn_sinks->abs_offset), attn_sinks->elements, 8, 0.0f, 0.2f);
    ds4_tw_fill_q8_0_dense(buf, attn_output_a, 3000u);
    ds4_tw_fill_q8_0_dense(buf, attn_output_b, 4000u);
    ds4_tw_fill_f16((uint16_t *)(buf + attn_compressor_ape->abs_offset), attn_compressor_ape->elements,
                    14, 0.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(buf + attn_compressor_kv->abs_offset), attn_compressor_kv->elements,
                    15, 0.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(buf + attn_compressor_gate->abs_offset), attn_compressor_gate->elements,
                    16, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + attn_compressor_norm->abs_offset), attn_compressor_norm->elements,
                    17, 1.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(buf + hc_ffn_fn->abs_offset), hc_ffn_fn->elements, 9, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + hc_ffn_scale->abs_offset), hc_ffn_scale->elements, 10, 0.5f, 0.2f);
    ds4_tw_fill_f32((float *)(buf + hc_ffn_base->abs_offset), hc_ffn_base->elements, 11, 0.0f, 0.1f);
    ds4_tw_fill_f32((float *)(buf + ffn_norm->abs_offset), ffn_norm->elements, 12, 1.0f, 0.1f);
    ds4_tw_fill_f16((uint16_t *)(buf + ffn_gate_inp->abs_offset), ffn_gate_inp->elements, 13, 0.0f, 0.2f);
    ds4_tw_fill_iq2_routed(buf, ffn_gate_exps, DS4_TW_PHASE_GATE);
    ds4_tw_fill_iq2_routed(buf, ffn_up_exps, DS4_TW_PHASE_UP);
    ds4_tw_fill_q2k_routed(buf, ffn_down_exps, DS4_TW_PHASE_DOWN);
    ds4_tw_fill_q8_0_dense(buf, ffn_gate_shexp, 5000u);
    ds4_tw_fill_q8_0_dense(buf, ffn_up_shexp, 6000u);
    ds4_tw_fill_q8_0_dense(buf, ffn_down_shexp, 7000u);

    /* ---- Metadata: every key config_validate_deepseek4_model requires
     * (ds4.c:5635-5745), matching DS4_SHAPE_FLASH exactly, plus the two
     * required per-layer arrays and a minimal tokenizer (vocab_load,
     * ds4.c:37700, runs unconditionally for a non-inspect DeepSeek4 open
     * regardless of load_slice). */
    char *kv_ptr = NULL;
    size_t kv_size = 0;
    FILE *kv_f = open_memstream(&kv_ptr, &kv_size);
    CHECK(kv_f != NULL, "open_memstream for metadata failed");
    writer kw = { kv_f };
    uint64_t n_kv = 0;

    w_kv_u32(&kw, "deepseek4.block_count", FLASH_N_LAYER, &n_kv);
    w_kv_u32(&kw, "deepseek4.embedding_length", FLASH_N_EMBD, &n_kv);
    w_kv_u32(&kw, "deepseek4.vocab_size", FLASH_N_VOCAB, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.head_count", FLASH_N_HEAD, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.head_count_kv", FLASH_N_HEAD_KV, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.key_length", FLASH_N_HEAD_DIM, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.value_length", FLASH_N_VALUE_DIM, &n_kv);
    w_kv_u32(&kw, "deepseek4.rope.dimension_count", FLASH_N_ROT, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.q_lora_rank", FLASH_N_LORA_Q, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.output_lora_rank", FLASH_N_LORA_O, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.output_group_count", FLASH_N_OUT_GROUP, &n_kv);
    w_kv_u32(&kw, "deepseek4.expert_count", FLASH_N_EXPERT, &n_kv);
    w_kv_u32(&kw, "deepseek4.expert_used_count", FLASH_N_EXPERT_USED, &n_kv);
    w_kv_u32(&kw, "deepseek4.expert_feed_forward_length", FLASH_N_FF_EXP, &n_kv);
    w_kv_u32(&kw, "deepseek4.expert_shared_count", FLASH_N_EXPERT_SHARED, &n_kv);
    w_kv_u32(&kw, "deepseek4.hash_layer_count", FLASH_N_HASH_LAYER, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.sliding_window", FLASH_N_SWA, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.indexer.head_count", FLASH_N_INDEXER_HEAD, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.indexer.key_length", FLASH_N_INDEXER_HEAD_DIM, &n_kv);
    w_kv_u32(&kw, "deepseek4.attention.indexer.top_k", FLASH_N_INDEXER_TOP_K, &n_kv);
    w_kv_u32(&kw, "deepseek4.hyper_connection.count", FLASH_N_HC, &n_kv);
    w_kv_u32(&kw, "deepseek4.hyper_connection.sinkhorn_iterations", FLASH_N_HC_SINKHORN_ITER, &n_kv);
    w_kv_f32(&kw, "deepseek4.rope.freq_base", FLASH_ROPE_FREQ_BASE, &n_kv);
    w_kv_f32(&kw, "deepseek4.attention.compress_rope_freq_base", FLASH_COMPRESS_ROPE_FREQ_BASE, &n_kv);
    w_kv_f32(&kw, "deepseek4.expert_weights_scale", FLASH_EXPERT_WEIGHT_SCALE, &n_kv);
    w_kv_f32(&kw, "deepseek4.attention.layer_norm_rms_epsilon", FLASH_RMS_EPS, &n_kv);
    w_kv_f32(&kw, "deepseek4.hyper_connection.epsilon", FLASH_HC_EPS, &n_kv);
    w_kv_bool(&kw, "deepseek4.expert_weights_norm", true, &n_kv);

    /* validate_compress_ratio_metadata (ds4.c:5536) requires this array at
     * DS4_N_LAYER entries, each matching ds4_expected_layer_compress_ratio
     * (ds4.c:1078): 0 for layers < 2, else 4 on even / 128 on odd. */
    uint32_t ratios[FLASH_N_LAYER];
    for (uint32_t il = 0; il < FLASH_N_LAYER; il++) {
        ratios[il] = il < 2u ? 0u : ((il & 1u) == 0u ? 4u : 128u);
    }
    assert(ratios[LOAD_LAYER] == LOAD_RATIO);
    w_kv_u32_array(&kw, "deepseek4.attention.compress_ratios", ratios, FLASH_N_LAYER, &n_kv);

    float clamps[FLASH_N_LAYER];
    for (uint32_t il = 0; il < FLASH_N_LAYER; il++) clamps[il] = FLASH_SWIGLU_CLAMP_EXP;
    w_kv_f32_array(&kw, "deepseek4.swiglu_clamp_exp", clamps, FLASH_N_LAYER, &n_kv);

    /* vocab_load's DeepSeek4 branch (ds4.c:37759-37773) looks up these
     * exact special-token strings by name after the token/merge arrays are
     * loaded and exit(1)s (vocab_lookup, ds4.c:37682) if any is absent;
     * unlike the id fields checked elsewhere, this is unconditional and
     * independent of load_slice. */
    /* Adjacent string literals split each \xHH escape from a following
     * character that is itself a hex digit (b, e, A, D): C's \x escape is
     * greedy and would otherwise swallow it into the same byte. */
    static const char *const tokens[] = {
        "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c" ">",
        "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c" ">",
        "<\xef\xbd\x9c" "User\xef\xbd\x9c" ">",
        "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c" ">",
        "<think>",
        "</think>",
        "\xef\xbd\x9c" "DSML\xef\xbd\x9c",
    };
    w_kv_string_array(&kw, "tokenizer.ggml.tokens", tokens, 7, &n_kv);
    w_kv_string_array(&kw, "tokenizer.ggml.merges", NULL, 0, &n_kv);

    fclose(kv_f);

    /* ---- Tensor directory: one entry per reserved tensor, rel_offset
     * taken straight from the blob reservation above so the directory and
     * the blob agree exactly. */
    char *dir_ptr = NULL;
    size_t dir_size = 0;
    FILE *dir_f = open_memstream(&dir_ptr, &dir_size);
    CHECK(dir_f != NULL, "open_memstream for tensor directory failed");
    writer dw = { dir_f };
    uint64_t rel_offset_pos_in_dir = 0;
    for (uint32_t i = 0; i < g_n_dir; i++) {
        const dir_entry *e = &g_dir[i];
        char name[64];
        snprintf(name, sizeof(name), "blk.%u.%s.weight", LOAD_LAYER, e->name);
        w_str(&dw, name);
        w_u32(&dw, e->t.ndim);
        for (uint32_t d = 0; d < e->t.ndim; d++) w_u64(&dw, e->t.dim[d]);
        w_u32(&dw, e->t.type);
        if (i == 0) {
            fflush(dir_f);
            rel_offset_pos_in_dir = (uint64_t)ftell(dir_f);
        }
        w_u64(&dw, e->t.abs_offset); /* rel_offset within the blob */
    }
    fclose(dir_f);

    /* ---- Assemble the file: header, metadata, tensor directory, padding
     * to the default 32-byte alignment (no general.alignment key), then
     * the tensor data blob verbatim -- its internal layout already matches
     * the rel_offsets just written, so a single sequential write suffices. */
    FILE *out = fopen(path, "wb");
    CHECK(out != NULL, "failed to create GGUF output file in .tmp/");
    writer ow = { out };
    w_u32(&ow, GGUF_MAGIC);
    w_u32(&ow, GGUF_VERSION);
    w_u64(&ow, (uint64_t)g_n_dir);
    w_u64(&ow, n_kv);
    w_bytes(&ow, kv_ptr, kv_size);
    const uint64_t dir_file_off = 24 + kv_size;
    w_bytes(&ow, dir_ptr, dir_size);

    uint64_t pos = 24 + kv_size + dir_size;
    uint64_t aligned = (pos % 32 == 0) ? pos : pos + (32 - pos % 32);
    static const uint8_t zero32[32] = {0};
    w_bytes(&ow, zero32, (size_t)(aligned - pos));

    w_bytes(&ow, buf, (size_t)blob_size);
    fclose(out);

    free(kv_ptr);
    free(dir_ptr);
    free(buf);

    *corrupt_offset_out = dir_file_off + rel_offset_pos_in_dir;
    *orig_rel_offset_out = g_dir[0].t.abs_offset;
    return 0;
}

/* Reads the file size once, used both to size the corrupted rel_offset and
 * to sanity-check the writer above actually produced a plausible file. */
static long file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long)st.st_size;
}

/* Attempts to open the (corrupted) file at `path` and exits: 0 if
 * ds4_engine_open wrongly accepted it, 9 if it correctly returned failure.
 * ds4_die exits(1) directly if the loader's own bounds check fires instead,
 * which also reads as rejection to the parent's waitpid check. Invoked as
 * a freshly exec'd process (see the "--ablation-attempt" handling in main)
 * rather than a bare fork: by the time the ablation runs, this process has
 * already taken a real GPU through ds4_gpu_init/decode/ds4_gpu_cleanup for
 * the baseline test above, and a bare fork() of that live SYCL/Level-Zero
 * state left the forked child spinning at 100% CPU forever instead of
 * exiting -- fork() duplicates memory, not the driver's own background
 * threads, so libze's atexit-registered teardown deadlocks in the child.
 * exec gives the ablation attempt a clean process image that never touches
 * the GPU at all, since ds4_die here fires inside model_open, before
 * ds4_gpu_init is ever reached. */
/* The one engine-open configuration this whole test uses: load only
 * LOAD_LAYER via the distributed-pipeline slice mechanism, no token
 * embedding, no output head. Shared by the baseline open in main() and the
 * ablation's freshly exec'd re-open so the two configurations cannot drift
 * apart. */
static ds4_engine_options make_opt(const char *path) {
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = DS4_BACKEND_CUDA; /* SYCL reuses this constant; spec section 4. */
    opt.context_size = 64;
    opt.load_slice = true;
    opt.load_layer_start = LOAD_LAYER;
    opt.load_layer_end = LOAD_LAYER;
    opt.load_output = false;
    return opt;
}

static int ablation_attempt(const char *path) {
    ds4_engine_options opt = make_opt(path);
    ds4_engine *e = NULL;
    int rc = ds4_engine_open(&e, &opt);
    if (rc == 0) {
        fprintf(stderr, "ablation: corrupted file was WRONGLY accepted\n");
        return 0;
    }
    return 9;
}

/* sycl/ds4_sycl_common.hpp test/report-only hook; not part of
 * the ABI. Cumulative bytes actually copied host-to-device by
 * sycl_stage_host_bytes, the chokepoint essentially all host-to-device
 * weight traffic in this backend passes through. */
extern uint64_t ds4_sycl_test_stage_host_bytes_total(void);
extern void     ds4_sycl_test_stage_host_bytes_reset(void);

static double now_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--ablation-attempt") == 0) {
        return ablation_attempt(argv[2]);
    }
    if (mkdir(".tmp", 0755) != 0 && errno != EEXIST) {
        perror("mkdir .tmp");
        return 1;
    }
    const char *path = ".tmp/test_sycl_gguf_load.gguf";

    uint64_t corrupt_offset = 0;
    uint64_t orig_rel_offset = 0;
    CHECK(build_gguf(path, &corrupt_offset, &orig_rel_offset) == 0,
          "failed to generate synthetic Flash GGUF");
    CHECK(file_size(path) > 0, "generated GGUF file is empty or missing");

    /* ---- Baseline: the real ds4_engine_open path, for the first time on
     * this backend driven by real offsets from a real file.
     *
     * ds4_gpu_init explicitly first: ds4_gpu_tier_free_vram reports 0 with
     * no device initialised (sycl/ds4_sycl_placement.hpp), so reading it
     * before any init call would record a false "0 free" baseline instead
     * of the tier's real headroom, making the after-open reading look
     * smaller by comparison for the wrong reason. ds4_gpu_init is
     * idempotent (ds4_sycl.cpp: `if (g_initialised) return 1`), so calling
     * it again inside ds4_engine_open below is a no-op. */
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for the free-VRAM baseline");
    const uint64_t free_vram_before_open = ds4_gpu_tier_free_vram(0);

    ds4_engine_options opt = make_opt(path);
    ds4_engine *e = NULL;
    int rc = ds4_engine_open(&e, &opt);
    CHECK(rc == 0, "ds4_engine_open failed on the generated real GGUF file");
    fprintf(stderr, "  ds4_engine_open OK (layer %u, real Flash shape)\n", LOAD_LAYER);

    /* Proves the model-range cache actually committed real VRAM
     * for this ~1.83 GiB real GGUF layer, on the real ds4_engine_open path
     * (accelerator_cache_model_tensors), not just the synthetic-model test
     * hooks in tests/test_sycl_full_token.c. Per design-spec 6w this is a
     * side-channel check (the shared VRAM ledger), not a logits
     * comparison: the whole point of this cache is that on-vs-off
     * produces identical output, so identical output alone would prove
     * nothing about whether caching happened. */
    const uint64_t free_vram_after_open = ds4_gpu_tier_free_vram(0);
    CHECK(free_vram_after_open < free_vram_before_open,
          "ds4_gpu_tier_free_vram did not drop after engine_open: the "
          "model-range cache did not commit any VRAM for this real GGUF "
          "file, so the run below cannot be exercising the cached path");

    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, opt.context_size) == 0,
          "ds4_session_create failed against the real graph allocator");
    fprintf(stderr, "  ds4_session_create OK\n");

    /* ---- Readback: one real per-token/per-chunk decode pass over the
     * loaded layer, exercising the real staging path (spec 6l) for every
     * quant type in this layer's tensors, then reading the resulting
     * hidden state back off the device. */
    const uint64_t hc_dim = (uint64_t)FLASH_N_HC * FLASH_N_EMBD;
    float *input_hc = (float *)malloc(hc_dim * sizeof(float));
    float *output_hc = (float *)calloc((size_t)hc_dim, sizeof(float));
    CHECK(input_hc != NULL && output_hc != NULL, "out of memory allocating hc buffers");
    for (uint64_t i = 0; i < hc_dim; i++) {
        input_hc[i] = fill_val((uint32_t)i, 99u);
    }

    int tokens[1] = { 0 };
    char err[256];
    err[0] = '\0';
    int eval_rc = ds4_session_eval_layer_slice(s, tokens, 1, /*pos0=*/0,
                                               LOAD_LAYER, LOAD_LAYER,
                                               input_hc, output_hc,
                                               /*output_logits=*/false, NULL,
                                               err, sizeof(err));
    CHECK(eval_rc == 0, err[0] ? err : "ds4_session_eval_layer_slice failed");

    bool any_nonzero = false;
    bool any_different = false;
    for (uint64_t i = 0; i < hc_dim; i++) {
        CHECK(isfinite(output_hc[i]), "layer-slice output contains a non-finite value");
        if (output_hc[i] != 0.0f) any_nonzero = true;
        if (output_hc[i] != input_hc[i]) any_different = true;
    }
    CHECK(any_nonzero, "layer-slice output is all zero -- spec 6l's unstaged-mmap-read trap");
    CHECK(any_different, "layer-slice output is identical to its input -- no real computation ran");
    fprintf(stderr, "  ds4_session_eval_layer_slice OK (real weight bytes staged and read back)\n");

    /* Measurement: steady-state per-layer bytes staged and
     * wall-clock, with the model-range cache already installed (this
     * session's engine_open already ran it above, so this loop measures
     * decode after the one-time cache-install cost, not blended with it).
     * This is ONE real Flash layer's routed-expert-heavy weight set
     * (~1.83 GiB of tensors); a full decode token touches all
     * DS4_N_LAYER layers, so the report scales this by that count rather
     * than claiming this number is already a whole token. */
    {
        const int repeats = 5;
        ds4_sycl_test_stage_host_bytes_reset();
        const double t0 = now_secs();
        for (int r = 0; r < repeats; r++) {
            char err2[256];
            err2[0] = '\0';
            /* pos0 increments per repeat: the session tracks an expected
             * next KV position, so replaying pos0=0 more than once
             * against the same live session (unlike the single-call
             * correctness check above, which only ever ran once) fails
             * with a KV position mismatch rather than measuring anything
             * useful. */
            int rc2 = ds4_session_eval_layer_slice(s, tokens, 1, /*pos0=*/(uint32_t)(r + 1),
                                                   LOAD_LAYER, LOAD_LAYER,
                                                   input_hc, output_hc,
                                                   /*output_logits=*/false, NULL,
                                                   err2, sizeof(err2));
            CHECK(rc2 == 0, err2[0] ? err2 : "measurement eval failed");
        }
        const double t1 = now_secs();
        const uint64_t bytes_per_call = ds4_sycl_test_stage_host_bytes_total() / (uint64_t)repeats;
        fprintf(stderr,
                "  measurement (cache ON, real 1.83 GiB layer, %d repeats): "
                "%.3f ms/layer-eval, %llu bytes staged/layer-eval\n",
                repeats, (t1 - t0) * 1000.0 / repeats, (unsigned long long)bytes_per_call);
    }

    free(input_hc);
    free(output_hc);
    ds4_session_free(s);
    ds4_engine_close(e);
    ds4_gpu_cleanup();
    fprintf(stderr, "  teardown OK\n");

    /* Measurement, cache OFF: DS4_CUDA_DIRECT_MODEL makes
     * accelerator_cache_model_tensors (ds4.c) return immediately without
     * ever calling ds4_gpu_cache_model_range, matching CUDA's own escape
     * hatch for this exact env var. Reopens the same generated file on a
     * fresh engine/session so this is a genuine independent run, not a
     * second pass reusing state the first run already warmed. */
    {
        CHECK(setenv("DS4_CUDA_DIRECT_MODEL", "1", 1) == 0, "setenv failed");
        ds4_engine_options opt2 = make_opt(path);
        ds4_engine *e2 = NULL;
        CHECK(ds4_engine_open(&e2, &opt2) == 0,
              "ds4_engine_open failed for the cache-OFF measurement run");
        ds4_session *s2 = NULL;
        CHECK(ds4_session_create(&s2, e2, opt2.context_size) == 0,
              "ds4_session_create failed for the cache-OFF measurement run");

        float *input_hc2 = (float *)malloc(hc_dim * sizeof(float));
        float *output_hc2 = (float *)calloc((size_t)hc_dim, sizeof(float));
        CHECK(input_hc2 != NULL && output_hc2 != NULL, "out of memory (cache-OFF run)");
        for (uint64_t i = 0; i < hc_dim; i++) input_hc2[i] = fill_val((uint32_t)i, 99u);

        const int repeats = 5;
        ds4_sycl_test_stage_host_bytes_reset();
        const double t0 = now_secs();
        for (int r = 0; r < repeats; r++) {
            char err2[256];
            err2[0] = '\0';
            /* This session is fresh (no prior eval call), so position
             * starts at 0 here, unlike the cache-ON loop above which
             * continues from the single correctness-check call already
             * made against its session at pos 0. */
            int rc2 = ds4_session_eval_layer_slice(s2, tokens, 1, /*pos0=*/(uint32_t)r,
                                                   LOAD_LAYER, LOAD_LAYER,
                                                   input_hc2, output_hc2,
                                                   /*output_logits=*/false, NULL,
                                                   err2, sizeof(err2));
            CHECK(rc2 == 0, err2[0] ? err2 : "cache-OFF measurement eval failed");
        }
        const double t1 = now_secs();
        const uint64_t bytes_per_call = ds4_sycl_test_stage_host_bytes_total() / (uint64_t)repeats;
        fprintf(stderr,
                "  measurement (cache OFF, real 1.83 GiB layer, %d repeats): "
                "%.3f ms/layer-eval, %llu bytes staged/layer-eval\n",
                repeats, (t1 - t0) * 1000.0 / repeats, (unsigned long long)bytes_per_call);

        free(input_hc2);
        free(output_hc2);
        ds4_session_free(s2);
        ds4_engine_close(e2);
        ds4_gpu_cleanup();
        CHECK(unsetenv("DS4_CUDA_DIRECT_MODEL") == 0, "unsetenv failed");
    }

    /* ---- Ablation: corrupt layer 3's first tensor's rel_offset so its
     * absolute file offset falls past the end of the file, confirm the
     * loader rejects it in a freshly exec'd child (ds4_die there would
     * otherwise exit this whole test process), then revert. The baseline
     * run above is the clean pre-ablation case spec 6n asks for. */
    long sz = file_size(path);
    CHECK(sz > 0, "cannot stat generated file for ablation");
    uint64_t bad_rel_offset = (uint64_t)sz + 4096u;

    FILE *rw = fopen(path, "r+b");
    CHECK(rw != NULL, "failed to reopen GGUF file for ablation");
    CHECK(fseek(rw, (long)corrupt_offset, SEEK_SET) == 0, "fseek to corrupt target failed");
    CHECK(fwrite(&bad_rel_offset, sizeof(bad_rel_offset), 1, rw) == 1,
          "failed to write corrupted rel_offset");
    fclose(rw);

    char self_path[4096];
    ssize_t self_len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    CHECK(self_len > 0, "readlink /proc/self/exe failed");
    self_path[self_len] = '\0';

    pid_t pid = fork();
    CHECK(pid >= 0, "fork failed for ablation child");
    if (pid == 0) {
        char *child_argv[] = { self_path, (char *)"--ablation-attempt", (char *)path, NULL };
        execv(self_path, child_argv);
        perror("execv");
        _exit(127);
    }
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid, "waitpid for ablation child failed");
    bool rejected = !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(rejected, "corrupted GGUF (offset past end of file) was accepted, not rejected");
    fprintf(stderr, "  ablation OK: corrupted tensor offset was rejected (child status %d)\n", status);

    /* Revert. */
    rw = fopen(path, "r+b");
    CHECK(rw != NULL, "failed to reopen GGUF file to revert ablation");
    CHECK(fseek(rw, (long)corrupt_offset, SEEK_SET) == 0, "fseek to revert target failed");
    CHECK(fwrite(&orig_rel_offset, sizeof(orig_rel_offset), 1, rw) == 1,
          "failed to revert corrupted rel_offset");
    fclose(rw);

    unlink(path);
    fprintf(stderr, "  test_sycl_gguf_load OK\n");
    return 0;
}
