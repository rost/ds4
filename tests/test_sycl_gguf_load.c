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
 * own distributed-pipeline slice mechanism (opt.load_slice, ds4.h): by
 * default layer 3 (compress ratio 128, so no compressor/indexer
 * complexity), or layer 2 (ratio 4, indexer included) when
 * DS4_TEST_GGUF_LAYER=2 is set (see configure_layer_mode below, added
 * to rank the ratio-4 indexed path), with no token embedding and
 * no output head, since ds4_session_create only requires that
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

/* Which layer this file loads is now a runtime choice, read from
 * DS4_TEST_GGUF_LAYER (default "3"). Two values are supported:
 *
 *   3 (default): ratio 128 (ds4_expected_layer_compress_ratio, ds4.c:1085:
 *     even layers >= 2 get 4, odd get 128) needs the attn_compressor_*
 *     tensors but not the indexer ones, and is not a hash layer
 *     (il >= DS4_N_HASH_LAYER == 3), so no ffn_gate_tid2eid either -- the
 *     smallest non-token_embd-requiring, non-trivial real Flash layer.
 *     This is the configuration used for the per-kernel-family measurement.
 *   2: ratio 4, which additionally needs the indexer_* tensors
 *     (weights_bind_layer, ds4.c:5926-5933) and, since 2 < DS4_N_HASH_LAYER,
 *     a real ffn_gate_tid2eid hash routing table (ds4.c:5943-5945) -- the
 *     configuration that makes the indexer subsystem and the ratio-4
 *     indexed attention path live.
 *
 * g_load_ratio is derived from g_load_layer by the same formula
 * ds4_expected_layer_compress_ratio uses, not hardcoded per mode, so the
 * two can never drift apart. g_comp_width follows weights_validate_layout's
 * coff rule (ds4.c:5058-5075): coff is 2 for ratio 4 (the compressor also
 * carries the indexer's rope-neighbour pair) and 1 for every other ratio. */
static uint32_t g_load_layer = 3u;
static uint32_t g_load_ratio = 128u;
static uint32_t g_comp_width = FLASH_N_HEAD_DIM;
static bool     g_load_layer_is_hash = false;

static void configure_layer_mode(void) {
    const char *env = getenv("DS4_TEST_GGUF_LAYER");
    uint32_t layer = 3u;
    if (env && env[0]) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (endp == env || *endp != '\0' || (v != 2 && v != 3)) {
            fprintf(stderr, "FAIL: DS4_TEST_GGUF_LAYER must be 2 or 3\n");
            exit(1);
        }
        layer = (uint32_t)v;
    }
    g_load_layer = layer;
    g_load_ratio = layer < 2u ? 0u : ((layer & 1u) == 0u ? 4u : 128u);
    const uint32_t coff = g_load_ratio == 4u ? 2u : 1u;
    g_comp_width = coff * FLASH_N_HEAD_DIM;
    g_load_layer_is_hash = layer < FLASH_N_HASH_LAYER;
}

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

/* 27 tensors for ratio 128 (layer 3), plus 6 indexer tensors and 1
 * ffn_gate_tid2eid hash table for ratio 4 (layer 2): 34, rounded up. */
#define MAX_TENSORS 40
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
 * g_load_layer, a minimal tokenizer, and every tensor weights_bind_layer
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
                                                 g_comp_width, g_load_ratio, 0);
    ds4_tw_tensor *attn_compressor_kv = reserve(&cursor, "attn_compressor_kv", DS4_TW_TYPE_F16, 2,
                                                FLASH_N_EMBD, g_comp_width, 0);
    ds4_tw_tensor *attn_compressor_gate = reserve(&cursor, "attn_compressor_gate", DS4_TW_TYPE_F16, 2,
                                                  FLASH_N_EMBD, g_comp_width, 0);
    ds4_tw_tensor *attn_compressor_norm = reserve(&cursor, "attn_compressor_norm", DS4_TW_TYPE_F32, 1,
                                                  FLASH_N_HEAD_DIM, 0, 0);

    /* Ratio 4 (layer 2) additionally needs the indexer
     * tensors (weights_bind_layer, ds4.c:5926-5933), shapes taken from
     * ds4_test_build_flash_layer_weights's own ratio==4 branch
     * (tests/test_sycl_layer_weights.h:645-654): index_width is always
     * 2 * n_indexer_head_dim (coff==2 for indexer tensors regardless of
     * the attention compressor's own coff), since the indexer keeps its
     * own compressed KV/score state separate from the attention
     * compressor's. NULL for ratio != 4 so the fill calls below can be
     * unconditional no-ops on zero elements, matching that header's own
     * pattern. */
    ds4_tw_tensor *indexer_attn_q_b = NULL;
    ds4_tw_tensor *indexer_proj = NULL;
    ds4_tw_tensor *indexer_compressor_ape = NULL;
    ds4_tw_tensor *indexer_compressor_kv = NULL;
    ds4_tw_tensor *indexer_compressor_gate = NULL;
    ds4_tw_tensor *indexer_compressor_norm = NULL;
    if (g_load_ratio == 4u) {
        const uint64_t index_q_dim = (uint64_t)FLASH_N_INDEXER_HEAD * FLASH_N_INDEXER_HEAD_DIM;
        const uint64_t index_width = 2u * FLASH_N_INDEXER_HEAD_DIM;
        indexer_attn_q_b = reserve(&cursor, "indexer.attn_q_b", DS4_TW_TYPE_F16, 2,
                                   FLASH_N_LORA_Q, index_q_dim, 0);
        indexer_proj = reserve(&cursor, "indexer.proj", DS4_TW_TYPE_F16, 2,
                               FLASH_N_EMBD, FLASH_N_INDEXER_HEAD, 0);
        indexer_compressor_ape = reserve(&cursor, "indexer_compressor_ape", DS4_TW_TYPE_F16, 2,
                                         index_width, g_load_ratio, 0);
        indexer_compressor_kv = reserve(&cursor, "indexer_compressor_kv", DS4_TW_TYPE_F16, 2,
                                        FLASH_N_EMBD, index_width, 0);
        indexer_compressor_gate = reserve(&cursor, "indexer_compressor_gate", DS4_TW_TYPE_F16, 2,
                                          FLASH_N_EMBD, index_width, 0);
        indexer_compressor_norm = reserve(&cursor, "indexer_compressor_norm", DS4_TW_TYPE_F32, 1,
                                          FLASH_N_INDEXER_HEAD_DIM, 0, 0);
    }
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

    /* weights_bind_layer, ds4.c:5943-5945: layers below DS4_N_HASH_LAYER
     * route by hash and require this table; layer 3 (ratio 128) does not,
     * layer 2 (ratio 4) does. */
    ds4_tw_tensor *ffn_gate_tid2eid = NULL;
    if (g_load_layer_is_hash) {
        ffn_gate_tid2eid = reserve(&cursor, "ffn_gate_tid2eid", DS4_TW_TYPE_I32, 2,
                                   FLASH_N_EXPERT_USED, FLASH_N_VOCAB, 0);
    }

    const uint64_t blob_size = cursor;
    fprintf(stderr, "ds4: generating layer-%u synthetic Flash GGUF, %.2f GiB tensor payload\n",
            g_load_layer, (double)blob_size / 1073741824.0);
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
    if (g_load_ratio == 4u) {
        ds4_tw_fill_f16((uint16_t *)(buf + indexer_attn_q_b->abs_offset),
                        indexer_attn_q_b->elements, 22, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(buf + indexer_proj->abs_offset),
                        indexer_proj->elements, 23, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(buf + indexer_compressor_ape->abs_offset),
                        indexer_compressor_ape->elements, 24, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(buf + indexer_compressor_kv->abs_offset),
                        indexer_compressor_kv->elements, 25, 0.0f, 0.1f);
        ds4_tw_fill_f16((uint16_t *)(buf + indexer_compressor_gate->abs_offset),
                        indexer_compressor_gate->elements, 26, 0.0f, 0.1f);
        ds4_tw_fill_f32((float *)(buf + indexer_compressor_norm->abs_offset),
                        indexer_compressor_norm->elements, 27, 1.0f, 0.1f);
    }
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
    if (g_load_layer_is_hash) {
        ds4_tw_fill_hash_tid2eid((int32_t *)(buf + ffn_gate_tid2eid->abs_offset),
                                 FLASH_N_VOCAB, FLASH_N_EXPERT_USED, FLASH_N_EXPERT,
                                 /*layer_salt=*/g_load_layer);
    }

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
    assert(ratios[g_load_layer] == g_load_ratio);
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
        snprintf(name, sizeof(name), "blk.%u.%s.weight", g_load_layer, e->name);
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
 * g_load_layer via the distributed-pipeline slice mechanism, no token
 * embedding, no output head. Shared by the baseline open in main() and the
 * ablation's freshly exec'd re-open so the two configurations cannot drift
 * apart.
 *
 * The ratio-4 layer needs a context large enough that a
 * single zero_prefix prefill call can push g->layer_n_comp[il] (which
 * ends the call at roughly (pos0 + n_tokens) / ratio, ds4.c:28858) past
 * DS4_N_INDEXER_TOP_K (512 in this harness's metadata) to make the
 * indexed-attention branch (ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K,
 * ds4.c:29384) live at all -- that needs n_tokens > 2048, which in turn
 * needs prefill_cap, and therefore context_size, at least that large
 * (ds4_prefill_cap_for_prompt, ds4.c:12165, caps prefill_cap at the
 * context size). The ratio-128 harness has no such requirement and keeps
 * the smaller context used before. */
static ds4_engine_options make_opt_ctx(const char *path, int context_size) {
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = DS4_BACKEND_CUDA; /* SYCL reuses this constant; spec section 4. */
    opt.context_size = context_size;
    opt.load_slice = true;
    opt.load_layer_start = g_load_layer;
    opt.load_layer_end = g_load_layer;
    opt.load_output = false;
    return opt;
}

static ds4_engine_options make_opt(const char *path) {
    return make_opt_ctx(path, g_load_ratio == 4u ? 8192 : 64);
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

/* sycl/ds4_sycl_common.hpp test/report-only hook; not
 * part of the ABI. Per-kernel device profiling, active only between
 * ds4_sycl_test_profile_enable(1) and ds4_sycl_test_profile_enable(0), so
 * only the measurement loop below is counted, not engine setup. Requires
 * DS4_SYCL_PROFILE to have been set before ds4_gpu_init built the queues:
 * property::queue::enable_profiling() is a construction-time property. */
extern void     ds4_sycl_test_profile_enable(int enable);
extern void     ds4_sycl_test_profile_reset(void);
extern uint64_t ds4_sycl_test_profile_kernel_ns(void);
extern uint64_t ds4_sycl_test_profile_kernel_count(void);

/* sycl/ds4_sycl_common.hpp test/report-only hook; not
 * part of the ABI. Per-kernel-family device time, layered on the
 * aggregate above, used to rank the dense-matmul kernel family before
 * deciding which one is worth rewriting first. */
extern uint64_t    ds4_sycl_test_profile_bucket_count(void);
extern const char *ds4_sycl_test_profile_bucket_name(uint64_t i);
extern uint64_t    ds4_sycl_test_profile_bucket_ns(uint64_t i);
extern uint64_t    ds4_sycl_test_profile_bucket_calls(uint64_t i);

/* Sums the calls/eval of every profile bucket whose name
 * starts with `prefix`, so "did the indexer subsystem fire" can be
 * answered by counter evidence (spec's own standard, matching how the
 * compressed-attention test proved which attention entry ran) instead of by re-deriving which of
 * the many indexer_topk_* / indexer_scores_* / attn_indexed_mixed*
 * bucket names (dc91266) a given call path happens to use. `repeats`
 * divides out multi-repeat accumulation the same way print_kernel_ranking
 * does; pass 1 to read a raw undivided total. */
static uint64_t sum_calls_with_prefix(const char *prefix, int repeats) {
    uint64_t total = 0;
    const uint64_t nb = ds4_sycl_test_profile_bucket_count();
    const size_t plen = strlen(prefix);
    for (uint64_t i = 0; i < nb; i++) {
        const char *nm = ds4_sycl_test_profile_bucket_name(i);
        if (nm && strncmp(nm, prefix, plen) == 0) {
            total += ds4_sycl_test_profile_bucket_calls(i) / (uint64_t)repeats;
        }
    }
    return total;
}

/* The gated top-k selection kernels specifically, as
 * distinct from the indexer's other, always-on per-token bucket
 * ("indexer_qat_hadamard_fp4", ds4_gpu_dsv4_indexer_qat_tensor: called
 * unconditionally to build the indexer's own Q/K representation every
 * time index data is computed, ds4.c:22924-29228, regardless of whether
 * n_comp has crossed DS4_N_INDEXER_TOP_K) and "indexer_argmax"
 * (ds4_gpu_argmax_tensor, ds4.c:31317 and friends: greedy-decode logits
 * argmax, unrelated to the indexer despite living in ds4_sycl_
 * indexer.hpp and sharing its bucket-name prefix). A bare "indexer"
 * prefix match on the first attempt at this check silently counted the
 * always-on QAT bucket and reported a false "the indexer fired" on a
 * plain 21-step decode loop; this file's own run caught it, which is
 * exactly spec 6i's warning that a check able to pass for the wrong
 * reason needs sharper discrimination, not just any nonzero count. */
static uint64_t sum_indexer_selection_calls(int repeats) {
    return sum_calls_with_prefix("indexer_topk", repeats) +
           sum_calls_with_prefix("indexer_scores", repeats);
}

/* Both indexed-attention families: the batched path (ds4.c:29331,
 * "attn_indexed_mixed*") and the single-token fast path (ds4.c:1560ish,
 * "attn_decode_indexed_mixed_one_fast_oldhip"), which a bare
 * "attn_indexed" prefix would miss. */
static uint64_t sum_indexed_attention_calls(int repeats) {
    return sum_calls_with_prefix("attn_indexed", repeats) +
           sum_calls_with_prefix("attn_decode_indexed", repeats);
}

/* Rank the named kernel-family buckets by device time, so
 * the most expensive kernel is identified from a real measurement before
 * any of them is rewritten. Simple insertion sort: at most kSyclProfileMaxBuckets
 * entries. Factored out so its three call sites (the original
 * single-decode ranking, plus the prefill-batch-size and ratio-4
 * rankings) cannot drift apart.
 *
 * The array size below must track ds4_sycl_common.hpp's
 * kSyclProfileMaxBuckets (160, raised from 16), not just
 * the bucket count this run happens to fill. The two are duplicated rather
 * than shared because this file is plain C compiled without that C++-only
 * header; if that cap is raised again without updating this
 * literal, the print loop below silently truncates the ranking instead of
 * failing loudly, so keep this comment as the tripwire. */
static void print_kernel_ranking(const char *label, int repeats) {
    const uint64_t nb = ds4_sycl_test_profile_bucket_count();
    enum { kMaxBuckets = 160 };
    const char *names[kMaxBuckets];
    double ms[kMaxBuckets];
    uint64_t calls[kMaxBuckets];
    uint64_t n = nb < kMaxBuckets ? nb : kMaxBuckets;
    for (uint64_t i = 0; i < n; i++) {
        names[i] = ds4_sycl_test_profile_bucket_name(i);
        ms[i] = (double)ds4_sycl_test_profile_bucket_ns(i) / 1.0e6 / repeats;
        calls[i] = ds4_sycl_test_profile_bucket_calls(i) / (uint64_t)repeats;
    }
    for (uint64_t i = 1; i < n; i++) {
        const char *nm = names[i];
        double mv = ms[i];
        uint64_t cv = calls[i];
        int64_t j = (int64_t)i - 1;
        while (j >= 0 && ms[j] < mv) {
            names[j + 1] = names[j];
            ms[j + 1] = ms[j];
            calls[j + 1] = calls[j];
            j--;
        }
        names[j + 1] = nm;
        ms[j + 1] = mv;
        calls[j + 1] = cv;
    }
    fprintf(stderr, "  %s (%d repeats):\n", label, repeats);
    for (uint64_t i = 0; i < n; i++) {
        fprintf(stderr, "    %2llu. %-40s %8.4f ms/eval (%llu calls/eval)\n",
                (unsigned long long)(i + 1), names[i], ms[i],
                (unsigned long long)calls[i]);
    }
}

static double now_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

int main(int argc, char **argv) {
    /* Must run before the ablation branch below: the ablation attempt is a
     * freshly exec'd child (see its own comment) that calls make_opt()
     * too, and DS4_TEST_GGUF_LAYER is inherited across exec, not re-read
     * from anywhere else. */
    configure_layer_mode();
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
    /* Must be set before this first ds4_gpu_init call,
     * which is what actually builds every tier's queue
     * (ds4_sycl_build_devices). property::queue::enable_profiling() is a
     * construction-time property; setting the env var later would build
     * a non-profiling queue that ds4_sycl_test_profile_enable(1) could not
     * retroactively fix. */
    CHECK(setenv("DS4_SYCL_PROFILE", "1", 1) == 0, "setenv DS4_SYCL_PROFILE failed");
    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init for the free-VRAM baseline");
    const uint64_t free_vram_before_open = ds4_gpu_tier_free_vram(0);

    ds4_engine_options opt = make_opt(path);
    ds4_engine *e = NULL;
    int rc = ds4_engine_open(&e, &opt);
    CHECK(rc == 0, "ds4_engine_open failed on the generated real GGUF file");
    fprintf(stderr, "  ds4_engine_open OK (layer %u, real Flash shape)\n", g_load_layer);

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
                                               g_load_layer, g_load_layer,
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
        const int repeats = 20;
        ds4_sycl_test_stage_host_bytes_reset();
        ds4_sycl_test_profile_reset();
        ds4_sycl_test_profile_enable(1);
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
                                                   g_load_layer, g_load_layer,
                                                   input_hc, output_hc,
                                                   /*output_logits=*/false, NULL,
                                                   err2, sizeof(err2));
            CHECK(rc2 == 0, err2[0] ? err2 : "measurement eval failed");
        }
        const double t1 = now_secs();
        ds4_sycl_test_profile_enable(0);
        const uint64_t bytes_per_call = ds4_sycl_test_stage_host_bytes_total() / (uint64_t)repeats;
        const double total_ms = (t1 - t0) * 1000.0;
        const double kernel_ms = (double)ds4_sycl_test_profile_kernel_ns() / 1.0e6;
        const uint64_t kernel_count = ds4_sycl_test_profile_kernel_count();
        /* "kernel" here is a LOWER BOUND: this instrumentation covers most
         * but not all wait_and_throw call sites in the backend (some
         * launch through a helper this measurement does not reach; see
         * ds4_sycl_profile_record's comment in ds4_sycl_common.hpp for why
         * an inter-kernel gap is not reported at all). "non-kernel" is
         * therefore an UPPER BOUND on drain/host/harness time combined,
         * not a clean measurement of drain overhead alone. */
        fprintf(stderr,
                "  profile (cache ON, real 1.83 GiB layer, %d repeats): "
                "total %.3f ms/layer-eval, summed kernel (lower bound) "
                "%.3f ms/layer-eval (%llu kernels/layer-eval captured), "
                "non-kernel (upper bound) %.3f ms/layer-eval\n",
                repeats, total_ms / repeats, kernel_ms / repeats,
                (unsigned long long)(kernel_count / (uint64_t)repeats),
                (total_ms - kernel_ms) / repeats);
        fprintf(stderr,
                "  measurement (cache ON, real 1.83 GiB layer, %d repeats): "
                "%.3f ms/layer-eval, %llu bytes staged/layer-eval\n",
                repeats, total_ms / repeats, (unsigned long long)bytes_per_call);

        print_kernel_ranking("per-kernel-family ranking (cache ON)", repeats);
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
                                                   g_load_layer, g_load_layer,
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

    /* ---- Rank the prefill kernels across batch sizes
     * that straddle the MoE dispatch thresholds. This layer's routed
     * format is IQ2_XXS gate/up, Q2_K down (iq2_path, sycl/ds4_sycl_
     * moe_launch.hpp's sycl_routed_moe_iq2_dispatch), whose only
     * sorted-pairs threshold is `n_tokens > 1u` -- read directly out of
     * that function rather than assumed. Q4_K's threshold there is
     * `n_tokens >= 32u` and MXFP4's is `n_tokens > 4u` (tiny_batch
     * `n_tokens <= 4u`), but neither format is present in this layer's
     * tensors regardless of n_tokens, so no batch size here can reach
     * them; the sizes below still straddle those thresholds anyway so
     * the coverage statement printed for each size is honest about
     * what it does and does not exercise, rather than silently leaving
     * two entire dispatch regimes unstated.
     *
     * Each size gets its own fresh engine (the one-time ~0.3s model-
     * range cache install the earlier engine_open traces show is paid
     * again, not folded into the per-call timing) and its own context,
     * sized only as large as that size's repeats need, since
     * prefill_cap is capped at context_size (ds4_prefill_cap_for_prompt,
     * ds4.c:12165). Skipped in ratio-4 mode: that configuration is
     * measured separately below with its own, much larger,
     * context. */
    if (g_load_ratio != 4u) {
        static const uint32_t batch_sizes[] = { 1u, 2u, 5u, 32u, 64u };
        static const char *const batch_notes[] = {
            "decode; no MoE sorted-pairs path exists at n_tokens==1 for any format",
            "crosses this layer's iq2_path sorted-pairs gate (n_tokens>1); MXFP4/Q4_K gates unreached (format absent)",
            "crosses MXFP4's n_tokens>4 tile gate; MXFP4 itself unreached (format absent in this layer)",
            "crosses Q4_K's n_tokens>=32 tile gate; Q4_K itself unreached (format absent in this layer)",
            "this harness's prefill_cap ceiling (context_size=64)",
        };
        const int n_sizes = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));
        for (int si = 0; si < n_sizes; si++) {
            const uint32_t b = batch_sizes[si];
            const int repeats = 5;
            uint32_t ctx = (uint32_t)(repeats + 1) * b;
            if (ctx < 64u) ctx = 64u;

            ds4_engine_options optb = make_opt_ctx(path, (int)ctx);
            ds4_engine *eb = NULL;
            CHECK(ds4_engine_open(&eb, &optb) == 0, "ds4_engine_open failed for the batch sweep");
            ds4_session *sb = NULL;
            CHECK(ds4_session_create(&sb, eb, optb.context_size) == 0,
                  "ds4_session_create failed for the batch sweep");

            const uint64_t bhc = (uint64_t)b * hc_dim;
            float *bin = (float *)malloc(bhc * sizeof(float));
            float *bout = (float *)calloc((size_t)bhc, sizeof(float));
            int *btok = (int *)malloc((size_t)b * sizeof(int));
            CHECK(bin != NULL && bout != NULL && btok != NULL,
                  "out of memory allocating batch-sweep buffers");
            for (uint64_t i = 0; i < bhc; i++) bin[i] = fill_val((uint32_t)i, 99u + b);
            for (uint32_t t = 0; t < b; t++) btok[t] = (int)(t % FLASH_N_VOCAB);

            double total_ms = 0.0;
            for (int r = 0; r <= repeats; r++) {
                char errb[256];
                errb[0] = '\0';
                if (r == 1) {
                    ds4_sycl_test_profile_reset();
                    ds4_sycl_test_profile_enable(1);
                }
                const double bt0 = now_secs();
                int rcb = ds4_session_eval_layer_slice(sb, btok, b, /*pos0=*/(uint32_t)r * b,
                                                       g_load_layer, g_load_layer,
                                                       bin, bout, /*output_logits=*/false, NULL,
                                                       errb, sizeof(errb));
                const double bt1 = now_secs();
                CHECK(rcb == 0, errb[0] ? errb : "batch-sweep eval failed");
                if (r == 0) {
                    /* Discard the first execution's timing (measurement
                     * discipline: a freshly linked binary's first
                     * execution reads far above its warm cost), but
                     * still check its output is real. */
                    for (uint64_t i = 0; i < bhc; i++) {
                        CHECK(isfinite(bout[i]), "batch-sweep output has a non-finite value");
                    }
                } else {
                    total_ms += (bt1 - bt0) * 1000.0;
                }
            }
            ds4_sycl_test_profile_enable(0);
            fprintf(stderr,
                    "  prefill batch=%-3u %.4f ms/eval (warm avg of %d) -- %s\n",
                    b, total_ms / repeats, repeats, batch_notes[si]);
            {
                char label[96];
                snprintf(label, sizeof(label), "kernel ranking, batch=%u", b);
                print_kernel_ranking(label, repeats);
            }

            free(bin);
            free(bout);
            free(btok);
            ds4_session_free(sb);
            ds4_engine_close(eb);
            ds4_gpu_cleanup();
        }
    }

    /* ---- Does the ratio-4 indexer subsystem, and the
     * ratio-4 indexed attention path, actually fire?
     *
     * (a) Single decode steps. The correctness-check call above (one
     * eval at pos0=0) plus the earlier 20-repeat decode loop
     * (pos0=1..20) are both n_tokens==1 calls against this same layer-2,
     * ratio-4 session, and their kernel-family bucket counts are still
     * live: nothing between there and here calls ds4_sycl_test_profile_
     * reset. The persistent-graph decode gate is
     * `layer_n_comp[il] > decode_sparse_threshold && layer_n_index_
     * comp[il] > DS4_N_INDEXER_TOP_K` (ds4.c:22946-22950). At ratio 4,
     * comp_counts[t] = (pos0 + t + 1) / ratio (ds4.c:28858), so after 21
     * decode steps layer_n_comp is at most 21 / 4 = 5, nowhere near
     * DS4_N_INDEXER_TOP_K (512 in this harness's metadata) -- the
     * counters below confirm that arithmetic instead of just asserting
     * it. */
    if (g_load_ratio == 4u) {
        const uint64_t decode_indexer_calls = sum_indexer_selection_calls(1);
        const uint64_t decode_indexed_attn_calls = sum_indexed_attention_calls(1);
        fprintf(stderr,
                "  (a) single-decode firing check: indexer "
                "kernel calls=%llu, attn_indexed_mixed calls=%llu over 21 "
                "n_tokens==1 calls at ratio 4 (layer_n_comp tops out at 5, "
                "gate needs > %u)\n",
                (unsigned long long)decode_indexer_calls,
                (unsigned long long)decode_indexed_attn_calls,
                (unsigned)FLASH_N_INDEXER_TOP_K);
        CHECK(decode_indexer_calls == 0 && decode_indexed_attn_calls == 0,
              "indexer/indexed-attention kernels fired on a single decode step at ratio 4 -- gate arithmetic above is wrong");

        /* (b) One large zero_prefix prefill call. n_comp ends the call at
         * (pos0 + n_tokens) / ratio = n_tokens / 4 for pos0 == 0
         * (ds4.c:28858); DS4_N_INDEXER_TOP_K is 512 here, so n_tokens
         * must exceed 2048 for `ratio == 4 && n_comp > DS4_N_INDEXER_
         * TOP_K` (ds4.c:29384) to go true within this one call.
         * big_n_tokens=2100 clears that (n_comp=525) with an
         * unambiguous integer margin while staying far inside this
         * session's prefill_cap (4096 at context_size=8192, per ds4_
         * prefill_cap_for_prompt's >4096-prompt-length branch,
         * ds4.c:12165 -- this harness's own engine_open trace confirms
         * prefill_cap=4096 for this context size).
         *
         * Each repeat needs its own fresh engine and session: a
         * zero_prefix (pos0 == 0) call is only legal once per session
         * (ds4_session_slice_check_timeline rejects a second pos0 == 0
         * against an already-advanced position, the same reason the
         * cache-OFF measurement above reopens per repeat), so there is
         * no way to average this measurement inside one persistent
         * session the way the decode loop does. */
        const uint32_t big_n_tokens = 2100u;
        const int repeats = 5;
        double total_ms = 0.0;
        for (int r = 0; r <= repeats; r++) {
            ds4_engine_options optc = make_opt(path);
            ds4_engine *ec = NULL;
            CHECK(ds4_engine_open(&ec, &optc) == 0, "ds4_engine_open failed for the ratio-4 prefill run");
            ds4_session *sc = NULL;
            CHECK(ds4_session_create(&sc, ec, optc.context_size) == 0,
                  "ds4_session_create failed for the ratio-4 prefill run");

            const uint64_t chc = (uint64_t)big_n_tokens * hc_dim;
            float *cin = (float *)malloc(chc * sizeof(float));
            float *cout = (float *)calloc((size_t)chc, sizeof(float));
            int *ctok = (int *)malloc((size_t)big_n_tokens * sizeof(int));
            CHECK(cin != NULL && cout != NULL && ctok != NULL,
                  "out of memory allocating prefill buffers");
            for (uint64_t i = 0; i < chc; i++) cin[i] = fill_val((uint32_t)i, 199u);
            for (uint32_t t = 0; t < big_n_tokens; t++) ctok[t] = (int)(t % FLASH_N_VOCAB);

            if (r == 1) {
                ds4_sycl_test_profile_reset();
                ds4_sycl_test_profile_enable(1);
            }
            char errc[256];
            errc[0] = '\0';
            const double ct0 = now_secs();
            int rcc = ds4_session_eval_layer_slice(sc, ctok, big_n_tokens, /*pos0=*/0,
                                                   g_load_layer, g_load_layer,
                                                   cin, cout, /*output_logits=*/false, NULL,
                                                   errc, sizeof(errc));
            const double ct1 = now_secs();
            CHECK(rcc == 0, errc[0] ? errc : "prefill eval failed");
            for (uint64_t i = 0; i < chc; i++) {
                CHECK(isfinite(cout[i]), "prefill output has a non-finite value");
            }
            if (r == 0) {
                /* Discard the first execution's timing (measurement
                 * discipline: a freshly linked binary's first execution
                 * reads far above its warm cost), but still check its
                 * output and leave profiling off during it. */
                ds4_sycl_test_profile_enable(0);
            } else {
                total_ms += (ct1 - ct0) * 1000.0;
            }

            free(cin);
            free(cout);
            free(ctok);
            ds4_session_free(sc);
            ds4_engine_close(ec);
            ds4_gpu_cleanup();
        }
        ds4_sycl_test_profile_enable(0);

        const uint64_t prefill_indexer_calls = sum_indexer_selection_calls(repeats);
        const uint64_t prefill_indexed_attn_calls = sum_indexed_attention_calls(repeats);
        fprintf(stderr,
                "  (b) prefill n_tokens=%u ratio4: %.3f ms/eval "
                "(warm avg of %d), n_comp=%u > DS4_N_INDEXER_TOP_K=%u, "
                "indexer kernel calls=%llu/eval, attn_indexed_mixed "
                "calls=%llu/eval\n",
                big_n_tokens, total_ms / repeats, repeats, big_n_tokens / 4u,
                (unsigned)FLASH_N_INDEXER_TOP_K,
                (unsigned long long)prefill_indexer_calls,
                (unsigned long long)prefill_indexed_attn_calls);
        CHECK(prefill_indexer_calls > 0 && prefill_indexed_attn_calls > 0,
              "indexer/indexed-attention kernels did NOT fire on the ratio-4 large prefill -- the ratio-4 firing premise failed");
        print_kernel_ranking("prefill kernel ranking (ratio4)", repeats);
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
