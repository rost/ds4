/* Entries the SYCL backend does not implement yet.  Each later plan removes
 * the lines it implements.  Variadic C-linkage stubs accept any argument
 * list because they never inspect their arguments, the same technique used
 * by ds4_rocm_unavailable.cu.
 *
 * ds4's GPU ABI is NOT uniform on success polarity: most entries return
 * nonzero for success and 0 for failure, but a minority (notably
 * ds4_gpu_device_cache_support_tensors, still stubbed below) return 0 for
 * success and nonzero for failure.  An "unavailable" stub
 * must always return ITS OWN entry's failure value, never a bare 0 or 1
 * picked without checking, or it silently reports success.  The two int
 * macros below make that convention explicit at the call site instead of
 * leaving it implicit in a single shared "return 0":
 *
 *   SYCL_UNAVAILABLE_NONZERO_OK(name) -- for entries where nonzero means
 *     success (the majority).  Stubs to 0, the failure value.
 *   SYCL_UNAVAILABLE_ZERO_OK(name) -- for entries where 0 means success.
 *     Stubs to 1, the failure value.
 *
 * ds4_gpu_decode_graph_begin/_end use a third, entry-specific convention
 * (0 = capture committed, -1 = capture failed; see ds4_gpu.h) and are
 * defined by hand below rather than through either macro. */

#include <cstdint>
#include <cstdio>

#define SYCL_UNAVAILABLE_NONZERO_OK(name) extern "C" int name(...) { return 0; }
#define SYCL_UNAVAILABLE_ZERO_OK(name)    extern "C" int name(...) { return 1; }
#define SYCL_UNAVAILABLE_VOID(name) extern "C" void name(...) { }

/* Everything below was harvested from real link errors: build, extract the
 * undefined ds4_gpu_ symbols, grep each one's declared return type in
 * ds4_gpu.h (ds4_gpu_args.h for ds4_gpu_args_probe_auto_cuda), and add the
 * matching stub. */

/* Opaque forward declaration, matching ds4_gpu.h.  Deliberately not
 * `#include "ds4_gpu.h"`: that header declares these same names with fixed
 * argument lists, which would conflict with the variadic definitions below
 * in the same translation unit.  A pointer to an incomplete type is enough
 * for stubs that only ever return nullptr. */
struct ds4_gpu_tensor;
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

/* void returns. */
SYCL_UNAVAILABLE_VOID(ds4_gpu_decode_graph_abort)
SYCL_UNAVAILABLE_VOID(ds4_gpu_decode_graphs_invalidate)
SYCL_UNAVAILABLE_VOID(ds4_gpu_enable_q8_dequant_gemm)
SYCL_UNAVAILABLE_VOID(ds4_gpu_model_residency_skip)
SYCL_UNAVAILABLE_VOID(ds4_gpu_print_memory_report)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_glm_model)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_quality)
SYCL_UNAVAILABLE_VOID(ds4_gpu_tp_keepalive_pause)
SYCL_UNAVAILABLE_VOID(ds4_gpu_tp_set_attn_head_split)

/* int returns. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_noncausal_raw_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_low_q8_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_build_derived_artifacts)
/* Entry-specific convention, not the usual nonzero-means-success family
 * (see ds4_gpu.h). -1 matches both ROCm's own stub (ds4_rocm_compat.cu)
 * and ds4.c's CPU-only fallback definitions of these same two functions,
 * which also return -1:
 *   ds4_gpu_decode_graph_begin: 1 = replayed, 0 = capturing, -1 = run
 *     eagerly. -1 here means "no graph support, always run eagerly".
 *   ds4_gpu_decode_graph_end: 0 = capture committed and launched,
 *     -1 = capture failed (caller must re-encode the island eagerly).
 *     Returning the macro's usual 0 here would falsely report a
 *     committed capture. */
extern "C" int ds4_gpu_decode_graph_begin(...) { return -1; }
extern "C" int ds4_gpu_decode_graph_end(...) { return -1; }
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_decode_graphs_supported)
/* ds4_gpu_device_cache_support_tensors: 0 means success, a positive value
 * is a distinct error code (see ds4_cuda.cu:4103, ds4.c:57069's
 * "if (rc != 0) ... failed"). Do not switch this back to
 * SYCL_UNAVAILABLE_NONZERO_OK: that would stub it to 0, which reports a
 * successful cache install when nothing was cached. ds4_gpu_device_cache_
 * tensors itself is implemented in sycl/ds4_sycl_placement.hpp, the same
 * polarity, real weight residency instead of a stub. */
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_device_cache_support_tensors)
/* ds4_gpu_dspark_markov_argmax_tensor: DSpark speculative-decoding
 * support-model path (DS4_SUPPORT_DSPARK, ds4.c:2532), not part of
 * baseline DeepSeek V4 Flash. Out of scope; stays stubbed. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dspark_markov_argmax_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_token_quant_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_tokens_quant_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_flash_staged_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_flash_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_full_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_indexed_batch_typed_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_attention_indexed_decode_typed_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_build_kv_cache_flash_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_build_kv_cache_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_fill_selected_range_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_fill_selected_range_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_indexer_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_indexer_score_one_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_indexer_scores_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_k_b_project_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_k_b_project_typed_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_kv_lora_rms_norm_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_qk_lowrank_typed_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_qk_lowrank_typed_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_qkv_norm_store_compact_kv_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_routed_moe_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_routed_moe_one_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_router_select_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_router_select_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_store_compact_kv_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_store_indexer_k_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_glm_value_project_typed_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_rms_fold_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_router_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor)
/* ds4_gpu_matmul_quant_decode_mpp_model_view_tensor stays stubbed: its only
 * call site (ds4.c:41241, glm_graph_matmul_q8_0_decode_tensor) is in the
 * glm_graph_* family, GLM-only, out of scope for this Flash-only backend
 * (checked directly: every caller of glm_graph_matmul_q8_0_decode_tensor
 * reads a ds4_glm_gpu_graph, not Flash's ds4_gpu_graph). */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_quant_decode_mpp_model_view_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_model_range_replaced)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_pack_slot_rows_f32_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_preload_q4_expert_tables)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_pro_q4_expert_table_auto_available)
/* ds4_gpu_q8_cache_suppressed / _set_q8_cache_suppressed and
 * ds4_gpu_register_model_map_no_copy are implemented in
 * sycl/ds4_sycl_placement.hpp. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_register_support_map)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_decode_fast_attention)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_decode_score_vec4)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_fd)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_fd_for_map)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_down_hc_expand_add_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor)
/* Deliberately left stubbed: implemented ONLY in ds4_cuda.cu (CUDA-only;
 * rocm/ has no definition at all, so ROCm itself hits its own unavailable
 * stub here and ds4.c falls back).  It is also multi-GPU aware
 * (ds4_tensor_device_idx, g_gpu_peer_ok, cuda_tmp_alloc_on), so it belongs
 * with the multi-GPU streaming plan if this backend ever wants it. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_should_use_managed_kv_cache)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_async)
