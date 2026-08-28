/* Entries the SYCL backend does not implement yet.  Each later plan removes
 * the lines it implements.  Variadic C-linkage stubs accept any argument
 * list because they never inspect their arguments, the same technique used
 * by ds4_rocm_unavailable.cu.
 *
 * ds4's GPU ABI is NOT uniform on success polarity: most entries return
 * nonzero for success and 0 for failure, but a minority (notably
 * ds4_gpu_set_current_device[_fenced] and ds4_gpu_args_probe_auto_cuda)
 * return 0 for success and nonzero for failure.  An "unavailable" stub
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

/* Tensor parallelism is Metal-only, permanently out of scope. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_kslice_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_q8_tp_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_add_tensor)

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(...) { return 0; }
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tp_big_gate_wait)

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

/* Return types other than int or void. */
extern "C" uint64_t ds4_gpu_recommended_working_set_size(...) { return 0; }
extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(...) { return 0; }
extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(...) { return 0; }
extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(...) { return 0; }
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(...) { return nullptr; }
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(...) { return nullptr; }
extern "C" uint64_t ds4_gpu_tier_free_vram(...) { return 0; }

/* void returns. */
SYCL_UNAVAILABLE_VOID(ds4_gpu_decode_graph_abort)
SYCL_UNAVAILABLE_VOID(ds4_gpu_decode_graphs_invalidate)
SYCL_UNAVAILABLE_VOID(ds4_gpu_enable_q8_dequant_gemm)
SYCL_UNAVAILABLE_VOID(ds4_gpu_model_residency_skip)
SYCL_UNAVAILABLE_VOID(ds4_gpu_print_memory_report)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_glm_model)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_glm_streaming_prefill_full_layer)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_q8_cache_suppressed)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_quality)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_ssd_streaming)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_streaming_expert_cache_budget)
SYCL_UNAVAILABLE_VOID(ds4_gpu_set_streaming_expert_cache_expert_bytes)
SYCL_UNAVAILABLE_VOID(ds4_gpu_stream_expert_cache_reset_route_hotness)
SYCL_UNAVAILABLE_VOID(ds4_gpu_tp_keepalive_pause)
SYCL_UNAVAILABLE_VOID(ds4_gpu_tp_set_attn_head_split)
SYCL_UNAVAILABLE_VOID(ds4_gpu_tp_suspend_expert_sharding)

/* int returns. */
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_add_rms_norm_weight_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_add_xdev_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_argmax_tensor)
/* ds4_gpu_args_probe_auto_cuda: 0 on success, nonzero on failure (see
 * ds4_gpu_args.h and ds4_rocm_compat.cu:146).  Failing loud here means
 * --gpu-vram auto gets a clean refusal instead of a zeroed config. */
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_args_probe_auto_cuda)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_decode_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_decode_mixed_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_decode_raw_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_decode_rows_rope_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_noncausal_raw_batch_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_low_q4_K_slice_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_low_q8_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_low_q8_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_q4_K_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_q8_batch_f16_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_output_q8_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_prefill_raw_heads_range_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_prefill_raw_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_prefill_static_mixed_heads_range_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attention_prefill_static_mixed_heads_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_begin_commands)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_build_derived_artifacts)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_cache_model_range)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_cache_q8_f16_range)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_commands_active)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_commit_and_wait_selected_readback)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_compressor_prefill_ratio4_replay_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_compressor_prefill_state_ratio4_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_compressor_prefill_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_compressor_update_tensor)
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
/* ds4_gpu_device_cache_tensors / ds4_gpu_device_cache_support_tensors:
 * 0 means success, a positive value is a distinct error code (see
 * ds4_cuda.cu:3947 and ds4_cuda.cu:4103). ds4.c:56888 and ds4.c:57069
 * both check "if (rc != 0) ... failed". Do not switch these back to
 * SYCL_UNAVAILABLE_NONZERO_OK: that would stub them to 0, which reports
 * a successful cache install when nothing was cached. */
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_device_cache_support_tensors)
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_device_cache_tensors)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_directional_steering_project_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dspark_markov_argmax_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dsv4_fp8_kv_quantize_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dsv4_indexer_qat_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_dsv4_topk_mask_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_token_hc_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_token_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_token_quant_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_tokens_hc_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_tokens_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_embed_tokens_quant_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_end_commands)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_flush_commands)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_flush_encoder)
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
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_add_split_half_add_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_add_split_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_split_half_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_split_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_expand_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_split_sinkhorn_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_split_weighted_sum_norm_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_split_weighted_sum_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_weighted_sum_split_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_hc_weighted_sum_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_head_rms_norm_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_head_rms_norm_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_indexer_score_one_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_indexer_scores_decode_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_indexer_scores_prefill_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_indexer_top1_value_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_indexer_topk_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_init_multi)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_kv_fp8_store_raw_decode_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_kv_fp8_store_raw_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_pair_compressor_store_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_pair_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_rms_fold_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_router_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f16_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_f32_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_f16_out_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_hc_expand_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_kslice_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_pair_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_q8_0_top1_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_quant_decode_mpp_model_view_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_quant_kslice_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_matmul_quant_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_model_range_replaced)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_moe_handoff_pack_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_output_hc_weights_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_pack_slot_rows_f32_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_preload_q4_expert_tables)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_pro_q4_expert_table_auto_available)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_q8_cache_suppressed)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_register_model_map_no_copy)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_register_support_map)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_repeat_hc_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_repeat_hc_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rms_norm_plain_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rms_norm_plain_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rms_norm_weight_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rms_norm_weight_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rope_tail_decode_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_rope_tail_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_batch_owned_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_one_owned_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_one_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_owned_packed_combine_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_owned_slots_combine_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_owned_slots_combine_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_routed_moe_set_selected_override)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_router_select_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_router_select_tensor)
/* ds4_gpu_set_current_device[_fenced]: 0 on success, nonzero on error or
 * out-of-range tier (see ds4_gpu_mgpu.h).  ds4.c's own CPU-only stub of
 * this entry returns -1; any nonzero value is a correct failure signal
 * here since callers only ever check for exactly 0, but 1 keeps this stub
 * consistent with every other ZERO_OK entry in this file. */
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_set_current_device)
SYCL_UNAVAILABLE_ZERO_OK(ds4_gpu_set_current_device_fenced)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_decode_fast_attention)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_decode_score_vec4)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_fd)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_fd_for_map)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_map_range)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_set_model_map_spans)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_down_hc_expand_add_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_down_hc_expand_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_shared_mid_swiglu_q8_0_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_should_use_managed_kv_cache)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_signal_selected_readback_ready)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_store_raw_kv_batch_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_store_raw_kv_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_stream_expert_cache_begin_selected_load)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_stream_expert_cache_prepare_selected_batch)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_stream_expert_cache_seed_experts)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_stream_expert_cache_seed_selected)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_swiglu_tensor)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_synchronize)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_async)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_xdev)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_xdev3)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_xdev3_default_dst)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_xdev_default)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_copy_xdev_ordered)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_wait_xdev)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tensor_wait_xdev_default)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tp_batch_gate_encode)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tp_big_gate_encode)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_tp_gate_encode)
SYCL_UNAVAILABLE_NONZERO_OK(ds4_gpu_wait_selected_readback_ready)
