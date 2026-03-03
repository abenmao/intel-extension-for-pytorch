
#pragma once

#include "./src/kernel_apis.hpp"

namespace esimd {
ESIMD_KERNEL_API cgf_t launch_fused_mha(
    sycl::half* query,
    sycl::half* key,
    sycl::half* value,
    sycl::half* output,
    uint8_t* mask,
    uint32_t num_batches,
    uint32_t num_heads_q,
    uint32_t num_heads_k,
    uint32_t head_size,
    uint32_t qo_len,
    uint32_t kv_len,
    bool is_head_first);

ESIMD_KERNEL_API cgf_t launch_ggemm_preprocess(
    int32_t* input,
    int32_t* output,
    int32_t n_expert);

ESIMD_KERNEL_API cgf_t launch_ggemm_preprocess_scale_align(
    float* scatter_per_token_scale,
    float* scatter_per_token_scale_aligned,
    int32_t* experts_token_count,
    int32_t* experts_token_start,
    int32_t* experts_token_start_aligned,
    int32_t n_expert);

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_256E_40T(
      int32_t* selected_experts,
      int32_t* token_to_scatter_offset,
      int32_t* experts_token_count,
      int32_t n_expert,
      int32_t shared_exp_num,
      int32_t n_tokens,
      int32_t topk);

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_256E_80T(
    int32_t* selected_experts,
    int32_t* token_to_scatter_offset,
    int32_t* experts_token_count,
    int32_t n_expert,
    int32_t shared_exp_num,
    int32_t n_tokens,
    int32_t topk);

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init(
    int32_t* selected_experts,
    int32_t* token_to_scatter_offset,
    int32_t* experts_token_count,
    int32_t n_expert,
    int32_t shared_exp_num,
    int32_t n_tokens,
    int32_t topk);
ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_k(
    int32_t* selected_experts,
    int32_t* token_to_scatter_offset,
    int32_t* experts_token_count,
    int32_t n_expert,
    int32_t shared_exp_num,
    int32_t n_tokens,
    int32_t topk);
ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_indexing(
      int32_t* experts_token_count,
      int32_t* experts_token_start,
      int32_t* experts_token_start_aligned,
      int32_t n_expert,
      int32_t shared_exp_num);

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_copy(
      sycl::half* hidden_states,
      float* experts_smooth_scale,
      int32_t* selected_experts,
      float* moe_weights,
      int32_t* token_to_scatter_offset,
      int32_t* experts_token_start,
      int8_t* scatter_tokens,
      float* scatter_per_token_scale,
      int32_t* scatter_tokens_offset,
      int32_t hd_size,
      int32_t n_expert,
      int32_t shared_exp_num,
      int32_t n_tokens,
      int32_t topk);
ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_copy_k(
      sycl::half* hidden_states,
      float* experts_smooth_scale,
      int32_t* selected_experts,
      float* moe_weights,
      int32_t* token_to_scatter_offset,
      int32_t* experts_token_start,
      int8_t* scatter_tokens,
      float* scatter_per_token_scale,
      int32_t* scatter_tokens_offset,
      int32_t hd_size,
      int32_t n_expert,
      int32_t shared_exp_num,
      int32_t n_tokens,
      int32_t topk);

ESIMD_KERNEL_API cgf_t launch_sage_attn(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    sycl::half* qScale,
    sycl::half* kScale,
    sycl::half* vScale,
    uint8_t* output,
    uint32_t kvSeqLen,
    uint32_t activationLength,
    uint32_t headQ,
    uint32_t headKv,
    uint32_t longestBatch
  );

ESIMD_KERNEL_API cgf_t launch_sage_attn_paged(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    sycl::half* qScale,
    sycl::half* kScale,
    sycl::half* vScale,
    uint32_t* cuLengthQ,
    uint32_t* cuLengthKv,
    uint32_t* blockTable,
    uint8_t* output,
    uint32_t headQ,
    uint32_t headKv,
    uint32_t longestBatch
  );

ESIMD_KERNEL_API cgf_t launch_sage_attn_decode_paged(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    float* qScale,
    float* kScale,
    float* vScale,
    uint32_t* cuLengthQ,
    uint32_t* cuLengthKv,
    uint32_t* blockTable,
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint32_t headQ,
    uint32_t headKv,
    uint32_t head_dim,
    uint32_t batch_num,
    uint32_t max_block_num_per_batch,
    uint32_t chunk_size
  );

ESIMD_KERNEL_API cgf_t launch_sage_attn_decode_paged_reduce(
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint8_t* output_final,
    uint32_t headQ,
    uint32_t head_dim,
    uint32_t batch_num,
    uint32_t chunk_count
  );

ESIMD_KERNEL_API cgf_t launch_sage_attn_decode_paged_large(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* qScale,
    uint8_t* kScale,
    uint8_t* vScale,
    uint32_t* cuLengthQ,
    uint32_t* cuLengthKv,
    uint32_t* blockTable,
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint32_t headQ,
    uint32_t headKv,
    uint32_t max_block_num_per_batch,
    uint32_t head_dim,
    uint32_t batch_num,
    uint32_t chunk_size,
    uint32_t acc_fp32
  );

ESIMD_KERNEL_API cgf_t launch_sage_attn_decode_paged_reduce_large(
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint8_t* output_final,
    uint32_t headQ,
    uint32_t head_dim,
    uint32_t batch_num,
    uint32_t chunk_count,
    uint32_t acc_fp32
  );


ESIMD_KERNEL_API cgf_t esimd_kernel_uni(
    uint8_t* p0,
    uint8_t* p1,
    uint8_t* p2,
    uint8_t* p3,
    uint8_t* p4,
    uint8_t* p5,
    uint8_t* p6,
    uint8_t* p7,
    uint8_t* p8,
    uint8_t* p9,

    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    int64_t i4,
    int64_t i5,
    int64_t i6,
    int64_t i7,
    int64_t i8,
    int64_t i9,

    float f0,
    float f1,
    float f2,
    float f3,
    float f4);


ESIMD_KERNEL_API cgf_t esimd_grouped_topk(
  uint8_t* gating_output, uint8_t* correction_bias, uint8_t* topk_weights, uint8_t* topk_ids,
  uint8_t* cpu_hidden_states, uint8_t* hidden_states,
  int64_t topk_in,
  int64_t topk_group_in,
  int64_t num_expert_group_in,
  int64_t input_len,
  int64_t renormalize,
  float routed_scaling_factor);

ESIMD_KERNEL_API cgf_t esimd_grouped_topk_kimi(
  uint8_t* gating_output, uint8_t* correction_bias, uint8_t* topk_weights, uint8_t* topk_ids,
  uint8_t* cpu_hidden_states, uint8_t* hidden_states,
  int64_t topk_in,
  int64_t topk_group_in,
  int64_t num_expert_group_in,
  int64_t input_len,
  int64_t renormalize,
  float routed_scaling_factor);

ESIMD_KERNEL_API cgf_t esimd_residual_kernel_rms_norm(
  sycl::half* weight, sycl::half* residual, sycl::half* hidden_states, sycl::half* after_res, sycl::half* hidden_states_out,
  const int hidden_size, const int num_tokens, const bool has_residual, float eps);

ESIMD_KERNEL_API cgf_t esimd_rotary_pos_emb_ds(uint8_t* qState, uint8_t* kState, uint8_t* cos_sin_cache, uint8_t* positions, uint8_t* offsets,
 int num_heads_q, int hidden_dim_q, int hd_stride_q,
 int num_heads_kv, int hidden_dim_kv, int hd_stride_kv, int head_hd_stride_kv, int input_len, int has_offset);

ESIMD_KERNEL_API cgf_t esimd_scale_dynamic_quant(const sycl::half* hidden_states,
  const float* smooth_scale,
  int8_t* quant_tokens,
  float* per_token_scale,
  const int64_t num_tokens,
  const int64_t hidden_size);

ESIMD_KERNEL_API cgf_t esimd_head_rms_norm(sycl::half* weight, sycl::half* input, sycl::half* output,
  const int num_tokens, const int total_head_num, const int head_num, const int head_dim, float eps);

ESIMD_KERNEL_API cgf_t esimd_store_kv_cache(
  sycl::half* packed_qkv,
  int32_t*    q_lens,
  int32_t*    accum_q_lens,
  int32_t*    cache_lens,
  int32_t*    cache_slot_ids,
  sycl::half*       k_cache,
  sycl::half*       v_cache,
  float*            k_scale,
  float*            v_scale,
  const int num_tokens,
  const int total_head_num,
  const int q_head_num,
  const int kv_head_num,
  const int batch_size,
  const int head_dim,
  const int64_t max_q_len,
  const int max_kv_len
);

ESIMD_KERNEL_API cgf_t esimd_store_kv_cache_int8(
    sycl::half* packed_qkv,
    int32_t* q_lens,
    int32_t* accum_q_lens,
    int32_t* cache_lens,
    int32_t* cache_slot_ids,
    int8_t* k_cache,
    int8_t* v_cache,
    float* k_scale,
    float* v_scale,
    const int num_tokens,
    const int total_head_num,
    const int q_head_num,
    const int kv_head_num,
    const int batch_size,
    const int head_dim,
    const int64_t max_q_len,
    const int max_kv_len
);

ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_int8(
    sycl::half* packed_qkv,
    int32_t*    q_lens,
    int32_t*    accum_q_lens,
    int32_t*    cache_lens,
    int32_t*    block_tables,
    int8_t*     k_cache,
    int8_t*     v_cache,
    float*      k_scale,
    float*      v_scale,
    const int   num_tokens,
    const int   total_head_num,
    const int   q_head_num,
    const int   kv_head_num,
    const int   batch_size,
    const int   head_dim,
    const int   block_size,
    const int   max_block_num_per_seq,
    const int64_t max_q_len
);

ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_fp16(
    sycl::half* packed_qkv,
    int32_t*    q_lens,
    int32_t*    accum_q_lens,
    int32_t*    cache_lens,
    int32_t*    block_tables,
    sycl::half* k_cache,
    sycl::half* v_cache,
    float*      k_scale,
    float*      v_scale,
    const int   num_tokens,
    const int   total_head_num,
    const int   q_head_num,
    const int   kv_head_num,
    const int   batch_size,
    const int   head_dim,
    const int   block_size,
    const int   max_block_num_per_seq,
    const int64_t max_q_len
);

ESIMD_KERNEL_API cgf_t esimd_moe_swiglu_dynamic_quant (
    sycl::half* scatter_tokens,
    float* smooth_scale,
    int32_t* experts_token_count,
    int32_t* experts_token_start,
    int8_t* quant_tokens,
    float* per_token_scale,
    const int total_experts_num,
    const int max_token_num,
    const int hidden_size
);

ESIMD_KERNEL_API cgf_t esimd_add_rms_norm_dynamic_quant(
  sycl::half* weight_ptr, sycl::half* input_ptr, sycl::half* residual_ptr, float* smooth_scale_ptr,
  sycl::half* after_res_ptr, sycl::half* after_norm_ptr,  int8_t* quant_tokens_ptr, float* per_token_scale_ptr,
  const int num_tokens, const int hidden_size, const bool has_residual, float eps);

ESIMD_KERNEL_API cgf_t esimd_dynamic_rotary_embedding(
  sycl::half* packed_qkv_ptr, int32_t* q_lens_ptr, int32_t* accum_q_lens_ptr,int32_t* cache_lens,
  sycl::half* cos_ptr, sycl::half* sin_ptr, sycl::half* y_ptr,
  const int batch_size, const int head_dim, const int total_head_num, const int head_num,
  const int rope_offset, const int rope_dim, const int max_q);

ESIMD_KERNEL_API cgf_t esimd_moe_gather_init(
  int32_t* scatter_tokens_offset_ptr, int32_t* tokens_index_ptr, int32_t* tokens_count_ptr,
  const int num_tokens, const int real_scatter_tokens, const int topk
);

ESIMD_KERNEL_API cgf_t esimd_moe_gather_kernel(
  sycl::half* scatter_tokens_ptr, int32_t* tokens_index_ptr, sycl::half* convergent_tokens_ptr,
  const int num_tokens, const int hidden_size, const int real_scatter_tokens, const int topk
);

ESIMD_KERNEL_API cgf_t esimd_moe_softmax_topk(
  float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
  const int num_tokens, const int num_experts, const int topk, const bool pre_softmax);

ESIMD_KERNEL_API cgf_t esimd_gating_gemm(
  uint8_t* weight, uint8_t* tokens, uint8_t* output,
  uint32_t outputRow, uint32_t outputCol, uint32_t hiddenDim);

ESIMD_KERNEL_API cgf_t esimd_flash_attn_simple(
  uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* normAlpha, uint8_t* output,
  uint32_t qLen, uint32_t seqLen, uint32_t headQ, uint32_t headKv, uint32_t headDim);

ESIMD_KERNEL_API cgf_t esimd_norm_fp16(
  uint8_t* in, uint8_t* out, uint8_t* normAlpha,
  uint32_t seqLen, uint32_t hiddenDim);
}
