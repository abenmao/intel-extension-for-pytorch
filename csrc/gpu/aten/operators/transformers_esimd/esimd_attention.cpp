
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <CL/sycl.hpp>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <stdlib.h>
#include <utils/oneMKLUtils.h>
#include "../comm/ATDispatch.h"
#include "../comm/AccumulateType.h"
#include "../esimd/api.hpp"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

static Tensor fmha_esimd(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& mask,
    bool is_head_first) {
  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

  uint32_t num_batches = query.size(0);
  uint32_t num_heads_q = query.size(1);
  uint32_t num_heads_k = key.size(1);
  uint32_t qo_len = query.size(-2);
  uint32_t kv_len = key.size(-2);
  uint32_t head_size = key.size(-1);

  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();

  // TODO(zw): support other datatype

  auto cgf = esimd::launch_fused_mha(
      reinterpret_cast<sycl::half*>(query.data_ptr()),
      reinterpret_cast<sycl::half*>(key.data_ptr()),
      reinterpret_cast<sycl::half*>(value.data_ptr()),
      reinterpret_cast<sycl::half*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(mask.data_ptr()),
      num_batches,
      num_heads_q,
      num_heads_k,
      head_size,
      qo_len,
      kv_len,
      is_head_first);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return output;
}

static Tensor ggemm_preprocess(
    Tensor& experts_token_count,
    Tensor& experts_token_start,
    Tensor& scatter_per_token_scale,
    Tensor& scatter_per_token_scale_aligned,
    Tensor& output) {

  auto dpcpp_queue = dpcppGetCurrentQueue();

  // TODO(zw): support other datatype
  int n_expert = experts_token_count.size(0);

  auto o_sz = experts_token_start.sizes().vec();
  Tensor experts_token_start_aligned;
  experts_token_start_aligned = at::empty(o_sz, experts_token_start.options());

  auto cgf = esimd::launch_ggemm_preprocess(
      reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
      reinterpret_cast<int32_t*>(output.data_ptr()),
      n_expert);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  c10::xpu::getCurrentXPUStream().synchronize();

  // reuse to cal cumsum
  auto cgf2 = esimd::launch_moe_scatter_dynamic_quant_indexing(
      reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start_aligned.data_ptr()),
      n_expert,
      0); // n_expert already including shared experts
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf2);
  
  // reuse to cal cumsum
  auto cgf3 = esimd::launch_ggemm_preprocess_scale_align(
      reinterpret_cast<float*>(scatter_per_token_scale.data_ptr()),
      reinterpret_cast<float*>(scatter_per_token_scale_aligned.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start_aligned.data_ptr()),
      n_expert);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf3);

  return output;
}

static Tensor moe_scatter_dynamic_quant(
    Tensor& selected_experts,
    Tensor& moe_weights,
    Tensor& token_to_scatter_offset,
    Tensor& experts_token_count,
    Tensor& experts_token_start,
    Tensor& hidden_states,
    Tensor& experts_smooth_scale,
    Tensor& scatter_tokens,
    Tensor& scatter_per_token_scale,
    Tensor& scatter_tokens_offset,
    int64_t shared_experts_num) {

  auto dpcpp_queue = dpcppGetCurrentQueue();

  auto o_sz = experts_token_start.sizes().vec();
  Tensor experts_token_start_aligned;
  experts_token_start_aligned = at::empty(o_sz, experts_token_start.options());

  // TODO(zw): support other datatype
  int topk = selected_experts.size(1);
  int n_tokens = selected_experts.size(0);
  int n_expert = experts_token_count.size(0);
  int hd_size = hidden_states.size(1);

  if (n_tokens == 40 && topk==8)
  {
      auto cgf = esimd::launch_moe_scatter_dynamic_quant_init_256E_40T(
          reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
          reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
          reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
          n_expert,
          shared_experts_num,
          n_tokens,
          topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
  else if (n_tokens == 80 && topk==8)
  {
      auto cgf = esimd::launch_moe_scatter_dynamic_quant_init_256E_80T(
          reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
          reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
          reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
          n_expert,
          shared_experts_num,
          n_tokens,
          topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
  else
  {
    if(topk==8){
      auto cgf = esimd::launch_moe_scatter_dynamic_quant_init(
          reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
          reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
          reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
          n_expert,
          shared_experts_num,
          n_tokens,
          topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
    else{
              auto cgf = esimd::launch_moe_scatter_dynamic_quant_init_k(
          reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
          reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
          reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
          n_expert,
          shared_experts_num,
          n_tokens,
          topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  }

  

auto cgf2 = esimd::launch_moe_scatter_dynamic_quant_indexing(
      reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start_aligned.data_ptr()),
      n_expert,
      shared_experts_num);
  
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf2);

if(topk==8){
      auto cgf3 = esimd::launch_moe_scatter_dynamic_quant_copy(
      reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
      reinterpret_cast<float*>(experts_smooth_scale.data_ptr()),
      reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
      reinterpret_cast<float*>(moe_weights.data_ptr()),
      reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
      reinterpret_cast<int8_t*>(scatter_tokens.data_ptr()),
      reinterpret_cast<float*>(scatter_per_token_scale.data_ptr()),
      reinterpret_cast<int32_t*>(scatter_tokens_offset.data_ptr()),
      hd_size,
      n_expert,
      shared_experts_num,
      n_tokens,
      topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf3);
}
else{
      auto cgf3 = esimd::launch_moe_scatter_dynamic_quant_copy_k(
      reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
      reinterpret_cast<float*>(experts_smooth_scale.data_ptr()),
      reinterpret_cast<int32_t*>(selected_experts.data_ptr()),
      reinterpret_cast<float*>(moe_weights.data_ptr()),
      reinterpret_cast<int32_t*>(token_to_scatter_offset.data_ptr()),
      reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
      reinterpret_cast<int8_t*>(scatter_tokens.data_ptr()),
      reinterpret_cast<float*>(scatter_per_token_scale.data_ptr()),
      reinterpret_cast<int32_t*>(scatter_tokens_offset.data_ptr()),
      hd_size,
      n_expert,
      shared_experts_num,
      n_tokens,
      topk);
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf3);
    }

  return selected_experts;
}

static Tensor sage_attn(
    Tensor& qState,
    Tensor& kState,
    Tensor& vState,
    Tensor& qScale,
    Tensor& kScale,
    Tensor& vScale,
    Tensor& output,
    int64_t longestBatch) {

  auto dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t kvSeqLen = kState.size(0);
  uint32_t activationLength = qState.size(0);
  uint32_t headQ = qState.size(1);
  uint32_t headKv = kState.size(1);

  auto cgf = esimd::launch_sage_attn(
      reinterpret_cast<uint8_t*>(qState.data_ptr()),
      reinterpret_cast<uint8_t*>(kState.data_ptr()),
      reinterpret_cast<uint8_t*>(vState.data_ptr()),
      reinterpret_cast<sycl::half*>(qScale.data_ptr()),
      reinterpret_cast<sycl::half*>(kScale.data_ptr()),
      reinterpret_cast<sycl::half*>(vScale.data_ptr()),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      kvSeqLen,
      activationLength,
      headQ,
      headKv,
      longestBatch);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return output;
}

static Tensor sage_attn_paged(
    Tensor& qState,   // [num_tokens_q, query_heads, head_dim]
    Tensor& kState,   // [num_blocks, block_size, key_heads, head_dim]
    Tensor& vState,   // [num_blocks, block_size, key_heads, head_dim]
    Tensor& qScale,   // [q_len, q_head, 1]
    Tensor& kScale,   // [kv_len, kv_head, 1]
    Tensor& vScale,   // [1, kv_head, head_dim]
    Tensor& cuLengthQ,    // [batch + 1]
    Tensor& cuLengthKv,   // [batch + 1]
    Tensor& blockTable,   // [batch, num_max_seq_block]
    Tensor& output,   // [num_tokens_q, query_heads, head_dim]
    int64_t longestBatch) {

  auto dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t headQ = qState.size(1);
  uint32_t headKv = kState.size(2);
  uint32_t headDim = qState.size(2);

  auto cgf = esimd::esimd_kernel_uni(
    reinterpret_cast<uint8_t*>(qState.data_ptr()),
    reinterpret_cast<uint8_t*>(kState.data_ptr()),
    reinterpret_cast<uint8_t*>(vState.data_ptr()),
    reinterpret_cast<uint8_t*>(qScale.data_ptr()),
    reinterpret_cast<uint8_t*>(kScale.data_ptr()),
    reinterpret_cast<uint8_t*>(vScale.data_ptr()),
    reinterpret_cast<uint8_t*>(cuLengthQ.data_ptr()),
    reinterpret_cast<uint8_t*>(cuLengthKv.data_ptr()),
    reinterpret_cast<uint8_t*>(blockTable.data_ptr()),
    reinterpret_cast<uint8_t*>(output.data_ptr()),
    static_cast<int64_t>(8888),
    longestBatch,
    static_cast<int64_t>(headQ),
    static_cast<int64_t>(headKv),
    static_cast<int64_t>(headDim),
    static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0),
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f
  );

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return output;
}

static Tensor sage_attn_decode_paged(
    Tensor& qState,   // [batch, num_tokens_q, query_heads, head_dim]
    Tensor& kState,   // [num_blocks, block_size, key_heads, head_dim]
    Tensor& vState,   // [num_blocks, block_size, key_heads, head_dim]
    Tensor& qScale,   // [batch, q_len, q_head, 1]
    Tensor& kScale,   // [batch, kv_len, kv_head, 1]
    Tensor& vScale,   // [1, kv_head, head_dim]
    Tensor& cuLengthQ,    // [batch + 1]
    Tensor& cuLengthKv,   // [batch + 1]
    Tensor& blockTable,   // [batch, num_max_seq_block]
    Tensor& output,   // [batch, chunk_num, num_tokens_q, query_heads, head_dim]
    Tensor& output_lse,  // [batch, chunk_num, num_tokens_q, query_heads, 1]
    Tensor& output_max,  // [batch, chunk_num, num_tokens_q, query_heads, 1]
    Tensor& output_final,  // [batch, num_tokens_q, query_heads, head_dim]
    int64_t batch_num,
    int64_t chunk_size) {

  auto dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t headQ = qState.size(2);  // qstate has batch dim: [batch, num_tokens_q, query_heads, head_dim]
  uint32_t headKv = kState.size(2);
  uint32_t head_dim = qState.size(3);
  uint32_t max_block_num_per_batch = blockTable.size(1);
  // assume block size is 512
  uint32_t block_size = 512;
  uint32_t chunk_count = (max_block_num_per_batch * block_size + chunk_size - 1) / chunk_size;
  chunk_count = output_max.size(1);


  uint32_t q_head_num_per_kv_head = headQ / headKv;
  uint32_t acc_fp32 = 0;
  c10::ScalarType acc_scalar_type = output.scalar_type();
  if(acc_scalar_type == at::kFloat)
    acc_fp32 = 1;

  if (head_dim == 64 || (q_head_num_per_kv_head == 16 || q_head_num_per_kv_head == 12 || q_head_num_per_kv_head == 10 || q_head_num_per_kv_head == 8))
  {
   auto cgf = esimd::launch_sage_attn_decode_paged_large(
      reinterpret_cast<uint8_t*>(qState.data_ptr()),
      reinterpret_cast<uint8_t*>(kState.data_ptr()),
      reinterpret_cast<uint8_t*>(vState.data_ptr()),
      reinterpret_cast<uint8_t*>(qScale.data_ptr()),
      reinterpret_cast<uint8_t*>(kScale.data_ptr()),
      reinterpret_cast<uint8_t*>(vScale.data_ptr()),
      reinterpret_cast<uint32_t*>(cuLengthQ.data_ptr()),
      reinterpret_cast<uint32_t*>(cuLengthKv.data_ptr()),
      reinterpret_cast<uint32_t*>(blockTable.data_ptr()),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(output_lse.data_ptr()),
      reinterpret_cast<uint8_t*>(output_max.data_ptr()),
      headQ,
      headKv,
      max_block_num_per_batch,
      head_dim,
      batch_num,
      chunk_size,
      acc_fp32
    );
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    
    auto cgf2 = esimd::launch_sage_attn_decode_paged_reduce_large(
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(output_lse.data_ptr()),
      reinterpret_cast<uint8_t*>(output_max.data_ptr()),
      reinterpret_cast<uint8_t*>(output_final.data_ptr()),
      headQ,
      head_dim,
      batch_num,
      chunk_count,
      acc_fp32);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf2);

     return output;
   }

  auto cgf = esimd::launch_sage_attn_decode_paged(
      reinterpret_cast<uint8_t*>(qState.data_ptr()),
      reinterpret_cast<uint8_t*>(kState.data_ptr()),
      reinterpret_cast<uint8_t*>(vState.data_ptr()),
      reinterpret_cast<float*>(qScale.data_ptr()),
      reinterpret_cast<float*>(kScale.data_ptr()),
      reinterpret_cast<float*>(vScale.data_ptr()),
      reinterpret_cast<uint32_t*>(cuLengthQ.data_ptr()),
      reinterpret_cast<uint32_t*>(cuLengthKv.data_ptr()),
      reinterpret_cast<uint32_t*>(blockTable.data_ptr()),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(output_lse.data_ptr()),
      reinterpret_cast<uint8_t*>(output_max.data_ptr()),
      headQ,
      headKv,
      head_dim,
      batch_num,
      max_block_num_per_batch,
      chunk_size);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  auto cgf2 = esimd::launch_sage_attn_decode_paged_reduce(
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(output_lse.data_ptr()),
      reinterpret_cast<uint8_t*>(output_max.data_ptr()),
      reinterpret_cast<uint8_t*>(output_final.data_ptr()),
      headQ,
      head_dim,
      batch_num,
      chunk_count);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf2);

  return output;
}

static Tensor residual_rms_norm(
    Tensor& weight, 
    Tensor& residual, 
    Tensor& hidden_states, 
    Tensor& after_res,
    Tensor& hidden_states_out,
    double variance_epsilon) {

  auto dpcpp_queue = dpcppGetCurrentQueue();
  bool add_residual = residual.defined() ? true : false;
  int hidden_size = hidden_states.size(1);
  int input_len = hidden_states.size(0);

  auto cgf = esimd::esimd_residual_kernel_rms_norm(
        reinterpret_cast<sycl::half*>(weight.data_ptr()),
        add_residual ? reinterpret_cast<sycl::half*>(residual.data_ptr()) : nullptr,
        reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
        reinterpret_cast<sycl::half*>(after_res.data_ptr()),
        reinterpret_cast<sycl::half*>(hidden_states_out.data_ptr()),
        hidden_size,
        input_len,
        add_residual,
        variance_epsilon);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return hidden_states_out;
}

static Tensor rotary_pos_emb_ds(
    Tensor& qState, Tensor& kState, Tensor& cos_sin_cache, Tensor& positions, Tensor& offsets) {

    auto dpcpp_queue = dpcppGetCurrentQueue();
    int num_heads_q = qState.size(1);
    int hidden_dim_q = qState.size(2);
    int hd_stride_q = qState.size(2);
    int num_heads_kv = kState.size(1);
    int hd_stride_kv = kState.size(2);
    int head_hd_stride_kv = hd_stride_kv * num_heads_kv;
    int input_len = qState.size(0);
    int has_offset = offsets.defined() ? 1 : 0;
    int hidden_dim_kv = hidden_dim_q;

    auto cgf = esimd::esimd_rotary_pos_emb_ds(
            reinterpret_cast<uint8_t*>(qState.data_ptr()),
            reinterpret_cast<uint8_t*>(kState.data_ptr()),
            reinterpret_cast<uint8_t*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<uint8_t*>(positions.data_ptr()),
            has_offset ? reinterpret_cast<uint8_t*>(offsets.data_ptr()) : nullptr,
            num_heads_q, hidden_dim_q, hd_stride_q,
            num_heads_kv, hidden_dim_kv, hd_stride_kv, head_hd_stride_kv, input_len, has_offset);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return qState;
}

static Tensor grouped_topk(
    Tensor& gating_output, Tensor& correction_bias, Tensor& topk_weights, Tensor& topk_ids,
    int64_t topk_in,
    int64_t topk_group_in,
    int64_t num_expert_group_in,
    int64_t renormalize) {

    if (topk_in != 8 || topk_group_in != 4 || num_expert_group_in != 8 || gating_output.size(1) != 256)
    {
        printf("only support topk 8, topk group 4, num_expert_group 8, num_expert 256! \n");
        return topk_ids;
    }

    auto dpcpp_queue = dpcppGetCurrentQueue();

    int input_len = gating_output.size(0);

    auto cgf = esimd::esimd_grouped_topk(
            reinterpret_cast<uint8_t*>(gating_output.data_ptr()),
            reinterpret_cast<uint8_t*>(correction_bias.data_ptr()),
            reinterpret_cast<uint8_t*>(topk_weights.data_ptr()),
            reinterpret_cast<uint8_t*>(topk_ids.data_ptr()),
            nullptr,
            nullptr,
            topk_in,
            topk_group_in,
            num_expert_group_in,
            input_len,
            renormalize, 0);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return topk_ids;
}

static Tensor grouped_topk_kimi(
    Tensor& gating_output, Tensor& correction_bias, Tensor& topk_weights, Tensor& topk_ids,
    int64_t topk_in,
    int64_t topk_group_in,
    int64_t num_expert_group_in,
    int64_t renormalize) {

    if (topk_in != 8 || topk_group_in != 1 || num_expert_group_in != 1 || gating_output.size(1) != 384)
    {
        printf("only support topk 8, topk group 1, num_expert_group 1, num_expert 384! \n");
        return topk_ids;
    }

    auto dpcpp_queue = dpcppGetCurrentQueue();

    int input_len = gating_output.size(0);

    auto cgf = esimd::esimd_grouped_topk_kimi(
            reinterpret_cast<uint8_t*>(gating_output.data_ptr()),
            reinterpret_cast<uint8_t*>(correction_bias.data_ptr()),
            reinterpret_cast<uint8_t*>(topk_weights.data_ptr()),
            reinterpret_cast<uint8_t*>(topk_ids.data_ptr()),
            nullptr,
            nullptr,
            topk_in,
            topk_group_in,
            num_expert_group_in,
            input_len,
            renormalize, 0);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return topk_ids;
}

static Tensor scale_dynamic_quant(
    Tensor& hidden_states,
    Tensor& smooth_scale,
    Tensor& quant_tokens,
    Tensor& per_token_scale
) {
    TORCH_CHECK(
      hidden_states.scalar_type() == at::kHalf, "scale_dynamic_quant hidden_states only supports half datatype");
    TORCH_CHECK(
      smooth_scale.scalar_type() == at::kFloat, "scale_dynamic_quant smooth_scale only supports float datatype");
    TORCH_CHECK(
      per_token_scale.scalar_type() == at::kFloat, "scale_dynamic_quant per_token_scale only supports float datatype");

    int64_t num_tokens = hidden_states.size(0);
    int64_t hidden_size = hidden_states.size(1);

    auto dpcpp_queue = dpcppGetCurrentQueue();


    auto cgf = esimd::esimd_scale_dynamic_quant(
            reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
            reinterpret_cast<float*>(smooth_scale.data_ptr()),
            reinterpret_cast<int8_t*>(quant_tokens.data_ptr()),
            reinterpret_cast<float*>(per_token_scale.data_ptr()),
            num_tokens, hidden_size);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return quant_tokens;
}


static Tensor head_rms_norm(
    Tensor& weight, Tensor& hidden_states, Tensor& output,
    double eps,
    int64_t head_num) {
    int num_tokens = hidden_states.size(0);
    int total_head_num = hidden_states.size(1);
    int head_dim = hidden_states.size(2);
    assert(head_dim == 32 || head_dim == 64 || head_dim == 128);
    assert(hidden_states.scalar_type() == at::kHalf);
    assert(hidden_states.scalar_type() == weight.scalar_type());
    auto dpcpp_queue = dpcppGetCurrentQueue();

    auto cgf = esimd::esimd_head_rms_norm(
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
            reinterpret_cast<sycl::half*>(output.data_ptr()),
            num_tokens,
            total_head_num,
            head_num,
            head_dim,
            eps);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return output;
}

static Tensor store_kv_cache(
    Tensor& packed_qkv,
    Tensor& q_lens,
    Tensor& accum_q_lens,
    Tensor& cache_lens,
    Tensor& cache_slot_ids,
    Tensor& k_cache,
    Tensor& v_cache,
    Tensor& k_scale,
    Tensor& v_scale,
    int64_t max_q_len
) {
    int num_tokens = packed_qkv.size(0);
    int total_head_num = packed_qkv.size(1);
    int head_dim = packed_qkv.size(2);
    int kv_head_num = k_cache.size(1);
    int max_kv_len = k_cache.size(2);
    int q_head_num = total_head_num - 2 * kv_head_num;
    int max_batch_size = q_lens.size(0);

    assert(head_dim == 32 || head_dim == 64 || head_dim == 128);
    assert(packed_qkv.scalar_type() == at::kHalf);

    auto dpcpp_queue = dpcppGetCurrentQueue();

    auto cgf = k_cache.scalar_type() == at::kHalf ? 
                esimd::esimd_store_kv_cache(
                    reinterpret_cast<sycl::half*>(packed_qkv.data_ptr()),
                    reinterpret_cast<int32_t*>(q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(accum_q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_slot_ids.data_ptr()),
                    reinterpret_cast<sycl::half*>(k_cache.data_ptr()),
                    reinterpret_cast<sycl::half*>(v_cache.data_ptr()),
                    k_scale.defined() ? reinterpret_cast<float*>(k_scale.data_ptr()): nullptr,
                    v_scale.defined() ? reinterpret_cast<float*>(v_scale.data_ptr()): nullptr,
                    num_tokens, total_head_num, q_head_num, kv_head_num,
                    max_batch_size, head_dim, max_q_len, max_kv_len
                ) :
                esimd::esimd_store_kv_cache_int8(
                    reinterpret_cast<sycl::half*>(packed_qkv.data_ptr()),
                    reinterpret_cast<int32_t*>(q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(accum_q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_slot_ids.data_ptr()),
                    reinterpret_cast<int8_t*>(k_cache.data_ptr()),
                    reinterpret_cast<int8_t*>(v_cache.data_ptr()),
                    k_scale.defined() ? reinterpret_cast<float*>(k_scale.data_ptr()): nullptr,
                    v_scale.defined() ? reinterpret_cast<float*>(v_scale.data_ptr()): nullptr,
                    num_tokens, total_head_num, q_head_num, kv_head_num,
                    max_batch_size, head_dim, max_q_len, max_kv_len
                );

    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return k_cache;
}


static Tensor store_paged_kv_cache(
    Tensor& packed_qkv,
    Tensor& q_lens,
    Tensor& accum_q_lens,
    Tensor& cache_lens,
    Tensor& block_table,
    Tensor& k_cache,
    Tensor& v_cache,
    Tensor& k_scale,
    Tensor& v_scale,
    int64_t max_q_len
) {
    int num_tokens = packed_qkv.size(0);
    int total_head_num = packed_qkv.size(1);
    int head_dim = packed_qkv.size(2);
    int kv_head_num = k_cache.size(2);
    int block_size = k_cache.size(1);
    int q_head_num = total_head_num - 2 * kv_head_num;
    int max_batch_size = q_lens.size(0);
    int max_block_num_per_seq = block_table.size(1);

    assert(head_dim == 32 || head_dim == 64 || head_dim == 128);
    assert(packed_qkv.scalar_type() == at::kHalf);
    assert(k_cache.scalar_type() == at::kHalf || k_cache.scalar_type() == at::kINT8);

    auto dpcpp_queue = dpcppGetCurrentQueue();

    // TODO: check k/v cache type
    auto cgf = k_cache.scalar_type() == at::kHalf ?
                esimd::esimd_store_paged_kv_cache_fp16(
                    reinterpret_cast<sycl::half*>(packed_qkv.data_ptr()),
                    reinterpret_cast<int32_t*>(q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(accum_q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(block_table.data_ptr()),
                    reinterpret_cast<sycl::half*>(k_cache.data_ptr()),
                    reinterpret_cast<sycl::half*>(v_cache.data_ptr()),
                    k_scale.defined() ? reinterpret_cast<float*>(k_scale.data_ptr()): nullptr,
                    v_scale.defined() ? reinterpret_cast<float*>(v_scale.data_ptr()): nullptr,
                    num_tokens, total_head_num, q_head_num, kv_head_num,
                    max_batch_size, head_dim, block_size, max_block_num_per_seq, max_q_len
                ) :
                esimd::esimd_store_paged_kv_cache_int8(
                    reinterpret_cast<sycl::half*>(packed_qkv.data_ptr()),
                    reinterpret_cast<int32_t*>(q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(accum_q_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(cache_lens.data_ptr()),
                    reinterpret_cast<int32_t*>(block_table.data_ptr()),
                    reinterpret_cast<int8_t*>(k_cache.data_ptr()),
                    reinterpret_cast<int8_t*>(v_cache.data_ptr()),
                    k_scale.defined() ? reinterpret_cast<float*>(k_scale.data_ptr()): nullptr,
                    v_scale.defined() ? reinterpret_cast<float*>(v_scale.data_ptr()): nullptr,
                    num_tokens, total_head_num, q_head_num, kv_head_num,
                    max_batch_size, head_dim, block_size, max_block_num_per_seq, max_q_len
                );

    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return k_cache;
}


static Tensor add_rms_norm_dynamic_quant(
    Tensor& weight, Tensor& hidden_states, Tensor& residual, Tensor& smooth_scale,
    Tensor& after_res, Tensor& after_norm, Tensor& quant_tokens, Tensor& per_token_scale,
    double eps) {
    int num_tokens = hidden_states.size(0);
    int hidden_size = hidden_states.size(1);
    assert(hidden_states.scalar_type() == at::kHalf);
    assert(hidden_states.scalar_type() == weight.scalar_type());
    auto dpcpp_queue = dpcppGetCurrentQueue();
    bool has_residual = residual.defined() ? true : false;
    auto cgf = esimd::esimd_add_rms_norm_dynamic_quant(
            reinterpret_cast<sycl::half*>(weight.data_ptr()),
            reinterpret_cast<sycl::half*>(hidden_states.data_ptr()),
            has_residual ? reinterpret_cast<sycl::half*>(residual.data_ptr()) : nullptr,
            reinterpret_cast<float*>(smooth_scale.data_ptr()),
            reinterpret_cast<sycl::half*>(after_res.data_ptr()),
            reinterpret_cast<sycl::half*>(after_norm.data_ptr()),
            reinterpret_cast<int8_t*>(quant_tokens.data_ptr()),
            reinterpret_cast<float*>(per_token_scale.data_ptr()),
            num_tokens,
            hidden_size,
            has_residual,
            eps);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return quant_tokens;
}

static Tensor dynamic_rotary_embedding(
    Tensor& packed_qkv, Tensor& q_lens, Tensor& accum_q_lens, Tensor& cache_lens,
    Tensor& cos, Tensor& sin, Tensor& y,
    int64_t q_head, int64_t kv_head, int64_t rope_offset, int64_t rope_dim,
    int64_t max_q) {
    int64_t batch_size = q_lens.size(0);
    int64_t all_head = packed_qkv.size(1);
    int64_t head_dim = packed_qkv.size(2);
    assert(packed_qkv.scalar_type() == y.scalar_type());
    assert(packed_qkv.scalar_type() == at::kHalf);
    assert(head_dim == 32 || head_dim == 64 || head_dim == 128);
    auto dpcpp_queue = dpcppGetCurrentQueue();

    auto cgf = esimd::esimd_dynamic_rotary_embedding(
            reinterpret_cast<sycl::half*>(packed_qkv.data_ptr()),
            reinterpret_cast<int32_t*>(q_lens.data_ptr()),
            reinterpret_cast<int32_t*>(accum_q_lens.data_ptr()),
            reinterpret_cast<int32_t*>(cache_lens.data_ptr()),
            reinterpret_cast<sycl::half*>(cos.data_ptr()),
            reinterpret_cast<sycl::half*>(sin.data_ptr()),
            reinterpret_cast<sycl::half*>(y.data_ptr()),
            batch_size,
            head_dim,
            all_head,
            q_head + kv_head,
            rope_offset,
            rope_dim,
            max_q);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return y;
}


static Tensor moe_swiglu_dynamic_quant(
    Tensor& scatter_tokens,
    Tensor& smooth_scale,
    Tensor& experts_token_count,
    Tensor& experts_token_start,
    Tensor& quant_tokens,
    Tensor& per_token_scale,
    int64_t total_experts_num,
    int64_t max_token_num) {
    assert(scatter_tokens.scalar_type() == at::kHalf && smooth_scale.scalar_type() == at::kFloat);
    int hidden_size = scatter_tokens.size(1)/2;
    auto dpcpp_queue = dpcppGetCurrentQueue();
    auto cgf = esimd::esimd_moe_swiglu_dynamic_quant(
            reinterpret_cast<sycl::half*>(scatter_tokens.data_ptr()),
            reinterpret_cast<float*>(smooth_scale.data_ptr()),
            reinterpret_cast<int32_t*>(experts_token_count.data_ptr()),
            reinterpret_cast<int32_t*>(experts_token_start.data_ptr()),
            reinterpret_cast<int8_t*>(quant_tokens.data_ptr()),
            reinterpret_cast<float*>(per_token_scale.data_ptr()),
            total_experts_num,
            max_token_num,
            hidden_size
            );
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return quant_tokens;
}

static Tensor moe_gather_function(
    Tensor& scatter_tokens,
    Tensor& scatter_tokens_offset,
    Tensor& convergent_tokens,
    int64_t topk) {
    int64_t num_tokens = convergent_tokens.size(0);
    int64_t hidden_size = convergent_tokens.size(1);
    int64_t real_scatter_tokens = scatter_tokens.size(0);
    auto dpcpp_queue = dpcppGetCurrentQueue();

    assert(real_scatter_tokens == topk * num_tokens);
    assert(scatter_tokens.scalar_type() == at::kHalf);

    Tensor tokens_index = at::zeros({num_tokens, topk}, at::device(scatter_tokens_offset.device()).dtype(scatter_tokens_offset.dtype()));
    Tensor tokens_count = at::zeros({num_tokens}, at::device(scatter_tokens_offset.device()).dtype(scatter_tokens_offset.dtype()));

    auto cgf1 = esimd::esimd_moe_gather_init(
        reinterpret_cast<int32_t*>(scatter_tokens_offset.data_ptr()),
        reinterpret_cast<int32_t*>(tokens_index.data_ptr()),
        reinterpret_cast<int32_t*>(tokens_count.data_ptr()),
        num_tokens,
        real_scatter_tokens,
        topk
        );
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf1);

    auto cgf2 = esimd::esimd_moe_gather_kernel(
            reinterpret_cast<sycl::half*>(scatter_tokens.data_ptr()),
            reinterpret_cast<int32_t*>(tokens_index.data_ptr()),
            reinterpret_cast<sycl::half*>(convergent_tokens.data_ptr()),
            num_tokens,
            hidden_size,
            real_scatter_tokens,
            topk
            );
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf2);

    return convergent_tokens;
}

static Tensor moe_softmax_topk(
    Tensor& gating_output, Tensor& selected_experts, Tensor& moe_weights,
    int64_t topk, bool pre_softmax) {
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    assert(gating_output.scalar_type() == at::kFloat);
    assert(gating_output.scalar_type() == moe_weights.scalar_type());
    auto dpcpp_queue = dpcppGetCurrentQueue();

    auto cgf = esimd::esimd_moe_softmax_topk(
            reinterpret_cast<float*>(gating_output.data_ptr()),
            reinterpret_cast<int*>(selected_experts.data_ptr()),
            reinterpret_cast<float*>(moe_weights.data_ptr()),
            num_tokens,
            num_experts,
            topk,
            pre_softmax);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return selected_experts;
}

static Tensor specialized_gating_gemm(
    Tensor& weight, Tensor& tokens, Tensor& output) {
    uint32_t outputRow = weight.size(0);
    uint32_t outputCol = tokens.size(0);
    uint32_t hiddenDim = weight.size(1);

    assert(output.scalar_type() == at::kFloat);
    assert(weight.scalar_type() == tokens.scalar_type());
    auto dpcpp_queue = dpcppGetCurrentQueue();

    auto cgf = esimd::esimd_gating_gemm(
            reinterpret_cast<uint8_t*>(weight.data_ptr()),
            reinterpret_cast<uint8_t*>(tokens.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            outputRow,
            outputCol,
            hiddenDim);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return output;
}

static Tensor flash_attn_simple(
    Tensor& query, Tensor& key, Tensor& value, int64_t flag) {
    assert(query.scalar_type() == at::kHalf);
    assert(key.scalar_type() == at::kHalf);
    assert(value.scalar_type() == at::kHalf);
    uint32_t headDim = query.size(3);
    uint32_t headQ = query.size(2);
    uint32_t qLen = query.size(1);
    uint32_t batchSize = query.size(0);
    uint32_t headKv = key.size(2);
    uint32_t kvLen = key.size(1);
    uint32_t hiddenDim = headDim * headQ;
    auto output = at::empty({batchSize, qLen, hiddenDim}, query.options());
    auto normAlpha = at::empty({hiddenDim}, at::device(at::kXPU).dtype(at::kFloat));
    auto normTemp = at::empty({batchSize, kvLen, hiddenDim}, key.options());
    auto dpcpp_queue = dpcppGetCurrentQueue();
    uint8_t* ptrTemp;

    if (flag == 1) {
        ptrTemp = reinterpret_cast<uint8_t*>(normTemp.data_ptr());
    } else {
        ptrTemp = reinterpret_cast<uint8_t*>(value.data_ptr());
    }

    auto cgf_norm = esimd::esimd_norm_fp16(
        reinterpret_cast<uint8_t*>(value.data_ptr()),
        ptrTemp,
        reinterpret_cast<uint8_t*>(normAlpha.data_ptr()),
        kvLen, hiddenDim);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf_norm);

    auto cgf_attn = esimd::esimd_flash_attn_simple(
        reinterpret_cast<uint8_t*>(query.data_ptr()), 
        reinterpret_cast<uint8_t*>(key.data_ptr()),
        ptrTemp,
        reinterpret_cast<uint8_t*>(normAlpha.data_ptr()),
        reinterpret_cast<uint8_t*>(output.data_ptr()),
        qLen, kvLen, headQ, headKv, headDim);
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf_attn);
    return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fmha_esimd.xpu", at::AtenIpexTypeXPU::fmha_esimd, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER("ggemm_preprocess.xpu", at::AtenIpexTypeXPU::ggemm_preprocess);
  IPEX_OP_REGISTER("residual_rms_norm.xpu", at::AtenIpexTypeXPU::residual_rms_norm);
  IPEX_OP_REGISTER("rotary_pos_emb_ds.xpu", at::AtenIpexTypeXPU::rotary_pos_emb_ds);
  IPEX_OP_REGISTER("grouped_topk.xpu", at::AtenIpexTypeXPU::grouped_topk);
  IPEX_OP_REGISTER("grouped_topk_kimi.xpu", at::AtenIpexTypeXPU::grouped_topk_kimi);
  IPEX_OP_REGISTER("moe_scatter_dynamic_quant.xpu", at::AtenIpexTypeXPU::moe_scatter_dynamic_quant);
  IPEX_OP_REGISTER("sage_attn.xpu", at::AtenIpexTypeXPU::sage_attn);
  IPEX_OP_REGISTER("sage_attn_paged.xpu", at::AtenIpexTypeXPU::sage_attn_paged);
  IPEX_OP_REGISTER("scale_dynamic_quant.xpu", at::AtenIpexTypeXPU::scale_dynamic_quant);
  IPEX_OP_REGISTER("head_rms_norm.xpu", at::AtenIpexTypeXPU::head_rms_norm);
  IPEX_OP_REGISTER("store_kv_cache.xpu", at::AtenIpexTypeXPU::store_kv_cache);
  IPEX_OP_REGISTER("add_rms_norm_dynamic_quant.xpu", at::AtenIpexTypeXPU::add_rms_norm_dynamic_quant);
  IPEX_OP_REGISTER("sage_attn_decode_paged.xpu", at::AtenIpexTypeXPU::sage_attn_decode_paged);
  IPEX_OP_REGISTER("dynamic_rotary_embedding.xpu", at::AtenIpexTypeXPU::dynamic_rotary_embedding);
  IPEX_OP_REGISTER("moe_swiglu_dynamic_quant.xpu", at::AtenIpexTypeXPU::moe_swiglu_dynamic_quant);
  IPEX_OP_REGISTER("store_paged_kv_cache.xpu", at::AtenIpexTypeXPU::store_paged_kv_cache);
  IPEX_OP_REGISTER("moe_gather_function.xpu", at::AtenIpexTypeXPU::moe_gather_function);
  IPEX_OP_REGISTER("moe_softmax_topk.xpu", at::AtenIpexTypeXPU::moe_softmax_topk);
  IPEX_OP_REGISTER("specialized_gating_gemm.xpu", at::AtenIpexTypeXPU::specialized_gating_gemm);
  IPEX_OP_REGISTER("flash_attn_simple.xpu", at::AtenIpexTypeXPU::flash_attn_simple);
}
} // namespace
