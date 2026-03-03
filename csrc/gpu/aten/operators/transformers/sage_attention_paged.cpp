/*
 * SageAttention with Paged KV Cache for XPU
 *
 * Implements torch.ops.torch_ipex.sage_attn_paged used by ByteMLPerf
 * sage_attention_page benchmark.
 *
 * Ported from frameworks.ai.client-ai.ipex-gpu-client reference implementation.
 * Uses ESIMD kernels for native paged KV, GQA, int8 dequant, and causal masking.
 */

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

using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor sage_attn_paged(
    const at::Tensor& qState,   // [num_tokens_q, query_heads, head_dim]  int8
    const at::Tensor& kState,   // [num_blocks, block_size, key_heads, head_dim] int8
    const at::Tensor& vState,   // [num_blocks, block_size, key_heads, head_dim] int8
    const at::Tensor& qScale,   // [q_len, q_head, 1]  float32
    const at::Tensor& kScale,   // [kv_len, kv_head, 1] float32
    const at::Tensor& vScale,   // [1, kv_head, head_dim] float32
    const at::Tensor& cuLengthQ,    // [batch + 1] int32
    const at::Tensor& cuLengthKv,   // [batch + 1] int32
    const at::Tensor& blockTable,   // [batch, num_max_seq_block] int32
    at::Tensor& output,             // [num_tokens_q, query_heads, head_dim] float16
    int64_t longestBatch) {

  RECORD_FUNCTION("sage_attn_paged", {});

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

at::Tensor sage_attn_decode_paged(
    at::Tensor& qState,       // [batch, num_tokens_q, query_heads, head_dim] int8
    at::Tensor& kState,       // [num_blocks, block_size, key_heads, head_dim] int8
    at::Tensor& vState,       // [num_blocks, block_size, key_heads, head_dim] int8
    at::Tensor& qScale,       // [batch, q_len, q_head, 1] float32
    at::Tensor& kScale,       // [batch, kv_len, kv_head, 1] float32
    at::Tensor& vScale,       // [1, kv_head, head_dim] float32
    at::Tensor& cuLengthQ,    // [batch + 1] int32
    at::Tensor& cuLengthKv,   // [batch + 1] int32
    at::Tensor& blockTable,   // [batch, num_max_seq_block] int32
    at::Tensor& output,       // [batch, chunk_num, num_tokens_q, query_heads, head_dim] float32
    at::Tensor& output_lse,   // [batch, chunk_num, num_tokens_q, query_heads, 1] float32
    at::Tensor& output_max,   // [batch, chunk_num, num_tokens_q, query_heads, 1] float32
    at::Tensor& output_final, // [batch, num_tokens_q, query_heads, head_dim] float16
    int64_t batch_num,
    int64_t chunk_size) {

  RECORD_FUNCTION("sage_attn_decode_paged", {});

  auto dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t headQ = qState.size(2);    // qstate: [batch, num_tokens_q, query_heads, head_dim]
  uint32_t headKv = kState.size(2);
  uint32_t head_dim = qState.size(3);
  uint32_t max_block_num_per_batch = blockTable.size(1);
  uint32_t block_size = 512;
  uint32_t chunk_count = output_max.size(1);

  uint32_t q_head_num_per_kv_head = headQ / headKv;
  uint32_t acc_fp32 = 0;
  c10::ScalarType acc_scalar_type = output.scalar_type();
  if (acc_scalar_type == at::kFloat)
    acc_fp32 = 1;

  if (head_dim == 64 || (q_head_num_per_kv_head == 16 || q_head_num_per_kv_head == 12
      || q_head_num_per_kv_head == 10 || q_head_num_per_kv_head == 8))
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
        acc_fp32);
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

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "sage_attn_paged",
      at::AtenIpexTypeXPU::sage_attn_paged);
  IPEX_OP_REGISTER(
      "sage_attn_decode_paged",
      at::AtenIpexTypeXPU::sage_attn_decode_paged);
}

} // namespace
