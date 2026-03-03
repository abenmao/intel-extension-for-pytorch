#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace esimd {
    ESIMD_KERNEL_API cgf_t esimd_moe_swiglu_dynamic_quant (
        fp16* scatter_tokens,
        float* smooth_scale,
        int32_t* experts_token_count,
        int32_t* experts_token_start,
        int8_t* quant_tokens,
        float* per_token_scale,
        const int total_experts_num,
        const int max_token_num,
        const int hidden_size
    ) {
        constexpr int BS = 256;
        constexpr int MAX_BN = 32;    // 8192 / 256
        assert(hidden_size % BS == 0);
        assert(hidden_size <= MAX_BN * BS);

        int nb = hidden_size / BS;

        sycl::range<3> GlobalRange(total_experts_num, max_token_num, nb);
        sycl::range<3> LocalRange(1, 1, nb);
        sycl::nd_range<3> Range(GlobalRange, LocalRange);

        cgf_t kernel_func = [=](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init(MAX_BN * sizeof(float));

                const int expert_idx = item.get_global_id(0);
                const int token_idx = item.get_global_id(1);
                const int bid = item.get_local_id(2);

                const int token_cnt = experts_token_count[expert_idx];

                if (token_idx >= token_cnt) {
                    return;
                }

                const int token_start = experts_token_start[expert_idx];

                fp16* scatter_token_base = scatter_tokens + (token_start + token_idx) * 2 * hidden_size;
                int8_t* output_base = quant_tokens + (token_start + token_idx) * hidden_size + bid * BS;

                simd<fp16, BS> x1 = block_load<fp16, BS>(scatter_token_base + bid * BS);
                simd<fp16, BS> x2 = block_load<fp16, BS>(scatter_token_base + hidden_size + bid * BS);
                simd<float, BS> scale = block_load<float, BS>(smooth_scale + expert_idx * hidden_size + bid * BS);

                simd<fp16, BS> x1_silu = x1 / (1 + exp(-x1));
                simd<fp16, BS> swiglu_tokens = x1_silu * x2;

                simd<float, BS> scaled_swiglu_tokens = swiglu_tokens * scale;
                simd<float, BS> scaled_swiglu_tokens_abs = abs(scaled_swiglu_tokens);
                float max_value = hmax<float, float, BS>(scaled_swiglu_tokens_abs);
                slm_block_store<float, 1>(bid * sizeof(float), max_value);

                barrier();

                simd<float, MAX_BN> max_value_full = slm_block_load<float, MAX_BN>(0);
                float max_value_final = hmax<float, float, MAX_BN>(max_value_full);
                float this_token_scale = max_value_final / 127.0;

                simd<float, BS> scaled_swiglu_tokens_quant = rnde<float>(scaled_swiglu_tokens / this_token_scale);
                simd<int8_t, BS> scaled_swiglu_tokens_quant_out = scaled_swiglu_tokens_quant;

                block_store<int8_t, BS>(output_base, scaled_swiglu_tokens_quant_out);

                if (bid == 0) {
                    block_store<float, 1>(per_token_scale + token_start + token_idx, this_token_scale);
                }
            });
        };
        return kernel_func;
    }

ESIMD_KERNEL_API cgf_t esimd_moe_gather_init(
  int32_t* scatter_tokens_offset_ptr, int32_t* tokens_index_ptr, int32_t* tokens_count_ptr,
  const int num_tokens, const int real_scatter_tokens, const int topk
) {
    sycl::range<1> GlobalRange(real_scatter_tokens);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(Range, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
        const int scatter_idx = item.get_global_id(0);
        const int token_index = scatter_tokens_offset_ptr[scatter_idx];
        // int old_count = atomic_update<atomic_op::add, int, 1>(tokens_count_ptr, token_index, 1);
        // 1. Define the value to add as a SIMD vector of size 1.
        __ESIMD_NS::simd<int32_t, 1> value_to_add = 1;

        // 2. Define a mask. Since we want the operation to happen, the mask is 1.
        __ESIMD_NS::simd_mask<1> mask = 1;

        __ESIMD_NS::simd<int32_t, 1> old_count = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
            tokens_count_ptr,   // The USM pointer
            token_index * sizeof(int32_t),  // offset
            value_to_add,  // The value to add
            mask           // The execution mask
        );

        tokens_index_ptr[token_index * topk + old_count] = scatter_idx;
        });
    };
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_moe_gather_kernel(
  fp16* scatter_tokens_ptr, int32_t* tokens_index_ptr, fp16* convergent_tokens_ptr,
  const int num_tokens, const int hidden_size, const int real_scatter_tokens, const int topk
) {
    constexpr int BS = 256;
    assert(hidden_size % BS == 0);
    sycl::range<2> GlobalRange(num_tokens, hidden_size / BS);
    sycl::range<2> LocalRange(1, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
        const int token_idx = item.get_global_id(0);
        const int batch_idx = item.get_global_id(1);
        simd<fp16, BS> output = block_load<fp16, BS>(convergent_tokens_ptr + token_idx * hidden_size + batch_idx * BS);

        for (int i = 0; i < topk; i ++) {
            const int scatter_index = tokens_index_ptr[token_idx * topk + i];
            simd<fp16, BS> tmp = block_load<fp16, BS>(scatter_tokens_ptr + scatter_index * hidden_size + batch_idx * BS);
            output += tmp;
        }

        block_store<fp16, BS>(convergent_tokens_ptr + token_idx * hidden_size + batch_idx * BS, output);
        });
    };
    return kernel_func;
}

}