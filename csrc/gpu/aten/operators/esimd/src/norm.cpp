#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace esimd {

template<const int head_dim>
ESIMD_KERNEL_API cgf_t esimd_head_rms_norm_kernel(
    fp16* weight_ptr, fp16* input_ptr, fp16* output_ptr,
    const int num_tokens, const int total_head_num, const int head_num, float eps
) {
    sycl::range<2> GlobalRange(num_tokens, head_num);
    sycl::range<2> LocalRange(1, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
            const int token_idx = item.get_global_id(0);
            const int head_idx = item.get_global_id(1);

            fp16* input_base = input_ptr + token_idx * total_head_num * head_dim + head_idx * head_dim;
            fp16* output_base = output_ptr + token_idx * total_head_num * head_dim + head_idx * head_dim;
            fp16* weight_base = weight_ptr + head_idx * head_dim;
            
            simd<fp16, head_dim> input = block_load<fp16, head_dim>(input_base);
            simd<fp16, head_dim> output;
            simd<fp16, head_dim> weight = block_load<fp16, head_dim>(weight_base);

            simd<float, head_dim> accv = 0;
            simd<float, head_dim> xv_f32 = input;
            accv = xv_f32 * xv_f32;
            float acc = sycl::ext::intel::esimd::detail::sum<float, float, head_dim>(accv) / head_dim;
            float scale = rsqrt(acc + eps);

            simd<float, head_dim> yv_f32 = weight;
            simd<fp16, head_dim> result = xv_f32 * scale * yv_f32;
            block_store<fp16, head_dim>(output_base, result);
        });
    };
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_head_rms_norm(
    fp16* weight_ptr, fp16* input_ptr, fp16* output_ptr,
    const int num_tokens, const int total_head_num, const int head_num, const int head_dim, float eps
) {
    auto func = [&](){
        switch (head_dim) {
            case 128: return esimd_head_rms_norm_kernel<128>;
            case 64: return esimd_head_rms_norm_kernel<64>;
            case 32: return esimd_head_rms_norm_kernel<32>;
        }
    } ();
    auto kernel_func = func(weight_ptr, input_ptr, output_ptr,
                            num_tokens, total_head_num, head_num, eps);
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_residual_kernel_rms_norm(
    fp16* weight_ptr, fp16* residual_ptr, fp16* input_ptr,
    fp16* after_res_ptr, fp16* after_norm_ptr,
    const int hidden_size, const int num_tokens, const bool has_residual, float eps
) {
    constexpr int block_size = 256;
    const int blocks = (hidden_size + block_size - 1) / block_size;
    assert(hidden_size % block_size == 0);
    assert(hidden_size <= 8192);
    constexpr int max_block = 32;  // 8192 / 256
    const int quant_offset = max_block * sizeof(float);
    sycl::range<2> GlobalRange(num_tokens, blocks);
    sycl::range<2> LocalRange(1, blocks);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);
    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
            slm_init(max_block * 2 * sizeof(float));

            const int token_idx = item.get_global_id(0);
            const int block_idx = item.get_local_id(1);

            fp16* input_base = input_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* after_res_base = after_res_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* after_norm_base = after_norm_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* weight_base = weight_ptr + block_idx * block_size;

            // residual stage
            simd<fp16, block_size> input = block_load<fp16, block_size>(input_base);
            if (has_residual) {
                fp16* residual_base = residual_ptr + token_idx * hidden_size + block_idx * block_size;
                simd<fp16, block_size> res = block_load<fp16, block_size>(residual_base);
                input += res;
            }
            block_store<fp16, block_size>(after_res_base, input);

            // rms norm stage
            simd<float, block_size> xv_f32 = input;
            simd<float, block_size> accv = xv_f32 * xv_f32;
            float acc = sycl::ext::intel::esimd::detail::sum<float, float, block_size>(accv) / hidden_size;

            slm_block_store<float, 1>(block_idx * sizeof(float), acc);

            barrier();

            simd<float, max_block> slm_sum = slm_block_load<float, max_block>(0);
            float all_sum = sycl::ext::intel::esimd::detail::sum<float, float, max_block>(slm_sum);
            float scale = rsqrt(all_sum + eps);

            simd<fp16, block_size> weight = block_load<fp16, block_size>(weight_base);
            simd<float, block_size> yv_f32 = weight;
            simd<fp16, block_size> result = xv_f32 * scale * yv_f32;
            block_store<fp16, block_size>(after_norm_base, result);
        });
    };
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_add_rms_norm_dynamic_quant(
    fp16* weight_ptr, fp16* input_ptr, fp16* residual_ptr, float* smooth_scale_ptr,
    fp16* after_res_ptr, fp16* after_norm_ptr,  int8_t* quant_tokens_ptr, float* per_token_scale_ptr,
    const int num_tokens, const int hidden_size, const bool has_residual, float eps
) {
    constexpr int block_size = 256;
    const int blocks = (hidden_size + block_size - 1) / block_size;
    assert(hidden_size % block_size == 0);
    assert(hidden_size <= 8192);
    constexpr int max_block = 32;  // 8192 / 256
    const int quant_offset = max_block * sizeof(float);
    sycl::range<2> GlobalRange(num_tokens, blocks);
    sycl::range<2> LocalRange(1, blocks);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);
    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
            slm_init(max_block * 2 * sizeof(float));

            const int token_idx = item.get_global_id(0);
            const int block_idx = item.get_local_id(1);

            fp16* input_base = input_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* after_res_base = after_res_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* after_norm_base = after_norm_ptr + token_idx * hidden_size + block_idx * block_size;
            fp16* weight_base = weight_ptr + block_idx * block_size;

            // residual stage
            simd<fp16, block_size> input = block_load<fp16, block_size>(input_base);
            if (has_residual) {
                fp16* residual_base = residual_ptr + token_idx * hidden_size + block_idx * block_size;
                simd<fp16, block_size> res = block_load<fp16, block_size>(residual_base);
                input += res;
            }
            block_store<fp16, block_size>(after_res_base, input);

            // rms norm stage
            simd<float, block_size> xv_f32 = input;
            simd<float, block_size> accv = xv_f32 * xv_f32;
            float acc = sycl::ext::intel::esimd::detail::sum<float, float, block_size>(accv) / hidden_size;

            slm_block_store<float, 1>(block_idx * sizeof(float), acc);

            barrier();

            simd<float, max_block> slm_sum = slm_block_load<float, max_block>(0);
            float all_sum = sycl::ext::intel::esimd::detail::sum<float, float, max_block>(slm_sum);
            float scale = rsqrt(all_sum + eps);

            simd<fp16, block_size> weight = block_load<fp16, block_size>(weight_base);
            simd<float, block_size> yv_f32 = weight;
            simd<fp16, block_size> result = xv_f32 * scale * yv_f32;
            block_store<fp16, block_size>(after_norm_base, result);

            // quant stage
            int8_t* quant_tokens_base = quant_tokens_ptr + token_idx * hidden_size + block_idx * block_size;
            float* per_token_scale_base = per_token_scale_ptr + token_idx;
            float* smooth_scale_base = smooth_scale_ptr + block_idx * block_size;
            simd<float, block_size> smooth_scale = block_load<float, block_size>(smooth_scale_base);
            simd<float, block_size> smooth_input = smooth_scale * result;
            simd<float, block_size> smooth_input_abs = __ESIMD_NS::abs(smooth_input);
            float maxvalue = hmax<float, float, block_size>(smooth_input_abs);
            slm_block_store<float, 1>(quant_offset + block_idx * sizeof(float), maxvalue);

            barrier();

            simd<float, max_block> all_maxvalue = slm_block_load<float, max_block>(quant_offset);
            float global_maxvalue = hmax<float, float, max_block>(all_maxvalue) / 127.0;
            if (block_idx == 0) {
                block_store<float, 1>(per_token_scale_base, global_maxvalue);
            }
            simd<float, block_size> quant = __ESIMD_NS::rnde<float>(smooth_input / global_maxvalue);
            simd<int8_t, block_size> quant_tokens = quant;
            block_store<int8_t, block_size>(quant_tokens_base, quant_tokens);
        });
    };
    return kernel_func;
}

}