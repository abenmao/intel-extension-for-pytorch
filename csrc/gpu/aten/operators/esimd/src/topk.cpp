#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <cfloat>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace esimd {

template<const int num_experts>
ESIMD_KERNEL_API cgf_t esimd_moe_softmax_topk_pre(
    float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
    const int num_tokens, const int topk
) {
    assert(topk <= num_experts);

    sycl::range<1> GlobalRange(num_tokens);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    // softmax_output = torch.softmax(gating_output, dim=-1)
    // moe_weights, selected_experts = torch.topk(softmax_output, topk, dim=-1)
    // moe_weights = moe_weights / moe_weights.sum(dim=-1, keepdim=True)

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
         
            const int nid = item.get_global_id(0);

            const float * logits_base = gating_output_ptr + nid * num_experts;
            int * topk_idx_base = selected_experts_ptr + nid * topk;
            float * topk_weight_base = moe_weights_ptr + nid * topk;

            // softmax
            simd<float, num_experts> logits = block_load<float, num_experts>(logits_base);
            simd<float, num_experts> logitsv = exp2(logits - hmax<float, float, num_experts>(logits));
            logitsv = logitsv / sycl::ext::intel::esimd::detail::sum<float, float, num_experts>(logitsv);

            // top_k
            simd<int, num_experts> topk_idxv;
            simd<float, num_experts> topk_weightv = 0;
            
            for (int i = 0; i < topk; ++i) {
                float max_logit = hmax<float, float, num_experts>(logitsv);

                uint32_t index = 0;
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {
                    simd_mask<32> mask = logitsv.template select<32, 1>(n) == max_logit;
                    uint32_t m = pack_mask(mask);
                    if (m != 0) {
                        index = n + fbl(m);
                        logitsv[index] = -10000.0f;
                        break;
                    }
                }

                topk_idxv[i] = index;
                topk_weightv[i] = max_logit;
            }

            float weight_sum = sycl::ext::intel::esimd::detail::sum<float, float, num_experts>(topk_weightv);
            topk_weightv = topk_weightv / weight_sum;

            for (int i = 0; i < topk; ++i) {
                topk_idx_base[i] = topk_idxv[i];
                topk_weight_base[i] = topk_weightv[i];
            }
            
        });
    };
    return kernel_func;
}
template<const int num_experts>

ESIMD_KERNEL_API cgf_t esimd_moe_softmax_topk_top8_pre(
    float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
    const int num_tokens
) {
    sycl::range<1> GlobalRange(num_tokens/2);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
        const int nid = item.get_global_id(0)<<1;
        float * logits_base = gating_output_ptr + nid * num_experts;
        int * topk_idx_base = selected_experts_ptr + nid * 8;
        float * topk_weight_base = moe_weights_ptr + nid * 8;
        simd<float, num_experts> logits = block_load<float, num_experts>(logits_base);
        simd<float, num_experts> logits1 = block_load<float, num_experts>(logits_base+num_experts);

        // softmax

        simd<float, num_experts> logitsv = exp2(logits - hmax<float, float, num_experts>(logits));
        logitsv = logitsv / sycl::ext::intel::esimd::detail::sum<float, float, num_experts>(logitsv);
        simd<float, num_experts> logitsv1 = exp2(logits1 - hmax<float, float, num_experts>(logits1));
        logitsv1 = logitsv1 / sycl::ext::intel::esimd::detail::sum<float, float, num_experts>(logitsv1);

            // top_k
            simd<int, 16> topk_idxv;
            simd<float, 16> topk_weightv;
 
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float max_logit = hmax<float, float, num_experts>(logitsv);
                float max_logit1 = hmax<float, float, num_experts>(logitsv1);
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {

                    simd_mask<32> mask = logitsv.template select<32, 1>(n) == max_logit;
                    uint32_t m = pack_mask(mask);
                    if(m!=0){
                        uint32_t index= n +  fbl(m);
                        topk_idxv[i] = index;
                        topk_weightv[i]=max_logit;
                        logitsv[index] = -10000.0f;
                        break;
                    }
                }
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {

                    simd_mask<32> mask = logitsv1.template select<32, 1>(n) == max_logit1;

                    uint32_t m = pack_mask(mask);
                    if(m!=0){

                        uint32_t index= n +  fbl(m);
                        topk_idxv[i+8] = index;
                        topk_weightv[i+8]=max_logit1;
                        logitsv1[index] = -10000.0f;
                        break;
                    }  
                }
            }


            float weight_sum = sycl::ext::intel::esimd::detail::sum<float, float, 8>(topk_weightv.select<8,1>(0));
            topk_weightv.select<8,1>(0) = topk_weightv.select<8,1>(0) / weight_sum;
            float weight_sum1 = sycl::ext::intel::esimd::detail::sum<float, float, 8>(topk_weightv.select<8,1>(8));
            topk_weightv.select<8,1>(8) = topk_weightv.select<8,1>(8) / weight_sum1;
            __ESIMD_ENS::lsc_block_store<int,16 ,__ESIMD_ENS::lsc_data_size::default_size,__ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>(topk_idx_base,topk_idxv);
            __ESIMD_ENS::lsc_block_store<float,16 ,__ESIMD_ENS::lsc_data_size::default_size,__ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>(topk_weight_base,topk_weightv);

        });
    };
    return kernel_func;
}
template<const int num_experts>
ESIMD_KERNEL_API cgf_t esimd_moe_softmax_topk_post(
    float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
    const int num_tokens, const int topk
) {
    // topk_output, selected_experts = torch.topk(gating_output, topk, dim=-1)
    // softmax_output = torch.softmax(topk_output, dim=-1)
    assert(topk <= num_experts);

    sycl::range<1> GlobalRange(num_tokens);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int nid = item.get_global_id(0);

            const float * logits_base = gating_output_ptr + nid * num_experts;
            int * topk_idx_base = selected_experts_ptr + nid * topk;
            float * topk_weight_base = moe_weights_ptr + nid * topk;

            simd<float, num_experts> logits = block_load<float, num_experts>(logits_base);

            // top_k
            simd<int, 32> topk_idxv;
            simd<float, 32> topk_weightv = -INFINITY;

            for (int i = 0; i < topk; ++i) {
                float max_logit = hmax<float, float, num_experts>(logits);

                uint32_t index = 0;
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {
                    simd_mask<32> mask = logits.template select<32, 1>(n) == max_logit;
                    uint32_t m = pack_mask(mask);
                    if (m != 0) {
                        index = n + fbl(m);
                        logits[index] = -INFINITY;
                        break;
                    }
                }

                topk_idxv[i] = index;
                topk_weightv[i] = max_logit;
            }

            // softmax
            simd<float, 32> logitsv = exp(topk_weightv - hmax<float, float, 32>(topk_weightv));
            logitsv = logitsv / sycl::ext::intel::esimd::detail::sum<float, float, 32>(logitsv);

            for (int i = 0; i < topk; ++i) {
                topk_idx_base[i] = topk_idxv[i];
                topk_weight_base[i] = logitsv[i];
            }
        });
    };
    return kernel_func;
}

template<const int num_experts>
ESIMD_KERNEL_API cgf_t esimd_moe_softmax_top8_post(
    float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
    const int num_tokens
) {
    // topk_output, selected_experts = torch.topk(gating_output, topk, dim=-1)
    // softmax_output = torch.softmax(topk_output, dim=-1)
 

    sycl::range<1> GlobalRange(num_tokens/2);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int nid = item.get_global_id(0)<<1;
            float * logits_base = gating_output_ptr + nid * num_experts;
            int * topk_idx_base = selected_experts_ptr + nid * 8;
            float * topk_weight_base = moe_weights_ptr + nid * 8;

            simd<float, num_experts> logits = block_load<float, num_experts>(logits_base);
            simd<float, num_experts> logits1 = block_load<float, num_experts>(logits_base+num_experts);
            // top_k
            simd<int, 16> topk_idxv;
            simd<float, 16> topk_weightv;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float max_logit = hmax<float, float, num_experts>(logits);
                float max_logit1 = hmax<float, float, num_experts>(logits1);
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {
                    simd_mask<32> mask = logits.template select<32, 1>(n) == max_logit;
                    uint32_t m = pack_mask(mask);
                    if (m != 0) {
                        uint32_t index = n + fbl(m);
                        logits[index] = -INFINITY;
                        topk_idxv[i] = index;
                        topk_weightv[i] = max_logit;
                        break;
                    }
                }
                #pragma unroll
                for (int n = 0; n < num_experts; n += 32) {
                    simd_mask<32> mask = logits1.template select<32, 1>(n) == max_logit1;
                    uint32_t m = pack_mask(mask);
                    if (m != 0) {
                        uint32_t index = n + fbl(m);
                        logits1[index] = -INFINITY;
                        topk_idxv[i+8] = index;
                        topk_weightv[i+8] = max_logit1;
                        break;
                    }
                }

            }
            __ESIMD_ENS::lsc_block_store<int,16 ,__ESIMD_ENS::lsc_data_size::default_size,__ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>(topk_idx_base,topk_idxv);
            // softmax
            simd<float, 16> logitsv;
            logitsv.select<8,1>(0) = exp(topk_weightv.select<8,1>(0) - hmax<float, float, 8>(topk_weightv.select<8,1>(0)));
            logitsv.select<8,1>(8) = exp(topk_weightv.select<8,1>(8) - hmax<float, float, 8>(topk_weightv.select<8,1>(8)));
            logitsv.select<8,1>(0) = logitsv.select<8,1>(0) / sycl::ext::intel::esimd::detail::sum<float, float, 8>(logitsv.select<8,1>(0));
            logitsv.select<8,1>(8) = logitsv.select<8,1>(8) / sycl::ext::intel::esimd::detail::sum<float, float, 8>(logitsv.select<8,1>(8));
            
            __ESIMD_ENS::lsc_block_store<float,16 ,__ESIMD_ENS::lsc_data_size::default_size,__ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>(topk_weight_base,logitsv);
        });
    };
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_moe_softmax_topk(
    float* gating_output_ptr, int* selected_experts_ptr, float* moe_weights_ptr,
    const int num_tokens, const int num_experts, const int topk, const bool pre_softmax
) {
    
    if (pre_softmax) {
        auto func = [&](){
            switch (num_experts) {
                case 256: return esimd_moe_softmax_topk_pre<256>;
                case 128: return esimd_moe_softmax_topk_pre<128>;
                case 64: return esimd_moe_softmax_topk_pre<64>;
                case 32: return esimd_moe_softmax_topk_pre<32>;
            }
        } ();
        auto kernel_func = func(gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
                                num_tokens, topk);
        auto func1 = [&](){
            switch (num_experts) {
                case 256: return esimd_moe_softmax_topk_top8_pre<256>;
                case 128: return esimd_moe_softmax_topk_top8_pre<128>;
                case 64: return esimd_moe_softmax_topk_top8_pre<64>;
                case 32: return esimd_moe_softmax_topk_top8_pre<32>;
            }
        } ();
        auto kernel_func1 = func1(gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
                                num_tokens);
        if(topk==8 && (num_tokens&0x1)==0&& (num_tokens>1023))
            return kernel_func1;
        return kernel_func;
    } else {
        auto func = [&](){
            switch (num_experts) {
                case 256: return esimd_moe_softmax_topk_post<256>;
                case 128: return esimd_moe_softmax_topk_post<128>;
                case 64: return esimd_moe_softmax_topk_post<64>;
                case 32: return esimd_moe_softmax_topk_post<32>;
            }
        } ();
        auto kernel_func = func(gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
                                num_tokens, topk);
        auto func1 = [&](){
            switch (num_experts) {
                case 256: return esimd_moe_softmax_top8_post<256>;
                case 128: return esimd_moe_softmax_top8_post<128>;
                case 64: return esimd_moe_softmax_top8_post<64>;
                case 32: return esimd_moe_softmax_top8_post<32>;
            }
        } ();
        auto kernel_func1 = func1(gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
                                num_tokens);
        if(topk==8 && (num_tokens&0x1)==0 && (num_tokens>1023))
            return kernel_func1;
        return kernel_func;
    }
}

}