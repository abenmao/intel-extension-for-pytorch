#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace esimd {

template<const int half_rope_dim>
ESIMD_KERNEL_API cgf_t esimd_dynamic_rotary_embedding_kernel(
    fp16* packed_qkv_ptr, int32_t* q_lens_ptr, int32_t* accum_q_lens_ptr,int32_t* cache_lens_ptr,
    fp16*cos_ptr, fp16* sin_ptr, fp16* y_ptr,
    const int batch_size, const int head_dim, const int total_head_num, const int head_num,
    const int rope_offset, const int rope_dim, const int q_len
) {
    const int hidden_dize = total_head_num * head_dim;
    constexpr int chunk = 128;

    sycl::range<3> GlobalRange(batch_size, (q_len + chunk - 1) / chunk, head_num);
    sycl::range<3> LocalRange(1, 1, 1);
    sycl::nd_range<3> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
            const int batch_idx = item.get_global_id(0);
            const int q_idx = item.get_global_id(1);
            const int head_idx = item.get_global_id(2);

            const int q_len = q_lens_ptr[batch_idx];
            const int accum_q_len = accum_q_lens_ptr[batch_idx];
            const int cache_len = cache_lens_ptr[batch_idx];

            fp16* input_base = packed_qkv_ptr + accum_q_len * hidden_dize + head_idx * head_dim + rope_offset;
            fp16* output_base = y_ptr + accum_q_len * hidden_dize + head_idx * head_dim + rope_offset;
            fp16* cos_base = cos_ptr + cache_len * half_rope_dim;
            fp16* sin_base = sin_ptr + cache_len * half_rope_dim;

            const int q_start = q_idx * chunk;
            const int q_end = min((q_idx + 1) * chunk, q_len);

            for (int i = q_start; i < q_end; i++) {
                simd<fp16, half_rope_dim> x1 = block_load<fp16, half_rope_dim>(input_base + i * hidden_dize);
                simd<fp16, half_rope_dim> x2 = block_load<fp16, half_rope_dim>(input_base + i * hidden_dize + half_rope_dim);
                simd<fp16, half_rope_dim> cos = block_load<fp16, half_rope_dim>(cos_base + i * half_rope_dim);
                simd<fp16, half_rope_dim> sin = block_load<fp16, half_rope_dim>(sin_base + i * half_rope_dim);
                simd<fp16, half_rope_dim> x1_rot = x1 * cos - x2 * sin;
                simd<fp16, half_rope_dim> x2_rot = x1 * sin + x2 * cos;
                block_store<fp16, half_rope_dim>(output_base + i * hidden_dize, x1_rot);
                block_store<fp16, half_rope_dim>(output_base + i * hidden_dize + half_rope_dim, x2_rot);
            }
        });
    };
    return kernel_func;
}

ESIMD_KERNEL_API cgf_t esimd_dynamic_rotary_embedding(
    fp16* packed_qkv_ptr, int32_t* q_lens_ptr, int32_t* accum_q_lens_ptr,int32_t* cache_lens_ptr,
    fp16* cos_ptr, fp16* sin_ptr, fp16* y_ptr,
    const int batch_size, const int head_dim, const int total_head_num, const int head_num,
    const int rope_offset, const int rope_dim, const int max_q
) {
    auto func = [&](){
        switch (rope_dim) {
            case 128: return esimd_dynamic_rotary_embedding_kernel<64>;
            case 112: return esimd_dynamic_rotary_embedding_kernel<56>;
            case 96: return esimd_dynamic_rotary_embedding_kernel<48>;
            case 80: return esimd_dynamic_rotary_embedding_kernel<40>;
            case 64: return esimd_dynamic_rotary_embedding_kernel<32>;
            case 48: return esimd_dynamic_rotary_embedding_kernel<24>;
            case 32: return esimd_dynamic_rotary_embedding_kernel<16>;
            case 16: return esimd_dynamic_rotary_embedding_kernel<8>;
        }
    } ();
    auto kernel_func = func(packed_qkv_ptr, q_lens_ptr, accum_q_lens_ptr, cache_lens_ptr,
                            cos_ptr, sin_ptr, y_ptr,
                            batch_size, head_dim, total_head_num, head_num,
                            rope_offset, rope_dim, max_q);
    return kernel_func;
}

}