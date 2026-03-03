#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

namespace esimd {
    template<const int HD>
    ESIMD_KERNEL_API cgf_t esimd_store_kv_cache_kernel(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* cache_slot_ids,
        fp16* k_cache,
        fp16* v_cache,
        float* k_scale,
        float* v_scale,
        const int num_tokens,
        const int total_head_num,
        const int q_head_num,
        const int kv_head_num,
        const int batch_size,
        const int64_t max_q_len,
        const int32_t max_kv_len
    ) {
        sycl::range<3> GlobalRange(batch_size, max_q_len, kv_head_num);
        sycl::range<3> LocalRange(1, 1, 1);
        sycl::nd_range<3> Range(GlobalRange, LocalRange);

        cgf_t kernel_func = [=](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int batch_idx = item.get_global_id(0);
                const int q_idx = item.get_global_id(1);
                const int kv_head_idx = item.get_global_id(2);

                const int q_len = q_lens[batch_idx];
                const int token_start = accum_q_lens[batch_idx];
                const int cur_cache_len = cache_lens[batch_idx];
                const int cur_slot_id = cache_slot_ids[batch_idx];

                if (q_idx >= q_len) {
                    return;
                }

                fp16* packed_qkv_base = packed_qkv + (token_start + q_idx) * total_head_num * HD;
                fp16* k_output_base = k_cache + cur_slot_id * kv_head_num * max_kv_len * HD + kv_head_idx * max_kv_len * HD + (cur_cache_len + q_idx) * HD;
                fp16* v_output_base = v_cache + cur_slot_id * kv_head_num * max_kv_len * HD + kv_head_idx * max_kv_len * HD + (cur_cache_len + q_idx) * HD;
                
                simd<fp16, HD> k = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_idx) * HD);
                simd<fp16, HD> v = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_num + kv_head_idx) * HD);

                if (k_scale) {
                    simd<float, HD> k_head_scale = block_load<float, HD>(k_scale + kv_head_idx * HD);
                    simd<float, HD> v_head_scale = block_load<float, HD>(v_scale + kv_head_idx * HD);

                    simd<float, HD> scaled_k = k * k_head_scale;
                    simd<float, HD> scaled_v = v * v_head_scale;

                    block_store<fp16, HD>(k_output_base, scaled_k);
                    block_store<fp16, HD>(v_output_base, scaled_v);
                } else {
                    block_store<fp16, HD>(k_output_base, k);
                    block_store<fp16, HD>(v_output_base, v);
                }
            });
        };
        return kernel_func;
    }


    template<const int HD>
    ESIMD_KERNEL_API cgf_t esimd_store_kv_cache_int8_kernel(
        fp16* packed_qkv,
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
        const int64_t max_q_len,
        const int32_t max_kv_len
    ) {
        sycl::range<3> GlobalRange(batch_size, max_q_len, kv_head_num);
        sycl::range<3> LocalRange(1, 1, 1);
        sycl::nd_range<3> Range(GlobalRange, LocalRange);

        cgf_t kernel_func = [=](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int batch_idx = item.get_global_id(0);
                const int q_idx = item.get_global_id(1);
                const int kv_head_idx = item.get_global_id(2);

                const int q_len = q_lens[batch_idx];
                const int token_start = accum_q_lens[batch_idx];
                const int cur_cache_len = cache_lens[batch_idx];
                const int cur_slot_id = cache_slot_ids[batch_idx];

                if (q_idx >= q_len) {
                    return;
                }

                fp16* packed_qkv_base = packed_qkv + (token_start + q_idx) * total_head_num * HD;
                int8_t* k_output_base = k_cache + cur_slot_id * kv_head_num * max_kv_len * HD + kv_head_idx * max_kv_len * HD + (cur_cache_len + q_idx) * HD;
                int8_t* v_output_base = v_cache + cur_slot_id * kv_head_num * max_kv_len * HD + kv_head_idx * max_kv_len * HD + (cur_cache_len + q_idx) * HD;
                
                simd<fp16, HD> k = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_idx) * HD);
                simd<fp16, HD> v = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_num + kv_head_idx) * HD);

                if (k_scale) {
                    simd<float, HD> k_head_scale = block_load<float, HD>(k_scale + kv_head_idx * HD);
                    simd<float, HD> v_head_scale = block_load<float, HD>(v_scale + kv_head_idx * HD);

                    simd<float, HD> scaled_k = k * k_head_scale;
                    simd<float, HD> scaled_v = v * v_head_scale;

                    simd<float, HD> round_k = min<float>(rnde<float>(scaled_k), 127.0);
                    simd<float, HD> round_v = min<float>(rnde<float>(scaled_v), 127.0);

                    simd<int8_t, HD> int8_k = round_k;
                    simd<int8_t, HD> int8_v = round_v;

                    block_store<int8_t, HD>(k_output_base, int8_k);
                    block_store<int8_t, HD>(v_output_base, int8_v);
                } else {
                    simd<float, HD> k_fp32 = k;
                    simd<float, HD> v_fp32 = v;

                    simd<float, HD> round_k = min<float>(rnde<float>(k_fp32), 127.0);
                    simd<float, HD> round_v = min<float>(rnde<float>(v_fp32), 127.0);

                    simd<int8_t, HD> int8_k = round_k;
                    simd<int8_t, HD> int8_v = round_v;

                    block_store<int8_t, HD>(k_output_base, int8_k);
                    block_store<int8_t, HD>(v_output_base, int8_v);
                }
            });
        };
        return kernel_func;
    }


    template<const int HD>
    ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_kernel(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* block_tables,
        fp16* k_cache,
        fp16* v_cache,
        float* k_scale,
        float* v_scale,
        const int num_tokens,
        const int total_head_num,
        const int q_head_num,
        const int kv_head_num,
        const int batch_size,
        const int block_size,
        const int max_block_num_per_seq,
        const int64_t max_q_len
    ) {
        sycl::range<3> GlobalRange(batch_size, max_q_len, kv_head_num);
        sycl::range<3> LocalRange(1, 1, 1);
        sycl::nd_range<3> Range(GlobalRange, LocalRange);

        cgf_t kernel_func = [=](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int batch_idx = item.get_global_id(0);
                const int q_idx = item.get_global_id(1);
                const int kv_head_idx = item.get_global_id(2);

                const int q_len = q_lens[batch_idx];
                const int token_start = accum_q_lens[batch_idx];
                const int cur_cache_len = cache_lens[batch_idx];

                if (q_idx >= q_len) {
                    return;
                }

                const int seq_block_idx = (cur_cache_len + q_idx) / block_size;
                const int seq_block_pos = (cur_cache_len + q_idx) % block_size;

                const int32_t global_block_idx = block_tables[batch_idx * max_block_num_per_seq + seq_block_idx];

                fp16* packed_qkv_base = packed_qkv + (token_start + q_idx) * total_head_num * HD;
                simd<fp16, HD> k = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_idx) * HD);
                simd<fp16, HD> v = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_num + kv_head_idx) * HD);

                fp16* k_output_base = k_cache + global_block_idx * block_size * kv_head_num * HD + seq_block_pos * kv_head_num * HD + kv_head_idx * HD;
                fp16* v_output_base = v_cache + global_block_idx * block_size * kv_head_num * HD + seq_block_pos * kv_head_num * HD + kv_head_idx * HD;

                if (k_scale) {
                    float k_scale_value = k_scale[global_block_idx * block_size * kv_head_num + seq_block_pos * kv_head_num + kv_head_idx];
                    simd<float, HD> v_head_scale = block_load<float, HD>(v_scale + kv_head_idx * HD);

                    simd<fp16, HD> scaled_k = k * k_scale_value;
                    simd<fp16, HD> scaled_v = v * v_head_scale;

                    block_store<fp16, HD>(k_output_base, scaled_k);
                    block_store<fp16, HD>(v_output_base, scaled_v);
                } else {
                    block_store<fp16, HD>(k_output_base, k);
                    block_store<fp16, HD>(v_output_base, v);
                }
            });
        };
        return kernel_func;
    }


    template<const int HD>
    ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_int8_kernel(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* block_tables,
        int8_t* k_cache,
        int8_t* v_cache,
        float* k_scale,
        float* v_scale,
        const int num_tokens,
        const int total_head_num,
        const int q_head_num,
        const int kv_head_num,
        const int batch_size,
        const int block_size,
        const int max_block_num_per_seq,
        const int64_t max_q_len
    ) {
        sycl::range<3> GlobalRange(batch_size, max_q_len, kv_head_num);
        sycl::range<3> LocalRange(1, 1, 1);
        sycl::nd_range<3> Range(GlobalRange, LocalRange);

        cgf_t kernel_func = [=](sycl::handler& cgh) {
            cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int batch_idx = item.get_global_id(0);
                const int q_idx = item.get_global_id(1);
                const int kv_head_idx = item.get_global_id(2);

                const int q_len = q_lens[batch_idx];
                const int token_start = accum_q_lens[batch_idx];
                const int cur_cache_len = cache_lens[batch_idx];

                if (q_idx >= q_len) {
                    return;
                }

                const int seq_block_idx = (cur_cache_len + q_idx) / block_size;
                const int seq_block_pos = (cur_cache_len + q_idx) % block_size;

                const int32_t global_block_idx = block_tables[batch_idx * max_block_num_per_seq + seq_block_idx];

                fp16* packed_qkv_base = packed_qkv + (token_start + q_idx) * total_head_num * HD;
                simd<fp16, HD> k = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_idx) * HD);
                simd<fp16, HD> v = block_load<fp16, HD>(packed_qkv_base + (q_head_num + kv_head_num + kv_head_idx) * HD);

                int8_t* k_output_base = k_cache + global_block_idx * block_size * kv_head_num * HD + seq_block_pos * kv_head_num * HD + kv_head_idx * HD;
                int8_t* v_output_base = v_cache + global_block_idx * block_size * kv_head_num * HD + seq_block_pos * kv_head_num * HD + kv_head_idx * HD;

                if (k_scale) {
                    float k_scale_value = k_scale[global_block_idx * block_size * kv_head_num + seq_block_pos * kv_head_num + kv_head_idx];
                    simd<float, HD> v_head_scale = block_load<float, HD>(v_scale + kv_head_idx * HD);

                    simd<float, HD> scaled_k = k * k_scale_value;
                    simd<float, HD> scaled_v = v * v_head_scale;

                    simd<float, HD> round_k = min<float>(rnde<float>(scaled_k), 127.0);
                    simd<float, HD> round_v = min<float>(rnde<float>(scaled_v), 127.0);

                    simd<int8_t, HD> int8_k = round_k;
                    simd<int8_t, HD> int8_v = round_v;

                    block_store<int8_t, HD>(k_output_base, int8_k);
                    block_store<int8_t, HD>(v_output_base, int8_v);
                } else {
                    simd<float, HD> scaled_k = k;
                    simd<float, HD> scaled_v = v;

                    simd<float, HD> round_k = min<float>(rnde<float>(scaled_k), 127.0);
                    simd<float, HD> round_v = min<float>(rnde<float>(scaled_v), 127.0);

                    simd<int8_t, HD> int8_k = round_k;
                    simd<int8_t, HD> int8_v = round_v;

                    block_store<int8_t, HD>(k_output_base, int8_k);
                    block_store<int8_t, HD>(v_output_base, int8_v);
                }
            });
        };
        return kernel_func;
    }


    ESIMD_KERNEL_API cgf_t esimd_store_kv_cache(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* cache_slot_ids,
        fp16* k_cache,
        fp16* v_cache,
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
    ) {
        auto func = [&](){
            switch (head_dim) {
                case 128: return esimd_store_kv_cache_kernel<128>;
                case 64: return esimd_store_kv_cache_kernel<64>;
                case 32: return esimd_store_kv_cache_kernel<32>;
            }
        } ();
        auto kernel_func = 
            func(packed_qkv, q_lens, accum_q_lens,
                cache_lens, cache_slot_ids, k_cache, v_cache,
                k_scale, v_scale, num_tokens, total_head_num,
                q_head_num, kv_head_num, batch_size, max_q_len, max_kv_len);
        return kernel_func;
    }

    ESIMD_KERNEL_API cgf_t esimd_store_kv_cache_int8(
        fp16* packed_qkv,
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
    ) {
        auto func = [&](){
            switch (head_dim) {
                case 128: return esimd_store_kv_cache_int8_kernel<128>;
                case 64: return esimd_store_kv_cache_int8_kernel<64>;
                case 32: return esimd_store_kv_cache_int8_kernel<32>;
            }
        } ();
        auto kernel_func = 
            func(packed_qkv, q_lens, accum_q_lens,
                cache_lens, cache_slot_ids, k_cache, v_cache,
                k_scale, v_scale, num_tokens, total_head_num,
                q_head_num, kv_head_num, batch_size, max_q_len, max_kv_len);
        return kernel_func;
    }


    ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_int8(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* block_tables,
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
        const int block_size,
        const int max_block_num_per_seq,
        const int64_t max_q_len
    ) {
        auto func = [&](){
            switch (head_dim) {
                case 128: return esimd_store_paged_kv_cache_int8_kernel<128>;
                case 64: return esimd_store_paged_kv_cache_int8_kernel<64>;
                case 32: return esimd_store_paged_kv_cache_int8_kernel<32>;
            }
        } ();
        auto kernel_func = 
            func(packed_qkv, q_lens, accum_q_lens,
                cache_lens, block_tables, k_cache, v_cache,
                k_scale, v_scale, num_tokens, total_head_num,
                q_head_num, kv_head_num, batch_size, block_size,
                max_block_num_per_seq, max_q_len);
        return kernel_func;
    }


    ESIMD_KERNEL_API cgf_t esimd_store_paged_kv_cache_fp16(
        fp16* packed_qkv,
        int32_t* q_lens,
        int32_t* accum_q_lens,
        int32_t* cache_lens,
        int32_t* block_tables,
        fp16* k_cache,
        fp16* v_cache,
        float* k_scale,
        float* v_scale,
        const int num_tokens,
        const int total_head_num,
        const int q_head_num,
        const int kv_head_num,
        const int batch_size,
        const int head_dim,
        const int block_size,
        const int max_block_num_per_seq,
        const int64_t max_q_len
    ) {
        auto func = [&](){
            switch (head_dim) {
                case 128: return esimd_store_paged_kv_cache_kernel<128>;
                case 64: return esimd_store_paged_kv_cache_kernel<64>;
                case 32: return esimd_store_paged_kv_cache_kernel<32>;
            }
        } ();
        auto kernel_func = 
            func(packed_qkv, q_lens, accum_q_lens,
                cache_lens, block_tables, k_cache, v_cache,
                k_scale, v_scale, num_tokens, total_head_num,
                q_head_num, kv_head_num, batch_size, block_size,
                max_block_num_per_seq, max_q_len);
        return kernel_func;
    }
}