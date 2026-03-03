#include "fmha.hpp"

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
    bool is_head_first) {
  // microsoft phi3 model
  if (num_heads_q == 32 && num_heads_k == 32 && head_size == 96) {
    if (is_head_first)
      return launch_phi_fused_mha<true, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
    else
      return launch_phi_fused_mha<false, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
  }
  // microsoft phi3-small model
  if (num_heads_q == 32 && num_heads_k == 8 && head_size == 128) {
    if (is_head_first)
      return launch_phi3small_fused_mha<true, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
    else
      return launch_phi3small_fused_mha<false, sycl::half>(
          query, key, value, output, mask, num_batches, qo_len, kv_len);
  } else {
    std::cout
        << "----------Error in mha: please add policy for currently running "
           "model-----------"
        << std::endl;
    // TODO(zw): exception handle heres
    return {};
  }
}

ESIMD_KERNEL_API cgf_t launch_ggemm_preprocess(
    int32_t* input,
    int32_t* output,
    int32_t n_expert) {
      
  int n_threads = (n_expert + 63) / 64;
  int rest = n_threads * 64 - n_expert;
  int last_thread_rest = n_expert - (n_threads-1) * 64;
  
  // printf("n_threads: %d\n", n_threads);
  // printf("rest: %d\n", rest);
  // printf("last_thread_rest: %d\n", last_thread_rest);

  sycl::range<1> GlobalRange(1 * n_threads);
  sycl::range<1> LocalRange(n_threads);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  if (rest == 0)
  {
    cgf_t kernel_func = [=](sycl::handler& cgh) {
      cgh.parallel_for(
        Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

          int h = ndi.get_group(0);
          int hh = ndi.get_local_id(0);

          __ESIMD_NS::simd<int32_t, 64> inputdata = __ESIMD_NS::block_load<int32_t, 64>(input + hh * 64);
          __ESIMD_NS::block_store<int32_t, 64>(output + hh * 64, inputdata);

        });
    };
    return kernel_func;
  }
  else if (last_thread_rest % 16 == 0)
  {
    int cnt_rest_16 = last_thread_rest / 16;
    cgf_t kernel_func = [=](sycl::handler& cgh) {
      cgh.parallel_for(
        Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

          int h = ndi.get_group(0);
          int hh = ndi.get_local_id(0);

          if (hh != (n_threads-1))
          {
            __ESIMD_NS::simd<int32_t, 64> inputdata = __ESIMD_NS::block_load<int32_t, 64>(input + hh * 64);
            __ESIMD_NS::block_store<int32_t, 64>(output + hh * 64, inputdata);
          }
          else
          {
            for (int k = 0; k < cnt_rest_16; k++) {
              __ESIMD_NS::simd<int32_t, 16> inputdata = __ESIMD_NS::block_load<int32_t, 16>(input + hh * 64 + 16*k);
              __ESIMD_NS::block_store<int32_t, 16>(output + hh * 64 + 16*k, inputdata);
            }
          }
        });
    };
    return kernel_func;
  }
  else
  {
    int cnt_rest_16 = (last_thread_rest + 15) / 16;
    int rest_16 = last_thread_rest - (cnt_rest_16-1) * 16;
    cgf_t kernel_func = [=](sycl::handler& cgh) {
      cgh.parallel_for(
        Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

          int h = ndi.get_group(0);
          int hh = ndi.get_local_id(0);

          if (hh != (n_threads-1))
          {
            __ESIMD_NS::simd<int32_t, 64> inputdata = __ESIMD_NS::block_load<int32_t, 64>(input + hh * 64);
            __ESIMD_NS::block_store<int32_t, 64>(output + hh * 64, inputdata);
          }
          else
          {
            if (rest_16 == 0)
            {
              for (int k = 0; k < cnt_rest_16; k++) {
                __ESIMD_NS::simd<int32_t, 16> inputdata = __ESIMD_NS::block_load<int32_t, 16>(input + hh * 64 + 16*k);
                __ESIMD_NS::block_store<int32_t, 16>(output + hh * 64 + 16*k, inputdata);
              }
            }
            else
            {
              for (int k = 0; k < cnt_rest_16-1; k++) {
                __ESIMD_NS::simd<int32_t, 16> inputdata = __ESIMD_NS::block_load<int32_t, 16>(input + hh * 64 + 16*k);
                __ESIMD_NS::block_store<int32_t, 16>(output + hh * 64 + 16*k, inputdata);
              }
              for (int k = 0; k < rest_16; k++) {
                __ESIMD_NS::simd<int32_t, 1> inputdata = __ESIMD_NS::block_load<int32_t, 1>(input + hh * 64 + 16*(cnt_rest_16-1) + k);
                __ESIMD_NS::block_store<int32_t, 1>(output + hh * 64 + 16*(cnt_rest_16-1) + k, inputdata);
              }
            }
          }
        });
    };
    return kernel_func;
  }
}

ESIMD_KERNEL_API cgf_t launch_ggemm_preprocess_scale_align(
    float* scatter_per_token_scale,
    float* scatter_per_token_scale_aligned,
    int32_t* experts_token_count,
    int32_t* experts_token_start,
    int32_t* experts_token_start_aligned,
    int32_t n_expert)
{
  int n_threads = 1;

  sycl::range<1> GlobalRange(n_expert);
  sycl::range<1> LocalRange(1);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        int h = ndi.get_group(0);

        int32_t token_count = experts_token_count[h];
        int32_t token_start = experts_token_start[h];
        int32_t token_start_aligned = experts_token_start_aligned[h];

        int loopcnt = 0;
        int32_t token_rest = token_count % 128;
        loopcnt = token_count / 128;
        for (int i = 0; i < loopcnt; i++)
        {
          __ESIMD_NS::simd<float, 128> inputdata = 
            __ESIMD_NS::block_load<float, 128>(scatter_per_token_scale + token_start + i * 128);
          __ESIMD_NS::block_store<float, 128>(scatter_per_token_scale_aligned + token_start_aligned + i * 128, inputdata);
        }

        for (int i = 0; i < token_rest; i++)
        {
          scatter_per_token_scale_aligned[token_start_aligned + loopcnt*128 + i] =
            scatter_per_token_scale[token_start + loopcnt*128 + i];
        }
      });
  };
  return kernel_func;
}

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_256E_80T(
    int32_t* selected_experts,
    int32_t* token_to_scatter_offset,
    int32_t* experts_token_count,
    int32_t n_expert,
    int32_t shared_exp_num,
    int32_t n_tokens,
    int32_t topk)
{
    int n_threads = 1;
    int n_tokens_per_thread = n_tokens / 20;

    // printf("n_threads: %d\n", n_threads);
    // printf("rest: %d\n", rest);
    // printf("last_thread_rest: %d\n", last_thread_rest);

    // topk must be 8 for now

    sycl::range<1> GlobalRange(20);
    sycl::range<1> LocalRange(20);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(
            Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            constexpr uint32_t slmSize = 256 * 4;
            __ESIMD_NS::slm_init(slmSize);

            int h = ndi.get_local_id(0);

            if (h < 16)
            {
                __ESIMD_NS::simd<int32_t, 16> zeroVector = 0;

                __ESIMD_NS::slm_block_store<int32_t, 16>(h * 64, zeroVector);
            }
            __ESIMD_NS::barrier();

            __ESIMD_NS::simd<int32_t, 32> inputdata = __ESIMD_NS::block_load<int32_t, 32>(selected_experts + h * 32);

            // 1. Define the value to add as a SIMD vector of size 1.
            __ESIMD_NS::simd<int32_t, 32> value_to_add = 1;

            // 2. Define a mask. Since we want the operation to happen, the mask is 1.
            __ESIMD_NS::simd_mask<32> mask = 1;

            __ESIMD_NS::simd<uint32_t, 32> offsets = inputdata * sizeof(int32_t);

            // 3. Call the ESIMD atomic intrinsic.
            // The return value (old_val) holds the value before the addition.
            __ESIMD_NS::simd<int32_t, 32> inputdata_write = __ESIMD_NS::slm_atomic_update<__ESIMD_NS::atomic_op::add,int32_t>(
                offsets,  // offset
                value_to_add,  // The value to add
                mask           // The execution mask
            );

            __ESIMD_NS::barrier();

            if (h < 4)
            {
                __ESIMD_NS::simd<uint32_t, 64> expertsTokenCountFromSLM;


                expertsTokenCountFromSLM = __ESIMD_NS::slm_block_load<uint32_t, 64>(h * 256);



                __ESIMD_NS::block_store<int32_t, 64>(experts_token_count + 64 * h, expertsTokenCountFromSLM);

            }

            __ESIMD_NS::block_store<int32_t, 32>(token_to_scatter_offset + h * 32, inputdata_write);

            });
        };

    return kernel_func;
}

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_256E_40T(
    int32_t* selected_experts,
    int32_t* token_to_scatter_offset,
    int32_t* experts_token_count,
    int32_t n_expert,
    int32_t shared_exp_num,
    int32_t n_tokens,
    int32_t topk)
{
    int n_threads = 1;
    int n_tokens_per_thread = n_tokens / 20;

    // printf("n_threads: %d\n", n_threads);
    // printf("rest: %d\n", rest);
    // printf("last_thread_rest: %d\n", last_thread_rest);

    // topk must be 8 for now

    sycl::range<1> GlobalRange(20);
    sycl::range<1> LocalRange(20);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(
            Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            constexpr uint32_t slmSize = 256 * 4;
            __ESIMD_NS::slm_init(slmSize);

            int h = ndi.get_local_id(0);

            if (h < 16)
            {
                __ESIMD_NS::simd<int32_t, 16> zeroVector = 0;

                __ESIMD_NS::slm_block_store<int32_t, 16>(h * 64, zeroVector);
            }
            __ESIMD_NS::barrier();

            __ESIMD_NS::simd<int32_t, 16> inputdata = __ESIMD_NS::block_load<int32_t, 16>(selected_experts + h * 16);

            // 1. Define the value to add as a SIMD vector of size 1.
            __ESIMD_NS::simd<int32_t, 16> value_to_add = 1;

            // 2. Define a mask. Since we want the operation to happen, the mask is 1.
            __ESIMD_NS::simd_mask<16> mask = 1;

            __ESIMD_NS::simd<uint32_t, 16> offsets = inputdata * sizeof(int32_t);

            // 3. Call the ESIMD atomic intrinsic.
            // The return value (old_val) holds the value before the addition.
            __ESIMD_NS::simd<int32_t, 16> inputdata_write = __ESIMD_NS::slm_atomic_update<__ESIMD_NS::atomic_op::add,int32_t>(
                offsets,  // offset
                value_to_add,  // The value to add
                mask           // The execution mask
            );

            __ESIMD_NS::barrier();

            if (h < 4)
            {
                __ESIMD_NS::simd<uint32_t, 64> expertsTokenCountFromSLM;


                expertsTokenCountFromSLM = __ESIMD_NS::slm_block_load<uint32_t, 64>(h * 256);



                __ESIMD_NS::block_store<int32_t, 64>(experts_token_count + 64 * h, expertsTokenCountFromSLM);

            }

            __ESIMD_NS::block_store<int32_t, 16>(token_to_scatter_offset + h * 16, inputdata_write);

            });
        };

    return kernel_func;
}

ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init(
      int32_t* selected_experts,
      int32_t* token_to_scatter_offset,
      int32_t* experts_token_count,
      int32_t n_expert,
      int32_t shared_exp_num,
      int32_t n_tokens,
      int32_t topk)
{
  int n_threads = 1;
  
  // printf("n_threads: %d\n", n_threads);
  // printf("rest: %d\n", rest);
  // printf("last_thread_rest: %d\n", last_thread_rest);

  // topk must be 8 for now

  sycl::range<1> GlobalRange(n_tokens);
  sycl::range<1> LocalRange(1);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        int h = ndi.get_group(0);

        __ESIMD_NS::simd<int32_t, 8> inputdata = __ESIMD_NS::block_load<int32_t, 8>(selected_experts + h*8);

        // 1. Define the value to add as a SIMD vector of size 1.
        __ESIMD_NS::simd<int32_t, 8> value_to_add = 1;

        // 2. Define a mask. Since we want the operation to happen, the mask is 1.
        __ESIMD_NS::simd_mask<8> mask = 1;

        // 3. Call the ESIMD atomic intrinsic.
        // The return value (old_val) holds the value before the addition.
        __ESIMD_NS::simd<int32_t, 8> inputdata_write = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
            experts_token_count + shared_exp_num,   // The USM pointer
            inputdata * sizeof(int32_t),  // offset
            value_to_add,  // The value to add
            mask           // The execution mask
        );

        if (h == 0)
        {
          for (int i = 0; i < shared_exp_num; i++)
          {
            __ESIMD_NS::block_store<int32_t, 1>(experts_token_count + i, n_tokens);
          }
        }

        __ESIMD_NS::block_store<int32_t, 8>(token_to_scatter_offset + h * 8, inputdata_write);
      });
  };
  return kernel_func;
}
ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_init_k(
      int32_t* selected_experts,
      int32_t* token_to_scatter_offset,
      int32_t* experts_token_count,
      int32_t n_expert,
      int32_t shared_exp_num,
      int32_t n_tokens,
      int32_t topk)
{
  int n_threads = 1;
  
  // printf("n_threads: %d\n", n_threads);
  // printf("rest: %d\n", rest);
  // printf("last_thread_rest: %d\n", last_thread_rest);

  // topk must be 8 for now

  sycl::range<1> GlobalRange(n_tokens);
  sycl::range<1> LocalRange(1);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        int h = ndi.get_group(0);
        int n=(topk+7)>>3;
        __ESIMD_NS::simd<int32_t, 8> value_to_add = 1;
        __ESIMD_NS::simd_mask<8> mask = 1;
        int lastN=topk&0x7;

        for(int i=0;i<n;++i){

          __ESIMD_NS::simd<int32_t, 8> inputdata=0;

          if( (lastN !=0) && (i==n-1)  ){
            mask=0;
            for(int j=0;j<lastN;++j){
              inputdata[j]=selected_experts[ h*topk + i * 8 + j];
              mask[j]=1;
            }
          } else {
             inputdata= __ESIMD_NS::block_load<int32_t, 8>(selected_experts + h*topk + i * 8);
          }
        

        // 1. Define the value to add as a SIMD vector of size 1.
  

        // 2. Define a mask. Since we want the operation to happen, the mask is 1.
     

        // 3. Call the ESIMD atomic intrinsic.
        // The return value (old_val) holds the value before the addition.
        __ESIMD_NS::simd<int32_t, 8> inputdata_write = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
            experts_token_count + shared_exp_num,   // The USM pointer
            inputdata * sizeof(int32_t),  // offset
            value_to_add,  // The value to add
            mask           // The execution mask
        );
            if( (lastN !=0) && (i==n-1)  ){
              for(int j=0;j<lastN;++j){
                token_to_scatter_offset[h*topk+i*8+j]=inputdata_write[j];
              }
            }
            else{
              __ESIMD_NS::block_store<int32_t, 8>(token_to_scatter_offset + h * topk + i*8, inputdata_write);
            }
      }

      if (h == 0)
      {
          for (int i = 0; i < shared_exp_num; i++)
          {
            __ESIMD_NS::block_store<int32_t, 1>(experts_token_count + i, n_tokens);
          }
      }

      });
  };
  return kernel_func;
}



ESIMD_KERNEL_API cgf_t launch_moe_scatter_dynamic_quant_indexing(
      int32_t* experts_token_count,
      int32_t* experts_token_start,
      int32_t* experts_token_start_aligned,
      int32_t n_expert,
      int32_t shared_exp_num)
{
  int n_expert_total = n_expert + shared_exp_num;
  int n_threads = (n_expert + 15) / 16;
  // support expert num <= 256 for now

  // topk must be 8 for now

  sycl::range<1> GlobalRange(n_threads);
  sycl::range<1> LocalRange(n_threads);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        __ESIMD_NS::slm_init(2 * 16 * sizeof(int32_t));

        int hh = ndi.get_local_id(0);

        __ESIMD_NS::simd<int32_t, 16> experts_token_count_data{0};

        if (n_expert_total % 16 != 0 && hh == n_threads-1)
        {
          int rest = n_expert_total % 16;
          for (int i = 0; i < rest; i++)
          {
            experts_token_count_data[i] = experts_token_count[hh*16 + i];
          }
        }
        else
        {
          experts_token_count_data = __ESIMD_NS::block_load<int32_t, 16>(experts_token_count + hh*16);
        }

        __ESIMD_NS::simd<int32_t, 16> experts_token_count_aligned_data;
          experts_token_count_aligned_data = (experts_token_count_data + 255) / 256 * 256;
        
        __ESIMD_NS::simd<int32_t, 1> experts_token_count_data_sum = 
          sycl::ext::intel::esimd::detail::sum<int32_t, int32_t, 16>(experts_token_count_data);
        __ESIMD_NS::simd<int32_t, 1> experts_token_count_aligned_sum = 
          sycl::ext::intel::esimd::detail::sum<int32_t, int32_t, 16>(experts_token_count_aligned_data);

        slm_block_store<int32_t, 1>(hh * sizeof(int32_t), experts_token_count_data_sum);
        slm_block_store<int32_t, 1>((16 + hh) * sizeof(int32_t), experts_token_count_aligned_sum);

        barrier();

        __ESIMD_NS::simd<int32_t, 16> all_acc = __ESIMD_NS::slm_block_load<int32_t, 16>(0);
        __ESIMD_NS::simd<int32_t, 16> all_acc_aligned = __ESIMD_NS::slm_block_load<int32_t, 16>(16 * sizeof(int32_t));

        int32_t acc = 0;
        int32_t acc_aligned = 0;
        for (int i = 0; i < hh; i++)
        {
          acc += all_acc[i];
          acc_aligned += all_acc_aligned[i];
        }

        __ESIMD_NS::simd<int32_t, 16> result{0};
        __ESIMD_NS::simd<int32_t, 16> result_aligned{0};

        int32_t result_acc = 0;
        int32_t result_acc_aligned = 0;
        #pragma unroll
        for (int i = 1; i < 16; i++)
        {
          result_acc += experts_token_count_data[i - 1];
          result_acc_aligned += experts_token_count_aligned_data[i - 1];
          result[i] = result_acc;
          result_aligned[i] += result_acc_aligned;
        }
        result += acc;
        result_aligned += acc_aligned;

        if (n_expert_total % 16 != 0 && hh == n_threads-1)
        {
          int rest = n_expert_total % 16;
          for (int i = 0; i < rest; i++)
          {
            experts_token_start[hh*16 + i] = result[i];
            experts_token_start_aligned[hh*16 + i] = result_aligned[i];
          }
        }
        else
        {
          __ESIMD_NS::block_store<int32_t, 16>(experts_token_start + hh * 16, result);
          __ESIMD_NS::block_store<int32_t, 16>(experts_token_start_aligned + hh * 16, result_aligned);
        }
      });
  };
  return kernel_func;
}

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
    int32_t topk)
{
  int n_threads = hd_size / 256;
  assert(n_threads <= 32);

  sycl::range<1> GlobalRange(n_tokens * n_threads);
  sycl::range<1> LocalRange(n_threads);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        __ESIMD_NS::slm_init(2 * 32 * sizeof(float));

        int h = ndi.get_group(0);
        int hh = ndi.get_local_id(0);

        __ESIMD_NS::simd<int32_t, 1> in_token_idx{h};

        // load any top-k values  exps id and weights 
        __ESIMD_NS::simd<int32_t, 8> exps = __ESIMD_NS::block_load<int32_t, 8>(selected_experts + h * 8);
        __ESIMD_NS::simd<float, 8> weights = __ESIMD_NS::block_load<float, 8>(moe_weights + h * 8);

        exps += shared_exp_num;
        
        __ESIMD_NS::simd<sycl::half, 256> hd_input_fp16 = __ESIMD_NS::block_load<sycl::half, 256>(hidden_states + h * hd_size + hh * 256);
        __ESIMD_NS::simd<float, 256> hd_input = hd_input_fp16;
        
        __ESIMD_NS::simd<float, 256 * 8> smooth_scale;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
          smooth_scale.template select<256, 1>(256 * i) = __ESIMD_NS::block_load<float, 256>(experts_smooth_scale + exps[i] * hd_size + hh * 256);
        }

        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
          float weight_in = weights[i];
          __ESIMD_NS::simd<float, 256> hd_input_smoothed = weight_in * hd_input * smooth_scale.select<256, 1>(256 * i);
          __ESIMD_NS::simd<float, 256> hd_input_abs = __ESIMD_NS::abs(hd_input_smoothed);

          float max_hd_value = hmax<float, float, 256>(hd_input_abs);

          int pingpong = i % 2;
          __ESIMD_NS::slm_block_store<float, 1>((pingpong * 32 + hh) * sizeof(float), max_hd_value);

          barrier();

          __ESIMD_NS::simd<float, 32> max_hd_value_full = __ESIMD_NS::slm_block_load<float, 32>(pingpong * 32 * sizeof(float));

          for (int j = n_threads; j < 32; j++)
          {
            max_hd_value_full[j] = -1;  // all value are > 0, mask out rest
          }

          float max_hd_value_final = hmax<float, float, 32>(max_hd_value_full);
          float per_token_scale = max_hd_value_final / 127.0;
          __ESIMD_NS::simd<float, 256> hd_quant = hd_input_smoothed / per_token_scale;
          __ESIMD_NS::simd<int8_t, 256> hd_quant_out = hd_quant;

          int32_t copy_dest_offset = __ESIMD_NS::block_load<int32_t, 1>(token_to_scatter_offset + h * 8 + i);

          int32_t exp_start = *(experts_token_start + exps[i]);

          __ESIMD_NS::block_store<int8_t, 256>(scatter_tokens + exp_start * hd_size + copy_dest_offset * hd_size + hh * 256, hd_quant_out);

          if (hh == 0)
          {
            __ESIMD_NS::block_store<float, 1>(scatter_per_token_scale + exp_start + copy_dest_offset, per_token_scale);
            __ESIMD_NS::block_store<int32_t, 1>(scatter_tokens_offset + exp_start + copy_dest_offset, in_token_idx);
          }
        }

        for (int i = 0; i < shared_exp_num; i++)
        {
          smooth_scale.template select<256, 1>(256 * i) = __ESIMD_NS::block_load<float, 256>(experts_smooth_scale + i * hd_size + hh * 256);
        }
        for (int i = 0; i < shared_exp_num; i++)
        {
          __ESIMD_NS::simd<float, 256> hd_input_smoothed = hd_input * smooth_scale.select<256, 1>(256 * i);
          __ESIMD_NS::simd<float, 256> hd_input_abs = __ESIMD_NS::abs(hd_input_smoothed);

          float max_hd_value = hmax<float, float, 256>(hd_input_abs);

          int pingpong = (8+i) % 2;
          __ESIMD_NS::slm_block_store<float, 1>((pingpong * 32 + hh) * sizeof(float), max_hd_value);

          barrier();

          __ESIMD_NS::simd<float, 32> max_hd_value_full = __ESIMD_NS::slm_block_load<float, 32>(pingpong * 32 * sizeof(float));

          for (int j = n_threads; j < 32; j++)
          {
            max_hd_value_full[j] = -1;  // all value are > 0, mask out rest
          }

          float max_hd_value_final = hmax<float, float, 32>(max_hd_value_full);
          float per_token_scale = max_hd_value_final / 127.0;
          __ESIMD_NS::simd<float, 256> hd_quant = hd_input_smoothed / per_token_scale;
          __ESIMD_NS::simd<int8_t, 256> hd_quant_out = hd_quant;

          int32_t shared_exp_start = *(experts_token_start + i);

          __ESIMD_NS::block_store<int8_t, 256>(scatter_tokens + (shared_exp_start + h) * hd_size + hh * 256, hd_quant_out);
          if (hh == 0)
          {
            __ESIMD_NS::block_store<float, 1>(scatter_per_token_scale + shared_exp_start + h, per_token_scale);
            __ESIMD_NS::block_store<int32_t, 1>(scatter_tokens_offset + shared_exp_start + h, in_token_idx);
          }
        }

      });
  };
  return kernel_func;
}
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
    int32_t topk)
{
  int n_threads = hd_size / 256;
  assert(n_threads <= 32);

  
  sycl::range<1> GlobalRange(n_tokens * n_threads);
  sycl::range<1> LocalRange(n_threads);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {

        __ESIMD_NS::slm_init(2 * 32 * sizeof(float));

        int h = ndi.get_group(0);
        int hh = ndi.get_local_id(0);

        __ESIMD_NS::simd<int32_t, 1> in_token_idx{h};

        int n=(topk+7)>>3;
        int lastN=topk&0x7;
        
        __ESIMD_NS::simd<int32_t, 8> exps;
        __ESIMD_NS::simd<float, 8> weights;
        __ESIMD_NS::simd<sycl::half, 256> hd_input_fp16 = __ESIMD_NS::block_load<sycl::half, 256>(hidden_states + h * hd_size + hh * 256);
        __ESIMD_NS::simd<float, 256> hd_input = hd_input_fp16;
        __ESIMD_NS::simd<float, 256 * 8> smooth_scale;
        for(int k =0; k<n; ++k){
          if( (lastN!=0) && (k==n-1) ){
              for(int j=0;j<lastN;++j){
                  exps[j]    = selected_experts[h * topk+k*8+j];
                  weights[j] = moe_weights[h * topk+k*8+j];
              }
              exps += shared_exp_num;
              

              for (int i = 0; i < lastN; i++){
                smooth_scale.template select<256, 1>(256 * i) = __ESIMD_NS::block_load<float, 256>(experts_smooth_scale + exps[i] * hd_size + hh * 256);
              }
              for (int i = 0; i < lastN; i++){
                float weight_in = weights[i];
                __ESIMD_NS::simd<float, 256> hd_input_smoothed = weight_in * hd_input * smooth_scale.select<256, 1>(256 * i);
                __ESIMD_NS::simd<float, 256> hd_input_abs = __ESIMD_NS::abs(hd_input_smoothed);

                float max_hd_value = hmax<float, float, 256>(hd_input_abs);

                int pingpong = i % 2;
                __ESIMD_NS::slm_block_store<float, 1>((pingpong * 32 + hh) * sizeof(float), max_hd_value);

                barrier();

                __ESIMD_NS::simd<float, 32> max_hd_value_full = __ESIMD_NS::slm_block_load<float, 32>(pingpong * 32 * sizeof(float));

                for (int j = n_threads; j < 32; j++)
                {
                  max_hd_value_full[j] = -1;  // all value are > 0, mask out rest
                }

                float max_hd_value_final = hmax<float, float, 32>(max_hd_value_full);
                float per_token_scale = max_hd_value_final / 127.0;
                __ESIMD_NS::simd<float, 256> hd_quant = hd_input_smoothed / per_token_scale;
                __ESIMD_NS::simd<int8_t, 256> hd_quant_out = hd_quant;

                int32_t copy_dest_offset = __ESIMD_NS::block_load<int32_t, 1>(token_to_scatter_offset + h * topk +8 * k + i);

                int32_t exp_start = *(experts_token_start + exps[i]);

                __ESIMD_NS::block_store<int8_t, 256>(scatter_tokens + exp_start * hd_size + copy_dest_offset * hd_size + hh * 256, hd_quant_out);

                if (hh == 0)
                {
                  __ESIMD_NS::block_store<float, 1>(scatter_per_token_scale + exp_start + copy_dest_offset, per_token_scale);
                  __ESIMD_NS::block_store<int32_t, 1>(scatter_tokens_offset + exp_start + copy_dest_offset, in_token_idx);
                }
              }
          }
          else{
            // load any top-k values  exps id and weights 
            __ESIMD_NS::simd<int32_t, 8> exps = __ESIMD_NS::block_load<int32_t, 8>(selected_experts +h * topk+ k * 8);
            __ESIMD_NS::simd<float, 8> weights = __ESIMD_NS::block_load<float, 8>(moe_weights +h * topk+ k * 8);

            exps += shared_exp_num;
        
            
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
              smooth_scale.template select<256, 1>(256 * i) = __ESIMD_NS::block_load<float, 256>(experts_smooth_scale + exps[i] * hd_size + hh * 256);
            }

            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
              float weight_in = weights[i];
              __ESIMD_NS::simd<float, 256> hd_input_smoothed = weight_in * hd_input * smooth_scale.select<256, 1>(256 * i);
              __ESIMD_NS::simd<float, 256> hd_input_abs = __ESIMD_NS::abs(hd_input_smoothed);

              float max_hd_value = hmax<float, float, 256>(hd_input_abs);

              int pingpong = i % 2;
              __ESIMD_NS::slm_block_store<float, 1>((pingpong * 32 + hh) * sizeof(float), max_hd_value);

              barrier();

              __ESIMD_NS::simd<float, 32> max_hd_value_full = __ESIMD_NS::slm_block_load<float, 32>(pingpong * 32 * sizeof(float));

              for (int j = n_threads; j < 32; j++)
              {
                max_hd_value_full[j] = -1;  // all value are > 0, mask out rest
              }

              float max_hd_value_final = hmax<float, float, 32>(max_hd_value_full);
              float per_token_scale = max_hd_value_final / 127.0;
              __ESIMD_NS::simd<float, 256> hd_quant = hd_input_smoothed / per_token_scale;
              __ESIMD_NS::simd<int8_t, 256> hd_quant_out = hd_quant;

              int32_t copy_dest_offset = __ESIMD_NS::block_load<int32_t, 1>(token_to_scatter_offset +h * topk +8 * k  + i);

              int32_t exp_start = *(experts_token_start + exps[i]);

              __ESIMD_NS::block_store<int8_t, 256>(scatter_tokens + exp_start * hd_size + copy_dest_offset * hd_size + hh * 256, hd_quant_out);

              if (hh == 0)
              {
                __ESIMD_NS::block_store<float, 1>(scatter_per_token_scale + exp_start + copy_dest_offset, per_token_scale);
                __ESIMD_NS::block_store<int32_t, 1>(scatter_tokens_offset + exp_start + copy_dest_offset, in_token_idx);
              }
            }
          }
        }
        for (int i = 0; i < shared_exp_num; i++)
        {
          smooth_scale.template select<256, 1>(256 * i) = __ESIMD_NS::block_load<float, 256>(experts_smooth_scale + i * hd_size + hh * 256);
        }
        for (int i = 0; i < shared_exp_num; i++)
        {
          __ESIMD_NS::simd<float, 256> hd_input_smoothed = hd_input * smooth_scale.select<256, 1>(256 * i);
          __ESIMD_NS::simd<float, 256> hd_input_abs = __ESIMD_NS::abs(hd_input_smoothed);

          float max_hd_value = hmax<float, float, 256>(hd_input_abs);

          int pingpong = (8+i) % 2;
          __ESIMD_NS::slm_block_store<float, 1>((pingpong * 32 + hh) * sizeof(float), max_hd_value);

          barrier();

          __ESIMD_NS::simd<float, 32> max_hd_value_full = __ESIMD_NS::slm_block_load<float, 32>(pingpong * 32 * sizeof(float));

          for (int j = n_threads; j < 32; j++)
          {
            max_hd_value_full[j] = -1;  // all value are > 0, mask out rest
          }

          float max_hd_value_final = hmax<float, float, 32>(max_hd_value_full);
          float per_token_scale = max_hd_value_final / 127.0;
          __ESIMD_NS::simd<float, 256> hd_quant = hd_input_smoothed / per_token_scale;
          __ESIMD_NS::simd<int8_t, 256> hd_quant_out = hd_quant;

          int32_t shared_exp_start = *(experts_token_start + i);

          __ESIMD_NS::block_store<int8_t, 256>(scatter_tokens + (shared_exp_start + h) * hd_size + hh * 256, hd_quant_out);
          if (hh == 0)
          {
            __ESIMD_NS::block_store<float, 1>(scatter_per_token_scale + shared_exp_start + h, per_token_scale);
            __ESIMD_NS::block_store<int32_t, 1>(scatter_tokens_offset + shared_exp_start + h, in_token_idx);
          }
        }

      });
  };
  return kernel_func;
}

// port from ds r1 opt ===============================================================================================

using fp16 = sycl::half;

ESIMD_KERNEL_API cgf_t esimd_grouped_topk(
  uint8_t* gating_output, uint8_t* correction_bias, uint8_t* topk_weights, uint8_t* topk_ids,
  uint8_t* cpu_hidden_states, uint8_t* hidden_states,
  int64_t topk_in,
  int64_t topk_group_in,
  int64_t num_expert_group_in,
  int64_t input_len,
  int64_t renormalize,
  float routed_scaling_factor)
{
  if (!(topk_in == 8 && topk_group_in == 4 && num_expert_group_in == 8))
  {
    std::cout << "[esimd_grouped_topk] Not supported topk topk_group num_expert_group "
      << topk_in << " " <<  topk_group_in << " " << num_expert_group_in << " " << std::endl;
    return;
  }

  // routed_scaling_factor is not used yet

  constexpr uint32_t topk = 8;
  constexpr uint32_t topk_group = 4;
  constexpr uint32_t num_expert_group = 8;
  // assume experts is 256, so each group is 32

  uint32_t copy_to_cpu_grp_num = 1;

  sycl::range<2> GlobalRange(copy_to_cpu_grp_num, input_len*num_expert_group);
  sycl::range<2> LocalRange(1, num_expert_group);
  sycl::nd_range<2> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        // do topk
        {

        // (8, 8) top8 for each group for choice   (8, 8) top8 idx for each group
        // 8 score for top2 for each group
        // (8, 8) top8 for each group real
        __ESIMD_NS::slm_init(topk * num_expert_group * sizeof(fp16) +
         topk * num_expert_group * sizeof(int16_t) +
        num_expert_group * sizeof(fp16) +
        topk * num_expert_group * sizeof(fp16));

        constexpr uint32_t top8_idx_for_each_group_offset = num_expert_group*8*sizeof(fp16);
        constexpr uint32_t groupscore_offset = top8_idx_for_each_group_offset + num_expert_group*8*sizeof(int16_t);
        constexpr uint32_t realscore_offset = groupscore_offset + num_expert_group * sizeof(fp16);

        int h = ndi.get_group(1);
        int cnt = ndi.get_local_id(1);

        __ESIMD_NS::simd<fp16, 32> input;
        __ESIMD_NS::simd<fp16, 32> input_forchoice;
        __ESIMD_NS::simd<fp16, 32> bias;
        __ESIMD_NS::simd<float, 32> input32;
        __ESIMD_NS::simd<fp16, 8> topkdata;
        __ESIMD_NS::simd<fp16, 8> topkdata_for_choice;
        topkdata = -INFINITY;
        topkdata_for_choice = -INFINITY;
        __ESIMD_NS::simd<int16_t, 8> max_k_idx;
        max_k_idx = -1;

        input.template bit_cast_view<uint8_t>().template select<64, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)gating_output + h * 32 * num_expert_group * sizeof(fp16) + cnt * 32 * sizeof(fp16));

        bias.template bit_cast_view<uint8_t>().template select<64, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)correction_bias + cnt * 32 * sizeof(fp16));


        // sigmoid
        input32 = input;
        input32.select<32, 1>(0) = pow<float, 32, float>(2.718f, (-1.0) * input32.select<32, 1>(0));
        input32.select<32, 1>(0) = 1.0 / (1.0 + input32.select<32, 1>(0));
        input = input32;

        input_forchoice = input + bias;

        // top 8 & top 2 sum in group 32 data
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
          #pragma unroll
          for (int k = 0; k < 32; k++)
          {
            if (topkdata_for_choice[j] < input_forchoice[k])
            {
              topkdata[j] = input[k]; // note, save the data, not for choice.
              topkdata_for_choice[j] = input_forchoice[k];
              max_k_idx[j] = k;
            }
          }
          if (max_k_idx[j] >= 0)
          {
            input_forchoice[max_k_idx[j]] = -INFINITY;
          }
        }
        max_k_idx += cnt*32;

        __ESIMD_NS::slm_block_store<fp16, 8>(cnt*8*sizeof(fp16), topkdata_for_choice);
        __ESIMD_NS::slm_block_store<int16_t, 8>(top8_idx_for_each_group_offset + cnt*8*sizeof(int16_t), max_k_idx);
        __ESIMD_NS::slm_block_store<fp16, 1>(groupscore_offset + cnt * sizeof(fp16), topkdata_for_choice[0] + topkdata_for_choice[1]);
        __ESIMD_NS::slm_block_store<fp16, 8>(realscore_offset + cnt*8*sizeof(fp16), topkdata);

        barrier(); // -----------------------------------------------

        if (cnt == 0)
        {
          topkdata = -INFINITY;
          __ESIMD_NS::simd<fp16, 8> group_score;
          group_score = __ESIMD_NS::slm_block_load<fp16, 8>(groupscore_offset);
          __ESIMD_NS::simd<fp16, 8*4> selected_group_data;
          __ESIMD_NS::simd<fp16, 8*4> selected_group_data_real;
          __ESIMD_NS::simd<fp16, 8*4> selected_group_idx;

          // __ESIMD_ENS::lsc_block_store<
          //   fp16,
          //   8,
          //   __ESIMD_ENS::lsc_data_size::default_size,
          //   __ESIMD_ENS::cache_hint::write_back,
          //   __ESIMD_ENS::cache_hint::write_back>((fp16*)debug_buffer + h * 8, group_score);
          // __ESIMD_NS::simd<fp16, 4> group_idx;

          // topk_group is 4
          // num_expert_group is 8

          int cur_sel_idx = 0;

          // select top 4 group and get data and index of those 4 groups (top8)
          #pragma unroll
          for (int j = 0; j < 4; j++)
          {
            #pragma unroll
            for (int k = 0; k < 8; k++)
            {
              if (topkdata[j] < group_score[k])
              {
                topkdata[j] = group_score[k];
                cur_sel_idx = k;
                // group_idx[j] = k;
              }
            }
            if (cur_sel_idx >= 0)
            {
              // select cur_sel_idx th group and set to j th GRF.
              selected_group_data.select<8, 1>(j*8) = __ESIMD_NS::slm_block_load<fp16, 8>(cur_sel_idx*8*sizeof(fp16));
              selected_group_data_real.select<8, 1>(j*8) = __ESIMD_NS::slm_block_load<fp16, 8>(realscore_offset + cur_sel_idx*8*sizeof(fp16));
              selected_group_idx.select<8, 1>(j*8) = __ESIMD_NS::slm_block_load<int16_t, 8>(top8_idx_for_each_group_offset + cur_sel_idx*8*sizeof(int16_t));
              group_score[cur_sel_idx] = -INFINITY;
            }
          }

          // __ESIMD_ENS::lsc_block_store<
          //   int32_t,
          //   4,
          //   __ESIMD_ENS::lsc_data_size::default_size,
          //   __ESIMD_ENS::cache_hint::write_back,
          //   __ESIMD_ENS::cache_hint::write_back>((int32_t*)debug_buffer_2 + h * 4, group_idx);

          max_k_idx = 0; // avoid the issue
          topkdata_for_choice = -INFINITY;
          topkdata = -INFINITY;

          // finally find top8
          #pragma unroll
          for (int j = 0; j < 8; j++)
          {
            #pragma unroll
            for (int k = 0; k < 4*8; k++)
            {
              if (topkdata_for_choice[j] < selected_group_data[k])
              {
                topkdata_for_choice[j] = selected_group_data[k];
                topkdata[j] = selected_group_data_real[k];  // get from real data, but sort by for choice data
                max_k_idx[j] = selected_group_idx[k];
                cur_sel_idx = k;
              }
            }
            if (cur_sel_idx >= 0)
            {
              selected_group_data[cur_sel_idx] = -INFINITY;
            }
          }

          if (renormalize)
          {
            __ESIMD_NS::simd<fp16, 8> topkdata_tmp;
            topkdata_tmp = topkdata;
            topkdata_tmp.select<4, 1>(0) += topkdata_tmp.select<4, 1>(4);
            topkdata_tmp.select<2, 1>(0) += topkdata_tmp.select<2, 1>(2);
            topkdata_tmp[0] += topkdata_tmp[1];

            fp16 sumdata = topkdata_tmp[0];

            topkdata = topkdata / sumdata;
          }

          __ESIMD_ENS::lsc_block_store<
            float,
            8,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((float*)topk_weights + 8 * h, topkdata.select<8, 1>(0));
          __ESIMD_ENS::lsc_block_store<
            int32_t,
            8,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((int32_t*)topk_ids + 8 * h, max_k_idx.select<8, 1>(0));
        }

        } // check hc

      });
  };
  return kernel_func;

}



ESIMD_KERNEL_API cgf_t esimd_grouped_topk_kimi(
  uint8_t* gating_output, uint8_t* correction_bias, uint8_t* topk_weights, uint8_t* topk_ids,
  uint8_t* cpu_hidden_states, uint8_t* hidden_states,
  int64_t topk_in,
  int64_t topk_group_in,
  int64_t num_expert_group_in,
  int64_t input_len,
  int64_t renormalize,
  float routed_scaling_factor)
{
  if (!(topk_in == 8 && topk_group_in == 1 && num_expert_group_in == 1))
  {
    std::cout << "[esimd_grouped_topk] Not supported topk topk_group num_expert_group "
      << topk_in << " " <<  topk_group_in << " " << num_expert_group_in << " " << std::endl;
    return;
  }

  // routed_scaling_factor is not used yet

  constexpr uint32_t topk = 8;
  constexpr uint32_t topk_group = 1;
  constexpr uint32_t num_expert_group = 1;
  // assume experts is 384, so each group is 384

  uint32_t copy_to_cpu_grp_num = 1;

  // 12 is 384/32
  sycl::range<2> GlobalRange(copy_to_cpu_grp_num, input_len*12);
  sycl::range<2> LocalRange(1, 12);
  sycl::nd_range<2> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        // do topk
        {

        // (12, 8) top8 for each group for choice   (12, 8) top8 idx for each group
        // (12, 8) top8 for each group real
        __ESIMD_NS::slm_init(topk * 12 * sizeof(fp16) +
         topk * 12 * sizeof(int16_t) +
        topk * 12 * sizeof(fp16));

        constexpr uint32_t top8_idx_for_each_group_offset = 12*8*sizeof(fp16);
        constexpr uint32_t realscore_offset = top8_idx_for_each_group_offset + 12*8*sizeof(int16_t);

        int h = ndi.get_group(1);
        int cnt = ndi.get_local_id(1);

        __ESIMD_NS::simd<fp16, 32> input;
        __ESIMD_NS::simd<fp16, 32> input_forchoice;
        __ESIMD_NS::simd<fp16, 32> bias;
        __ESIMD_NS::simd<float, 32> input32;
        __ESIMD_NS::simd<fp16, 8> topkdata;
        __ESIMD_NS::simd<fp16, 8> topkdata_for_choice;
        topkdata = -INFINITY;
        topkdata_for_choice = -INFINITY;
        __ESIMD_NS::simd<int16_t, 8> max_k_idx;
        max_k_idx = -1;

        input.template bit_cast_view<uint8_t>().template select<64, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)gating_output + h * 384 * sizeof(fp16) + cnt * 32 * sizeof(fp16));

        bias.template bit_cast_view<uint8_t>().template select<64, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)correction_bias + cnt * 32 * sizeof(fp16));


        // sigmoid
        input32 = input;
        input32.select<32, 1>(0) = pow<float, 32, float>(2.718f, (-1.0) * input32.select<32, 1>(0));
        input32.select<32, 1>(0) = 1.0 / (1.0 + input32.select<32, 1>(0));
        input = input32;

        input_forchoice = input + bias;

        // top 8 & top 2 sum in group 32 data
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
          #pragma unroll
          for (int k = 0; k < 32; k++)
          {
            if (topkdata_for_choice[j] < input_forchoice[k])
            {
              topkdata[j] = input[k]; // note, save the data, not for choice.
              topkdata_for_choice[j] = input_forchoice[k];
              max_k_idx[j] = k;
            }
          }
          if (max_k_idx[j] >= 0)
          {
            input_forchoice[max_k_idx[j]] = -INFINITY;
          }
        }
        max_k_idx += cnt*32;

        __ESIMD_NS::slm_block_store<fp16, 8>(cnt*8*sizeof(fp16), topkdata_for_choice);
        __ESIMD_NS::slm_block_store<int16_t, 8>(top8_idx_for_each_group_offset + cnt*8*sizeof(int16_t), max_k_idx);
        __ESIMD_NS::slm_block_store<fp16, 8>(realscore_offset + cnt*8*sizeof(fp16), topkdata);

        barrier(); // -----------------------------------------------

        if (cnt == 0)
        {
          topkdata = -INFINITY;
          __ESIMD_NS::simd<fp16, 8*12> selected_group_data;
          __ESIMD_NS::simd<fp16, 8*12> selected_group_data_real;
          __ESIMD_NS::simd<fp16, 8*12> selected_group_idx;

          // __ESIMD_ENS::lsc_block_store<
          //   fp16,
          //   8,
          //   __ESIMD_ENS::lsc_data_size::default_size,
          //   __ESIMD_ENS::cache_hint::write_back,
          //   __ESIMD_ENS::cache_hint::write_back>((fp16*)debug_buffer + h * 8, group_score);
          // __ESIMD_NS::simd<fp16, 4> group_idx;

          // topk_group is 1
          // num_expert_group is 1

          int cur_sel_idx = 0;

          // select all group since only one group
          selected_group_data.select<12*8, 1>(0) = __ESIMD_NS::slm_block_load<fp16, 12*8>(0);
          selected_group_data_real.select<12*8, 1>(0) = __ESIMD_NS::slm_block_load<fp16, 12*8>(realscore_offset);
          selected_group_idx.select<12*8, 1>(0) = __ESIMD_NS::slm_block_load<int16_t, 12*8>(top8_idx_for_each_group_offset);

          // __ESIMD_ENS::lsc_block_store<
          //   int32_t,
          //   4,
          //   __ESIMD_ENS::lsc_data_size::default_size,
          //   __ESIMD_ENS::cache_hint::write_back,
          //   __ESIMD_ENS::cache_hint::write_back>((int32_t*)debug_buffer_2 + h * 4, group_idx);

          max_k_idx = 0; // avoid the issue
          topkdata_for_choice = -INFINITY;
          topkdata = -INFINITY;

          // finally find top8
          #pragma unroll
          for (int j = 0; j < 8; j++)
          {
            #pragma unroll
            for (int k = 0; k < 12*8; k++)
            {
              if (topkdata_for_choice[j] < selected_group_data[k])
              {
                topkdata_for_choice[j] = selected_group_data[k];
                topkdata[j] = selected_group_data_real[k];  // get from real data, but sort by for choice data
                max_k_idx[j] = selected_group_idx[k];
                cur_sel_idx = k;
              }
            }
            if (cur_sel_idx >= 0)
            {
              selected_group_data[cur_sel_idx] = -INFINITY;
            }
          }

          if (renormalize)
          {
            __ESIMD_NS::simd<fp16, 8> topkdata_tmp;
            topkdata_tmp = topkdata;
            topkdata_tmp.select<4, 1>(0) += topkdata_tmp.select<4, 1>(4);
            topkdata_tmp.select<2, 1>(0) += topkdata_tmp.select<2, 1>(2);
            topkdata_tmp[0] += topkdata_tmp[1];

            fp16 sumdata = topkdata_tmp[0];

            topkdata = topkdata / sumdata;
          }

          __ESIMD_ENS::lsc_block_store<
            float,
            8,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((float*)topk_weights + 8 * h, topkdata.select<8, 1>(0));
          __ESIMD_ENS::lsc_block_store<
            int32_t,
            8,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((int32_t*)topk_ids + 8 * h, max_k_idx.select<8, 1>(0));
        }

        } // check hc

      });
  };
  return kernel_func;


}


/**
 *
 *
 *
 * def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

 *
        dtype = query.dtype
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query.to(dtype), key.to(dtype)

*/
ESIMD_INLINE void rotary_pos_emb_ds(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* cos_sin_cache,
    uint8_t* positions,
    uint8_t* offsets,
    int num_heads_q,
    int hidden_dim_q,
    int hd_stride_q,
    int num_heads_kv,
    int hidden_dim_kv,
    int hd_stride_kv,
    int head_hd_stride_kv,
    int has_offset,
    sycl::nd_item<2>& ndi) {
  int h = ndi.get_group(1);  // [0, q_heads+k_heads)
  int v = ndi.get_group(0);  // [0, input_len)

  __ESIMD_NS::simd<fp16, 64> input;
  __ESIMD_NS::simd<fp16, 64> input_rotate_half;
  __ESIMD_NS::simd<fp16, 64> output;
  __ESIMD_NS::simd<fp16, 64> cos_value;
  __ESIMD_NS::simd<fp16, 64> sin_value;
  uint64_t positions_index = ((uint64_t*)positions)[v];
  if (has_offset) {
    positions_index += ((uint64_t*)offsets)[v];
  }
  unsigned int offsetCosSin = positions_index * hidden_dim_q;

  __ESIMD_NS::simd<fp16, 64> cos_sin_value;
  cos_sin_value.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
      fp16,
      64,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((fp16*)cos_sin_cache + offsetCosSin);

  cos_value.select<32, 2>(0) = cos_sin_value.select<32, 1>(0);
  cos_value.select<32, 2>(1) = cos_sin_value.select<32, 1>(0);
  sin_value.select<32, 2>(0) = cos_sin_value.select<32, 1>(32);
  sin_value.select<32, 2>(1) = cos_sin_value.select<32, 1>(32);

  if (h < num_heads_q)  // q
  {
    unsigned int InOffset = h * hd_stride_q + v * hd_stride_q * num_heads_q;
    input.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((fp16*)qState + InOffset);

    input_rotate_half.select<32, 2>(1) = input.select<32, 2>(0);
    input_rotate_half.select<32, 2>(0) = input.select<32, 2>(1) * -1.0;

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)qState + InOffset, output.select<64, 1>(0));
  } else if (h < num_heads_q + num_heads_kv)  // k
  {
    unsigned int InOffset = (h - num_heads_q) * hd_stride_kv + v * head_hd_stride_kv;
    input.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((fp16*)kState + InOffset);

    input_rotate_half.select<32, 2>(1) = input.select<32, 2>(0);
    input_rotate_half.select<32, 2>(0) = input.select<32, 2>(1) * -1.0;

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)kState + InOffset, output.select<64, 1>(0));
  }
}




ESIMD_KERNEL_API cgf_t esimd_rotary_pos_emb_ds(uint8_t* qState, uint8_t* kState, uint8_t* cos_sin_cache, uint8_t* positions, uint8_t* offsets,
 int num_heads_q, int hidden_dim_q, int hd_stride_q,
 int num_heads_kv, int hidden_dim_kv, int hd_stride_kv, int head_hd_stride_kv, int input_len, int has_offset) {

  // printf("input_len = %d", (int)input_len);
  sycl::range<2> GlobalRange(input_len, num_heads_q + num_heads_kv);
  sycl::range<2> LocalRange(1, 1);
  sycl::nd_range<2> Range(GlobalRange, LocalRange);
  static int callCount = 0;
  // printf("---------- esimd_rotary_pos_emb, GlobalRange: %d, %d ----------\n", num_heads_q + num_heads_kv, 1 );
  // currently only support hidden_dim_q and hidden_dim_kv 64

  cgf_t kernel_func = [=](sycl::handler& cgh) {
      cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
          rotary_pos_emb_ds(qState, kState, cos_sin_cache, positions, offsets,
            num_heads_q, hidden_dim_q, hd_stride_q, num_heads_kv, hidden_dim_kv, hd_stride_kv, head_hd_stride_kv, has_offset, ndi);
      });
  };
  return kernel_func;

}

// ESIMD_INLINE void residual_rmsNorm128PerThread_64t(uint8_t* weight, uint8_t* residual, uint8_t* hidden_states, uint8_t* hidden_states_out, int64_t hidden_size, int64_t input_len, int64_t add_residual, float variance_epsilon ,sycl::nd_item<1>& ndi) {

//   __ESIMD_NS::slm_init(32 * sizeof(float));

//   int h = ndi.get_group(0);
//   int hh = ndi.get_local_linear_id();
//   uint32_t inputOffset = 256 * hh * sizeof(fp16);
//   int active_thread_num = hidden_size / 256;
//   if (h >= input_len) return;

//   __ESIMD_NS::simd<fp16, 256> input_FP16;
//   __ESIMD_NS::simd<float, 256> input;
//   __ESIMD_NS::simd<float, 256> input_powered;
//   __ESIMD_NS::simd<fp16, 256> weight_FP16;
//   __ESIMD_NS::simd<float, 16> variance = 0;
//   __ESIMD_NS::simd<fp16, 256> residual_FP16;

//   __ESIMD_NS::simd<float, 32> varianceSum = 0;
//   int no_active_cnt = 32 - active_thread_num;

//   if (hh < active_thread_num)
//   {
//     input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
//           __ESIMD_ENS::lsc_block_load<
//           uint8_t,
//           256,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::cached,
//           __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset);
//     input_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(256) =
//           __ESIMD_ENS::lsc_block_load<
//           uint8_t,
//           256,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::cached,
//           __ESIMD_ENS::cache_hint::cached>((uint8_t*)hidden_states + h * hidden_size * sizeof(fp16) + inputOffset + 128 * sizeof(fp16));
    
//     if (add_residual == 1)
//     {
//       residual_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
//             __ESIMD_ENS::lsc_block_load<
//             uint8_t,
//             256,
//             __ESIMD_ENS::lsc_data_size::default_size,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached>((uint8_t*)residual + h * hidden_size * sizeof(fp16) + inputOffset);
//       residual_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(256) =
//             __ESIMD_ENS::lsc_block_load<
//             uint8_t,
//             256,
//             __ESIMD_ENS::lsc_data_size::default_size,
//             __ESIMD_ENS::cache_hint::cached,
//             __ESIMD_ENS::cache_hint::cached>((uint8_t*)residual + h * hidden_size * sizeof(fp16) + inputOffset + 128 * sizeof(fp16));
//     }

//     weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
//           __ESIMD_ENS::lsc_block_load<
//           uint8_t,
//           256,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::cached,
//           __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset);
//     weight_FP16.template bit_cast_view<uint8_t>().template select<256, 1>(256) =
//           __ESIMD_ENS::lsc_block_load<
//           uint8_t,
//           256,
//           __ESIMD_ENS::lsc_data_size::default_size,
//           __ESIMD_ENS::cache_hint::cached,
//           __ESIMD_ENS::cache_hint::cached>((uint8_t*)weight + inputOffset + 128 * sizeof(fp16));
//   }
//   else
//   {
//     input_FP16 = 0;
//     weight_FP16 = 0;
//     residual_FP16 = 0;
//   }

//   if (add_residual == 1)
//   {
//     input = input_FP16 + residual_FP16;
//   }
//   else
//   {
//     input = input_FP16;
//   }
  
//   // write back new residual
//   if (hh < active_thread_num)
//   {
//     __ESIMD_ENS::lsc_block_store<
//       fp16,
//       128,
//       __ESIMD_ENS::lsc_data_size::default_size,
//       __ESIMD_ENS::cache_hint::write_back,
//       __ESIMD_ENS::cache_hint::write_back>((fp16*)residual + h * hidden_size + 256 * hh, input.select<128, 1>(0));
//     __ESIMD_ENS::lsc_block_store<
//       fp16,
//       128,
//       __ESIMD_ENS::lsc_data_size::default_size,
//       __ESIMD_ENS::cache_hint::write_back,
//       __ESIMD_ENS::cache_hint::write_back>((fp16*)residual + h * hidden_size + 256 * hh + 128, input.select<128, 1>(128));
//   }

// #pragma unroll
//   for (int ll = 0; ll < 16; /*16*16=256*/ ll++) {
//     input_powered.select<16, 1>(ll *16) = pow<float, 16, float>(input.select<16, 1>(ll *16), 2.0f);
//     variance += input_powered.select<16, 1>(ll *16);
//   }

//   variance.select<8, 1>(0) += variance.select<8, 1>(8);
//   variance.select<4, 1>(0) += variance.select<4, 1>(4);
//   variance.select<2, 1>(0) += variance.select<2, 1>(2);
//   variance[0] += variance[1];

//   __ESIMD_NS::slm_block_store<float, 1>(hh * sizeof(float), variance[0]);

//   barrier();
  
//   varianceSum.select<32, 1>(0) = __ESIMD_NS::slm_block_load<float, 32>(0);

//   for (int i = active_thread_num; i < 32; i++)
//   {
//     varianceSum[i] = 0;
//   }
  
//   varianceSum.select<16, 1>(0) += varianceSum.select<16, 1>(16);
//   varianceSum.select<8, 1>(0) += varianceSum.select<8, 1>(8);
//   varianceSum.select<4, 1>(0) += varianceSum.select<4, 1>(4);
//   varianceSum.select<2, 1>(0) += varianceSum.select<2, 1>(2);
//   varianceSum[0] += varianceSum[1];

//   varianceSum[0] = varianceSum[0] / hidden_size;
//   varianceSum[0] =  sqrt(varianceSum[0] + variance_epsilon);

//   __ESIMD_NS::simd<float, 256> varianceAVG;
//   // fill 128 length varianceAVG
//   varianceAVG[0] = varianceSum[0];
//   varianceAVG[1] = varianceAVG[0];
//   varianceAVG.select<2, 1>(2) = varianceAVG.select<2, 1>(0);
//   varianceAVG.select<4, 1>(4) = varianceAVG.select<4, 1>(0);
//   varianceAVG.select<8, 1>(8) = varianceAVG.select<8, 1>(0);
//   varianceAVG.select<16, 1>(16) = varianceAVG.select<16, 1>(0);
//   varianceAVG.select<32, 1>(32) = varianceAVG.select<32, 1>(0);
//   varianceAVG.select<64, 1>(64) = varianceAVG.select<64, 1>(0);
//   varianceAVG.select<128, 1>(128) = varianceAVG.select<128, 1>(0);
//   // simd<float, 128> varianceAVG{varianceSum[0]};

//   input = input / varianceAVG;

//   input_FP16 = input;
//   input_FP16 = input_FP16 * weight_FP16;

//   if (hh < active_thread_num)
//   {
//     __ESIMD_ENS::lsc_block_store<
//       fp16,
//       128,
//       __ESIMD_ENS::lsc_data_size::default_size,
//       __ESIMD_ENS::cache_hint::write_back,
//       __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 256 * hh, input_FP16.select<128, 1>(0));
//     __ESIMD_ENS::lsc_block_store<
//       fp16,
//       128,
//       __ESIMD_ENS::lsc_data_size::default_size,
//       __ESIMD_ENS::cache_hint::write_back,
//       __ESIMD_ENS::cache_hint::write_back>((fp16*)hidden_states_out + h * hidden_size + 256 * hh + 128, input_FP16.select<128, 1>(128));
//   }
// }

// ESIMD_KERNEL_API cgf_t esimd_residual_kernel_rms_norm(
//   uint8_t* weight, uint8_t* residual, uint8_t* hidden_states, uint8_t* hidden_states_out,

//   int64_t hidden_size,
//   int64_t input_len,
//   int64_t add_residual,
//   float variance_epsilon)
// {

// {
//   int threads = hidden_size / 256;

//   sycl::range<1> GlobalRange(input_len*threads); // input_len
//   sycl::range<1> LocalRange(threads);
//   sycl::nd_range<1> Range(GlobalRange, LocalRange);

//   sycl::event e;
//   cgf_t kernel_func = [=](sycl::handler& cgh) {
//     cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
//       residual_rmsNorm128PerThread_64t(weight, residual, hidden_states, hidden_states_out, hidden_size, input_len, add_residual,  variance_epsilon, ndi);
//       });
//   };
//   return kernel_func;
// }
// }

#define FP32_MAX (1.7e+38)
#define FP32_MIN (-1.7e+38)
#define FP16_MAX (65504.0f)
#define FP16_MIN (-65504.0f)
#include "sage.attn.kernel.h"
#include "sage.decode.kernel.h"
#include "sage.attn.kernel_decoder.h"

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
  )
{
  if (activationLength > kvSeqLen) {
    std::cout << "wrong activationLength" << std::endl;
  }
  uint32_t startActivationIdx = kvSeqLen - activationLength;
  uint32_t gqaRatio = headQ / headKv;
  int groupH;
  int groupV;
  int localH;
  int localV;
  switch (gqaRatio) {
  case 12:
    groupH = (longestBatch + 63) / 64;
    groupV = headQ / 4;
    localH = 16;
    localV = 1;
    break;
  default:
    groupH = (longestBatch + 31) / 32;
    groupV = headKv;
    localH = 16;
    localV = 1;
    break;
  }

  sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV); // 32 x kv_len, batch size
  sycl::range<2> LocalRangeGqa(localH, localV);       // kv_len, x
  sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);
 
  // std::cout << "running sage attn: " << std::endl
  //   << "gqaRatio = " << gqaRatio << ", " << std::endl
  //   << "activationLength = " << activationLength << ", " << std::endl
  //   << "kvSeqLen = " << kvSeqLen << ", " << std::endl
  //   << "groupH = " << groupH << ", " << std::endl
  //   << "groupV = " << groupV << ", " << std::endl;

  // always create this first
  cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            sageAttnGqa12(
              qState, kState, vState, (fp16*)qScale, (fp16*)kScale, (fp16*)vScale, output,
              kvSeqLen, startActivationIdx, activationLength, headQ, headKv, ndi);
              });
        };

  switch (gqaRatio) {
      case 8:
        std::cout << "running gqaRatio 8, not supported " << std::endl;
        return kernel_func;
        break;
      case 12:
        return kernel_func;
        break;
      default:
        std::cout << "running gqaRatio not supported " << std::endl;
        return kernel_func;
        break;
      }
  return kernel_func; 
}

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
  )
{
  uint32_t gqaRatio = headQ / headKv;
  int groupH;
  int groupV;
  int localH;
  int localV;
  switch (gqaRatio) {
  case 12:
    groupH = (longestBatch + 63) / 64;
    groupV = headQ / 4;
    localH = 16;
    localV = 1;
    break;
  default:
    groupH = (longestBatch + 31) / 32;
    groupV = headKv;
    localH = 16;
    localV = 1;
    break;
  }

  sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV); // 32 x kv_len, batch size
  sycl::range<2> LocalRangeGqa(localH, localV);       // kv_len, x
  sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);
 
  // std::cout << "running sage attn paged: " << std::endl
  //   << "gqaRatio = " << gqaRatio << ", " << std::endl
  //   << "groupH = " << groupH << ", " << std::endl
  //   << "groupV = " << groupV << ", " << std::endl;

  // always create this first
  cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            sageAttnGqa12paged(
              qState, kState, vState, (fp16*)qScale, (fp16*)kScale, (fp16*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
              headQ, headKv, ndi);
              });
        };

  switch (gqaRatio) {
      case 8:
        std::cout << "running gqaRatio 8, not supported " << std::endl;
        return kernel_func;
        break;
      case 12:
        return kernel_func;
        break;
      default:
        std::cout << "running gqaRatio not supported " << std::endl;
        return kernel_func;
        break;
      }
  return kernel_func; 

}


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
    float f4
    )
    { 

    uint8_t* qState = p0;
    uint8_t* kState = p1;
    uint8_t* vState = p2;
    uint8_t* qScale = p3;
    uint8_t* kScale = p4;
    uint8_t* vScale = p5;
    uint32_t* cuLengthQ = (uint32_t*)p6;
    uint32_t* cuLengthKv = (uint32_t*)p7;
    uint32_t* blockTable = (uint32_t*)p8; 
    uint8_t* output = p9;

    uint32_t longestBatch = i1;
    uint32_t headQ = i2;
    uint32_t headKv = i3;
    uint32_t headDim = i4;

    //std::cout << "[headDim]: " << headDim << std::endl;    

    cgf_t kernel_func;

    switch(i0){        
        case 8888:
        {
            /*
            torch.ops.torch_ipex.sage_attn_paged(
                q_state, k_state, v_state,
                q_scale, k_scale, v_scale,
                cu_seqlen_q,
                cu_seqlen_k,
                block_tables,
                output,
                longestBatch
            )
            torch.ops.torch_ipex.esimd_kernel_uni(
                q_state, k_state, v_state, q_scale, k_scale, v_scale, cu_seqlen_q, cu_seqlen_k, block_tables, output,
                8888,
                longestBatch,
                q_head,
                kv_head,
                head_dim,
                0, 0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0
                )
            */

            uint32_t gqaRatio = headQ / headKv;
            int groupH;
            int groupV;
            int localH;
            int localV;

            if (headDim == 128)
            {
                switch (gqaRatio) {
                    case 16:
                        groupH = (longestBatch + 63) / 64;
                        groupV = headQ / 4;
                        localH = 16;
                        localV = 1;
                        break;
                    case 12:
                        groupH = (longestBatch + 63) / 64;
                        groupV = headQ / 4;
                        localH = 16;
                        localV = 1;
                        break;
                    case 10:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 2;
                        localH = 16;
                        localV = 1;
                        break;
                    case 8:
                        groupH = (longestBatch + 63) / 64;
                        groupV = headQ / 4;
                        localH = 16;
                        localV = 1;
                        break;
                    case 4:
                        groupH = (longestBatch + 63) / 64;
                        groupV = headQ / 4;
                        localH = 16;
                        localV = 1;
                        break;
                    case 2:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 2;
                        localH = 16;
                        localV = 1;
                        break;
                    default:
                        groupH = (longestBatch + 31) / 32;
                        groupV = headKv;
                        localH = 16;
                        localV = 1;
                        break;
                }

                //std::cout << "[gqaRatio]: " << gqaRatio << std::endl;
                //std::cout << "[groupV]: " << groupV << std::endl;

                sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV); // 32 x kv_len, batch size
                sycl::range<2> LocalRangeGqa(localH, localV);       // kv_len, x
                sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);

                switch (gqaRatio) {
                    case 2:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_2X<1>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 4:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X<1>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 8:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X<2>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 10:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_2X<5>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 12:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X<3>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi);
                                             });
                                           };
                        break;
                    case 16:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X<4>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi);
                                             });
                                           };
                        break;
                    default:
                        break;
                }
            }
            else if (headDim == 64)
            {
                switch (gqaRatio) {
                    case 16:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 4;
                        localH = 32;
                        localV = 1;
                        break;
                    case 12:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 4;
                        localH = 32;
                        localV = 1;
                        break;
                    case 10:
                        groupH = (longestBatch + 255) / 256;
                        groupV = headQ / 2;
                        localH = 32;
                        localV = 1;
                        break;
                    case 8:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 4;
                        localH = 32;
                        localV = 1;
                        break;
                    case 4:
                        groupH = (longestBatch + 127) / 128;
                        groupV = headQ / 4;
                        localH = 32;
                        localV = 1;
                        break;
                    case 2:
                        groupH = (longestBatch + 255) / 256;
                        groupV = headQ / 2;
                        localH = 32;
                        localV = 1;
                        break;
                    default:
                        groupH = (longestBatch + 31) / 32;
                        groupV = headKv;
                        localH = 32;
                        localV = 1;
                        break;
                }

                //std::cout << "[gqaRatio]: " << gqaRatio << std::endl;
                //std::cout << "[groupV]: " << groupV << std::endl;

                sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV); // 32 x kv_len, batch size
                sycl::range<2> LocalRangeGqa(localH, localV);       // kv_len, x
                sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);

                switch (gqaRatio) {
                    case 2:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_2X_64<1>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 4:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X_64<1>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 8:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X_64<2>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 10:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_2X_64<5>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi); });
                                           };
                        break;
                    case 12:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X_64<3>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi);
                                             });
                                           };
                        break;
                    case 16:
                        kernel_func = [=](sycl::handler& cgh) {
                            cgh.parallel_for(qkvMatMatRangeGqa, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                            sageAttnGqa_4X_64<4>(
                            qState, kState, vState, (float*)qScale, (float*)kScale, (float*)vScale, cuLengthQ, cuLengthKv, blockTable, output,
                            headQ, headKv, ndi);
                                             });
                                           };
                        break;
                    default:
                        break;
                }
            }
            break;
        }
        default:
            printf("---------- esimd kernel op not supported, op is %lld ----------\n", i0);
            break;
      }
      return kernel_func;
    }


ESIMD_KERNEL_API cgf_t esimd_scale_dynamic_quant(
  const sycl::half* hidden_states,
  const float* smooth_scale,
  int8_t* quant_tokens,
  float* per_token_scale,
  const int64_t num_tokens,
  const int64_t hidden_size
) {
    constexpr int BS = 256;
    constexpr int MAX_BS_PER_THREAD = 4;
    constexpr int MAX_BN = 32;    // 8192 / 256
    assert(hidden_size % BS == 0);
    assert(hidden_size <= MAX_BS_PER_THREAD * MAX_BN * BS);

    int nb = hidden_size / BS;
    nb = nb > MAX_BN ? MAX_BN : nb;

    sycl::range<2> GlobalRange(num_tokens, nb);
    sycl::range<2> LocalRange(1, nb);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
      cgh.parallel_for(
        Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
          __ESIMD_NS::slm_init(MAX_BN * sizeof(float));

          const int rid = item.get_global_id(0);
          const int bid = item.get_local_id(1);

          const sycl::half * hidden_states_head = hidden_states + (size_t)rid * hidden_size + bid * BS;
          int8_t * output = quant_tokens + (size_t)rid * hidden_size + bid * BS;

          __ESIMD_NS::simd<sycl::half, BS> input = __ESIMD_NS::block_load<sycl::half, BS>(hidden_states_head);
          __ESIMD_NS::simd<float, BS> scales = __ESIMD_NS::block_load<float, BS>(smooth_scale + bid * BS);

          __ESIMD_NS::simd<float, BS> input_smoothed = input * scales;
          __ESIMD_NS::simd<float, BS> input_abs = __ESIMD_NS::abs(input_smoothed);
          float max_value = __ESIMD_NS::hmax<float, float, BS>(input_abs);
          __ESIMD_NS::slm_block_store<float, 1>(bid * sizeof(float), max_value);

          barrier();

          __ESIMD_NS::simd<float, MAX_BN> max_hd_value_full = __ESIMD_NS::slm_block_load<float, MAX_BN>(0);
          float max_value_final = hmax<float, float, MAX_BN>(max_hd_value_full);
          float this_token_scale = max_value_final / 127.0;

          __ESIMD_NS::simd<float, BS> hd_quant = __ESIMD_NS::rnde<float>(input_smoothed / this_token_scale);
          __ESIMD_NS::simd<int8_t, BS> hd_quant_out = hd_quant;

           __ESIMD_NS::block_store<int8_t, BS>(output, hd_quant_out);

          if (bid == 0){
            __ESIMD_NS::block_store<float, 1>(per_token_scale + rid, this_token_scale);
          }
        }
      );
    };

    cgf_t kernel_func_long = [=](sycl::handler& cgh) {
      cgh.parallel_for(
        Range, [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
          __ESIMD_NS::slm_init(MAX_BN * sizeof(float));

          const int rid = item.get_global_id(0);
          const int bid = item.get_local_id(1);

          uint32_t inputOutputOffsetBase = rid * hidden_size + bid * BS;
          uint32_t inOutBoundary = rid * hidden_size + hidden_size;
          uint32_t scaleOffset = bid * BS;

          __ESIMD_NS::simd<sycl::half, BS * MAX_BS_PER_THREAD> input;
          __ESIMD_NS::simd<float, BS * MAX_BS_PER_THREAD> scales;
          __ESIMD_NS::simd<float, BS * MAX_BS_PER_THREAD> input_smoothed;
          float max_value = 0;
          uint32_t inputOffset = inputOutputOffsetBase;
#pragma unroll
          for (int kk = 0; kk < MAX_BS_PER_THREAD; kk++) {
            if (inputOffset < inOutBoundary) {
              input.select<BS, 1>(BS * kk) = __ESIMD_NS::block_load<sycl::half, BS>(hidden_states + inputOffset);
              scales.select<BS, 1>(BS * kk) = __ESIMD_NS::block_load<float, BS>(smooth_scale + scaleOffset);
              input_smoothed.select<BS, 1>(BS * kk) = input.select<BS, 1>(BS * kk) * scales.select<BS, 1>(BS * kk);
              __ESIMD_NS::simd<float, BS> input_abs = __ESIMD_NS::abs<float, BS>(input_smoothed.select<BS, 1>(BS * kk));
              float max_value_temp = __ESIMD_NS::hmax<float, float, BS>(input_abs);
              max_value = max_value_temp > max_value ? max_value_temp : max_value;
              inputOffset += MAX_BN * BS;
              scaleOffset += MAX_BN * BS;
            }
          }

          __ESIMD_NS::slm_block_store<float, 1>(bid * sizeof(float), max_value);

          barrier();

          __ESIMD_NS::simd<float, MAX_BN> max_hd_value_full = __ESIMD_NS::slm_block_load<float, MAX_BN>(0);
          float max_value_final = hmax<float, float, MAX_BN>(max_hd_value_full);
          float this_token_scale = max_value_final / 127.0;
          this_token_scale = 1.0f / this_token_scale;

          uint32_t outputOffset = inputOutputOffsetBase;
#pragma unroll
          for (int kk = 0; kk < MAX_BS_PER_THREAD; kk++) {
            if (outputOffset < inOutBoundary) {
              __ESIMD_NS::simd<float, BS> hd_quant = __ESIMD_NS::rnde<float>(input_smoothed.select<BS, 1>(BS * kk) * this_token_scale);
              __ESIMD_NS::simd<int8_t, BS> hd_quant_out = hd_quant;

              __ESIMD_NS::block_store<int8_t, BS>(quant_tokens + outputOffset, hd_quant_out);
              outputOffset += MAX_BN * BS;
            }
          }

          if (bid == 0){
            __ESIMD_NS::block_store<float, 1>(per_token_scale + rid, this_token_scale);
          }
        }
      );
    };

    if (hidden_size <= MAX_BN * BS) {
      return kernel_func;
    } else {
      return kernel_func_long;
    }
}

#define LARGE_HEAD_KV_STEP 64
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
  )
{
  uint32_t q_head_num_per_kv_head = headQ / headKv;
  uint32_t chunk_num = (max_block_num_per_batch * BLOCK_SIZE + chunk_size - 1) / chunk_size;

  sycl::range<3> GlobalRange(batch_num, chunk_num, headKv * q_head_num_per_kv_head);
  sycl::range<3> LocalRange(1, 1, q_head_num_per_kv_head);   
  sycl::nd_range<3> Range(GlobalRange, LocalRange);

  if (head_dim == 128)
  {  // head_dim == 128 --------------------------------------------------------------------------------------------------

  if (q_head_num_per_kv_head == 2)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 32;
    constexpr int V_SUB_STEP = 32;
    constexpr int QHEAD_PER_THREAD = 1;
    constexpr int SMALL_HEAD_KV_STEP = 64;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 4)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 8)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 4;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 10)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr bool SUM_OPT = true;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 4;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 12)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 3;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 4;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 16)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 4;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<128, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }


  } // head_dim == 128 --------------------------------------------------------------------------------------------------
  else
  { // head_dim == 64 --------------------------------------------------------------------------------------------------

  if (q_head_num_per_kv_head == 2)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 64;
    constexpr int V_SUB_STEP = 64;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 128;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  // else if (q_head_num_per_kv_head == 4 && batch_num <= 10 && max_block_num_per_batch * BLOCK_SIZE <= 8192)
  // {
  //   constexpr bool SIMD32_ALIGN = false;
  //   constexpr int K_SUB_STEP = 64;
  //   constexpr int V_SUB_STEP = 64;
  //   constexpr int QHEAD_PER_THREAD = 2;
  //   constexpr int SMALL_HEAD_KV_STEP = 128;
  //   sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
  //   sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
  //   sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

  //   switch (chunk_size)
  //   {
  //   case 512:
  //   {
  //     cgf_t kernel_func = [=](sycl::handler& cgh) {
  //         cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
  //           sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
  //             qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
  //             output, output_lse, output_max,
  //             headQ, headKv, batch_num, max_block_num_per_batch, ndi);
  //             });
  //       };
  //     return kernel_func; 
  //   }
  //     break;
  //   case 1024:
  //   {
  //     cgf_t kernel_func = [=](sycl::handler& cgh) {
  //         cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
  //           sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
  //             qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
  //             output, output_lse, output_max,
  //             headQ, headKv, batch_num, max_block_num_per_batch, ndi);
  //             });
  //       };
  //     return kernel_func; 
  //   }
  //     break;
  //   case 2048:
  //   {
  //     cgf_t kernel_func = [=](sycl::handler& cgh) {
  //         cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
  //           sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
  //             qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
  //             output, output_lse, output_max,
  //             headQ, headKv, batch_num, max_block_num_per_batch, ndi);
  //             });
  //       };
  //     return kernel_func; 
  //   }
  //     break;
  //   case 4096:
  //   {
  //     cgf_t kernel_func = [=](sycl::handler& cgh) {
  //         cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
  //           sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
  //             qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
  //             output, output_lse, output_max,
  //             headQ, headKv, batch_num, max_block_num_per_batch, ndi);
  //             });
  //       };
  //     return kernel_func; 
  //   }
  //     break;
    
  //   default:
  //     break;
  //   }
  // }
  else if (q_head_num_per_kv_head == 4)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 32;
    constexpr int V_SUB_STEP = 32;
    constexpr int QHEAD_PER_THREAD = 2;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 8)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 32;
    constexpr int V_SUB_STEP = 32;
    constexpr int QHEAD_PER_THREAD = 4;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 10)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr bool SUM_OPT = false;
    constexpr int K_SUB_STEP = 32;
    constexpr int V_SUB_STEP = 32;
    constexpr int QHEAD_PER_THREAD = 5;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 2;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, SUM_OPT, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 12)
  {
    constexpr bool SIMD32_ALIGN = false;
    constexpr int K_SUB_STEP = 16;
    constexpr int V_SUB_STEP = 16;
    constexpr int QHEAD_PER_THREAD = 6;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 2;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }
  else if (q_head_num_per_kv_head == 16)
  {
    constexpr bool SIMD32_ALIGN = false;  // faster vs turned on.
    constexpr int K_SUB_STEP = 32;
    constexpr int V_SUB_STEP = 32;
    constexpr int QHEAD_PER_THREAD = 4;
    constexpr int SMALL_HEAD_KV_STEP = 32;
    constexpr int PREFETCH_THREAD_CNT = 4;
    sycl::range<3> GlobalRangeSmall(batch_num, chunk_num, headKv * q_head_num_per_kv_head / QHEAD_PER_THREAD);
    sycl::range<3> LocalRangeSmall(1, 1, q_head_num_per_kv_head / QHEAD_PER_THREAD);   
    sycl::nd_range<3> RangeSmall(GlobalRangeSmall, LocalRangeSmall);

    switch (chunk_size)
    {
    case 512:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 512, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 1024:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 1024, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 2048:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 2048, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    case 4096:
    {
      cgf_t kernel_func = [=](sycl::handler& cgh) {
          cgh.parallel_for(RangeSmall, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{
            sageAttnDecodePagedNoSLM<64, 4, QHEAD_PER_THREAD, SIMD32_ALIGN, SMALL_HEAD_KV_STEP, K_SUB_STEP,  V_SUB_STEP, 4096, false, PREFETCH_THREAD_CNT>(
              qState, kState, vState, qScale, kScale, vScale, cuLengthQ, cuLengthKv, blockTable, 
              output, output_lse, output_max,
              headQ, headKv, batch_num, max_block_num_per_batch, ndi);
              });
        };
      return kernel_func; 
    }
      break;
    
    default:
      break;
    }
  }

  } // head_dim == 64 --------------------------------------------------------------------------------------------------

}

ESIMD_KERNEL_API cgf_t launch_sage_attn_decode_paged_reduce(
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint8_t* output_final,
    uint32_t headQ,
    uint32_t head_dim,
    uint32_t batch_num,
    uint32_t chunk_count
  )
{
  if (head_dim == 128)
  {
    return launch_sage_attn_decode_paged_reduce_kernel<128>(
      output,
      output_lse,
      output_max,
      output_final,
      headQ,
      batch_num,
      chunk_count);
  }
  else
  {
    return launch_sage_attn_decode_paged_reduce_kernel<64>(
      output,
      output_lse,
      output_max,
      output_final,
      headQ,
      batch_num,
      chunk_count);
  }
  
}

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
  )
{
  uint32_t q_head_num_per_kv_head = headQ / headKv;
  if (head_dim == 128)
  {
    if (q_head_num_per_kv_head == 16)
    {
      switch(chunk_size)
      {
	      case 1024:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 16, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 16, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 16, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else{
            if(headKv==2){
                return sageAttnDecoder<128, 4, 16, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder<128, 4, 16, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder<128, 4, 16, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
	      case 2048:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 16, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 16, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
              return sageAttnDecoder_acc32<128, 4, 16, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else {
          if(headKv==2){
              return sageAttnDecoder<128, 4, 16, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          else if(headKv==4){
              return sageAttnDecoder<128, 4, 16, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          }
          else if(headKv==8){
                  return sageAttnDecoder<128, 4, 16, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
        case 4096:
	        if(acc_fp32){
            if(headKv==2){
                return sageAttnDecoder_acc32<128, 4, 16, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder_acc32<128, 4, 16, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 16, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            }
	        else {
            
            if(headKv==2){
                return sageAttnDecoder<128, 4, 16, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
              else if(headKv==4){
                return sageAttnDecoder<128, 4, 16, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }else if(headKv==8){
                return sageAttnDecoder<128, 4, 16, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
            }
	        break;
	      default:
	      break;
      }
    }
    else if (q_head_num_per_kv_head == 12)
    {

      switch(chunk_size)
      {
        case 512:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 12, 512,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 12, 512,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 12, 512,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else{
            if(headKv==2){
                return sageAttnDecoder<128, 4, 12, 512,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder<128, 4, 12, 512,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder<128, 4, 12, 512,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
	      case 1024:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 12, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 12, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 12, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else{
            if(headKv==2){
                return sageAttnDecoder<128, 4, 12, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder<128, 4, 12, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder<128, 4, 12, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
	      case 2048:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 12, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 12, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
              return sageAttnDecoder_acc32<128, 4, 12, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else {
          if(headKv==2){
              return sageAttnDecoder<128, 4, 12, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          else if(headKv==4){
              return sageAttnDecoder<128, 4, 12, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          }
          else if(headKv==8){
                  return sageAttnDecoder<128, 4, 12, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
        case 4096:
	        if(acc_fp32){
            if(headKv==2){
                return sageAttnDecoder_acc32<128, 4, 12, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder_acc32<128, 4, 12, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 12, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            }
	        else {
            if(headKv==2){
                return sageAttnDecoder<128, 4, 12, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
              else if(headKv==4){
                return sageAttnDecoder<128, 4, 12, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }else if(headKv==8){
                return sageAttnDecoder<128, 4, 12, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
            }
	        break;
	      default:
	      break;
      }
    }
    else if (q_head_num_per_kv_head == 10)
    {
      switch(chunk_size)
      {
	      case 1024:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 10, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 10, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 10, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else{
            if(headKv==2){
                return sageAttnDecoder<128, 4, 10, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder<128, 4, 10, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder<128, 4, 10, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
	      case 2048:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 10, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 10, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
              return sageAttnDecoder_acc32<128, 4, 10, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else {
          if(headKv==2){
              return sageAttnDecoder<128, 4, 10, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          else if(headKv==4){
              return sageAttnDecoder<128, 4, 10, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          }
          else if(headKv==8){
                  return sageAttnDecoder<128, 4, 10, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
        case 4096:
	        if(acc_fp32){
            if(headKv==2){
                return sageAttnDecoder_acc32<128, 4, 10, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder_acc32<128, 4, 10, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 10, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            }
	        else {
            if(headKv==2){
                return sageAttnDecoder<128, 4, 10, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
              else if(headKv==4){
                return sageAttnDecoder<128, 4, 10, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }else if(headKv==8){
                return sageAttnDecoder<128, 4, 10, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
            }
	        break;
	      default:
	        break;
      }
    }
      else if (q_head_num_per_kv_head == 8)
    {
      switch(chunk_size)
      {
	      case 1024:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 8, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 8, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 8, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else{
            if(headKv==2){
                return sageAttnDecoder<128, 4, 8, 1024,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder<128, 4, 8, 1024,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder<128, 4, 8, 1024,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
	      case 2048:
	        if(acc_fp32){
            if(headKv==2){
              return sageAttnDecoder_acc32<128, 4, 8, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
              return sageAttnDecoder_acc32<128, 4, 8, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
              return sageAttnDecoder_acc32<128, 4, 8, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        else {
          if(headKv==2){
              return sageAttnDecoder<128, 4, 8, 2048,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          else if(headKv==4){
              return sageAttnDecoder<128, 4, 8, 2048,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          }
          else if(headKv==8){
                  return sageAttnDecoder<128, 4, 8, 2048,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
          }
	        break;
        case 4096:
	        if(acc_fp32){
            if(headKv==2){
                return sageAttnDecoder_acc32<128, 4, 8, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            else if(headKv==4){
                return sageAttnDecoder_acc32<128, 4, 8, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }else if(headKv==8){
                return sageAttnDecoder_acc32<128, 4, 8, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
            }
            }
	        else {
            if(headKv==2){
                return sageAttnDecoder<128, 4, 8, 4096,2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
              else if(headKv==4){
                return sageAttnDecoder<128, 4, 8, 4096,4>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }else if(headKv==8){
                return sageAttnDecoder<128, 4, 8, 4096,8>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
              }
            }
	        break;
	      default:
	        break;
      }
    }
    else if (q_head_num_per_kv_head == 4)
    {
      switch(chunk_size)
      {
        case 2048:
          return sageAttnDecoder_acc32<128, 4, 4, 2048>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32<128, 4, 4, 4096>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }

  }
  else if(head_dim ==64)
  {
    if (q_head_num_per_kv_head == 16)
    {
    switch(chunk_size)
    {
	  case 2048:
	  return sageAttnDecoder_acc32_dim64<64, 4, 16, 2048>(
      		qState,kState,vState,qScale,kScale,vScale,
      		cuLengthQ,cuLengthKv,blockTable,
      		output,output_lse,output_max,
      		headQ,headKv,max_block_num_per_batch,batch_num);
	  break;
	  case 4096:
	  return sageAttnDecoder_acc32_dim64<64, 4, 16, 4096>(
      		qState,kState,vState,qScale,kScale,vScale,
      		cuLengthQ,cuLengthKv,blockTable,
      		output,output_lse,output_max,
      		headQ,headKv,max_block_num_per_batch,batch_num);
	  break;
	  default:
	  break;
    }
    }
    else if (q_head_num_per_kv_head == 12)
    {
      switch(chunk_size)
      {
        case 2048:
          return sageAttnDecoder_acc32_dim64<64, 4, 12, 2048>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32_dim64<64, 4, 12, 4096>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }
    else if (q_head_num_per_kv_head == 10)
    {
      switch(chunk_size)
      {
        case 2048:
          return sageAttnDecoder_acc32_dim64<64, 4, 10, 2048>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32_dim64<64, 4, 10, 4096>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }
    else if (q_head_num_per_kv_head == 8)
    {
      switch(chunk_size)
      {
        case 1024:
          return sageAttnDecoder_acc32_dim64<64, 4, 8, 1024, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;

        case 2048:
          return sageAttnDecoder_acc32_dim64<64, 4, 8, 2048, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32_dim64<64, 4, 8, 4096, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }
    else if (q_head_num_per_kv_head == 4)
    {
      switch(chunk_size)
      {
        case 2048:
          return sageAttnDecoder_acc32_dim64<64, 4, 4, 2048, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32_dim64<64, 4, 4, 4096, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }

    else if (q_head_num_per_kv_head == 2)
    {
      switch(chunk_size)
      {
        case 2048:
          return sageAttnDecoder_acc32_dim64<64, 4, 2, 2048, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        case 4096:
          return sageAttnDecoder_acc32_dim64<64, 4, 2, 4096, 2>(
                qState,kState,vState,qScale,kScale,vScale,
                cuLengthQ,cuLengthKv,blockTable,
                output,output_lse,output_max,
                headQ,headKv,max_block_num_per_batch,batch_num);
          break;
        default:
          break;
      }
      }
 
  }
  
}

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
  )
{
  if (head_dim == 128)
  {
   if(acc_fp32)
     return sage_attn_decode_paged_reduce_large_acc32<128>(
      output,
      output_lse,
      output_max,
      output_final,
      headQ,
      batch_num,
      chunk_count);
   else
     return sage_attn_decode_paged_reduce_large<128>(
      output,
      output_lse,
      output_max,
      output_final,
      headQ,
      batch_num,
      chunk_count);

  }
  else
  {
    return sage_attn_decode_paged_reduce_large_acc32<64>(
      output,
      output_lse,
      output_max,
      output_final,
      headQ,
      batch_num,
      chunk_count);
  }

}

// ===================================================================================================================

} // namespace esimd
