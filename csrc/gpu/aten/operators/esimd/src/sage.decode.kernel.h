#define TYPE_XVE_SOFTMAX  float
#define TYPE_XVE  float
#define BLOCK_SIZE  512
#define PREFETCH_DEPTH  1
#define XMX_MERGE 2

template<uint32_t HEAD_DIM, uint32_t Q_LEN, uint32_t Q_HEAD_PER_THREAD, bool SIMD32_ALIGN,
uint32_t KV_STEP, uint32_t K_SUB_STEP, uint32_t V_SUB_STEP, uint32_t CHUNK_SIZE, bool SUM_OPT=false, uint32_t PREFETCH_THREAD_CNT=0>
ESIMD_INLINE void sageAttnDecodePagedNoSLM(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  float* qs,
  float* ks,
  float* vs,
  uint32_t* cuSeqQuery,
  uint32_t* cuSeqKv,
  uint32_t* kvCacheBlockTable,
  uint8_t* out,
  uint8_t* out_lse,
  uint8_t* out_max,
  uint32_t headQ,
  uint32_t headKv,
  uint32_t batch_num,
  uint32_t max_block_num_per_batch,
  sycl::nd_item<3>& ndi
  ) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
  constexpr float matMulQuantCoeff = (HEAD_DIM == 128) ? 0.08838834764831844f : 0.125; // 1.0f / sqrt(128.0f) : 1.0f / sqrt(64.0f) ;
  constexpr int num_block_in_chunk = CHUNK_SIZE / BLOCK_SIZE;

  int batch_idx = ndi.get_group(0);
  int chunk_idx = ndi.get_group(1);
  int kv_head_idx = ndi.get_group(2);

  int head_group_idx = ndi.get_local_id(2);
  
  int q_head_num_per_kv_head = headQ / headKv;
  int q_head_idx = kv_head_idx * q_head_num_per_kv_head + head_group_idx * Q_HEAD_PER_THREAD;
  int start_block_idx = chunk_idx * CHUNK_SIZE / BLOCK_SIZE;

  __ESIMD_NS::simd<uint32_t, 2> kv_len_cord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqKv + batch_idx);
  __ESIMD_NS::simd<uint32_t, 2> q_len_cord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqQuery + batch_idx);
  
  int kv_len = kv_len_cord[1] - kv_len_cord[0];
  int q_len = q_len_cord[1] - q_len_cord[0];

  int total_chunk_num_in_batch = (kv_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
  if (chunk_idx >= total_chunk_num_in_batch) return;

  int block_num = num_block_in_chunk;
  int step_num = CHUNK_SIZE / KV_STEP;
  int rest_in_chunk = 0;
  int rest_in_step = 0;

  // last chunk boundary handling
  if ((chunk_idx + 1) * CHUNK_SIZE > kv_len)
  {
    rest_in_chunk = kv_len % CHUNK_SIZE;
    step_num = (rest_in_chunk + KV_STEP - 1) / KV_STEP;
    rest_in_step = kv_len % KV_STEP;
    block_num = (rest_in_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }
  int last_step_idx = step_num - 1;

  __ESIMD_NS::simd<uint32_t, num_block_in_chunk> block_real_idx;
  if (block_num != num_block_in_chunk)
  {
    for (int32_t k = 0; k < block_num; k++) 
    {
      block_real_idx[k] = kvCacheBlockTable[batch_idx * max_block_num_per_batch + start_block_idx + k];
    } 
  }
  else
  {
    block_real_idx = __ESIMD_NS::block_load<uint32_t, num_block_in_chunk>(
      kvCacheBlockTable + batch_idx * max_block_num_per_batch + start_block_idx);
  }

  __ESIMD_NS::simd<int8_t, Q_HEAD_PER_THREAD * HEAD_DIM*Q_LEN> qInput;
  
  // read q and shuffle   (Q_LEN, HEAD_DIM) => (HEAD_DIM/32, Q_LEN, 32)
  #pragma unroll
  for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
  {
    #pragma unroll
    for (int32_t k = 0; k < HEAD_DIM/32; k++)
    {
      qInput.template select<Q_LEN*32, 1>(h*HEAD_DIM*Q_LEN + k * Q_LEN*32) = __ESIMD_ENS::lsc_load_2d<
        int8_t, 32, Q_LEN, 1,  //Q_LENx32
        false, false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((int8_t*)qState,
            headQ * HEAD_DIM - 1, batch_num*Q_LEN - 1, headQ * HEAD_DIM - 1,
            (q_head_idx + h) * HEAD_DIM + k * 32, // X cord
            batch_idx * Q_LEN);  // Y cord
    }
  }

  // read q scales  (Q_LEN)
  // read v scale (HEAD_DIM, 1)
  __ESIMD_NS::simd<uint32_t, 4> gather_offset_qscale(baseOffsetInc4);
  gather_offset_qscale = gather_offset_qscale * headQ * sizeof(float);

  sycl::ext::intel::esimd::simd_mask<4> quantPred = 0;
  #pragma unroll
  for (int32_t k = 0; k < Q_LEN; k++)
  {
    quantPred[k] = 1;
  }
  __ESIMD_NS::simd<float, Q_HEAD_PER_THREAD * 4> qScale;
  #pragma unroll
  for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
  {
    qScale.template select<4, 1>(h*4) = __ESIMD_ENS::lsc_gather<
      float,
      1,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      4,
      uint32_t
      >(qs, gather_offset_qscale + (batch_idx * Q_LEN * headQ + (q_head_idx + h))*sizeof(float), quantPred);
  }

  __ESIMD_NS::simd<float, HEAD_DIM> vScale = __ESIMD_NS::block_load<float, HEAD_DIM>(vs + kv_head_idx * HEAD_DIM);

  __ESIMD_NS::simd<TYPE_XVE_SOFTMAX, Q_HEAD_PER_THREAD * Q_LEN*1> maxKq = -65504.0;
  __ESIMD_NS::simd<TYPE_XVE_SOFTMAX, Q_HEAD_PER_THREAD * Q_LEN*1> old_maxKq = -65504.0;
  __ESIMD_NS::simd<TYPE_XVE_SOFTMAX, Q_HEAD_PER_THREAD * Q_LEN*1> max_correction = -65504.0;
  // (HEAD_DIM/16, Q_LEN, 16)
  __ESIMD_NS::simd<TYPE_XVE, Q_HEAD_PER_THREAD * HEAD_DIM * Q_LEN> pv_acc = 0;
  // (Q_LEN, 16)
  __ESIMD_NS::simd<TYPE_XVE_SOFTMAX, Q_HEAD_PER_THREAD * 16 * Q_LEN> softMaxSumTemp = 0.0000000001;
  __ESIMD_NS::simd<TYPE_XVE_SOFTMAX, 16 * Q_LEN> softMaxSumTempLocal = 0.0000000001;

  __ESIMD_NS::simd<uint32_t, 16> gather_offset(baseOffsetInc16);
  gather_offset = gather_offset * headKv * HEAD_DIM * sizeof(int8_t);
  __ESIMD_NS::simd<uint32_t, 16> gather_offset_kvscale(baseOffsetInc16);
  gather_offset_kvscale = gather_offset_kvscale * headKv * sizeof(float);
  
  int step_last_idx = step_num-1;
  for (int step_idx=0; step_idx < step_num; step_idx++)
  {
    int block_idx = block_real_idx[step_idx * KV_STEP / BLOCK_SIZE];

    if (PREFETCH_THREAD_CNT != 0)
    {
      int step_idx_prefetch = step_idx + PREFETCH_DEPTH;
      if (PREFETCH_THREAD_CNT == 4 && HEAD_DIM == 64)
      {
        if (step_idx_prefetch <= step_last_idx)
        {
          int block_idx_prefetch = block_real_idx[step_idx_prefetch * KV_STEP / BLOCK_SIZE]; 
          int in_block_offset_prefetch = (step_idx_prefetch * KV_STEP) % BLOCK_SIZE;

          // prefetch --------------------------------------------------------
          // assume K_SUB_STEP & V_SUB_STEP is 32 and HEAD_DIM is 64 and PREFETCH_THREAD_CNT = 4

          uint64_t offsetK = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM + kv_head_idx * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/K_SUB_STEP; k++) 
          {
            // prefetch k current  (num_in_sub_step, 4, 16, 32) gather
            int32_t kk = head_group_idx / 2;  // K_SUB_STEP/16
            int32_t kkk = head_group_idx % 2;   // HEAD_DIM/32
            
            __ESIMD_ENS::lsc_prefetch<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)kState, gather_offset + offsetK + (k * K_SUB_STEP * headKv * HEAD_DIM + kk * 16 * headKv * HEAD_DIM + kkk * 32) * sizeof(int8_t));
          }

          int32_t offsetVBase = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/V_SUB_STEP; k++) 
          {
            // prefetch v current  (num_in_sub_step, 8, 16, 16)  2D
            __ESIMD_NS::simd<int8_t, V_SUB_STEP/16*8*16*16> vInput;
            #pragma unroll
            for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
            {
              int32_t kkk = head_group_idx; // HEAD_DIM/16
              __ESIMD_ENS::lsc_prefetch_2d<
                int8_t, 16, 16, 1,  //16x16
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((int8_t*)vState + offsetVBase,
                    headKv * HEAD_DIM - 1, KV_STEP - 1, headKv * HEAD_DIM - 1,
                    kv_head_idx * HEAD_DIM + kkk * 16, // X cord
                    16*(k*V_SUB_STEP/16 + kk));  // Y cord
            }
          }
        } // if (step_idx_prefetch <= step_last_idx)
      }
      else if (HEAD_DIM == 128)
      {
        if (step_idx_prefetch <= step_last_idx && head_group_idx < 4)
        {
          int block_idx_prefetch = block_real_idx[step_idx_prefetch * KV_STEP / BLOCK_SIZE]; 
          int in_block_offset_prefetch = (step_idx_prefetch * KV_STEP) % BLOCK_SIZE;

          // prefetch --------------------------------------------------------
          // assume HEAD_DIM is 128 and PREFETCH_THREAD_CNT = 2 or 4

          uint64_t offsetK = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM + kv_head_idx * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/K_SUB_STEP; k++) 
          {
            // prefetch k current  (num_in_sub_step, 4, 16, 32) gather
            #pragma unroll
            for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
            {
              int32_t kkk = head_group_idx;
              #pragma unroll
              for (int32_t kkkk = 0; kkkk < HEAD_DIM/(32*PREFETCH_THREAD_CNT); kkkk++) 
              {
                __ESIMD_ENS::lsc_prefetch<
                uint32_t,
                8,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)kState, gather_offset + offsetK + (k * K_SUB_STEP * headKv * HEAD_DIM + kk * 16 * headKv * HEAD_DIM + (kkk*HEAD_DIM/(32*PREFETCH_THREAD_CNT) + kkkk) * 32) * sizeof(int8_t));
              }
            }
          }

          int32_t offsetVBase = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/V_SUB_STEP; k++) 
          {
            // prefetch v current  (num_in_sub_step, 8, 16, 16)  2D
            __ESIMD_NS::simd<int8_t, V_SUB_STEP/16*8*16*16> vInput;
            #pragma unroll
            for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
            {
              int32_t kkk = head_group_idx;
              #pragma unroll
              for (int32_t kkkk = 0; kkkk < HEAD_DIM/(16*PREFETCH_THREAD_CNT); kkkk++) 
              {
              __ESIMD_ENS::lsc_prefetch_2d<
                int8_t, 16, 16, 1,  //16x16
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((int8_t*)vState + offsetVBase,
                    headKv * HEAD_DIM - 1, KV_STEP - 1, headKv * HEAD_DIM - 1,
                    kv_head_idx * HEAD_DIM + (kkk*HEAD_DIM/(16*PREFETCH_THREAD_CNT) + kkkk) * 16, // X cord
                    16*(k*V_SUB_STEP/16 + kk));  // Y cord
              }
            }
          }
        } // if (step_idx_prefetch <= step_last_idx)
      }
      else if (PREFETCH_THREAD_CNT == 2 && HEAD_DIM == 64)
      {
        if (step_idx_prefetch <= step_last_idx)
        {
          int block_idx_prefetch = block_real_idx[step_idx_prefetch * KV_STEP / BLOCK_SIZE]; 
          int in_block_offset_prefetch = (step_idx_prefetch * KV_STEP) % BLOCK_SIZE;

          // prefetch --------------------------------------------------------
          // assume HEAD_DIM is 64 and PREFETCH_THREAD_CNT = 2

          uint64_t offsetK = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM + kv_head_idx * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/K_SUB_STEP; k++) 
          {
            // prefetch k current  (num_in_sub_step, 4, 16, 32) gather
            #pragma unroll
            for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
            {
              int32_t kkk = head_group_idx;   // HEAD_DIM/32
            
              __ESIMD_ENS::lsc_prefetch<
              uint32_t,
              8,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              16,
              uint32_t
              >((uint32_t*)kState, gather_offset + offsetK + (k * K_SUB_STEP * headKv * HEAD_DIM + kk * 16 * headKv * HEAD_DIM + kkk * 32) * sizeof(int8_t));
            }
          }

          int32_t offsetVBase = (block_idx_prefetch * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset_prefetch * headKv * HEAD_DIM) * sizeof(int8_t);
          #pragma unroll
          for (int32_t k = 0; k < KV_STEP/V_SUB_STEP; k++) 
          {
            // prefetch v current  (num_in_sub_step, 8, 16, 16)  2D
            __ESIMD_NS::simd<int8_t, V_SUB_STEP/16*8*16*16> vInput;
            #pragma unroll
            for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
            {
              int32_t kkk = head_group_idx;
              #pragma unroll
              for (int32_t kkkk = 0; kkkk < HEAD_DIM/(16*PREFETCH_THREAD_CNT); kkkk++) 
              {
                __ESIMD_ENS::lsc_prefetch_2d<
                  int8_t, 16, 16, 1,  //16x16
                  __ESIMD_ENS::cache_hint::cached,
                  __ESIMD_ENS::cache_hint::cached>((int8_t*)vState + offsetVBase,
                      headKv * HEAD_DIM - 1, KV_STEP - 1, headKv * HEAD_DIM - 1,
                      kv_head_idx * HEAD_DIM + (kkk*HEAD_DIM/(16*PREFETCH_THREAD_CNT) + kkkk) * 16, // X cord
                      16*(k*V_SUB_STEP/16 + kk));  // Y cord
              }
            }
          }
        } // if (step_idx_prefetch <= step_last_idx)
      } 
    } // prefetch --------------------------------------------------------



    int in_block_offset = (step_idx * KV_STEP) % BLOCK_SIZE;
    uint64_t offsetK = (block_idx * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset * headKv * HEAD_DIM + kv_head_idx * HEAD_DIM) * sizeof(int8_t);
    // read k scale (KV_STEP, 1)
    __ESIMD_NS::simd<float, KV_STEP> kScale;

    // kScale.template select<KV_STEP, 1>(0) = 1;

    __ESIMD_NS::simd<TYPE_XVE, Q_HEAD_PER_THREAD * KV_STEP/16*Q_LEN*16> kq_out;
    __ESIMD_NS::simd<fp16, Q_HEAD_PER_THREAD * KV_STEP/16*Q_LEN*16> kq_out_fp16;
    __ESIMD_NS::simd<uint32_t, 16> ksOffset(baseOffsetInc16);
    ksOffset = ksOffset + KV_STEP * step_idx + chunk_idx * CHUNK_SIZE ;
    #pragma unroll
    for (int32_t k = 0; k < KV_STEP/16; k++) 
    {
    
        ksOffset.merge(kv_len-1, ksOffset >= kv_len);
        __ESIMD_NS::simd<uint32_t, 16>  offsets = (ksOffset * headKv  + kv_head_idx + batch_idx * kv_len * headKv )* sizeof(fp32) ;

      kScale.template select<16, 1>(k*16) = __ESIMD_ENS::lsc_gather<
        float,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >(ks,offsets );
        ksOffset+=16;
    }

    #pragma unroll
    for (int32_t k = 0; k < KV_STEP/K_SUB_STEP; k++) 
    {
      // read k current  (num_in_sub_step, 4, 16, 32) gather
      __ESIMD_NS::simd<int8_t, K_SUB_STEP/16*4*16*32> kInput;
      #pragma unroll
      for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
      {
        #pragma unroll
        for (int32_t kkk = 0; kkk < HEAD_DIM/32; kkk++) 
        {
          kInput.template bit_cast_view<uint32_t>().template select<128, 1>((kk*HEAD_DIM/32*16*32 + kkk*16*32)/sizeof(uint32_t)) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)kState, gather_offset + offsetK + (k * K_SUB_STEP * headKv * HEAD_DIM + kk * 16 * headKv * HEAD_DIM + kkk * 32) * sizeof(int8_t));
        }
      }

      __ESIMD_NS::simd<int32_t, Q_HEAD_PER_THREAD * K_SUB_STEP/16*Q_LEN*16> kq_out_int32{0};

      #pragma unroll
      for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
      {
        #pragma unroll
        for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
        {
          __ESIMD_NS::simd<int32_t, Q_LEN * 16> cc_xmx{0};
          cc_xmx = kq_out_int32.template select<Q_LEN*16, 1>(h*K_SUB_STEP/16*Q_LEN*16 + kk*Q_LEN*16);
          #pragma unroll
          for (int32_t kkk = 0; kkk < HEAD_DIM/32; kkk++) 
          {
            // dpas
            __ESIMD_NS::simd<int8_t, Q_LEN * 32> bb_xmx{0};
            __ESIMD_NS::simd<int8_t, 16 * 32> aa_xmx{0};

            bb_xmx = qInput.template select<Q_LEN*32,1>(h*HEAD_DIM*Q_LEN +  + kkk * Q_LEN*32);
            aa_xmx = kInput.template select<16*32, 1>(kk*HEAD_DIM/32*16*32 + kkk*16*32);
            cc_xmx = sycl::ext::intel::esimd::xmx::dpas<8, Q_LEN, int32_t, int32_t, int8_t, int8_t>(cc_xmx, aa_xmx, bb_xmx);
          }
          kq_out_int32.template select<Q_LEN*16, 1>(h*K_SUB_STEP/16*Q_LEN*16 + kk*Q_LEN*16) = cc_xmx.template select<16*Q_LEN,1>(0);
        }
      }
      
      // dequant
      if (SIMD32_ALIGN)
      {
        #pragma unroll
        for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
        {
          #pragma unroll
          for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
          {
            kq_out.template select<16*Q_LEN, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16) =
              kq_out_int32.template select<16*Q_LEN, 1>(h*K_SUB_STEP/16*Q_LEN*16 + kk*Q_LEN*16);
          }
        }

        #pragma unroll
        for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
        {
          #pragma unroll
          for (int32_t kkk = 0; kkk < Q_LEN/2; kkk++) 
          {
            __ESIMD_NS::simd<TYPE_XVE, 2*16> qScaleTemp;
            qScaleTemp.template select<16, 1>(0) = qScale.template select<1, 1>(h*Q_LEN + kkk*2).template replicate<16>();
            qScaleTemp.template select<16, 1>(16) = qScale.template select<1, 1>(h*Q_LEN + kkk*2+1).template replicate<16>();
                      
            #pragma unroll
            for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
            {
              __ESIMD_NS::simd<TYPE_XVE, 2*16> kScaleTemp;
              kScaleTemp.template select<16, 1>(0) = kScale.template select<16, 1>((k * K_SUB_STEP/16 + kk) * 16);
              kScaleTemp.template select<16, 1>(16) = kScale.template select<16, 1>((k * K_SUB_STEP/16 + kk) * 16);

              kq_out.template select<16*2, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16*2) =
                kq_out.template select<16*2, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16*2) * 
                kScaleTemp;
              kq_out.template select<2*16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16*2) =
                kq_out.template select<2*16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16*2) * qScaleTemp;
            }
          }
        }
      }
      else
      {
        #pragma unroll
        for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
        {
          #pragma unroll
          for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++) 
          {
            kq_out.template select<16*Q_LEN, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16) =
              kq_out_int32.template select<16*Q_LEN, 1>(h*K_SUB_STEP/16*Q_LEN*16 + kk*Q_LEN*16);
          }
        }

        #pragma unroll
        for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
        {
          #pragma unroll
          for (int32_t kk = 0; kk < (K_SUB_STEP/16); kk++)
          {
            #pragma unroll
            for (int32_t kkk = 0; kkk < Q_LEN; kkk++)
            {
              kq_out.template select<16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16) =
                kq_out.template select<16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16) *
                kScale.template select<16, 1>((k * K_SUB_STEP/16 + kk) * 16);
              kq_out.template select<16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16) =
                kq_out.template select<16, 1>(h*KV_STEP/16*Q_LEN*16 + (k*K_SUB_STEP/16 + kk)*Q_LEN*16 + kkk*16) * qScale[h*Q_LEN + kkk];
            }
          }
        }
      }
      
    }

    kq_out = kq_out * matMulQuantCoeff;

    // softmax (KV_STEP/16, Q_LEN, 16)
    old_maxKq = maxKq;
    // corner case check and support cacusal decoding 
   // if(((step_idx+1) * KV_STEP+CHUNK_SIZE*chunk_idx)>=(kv_len-3))
    {
        int curPos=step_idx * KV_STEP+CHUNK_SIZE*chunk_idx;
        __ESIMD_NS::simd<uint32_t, 16> simd16_offset(baseOffsetInc16);
         simd_mask<16*Q_LEN> casual_mask;
        #pragma unroll
        for (int32_t k = 0; k < KV_STEP/16; k++){
         // #pragma unroll
          for (int32_t kkk = 0; kkk < Q_LEN; kkk++)
          {
            casual_mask.template select<16, 1>(kkk*16)=(curPos+k*16+simd16_offset)>=(kv_len-3+kkk);
          }
          //#pragma unroll
          for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
          {
           // #pragma unroll
            for (int32_t kkk = 0; kkk < Q_LEN; kkk++)
            {
              kq_out.template select<16, 1>(h*KV_STEP/16*Q_LEN*16 + k*Q_LEN*16 + kkk*16).merge(FP32_MIN,casual_mask.template select<16, 1>(kkk*16));
            }
          }
        }
    }

    #pragma unroll
    for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
    { 
      // get max  (Q_LEN, 1)   8421
      __ESIMD_NS::simd<TYPE_XVE, Q_LEN*16> maxKq_8421 = kq_out.template select<Q_LEN*16, 1>(h*KV_STEP/16*Q_LEN*16 + 0*Q_LEN*16);
      #pragma unroll
      for (int ll = 1; ll < KV_STEP/16; ll++) {
        maxKq_8421 = __ESIMD_NS::max<TYPE_XVE, Q_LEN*16, TYPE_XVE>(kq_out.template select<Q_LEN*16, 1>(h*KV_STEP/16*Q_LEN*16 + ll*Q_LEN*16), maxKq_8421);
      }
      #pragma unroll
      for (int ll = 0; ll < Q_LEN; ll++) {
        maxKq_8421.template select<8, 1>(ll*16) = __ESIMD_NS::max<TYPE_XVE, 8, TYPE_XVE>(maxKq_8421.template select<8, 1>(ll*16+8), maxKq_8421.template select<8, 1>(ll*16));
        maxKq_8421.template select<4, 1>(ll*16) = __ESIMD_NS::max<TYPE_XVE, 4, TYPE_XVE>(maxKq_8421.template select<4, 1>(ll*16+4), maxKq_8421.template select<4, 1>(ll*16));
        maxKq_8421.template select<2, 1>(ll*16) = __ESIMD_NS::max<TYPE_XVE, 2, TYPE_XVE>(maxKq_8421.template select<2, 1>(ll*16+2), maxKq_8421.template select<2, 1>(ll*16));
        maxKq.template select<1, 1>(h*Q_LEN*1 + ll) = std::max(maxKq_8421.template select<1, 1>(ll*16+1), maxKq_8421.template select<1, 1>(ll*16));
      }
    }
    maxKq = __ESIMD_NS::max<TYPE_XVE, Q_HEAD_PER_THREAD * Q_LEN, TYPE_XVE>(maxKq, old_maxKq); 

    #pragma unroll
    for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
    { 
      #pragma unroll
      for (int mm = 0; mm < Q_LEN/2; mm++) {
        __ESIMD_NS::simd<TYPE_XVE, 2*16> maxKq_temp;
        maxKq_temp.template select<16, 1>(0) = maxKq.template select<1, 1>(h*Q_LEN*1 + mm*2).template replicate<16>();
        maxKq_temp.template select<16, 1>(16) = maxKq.template select<1, 1>(h*Q_LEN*1 + mm*2+1).template replicate<16>();
        // minus max  (KV_STEP/16, Q_LEN, 16)
        #pragma unroll
        for (int ll = 0; ll < KV_STEP/16; ll++) {
            kq_out.template select<16*2, 1>(h*KV_STEP/16*Q_LEN*16 + ll*Q_LEN*16 + mm*16*2) =
            kq_out.template select<16*2, 1>(h*KV_STEP/16*Q_LEN*16 + ll*Q_LEN*16 + mm*16*2) - maxKq_temp;
          }
      }
    }

    // exp (KV_STEP/16, Q_LEN, 16)
    kq_out = __ESIMD_NS::exp2<TYPE_XVE, Q_HEAD_PER_THREAD * KV_STEP*Q_LEN, TYPE_XVE>(kq_out * sycl::ext::intel::esimd::detail::log2e);
    // kq_out = pow<TYPE_XVE, Q_HEAD_PER_THREAD * KV_STEP*Q_LEN, TYPE_XVE>(2.718f, kq_out);

    // lse correction (Q_LEN, 1)
    // pv acc correction  (HEAD_DIM/16, Q_LEN, 16)
    if (step_idx >= 1)
    {
      max_correction = old_maxKq - maxKq;
      max_correction = __ESIMD_NS::exp2<TYPE_XVE, Q_HEAD_PER_THREAD * Q_LEN, TYPE_XVE>(max_correction * sycl::ext::intel::esimd::detail::log2e);
      // max_correction = pow<TYPE_XVE, Q_HEAD_PER_THREAD * Q_LEN, TYPE_XVE>(2.718f, max_correction);

      #pragma unroll
      for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
      { 
        if (SIMD32_ALIGN)
        {
          #pragma unroll
          for (int mm = 0; mm < Q_LEN/2; mm++) {
            #pragma unroll
            for (int ll = 0; ll < HEAD_DIM/16; ll++) {
              __ESIMD_NS::simd<TYPE_XVE, 2*16> max_correction_temp;
              max_correction_temp.template select<16, 1>(0) = max_correction.template select<1, 1>(h*Q_LEN*1 + mm*2).template replicate<16>();
              max_correction_temp.template select<16, 1>(16) = max_correction.template select<1, 1>(h*Q_LEN*1 + mm*2+1).template replicate<16>();
              pv_acc.template select<2*16, 1>(h*HEAD_DIM * Q_LEN + ll * Q_LEN*16 + mm * 16*2) = pv_acc.template select<2*16, 1>(h*HEAD_DIM * Q_LEN + ll * Q_LEN*16 + mm * 16*2) * max_correction_temp;
            } 
          }
        }
        else
        {
          #pragma unroll
          for (int ll = 0; ll < HEAD_DIM/16; ll++) {
            #pragma unroll
            for (int mm = 0; mm < Q_LEN; mm++) {
              pv_acc.template select<16, 1>(h*HEAD_DIM * Q_LEN + ll * Q_LEN*16 + mm * 16) = pv_acc.template select<16, 1>(h*HEAD_DIM * Q_LEN + ll * Q_LEN*16 + mm * 16) * max_correction[h*Q_LEN*1 + mm];
            }
          }
        }
        
        if (SUM_OPT)
        {
          softMaxSumTemp.template select<Q_LEN, 1>(h * Q_LEN) = softMaxSumTemp.template select<Q_LEN, 1>(h*Q_LEN) * max_correction.template select<Q_LEN, 1>(h*Q_LEN*1);
        }
        else
        {
          #pragma unroll
          for (int mm = 0; mm < Q_LEN; mm++) {
            softMaxSumTemp.template select<16, 1>(h*16 * Q_LEN + mm * 16) = softMaxSumTemp.template select<16, 1>(h*16 * Q_LEN + mm * 16) * max_correction[h*Q_LEN*1 + mm];
          }
        }
      }
    }

    // lse add (Q_LEN, 1)
    #pragma unroll
    for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
    {
      #pragma unroll
      for (int ll = 0; ll < KV_STEP/16; ll++) {
        if (SUM_OPT)
        {
          softMaxSumTempLocal = kq_out.template select<Q_LEN*16, 1>(h*KV_STEP/16*Q_LEN*16 + ll*Q_LEN*16);
          __ESIMD_NS::simd<TYPE_XVE, Q_LEN> sum_temp;
          #pragma unroll
          for (int qq = 0; qq < Q_LEN; qq++) {
            sum_temp[qq] = sycl::ext::intel::esimd::reduce<TYPE_XVE, TYPE_XVE, 16, std::plus<>>(softMaxSumTempLocal.template select<16, 1>(qq * 16), std::plus<>());
          }
          softMaxSumTemp.template select<Q_LEN, 1>(h * Q_LEN) += sum_temp;
        }
        else
        {
          softMaxSumTemp.template select<Q_LEN*16, 1>(h*16 * Q_LEN) += kq_out.template select<Q_LEN*16, 1>(h*KV_STEP/16*Q_LEN*16 + ll*Q_LEN*16);
        }
      }
    }

    kq_out_fp16 = kq_out;

    int32_t offsetVBase = (block_idx * BLOCK_SIZE * headKv * HEAD_DIM + in_block_offset * headKv * HEAD_DIM) * sizeof(int8_t);
    #pragma unroll
    for (int32_t k = 0; k < KV_STEP/V_SUB_STEP; k++) 
    {
      // read v current  (num_in_sub_step, 8, 16, 16)  2D
      __ESIMD_NS::simd<int8_t, V_SUB_STEP/16*8*16*16> vInput;
      #pragma unroll
      for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
      {
        #pragma unroll
        for (int32_t kkk = 0; kkk < HEAD_DIM/16; kkk++) 
        {
          vInput.template select<16*16, 1>(kk*16*16*HEAD_DIM/16 + kkk*16*16) = __ESIMD_ENS::lsc_load_2d<
            int8_t, 16, 16, 1,  //16x16
            false, false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((int8_t*)vState + offsetVBase,
                headKv * HEAD_DIM - 1, KV_STEP - 1, headKv * HEAD_DIM - 1,
                kv_head_idx * HEAD_DIM + kkk * 16, // X cord
                16*(k*V_SUB_STEP/16 + kk));  // Y cord
        }
      }

      // dequant and shuffle v
      __ESIMD_NS::simd<fp16, V_SUB_STEP/16*8*16*16> vDequanted;
      __ESIMD_NS::simd<fp16, V_SUB_STEP/16*8*16*16> vInputFP16 = vInput;
      // shuffle (V_SUB_STEP/16, 8, 16, 16) => (V_SUB_STEP/16, 8, 8, 32)
      #pragma unroll
      for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
      {
        #pragma unroll
        for (int32_t kkk = 0; kkk < HEAD_DIM/16; kkk++) 
        {
          #pragma unroll
          for (int32_t ss = 0; ss < 8; ss++)
          {
            vDequanted.template select<16, 2>(kk*16*16*HEAD_DIM/16 + kkk*16*16 + ss*2*16) = 
              vInputFP16.template select<16, 1>(kk*16*16*HEAD_DIM/16 + kkk*16*16 + ss*2*16) * vScale.template select<16, 1>(kkk*16);
            vDequanted.template select<16, 2>(kk*16*16*HEAD_DIM/16 + kkk*16*16 + ss*2*16 + 1) = 
              vInputFP16.template select<16, 1>(kk*16*16*HEAD_DIM/16 + kkk*16*16 + ss*2*16 + 16) * vScale.template select<16, 1>(kkk*16);
          }
        }
      }

      #pragma unroll
      for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
      {
        #pragma unroll
        for (int32_t kk = 0; kk < (V_SUB_STEP/16); kk++) 
        {
          #pragma unroll
          for (int32_t kkk = 0; kkk < HEAD_DIM/16; kkk++) 
          {
            // dpas
            __ESIMD_NS::simd<TYPE_XVE, Q_LEN * 16> cc_xmx{0};
            __ESIMD_NS::simd<fp16, Q_LEN * 16> bb_xmx{0};
            __ESIMD_NS::simd<fp16, 16 * 16> aa_xmx{0};

            // pv_acc (8, Q_LEN, 16)
            cc_xmx.template select<16*Q_LEN,1>(0) = pv_acc.template select<16*Q_LEN,1>(h*HEAD_DIM * Q_LEN + kkk*16*Q_LEN);

            // kq_out (KV_STEP/16, Q_LEN, 16)
            bb_xmx= kq_out_fp16.template select<Q_LEN*16,1>(h*KV_STEP/16*Q_LEN*16 + (k*V_SUB_STEP/16 + kk)*Q_LEN*16);
            // vDequanted (num_in_sub_step, 8, 16, 16)
            aa_xmx = vDequanted.template select<16*16, 1>(kk*16*16*HEAD_DIM/16 + kkk*16*16);
            cc_xmx = sycl::ext::intel::esimd::xmx::dpas<8, Q_LEN, TYPE_XVE, TYPE_XVE, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);

            pv_acc.template select<16*Q_LEN,1>(h*HEAD_DIM * Q_LEN + kkk*16*Q_LEN) = cc_xmx.template select<16*Q_LEN,1>(0);
          }
        }
      }
    }
  }

  // do sum8421 for lse (Q_LEN, 16)
  #pragma unroll
  for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
  {
    int grf_h_offset = h*16 * Q_LEN;
    #pragma unroll
    for (int qq = 0; qq < Q_LEN; qq++) {
      if (SUM_OPT)
      {
        // write lse and max  (Q_LEN, 1) (Q_LEN, 1)
        __ESIMD_NS::block_store<TYPE_XVE, 1>((TYPE_XVE*)out_lse + batch_idx*Q_LEN*headQ*total_chunk_num_in_batch +
          chunk_idx*Q_LEN*headQ + (q_head_idx + h) + qq * headQ, softMaxSumTemp[h * Q_LEN + qq]);
        __ESIMD_NS::block_store<TYPE_XVE, 1>((TYPE_XVE*)out_max + batch_idx*Q_LEN*headQ*total_chunk_num_in_batch +
          chunk_idx*Q_LEN*headQ + (q_head_idx + h) + qq * headQ, maxKq[h*Q_LEN*1 + qq]);
      }
      else
      {
        __ESIMD_NS::simd<TYPE_XVE, Q_LEN> sum_temp;
        #pragma unroll
        for (int qq = 0; qq < Q_LEN; qq++) {
          sum_temp[qq] = sycl::ext::intel::esimd::reduce<TYPE_XVE, TYPE_XVE, 16, std::plus<>>(softMaxSumTemp.template select<16, 1>(grf_h_offset + qq * 16), std::plus<>());
        }
        // write lse and max  (Q_LEN, 1) (Q_LEN, 1)
        __ESIMD_NS::block_store<TYPE_XVE, 1>((TYPE_XVE*)out_lse + batch_idx*Q_LEN*headQ*total_chunk_num_in_batch +
          chunk_idx*Q_LEN*headQ + (q_head_idx + h) + qq * headQ, sum_temp[qq]);
        __ESIMD_NS::block_store<TYPE_XVE, 1>((TYPE_XVE*)out_max + batch_idx*Q_LEN*headQ*total_chunk_num_in_batch +
          chunk_idx*Q_LEN*headQ + (q_head_idx + h) + qq * headQ, maxKq[h*Q_LEN*1 + qq]);
      }
    }
  }

  // write out (HEAD_DIM/16, Q_LEN, 16) => (Q_LEN, HEAD_DIM)
  #pragma unroll
  for (int32_t h = 0; h < Q_HEAD_PER_THREAD; h++)
  {
    #pragma unroll
    for (int32_t k = 0; k < HEAD_DIM/16; k++)
    {
      __ESIMD_NS::simd<TYPE_XVE, 16*Q_LEN> qOutput = pv_acc.template select<16*Q_LEN, 1>(h*HEAD_DIM * Q_LEN + k*16*Q_LEN);

      __ESIMD_ENS::lsc_store_2d<
        TYPE_XVE, 16, Q_LEN,  //Q_LENx32
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((TYPE_XVE*)out + batch_idx*total_chunk_num_in_batch*Q_LEN*headQ*HEAD_DIM + chunk_idx*Q_LEN*headQ*HEAD_DIM,
            headQ * HEAD_DIM * sizeof(TYPE_XVE) - 1, Q_LEN - 1, headQ * HEAD_DIM * sizeof(TYPE_XVE) - 1,
            (q_head_idx + h) * HEAD_DIM + k * 16, // X cord
            0, // Y cord
            qOutput);
    }
  }

}

template<uint32_t HEAD_DIM>
ESIMD_INLINE cgf_t launch_sage_attn_decode_paged_reduce_kernel(
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint8_t* output_final,
    uint32_t headQ,
    uint32_t batch_num,
    uint32_t chunk_count
  )
{
  constexpr uint32_t QLEN = 4;
  sycl::range<3> GlobalRange(batch_num, 4, headQ);
  sycl::range<3> LocalRange(1, 1, 1);   
  sycl::nd_range<3> Range(GlobalRange, LocalRange);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(
      Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {

        
        int batch_idx = ndi.get_group(0);
        int q_idx = ndi.get_group(1);
        int q_head_idx = ndi.get_group(2);
        int32_t offset_output_reduce = batch_idx * chunk_count * QLEN * headQ * HEAD_DIM
          + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
        int32_t offset_output_reduce_final = batch_idx * 1 * QLEN * headQ * HEAD_DIM
          + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
        int32_t offset_output_lse_max = batch_idx * chunk_count * QLEN * headQ
          + q_idx * headQ + q_head_idx;

        __ESIMD_NS::simd<TYPE_XVE, 1> max_final = -65504.0;
        __ESIMD_NS::simd<TYPE_XVE, HEAD_DIM> reduce_final = 0;
        __ESIMD_NS::simd<TYPE_XVE, 1> lse_final = 0;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
        {
          int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

          __ESIMD_NS::simd<TYPE_XVE, 1> cur_max = __ESIMD_NS::block_load<TYPE_XVE, 1>((TYPE_XVE*)output_max + offset_output_lse_max_ck);
          max_final = __ESIMD_NS::max<TYPE_XVE, 1, TYPE_XVE>(cur_max, max_final);
        }
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
        {
          int32_t offset_output_reduce_ck = offset_output_reduce + chunk_idx * QLEN * headQ * HEAD_DIM;
          int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

          __ESIMD_NS::simd<TYPE_XVE, 1> cur_max = __ESIMD_NS::block_load<TYPE_XVE, 1>((TYPE_XVE*)output_max + offset_output_lse_max_ck);
          __ESIMD_NS::simd<TYPE_XVE, 1> cur_lse = __ESIMD_NS::block_load<TYPE_XVE, 1>((TYPE_XVE*)output_lse + offset_output_lse_max_ck);
          __ESIMD_NS::simd<TYPE_XVE, HEAD_DIM> cur_reduce = __ESIMD_NS::block_load<TYPE_XVE, HEAD_DIM>((TYPE_XVE*)output + offset_output_reduce_ck);
          
          __ESIMD_NS::simd<TYPE_XVE, 1> correction = pow<TYPE_XVE, 1, TYPE_XVE>(2.718f, cur_max - max_final);

          lse_final = lse_final + cur_lse * correction;
          
          cur_reduce.template select<HEAD_DIM, 1>(0) = cur_reduce.template select<HEAD_DIM, 1>(0) * correction[0];
          reduce_final = reduce_final + cur_reduce;
        }
  
        reduce_final.template select<HEAD_DIM, 1>(0) = reduce_final.template select<HEAD_DIM, 1>(0) / lse_final[0];
        __ESIMD_NS::block_store<fp16, HEAD_DIM>((fp16*)output_final + offset_output_reduce_final, reduce_final);
      });
  };
 sycl::range<3> GlobalRange1(batch_num, QLEN, headQ / 16);
    sycl::range<3> LocalRange1(1, 1, 1);
    sycl::nd_range<3> Range1(GlobalRange1, LocalRange1);
    cgf_t kernel_func1 = [=](sycl::handler& cgh) {
        cgh.parallel_for(
          Range1, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{

         
            int batch_idx = ndi.get_group(0);
            int q_idx = ndi.get_group(1);
            int q_head_idx = ndi.get_group(2) << 4;

            int32_t offset_output_reduce = batch_idx * chunk_count * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_reduce_final = batch_idx * 1 * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_lse_max = batch_idx * chunk_count * QLEN * headQ
              + q_idx * headQ + q_head_idx;

            __ESIMD_NS::simd<fp32, 16> max_final = FP32_MIN;
            __ESIMD_NS::simd<fp32, HEAD_DIM * 16> reduce_final = 0;
            __ESIMD_NS::simd<fp32, HEAD_DIM * 16> cur_reduce;
            __ESIMD_NS::simd<fp32, 16> lse_final = 0;

            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<fp32, 16> cur_max = __ESIMD_NS::block_load<fp32, 16>((fp32*)output_max + offset_output_lse_max_ck);
              max_final = __ESIMD_NS::max<fp32, 16, fp32>(cur_max, max_final);
            }
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_reduce_ck = offset_output_reduce + chunk_idx * QLEN * headQ * HEAD_DIM;
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<fp32, 16> cur_max = __ESIMD_NS::block_load<fp32, 16>((fp32*)output_max + offset_output_lse_max_ck);
              __ESIMD_NS::simd<fp32, 16> cur_lse = __ESIMD_NS::block_load<fp32, 16>((fp32*)output_lse + offset_output_lse_max_ck);
              __ESIMD_NS::simd<fp32, 16> correction = __ESIMD_NS::pow<fp32, 16, fp32>(2.718f,(cur_max - max_final));
              lse_final = lse_final + cur_lse * correction;

                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                  cur_reduce.template select<HEAD_DIM,1>(HEAD_DIM * j) = __ESIMD_NS::block_load<fp32, HEAD_DIM>((fp32*)output + offset_output_reduce_ck + j * HEAD_DIM);
                  reduce_final.template select<HEAD_DIM,1>(HEAD_DIM *  j) = cur_reduce.template select<HEAD_DIM,1>(HEAD_DIM * j) * correction[ j] + reduce_final.template select<HEAD_DIM,1>(HEAD_DIM * j);
                }
              
            }
            #pragma unroll
            for (int i = 0; i < 16 ; ++i) {
              reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM) = reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM) / lse_final[i];
              __ESIMD_NS::block_store<fp16, HEAD_DIM>((fp16*)output_final + offset_output_reduce_final + i * HEAD_DIM, reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM));
            }
          });
        };
    if ( (headQ &15) ==0) {
        return kernel_func1;
    }
  return kernel_func; 
}