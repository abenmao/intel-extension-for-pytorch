#define MERGE_QUANT
#define GS_Q_LEN 4  //q_len
#define QKV_INT8_DEC
#define KV_LOOP_STEP 64

#define XMX_M           8  //  RepeatCount
#define XMX_N           16 //  ExecutionSize
#define XMX_K           32 //  SystolicDepth * OperationsPerChannel


using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
//using fp16 = sycl::ext::oneapi::bfloat16;
using fp32 = float;
using REDUCE_TYPE = fp16;
using namespace sycl;
template <uint32_t HEAD_DIM, uint32_t q_len, uint32_t gqaRatio, uint32_t kvChunkSize, uint32_t HEAD_KV=8>
ESIMD_INLINE cgf_t sageAttnDecoder(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* qs,
  uint8_t* ks,
  uint8_t* vs,
  uint32_t* cuSeqQuery,
  uint32_t* cuSeqKv,
  uint32_t* kvCacheBlockTable,
  uint8_t* out,
  uint8_t* out_lse,
  uint8_t* out_max,
  uint32_t headQ,
  uint32_t headKv,
  uint32_t kvCacheBlockPerBatch,  //how many blocks in the batch, max?
  uint32_t batch_size
  ) {
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;
    uint32_t maxKvChunkNum = (kvCacheBlockPerBatch * blockSize + kvChunkSize - 1) / kvChunkSize;
    constexpr uint32_t numBlocksInKvChunk = (kvChunkSize + blockSize - 1) / blockSize;
    int groupH;
    int groupV;
    int localH;
    int localV;

    groupH =  batch_size;
    groupV =  maxKvChunkNum; 
    localH = GS_Q_LEN;
    localV = headKv;

    sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV); 
    sycl::range<2> LocalRangeGqa(localH, localV);   
    sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(qkvMatMatRangeGqa, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{

   	float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  //----------------- kernel start ----------------
    if constexpr (HEAD_DIM == 64) {
        matMulQuantCoeff = 0.125f;
    }


  constexpr uint32_t slmSizeK = 0 ;
#ifdef QKV_INT8_DEC
  constexpr uint32_t slmSizeV = KV_LOOP_STEP * HEAD_DIM * sizeof(int8_t) * HEAD_KV;
#else
  constexpr uint32_t slmSizeV = KV_LOOP_STEP * HEAD_DIM * sizeof(fp16)* HEAD_KV;
#endif

  constexpr uint32_t slmSizeQ = 16 * HEAD_DIM * GS_Q_LEN * sizeof(int8_t)* HEAD_KV;

  constexpr uint32_t slmSize = slmSizeV + slmSizeQ;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  named_barrier_init<1+HEAD_KV>();
  constexpr uint32_t slmOffsetBaseV = 0;
  constexpr uint32_t slmOffsetBaseQ = slmOffsetBaseV + slmSizeV;


  constexpr uint32_t outloop = (HEAD_DIM / 8) / GS_Q_LEN;
  constexpr uint32_t boundaryQ = gqaRatio - 1;
  float MIN_VALUE = FP32_MIN * matMulQuantCoeff; 


  int32_t localLinearId = ndi.get_local_id(0); //q 0 ,1,2,3
  //int32_t hhq = localLinearId & 0x3;
  //int32_t vvq = localLinearId >> 2;
  int32_t vvq = localLinearId;
  int32_t hhv = localLinearId;
  //int32_t h =  ndi.get_group(1);
  int32_t batchIdx = ndi.get_group(0); // kv headers * batch
  int32_t kvChunkIdx = ndi.get_group(1);
    
  //int32_t headIdxKv = v;

  int32_t headIdxKv =ndi.get_local_id(1);
  int32_t headIdxQ = gqaRatio * headIdxKv;


  __ESIMD_NS::simd<uint32_t, 2> kvCoord;

  __ESIMD_NS::simd<int8_t, 16 * HEAD_DIM> i8QState;
  __ESIMD_NS::simd<fp32, 32> fp32Qs;   //to simd 32
  __ESIMD_NS::simd<int32_t, 16 * KV_LOOP_STEP> i32TempBuffer;   //xmx C result
//  auto qkTemp = i32TempBuffer.select<16 * KV_LOOP_STEP, 1>(0);
  //__ESIMD_NS::simd<float, 512> tempBuffer;  //512*4 int8_t
  //auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
  //auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 3);
  //auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512*1);
  __ESIMD_NS::simd<int8_t, KV_LOOP_STEP * 8> i8VState;

  //auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
  __ESIMD_NS::simd<int8_t, 512*4> i8TempBuffer; //to load K and V from SLM
  //auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  __ESIMD_NS::simd<fp16, 16 * HEAD_DIM> finalOutput = 0;
  __ESIMD_NS::simd<fp32, HEAD_DIM> fp32Qv;

  __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
  //simd<uint32_t, 16> blockIdx;
  uint32_t batchStartKv;
  uint32_t batchEndKv;

  batchStartKv = cuSeqKv[batchIdx];
  batchEndKv = cuSeqKv[batchIdx+1];
  int32_t totalKvSeqLen = batchEndKv - batchStartKv - 3 + localLinearId;
  //int32_t totalKvSeqLen = batchEndKv - batchStartKv;   //for attn mask

  uint32_t kvChunkNum = (totalKvSeqLen + kvChunkSize - 1) / kvChunkSize;
  if (kvChunkIdx >= kvChunkNum)
      return;

  uint32_t blockIdxStart = kvChunkIdx * kvChunkSize / blockSize;
  uint32_t block_num = numBlocksInKvChunk;
  uint32_t kVloopNum = (kvChunkSize + KV_LOOP_STEP - 1) / KV_LOOP_STEP;
  uint32_t kvSeqLen = kvChunkSize;
  uint32_t block2dHeightBoundary = 63;
  if ((kvChunkIdx + 1) * kvChunkSize > totalKvSeqLen)
  {
      uint32_t restInKvChunk = totalKvSeqLen % kvChunkSize;
      kVloopNum = (restInKvChunk + KV_LOOP_STEP -1) / KV_LOOP_STEP;
      //block2dHeightBoundary= (restInKvChunk - 3 + localLinearId) &(KV_LOOP_STEP-1);
      block2dHeightBoundary= restInKvChunk &(KV_LOOP_STEP-1);
      block2dHeightBoundary=(block2dHeightBoundary==0)?63:block2dHeightBoundary;
      block_num = (restInKvChunk + blockSize - 1) / blockSize;
      kvSeqLen = totalKvSeqLen - kvChunkIdx * kvChunkSize;
  }
  __ESIMD_NS::simd<uint32_t, numBlocksInKvChunk> blockIdx;

  // the last not aligned kv chunk
  if (block_num != numBlocksInKvChunk)
  {
      for (int32_t k = 0; k < block_num; k++)
      {
          blockIdx[k] = kvCacheBlockTable[batchIdx * kvCacheBlockPerBatch + blockIdxStart + k];
      }
  }
  else
  {
      blockIdx = __ESIMD_NS::block_load<uint32_t, numBlocksInKvChunk>(
        kvCacheBlockTable + batchIdx * kvCacheBlockPerBatch + blockIdxStart);
  }

  unsigned int offsetQBase =  headIdxQ * HEAD_DIM + vvq * headQ * HEAD_DIM/*token*/ + batchIdx *q_len*headQ * HEAD_DIM/*batch*/;
  unsigned int kvHiddenDim = headKv * HEAD_DIM;
  unsigned int offsetBaseK = (headIdxKv * HEAD_DIM + localLinearId * outloop * headKv * HEAD_DIM) * sizeof(int8_t);
  unsigned int offsetK;

#ifdef QKV_INT8_DEC
  unsigned int slmOffsetV = slmOffsetBaseV + hhv * outloop * 256 * sizeof(int8_t) + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP;
#else
  unsigned int slmOffsetV = slmOffsetBaseV + hhv * outloop * HEAD_DIM * sizeof(fp16)+ndi.get_local_id(1)*HEAD_DIM* KV_LOOP_STEP *2;
#endif

  unsigned int slmOffsetQ = slmOffsetBaseQ + (headIdxKv * GS_Q_LEN + localLinearId) * 16 * HEAD_DIM * sizeof(int8_t);

  const fp32* qscale_head = (const fp32*)qs + batchIdx * q_len * headQ + vvq * headQ + headIdxQ;

  //16 headers, 1 token
  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * HEAD_DIM + offsetQBase;
#pragma unroll
  for (int k = 0; k < HEAD_DIM / 32; k++) {
    //512
    i8QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * k) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)qState, simdOffsets);

    simdOffsets += 8 * sizeof(uint32_t);
  }

  //[q_len, q_head_num, 1]
  fp32Qs.select<gqaRatio, 1>(0) = block_load<fp32, gqaRatio>(qscale_head);
  fp32Qs.select<16, 1>(16) = fp32Qs.select<16, 1>(0);

  __ESIMD_NS::simd<float, 16> fp32SoftMaxTemp = 0.0f;
  __ESIMD_NS::simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
  __ESIMD_NS::simd<float, 16> fp32CurrentMaxTemp;
  __ESIMD_NS::simd<float, 16> fp32SoftMaxCompensation;

  //blockIdx = __ESIMD_ENS::lsc_block_load<
  //  uint32_t,
  //  16,
  //  __ESIMD_ENS::lsc_data_size::default_size,
  //  __ESIMD_ENS::cache_hint::cached,
  //  __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + batchIdx * kvCacheBlockPerBatch);

  //temp WA
  //blockIdx = blockIdx - batchIdx * kvCacheBlockPerBatch;
#ifdef QKV_INT8_DEC

#pragma unroll
  for (int k = 0; k < HEAD_DIM / 64; k++) {
      fp32Qv.template select<64, 1>(64*k) =
          __ESIMD_ENS::lsc_block_load<
          fp32,
          64,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((fp32*)vs + HEAD_DIM * headIdxKv + 64 *k);
  }

  
 fp32Qv = fp32Qv * (1.0f / 255.0f);
#else
  //[1, kv_head_num, kv_head_dim]
  //only this thread data, 8 
//  fp16TempBuffer.select<HEAD_DIM / GS, 1>(0) =
//    __ESIMD_ENS::lsc_block_load<
//    fp16,
//    HEAD_DIM/GS,
//    __ESIMD_ENS::lsc_data_size::default_size,
//    __ESIMD_ENS::cache_hint::cached,
//    __ESIMD_ENS::cache_hint::cached>((fp16*)vs + HEAD_DIM * headIdxKv + 8 * hhv * outloop);
//
//#pragma unroll
//  for (int kk = 0; kk < 4; kk++) {
//    //fp32Qv.select<32, 1>(32 * kk) = fp16TempBuffer.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
//      fp32Qv.select<32, 1>(32 * kk) = fp16TempBuffer.select<HEAD_DIM / GS, 1>(0);
//  }
#endif

#pragma unroll
  for (int32_t kk = 0; kk < HEAD_DIM / 32; kk++) {
      slm_block_store<uint32_t, 128>(slmOffsetQ + kk * 128 * sizeof(uint32_t), i8QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * kk));
  }

  __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);

  for (int loopIdx = 0; loopIdx < kVloopNum; loopIdx++)
  { 
      //which table[][?]    0x7F  , 8K/KV_BLOCK = 128,  KV_BLOCK * 8 = 512
      uint32_t whichIdx = loopIdx;// &blockTableReloadCheckMask; //change to next 8K, block_table[next][]
      //start paged block
      //uint32_t macroBlock = kvCacheBlockTable[batchIdx * kvCacheBlockPerBatch + (whichIdx >> loopsPerMacroBlockShift)] * blockSize;
      uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;

      //which KV_BLOCK,  0x7   512/KV_BLOCK = 8
      uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * KV_LOOP_STEP;
      subBlock = macroBlock + subBlock; //real row
      offsetK = subBlock * kvHiddenDim + offsetBaseK;

      uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;

      unsigned int offsetK_outloop = offsetK;
      // Load KS
      simd<fp32, KV_LOOP_STEP> ksTemp;

      simd<uint32_t, 16> ksOffset;
      ksOffset = baseOffsetInc16AsVector + KV_LOOP_STEP * loopIdx + kvChunkIdx * kvChunkSize;
      ksOffset.merge(0, ksOffset >= totalKvSeqLen);
      ksOffset = ksOffset * headKv * sizeof(fp32) + headIdxKv * sizeof(fp32);
      ksOffset += batchStartKv * headKv * sizeof(fp32)/*batch*/;

#pragma unroll
      for (int32_t n = 0; n < KV_LOOP_STEP/16; n++)
      {
        ksTemp.select<16,1>(16*n) =
        __ESIMD_ENS::lsc_gather<
        fp32,
        1,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((fp32*)ks, ksOffset);

        ksOffset += 16 * headKv * sizeof(fp32);
      }

    auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * KV_LOOP_STEP, 1>(0);
    auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();  //quant and VNNI to INT8
    auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();  //VNNI to fp16

    //barrier();

    //next 8K
    //if (whichIdx == blockTableReloadCheckMask && loopIdx < kvSeqOutLoopCount-1) {
    //    uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
    //    blockIdx =
    //        __ESIMD_ENS::lsc_block_load<
    //        uint32_t,
    //        16,
    //        __ESIMD_ENS::lsc_data_size::default_size,
    //        __ESIMD_ENS::cache_hint::cached,
    //        __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable + batchIdx * kvCacheBlockPerBatch);
    //    //blockIdx = blockIdx - batchIdx * kvCacheBlockPerBatch;
    //}

    i32TempBuffer = 0;
    //1 group 4*16*128, 1 thread 1*16*128

    //XMX1
    {
        //simd<fp16, 64> ksAllTemp = slm_block_load<fp16, 64>(slmOffsetBaseKs);

        offsetK_outloop =  subBlock * kvHiddenDim;
        auto ksAllTemp=ksTemp;

#pragma unroll
        for (int32_t nn = 0; nn < KV_LOOP_STEP / 16; nn++) {
//#pragma unroll
//            for (int32_t ll = 0; ll < HEAD_DIM / 32; ll++) {
//
//                i8TempBuffer.select<512, 1>(512 * ll) = __ESIMD_ENS::lsc_load_2d<
//                    uint8_t, 32, 16, 1,  //16x32
//                    false, false,
//                    __ESIMD_ENS::cache_hint::cached,
//                    __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK_outloop,
//                    headKv * HEAD_DIM - 1, totalKvSeqLen - 1, headKv * HEAD_DIM - 1,
//                    ll * 32 + headIdxKv * HEAD_DIM, nn * 16);
//
//            }

#pragma unroll
            for (int32_t l = 0; l < HEAD_DIM / (XMX_K); l++) {
#pragma unroll
                //4 xmx [8*32]
                for (int32_t ll = 0; ll < 1; ll++) {

                    i8TempBuffer.select<512, 1>(512* ll) = __ESIMD_ENS::lsc_load_2d<
                    uint8_t, 32, 16, 1,  //16x32
                    false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK_outloop,
                    headKv * HEAD_DIM - 1, totalKvSeqLen - 1, headKv * HEAD_DIM - 1,
                        l* 32+headIdxKv * HEAD_DIM,  nn * 16);

                    }
#pragma unroll
                for (int32_t ll = 0; ll < 1; ll++) {
                    i8TempBuffer.select<512, 1>(512 + 512 * ll) = slm_block_load<int8_t, 512>(
                      slmOffsetQ +
                      512 * l * sizeof(int8_t));
                }

#pragma unroll
                //4 xmx
                for (int32_t k = 0; k < 1; k++) {
#pragma unroll
                    for (int32_t kk = 0; kk < 2; kk++) {
                        auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
                        auto aaTile = i8TempBuffer.template bit_cast_view<int32_t>().select<128, 1>(128 + 128 * k);
                        auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk);

                        ccTile =
                            dpas
                            <8, 8, int32_t, int32_t, int32_t, int32_t,
                            dpas_argument_type::s8,
                            dpas_argument_type::s8
                            >(
                                __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                                __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                                __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                            );
                    }
                }
            }
#pragma unroll
            //Q&K dequant,  2 rows (16 kv) xmx acc
            for (int32_t kk = 0; kk < 8; kk++) {   //16*16
                __ESIMD_NS::simd<float, 32> fp32Temp0;
                __ESIMD_NS::simd<float, 32> fp32Temp1;
                fp32Temp0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk) * fp32Qs;
                fp32Temp1 = i32TempBuffer.select<32, 1>(256 * nn + 32 * kk);
#ifdef MERGE_QUANT
                tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1 * matMulQuantCoeff;
#else
                tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1
#endif
            }

        }

#if 1
	#pragma unroll
    for (int kk = 0; kk < KV_LOOP_STEP/16; kk++) {
      __ESIMD_ENS::lsc_prefetch<
        uint8_t,
        HEAD_DIM,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
    }
#endif
        //if (whichIdx == blockTableReloadCheckMask) {
        //    offsetK = blockIdx[0] * kvHiddenDim * blockSize + offsetBaseK;
        //}

    }


/*	    if(batchIdx == 0 && kvChunkIdx == 0 && headIdxKv == 0)
                {

                    float ttt1 = tempOutput[0];
                    float ttt2 = tempOutput[1];
                    float ttt3 = tempOutput[2];
                    float ttt4 = tempOutput[3];

                    sycl::ext::oneapi::experimental::printf("thread_idx %d tempOutput[0] %f %f %f %f \n",localLinearId,  ttt1, ttt2, ttt3, ttt4);
                }
 */
    //Load V
#ifdef QKV_INT8_DEC
    __ESIMD_NS::simd<uint32_t, KV_LOOP_STEP> simd32Offsets00;
    //simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
    //simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
    //simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + KV_LOOP_STEP * loopIdx;

    //simd<uint32_t, 32> simd32Offsets00_outloop = simd32Offsets00.select<32, 1>(0);

#pragma unroll
    for (int n = 0; n < outloop; n++)
    {
        //simd32Offsets00.select<32, 1>(0) = simd32Offsets00_outloop;
#pragma unroll
        //64*8=512
        for (int kk = 0; kk < KV_LOOP_STEP/32; kk++) {
            simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
            simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
            //simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * kk;
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * kk + subBlock;
            //simd_mask<32> mask;
            //mask.merge(1, 0, simd32Offsets00.select<32, 1>(32) < (kvSeqLen + 3 - localLinearId)); //WA, TODO

            //simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
            //simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim +
            //    subBlock * kvHiddenDim + (headIdxKv * HEAD_DIM) + hhv * outloop * 8 + n * 8;

            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim +
                (headIdxKv * HEAD_DIM) + hhv * outloop * 8 + n * 8;
            //32*8, always block_size algined
            i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                32,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32) * sizeof(int8_t));// mask);

            //simd32Offsets00.select<32, 1>(0) += 32;
        }


#pragma unroll
        //write 512 to SLM, 64*8
        for (int k = 0; k < KV_LOOP_STEP*8/256; k++) {
        __ESIMD_NS::simd<int8_t, 256> shuffleTemp;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                //128 =  32 * sizeof(uint32_t) not head_dim
                shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
            }
//	    if(batchIdx == 0 && kvChunkIdx == 0 && headIdxKv == 0)
//	       {
//    uint32_t  t1 = slmOffsetV + n * 256 * sizeof(int8_t) + k * HEAD_DIM * XMX_K * sizeof(int8_t);


 //   sycl::ext::oneapi::experimental::printf("batchIdx %d, chunk_idx %d kv_idx %d,thread_idx %d, outloop %d  n %d Vslmoffset %d  \n", batchIdx, kvChunkIdx,headIdxKv, localLinearId, outloop, n, t1);
 //   }
            slm_block_store<int32_t, 64>(slmOffsetV + n * 256 * sizeof(int8_t) + k * HEAD_DIM * XMX_K * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
        }

    }
//    if(loopIdx > 64)
//			    return;
//
#else
    __ESIMD_NS::simd<uint32_t,32> simd32Offsets00;
    simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
    simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + KV_LOOP_STEP * loopIdx;

    __ESIMD_NS::simd<uint32_t, 16> simd32Offsets00_outloop = simd32Offsets00.select<16, 1>(0);
#pragma unroll
    for (int n = 0; n < outloop; n++)
    {
        simd32Offsets00.select<16, 1>(0) = simd32Offsets00_outloop;
        //to do 64*8
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk;
            simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<16, 1>(16) = 
          simd32Offsets00.select<16, 1>(16) * kvHiddenDim  + subBlock * kvHiddenDim  + 
                (headIdxKv * HEAD_DIM) + hhv * outloop *  8  + n * 8;

            //16*8
            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
        >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16) * sizeof(int8_t));
            simd32Offsets00.select<16, 1>(0) += 16;
        }
          //Load vs
        fp16TempBuffer.select<8, 1>(0) =
            __ESIMD_ENS::lsc_block_load<
            fp16,
            8,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((fp16*)vs + HEAD_DIM * headIdxKv + 8 * hhv * outloop + n * 8);

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            fp32Qv.select<32, 1>(32 * kk) = fp16TempBuffer.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
        }
#pragma unroll
        for (int k = 0; k < HEAD_DIM/32; k++) {
            __ESIMD_NS::simd<fp16, 128> shuffleTemp;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                shuffleTemp.select<16, 1>(64 * kk + 0) = i8VState.select<16, 4>(128 * k + 64 * kk + 0);
                shuffleTemp.select<16, 1>(64 * kk + 16 * 1) = i8VState.select<16, 4>(128 * k + 64 * kk + 1);
                shuffleTemp.select<16, 1>(64 * kk + 16 * 2) = i8VState.select<16, 4>(128 * k + 64 * kk + 2);
                shuffleTemp.select<16, 1>(64 * kk + 16 * 3) = i8VState.select<16, 4>(128 * k + 64 * kk + 3);
            }

#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp32Qv.select<32, 1>(32 * kk);
            }
            slm_block_store<fp16, 128>(slmOffsetV + n * 128 * sizeof(fp16) + k * HEAD_DIM * 16 * sizeof(fp16), shuffleTemp);
        }
    }
#endif

    //barrier();

	//soft max
    {
        //   auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
        //   auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);

        //auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
        //auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
        //auto softmaxPositions = ui32Temp.select<KV_LOOP_STEP, 1>(KV_LOOP_STEP);

        //  fp32SoftMaxTemp = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
        //  fp32HistoricMaxTemp = slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
    //TODO add attn mask here
#if  VERSION_0

//        softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * KV_LOOP_STEP;
//#pragma unroll
//        for (int k = 0; k < 4; k++) {
//#pragma unroll
//            for (int kk = 0; kk < 16; kk++) {
//                tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
//                tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
//            }
//            softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
//        }


#else
//            if (loopIdx >= myLastFullAttnLoopIdx) {
//                softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
//                softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
//                softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
//#pragma unroll
//                for (int kk = 0; kk < 64; kk++) {
//                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
//                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
//                }
//            }
#endif
        if((block2dHeightBoundary < 63) &&  (loopIdx+1)==kVloopNum)
        {
#pragma unroll    
            for (int k = 0; k < KV_LOOP_STEP; k++) {
                tempOutput.select<16, 1>(16 * k).merge(FP32_MIN, k >= block2dHeightBoundary);
            }
        }

#ifndef MERGE_QUANT
        tempOutput.select<16 * KV_LOOP_STEP, 1>(0) = tempOutput.select<16 * KV_LOOP_STEP, 1>(0) * matMulQuantCoeff;

#endif
        fp32CurrentMaxTemp = fp32HistoricMaxTemp;

        __ESIMD_NS::simd<float, 8 * 16> ttemp;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32 * kk), tempOutput.select<16, 1>(32 * kk + 16));
        }
#pragma unroll
        for (int kkk = 0; kkk < (KV_LOOP_STEP-16)/8; ++kkk) {
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16 * kk), tempOutput.select<16, 1>((8 * kkk + kk) * 16 + 16 * 16));
            }
        }
        ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
        ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
        ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));

        fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);



#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
            tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
        }
#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
            tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
        }

#if 0
            fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
            fp32SoftMaxCompensation = __ESIMD_NS::pow<float, 16, float>(2.718281828459f, fp32SoftMaxCompensation);
#else
            fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
        fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);
#endif
        if (loopIdx != 0) {
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
        }

#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(32 * kk) + tempOutput.select<16, 1>(32 * kk + 16);
        }
#pragma unroll
        for (int k = 0; k < (KV_LOOP_STEP - 16) / 8; ++k) {
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = ttemp.select<16, 1>(16 * kk) + tempOutput.select<16, 1>((8 * k + kk) * 16 + 16 * 16);
            }
        }
        ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(16 * 4);
        ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(16 * 2);
        ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);

        fp32HistoricMaxTemp = fp32CurrentMaxTemp;
        // slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
        // slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
        // slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#ifdef QKV_INT8_DEC
#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP/4; kk++) {

            __ESIMD_NS::simd<float, 64> shuffleTemp;       //4 rows?
#if  0
            shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f;
            shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f;

            shuffleTemp = shuffleTemp + 0.5f;
#else
            shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
            shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f + 0.5f;

#endif
            shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
            tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
            tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
        }
#else
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            __ESIMD_NS::simd<float, 32> shuffleTemp;
            shuffleTemp = tempOutput.select<32, 1>(32 * kk);
            tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
            tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
        }

#endif

        __ESIMD_NS::simd<fp16, 32> tempCompensation;
        tempCompensation.select<16, 1>(0) = fp32SoftMaxCompensation;
        tempCompensation.select<16, 1>(16) = tempCompensation.select<16, 1>(0);
        if (loopIdx != 0) {
            __ESIMD_NS::simd<fp16, 32> fp16Temp0 = tempCompensation;
#pragma unroll
            for (int kk = 0; kk < HEAD_DIM/2; kk++) {
                finalOutput.template select<32, 1>(32 * kk) = finalOutput.template select<32, 1>(32 * kk) * fp16Temp0;
            }
        }
    }
	
    //barrier();
    named_barrier_signal(headIdxKv+1, 0, 4, 4);
    named_barrier_wait(headIdxKv+1); // consumers waiting for signal

    //XMX2
#ifdef QKV_INT8_DEC
#pragma unroll
        for (int nn = 0; nn < KV_LOOP_STEP* HEAD_DIM/1024; nn++) { // HEAD_DIM/ (4*8)

            // add one more loop? and no C= A*B
#pragma unroll
            //Load 2048 from SLM,  8 xmx
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 1; ll++) {
                    i8TempBuffer.select<512, 1>(512 * l + 512 * ll) = slm_block_load<int8_t, 512>(    //2048
                      slmOffsetBaseV + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP +
                      (KV_LOOP_STEP/2) * HEAD_DIM * l * sizeof(int8_t) +     // next half
                      //1024 * nn * sizeof(int8_t) +   // 2048/2
                      nn * 512 * sizeof(int8_t));
                }
            }

/*	    if(batchIdx == 0 && kvChunkIdx == 0 && headIdxKv == 0)
                {

                    int32_t ttt1 = i8TempBuffer[0];
                    int32_t ttt2 = i8TempBuffer[1];
                    int32_t ttt3 = i8TempBuffer[2];
                    int32_t ttt4 = i8TempBuffer[3];

                    sycl::ext::oneapi::experimental::printf("thread_idx %d nn %d i8TempBuffer %d %d %d %d \n",localLinearId, nn, ttt1, ttt2, ttt3, ttt4);
                }
*/	    
#pragma unroll
            //C= A*B  
            for (int ll = 0; ll < 2; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 256);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);          //1st 32*16
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                      __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                      __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                    );
            }

            //TODO, add more XMX
#pragma unroll
            //C= C+ A*B   
            for (int ll = 0; ll < 2; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 256);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);    //2nd 32*16
                //8*32
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 + 64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                      __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                      __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                      __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                    );
            }

            //float vsTempFp32 = vsTemp * (1.0f / 255.0f);
#pragma unroll
            for (int32_t kk = 0; kk < 8; kk++) { //32*16
                __ESIMD_NS::simd<fp32, 32> fp32Temp1;
                fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk + 256);
                //fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk + 512) * 0.003921568627451f;
                finalOutput.template select<32, 1>(8 * 32 * nn + 32 * kk) += fp32Temp1 * fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk);
            }
        }
#else
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 4; ll++) {
                    fp16TempBuffer.select<256, 1>(256 * ll) = slm_block_load<fp16, 256>(slmOffsetBaseV +ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP * 2 +
                      16 * HEAD_DIM * nn * sizeof(fp16) +
                      1024 * l * sizeof(fp16) +
                      ll * 256 * sizeof(fp16));
                }


#pragma unroll
                for (int ll = 0; ll < 8; ll++) {
                    auto ccTile = finalOutput.template select<128, 1>(1024 * l + 128 * ll);
                    auto aaTile = tempQkFp16.select<256, 1>(256 * nn);
                    auto bbTile = fp16TempBuffer.select<128, 1>(128 * ll);

                    ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                      __ESIMD_NS::simd<fp16, 128>(ccTile.data()),
                      __ESIMD_NS::simd<fp16, 256>(aaTile.data()),
                      __ESIMD_NS::simd<fp16, 128>(bbTile.data()));
                }
            }
        }
#endif

#if 1
        offsetK += KV_LOOP_STEP * kvHiddenDim * sizeof(int8_t);
        if (singpagelastIdx < singlePageMaxMask) {
#pragma unroll
            for (int n = 0; n < outloop; n++)
            {
#pragma unroll
                for (int kk = 0; kk < KV_LOOP_STEP/16; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        HEAD_DIM,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetK + n * headKv * HEAD_DIM +
                        kk * 16 * kvHiddenDim * sizeof(int8_t));
            }
        }
                }
#endif

    //barrier();
  }

      __ESIMD_NS::simd<float, 32> softMaxDividor;
      simd_mask<16> mask;
      // Only one divide simd6
      //softMaxDividor.select<16, 1>(0) =  1.0f /fp32SoftMaxTemp;
      //softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
    
#pragma unroll
      for (int kk = 0; kk < (HEAD_DIM / 2); kk++) {
          __ESIMD_NS::simd<float, 32> f16Temp = finalOutput.template select<32, 1>(32 * kk);
          //f16Temp = f16Temp * softMaxDividor;
          finalOutput.template select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
          finalOutput.template select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
      }

      uint32_t offsetOutputBase = batchIdx * kvChunkNum * q_len * headQ * HEAD_DIM/*batch*/ +
        kvChunkIdx * q_len * headQ * HEAD_DIM/*kv chunk*/ +
        vvq * headQ * HEAD_DIM/*token*/ + headIdxQ * HEAD_DIM/*header*/;
      simdOffsets = baseOffsetInc16AsVector;
      //simdOffsets = simdOffsets + 64 * h + 16 * hhq;
      mask = simdOffsets <= boundaryQ;
      simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
      simdOffsets = simdOffsets * HEAD_DIM *sizeof(fp16) + offsetOutputBase * sizeof(fp16);

#pragma unroll
      for (int kk = 0; kk < HEAD_DIM/16; kk++) {
          __ESIMD_ENS::lsc_scatter<
              uint32_t,
              8,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::write_back,
              __ESIMD_ENS::cache_hint::write_back,
              16,
              uint32_t
          >((uint32_t*)out, simdOffsets, finalOutput.template bit_cast_view<uint32_t>().template select<128, 1>(128 * kk), mask);
          simdOffsets += 8 * sizeof(uint32_t);
      }

      uint32_t offsetOutputSoftmax = batchIdx * kvChunkNum * q_len * headQ/*batch*/ +
          kvChunkIdx * q_len * headQ /*kv chunk*/ +
          vvq * headQ /*token*/ + headIdxQ /*header*/;

      __ESIMD_NS::block_store<fp16, gqaRatio>((fp16*)out_lse + offsetOutputSoftmax, fp32SoftMaxTemp.template select<gqaRatio, 1>(0));
      __ESIMD_NS::block_store<fp16, gqaRatio>((fp16*)out_max + offsetOutputSoftmax, fp32HistoricMaxTemp.template select<gqaRatio,1>(0));

              });
          };
    return kernel_func;
}

template<uint32_t HEAD_DIM>
ESIMD_INLINE  cgf_t sage_attn_decode_paged_reduce_large(
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
    sycl::range<3> GlobalRange(batch_num, QLEN, headQ);
    sycl::range<3> LocalRange(1, 1, 1);
    sycl::nd_range<3> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(
          Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{

    
            int batch_idx = ndi.get_group(0);
            int q_idx = ndi.get_group(1);
            int q_head_idx = ndi.get_group(2);
            int32_t offset_output_reduce = batch_idx * chunk_count * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_reduce_final = batch_idx * 1 * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_lse_max = batch_idx * chunk_count * QLEN * headQ
              + q_idx * headQ + q_head_idx;

            __ESIMD_NS::simd<REDUCE_TYPE, 1> max_final = -65504.0;
            __ESIMD_NS::simd<REDUCE_TYPE, HEAD_DIM> reduce_final = 0;
            __ESIMD_NS::simd<REDUCE_TYPE, 1> lse_final = 0;

            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<REDUCE_TYPE, 1> cur_max = __ESIMD_NS::block_load<REDUCE_TYPE, 1>((REDUCE_TYPE*)output_max + offset_output_lse_max_ck);
              max_final = __ESIMD_NS::max<REDUCE_TYPE, 1, REDUCE_TYPE>(cur_max, max_final);
            }
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_reduce_ck = offset_output_reduce + chunk_idx * QLEN * headQ * HEAD_DIM;
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<REDUCE_TYPE, 1> cur_max = __ESIMD_NS::block_load<REDUCE_TYPE, 1>((REDUCE_TYPE*)output_max + offset_output_lse_max_ck);
              __ESIMD_NS::simd<REDUCE_TYPE, 1> cur_lse = __ESIMD_NS::block_load<REDUCE_TYPE, 1>((REDUCE_TYPE*)output_lse + offset_output_lse_max_ck);
              __ESIMD_NS::simd<REDUCE_TYPE, HEAD_DIM> cur_reduce = __ESIMD_NS::block_load<REDUCE_TYPE, HEAD_DIM>((REDUCE_TYPE*)output + offset_output_reduce_ck);

              __ESIMD_NS::simd<REDUCE_TYPE, 1> correction = exp2<REDUCE_TYPE, 1, REDUCE_TYPE>( cur_max - max_final);

              lse_final = lse_final + cur_lse * correction;

              cur_reduce.template select<HEAD_DIM, 1>(0) = cur_reduce.template select<HEAD_DIM, 1>(0) * correction[0];
              reduce_final = reduce_final + cur_reduce;
            }

            reduce_final.template select<HEAD_DIM, 1>(0) = reduce_final.template select<HEAD_DIM, 1>(0) / lse_final[0];
            __ESIMD_NS::block_store<REDUCE_TYPE, HEAD_DIM>((REDUCE_TYPE*)output_final + offset_output_reduce_final, reduce_final);
          });
        };


    sycl::range<3> GlobalRange1(batch_num, QLEN, headQ / 32);
    sycl::range<3> LocalRange1(1, 1, 1);
    sycl::nd_range<3> Range1(GlobalRange1, LocalRange1);
    cgf_t kernel_func1 = [=](sycl::handler& cgh) {
        cgh.parallel_for(
          Range1, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{

         
            int batch_idx = ndi.get_group(0);
            int q_idx = ndi.get_group(1);
            int q_head_idx = ndi.get_group(2) << 5;

            int32_t offset_output_reduce = batch_idx * chunk_count * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_reduce_final = batch_idx * 1 * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_lse_max = batch_idx * chunk_count * QLEN * headQ
              + q_idx * headQ + q_head_idx;

            __ESIMD_NS::simd<REDUCE_TYPE, 32> max_final = -65504.0;
            __ESIMD_NS::simd<REDUCE_TYPE, HEAD_DIM * 32> reduce_final = 0;
            __ESIMD_NS::simd<REDUCE_TYPE, HEAD_DIM * 16> cur_reduce;
            __ESIMD_NS::simd<REDUCE_TYPE, 32> lse_final = 0;

            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<REDUCE_TYPE, 32> cur_max = __ESIMD_NS::block_load<REDUCE_TYPE, 32>((REDUCE_TYPE*)output_max + offset_output_lse_max_ck);
              max_final = __ESIMD_NS::max<REDUCE_TYPE, 32, REDUCE_TYPE>(cur_max, max_final);
            }
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_reduce_ck = offset_output_reduce + chunk_idx * QLEN * headQ * HEAD_DIM;
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<REDUCE_TYPE, 32> cur_max = __ESIMD_NS::block_load<REDUCE_TYPE, 32>((REDUCE_TYPE*)output_max + offset_output_lse_max_ck);
              __ESIMD_NS::simd<REDUCE_TYPE, 32> cur_lse = __ESIMD_NS::block_load<REDUCE_TYPE, 32>((REDUCE_TYPE*)output_lse + offset_output_lse_max_ck);
              __ESIMD_NS::simd<REDUCE_TYPE, 32> correction = __ESIMD_NS::exp2<REDUCE_TYPE, 32, REDUCE_TYPE>(cur_max - max_final);
              lse_final = lse_final + cur_lse * correction;
              #pragma unroll
              for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                  cur_reduce.template select<HEAD_DIM,1>(HEAD_DIM * j) = __ESIMD_NS::block_load<REDUCE_TYPE, HEAD_DIM>((REDUCE_TYPE*)output + offset_output_reduce_ck + (i * 16 + j) * HEAD_DIM);
                  reduce_final.template select<HEAD_DIM,1>(HEAD_DIM * (i * 16 + j)) = cur_reduce.template select<HEAD_DIM,1>(HEAD_DIM * j) * correction[i * 16 + j] + reduce_final.template select<HEAD_DIM,1>(HEAD_DIM * (i * 16 + j));
                }
              }
            }
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
              reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM) = reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM) / lse_final[i];
              __ESIMD_NS::block_store<REDUCE_TYPE, HEAD_DIM>((REDUCE_TYPE*)output_final + offset_output_reduce_final + i * HEAD_DIM, reduce_final.template select<HEAD_DIM, 1>(i * HEAD_DIM));
            }
          });
        };
    if  ( (headQ & 31)== 0 ) {
        return kernel_func1;
    }
    return kernel_func;
}

template <uint32_t HEAD_DIM, uint32_t q_len, uint32_t gqaRatio, uint32_t kvChunkSize, uint32_t HEAD_KV = 8>
ESIMD_INLINE cgf_t sageAttnDecoder_acc32(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* qs,
  uint8_t* ks,
  uint8_t* vs,
  uint32_t* cuSeqQuery,
  uint32_t* cuSeqKv,
  uint32_t* kvCacheBlockTable,
  uint8_t* out,
  uint8_t* out_lse,
  uint8_t* out_max,
  uint32_t headQ,
  uint32_t headKv,
  uint32_t kvCacheBlockPerBatch,
  uint32_t batch_size
) {
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;
    uint32_t maxKvChunkNum = (kvCacheBlockPerBatch * blockSize + kvChunkSize - 1) / kvChunkSize;
    constexpr uint32_t numBlocksInKvChunk = (kvChunkSize + blockSize - 1) / blockSize;
    int groupH;
    int groupV;
    int localH;
    int localV;

    groupH = batch_size;
    groupV = maxKvChunkNum;
    localH = q_len;
    localV = headKv;

    sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV);
    sycl::range<2> LocalRangeGqa(localH, localV);
    sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(qkvMatMatRangeGqa, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
    
    //----------------- kernel start ----------------
    float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
    if constexpr (HEAD_DIM == 64) {
        matMulQuantCoeff = 0.125f;
    }

    constexpr uint32_t slmSizeK = 0;
    constexpr uint32_t slmSizeV = KV_LOOP_STEP * HEAD_DIM * sizeof(int8_t) * HEAD_KV;

    //Q SLM
    constexpr uint32_t slmSizeQ = 16 * HEAD_DIM * q_len * sizeof(int8_t) * HEAD_KV;
    //constexpr uint32_t slmSizeQ = 0;

    constexpr uint32_t slmSize = slmSizeV + slmSizeQ;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);

    named_barrier_init<1 + HEAD_KV>();
    constexpr uint32_t slmOffsetBaseV = 0;
    constexpr uint32_t slmOffsetBaseQ = slmOffsetBaseV + slmSizeV;


    constexpr uint32_t outloop = (HEAD_DIM / 8) / q_len;
    constexpr uint32_t boundaryQ = gqaRatio - 1;
    float MIN_VALUE = FP32_MIN * matMulQuantCoeff;


    int32_t localLinearId = ndi.get_local_id(0);
    int32_t vvq = localLinearId;
    int32_t hhv = localLinearId;
    int32_t batchIdx = ndi.get_group(0);
    int32_t kvChunkIdx = ndi.get_group(1);

    int32_t headIdxKv = ndi.get_local_id(1);
    int32_t headIdxQ = gqaRatio * headIdxKv;


    __ESIMD_NS::simd<uint32_t, 2> kvCoord;

    __ESIMD_NS::simd<int8_t, 16 * HEAD_DIM> i8QState;
    __ESIMD_NS::simd<fp32, 32> fp32Qs;
    __ESIMD_NS::simd<int32_t, 16 * KV_LOOP_STEP> i32TempBuffer;
    __ESIMD_NS::simd<int8_t, KV_LOOP_STEP * 8> i8VState;
    __ESIMD_NS::simd<int8_t, 512 * 4> i8TempBuffer; //to load K and V from SLM
    __ESIMD_NS::simd<fp32, 16 * HEAD_DIM> finalOutput = 0;
    __ESIMD_NS::simd<fp32, HEAD_DIM> fp32Qv;

    __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
    __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    uint32_t batchStartKv;
    uint32_t batchEndKv;

    batchStartKv = cuSeqKv[batchIdx];
    batchEndKv = cuSeqKv[batchIdx + 1];
    int32_t totalKvSeqLen = batchEndKv - batchStartKv;

    uint32_t kvChunkNum = (totalKvSeqLen + kvChunkSize - 1) / kvChunkSize;
    if (kvChunkIdx >= kvChunkNum || totalKvSeqLen < q_len)
        return;

    //int32_t totalKvSeqLen = totalKvSeqLen1 - 3 + localLinearId;
    uint32_t blockIdxStart = kvChunkIdx * kvChunkSize / blockSize;
    uint32_t block_num = numBlocksInKvChunk;
    uint32_t kVloopNum = (kvChunkSize + KV_LOOP_STEP - 1) / KV_LOOP_STEP;

    if ((kvChunkIdx + 1) * kvChunkSize > totalKvSeqLen)
    {
        uint32_t restInKvChunk = totalKvSeqLen % kvChunkSize;
        kVloopNum = (restInKvChunk + KV_LOOP_STEP - 1) / KV_LOOP_STEP;
        block_num = (restInKvChunk + blockSize - 1) / blockSize;
    }
    int curKVlength=totalKvSeqLen-3+localLinearId;
    

    __ESIMD_NS::simd<uint32_t, numBlocksInKvChunk> blockIdx;
    // the last not aligned kv chunk
    if (block_num != numBlocksInKvChunk)
    {
        for (int32_t k = 0; k < block_num; k++)
        {
            blockIdx[k] = kvCacheBlockTable[batchIdx * kvCacheBlockPerBatch + blockIdxStart + k];
        }
    }
    else
    {
        blockIdx = __ESIMD_NS::block_load<uint32_t, numBlocksInKvChunk>(
          kvCacheBlockTable + batchIdx * kvCacheBlockPerBatch + blockIdxStart);
    }

    unsigned int offsetQBase = headIdxQ * HEAD_DIM + vvq * headQ * HEAD_DIM/*token*/ + batchIdx * q_len * headQ * HEAD_DIM/*batch*/;
    unsigned int kvHiddenDim = headKv * HEAD_DIM;
    unsigned int offsetBaseK = (headIdxKv * HEAD_DIM + localLinearId * outloop * headKv * HEAD_DIM) * sizeof(int8_t);
    unsigned int offsetK;
    unsigned int slmOffsetV = slmOffsetBaseV + hhv * outloop * 256 * sizeof(int8_t) + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP;
    unsigned int slmOffsetQ = slmOffsetBaseQ + (headIdxKv * q_len + localLinearId) * 16 * HEAD_DIM * sizeof(int8_t);

    const fp32* qscale_head = (const fp32*)qs + batchIdx * q_len * headQ + vvq * headQ + headIdxQ;

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * HEAD_DIM + offsetQBase;
#pragma unroll
    for (int k = 0; k < HEAD_DIM / 32; k++) {
        //512
        i8QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * k) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)qState, simdOffsets);

        simdOffsets += 8 * sizeof(uint32_t);
    }

    //[q_len, q_head_num, 1]
    fp32Qs.select<gqaRatio, 1>(0) = block_load<fp32, gqaRatio>(qscale_head);
    fp32Qs.select<16, 1>(16) = fp32Qs.select<16, 1>(0);

    __ESIMD_NS::simd<float, 16> fp32SoftMaxTemp = 0.0f;
    __ESIMD_NS::simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    __ESIMD_NS::simd<float, 16> fp32CurrentMaxTemp;
    __ESIMD_NS::simd<float, 16> fp32SoftMaxCompensation;
        

#pragma unroll
    for (int k = 0; k < HEAD_DIM / 64; k++) {
        fp32Qv.template select<64, 1>(64 * k) =
            __ESIMD_ENS::lsc_block_load<
            fp32,
            64,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((fp32*)vs + HEAD_DIM * headIdxKv + 64 * k);
    }


    fp32Qv = fp32Qv * (1.0f / 255.0f);

#pragma unroll
    for (int32_t kk = 0; kk < HEAD_DIM / 32; kk++) {
        slm_block_store<uint32_t, 128>(slmOffsetQ + kk * 128 * sizeof(uint32_t), i8QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * kk));
    }

    __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);


    for (int loopIdx = 0; loopIdx < kVloopNum; loopIdx++)
    {
        uint32_t whichIdx = loopIdx;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * KV_LOOP_STEP;
        subBlock = macroBlock + subBlock;
        offsetK = subBlock * kvHiddenDim + offsetBaseK;
        uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;

        unsigned int offsetK_outloop = offsetK;
        // Load KS
        simd<fp32, KV_LOOP_STEP> ksTemp;

        simd<uint32_t, 16> ksOffset;
        ksOffset = baseOffsetInc16AsVector + KV_LOOP_STEP * loopIdx + kvChunkIdx * kvChunkSize;
        ksOffset.merge(0, ksOffset >= totalKvSeqLen);
        ksOffset = ksOffset * headKv * sizeof(fp32) + headIdxKv * sizeof(fp32);
        ksOffset += batchStartKv * headKv * sizeof(fp32)/*batch*/;

        if constexpr (HEAD_DIM == 64) {
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                __ESIMD_ENS::lsc_prefetch<
                    uint8_t,
                    HEAD_DIM,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached
                >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
            }
        }


#pragma unroll
        for (int32_t n = 0; n < KV_LOOP_STEP / 16; n++)
        {
            ksTemp.select<16, 1>(16 * n) =
                __ESIMD_ENS::lsc_gather<
                fp32,
                1,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((fp32*)ks, ksOffset);

            ksOffset += 16 * headKv * sizeof(fp32);
        }

        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * KV_LOOP_STEP, 1>(0);
        auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();  //quant and VNNI to INT8
        auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();  //VNNI to fp16

        i32TempBuffer = 0;

        //XMX1
        {
            offsetK_outloop = subBlock * kvHiddenDim;
            auto ksAllTemp = ksTemp;

#pragma unroll
            for (int32_t nn = 0; nn < KV_LOOP_STEP / 16; nn++) {

#pragma unroll
                for (int32_t l = 0; l < HEAD_DIM / (XMX_K); l++) { //4
#pragma unroll
                    //2 xmx [8*32]
                    for (int32_t ll = 0; ll < 1; ll++) {

                        i8TempBuffer.select<512, 1>(512 * ll) = __ESIMD_ENS::lsc_load_2d<
                            uint8_t, 32, 16, 1,  //16x32
                            false, false,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK_outloop,
                            headKv * HEAD_DIM - 1, totalKvSeqLen - 1, headKv * HEAD_DIM - 1,
                                l * 32 + headIdxKv * HEAD_DIM, nn * 16);

                    }
#pragma unroll
                    for (int32_t ll = 0; ll < 1; ll++) {
                        i8TempBuffer.select<512, 1>(512 + 512 * ll) = slm_block_load<int8_t, 512>(
                          slmOffsetQ +
                          512 * l * sizeof(int8_t));

                    }

#pragma unroll
                    //2 xmx
                    for (int32_t k = 0; k < 1; k++) {
#pragma unroll
                        for (int32_t kk = 0; kk < 2; kk++) {
                            auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
                            auto aaTile = i8TempBuffer.template bit_cast_view<int32_t>().select<128, 1>(128 + 128 * k);    //Q
                            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk); //K

                            ccTile =
                                dpas
                                <8, 8, int32_t, int32_t, int32_t, int32_t,
                                dpas_argument_type::s8,
                                dpas_argument_type::s8
                                >(
                                    __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                                    __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                                    __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                                );
                        }
                    }
                }

#pragma unroll
                //Q&K dequant,  2 rows (16 kv) xmx acc
                for (int32_t kk = 0; kk < 8; kk++) {   //16*16
                    __ESIMD_NS::simd<float, 32> fp32Temp0;
                    __ESIMD_NS::simd<float, 32> fp32Temp1;
                    fp32Temp0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk) * fp32Qs;
                    fp32Temp1 = i32TempBuffer.select<32, 1>(256 * nn + 32 * kk);
                    tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1 * matMulQuantCoeff;
                }

            }
        }

        if constexpr (HEAD_DIM == 128) {
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                __ESIMD_ENS::lsc_prefetch<
                    uint8_t,
                    HEAD_DIM,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached
                >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
            }
        }

        //Load V
        __ESIMD_NS::simd<uint32_t, 32> simd32Offsets00;  

#pragma unroll
        for (int n = 0; n < outloop; n++)
        {
#pragma unroll
            for (int kk = 0; kk < KV_LOOP_STEP / 32; kk++) {
                simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
                simd32Offsets00.select<16, 1>(0 + 16) = baseOffsetInc16AsVector + 16;
                simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 32 * kk + subBlock;

                simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) * kvHiddenDim +
                    (headIdxKv * HEAD_DIM) + hhv * outloop * 8 + n * 8;
                
                //32*8, always block_size algined
                i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
                    __ESIMD_ENS::lsc_gather<
                    uint32_t,
                    2,
                    __ESIMD_ENS::lsc_data_size::u32,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    32,
                    uint32_t
                    >((uint32_t*)vState, simd32Offsets00.select<32, 1>(0) * sizeof(int8_t));// mask);
            }


#pragma unroll
            //write 512 to SLM, 64*8
            for (int k = 0; k < KV_LOOP_STEP * 8 / 256; k++) {
                __ESIMD_NS::simd<int8_t, 256> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {

                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
                }

                slm_block_store<int32_t, 64>(slmOffsetV + n * 256 * sizeof(int8_t) + k * HEAD_DIM * XMX_K * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
            }

        }

        //softmax
        { 
            int restInLoop=kvChunkIdx*kvChunkSize+(loopIdx+1)* KV_LOOP_STEP -curKVlength;
            if( restInLoop > 0 ){
            #pragma unroll    
                for (int k = 0; k < KV_LOOP_STEP; k++) {
                    tempOutput.select<16, 1>(16 * k).merge(FP32_MIN, k >= (KV_LOOP_STEP - restInLoop));
                }
            }

            fp32CurrentMaxTemp = fp32HistoricMaxTemp;

            __ESIMD_NS::simd<float, 8 * 16> ttemp;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32 * kk), tempOutput.select<16, 1>(32 * kk + 16));
            }
#pragma unroll
            for (int kkk = 0; kkk < (KV_LOOP_STEP - 16) / 8; ++kkk) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16 * kk), tempOutput.select<16, 1>((8 * kkk + kk) * 16 + 16 * 16));
                }
            }
            ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
            ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
            ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));

            fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);

#pragma unroll
            for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
            }
#pragma unroll
            for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }

            fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
            fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

            if (loopIdx != 0) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
            }

#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(32 * kk) + tempOutput.select<16, 1>(32 * kk + 16);
            }
#pragma unroll
            for (int k = 0; k < (KV_LOOP_STEP - 16) / 8; ++k) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = ttemp.select<16, 1>(16 * kk) + tempOutput.select<16, 1>((8 * k + kk) * 16 + 16 * 16);
                }
            }
            ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(16 * 4);
            ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(16 * 2);
            ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);

            fp32HistoricMaxTemp = fp32CurrentMaxTemp;

    #pragma unroll
            for (int kk = 0; kk < KV_LOOP_STEP / 4; kk++) {

                __ESIMD_NS::simd<float, 64> shuffleTemp;

                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f + 0.5f;
                shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
                tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
                tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
            }

            __ESIMD_NS::simd<fp32, 32> tempCompensation;
            tempCompensation.select<16, 1>(0) = fp32SoftMaxCompensation;
            tempCompensation.select<16, 1>(16) = tempCompensation.select<16, 1>(0);
            if (loopIdx != 0) {
                __ESIMD_NS::simd<fp32, 32> fp16Temp0 = tempCompensation;
#pragma unroll
                for (int kk = 0; kk < HEAD_DIM / 2; kk++) {
                    finalOutput.template select<32, 1>(32 * kk) = finalOutput.template select<32, 1>(32 * kk) * fp16Temp0;
                }
            }
        }

        named_barrier_signal(headIdxKv + 1, 0, 4, 4);
        named_barrier_wait(headIdxKv + 1); // consumers waiting for signal


        //XMX2
#pragma unroll
        for (int nn = 0; nn < KV_LOOP_STEP * HEAD_DIM / 1024; nn++) { //8
#pragma unroll
            //Load 1024 from SLM,  4 xmx
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 1; ll++) {
                    i8TempBuffer.select<512, 1>(512 * l + 512 * ll) = slm_block_load<int8_t, 512>(    //1024
                      slmOffsetBaseV + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP +
                      (KV_LOOP_STEP / 2) * HEAD_DIM * l * sizeof(int8_t) +     // next half
                      512 * nn * sizeof(int8_t));  // 1024/2
                }
            }

#pragma unroll
            //C= A*B  
            for (int ll = 0; ll < 2; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 256);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);          //1st 32*16
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                      __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                      __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                    );
            }

            //TODO, add more XMX
#pragma unroll
            //C= C+ A*B   
            for (int ll = 0; ll < 2; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 256);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);    //2nd 32*16
                //8*32
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 + 64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                      __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                      __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                      __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                    );
            }

#pragma unroll
            for (int32_t kk = 0; kk < 8; kk++) { //16*16
                __ESIMD_NS::simd<fp32, 32> fp32Temp1;
                fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk + 256);
                finalOutput.template select<32, 1>(8 * 32 * nn + 32 * kk) += fp32Temp1 * fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk);
            }
        }

#if 1
        offsetK += KV_LOOP_STEP * kvHiddenDim * sizeof(int8_t);
        if (singpagelastIdx < singlePageMaxMask) {
#pragma unroll
            for (int n = 0; n < outloop; n++)
            {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        HEAD_DIM,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetK + n * headKv * HEAD_DIM +
                        kk * 16 * kvHiddenDim * sizeof(int8_t));
                }
            }
        }
#endif

    }

    __ESIMD_NS::simd<float, 32> softMaxDividor;
    simd_mask<16> mask;

    uint32_t offsetOutputBase = batchIdx * kvChunkNum * q_len * headQ * HEAD_DIM/*batch*/ +
        kvChunkIdx * q_len * headQ * HEAD_DIM/*kv chunk*/ +
        vvq * headQ * HEAD_DIM/*token*/ + headIdxQ * HEAD_DIM/*header*/;
    simdOffsets = baseOffsetInc16AsVector;
    mask = simdOffsets <= boundaryQ;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * HEAD_DIM * sizeof(fp32) + offsetOutputBase * sizeof(fp32);

#pragma unroll
    for (int kk = 0; kk < HEAD_DIM / 8; kk++) {
        __ESIMD_ENS::lsc_scatter<
            fp32,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back,
            16,
            uint32_t
        >((fp32*)out, simdOffsets, finalOutput.template select<128, 1>(128 * kk), mask);
        simdOffsets += 8 * sizeof(uint32_t);
    }
    uint32_t offsetOutputSoftmax = batchIdx * kvChunkNum * q_len * headQ/*batch*/ +
        kvChunkIdx * q_len * headQ /*kv chunk*/ +
        vvq * headQ /*token*/ + headIdxQ /*header*/;

    __ESIMD_NS::block_store<fp32, gqaRatio>((fp32*)out_lse + offsetOutputSoftmax, fp32SoftMaxTemp.template select<gqaRatio, 1>(0));
    __ESIMD_NS::block_store<fp32, gqaRatio>((fp32*)out_max + offsetOutputSoftmax, fp32HistoricMaxTemp.template select<gqaRatio, 1>(0));

     }); //parallel_for
  };  //kernel_func
    return kernel_func;
}

//TODO 32 Q
template <uint32_t HEAD_DIM, uint32_t q_len, uint32_t gqaRatio, uint32_t kvChunkSize, uint32_t Q_LEN_PER_THREAD=1, uint32_t HEAD_KV = 8>
ESIMD_INLINE cgf_t sageAttnDecoder_acc32_dim64(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* qs,
  uint8_t* ks,
  uint8_t* vs,
  uint32_t* cuSeqQuery,
  uint32_t* cuSeqKv,
  uint32_t* kvCacheBlockTable,
  uint8_t* out,
  uint8_t* out_lse,
  uint8_t* out_max,
  uint32_t headQ,
  uint32_t headKv,
  uint32_t kvCacheBlockPerBatch,
  uint32_t batch_size
) {
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;
    constexpr uint32_t numBlocksInKvChunk = (kvChunkSize + blockSize - 1) / blockSize;
    constexpr uint32_t localThreads_qlen = q_len / Q_LEN_PER_THREAD;
    uint32_t maxKvChunkNum = (kvCacheBlockPerBatch * blockSize + kvChunkSize - 1) / kvChunkSize;
    int groupH;
    int groupV;
    int localH;
    int localV;

    groupH = batch_size;
    groupV = maxKvChunkNum;
    localH = localThreads_qlen;
    localV = headKv;

    sycl::range<2> GlobalRangeGqa(groupH * localH, groupV * localV);
    sycl::range<2> LocalRangeGqa(localH, localV);
    sycl::nd_range<2> qkvMatMatRangeGqa(GlobalRangeGqa, LocalRangeGqa);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(qkvMatMatRangeGqa, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
    
    //----------------- kernel start ----------------
    constexpr float matMulQuantCoeff = 0.125f; // 1.0f / sqrt(64.0f);
    constexpr uint32_t slmSizeK = 0;
    constexpr uint32_t slmSizeV = KV_LOOP_STEP * HEAD_DIM * sizeof(int8_t) * HEAD_KV;
    constexpr uint32_t slmSize = slmSizeV;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);

    named_barrier_init<1 + HEAD_KV>();
    constexpr uint32_t slmOffsetBaseV = 0;
    constexpr uint32_t outloop = (HEAD_DIM / 8) / localThreads_qlen;
    constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff;

    constexpr uint32_t boundaryQ = Q_LEN_PER_THREAD * gqaRatio - 1;

    int32_t localLinearId = ndi.get_local_id(0);    //0, 1
    int32_t batchIdx = ndi.get_group(0);
    int32_t kvChunkIdx = ndi.get_group(1);

    int32_t headIdxKv = ndi.get_local_id(1);
    int32_t headIdxQ = gqaRatio * headIdxKv;

    __ESIMD_NS::simd<uint32_t, 2> kvCoord;

    __ESIMD_NS::simd<int8_t, 16 * HEAD_DIM> i8QState;
    __ESIMD_NS::simd<fp32, 32> fp32Qs;
    __ESIMD_NS::simd<int32_t, 16 * KV_LOOP_STEP> i32TempBuffer;
    //__ESIMD_NS::simd<int8_t, 512 * 8> i8VState;
    __ESIMD_NS::simd<int8_t, 512 * 8> i8TempBuffer; //to load K and V from SLM
    __ESIMD_NS::simd<fp32, 16 * HEAD_DIM> finalOutput = 0;
    __ESIMD_NS::simd<fp32, HEAD_DIM> fp32Qv;

    __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
    __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    uint32_t batchStartKv;
    uint32_t batchEndKv;

    batchStartKv = cuSeqKv[batchIdx];
    batchEndKv = cuSeqKv[batchIdx + 1];
    int32_t totalKvSeqLen = batchEndKv - batchStartKv;

    uint32_t kvChunkNum = (totalKvSeqLen + kvChunkSize - 1) / kvChunkSize;
    if (kvChunkIdx >= kvChunkNum || totalKvSeqLen < q_len)
        return;

    uint32_t blockIdxStart = kvChunkIdx * kvChunkSize / blockSize;
    uint32_t block_num = numBlocksInKvChunk;
    uint32_t kVloopNum = (kvChunkSize + KV_LOOP_STEP - 1) / KV_LOOP_STEP;

    if ((kvChunkIdx + 1) * kvChunkSize > totalKvSeqLen)
    {
        uint32_t restInKvChunk = totalKvSeqLen % kvChunkSize;
        kVloopNum = (restInKvChunk + KV_LOOP_STEP - 1) / KV_LOOP_STEP;
        block_num = (restInKvChunk + blockSize - 1) / blockSize;
    }
    int curKVlength = totalKvSeqLen - 4 + Q_LEN_PER_THREAD + Q_LEN_PER_THREAD * localLinearId;
    

    __ESIMD_NS::simd<uint32_t, numBlocksInKvChunk> blockIdx;
    // the last not aligned kv chunk
    if (block_num != numBlocksInKvChunk) {
        for (int32_t k = 0; k < block_num; k++) {
            blockIdx[k] = kvCacheBlockTable[batchIdx * kvCacheBlockPerBatch + blockIdxStart + k];
        }
    } else {
        blockIdx = __ESIMD_NS::block_load<uint32_t, numBlocksInKvChunk>(
          kvCacheBlockTable + batchIdx * kvCacheBlockPerBatch + blockIdxStart);
    }

    unsigned int offsetQBase = headIdxQ * HEAD_DIM + Q_LEN_PER_THREAD *localLinearId * headQ * HEAD_DIM/*token*/ + batchIdx * q_len * headQ * HEAD_DIM/*batch*/;
    unsigned int kvHiddenDim = headKv * HEAD_DIM;
    unsigned int offsetBaseKVPrefetch = (headIdxKv * HEAD_DIM + localLinearId * outloop * headKv * HEAD_DIM) * sizeof(int8_t); //for K/V prefetch
    unsigned int offsetKVPrefetch;
    unsigned int slmOffsetV = slmOffsetBaseV + localLinearId * outloop * 256 * sizeof(int8_t) + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP;

    const fp32* qscale_head = (const fp32*)qs + batchIdx * q_len * headQ + Q_LEN_PER_THREAD *localLinearId * headQ + headIdxQ;

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets.merge(gqaRatio-1, simdOffsets >= gqaRatio);
    //simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets *HEAD_DIM + offsetQBase;
#pragma unroll
    for (int k = 1; k < Q_LEN_PER_THREAD; k++) {
        simdOffsets.select<gqaRatio, 1>(gqaRatio * k) = simdOffsets.select<gqaRatio, 1>(gqaRatio * (k - 1)) + headQ * HEAD_DIM;
    }

#pragma unroll
    for (int k = 0; k < HEAD_DIM / 32; k++) {
        //512
        i8QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * k) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)qState, simdOffsets);

        simdOffsets += 8 * sizeof(uint32_t);
    }

    //[q_len, q_head_num, 1]
#pragma unroll
    for (int k = 0; k < Q_LEN_PER_THREAD; k++) {
        fp32Qs.select<gqaRatio, 1>(k* gqaRatio) = block_load<fp32, gqaRatio>(qscale_head + k*headQ);
    }

    fp32Qs.select<16, 1>(16) = fp32Qs.select<16, 1>(0);

    __ESIMD_NS::simd<float, 16> fp32SoftMaxTemp = 0.0f;
    __ESIMD_NS::simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    __ESIMD_NS::simd<float, 16> fp32CurrentMaxTemp;
    __ESIMD_NS::simd<float, 16> fp32SoftMaxCompensation;
        

#pragma unroll
    for (int k = 0; k < HEAD_DIM / 64; k++) {
        fp32Qv.template select<64, 1>(64 * k) =
            __ESIMD_ENS::lsc_block_load<
            fp32,
            64,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((fp32*)vs + HEAD_DIM * headIdxKv + 64 * k);
    }
    __ESIMD_ENS::config_2d_mem_access<uint8_t, 32, 16, 1> Block2DLoadKPayload(
         (uint8_t*)kState, headKv * HEAD_DIM - 1, totalKvSeqLen - 1, headKv * HEAD_DIM - 1, 0, 0);

    fp32Qv = fp32Qv * (1.0f / 255.0f);

    for (int loopIdx = 0; loopIdx < kVloopNum; loopIdx++)
    {
        uint32_t whichIdx = loopIdx;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * KV_LOOP_STEP;
        subBlock = macroBlock + subBlock;
        offsetKVPrefetch = subBlock * kvHiddenDim + offsetBaseKVPrefetch;
        uint32_t singlePagelastIdx = whichIdx & singlePageMaxMask;

        // Load KS
        simd<fp32, KV_LOOP_STEP> fp32Ks;

        simd<uint32_t, 16> ksOffset;
        ksOffset = baseOffsetInc16AsVector + KV_LOOP_STEP * loopIdx + kvChunkIdx * kvChunkSize;
        ksOffset.merge(0, ksOffset >= totalKvSeqLen);
        ksOffset = ksOffset * headKv * sizeof(fp32) + headIdxKv * sizeof(fp32);
        ksOffset += batchStartKv * headKv * sizeof(fp32)/*batch*/;

#pragma unroll
        for (int32_t n = 0; n < KV_LOOP_STEP / 16; n++)
        {
            fp32Ks.select<16, 1>(16 * n) =
                __ESIMD_ENS::lsc_gather<
                fp32,
                1,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((fp32*)ks, ksOffset);

            ksOffset += 16 * headKv * sizeof(fp32);
        }

        //Load K
        Block2DLoadKPayload.set_data_pointer((uint8_t*)kState + subBlock * kvHiddenDim);
#pragma unroll
        for (int32_t ll = 0; ll < 2; ll++) {
            Block2DLoadKPayload.set_x(headIdxKv* HEAD_DIM + ll * 32);
#pragma unroll
            for (int32_t nn = 0; nn < 4; nn++) {
                Block2DLoadKPayload.set_y(nn * 16);

                i8TempBuffer.select<512, 1>(2048 * ll + 512 * nn) = __ESIMD_ENS::lsc_load_2d<
                    uint8_t, 32, 16, 1,  //16x32
                    false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>(Block2DLoadKPayload);
            }
        }

#if 1
        //prefetch V
#pragma unroll
        for (int n = 0; n < outloop; n++)
        {
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                __ESIMD_ENS::lsc_prefetch<
                    uint8_t,
                    HEAD_DIM,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached
                >((uint8_t*)vState + offsetKVPrefetch + n * headKv * HEAD_DIM + kk * 16 * kvHiddenDim * sizeof(int8_t));
            }
        }
#endif

        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * KV_LOOP_STEP, 1>(0);

        //i32TempBuffer = 0;

        //XMX1
#pragma unroll
        //C= A*B  
        for (int kk = 0; kk < 8; kk++) {
            auto ccTile = i32TempBuffer.select<128, 1>(128 * kk);
            auto aaTile = i8QState.template bit_cast_view<int32_t>().template select<128, 1>(0);
            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().template select<64, 1>(64 * kk);

            ccTile =
                dpas
                <8, 8, int32_t, int32_t, int32_t,
                dpas_argument_type::s8,
                dpas_argument_type::s8
                >(
                  __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                  __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
        }

#pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
            auto ccTile = i32TempBuffer.select<128, 1>(128 * kk);
            auto aaTile = i8QState.template bit_cast_view<int32_t>().template select<128, 1>(128);
            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().template select<64, 1>(512 + 64 * kk);

            ccTile =
                dpas
                <8, 8, int32_t, int32_t, int32_t, int32_t,
                dpas_argument_type::s8,
                dpas_argument_type::s8
                >(
                    __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                    __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                    __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
        }

#pragma unroll
        //Q&K dequant
        for (int32_t kk = 0; kk < 32; kk++) {
            __ESIMD_NS::simd<float, 32> fp32Temp0;
            __ESIMD_NS::simd<float, 32> fp32Temp1;
            fp32Temp0 = fp32Ks.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk) * fp32Qs;
            fp32Temp1 = i32TempBuffer.select<32, 1>(32 * kk);
            tempOutput.select<32, 1>(32 * kk) = fp32Temp0 * fp32Temp1 * matMulQuantCoeff;
        }


        //Load V  
        __ESIMD_NS::simd<uint32_t, 32> simd32Offsets00;

#pragma unroll
        for (int n = 0; n < outloop; n++)    //2
        {
#pragma unroll
            for (int kk = 0; kk < KV_LOOP_STEP / 32; kk++) {  //TODO 16 offset to 32 offset
                simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
                simd32Offsets00.select<16, 1>(0 + 16) = baseOffsetInc16AsVector + 16;
                simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 32 * kk + subBlock;

                simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) * kvHiddenDim +
                    (headIdxKv * HEAD_DIM) + localLinearId * outloop * 8 + n * 8;

                //32*8, always block_size algined
                i8TempBuffer.template bit_cast_view<uint32_t>().template select<64, 1>(128 * n + 64 * kk) = 
                    __ESIMD_ENS::lsc_gather<
                    uint32_t,
                    2,
                    __ESIMD_ENS::lsc_data_size::u32,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    32,
                    uint32_t
                    >((uint32_t*)vState, simd32Offsets00.select<32, 1>(0) * sizeof(int8_t));// mask);
            }
        }


        __ESIMD_NS::simd<uint8_t, 16 * KV_LOOP_STEP> tempQkInt8;
        //softmax
        int restInLoop = kvChunkIdx * kvChunkSize + (loopIdx + 1) * KV_LOOP_STEP - curKVlength;

        if (restInLoop > 0) {
#pragma unroll    
            for (int k = 0; k < KV_LOOP_STEP; k++) {
                if constexpr (localThreads_qlen == 2) {
                    tempOutput.select<gqaRatio, 1>(16 * k).merge(FP32_MIN, k == (KV_LOOP_STEP - restInLoop - 1));
                }
                tempOutput.select<16, 1>(16 * k).merge(FP32_MIN, k >= (KV_LOOP_STEP - restInLoop));
            }

        }else if (restInLoop == 0) {
            if constexpr (localThreads_qlen == 2) {
                tempOutput.select<gqaRatio, 1>(16 * (KV_LOOP_STEP - 1)).merge(FP32_MIN, 1);
            }
        }

        fp32CurrentMaxTemp = fp32HistoricMaxTemp;
#if 0
#pragma unroll
        for (int kk = 0; kk < 64; kk++) {
            fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * kk), fp32CurrentMaxTemp);
        }
        //fp32CurrentMaxTemp = fp32CurrentMaxTemp * sycl::ext::intel::esimd::detail::log2e;
#endif

#if 1
        __ESIMD_NS::simd<float, 8 * 16> ttemp;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32 * kk), tempOutput.select<16, 1>(32 * kk + 16));
        }
#pragma unroll
        for (int kkk = 0; kkk < (KV_LOOP_STEP - 16) / 8; ++kkk) {
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16 * kk), tempOutput.select<16, 1>((8 * kkk + kk) * 16 + 16 * 16));
            }
        }
        ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
        ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
        ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));

        fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);
#endif

#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
            tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
        }
#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP; kk++) {
            tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
        }

        fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
        fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

        if (loopIdx != 0) {
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
        }

#if 0
#pragma unroll
        for (int kk = 0; kk < 64; kk++) {
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + tempOutput.select<16, 1>(16 * kk);
        }
#endif

#if 1
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(32 * kk) + tempOutput.select<16, 1>(32 * kk + 16);
        }
#pragma unroll
        for (int k = 0; k < (KV_LOOP_STEP - 16) / 8; ++k) {
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = ttemp.select<16, 1>(16 * kk) + tempOutput.select<16, 1>((8 * k + kk) * 16 + 16 * 16);
            }
        }
        ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(16 * 4);
        ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(16 * 2);
        ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);
#endif
        fp32HistoricMaxTemp = fp32CurrentMaxTemp;

#pragma unroll
        for (int kk = 0; kk < KV_LOOP_STEP / 4; kk++) {

            __ESIMD_NS::simd<float, 64> shuffleTemp;

            shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
            shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f + 0.5f;
            shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
            tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
            tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
        }

#pragma unroll
        for (int n = 0; n < outloop; n++) { 
#pragma unroll
            for (int k = 0; k < KV_LOOP_STEP * 8 / 256; k++) { 
                __ESIMD_NS::simd<int8_t, 256> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {

                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8TempBuffer.select<32, 4>(512 * n + 256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8TempBuffer.select<32, 4>(512 * n + 256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8TempBuffer.select<32, 4>(512 * n + 256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8TempBuffer.select<32, 4>(512 * n + 256 * k + 128 * kk + 3);
                }

                slm_block_store<int32_t, 64>(slmOffsetV + n * 256 * sizeof(int8_t) + k * HEAD_DIM * XMX_K * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
            }
        }

        __ESIMD_NS::simd<fp32, 32> tempCompensation;
        tempCompensation.select<16, 1>(0) = fp32SoftMaxCompensation;
        tempCompensation.select<16, 1>(16) = tempCompensation.select<16, 1>(0);
        if (loopIdx != 0) {
            __ESIMD_NS::simd<fp32, 32> fp16Temp0 = tempCompensation;
#pragma unroll
            for (int kk = 0; kk < HEAD_DIM / 2; kk++) {
                finalOutput.template select<32, 1>(32 * kk) = finalOutput.template select<32, 1>(32 * kk) * fp16Temp0;
            }
        }

        named_barrier_signal(headIdxKv + 1, 0, localThreads_qlen, localThreads_qlen);
        named_barrier_wait(headIdxKv + 1); // consumers waiting for signal


#if 1
        //prefetch K 
        offsetKVPrefetch += KV_LOOP_STEP * kvHiddenDim * sizeof(int8_t);
        if (singlePagelastIdx < singlePageMaxMask) {
#pragma unroll
            for (int n = 0; n < outloop; n++)
            {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        HEAD_DIM,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetKVPrefetch + n * headKv * HEAD_DIM +
                        kk * 16 * kvHiddenDim * sizeof(int8_t));
                }
            }
        }
#endif

        //XMX2
#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
            i8TempBuffer.select<512, 1>(512 * ll) = slm_block_load<int8_t, 512>( 
             slmOffsetBaseV + ndi.get_local_id(1) * HEAD_DIM * KV_LOOP_STEP + ll * 512 * sizeof(int8_t));
        }

#pragma unroll
        //C= A*B  
        for (int ll = 0; ll < 8; ll++) {
            auto ccTile = i32TempBuffer.select<128, 1>(128 * ll);
            auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);          //1st 32*16
            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

            ccTile =
                dpas
                <8, 8, int32_t, uint32_t, int32_t,
                dpas_argument_type::u8,
                dpas_argument_type::s8
                >(
                  __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                  __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
        }

#if 1
#pragma unroll
        for (int k = 0; k < Q_LEN_PER_THREAD; k++) {
            //prefetch Ks
            ksOffset += (Q_LEN_PER_THREAD * localLinearId + k) * 16 * headKv * sizeof(fp32);
            ksOffset.merge(0, ksOffset >= totalKvSeqLen);
            __ESIMD_ENS::lsc_prefetch <
                fp32,
                1,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
            >((fp32*)ks, ksOffset);
        }
#endif


#pragma unroll
        //C= C+ A*B   
        for (int ll = 0; ll < 8; ll++) {
            auto ccTile = i32TempBuffer.select<128, 1>(128 * ll);
            auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);    //2nd 32*16
            //8*32
            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(512 + 64 * ll);

            ccTile =
                dpas
                <8, 8, int32_t, int32_t, uint32_t, int32_t,
                dpas_argument_type::u8,
                dpas_argument_type::s8
                >(
                  __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                  __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
                  __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
        }

#pragma unroll
        for (int32_t kk = 0; kk < 32; kk++) {
            __ESIMD_NS::simd<fp32, 32> fp32Temp1;
            fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk);
            finalOutput.template select<32, 1>(32 * kk) += fp32Temp1 * fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
        }

    }

    __ESIMD_NS::simd<float, 32> softMaxDividor;
    simd_mask<16> mask;

    uint32_t offsetOutputBase = batchIdx * kvChunkNum * q_len * headQ * HEAD_DIM/*batch*/ +
        kvChunkIdx * q_len * headQ * HEAD_DIM/*kv chunk*/ +
        Q_LEN_PER_THREAD *localLinearId * headQ * HEAD_DIM/*token*/ + headIdxQ * HEAD_DIM/*header*/;
    

    simdOffsets = baseOffsetInc16AsVector;
    mask = simdOffsets <= boundaryQ;
    //simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);

    simdOffsets = (simdOffsets * HEAD_DIM + offsetOutputBase) * sizeof(fp32);
#pragma unroll
    for (int k = 1; k < Q_LEN_PER_THREAD; k++) {
        simdOffsets.select<gqaRatio, 1>(gqaRatio* k) = simdOffsets.select<gqaRatio, 1>(gqaRatio * (k - 1)) + headQ * HEAD_DIM * sizeof(fp32);
    }
    
#pragma unroll
    for (int kk = 0; kk < HEAD_DIM / 8; kk++) {
        __ESIMD_ENS::lsc_scatter<
            fp32,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back,
            16,
            uint32_t
        >((fp32*)out, simdOffsets, finalOutput.template select<128, 1>(128 * kk), mask);
        simdOffsets += 8 * sizeof(uint32_t);
    }
    uint32_t offsetOutputSoftmax = batchIdx * kvChunkNum * q_len * headQ/*batch*/ +
        kvChunkIdx * q_len * headQ /*kv chunk*/ +
        Q_LEN_PER_THREAD *localLinearId * headQ /*token*/ + headIdxQ /*header*/;

#pragma unroll
    for (int k = 0; k < Q_LEN_PER_THREAD; k++) {
        //token x
        __ESIMD_NS::block_store<fp32, gqaRatio>((fp32*)out_lse + offsetOutputSoftmax + k*headQ, fp32SoftMaxTemp.template select<gqaRatio, 1>(k* gqaRatio));
        //token next
        __ESIMD_NS::block_store<fp32, gqaRatio>((fp32*)out_max + offsetOutputSoftmax + k*headQ, fp32HistoricMaxTemp.template select<gqaRatio, 1>(k*gqaRatio));
    }

     }); //parallel_for
  };  //kernel_func
    return kernel_func;
}

template<uint32_t HEAD_DIM, uint32_t QLEN = 4>
ESIMD_INLINE  cgf_t sage_attn_decode_paged_reduce_large_acc32(
    uint8_t* output,
    uint8_t* output_lse,
    uint8_t* output_max,
    uint8_t* output_final,
    uint32_t headQ,
    uint32_t batch_num,
    uint32_t chunk_count
)
{
    sycl::range<3> GlobalRange(batch_num, QLEN, headQ);
    sycl::range<3> LocalRange(1, 1, 1);
    sycl::nd_range<3> Range(GlobalRange, LocalRange);

    cgf_t kernel_func = [=](sycl::handler& cgh) {
        cgh.parallel_for(
          Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL{

   
            int batch_idx = ndi.get_group(0);
            int q_idx = ndi.get_group(1);
            int q_head_idx = ndi.get_group(2);
            int32_t offset_output_reduce = batch_idx * chunk_count * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_reduce_final = batch_idx * 1 * QLEN * headQ * HEAD_DIM
              + q_idx * headQ * HEAD_DIM + q_head_idx * HEAD_DIM;
            int32_t offset_output_lse_max = batch_idx * chunk_count * QLEN * headQ
              + q_idx * headQ + q_head_idx;

            __ESIMD_NS::simd<fp32, 1> max_final = FP32_MIN;
            __ESIMD_NS::simd<fp32, HEAD_DIM> reduce_final = 0;
            __ESIMD_NS::simd<fp32, 1> lse_final = 0;

            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<fp32, 1> cur_max = __ESIMD_NS::block_load<fp32, 1>((fp32*)output_max + offset_output_lse_max_ck);
              max_final = __ESIMD_NS::max<fp32, 1, fp32>(cur_max, max_final);
            }
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++)
            {
              int32_t offset_output_reduce_ck = offset_output_reduce + chunk_idx * QLEN * headQ * HEAD_DIM;
              int32_t offset_output_lse_max_ck = offset_output_lse_max + chunk_idx * QLEN * headQ;

              __ESIMD_NS::simd<fp32, 1> cur_max = __ESIMD_NS::block_load<fp32, 1>((fp32*)output_max + offset_output_lse_max_ck);
              __ESIMD_NS::simd<fp32, 1> cur_lse = __ESIMD_NS::block_load<fp32, 1>((fp32*)output_lse + offset_output_lse_max_ck);
              __ESIMD_NS::simd<fp32, HEAD_DIM> cur_reduce = __ESIMD_NS::block_load<fp32, HEAD_DIM>((fp32*)output + offset_output_reduce_ck);

              __ESIMD_NS::simd<fp32, 1> correction = exp2<fp32, 1, fp32>(cur_max - max_final);

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
              __ESIMD_NS::simd<fp32, 16> correction = __ESIMD_NS::exp2<fp32, 16, fp32>(cur_max - max_final);
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

