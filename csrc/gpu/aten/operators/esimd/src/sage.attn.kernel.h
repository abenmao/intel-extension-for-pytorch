// Shape Q [activation token length, 16, 128]  INT8,
// Shape K:  [kv len, 4, 128] INT8,
// Shape V : [kv len, 4, 128] INT8,
// output: [activation token length, 16, 128]  INT8,
#define MERGE_QUANT
#define QKV_INT8
#define FP32_MAX (1.7e+38)
#define FP32_MIN (-1.7e+38)
#define FP16_MAX (65504.0f)
#define FP16_MIN (-65504.0f)

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
//using fp16 = sycl::ext::oneapi::bfloat16;
using fp32 = float;
using namespace sycl;

template <uint32_t GPA_DIV_4>
ESIMD_INLINE void sageAttnGqa_4X(
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
    uint32_t headQ,
    uint32_t headKv,
    sycl::nd_item<2>& ndi
) {
    constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
    constexpr uint32_t slmSizeK = 64 * 128 * sizeof(int8_t);
    constexpr uint32_t slmSizeV = 64 * 128 * sizeof(fp16);
    constexpr uint32_t slmSizeSoftMaxSum = 256 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxHistoric = 256 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxCompensation = 256 * sizeof(float);
    constexpr uint32_t slmSizeKs = 64 * sizeof(float);
    constexpr uint32_t slmSizeVs = 128 * sizeof(float);
    constexpr uint32_t slmSizeQ = 16 * 128 * 16 * sizeof(int8_t);
    constexpr uint32_t slmSize =
        slmSizeK +
        slmSizeV +
        slmSizeSoftMaxSum +
        slmSizeSoftMaxHistoric +
        slmSizeSoftMaxCompensation +
        slmSizeKs +
        slmSizeVs +
        slmSizeQ;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
    constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
    constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
    constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
    constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
    constexpr uint32_t slmOffsetBaseVs = slmOffsetBaseKs + slmSizeKs;
    constexpr uint32_t slmOffsetBaseQ = slmOffsetBaseVs + slmSizeVs;
    constexpr int32_t batchIdx = 0; // v / headKv;
    constexpr int32_t groupGqaRatio = 4;
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
    constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;


    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq = localLinearId & 0x3;
    int32_t vvq = localLinearId >> 2;
    int32_t hhv = localLinearId & 0xf;
    int32_t vvv = localLinearId >> 4;
    int32_t h = ndi.get_group(0);
    int32_t v = ndi.get_group(1);
    int32_t headIdxQ = v * 4;
    int32_t headIdxKv = v / GPA_DIV_4;
    uint32_t kvSeqLen;
    uint32_t activationLength;
    uint32_t startActivationIdx;
    simd<uint32_t, 2> kvCoord;
    simd<uint32_t, 2> queryCoord;
    kvCoord = block_load<uint32_t, 2>(cuSeqKv);
    queryCoord = block_load<uint32_t, 2>(cuSeqQuery);
    //if (kvSeqLen >= activationLength) {
    //  startActivationIdx = kvSeqLen - activationLength;
    //} else {
    //  startActivationIdx = 0;
    //}
    headIdxQ = headIdxQ + vvq;

    //sycl::ext::oneapi::experimental::printf("h = %d, v = %d, headIdxQ %d\n", h, v, headIdxQ);

    //simd<int8_t, 16 * 128> i8QState;
    simd<float, 16> fp32Qs;
    simd<int32_t, 16 * 64> i32TempBuffer;
    //  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
    simd<float, 640> tempBuffer;
    auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
    auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 4);
    auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
    auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
    auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
    simd<float, 16 * 128> finalOutput = 0;
    simd<float, 8> fp32Qv;
    simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
    simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    simd<uint32_t, 16> blockIdx;

    kvSeqLen = kvCoord[1] - kvCoord[0];
    activationLength = queryCoord[1] - queryCoord[0];
    startActivationIdx = kvSeqLen - activationLength;

    int32_t kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 63) >> 6;
    uint32_t myLastFullAttnLoopIdx = (64 * h + startActivationIdx + 16 * hhq) >> 6;
    unsigned int offsetQBase = headIdxQ * 128 * sizeof(int8_t);
    unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
    unsigned int kvHiddenDim = headKv * 128;
    unsigned int offsetBaseK = (headIdxKv * 128 + localLinearId * headKv * 128) * sizeof(int8_t);
    unsigned int offsetK;
    unsigned int slmOffsetK = slmOffsetBaseK + localLinearId * 32 * sizeof(int8_t);
    unsigned int slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16);
    unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
    unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

    unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

    uint32_t boundaryQ = activationLength - 1;

    if (64 * h >= activationLength) {
        return;
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 64 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 128 * sizeof(int8_t) + offsetQBase;

#pragma unroll
    for (int k = 0; k < 4; k++) {
        tempBuffer.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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
    constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff;
    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 64 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * sizeof(float) + headIdxQ * sizeof(float);
    fp32Qs.select<16, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        float,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((float*)qs, simdOffsets);

    softMaxThresh = softMaxThresh + 16 * hhq + 64 * h + startActivationIdx;

    simd<float, 16> fp32SoftMaxTemp = 0.0f;
    simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    blockIdx = __ESIMD_ENS::lsc_block_load<
        uint32_t,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + 0);

    fp32Qv.select<8, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        8,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 128 * headIdxKv + 8 * hhv);
#ifdef QKV_INT8
    fp32Qv = fp32Qv * (1.0f / 255.0f);
    slm_block_store<float, 8>(slmOffsetBaseVs + 8 * hhv * sizeof(float), fp32Qv);
#endif

#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
        slm_block_store<uint32_t, 128>(
            slmOffsetBaseQ + localLinearId * 16 * 128 * sizeof(int8_t) + kk * 128 * sizeof(uint32_t),
            tempBuffer.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
    }

    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        uint32_t whichIdx = loopIdx & blockTableReloadCheckMask;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * 64;
        subBlock = macroBlock + subBlock;
        offsetK = subBlock * kvHiddenDim + offsetBaseK;

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            __ESIMD_ENS::lsc_prefetch<
                uint8_t,
                128,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached
            >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
        }

#if 1
        {
            simd<uint32_t, 16> ksOffset;
            simd<float, 4> ksTemp;
            ksOffset = localLinearId + baseOffsetInc16AsVector * 16 + 64 * loopIdx;
            ksOffset.merge(0, ksOffset >= kvSeqLen);
            ksOffset = ksOffset * headKv * sizeof(float) + headIdxKv * sizeof(float);
            ksTemp =
                __ESIMD_ENS::lsc_gather<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                4,
                uint32_t
                >((float*)ks, ksOffset.select<4, 1>(0));
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                //        offsetK = __ESIMD_NS::min(offsetK, boundaryKv);
                if (loopIdx * 64 + kk * 16 + localLinearId >= kvSeqLen) {
                    offsetK = subBlock * kvHiddenDim;
                }
                i8KState.template bit_cast_view<uint8_t>().template select<128, 1>(128 * kk) =
                    __ESIMD_ENS::lsc_block_load<
                    uint8_t,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK);

                offsetK += 16 * kvHiddenDim * sizeof(int8_t);
            }
            ksOffset = baseOffsetInc16AsVector * 16 + localLinearId;
            ksOffset = ksOffset * sizeof(float) + slmOffsetBaseKs;
            __ESIMD_ENS::lsc_slm_scatter<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                4>(ksOffset.select<4, 1>(0), ksTemp.select<4, 1>(0));

            if (whichIdx == blockTableReloadCheckMask) {
                uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
                blockIdx =
                    __ESIMD_ENS::lsc_block_load<
                    uint32_t,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable);
            }

#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    slm_block_store<int8_t, 32>(slmOffsetK + k * 16 * 128 * sizeof(int8_t) + kk * 16 * 32 * sizeof(int8_t), i8KState.select<32, 1>(128 * k + 32 * kk));
                }
            }
        }
#else
#pragma unroll
        //16 x4  64 Ks
        for (int kk = 0; kk < 4; kk++) {
            uint32_t ksOffset = localLinearId + 16 * kk + 64 * loopIdx;
            fp16 ksTemp;
            ksTemp = ks[ksOffset];
            slm_block_store<fp16, 1>(slmOffsetBaseKs + localLinearId * sizeof(fp16) + 16 * kk * sizeof(fp16), ksTemp);
        }
        uint32_t kOffset_2d = (64 * loopIdx + (localLinearId / 4) * 16) * headKv * 128 /*token*/ + headIdxKv * 128 /*header*/ + (localLinearId % 4) * 32/*block*/;
        i8KState.select<512, 1>(0) = __ESIMD_ENS::lsc_load_2d<
            uint8_t, 32, 16, 1,  //8x32
            false, false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState,
                                             headKv * 128 - 1, kvSeqLen - 1, headKv * 128 - 1,
                                             headIdxKv * 128 + (localLinearId % 4) * 32, 64 * loopIdx + (localLinearId / 4) * 16);
        slm_block_store<int8_t, 512>(slmOffsetBaseK + (localLinearId / 2) * 2 * 512 + (localLinearId % 2) * 512, i8KState.select<512, 1>(0));
#endif
        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
        auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
        auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();
        barrier();
        i32TempBuffer = 0;

        {
            simd<float, 64> ksAllTemp = slm_block_load<float, 64>(slmOffsetBaseKs);
#pragma unroll
            for (int32_t nn = 0; nn < 4; nn++) {
#pragma unroll
                for (int32_t l = 0; l < 2; l++) {
#pragma unroll
                    for (int32_t ll = 0; ll < 2; ll++) {
                        i8TempBuffer.select<512, 1>(512 * ll) = slm_block_load<int8_t, 512>(
                            slmOffsetBaseK + 16 * 128 * nn * sizeof(int8_t) +
                            1024 * l * sizeof(int8_t) +
                            512 * ll * sizeof(int8_t));
                    }

#pragma unroll
                    for (int32_t ll = 0; ll < 2; ll++) {
                        i8TempBuffer.select<512, 1>(1024 + 512 * ll) = slm_block_load<int8_t, 512>(
                            slmOffsetBaseQ + localLinearId * 16 * 128 * sizeof(int8_t) +
                            1024 * l * sizeof(int8_t) +
                            512 * ll * sizeof(int8_t));
                    }

#pragma unroll
                    for (int32_t k = 0; k < 2; k++) {
#pragma unroll
                        for (int32_t kk = 0; kk < 2; kk++) {
                            auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
                            auto aaTile = i8TempBuffer.template bit_cast_view<int32_t>().select<128, 1>(256 + 128 * k);
                            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk);

                            ccTile =
                                dpas
                                <8, 8, int32_t, int32_t, int32_t, int32_t,
                                dpas_argument_type::s8,
                                dpas_argument_type::s8
                                >(
                                    simd<int32_t, 128>(ccTile.data()),
                                    simd<int32_t, 128>(aaTile.data()),
                                    simd<int32_t, 64>(bbTile.data())
                                );
                        }
                    }
                }
#pragma unroll
                for (int32_t kk = 0; kk < 16; kk++) {
                    simd<float, 16> fp32Temp0;
                    simd<float, 16> fp32Temp1;
                    fp32Temp0.select<16, 1>(0) = ksAllTemp[16 * nn + kk] * fp32Qs.select<16, 1>(0);
                    fp32Temp1.select<16, 1>(0) = i32TempBuffer.select<16, 1>(256 * nn + 16 * kk);
#ifdef MERGE_QUANT
                    tempOutput.select<16, 1>(256 * nn + 16 * kk) = fp32Temp0 * fp32Temp1 * matMulQuantCoeff;
#else
                    tempOutput.select<16, 1>(256 * nn + 16 * kk) = fp32Temp0 * fp32Temp1
#endif
                }
            }

            if (whichIdx == blockTableReloadCheckMask) {
                offsetK = blockIdx[0] * kvHiddenDim * blockSize + offsetBaseK;
            }

            uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;

            if (singpagelastIdx < singlePageMaxMask)
            {
#if 1
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        128,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
                }
#endif
            }

        }

#ifdef QKV_INT8
        //    vsTemp = vs[loopIdx];
        simd<uint32_t, 64> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
        simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 64 * loopIdx;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
            simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * kk;
            simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim * sizeof(int8_t) +
                subBlock * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 128) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                32,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32));

            simd32Offsets00.select<32, 1>(0) += 32;
        }
#else
        simd<uint32_t, 32> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk;
            simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * kvHiddenDim * sizeof(int8_t) +
                subBlock * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 128) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16));
            simd32Offsets00.select<16, 1>(0) += 16;
        }
#endif

        {
            //   auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
            //   auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
            auto softmaxPositions = ui32Temp.select<64, 1>(64);

            //  fp32SoftMaxTemp = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
            //  fp32HistoricMaxTemp = slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
#if  VERSION_0

            softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
                }
                softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
            }


#else
            if (loopIdx >= myLastFullAttnLoopIdx) {
                softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
                softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
                softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
                }
            }
#endif
#ifndef MERGE_QUANT
            tempOutput.select<16 * 64, 1>(0) = tempOutput.select<16 * 64, 1>(0) * matMulQuantCoeff;

#endif
            fp32CurrentMaxTemp = fp32HistoricMaxTemp;

#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * kk), fp32CurrentMaxTemp);
            }

#else
            simd<float, 8 * 16> ttemp;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32 * kk), tempOutput.select<16, 1>(32 * kk + 16));
            }
#pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16 * kk), tempOutput.select<16, 1>((8 * kkk + kk) * 16 + 16 * 16));
                }
            }
            ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
            ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
            ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
#if  VERSION_0// Positive 
            fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), fp32CurrentMaxTemp);
#else
            fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);
#endif

#endif

#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#elif 1  // Positive 
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#else
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#endif

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
#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + tempOutput.select<16, 1>(16 * kk);
            }
#else  
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(32 * kk) + tempOutput.select<16, 1>(32 * kk + 16);
            }
#pragma unroll
            for (int k = 0; k < 6; ++k) {
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
            // slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
            // slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
            // slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#ifdef QKV_INT8
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {

                simd<fp16, 64> shuffleTemp;
#if  0
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f;

                shuffleTemp = shuffleTemp + 0.5f;
#else
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * (fp16)255.0f + (fp16)0.5f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * (fp16)255.0f + (fp16)0.5f;

#endif
                tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
                tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
            }

#pragma unroll
            for (int k = 0; k < 2; k++) {
                simd<int8_t, 256> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
                }
                slm_block_store<int32_t, 64>(slmOffsetV + k * 128 * 32 * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
            }
#else
#pragma unroll
            for (int kk = 0; kk < 32; kk++) {
                simd<float, 32> shuffleTemp;
                shuffleTemp = tempOutput.select<32, 1>(32 * kk);
                tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
                tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
            }

            simd<fp16, 128> fp16Qv;

#pragma unroll
            for (int k = 0; k < 4; k++) {
                fp16Qv.select<32, 1>(32 * k) = fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(2 * k);
            }
#pragma unroll
            for (int k = 0; k < 4; k++) {
                simd<fp16, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<16, 1>(64 * kk + 0) = i8VState.select<16, 4>(128 * k + 64 * kk + 0);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 1) = i8VState.select<16, 4>(128 * k + 64 * kk + 1);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 2) = i8VState.select<16, 4>(128 * k + 64 * kk + 2);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 3) = i8VState.select<16, 4>(128 * k + 64 * kk + 3);
                }

#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
                }
                slm_block_store<fp16, 128>(slmOffsetV + k * 128 * 16 * sizeof(fp16), shuffleTemp);
            }
#endif

            tempBuffer.select<16, 1>(0) = fp32SoftMaxCompensation;
            tempBuffer.select<16, 1>(16) = tempBuffer.select<16, 1>(0);
            if (loopIdx != 0) {
                simd<fp16, 32> fp16Temp0;
                fp16Temp0 = tempBuffer.select<32, 1>(0);
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * fp16Temp0;
                }
            }
        }

        barrier();
#ifdef QKV_INT8
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            simd<float, 32> fp32QvTemp = slm_block_load<float, 32>(slmOffsetBaseVs + nn * 32 * sizeof(float));
#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 2; ll++) {
                    i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseV +
                        32 * 128 * l * sizeof(int8_t) +
                        32 * 32 * nn * sizeof(int8_t) +
                        ll * 512 * sizeof(int8_t));
                }
            }
#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                        simd<uint32_t, 128>(aaTile.data()),
                        simd<int32_t, 64>(bbTile.data())
                    );
            }

#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                        simd<int32_t, 128>(ccTile.data()),
                        simd<uint32_t, 128>(aaTile.data()),
                        simd<int32_t, 64>(bbTile.data())
                    );
            }

            //float vsTempFp32 = vsTemp * (1.0f / 255.0f);
#pragma unroll
            for (int32_t kk = 0; kk < 32; kk++) {
                simd<float, 16> fp32Temp1;
                fp32Temp1 = i32TempBuffer.select<16, 1>(16 * kk + 512);
                finalOutput.select<16, 1>(16 * 32 * nn + 16 * kk) += fp32Temp1.select<16, 1>(0) * fp32QvTemp[kk];
            }
        }
#else
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 4; ll++) {
                    fp16TempBuffer.select<256, 1>(256 * ll) = slm_block_load<fp16, 256>(slmOffsetBaseV +
                                                                                        16 * 128 * nn * sizeof(fp16) +
                                                                                        1024 * l * sizeof(fp16) +
                                                                                        ll * 256 * sizeof(fp16));
                }

#pragma unroll
                for (int ll = 0; ll < 8; ll++) {
                    auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                    auto aaTile = tempQkFp16.select<256, 1>(256 * nn);
                    auto bbTile = fp16TempBuffer.select<128, 1>(128 * ll);

                    ccTile = dpas<8, 8, float, float, fp16, fp16>(
                        simd<float, 128>(ccTile.data()),
                        simd<fp16, 256>(aaTile.data()),
                        simd<fp16, 128>(bbTile.data()));
                }
            }
        }
#endif
        //barrier();
    }

    simd<fp16, 16 * 128> finalOutputFp16;

    simd<float, 32> softMaxDividor;
    simd_mask<16> mask;
    //softMaxDividor.select<16, 1>(0) = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
    softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
    softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
    softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
    for (int kk = 0; kk < 64; kk++) {
        simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
        f16Temp = f16Temp * softMaxDividor;
        finalOutputFp16.select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
        finalOutputFp16.select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 64 * h + 16 * hhq;
    mask = simdOffsets <= boundaryQ;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        __ESIMD_ENS::lsc_scatter<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back,
            16,
            uint32_t
        >((uint32_t*)out, simdOffsets, finalOutputFp16.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk), mask);
        simdOffsets += 8 * sizeof(uint32_t);
    }
}

template <uint32_t GPA_DIV_2>
ESIMD_INLINE void sageAttnGqa_2X(
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
    uint32_t headQ,
    uint32_t headKv,
    sycl::nd_item<2>& ndi
) {
    constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
    constexpr uint32_t slmSizeK = 64 * 128 * sizeof(int8_t);
    constexpr uint32_t slmSizeV = 64 * 128 * sizeof(fp16);
    constexpr uint32_t slmSizeSoftMaxSum = 256 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxHistoric = 256 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxCompensation = 256 * sizeof(float);
    constexpr uint32_t slmSizeKs = 64 * sizeof(float);
    constexpr uint32_t slmSizeVs = 128 * sizeof(float);
    constexpr uint32_t slmSizeQ = 16 * 128 * 16 * sizeof(int8_t);
    constexpr uint32_t slmSize =
        slmSizeK +
        slmSizeV +
        slmSizeSoftMaxSum +
        slmSizeSoftMaxHistoric +
        slmSizeSoftMaxCompensation +
        slmSizeKs +
        slmSizeVs +
        slmSizeQ;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
    constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
    constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
    constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
    constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
    constexpr uint32_t slmOffsetBaseVs = slmOffsetBaseKs + slmSizeKs;
    constexpr uint32_t slmOffsetBaseQ = slmOffsetBaseVs + slmSizeVs;
    constexpr int32_t batchIdx = 0; // v / headKv;
    constexpr int32_t groupGqaRatio = 4;
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
    constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;


    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq = localLinearId & 0x7;
    int32_t vvq = localLinearId >> 3;
    int32_t hhv = localLinearId & 0xf;
    int32_t vvv = localLinearId >> 4;
    int32_t h = ndi.get_group(0);
    int32_t v = ndi.get_group(1);
    int32_t headIdxQ = v * 2;
    int32_t headIdxKv = v / GPA_DIV_2;
    uint32_t kvSeqLen;
    uint32_t activationLength;
    uint32_t startActivationIdx;
    simd<uint32_t, 2> kvCoord;
    simd<uint32_t, 2> queryCoord;
    kvCoord = block_load<uint32_t, 2>(cuSeqKv);
    queryCoord = block_load<uint32_t, 2>(cuSeqQuery);
    //if (kvSeqLen >= activationLength) {
    //  startActivationIdx = kvSeqLen - activationLength;
    //} else {
    //  startActivationIdx = 0;
    //}
    headIdxQ = headIdxQ + vvq;

    //sycl::ext::oneapi::experimental::printf("h = %d, v = %d, headIdxQ %d\n", h, v, headIdxQ);

    //simd<int8_t, 16 * 128> i8QState;
    simd<float, 16> fp32Qs;
    simd<int32_t, 16 * 64> i32TempBuffer;
    //  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
    simd<float, 640> tempBuffer;
    auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
    auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 4);
    auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
    auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
    auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
    simd<float, 16 * 128> finalOutput = 0;
    simd<float, 8> fp32Qv;
    simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
    simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    simd<uint32_t, 16> blockIdx;

    kvSeqLen = kvCoord[1] - kvCoord[0];
    activationLength = queryCoord[1] - queryCoord[0];
    startActivationIdx = kvSeqLen - activationLength;

    int32_t kvSeqOutLoopCount = 128 * h + 128 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 63) >> 6;
    uint32_t myLastFullAttnLoopIdx = (128 * h + startActivationIdx + 16 * hhq) >> 6;
    unsigned int offsetQBase = headIdxQ * 128 * sizeof(int8_t);
    unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
    unsigned int kvHiddenDim = headKv * 128;
    unsigned int offsetBaseK = (headIdxKv * 128 + localLinearId * headKv * 128) * sizeof(int8_t);
    unsigned int offsetK;
    unsigned int slmOffsetK = slmOffsetBaseK + localLinearId * 32 * sizeof(int8_t);
    unsigned int slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16);
    unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
    unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

    unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

    uint32_t boundaryQ = activationLength - 1;

    if (128 * h >= activationLength) {
        return;
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 128 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 128 * sizeof(int8_t) + offsetQBase;

#pragma unroll
    for (int k = 0; k < 4; k++) {
        tempBuffer.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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
    constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff;
    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 128 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * sizeof(float) + headIdxQ * sizeof(float);
    fp32Qs.select<16, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        float,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((float*)qs, simdOffsets);

    softMaxThresh = softMaxThresh + 16 * hhq + 128 * h + startActivationIdx;

    simd<float, 16> fp32SoftMaxTemp = 0.0f;
    simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    blockIdx = __ESIMD_ENS::lsc_block_load<
        uint32_t,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + 0);

    fp32Qv.select<8, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        8,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 128 * headIdxKv + 8 * hhv);
#ifdef QKV_INT8
    fp32Qv = fp32Qv * (1.0f / 255.0f);
    slm_block_store<float, 8>(slmOffsetBaseVs + 8 * hhv * sizeof(float), fp32Qv);
#endif

#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
        slm_block_store<uint32_t, 128>(
            slmOffsetBaseQ + localLinearId * 16 * 128 * sizeof(int8_t) + kk * 128 * sizeof(uint32_t),
            tempBuffer.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
    }

    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        uint32_t whichIdx = loopIdx & blockTableReloadCheckMask;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * 64;
        subBlock = macroBlock + subBlock;
        offsetK = subBlock * kvHiddenDim + offsetBaseK;

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            __ESIMD_ENS::lsc_prefetch<
                uint8_t,
                128,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached
            >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
        }

#if 1
        {
            simd<uint32_t, 16> ksOffset;
            simd<float, 4> ksTemp;
            ksOffset = localLinearId + baseOffsetInc16AsVector * 16 + 64 * loopIdx;
            ksOffset.merge(0, ksOffset >= kvSeqLen);
            ksOffset = ksOffset * headKv * sizeof(float) + headIdxKv * sizeof(float);
            ksTemp =
                __ESIMD_ENS::lsc_gather<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                4,
                uint32_t
                >((float*)ks, ksOffset.select<4, 1>(0));
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                //        offsetK = __ESIMD_NS::min(offsetK, boundaryKv);
                if (loopIdx * 64 + kk * 16 + localLinearId >= kvSeqLen) {
                    offsetK = subBlock * kvHiddenDim;
                }
                i8KState.template bit_cast_view<uint8_t>().template select<128, 1>(128 * kk) =
                    __ESIMD_ENS::lsc_block_load<
                    uint8_t,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK);

                offsetK += 16 * kvHiddenDim * sizeof(int8_t);
            }
            ksOffset = baseOffsetInc16AsVector * 16 + localLinearId;
            ksOffset = ksOffset * sizeof(float) + slmOffsetBaseKs;
            __ESIMD_ENS::lsc_slm_scatter<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                4>(ksOffset.select<4, 1>(0), ksTemp.select<4, 1>(0));

            if (whichIdx == blockTableReloadCheckMask) {
                uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
                blockIdx =
                    __ESIMD_ENS::lsc_block_load<
                    uint32_t,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable);
            }

#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    slm_block_store<int8_t, 32>(slmOffsetK + k * 16 * 128 * sizeof(int8_t) + kk * 16 * 32 * sizeof(int8_t), i8KState.select<32, 1>(128 * k + 32 * kk));
                }
            }
        }
#else
#pragma unroll
        //16 x4  64 Ks
        for (int kk = 0; kk < 4; kk++) {
            uint32_t ksOffset = localLinearId + 16 * kk + 64 * loopIdx;
            fp16 ksTemp;
            ksTemp = ks[ksOffset];
            slm_block_store<fp16, 1>(slmOffsetBaseKs + localLinearId * sizeof(fp16) + 16 * kk * sizeof(fp16), ksTemp);
        }
        uint32_t kOffset_2d = (64 * loopIdx + (localLinearId / 4) * 16) * headKv * 128 /*token*/ + headIdxKv * 128 /*header*/ + (localLinearId % 4) * 32/*block*/;
        i8KState.select<512, 1>(0) = __ESIMD_ENS::lsc_load_2d<
            uint8_t, 32, 16, 1,  //8x32
            false, false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState,
                                             headKv * 128 - 1, kvSeqLen - 1, headKv * 128 - 1,
                                             headIdxKv * 128 + (localLinearId % 4) * 32, 64 * loopIdx + (localLinearId / 4) * 16);
        slm_block_store<int8_t, 512>(slmOffsetBaseK + (localLinearId / 2) * 2 * 512 + (localLinearId % 2) * 512, i8KState.select<512, 1>(0));
#endif
        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
        auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
        auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();
        barrier();
        i32TempBuffer = 0;

        {
            simd<float, 64> ksAllTemp = slm_block_load<float, 64>(slmOffsetBaseKs);
#pragma unroll
            for (int32_t nn = 0; nn < 4; nn++) {
#pragma unroll
                for (int32_t l = 0; l < 2; l++) {
#pragma unroll
                    for (int32_t ll = 0; ll < 2; ll++) {
                        i8TempBuffer.select<512, 1>(512 * ll) = slm_block_load<int8_t, 512>(
                            slmOffsetBaseK + 16 * 128 * nn * sizeof(int8_t) +
                            1024 * l * sizeof(int8_t) +
                            512 * ll * sizeof(int8_t));
                    }

#pragma unroll
                    for (int32_t ll = 0; ll < 2; ll++) {
                        i8TempBuffer.select<512, 1>(1024 + 512 * ll) = slm_block_load<int8_t, 512>(
                            slmOffsetBaseQ + localLinearId * 16 * 128 * sizeof(int8_t) +
                            1024 * l * sizeof(int8_t) +
                            512 * ll * sizeof(int8_t));
                    }

#pragma unroll
                    for (int32_t k = 0; k < 2; k++) {
#pragma unroll
                        for (int32_t kk = 0; kk < 2; kk++) {
                            auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
                            auto aaTile = i8TempBuffer.template bit_cast_view<int32_t>().select<128, 1>(256 + 128 * k);
                            auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk);

                            ccTile =
                                dpas
                                <8, 8, int32_t, int32_t, int32_t, int32_t,
                                dpas_argument_type::s8,
                                dpas_argument_type::s8
                                >(
                                    simd<int32_t, 128>(ccTile.data()),
                                    simd<int32_t, 128>(aaTile.data()),
                                    simd<int32_t, 64>(bbTile.data())
                                );
                        }
                    }
                }
#pragma unroll
                for (int32_t kk = 0; kk < 16; kk++) {
                    simd<float, 16> fp32Temp0;
                    simd<float, 16> fp32Temp1;
                    fp32Temp0.select<16, 1>(0) = ksAllTemp[16 * nn + kk] * fp32Qs.select<16, 1>(0);
                    fp32Temp1.select<16, 1>(0) = i32TempBuffer.select<16, 1>(256 * nn + 16 * kk);
#ifdef MERGE_QUANT
                    tempOutput.select<16, 1>(256 * nn + 16 * kk) = fp32Temp0 * fp32Temp1 * matMulQuantCoeff;
#else
                    tempOutput.select<16, 1>(256 * nn + 16 * kk) = fp32Temp0 * fp32Temp1
#endif
                }
            }

            if (whichIdx == blockTableReloadCheckMask) {
                offsetK = blockIdx[0] * kvHiddenDim * blockSize + offsetBaseK;
            }

            uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;

            if (singpagelastIdx < singlePageMaxMask)
            {
#if 1
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        128,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
                }
#endif
            }

        }

#ifdef QKV_INT8
        //    vsTemp = vs[loopIdx];
        simd<uint32_t, 64> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
        simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 64 * loopIdx;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
            simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * kk;
            simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim * sizeof(int8_t) +
                subBlock * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 128) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                32,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32));

            simd32Offsets00.select<32, 1>(0) += 32;
        }
#else
        simd<uint32_t, 32> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk;
            simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * kvHiddenDim * sizeof(int8_t) +
                subBlock * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 128) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16));
            simd32Offsets00.select<16, 1>(0) += 16;
        }
#endif

        {
            //   auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
            //   auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
            auto softmaxPositions = ui32Temp.select<64, 1>(64);

            //  fp32SoftMaxTemp = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
            //  fp32HistoricMaxTemp = slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
#if  VERSION_0

            softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
#pragma unroll
            for (int k = 0; k < 4; k++) {
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
                }
                softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
            }


#else
            if (loopIdx >= myLastFullAttnLoopIdx) {
                softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
                softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
                softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
                }
            }
#endif
#ifndef MERGE_QUANT
            tempOutput.select<16 * 64, 1>(0) = tempOutput.select<16 * 64, 1>(0) * matMulQuantCoeff;

#endif
            fp32CurrentMaxTemp = fp32HistoricMaxTemp;

#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * kk), fp32CurrentMaxTemp);
            }

#else
            simd<float, 8 * 16> ttemp;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32 * kk), tempOutput.select<16, 1>(32 * kk + 16));
            }
#pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16 * kk), tempOutput.select<16, 1>((8 * kkk + kk) * 16 + 16 * 16));
                }
            }
            ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
            ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
            ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
#if  VERSION_0// Positive 
            fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), fp32CurrentMaxTemp);
#else
            fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);
#endif

#endif

#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#elif 1  // Positive 
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#else
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
            }
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
            }
#endif

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
#if VERSION_0
#pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + tempOutput.select<16, 1>(16 * kk);
            }
#else  
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(32 * kk) + tempOutput.select<16, 1>(32 * kk + 16);
            }
#pragma unroll
            for (int k = 0; k < 6; ++k) {
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
            // slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
            // slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
            // slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#ifdef QKV_INT8
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {

                simd<fp16, 64> shuffleTemp;
#if  0
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f;

                shuffleTemp = shuffleTemp + 0.5f;
#else
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * (fp16)255.0f + (fp16)0.5f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * (fp16)255.0f + (fp16)0.5f;

#endif

                tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
                tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
            }

#pragma unroll
            for (int k = 0; k < 2; k++) {
                simd<int8_t, 256> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
                }
                slm_block_store<int32_t, 64>(slmOffsetV + k * 128 * 32 * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
            }
#else
#pragma unroll
            for (int kk = 0; kk < 32; kk++) {
                simd<float, 32> shuffleTemp;
                shuffleTemp = tempOutput.select<32, 1>(32 * kk);
                tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
                tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
            }

            simd<fp16, 128> fp16Qv;

#pragma unroll
            for (int k = 0; k < 4; k++) {
                fp16Qv.select<32, 1>(32 * k) = fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(2 * k);
            }
#pragma unroll
            for (int k = 0; k < 4; k++) {
                simd<fp16, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<16, 1>(64 * kk + 0) = i8VState.select<16, 4>(128 * k + 64 * kk + 0);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 1) = i8VState.select<16, 4>(128 * k + 64 * kk + 1);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 2) = i8VState.select<16, 4>(128 * k + 64 * kk + 2);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 3) = i8VState.select<16, 4>(128 * k + 64 * kk + 3);
                }

#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
                }
                slm_block_store<fp16, 128>(slmOffsetV + k * 128 * 16 * sizeof(fp16), shuffleTemp);
            }
#endif

            tempBuffer.select<16, 1>(0) = fp32SoftMaxCompensation;
            tempBuffer.select<16, 1>(16) = tempBuffer.select<16, 1>(0);
            if (loopIdx != 0) {
                simd<fp16, 32> fp16Temp0;
                fp16Temp0 = tempBuffer.select<32, 1>(0);
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * fp16Temp0;
                }
            }
        }

        barrier();
#ifdef QKV_INT8
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            simd<float, 32> fp32QvTemp = slm_block_load<float, 32>(slmOffsetBaseVs + nn * 32 * sizeof(float));
#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 2; ll++) {
                    i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseV +
                        32 * 128 * l * sizeof(int8_t) +
                        32 * 32 * nn * sizeof(int8_t) +
                        ll * 512 * sizeof(int8_t));
                }
            }
#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                        simd<uint32_t, 128>(aaTile.data()),
                        simd<int32_t, 64>(bbTile.data())
                    );
            }

#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
                auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);
                auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * ll);

                ccTile =
                    dpas
                    <8, 8, int32_t, int32_t, uint32_t, int32_t,
                    dpas_argument_type::u8,
                    dpas_argument_type::s8
                    >(
                        simd<int32_t, 128>(ccTile.data()),
                        simd<uint32_t, 128>(aaTile.data()),
                        simd<int32_t, 64>(bbTile.data())
                    );
            }

            //float vsTempFp32 = vsTemp * (1.0f / 255.0f);
#pragma unroll
            for (int32_t kk = 0; kk < 32; kk++) {
                simd<float, 16> fp32Temp1;
                fp32Temp1 = i32TempBuffer.select<16, 1>(16 * kk + 512);
                finalOutput.select<16, 1>(16 * 32 * nn + 16 * kk) += fp32Temp1.select<16, 1>(0) * fp32QvTemp[kk];
            }
        }
#else
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 4; ll++) {
                    fp16TempBuffer.select<256, 1>(256 * ll) = slm_block_load<fp16, 256>(slmOffsetBaseV +
                                                                                        16 * 128 * nn * sizeof(fp16) +
                                                                                        1024 * l * sizeof(fp16) +
                                                                                        ll * 256 * sizeof(fp16));
                }

#pragma unroll
                for (int ll = 0; ll < 8; ll++) {
                    auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                    auto aaTile = tempQkFp16.select<256, 1>(256 * nn);
                    auto bbTile = fp16TempBuffer.select<128, 1>(128 * ll);

                    ccTile = dpas<8, 8, float, float, fp16, fp16>(
                        simd<float, 128>(ccTile.data()),
                        simd<fp16, 256>(aaTile.data()),
                        simd<fp16, 128>(bbTile.data()));
                }
            }
        }
#endif
        //barrier();
    }

    simd<fp16, 16 * 128> finalOutputFp16;

    simd<float, 32> softMaxDividor;
    simd_mask<16> mask;
    //softMaxDividor.select<16, 1>(0) = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
    softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
    softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
    softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
    for (int kk = 0; kk < 64; kk++) {
        simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
        f16Temp = f16Temp * softMaxDividor;
        finalOutputFp16.select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
        finalOutputFp16.select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 128 * h + 16 * hhq;
    mask = simdOffsets <= boundaryQ;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        __ESIMD_ENS::lsc_scatter<
            uint32_t,
            8,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back,
            16,
            uint32_t
        >((uint32_t*)out, simdOffsets, finalOutputFp16.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk), mask);
        simdOffsets += 8 * sizeof(uint32_t);
    }
}

// Shape Q [activation token length, 96, 64] INT8,
// Shape K:  [kv len, 8, 64] INT8,
// Shape V : [kv len, 8, 64] INT8,
// output: [activation token length, 96, 64] FP16,
template <uint32_t GPA_DIV_4>
ESIMD_INLINE void sageAttnGqa_4X_64(
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
    uint32_t headQ,
    uint32_t headKv,
    nd_item<2>& ndi
) {
    constexpr float matMulQuantCoeff = 0.125f; // 1.0f / sqrt(64.0f);
    constexpr uint32_t slmSizeK = 64 * 64 * sizeof(int8_t);
    constexpr uint32_t slmSizeV = 64 * 64 * sizeof(fp16);
    constexpr uint32_t slmSizeSoftMaxSum = 512 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxHistoric = 512 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxCompensation = 512 * sizeof(float);
    constexpr uint32_t slmSizeKs = 64 * sizeof(float);
    //  constexpr uint32_t slmSizeQ = 32 * 64 * 16 * sizeof(int8_t);
    constexpr uint32_t slmSizeVs = 64 * sizeof(float);
    constexpr uint32_t slmSize =
        slmSizeK +
        slmSizeV +
        slmSizeSoftMaxSum +
        slmSizeSoftMaxHistoric +
        slmSizeSoftMaxCompensation +
        slmSizeKs +
        slmSizeVs;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
    constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
    constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
    constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
    constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
    constexpr uint32_t slmOffsetBaseVs = slmOffsetBaseKs + slmSizeKs;
    constexpr int32_t batchIdx = 0; // v / headKv;
    constexpr int32_t groupGqaRatio = 4;
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
    constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;
    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq = localLinearId & 0x7;
    int32_t vvq = localLinearId >> 3;
    int32_t hhk = localLinearId & 0x7; // [0, 4)
    int32_t vvk = localLinearId >> 3; // [0, 4)
    int32_t hhv;
    int32_t vvv;
#ifdef QKV_INT8
    hhv = localLinearId & 0xf;
    vvv = localLinearId >> 4;
#else
    hhv = localLinearId & 0x7;
    vvv = localLinearId >> 3;
#endif
    int32_t h = ndi.get_group(0);
    int32_t v = ndi.get_group(1); // [0, 24)
    int32_t headIdxQ = v * 4;
    //int32_t gqaRatio = (headQ / headKv) >> 2;
    int32_t headIdxKv = v / GPA_DIV_4;
    uint32_t kvSeqLen;
    uint32_t activationLength;
    uint32_t startActivationIdx;
    simd<uint32_t, 2> kvCoord;
    simd<uint32_t, 2> queryCoord;
    kvCoord = block_load<uint32_t, 2>(cuSeqKv);
    queryCoord = block_load<uint32_t, 2>(cuSeqQuery);
    headIdxQ = headIdxQ + vvq;
    simd<int8_t, 16 * 64> i8QState;
    simd<float, 16> fp32Qs;
    simd<int32_t, 16 * 64> i32TempBuffer;
    //  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
    simd<float, 512> tempBuffer;
    simd<int8_t, 128> i8VState;
    auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<128, 1>(0);
    auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
    auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
    auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
    simd<float, 16 * 64> finalOutput = 0;
    simd<float, 8> fp32Qv;
    simd<fp16, 128> fp16Qv;
    fp16 vsTemp = 1.0f;
    simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    simd<uint32_t, 32> softMaxThresh(baseOffsetInc16);
    simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    simd<uint32_t, 16> blockIdx;

    kvSeqLen = kvCoord[1] - kvCoord[0];
    activationLength = queryCoord[1] - queryCoord[0];
    startActivationIdx = kvSeqLen - activationLength;

    int32_t kvSeqOutLoopCount = 128 * h + 128 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 63) >> 6;
    uint32_t myLastFullAttnLoopIdx = (128 * h + startActivationIdx + 16 * hhq) >> 6;
    unsigned int offsetQBase = headIdxQ * 64 * sizeof(int8_t);
    unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
    unsigned int kvHiddenDim = headKv * 64;
    unsigned int offsetBasePref = (headIdxKv * 64 + localLinearId * headKv * 64) * sizeof(int8_t);
    unsigned int slmOffsetK = slmOffsetBaseK + (hhk & 0x3) * 2 * sizeof(uint32_t) + (hhk >> 2) * 64 * 32 * sizeof(uint8_t) + vvk * 16 * 8 * sizeof(uint32_t);
    unsigned int slmOffsetV;
#ifdef QKV_INT8
    slmOffsetV = slmOffsetBaseV + localLinearId * 128 * sizeof(int8_t);
#else
    slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16) + vvv * 16 * 64 * sizeof(fp16);
#endif
    unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
    unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
    //  unsigned int slmOffsetC0 = slmOffsetBaseV;
    unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
    //  uint32_t boundaryKv = (headIdxKv * 128 + (kvSeqLen - 1) * headKv * 128) * sizeof(int8_t);
    uint32_t boundaryQ = activationLength - 1;

    if (128 * h >= activationLength) {
        return;
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 128 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 64 * sizeof(int8_t) + offsetQBase;

#pragma unroll
    for (int k = 0; k < 2; k++) {
        i8QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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

    //  constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff;
    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 128 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * sizeof(float) + headIdxQ * sizeof(float);
    fp32Qs.select<16, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        float,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((float*)qs, simdOffsets);

    //  simd<float, 16> fp32SoftMaxTemp = 0.0f;
    //  simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    blockIdx = __ESIMD_ENS::lsc_block_load<
        uint32_t,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + 0);

#ifdef QKV_INT8
    fp32Qv.select<4, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        4,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 64 * headIdxKv + 4 * hhv);
#else
    fp32Qv.select<8, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        8,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 64 * headIdxKv + 8 * hhv);
#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
        fp16Qv.select<32, 1>(32 * kk) = fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
    }
#endif

    {
#ifdef QKV_INT8
        if (vvv == 0) {
            fp32Qv.select<4, 1>(0) = fp32Qv.select<4, 1>(0) * (1.0f / 255.0f);
            slm_block_store<float, 4>(slmOffsetBaseVs + hhv * 4 * sizeof(float), fp32Qv.select<4, 1>(0));
        }
#else
#endif
        simd<float, 16> slmInitTemp = 0;
        slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), slmInitTemp);
        slmInitTemp = FP32_MIN;
        slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, slmInitTemp);
    }

    softMaxThresh.select<16, 1>(0) = baseOffsetInc16AsVector;
    softMaxThresh = softMaxThresh + 16 * hhq + 128 * h + startActivationIdx;
    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        uint32_t whichIdx = loopIdx & blockTableReloadCheckMask;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift];
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask);
        uint32_t blockOffset = macroBlock * blockSize + subBlock * 64;
        uint32_t offsetPref = blockOffset * kvHiddenDim + offsetBasePref;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            __ESIMD_ENS::lsc_prefetch<
                uint8_t,
                64,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached
            >((uint8_t*)vState + offsetPref + kk * 32 * kvHiddenDim * sizeof(int8_t));
        }

        {
            simd<uint32_t, 16> ksOffset;
            simd<uint32_t, 16> coordK;
            simd<float, 2> ksTemp;
            ksOffset = localLinearId + baseOffsetInc16AsVector * 32 + 64 * loopIdx;
            ksOffset.merge(0, ksOffset >= kvSeqLen);
            ksOffset = ksOffset * headKv * sizeof(float) + headIdxKv * sizeof(float);
            ksTemp =
                __ESIMD_ENS::lsc_gather<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                2,
                uint32_t
                >((float*)ks, ksOffset.select<2, 1>(0));
            coordK.select<16, 1>(0) = baseOffsetInc16AsVector;
            coordK.select<16, 1>(0) = coordK.select<16, 1>(0) + 64 * loopIdx + vvk * 16;
            simdOffsets.select<16, 1>(0) = baseOffsetInc16AsVector + vvk * 16 + subBlock * 64;
            simdOffsets.merge(0, coordK >= kvSeqLen);
            simdOffsets = simdOffsets + macroBlock * blockSize;
            simdOffsets = simdOffsets * kvHiddenDim * sizeof(int8_t) +
                headIdxKv * 64 * sizeof(int8_t) +
                hhk * 2 * sizeof(uint32_t);

            i8KState.template bit_cast_view<uint32_t>().select<32, 1>(0) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)kState, simdOffsets.select<16, 1>(0));

            ksOffset = baseOffsetInc16AsVector * 32 + localLinearId;
            ksOffset = ksOffset * sizeof(float) + slmOffsetBaseKs;
            __ESIMD_ENS::lsc_slm_scatter<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                2>(ksOffset.select<2, 1>(0), ksTemp.select<2, 1>(0));

            if (whichIdx == blockTableReloadCheckMask) {
                uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
                blockIdx =
                    __ESIMD_ENS::lsc_block_load<
                    uint32_t,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable);
            }

            coordK = baseOffsetInc16AsVector * 8 * sizeof(uint32_t) + slmOffsetK;
            __ESIMD_ENS::lsc_slm_scatter<
                int32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                16>(coordK.select<16, 1>(0), i8KState.template bit_cast_view<int32_t>().select<32, 1>(0));
        }

        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
        auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
        auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();

        barrier();
        i32TempBuffer = 0;

        {
            simd<float, 64> ksAllTemp = slm_block_load<float, 64>(slmOffsetBaseKs);
#pragma unroll
            for (int32_t nn = 0; nn < 2; nn++) {
#pragma unroll
                for (int32_t ll = 0; ll < 4; ll++) {
                    i8TempBuffer.select<512, 1>(512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseK +
                        32 * 64 * nn * sizeof(int8_t) +
                        512 * ll * sizeof(int8_t));
                }

#pragma unroll
                for (int32_t kk = 0; kk < 8; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(128 * kk);
                    auto aaTile = i8QState.template bit_cast_view<int32_t>().select<128, 1>(128 * nn);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * kk);

                    ccTile =
                        dpas
                        <8, 8, int32_t, int32_t, int32_t, int32_t,
                        dpas_argument_type::s8,
                        dpas_argument_type::s8
                        >(
                            simd<int32_t, 128>(ccTile.data()),
                            simd<int32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
    }
}

            simd<float, 32> fp32TempQs0;
            fp32TempQs0.select<16, 1>(0) = fp32Qs.select<16, 1>(0);
            fp32TempQs0.select<16, 1>(16) = fp32Qs.select<16, 1>(0);

#pragma unroll
            for (int32_t kk = 0; kk < 32; kk++) {
                simd<float, 32> fp32Temp0;
                simd<float, 32> fp32Temp1;
                simd<float, 32> fp32TempOuter0;

                fp32TempOuter0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
                fp32Temp0 = i32TempBuffer.select<32, 1>(32 * kk) * matMulQuantCoeff;
                fp32Temp1 = fp32TempOuter0 * fp32TempQs0;
                tempOutput.select<32, 1>(32 * kk) = fp32Temp0 * fp32Temp1;
            }

            if (whichIdx == blockTableReloadCheckMask) {
                offsetPref = blockIdx[0] * kvHiddenDim * blockSize + offsetBasePref;
            }

            uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;
            if (singpagelastIdx < singlePageMaxMask) {
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        64,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetPref + kk * 32 * kvHiddenDim * sizeof(int8_t));
                }
            }
        }

#ifdef QKV_INT8
        simd<uint32_t, 64> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
        simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 64 * loopIdx + 32 * vvv;
#pragma unroll
        for (int kk = 0; kk < 1; kk++) {
            simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
            simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * vvv + 64 * subBlock;
            simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim * sizeof(int8_t) +
                macroBlock * blockSize * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 64) * sizeof(int8_t) +
                hhv * 4 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                32,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32));

            simd32Offsets00.select<32, 1>(0) += 32;
        }
#else
        simd<uint32_t, 32> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx + 16 * vvv;
#pragma unroll
        for (int kk = 0; kk < 1; kk++) {
            simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk + 16 * vvv + 64 * subBlock;
            simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * kvHiddenDim * sizeof(int8_t) +
                macroBlock * blockSize * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 64) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16));
            simd32Offsets00.select<16, 1>(0) += 16;
        }
#endif

        {
            auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
            auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
            auto softmaxPositions = ui32Temp.select<64, 1>(64);

            fp32SoftMaxTemp = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
            fp32HistoricMaxTemp = slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
            if (loopIdx >= myLastFullAttnLoopIdx) {
                softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
                softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
                softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
                }
            }

            fp32CurrentMaxTemp = fp32HistoricMaxTemp;
            simd<float, 8 * 16> ttemp;

#pragma unroll
            for (int k = 0; k < 1; k++) {
                //#pragma unroll
                //        for (int kk = 0; kk < 64; kk++) {
                //          fp32CurrentMaxTemp.select<16, 1>(16 * k) =
                //            __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * 64 * k + 16 * kk),
                //              fp32CurrentMaxTemp.select<16, 1>(16 * k));
                //        }

#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) =
                        __ESIMD_NS::max<float, 16, float>(
                            tempOutput.select<16, 1>(16 * 64 * k + 32 * kk),
                            tempOutput.select<16, 1>(16 * 64 * k + 32 * kk + 16));
                }
#pragma unroll
                for (int kkk = 0; kkk < 6; ++kkk) {
#pragma unroll
                    for (int kk = 0; kk < 8; kk++) {
                        ttemp.select<16, 1>(16 * kk) =
                            __ESIMD_NS::max<float, 16, float>(
                                ttemp.select<16, 1>(16 * kk),
                                tempOutput.select<16, 1>(16 * 64 * k + (8 * kkk + kk) * 16 + 16 * 16));
                    }
                }
                ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
                ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
                ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
                fp32CurrentMaxTemp.select<16, 1>(16 * k).merge(
                    ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e,
                    ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp.select<16, 1>(16 * k));

                }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) =
                        tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) * sycl::ext::intel::esimd::detail::log2e -
                        fp32CurrentMaxTemp.select<16, 1>(16 * k);
                }
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * 64 * k + 16 * kk));
                }
            }

            fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
            fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

            if (loopIdx != 0) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
            }

#pragma unroll
            for (int nn = 0; nn < 1; nn++) {
                //#pragma unroll
                //        for (int kk = 0; kk < 64; kk++) {
                //          fp32SoftMaxTemp.select<16, 1>(16 * nn) =
                //            fp32SoftMaxTemp.select<16, 1>(16 * nn) +
                //            tempOutput.select<16, 1>(16 * kk);
                //        }
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * 64 * nn + 32 * kk) + tempOutput.select<16, 1>(16 * 64 * nn + 32 * kk + 16);
                }
#pragma unroll
                for (int k = 0; k < 6; ++k) {
#pragma unroll
                    for (int kk = 0; kk < 8; kk++) {
                        ttemp.select<16, 1>(16 * kk) = ttemp.select<16, 1>(16 * kk) + tempOutput.select<16, 1>(16 * 64 * nn + (8 * k + kk) * 16 + 16 * 16);
                    }
                }
                ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(16 * 4);
                ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(16 * 2);
                ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
                fp32SoftMaxTemp.select<16, 1>(16 * nn) = fp32SoftMaxTemp.select<16, 1>(16 * nn) + ttemp.select<16, 1>(0);
                    }

            fp32HistoricMaxTemp = fp32CurrentMaxTemp;
            slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
            slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
            slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#pragma unroll
            for (int k = 0; k < 1; k++) {
                if (loopIdx != 0) {
                    simd<float, 32> fp32Temp0;
                    fp32Temp0.select<16, 1>(0) = fp32SoftMaxCompensation.select<16, 1>(16 * k);
                    fp32Temp0.select<16, 1>(16) = fp32SoftMaxCompensation.select<16, 1>(16 * k);
#pragma unroll
                    for (int kk = 0; kk < 32; kk++) {
                        finalOutput.select<32, 1>(16 * 64 * k + 32 * kk) = finalOutput.select<32, 1>(16 * 64 * k + 32 * kk) * fp32Temp0;
                    }
                }
            }
        }

        {
#ifdef QKV_INT8
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                simd<float, 64> shuffleTemp;
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f + 0.5f;

                shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
                tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
                tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
                }

#pragma unroll
            for (int k = 0; k < 1; k++) {
                simd<int8_t, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 1; kk++) {
                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
                }
                slm_block_store<int32_t, 32>(slmOffsetV, shuffleTemp.template bit_cast_view<int32_t>());
            }
#else
#pragma unroll
            for (int kk = 0; kk < 32; kk++) {
                simd<float, 32> shuffleTemp;
                shuffleTemp = tempOutput.select<32, 1>(32 * kk);
                tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
                tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
                simd<fp16, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<16, 1>(64 * kk + 0) = i8VState.select<16, 4>(128 * k + 64 * kk + 0);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 1) = i8VState.select<16, 4>(128 * k + 64 * kk + 1);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 2) = i8VState.select<16, 4>(128 * k + 64 * kk + 2);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 3) = i8VState.select<16, 4>(128 * k + 64 * kk + 3);
                }

#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
                }
                slm_block_store<fp16, 128>(slmOffsetV + k * 64 * 16 * sizeof(fp16), shuffleTemp);
            }
#endif
                }

        barrier();

#ifdef QKV_INT8
#pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            simd<float, 32> fp32QvAll = slm_block_load<float, 32>(slmOffsetBaseVs + nn * 32 * sizeof(float));

#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 2; ll++) {
                    i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseV +
                        32 * 64 * l * sizeof(int8_t) +
                        32 * 32 * nn * sizeof(int8_t) +
                        ll * 512 * sizeof(int8_t));
                }
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(512 * k + 128 * kk + 512);
                    auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(256 * k);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * kk);
                    ccTile =
                        dpas
                        <8, 8, int32_t, uint32_t, int32_t,
                        dpas_argument_type::u8,
                        dpas_argument_type::s8
                        >(
                            simd<uint32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
                }
            }
#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(512 * k + 128 * kk + 512);
                    auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(256 * k + 128);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * kk);

                    ccTile =
                        dpas
                        <8, 8, int32_t, int32_t, uint32_t, int32_t,
                        dpas_argument_type::u8,
                        dpas_argument_type::s8
                        >(
                            simd<int32_t, 128>(ccTile.data()),
                            simd<uint32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
                }
            }

#pragma unroll
            for (int32_t kk = 0; kk < 16; kk++) {
                simd<float, 32> fp32Temp1;
                fp32Temp1 = i32TempBuffer.select<32, 1>(32 * kk + 512);
                finalOutput.select<32, 1>(16 * 32 * nn + 32 * kk) += fp32Temp1 * fp32QvAll.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
            }
            }
#else
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                fp16TempBuffer.select<256, 1>(256 * ll) = slm_block_load<fp16, 256>(slmOffsetBaseV +
                                                                                    16 * 64 * nn * sizeof(fp16) +
                                                                                    ll * 256 * sizeof(fp16));
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    auto ccTile = finalOutput.select<128, 1>(1024 * k + 128 * kk);
                    auto aaTile = tempQkFp16.select<256, 1>(1024 * k + 256 * nn);
                    auto bbTile = fp16TempBuffer.select<128, 1>(128 * kk);

                    ccTile = dpas<8, 8, float, float, fp16, fp16>(
                        simd<float, 128>(ccTile.data()),
                        simd<fp16, 256>(aaTile.data()),
                        simd<fp16, 128>(bbTile.data()));
                }
            }
        }
#endif
        //barrier();
                }

    simd<float, 32> softMaxDividor;
    simd<fp16, 16 * 64> finalOutputFp16;
    simd_mask<16> mask;
#pragma unroll
    for (int k = 0; k < 1; k++) {
        softMaxDividor.select<16, 1>(0) = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
        softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
        softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            simd<float, 32> f16Temp = finalOutput.select<32, 1>(16 * 64 * k + 32 * kk);
            f16Temp = f16Temp * softMaxDividor;
            finalOutputFp16.select<16, 2>(16 * 64 * k + 32 * kk) = f16Temp.select<16, 1>(0);
            finalOutputFp16.select<16, 2>(16 * 64 * k + 32 * kk + 1) = f16Temp.select<16, 1>(16);
        }

        simdOffsets = baseOffsetInc16AsVector;
        simdOffsets = simdOffsets + 128 * h + 16 * hhq;
        mask = simdOffsets <= boundaryQ;
        simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
        simdOffsets = simdOffsets * headQ * 64 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            __ESIMD_ENS::lsc_scatter<
                uint32_t,
                8,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back,
                16,
                uint32_t
            >((uint32_t*)out, simdOffsets, finalOutputFp16.template bit_cast_view<uint32_t>().select<128, 1>(512 * k + 128 * kk), mask);
            simdOffsets += 8 * sizeof(uint32_t);
        }
    }
}


// Shape Q [activation token length, 96, 64] INT8,
// Shape K:  [kv len, 8, 64] INT8,
// Shape V : [kv len, 8, 64] INT8,
// output: [activation token length, 96, 64] FP16,
template <uint32_t GPA_DIV_2>
ESIMD_INLINE void sageAttnGqa_2X_64(
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
    uint32_t headQ,
    uint32_t headKv,
    nd_item<2>& ndi
) {
    constexpr float matMulQuantCoeff = 0.125f; // 1.0f / sqrt(64.0f);
    constexpr uint32_t slmSizeK = 64 * 64 * sizeof(int8_t);
    constexpr uint32_t slmSizeV = 64 * 64 * sizeof(fp16);
    constexpr uint32_t slmSizeSoftMaxSum = 512 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxHistoric = 512 * sizeof(float);
    constexpr uint32_t slmSizeSoftMaxCompensation = 512 * sizeof(float);
    constexpr uint32_t slmSizeKs = 64 * sizeof(float);
    //  constexpr uint32_t slmSizeQ = 32 * 64 * 16 * sizeof(int8_t);
    constexpr uint32_t slmSizeVs = 64 * sizeof(float);
    constexpr uint32_t slmSize =
        slmSizeK +
        slmSizeV +
        slmSizeSoftMaxSum +
        slmSizeSoftMaxHistoric +
        slmSizeSoftMaxCompensation +
        slmSizeKs +
        slmSizeVs;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    __ESIMD_NS::slm_init(slmSize);
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
    constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
    constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
    constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
    constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
    constexpr uint32_t slmOffsetBaseVs = slmOffsetBaseKs + slmSizeKs;
    constexpr int32_t batchIdx = 0; // v / headKv;
    constexpr int32_t groupGqaRatio = 4;
    constexpr uint32_t powerOf2BlockSize = 9;
    constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
    constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
    constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
    constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
    constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16
    constexpr uint32_t singlePageMaxMask = (1 << (powerOf2BlockSize - 6)) - 1;
    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq = localLinearId & 0xf;
    int32_t vvq = localLinearId >> 4;
    int32_t hhk = localLinearId & 0x7; // [0, 4)
    int32_t vvk = localLinearId >> 3; // [0, 4)
    int32_t hhv;
    int32_t vvv;
#ifdef QKV_INT8
    hhv = localLinearId & 0xf;
    vvv = localLinearId >> 4;
#else
    hhv = localLinearId & 0x7;
    vvv = localLinearId >> 3;
#endif
    int32_t h = ndi.get_group(0);
    int32_t v = ndi.get_group(1); // [0, 24)
    int32_t headIdxQ = v * 2;
    //int32_t gqaRatio = (headQ / headKv) >> 2;
    int32_t headIdxKv = v / GPA_DIV_2;
    uint32_t kvSeqLen;
    uint32_t activationLength;
    uint32_t startActivationIdx;
    simd<uint32_t, 2> kvCoord;
    simd<uint32_t, 2> queryCoord;
    kvCoord = block_load<uint32_t, 2>(cuSeqKv);
    queryCoord = block_load<uint32_t, 2>(cuSeqQuery);
    headIdxQ = headIdxQ + vvq;
    simd<int8_t, 16 * 64> i8QState;
    simd<float, 16> fp32Qs;
    simd<int32_t, 16 * 64> i32TempBuffer;
    //  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
    simd<float, 512> tempBuffer;
    simd<int8_t, 128> i8VState;
    auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<128, 1>(0);
    auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
    auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
    auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
    simd<float, 16 * 64> finalOutput = 0;
    simd<float, 8> fp32Qv;
    simd<fp16, 128> fp16Qv;
    fp16 vsTemp = 1.0f;
    simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
    simd<uint32_t, 32> softMaxThresh(baseOffsetInc16);
    simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
    simd<uint32_t, 16> blockIdx;

    kvSeqLen = kvCoord[1] - kvCoord[0];
    activationLength = queryCoord[1] - queryCoord[0];
    startActivationIdx = kvSeqLen - activationLength;

    int32_t kvSeqOutLoopCount = 256 * h + 256 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 63) >> 6;
    uint32_t myLastFullAttnLoopIdx = (256 * h + startActivationIdx + 16 * hhq) >> 6;
    unsigned int offsetQBase = headIdxQ * 64 * sizeof(int8_t);
    unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
    unsigned int kvHiddenDim = headKv * 64;
    unsigned int offsetBasePref = (headIdxKv * 64 + localLinearId * headKv * 64) * sizeof(int8_t);
    unsigned int slmOffsetK = slmOffsetBaseK + (hhk & 0x3) * 2 * sizeof(uint32_t) + (hhk >> 2) * 64 * 32 * sizeof(uint8_t) + vvk * 16 * 8 * sizeof(uint32_t);
    unsigned int slmOffsetV;
#ifdef QKV_INT8
    slmOffsetV = slmOffsetBaseV + localLinearId * 128 * sizeof(int8_t);
#else
    slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16) + vvv * 16 * 64 * sizeof(fp16);
#endif
    unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
    unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
    //  unsigned int slmOffsetC0 = slmOffsetBaseV;
    unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
    //  uint32_t boundaryKv = (headIdxKv * 128 + (kvSeqLen - 1) * headKv * 128) * sizeof(int8_t);
    uint32_t boundaryQ = activationLength - 1;

    if (256 * h >= activationLength) {
        return;
    }

    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 256 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * 64 * sizeof(int8_t) + offsetQBase;

#pragma unroll
    for (int k = 0; k < 2; k++) {
        i8QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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

    //  constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff;
    simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + 256 * h + 16 * hhq;
    simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
    simdOffsets = simdOffsets * headQ * sizeof(float) + headIdxQ * sizeof(float);
    fp32Qs.select<16, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        float,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((float*)qs, simdOffsets);

    //  simd<float, 16> fp32SoftMaxTemp = 0.0f;
    //  simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
    blockIdx = __ESIMD_ENS::lsc_block_load<
        uint32_t,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + 0);

#ifdef QKV_INT8
    fp32Qv.select<4, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        4,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 64 * headIdxKv + 4 * hhv);
#else
    fp32Qv.select<8, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        8,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)vs + 64 * headIdxKv + 8 * hhv);
#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
        fp16Qv.select<32, 1>(32 * kk) = fp32Qv.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
    }
#endif

    {
#ifdef QKV_INT8
        if (vvv == 0) {
            fp32Qv.select<4, 1>(0) = fp32Qv.select<4, 1>(0) * (1.0f / 255.0f);
            slm_block_store<float, 4>(slmOffsetBaseVs + hhv * 4 * sizeof(float), fp32Qv.select<4, 1>(0));
        }
#else
#endif
        simd<float, 16> slmInitTemp = 0;
        slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), slmInitTemp);
        slmInitTemp = FP32_MIN;
        slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, slmInitTemp);
    }

    softMaxThresh.select<16, 1>(0) = baseOffsetInc16AsVector;
    softMaxThresh = softMaxThresh + 16 * hhq + 256 * h + startActivationIdx;
    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        uint32_t whichIdx = loopIdx & blockTableReloadCheckMask;
        uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift];
        uint32_t subBlock = (whichIdx & perLoopMacroBlockMask);
        uint32_t blockOffset = macroBlock * blockSize + subBlock * 64;
        uint32_t offsetPref = blockOffset * kvHiddenDim + offsetBasePref;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            __ESIMD_ENS::lsc_prefetch<
                uint8_t,
                64,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached
            >((uint8_t*)vState + offsetPref + kk * 32 * kvHiddenDim * sizeof(int8_t));
        }

        {
            simd<uint32_t, 16> ksOffset;
            simd<uint32_t, 16> coordK;
            simd<float, 2> ksTemp;
            ksOffset = localLinearId + baseOffsetInc16AsVector * 32 + 64 * loopIdx;
            ksOffset.merge(0, ksOffset >= kvSeqLen);
            ksOffset = ksOffset * headKv * sizeof(float) + headIdxKv * sizeof(float);
            ksTemp =
                __ESIMD_ENS::lsc_gather<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                2,
                uint32_t
                >((float*)ks, ksOffset.select<2, 1>(0));
            coordK.select<16, 1>(0) = baseOffsetInc16AsVector;
            coordK.select<16, 1>(0) = coordK.select<16, 1>(0) + 64 * loopIdx + vvk * 16;
            simdOffsets.select<16, 1>(0) = baseOffsetInc16AsVector + vvk * 16 + subBlock * 64;
            simdOffsets.merge(0, coordK >= kvSeqLen);
            simdOffsets = simdOffsets + macroBlock * blockSize;
            simdOffsets = simdOffsets * kvHiddenDim * sizeof(int8_t) +
                headIdxKv * 64 * sizeof(int8_t) +
                hhk * 2 * sizeof(uint32_t);

            i8KState.template bit_cast_view<uint32_t>().select<32, 1>(0) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)kState, simdOffsets.select<16, 1>(0));

            ksOffset = baseOffsetInc16AsVector * 32 + localLinearId;
            ksOffset = ksOffset * sizeof(float) + slmOffsetBaseKs;
            __ESIMD_ENS::lsc_slm_scatter<
                float,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                2>(ksOffset.select<2, 1>(0), ksTemp.select<2, 1>(0));

            if (whichIdx == blockTableReloadCheckMask) {
                uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
                blockIdx =
                    __ESIMD_ENS::lsc_block_load<
                    uint32_t,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable);
            }

            coordK = baseOffsetInc16AsVector * 8 * sizeof(uint32_t) + slmOffsetK;
            __ESIMD_ENS::lsc_slm_scatter<
                int32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                16>(coordK.select<16, 1>(0), i8KState.template bit_cast_view<int32_t>().select<32, 1>(0));
        }

        auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
        auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
        auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();

        barrier();
        i32TempBuffer = 0;

        {
            simd<float, 64> ksAllTemp = slm_block_load<float, 64>(slmOffsetBaseKs);
#pragma unroll
            for (int32_t nn = 0; nn < 2; nn++) {
#pragma unroll
                for (int32_t ll = 0; ll < 4; ll++) {
                    i8TempBuffer.select<512, 1>(512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseK +
                        32 * 64 * nn * sizeof(int8_t) +
                        512 * ll * sizeof(int8_t));
                }

#pragma unroll
                for (int32_t kk = 0; kk < 8; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(128 * kk);
                    auto aaTile = i8QState.template bit_cast_view<int32_t>().select<128, 1>(128 * nn);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * kk);

                    ccTile =
                        dpas
                        <8, 8, int32_t, int32_t, int32_t, int32_t,
                        dpas_argument_type::s8,
                        dpas_argument_type::s8
                        >(
                            simd<int32_t, 128>(ccTile.data()),
                            simd<int32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
                }
            }

            simd<float, 32> fp32TempQs0;
            fp32TempQs0.select<16, 1>(0) = fp32Qs.select<16, 1>(0);
            fp32TempQs0.select<16, 1>(16) = fp32Qs.select<16, 1>(0);

#pragma unroll
            for (int32_t kk = 0; kk < 32; kk++) {
                simd<float, 32> fp32Temp0;
                simd<float, 32> fp32Temp1;
                simd<float, 32> fp32TempOuter0;

                fp32TempOuter0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
                fp32Temp0 = i32TempBuffer.select<32, 1>(32 * kk) * matMulQuantCoeff;
                fp32Temp1 = fp32TempOuter0 * fp32TempQs0;
                tempOutput.select<32, 1>(32 * kk) = fp32Temp0 * fp32Temp1;
            }

            if (whichIdx == blockTableReloadCheckMask) {
                offsetPref = blockIdx[0] * kvHiddenDim * blockSize + offsetBasePref;
            }

            uint32_t singpagelastIdx = whichIdx & singlePageMaxMask;
            if (singpagelastIdx < singlePageMaxMask) {
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    __ESIMD_ENS::lsc_prefetch<
                        uint8_t,
                        64,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached
                    >((uint8_t*)kState + offsetPref + kk * 32 * kvHiddenDim * sizeof(int8_t));
                }
            }
        }

#ifdef QKV_INT8
        simd<uint32_t, 64> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
        simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 64 * loopIdx + 32 * vvv;
#pragma unroll
        for (int kk = 0; kk < 1; kk++) {
            simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
            simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * vvv + 64 * subBlock;
            simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim * sizeof(int8_t) +
                macroBlock * blockSize * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 64) * sizeof(int8_t) +
                hhv * 4 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                1,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                32,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32));

            simd32Offsets00.select<32, 1>(0) += 32;
        }
#else
        simd<uint32_t, 32> simd32Offsets00;
        simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
        simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx + 16 * vvv;
#pragma unroll
        for (int kk = 0; kk < 1; kk++) {
            simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk + 16 * vvv + 64 * subBlock;
            simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
            simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * kvHiddenDim * sizeof(int8_t) +
                macroBlock * blockSize * kvHiddenDim * sizeof(int8_t) +
                (headIdxKv * 64) * sizeof(int8_t) +
                hhv * 8 * sizeof(int8_t);

            i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
                __ESIMD_ENS::lsc_gather<
                uint32_t,
                2,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16));
            simd32Offsets00.select<16, 1>(0) += 16;
        }
#endif

        {
            auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
            auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
            auto softmaxPositions = ui32Temp.select<64, 1>(64);

            fp32SoftMaxTemp = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
            fp32HistoricMaxTemp = slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
            if (loopIdx >= myLastFullAttnLoopIdx) {
                softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
                softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
                softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
                    tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
                }
            }

            fp32CurrentMaxTemp = fp32HistoricMaxTemp;
            simd<float, 8 * 16> ttemp;

#pragma unroll
            for (int k = 0; k < 1; k++) {
                //#pragma unroll
                //        for (int kk = 0; kk < 64; kk++) {
                //          fp32CurrentMaxTemp.select<16, 1>(16 * k) =
                //            __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * 64 * k + 16 * kk),
                //              fp32CurrentMaxTemp.select<16, 1>(16 * k));
                //        }

#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) =
                        __ESIMD_NS::max<float, 16, float>(
                            tempOutput.select<16, 1>(16 * 64 * k + 32 * kk),
                            tempOutput.select<16, 1>(16 * 64 * k + 32 * kk + 16));
                }
#pragma unroll
                for (int kkk = 0; kkk < 6; ++kkk) {
#pragma unroll
                    for (int kk = 0; kk < 8; kk++) {
                        ttemp.select<16, 1>(16 * kk) =
                            __ESIMD_NS::max<float, 16, float>(
                                ttemp.select<16, 1>(16 * kk),
                                tempOutput.select<16, 1>(16 * 64 * k + (8 * kkk + kk) * 16 + 16 * 16));
                    }
                }
                ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(16 * 4));
                ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(16 * 2));
                ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
                fp32CurrentMaxTemp.select<16, 1>(16 * k).merge(
                    ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e,
                    ttemp.select<16, 1>(0) * sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp.select<16, 1>(16 * k));

            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) =
                        tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) * sycl::ext::intel::esimd::detail::log2e -
                        fp32CurrentMaxTemp.select<16, 1>(16 * k);
                }
#pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    tempOutput.select<16, 1>(16 * 64 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * 64 * k + 16 * kk));
                }
            }

            fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
            fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

            if (loopIdx != 0) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
            }

#pragma unroll
            for (int nn = 0; nn < 1; nn++) {
                //#pragma unroll
                //        for (int kk = 0; kk < 64; kk++) {
                //          fp32SoftMaxTemp.select<16, 1>(16 * nn) =
                //            fp32SoftMaxTemp.select<16, 1>(16 * nn) +
                //            tempOutput.select<16, 1>(16 * kk);
                //        }
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * 64 * nn + 32 * kk) + tempOutput.select<16, 1>(16 * 64 * nn + 32 * kk + 16);
                }
#pragma unroll
                for (int k = 0; k < 6; ++k) {
#pragma unroll
                    for (int kk = 0; kk < 8; kk++) {
                        ttemp.select<16, 1>(16 * kk) = ttemp.select<16, 1>(16 * kk) + tempOutput.select<16, 1>(16 * 64 * nn + (8 * k + kk) * 16 + 16 * 16);
                    }
                }
                ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(16 * 4);
                ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(16 * 2);
                ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
                fp32SoftMaxTemp.select<16, 1>(16 * nn) = fp32SoftMaxTemp.select<16, 1>(16 * nn) + ttemp.select<16, 1>(0);
            }

            fp32HistoricMaxTemp = fp32CurrentMaxTemp;
            slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
            slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
            slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#pragma unroll
            for (int k = 0; k < 1; k++) {
                if (loopIdx != 0) {
                    simd<float, 32> fp32Temp0;
                    fp32Temp0.select<16, 1>(0) = fp32SoftMaxCompensation.select<16, 1>(16 * k);
                    fp32Temp0.select<16, 1>(16) = fp32SoftMaxCompensation.select<16, 1>(16 * k);
#pragma unroll
                    for (int kk = 0; kk < 32; kk++) {
                        finalOutput.select<32, 1>(16 * 64 * k + 32 * kk) = finalOutput.select<32, 1>(16 * 64 * k + 32 * kk) * fp32Temp0;
                    }
                }
            }
        }

        {
#ifdef QKV_INT8
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                simd<float, 64> shuffleTemp;
                shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
                shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f + 0.5f;

                shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
                tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
                tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
                simd<int8_t, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 1; kk++) {
                    shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
                    shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
                }
                slm_block_store<int32_t, 32>(slmOffsetV, shuffleTemp.template bit_cast_view<int32_t>());
            }
#else
#pragma unroll
            for (int kk = 0; kk < 32; kk++) {
                simd<float, 32> shuffleTemp;
                shuffleTemp = tempOutput.select<32, 1>(32 * kk);
                tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
                tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
                simd<fp16, 128> shuffleTemp;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    shuffleTemp.select<16, 1>(64 * kk + 0) = i8VState.select<16, 4>(128 * k + 64 * kk + 0);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 1) = i8VState.select<16, 4>(128 * k + 64 * kk + 1);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 2) = i8VState.select<16, 4>(128 * k + 64 * kk + 2);
                    shuffleTemp.select<16, 1>(64 * kk + 16 * 3) = i8VState.select<16, 4>(128 * k + 64 * kk + 3);
                }

#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
                }
                slm_block_store<fp16, 128>(slmOffsetV + k * 64 * 16 * sizeof(fp16), shuffleTemp);
            }
#endif
        }

        barrier();

#ifdef QKV_INT8
#pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            simd<float, 32> fp32QvAll = slm_block_load<float, 32>(slmOffsetBaseVs + nn * 32 * sizeof(float));

#pragma unroll
            for (int l = 0; l < 2; l++) {
#pragma unroll
                for (int ll = 0; ll < 2; ll++) {
                    i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = slm_block_load<int8_t, 512>(
                        slmOffsetBaseV +
                        32 * 64 * l * sizeof(int8_t) +
                        32 * 32 * nn * sizeof(int8_t) +
                        ll * 512 * sizeof(int8_t));
                }
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(512 * k + 128 * kk + 512);
                    auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(256 * k);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * kk);
                    ccTile =
                        dpas
                        <8, 8, int32_t, uint32_t, int32_t,
                        dpas_argument_type::u8,
                        dpas_argument_type::s8
                        >(
                            simd<uint32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
                }
            }
#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    auto ccTile = i32TempBuffer.select<128, 1>(512 * k + 128 * kk + 512);
                    auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(256 * k + 128);
                    auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * kk);

                    ccTile =
                        dpas
                        <8, 8, int32_t, int32_t, uint32_t, int32_t,
                        dpas_argument_type::u8,
                        dpas_argument_type::s8
                        >(
                            simd<int32_t, 128>(ccTile.data()),
                            simd<uint32_t, 128>(aaTile.data()),
                            simd<int32_t, 64>(bbTile.data())
                        );
                }
            }

#pragma unroll
            for (int32_t kk = 0; kk < 16; kk++) {
                simd<float, 32> fp32Temp1;
                fp32Temp1 = i32TempBuffer.select<32, 1>(32 * kk + 512);
                finalOutput.select<32, 1>(16 * 32 * nn + 32 * kk) += fp32Temp1 * fp32QvAll.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
            }
        }
#else
#pragma unroll
        for (int nn = 0; nn < 4; nn++) {
#pragma unroll
            for (int ll = 0; ll < 4; ll++) {
                fp16TempBuffer.select<256, 1>(256 * ll) = slm_block_load<fp16, 256>(slmOffsetBaseV +
                                                                                    16 * 64 * nn * sizeof(fp16) +
                                                                                    ll * 256 * sizeof(fp16));
            }

#pragma unroll
            for (int k = 0; k < 1; k++) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    auto ccTile = finalOutput.select<128, 1>(1024 * k + 128 * kk);
                    auto aaTile = tempQkFp16.select<256, 1>(1024 * k + 256 * nn);
                    auto bbTile = fp16TempBuffer.select<128, 1>(128 * kk);

                    ccTile = dpas<8, 8, float, float, fp16, fp16>(
                        simd<float, 128>(ccTile.data()),
                        simd<fp16, 256>(aaTile.data()),
                        simd<fp16, 128>(bbTile.data()));
                }
            }
        }
#endif
        //barrier();
                }

    simd<float, 32> softMaxDividor;
    simd<fp16, 16 * 64> finalOutputFp16;
    simd_mask<16> mask;
#pragma unroll
    for (int k = 0; k < 1; k++) {
        softMaxDividor.select<16, 1>(0) = slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
        softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
        softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            simd<float, 32> f16Temp = finalOutput.select<32, 1>(16 * 64 * k + 32 * kk);
            f16Temp = f16Temp * softMaxDividor;
            finalOutputFp16.select<16, 2>(16 * 64 * k + 32 * kk) = f16Temp.select<16, 1>(0);
            finalOutputFp16.select<16, 2>(16 * 64 * k + 32 * kk + 1) = f16Temp.select<16, 1>(16);
        }

        simdOffsets = baseOffsetInc16AsVector;
        simdOffsets = simdOffsets + 256 * h + 16 * hhq;
        mask = simdOffsets <= boundaryQ;
        simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
        simdOffsets = simdOffsets * headQ * 64 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            __ESIMD_ENS::lsc_scatter<
                uint32_t,
                8,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back,
                16,
                uint32_t
            >((uint32_t*)out, simdOffsets, finalOutputFp16.template bit_cast_view<uint32_t>().select<128, 1>(512 * k + 128 * kk), mask);
            simdOffsets += 8 * sizeof(uint32_t);
        }
    }
}


ESIMD_INLINE void sageAttnGqa12(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  fp16* qs,
  fp16* ks,
  fp16* vs,
  uint8_t* out,
  uint32_t kvSeqLen,
  uint32_t startActivationIdx,
  uint32_t activationLength,
  uint32_t headQ,
  uint32_t headKv,
  sycl::nd_item<2>& ndi
  ) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr uint32_t slmSizeK = 64 * 128 * sizeof(int8_t);
  constexpr uint32_t slmSizeV = 64 * 128 * sizeof(fp16);
  constexpr uint32_t slmSizeSoftMaxSum = 256 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxHistoric = 256 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxCompensation = 256 * sizeof(float);
  constexpr uint32_t slmSizeKs = 64 * sizeof(fp16);
  constexpr uint32_t slmSize =
    slmSizeK +
    slmSizeV +
    slmSizeSoftMaxSum +
    slmSizeSoftMaxHistoric + 
    slmSizeSoftMaxCompensation +
    slmSizeKs;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseK = 0;
  constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
  constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
  constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
  constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
  constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
  constexpr int32_t batchIdx = 0; // v / headKv;
  constexpr int32_t groupGqaRatio = 4;
  int32_t localLinearId = ndi.get_local_id(0);
  int32_t hh = localLinearId & 0x3;
  int32_t vv = localLinearId >> 2;
  int32_t hhq = localLinearId & 0x3;
  int32_t vvq = localLinearId >> 2;
  int32_t hhv = localLinearId & 0xf;
  int32_t vvv = localLinearId >> 4;
  int32_t h =  ndi.get_group(0);
  int32_t v = ndi.get_group(1); // [0, 24)
  int32_t headIdxQ = v * 4;
  int32_t headIdxKv = v / 3;
  headIdxQ = headIdxQ + vvq;
  __ESIMD_NS::simd<int8_t, 16 * 128> i8QState;
  __ESIMD_NS::simd<fp16, 32> fp16Qs;
  __ESIMD_NS::simd<int32_t, 16 * 64> i32TempBuffer;
//  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
  __ESIMD_NS::simd<float, 512> tempBuffer;
  auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
  auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 3);
  auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
  auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  __ESIMD_NS::simd<fp16, 16 * 128> finalOutput = 0;
  __ESIMD_NS::simd<fp16, 128> fp16Qv;
  fp16 vsTemp;
  __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
  
  int32_t kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
  kvSeqOutLoopCount = kvSeqOutLoopCount >> 6;
  unsigned int offsetQBase = headIdxQ * 128 * sizeof(int8_t);
  unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
  unsigned int offsetK = (headIdxKv * 128 + localLinearId * headKv * 128) * sizeof(int8_t);
  unsigned int slmOffsetK = slmOffsetBaseK + localLinearId * 32 * sizeof(int8_t);
  unsigned int slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16);
  unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
  unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
  unsigned int slmOffsetC0 = slmOffsetBaseV;
  unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);
  uint32_t boundaryKv = (headIdxKv * 128 + (kvSeqLen - 1) * headKv * 128) * sizeof(int8_t);
  uint32_t boundaryQ = activationLength - 1;

  if (64 * h >= activationLength) {
    return;
  }

  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * 128 * sizeof(int8_t) + offsetQBase;

#pragma unroll
  for (int k = 0; k < 4; k++) {
    i8QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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
  constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff; 
  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * sizeof(fp16) + headIdxQ * sizeof(fp16);
  fp16Qs.select<16, 1>(0) =
    __ESIMD_ENS::lsc_gather<
    fp16,
    1,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached,
    16,
    uint32_t
    >((fp16*)qs, simdOffsets);

  softMaxThresh = softMaxThresh + 16 * hhq + 64 * h + startActivationIdx;
  
  __ESIMD_NS::simd<float, 16> fp32SoftMaxTemp = 0.0f;
  __ESIMD_NS::simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;

#ifdef QKV_INT8
  fp16Qv = 
    __ESIMD_ENS::lsc_block_load<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((fp16*)vs + 128 * headIdxKv);

  fp16Qv = fp16Qv * (fp16)(1.0f / 255.0f);
#else
  fp16TempBuffer.select<8, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    fp16,
    8,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((fp16*)vs + 128 * headIdxKv + 8 * hhv);

#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    fp16Qv.select<32, 1>(32 * kk) = fp16TempBuffer.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
  }
#endif

  fp16Qs.select<16, 1>(16) = fp16Qs.select<16, 1>(0);

  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    auto simd32Offsets = i32TempBuffer.select<32, 1>(0);
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      __ESIMD_ENS::lsc_prefetch<
        uint8_t,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >((uint8_t*)vState + offsetK + kk * 16 * headKv * 128 * sizeof(int8_t));
    }
#if 1
    {
      __ESIMD_NS::simd<uint32_t, 16> ksOffset;
      __ESIMD_NS::simd<fp16, 4> ksTemp;
      ksOffset = localLinearId + baseOffsetInc16AsVector * 16 + 64 * loopIdx;
      ksOffset.merge(0, ksOffset >= kvSeqLen);
      ksOffset = ksOffset * headKv * sizeof(fp16) + headIdxKv * sizeof(fp16);
      ksTemp =
        __ESIMD_ENS::lsc_gather<
        fp16,
        1,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        4,
        uint32_t
        >((fp16*)ks, ksOffset.select<4, 1>(0));
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        offsetK = __ESIMD_NS::min(offsetK, boundaryKv);

        i8KState.template bit_cast_view<uint8_t>().template select<128, 1>(128 * kk) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK);

        offsetK += 16 * headKv * 128 * sizeof(int8_t);
      }
      ksOffset = baseOffsetInc16AsVector * 16 + localLinearId;
      ksOffset = ksOffset * sizeof(fp16) + slmOffsetBaseKs;
      __ESIMD_ENS::lsc_slm_scatter<
        fp16,
        1,
        __ESIMD_ENS::lsc_data_size::u16,
        4>(ksOffset.select<4, 1>(0), ksTemp.select<4, 1>(0));
#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          slm_block_store<int8_t, 32>(slmOffsetK + k * 16 * 128 * sizeof(int8_t) + kk * 16 * 32 * sizeof(int8_t), i8KState.select<32, 1>(128 * k + 32 * kk));
        }
      }
    }
#else
#pragma unroll
      //16 x4  64 Ks
      for (int kk = 0; kk < 4; kk++) {
          uint32_t ksOffset = localLinearId + 16 * kk + 64 * loopIdx;
          fp16 ksTemp;
          ksTemp = ks[ksOffset];
          slm_block_store<fp16, 1>(slmOffsetBaseKs + localLinearId * sizeof(fp16) + 16 * kk * sizeof(fp16), ksTemp);
      }
      uint32_t kOffset_2d = (64 * loopIdx + (localLinearId / 4) * 16) * headKv * 128 /*token*/ + headIdxKv * 128 /*header*/ + (localLinearId % 4) * 32/*block*/;
      i8KState.select<512, 1>(0) = __ESIMD_ENS::lsc_load_2d<
          uint8_t, 32, 16, 1,  //8x32
          false, false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState,
              headKv * 128 - 1, kvSeqLen - 1, headKv * 128 - 1,
              headIdxKv * 128 + (localLinearId % 4) * 32, 64 * loopIdx + (localLinearId / 4) * 16);
      slm_block_store<int8_t, 512>(slmOffsetBaseK + (localLinearId / 2) * 2 * 512 + (localLinearId % 2) * 512, i8KState.select<512, 1>(0));
#endif
    auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
    auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
    auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();
    barrier();
    i32TempBuffer = 0;

    {
      __ESIMD_NS::simd<fp16, 64> ksAllTemp = __ESIMD_NS::slm_block_load<fp16, 64>(slmOffsetBaseKs);
#pragma unroll
      for (int32_t nn = 0; nn < 4; nn++) {
#pragma unroll
        for (int32_t l = 0; l < 2; l++) {
#pragma unroll
          for (int32_t ll = 0; ll < 2; ll++) {
            i8TempBuffer.select<512, 1>(512 * ll) = __ESIMD_NS::slm_block_load<int8_t, 512>(
              slmOffsetBaseK + 16 * 128 * nn * sizeof(int8_t) +
              1024 * l * sizeof(int8_t) +
              512 * ll * sizeof(int8_t));
          }

#pragma unroll
          for (int32_t k = 0; k < 2; k++) {
#pragma unroll
            for (int32_t kk = 0; kk < 2; kk++) {
              auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
              auto aaTile = i8QState.template bit_cast_view<int32_t>().select<128, 1>(256 * l + 128 * k);
              auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk);

              ccTile =
                sycl::ext::intel::esimd::xmx::dpas
                <8, 8, int32_t, int32_t, int32_t, int32_t,
                sycl::ext::intel::esimd::xmx::dpas_argument_type::s8,
                sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
                >(
                  __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                  __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                  __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
            }
          }
        }
#pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          __ESIMD_NS::simd<float, 32> fp32Temp0;
          __ESIMD_NS::simd<float, 32> fp32Temp1;
          fp32Temp0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk) * fp16Qs;
          fp32Temp1 = i32TempBuffer.select<32, 1>(256 * nn + 32 * kk);
  #ifdef MERGE_QUANT
          tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1*matMulQuantCoeff;
  #else
          tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1
  #endif
        }
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        __ESIMD_ENS::lsc_prefetch<
          uint8_t,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached
        >((uint8_t*)kState + offsetK + kk * 16 * headKv * 128 * sizeof(int8_t));
      }
    }

#ifdef QKV_INT8
//    vsTemp = vs[loopIdx];
	  __ESIMD_NS::simd<uint32_t,32> simd32Offsets00;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
      simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
      simd32Offsets00 = simd32Offsets00 + 64 * loopIdx + 32 * kk;
      simd32Offsets00.merge(0, simd32Offsets00 >= kvSeqLen);
      simd32Offsets00 = simd32Offsets00 * headKv * 128 * sizeof(int8_t) +
        (headIdxKv * 128) * sizeof(int8_t) +
        hhv * 8 * sizeof(int8_t);

      i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        2,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        32,
        uint32_t
        >((uint32_t*)vState, simd32Offsets00);
    }
#else
    __ESIMD_NS::simd<uint32_t,32> simd32Offsets00;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
      simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx + 16 * kk;
      simd32Offsets00.select<16, 1>(0).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
      simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0);

      simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) * headKv * 128 * sizeof(int8_t) +
        (headIdxKv * 128) * sizeof(int8_t) +
        hhv * 8 * sizeof(int8_t);

      simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * headKv * sizeof(fp16) + 
        (headIdxKv) * sizeof(fp16);

      i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        2,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)vState, simd32Offsets00.select<16, 1>(0));
    }
#endif

    {
     //   auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
     //   auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
      auto softmaxPositions = ui32Temp.select<16, 1>(64);

    //  fp32SoftMaxTemp = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
    //  fp32HistoricMaxTemp = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
#if  VERSION_0
 
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
        }
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
      }
   

#else
      if(loopIdx==kvSeqOutLoopCount-1){
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
        }
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
      }
    }
#endif
#ifndef MERGE_QUANT
     tempOutput.select<16 * 64, 1>(0) = tempOutput.select<16 * 64, 1>(0) * matMulQuantCoeff;

#endif
  fp32CurrentMaxTemp = fp32HistoricMaxTemp;

#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * kk), fp32CurrentMaxTemp);
      }

#else
      __ESIMD_NS::simd<float,8*16> ttemp;
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32*kk), tempOutput.select<16, 1>(32*kk+16));
      }
#pragma unroll
    for(int kkk=0;kkk<6;++kkk){
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16*kk), tempOutput.select<16, 1>((8*kkk+kk)*16+16*16));
      }
    }
    ttemp.select<64, 1>(0)=__ESIMD_NS::max<float, 64, float>( ttemp.select<64, 1>(0), ttemp.select<64, 1>(16*4));
    ttemp.select<32, 1>(0)=__ESIMD_NS::max<float, 32, float>( ttemp.select<32, 1>(0), ttemp.select<32, 1>(16*2));
    ttemp.select<16, 1>(0)=__ESIMD_NS::max<float, 16, float>( ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
    #if  VERSION_0// Positive 
        fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), fp32CurrentMaxTemp);
    #else
        fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0)*sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0)*sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);
    #endif

#endif

#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#elif 1  // Positive 
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
      }
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#else
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk)  - fp32CurrentMaxTemp.select<16, 1>(0);
      }
#pragma unroll
     for (int kk = 0; kk < 64; kk++) { 
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
      }
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#endif

#if 0
      fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
      fp32SoftMaxCompensation = __ESIMD_NS::pow<float, 16, float>(2.718281828459f, fp32SoftMaxCompensation);
#else
      fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>( fp32SoftMaxCompensation);
#endif
      if (loopIdx != 0) {
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
      }
#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + tempOutput.select<16, 1>(16 * kk);
      }
#else  
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = tempOutput.select<16, 1>(32*kk)+ tempOutput.select<16, 1>(32*kk+16);
      }
#pragma unroll
      for(int k=0;k<6;++k){
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16*kk) = ttemp.select<16, 1>(16*kk)+ tempOutput.select<16, 1>((8*k+kk)*16+16*16);
        }
    }
    ttemp.select<64, 1>(0)= ttemp.select<64, 1>(0)+ ttemp.select<64, 1>(16*4);
    ttemp.select<32, 1>(0)= ttemp.select<32, 1>(0)+ ttemp.select<32, 1>(16*2);
    ttemp.select<16, 1>(0)= ttemp.select<16, 1>(0)+ ttemp.select<16, 1>(16);
    fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) +ttemp.select<16, 1>(0); 
#endif
      fp32HistoricMaxTemp =fp32CurrentMaxTemp;
     // slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
     // slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
     // slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#ifdef QKV_INT8
#pragma unroll
      for (int kk = 0; kk < 16; kk++) {

        __ESIMD_NS::simd<float, 64> shuffleTemp;
#if  0
        shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f;
        shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f;

        shuffleTemp = shuffleTemp + 0.5f;
#else
        shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
        shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f+0.5f;

#endif
        shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
        tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
        tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
      }

#pragma unroll
      for (int k = 0; k < 2; k++) {
        __ESIMD_NS::simd<int8_t, 256> shuffleTemp;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
        }
        slm_block_store<int32_t, 64>(slmOffsetV + k * 128 * 32 * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
      }
#else
#pragma unroll
      for (int kk = 0; kk < 32; kk++) {
        __ESIMD_NS::simd<float, 32> shuffleTemp;
        shuffleTemp = tempOutput.select<32, 1>(32 * kk);
        tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
        tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
      }

#pragma unroll
      for (int k = 0; k < 4; k++) {
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
          shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
        }
        slm_block_store<fp16, 128>(slmOffsetV + k * 128 * 16 * sizeof(fp16), shuffleTemp);
      }
#endif

      tempBuffer.select<16, 1>(0) = fp32SoftMaxCompensation;
      tempBuffer.select<16, 1>(16) = tempBuffer.select<16, 1>(0);
      if (loopIdx != 0) {
        __ESIMD_NS::simd<fp16, 32> fp16Temp0;
        fp16Temp0 = tempBuffer.select<32, 1>(0);
#pragma unroll
        for (int kk = 0; kk < 64; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * fp16Temp0;
        }
      }
    }

    barrier();
#ifdef QKV_INT8
#pragma unroll
    for (int nn = 0; nn < 4; nn++) {
#pragma unroll
      for (int l = 0; l < 2; l++) {
#pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = __ESIMD_NS::slm_block_load<int8_t, 512>(
            slmOffsetC0 +
            32 * 128 * l * sizeof(int8_t) +
            32 * 32 * nn * sizeof(int8_t) +
            ll * 512 * sizeof(int8_t));
        }
      }
#pragma unroll
      for (int ll = 0; ll < 4; ll++) {
        auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
        auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);
        auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

        ccTile =
          sycl::ext::intel::esimd::xmx::dpas
          <8, 8, int32_t, uint32_t, int32_t,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::u8,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
          >(
            __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
            __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
          );
      }

#pragma unroll
      for (int ll = 0; ll < 4; ll++) {
        auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
        auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);
        auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * ll);

        ccTile =
          sycl::ext::intel::esimd::xmx::dpas
          <8, 8, int32_t, int32_t, uint32_t, int32_t,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::u8,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
          >(
            __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
            __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
            __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
          );
      }

      //float vsTempFp32 = vsTemp * (1.0f / 255.0f);
#pragma unroll
      for (int32_t kk = 0; kk < 16; kk++) {
        __ESIMD_NS::simd<fp16, 32> fp32Temp1;
        fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk + 512);
        finalOutput.select<32, 1>(16 * 32 * nn + 32 * kk) += fp32Temp1 * fp16Qv.template replicate_vs_w_hs<2, 1, 16, 0>(32 * nn + 2 * kk);
      }
    }
#else
#pragma unroll
    for (int nn = 0; nn < 4; nn++) {
#pragma unroll
      for (int l = 0; l < 2; l++) {
#pragma unroll
        for (int ll = 0; ll < 4; ll++) {
          fp16TempBuffer.select<256, 1>(256 * ll) = __ESIMD_NS::slm_block_load<fp16, 256>(slmOffsetC0 +
            16 * 128 * nn * sizeof(fp16) +
            1024 * l * sizeof(fp16) +
            ll * 256 * sizeof(fp16));
        }

#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
          auto aaTile = tempQkFp16.select<256, 1>(256 * nn);
          auto bbTile = fp16TempBuffer.select<128, 1>(128 * ll);

          ccTile = sycl::ext::intel::esimd::xmx::dpas<8, 8, fp16, fp16, fp16, fp16>(
            __ESIMD_NS::simd<fp16, 128>(ccTile.data()),
            __ESIMD_NS::simd<fp16, 256>(aaTile.data()),
            __ESIMD_NS::simd<fp16, 128>(bbTile.data()));
        }
      }
    }
#endif
    //barrier();
  }

  __ESIMD_NS::simd<float, 32> softMaxDividor;
  __ESIMD_NS::simd_mask<16> mask;
  //softMaxDividor.select<16, 1>(0) = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
  softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
  softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
  softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
  for (int kk = 0; kk < 64; kk++) {
    __ESIMD_NS::simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
    f16Temp = f16Temp * softMaxDividor;
    finalOutput.select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
    finalOutput.select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
  }

  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  mask = simdOffsets <= boundaryQ;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    __ESIMD_ENS::lsc_scatter<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back,
      16,
      uint32_t
    >((uint32_t*)out, simdOffsets, finalOutput.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk), mask);
    simdOffsets += 8 * sizeof(uint32_t);
  }
}



// ESIMD_INLINE void sageAttnGqa12paged(
//   uint8_t* qState,
//   uint8_t* kState,
//   uint8_t* vState,
//   fp16* qs,
//   fp16* ks,
//   fp16* vs,
//   uint32_t* cuSeqQuery,
//   uint32_t* cuSeqKv,
//   uint32_t* kvCacheBlockTable,
//   uint8_t* out,
//   uint32_t headQ,
//   uint32_t headKv,
//   sycl::nd_item<2>& ndi
//   ) {
//   constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
//   constexpr uint32_t slmSizeK = 64 * 128 * sizeof(int8_t);
//   constexpr uint32_t slmSizeV = 64 * 128 * sizeof(fp16);
//   constexpr uint32_t slmSizeSoftMaxSum = 256 * sizeof(float);
//   constexpr uint32_t slmSizeSoftMaxHistoric = 256 * sizeof(float);
//   constexpr uint32_t slmSizeSoftMaxCompensation = 256 * sizeof(float);
//   constexpr uint32_t slmSizeKs = 64 * sizeof(fp16);
//   constexpr uint32_t slmSize =
//     slmSizeK +
//     slmSizeV +
//     slmSizeSoftMaxSum +
//     slmSizeSoftMaxHistoric + 
//     slmSizeSoftMaxCompensation +
//     slmSizeKs;
//   constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
//   __ESIMD_NS::slm_init(slmSize);
//   constexpr uint32_t slmOffsetBaseK = 0;
//   constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
//   constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
//   constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
//   constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
//   constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
//   constexpr int32_t batchIdx = 0; // v / headKv;
//   constexpr int32_t groupGqaRatio = 4;
//   constexpr uint32_t powerOf2BlockSize = 9;
//   constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
//   constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
//   constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
//   constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
//   constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16

//   int32_t localLinearId = ndi.get_local_id(0);
//   int32_t hh = localLinearId & 0x3;
//   int32_t vv = localLinearId >> 2;
//   int32_t hhq = localLinearId & 0x3;
//   int32_t vvq = localLinearId >> 2;
//   int32_t hhv = localLinearId & 0xf;
//   int32_t vvv = localLinearId >> 4;
//   int32_t h =  ndi.get_group(0);
//   int32_t v = ndi.get_group(1); // [0, 24)
//   int32_t headIdxQ = v * 4;
//   int32_t headIdxKv = v / 3;
//   uint32_t kvSeqLen;
//   uint32_t activationLength;
//   uint32_t startActivationIdx;
//   __ESIMD_NS::simd<uint32_t, 2> kvCoord;
//   __ESIMD_NS::simd<uint32_t, 2> queryCoord;
//   kvCoord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqKv);
//   queryCoord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqQuery);
//   //if (kvSeqLen >= activationLength) {
//   //  startActivationIdx = kvSeqLen - activationLength;
//   //} else {
//   //  startActivationIdx = 0;
//   //}
//   headIdxQ = headIdxQ + vvq;
//   __ESIMD_NS::simd<int8_t, 16 * 128> i8QState;
//   __ESIMD_NS::simd<fp16, 32> fp16Qs;
//   __ESIMD_NS::simd<int32_t, 16 * 64> i32TempBuffer;
// //  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
//   __ESIMD_NS::simd<float, 512> tempBuffer;
//   auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
//   auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 3);
//   auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
//   auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
//   auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
//   __ESIMD_NS::simd<fp16, 16 * 128> finalOutput = 0;
//   __ESIMD_NS::simd<fp16, 128> fp16Qv;
//   fp16 vsTemp = 1.0f;
//   __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
//   __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
//   __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
//   __ESIMD_NS::simd<uint32_t, 16> blockIdx;

//   kvSeqLen = kvCoord[1] - kvCoord[0];
//   activationLength = queryCoord[1] - queryCoord[0];
//   startActivationIdx = kvSeqLen - activationLength;
  
//   int32_t kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
//   kvSeqOutLoopCount = kvSeqOutLoopCount >> 6;
//   unsigned int offsetQBase = headIdxQ * 128 * sizeof(int8_t);
//   uint32_t boundaryQ = activationLength - 1;
//   unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
//   __ESIMD_NS::simd_mask<16> mask;
//   simdOffsets = baseOffsetInc16AsVector;
//   simdOffsets = simdOffsets + 64 * h + 16 * hhq;
//   mask = simdOffsets <= boundaryQ;
//   simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
//   simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + offsetOutputBase;

//   finalOutput.template bit_cast_view<uint32_t>() = 0x12345678;

// #pragma unroll
//   for (int kk = 0; kk < 8; kk++) {
//     __ESIMD_ENS::lsc_scatter<
//       uint32_t,
//       8,
//       __ESIMD_ENS::lsc_data_size::u32,
//       __ESIMD_ENS::cache_hint::write_back,
//       __ESIMD_ENS::cache_hint::write_back,
//       16,
//       uint32_t
//     >((uint32_t*)out, simdOffsets, finalOutput.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk), mask);
//     simdOffsets += 8 * sizeof(uint32_t);
//   }
// }

ESIMD_INLINE void sageAttnGqa12paged(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  fp16* qs,
  fp16* ks,
  fp16* vs,
  uint32_t* cuSeqQuery,
  uint32_t* cuSeqKv,
  uint32_t* kvCacheBlockTable,
  uint8_t* out,
  uint32_t headQ,
  uint32_t headKv,
  sycl::nd_item<2>& ndi
  ) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr uint32_t slmSizeK = 64 * 128 * sizeof(int8_t);
  constexpr uint32_t slmSizeV = 64 * 128 * sizeof(fp16);
  constexpr uint32_t slmSizeSoftMaxSum = 256 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxHistoric = 256 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxCompensation = 256 * sizeof(float);
  constexpr uint32_t slmSizeKs = 64 * sizeof(fp16);
  constexpr uint32_t slmSize =
    slmSizeK +
    slmSizeV +
    slmSizeSoftMaxSum +
    slmSizeSoftMaxHistoric + 
    slmSizeSoftMaxCompensation +
    slmSizeKs;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseK = 0;
  constexpr uint32_t slmOffsetBaseV = slmOffsetBaseK + slmSizeK;
  constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
  constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
  constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
  constexpr uint32_t slmOffsetBaseKs = slmOffsetBaseSoftMaxCompensation + slmSizeSoftMaxCompensation;
  constexpr int32_t batchIdx = 0; // v / headKv;
  constexpr int32_t groupGqaRatio = 4;
  constexpr uint32_t powerOf2BlockSize = 9;
  constexpr uint32_t loopsPerMacroBlockShift = powerOf2BlockSize - 6; // 3 for 512, 
  constexpr uint32_t perLoopMacroBlockMask = (1 << loopsPerMacroBlockShift) - 1; // 0x7
  constexpr uint32_t blockSize = 1 << powerOf2BlockSize; // 9 for 512
  constexpr uint32_t blockTableReloadShift = (powerOf2BlockSize - 6) + 4; // 7 for 512x16
  constexpr uint32_t blockTableReloadCheckMask = (1 << blockTableReloadShift) - 1; // 127 for 512x16

  int32_t localLinearId = ndi.get_local_id(0);
  int32_t hhq = localLinearId & 0x3;
  int32_t vvq = localLinearId >> 2;
  int32_t hhv = localLinearId & 0xf;
  int32_t vvv = localLinearId >> 4;
  int32_t h =  ndi.get_group(0);
  int32_t v = ndi.get_group(1); // [0, 24)
  int32_t headIdxQ = v * 4;
  int32_t headIdxKv = v / 3;
  uint32_t kvSeqLen;
  uint32_t activationLength;
  uint32_t startActivationIdx;
  __ESIMD_NS::simd<uint32_t, 2> kvCoord;
  __ESIMD_NS::simd<uint32_t, 2> queryCoord;
  kvCoord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqKv);
  queryCoord = __ESIMD_NS::block_load<uint32_t, 2>(cuSeqQuery);
  //if (kvSeqLen >= activationLength) {
  //  startActivationIdx = kvSeqLen - activationLength;
  //} else {
  //  startActivationIdx = 0;
  //}
  headIdxQ = headIdxQ + vvq;
  __ESIMD_NS::simd<int8_t, 16 * 128> i8QState;
  __ESIMD_NS::simd<fp16, 32> fp16Qs;
  __ESIMD_NS::simd<int32_t, 16 * 64> i32TempBuffer;
//  auto qkTemp = i32TempBuffer.select<16 * 64, 1>(0);
  __ESIMD_NS::simd<float, 512> tempBuffer;
  auto i8KState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(0);
  auto i8VState = tempBuffer.template bit_cast_view<int8_t>().select<512, 1>(512 * 3);
  auto fp16TempBuffer = tempBuffer.template bit_cast_view<fp16>();
  auto i8TempBuffer = tempBuffer.template bit_cast_view<int8_t>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  __ESIMD_NS::simd<fp16, 16 * 128> finalOutput = 0;
  __ESIMD_NS::simd<fp16, 128> fp16Qv;
  fp16 vsTemp;
  __ESIMD_NS::simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> softMaxThresh(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);
  __ESIMD_NS::simd<uint32_t, 16> blockIdx;

  kvSeqLen = kvCoord[1] - kvCoord[0];
  activationLength = queryCoord[1] - queryCoord[0];
  startActivationIdx = kvSeqLen - activationLength;
  
  int32_t kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
  kvSeqOutLoopCount = (kvSeqOutLoopCount + 63) >> 6;
  uint32_t myLastFullAttnLoopIdx = (64 * h + startActivationIdx + 16 * hhq) >> 6;
  unsigned int offsetQBase = headIdxQ * 128 * sizeof(int8_t);
  unsigned int offsetOutputBase = offsetQBase * sizeof(fp16);
  unsigned int kvHiddenDim = headKv * 128;
  unsigned int offsetBaseK = (headIdxKv * 128 + localLinearId * headKv * 128) * sizeof(int8_t);
  unsigned int offsetK;
  unsigned int slmOffsetK = slmOffsetBaseK + localLinearId * 32 * sizeof(int8_t);
  unsigned int slmOffsetV = slmOffsetBaseV + hhv * 128 * sizeof(fp16);
  unsigned int slmOffsetSoftMaxHistoric = slmOffsetBaseSoftMaxHistoric + localLinearId * 16 * sizeof(float);
  unsigned int slmOffsetSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

  unsigned int slmOffsetCSoftMaxCompensation = slmOffsetBaseSoftMaxCompensation + localLinearId * 16 * sizeof(float);

  uint32_t boundaryQ = activationLength - 1;

  if (64 * h >= activationLength) {
    return;
  }

  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * 128 * sizeof(int8_t) + offsetQBase;

#pragma unroll
  for (int k = 0; k < 4; k++) {
    i8QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * k) =
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
  constexpr float MIN_VALUE = FP32_MIN * matMulQuantCoeff; 
  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * sizeof(fp16) + headIdxQ * sizeof(fp16);
  fp16Qs.select<16, 1>(0) =
    __ESIMD_ENS::lsc_gather<
    fp16,
    1,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached,
    16,
    uint32_t
    >((fp16*)qs, simdOffsets);

  softMaxThresh = softMaxThresh + 16 * hhq + 64 * h + startActivationIdx;
  
  __ESIMD_NS::simd<float, 16> fp32SoftMaxTemp = 0.0f;
  __ESIMD_NS::simd<float, 16> fp32HistoricMaxTemp = MIN_VALUE;
  blockIdx = __ESIMD_ENS::lsc_block_load<
    uint32_t,
    16,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + 0);

#ifdef QKV_INT8
  fp16Qv = 
    __ESIMD_ENS::lsc_block_load<
    fp16,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((fp16*)vs + 128 * headIdxKv);

  fp16Qv = fp16Qv * (fp16)(1.0f / 255.0f);
#else
  fp16TempBuffer.select<8, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    fp16,
    8,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((fp16*)vs + 128 * headIdxKv + 8 * hhv);

#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    fp16Qv.select<32, 1>(32 * kk) = fp16TempBuffer.template replicate_vs_w_hs<2, 1, 16, 0>(2 * kk);
  }
#endif

  fp16Qs.select<16, 1>(16) = fp16Qs.select<16, 1>(0);

  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    uint32_t whichIdx = loopIdx & blockTableReloadCheckMask;
    uint32_t macroBlock = blockIdx[whichIdx >> loopsPerMacroBlockShift] * blockSize;
    uint32_t subBlock = (whichIdx & perLoopMacroBlockMask) * 64;
    subBlock = macroBlock + subBlock;
    offsetK = subBlock * kvHiddenDim + offsetBaseK;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      __ESIMD_ENS::lsc_prefetch<
        uint8_t,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >((uint8_t*)vState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
    }
#if 1
    {
      __ESIMD_NS::simd<uint32_t, 16> ksOffset;
      __ESIMD_NS::simd<fp16, 4> ksTemp;
      ksOffset = localLinearId + baseOffsetInc16AsVector * 16 + 64 * loopIdx;
      ksOffset.merge(0, ksOffset >= kvSeqLen);
      ksOffset = ksOffset * headKv * sizeof(fp16) + headIdxKv * sizeof(fp16);
      ksTemp =
        __ESIMD_ENS::lsc_gather<
        fp16,
        1,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        4,
        uint32_t
        >((fp16*)ks, ksOffset.select<4, 1>(0));
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
//        offsetK = __ESIMD_NS::min(offsetK, boundaryKv);
        if (loopIdx * 64 + kk * 16 + localLinearId >= kvSeqLen) {
          offsetK = subBlock * kvHiddenDim;
        }
        i8KState.template bit_cast_view<uint8_t>().template select<128, 1>(128 * kk) =
          __ESIMD_ENS::lsc_block_load<
          uint8_t,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState + offsetK);

        offsetK += 16 * kvHiddenDim * sizeof(int8_t);
      }
      ksOffset = baseOffsetInc16AsVector * 16 + localLinearId;
      ksOffset = ksOffset * sizeof(fp16) + slmOffsetBaseKs;
      __ESIMD_ENS::lsc_slm_scatter<
        fp16,
        1,
        __ESIMD_ENS::lsc_data_size::u16,
        4>(ksOffset.select<4, 1>(0), ksTemp.select<4, 1>(0));

      if (whichIdx == blockTableReloadCheckMask) {
        uint32_t tempOffsetForBlockTable = ((loopIdx + 1) >> blockTableReloadShift) << 4;
        blockIdx =
          __ESIMD_ENS::lsc_block_load<
          uint32_t,
          16,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint32_t*)kvCacheBlockTable + tempOffsetForBlockTable);
      }

#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          slm_block_store<int8_t, 32>(slmOffsetK + k * 16 * 128 * sizeof(int8_t) + kk * 16 * 32 * sizeof(int8_t), i8KState.select<32, 1>(128 * k + 32 * kk));
        }
      }
    }
#else
#pragma unroll
      //16 x4  64 Ks
      for (int kk = 0; kk < 4; kk++) {
          uint32_t ksOffset = localLinearId + 16 * kk + 64 * loopIdx;
          fp16 ksTemp;
          ksTemp = ks[ksOffset];
          slm_block_store<fp16, 1>(slmOffsetBaseKs + localLinearId * sizeof(fp16) + 16 * kk * sizeof(fp16), ksTemp);
      }
      uint32_t kOffset_2d = (64 * loopIdx + (localLinearId / 4) * 16) * headKv * 128 /*token*/ + headIdxKv * 128 /*header*/ + (localLinearId % 4) * 32/*block*/;
      i8KState.select<512, 1>(0) = __ESIMD_ENS::lsc_load_2d<
          uint8_t, 32, 16, 1,  //8x32
          false, false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>((uint8_t*)kState,
              headKv * 128 - 1, kvSeqLen - 1, headKv * 128 - 1,
              headIdxKv * 128 + (localLinearId % 4) * 32, 64 * loopIdx + (localLinearId / 4) * 16);
      slm_block_store<int8_t, 512>(slmOffsetBaseK + (localLinearId / 2) * 2 * 512 + (localLinearId % 2) * 512, i8KState.select<512, 1>(0));
#endif
    auto tempOutput = i32TempBuffer.template bit_cast_view<float>().select<16 * 64, 1>(0);
    auto tempQkInt8 = i32TempBuffer.template bit_cast_view<uint8_t>();
    auto tempQkFp16 = i32TempBuffer.template bit_cast_view<fp16>();
    barrier();
    i32TempBuffer = 0;

    {
      __ESIMD_NS::simd<fp16, 64> ksAllTemp = __ESIMD_NS::slm_block_load<fp16, 64>(slmOffsetBaseKs);
#pragma unroll
      for (int32_t nn = 0; nn < 4; nn++) {
#pragma unroll
        for (int32_t l = 0; l < 2; l++) {
#pragma unroll
          for (int32_t ll = 0; ll < 2; ll++) {
            i8TempBuffer.select<512, 1>(512 * ll) = __ESIMD_NS::slm_block_load<int8_t, 512>(
              slmOffsetBaseK + 16 * 128 * nn * sizeof(int8_t) +
              1024 * l * sizeof(int8_t) +
              512 * ll * sizeof(int8_t));
          }

#pragma unroll
          for (int32_t k = 0; k < 2; k++) {
#pragma unroll
            for (int32_t kk = 0; kk < 2; kk++) {
              auto ccTile = i32TempBuffer.select<128, 1>(256 * nn + 128 * kk);
              auto aaTile = i8QState.template bit_cast_view<int32_t>().select<128, 1>(256 * l + 128 * k);
              auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(128 * k + 64 * kk);

              ccTile =
                sycl::ext::intel::esimd::xmx::dpas
                <8, 8, int32_t, int32_t, int32_t, int32_t,
                sycl::ext::intel::esimd::xmx::dpas_argument_type::s8,
                sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
                >(
                  __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
                  __ESIMD_NS::simd<int32_t, 128>(aaTile.data()),
                  __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
                );
            }
          }
        }
#pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          __ESIMD_NS::simd<float, 32> fp32Temp0;
          __ESIMD_NS::simd<float, 32> fp32Temp1;
          fp32Temp0 = ksAllTemp.template replicate_vs_w_hs<2, 1, 16, 0>(16 * nn + 2 * kk) * fp16Qs;
          fp32Temp1 = i32TempBuffer.select<32, 1>(256 * nn + 32 * kk);
  #ifdef MERGE_QUANT
          tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1*matMulQuantCoeff;
  #else
          tempOutput.select<32, 1>(256 * nn + 32 * kk) = fp32Temp0 * fp32Temp1
  #endif
        }
      }
	  
      if (whichIdx == blockTableReloadCheckMask) {
        offsetK = blockIdx[0] * kvHiddenDim * blockSize + offsetBaseK;
      }

#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        __ESIMD_ENS::lsc_prefetch<
          uint8_t,
          128,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached
        >((uint8_t*)kState + offsetK + kk * 16 * kvHiddenDim * sizeof(int8_t));
      }
    }

#ifdef QKV_INT8
//    vsTemp = vs[loopIdx];
	  __ESIMD_NS::simd<uint32_t,64> simd32Offsets00;
    simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
    simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(0) + 16;
    simd32Offsets00.select<32, 1>(0) = simd32Offsets00.select<32, 1>(0) + 64 * loopIdx;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      simd32Offsets00.select<16, 1>(32) = baseOffsetInc16AsVector;
      simd32Offsets00.select<16, 1>(32 + 16) = baseOffsetInc16AsVector + 16;
      simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) + 32 * kk;
      simd32Offsets00.select<32, 1>(32).merge(0, simd32Offsets00.select<32, 1>(0) >= kvSeqLen);
      simd32Offsets00.select<32, 1>(32) = simd32Offsets00.select<32, 1>(32) * kvHiddenDim * sizeof(int8_t) +
        subBlock * kvHiddenDim * sizeof(int8_t) +
        (headIdxKv * 128) * sizeof(int8_t) +
        hhv * 8 * sizeof(int8_t);

      i8VState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        2,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        32,
        uint32_t
        >((uint32_t*)vState, simd32Offsets00.select<32, 1>(32));

      simd32Offsets00.select<32, 1>(0) += 32;
    }
#else
    __ESIMD_NS::simd<uint32_t,32> simd32Offsets00;
    simd32Offsets00.select<16, 1>(0) = baseOffsetInc16AsVector;
    simd32Offsets00.select<16, 1>(0) = simd32Offsets00.select<16, 1>(0) + 64 * loopIdx;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      simd32Offsets00.select<16, 1>(16) = baseOffsetInc16AsVector + 16 * kk;
      simd32Offsets00.select<16, 1>(16).merge(0, simd32Offsets00.select<16, 1>(0) >= kvSeqLen);
      simd32Offsets00.select<16, 1>(16) = simd32Offsets00.select<16, 1>(16) * kvHiddenDim * sizeof(int8_t) +
        subBlock * kvHiddenDim * sizeof(int8_t) + 
        (headIdxKv * 128) * sizeof(int8_t) +
        hhv * 8 * sizeof(int8_t);

      i8VState.template bit_cast_view<uint32_t>().template select<32, 1>(32 * kk) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        2,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)vState, simd32Offsets00.select<16, 1>(16));
      simd32Offsets00.select<16, 1>(0) += 16;
    }
#endif

    {
     //   auto fp32SoftMaxTemp = tempBuffer.select<16, 1>(0);
     //   auto fp32HistoricMaxTemp = tempBuffer.select<16, 1>(16);
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(32);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(48);
      auto softmaxPositions = ui32Temp.select<64, 1>(64);

    //  fp32SoftMaxTemp = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
    //  fp32HistoricMaxTemp = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetSoftMaxHistoric);
#if  VERSION_0
 
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions[kk]);
        }
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
      }
   

#else
    if (loopIdx >= myLastFullAttnLoopIdx) {
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
      softmaxPositions.select<16, 1>(16) = softmaxPositions.select<16, 1>(0) + 16;
      softmaxPositions.select<32, 1>(32) = softmaxPositions.select<32, 1>(0) + 16 * 2;
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softmaxPositions.replicate_w<16, 1>(kk) >= kvSeqLen);
        tempOutput.select<16, 1>(16 * kk).merge(FP32_MIN, softMaxThresh.select<16, 1>(0) < softmaxPositions.template replicate_w<16, 1>(kk));
      }
    }
#endif
#ifndef MERGE_QUANT
     tempOutput.select<16 * 64, 1>(0) = tempOutput.select<16 * 64, 1>(0) * matMulQuantCoeff;

#endif
  fp32CurrentMaxTemp = fp32HistoricMaxTemp;

#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(16 * kk), fp32CurrentMaxTemp);
      }

#else
      __ESIMD_NS::simd<float,8*16> ttemp;
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = __ESIMD_NS::max<float, 16, float>(tempOutput.select<16, 1>(32*kk), tempOutput.select<16, 1>(32*kk+16));
      }
#pragma unroll
    for(int kkk=0;kkk<6;++kkk){
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(16*kk), tempOutput.select<16, 1>((8*kkk+kk)*16+16*16));
      }
    }
    ttemp.select<64, 1>(0)=__ESIMD_NS::max<float, 64, float>( ttemp.select<64, 1>(0), ttemp.select<64, 1>(16*4));
    ttemp.select<32, 1>(0)=__ESIMD_NS::max<float, 32, float>( ttemp.select<32, 1>(0), ttemp.select<32, 1>(16*2));
    ttemp.select<16, 1>(0)=__ESIMD_NS::max<float, 16, float>( ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
    #if  VERSION_0// Positive 
        fp32CurrentMaxTemp = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), fp32CurrentMaxTemp);
    #else
        fp32CurrentMaxTemp.merge(ttemp.select<16, 1>(0)*sycl::ext::intel::esimd::detail::log2e, ttemp.select<16, 1>(0)*sycl::ext::intel::esimd::detail::log2e > fp32CurrentMaxTemp);
    #endif

#endif

#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) - fp32CurrentMaxTemp.select<16, 1>(0);
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#elif 1  // Positive 
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e - fp32CurrentMaxTemp.select<16, 1>(0);
      }
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#else
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk)  - fp32CurrentMaxTemp.select<16, 1>(0);
      }
#pragma unroll
     for (int kk = 0; kk < 64; kk++) { 
        tempOutput.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(16 * kk) * sycl::ext::intel::esimd::detail::log2e;
      }
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        tempOutput.select<16, 1>(16 * kk) = __ESIMD_NS::exp2<float, 16, float>(tempOutput.select<16, 1>(16 * kk));
      }
#endif

#if 0
      fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
      fp32SoftMaxCompensation = __ESIMD_NS::pow<float, 16, float>(2.718281828459f, fp32SoftMaxCompensation);
#else
      fp32SoftMaxCompensation = fp32HistoricMaxTemp - fp32CurrentMaxTemp;
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>( fp32SoftMaxCompensation);
#endif
      if (loopIdx != 0) {
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
      }
#if VERSION_0
#pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + tempOutput.select<16, 1>(16 * kk);
      }
#else  
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        ttemp.select<16, 1>(16*kk) = tempOutput.select<16, 1>(32*kk)+ tempOutput.select<16, 1>(32*kk+16);
      }
#pragma unroll
      for(int k=0;k<6;++k){
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            ttemp.select<16, 1>(16*kk) = ttemp.select<16, 1>(16*kk)+ tempOutput.select<16, 1>((8*k+kk)*16+16*16);
        }
    }
    ttemp.select<64, 1>(0)= ttemp.select<64, 1>(0)+ ttemp.select<64, 1>(16*4);
    ttemp.select<32, 1>(0)= ttemp.select<32, 1>(0)+ ttemp.select<32, 1>(16*2);
    ttemp.select<16, 1>(0)= ttemp.select<16, 1>(0)+ ttemp.select<16, 1>(16);
    fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) +ttemp.select<16, 1>(0); 
#endif
      fp32HistoricMaxTemp =fp32CurrentMaxTemp;
     // slm_block_store<float, 16>(slmOffsetSoftMaxHistoric, fp32CurrentMaxTemp);
     // slm_block_store<float, 16>(slmOffsetSoftMaxCompensation, fp32SoftMaxCompensation);
     // slm_block_store<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float), fp32SoftMaxTemp);

#ifdef QKV_INT8
#pragma unroll
      for (int kk = 0; kk < 16; kk++) {

        __ESIMD_NS::simd<float, 64> shuffleTemp;
#if  0
        shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f;
        shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f;

        shuffleTemp = shuffleTemp + 0.5f;
#else
        shuffleTemp.select<32, 2>(0) = tempOutput.select<32, 1>(64 * kk) * 255.0f + 0.5f;
        shuffleTemp.select<32, 2>(1) = tempOutput.select<32, 1>(64 * kk + 32) * 255.0f+0.5f;

#endif
        shuffleTemp = __ESIMD_NS::rndd<float>(shuffleTemp);
        tempQkInt8.select<32, 2>(64 * kk) = shuffleTemp.select<32, 1>(0);
        tempQkInt8.select<32, 2>(64 * kk + 1) = shuffleTemp.select<32, 1>(32);
      }

#pragma unroll
      for (int k = 0; k < 2; k++) {
        __ESIMD_NS::simd<int8_t, 256> shuffleTemp;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          shuffleTemp.select<32, 1>(128 * kk + 0) = i8VState.select<32, 4>(256 * k + 128 * kk + 0);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 1) = i8VState.select<32, 4>(256 * k + 128 * kk + 1);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 2) = i8VState.select<32, 4>(256 * k + 128 * kk + 2);
          shuffleTemp.select<32, 1>(128 * kk + 32 * 3) = i8VState.select<32, 4>(256 * k + 128 * kk + 3);
        }
        slm_block_store<int32_t, 64>(slmOffsetV + k * 128 * 32 * sizeof(int8_t), shuffleTemp.template bit_cast_view<int32_t>());
      }
#else
#pragma unroll
      for (int kk = 0; kk < 32; kk++) {
        __ESIMD_NS::simd<float, 32> shuffleTemp;
        shuffleTemp = tempOutput.select<32, 1>(32 * kk);
        tempQkFp16.select<16, 2>(32 * kk) = shuffleTemp.select<16, 1>(0);
        tempQkFp16.select<16, 2>(32 * kk + 1) = shuffleTemp.select<16, 1>(16);
      }

#pragma unroll
      for (int k = 0; k < 4; k++) {
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
          shuffleTemp.select<32, 1>(32 * kk) = shuffleTemp.select<32, 1>(32 * kk) * fp16Qv.select<32, 1>(32 * kk);
        }
        slm_block_store<fp16, 128>(slmOffsetV + k * 128 * 16 * sizeof(fp16), shuffleTemp);
      }
#endif

      tempBuffer.select<16, 1>(0) = fp32SoftMaxCompensation;
      tempBuffer.select<16, 1>(16) = tempBuffer.select<16, 1>(0);
      if (loopIdx != 0) {
        __ESIMD_NS::simd<fp16, 32> fp16Temp0;
        fp16Temp0 = tempBuffer.select<32, 1>(0);
#pragma unroll
        for (int kk = 0; kk < 64; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * fp16Temp0;
        }
      }
    }

    barrier();
#ifdef QKV_INT8
#pragma unroll
    for (int nn = 0; nn < 4; nn++) {
#pragma unroll
      for (int l = 0; l < 2; l++) {
#pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          i8TempBuffer.select<512, 1>(1024 * l + 512 * ll) = __ESIMD_NS::slm_block_load<int8_t, 512>(
            slmOffsetBaseV +
            32 * 128 * l * sizeof(int8_t) +
            32 * 32 * nn * sizeof(int8_t) +
            ll * 512 * sizeof(int8_t));
        }
      }
#pragma unroll
      for (int ll = 0; ll < 4; ll++) {
        auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
        auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(0);
        auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(64 * ll);

        ccTile =
          sycl::ext::intel::esimd::xmx::dpas
          <8, 8, int32_t, uint32_t, int32_t,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::u8,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
          >(
            __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
            __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
          );
      }

#pragma unroll
      for (int ll = 0; ll < 4; ll++) {
        auto ccTile = i32TempBuffer.select<128, 1>(128 * ll + 512);
        auto aaTile = tempQkInt8.template bit_cast_view<uint32_t>().select<128, 1>(128);
        auto bbTile = i8TempBuffer.template bit_cast_view<int32_t>().select<64, 1>(256 + 64 * ll);

        ccTile =
          sycl::ext::intel::esimd::xmx::dpas
          <8, 8, int32_t, int32_t, uint32_t, int32_t,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::u8,
          sycl::ext::intel::esimd::xmx::dpas_argument_type::s8
          >(
            __ESIMD_NS::simd<int32_t, 128>(ccTile.data()),
            __ESIMD_NS::simd<uint32_t, 128>(aaTile.data()),
            __ESIMD_NS::simd<int32_t, 64>(bbTile.data())
          );
      }

      //float vsTempFp32 = vsTemp * (1.0f / 255.0f);
#pragma unroll
      for (int32_t kk = 0; kk < 16; kk++) {
        __ESIMD_NS::simd<fp16, 32> fp32Temp1;
        fp32Temp1.select<32, 1>(0) = i32TempBuffer.select<32, 1>(32 * kk + 512);
        finalOutput.select<32, 1>(16 * 32 * nn + 32 * kk) += fp32Temp1 * fp16Qv.template replicate_vs_w_hs<2, 1, 16, 0>(32 * nn + 2 * kk);
      }
    }
#else
#pragma unroll
    for (int nn = 0; nn < 4; nn++) {
#pragma unroll
      for (int l = 0; l < 2; l++) {
#pragma unroll
        for (int ll = 0; ll < 4; ll++) {
          fp16TempBuffer.select<256, 1>(256 * ll) = __ESIMD_NS::slm_block_load<fp16, 256>(slmOffsetBaseV +
            16 * 128 * nn * sizeof(fp16) +
            1024 * l * sizeof(fp16) +
            ll * 256 * sizeof(fp16));
        }

#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
          auto aaTile = tempQkFp16.select<256, 1>(256 * nn);
          auto bbTile = fp16TempBuffer.select<128, 1>(128 * ll);

          ccTile = sycl::ext::intel::esimd::xmx::dpas<8, 8, fp16, fp16, fp16, fp16>(
            __ESIMD_NS::simd<fp16, 128>(ccTile.data()),
            __ESIMD_NS::simd<fp16, 256>(aaTile.data()),
            __ESIMD_NS::simd<fp16, 128>(bbTile.data()));
        }
      }
    }
#endif
    //barrier();
  }

  __ESIMD_NS::simd<float, 32> softMaxDividor;
  __ESIMD_NS::simd_mask<16> mask;
  //softMaxDividor.select<16, 1>(0) = __ESIMD_NS::slm_block_load<float, 16>(slmOffsetBaseSoftMaxSum + localLinearId * 16 * sizeof(float));
  softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
  softMaxDividor.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
  softMaxDividor = 1.0f / softMaxDividor;
#pragma unroll
  for (int kk = 0; kk < 64; kk++) {
    __ESIMD_NS::simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
    f16Temp = f16Temp * softMaxDividor;
    finalOutput.select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
    finalOutput.select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
  }

  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 64 * h + 16 * hhq;
  mask = simdOffsets <= boundaryQ;
  simdOffsets = __ESIMD_NS::min<uint32_t, 16, uint32_t>(simdOffsets, boundaryQ);
  simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + offsetOutputBase;

#pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    __ESIMD_ENS::lsc_scatter<
      uint32_t,
      8,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back,
      16,
      uint32_t
    >((uint32_t*)out, simdOffsets, finalOutput.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk), mask);
    simdOffsets += 8 * sizeof(uint32_t);
  }
}