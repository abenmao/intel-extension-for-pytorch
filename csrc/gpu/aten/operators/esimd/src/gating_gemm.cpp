#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <cfloat>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;

namespace esimd {
ESIMD_KERNEL_API cgf_t esimd_gating_gemm(
    uint8_t* weight,
    uint8_t* tokens,
    uint8_t* output,
    uint32_t outputRow,
    uint32_t outputCol,
    uint32_t hiddenDim) {
  assert(outputRow == 256);
  assert(hiddenDim == 7168);
//  assert((outputCol == 40) || (outputCol == 80));
  int32_t groupH;
  int32_t groupV;
  int32_t groupD;
  int32_t localH;
  int32_t localV;
  int32_t localD;

  groupH = 16;
  groupV = (outputCol + 15) / 16;
  localH = 8;
  localV = 1;

  sycl::range<2> GlobalRangeGatingGemm(
      localH * groupH, groupV * localV);
  sycl::range<2> LocalRangeGatingGemm(localH, localV);
  sycl::nd_range<2> RangeGatingGemm(
      GlobalRangeGatingGemm, LocalRangeGatingGemm);

  groupH = (outputCol + 207) / 208;
  groupV = (outputRow + 255) / 256;
  localH = 32;
  localV = 1;
    
  sycl::range<2> GlobalRangeGatingGemmPref(localH * groupH, groupV * localV);
  sycl::range<2> LocalRangeGatingGemmPref(localH, localV);
  sycl::nd_range<2> RangeGatingGemmPref(GlobalRangeGatingGemmPref, LocalRangeGatingGemmPref);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(RangeGatingGemm, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
      constexpr uint32_t slmSize = 16 * 16 * sizeof(float) * 8;
      constexpr uint32_t strideK = 128;
      __ESIMD_NS::slm_init(slmSize);
      uint32_t hiddenDimSizeInByte = hiddenDim * sizeof(fp16);
      int h = ndi.get_group(0);
      int v = ndi.get_group(1);
      int hh = ndi.get_local_id(0);
      int hha = hh;
      int hhpref = hh & 0x1;
      int vvpref = hh >> 1;
      uint32_t AYBase = h * 16;
      uint32_t BYBase = v * 16;
      uint32_t CYBase = v * 16;
      uint32_t APrefYBase = h * 16 + hhpref * 8;
      uint32_t BPrefYBase = v * 16 + hhpref * 8;
    
      uint32_t AXBase = hha * 16;
      uint32_t BXBase = hha * 32;
      uint32_t CXBase = h * 16;
      uint32_t APrefXBase = 128 + vvpref * 32;
      uint32_t loopCount = 28;
      simd<fp16, 16 * 32> aa;
      simd<fp16, 16 * 32> bb;
      simd<float, 16 * 16> cc = 0;
    
      uint32_t widthA = hiddenDimSizeInByte - 1;
      uint32_t heightA = outputRow - 1;
      uint32_t widthB = hiddenDimSizeInByte - 1;
      uint32_t heightB = outputCol - 1;
      uint32_t widthC = outputRow * sizeof(float) - 1;
      uint32_t heightC = outputCol - 1;
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA0(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase, AYBase);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA1(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase + 8, AYBase);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefA(
        (uint32_t*)weight, widthA, heightA, widthA, APrefXBase, APrefYBase);
    
      __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadB0(
        (fp16*)tokens, widthB, heightB, widthB, BXBase, BYBase);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefB(
        (uint32_t*)tokens, widthB, heightB, widthB, APrefXBase, BPrefYBase);
    
    #pragma unroll
      for (int nn = 0; nn < 28; nn++) {
        bb.select<512, 1>(512 * 0) =
          __ESIMD_ENS::lsc_load_2d<
          fp16,
          16,
          16,
          2,
          false,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(payloadB0);
    
        aa.template bit_cast_view<uint32_t>().select<128, 1>(128 * 0) =
          __ESIMD_ENS::lsc_load_2d<
          uint32_t,
          8,
          16,
          1,
          true,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(payloadA0);
    
        aa.template bit_cast_view<uint32_t>().select<128, 1>(128 * 1) =
          __ESIMD_ENS::lsc_load_2d<
          uint32_t,
          8,
          16,
          1,
          true,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(payloadA1);
    
    #pragma unroll
        for (int k = 0; k < 2; k++) {
    #pragma unroll
          for (int m = 0; m < 1; m++) {
    #pragma unroll
            for (int n = 0; n < 1; n++) {
              auto ccTile0 = cc.select<128, 1>(640 * m + 256 * n);
              auto ccTile1 = cc.select<128, 1>(640 * m + 256 * n + 128);
              auto aaTile = aa.select<256, 1>(512 * m + 256 * k);
              auto bbTile0 = bb.select<128, 1>(512 * n + 256 * k);
              auto bbTile1 = bb.select<128, 1>(512 * n + 256 * k + 128);
    
              ccTile0 = dpas<8, 8, float, float, fp16, fp16>(
                simd<float, 128>(ccTile0.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 128>(bbTile0.data()));
    
              ccTile1 = dpas<8, 8, float, float, fp16, fp16>(
                simd<float, 128>(ccTile1.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 128>(bbTile1.data()));
            }
          }
    
          payloadPrefA.set_x(APrefXBase);
          payloadPrefB.set_x(APrefXBase);
          __ESIMD_ENS::lsc_prefetch_2d<
            uint32_t,
            16,
            8,
            1,
            false,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached
          >(payloadPrefA);
    
          __ESIMD_ENS::lsc_prefetch_2d<
            uint32_t,
            16,
            8,
            1,
            false,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached
          >(payloadPrefB);
          APrefXBase += 16;
        }
    
        APrefXBase += 96;
        AXBase += strideK;
        BXBase += strideK * 2;
        payloadB0.set_x(BXBase);
        payloadA0.set_x(AXBase);
        payloadA1.set_x(AXBase + 8);
      }
    
      {
    #pragma unroll
        for (int k = 0; k < 16; k++) {
          slm_block_store<float, 16>(
            hh * 16 * sizeof(float) + 128 * k * sizeof(float),
            cc.select<16, 1>(16 * k));
        }
      }
    
      barrier();
      {
        uint32_t coorC0 = v * 16 + hh * 2;
        uint32_t coorC1 = coorC0 + 1;
    
        uint32_t offsetC0 = h * 16 + coorC0 * outputRow;
        uint32_t offsetC1 = h * 16 + coorC1 * outputRow;
    
        simd<float, 16 * 16> cc1;
    #pragma unroll
        for (int k = 0; k < 1; k++) {
          cc1.select<256, 1>(256 * k) = slm_block_load<float, 256>(
            256 * hh * sizeof(float));
        }
    
    #pragma unroll
        for (int k = 1; k < 4; k++) {
          cc1.select<32, 1>(0) = cc1.select<32, 1>(0) + cc1.select<32, 1>(32 * k);
          cc1.select<32, 1>(128) = cc1.select<32, 1>(128) + cc1.select<32, 1>(128 + 32 * k);
        }
    
        cc1.select<16, 1>(0) = cc1.select<16, 1>(0) + cc1.select<16, 1>(16);
        cc1.select<16, 1>(128) = cc1.select<16, 1>(128) + cc1.select<16, 1>(128 + 16);
    
        if (coorC0 < outputCol) {
          block_store<float, 16>((float*)output + offsetC0, cc1.select<16, 1>(0));
        }
        if (coorC1 < outputCol) {
          block_store<float, 16>((float*)output + offsetC1, cc1.select<16, 1>(128));
        }
      }
    });
  };

  cgf_t kernel_func_pref = [=](sycl::handler& cgh) {
    cgh.parallel_for(RangeGatingGemmPref, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
      constexpr uint32_t strideK = 16;
      uint32_t hiddenDimSizeInByte = hiddenDim * sizeof(fp16);
      int h = ndi.get_group(0);
      int v = ndi.get_group(1);
      int localLinearId = ndi.get_local_id(0);
      int hhc = localLinearId & 0x3;
      int vvc = localLinearId >> 2;
      int hhpref = localLinearId;
      int vvpref = 0;
      uint32_t AYBase = v * 256 + 64 * hhc;
      uint32_t BYBase = h * 208 + 26 * vvc;
      uint32_t CYBase = h * 208 + 26 * vvc;
      uint32_t APrefYBase = v * 256 + hhpref * 8;
      uint32_t BPrefYBase = h * 208 + hhpref * 8;
    
      uint32_t AXBase = 0;
      uint32_t BXBase = 0;
      uint32_t CXBase = h * 16;
      uint32_t APrefXBase = 0;
      uint32_t loopCount = hiddenDim >> 5;
      simd<fp16, 64 * 32> aa;
      simd<fp16, 26 * 32> bb;
      simd<float, 26 * 64> cc = 0;
    
      uint32_t widthA = hiddenDimSizeInByte - 1;
      uint32_t heightA = outputRow - 1;
      uint32_t widthB = hiddenDimSizeInByte - 1;
      uint32_t heightB = outputCol - 1;
      uint32_t widthC = outputRow * sizeof(float) - 1;
      uint32_t heightC = outputCol - 1;
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA0(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase, AYBase);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA1(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase, AYBase + 16 * 1);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA2(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase, AYBase + 16 * 2);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadA3(
        (uint32_t*)weight, widthA, heightA, widthA, AXBase, AYBase + 16 * 3);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefA(
        (uint32_t*)weight, widthA, heightA, widthA, APrefXBase, APrefYBase);
    
      __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadB0(
        (fp16*)tokens, widthB, heightB, widthB, BXBase, BYBase);
    
      __ESIMD_ENS::config_2d_mem_access<fp16, 16, 10, 2> payloadB1(
        (fp16*)tokens, widthB, heightB, widthB, BXBase, BYBase + 16);
    
      __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefB(
        (uint32_t*)tokens, widthB, heightB, widthB, APrefXBase, BPrefYBase);
    
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefA);
    
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefB);
    
      APrefXBase += 16;
      payloadPrefA.set_x(APrefXBase);
      payloadPrefB.set_x(APrefXBase);
    
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefA);
    
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefB);
      APrefXBase += 16;
      payloadPrefA.set_x(APrefXBase);
      payloadPrefB.set_x(APrefXBase);
      __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);
    
      for (int nn = 0; nn < loopCount; nn++) {
        bb.select<512, 1>(512 * 0) =
          __ESIMD_ENS::lsc_load_2d<
          fp16,
          16,
          16,
          2,
          false,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(payloadB0);
    
        bb.select<320, 1>(512 * 1) =
          __ESIMD_ENS::lsc_load_2d<
          fp16,
          16,
          10,
          2,
          false,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(payloadB1);
    
    #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          aa.template bit_cast_view<uint32_t>().select<128, 1>(512 * kk + 128 * 0) =
            __ESIMD_ENS::lsc_load_2d<
            uint32_t,
            8,
            16,
            1,
            true,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadA0);
    
          aa.template bit_cast_view<uint32_t>().select<128, 1>(512 * kk + 128 * 1) =
            __ESIMD_ENS::lsc_load_2d<
            uint32_t,
            8,
            16,
            1,
            true,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadA1);
    
          aa.template bit_cast_view<uint32_t>().select<128, 1>(512 * kk + 128 * 2) =
            __ESIMD_ENS::lsc_load_2d<
            uint32_t,
            8,
            16,
            1,
            true,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadA2);
    
          aa.template bit_cast_view<uint32_t>().select<128, 1>(512 * kk + 128 * 3) =
            __ESIMD_ENS::lsc_load_2d<
            uint32_t,
            8,
            16,
            1,
            true,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadA3);
    
          AXBase += 8;
          payloadA0.set_x(AXBase);
          payloadA1.set_x(AXBase);
          payloadA2.set_x(AXBase);
          payloadA3.set_x(AXBase);
        }
    
    #pragma unroll
        for (int k = 0; k < 2; k++) {
    #pragma unroll
          for (int m = 0; m < 4; m++) {
    #pragma unroll
            for (int n = 0; n < 1; n++) {
              auto ccTile0 = cc.select<128, 1>(26 * 16 * m + 256 * n);
              auto ccTile1 = cc.select<128, 1>(26 * 16 * m + 256 * n + 128);
              auto ccTile2 = cc.select<128, 1>(26 * 16 * m + 256 * n + 256);
              auto ccTile3 = cc.select<32, 1>(26 * 16 * m + 256 * n + 384);
    
              auto aaTile = aa.select<256, 1>(1024 * k + 256 * m);
              auto bbTile0 = bb.select<128, 1>(512 * 0 + 256 * k);
              auto bbTile1 = bb.select<128, 1>(512 * 0 + 256 * k + 128);
              auto bbTile2 = bb.select<128, 1>(512 * 1 + 160 * k);
              auto bbTile3 = bb.select<32, 1>(512 * 1 + 160 * k + 128);
    
              ccTile0 = dpas<8, 8, float, float, fp16, fp16>(
                simd<float, 128>(ccTile0.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 128>(bbTile0.data()));
    
              ccTile1 = dpas<8, 8, float, float, fp16, fp16>(
                simd<float, 128>(ccTile1.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 128>(bbTile1.data()));
    
              ccTile2 = dpas<8, 8, float, float, fp16, fp16>(
                simd<float, 128>(ccTile2.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 128>(bbTile2.data()));
    
              ccTile3 = dpas<8, 2, float, float, fp16, fp16>(
                simd<float, 32>(ccTile3.data()),
                simd<fp16, 256>(aaTile.data()),
                simd<fp16, 32>(bbTile3.data()));
            }
          }
        }
        __ESIMD_ENS::lsc_prefetch_2d<
          uint32_t,
          16,
          8,
          1,
          false,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached
        >(payloadPrefA);
    
        __ESIMD_ENS::lsc_prefetch_2d<
          uint32_t,
          16,
          8,
          1,
          false,
          false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached
        >(payloadPrefB);
        APrefXBase += 16;
        payloadPrefA.set_x(APrefXBase);
        payloadPrefB.set_x(APrefXBase);
    
    //    AXBase += strideK;
        BXBase += 32;
        payloadB0.set_x(BXBase);
        payloadB1.set_x(BXBase);
      }
    
      uint32_t coorC0 = h * 208 + vvc * 26;
      uint32_t offsetC0 = v * 256 + hhc * 64;
      offsetC0 = offsetC0 + coorC0 * outputRow;
    #pragma unroll
      for (int k = 0; k < 26; k++) {
        if (coorC0 < outputCol) {
          simd<float, 64> fp32Out;
    #pragma unroll
          for (int kk = 0; kk < 4; kk++) {
            fp32Out.select<16, 1>(16 * kk) = cc.select<16, 1>(16 * 26 * kk + 16 * k);
          }
          //fp32Out = 1.0f;
          block_store<float, 64>((float*)output + offsetC0, fp32Out.select<64, 1>(0));
          coorC0++;
          offsetC0 = offsetC0 + outputRow;
        }
      }
    });
  };
  if ((outputCol == 40) || (outputCol == 80)) {
    return kernel_func;
  } else {
    return kernel_func_pref;
  }
}
}