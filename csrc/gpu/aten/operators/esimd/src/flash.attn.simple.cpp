#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "kernel_apis.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <cfloat>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
#define FP32_MIN (-1.7e+38)
#include "flash.attn.b.mha128.h"

namespace esimd {
ESIMD_KERNEL_API cgf_t esimd_flash_attn_simple(
  uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* normAlpha, uint8_t* output,
  uint32_t qLen, uint32_t seqLen, uint32_t headQ, uint32_t headKv, uint32_t headDim) {
  assert(headDim == 128);
  int32_t groupH;
  int32_t groupV;
  int32_t groupD;
  int32_t localH;
  int32_t localV;
  int32_t localD;

  groupH = headQ;
  groupV = (qLen + 255) / 256;
  localH = 16;
  localV = 1;

  sycl::range<2> GlobalRangeFlashAttn(
      localH * groupH, groupV * localV);
  sycl::range<2> LocalRangeFlashAttn(localH, localV);
  sycl::nd_range<2> RangeFlashAttn(
    GlobalRangeFlashAttn, LocalRangeFlashAttn);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(RangeFlashAttn, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
      flashAttnBMha128(
        qState, kState, vState, normAlpha, output,
        qLen, seqLen, headQ, ndi);
    });
  };

  switch(headDim) {
  case 128:
    return kernel_func;
    break;
  default:
    return kernel_func;
    break;
  }
}

ESIMD_KERNEL_API cgf_t esimd_norm_fp16(
  uint8_t* in, uint8_t* out, uint8_t* normAlpha,
  uint32_t seqLen, uint32_t hiddenDim) {
  int32_t groupH;
  int32_t groupV;
  int32_t groupD;
  int32_t localH;
  int32_t localV;
  int32_t localD;

  localH = 8;
  groupH = hiddenDim / 16;
  groupV = 1;
  localV = 1;
  sycl::range<2> GlobalRangeInplaceNorm(groupH * localH, groupV * localV);
  sycl::range<2> LocalRangeInplaceNorm(localH, localV);
  sycl::nd_range<2> RangeInplaceNorm(GlobalRangeInplaceNorm, LocalRangeInplaceNorm);

  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for(RangeInplaceNorm, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
      inplaceNormFp16(
        in, out, normAlpha,
        seqLen, hiddenDim, ndi);
    });
  };
  return kernel_func;
}
}