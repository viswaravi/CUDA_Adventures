#pragma once

#include "experiment_types.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// CUDA dtype conversion helpers shared by typed backend launchers.
//
// Host-side initialization and validation should use these instead of direct
// casts so half and bfloat16 follow the CUDA-supported conversion paths.

template <typename T> __host__ __device__ T cudaValueFromDouble(double value) {
  return static_cast<T>(value);
}

template <> __host__ __device__ half cudaValueFromDouble<half>(double value) {
  return __float2half(static_cast<float>(value));
}

template <>
__host__ __device__ __nv_bfloat16
cudaValueFromDouble<__nv_bfloat16>(double value) {
  return __float2bfloat16(static_cast<float>(value));
}

template <typename T> double cudaValueToDouble(T value) {
  return static_cast<double>(value);
}

template <> double cudaValueToDouble<half>(half value) {
  return static_cast<double>(__half2float(value));
}

template <> double cudaValueToDouble<__nv_bfloat16>(__nv_bfloat16 value) {
  return static_cast<double>(__bfloat162float(value));
}

inline double readDTypeValueAsDouble(const void *data, std::uint64_t idx,
                                     DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return static_cast<double>(static_cast<const int *>(data)[idx]);
  case DType::Float32:
    return static_cast<double>(static_cast<const float *>(data)[idx]);
  case DType::Float16:
    return static_cast<double>(
        __half2float(static_cast<const half *>(data)[idx]));
  case DType::BFloat16:
    return static_cast<double>(
        __bfloat162float(static_cast<const __nv_bfloat16 *>(data)[idx]));
  }
  throw std::runtime_error("Unsupported dtype");
}
