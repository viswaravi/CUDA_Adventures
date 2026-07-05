#pragma once

#include "cuda_runtime.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define MAX_BLOCK_DIM 1024

// int32 kernels
__global__ void add_i32_scalar_kernel(const int *a, const int *b, int *c,
                                      std::uint64_t n);
__global__ void add_i32_vec4_kernel(const int *a, const int *b, int *c,
                                    std::uint64_t n);

// float32 kernels
__global__ void add_f32_scalar_kernel(const float *a, const float *b, float *c,
                                      std::uint64_t n);
__global__ void add_f32_vec4_kernel(const float *a, const float *b, float *c,
                                    std::uint64_t n);

// float16 kernels
__global__ void add_f16_scalar_kernel(const half *a, const half *b, half *c,
                                      std::uint64_t n);
__global__ void add_f16_half2_kernel(const half *a, const half *b, half *c,
                                     std::uint64_t n);

// bfloat16 kernels
__global__ void add_bf16_scalar_kernel(const __nv_bfloat16 *a,
                                       const __nv_bfloat16 *b,
                                       __nv_bfloat16 *c, std::uint64_t n);
__global__ void add_bf16_pair_kernel(const __nv_bfloat16 *a,
                                     const __nv_bfloat16 *b, __nv_bfloat16 *c,
                                     std::uint64_t n);
