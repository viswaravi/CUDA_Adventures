// Explicit CUDA vector_add kernels used by the runtime runner and PTX/SASS
// export targets. Keep these non-templated so generated symbols are stable and
// easy to inspect while learning CUDA kernel development.

#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add_i32_scalar_kernel(const int *a, const int *b, int *c,
                                      std::uint64_t n) {
  std::uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void add_i32_vec4_kernel(const int *a, const int *b, int *c,
                                    std::uint64_t n) {
  std::uint64_t base = ((blockIdx.x * blockDim.x) + threadIdx.x) * 4;
  if (base + 4 <= n) {
    int4 av = reinterpret_cast<const int4 *>(a)[base / 4];
    int4 bv = reinterpret_cast<const int4 *>(b)[base / 4];
    int4 cv{av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w};
    reinterpret_cast<int4 *>(c)[base / 4] = cv;
    return;
  }
  for (int i = 0; i < 4 && base + i < n; ++i) {
    c[base + i] = a[base + i] + b[base + i];
  }
}

__global__ void add_f32_scalar_kernel(const float *a, const float *b, float *c,
                                      std::uint64_t n) {
  std::uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void add_f32_vec4_kernel(const float *a, const float *b, float *c,
                                    std::uint64_t n) {
  std::uint64_t base = ((blockIdx.x * blockDim.x) + threadIdx.x) * 4;
  if (base + 4 <= n) {
    float4 av = reinterpret_cast<const float4 *>(a)[base / 4];
    float4 bv = reinterpret_cast<const float4 *>(b)[base / 4];
    float4 cv{av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w};
    reinterpret_cast<float4 *>(c)[base / 4] = cv;
    return;
  }
  for (int i = 0; i < 4 && base + i < n; ++i) {
    c[base + i] = a[base + i] + b[base + i];
  }
}

__global__ void add_f16_scalar_kernel(const half *a, const half *b, half *c,
                                      std::uint64_t n) {
  std::uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

__global__ void add_f16_half2_kernel(const half *a, const half *b, half *c,
                                     std::uint64_t n) {
  std::uint64_t base = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
  if (base + 2 <= n) {
    half2 av = reinterpret_cast<const half2 *>(a)[base / 2];
    half2 bv = reinterpret_cast<const half2 *>(b)[base / 2];
    reinterpret_cast<half2 *>(c)[base / 2] = __hadd2(av, bv);
    return;
  }
  if (base < n) {
    c[base] = __hadd(a[base], b[base]);
  }
}

__global__ void add_bf16_scalar_kernel(const __nv_bfloat16 *a,
                                       const __nv_bfloat16 *b,
                                       __nv_bfloat16 *c, std::uint64_t n) {
  std::uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = __float2bfloat16(__bfloat162float(a[idx]) +
                              __bfloat162float(b[idx]));
  }
}

__global__ void add_bf16_pair_kernel(const __nv_bfloat16 *a,
                                     const __nv_bfloat16 *b, __nv_bfloat16 *c,
                                     std::uint64_t n) {
  std::uint64_t base = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
  if (base < n) {
    c[base] = __float2bfloat16(__bfloat162float(a[base]) +
                               __bfloat162float(b[base]));
  }
  if (base + 1 < n) {
    c[base + 1] = __float2bfloat16(__bfloat162float(a[base + 1]) +
                                   __bfloat162float(b[base + 1]));
  }
}
