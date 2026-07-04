#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

#define MAX_BLOCK_DIM 1024

__global__ void addKernel(int *c, const int *a, const int *b,
                          const unsigned long long length)
{
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < length) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void addKernelVectorized(int *c, const int *a, const int *b,
                                    const unsigned long long length)
{
  unsigned long long idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (idx < length) {
    int4 a_vec = *((int4 *)&a[idx]);
    int4 b_vec = *((int4 *)&b[idx]);
    int4 c_vec;

    c_vec.x = a_vec.x + b_vec.x;
    c_vec.y = a_vec.y + b_vec.y;
    c_vec.z = a_vec.z + b_vec.z;
    c_vec.w = a_vec.w + b_vec.w;

    *((int4 *)&c[idx]) = c_vec;
  }
}
