#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <kernel_lab/utils/cuda_utils.cuh>

#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

float reduceCPU(float *A, const unsigned long long length)
{
  float sum = 0.0f;
  for (unsigned long long i = 0; i < length; i++) sum += A[i];
  return sum;
}

void inclusiveScanCPU(float *A, float *result, const unsigned long long length)
{
  float sum = 0.0f;
  for (unsigned long long i = 0; i < length; i++) {
    sum += A[i];
    result[i] = sum;
  }
}

__global__ void inclusiveScan_naive(float *A, float *result, const unsigned long long length)
{
  __shared__ float partial[BLOCK_WIDTH];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partial[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride <= (unsigned int)tid; stride *= 2) {
    __syncthreads();
    if (tid - stride >= 0) partial[tid] += partial[tid - stride];
  }
  if (idx < length) result[idx] = partial[tid];
}

__global__ void inclusiveScan_fast(float *A, float *result, const unsigned long long length)
{
  __shared__ float partial[BLOCK_WIDTH];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partial[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if ((tid + 1) % (2 * stride) == 0) partial[tid] += partial[tid - stride];
  }
  for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if ((index + stride) < (int)blockDim.x) partial[index + stride] += partial[index];
  }
  __syncthreads();
  if (idx < length) result[idx] = partial[tid];
}

__global__ void inclusiveScan_fast2(float *A, float *result, const unsigned long long length)
{
  __shared__ float partial[BLOCK_WIDTH];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partial[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if (index < (int)blockDim.x) partial[index] += partial[index - stride];
  }
  for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if ((index + stride) < (int)blockDim.x) partial[index + stride] += partial[index];
  }
  if (idx < length) result[idx] = partial[tid];
}

__global__ void inclusiveScan_double(float *A, float *result, const unsigned long long length)
{
  extern __shared__ float partial[];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  int bx = blockDim.x;
  int sharedDim = 2 * blockDim.x;

  partial[tid]      = idx < length       ? A[idx]      : 0.0f;
  partial[tid + bx] = idx + bx < length  ? A[idx + bx] : 0.0f;

  for (unsigned int stride = 1; stride <= (unsigned int)bx; stride *= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if (index < sharedDim) partial[index] += partial[index - stride];
  }
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if ((index + stride) < sharedDim) partial[index + stride] += partial[index];
  }
  __syncthreads();
  if (idx < length)      result[idx]      = partial[tid];
  if (idx + bx < length) result[idx + bx] = partial[tid + bx];
}

__global__ void inclusiveScan_block_reduce(float *A, float *block_sums, const unsigned long long length)
{
  extern __shared__ float partial[];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  int bx = blockDim.x;
  partial[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < (unsigned int)bx; stride *= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if (index < bx) partial[index] += partial[index - stride];
  }
  for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = (tid + 1) * (2 * stride) - 1;
    if ((index + stride) < (int)blockDim.x) partial[index + stride] += partial[index];
  }
  __syncthreads();
  if (idx < length) A[idx] = partial[tid];
  block_sums[blockIdx.x] = partial[bx - 1];
}

__global__ void add_block_reduce(float *A, const unsigned long long length, float *d_block_sums)
{
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (blockIdx.x > 0 && idx < length) A[idx] += d_block_sums[blockIdx.x - 1];
}
