#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include <stdio.h>

#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

using ReductionKernel = void (*)(float *, float *, unsigned long long);

double reduceCPU(float *A, const unsigned long long length)
{
  double sum = 0.0f;
  for (unsigned long long i = 0; i < length; i++) {
    sum += A[i];
  }
  return sum;
}

// 1. Interleaved Addressing with Divergent Branches
__global__ void reduce1(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (tid % (2 * stride) == 0 && (tid + stride < blockDim.x)) {
      partialSum[tid] += partialSum[tid + stride];
    }
  }
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// 2. Interleaved Addressing with Bank Conflicts
__global__ void reduce2(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    int index = 2 * stride * tid;
    if (index + stride < blockDim.x) {
      partialSum[index] += partialSum[index + stride];
    }
  }
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// 3. Sequential Addressing with Idle Threads
__global__ void reduce3(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride && (tid + stride < blockDim.x)) {
      partialSum[tid] += partialSum[tid + stride];
    }
  }
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// 4. Sequential Addressing, Add during load
__global__ void reduce4(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride && (tid + stride < blockDim.x)) {
      partialSum[tid] += partialSum[tid + stride];
    }
  }
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// 5. Sequential Addressing, Add during load, Unrolling last loop
__global__ void reduce5(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;

  for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
    __syncthreads();
    if (tid < stride && (tid + stride < blockDim.x)) {
      partialSum[tid] += partialSum[tid + stride];
    }
  }
  __syncthreads();
  if (tid < 32) {
    partialSum[tid] += partialSum[tid + 32];
    partialSum[tid] += partialSum[tid + 16];
    partialSum[tid] += partialSum[tid + 8];
    partialSum[tid] += partialSum[tid + 4];
    partialSum[tid] += partialSum[tid + 2];
    partialSum[tid] += partialSum[tid + 1];
  }
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// Warp Reduction function, unrolled last 6 steps for 1024 block size
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid)
{
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

__device__ float warpReduceShuffle(float value)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

// 6. Sequential Addressing, Add during load, Full loop unrolling
__global__ void reduce6(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;
  __syncthreads();

  unsigned int blockSize = blockDim.x;
  if (blockSize >= 512) {
    if (tid < 256) partialSum[tid] += partialSum[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) partialSum[tid] += partialSum[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) partialSum[tid] += partialSum[tid + 64];
    __syncthreads();
  }
  if (tid < 32) warpReduce<BLOCK_WIDTH / 2>(partialSum, tid);
  __syncthreads();
  if (tid == 0) blockSums[blockIdx.x] = partialSum[0];
}

// 7. Sequential Addressing, Multiple elements per thread, Warp Shuffle Reduction
__global__ void reduce7(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];
  int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int lane = tid % warpSize;
  unsigned int warpId = tid / warpSize;
  unsigned int numWarps = (blockSize + warpSize - 1) / warpSize;
  unsigned long long gridSize = static_cast<unsigned long long>(blockSize) * 2ULL * gridDim.x;
  unsigned long long idx = (blockIdx.x * blockSize * 2) + threadIdx.x;
  float sum = 0.0f;

  while (idx < length) {
    sum += A[idx];
    if (idx + blockSize < length) sum += A[idx + blockSize];
    idx += gridSize;
  }

  sum = warpReduceShuffle(sum);
  if (lane == 0) partialSum[warpId] = sum;
  __syncthreads();

  if (warpId == 0) {
    sum = (lane < numWarps) ? partialSum[lane] : 0.0f;
    sum = warpReduceShuffle(sum);
    if (lane == 0) blockSums[blockIdx.x] = sum;
  }
}

// Test kernel to exercise warp-level primitives
__global__ void warpPrimitives(float *A, float *result,
                               const unsigned long long length)
{
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  if (tid < 32) {
    int sum = A[idx];
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (__any_sync(0xFFFFFFFF, sum == 32)) {
      printf("At least one thread has a value 32!\n");
    }
    if (tid == 0) *result = sum;
  }
}

// 8. Strided Reduction with Atomic Result
__global__ void reduceAtomic(float *A, float *result, const unsigned long long length)
{
  __shared__ float partialSum[BLOCK_WIDTH];
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (tid % (2 * stride) == 0 && (tid + stride < blockDim.x)) {
      partialSum[tid] += partialSum[tid + stride];
    }
  }
  if (tid == 0) atomicAdd(result, partialSum[0]);
}
