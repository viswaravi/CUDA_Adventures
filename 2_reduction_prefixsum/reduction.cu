
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

#include <assert.h>
#include <stdio.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

using ReductionKernel = void (*)(float *, float *, unsigned long long);

// cpu reduction
double reduceCPU(float *A, const unsigned long long length)
{
  double sum = 0.0f;
  for (unsigned long long i = 0; i < length; i++)
  {
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

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction in Strides
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();

    // Choose active threads in strides 2,4,8,16,...
    // Divergent threads
    if (tid % (2 * stride) == 0 && (tid + stride < blockDim.x))
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 2. Interleaved Addressing with Bank Conflicts
__global__ void reduce2(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];

  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction in Strides
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();

    // All threads are active with strided indexing (no divergent branches)
    int index = 2 * stride * tid;
    if (index + stride < blockDim.x)
    {
      partialSum[index] += partialSum[index + stride];
    }
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 3. Sequential Addressing with Idle Threads
__global__ void reduce3(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];

  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction in Strides - Stop at 32 for final warp reduction later
  // Only half of threads are active
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    __syncthreads();

    if (tid < stride && (tid + stride < blockDim.x))
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 4. Sequential Addressing, Add during load
__global__ void reduce4(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];

  // Double Block Dimension
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;

  // Each thread loads sum of two elements from global memory
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;

  // Block Reduction in Strides - Stop at 32 for final warp reduction later
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    __syncthreads();

    if (tid < stride && (tid + stride < blockDim.x))
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 5. Sequential Addressig, Loop unrolling last loop
__global__ void reduce5(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];

  // Double Block Dimension
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;

  // Each thread loads sum of two elements from global memory
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;

  // Block Reduction in Strides - Stop at 32 for final warp reduction later
  for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
  {
    __syncthreads();

    if (tid < stride && (tid + stride < blockDim.x))
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  __syncthreads();

  if (tid < 32)
  {
    // Unroll last loop
    partialSum[tid] += partialSum[tid + 32];
    partialSum[tid] += partialSum[tid + 16];
    partialSum[tid] += partialSum[tid + 8];
    partialSum[tid] += partialSum[tid + 4];
    partialSum[tid] += partialSum[tid + 2];
    partialSum[tid] += partialSum[tid + 1];
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 6. Sequential Addressig, Full loop unrolling
// Templated Warp Reduction function
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid)
{
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

// Warp Reduce implemenation using warp shuffle instructions
__device__ void warpReduceShuffle(float *sdata, int tid)
{
  if (tid < 32)
  {
    int sum = sdata[tid];

    for (int offset = 32; offset > 0; offset /= 2)
    {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    sdata[tid] = sum;
  }
}

__global__ void reduce6(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];

  // Double Block Dimension
  unsigned long long idx = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
  int tid = threadIdx.x;

  // Each thread loads sum of two elements from global memory
  partialSum[tid] = idx < length ? A[idx] + A[idx + blockDim.x] : 0.0f;

  __syncthreads();

  // Fully Unrolled Reduce
  unsigned int blockSize = blockDim.x;

  if (blockSize >= 512)
  {
    // For stride = 256
    if (tid < 256)
    {
      partialSum[tid] += partialSum[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256)
  {
    // For stride = 128
    if (tid < 128)
    {
      partialSum[tid] += partialSum[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128)
  {
    // For stride = 64
    if (tid < 64)
    {
      partialSum[tid] += partialSum[tid + 64];
    }
    __syncthreads();
  }

  // For stride = 32
  if (tid < 32)
  {
    warpReduce<BLOCK_WIDTH / 2>(partialSum, tid);
  }

  __syncthreads();

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 7. Sequential Addressig, load Multiple elements per thread
__global__ void reduce7(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];
  int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  unsigned long long idx = (blockIdx.x * blockSize * 2) + threadIdx.x;

  // Each thread loads sum of n elements from global memory
  while (idx < length)
  {
    partialSum[tid] = A[idx] + A[idx + blockSize];
    idx += gridSize;
  }

  __syncthreads();

  // Fully Unrolled Reduce
  if (blockSize >= 512)
  {
    // For stride = 256
    if (tid < 256)
    {
      partialSum[tid] += partialSum[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256)
  {
    // For stride = 128
    if (tid < 128)
    {
      partialSum[tid] += partialSum[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128)
  {
    // For stride = 64
    if (tid < 64)
    {
      partialSum[tid] += partialSum[tid + 64];
    }
    __syncthreads();
  }

  // For stride = 32
  if (tid < 32)
  {
    warpReduce<BLOCK_WIDTH / 2>(partialSum, tid);
  }

  __syncthreads();

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// Test kernel to play with warp functions
__global__ void warpPrimitives(float *A, float *result,
                               const unsigned long long length)
{
  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  if (tid < 32)
  {
    int sum = A[idx];

    // Reduce Sum Value
    for (int offset = 16; offset > 0; offset /= 2)
    {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Verify value using Voting
    if (__any_sync(0xFFFFFFFF, sum == 32))
    {
      printf("At least one thread has a value 32!\n");
    }

    // Add final reduce value directly to global memory
    if (tid == 0)
    {
      *result = sum;
    }
  }
}

// 8. Strided Reduction with Atomic Result
__global__ void reduceAtomic(float *A, float *result, const unsigned long long length)
{
  __shared__ float partialSum[BLOCK_WIDTH];

  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction in Strides
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();

    // Choose active threads in strides 2,4,8,16,...
    // Divergent threads
    if (tid % (2 * stride) == 0 && (tid + stride < blockDim.x))
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  // Add block reduce value directly to global memory
  // No block memory storage
  if (tid == 0)
  {
    atomicAdd(result, partialSum[0]);
  }
}

void printArray(float *A, const unsigned long long length)
{
  for (unsigned long long i = 0; i < length; i++)
  {
    std::cout << std::fixed << std::setprecision(2) << A[i] << " ";
  }
  std::cout << std::endl;
}

void recursiveReduceLauncher(ReductionKernel kernel, float *h_A, CudaMemory<float> &d_A, unsigned long long array_len, bool is_half_blocks = false)
{
  // Kernel Config
  int blockWidth = BLOCK_WIDTH;
  dim3 blockDim(blockWidth);
  dim3 gridDim((array_len + blockWidth - 1) / blockWidth);

  if (is_half_blocks)
  {
    // Reduce block dimension for kernels which loads two elements per thread
    blockDim.x = blockDim.x / 2;
  }

  // Block Memory for Recursive Reduce
  CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
  int numBlocks = gridDim.x;
  float *result;
  result = (float *)malloc(sizeof(float));

  // First Reduce
  printKernelConfig(gridDim, blockDim);
  kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_A.get(), d_blockSums.get(), array_len);
  cudaDeviceSynchronize();
  // Recursive Reduce
  while (numBlocks > 1)
  {
    // Compute new grid dimension
    gridDim.x = (numBlocks + blockWidth - 1) / blockWidth;
    // New Block Memory - out
    CudaMemory<float> d_blockSums_out(gridDim.x * sizeof(float));

    printKernelConfig(gridDim, blockDim);
    kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_blockSums.get(), d_blockSums_out.get(), numBlocks);
    cudaDeviceSynchronize();

    // Update next
    d_blockSums = std::move(d_blockSums_out);
    numBlocks = gridDim.x;
  }
  CUDA_CALL(cudaMemcpy(result, d_blockSums.get(), sizeof(float), cudaMemcpyDeviceToHost));

  double result_cpu = reduceCPU(h_A, array_len);

  std::cout << "Recursive Reduce Result: " << *result << " : " << result_cpu << std::endl;
  assert(std::abs(*result - result_cpu) < 1e-5);
  free(result);
}

void atomicReduceLauncher(ReductionKernel kernel, float *h_A, CudaMemory<float> &d_A, unsigned long long array_len)
{
  // Global Atomic result
  float *result;
  result = (float *)malloc(sizeof(float));
  *result = 0.0f;

  // Kernel Config
  dim3 blockDim(BLOCK_WIDTH);
  dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);

  // Block Memory for Recursive Reduce
  CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
  int numBlocks = gridDim.x;

  // First Reduce
  kernel<<<gridDim, blockDim>>>(d_A.get(), d_blockSums.get(), array_len);
  cudaDeviceSynchronize();
  // Recursive Reduce
  while (numBlocks > 1)
  {
    // Compute new grid dimension
    gridDim.x = (numBlocks + blockDim.x - 1) / blockDim.x;
    // New Block Memory - out
    CudaMemory<float> d_blockSums_out(gridDim.x * sizeof(float));

    kernel<<<gridDim, blockDim>>>(d_blockSums.get(), d_blockSums_out.get(), numBlocks);
    cudaDeviceSynchronize();

    // Update next
    d_blockSums = std::move(d_blockSums_out);
    numBlocks = gridDim.x;
  }
  CUDA_CALL(cudaMemcpy(result, d_blockSums.get(), sizeof(float), cudaMemcpyDeviceToHost));

  double result_cpu = reduceCPU(h_A, array_len);
  // std::cout << "Recursive Reduce Result: " << *result << " : " << result_cpu << std::endl;
  assert(std::abs(*result - result_cpu) < 1e-5);
  free(result);
}

int main()
{
  // Choose GPU
  CUDA_CALL(cudaSetDevice(0));

  try
  {
    enum Options
    {
      REDUCTION_Interleaved_Divergent,
      REDUCTION_Interleaved_BankConflicts,
      REDUCTION_Sequential_Idle,
      REDUCTION_Sequential_AddLoad,
      REDUCTION_Sequential_LastUnroll,
      REDUCTION_Sequential_FullUnroll,
      REDUCTION_Sequential_MultipleElts,

      REDUCTION_Atomic,
      WarpPrimitives,
    };

    Options option = REDUCTION_Sequential_MultipleElts;

    unsigned long long array_len = 1024 * 1024 * 512;
    size_t mem_size = array_len * sizeof(float);

    // Host Data Reduction
    float *h_A;
    h_A = (float *)malloc(mem_size);

    // Initialize Host Data
    std::fill(h_A, h_A + array_len, 1);

    // Device Data
    CudaMemory<float> d_A(mem_size), d_result(sizeof(float)), d_result_scan(mem_size);

    // Copy to Device
    CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

    // kernel config for double procesing
    // int block_size = array_len / 2; // process using half the threads
    // int shared_size = array_len * sizeof(float);
    // dim3 blockDimD(block_size);
    // dim3 gridDimD((block_size + block_size - 1) / block_size);

    switch (option)
    {
    case REDUCTION_Interleaved_Divergent:
      recursiveReduceLauncher(reduce1, h_A, d_A, array_len);
      break;

    case REDUCTION_Interleaved_BankConflicts:
      recursiveReduceLauncher(reduce2, h_A, d_A, array_len);
      break;

    case REDUCTION_Sequential_Idle:
      recursiveReduceLauncher(reduce3, h_A, d_A, array_len);
      break;

    case REDUCTION_Sequential_AddLoad:
      recursiveReduceLauncher(reduce4, h_A, d_A, array_len, true);
      break;

    case REDUCTION_Sequential_LastUnroll:
      recursiveReduceLauncher(reduce5, h_A, d_A, array_len, true);
      break;

    case REDUCTION_Sequential_FullUnroll:
      recursiveReduceLauncher(reduce6, h_A, d_A, array_len, true);
      break;

    case REDUCTION_Sequential_MultipleElts:
      recursiveReduceLauncher(reduce7, h_A, d_A, array_len, true);
      break;

    case REDUCTION_Atomic:
      atomicReduceLauncher(reduceAtomic, h_A, d_A, array_len);
      break;

      // Test kernel
      // case WarpPrimitives:
      // warpPrimitives<<<gridDim, blockDim>>>(d_A.get(), d_result.get(), array_len);
      // CUDA_CALL(cudaMemcpy(result, d_result.get(), sizeof(float), cudaMemcpyDeviceToHost));
      // std::cout << "Atomic Reduce Result: " << *result << std::endl;
      // break;

    default:
      break;
    }

    // Free memory
    free(h_A);
    // cudaDeviceReset - for profiling
    CUDA_CALL(cudaDeviceReset());
  }
  catch (std::exception &e)
  {
    fprintf(stderr, "Exception: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
