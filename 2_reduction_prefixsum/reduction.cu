
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
#include <vector>
#include "cli_args.hpp"
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

    // Choose active threads in strides 1,2,4,8,16,...
    // stall: divergent threads within a warp - some threads are idle while others are active at each cycle
    if (tid % (2 * stride) == 0 && (tid + stride < blockDim.x))
    {
      // stall: shared memory bank conflict, not a big issue, already serialized by divergent threads
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
// Focus on providing uniform load to threads within a warp to reduce divergence
__global__ void reduce2(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];

  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction in Strides
  // strides 1,2,4,8,... lead to all threads being active
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    
    // interleaved addressing pattern but as stride increases only 
    // lower warps in the block do the work so reduced warp divergence
    // stall: higher indexed warps are idle
    // metric: smsp__average_warps_active_per_inst_executed.ratio
    int index = 2 * stride * tid;
    if (index + stride < blockDim.x)
    {
      // stall: Bank Conflicts - threads access shared memory in a pattern that causes multiple 
      // threads to access the same memory bank, Increases warp stall
      // metric: derived__memory_l1_conflicts_shared_way
      partialSum[index] += partialSum[index + stride];
    }
  }

  // Add block reduce value to global memory
  if (tid == 0)
  {
    blockSums[blockIdx.x] = partialSum[0];
  }
}

// 3. Sequential Addressing with Idle Threads (uneven work distribution - Idle threads & warps)
// Achieved uniform workload within a warp but uneven workload within the block.
__global__ void reduce3(float *A, float *blockSums, const unsigned long long length)
{
  extern __shared__ float partialSum[];

  unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;

  // Load data into shared memory
  partialSum[tid] = idx < length ? A[idx] : 0.0f;

  // Block Reduction - only half of threads are active in a block even at start and decreases by half 
  // every stride leads to many idle threads, especially in later strides when only a few threads are active
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    __syncthreads();

    // stall: even fewer warps than reduce2 are active at each stride, as only threads with tid < stride are active
    // all those tail warps which cannot execute any instruction also need to participate in synchronization 
    // leading to increased barrier wait in __syncthreads() until block reaches end
    // metric: smsp__thread_inst_executed_per_inst_executed.ratio
    // reduced instruction issue efficiency in warp
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
// Reducing the block size to compensate for uneven work distribution for tail warps which happened due to sequential addressing
__global__ void reduce4(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];

  // Double Block Dimension - Each thread loads two elements from global memory and adds them during load, 
  // reducing the number of iterations in the reduction loop by half.
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

// 5. Sequential Addressing, Add during load, Unrolling last loop
__global__ void reduce5(float *A, float *blockSums, const unsigned long long length)
{
  // shared memory is halved, as each thread loads two elements
  extern __shared__ float partialSum[];

  // Double Block Dimension - Each thread loads two elements from global memory and adds them during load,
  // reducing the number of iterations in the reduction loop by half. 
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


// Warp Reduction function, unrolled last 6 steps for 1024 block size
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

// Warp Reduce implementation using warp shuffle instructions
__device__ float warpReduceShuffle(float value)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
  {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }

  return value;
}

// 6. Sequential Addressing, Add during load, Full loop unrolling
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

// 7. Sequential Addressing, Add during load, load Multiple elements per thread, with Warp Shuffle Reduction
__global__ void reduce7(float *A, float *blockSums, const unsigned long long length)
{
  // Shared memory stores one partial sum per warp.
  extern __shared__ float partialSum[];
  int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int lane = tid % warpSize;
  unsigned int warpId = tid / warpSize;
  unsigned int numWarps = (blockSize + warpSize - 1) / warpSize;
  unsigned long long gridSize = static_cast<unsigned long long>(blockSize) * 2ULL * gridDim.x;
  unsigned long long idx = (blockIdx.x * blockSize * 2) + threadIdx.x;
  float sum = 0.0f;

  // Each thread accumulates multiple elements from global memory.
  while (idx < length)
  {
    sum += A[idx];

    if (idx + blockSize < length)
    {
      sum += A[idx + blockSize];
    }

    idx += gridSize;
  }

  // Reduce inside each warp using shuffle instructions.
  sum = warpReduceShuffle(sum);

  if (lane == 0)
  {
    partialSum[warpId] = sum;
  }

  __syncthreads();

  // First warp reduces the warp-level partial sums.
  if (warpId == 0)
  {
    sum = (lane < numWarps) ? partialSum[lane] : 0.0f;
    sum = warpReduceShuffle(sum);

    if (lane == 0)
    {
      blockSums[blockIdx.x] = sum;
    }
  }
}

// Test kernel to play with warp functions
__global__ void  warpPrimitives(float *A, float *result,
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

void recursiveReduceLauncher(ReductionKernel kernel, float *h_A, CudaMemory<float> &d_A, unsigned long long array_len,
                             bool is_half_blocks = false, bool is_load_multiple = false)
{
  double result_cpu = reduceCPU(h_A, array_len);

  // Kernel Config
  int blockWidth = BLOCK_WIDTH;
  dim3 blockDim(blockWidth);
  dim3 gridDim((array_len + blockWidth - 1) / blockWidth);

  if (is_half_blocks)
  {
    // Reduce block dimension for kernels which loads two elements per thread
    blockDim.x = blockDim.x / 2;
  }

  if (is_load_multiple && gridDim.x > 1)
  {
    gridDim.x = gridDim.x / 2;
  }

  // Block Memory for Recursive Reduce
  CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
  int numBlocks = gridDim.x;
  float *result;
  result = (float *)malloc(sizeof(float));

  // First Reduce
  printKernelConfig(gridDim, blockDim);
  std::cout << "Array Length: " << array_len << std::endl;
  kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_A.get(), d_blockSums.get(), array_len);
  cudaDeviceSynchronize();
  // Recursive Reduce
  while (numBlocks > 1)
  {
    // Compute new grid dimension
    gridDim.x = (numBlocks + blockWidth - 1) / blockWidth;

    if (is_load_multiple && gridDim.x > 1)
    {
      gridDim.x = gridDim.x / 2;
    }

    // New Block Memory - out
    CudaMemory<float> d_blockSums_out(gridDim.x * sizeof(float));

    printKernelConfig(gridDim, blockDim);
    std::cout << "Array Length: " << numBlocks << std::endl;
    kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_blockSums.get(), d_blockSums_out.get(), numBlocks);
    cudaDeviceSynchronize();

    // Update next
    d_blockSums = std::move(d_blockSums_out);
    numBlocks = gridDim.x;
  }
  CUDA_CALL(cudaMemcpy(result, d_blockSums.get(), sizeof(float), cudaMemcpyDeviceToHost));

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

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> reduction_args = {
      {"--n", "uint64", "8192", "Array length (elements)"},
  };

  const VariantRegistry variants = {
      {
          "interleaved-divergent",
          "Interleaved addressing with divergent branches",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce1, h_A, d_A, array_len);
            free(h_A);
          },
      },
      {
          "interleaved-bank-conflicts",
          "Interleaved addressing without divergent branches with shared memory bank conflicts",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce2, h_A, d_A, array_len);
            free(h_A);
          },
      },
      {
          "sequential-idle",
          "Sequential addressing with idle threads reduced shared memory bank conflicts",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce3, h_A, d_A, array_len);
            free(h_A);
          },
      },
      {
          "sequential-add-load",
          "Sequential addressing with add-during-load",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce4, h_A, d_A, array_len, true);
            free(h_A);
          },
      },
      {
          "sequential-last-unroll",
          "Sequential addressing with last-warp loop unrolling",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce5, h_A, d_A, array_len, true);
            free(h_A);
          },
      },
      {
          "sequential-full-unroll",
          "Sequential addressing with full unrolling",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce6, h_A, d_A, array_len, true);
            free(h_A);
          },
      },
       {
          "sequential-multiple-load-warpshuffle",
          "Sequential addressing with add-during-load and multiple elements per thread along with warp shuffle reduction",
          reduction_args,
          [](const RunConfig &c)
          {
            std::cout << "Running Sequential Addressing with add-during-load and multiple elements per thread along with warp shuffle reduction" << std::endl;
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            recursiveReduceLauncher(reduce7, h_A, d_A, array_len, true, true);
            free(h_A);
          },
      },
      {
          "atomic",
          "Strided reduction using global atomic accumulation",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            atomicReduceLauncher(reduceAtomic, h_A, d_A, array_len);
            free(h_A);
          },
      },
      {
          "warp-primitives",
          "Reduction using warp-level primitives",
          reduction_args,
          [](const RunConfig &c)
          {
            unsigned long long array_len = get_ull(c, "--n", 8192ULL);
            size_t mem_size = array_len * sizeof(float);
            float *h_A = (float *)malloc(mem_size);
            std::fill(h_A, h_A + array_len, 1.0f);
            CudaMemory<float> d_A(mem_size);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));
            CudaMemory<float> d_result(sizeof(float));
            warpPrimitives<<<1, 32>>>(d_A.get(), d_result.get(), array_len);
            cudaDeviceSynchronize();
            float h_result;
            CUDA_CALL(cudaMemcpy(&h_result, d_result.get(), sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "Warp Primitives Reduce Result: " << h_result << std::endl;
            assert(std::abs(h_result - 32.0f) < 1e-5);
            free(h_A);
          },
      },
  };

  try
  {
    RunConfig cfg = parse_args(argc, argv, variants);
    if (cfg.print_help)
    {
      print_usage(argv[0], variants);
      return EXIT_SUCCESS;
    }
    if (cfg.list_variants)
    {
      print_variants(variants);
      return EXIT_SUCCESS;
    }

    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);

    if (cfg.reset_device)
    {
      CUDA_CALL(cudaDeviceReset());
    }
  }
  catch (const std::exception &e)
  {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], variants);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


