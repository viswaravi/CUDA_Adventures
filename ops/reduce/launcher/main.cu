#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include "kernels.cuh"

#include <assert.h>
#include <stdio.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "cli_args.hpp"

void printArray(float *A, const unsigned long long length)
{
  for (unsigned long long i = 0; i < length; i++) {
    std::cout << std::fixed << std::setprecision(2) << A[i] << " ";
  }
  std::cout << std::endl;
}

void recursiveReduceLauncher(ReductionKernel kernel, float *h_A,
                             CudaMemory<float> &d_A, unsigned long long array_len,
                             bool is_half_blocks = false,
                             bool is_load_multiple = false)
{
  double result_cpu = reduceCPU(h_A, array_len);

  int blockWidth = BLOCK_WIDTH;
  dim3 blockDim(blockWidth);
  dim3 gridDim((array_len + blockWidth - 1) / blockWidth);

  if (is_half_blocks) blockDim.x = blockDim.x / 2;
  if (is_load_multiple && gridDim.x > 1) gridDim.x = gridDim.x / 2;

  CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
  int numBlocks = gridDim.x;
  float *result = (float *)malloc(sizeof(float));

  printKernelConfig(gridDim, blockDim);
  std::cout << "Array Length: " << array_len << std::endl;
  kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_A.get(), d_blockSums.get(), array_len);
  cudaDeviceSynchronize();

  while (numBlocks > 1) {
    gridDim.x = (numBlocks + blockWidth - 1) / blockWidth;
    if (is_load_multiple && gridDim.x > 1) gridDim.x = gridDim.x / 2;

    CudaMemory<float> d_blockSums_out(gridDim.x * sizeof(float));
    printKernelConfig(gridDim, blockDim);
    std::cout << "Array Length: " << numBlocks << std::endl;
    kernel<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_blockSums.get(), d_blockSums_out.get(), numBlocks);
    cudaDeviceSynchronize();
    d_blockSums = std::move(d_blockSums_out);
    numBlocks = gridDim.x;
  }

  CUDA_CALL(cudaMemcpy(result, d_blockSums.get(), sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "Recursive Reduce Result: " << *result << " : " << result_cpu << std::endl;
  assert(std::abs(*result - result_cpu) < 1e-5);
  free(result);
}

void atomicReduceLauncher(ReductionKernel kernel, float *h_A,
                          CudaMemory<float> &d_A, unsigned long long array_len)
{
  float *result = (float *)malloc(sizeof(float));
  *result = 0.0f;

  dim3 blockDim(BLOCK_WIDTH);
  dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);

  CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
  int numBlocks = gridDim.x;

  kernel<<<gridDim, blockDim>>>(d_A.get(), d_blockSums.get(), array_len);
  cudaDeviceSynchronize();

  while (numBlocks > 1) {
    gridDim.x = (numBlocks + blockDim.x - 1) / blockDim.x;
    CudaMemory<float> d_blockSums_out(gridDim.x * sizeof(float));
    kernel<<<gridDim, blockDim>>>(d_blockSums.get(), d_blockSums_out.get(), numBlocks);
    cudaDeviceSynchronize();
    d_blockSums = std::move(d_blockSums_out);
    numBlocks = gridDim.x;
  }

  CUDA_CALL(cudaMemcpy(result, d_blockSums.get(), sizeof(float), cudaMemcpyDeviceToHost));
  double result_cpu = reduceCPU(h_A, array_len);
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
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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
          "Sequential addressing with idle threads",
          reduction_args,
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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
          "Sequential addressing with multiple elements per thread and warp shuffle reduction",
          reduction_args,
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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
          [](const RunConfig &c) {
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

  try {
    RunConfig cfg = parse_args(argc, argv, variants);
    if (cfg.print_help) {
      print_usage(argv[0], variants);
      return EXIT_SUCCESS;
    }
    if (cfg.list_variants) {
      print_variants(variants);
      return EXIT_SUCCESS;
    }
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);
    if (cfg.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], variants);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
