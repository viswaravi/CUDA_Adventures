#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include "kernels.cuh"

#include <assert.h>
#include <stdio.h>
#include <algorithm>
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

void runReductionKernel(const RunConfig &cfg)
{
  const std::string kernel_name = get_string(cfg, "--kernel", "interleaved-divergent");
  unsigned long long array_len = get_ull(cfg, "--n", 8192ULL);
  size_t mem_size = array_len * sizeof(float);
  float *h_A = (float *)malloc(mem_size);
  std::fill(h_A, h_A + array_len, 1.0f);
  CudaMemory<float> d_A(mem_size);
  CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

  if (kernel_name == "interleaved-divergent") {
    recursiveReduceLauncher(reduce1, h_A, d_A, array_len);
  } else if (kernel_name == "interleaved-bank-conflicts") {
    recursiveReduceLauncher(reduce2, h_A, d_A, array_len);
  } else if (kernel_name == "sequential-idle") {
    recursiveReduceLauncher(reduce3, h_A, d_A, array_len);
  } else if (kernel_name == "sequential-add-load") {
    recursiveReduceLauncher(reduce4, h_A, d_A, array_len, true);
  } else if (kernel_name == "sequential-last-unroll") {
    recursiveReduceLauncher(reduce5, h_A, d_A, array_len, true);
  } else if (kernel_name == "sequential-full-unroll") {
    recursiveReduceLauncher(reduce6, h_A, d_A, array_len, true);
  } else if (kernel_name == "sequential-multiple-load-warpshuffle") {
    recursiveReduceLauncher(reduce7, h_A, d_A, array_len, true, true);
  } else if (kernel_name == "atomic") {
    atomicReduceLauncher(reduceAtomic, h_A, d_A, array_len);
  } else if (kernel_name == "warp-primitives") {
    CudaMemory<float> d_result(sizeof(float));
    warpPrimitives<<<1, 32>>>(d_A.get(), d_result.get(), array_len);
    cudaDeviceSynchronize();
    float h_result;
    CUDA_CALL(cudaMemcpy(&h_result, d_result.get(), sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Warp Primitives Reduce Result: " << h_result << std::endl;
    assert(std::abs(h_result - 32.0f) < 1e-5);
  } else {
    free(h_A);
    throw std::invalid_argument("Unsupported --kernel for reduce: " + kernel_name);
  }

  free(h_A);
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> reduction_args = {
      {"--op", "string", "reduce", "Operation name"},
      {"--kernel", "string", "interleaved-divergent", "Reduction kernel name"},
      {"--n", "uint64", "8192", "Array length (elements)"},
  };

  try {
    RunConfig cfg = parse_args(argc, argv, reduction_args);
    if (cfg.print_help) {
      print_usage(argv[0], reduction_args, "CUDA reduce backend runner.");
      return EXIT_SUCCESS;
    }
    if (get_string(cfg, "--op", "reduce") != "reduce")
      throw std::invalid_argument("--op must be reduce");
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    runReductionKernel(cfg);
    if (cfg.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], reduction_args, "CUDA reduce backend runner.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
