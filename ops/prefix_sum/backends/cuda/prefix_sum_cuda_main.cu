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

void verifyInclusiveScan(float *A, float *B, const unsigned long long length)
{
  for (unsigned long long i = 0; i < length; i++) {
    if (A[i] - B[i] != 0.0) {
      std::cout << "Inclusive Scan Verification Failed at Index: " << i
                << " A:" << A[i] << " B:" << B[i] << std::endl;
      return;
    }
  }
  std::cout << "Inclusive Scan Verification Passed!" << std::endl;
}

void runPrefixSumKernel(const RunConfig &c)
{
  const std::string kernel_name = get_string(c, "--kernel", "naive");
  unsigned long long array_len = get_ull(c, "--n", 1024ULL);
  size_t mem_size = array_len * sizeof(float);
  float *h_A = (float *)malloc(mem_size);
  float *result_scan = (float *)malloc(mem_size);
  float *result_scan_cpu = (float *)malloc(mem_size);
  std::fill(h_A, h_A + array_len, 1.0f);
  CudaMemory<float> d_A(mem_size), d_result_scan(mem_size);
  CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

  if (kernel_name == "naive") {
    assert(array_len <= BLOCK_WIDTH);
    dim3 blockDim(BLOCK_WIDTH);
    dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
    inclusiveScan_naive<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(),
                                               array_len);
  } else if (kernel_name == "fast") {
    assert(array_len <= BLOCK_WIDTH);
    dim3 blockDim(BLOCK_WIDTH);
    dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
    inclusiveScan_fast<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(),
                                              array_len);
  } else if (kernel_name == "fast2") {
    assert(array_len <= BLOCK_WIDTH);
    dim3 blockDim(BLOCK_WIDTH);
    dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
    inclusiveScan_fast2<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(),
                                               array_len);
  } else if (kernel_name == "double") {
    assert(array_len <= BLOCK_WIDTH);
    int block_size = static_cast<int>(array_len / 2);
    assert(block_size > 0);
    int shared_size = static_cast<int>(array_len * sizeof(float));
    dim3 blockDim(block_size);
    dim3 gridDim((block_size + block_size - 1) / block_size);
    inclusiveScan_double<<<gridDim, blockDim, shared_size>>>(
        d_A.get(), d_result_scan.get(), array_len);
  } else if (kernel_name == "hier") {
    dim3 blockDim(BLOCK_WIDTH);
    dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
    int numBlocks = static_cast<int>(gridDim.x);
    assert(numBlocks <= BLOCK_WIDTH);
    size_t block_sum_size = numBlocks * sizeof(float);
    CudaMemory<float> d_block_sums(block_sum_size);
    inclusiveScan_block_reduce<<<gridDim, blockDim, BLOCK_WIDTH * sizeof(float)>>>(
        d_A.get(), d_block_sums.get(), array_len);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    dim3 blockDimH(BLOCK_WIDTH);
    dim3 gridDimH((numBlocks + blockDimH.x - 1) / blockDimH.x);
    inclusiveScan_fast2<<<gridDimH, blockDimH>>>(d_block_sums.get(),
                                                 d_block_sums.get(), numBlocks);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    add_block_reduce<<<gridDim, blockDim>>>(d_A.get(), array_len,
                                            d_block_sums.get());
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(result_scan, d_A.get(), mem_size,
                         cudaMemcpyDeviceToHost));
    inclusiveScanCPU(h_A, result_scan_cpu, array_len);
    verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
    free(h_A);
    free(result_scan);
    free(result_scan_cpu);
    return;
  } else {
    free(h_A);
    free(result_scan);
    free(result_scan_cpu);
    throw std::invalid_argument("Unsupported --kernel for prefix_sum: " +
                                kernel_name);
  }

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size,
                       cudaMemcpyDeviceToHost));
  inclusiveScanCPU(h_A, result_scan_cpu, array_len);
  verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
  free(h_A);
  free(result_scan);
  free(result_scan_cpu);
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> scan_args = {
      {"--op", "string", "prefix_sum", "Operation name"},
      {"--kernel", "string", "naive", "Prefix-sum kernel name"},
      {"--n", "uint64", "1024", "Array length (elements)"},
  };

  try {
    RunConfig cfg = parse_args(argc, argv, scan_args);
    if (cfg.print_help) {
      print_usage(argv[0], scan_args, "CUDA prefix_sum backend runner.");
      return EXIT_SUCCESS;
    }
    if (get_string(cfg, "--op", "prefix_sum") != "prefix_sum")
      throw std::invalid_argument("--op must be prefix_sum");
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    runPrefixSumKernel(cfg);
    if (cfg.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], scan_args, "CUDA prefix_sum backend runner.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
