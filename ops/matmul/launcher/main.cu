#include <assert.h>
#include <device_functions.h>
#include <stdio.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cli_args.hpp"
#include "utils.cuh"
#include "kernels.cuh"

void printMemoryRequirements(unsigned long M, unsigned long N, unsigned long K)
{
  const int mat_elt_count = (M * K) + (K * N) + (M * N);
  long double array_size_bytes = mat_elt_count * sizeof(float);
  long double array_size_gbytes = array_size_bytes / (1024 * 1024 * 1024);
  long double array_size_mbytes = array_size_bytes / (1024 * 1024);
  std::cout << "---Memory Requirements---" << std::endl;
  std::cout << "Matrix Dimensions=>  A:" << M * K << " B:" << K * N << " C:" << M * N << std::endl;
  std::cout << "Total Memory Size: " << array_size_gbytes << "GB, " << array_size_mbytes << "MB" << std::endl;
}

bool verifyResults(const float *cpu_C, const float *gpu_C, int N, float tolerance = 1e-5)
{
  for (int i = 0; i < N * N; i++) {
    if (fabs(cpu_C[i] - gpu_C[i]) > tolerance) {
      std::cout << "Mismatch at index " << i << " | CPU: " << cpu_C[i] << " vs GPU: " << gpu_C[i] << "\n";
      return false;
    }
  }
  std::cout << "GPU results match CPU results within tolerance " << tolerance << "\n";
  return true;
}

void printMatrix(const float *matrix, int M, int N)
{
  std::cout << "Matrix (" << M << "x" << N << "):\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) std::cout << matrix[i * N + j] << "\t";
    std::cout << "\n";
  }
  std::cout << "-----------------------------------\n";
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> matmul_args = {
      {"--m", "int", "2048", "Rows for matrix A and C"},
      {"--n", "int", "2048", "Cols for matrix B and C"},
      {"--k", "int", "2048", "Cols for matrix A / rows for matrix B"},
  };

  const VariantRegistry variants = {
      {
          "matmul-naive",
          "Naive global-memory matrix multiplication",
          matmul_args,
          [](const RunConfig &c) {
            const int M = get_int(c, "--m", 2048), N = get_int(c, "--n", 2048), K = get_int(c, "--k", 2048);
            if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("--m, --n and --k must all be > 0");
            int mem_size_A = M * K * sizeof(float), mem_size_B = K * N * sizeof(float), mem_size_C = M * N * sizeof(float);
            printMemoryRequirements(M, N, K);
            float *h_A = (float *)malloc(mem_size_A), *h_B = (float *)malloc(mem_size_B);
            float *h_C = (float *)malloc(mem_size_C), *h_C_cpu = (float *)malloc(mem_size_C);
            std::fill(h_A, h_A + M * K, 2.0f); std::fill(h_B, h_B + K * N, 2.0f);
            std::fill(h_C, h_C + M * N, 0.0f); std::fill(h_C_cpu, h_C_cpu + M * N, 0.0f);
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
            matmulKernelNaive<<<dimGrid, dimBlock>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError()); CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));
            free(h_A); free(h_B); free(h_C); free(h_C_cpu);
          },
      },
      {
          "matmul-shared",
          "Shared-memory tiled matrix multiplication",
          matmul_args,
          [](const RunConfig &c) {
            const int M = get_int(c, "--m", 2048), N = get_int(c, "--n", 2048), K = get_int(c, "--k", 2048);
            if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("--m, --n and --k must all be > 0");
            int mem_size_A = M * K * sizeof(float), mem_size_B = K * N * sizeof(float), mem_size_C = M * N * sizeof(float);
            printMemoryRequirements(M, N, K);
            float *h_A = (float *)malloc(mem_size_A), *h_B = (float *)malloc(mem_size_B);
            float *h_C = (float *)malloc(mem_size_C), *h_C_cpu = (float *)malloc(mem_size_C);
            std::fill(h_A, h_A + M * K, 2.0f); std::fill(h_B, h_B + K * N, 2.0f);
            std::fill(h_C, h_C + M * N, 0.0f); std::fill(h_C_cpu, h_C_cpu + M * N, 0.0f);
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
            matmulKernelShared<<<dimGrid, dimBlock>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError()); CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));
            free(h_A); free(h_B); free(h_C); free(h_C_cpu);
          },
      },
      {
          "transpose-naive",
          "Shared-memory transpose without padding",
          matmul_args,
          [](const RunConfig &c) {
            const int M = get_int(c, "--m", 2048), N = get_int(c, "--n", 2048), K = get_int(c, "--k", 2048);
            int mem_size_A = M * K * sizeof(float), mem_size_C = M * N * sizeof(float);
            printMemoryRequirements(M, N, K);
            float *h_A = (float *)malloc(mem_size_A), *h_C = (float *)malloc(mem_size_C);
            std::fill(h_A, h_A + M * K, 2.0f); std::fill(h_C, h_C + M * N, 0.0f);
            CudaMemory<float> d_A(mem_size_A), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
            matTransposeNaive<<<dimGrid, dimBlock>>>(d_A.get(), d_C.get(), M, K);
            CUDA_CALL(cudaGetLastError()); CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));
            free(h_A); free(h_C);
          },
      },
      {
          "transpose-padded",
          "Shared-memory transpose with bank-conflict padding",
          matmul_args,
          [](const RunConfig &c) {
            const int M = get_int(c, "--m", 2048), N = get_int(c, "--n", 2048), K = get_int(c, "--k", 2048);
            int mem_size_A = M * K * sizeof(float), mem_size_C = M * N * sizeof(float);
            printMemoryRequirements(M, N, K);
            float *h_A = (float *)malloc(mem_size_A), *h_C = (float *)malloc(mem_size_C);
            std::fill(h_A, h_A + M * K, 2.0f); std::fill(h_C, h_C + M * N, 0.0f);
            CudaMemory<float> d_A(mem_size_A), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
            matTransposePadded<<<dimGrid, dimBlock>>>(d_A.get(), d_C.get(), M, K);
            CUDA_CALL(cudaGetLastError()); CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));
            free(h_A); free(h_C);
          },
      },
  };

  try {
    RunConfig cfg = parse_args(argc, argv, variants);
    if (cfg.print_help) { print_usage(argv[0], variants); return EXIT_SUCCESS; }
    if (cfg.list_variants) { print_variants(variants); return EXIT_SUCCESS; }
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);
    if (cfg.reset_device) CUDA_CALL(cudaDeviceReset());
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], variants);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
