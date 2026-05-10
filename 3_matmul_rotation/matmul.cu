
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

#define BLOCK_WIDTH 32

// CPU Matmul
void cpuMatMul(const float *A, const float *B, float *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0.0f;
      for (int k = 0; k < K; k++)
      {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void cpuMatTranspose(const float *A, float *B, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B[j * M + i] = A[i * N + j];
    }
  }
}

bool verifyResults(const float *cpu_C, const float *gpu_C, int N,
                   float tolerance = 1e-5)
{
  for (int i = 0; i < N * N; i++)
  {
    if (fabs(cpu_C[i] - gpu_C[i]) > tolerance)
    {
      std::cout << "Mismatch at index " << i << " | CPU: " << cpu_C[i]
                << " vs GPU: " << gpu_C[i] << "\n";
      return false;
    }
  }
  std::cout << "GPU results match CPU results within tolerance " << tolerance
            << "\n";
  return true;
}

// Function to print a matrix
void printMatrix(const float *matrix, int M, int N)
{
  std::cout << "Matrix (" << M << "x" << N << "):\n";
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      std::cout << matrix[i * N + j] << "\t";
    }
    std::cout << "\n";
  }
  std::cout << "-----------------------------------\n";
}

// Test kernel to fill and check 2D indexing
__global__ void fill2DIdx(float *A, float *B, float *C, int N)
{
  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int width_x = (gridDim.x * blockDim.x);
  int globalIdx = (row * width_x) + col;

  C[globalIdx] = globalIdx;
}

// Matmul - Global memory Access
__global__ void matmulKernelNaive(float *A, float *B, float *C, int M, int N,
                                  int K)
{
  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  // global idx for Matrix Dimension, not thread dimension
  int globalIdx = (row * N) + col;

  if (row < M && col < N)
  {
    float value = 0.0f;
    for (int k = 0; k < K; k++)
    {
      value += (A[row * N + k] * B[k * N + col]);
    }
    C[globalIdx] = value;
  }
}

// Matmul - Shared memory Access
__global__ void matmulKernelShared(float *A, float *B, float *C, int M, int N,
                                   int K)
{
  // Shared Memory for tiles
  __shared__ float As[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ float Bs[BLOCK_WIDTH][BLOCK_WIDTH];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int globalIdx =
      (row * N) + col; // global idx for Matrix Dimension, not thread dimension
  float value = 0.0f;

  // loop through tiles
  int num_tiles = (K + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  for (unsigned int tile = 0; tile < num_tiles; tile++)
  {
    // Load into Shared Memory
    if (row < M && (tile * BLOCK_WIDTH + threadIdx.x) < K)
    {
      // tiling in x_direction
      //                          row  * width       + col
      int tile_idx_A = row * K + (tile * BLOCK_WIDTH + threadIdx.x);
      As[threadIdx.y][threadIdx.x] = A[tile_idx_A];
    }
    if ((tile * BLOCK_WIDTH + threadIdx.y) < K && col < N)
    {
      // tiling in y_direction
      //                row  * width       + col
      int tile_idx_B = (tile * BLOCK_WIDTH + threadIdx.y) * N + col;
      Bs[threadIdx.y][threadIdx.x] = B[tile_idx_B];
    }

    __syncthreads();

    // Compute from shared memory
    for (int k = 0; k < BLOCK_WIDTH; k++)
    {
      value += (As[threadIdx.y][k] * Bs[k][threadIdx.x]);
    }

    __syncthreads();
  }

  C[globalIdx] = value;
}

// Transpose - Shared Memory
__global__ void matTransposeNaive(float *A, float *B, int M, int N)
{
  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int globalIdx =
      (row * N) + col; // global idx for Matrix Dimension, not thread dimension

  // Load data into shared memory
  if (col < N && row < M)
  {
    tile[threadIdx.y][threadIdx.x] = A[globalIdx];
  }

  __syncthreads();

  // Write data back to global memory
  col = blockIdx.y * blockDim.x + threadIdx.x;
  row = blockIdx.x * blockDim.y + threadIdx.y;
  globalIdx = (row * N) + col;

  if (col < M && row < N)
  {
    B[globalIdx] = tile[threadIdx.x][threadIdx.y];
  }
}

// Transpose - Shared Memory Padded
__global__ void matTransposePadded(float *A, float *B, int M, int N)
{
  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH + 1];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int globalIdx =
      (row * N) + col; // global idx for Matrix Dimension, not thread dimension

  // Load data into shared memory
  if (col < N && row < M)
  {
    tile[threadIdx.y][threadIdx.x] = A[globalIdx];
  }

  __syncthreads();

  // Write data back to global memory
  col = blockIdx.y * blockDim.x + threadIdx.x;
  row = blockIdx.x * blockDim.y + threadIdx.y;
  globalIdx = (row * N) + col;

  if (col < M && row < N)
  {
    B[globalIdx] = tile[threadIdx.x][threadIdx.y];
  }
}

void printMemoryRequirements(unsigned long M, unsigned long N,
                             unsigned long K)
{
  const int mat_elt_count = (M * K) + (K * N) + (M * N); // square matrix A+B+C
  long double array_size_bytes = mat_elt_count * sizeof(float);
  long double array_size_gbytes = array_size_bytes / (1024 * 1024 * 1024);
  long double array_size_mbytes = array_size_bytes / (1024 * 1024);

  std::cout << "---Memory Requirements---" << std::endl;
  std::cout << "Matrix Dimensions=>  A:" << M * K << " B:" << K * N
            << " C:" << M * N << std::endl;
  std::cout << "Total Memory Size: " << array_size_gbytes << "GB, "
            << array_size_mbytes << "MB" << std::endl;
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
          [](const RunConfig &c)
          {
            // Parse matrix dimensions from CLI args
            const int M = get_int(c, "--m", 2048);
            const int N = get_int(c, "--n", 2048);
            const int K = get_int(c, "--k", 2048);

            if (M <= 0 || N <= 0 || K <= 0)
            {
              throw std::invalid_argument("--m, --n and --k must all be > 0");
            }

            int elts_A = M * K;
            int elts_B = K * N;
            int elts_C = M * N;
            int mem_size_A = elts_A * sizeof(float);
            int mem_size_B = elts_B * sizeof(float);
            int mem_size_C = elts_C * sizeof(float);

            // Debug memory summary for this run
            printMemoryRequirements(M, N, K);

            // Allocate host memory
            float *h_A, *h_B, *h_C, *h_C_cpu;
            h_A = (float *)malloc(mem_size_A);
            h_B = (float *)malloc(mem_size_B);
            h_C = (float *)malloc(mem_size_C);
            h_C_cpu = (float *)malloc(mem_size_C);

            // Initialize host buffers
            std::fill(h_A, h_A + elts_A, 2.0f);
            std::fill(h_B, h_B + elts_B, 2.0f);
            std::fill(h_C, h_C + elts_C, 0.0f);
            std::fill(h_C_cpu, h_C_cpu + elts_C, 0.0f);

            // Allocate/copy device buffers
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

            // Kernel launch configuration
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                         (M + dimBlock.y - 1) / dimBlock.y);

            // Launch variant kernel
            matmulKernelNaive<<<dimGrid, dimBlock>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

            // Cleanup host allocations
            free(h_A);
            free(h_B);
            free(h_C);
            free(h_C_cpu);
          },
      },
      {
          "matmul-shared",
          "Shared-memory tiled matrix multiplication",
          matmul_args,
          [](const RunConfig &c)
          {
            // Parse matrix dimensions from CLI args
            const int M = get_int(c, "--m", 2048);
            const int N = get_int(c, "--n", 2048);
            const int K = get_int(c, "--k", 2048);

            if (M <= 0 || N <= 0 || K <= 0)
            {
              throw std::invalid_argument("--m, --n and --k must all be > 0");
            }

            int elts_A = M * K;
            int elts_B = K * N;
            int elts_C = M * N;
            int mem_size_A = elts_A * sizeof(float);
            int mem_size_B = elts_B * sizeof(float);
            int mem_size_C = elts_C * sizeof(float);

            // Debug memory summary for this run
            printMemoryRequirements(M, N, K);

            // Allocate host memory
            float *h_A, *h_B, *h_C, *h_C_cpu;
            h_A = (float *)malloc(mem_size_A);
            h_B = (float *)malloc(mem_size_B);
            h_C = (float *)malloc(mem_size_C);
            h_C_cpu = (float *)malloc(mem_size_C);

            // Initialize host buffers
            std::fill(h_A, h_A + elts_A, 2.0f);
            std::fill(h_B, h_B + elts_B, 2.0f);
            std::fill(h_C, h_C + elts_C, 0.0f);
            std::fill(h_C_cpu, h_C_cpu + elts_C, 0.0f);

            // Allocate/copy device buffers
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

            // Kernel launch configuration
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                         (M + dimBlock.y - 1) / dimBlock.y);

            // Launch variant kernel
            matmulKernelShared<<<dimGrid, dimBlock>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

            // Cleanup host allocations
            free(h_A);
            free(h_B);
            free(h_C);
            free(h_C_cpu);
          },
      },
      {
          "transpose-naive",
          "Shared-memory transpose without padding",
          matmul_args,
          [](const RunConfig &c)
          {
            // Parse matrix dimensions from CLI args
            const int M = get_int(c, "--m", 2048);
            const int N = get_int(c, "--n", 2048);
            const int K = get_int(c, "--k", 2048);

            if (M <= 0 || N <= 0 || K <= 0)
            {
              throw std::invalid_argument("--m, --n and --k must all be > 0");
            }

            int elts_A = M * K;
            int elts_B = K * N;
            int elts_C = M * N;
            int mem_size_A = elts_A * sizeof(float);
            int mem_size_B = elts_B * sizeof(float);
            int mem_size_C = elts_C * sizeof(float);

            // Debug memory summary for this run
            printMemoryRequirements(M, N, K);

            // Allocate host memory
            float *h_A, *h_B, *h_C, *h_C_cpu;
            h_A = (float *)malloc(mem_size_A);
            h_B = (float *)malloc(mem_size_B);
            h_C = (float *)malloc(mem_size_C);
            h_C_cpu = (float *)malloc(mem_size_C);

            // Initialize host buffers
            std::fill(h_A, h_A + elts_A, 2.0f);
            std::fill(h_B, h_B + elts_B, 2.0f);
            std::fill(h_C, h_C + elts_C, 0.0f);
            std::fill(h_C_cpu, h_C_cpu + elts_C, 0.0f);

            // Allocate/copy device buffers
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

            // Kernel launch configuration
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                         (M + dimBlock.y - 1) / dimBlock.y);

            // Launch variant kernel
            matTransposeNaive<<<dimGrid, dimBlock>>>(d_A.get(), d_C.get(), M, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

            // Cleanup host allocations
            free(h_A);
            free(h_B);
            free(h_C);
            free(h_C_cpu);
          },
      },
      {
          "transpose-padded",
          "Shared-memory transpose with bank-conflict padding",
          matmul_args,
          [](const RunConfig &c)
          {
            // Parse matrix dimensions from CLI args
            const int M = get_int(c, "--m", 2048);
            const int N = get_int(c, "--n", 2048);
            const int K = get_int(c, "--k", 2048);

            if (M <= 0 || N <= 0 || K <= 0)
            {
              throw std::invalid_argument("--m, --n and --k must all be > 0");
            }

            int elts_A = M * K;
            int elts_B = K * N;
            int elts_C = M * N;
            int mem_size_A = elts_A * sizeof(float);
            int mem_size_B = elts_B * sizeof(float);
            int mem_size_C = elts_C * sizeof(float);

            // Debug memory summary for this run
            printMemoryRequirements(M, N, K);

            // Allocate host memory
            float *h_A, *h_B, *h_C, *h_C_cpu;
            h_A = (float *)malloc(mem_size_A);
            h_B = (float *)malloc(mem_size_B);
            h_C = (float *)malloc(mem_size_C);
            h_C_cpu = (float *)malloc(mem_size_C);

            // Initialize host buffers
            std::fill(h_A, h_A + elts_A, 2.0f);
            std::fill(h_B, h_B + elts_B, 2.0f);
            std::fill(h_C, h_C + elts_C, 0.0f);
            std::fill(h_C_cpu, h_C_cpu + elts_C, 0.0f);

            // Allocate/copy device buffers
            CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);
            CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

            // Kernel launch configuration
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                         (M + dimBlock.y - 1) / dimBlock.y);

            // Launch variant kernel
            matTransposePadded<<<dimGrid, dimBlock>>>(d_A.get(), d_C.get(), M, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

            // Cleanup host allocations
            free(h_A);
            free(h_B);
            free(h_C);
            free(h_C_cpu);
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

    // Set device and run selected variant
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);

    // Reset device after execution (unless explicitly disabled)
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
