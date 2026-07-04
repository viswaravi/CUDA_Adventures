#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "utils.cuh"

#define BLOCK_WIDTH 32

void cpuMatMul(const float *A, const float *B, float *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++)
        sum += A[i * N + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

void cpuMatTranspose(const float *A, float *B, int M, int N)
{
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      B[j * M + i] = A[i * N + j];
}

// Test kernel: fill with 2D global index
__global__ void fill2DIdx(float *A, float *B, float *C, int N)
{
  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int width_x = (gridDim.x * blockDim.x);
  int globalIdx = (row * width_x) + col;
  C[globalIdx] = globalIdx;
}

// Matmul — global memory only
__global__ void matmulKernelNaive(float *A, float *B, float *C, int M, int N, int K)
{
  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  int globalIdx = (row * N) + col;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int k = 0; k < K; k++)
      value += (A[row * N + k] * B[k * N + col]);
    C[globalIdx] = value;
  }
}

// Matmul — tiled shared memory
__global__ void matmulKernelShared(float *A, float *B, float *C, int M, int N, int K)
{
  __shared__ float As[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ float Bs[BLOCK_WIDTH][BLOCK_WIDTH];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;
  float value = 0.0f;

  int num_tiles = (K + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  for (unsigned int tile = 0; tile < (unsigned int)num_tiles; tile++) {
    if (row < M && (tile * BLOCK_WIDTH + threadIdx.x) < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + (tile * BLOCK_WIDTH + threadIdx.x)];
    if ((tile * BLOCK_WIDTH + threadIdx.y) < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_WIDTH + threadIdx.y) * N + col];

    __syncthreads();
    for (int k = 0; k < BLOCK_WIDTH; k++)
      value += (As[threadIdx.y][k] * Bs[k][threadIdx.x]);
    __syncthreads();
  }

  if (row < M && col < N)
    C[(row * N) + col] = value;
}

// Transpose — shared memory (no padding)
__global__ void matTransposeNaive(float *A, float *B, int M, int N)
{
  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;

  if (col < N && row < M)
    tile[threadIdx.y][threadIdx.x] = A[(row * N) + col];
  __syncthreads();

  col = blockIdx.y * blockDim.x + threadIdx.x;
  row = blockIdx.x * blockDim.y + threadIdx.y;
  if (col < M && row < N)
    B[(row * N) + col] = tile[threadIdx.x][threadIdx.y];
}

// Transpose — shared memory with +1 padding (eliminates bank conflicts)
__global__ void matTransposePadded(float *A, float *B, int M, int N)
{
  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH + 1];

  int col = (blockDim.x * blockIdx.x) + threadIdx.x;
  int row = (blockDim.y * blockIdx.y) + threadIdx.y;

  if (col < N && row < M)
    tile[threadIdx.y][threadIdx.x] = A[(row * N) + col];
  __syncthreads();

  col = blockIdx.y * blockDim.x + threadIdx.x;
  row = blockIdx.x * blockDim.y + threadIdx.y;
  if (col < M && row < N)
    B[(row * N) + col] = tile[threadIdx.x][threadIdx.y];
}
