#pragma once

#include "cuda_runtime.h"

#define BLOCK_WIDTH 32

// CPU reference implementations
void cpuMatMul(const float *A, const float *B, float *C, int M, int N, int K);
void cpuMatTranspose(const float *A, float *B, int M, int N);

// Kernel variants
__global__ void fill2DIdx(float *A, float *B, float *C, int N);
__global__ void matmulKernelNaive(float *A, float *B, float *C, int M, int N, int K);
__global__ void matmulKernelShared(float *A, float *B, float *C, int M, int N, int K);
__global__ void matTransposeNaive(float *A, float *B, int M, int N);
__global__ void matTransposePadded(float *A, float *B, int M, int N);
