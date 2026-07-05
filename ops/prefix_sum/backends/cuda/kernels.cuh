#pragma once

#include <cuda_runtime.h>

#define BLOCK_WIDTH 1024

// CPU reference implementations
float reduceCPU(float *A, const unsigned long long length);
void inclusiveScanCPU(float *A, float *result, const unsigned long long length);

// Kernel variants
__global__ void inclusiveScan_naive(float *A, float *result, const unsigned long long length);
__global__ void inclusiveScan_fast(float *A, float *result, const unsigned long long length);
__global__ void inclusiveScan_fast2(float *A, float *result, const unsigned long long length);
__global__ void inclusiveScan_double(float *A, float *result, const unsigned long long length);
__global__ void inclusiveScan_block_reduce(float *A, float *block_sums, const unsigned long long length);
__global__ void add_block_reduce(float *A, const unsigned long long length, float *d_block_sums);
