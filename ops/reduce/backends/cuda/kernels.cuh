#pragma once

#include <cuda_runtime.h>

#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

using ReductionKernel = void (*)(float *, float *, unsigned long long);

// CPU reference implementation
double reduceCPU(float *A, const unsigned long long length);

// Kernel variants (reduce1..7: recursive block-sum pattern;
//                  reduceAtomic: single-pass atomic;
//                  warpPrimitives: warp-level demo)
__global__ void reduce1(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce2(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce3(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce4(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce5(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce6(float *A, float *blockSums, const unsigned long long length);
__global__ void reduce7(float *A, float *blockSums, const unsigned long long length);
__global__ void warpPrimitives(float *A, float *result, const unsigned long long length);
__global__ void reduceAtomic(float *A, float *result, const unsigned long long length);
