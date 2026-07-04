#pragma once
// Legacy CUDA vector_add kernels kept next to the CUDA backend runner.
//
// The normalized runner uses equivalent int32 kernels internally; these
// declarations remain available for analysis/PTX export targets.

#include "cuda_runtime.h"

#define MAX_BLOCK_DIM 1024

// Elementwise addition — one element per thread
__global__ void addKernel(int *c, const int *a, const int *b,
                          const unsigned long long length);

// Vectorised addition — four int elements per thread (int4 loads/stores)
__global__ void addKernelVectorized(int *c, const int *a, const int *b,
                                    const unsigned long long length);
