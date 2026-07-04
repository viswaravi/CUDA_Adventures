#pragma once

#include "cuda_runtime.h"

#define PI 3.14159265358979323846
#define BLOCK_WIDTH 32
#define MAX_SHARED_WIDTH 50

__global__ void rgb_to_gray(float *d_in, float *d_out, int width, int height, int channels);
__global__ void rotation_naive(float *d_in, float *d_out, int width, int height, int channels, int angle);
__global__ void tex_interpolation(cudaTextureObject_t texObj, float *output, int width, int height, int channels);
__global__ void rotation_bbox_coop(float *d_in, float *d_out, int width, int height, int channels, int angle);
__global__ void rotation_bbox_strided(float *d_in, float *d_out, int width, int height, int channels, int angle);
