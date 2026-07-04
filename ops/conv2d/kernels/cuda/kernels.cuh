#pragma once

#include "cuda_runtime.h"

__global__ void rgb_to_gray(float *d_in, float *d_out, int width, int height, int channels);
__global__ void convolution_2d_naive(float *d_in, float *d_out, int width, int height,
                                     int channels, float *d_kernel, int kernel_width);
__global__ void convolution_2d_shared_single(float *d_in, float *d_out, int width, int height,
                                             int channels, float *d_kernel, int kernel_width);
__global__ void convolution_2d_shared_coop(float *d_in, float *d_out, int width, int height,
                                           int channels, float *d_kernel, int kernel_width);
__global__ void convolution_2d_shared_struct(float *d_in, float *d_out, int width, int height,
                                             int channels, float *d_kernel, int kernel_width);
