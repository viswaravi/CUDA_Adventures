#include <cuda_runtime.h>
#include <kernel_lab/utils/cuda_utils.cuh>

// ─── Device helpers ──────────────────────────────────────────────────────────

__device__ int d_clamp(int x, int a, int b)
{
  return fmaxf(a, fminf(b, x));
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

__global__ void rgb_to_gray(float *d_in, float *d_out, int width, int height, int channels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = (y * width + x) * channels;
  if (x < width && y < height) {
    float r = d_in[idx], g = d_in[idx + 1], b = d_in[idx + 2];
    d_out[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
  }
}

// 2D Convolution — global memory only
__global__ void convolution_2d_naive(float *d_in, float *d_out, int width,
                                     int height, int channels, float *d_kernel,
                                     int kernel_width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int pad = kernel_width / 2;

  if (x < width && y < height) {
    for (int c = 0; c < channels; c++) {
      float sum = 0.0f;
      for (int ky = -pad; ky <= pad; ky++) {
        for (int kx = -pad; kx <= pad; kx++) {
          int in_y = d_clamp(y + ky, 0, height - 1);
          int in_x = d_clamp(x + kx, 0, width - 1);
          sum += d_in[(in_y * width + in_x) * channels + c]
               * d_kernel[(ky + pad) * kernel_width + (kx + pad)];
        }
      }
      d_out[(y * width + x) * channels + c] = sum;
    }
  }
}

// 2D Convolution — shared memory, single-threaded tile load
__global__ void convolution_2d_shared_single(float *d_in, float *d_out,
                                             int width, int height,
                                             int channels, float *d_kernel,
                                             int kernel_width)
{
  extern __shared__ float tile[];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int pad = kernel_width / 2;
  int tw = blockDim.x;
  int shared_width = tw + (2 * pad);

  if (tx == 0 && ty == 0) {
    for (int c = 0; c < channels; c++) {
      for (int i = -pad; i < pad + tw; ++i) {
        for (int j = -pad; j < pad + tw; ++j) {
          int gy = d_clamp(y + i, 0, height - 1);
          int gx = d_clamp(x + j, 0, width - 1);
          tile[((i + pad) * shared_width + (j + pad)) * channels + c] = d_in[(gy * width + gx) * channels + c];
        }
      }
    }
  }
  __syncthreads();

  for (int c = 0; c < channels; c++) {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++) {
      for (int kx = -pad; kx <= pad; kx++) {
        int in_y = d_clamp((ty + pad) + ky, 0, shared_width - 1);
        int in_x = d_clamp((tx + pad) + kx, 0, shared_width - 1);
        sum += tile[(in_y * shared_width + in_x) * channels + c]
             * d_kernel[(ky + pad) * kernel_width + (kx + pad)];
      }
    }
    if (x < width && y < height)
      d_out[(y * width + x) * channels + c] = sum;
  }
}

// 2D Convolution — shared memory, cooperative tile load
__global__ void convolution_2d_shared_coop(float *d_in, float *d_out, int width,
                                           int height, int channels,
                                           float *d_kernel, int kernel_width)
{
  extern __shared__ float tile[];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int pad = kernel_width / 2;
  int tw = blockDim.x;
  int shared_width = tw + (2 * pad);
  int num_threads = tw * tw;
  int total_elems = shared_width * shared_width;

  for (int i = ty * tw + tx; i < total_elems; i += num_threads) {
    int tY = i / shared_width, tX = i % shared_width;
    int gX = d_clamp((int)(blockIdx.x * blockDim.x) + (tX - pad), 0, width - 1);
    int gY = d_clamp((int)(blockIdx.y * blockDim.y) + (tY - pad), 0, height - 1);
    for (int c = 0; c < channels; c++)
      tile[(tY * shared_width + tX) * channels + c] = d_in[(gY * width + gX) * channels + c];
  }
  __syncthreads();

  for (int c = 0; c < channels; c++) {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++) {
      for (int kx = -pad; kx <= pad; kx++) {
        int in_y = d_clamp((ty + pad) + ky, 0, shared_width - 1);
        int in_x = d_clamp((tx + pad) + kx, 0, shared_width - 1);
        sum += tile[(in_y * shared_width + in_x) * channels + c]
             * d_kernel[(ky + pad) * kernel_width + (kx + pad)];
      }
    }
    if (x < width && y < height)
      d_out[(y * width + x) * channels + c] = sum;
  }
}

// 2D Convolution — shared memory, structured halo load
__global__ void convolution_2d_shared_struct(float *d_in, float *d_out,
                                             int width, int height,
                                             int channels, float *d_kernel,
                                             int kernel_width)
{
  extern __shared__ float tile[];
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
  int x = d_clamp(bx + tx, 0, width - 1);
  int y = d_clamp(by + ty, 0, height - 1);
  int pad = kernel_width / 2;
  int tw = blockDim.x;
  int shared_width = tw + (2 * pad);

  for (int c = 0; c < channels; c++) {
    tile[((ty + pad) * shared_width + (tx + pad)) * channels + c] = d_in[(y * width + x) * channels + c];

    if (ty < pad) {
      int y_top = d_clamp(by - pad + ty, 0, height - 1);
      int y_bot = d_clamp(by + tw + ty, 0, height - 1);
      if (x < width) {
        tile[(ty * shared_width + (tx + pad)) * channels + c] = d_in[(y_top * width + x) * channels + c];
        tile[((ty + tw + pad) * shared_width + (tx + pad)) * channels + c] = d_in[(y_bot * width + x) * channels + c];
      }
    }
    if (tx < pad) {
      int x_left = d_clamp(bx - pad + tx, 0, width - 1);
      int x_right = d_clamp(bx + tw + tx, 0, width - 1);
      if (y < height) {
        tile[((ty + pad) * shared_width + tx) * channels + c] = d_in[(y * width + x_left) * channels + c];
        tile[((ty + pad) * shared_width + (pad + tw + tx)) * channels + c] = d_in[(y * width + x_right) * channels + c];
      }
    }
    if (tx < pad && ty < pad) {
      int y_tl = d_clamp(by - pad + ty, 0, height - 1), x_tl = d_clamp(bx - pad + tx, 0, width - 1);
      int y_tr = y_tl,                                   x_tr = d_clamp(bx + tw + tx, 0, width - 1);
      int y_bl = d_clamp(by + tw + ty, 0, height - 1),  x_bl = x_tl;
      int y_br = y_bl,                                   x_br = x_tr;
      tile[(ty * shared_width + tx) * channels + c] = d_in[(y_tl * width + x_tl) * channels + c];
      tile[(ty * shared_width + (pad + tw + tx)) * channels + c] = d_in[(y_tr * width + x_tr) * channels + c];
      tile[((ty + pad + tw) * shared_width + tx) * channels + c] = d_in[(y_bl * width + x_bl) * channels + c];
      tile[((ty + pad + tw) * shared_width + (pad + tw + tx)) * channels + c] = d_in[(y_br * width + x_br) * channels + c];
    }
  }
  __syncthreads();

  for (int c = 0; c < channels; c++) {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++) {
      for (int kx = -pad; kx <= pad; kx++) {
        int in_y = (ty + pad) + ky, in_x = (tx + pad) + kx;
        sum += tile[(in_y * shared_width + in_x) * channels + c]
             * d_kernel[(ky + pad) * kernel_width + (kx + pad)];
      }
    }
    if (x < width && y < height)
      d_out[(y * width + x) * channels + c] = sum;
  }
}
