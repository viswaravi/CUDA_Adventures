#include <cuda_runtime.h>
#include <kernel_lab/utils/cuda_utils.cuh>
#include <cmath>

#define PI 3.14159265358979323846
#define BLOCK_WIDTH 32
#define MAX_SHARED_WIDTH 50

// ─── Device helpers ──────────────────────────────────────────────────────────

__device__ int d_clamp(int x, int a, int b)
{
  return fmaxf(a, fminf(b, x));
}

__device__ float d_bilinearInterpolate(float *d_in, int width, int height,
                                       int channels, int c, float x, float y)
{
  int x1 = (int)floor(x), y1 = (int)floor(y);
  int x2 = min(x1 + 1, width - 1);
  int y2 = min(y1 + 1, height - 1);
  if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height) return 0;

  float a = x - x1, b = y - y1;
  float p1 = d_in[(y1 * width + x1) * channels + c];
  float p2 = d_in[(y1 * width + x2) * channels + c];
  float p3 = d_in[(y2 * width + x1) * channels + c];
  float p4 = d_in[(y2 * width + x2) * channels + c];
  return (1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4;
}

__device__ float2 computeSrcCoord(int x, int y, float angle, int width, int height)
{
  float radians = angle * PI / 180.0;
  float cosA = cos(radians), sinA = sin(radians);
  int cx = width / 2, cy = height / 2;
  float srcX = ((x - cx) * cosA + (y - cy) * -sinA) + cx;
  float srcY = ((x - cx) * sinA + (y - cy) * cosA) + cy;
  return make_float2(srcX, srcY);
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

__global__ void rotation_naive(float *d_in, float *d_out, int width, int height,
                               int channels, int angle)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float2 srcCoord = computeSrcCoord(x, y, angle, width, height);
    for (unsigned int c = 0; c < (unsigned int)channels; c++)
      d_out[(y * width + x) * channels + c] =
          d_bilinearInterpolate(d_in, width, height, channels, c, srcCoord.x, srcCoord.y);
  }
}

__global__ void tex_interpolation(cudaTextureObject_t texObj, float *output,
                                  int width, int height, int channels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float u = x / (float)width, v = y / (float)height;
    float4 value = tex2D<float4>(texObj, u, v);
    int idx = (y * width + x) * channels;
    output[idx + 0] = value.x;
    output[idx + 1] = value.y;
    output[idx + 2] = value.z;
  }
}

__global__ void rotation_bbox_coop(float *d_in, float *d_out, int width,
                                   int height, int channels, int angle)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  int tile_width = blockDim.x, tile_height = blockDim.y;
  int x = blockIdx.x * tile_width + tx;
  int y = blockIdx.y * tile_height + ty;

  extern __shared__ char sharedMem[];
  int *bbox = (int *)sharedMem;
  float *tile = (float *)((char *)sharedMem + 4 * sizeof(int));

  float2 currentSrcCoord = computeSrcCoord(x, y, angle, width, height);
  if (tx == 0 && ty == 0) {
    float2 c1 = currentSrcCoord;
    float2 c2 = computeSrcCoord(x + tile_width, y, angle, width, height);
    float2 c3 = computeSrcCoord(x, y + tile_height, angle, width, height);
    float2 c4 = computeSrcCoord(x + tile_width, y + tile_height, angle, width, height);
    bbox[0] = (int)floorf(fminf(fminf(c1.x, c2.x), fminf(c3.x, c4.x)));
    bbox[1] = (int)floorf(fminf(fminf(c1.y, c2.y), fminf(c3.y, c4.y)));
    bbox[2] = (int)ceilf(fmaxf(fmaxf(c1.x, c2.x), fmaxf(c3.x, c4.x)));
    bbox[3] = (int)ceilf(fmaxf(fmaxf(c1.y, c2.y), fmaxf(c3.y, c4.y)));
  }
  __syncthreads();

  int bbox_width = bbox[2] - bbox[0] + 1;
  int bbox_height = bbox[3] - bbox[1] + 1;

  for (int i = ty; i < bbox_height; i += blockDim.y) {
    for (int j = tx; j < bbox_width; j += blockDim.x) {
      int srcX = bbox[0] + j, srcY = bbox[1] + i;
      for (int c = 0; c < channels; c++) {
        tile[(i * bbox_width + j) * channels + c] =
            (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height)
            ? d_in[(srcY * width + srcX) * channels + c]
            : 0.0f;
      }
    }
  }
  __syncthreads();

  if (x < width && y < height) {
    float srcX_tile = currentSrcCoord.x - bbox[0];
    float srcY_tile = currentSrcCoord.y - bbox[1];
    for (unsigned int c = 0; c < (unsigned int)channels; c++)
      d_out[(y * width + x) * channels + c] =
          d_bilinearInterpolate(tile, bbox_width, bbox_height, channels, c, srcX_tile, srcY_tile);
  }
}

__global__ void rotation_bbox_strided(float *d_in, float *d_out, int width,
                                      int height, int channels, int angle)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  int tile_width = blockDim.x, tile_height = blockDim.y;
  int x = blockIdx.x * tile_width + tx;
  int y = blockIdx.y * tile_height + ty;

  extern __shared__ char sharedMem[];
  int *bbox = (int *)sharedMem;
  float *tile = (float *)((char *)sharedMem + 4 * sizeof(int));

  float2 currentSrcCoord = computeSrcCoord(x, y, angle, width, height);
  if (tx == 0 && ty == 0) {
    float2 c1 = currentSrcCoord;
    float2 c2 = computeSrcCoord(x + tile_width, y, angle, width, height);
    float2 c3 = computeSrcCoord(x, y + tile_height, angle, width, height);
    float2 c4 = computeSrcCoord(x + tile_width, y + tile_height, angle, width, height);
    bbox[0] = (int)floorf(fminf(fminf(c1.x, c2.x), fminf(c3.x, c4.x)));
    bbox[1] = (int)floorf(fminf(fminf(c1.y, c2.y), fminf(c3.y, c4.y)));
    bbox[2] = (int)ceilf(fmaxf(fmaxf(c1.x, c2.x), fmaxf(c3.x, c4.x)));
    bbox[3] = (int)ceilf(fmaxf(fmaxf(c1.y, c2.y), fmaxf(c3.y, c4.y)));
  }
  __syncthreads();

  int bbox_width = bbox[2] - bbox[0] + 1;
  int bbox_height = bbox[3] - bbox[1] + 1;
  int thread_id = ty * blockDim.x + tx;
  int total_threads = blockDim.x * blockDim.y;

  for (int i = thread_id; i < bbox_width * bbox_height; i += total_threads) {
    int local_x = i % bbox_width, local_y = i / bbox_width;
    int srcX = bbox[0] + local_x, srcY = bbox[1] + local_y;
    for (int c = 0; c < channels; c++) {
      tile[(local_y * bbox_width + local_x) * channels + c] =
          (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height)
          ? d_in[(srcY * width + srcX) * channels + c]
          : 0.0f;
    }
  }
  __syncthreads();

  if (x < width && y < height) {
    float srcX_tile = currentSrcCoord.x - bbox[0];
    float srcY_tile = currentSrcCoord.y - bbox[1];
    for (unsigned int c = 0; c < (unsigned int)channels; c++)
      d_out[(y * width + x) * channels + c] =
          d_bilinearInterpolate(tile, bbox_width, bbox_height, channels, c, srcX_tile, srcY_tile);
  }
}
