#include <cuda_runtime.h>
#include <iostream>
#include "utils.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "stb_image_write.h"
#define PI 3.14159265358979323846

// Alignment check function
bool h_is_aligned(const void *ptr, size_t alignment)
{
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

__device__ bool d_is_aligned(const void *ptr, size_t alignment)
{
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// RGB to Gray kernel
__global__ void rgb_to_gray(float *d_in, float *d_out, int width, int height,
                            int channels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = (y * width + x) * channels;

  if (x < width && y < height)
  {
    float r = d_in[idx];
    float g = d_in[idx + 1];
    float b = d_in[idx + 2];
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    d_out[y * width + x] = gray;
  }
}

template <typename T>
T clamp(T value, T min_val, T max_val)
{
  return (value < min_val) ? min_val : (value > max_val ? max_val : value);
}

unsigned char float_to_uchar(float pixel)
{
  return static_cast<unsigned char>(
      std::round(clamp(pixel * 255.0f, 0.0f, 255.0f)));
}

void initialize_kernel(int option, float *kernel, float strength = 1)
{
  if (!kernel)
    return; // Safety check for null pointer

  switch (option)
  {
  case 1: // Gaussian Blur (3x3)
  {
    float base_kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    float factor = strength / 16.0f; // Normalize
    for (int i = 0; i < 9; ++i)
      kernel[i] = base_kernel[i] * factor;
  }
  break;

  case 2: // Sobel X (Vertical Edges)
  {
    float base_kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    for (int i = 0; i < 9; ++i)
      kernel[i] = base_kernel[i] * strength;
  }
  break;

  case 3: // Sobel Y (Horizontal Edges)
  {
    float base_kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    for (int i = 0; i < 9; ++i)
      kernel[i] = base_kernel[i] * strength;
  }
  break;

  case 4: // Sharpen
  {
    float base_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for (int i = 0; i < 9; ++i)
      kernel[i] = base_kernel[i] * strength;
  }
  break;

  case 5: // Laplacian of Gaussian
  {
    float base_kernel[9] = {1, 2, -1, 2, -16, 2, 1, 2, 1};
    for (int i = 0; i < 9; ++i)
      kernel[i] = base_kernel[i] * strength;
  }
  break;

  default:
    throw std::invalid_argument("Invalid kernel option!");
  }
}

void convolution_2d_cpu(float *d_in, float *d_out, int width, int height,
                        int channels, float *kernel, int kernel_width)
{
  int pad = kernel_width / 2;
  // For each Pixel
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      // For each Channel
      for (int c = 0; c < channels; ++c)
      {
        float sum = 0.0f;
        // For each Kernel Element
        for (int ky = -pad; ky <= pad; ++ky)
        {
          for (int kx = -pad; kx <= pad; ++kx)
          {
            // Handle Border - Halo Pixels
            int ix = clamp(x + kx, 0, width - 1);
            int iy = clamp(y + ky, 0, height - 1);
            int image_idx = (iy * width + ix) * channels + c;
            int kernel_idx = (ky + pad) * kernel_width + (kx + pad);
            sum += d_in[image_idx] * kernel[kernel_idx];
          }
        }
        d_out[(y * width + x) * channels + c] = sum;
      }
    }
  }
}

// Bilinear interpolation function cpu
float bilinearInterpolate(float *d_in, int width, int height, int channels,
                          int c, float x, float y)
{
  int x1 = (int)x;
  int y1 = (int)y;
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  if (x1 < 0 || x2 >= width || y1 < 0 || y2 >= height)
    return 0; // Out of bounds handling

  float a = x - x1;
  float b = y - y1;

  uint8_t p1 = d_in[(y1 * width + x1) * channels + c]; // Top-left
  uint8_t p2 = d_in[(y1 * width + x2) * channels + c]; // Top-right
  uint8_t p3 = d_in[(y2 * width + x1) * channels + c]; // Bottom-left
  uint8_t p4 = d_in[(y2 * width + x2) * channels + c]; // Bottom-right

  // Bilinear interpolation formula
  float interpolatedValue =
      (1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4;
  return interpolatedValue;
}

void rotation_cpu(float *d_in, float *d_out, int width, int height,
                  int channels, float angle)
{
  float radians = angle * PI / 180.0;
  float cosA = cos(radians);
  float sinA = sin(radians);

  int cx = width / 2;
  int cy = height / 2;

  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      // Compute the coordinates in the original image
      float srcX = (x - cx) * cosA + (y - cy) * -sinA + cx;
      float srcY = (x - cx) * sinA + (y - cy) * cosA + cy;

      // Interpolate and set pixel
      for (unsigned int c = 0; c < channels; c++)
      {
        int in_index = (y * width + x) * channels + c;
        d_out[in_index] =
            bilinearInterpolate(d_in, width, height, channels, c, srcX, srcY);
      }
    }
  }
}

__device__ int d_clamp(int x, int a, int b)
{
  return fmaxf(a, fminf(b, x));
}

// 2D Convolution Kernel Naive
__global__ void convolution_2d_naive(float *d_in, float *d_out, int width,
                                     int height, int channels, float *d_kernel,
                                     int kernel_width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = (y * width + x) * channels;

  int pad = kernel_width / 2;

  if (x < width && y < height)
  {
    for (int c = 0; c < channels; c++)
    {
      float sum = 0.0f;
      for (int ky = -pad; ky <= pad; ky++)
      {
        for (int kx = -pad; kx <= pad; kx++)
        {
          int in_y = d_clamp(y + ky, 0, height - 1);
          int in_x = d_clamp(x + kx, 0, width - 1);
          int in_idx = (in_y * width + in_x) * channels + c;
          int kernel_idx = (ky + pad) * kernel_width + (kx + pad);
          sum += d_in[in_idx] * d_kernel[kernel_idx];
        }
      }
      int out_idx = (y * width + x) * channels + c;
      d_out[out_idx] = sum;
    }
  }
}

// 2D Convolution - Single Threaded
__global__ void convolution_2d_shared_single(float *d_in, float *d_out,
                                             int width, int height,
                                             int channels, float *d_kernel,
                                             int kernel_width)
{
  // Shared Memory for Current Block
  extern __shared__ float tile[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int pad = kernel_width / 2;
  int tw = blockDim.x; // sqaure tile width
  int shared_width = tw + (2 * pad);

  // Single Threaded Loading
  if (tx == 0 && ty == 0)
  {
    for (int c = 0; c < channels; c++)
    {
      for (int i = -pad; i < pad + tw; ++i)
      {
        for (int j = -pad; j < pad + tw; ++j)
        {
          int gy = d_clamp(y + i, 0, height - 1);
          int gx = d_clamp(x + j, 0, width - 1);
          int in_idx = (gy * width + gx) * channels + c;
          int tile_idx = ((i + pad) * shared_width + (j + pad)) * channels + c;
          tile[tile_idx] = d_in[in_idx];
        }
      }
    }
  }

  __syncthreads();

  // Perform Convolution
  for (int c = 0; c < channels; c++)
  {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++)
    {
      for (int kx = -pad; kx <= pad; kx++)
      {
        // use input from shared memory
        int in_y = d_clamp((ty + pad) + ky, 0, shared_width - 1);
        int in_x = d_clamp((tx + pad) + kx, 0, shared_width - 1);
        int in_idx = (in_y * shared_width + in_x) * channels + c;
        int kernel_idx = (ky + pad) * kernel_width + (kx + pad);
        sum += tile[in_idx] * d_kernel[kernel_idx];
      }
    }

    if (x < width && y < height)
    {
      int out_idx = (y * width + x) * channels + c;
      d_out[out_idx] = sum;
    }
  }
}

// 2D Convolution - Cooperative Loading
__global__ void convolution_2d_shared_coop(float *d_in, float *d_out, int width,
                                           int height, int channels,
                                           float *d_kernel, int kernel_width)
{
  // Shared Memory for Current Block
  extern __shared__ float tile[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int pad = kernel_width / 2;
  int tw = blockDim.x; // sqaure tile width
  int shared_width = tw + (2 * pad);

  // Cooperative Loading
  int local_id = ty * tw + tx;
  int num_threads = tw * tw;
  int total_elems_in_tile = shared_width * shared_width;
  for (int i = local_id; i < total_elems_in_tile; i += num_threads)
  {
    // tile coordinates (tX, tY) in [0..shared_width-1], [0..shared_height-1]
    int tY = i / shared_width; // row within shared tile
    int tX = i % shared_width; // col within shared tile

    // Convert tile coords (tX, tY) into global input coords (gX, gY)
    // We shift by (-pad) so that tile(0,0) corresponds to input(x-pad, y-pad).
    int gX = blockIdx.x * blockDim.x + (tX - pad);
    int gY = blockIdx.y * blockDim.y + (tY - pad);
    gX = d_clamp(gX, 0, width - 1);
    gY = d_clamp(gY, 0, height - 1);

    // Load each channel
    for (int c = 0; c < channels; c++)
    {
      int tile_id = (tY * shared_width + tX) * channels + c;
      int in_idx = (gY * width + gX) * channels + c;
      tile[tile_id] = d_in[in_idx];
    }
  }

  __syncthreads();

  // Perform Convolution
  for (int c = 0; c < channels; c++)
  {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++)
    {
      for (int kx = -pad; kx <= pad; kx++)
      {
        // use input from shared memory
        int in_y = d_clamp((ty + pad) + ky, 0, shared_width - 1);
        int in_x = d_clamp((tx + pad) + kx, 0, shared_width - 1);
        int in_idx = (in_y * shared_width + in_x) * channels + c;
        int kernel_idx = (ky + pad) * kernel_width + (kx + pad);
        sum += tile[in_idx] * d_kernel[kernel_idx];
      }
    }

    if (x < width && y < height)
    {
      int out_idx = (y * width + x) * channels + c;
      d_out[out_idx] = sum;
    }
  }
}

// 2D Convolution - Structured Loading
__global__ void convolution_2d_shared_struct(float *d_in, float *d_out,
                                             int width, int height,
                                             int channels, float *d_kernel,
                                             int kernel_width)
{
  // Shared Memory for Current Block
  extern __shared__ float tile[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;

  // For blocks with partially valid data, the rest of the shared memory will go uninitialized
  // Thus, clamp the global x,y to mirror edge pixels
  int x = d_clamp(bx + tx, 0, width - 1);
  int y = d_clamp(by + ty, 0, height - 1);

  int pad = kernel_width / 2;
  int tw = blockDim.x; // Square tile width
  int shared_width = tw + (2 * pad);

  // Structured Loading for All Channels
  for (int c = 0; c < channels; c++)
  {
    // Load Main Tile Element
    int tile_idx = ((ty + pad) * shared_width + (tx + pad)) * channels + c;
    int in_idx = (y * width + x) * channels + c;
    tile[tile_idx] = d_in[in_idx];

    // Load Top & Bottom Halos
    if (ty < pad)
    {
      int y_in_top = d_clamp(by - pad + ty, 0, height - 1);
      int y_in_bottom = d_clamp(by + tw + ty, 0, height - 1);

      if (y_in_top >= 0 && x < width)
      {
        int tile_idx_top = (ty * shared_width + (tx + pad)) * channels + c;
        int in_idx_top = (y_in_top * width + x) * channels + c;

        tile[tile_idx_top] = d_in[in_idx_top];
      }

      if (y_in_bottom < height && x < width)
      {
        int tile_idx_bottom = ((ty + tw + pad) * shared_width + (tx + pad)) * channels + c;
        int in_idx_bottom = (y_in_bottom * width + x) * channels + c;
        tile[tile_idx_bottom] = d_in[in_idx_bottom];
      }
    }

    // Load Left & Right Halos
    if (tx < pad)
    {
      int x_in_left = d_clamp(bx - pad + tx, 0, width - 1);
      int x_in_right = d_clamp(bx + tw + tx, 0, width - 1);

      if (x_in_left >= 0 && y < height)
      {
        int tile_idx_left = ((ty + pad) * shared_width + tx) * channels + c;
        int in_idx_left = (y * width + x_in_left) * channels + c;
        tile[tile_idx_left] = d_in[in_idx_left];
      }

      if (x_in_right < width && y < height)
      {
        int tile_idx_right =
            ((ty + pad) * shared_width + (pad + tw + tx)) * channels + c;
        int in_idx_right = (y * width + x_in_right) * channels + c;
        tile[tile_idx_right] = d_in[in_idx_right];
      }
    }

    // Load Corner Elements
    if (tx < pad && ty < pad)
    {
      int y_tl = d_clamp(by - pad + ty, 0, height - 1);
      int x_tl = d_clamp(bx - pad + tx, 0, width - 1);

      int y_tr = d_clamp(by - pad + ty, 0, height - 1);
      int x_tr = d_clamp(bx + tw + tx, 0, width - 1);

      int y_bl = d_clamp(by + tw + ty, 0, height - 1);
      int x_bl = d_clamp(bx - pad + tx, 0, width - 1);

      int y_br = d_clamp(by + tw + ty, 0, height - 1);
      int x_br = d_clamp(bx + tw + tx, 0, width - 1);

      // Top Left
      if (x_tl >= 0 && y_tl >= 0)
      {
        int tile_idx_tl = (ty * shared_width + tx) * channels + c;
        int in_idx_tl = (y_tl * width + x_tl) * channels + c;
        tile[tile_idx_tl] = d_in[in_idx_tl];
      }

      // Top Right
      if (x_tr < width && y_tr >= 0)
      {
        int tile_idx_tr = (ty * shared_width + (pad + tw + tx)) * channels + c;
        int in_idx_tr = (y_tr * width + x_tr) * channels + c;
        tile[tile_idx_tr] = d_in[in_idx_tr];
      }

      // Bottom Left
      if (x_bl >= 0 && y_bl < height)
      {
        int tile_idx_bl = ((ty + pad + tw) * shared_width + tx) * channels + c;
        int in_idx_bl = (y_bl * width + x_bl) * channels + c;
        tile[tile_idx_bl] = d_in[in_idx_bl];
      }

      // Bottom Right
      if (x_br < width && y_br < height)
      {
        int tile_idx_br = ((ty + pad + tw) * shared_width + (pad + tw + tx)) * channels + c;
        int in_idx_br = (y_br * width + x_br) * channels + c;
        tile[tile_idx_br] = d_in[in_idx_br];
      }
    }
  }

  __syncthreads();

  // Perform Convolution (Same as Before)
  for (int c = 0; c < channels; c++)
  {
    float sum = 0.0f;
    for (int ky = -pad; ky <= pad; ky++)
    {
      for (int kx = -pad; kx <= pad; kx++)
      {
        int in_y = (ty + pad) + ky;
        int in_x = (tx + pad) + kx;
        int in_idx = (in_y * shared_width + in_x) * channels + c;
        int kernel_idx = (ky + pad) * kernel_width + (kx + pad);
        sum += tile[in_idx] * d_kernel[kernel_idx];
      }
    }

    if (x < width && y < height)
    {
      int out_idx = (y * width + x) * channels + c;
      d_out[out_idx] = sum;
    }
  }
}

int main()
{
  // Choose GPU
  CUDA_CALL(cudaSetDevice(0));

  try
  {
    enum Options
    {
      RGB2GREY,
      TWO_D_CONV_NAIVE,
      TWO_D_CONV_SHARED_SINGLE,
      TWO_D_CONV_SHARED_COOP,
      TWO_D_CONV_SHARED_STRUCT
    };
    Options option = TWO_D_CONV_SHARED_COOP;

    // Load Image
    int width, height, channels;
    unsigned char *h_in_char, *h_out_char;

    h_in_char = stbi_load("C:\\Users\\rvisw\\Pictures\\Screenshots\\cuda.png",
                          &width, &height, &channels, 0);
    if (!h_in_char)
    {
      throw std::runtime_error(std::string("Error loading image"));
    }

    // Allocate GPU Memory
    size_t img_pixel_count = width * height * channels;
    size_t img_size = img_pixel_count * sizeof(float);
    size_t gray_img_size = width * height * sizeof(unsigned char);

    std::cout << "Image Width: " << width << "  Height:" << height
              << "  Channels:" << channels << std::endl;

    // Host Data
    // Gray
    unsigned char *h_out_gray = new unsigned char[width * height];
    float *h_out_gray_f = new float[width * height];

    // RGB
    h_out_char = new unsigned char[img_pixel_count];
    float *h_out_cpu, *h_in, *h_out;

    h_out_cpu = (float *)malloc(img_size);
    h_in = (float *)malloc(img_size);
    h_out = (float *)malloc(img_size);

    // GPU
    CudaMemory<float> d_in(img_size);
    CudaMemory<float> d_out(img_size);
    CudaMemory<float> d_out_gray(width * height * sizeof(float));

    // Convert Char to Float for precise convolution
    std::transform(h_in_char, h_in_char + img_pixel_count, h_in,
                   [](unsigned char pixel)
                   { return pixel / 255.0f; });

    // Load in GPU
    CUDA_CALL(cudaMemcpy(d_in.get(), h_in, img_size, cudaMemcpyHostToDevice));

    // Initialize kernel for Convolution
    float kernel[9];
    int kernel_len = 3;
    initialize_kernel(1, kernel);
    // Move kernel to GPU
    size_t kernel_size = kernel_len * kernel_len * sizeof(float);
    CudaMemory<float> d_kernel(kernel_size);
    CUDA_CALL(cudaMemcpy(d_kernel.get(), kernel, kernel_size,
                         cudaMemcpyHostToDevice));

    // Perform GT Conv CPU version
    convolution_2d_cpu(h_in, h_out_cpu, width, height, channels, kernel,
                       kernel_len);
    std::transform(h_out_cpu, h_out_cpu + img_pixel_count, h_out_char,
                   float_to_uchar);
    stbi_write_jpg("conv_cpu.jpg", width, height, channels, h_out_char, 100);

    // Perform GPU Variant
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    int pad = kernel_len / 2;
    int shared_block_size =
        ((blockSize.x + 2 * pad) * (blockSize.y + 2 * pad)) *
        channels; // W * H * C

    printKernelConfig(gridSize, blockSize);

    std::cout << "Host Data Alignment: " << h_is_aligned(h_in, 4) << std::endl;

    std::cout << "Shared Block Size:" << shared_block_size << std::endl;

    switch (option)
    {
    case RGB2GREY:
      rgb_to_gray<<<gridSize, blockSize>>>(d_in.get(), d_out_gray.get(),
                                           width, height, channels);
      CUDA_CALL(cudaMemcpy(h_out_gray_f, d_out_gray.get(),
                           width * height * sizeof(float),
                           cudaMemcpyDeviceToHost));
      std::transform(h_out_gray_f, h_out_gray_f + (width * height),
                     h_out_gray, float_to_uchar);
      stbi_write_jpg("gray.jpg", width, height, 1, h_out_gray, 100);
      break;

    // Convolution
    case TWO_D_CONV_NAIVE:
      convolution_2d_naive<<<gridSize, blockSize>>>(
          d_in.get(), d_out.get(), width, height, channels, d_kernel.get(),
          kernel_len);
      CUDA_CALL(
          cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
      std::transform(h_out, h_out + img_pixel_count, h_out_char,
                     float_to_uchar);
      stbi_write_jpg("conv_naive.jpg", width, height, channels, h_out_char,
                     100);
      break;
    case TWO_D_CONV_SHARED_SINGLE:
      convolution_2d_shared_single<<<gridSize, blockSize,
                                     shared_block_size * sizeof(float)>>>(
          d_in.get(), d_out.get(), width, height, channels, d_kernel.get(),
          kernel_len);
      CUDA_CALL(
          cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
      std::transform(h_out, h_out + img_pixel_count, h_out_char,
                     float_to_uchar);
      stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char,
                     100);
      break;

    case TWO_D_CONV_SHARED_COOP:
      convolution_2d_shared_coop<<<gridSize, blockSize,
                                   shared_block_size * sizeof(float)>>>(
          d_in.get(), d_out.get(), width, height, channels, d_kernel.get(),
          kernel_len);
      CUDA_CALL(
          cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
      std::transform(h_out, h_out + img_pixel_count, h_out_char,
                     float_to_uchar);
      stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char,
                     100);
      break;

    case TWO_D_CONV_SHARED_STRUCT:
      convolution_2d_shared_struct<<<gridSize, blockSize,
                                     shared_block_size * sizeof(float)>>>(
          d_in.get(), d_out.get(), width, height, channels, d_kernel.get(),
          kernel_len);
      CUDA_CALL(
          cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
      std::transform(h_out, h_out + img_pixel_count, h_out_char,
                     float_to_uchar);
      stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char,
                     100);
      break;

    default:
      break;
    }

    // Free Memory
    stbi_image_free(h_in_char);
    free(h_in);
    free(h_out);
    free(h_out_cpu);

    // cudaDeviceReset - for profiling
    CUDA_CALL(cudaDeviceReset());
  }
  catch (std::exception &e)
  {
    fprintf(stderr, "Exception: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return 0;
}