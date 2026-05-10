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
#include <vector>
#include "stb_image_write.h"
#include "cli_args.hpp"
#define PI 3.14159265358979323846
#define BLOCK_WIDTH 32
#define MAX_SHARED_WIDTH 50

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

// Bilinear interpolation function cpu
float bilinearInterpolate(float *d_in, int width, int height, int channels,
                          int c, float x, float y)
{
  int x1 = (int)floor(x);
  int y1 = (int)floor(y);
  int x2 = std::min(x1 + 1, width - 1);
  int y2 = std::min(y1 + 1, height - 1);

  // Ensure valid bounds
  if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height)
    return 0.0f;

  float a = x - x1;
  float b = y - y1;

  // Sample four pixels
  float p1 = d_in[(y1 * width + x1) * channels + c]; // Top-left
  float p2 = d_in[(y1 * width + x2) * channels + c]; // Top-right
  float p3 = d_in[(y2 * width + x1) * channels + c]; // Bottom-left
  float p4 = d_in[(y2 * width + x2) * channels + c]; // Bottom-right

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

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      // Compute the coordinates in the original image
      // Inverse transform to find the target location to sample from
      float srcX = ((x - cx) * cosA + (y - cy) * -sinA) + cx;
      float srcY = ((x - cx) * sinA + (y - cy) * cosA) + cy;

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

void verifyRotationImages(float *h_out, float *h_out_cpu, int width, int height,
                          int channels)
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      for (unsigned int c = 0; c < channels; c++)
      {
        int idx = (y * width + x) * channels + c;
        int diff = (int)abs(h_out[idx] - h_out_cpu[idx]);
        if (diff > 0)
        {
          std::cout << "Mismatch at X:" << x << " Y:" << y << "  C:" << c
                    << " Value:" << diff << " " << h_out[idx] << " "
                    << h_out_cpu[idx] << std::endl;
          return;
        }
      }
    }
  }
}

__device__ float d_bilinearInterpolate(float *d_in, int width, int height,
                                       int channels, int c, float x, float y)
{
  int x1 = (int)floor(x);
  int y1 = (int)floor(y);
  int x2 = min(x1 + 1, width - 1);
  int y2 = min(y1 + 1, height - 1);

  // Ensure valid bounds
  if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height)
    return 0;

  float a = x - x1;
  float b = y - y1;

  // Sample four pixels
  float p1 = d_in[(y1 * width + x1) * channels + c]; // Top-left
  float p2 = d_in[(y1 * width + x2) * channels + c]; // Top-right
  float p3 = d_in[(y2 * width + x1) * channels + c]; // Bottom-left
  float p4 = d_in[(y2 * width + x2) * channels + c]; // Bottom-right

  // Bilinear interpolation formula
  float interpolatedValue =
      (1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4;
  return interpolatedValue;
}

__device__ float2 computeSrcCoord(int x, int y, float angle, int width,
                                  int height)
{
  float radians = angle * PI / 180.0;
  float cosA = cos(radians);
  float sinA = sin(radians);
  int cx = width / 2;
  int cy = height / 2;

  // Compute the coordinates in the original image
  // Inverse transform to find the target location to sample from
  float srcX = ((x - cx) * cosA + (y - cy) * -sinA) + cx;
  float srcY = ((x - cx) * sinA + (y - cy) * cosA) + cy;

  return make_float2(srcX, srcY);
}

__global__ void rotation_naive(float *d_in, float *d_out, int width, int height,
                               int channels, int angle)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = (y * width + x) * channels;

  if (x < width && y < height)
  {
    // Compute the coordinate from the source image to get from
    float2 srcCoord = computeSrcCoord(x, y, angle, width, height);

    for (unsigned int c = 0; c < channels; c++)
    {
      int in_index = (y * width + x) * channels + c;
      d_out[in_index] = d_bilinearInterpolate(d_in, width, height, channels, c,
                                              srcCoord.x, srcCoord.y);
    }
  }
}

__global__ void tex_interpolation(cudaTextureObject_t texObj, float *output,
                                  int width, int height, int channels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    float u = x / (float)width;
    float v = y / (float)height;

    // tex2D automatically performs bilinear interpolation when we provide
    // floating-point coordinates that don't align exactly with pixel centers
    float4 value = tex2D<float4>(texObj, u, v);

    // Write the interpolated value to the output
    int idx = (y * width + x) * channels;
    output[idx + 0] = value.x; // Red
    output[idx + 1] = value.y; // Green
    output[idx + 2] = value.z; // Blue
  }
}

__global__ void rotation_bbox_coop(float *d_in, float *d_out, int width,
                                   int height, int channels, int angle)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tile_width = blockDim.x;
  int tile_height = blockDim.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int x = bx * tile_width + tx;
  int y = by * tile_height + ty;
  int idx = (y * width + x) * channels;

  // Shared Memory
  extern __shared__ char sharedMem[];
  int *bbox = (int *)sharedMem;
  int bbox_size = 4 * sizeof(int);
  float *tile = (float *)((char *)sharedMem + bbox_size);

  // Compute BBOX using single thread
  float2 currentSrcCoord = computeSrcCoord(x, y, angle, width, height);
  if (tx == 0 && ty == 0)
  {
    float2 corner1 = currentSrcCoord;
    float2 corner2 = computeSrcCoord(x + tile_width, y, angle, width, height);
    float2 corner3 = computeSrcCoord(x, y + tile_height, angle, width, height);
    float2 corner4 = computeSrcCoord(x + tile_width, y + tile_height, angle, width, height);

    float minX =
        fminf(fminf(corner1.x, corner2.x), fminf(corner3.x, corner4.x));
    float maxX =
        fmaxf(fmaxf(corner1.x, corner2.x), fmaxf(corner3.x, corner4.x));
    float minY =
        fminf(fminf(corner1.y, corner2.y), fminf(corner3.y, corner4.y));
    float maxY =
        fmaxf(fmaxf(corner1.y, corner2.y), fmaxf(corner3.y, corner4.y));

    // Store BBox coords in shared memory
    bbox[0] = (int)floorf(minX);
    bbox[1] = (int)floorf(minY);
    bbox[2] = (int)ceilf(maxX);
    bbox[3] = (int)ceilf(maxY);

    // printf("Bounding Box Size: %d %d\n", bbox_width, bbox_height);
  }

  __syncthreads();

  int bbox_width = bbox[2] - bbox[0] + 1;
  int bbox_height = bbox[3] - bbox[1] + 1;

  // Cooperative loading into shared memory
  for (int i = ty; i < bbox_height; i += blockDim.y)
  {
    for (int j = tx; j < bbox_width; j += blockDim.x)
    {
      int srcX = bbox[0] + j;
      int srcY = bbox[1] + i;
      if (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height)
      {
        for (int c = 0; c < channels; c++)
        {
          tile[(i * bbox_width + j) * channels + c] =
              d_in[(srcY * width + srcX) * channels + c];
        }
      }
      else // Out of Bounds
      {
        for (int c = 0; c < channels; c++)
        {
          tile[(i * bbox_width + j) * channels + c] = 0.0f;
        }
      }
    }
  }

  __syncthreads();

  // Bilinear Interpolate from shared memory
  if (x < width && y < height)
  {
    float srcX_tile = currentSrcCoord.x - bbox[0];
    float srcY_tile = currentSrcCoord.y - bbox[1];

    for (unsigned int c = 0; c < channels; c++)
    {
      int in_index = (y * width + x) * channels + c;
      d_out[in_index] = d_bilinearInterpolate(
          tile, bbox_width, bbox_height, channels, c, srcX_tile, srcY_tile);
    }
  }
}

__global__ void rotation_bbox_strided(float *d_in, float *d_out, int width,
                                      int height, int channels, int angle)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tile_width = blockDim.x;
  int tile_height = blockDim.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int x = bx * tile_width + tx;
  int y = by * tile_height + ty;
  int idx = (y * width + x) * channels;

  // Shared Memory
  extern __shared__ char sharedMem[];
  int *bbox = (int *)sharedMem;
  int bbox_size = 4 * sizeof(int);
  float *tile = (float *)((char *)sharedMem + bbox_size);

  // Compute BBOX using single thread
  float2 currentSrcCoord = computeSrcCoord(x, y, angle, width, height);
  if (tx == 0 && ty == 0)
  {
    float2 corner1 = currentSrcCoord;
    float2 corner2 = computeSrcCoord(x + tile_width, y, angle, width, height);
    float2 corner3 = computeSrcCoord(x, y + tile_height, angle, width, height);
    float2 corner4 =
        computeSrcCoord(x + tile_width, y + tile_height, angle, width, height);

    float minX = fminf(fminf(corner1.x, corner2.x), fminf(corner3.x, corner4.x));
    float maxX = fmaxf(fmaxf(corner1.x, corner2.x), fmaxf(corner3.x, corner4.x));
    float minY = fminf(fminf(corner1.y, corner2.y), fminf(corner3.y, corner4.y));
    float maxY = fmaxf(fmaxf(corner1.y, corner2.y), fmaxf(corner3.y, corner4.y));

    // Store BBox coords in shared memory
    bbox[0] = (int)floorf(minX);
    bbox[1] = (int)floorf(minY);
    bbox[2] = (int)ceilf(maxX);
    bbox[3] = (int)ceilf(maxY);

    // printf("Bounding Box Size: %d %d\n", bbox_width, bbox_height);
  }

  __syncthreads();

  int bbox_width = bbox[2] - bbox[0] + 1;
  int bbox_height = bbox[3] - bbox[1] + 1;

  // Cooperative loading into shared memory
  int thread_id = ty * blockDim.x + tx;
  int total_threads = blockDim.x * blockDim.y;
  int total_elements = bbox_width * bbox_height;

  for (int i = thread_id; i < total_elements; i += total_threads)
  {
    // Convert linear index back to 2D within bounding box
    int local_x = i % bbox_width;
    int local_y = i / bbox_width;

    // Map to source image coordinates
    int srcX = bbox[0] + local_x;
    int srcY = bbox[1] + local_y;

    if (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height)
    {
      // Read all channels with coalesced accesses
      for (int c = 0; c < channels; c++)
      {
        tile[(local_y * bbox_width + local_x) * channels + c] =
            d_in[(srcY * width + srcX) * channels + c];
      }
    }
    else
    {
      for (int c = 0; c < channels; c++)
      {
        tile[(local_y * bbox_width + local_x) * channels + c] = 0.0f;
      }
    }
  }

  __syncthreads();

  // Bilinear Interpolate from shared memory
  if (x < width && y < height)
  {
    float srcX_tile = currentSrcCoord.x - bbox[0];
    float srcY_tile = currentSrcCoord.y - bbox[1];

    for (unsigned int c = 0; c < channels; c++)
    {
      int in_index = (y * width + x) * channels + c;
      d_out[in_index] = d_bilinearInterpolate(
          tile, bbox_width, bbox_height, channels, c, srcX_tile, srcY_tile);
    }
  }
}

std::string get_string(const RunConfig &cfg, const std::string &flag,
                       const std::string &default_val)
{
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end())
    return default_val;
  return it->second;
}

enum RotationOption
{
  IMG_ROT_NAIVE,
  TEX_TEST,
  IMG_ROT_BBOX_COOP,
  IMG_ROT_BBOX_STRIDED
};

void run_rotation(RotationOption option, const RunConfig &cfg)
{
  // Parse variant-specific runtime arguments
  const std::string image_path = get_string(cfg, "--image", "res/cuda.png");
  const int rotation_angle_degrees = get_int(cfg, "--angle", 45);

  if (rotation_angle_degrees < -360 || rotation_angle_degrees > 360)
  {
    throw std::invalid_argument("--angle must be in range [-360, 360]");
  }

  int width, height, channels;
  unsigned char *h_in_char, *h_out_char;

  // Load input image
  h_in_char = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!h_in_char)
  {
    throw std::runtime_error(std::string("Error loading image: ") + image_path);
  }

  // Compute image memory footprint
  size_t img_pixel_count = width * height * channels;
  size_t img_size = img_pixel_count * sizeof(float);

  std::cout << "Image Width: " << width << "  Height:" << height
            << "  Channels:" << channels << std::endl;

  // Host buffers
  unsigned char *h_out_gray = new unsigned char[width * height];
  float *h_out_gray_f = new float[width * height];

  h_out_char = new unsigned char[img_pixel_count];
  float *h_out_cpu, *h_in, *h_out;

  h_out_cpu = (float *)malloc(img_size);
  h_in = (float *)malloc(img_size);
  h_out = (float *)malloc(img_size);

  // Convert input image to float domain for interpolation math
  std::transform(h_in_char, h_in_char + img_pixel_count, h_in,
                 [](unsigned char pixel)
                 { return pixel / 255.0f; });

  // Allocate GPU buffers
  CudaMemory<float> d_in(img_size);
  CudaMemory<float> d_out(img_size);

  // Allocate/copy CUDA array for texture path
  cudaArray *cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpyToArray(cuArray, 0, 0, h_in, img_size, cudaMemcpyHostToDevice);

  // Texture resource descriptor
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceType::cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Texture sampling descriptor
  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Copy input image to linear device buffer
  CUDA_CALL(cudaMemcpy(d_in.get(), h_in, img_size, cudaMemcpyHostToDevice));

  // CPU reference output for quick visual/debug comparison
  rotation_cpu(h_in, h_out_cpu, width, height, channels,
               rotation_angle_degrees);
  std::transform(h_out_cpu, h_out_cpu + img_pixel_count, h_out_char,
                 float_to_uchar);
  stbi_write_jpg("rot_cpu.jpg", width, height, channels, h_out_char, 100);

  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  if (option == IMG_ROT_BBOX_COOP || option == IMG_ROT_BBOX_STRIDED)
  {
    // Shared-memory bbox variants need a smaller block for resource limits
    blockSize = dim3(16, 16);
  }
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  printKernelConfig(gridSize, blockSize);

  size_t bbox_size = 4 * sizeof(int);
  size_t tile_size =
      MAX_SHARED_WIDTH * MAX_SHARED_WIDTH * sizeof(float) * channels;
  size_t shared_mem_size = bbox_size + tile_size;

  // Launch selected variant
  switch (option)
  {
  case IMG_ROT_NAIVE:
    rotation_naive<<<gridSize, blockSize>>>(d_in.get(), d_out.get(), width,
                                            height, channels,
                                            rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char,
                   float_to_uchar);
    stbi_write_jpg("rot_naive.jpg", width, height, channels, h_out_char,
                   100);
    break;
  case TEX_TEST:
    tex_interpolation<<<gridSize, blockSize>>>(texObj, d_out.get(), width,
                                               height, channels);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char,
                   float_to_uchar);
    stbi_write_jpg("tex_test.jpg", width, height, channels, h_out_char,
                   100);
    break;
  case IMG_ROT_BBOX_COOP:
    rotation_bbox_coop<<<gridSize, blockSize, shared_mem_size>>>(
        d_in.get(), d_out.get(), width, height, channels,
        rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char,
                   float_to_uchar);
    stbi_write_jpg("rot_bbox_coop.jpg", width, height, channels, h_out_char,
                   100);
    break;

  case IMG_ROT_BBOX_STRIDED:
    rotation_bbox_strided<<<gridSize, blockSize, shared_mem_size>>>(
        d_in.get(), d_out.get(), width, height, channels,
        rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char,
                   float_to_uchar);
    stbi_write_jpg("rot_bbox_strided.jpg", width, height, channels, h_out_char,
                   100);
    break;

  default:
    break;
  }

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  // Sanity compare against CPU path
  verifyRotationImages(h_out_cpu, h_out, width, height, channels);

  // Cleanup host/device resources
  stbi_image_free(h_in_char);
  free(h_in);
  free(h_out);
  free(h_out_cpu);

  delete[] h_out_gray;
  delete[] h_out_gray_f;
  delete[] h_out_char;

  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> rotation_args = {
      {"--image", "path", "res/cuda.png", "Input image path"},
      {"--angle", "int", "45", "Rotation angle in degrees"},
  };

  const std::vector<ArgSpec> texture_args = {
      {"--image", "path", "res/cuda.png", "Input image path"},
  };

  const VariantRegistry variants = {
      {
          "rotation-naive",
          "Image rotation with direct bilinear interpolation",
          rotation_args,
          [](const RunConfig &c)
          { run_rotation(IMG_ROT_NAIVE, c); },
      },
      {
          "texture-interpolation",
          "Texture interpolation sampling test",
          texture_args,
          [](const RunConfig &c)
          { run_rotation(TEX_TEST, c); },
      },
      {
          "rotation-bbox-coop",
          "Rotation with cooperative shared-memory BBox loading",
          rotation_args,
          [](const RunConfig &c)
          { run_rotation(IMG_ROT_BBOX_COOP, c); },
      },
      {
          "rotation-bbox-strided",
          "Rotation with strided shared-memory BBox loading",
          rotation_args,
          [](const RunConfig &c)
          { run_rotation(IMG_ROT_BBOX_STRIDED, c); },
      },
  };

  try
  {
    RunConfig cfg = parse_args(argc, argv, variants);
    if (cfg.print_help)
    {
      print_usage(argv[0], variants);
      return EXIT_SUCCESS;
    }
    if (cfg.list_variants)
    {
      print_variants(variants);
      return EXIT_SUCCESS;
    }

    // Set CUDA device and execute requested variant
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);

    // Reset device after execution (unless explicitly disabled)
    if (cfg.reset_device)
    {
      CUDA_CALL(cudaDeviceReset());
    }
  }
  catch (const std::exception &e)
  {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], variants);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}