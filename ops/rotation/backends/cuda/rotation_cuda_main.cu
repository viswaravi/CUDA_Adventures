#include <cuda_runtime.h>
#include "utils.cuh"
#include "kernels.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cli_args.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

bool h_is_aligned(const void *ptr, size_t alignment)
{
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
T clamp(T value, T min_val, T max_val)
{
  return (value < min_val) ? min_val : (value > max_val ? max_val : value);
}

unsigned char float_to_uchar(float pixel)
{
  return static_cast<unsigned char>(std::round(clamp(pixel * 255.0f, 0.0f, 255.0f)));
}

float bilinearInterpolate(float *d_in, int width, int height, int channels,
                          int c, float x, float y)
{
  int x1 = (int)floor(x), y1 = (int)floor(y);
  int x2 = std::min(x1 + 1, width - 1);
  int y2 = std::min(y1 + 1, height - 1);
  if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height) return 0.0f;
  float a = x - x1, b = y - y1;
  float p1 = d_in[(y1 * width + x1) * channels + c];
  float p2 = d_in[(y1 * width + x2) * channels + c];
  float p3 = d_in[(y2 * width + x1) * channels + c];
  float p4 = d_in[(y2 * width + x2) * channels + c];
  return (1 - a) * (1 - b) * p1 + a * (1 - b) * p2 + (1 - a) * b * p3 + a * b * p4;
}

void rotation_cpu(float *d_in, float *d_out, int width, int height,
                  int channels, float angle)
{
  float radians = angle * PI / 180.0;
  float cosA = cos(radians), sinA = sin(radians);
  int cx = width / 2, cy = height / 2;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float srcX = ((x - cx) * cosA + (y - cy) * -sinA) + cx;
      float srcY = ((x - cx) * sinA + (y - cy) * cosA) + cy;
      for (unsigned int c = 0; c < (unsigned int)channels; c++)
        d_out[(y * width + x) * channels + c] =
            bilinearInterpolate(d_in, width, height, channels, c, srcX, srcY);
    }
  }
}

void verifyRotationImages(float *h_out, float *h_out_cpu, int width, int height, int channels)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        int idx = (y * width + x) * channels + c;
        if ((int)abs(h_out[idx] - h_out_cpu[idx]) > 0) {
          std::cout << "Mismatch at X:" << x << " Y:" << y << " C:" << c
                    << " Value:" << (int)abs(h_out[idx] - h_out_cpu[idx])
                    << std::endl;
          return;
        }
      }
    }
  }
}

enum RotationOption { IMG_ROT_NAIVE, TEX_TEST, IMG_ROT_BBOX_COOP, IMG_ROT_BBOX_STRIDED };

void run_rotation(RotationOption option, const RunConfig &cfg)
{
  const std::string image_path = get_string(cfg, "--image", "res/cuda.png");
  const int rotation_angle_degrees = get_int(cfg, "--angle", 45);

  if (rotation_angle_degrees < -360 || rotation_angle_degrees > 360)
    throw std::invalid_argument("--angle must be in range [-360, 360]");

  int width, height, channels;
  unsigned char *h_in_char = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!h_in_char)
    throw std::runtime_error(std::string("Error loading image: ") + image_path);

  size_t img_pixel_count = width * height * channels;
  size_t img_size = img_pixel_count * sizeof(float);
  std::cout << "Image Width: " << width << "  Height:" << height << "  Channels:" << channels << std::endl;

  unsigned char *h_out_gray = new unsigned char[width * height];
  float *h_out_gray_f = new float[width * height];
  unsigned char *h_out_char = new unsigned char[img_pixel_count];
  float *h_out_cpu = (float *)malloc(img_size);
  float *h_in = (float *)malloc(img_size);
  float *h_out = (float *)malloc(img_size);

  std::transform(h_in_char, h_in_char + img_pixel_count, h_in,
                 [](unsigned char pixel) { return pixel / 255.0f; });

  CudaMemory<float> d_in(img_size), d_out(img_size);

  cudaArray *cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpyToArray(cuArray, 0, 0, h_in, img_size, cudaMemcpyHostToDevice);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  CUDA_CALL(cudaMemcpy(d_in.get(), h_in, img_size, cudaMemcpyHostToDevice));

  rotation_cpu(h_in, h_out_cpu, width, height, channels, rotation_angle_degrees);
  std::transform(h_out_cpu, h_out_cpu + img_pixel_count, h_out_char, float_to_uchar);
  stbi_write_jpg("rot_cpu.jpg", width, height, channels, h_out_char, 100);

  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  if (option == IMG_ROT_BBOX_COOP || option == IMG_ROT_BBOX_STRIDED) blockSize = dim3(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
  printKernelConfig(gridSize, blockSize);

  size_t shared_mem_size = 4 * sizeof(int) + MAX_SHARED_WIDTH * MAX_SHARED_WIDTH * sizeof(float) * channels;

  switch (option) {
  case IMG_ROT_NAIVE:
    rotation_naive<<<gridSize, blockSize>>>(d_in.get(), d_out.get(), width, height, channels, rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("rot_naive.jpg", width, height, channels, h_out_char, 100);
    break;
  case TEX_TEST:
    tex_interpolation<<<gridSize, blockSize>>>(texObj, d_out.get(), width, height, channels);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("tex_test.jpg", width, height, channels, h_out_char, 100);
    break;
  case IMG_ROT_BBOX_COOP:
    rotation_bbox_coop<<<gridSize, blockSize, shared_mem_size>>>(d_in.get(), d_out.get(), width, height, channels, rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("rot_bbox_coop.jpg", width, height, channels, h_out_char, 100);
    break;
  case IMG_ROT_BBOX_STRIDED:
    rotation_bbox_strided<<<gridSize, blockSize, shared_mem_size>>>(d_in.get(), d_out.get(), width, height, channels, rotation_angle_degrees);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("rot_bbox_strided.jpg", width, height, channels, h_out_char, 100);
    break;
  default: break;
  }

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  verifyRotationImages(h_out_cpu, h_out, width, height, channels);

  stbi_image_free(h_in_char);
  free(h_in); free(h_out); free(h_out_cpu);
  delete[] h_out_gray; delete[] h_out_gray_f; delete[] h_out_char;
  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> rotation_args = {
      {"--op", "string", "rotation", "Operation name"},
      {"--kernel", "string", "rotation-naive", "Rotation kernel name"},
      {"--image", "path", "res/cuda.png", "Input image path"},
      {"--angle", "int", "45", "Rotation angle in degrees"},
  };

  try {
    RunConfig cfg = parse_args(argc, argv, rotation_args);
    if (cfg.print_help) { print_usage(argv[0], rotation_args, "CUDA rotation backend runner."); return EXIT_SUCCESS; }
    if (get_string(cfg, "--op", "rotation") != "rotation")
      throw std::invalid_argument("--op must be rotation");
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    const std::string kernel = get_string(cfg, "--kernel", "rotation-naive");
    if (kernel == "rotation-naive") {
      run_rotation(IMG_ROT_NAIVE, cfg);
    } else if (kernel == "texture-interpolation") {
      run_rotation(TEX_TEST, cfg);
    } else if (kernel == "rotation-bbox-coop") {
      run_rotation(IMG_ROT_BBOX_COOP, cfg);
    } else if (kernel == "rotation-bbox-strided") {
      run_rotation(IMG_ROT_BBOX_STRIDED, cfg);
    } else {
      throw std::invalid_argument("Unsupported --kernel for rotation: " + kernel);
    }
    if (cfg.reset_device) CUDA_CALL(cudaDeviceReset());
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], rotation_args, "CUDA rotation backend runner.");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
