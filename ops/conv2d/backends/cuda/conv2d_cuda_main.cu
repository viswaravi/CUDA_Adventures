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

#define PI 3.14159265358979323846

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

void initialize_kernel(int option, float *kernel, float strength = 1)
{
  if (!kernel) return;
  switch (option) {
  case 1: { float b[9] = {1,2,1,2,4,2,1,2,1}; float f = strength / 16.0f; for (int i=0;i<9;i++) kernel[i]=b[i]*f; } break;
  case 2: { float b[9] = {-1,0,1,-2,0,2,-1,0,1}; for (int i=0;i<9;i++) kernel[i]=b[i]*strength; } break;
  case 3: { float b[9] = {-1,-2,-1,0,0,0,1,2,1}; for (int i=0;i<9;i++) kernel[i]=b[i]*strength; } break;
  case 4: { float b[9] = {0,-1,0,-1,5,-1,0,-1,0}; for (int i=0;i<9;i++) kernel[i]=b[i]*strength; } break;
  case 5: { float b[9] = {1,2,-1,2,-16,2,1,2,1}; for (int i=0;i<9;i++) kernel[i]=b[i]*strength; } break;
  default: throw std::invalid_argument("Invalid kernel option!");
  }
}

void convolution_2d_cpu(float *d_in, float *d_out, int width, int height,
                        int channels, float *kernel, int kernel_width)
{
  int pad = kernel_width / 2;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int ky = -pad; ky <= pad; ky++) {
          for (int kx = -pad; kx <= pad; kx++) {
            int ix = clamp(x + kx, 0, width - 1);
            int iy = clamp(y + ky, 0, height - 1);
            sum += d_in[(iy * width + ix) * channels + c]
                 * kernel[(ky + pad) * kernel_width + (kx + pad)];
          }
        }
        d_out[(y * width + x) * channels + c] = sum;
      }
    }
  }
}

enum ConvolutionOption { RGB2GREY, TWO_D_CONV_NAIVE, TWO_D_CONV_SHARED_SINGLE, TWO_D_CONV_SHARED_COOP, TWO_D_CONV_SHARED_STRUCT };

void run_convolution(ConvolutionOption option, const RunConfig &cfg)
{
  const std::string image_path = get_string(cfg, "--image", "res/cuda.png");
  const int kernel_option = get_int(cfg, "--filter", 1);

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

  CudaMemory<float> d_in(img_size), d_out(img_size);
  CudaMemory<float> d_out_gray(width * height * sizeof(float));

  std::transform(h_in_char, h_in_char + img_pixel_count, h_in,
                 [](unsigned char pixel) { return pixel / 255.0f; });
  CUDA_CALL(cudaMemcpy(d_in.get(), h_in, img_size, cudaMemcpyHostToDevice));

  float kernel[9];
  int kernel_len = 3;
  initialize_kernel(kernel_option, kernel);
  size_t kernel_size = kernel_len * kernel_len * sizeof(float);
  CudaMemory<float> d_kernel(kernel_size);
  CUDA_CALL(cudaMemcpy(d_kernel.get(), kernel, kernel_size, cudaMemcpyHostToDevice));

  convolution_2d_cpu(h_in, h_out_cpu, width, height, channels, kernel, kernel_len);
  std::transform(h_out_cpu, h_out_cpu + img_pixel_count, h_out_char, float_to_uchar);
  stbi_write_jpg("conv_cpu.jpg", width, height, channels, h_out_char, 100);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
  int pad = kernel_len / 2;
  int shared_block_size = ((blockSize.x + 2 * pad) * (blockSize.y + 2 * pad)) * channels;

  printKernelConfig(gridSize, blockSize);
  std::cout << "Host Data Alignment: " << h_is_aligned(h_in, 4) << std::endl;
  std::cout << "Shared Block Size:" << shared_block_size << std::endl;

  switch (option) {
  case RGB2GREY:
    rgb_to_gray<<<gridSize, blockSize>>>(d_in.get(), d_out_gray.get(), width, height, channels);
    CUDA_CALL(cudaMemcpy(h_out_gray_f, d_out_gray.get(), width * height * sizeof(float), cudaMemcpyDeviceToHost));
    std::transform(h_out_gray_f, h_out_gray_f + (width * height), h_out_gray, float_to_uchar);
    stbi_write_jpg("gray.jpg", width, height, 1, h_out_gray, 100);
    break;
  case TWO_D_CONV_NAIVE:
    convolution_2d_naive<<<gridSize, blockSize>>>(d_in.get(), d_out.get(), width, height, channels, d_kernel.get(), kernel_len);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("conv_naive.jpg", width, height, channels, h_out_char, 100);
    break;
  case TWO_D_CONV_SHARED_SINGLE:
    convolution_2d_shared_single<<<gridSize, blockSize, shared_block_size * sizeof(float)>>>(d_in.get(), d_out.get(), width, height, channels, d_kernel.get(), kernel_len);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char, 100);
    break;
  case TWO_D_CONV_SHARED_COOP:
    convolution_2d_shared_coop<<<gridSize, blockSize, shared_block_size * sizeof(float)>>>(d_in.get(), d_out.get(), width, height, channels, d_kernel.get(), kernel_len);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char, 100);
    break;
  case TWO_D_CONV_SHARED_STRUCT:
    convolution_2d_shared_struct<<<gridSize, blockSize, shared_block_size * sizeof(float)>>>(d_in.get(), d_out.get(), width, height, channels, d_kernel.get(), kernel_len);
    CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    stbi_write_jpg("conv_shared.jpg", width, height, channels, h_out_char, 100);
    break;
  default: break;
  }

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  stbi_image_free(h_in_char);
  free(h_in); free(h_out); free(h_out_cpu);
  delete[] h_out_gray; delete[] h_out_gray_f; delete[] h_out_char;
}

int main(int argc, char **argv)
{
  const std::vector<ArgSpec> conv_args = {
      {"--op", "string", "conv2d", "Operation name"},
      {"--kernel", "string", "conv-naive", "Conv2D kernel name"},
      {"--image", "path", "res/cuda.png", "Input image path"},
      {"--filter", "int", "1", "Filter type: 1=Gaussian, 2=SobelX, 3=SobelY, 4=Sharpen, 5=LoG"},
  };

  try {
    RunConfig cfg = parse_args(argc, argv, conv_args);
    if (cfg.print_help) { print_usage(argv[0], conv_args, "CUDA conv2d backend runner."); return EXIT_SUCCESS; }
    if (get_string(cfg, "--op", "conv2d") != "conv2d")
      throw std::invalid_argument("--op must be conv2d");
    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    const std::string kernel = get_string(cfg, "--kernel", "conv-naive");
    if (kernel == "rgb2gray") {
      run_convolution(RGB2GREY, cfg);
    } else if (kernel == "conv-naive") {
      run_convolution(TWO_D_CONV_NAIVE, cfg);
    } else if (kernel == "conv-shared-single") {
      run_convolution(TWO_D_CONV_SHARED_SINGLE, cfg);
    } else if (kernel == "conv-shared-coop") {
      run_convolution(TWO_D_CONV_SHARED_COOP, cfg);
    } else if (kernel == "conv-shared-struct") {
      run_convolution(TWO_D_CONV_SHARED_STRUCT, cfg);
    } else {
      throw std::invalid_argument("Unsupported --kernel for conv2d: " + kernel);
    }
    if (cfg.reset_device) CUDA_CALL(cudaDeviceReset());
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], conv_args, "CUDA conv2d backend runner.");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
