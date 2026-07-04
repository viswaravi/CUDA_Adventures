#include <cuda_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "stb_image_write.h"
#define PI 3.14159265358979323846

// Error Handling Wrapper
inline void checkCuda(cudaError_t result, const char *func, const char *file,
                      int line)
{
  if (result != cudaSuccess)
  {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + " (" + func +
                             "): " + cudaGetErrorString(result));
  }
}

#define CUDA_CALL(func) checkCuda((func), #func, __FILE__, __LINE__)
#define CHECK_CUFFT(call)                               \
  {                                                     \
    cufftResult err = call;                             \
    if (err != CUFFT_SUCCESS)                           \
    {                                                   \
      std::cerr << "cuFFT error: " << err << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

// Memory Handling Wrapper - RAII
template <typename T>
class CudaMemory
{
public:
  explicit CudaMemory(size_t size)
  {
    ptr = nullptr;
    CUDA_CALL(cudaMalloc((void **)&ptr, size));
  }

  // Prevent copying
  CudaMemory(const CudaMemory &) = delete;
  CudaMemory &operator=(const CudaMemory &) = delete;

  // Allow Moving
  CudaMemory(CudaMemory &&other) noexcept : ptr(other.ptr)
  {
    other.ptr = nullptr;
  }
  CudaMemory &operator=(CudaMemory &&other) noexcept
  {
    if (this != &other)
    {
      if (ptr)
        cudaFree(ptr);
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  void free()
  {
    if (ptr)
      cudaFree(ptr);
  }

  ~CudaMemory() { free(); }

  T *get() const { return ptr; }

private:
  T *ptr;
};

void printKernelConfig(dim3 grid, dim3 block)
{
  std::cout << "Grid: " << grid.x << " " << grid.y << " " << grid.z
            << std::endl;
  std::cout << "Block: " << block.x << " " << block.y << " " << block.z
            << std::endl;
  std::cout << "Threads in X:" << grid.x * block.x << " Y:" << grid.y * block.y
            << " Z:" << grid.z * block.z << std::endl;
}

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
__global__ void fconvolution_2d_naive(float *d_in, float *d_out, int width,
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

int computePaddingSize(int input_len, int kernel_len)
{
  int out_len = input_len + kernel_len - 1;

  int exp = (int)log2(out_len);

  if (pow(2, exp) < out_len)
  {
    exp += 1;
  }

  return pow(2, exp);
}

void copyToPadded(float *h_in, float *h_out, int width, int height,
                  int channels, int out_width, int out_height)
{
  for (int y = 0; y < out_height; y++)
  {
    for (int x = 0; x < out_width; x++)
    {
      for (int c = 0; c < channels; c++)
      {
        h_out[(y * out_width + x) * channels + c] =
            (y < height && x < width) ? h_in[(y * width + x) * channels + c]
                                      : 0;
      }
    }
  }
}

__global__ void normalize(float *d_out, int width, int height)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width)
  {
    int idx = i * width + j;
    d_out[idx] /= (float)(width * height);
  }
}

__global__ void complexPointwiseMulBatched(cufftComplex *a, cufftComplex *b,
                                           int nx, int ny_half, int batch)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y; // Row
  int j = blockIdx.x * blockDim.x + threadIdx.x; // Col
  int c = blockIdx.z;                            // Channel
  if (i < nx && j < ny_half && c < batch)
  {
    int idx = c * nx * ny_half + i * ny_half + j;
    float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
    float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
    a[idx].x = real;
    a[idx].y = imag;
  }
}

__global__ void complexPointwiseMul(cufftComplex *a, cufftComplex *b, int width,
                                    int height)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y; // Row
  int col = blockIdx.x * blockDim.x + threadIdx.x; // Col

  if (col < width && row < ny_half)
  {
    int idx = width * ny_half + i * ny_half + j;
    float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
    float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
    a[idx].x = real;
    a[idx].y = imag;
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
      TWO_D_FCONV_NAIVE
    };
    Options option = TWO_D_FCONV_NAIVE;

    // Load Image
    int width, height, channels;
    unsigned char *h_in_char, *h_out_char;

    h_in_char = stbi_load("C:\\Users\\rvisw\\Pictures\\Screenshots\\cuda.png",
                          &width, &height, &channels, 0);
    if (!h_in_char)
    {
      throw std::runtime_error(std::string("Error loading image"));
    }

    // Input Image Size
    size_t img_pixel_count = width * height * channels;
    size_t img_size = img_pixel_count * sizeof(float);
    std::cout << "Input Image Width: " << width << "  Height:" << height
              << "  Channels:" << channels << std::endl;

    // Initialize kernel for Convolution
    int kernel_len = 3;

    // Padded Image Size for Fast Convolution
    int out_width = computePaddingSize(width, kernel_len);
    int out_height = computePaddingSize(height, kernel_len);
    size_t padded_img_pixel_count = out_width * out_height * channels;
    size_t padded_img_size_R = padded_img_pixel_count * sizeof(float);
    size_t padded_img_size_C =
        out_width * (out_height / 2) * channels * sizeof(cufftComplex);

    std::cout << "Padded Image W:" << out_width << " H:" << out_height
              << std::endl;

    // Allocate Host Memory
    // Image
    h_out_char = new unsigned char[img_pixel_count];
    float *h_out_cpu, *h_in, *h_out, *h_in_padded, *h_out_padded;

    h_out_cpu = (float *)malloc(img_size);
    h_in = (float *)malloc(img_size);
    h_out = (float *)malloc(img_size);
    h_in_padded = (float *)malloc(padded_img_size_R);
    h_out_padded = (float *)malloc(padded_img_size_R);
    // Convert Char to Float for precise convolution
    std::transform(h_in_char, h_in_char + img_pixel_count, h_in,
                   [](unsigned char pixel)
                   { return pixel / 255.0f; });
    copyToPadded(h_in, h_in_padded, width, height, channels, out_width,
                 out_height);

    // Allocate Host Memory Kernel
    float kernel[9];
    initialize_kernel(1, kernel);
    size_t padded_kernel_size_R = out_width * out_height * sizeof(float);
    size_t padded_kernel_size_C =
        out_width * (out_height / 2) * sizeof(cufftComplex);
    float *kernel_padded = (float *)malloc(padded_kernel_size_R);
    copyToPadded(kernel, kernel_padded, kernel_len, kernel_len, 1, out_width,
                 out_height);

    // Allocate GPU Memory
    // Image
    CudaMemory<float> d_in_padded_R(padded_img_size_R);
    CudaMemory<cufftComplex> d_in_padded_C(padded_img_size_C);
    CudaMemory<cufftComplex> d_out_padded_C(padded_img_size_C);
    CudaMemory<float> d_out_padded_R(padded_img_size_R);
    // Kernel
    CudaMemory<float> d_kernel_R(padded_kernel_size_R);
    CudaMemory<cufftComplex> d_kernel_C(padded_kernel_size_C);

    // Copy Data to Device
    CUDA_CALL(cudaMemcpy(d_in_padded_R.get(), h_in_padded, padded_img_size_R,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kernel_R.get(), kernel_padded, padded_kernel_size_R,
                         cudaMemcpyHostToDevice));

    // Interleaved to Planar
    cufftHandle plan_r2c, plan_c2r;
    int n[2] = {nx, ny}; // 2D size
    CHECK_CUFFT(cufftPlanMany(&plan_r2c, 2, n, NULL, 1, nx * ny, NULL, 1,
                              nx * (ny / 2 + 1), CUFFT_R2C, batch));
    CHECK_CUFFT(cufftPlanMany(&plan_c2r, 2, n, NULL, 1, nx * (ny / 2 + 1), NULL,
                              1, nx * ny, CUFFT_C2R, batch));

    // R2C

    // Batched - Create cuFFT plans for the image
    cufftHandle plan_r2c_image, plan_c2r_image;
    CHECK_CUFFT(cufftPlan1d(&plan_r2c_image, image_size, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&plan_c2r_image, image_size, CUFFT_C2R, 1));

    // Create cuFFT plans for the kernel
    cufftHandle plan_r2c_kernel, plan_c2r_kernel;
    CHECK_CUFFT(cufftPlan1d(&plan_r2c_kernel, kernel_size, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&plan_c2r_kernel, kernel_size, CUFFT_C2R, 1));

    // Point wise Multiplication
    // dim3 blockSize(16, 16);
    // dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // printKernelConfig(gridSize, blockSize);
    //  fconvolution_2d_naive <<<gridSize, blockSize>>>(d_in.get(), d_out.get(), width, height, channels, d_kernel.get(), kernel_len);

    // C2R

    // Deinterleave

    // Copy back to Host
    //  CUDA_CALL(cudaMemcpy(h_out, d_out.get(), img_size, cudaMemcpyDeviceToHost));
    //  std::transform(h_out, h_out + img_pixel_count, h_out_char, float_to_uchar);
    //  stbi_write_jpg("fconv_naive.jpg", width, height, channels, h_out_char, 100);

    // Perform GT Conv CPU version
    // convolution_2d_cpu(h_in, h_out_cpu, width, height, channels, kernel, kernel_len);
    // std::transform(h_out_cpu, h_out_cpu + img_pixel_count, h_out_char, float_to_uchar);
    // stbi_write_jpg("conv_cpu.jpg", width, height, channels, h_out_char, 100);

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