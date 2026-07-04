#include <cuda_runtime.h>
#include <iostream>
#include "utils.cuh"

// Kernel to double each element in the 2D array
__global__ void doubleValues(float* data, size_t pitch, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Access using pitch (pitch is in bytes, so convert to float pointer offset)
    float* row = (float*)((char*)data + y * pitch);
    row[x] = row[x] * 2.0f;
  }
}

int main() {
  const int WIDTH = 5;   // Number of columns
  const int HEIGHT = 4;  // Number of rows
  const size_t WIDTH_BYTES = WIDTH * sizeof(float);

  // Host array
  float h_data[HEIGHT][WIDTH] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                 {6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                                 {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
                                 {16.0f, 17.0f, 18.0f, 19.0f, 20.0f}};

  // Device pointer and pitch
  float* d_data;
  size_t pitch;

  // Allocate pitched memory on GPU
  CHECK_CUDA_ERROR(cudaMallocPitch(&d_data, &pitch, WIDTH_BYTES, HEIGHT));
  std::cout << "Pitch: " << pitch
            << " bytes (vs requested width: " << WIDTH_BYTES << " bytes)"
            << std::endl;

  // Copy data from host to device
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_data, pitch, h_data, WIDTH_BYTES, WIDTH_BYTES,
                                HEIGHT, cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
  doubleValues<<<grid, block>>>(d_data, pitch, WIDTH, HEIGHT);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy result back to host
  float h_result[HEIGHT][WIDTH];
  CHECK_CUDA_ERROR(cudaMemcpy2D(h_result, WIDTH_BYTES, d_data, pitch,
                                WIDTH_BYTES, HEIGHT, cudaMemcpyDeviceToHost));

  // Print result
  std::cout << "Result after doubling:\n";
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      std::cout << h_result[y][x] << " ";
    }
    std::cout << "\n";
  }

  // Free device memory
  CHECK_CUDA_ERROR(cudaFree(d_data));

  return 0;
}