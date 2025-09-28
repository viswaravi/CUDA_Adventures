#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#define CHECK_CUDA(call)                                                   \
  {                                                                        \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      exit(1);                                                             \
    }                                                                      \
  }
  
#define CHECK_CUFFT(call)                               \
  {                                                     \
    cufftResult err = call;                             \
    if (err != CUFFT_SUCCESS) {                         \
      std::cerr << "cuFFT error: " << err << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

__global__ void complexPointwiseMul(cufftComplex* a, cufftComplex* b,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;  // Real part
    float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;  // Imaginary part
    a[idx].x = real;
    a[idx].y = imag;
  }
}

int main() {
  // Input and kernel
  float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};  // Input signal
  float h_h[] = {1.0f, -1.0f};             // Kernel
  int N = 4;                               // Input length
  int M = 2;                               // Kernel length
  int padded_size = 8;                     // Next power of 2 >= N + M - 1

  // Allocate host padded arrays
  float* h_x_padded = new float[padded_size]();
  float* h_h_padded = new float[padded_size]();
  for (int i = 0; i < N; i++)
    h_x_padded[i] = h_x[i];
  for (int i = 0; i < M; i++)
    h_h_padded[i] = h_h[i];

  // Allocate device memory
  float *d_x, *d_h;
  cufftComplex *d_X, *d_H;
  CHECK_CUDA(cudaMalloc(&d_x, padded_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_h, padded_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_X, padded_size * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMalloc(&d_H, padded_size * sizeof(cufftComplex)));

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(d_x, h_x_padded, padded_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_h, h_h_padded, padded_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Create cuFFT plans
  cufftHandle plan_r2c, plan_c2r;
  CHECK_CUFFT(cufftPlan1d(&plan_r2c, padded_size, CUFFT_R2C, 1));
  CHECK_CUFFT(cufftPlan1d(&plan_c2r, padded_size, CUFFT_C2R, 1));

  // Execute FFT (real to complex)
  CHECK_CUFFT(cufftExecR2C(plan_r2c, d_x, d_X));
  CHECK_CUFFT(cufftExecR2C(plan_r2c, d_h, d_H));

  // Pointwise multiplication
  int threadsPerBlock = 256;
  int blocks = (padded_size + threadsPerBlock - 1) / threadsPerBlock;
  complexPointwiseMul<<<blocks, threadsPerBlock>>>(d_X, d_H, padded_size);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Inverse FFT (complex to real)
  float* d_y = d_x;  // Reuse d_x for output
  CHECK_CUFFT(cufftExecC2R(plan_c2r, d_X, d_y));

  // Copy result back
  float* h_y = new float[padded_size];
  CHECK_CUDA(cudaMemcpy(h_y, d_y, padded_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Normalize (cuFFT doesn't scale by 1/N)
  int out_size = N + M - 1;
  for (int i = 0; i < out_size; i++) {
    h_y[i] /= padded_size;
    std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
  }

  // Cleanup
  CHECK_CUFFT(cufftDestroy(plan_r2c));
  CHECK_CUFFT(cufftDestroy(plan_c2r));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_h));
  CHECK_CUDA(cudaFree(d_X));
  CHECK_CUDA(cudaFree(d_H));
  delete[] h_x_padded;
  delete[] h_h_padded;
  delete[] h_y;

  return 0;
}