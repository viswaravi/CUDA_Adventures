#pragma once
#include "cuda_runtime.h"
#define CUDA_CALL(func) checkCuda((func), #func, __FILE__, __LINE__)

inline void checkCuda(cudaError_t result, const char *func, const char *file,
                      int line);

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

  ~CudaMemory()
  {
    if (ptr)
      cudaFree(ptr);
  }

  T *get() const { return ptr; }

private:
  T *ptr;
};

void printDeviceDetails();

void printKernelConfig(dim3 grid, dim3 block);
