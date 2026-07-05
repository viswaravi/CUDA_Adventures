#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>

#define CUDA_CALL(func) ::kernel_lab::checkCuda((func), #func, __FILE__, __LINE__)

namespace kernel_lab {

void checkCuda(cudaError_t result, const char *func, const char *file,
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

template <>
class CudaMemory<void>
{
public:
  explicit CudaMemory(size_t size)
  {
    ptr = nullptr;
    CUDA_CALL(cudaMalloc(&ptr, size));
  }

  CudaMemory(const CudaMemory &) = delete;
  CudaMemory &operator=(const CudaMemory &) = delete;

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

  void *get() const { return ptr; }

private:
  void *ptr;
};

class HostMemory
{
public:
  HostMemory(size_t bytes, bool pinned) : ptr(nullptr), pinned(pinned)
  {
    if (pinned)
    {
      CUDA_CALL(cudaMallocHost(&ptr, bytes));
      return;
    }
    ptr = std::malloc(bytes);
    if (ptr == nullptr)
    {
      throw std::runtime_error("malloc failed");
    }
  }

  HostMemory(const HostMemory &) = delete;
  HostMemory &operator=(const HostMemory &) = delete;
  HostMemory(HostMemory &&other) = delete;
  HostMemory &operator=(HostMemory &&other) = delete;

  ~HostMemory()
  {
    if (ptr == nullptr)
      return;
    if (pinned)
      cudaFreeHost(ptr);
    else
      std::free(ptr);
  }

  void *get() const { return ptr; }

private:
  void *ptr;
  bool pinned;
};

void printDeviceDetails();

void printKernelConfig(dim3 grid, dim3 block);

} // namespace kernel_lab
