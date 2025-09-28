#include <stdio.h>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

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

void printDeviceDetails()
{
  cudaDeviceProp prop;
  int deviceId = 0; // Use 0 for the first GPU

  size_t free_mem, total_mem;

  CUDA_CALL(cudaGetDeviceProperties(&prop, deviceId));
  CUDA_CALL(cudaMemGetInfo(&free_mem, nullptr));

  // Convert to MB
  total_mem = prop.totalGlobalMem / (1024 * 1024);
  free_mem = free_mem / (1024 * 1024);

  printf("Device Name: %s\n", prop.name);
  printf("SM Count: %d\n", prop.multiProcessorCount);
  printf("Total Global Memory: %zu Mbytes\n", total_mem);
  printf("Current Free Memory: %zu Mbytes\n", free_mem);

  printf("\n---Block Limits---\n");
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Block Dimension: %d %d %d\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max Grid Size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
         prop.maxGridSize[2]);

  printf("\n---Memory Limits per Block---\n");
  printf("Registers per Block: %d\n", prop.regsPerBlock);
  printf("Shared Memory per Block: %d\n", prop.sharedMemPerBlock);

  // printf("\n---Clock Rates---\n");
  // printf("Clock Rate: %d KHz\n", prop.clockRate);
  // printf("Memory Clock Rate: %d KHz\n", prop.memoryClockRate);
  // printf("\n");
}

void printKernelConfig(dim3 grid, dim3 block)
{
  std::cout << "Grid: " << grid.x << " " << grid.y << " " << grid.z
            << std::endl;
  std::cout << "Block: " << block.x << " " << block.y << " " << block.z
            << std::endl;
  std::cout << "Threads in X:" << grid.x * block.x << " Y:" << grid.y * block.y
            << " Z:" << grid.z * block.z << std::endl;
}