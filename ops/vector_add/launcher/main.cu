#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cli_args.hpp"
#include "utils.cuh"
#include "kernels.cuh"
#include <algorithm>
#include <assert.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

void printMemoryRequirements(unsigned long long array_len,
                             unsigned long long chunk_len = 0) {
  long double array_size_bytes = array_len * sizeof(int);
  long double array_size_gbytes = array_size_bytes / (1024 * 1024 * 1024);
  long double array_size_mbytes = array_size_bytes / (1024 * 1024);

  std::cout << "---Memory Requirements---" << std::endl;
  std::cout << "Array Length: " << array_len << std::endl;
  std::cout << "Host Array Memory Size: " << array_size_gbytes << "GB, "
            << array_size_mbytes << "MB" << std::endl;

  if (chunk_len > 0) {
    long double chunk_size_bytes = chunk_len * sizeof(int);
    long double chunk_size_gbytes = chunk_size_bytes / (1024 * 1024 * 1024);
    long double chunk_size_mbytes = chunk_size_bytes / (1024 * 1024);

    std::cout << "Chunk Length: " << chunk_len << std::endl;
    std::cout << "GPU  Chunk Per Array Memory: " << chunk_size_gbytes << " GB"
              << std::endl;
    std::cout << "GPU  Chunk Total Array Memory: " << chunk_size_gbytes * 3
              << "GB, " << chunk_size_mbytes * 3 << "MB" << std::endl
              << std::endl;
  }
}

void addWithCuda(int *c, const int *a, const int *b,
                 unsigned long long array_len) {
  int *dev_a = 0;
  int *dev_b = 0;
  int *dev_c = 0;

  int mem_size = array_len * sizeof(int);
  CUDA_CALL(cudaMalloc((void **)&dev_c, mem_size));
  CUDA_CALL(cudaMalloc((void **)&dev_a, mem_size));
  CUDA_CALL(cudaMalloc((void **)&dev_b, mem_size));

  CUDA_CALL(cudaMemcpy(dev_a, a, mem_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_b, b, mem_size, cudaMemcpyHostToDevice));

  int blockDim = MAX_BLOCK_DIM;
  int gridDim = (array_len + blockDim - 1) / blockDim;
  addKernel<<<gridDim, blockDim>>>(dev_c, dev_a, dev_b, array_len);

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(c, dev_c, mem_size, cudaMemcpyDeviceToHost));

  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);
}

void addWithCudaMemSafe(int *c, const int *a, const int *b,
                        unsigned long long array_len,
                        bool use_vectorized_kernel = false) {
  size_t mem_size = array_len * sizeof(int);
  CudaMemory<int> dev_a(mem_size), dev_b(mem_size), dev_c(mem_size);

  CUDA_CALL(cudaMemcpy(dev_a.get(), a, mem_size, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_b.get(), b, mem_size, cudaMemcpyHostToDevice));

  if (use_vectorized_kernel) {
    if (array_len % 4 != 0) {
      throw std::runtime_error("Array length must be a multiple of 4 for vectorized kernel");
    }
    dim3 blockDim(MAX_BLOCK_DIM);
    dim3 gridDim(((array_len / 4) + blockDim.x - 1) / blockDim.x);
    addKernelVectorized<<<gridDim, blockDim>>>(dev_c.get(), dev_a.get(), dev_b.get(), array_len);
  } else {
    dim3 blockDim(MAX_BLOCK_DIM);
    dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
    addKernel<<<gridDim, blockDim>>>(dev_c.get(), dev_a.get(), dev_b.get(), array_len);
  }

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(c, dev_c.get(), mem_size, cudaMemcpyDeviceToHost));
}

void pageableMemoryAddition(unsigned long long array_len, bool use_vectorized_kernel = false) {
  int *h_A, *h_B, *h_C;
  size_t mem_size = array_len * sizeof(int);

  h_A = (int *)malloc(mem_size);
  h_B = (int *)malloc(mem_size);
  h_C = (int *)malloc(mem_size);

  std::fill(h_A, h_A + array_len, 42);
  std::fill(h_B, h_B + array_len, 38);
  std::fill(h_C, h_C + array_len, 0);

  addWithCudaMemSafe(h_C, h_A, h_B, array_len, use_vectorized_kernel);

  std::cout << "Result: " << h_C[0] << " " << h_C[array_len - 1] << std::endl;

  free(h_A);
  free(h_B);
  free(h_C);
}

void pinnedMemoryAddition(unsigned long long array_len, bool use_vectorized_kernel = false) {
  int *h_A, *h_B, *h_C;
  size_t mem_size = array_len * sizeof(int);

  CUDA_CALL(cudaMallocHost(&h_A, mem_size));
  CUDA_CALL(cudaMallocHost(&h_B, mem_size));
  CUDA_CALL(cudaMallocHost(&h_C, mem_size));

  std::fill(h_A, h_A + array_len, 42);
  std::fill(h_B, h_B + array_len, 38);
  std::fill(h_C, h_C + array_len, 0);

  std::cout << "Initialized A:" << h_A[0] << "  B:" << h_B[0] << std::endl;

  addWithCudaMemSafe(h_C, h_A, h_B, array_len, use_vectorized_kernel);

  std::cout << "Result: " << h_C[0] << " " << h_C[array_len - 1] << std::endl;

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
}

void pageableMemoryAdditionLarge(unsigned long long array_len,
                                 float free_mem_threshold = 0.2,
                                 bool use_vectorized_kernel = false) {
  std::srand(std::time(nullptr));

  size_t free_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, nullptr));

  size_t total_mem = free_mem * free_mem_threshold;
  size_t vec_max_mem = total_mem / 3;
  size_t vec_max_len = vec_max_mem / sizeof(int);

  unsigned long long chunk_len = vec_max_len - (vec_max_len % MAX_BLOCK_DIM);
  size_t chunk_num = array_len / chunk_len;
  unsigned long long final_chunk_len = array_len - (chunk_num * chunk_len);
  assert(((chunk_num * chunk_len) + final_chunk_len) == array_len);
  assert(chunk_len >= MAX_BLOCK_DIM);

  std::cout << "Number of Chunks: " << chunk_num << std::endl;

  int *h_A, *h_B, *h_C;
  size_t mem_size = array_len * sizeof(int);

  h_A = (int *)malloc(mem_size);
  h_B = (int *)malloc(mem_size);
  h_C = (int *)malloc(mem_size);

  std::fill(h_A, h_A + array_len, 42);
  std::fill(h_B, h_B + array_len, 100);
  std::fill(h_C, h_C + array_len, 0);

  std::cout << "Host Initialized -> Launching Kernels" << std::endl;

  size_t chunk_mem_size = chunk_len * sizeof(int);
  CudaMemory<int> dev_a(chunk_mem_size), dev_b(chunk_mem_size), dev_c(chunk_mem_size);

  unsigned long long chunk_offset = 0;
  for (int chunk_index = 0; chunk_index < (int)chunk_num; chunk_index++) {
    chunk_offset = (chunk_index * chunk_len);
    CUDA_CALL(cudaMemcpy(dev_a.get(), h_A + chunk_offset, chunk_mem_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_b.get(), h_B + chunk_offset, chunk_mem_size, cudaMemcpyHostToDevice));
    dim3 blockDim(MAX_BLOCK_DIM);
    dim3 gridDim((chunk_len + blockDim.x - 1) / blockDim.x);
    addKernel<<<gridDim, blockDim>>>(dev_c.get(), dev_a.get(), dev_b.get(), chunk_len);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(h_C + chunk_offset, dev_c.get(), chunk_mem_size, cudaMemcpyDeviceToHost));
  }

  if (final_chunk_len > 0) {
    chunk_offset = chunk_num * chunk_len;
    addWithCudaMemSafe(h_C + chunk_offset, h_A + chunk_offset, h_B + chunk_offset,
                       final_chunk_len, use_vectorized_kernel);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void streamedVectorAdditionLarge(unsigned long long array_len,
                                 float free_mem_threshold = 0.2,
                                 int num_streams = 4,
                                 bool use_vectorized_kernel = false) {
  assert(free_mem_threshold <= 1);
  assert(num_streams > 0);

  int *h_A, *h_B, *h_C;
  size_t mem_size = array_len * sizeof(int);

  CUDA_CALL(cudaMallocHost(&h_A, mem_size));
  CUDA_CALL(cudaMallocHost(&h_B, mem_size));
  CUDA_CALL(cudaMallocHost(&h_C, mem_size));

  std::fill(h_A, h_A + array_len, 42);
  std::fill(h_B, h_B + array_len, 100);
  std::fill(h_C, h_C + array_len, 0);

  size_t free_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, nullptr));

  size_t total_mem = free_mem * free_mem_threshold;
  int num_vecs_in_device = 3 * num_streams;
  size_t chunk_max_mem = total_mem / num_vecs_in_device;
  size_t chunk_max_len = chunk_max_mem / sizeof(int);

  unsigned long long chunk_len = chunk_max_len - (chunk_max_len % MAX_BLOCK_DIM);
  assert(chunk_len >= MAX_BLOCK_DIM);
  size_t chunk_mem_size = chunk_len * sizeof(int);

  size_t chunk_num = array_len / chunk_len;
  std::cout << "Number of Chunks: " << chunk_num << std::endl;
  std::cout << "Chunk Length: " << chunk_len << std::endl;
  std::cout << "Host Initialized -> Launching Kernels" << std::endl;

  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }

  std::vector<CudaMemory<int>> dev_a, dev_b, dev_c;
  dev_a.reserve(num_streams);
  dev_b.reserve(num_streams);
  dev_c.reserve(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    dev_a.emplace_back(chunk_mem_size);
    dev_b.emplace_back(chunk_mem_size);
    dev_c.emplace_back(chunk_mem_size);
  }

  for (unsigned long long chunk_start = 0; chunk_start < array_len;
       chunk_start += (chunk_len * num_streams)) {
    for (int stream_index = 0; stream_index < num_streams; stream_index++) {
      unsigned long long chunk_offset = chunk_start + (stream_index * chunk_len);
      if (chunk_offset >= array_len) {
        break;
      }

      unsigned long long current_chunk_len = std::min(chunk_len, array_len - chunk_offset);
      size_t current_chunk_mem_size = current_chunk_len * sizeof(int);

      CUDA_CALL(cudaMemcpyAsync(dev_a[stream_index].get(), h_A + chunk_offset,
                                current_chunk_mem_size, cudaMemcpyHostToDevice,
                                streams[stream_index]));
      CUDA_CALL(cudaMemcpyAsync(dev_b[stream_index].get(), h_B + chunk_offset,
                                current_chunk_mem_size, cudaMemcpyHostToDevice,
                                streams[stream_index]));

      dim3 blockDim(MAX_BLOCK_DIM);
      dim3 gridDim((current_chunk_len + blockDim.x - 1) / blockDim.x);

      addKernel<<<gridDim, blockDim, 0, streams[stream_index]>>>(
          dev_c[stream_index].get(), dev_a[stream_index].get(),
          dev_b[stream_index].get(), current_chunk_len);
      CUDA_CALL(cudaGetLastError());

      CUDA_CALL(cudaMemcpyAsync(h_C + chunk_offset, dev_c[stream_index].get(),
                                current_chunk_mem_size, cudaMemcpyDeviceToHost,
                                streams[stream_index]));
    }
  }

  for (int i = 0; i < num_streams; i++) {
    CUDA_CALL(cudaStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < num_streams; i++) {
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }

  std::cout << "Result: " << h_C[0] << " " << h_C[array_len - 1] << std::endl;

  CUDA_CALL(cudaFreeHost(h_A));
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C));
}

int main(int argc, char **argv) {
  const std::vector<ArgSpec> pageable_args = {
      {"--n", "uint64", "268435456", "Array length (elements)"},
      {"--vectorized", "bool", "false", "Use vectorized addition kernel"},
  };
  const std::vector<ArgSpec> pinned_args = {
      {"--n", "uint64", "268435456", "Array length (elements)"},
      {"--vectorized", "bool", "false", "Use vectorized addition kernel"},
  };
  const std::vector<ArgSpec> chunked_args = {
      {"--n", "uint64", "536870912", "Array length (elements)"},
      {"--free-mem-threshold", "float", "0.2",
       "Fraction of free GPU memory to use per chunk (0, 1]"},
      {"--vectorized", "bool", "false", "Use vectorized addition kernel"},
  };
  const std::vector<ArgSpec> streamed_args = {
      {"--n", "uint64", "536870912", "Array length (elements)"},
      {"--free-mem-threshold", "float", "0.2",
       "Fraction of free GPU memory to use per chunk (0, 1]"},
      {"--streams", "int", "4", "Number of concurrent CUDA streams"},
      {"--vectorized", "bool", "false", "Use vectorized addition kernel"},
  };

  const VariantRegistry variants = {
      {
          "pageable",
          "Pageable host memory, no chunking",
          pageable_args,
          [](const RunConfig &c) {
            unsigned long long n = get_ull(c, "--n", 1024ULL * 1024 * 256);
            bool use_vectorized = get_bool(c, "--vectorized", false);
            printMemoryRequirements(n);
            pageableMemoryAddition(n, use_vectorized);
          },
      },
      {
          "pinned",
          "Pinned host memory, no chunking",
          pinned_args,
          [](const RunConfig &c) {
            unsigned long long n = get_ull(c, "--n", 1024ULL * 1024 * 256);
            bool use_vectorized = get_bool(c, "--vectorized", false);
            printMemoryRequirements(n);
            pinnedMemoryAddition(n, use_vectorized);
          },
      },
      {
          "pageable-large",
          "Pageable memory with chunked GPU transfers",
          chunked_args,
          [](const RunConfig &c) {
            unsigned long long n = get_ull(c, "--n", 1ULL * 1024 * 1024 * 512);
            bool use_vectorized = get_bool(c, "--vectorized", false);
            printMemoryRequirements(n);
            pageableMemoryAdditionLarge(n, get_float(c, "--free-mem-threshold", 0.2f),
                                        use_vectorized);
          },
      },
      {
          "pinned-large",
          "Pinned memory with multi-stream async transfers",
          streamed_args,
          [](const RunConfig &c) {
            unsigned long long n = get_ull(c, "--n", 1ULL * 1024 * 1024 * 512);
            bool use_vectorized = get_bool(c, "--vectorized", false);
            printMemoryRequirements(n);
            streamedVectorAdditionLarge(n, get_float(c, "--free-mem-threshold", 0.2f),
                                        get_int(c, "--streams", 4), use_vectorized);
          },
      },
  };

  try {
    RunConfig cfg = parse_args(argc, argv, variants);
    if (cfg.print_help) {
      print_usage(argv[0], variants);
      return EXIT_SUCCESS;
    }
    if (cfg.list_variants) {
      print_variants(variants);
      return EXIT_SUCCESS;
    }

    CUDA_CALL(cudaSetDevice(cfg.device));
    printDeviceDetails();
    run_variant(cfg, variants);

    if (cfg.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    print_usage(argv[0], variants);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
