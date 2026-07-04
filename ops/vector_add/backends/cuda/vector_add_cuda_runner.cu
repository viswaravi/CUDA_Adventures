// CUDA vector_add backend runner.
//
// This executable owns the vector_add CUDA CLI contract and implementation.
// Shared op-level parsing lives in ops/vector_add/common/vector_add_config.hpp;
// generic cross-op CLI/dtype utilities live in utils/.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cli_args.hpp"
#include "experiment_types.hpp"
#include "utils.cuh"
#include "vector_add_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define MAX_BLOCK_DIM 1024

// Vector-add is the operation-specific math policy. The CUDA backend currently
// supports int32 only; keep this local so dtype expansion is explicit.
template <typename T> struct AddOp {
  __device__ static T apply(T a, T b) { return a + b; }
};

// Stable externally named kernels; dtype specialization happens through T.
template <typename T>
__global__ void vectorAddScalarKernel(const T *a, const T *b, T *c,
                                      std::uint64_t n) {
  std::uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = AddOp<T>::apply(a[idx], b[idx]);
  }
}

// Vectorized loads use a 16-byte pack. For int32, the default VecWidth is 4.
template <typename T, int VecWidth> struct alignas(16) VectorPack {
  T values[VecWidth];
};

template <typename T, int VecWidth>
__global__ void vectorAddVectorizedKernel(const T *a, const T *b, T *c,
                                          std::uint64_t n) {
  using Pack = VectorPack<T, VecWidth>;
  std::uint64_t pack_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  std::uint64_t base = pack_idx * VecWidth;

  if (base + VecWidth <= n) {
    Pack a_pack = reinterpret_cast<const Pack *>(a)[pack_idx];
    Pack b_pack = reinterpret_cast<const Pack *>(b)[pack_idx];
    Pack c_pack;
    for (int i = 0; i < VecWidth; ++i) {
      c_pack.values[i] = AddOp<T>::apply(a_pack.values[i], b_pack.values[i]);
    }
    reinterpret_cast<Pack *>(c)[pack_idx] = c_pack;
    return;
  }

  for (int i = 0; i < VecWidth && base + i < n; ++i) {
    c[base + i] = AddOp<T>::apply(a[base + i], b[base + i]);
  }
}

template <typename T> void initializeInputs(T *a, T *b, T *c, std::uint64_t n) {
  for (std::uint64_t i = 0; i < n; ++i) {
    a[i] = static_cast<T>(1);
    b[i] = static_cast<T>(2);
    c[i] = static_cast<T>(0);
  }
}

// Validate a few sentinel indices. Integer vector-add should be exact.
template <typename T>
bool validateResult(const T *c, std::uint64_t n, DType dtype) {
  if (n == 0) {
    return true;
  }
  Tolerance tolerance = toleranceFor(dtype);
  double expected = 3.0;
  const std::uint64_t indices[] = {0, n / 2, n - 1};
  for (std::uint64_t idx : indices) {
    double actual = static_cast<double>(c[idx]);
    double diff = std::abs(actual - expected);
    double limit = tolerance.abs + tolerance.rel * std::abs(expected);
    if (diff > limit) {
      std::cerr << "Validation failed at index " << idx << ": expected "
                << expected << ", got " << actual << ", tolerance " << limit
                << std::endl;
      return false;
    }
  }
  return true;
}

// Compile-time vector widths keep the vectorized kernel simple while the
// runtime config still controls which specialization is launched.
template <typename T, int VecWidth>
void launchVectorized(const T *dev_a, const T *dev_b, T *dev_c,
                      std::uint64_t n, cudaStream_t stream = 0) {
  dim3 blockDim(MAX_BLOCK_DIM);
  std::uint64_t packs = (n + VecWidth - 1) / VecWidth;
  dim3 gridDim((packs + blockDim.x - 1) / blockDim.x);
  vectorAddVectorizedKernel<T, VecWidth>
      <<<gridDim, blockDim, 0, stream>>>(dev_a, dev_b, dev_c, n);
}

// Kernel dispatch is separate from execution-mode dispatch so all memory paths
// share identical scalar/vectorized launch behavior.
template <typename T>
void launchVectorAddKernel(const VectorAddConfig &cfg, const T *dev_a,
                           const T *dev_b, T *dev_c, std::uint64_t n,
                           cudaStream_t stream = 0) {
  if (cfg.kernel == "scalar") {
    dim3 blockDim(MAX_BLOCK_DIM);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    vectorAddScalarKernel<T><<<gridDim, blockDim, 0, stream>>>(dev_a, dev_b,
                                                               dev_c, n);
    return;
  }

  if (cfg.kernel == "vectorized") {
    int width =
        cfg.vector_width > 0 ? cfg.vector_width : defaultVectorWidth(cfg.dtype);
    if (static_cast<int>(sizeof(T)) * width != 16) {
      throw std::runtime_error(
          "--vector-width must select a 16-byte vector for the chosen dtype");
    }
    if (width == 4) {
      launchVectorized<T, 4>(dev_a, dev_b, dev_c, n, stream);
      return;
    }
    if (width == 8) {
      launchVectorized<T, 8>(dev_a, dev_b, dev_c, n, stream);
      return;
    }
    if (width == 16) {
      launchVectorized<T, 16>(dev_a, dev_b, dev_c, n, stream);
      return;
    }
    throw std::runtime_error("Unsupported --vector-width: " +
                             std::to_string(width));
  }

  throw std::runtime_error("Unsupported --kernel for vector_add cuda: " +
                           cfg.kernel);
}

// Single-stream path: copy the whole problem once, run warmup/repeat launches,
// then copy the full result back for validation.
template <typename T>
Status runSingleStreamTyped(const VectorAddConfig &cfg, bool pinned) {
  T *h_a = nullptr;
  T *h_b = nullptr;
  T *h_c = nullptr;
  size_t bytes = cfg.n * sizeof(T);

  if (pinned) {
    CUDA_CALL(cudaMallocHost(&h_a, bytes));
    CUDA_CALL(cudaMallocHost(&h_b, bytes));
    CUDA_CALL(cudaMallocHost(&h_c, bytes));
  } else {
    h_a = static_cast<T *>(malloc(bytes));
    h_b = static_cast<T *>(malloc(bytes));
    h_c = static_cast<T *>(malloc(bytes));
  }
  initializeInputs(h_a, h_b, h_c, cfg.n);

  CudaMemory<T> dev_a(bytes), dev_b(bytes), dev_c(bytes);
  CUDA_CALL(cudaMemcpy(dev_a.get(), h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_b.get(), h_b, bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < cfg.warmup; ++i) {
    launchVectorAddKernel(cfg, dev_a.get(), dev_b.get(), dev_c.get(), cfg.n);
  }
  for (int i = 0; i < cfg.repeats; ++i) {
    launchVectorAddKernel(cfg, dev_a.get(), dev_b.get(), dev_c.get(), cfg.n);
  }
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(h_c, dev_c.get(), bytes, cudaMemcpyDeviceToHost));

  bool valid = !cfg.validate || validateResult(h_c, cfg.n, cfg.dtype);
  std::cout << "Result: " << static_cast<double>(h_c[0]) << " "
            << static_cast<double>(h_c[cfg.n - 1]) << std::endl;

  if (pinned) {
    CUDA_CALL(cudaFreeHost(h_a));
    CUDA_CALL(cudaFreeHost(h_b));
    CUDA_CALL(cudaFreeHost(h_c));
  } else {
    free(h_a);
    free(h_b);
    free(h_c);
  }
  return valid ? Status::Ok : Status::ValidationFailed;
}

// Chunked pageable path: reuse one device chunk when the full problem should
// not be resident on the GPU at once.
template <typename T>
Status runChunkedTyped(const VectorAddConfig &cfg) {
  T *h_a = nullptr;
  T *h_b = nullptr;
  T *h_c = nullptr;
  size_t bytes = cfg.n * sizeof(T);
  h_a = static_cast<T *>(malloc(bytes));
  h_b = static_cast<T *>(malloc(bytes));
  h_c = static_cast<T *>(malloc(bytes));
  initializeInputs(h_a, h_b, h_c, cfg.n);

  size_t free_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, nullptr));
  size_t total_mem = static_cast<size_t>(free_mem * cfg.free_mem_threshold);
  size_t chunk_max_len = (total_mem / 3) / sizeof(T);
  std::uint64_t chunk_len = std::max<std::uint64_t>(
      MAX_BLOCK_DIM, chunk_max_len - (chunk_max_len % MAX_BLOCK_DIM));
  size_t chunk_bytes = chunk_len * sizeof(T);

  CudaMemory<T> dev_a(chunk_bytes), dev_b(chunk_bytes), dev_c(chunk_bytes);
  for (std::uint64_t offset = 0; offset < cfg.n; offset += chunk_len) {
    std::uint64_t current_len =
        std::min<std::uint64_t>(chunk_len, cfg.n - offset);
    size_t current_bytes = current_len * sizeof(T);
    CUDA_CALL(cudaMemcpy(dev_a.get(), h_a + offset, current_bytes,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_b.get(), h_b + offset, current_bytes,
                         cudaMemcpyHostToDevice));
    launchVectorAddKernel(cfg, dev_a.get(), dev_b.get(), dev_c.get(),
                          current_len);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(h_c + offset, dev_c.get(), current_bytes,
                         cudaMemcpyDeviceToHost));
  }

  bool valid = !cfg.validate || validateResult(h_c, cfg.n, cfg.dtype);
  std::cout << "Result: " << static_cast<double>(h_c[0]) << " "
            << static_cast<double>(h_c[cfg.n - 1]) << std::endl;
  free(h_a);
  free(h_b);
  free(h_c);
  return valid ? Status::Ok : Status::ValidationFailed;
}

// Pinned multistream path: divide the chunk budget across streams and overlap
// H2D copy, kernel launch, and D2H copy for consecutive chunks.
template <typename T>
Status runChunkedMultiStreamTyped(const VectorAddConfig &cfg) {
  T *h_a = nullptr;
  T *h_b = nullptr;
  T *h_c = nullptr;
  size_t bytes = cfg.n * sizeof(T);
  CUDA_CALL(cudaMallocHost(&h_a, bytes));
  CUDA_CALL(cudaMallocHost(&h_b, bytes));
  CUDA_CALL(cudaMallocHost(&h_c, bytes));
  initializeInputs(h_a, h_b, h_c, cfg.n);

  size_t free_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, nullptr));
  size_t total_mem = static_cast<size_t>(free_mem * cfg.free_mem_threshold);
  size_t chunk_max_len = (total_mem / (3 * cfg.streams)) / sizeof(T);
  std::uint64_t chunk_len = std::max<std::uint64_t>(
      MAX_BLOCK_DIM, chunk_max_len - (chunk_max_len % MAX_BLOCK_DIM));
  size_t chunk_bytes = chunk_len * sizeof(T);

  std::vector<cudaStream_t> streams(cfg.streams);
  std::vector<CudaMemory<T>> dev_a, dev_b, dev_c;
  dev_a.reserve(cfg.streams);
  dev_b.reserve(cfg.streams);
  dev_c.reserve(cfg.streams);
  for (int i = 0; i < cfg.streams; ++i) {
    CUDA_CALL(cudaStreamCreate(&streams[i]));
    dev_a.emplace_back(chunk_bytes);
    dev_b.emplace_back(chunk_bytes);
    dev_c.emplace_back(chunk_bytes);
  }

  for (std::uint64_t chunk_start = 0; chunk_start < cfg.n;
       chunk_start += chunk_len * cfg.streams) {
    for (int stream_index = 0; stream_index < cfg.streams; ++stream_index) {
      std::uint64_t offset = chunk_start + stream_index * chunk_len;
      if (offset >= cfg.n) {
        break;
      }
      std::uint64_t current_len =
          std::min<std::uint64_t>(chunk_len, cfg.n - offset);
      size_t current_bytes = current_len * sizeof(T);
      CUDA_CALL(cudaMemcpyAsync(dev_a[stream_index].get(), h_a + offset,
                                current_bytes, cudaMemcpyHostToDevice,
                                streams[stream_index]));
      CUDA_CALL(cudaMemcpyAsync(dev_b[stream_index].get(), h_b + offset,
                                current_bytes, cudaMemcpyHostToDevice,
                                streams[stream_index]));
      launchVectorAddKernel(cfg, dev_a[stream_index].get(),
                            dev_b[stream_index].get(),
                            dev_c[stream_index].get(), current_len,
                            streams[stream_index]);
      CUDA_CALL(cudaMemcpyAsync(h_c + offset, dev_c[stream_index].get(),
                                current_bytes, cudaMemcpyDeviceToHost,
                                streams[stream_index]));
    }
  }

  for (cudaStream_t stream : streams) {
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaStreamDestroy(stream));
  }
  CUDA_CALL(cudaGetLastError());

  bool valid = !cfg.validate || validateResult(h_c, cfg.n, cfg.dtype);
  std::cout << "Result: " << static_cast<double>(h_c[0]) << " "
            << static_cast<double>(h_c[cfg.n - 1]) << std::endl;
  CUDA_CALL(cudaFreeHost(h_a));
  CUDA_CALL(cudaFreeHost(h_b));
  CUDA_CALL(cudaFreeHost(h_c));
  return valid ? Status::Ok : Status::ValidationFailed;
}

// Execution-mode dispatch is dtype-independent after runCudaVectorAdd chooses T.
template <typename T> Status runCudaVectorAddTyped(const VectorAddConfig &cfg) {
  if (cfg.execution_mode == "single_stream" && cfg.memory_host == "pageable") {
    return runSingleStreamTyped<T>(cfg, false);
  }
  if (cfg.execution_mode == "single_stream" && cfg.memory_host == "pinned") {
    return runSingleStreamTyped<T>(cfg, true);
  }
  if (cfg.execution_mode == "chunked" && cfg.memory_host == "pageable") {
    return runChunkedTyped<T>(cfg);
  }
  if (cfg.execution_mode == "chunked_multistream" && cfg.memory_host == "pinned") {
    return runChunkedMultiStreamTyped<T>(cfg);
  }
  throw std::runtime_error("Unsupported CUDA vector_add combination: memory.host=" +
                           cfg.memory_host + ", execution.mode=" + cfg.execution_mode);
}

// Public CUDA vector-add entrypoint used by the runner CLI.
Status runCudaVectorAdd(const VectorAddConfig &cfg) {
  switch (cfg.dtype) {
  case DType::Int32:
    return runCudaVectorAddTyped<int>(cfg);
  case DType::Float32:
  case DType::Float16:
  case DType::BFloat16:
    return Status::UnsupportedDType;
  default:
    return Status::UnsupportedDType;
  }
}

namespace {

constexpr const char *kDescription =
    "CUDA vector_add backend with  ernel, dtype, memory, and "
    "execution args.";

} // namespace

int main(int argc, char **argv) {
  const std::vector<ArgSpec> args = vectorAddArgs("scalar");

  try {
    RunConfig run_config = parse_args(argc, argv, args);
    if (run_config.print_help) {
      print_usage(argv[0], args, kDescription);
      return EXIT_SUCCESS;
    }

    CUDA_CALL(cudaSetDevice(run_config.device));
    printDeviceDetails();

    VectorAddConfig cfg = vectorAddConfigFromRunConfig(run_config);
    std::cout << "vector_add cuda dtype=" << dtypeName(cfg.dtype)
              << " kernel=" << cfg.kernel
              << " memory.host=" << cfg.memory_host
              << " execution.mode=" << cfg.execution_mode << std::endl;

    Status status = runCudaVectorAdd(cfg);
    if (status != Status::Ok) {
      throw std::runtime_error("vector_add cuda failed");
    }

    if (run_config.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    print_usage(argv[0], args, kDescription);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
