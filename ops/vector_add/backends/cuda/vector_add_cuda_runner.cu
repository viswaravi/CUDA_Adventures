// CUDA vector_add backend runner.
//
// This executable owns the vector_add CUDA CLI contract and runtime dispatch.
// The actual CUDA kernels live in kernels.cu with explicit dtype-specific
// symbols so PTX/SASS output is easy to inspect.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernel_lab/utils/cli_args.hpp>
#include <kernel_lab/utils/cuda_dtype.cuh>
#include <kernel_lab/utils/experiment_types.hpp>
#include "kernels.cuh"
#include <kernel_lab/utils/cuda_utils.cuh>
#include <kernel_lab/ops/vector_add/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kernel_lab;
using namespace kernel_lab::ops::vector_add;

namespace {

constexpr const char *kDescription = "CUDA vector_add backend with explicit dtype kernels and execution args.";

// Fill host buffers with small values that are exactly representable for all
// supported dtypes. That keeps validation focused on kernel behavior.
void initializeInputs(void *a, void *b, void *c, std::uint64_t n, DType dtype) {
  switch (dtype) {
  case DType::Int32: {
    auto *aa = static_cast<int *>(a);
    
    auto *bb = static_cast<int *>(b);
    auto *cc = static_cast<int *>(c);
    for (std::uint64_t i = 0; i < n; ++i) {
      aa[i] = 1;
      bb[i] = 2;
      cc[i] = 0;
    }
    return;
  }
  case DType::Float32: {
    auto *aa = static_cast<float *>(a);
    auto *bb = static_cast<float *>(b);
    auto *cc = static_cast<float *>(c);
    for (std::uint64_t i = 0; i < n; ++i) {
      aa[i] = 1.0f;
      bb[i] = 2.0f;
      cc[i] = 0.0f;
    }
    return;
  }
  case DType::Float16: {
    auto *aa = static_cast<half *>(a);
    auto *bb = static_cast<half *>(b);
    auto *cc = static_cast<half *>(c);
    for (std::uint64_t i = 0; i < n; ++i) {
      aa[i] = __float2half(1.0f);
      bb[i] = __float2half(2.0f);
      cc[i] = __float2half(0.0f);
    }
    return;
  }
  case DType::BFloat16: {
    auto *aa = static_cast<__nv_bfloat16 *>(a);
    auto *bb = static_cast<__nv_bfloat16 *>(b);
    auto *cc = static_cast<__nv_bfloat16 *>(c);
    for (std::uint64_t i = 0; i < n; ++i) {
      aa[i] = __float2bfloat16(1.0f);
      bb[i] = __float2bfloat16(2.0f);
      cc[i] = __float2bfloat16(0.0f);
    }
    return;
  }
  }
  throw std::runtime_error("Unsupported dtype");
}

// Check sentinel positions instead of the full output buffer. These experiments
// focus on profiling; this catches common launch/copy bugs with low overhead.
bool validateResult(const void *c, std::uint64_t n, DType dtype) {
  if (n == 0) {
    return true;
  }
  Tolerance tolerance = toleranceFor(dtype);
  const double expected = 3.0;
  const std::uint64_t indices[] = {0, n / 2, n - 1};
  for (std::uint64_t idx : indices) {
    double actual = readDTypeValueAsDouble(c, idx, dtype);
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

dim3 gridForElements(std::uint64_t elements, int elements_per_thread) {
  dim3 block(MAX_BLOCK_DIM);
  std::uint64_t threads = (elements + elements_per_thread - 1) / elements_per_thread;
  return dim3((threads + block.x - 1) / block.x);
}

// The explicit vector kernels have fixed widths. Reject mismatched tuning so a
// YAML typo does not silently profile a different memory access pattern.
void validateVectorWidth(const VectorAddConfig &cfg, int expected) {
  int width = cfg.vector_width > 0 ? cfg.vector_width : expected;
  if (width != expected) {
    throw std::runtime_error("--vector-width must be " +
                             std::to_string(expected) +
                             " for dtype " + dtypeName(cfg.dtype));
  }
}

// Scalar dispatch keeps kernel symbols explicit while sharing the surrounding
// launch setup across dtypes.
void launchScalarKernel(const VectorAddConfig &cfg, const void *a,
                        const void *b, void *c, std::uint64_t n,
                        cudaStream_t stream) {
  dim3 block(MAX_BLOCK_DIM);
  dim3 grid = gridForElements(n, 1);

  switch (cfg.dtype) {
  case DType::Int32:
    add_i32_scalar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const int *>(a), static_cast<const int *>(b),
        static_cast<int *>(c), n);
    return;
  case DType::Float32:
    add_f32_scalar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float *>(a), static_cast<const float *>(b),
        static_cast<float *>(c), n);
    return;
  case DType::Float16:
    add_f16_scalar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half *>(a), static_cast<const half *>(b),
        static_cast<half *>(c), n);
    return;
  case DType::BFloat16:
    add_bf16_scalar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(a),
        static_cast<const __nv_bfloat16 *>(b),
        static_cast<__nv_bfloat16 *>(c), n);
    return;
  }
  throw std::runtime_error("Unsupported dtype");
}

// Vector dispatch maps normalized "vectorized" experiments onto the concrete
// kernel variant for the selected dtype.
void launchVectorKernel(const VectorAddConfig &cfg, const void *a,
                        const void *b, void *c, std::uint64_t n,
                        cudaStream_t stream) {
  dim3 block(MAX_BLOCK_DIM);

  switch (cfg.dtype) {
  case DType::Int32:
    validateVectorWidth(cfg, 4);
    add_i32_vec4_kernel<<<gridForElements(n, 4), block, 0, stream>>>(
        static_cast<const int *>(a), static_cast<const int *>(b),
        static_cast<int *>(c), n);
    return;
  case DType::Float32:
    validateVectorWidth(cfg, 4);
    add_f32_vec4_kernel<<<gridForElements(n, 4), block, 0, stream>>>(
        static_cast<const float *>(a), static_cast<const float *>(b),
        static_cast<float *>(c), n);
    return;
  case DType::Float16:
    validateVectorWidth(cfg, 2);
    add_f16_half2_kernel<<<gridForElements(n, 2), block, 0, stream>>>(
        static_cast<const half *>(a), static_cast<const half *>(b),
        static_cast<half *>(c), n);
    return;
  case DType::BFloat16:
    validateVectorWidth(cfg, 2);
    add_bf16_pair_kernel<<<gridForElements(n, 2), block, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(a),
        static_cast<const __nv_bfloat16 *>(b),
        static_cast<__nv_bfloat16 *>(c), n);
    return;
  }
  throw std::runtime_error("Unsupported dtype");
}

// Kernel selection is intentionally string-based at the edge of the runner so
// experiment YAML names map directly to the profiled CUDA symbol.
void launchVectorAddKernel(const VectorAddConfig &cfg, const void *a,
                           const void *b, void *c, std::uint64_t n,
                           cudaStream_t stream = 0) {
  if (cfg.kernel == "scalar") {
    launchScalarKernel(cfg, a, b, c, n, stream);
    return;
  }
  if (cfg.kernel == "vectorized") {
    launchVectorKernel(cfg, a, b, c, n, stream);
    return;
  }
  throw std::runtime_error("Unsupported --kernel for vector_add cuda: " +
                           cfg.kernel);
}

// Whole-buffer path used by pageable and pinned single-stream experiments.
Status runSingleStream(const VectorAddConfig &cfg, bool pinned) {
  const size_t bytes = cfg.n * dtypeSize(cfg.dtype);
  HostMemory h_a(bytes, pinned);
  HostMemory h_b(bytes, pinned);
  HostMemory h_c(bytes, pinned);
  initializeInputs(h_a.get(), h_b.get(), h_c.get(), cfg.n, cfg.dtype);

  CudaMemory<void> d_a(bytes), d_b(bytes), d_c(bytes);
  CUDA_CALL(cudaMemcpy(d_a.get(), h_a.get(), bytes, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_b.get(), h_b.get(), bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < cfg.warmup; ++i) {
    launchVectorAddKernel(cfg, d_a.get(), d_b.get(), d_c.get(), cfg.n);
  }
  for (int i = 0; i < cfg.repeats; ++i) {
    launchVectorAddKernel(cfg, d_a.get(), d_b.get(), d_c.get(), cfg.n);
  }
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(h_c.get(), d_c.get(), bytes, cudaMemcpyDeviceToHost));

  bool valid = !cfg.validate || validateResult(h_c.get(), cfg.n, cfg.dtype);
  std::cout << "Result: " << readDTypeValueAsDouble(h_c.get(), 0, cfg.dtype)
            << " "
            << readDTypeValueAsDouble(h_c.get(), cfg.n - 1, cfg.dtype)
            << std::endl;

  return valid ? Status::Ok : Status::ValidationFailed;
}

// Pageable chunking keeps only one chunk resident on the GPU at a time. This is
// useful for profiling copy/compute behavior on problem sizes larger than the
// desired device memory budget.
Status runChunked(const VectorAddConfig &cfg) {
  const size_t element_size = dtypeSize(cfg.dtype);
  const size_t bytes = cfg.n * element_size;
  HostMemory h_a(bytes, false);
  HostMemory h_b(bytes, false);
  HostMemory h_c(bytes, false);
  initializeInputs(h_a.get(), h_b.get(), h_c.get(), cfg.n, cfg.dtype);

  size_t free_mem = 0;
  size_t total_device_mem = 0;
  CUDA_CALL(cudaMemGetInfo(&free_mem, &total_device_mem));
  size_t total_mem = static_cast<size_t>(free_mem * cfg.free_mem_threshold);
  size_t chunk_max_len = (total_mem / 3) / element_size;
  std::uint64_t chunk_len = std::max<std::uint64_t>(MAX_BLOCK_DIM, chunk_max_len - (chunk_max_len % MAX_BLOCK_DIM));
  size_t chunk_bytes = chunk_len * element_size;

  CudaMemory<void> d_a(chunk_bytes), d_b(chunk_bytes), d_c(chunk_bytes);
  auto *base_a = static_cast<char *>(h_a.get());
  auto *base_b = static_cast<char *>(h_b.get());
  auto *base_c = static_cast<char *>(h_c.get());

  for (std::uint64_t offset = 0; offset < cfg.n; offset += chunk_len) {
    std::uint64_t current_len =
        std::min<std::uint64_t>(chunk_len, cfg.n - offset);
    size_t current_bytes = current_len * element_size;
    CUDA_CALL(cudaMemcpy(d_a.get(), base_a + offset * element_size, current_bytes,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_b.get(), base_b + offset * element_size, current_bytes,
                         cudaMemcpyHostToDevice));
    launchVectorAddKernel(cfg, d_a.get(), d_b.get(), d_c.get(), current_len);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(base_c + offset * element_size, d_c.get(), current_bytes,
                         cudaMemcpyDeviceToHost));
  }

  bool valid = !cfg.validate || validateResult(h_c.get(), cfg.n, cfg.dtype);
  std::cout << "Result: " << readDTypeValueAsDouble(h_c.get(), 0, cfg.dtype)
            << " "
            << readDTypeValueAsDouble(h_c.get(), cfg.n - 1, cfg.dtype)
            << std::endl;

  return valid ? Status::Ok : Status::ValidationFailed;
}

// Pinned multistream chunking overlaps H2D copy, kernel launch, and D2H copy
// across independent streams.
Status runChunkedMultiStream(const VectorAddConfig &cfg) {
  const size_t element_size = dtypeSize(cfg.dtype);
  const size_t bytes = cfg.n * element_size;
  HostMemory h_a(bytes, true);
  HostMemory h_b(bytes, true);
  HostMemory h_c(bytes, true);
  initializeInputs(h_a.get(), h_b.get(), h_c.get(), cfg.n, cfg.dtype);

  size_t free_mem = 0;
  size_t total_device_mem = 0;
  CUDA_CALL(cudaMemGetInfo(&free_mem, &total_device_mem));
  size_t total_mem = static_cast<size_t>(free_mem * cfg.free_mem_threshold);
  size_t chunk_max_len = (total_mem / (3 * cfg.streams)) / element_size;
  std::uint64_t chunk_len = std::max<std::uint64_t>(
      MAX_BLOCK_DIM, chunk_max_len - (chunk_max_len % MAX_BLOCK_DIM));
  size_t chunk_bytes = chunk_len * element_size;

  std::vector<cudaStream_t> streams(cfg.streams);
  std::vector<CudaMemory<void>> d_a;
  std::vector<CudaMemory<void>> d_b;
  std::vector<CudaMemory<void>> d_c;
  d_a.reserve(cfg.streams);
  d_b.reserve(cfg.streams);
  d_c.reserve(cfg.streams);
  for (int i = 0; i < cfg.streams; ++i) {
    CUDA_CALL(cudaStreamCreate(&streams[i]));
    d_a.emplace_back(chunk_bytes);
    d_b.emplace_back(chunk_bytes);
    d_c.emplace_back(chunk_bytes);
  }

  auto *base_a = static_cast<char *>(h_a.get());
  auto *base_b = static_cast<char *>(h_b.get());
  auto *base_c = static_cast<char *>(h_c.get());
  for (std::uint64_t chunk_start = 0; chunk_start < cfg.n;
       chunk_start += chunk_len * cfg.streams) {
    for (int stream_index = 0; stream_index < cfg.streams; ++stream_index) {
      std::uint64_t offset = chunk_start + stream_index * chunk_len;
      if (offset >= cfg.n) {
        break;
      }
      std::uint64_t current_len =
          std::min<std::uint64_t>(chunk_len, cfg.n - offset);
      size_t current_bytes = current_len * element_size;
      cudaStream_t stream = streams[stream_index];
      CUDA_CALL(cudaMemcpyAsync(d_a[stream_index].get(),
                                base_a + offset * element_size, current_bytes,
                                cudaMemcpyHostToDevice, stream));
      CUDA_CALL(cudaMemcpyAsync(d_b[stream_index].get(),
                                base_b + offset * element_size, current_bytes,
                                cudaMemcpyHostToDevice, stream));
      launchVectorAddKernel(cfg, d_a[stream_index].get(),
                            d_b[stream_index].get(),
                            d_c[stream_index].get(), current_len, stream);
      CUDA_CALL(cudaMemcpyAsync(base_c + offset * element_size,
                                d_c[stream_index].get(), current_bytes,
                                cudaMemcpyDeviceToHost, stream));
    }
  }

  for (int i = 0; i < cfg.streams; ++i) {
    CUDA_CALL(cudaStreamSynchronize(streams[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
  CUDA_CALL(cudaGetLastError());

  bool valid = !cfg.validate || validateResult(h_c.get(), cfg.n, cfg.dtype);
  std::cout << "Result: " << readDTypeValueAsDouble(h_c.get(), 0, cfg.dtype)
            << " "
            << readDTypeValueAsDouble(h_c.get(), cfg.n - 1, cfg.dtype)
            << std::endl;

  return valid ? Status::Ok : Status::ValidationFailed;
}

// Execution dispatch is separate from kernel dispatch so memory strategies can
// be compared without changing the concrete CUDA kernel symbol.
Status runCudaVectorAdd(const VectorAddConfig &cfg) {
  if (cfg.execution_mode == "single_stream" && cfg.memory_host == "pageable") {
    return runSingleStream(cfg, false);
  }
  if (cfg.execution_mode == "single_stream" && cfg.memory_host == "pinned") {
    return runSingleStream(cfg, true);
  }
  if (cfg.execution_mode == "chunked" && cfg.memory_host == "pageable") {
    return runChunked(cfg);
  }
  if (cfg.execution_mode == "chunked_multistream" &&
      cfg.memory_host == "pinned") {
    return runChunkedMultiStream(cfg);
  }
  throw std::runtime_error("Unsupported CUDA vector_add combination: memory.host=" +
                           cfg.memory_host + ", execution.mode=" +
                           cfg.execution_mode);
}

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

    // parse config from CLI args
    VectorAddConfig cfg = vectorAddConfigFromRunConfig(run_config);
    std::cout << "vector_add cuda dtype=" << dtypeName(cfg.dtype)
              << " kernel=" << cfg.kernel
              << " memory.host=" << cfg.memory_host
              << " execution.mode=" << cfg.execution_mode 
              << " n=" << cfg.n  
              << std::endl;

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
