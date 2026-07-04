// cuBLAS vector_add backend runner.
//
// Implements the normalized backend runner contract for the library reference
// case using cublasSaxpy: y = alpha * x + y. It currently supports float32.

#include "cuda_runtime.h"
#include "cli_args.hpp"
#include "experiment_types.hpp"
#include "utils.cuh"
#include "vector_add_config.hpp"

#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static void checkCublas(cublasStatus_t status, const char *func) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(func) + " failed");
  }
}

#define CUBLAS_CALL(func) checkCublas((func), #func)

int main(int argc, char **argv) {
  const std::vector<ArgSpec> args = vectorAddArgs("saxpy_reference", "float32");

  try {
    RunConfig run_config = parse_args(argc, argv, args);
    if (run_config.print_help) {
      print_usage(argv[0], args, "cuBLAS vector_add backend runner.");
      return EXIT_SUCCESS;
    }

    CUDA_CALL(cudaSetDevice(run_config.device));
    VectorAddConfig cfg = vectorAddConfigFromRunConfig(run_config, "float32");
    if (cfg.kernel != "saxpy_reference") {
      throw std::invalid_argument("Unsupported --kernel: " + cfg.kernel);
    }
    if (cfg.dtype != DType::Float32) {
      throw std::invalid_argument("cuBLAS vector_add currently supports dtype=float32 only");
    }
    if (cfg.execution_mode != "single_stream") {
      throw std::invalid_argument("cuBLAS vector_add supports execution_mode=single_stream only");
    }

    std::vector<float> h_x(cfg.n, 1.25f);
    std::vector<float> h_y(cfg.n, 2.5f);
    float *d_x = nullptr;
    float *d_y = nullptr;
    CUDA_CALL(cudaMalloc(&d_x, cfg.n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_y, cfg.n * sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_x, h_x.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_y, h_y.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSaxpy(handle, static_cast<int>(cfg.n), &cfg.alpha, d_x, 1, d_y, 1));
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(h_y.data(), d_y, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    CUBLAS_CALL(cublasDestroy(handle));
    CUDA_CALL(cudaFree(d_x));
    CUDA_CALL(cudaFree(d_y));

    if (cfg.validate) {
      float expected = 2.5f + cfg.alpha * 1.25f;
      if (std::abs(h_y.front() - expected) > 1e-5f ||
          std::abs(h_y.back() - expected) > 1e-5f) {
        throw std::runtime_error("Validation failed");
      }
    }

    std::cout << "Result: " << h_y.front() << " " << h_y.back() << std::endl;
    if (run_config.reset_device) {
      CUDA_CALL(cudaDeviceReset());
    }
  } catch (const std::exception &exc) {
    std::cerr << "Error: " << exc.what() << std::endl;
    print_usage(argv[0], args, "cuBLAS vector_add backend runner.");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
