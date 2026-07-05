// CUDA prefix_sum backend runner.
//
// This executable owns the normalized prefix_sum CUDA CLI contract and
// dispatches to the explicit kernel variants declared by experiments.yaml.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels.cuh"
#include <kernel_lab/ops/prefix_sum/config.hpp>
#include <kernel_lab/utils/cli_args.hpp>
#include <kernel_lab/utils/cuda_utils.cuh>
#include <kernel_lab/utils/experiment_types.hpp>
#include <kernel_lab/utils/host_algorithms.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kernel_lab;
using namespace kernel_lab::ops::prefix_sum;

namespace {

constexpr const char *kDescription =
    "CUDA prefix_sum backend with normalized dtype, validation, warmup, and repeat args.";

void initializeInput(float *data, std::uint64_t n) { std::fill(data, data + n, 1.0f); }

void validateKernelConstraints(const PrefixSumConfig &cfg) {
    if (cfg.kernel == "naive" || cfg.kernel == "fast" || cfg.kernel == "fast2") {
        if (cfg.n > BLOCK_WIDTH) {
            throw std::runtime_error("--kernel " + cfg.kernel + " requires --n <= " + std::to_string(BLOCK_WIDTH));
        }
        return;
    }

    if (cfg.kernel == "double") {
        if (cfg.n > BLOCK_WIDTH) {
            throw std::runtime_error("--kernel double requires --n <= " + std::to_string(BLOCK_WIDTH));
        }
        if (cfg.n < 2 || (cfg.n % 2) != 0) {
            throw std::runtime_error("--kernel double requires even --n >= 2");
        }
        return;
    }

    if (cfg.kernel == "hier") {
        std::uint64_t num_blocks = (cfg.n + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
        if (num_blocks > BLOCK_WIDTH) {
            throw std::runtime_error("--kernel hier requires ceil(n / BLOCK_WIDTH) <= BLOCK_WIDTH");
        }
        return;
    }

    throw std::runtime_error("Unsupported --kernel for prefix_sum cuda: " + cfg.kernel);
}

void launchPrefixSumKernel(const PrefixSumConfig &cfg, CudaMemory<float> &d_input, CudaMemory<float> &d_output) {
    validateKernelConstraints(cfg);

    if (cfg.kernel == "naive") {
        dim3 block(BLOCK_WIDTH);
        dim3 grid((cfg.n + block.x - 1) / block.x);
        inclusiveScan_naive<<<grid, block>>>(d_input.get(), d_output.get(), cfg.n);
        return;
    }

    if (cfg.kernel == "fast") {
        dim3 block(BLOCK_WIDTH);
        dim3 grid((cfg.n + block.x - 1) / block.x);
        inclusiveScan_fast<<<grid, block>>>(d_input.get(), d_output.get(), cfg.n);
        return;
    }

    if (cfg.kernel == "fast2") {
        dim3 block(BLOCK_WIDTH);
        dim3 grid((cfg.n + block.x - 1) / block.x);
        inclusiveScan_fast2<<<grid, block>>>(d_input.get(), d_output.get(), cfg.n);
        return;
    }

    if (cfg.kernel == "double") {
        int block_size = static_cast<int>(cfg.n / 2);
        int shared_size = static_cast<int>(cfg.n * sizeof(float));
        dim3 block(block_size);
        dim3 grid(1);
        inclusiveScan_double<<<grid, block, shared_size>>>(d_input.get(), d_output.get(), cfg.n);
        return;
    }

    if (cfg.kernel == "hier") {
        dim3 block(BLOCK_WIDTH);
        dim3 grid((cfg.n + block.x - 1) / block.x);
        int num_blocks = static_cast<int>(grid.x);
        CudaMemory<float> d_block_sums(num_blocks * sizeof(float));
        inclusiveScan_block_reduce<<<grid, block, BLOCK_WIDTH * sizeof(float)>>>(d_input.get(), d_block_sums.get(),
                                                                                 cfg.n);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        dim3 block_h(BLOCK_WIDTH);
        dim3 grid_h((num_blocks + block_h.x - 1) / block_h.x);
        inclusiveScan_fast2<<<grid_h, block_h>>>(d_block_sums.get(), d_block_sums.get(), num_blocks);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        add_block_reduce<<<grid, block>>>(d_input.get(), cfg.n, d_block_sums.get());
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(d_output.get(), d_input.get(), cfg.n * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }

    throw std::runtime_error("Unsupported --kernel for prefix_sum cuda: " + cfg.kernel);
}

bool validateResult(const float *actual, const float *expected, std::uint64_t n) {
    Tolerance tolerance = toleranceFor(DType::Float32);
    for (std::uint64_t i = 0; i < n; ++i) {
        double want = static_cast<double>(expected[i]);
        double got = static_cast<double>(actual[i]);
        double diff = std::abs(got - want);
        double limit = tolerance.abs + tolerance.rel * std::abs(want);
        if (diff > limit) {
            std::cerr << "Validation failed at index " << i << ": expected " << want << ", got " << got
                      << ", tolerance " << limit << std::endl;
            return false;
        }
    }
    return true;
}

Status runCudaPrefixSum(const PrefixSumConfig &cfg) {
    const size_t bytes = cfg.n * sizeof(float);
    HostMemory h_input(bytes, false);
    HostMemory h_output(bytes, false);
    HostMemory h_expected(bytes, false);
    auto *input = static_cast<float *>(h_input.get());
    auto *output = static_cast<float *>(h_output.get());
    auto *expected = static_cast<float *>(h_expected.get());

    initializeInput(input, cfg.n);
    hostInclusiveScan(input, expected, static_cast<size_t>(cfg.n));

    CudaMemory<float> d_input(bytes);
    CudaMemory<float> d_output(bytes);

    for (int i = 0; i < cfg.warmup; ++i) {
        CUDA_CALL(cudaMemcpy(d_input.get(), input, bytes, cudaMemcpyHostToDevice));
        launchPrefixSumKernel(cfg, d_input, d_output);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (int i = 0; i < cfg.repeats; ++i) {
        CUDA_CALL(cudaMemcpy(d_input.get(), input, bytes, cudaMemcpyHostToDevice));
        launchPrefixSumKernel(cfg, d_input, d_output);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    CUDA_CALL(cudaMemcpy(output, d_output.get(), bytes, cudaMemcpyDeviceToHost));
    std::cout << "Prefix Sum Result: " << output[0] << " " << output[cfg.n - 1] << std::endl;
    bool valid = !cfg.validate || validateResult(output, expected, cfg.n);
    return valid ? Status::Ok : Status::ValidationFailed;
}

} // namespace

int main(int argc, char **argv) {
    const std::vector<ArgSpec> args = prefixSumArgs();

    try {
        RunConfig run_config = parse_args(argc, argv, args);
        if (run_config.print_help) {
            print_usage(argv[0], args, kDescription);
            return EXIT_SUCCESS;
        }

        CUDA_CALL(cudaSetDevice(run_config.device));
        printDeviceDetails();

        PrefixSumConfig cfg = prefixSumConfigFromRunConfig(run_config);
        std::cout << "prefix_sum cuda dtype=" << dtypeName(cfg.dtype) << " kernel=" << cfg.kernel << " n=" << cfg.n
                  << " validate=" << (cfg.validate ? "true" : "false") << " warmup=" << cfg.warmup
                  << " repeats=" << cfg.repeats << std::endl;

        Status status = runCudaPrefixSum(cfg);
        if (status != Status::Ok) {
            throw std::runtime_error("prefix_sum cuda failed");
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
