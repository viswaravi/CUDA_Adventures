// CUDA reduce backend runner.
//
// This executable owns the normalized reduce CUDA CLI contract and dispatches
// to the explicit kernel variants declared by ops/reduce/experiments.yaml.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels.cuh"
#include <kernel_lab/ops/reduce/config.hpp>
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
using namespace kernel_lab::ops::reduce;

namespace {

constexpr const char *kDescription = "CUDA reduce backend with normalized dtype, validation, warmup, and repeat args.";

struct KernelPlan {
    ReductionKernel kernel = nullptr;
    bool half_blocks = false;
    bool load_multiple = false;
    bool atomic = false;
};

void initializeInput(float *data, std::uint64_t n) { std::fill(data, data + n, 1.0f); }

dim3 initialGrid(std::uint64_t n, unsigned int block_width, bool load_multiple) {
    std::uint64_t elements_per_block = load_multiple ? block_width * 2ULL : block_width;
    return dim3((n + elements_per_block - 1) / elements_per_block);
}

KernelPlan planForKernel(const std::string &kernel_name) {
    if (kernel_name == "interleaved-divergent") {
        return {reduce1, false, false, false};
    }
    if (kernel_name == "interleaved-bank-conflicts") {
        return {reduce2, false, false, false};
    }
    if (kernel_name == "sequential-idle") {
        return {reduce3, false, false, false};
    }
    if (kernel_name == "sequential-add-load") {
        return {reduce4, true, true, false};
    }
    if (kernel_name == "sequential-last-unroll") {
        return {reduce5, true, true, false};
    }
    if (kernel_name == "sequential-full-unroll") {
        return {reduce6, true, true, false};
    }
    if (kernel_name == "sequential-multiple-load-warpshuffle") {
        return {reduce7, true, true, false};
    }
    if (kernel_name == "atomic") {
        return {reduceAtomic, false, false, true};
    }
    if (kernel_name == "warp-primitives") {
        return {warpPrimitives, false, false, false};
    }
    throw std::runtime_error("Unsupported --kernel for reduce cuda: " + kernel_name);
}

float launchRecursiveReduction(const ReduceConfig &cfg, CudaMemory<float> &d_input) {
    const KernelPlan plan = planForKernel(cfg.kernel);
    unsigned int block_width = BLOCK_WIDTH;
    dim3 block(block_width);
    if (plan.half_blocks) {
        block.x /= 2;
    }

    dim3 grid = initialGrid(cfg.n, block.x, plan.load_multiple);
    CudaMemory<float> d_block_sums(grid.x * sizeof(float));
    unsigned int num_blocks = grid.x;

    if (plan.atomic) {
        CUDA_CALL(cudaMemset(d_block_sums.get(), 0, grid.x * sizeof(float)));
    }

    printKernelConfig(grid, block);
    std::cout << "Array Length: " << cfg.n << std::endl;
    plan.kernel<<<grid, block, block.x * sizeof(float)>>>(d_input.get(), d_block_sums.get(), cfg.n);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    while (num_blocks > 1) {
        grid.x = initialGrid(num_blocks, block.x, plan.load_multiple).x;
        CudaMemory<float> d_block_sums_out(grid.x * sizeof(float));
        if (plan.atomic) {
            CUDA_CALL(cudaMemset(d_block_sums_out.get(), 0, grid.x * sizeof(float)));
        }

        printKernelConfig(grid, block);
        std::cout << "Array Length: " << num_blocks << std::endl;
        plan.kernel<<<grid, block, block.x * sizeof(float)>>>(d_block_sums.get(), d_block_sums_out.get(), num_blocks);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        d_block_sums = std::move(d_block_sums_out);
        num_blocks = grid.x;
    }

    float result = 0.0f;
    CUDA_CALL(cudaMemcpy(&result, d_block_sums.get(), sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

bool validateResult(float actual, double expected) {
    Tolerance tolerance = toleranceFor(DType::Float32);
    double diff = std::abs(static_cast<double>(actual) - expected);
    double limit = tolerance.abs + tolerance.rel * std::abs(expected);
    if (diff > limit) {
        std::cerr << "Validation failed: expected " << expected << ", got " << actual << ", tolerance " << limit
                  << std::endl;
        return false;
    }
    return true;
}

Status runCudaReduce(const ReduceConfig &cfg) {
    const size_t bytes = cfg.n * sizeof(float);
    HostMemory h_input(bytes, false);
    auto *input = static_cast<float *>(h_input.get());
    initializeInput(input, cfg.n);
    const double expected = hostReduceSum(input, static_cast<size_t>(cfg.n));

    CudaMemory<float> d_input(bytes);
    float result = 0.0f;

    for (int i = 0; i < cfg.warmup; ++i) {
        CUDA_CALL(cudaMemcpy(d_input.get(), input, bytes, cudaMemcpyHostToDevice));
        result = launchRecursiveReduction(cfg, d_input);
    }
    for (int i = 0; i < cfg.repeats; ++i) {
        CUDA_CALL(cudaMemcpy(d_input.get(), input, bytes, cudaMemcpyHostToDevice));
        result = launchRecursiveReduction(cfg, d_input);
    }

    std::cout << "Reduce Result: " << result << " expected " << expected << std::endl;
    bool valid = !cfg.validate || validateResult(result, expected);
    return valid ? Status::Ok : Status::ValidationFailed;
}

} // namespace

int main(int argc, char **argv) {
    const std::vector<ArgSpec> args = reduceArgs();

    try {
        RunConfig run_config = parse_args(argc, argv, args);
        if (run_config.print_help) {
            print_usage(argv[0], args, kDescription);
            return EXIT_SUCCESS;
        }

        CUDA_CALL(cudaSetDevice(run_config.device));
        printDeviceDetails();

        ReduceConfig cfg = reduceConfigFromRunConfig(run_config);
        std::cout << "reduce cuda dtype=" << dtypeName(cfg.dtype) << " kernel=" << cfg.kernel << " n=" << cfg.n
                  << " validate=" << (cfg.validate ? "true" : "false") << " warmup=" << cfg.warmup
                  << " repeats=" << cfg.repeats << std::endl;

        Status status = runCudaReduce(cfg);
        if (status != Status::Ok) {
            throw std::runtime_error("reduce cuda failed");
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
