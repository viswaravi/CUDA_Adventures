
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

#include <assert.h>
#include <stdio.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "cli_args.hpp"
#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

__global__ void inclusiveScan_naive(float *A, float *result, const unsigned long long length)
{
    __shared__ float partial[BLOCK_WIDTH];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;

    // load into shared memory
    partial[tid] = idx < length ? A[idx] : 0.0f;

    // iterative scan
    for (unsigned int stride = 1; stride <= tid; stride *= 2)
    {
        __syncthreads();

        if (tid - stride >= 0)
        {
            partial[tid] += partial[tid - stride];
        }
    }

    if (idx < length)
    {
        result[idx] = partial[tid];
    }
}

// conditionally divergent reduction - scan
__global__ void inclusiveScan_fast(float *A, float *result, const unsigned long long length)
{
    __shared__ float partial[BLOCK_WIDTH];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;

    // load into shared memory
    partial[tid] = idx < length ? A[idx] : 0.0f;

    // reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        if ((tid + 1) % (2 * stride) == 0) // elements at (2n-1,4n-1,8n-1,...)
        {
            partial[tid] += partial[tid - stride];
        }
    }

    // reverse tree distribution
    for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if ((index + stride) < blockDim.x)
        {
            partial[index + stride] += partial[index];
        }
    }

    __syncthreads();

    if (idx < length)
    {
        result[idx] = partial[tid];
    }
}

// index - scan
__global__ void inclusiveScan_fast2(float *A, float *result, const unsigned long long length)
{
    __shared__ float partial[BLOCK_WIDTH];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;

    // load into shared memory
    partial[tid] = idx < length ? A[idx] : 0.0f;

    // reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1; // 2n-1, 4n-1, 8n-1,...

        if (index < blockDim.x)
        {
            partial[index] += partial[index - stride];
        }
    }

    // reverse tree distribution
    for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if ((index + stride) < blockDim.x)
        {
            partial[index + stride] += partial[index];
        }
    }

    if (idx < length)
    {
        result[idx] = partial[tid];
    }
}

// double data processing
__global__ void inclusiveScan_double(float *A, float *result, const unsigned long long length)
{
    extern __shared__ float partial[];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;
    int bx = blockDim.x;
    int sharedDim = 2 * blockDim.x;

    // load two indices into shared memory
    partial[tid] = idx < length ? A[idx] : 0.0f;
    partial[tid + bx] = idx + bx < length ? A[idx + bx] : 0.0f;

    // reduction
    // Extra one step to reduce until second batch
    for (unsigned int stride = 1; stride <= bx; stride *= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if (index < sharedDim)
        {
            partial[index] += partial[index - stride];
        }
    }

    // reverse tree distribution
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if ((index + stride) < sharedDim)
        {
            partial[index + stride] += partial[index];
        }
    }

    __syncthreads();

    if (idx < length)
    {
        result[idx] = partial[tid];
    }
    if (idx + bx < length)
    {
        result[idx + bx] = partial[tid + bx];
    }
}

__global__ void inclusiveScan_block_reduce(float *A, float *block_sums, const unsigned long long length)
{
    extern __shared__ float partial[];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;
    int bx = blockDim.x;

    // load two indices into shared memory
    partial[tid] = idx < length ? A[idx] : 0.0f;

    // reduction
    // Extra one step to reduce until second batch
    for (unsigned int stride = 1; stride < bx; stride *= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if (index < bx)
        {
            partial[index] += partial[index - stride];
        }
    }

    // reverse tree distribution
    for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2)
    {
        __syncthreads();

        int index = (tid + 1) * (2 * stride) - 1;

        if ((index + stride) < blockDim.x)
        {
            partial[index + stride] += partial[index];
        }
    }

    __syncthreads();

    if (idx < length)
    {
        A[idx] = partial[tid];
    }

    // Add final index to  block sum
    block_sums[blockIdx.x] = partial[bx - 1];
}

__global__ void add_block_reduce(float *A, const unsigned long long length, float *d_block_sums)
{
    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (blockIdx.x > 0 && idx < length)
    {
        A[idx] += d_block_sums[blockIdx.x - 1];
    }
}

float reduceCPU(float *A, const unsigned long long length)
{
    float sum = 0.0f;
    for (unsigned long long i = 0; i < length; i++)
    {
        sum += A[i];
    }
    return sum;
}

void inclusiveScanCPU(float *A, float *result, const unsigned long long length)
{
    float sum = 0.0f;
    for (unsigned long long i = 0; i < length; i++)
    {
        sum += A[i];
        result[i] = sum;
    }
}

void verifyInclusiveScan(float *A, float *B, const unsigned long long length)
{
    for (unsigned long long i = 0; i < length; i++)
    {
        if (A[i] - B[i] != 0.0)
        {
            std::cout << "Inclusive Scan Verification Failed at Index: " << i
                      << " A:" << A[i] << " B:" << B[i] << std::endl;
            return;
        }
    }
    std::cout << "Inclusive Scan Verification Passed!" << std::endl;
}

void printArray(float *A, const unsigned long long length)
{
    for (unsigned long long i = 0; i < length; i++)
    {
        std::cout << std::fixed << std::setprecision(2) << A[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    const std::vector<ArgSpec> scan_args = {
        {"--n", "uint64", "1024", "Array length (elements)"},
    };

    const VariantRegistry variants = {
        {
            "naive",
            "Inclusive scan with iterative shared-memory accumulation",
            scan_args,
            [](const RunConfig &c)
            {
                unsigned long long array_len = get_ull(c, "--n", 1024ULL);
                assert(array_len <= BLOCK_WIDTH);
                size_t mem_size = array_len * sizeof(float);
                float *h_A = (float *)malloc(mem_size);
                float *result_scan = (float *)malloc(mem_size);
                float *result_scan_cpu = (float *)malloc(mem_size);

                std::fill(h_A, h_A + array_len, 1.0f);

                CudaMemory<float> d_A(mem_size), d_result_scan(mem_size);
                CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

                dim3 blockDim(BLOCK_WIDTH);
                dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
                inclusiveScan_naive<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(), array_len);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());
                CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size, cudaMemcpyDeviceToHost));

                inclusiveScanCPU(h_A, result_scan_cpu, array_len);
                verifyInclusiveScan(result_scan, result_scan_cpu, array_len);

                free(h_A);
                free(result_scan);
                free(result_scan_cpu);
            },
        },
        {
            "fast",
            "Inclusive scan with tree reduction and distribution",
            scan_args,
            [](const RunConfig &c)
            {
                unsigned long long array_len = get_ull(c, "--n", 1024ULL);
                assert(array_len <= BLOCK_WIDTH);
                size_t mem_size = array_len * sizeof(float);
                float *h_A = (float *)malloc(mem_size);
                float *result_scan = (float *)malloc(mem_size);
                float *result_scan_cpu = (float *)malloc(mem_size);

                std::fill(h_A, h_A + array_len, 1.0f);

                CudaMemory<float> d_A(mem_size), d_result_scan(mem_size);
                CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

                dim3 blockDim(BLOCK_WIDTH);
                dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
                inclusiveScan_fast<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(), array_len);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());
                CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size, cudaMemcpyDeviceToHost));

                inclusiveScanCPU(h_A, result_scan_cpu, array_len);
                verifyInclusiveScan(result_scan, result_scan_cpu, array_len);

                free(h_A);
                free(result_scan);
                free(result_scan_cpu);
            },
        },
        {
            "fast2",
            "Inclusive scan using index-based tree traversal",
            scan_args,
            [](const RunConfig &c)
            {
                unsigned long long array_len = get_ull(c, "--n", 1024ULL);
                assert(array_len <= BLOCK_WIDTH);
                size_t mem_size = array_len * sizeof(float);
                float *h_A = (float *)malloc(mem_size);
                float *result_scan = (float *)malloc(mem_size);
                float *result_scan_cpu = (float *)malloc(mem_size);

                std::fill(h_A, h_A + array_len, 1.0f);

                CudaMemory<float> d_A(mem_size), d_result_scan(mem_size);
                CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

                dim3 blockDim(BLOCK_WIDTH);
                dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
                inclusiveScan_fast2<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(), array_len);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());
                CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size, cudaMemcpyDeviceToHost));

                inclusiveScanCPU(h_A, result_scan_cpu, array_len);
                verifyInclusiveScan(result_scan, result_scan_cpu, array_len);

                free(h_A);
                free(result_scan);
                free(result_scan_cpu);
            },
        },
        {
            "double",
            "Inclusive scan processing two elements per thread",
            scan_args,
            [](const RunConfig &c)
            {
                unsigned long long array_len = get_ull(c, "--n", 1024ULL);
                assert(array_len <= BLOCK_WIDTH);
                size_t mem_size = array_len * sizeof(float);
                float *h_A = (float *)malloc(mem_size);
                float *result_scan = (float *)malloc(mem_size);
                float *result_scan_cpu = (float *)malloc(mem_size);

                std::fill(h_A, h_A + array_len, 1.0f);

                CudaMemory<float> d_A(mem_size), d_result_scan(mem_size);
                CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

                int block_size = static_cast<int>(array_len / 2);
                assert(block_size > 0);
                int shared_size = static_cast<int>(array_len * sizeof(float));
                dim3 blockDimD(block_size);
                dim3 gridDimD((block_size + block_size - 1) / block_size);
                inclusiveScan_double<<<gridDimD, blockDimD, shared_size>>>(d_A.get(), d_result_scan.get(), array_len);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());
                CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size, cudaMemcpyDeviceToHost));

                inclusiveScanCPU(h_A, result_scan_cpu, array_len);
                verifyInclusiveScan(result_scan, result_scan_cpu, array_len);

                free(h_A);
                free(result_scan);
                free(result_scan_cpu);
            },
        },
        {
            "hier",
            "Hierarchical inclusive scan for multi-block arrays",
            scan_args,
            [](const RunConfig &c)
            {
                unsigned long long array_len = get_ull(c, "--n", 1048576ULL);
                size_t mem_size = array_len * sizeof(float);
                float *h_A = (float *)malloc(mem_size);
                float *result_scan = (float *)malloc(mem_size);
                float *result_scan_cpu = (float *)malloc(mem_size);

                std::fill(h_A, h_A + array_len, 1.0f);

                CudaMemory<float> d_A(mem_size);
                CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

                dim3 blockDim(BLOCK_WIDTH);
                dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);
                int numBlocks = static_cast<int>(gridDim.x);
                assert(numBlocks <= BLOCK_WIDTH);

                size_t block_sum_size = numBlocks * sizeof(float);
                CudaMemory<float> d_block_sums(block_sum_size);

                inclusiveScan_block_reduce<<<gridDim, blockDim, BLOCK_WIDTH * sizeof(float)>>>(
                    d_A.get(), d_block_sums.get(), array_len);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                dim3 blockDimH(BLOCK_WIDTH);
                dim3 gridDimH((numBlocks + blockDimH.x - 1) / blockDimH.x);
                inclusiveScan_fast2<<<gridDimH, blockDimH>>>(d_block_sums.get(), d_block_sums.get(), numBlocks);
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                add_block_reduce<<<gridDim, blockDim>>>(d_A.get(), array_len, d_block_sums.get());
                CUDA_CALL(cudaGetLastError());
                CUDA_CALL(cudaDeviceSynchronize());

                CUDA_CALL(cudaMemcpy(result_scan, d_A.get(), mem_size, cudaMemcpyDeviceToHost));
                inclusiveScanCPU(h_A, result_scan_cpu, array_len);
                verifyInclusiveScan(result_scan, result_scan_cpu, array_len);

                free(h_A);
                free(result_scan);
                free(result_scan_cpu);
            },
        },
    };

    try
    {
        RunConfig cfg = parse_args(argc, argv, variants);
        if (cfg.print_help)
        {
            print_usage(argv[0], variants);
            return EXIT_SUCCESS;
        }
        if (cfg.list_variants)
        {
            print_variants(variants);
            return EXIT_SUCCESS;
        }

        CUDA_CALL(cudaSetDevice(cfg.device));
        printDeviceDetails();
        run_variant(cfg, variants);

        if (cfg.reset_device)
        {
            CUDA_CALL(cudaDeviceReset());
        }
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
        print_usage(argv[0], variants);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
