
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
#define BLOCK_WIDTH 1024
#define FULL_MASK 0xffffffff

__global__ void inclusiveScan_naive(float *A, float *result, const unsigned long long length)
{
    __shared__ float partial[BLOCK_WIDTH];

    unsigned long long idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid = threadIdx.x;

    // load into shared memory
    if (idx < length)
    {
        partial[tid] = A[idx];
    }

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
    if (idx < length)
    {
        partial[tid] = A[idx];
    }

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
    if (idx < length)
    {
        partial[tid] = A[idx];
    }

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

    result[idx] = partial[tid];
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
    if (idx < length)
    {
        partial[tid] = A[idx];
    }
    if (idx + bx < length)
    {
        partial[tid + bx] = A[idx + bx];
    }

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
    if (idx < length)
    {
        partial[tid] = A[idx];
    }

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

int main()
{
    // Choose GPU
    CUDA_CALL(cudaSetDevice(0));

    try
    {
        enum Options
        {
            INCLUSIVE_SCAN_NAIVE,
            INCLUSIVE_SCAN_FAST,
            INCLUSIVE_SCAN_FAST2,
            INCLUSIVE_SCAN_DOUBLE,
            INCLUSIVE_SCAN_HIER
        };

        Options option = INCLUSIVE_SCAN_NAIVE;

        unsigned long long array_len = 1024;
        size_t mem_size = array_len * sizeof(float);

        // Host Data Reduction
        float *h_A, *result;
        float result_cpu = 0.0f;
        h_A = (float *)malloc(mem_size);
        result = (float *)malloc(sizeof(float));
        // Host Data Prefix sum
        float *result_scan, *result_scan_cpu, *block_sums_cpu;
        result_scan = (float *)malloc(mem_size);
        result_scan_cpu = (float *)malloc(mem_size);

        // Init
        std::fill(h_A, h_A + array_len, 1);
        *result = 0.0f;

        // Device Data
        CudaMemory<float> d_A(mem_size), d_result(sizeof(float)),
            d_result_scan(mem_size);

        // Copy to Device
        CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size, cudaMemcpyHostToDevice));

        // Kernel Config
        dim3 blockDim(BLOCK_WIDTH);
        dim3 gridDim((array_len + blockDim.x - 1) / blockDim.x);

        // Block Memory for Recursive Reduce
        CudaMemory<float> d_blockSums(gridDim.x * sizeof(float));
        int numBlocks = gridDim.x;

        // kernel config for double procesing
        int block_size = array_len / 2; // process using half the threads
        int shared_size = array_len * sizeof(float);
        dim3 blockDimD(block_size);
        dim3 gridDimD((block_size + block_size - 1) / block_size);

        // Hierarchical prefix sum
        if (option == INCLUSIVE_SCAN_HIER)
        {
            inclusiveScanCPU(h_A, result_scan_cpu, array_len);
            int num_blocks = array_len / 1024;
            size_t block_sum_size = numBlocks * sizeof(float);
            block_sums_cpu = (float *)malloc(block_sum_size);
            CudaMemory<float> d_block_sums(block_sum_size);

            // Block wise Reduction
            printKernelConfig(gridDim, blockDim);
            inclusiveScan_block_reduce<<<gridDim, blockDim,
                                         BLOCK_WIDTH * sizeof(float)>>>(
                d_A.get(), d_block_sums.get(), array_len);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            /*     CUDA_CALL(cudaMemcpy(block_sums_cpu, d_block_sums.get(), block_sum_size, cudaMemcpyDeviceToHost));
                  printArray(block_sums_cpu, numBlocks);*/

            // Block Reduce
            dim3 blockDimH(numBlocks);
            dim3 gridDimH(1);
            inclusiveScan_fast2<<<gridDim, blockDim>>>(d_block_sums.get(), d_block_sums.get(), numBlocks);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            /*CUDA_CALL(cudaMemcpy(block_sums_cpu, d_block_sums.get(), block_sum_size, cudaMemcpyDeviceToHost));
                  printArray(block_sums_cpu, numBlocks);*/

            // Add Blocksum to array
            add_block_reduce<<<gridDim, blockDim>>>(d_A.get(), array_len, d_block_sums.get());
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            // Check Result
            CUDA_CALL(cudaMemcpy(result_scan, d_A.get(), mem_size, cudaMemcpyDeviceToHost));
            verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
            // printArray(result_scan, array_len);

            // Free Memory
            free(block_sums_cpu);
        }

        switch (option)
        {
        case INCLUSIVE_SCAN_NAIVE:
            assert(array_len <= 1024);
            inclusiveScan_naive<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(), array_len);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size,
                                 cudaMemcpyDeviceToHost));
            inclusiveScanCPU(h_A, result_scan_cpu, array_len);
            verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
            break;

        case INCLUSIVE_SCAN_FAST:
            assert(array_len <= 1024);
            inclusiveScan_fast<<<gridDim, blockDim>>>(d_A.get(), d_result_scan.get(), array_len);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size,
                                 cudaMemcpyDeviceToHost));
            inclusiveScanCPU(h_A, result_scan_cpu, array_len);
            verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
            break;

        case INCLUSIVE_SCAN_FAST2:
            assert(array_len <= 1024);
            inclusiveScan_fast2<<<gridDim, blockDim>>>(
                d_A.get(), d_result_scan.get(), array_len);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size,
                                 cudaMemcpyDeviceToHost));
            inclusiveScanCPU(h_A, result_scan_cpu, array_len);
            verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
            break;

        case INCLUSIVE_SCAN_DOUBLE:
            assert(array_len <= 1024);
            inclusiveScan_double<<<gridDimD, blockDimD, shared_size>>>(d_A.get(), d_result_scan.get(), array_len);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(result_scan, d_result_scan.get(), mem_size,
                                 cudaMemcpyDeviceToHost));
            inclusiveScanCPU(h_A, result_scan_cpu, array_len);
            // printArray(result_scan, array_len);
            // printArray(result_scan_cpu, array_len);
            verifyInclusiveScan(result_scan, result_scan_cpu, array_len);
            break;

        default:
            break;
        }

        // Free memory
        free(h_A);
        free(result);
        free(result_scan);
        free(result_scan_cpu);

        // cudaDeviceReset - for profiling
        CUDA_CALL(cudaDeviceReset());
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "Exception: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
