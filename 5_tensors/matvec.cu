
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <assert.h>
#include <exception>
#include <string>
#include <iomanip>
#include <mma.h>
# define BLOCK_WIDTH 4

// Error Handling Wrapper
inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " (" + func + "): " + cudaGetErrorString(result)
        );
    }
}

#define CUDA_CALL(func) checkCuda((func), #func, __FILE__, __LINE__)

// Memory Handling Wrapper - RAII
template <typename T>
class CudaMemory {
public:
    explicit CudaMemory(size_t size)
    {
        ptr = nullptr;
        CUDA_CALL(cudaMalloc((void**)&ptr, size));
    }

    // Prevent copying
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Allow Moving
    CudaMemory(CudaMemory&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~CudaMemory() { if (ptr) cudaFree(ptr); }
    T* get() const { return ptr; }
private:
    T* ptr;
};

void printKernelConfig(dim3 grid, dim3 block)
{
    std::cout << "Grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    std::cout << "Block: " << block.x << " " << block.y << " " << block.z << std::endl;
    std::cout << "Threads in X:" << grid.x * block.x << " Y:" << grid.y * block.y << " Z:" << grid.z * block.z << std::endl;
}

// CPU Matrix Vector Multiplication
void cpuMatVecMul(const float* A, const float* B, float* C, int M, int N) {
    for (int i = 0; i < M; i++) {
        C[i] = 0;
        for (int j = 0; j < N; j++) {
            C[i] += A[i * N + j] * B[j];
        }
    }
}

bool verifyResults(const float* cpu_C, const float* gpu_C, int N, float tolerance = 1e-5) {
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_C[i] - gpu_C[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << " | CPU: " << cpu_C[i] << " vs GPU: " << gpu_C[i] << "\n";
            return false;
        }
    }
    std::cout << "GPU results match CPU results within tolerance " << tolerance << "\n";
    return true;
}

// Function to print a matrix
void printMatrix(const float* matrix, int M, int N)
{
    std::cout << "Matrix (" << M << "x" << N << "):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "-----------------------------------\n";
}

// MatVecmul - Global memory Access
__global__ void matvecmulKernelNaive(float* A, float* B, float* C, int M, int N)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int idx = (row * 1) + col; // Vector Dimension

    //printf("Row:%d Col:%d idx:%d  BID:%d\n", row, col ,idx, blockIdx.y);

    if (col < 1 && row < M)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++)
        {
            int in_idx = (row * N) + i;
            C[idx] += A[in_idx] * B[idx];
        }
    }
}

// MatVecmul - Tiled shared memory Access
__global__ void matvecmulKernelShared(float* A, float* B, float* C, int M, int N)
{
    // Shared Memory for tiles
    __shared__ float As[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float Bs[BLOCK_WIDTH];

    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int idx = (row * 1) + col; // Vector Dimension

    //printf("Row:%d Col:%d idx:%d  BID:%d\n", row, col ,idx, blockIdx.y);

    if (col < 1 && row < M)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++)
        {
            int in_idx = (row * N) + i;
            C[idx] += A[in_idx] * B[idx];
        }
    }
}


int main()
{
    // Choose GPU
    CUDA_CALL(cudaSetDevice(0));

    try
    {
        enum Options { MAT_VEC_MUL_NAIVE, MAT_VEC_MUL_SHARED };
        Options option = MAT_VEC_MUL_SHARED;

        // A->MxK, B->KxN, C->MxN
        const int M = 8;  // Rows - A
        const int N = 8;  // Cols - A, Rows - B
        int elts_A = M * N;  // Matrix
        int elts_B = N * 1;  // Vector
        int mem_size_A = elts_A * sizeof(float); 
        int mem_size_B = elts_B * sizeof(float);
        int mem_size_C = elts_B * sizeof(float);

        // Allocate host memory
        float* h_A, * h_B, * h_C, * h_C_cpu;
        h_A = (float*)malloc(mem_size_A);
        h_B = (float*)malloc(mem_size_B);
        h_C = (float*)malloc(mem_size_C);
        h_C_cpu = (float*)malloc(mem_size_C);

        // Initialize Host Data
        std::fill(h_A, h_A + elts_A, 2);
        std::fill(h_B, h_B + elts_B, 2);
        std::fill(h_C, h_C + elts_B, 0);
        std::fill(h_C_cpu, h_C_cpu + elts_B, 0);

        // CPU variant
        cpuMatVecMul(h_A, h_B, h_C_cpu, M, N);
        //printMatrix(h_A, M, N);
        //printMatrix(h_C_cpu, N, 1);

        // Device Memory
        CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);

        // Copy data to device
        CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

        // Define grid and block dimensions
        dim3 dimBlock(1, BLOCK_WIDTH); // tile size
        dim3 dimGrid(1, (N + dimBlock.y - 1) / dimBlock.y);

        printKernelConfig(dimGrid, dimBlock);

        // Launch the matrix multiplication kernel
        if (option == MAT_VEC_MUL_NAIVE)
        {
            matvecmulKernelNaive<<<dimGrid, dimBlock >> > (d_A.get(), d_B.get(), d_C.get(), M, N);
        }
        else if(option == MAT_VEC_MUL_SHARED) 
        {
            matvecmulKernelShared<<<dimGrid, dimBlock >> > (d_A.get(), d_B.get(), d_C.get(), M, N);
        }

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

        // Verify GPU results
        verifyResults(h_C_cpu, h_C, N);

        // cudaDeviceReset - for profiling
        CUDA_CALL(cudaDeviceReset());
    }
    catch (std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

