
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
#include <cublas_v2.h>

# define BLOCK_WIDTH 32

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " (" + func + "): " + cudaGetErrorString(result)
        );
    }
}

inline void checkCUBLAS(cublasStatus_t status, const char* func, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line) +
            " (" + func + "): " + cublasGetStatusString(status)
        );
    }
}

#define CUDA_CALL(func) checkCuda((func), #func, __FILE__, __LINE__)
#define CUBLAS_CALL(func) checkCUBLAS((func), #func, __FILE__, __LINE__)


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
    std::cout << "Threds in X:" << grid.x * block.x << " Y:" << grid.y * block.y << " Z:" << grid.z * block.z << std::endl;
}

// CPU Matmul
void cpuMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void cpuMatTranspose(const float* A, float* B, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

bool verifyResults(const float* cpu_C, const float* gpu_C, int N, float tolerance = 1e-5) {
    for (int i = 0; i < N * N; i++) {
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

// Test kernel to fill and check 2D indexing
__global__ void fill2DIdx(float* A, float* B, float* C, int N)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int width_x = (gridDim.x * blockDim.x);
    int globalIdx = (row * width_x) + col;

    C[globalIdx] = globalIdx;
}

// Matmul - Global memory Access
__global__ void matmulKernelNaive(float* A, float* B, float* C, int M, int N, int K)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int globalIdx = (row * N) + col; // global idx for Matrix Dimension, not thread dimension

    if (row < M && col < N)
    {
        float value = 0.0f;
        for (int k = 0; k < K; k++)
        {
            value += (A[row * N + k] * B[k * N + col]);
        }
        C[globalIdx] = value;
    }
}

// Matmul - Shared memory Access
__global__ void matmulKernelShared(float* A, float* B, float* C, int M, int N, int K)
{
    // Shared Memory for tiles
    __shared__ float As[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float Bs[BLOCK_WIDTH][BLOCK_WIDTH];

    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int globalIdx = (row * N) + col; // global idx for Matrix Dimension, not thread dimension
    float value = 0.0f;

    // loop through tiles
    int num_tiles = (K + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    for (unsigned int tile = 0; tile < num_tiles; tile++)
    {
        // Load into Shared Memory
        if (row < M && (tile * BLOCK_WIDTH + threadIdx.x) < K)
        {
            //  row                  * width + col
            int tile_idx_A = row * K + (tile * BLOCK_WIDTH + threadIdx.x); // tiling in x_direction 
            As[threadIdx.y][threadIdx.x] = A[tile_idx_A];

        }
        if ((tile * BLOCK_WIDTH + threadIdx.y) < K && col < N)
        {
            //  row                  * width +  col
            int tile_idx_B = (tile * BLOCK_WIDTH + threadIdx.y) * N + col;                             // tiling in y_direction
            Bs[threadIdx.y][threadIdx.x] = B[tile_idx_B];
        }

        __syncthreads();


        // Compute from shared memory
        for (int k = 0; k < BLOCK_WIDTH; k++)
        {
            value += (As[threadIdx.y][k] * Bs[k][threadIdx.x]);
        }

        __syncthreads();
    }

    C[globalIdx] = value;
}


int mains()
{
    // Choose GPU
    CUDA_CALL(cudaSetDevice(0));

    try
    {
        enum Options { MATMUL_NAIVE, MATMUL_SHARED, MATMUL_cuBLAS, MATMUL_CUTLASS, MATMUL_WMMA };
        Options option = MATMUL_SHARED;

        // A->MxK, B->KxN, C->MxN
        const int M = 8; // Rows - A
        const int N = 8; // Cols - B
        const int K = 8; // Cols-A,Rows-B
        int elts_A = M * K;
        int elts_B = K * N;
        int elts_C = M * N;
        int mem_size_A = elts_A * sizeof(float);
        int mem_size_B = elts_B * sizeof(float);
        int mem_size_C = elts_C * sizeof(float);

        // Allocate host memory
        float* h_A, * h_B, * h_C, * h_C_cpu;
        h_A = (float*)malloc(mem_size_A);
        h_B = (float*)malloc(mem_size_B);
        h_C = (float*)malloc(mem_size_C);
        h_C_cpu = (float*)malloc(mem_size_C);

        // Initialize Host Data
        std::fill(h_A, h_A + elts_A, 2);
        std::fill(h_B, h_B + elts_B, 2);
        std::fill(h_C, h_C + elts_C, 0);
        std::fill(h_C_cpu, h_C_cpu + elts_C, 0);

        /*printMatrix(h_A,N);
        printMatrix(h_B, N);*/
        // CPU implementation
        //cpuMatMul(h_A, h_B, h_C_cpu, M, N, K);
        //cpuMatTranspose(h_A, h_C_cpu, M, K);


        // Device Memory
        CudaMemory<float> d_A(mem_size_A), d_B(mem_size_B), d_C(mem_size_C);

        // Copy data to device
        CUDA_CALL(cudaMemcpy(d_A.get(), h_A, mem_size_A, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_B.get(), h_B, mem_size_B, cudaMemcpyHostToDevice));

        // Define grid and block dimensions
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH); // tile size
        dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
            (M + dimBlock.y - 1) / dimBlock.y);

        //printKernelConfig(dimGrid, dimBlock);

        // Launch the matrix multiplication kernel
        if (option == MATMUL_NAIVE)
        {
            matmulKernelNaive << <dimGrid, dimBlock >> > (d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
        }
        else if (option == MATMUL_SHARED)
        {
            matmulKernelShared << <dimGrid, dimBlock >> > (d_A.get(), d_B.get(), d_C.get(), M, N, K);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
        }
        else if (option == MATMUL_cuBLAS)
        {
            // Create cuBLAS handle
            cublasHandle_t handle;
            CUBLAS_CALL(cublasCreate(&handle));

            // cuBLAS uses column-major order, but our matrices are in row-major order
            // We can compute B * A instead of A * B to compensate for this
            // C(m, n) = B(n, k) * A(k, m) is equivalent to C ^ T = A ^ T * B ^ T in row - major

            float alpha = 1.0f;
            float beta = 0.0f;

            CUBLAS_CALL(cublasSgemm(
                handle,                  // cuBLAS handle
                CUBLAS_OP_N,             // Operation on B: no transpose
                CUBLAS_OP_N,             // Operation on A: no transpose
                N, M, K,                 // Dimensions (n x m = n x k * k x m)
                &alpha,                  // Scaling factor for multiplication
                d_B.get(), N,            // B matrix and leading dimension
                d_A.get(), K,            // A matrix and leading dimension
                &beta,                   // Scaling factor for C
                d_C.get(), N             // C matrix and leading dimension
            ));

            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());

            CUBLAS_CALL(cublasDestroy(handle));
        }
        else if (option == MATMUL_cuBLAS)
        {


        }


        // Copy result back to host
        CUDA_CALL(cudaMemcpy(h_C, d_C.get(), mem_size_C, cudaMemcpyDeviceToHost));

        // Verify GPU results
        verifyResults(h_C_cpu, h_C, N);
        printMatrix(h_C, M, N);

        // cudaDeviceReset - for profiling
        CUDA_CALL(cudaDeviceReset());

        // Free Memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_cpu);
    }
    catch (std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

