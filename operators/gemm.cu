#include <cuda_runtime.h>
#include <iostream>

//vectorized and shared mem optimized matrix multiplication
#define TILE_SIZE 16

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void gemm_kernel_optimized(float* A, float* B, float* C, int M, int N, int K) {
    // 定义共享内存，用于存储 A 和 B 的子块
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // 当前线程的行和列索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 用于累积结果
    float value = 0.0f;

    // 遍历 A 和 B 的所有子块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 和 B 的子块到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步线程，确保所有线程都加载完成
        __syncthreads();

        // 计算子块的部分结果
        for (int i = 0; i < TILE_SIZE; i++) {
            value += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }

        // 同步线程，确保所有线程完成当前子块的计算
        __syncthreads();
    }

    // 将结果写入矩阵 C
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}



extern "C" void gemm_cuda(float* A, float* B, float* C, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);//256thread x-col y-row
    // how much block per grid to cover the whole matrix
    // gridDim.x = ceil(N / TILE_SIZE)
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel_optimized<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}