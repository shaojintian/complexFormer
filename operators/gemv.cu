#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 256

__global__ void gemv_kernel(float* A, float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float value = 0.0f;
        for (int col = 0; col < N; col++) {
            value += A[row * N + col] * x[col];
        }
        y[row] = value;
    }
}

extern "C" void gemv(float* A, float* x, float* y, int M, int N) {
    dim3 blockDim(TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE);

    gemv_kernel<<<gridDim, blockDim>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}