#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define TILE_SIZE 256

__global__ void ffn_kernel(float* input, float* weight, float* bias, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float value = 0.0f;
        for (int col = 0; col < N; col++) {
            value += input[row * N + col] * weight[col];
        }
        output[row] = fmaxf(0.0f, value + bias[row]); // ReLU 激活函数
    }
}

extern "C" void ffn(float* input, float* weight, float* bias, float* output, int M, int N) {
    dim3 blockDim(TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE);

    ffn_kernel<<<gridDim, blockDim>>>(input, weight, bias, output, M, N);
    cudaDeviceSynchronize();
}