#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void conv2d_kernel(float* input, float* kernel, float* output, int H, int W, int KH, int KW) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        float value = 0.0f;
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j < KW; j++) {
                int input_row = row + i - KH / 2;
                int input_col = col + j - KW / 2;
                if (input_row >= 0 && input_row < H && input_col >= 0 && input_col < W) {
                    value += input[input_row * W + input_col] * kernel[i * KW + j];
                }
            }
        }
        output[row * W + col] = value;
    }
}

extern "C" void conv2d(float* input, float* kernel, float* output, int H, int W, int KH, int KW) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

    conv2d_kernel<<<gridDim, blockDim>>>(input, kernel, output, H, W, KH, KW);
    cudaDeviceSynchronize();
}