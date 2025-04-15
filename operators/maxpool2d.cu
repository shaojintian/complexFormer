#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void maxpool2d_kernel(float* input, float* output, int H, int W, int PH, int PW) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H / PH && col < W / PW) {
        float max_value = -FLT_MAX;
        for (int i = 0; i < PH; i++) {
            for (int j = 0; j < PW; j++) {
                int input_row = row * PH + i;
                int input_col = col * PW + j;
                if (input_row < H && input_col < W) {
                    max_value = fmaxf(max_value, input[input_row * W + input_col]);
                }
            }
        }
        output[row * (W / PW) + col] = max_value;
    }
}

extern "C" void maxpool2d(float* input, float* output, int H, int W, int PH, int PW) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W / PW + TILE_SIZE - 1) / TILE_SIZE, (H / PH + TILE_SIZE - 1) / TILE_SIZE);

    maxpool2d_kernel<<<gridDim, blockDim>>>(input, output, H, W, PH, PW);
    cudaDeviceSynchronize();
}