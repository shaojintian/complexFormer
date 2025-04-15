#include <cuda_runtime.h>

__global__ void int8_gemv_kernel(
    int32_t* y,           // 输出向量 (int32)
    const int8_t* A,      // 量化矩阵 (m x n, 行优先)
    const int8_t* x,      // 量化向量 (n x 1)
    const float* scale_A, // 矩阵A的缩放因子
    const float* scale_x, // 向量x的缩放因子
    int m, int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    // 使用共享内存缓存向量x
    extern __shared__ int8_t s_x[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // 混合精度累加
    int32_t sum = 0;
    for (int col = 0; col < n; ++col) {
        sum += static_cast<int32_t>(A[row * n + col]) * static_cast<int32_t>(s_x[col]);
    }

    // 应用缩放因子并存储
    y[row] = sum;
}