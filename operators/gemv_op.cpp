#include <torch/extension.h>
#include <vector>

// 声明 CUDA 核函数
extern "C" void gemv(float* A, float* x, float* y, int M, int N);

// C++ 接口函数
void gemv(float* A, float* x, float* y, int M, int N) {
    // 检查输入是否在 GPU 上
    //TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");

    // 调用 CUDA 核函数
    gemv(A, x, y, M, N);
}

// 注册算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &gemv, "GEMV kernel (CUDA)");
}
