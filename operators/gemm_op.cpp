#include <torch/extension.h>
#include <vector>

// 声明 CUDA 核函数
extern "C" void gemm_cuda(float* A,float* B, float* C, int M, int N, int K);

// C++ 接口函数
void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K) {
    // 检查输入是否在 GPU 上
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");

    // 调用 CUDA 核函数
    gemm_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

// 注册算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "GEMM kernel (CUDA)");
}
