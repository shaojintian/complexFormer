from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="operators",
    ext_modules=[
        CUDAExtension(
            name="gemm_op",
            include_dirs=["include"],
            sources=["operators/gemm_op.cpp", "operators/gemm.cu"],
        ),
        CUDAExtension(
            name="gemv_op",
            include_dirs=["include"],
            sources=["operators/gemv_op.cpp", "operators/gemv.cu"],
        ),
        CUDAExtension(
            name="ffn_op",
            include_dirs=["include"],
            sources=["operators/ffn_op.cpp", "operators/ffn.cu"],
        ),
        CUDAExtension(
            name="mha_op",
            include_dirs=["include"],
            sources=["operators/mha_op.cpp", "operators/mha_op.cu"],
        ),
        CUDAExtension(
            name="conv2d_op",
            include_dirs=["include"],
            sources=["operators/conv2d_op.cpp", "operators/conv2d_op.cu"],
        ),
        CUDAExtension(
            name="rope_op",
            include_dirs=["include"],
            sources=["operators/rope_op.cpp", "operators/rope_op.cu"],
        ),
        CUDAExtension(
            name="maxpool2d_op",
            include_dirs=["include"],
            sources=["operators/maxpool2d_op.cpp", "operators/maxpool2d_op.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


