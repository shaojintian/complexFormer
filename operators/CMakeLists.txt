cmake_minimum_required(VERSION 3.18)
project(RopeExtension LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 查找 PyTorch
find_package(Torch REQUIRED)

# 添加 CUDA 和 C++ 源文件
set(SOURCES
    rope.cpp
    rope.cu
)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# 生成共享库
add_library(rope_extension SHARED ${SOURCES})

# 链接 PyTorch 和 CUDA 库
target_include_directories(rope_extension PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(rope_extension PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# 设置输出路径
set_target_properties(rope_extension PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    PREFIX ""  # 不添加 "lib" 前缀
    SUFFIX ".so"  # 生成 .so 文件
)