#pragma once

#include <cstddef>
#include <stdexcept>
#include "gpumemoryallocator/mempool.cpp" // 假设 CUDAMemoryPool 定义在 mempool.cpp 中

class CUDABuffer {
public:
    // 构造函数：分配显存
    explicit CUDABuffer(CUDAMemoryPool& memory_pool, size_t size)
        : memory_pool_(memory_pool), size_(size), device_ptr_(nullptr) {
        device_ptr_ = memory_pool_.allocate(size_);
        if (!device_ptr_) {
            throw std::runtime_error("Failed to allocate GPU memory.");
        }
    }

    // 禁止拷贝构造和赋值
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    // 允许移动构造和赋值
    CUDABuffer(CUDABuffer&& other) noexcept
        : memory_pool_(other.memory_pool_), size_(other.size_), device_ptr_(other.device_ptr_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    CUDABuffer& operator=(CUDABuffer&& other) noexcept {
        if (this != &other) {
            release(); // 释放当前资源
            memory_pool_ = other.memory_pool_;
            size_ = other.size_;
            device_ptr_ = other.device_ptr_;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 析构函数：释放显存
    ~CUDABuffer() {
        release();
    }

    // 获取显存指针
    void* data() const {
        return device_ptr_;
    }

    // 获取缓冲区大小
    size_t size() const {
        return size_;
    }

private:
    CUDAMemoryPool& memory_pool_; // 引用显存池
    size_t size_;                 // 缓冲区大小
    void* device_ptr_;            // 显存指针

    // 释放显存
    void release() {
        if (device_ptr_) {
            memory_pool_.deallocate(device_ptr_, size_);
            device_ptr_ = nullptr;
            size_ = 0;
        }
    }
};

#endif // CUDA_BUFFER_H