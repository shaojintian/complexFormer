#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <string>
#include "buffer/cudabuffer.h" // 假设 CUDABuffer 定义在 cudabuffer.h 中

class Tensor {
public:
    // 构造函数：创建张量
    Tensor(CUDAMemoryPool& memory_pool, const std::vector<size_t>& shape, size_t element_size)
        : memory_pool_(memory_pool), shape_(shape), element_size_(element_size), buffer_(nullptr) {
        // 计算张量的总大小（字节数）
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        size_t total_size = total_elements * element_size;

        // 分配显存
        buffer_ = std::make_unique<CUDABuffer>(memory_pool_, total_size);
    }

    // 禁止拷贝构造和赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 允许移动构造和赋值
    Tensor(Tensor&& other) noexcept
        : memory_pool_(other.memory_pool_), shape_(std::move(other.shape_)),
          element_size_(other.element_size_), buffer_(std::move(other.buffer_)) {}

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            memory_pool_ = other.memory_pool_;
            shape_ = std::move(other.shape_);
            element_size_ = other.element_size_;
            buffer_ = std::move(other.buffer_);
        }
        return *this;
    }

    // 获取张量的形状
    const std::vector<size_t>& shape() const {
        return shape_;
    }

    // 获取张量的总元素数
    size_t num_elements() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    }

    // 获取张量的总大小（字节数）
    size_t size_in_bytes() const {
        return num_elements() * element_size_;
    }

    // 获取张量的数据指针
    void* data() const {
        return buffer_->data();
    }

    // 获取张量的元素大小
    size_t element_size() const {
        return element_size_;
    }

    // 打印张量信息
    std::string info() const {
        std::string info = "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            info += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                info += ", ";
            }
        }
        info += "], element_size=" + std::to_string(element_size_) + " bytes)";
        return info;
    }

private:
    CUDAMemoryPool& memory_pool_;               // 引用显存池
    std::vector<size_t> shape_;                // 张量的形状
    size_t element_size_;                      // 每个元素的大小（字节）
    std::unique_ptr<CUDABuffer> buffer_;       // 显存缓冲区
};

