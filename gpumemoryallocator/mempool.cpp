#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

class CUDAMemoryPool {
public:
    // 初始化显存池：预分配大块显存，按对齐大小分割
    explicit CUDAMemoryPool(size_t pool_size = 64 * 1024 * 1024, size_t alignment = 256)
        : pool_size_(pool_size), alignment_(alignment) {
        // 预分配一大块显存
        cudaMalloc(&pool_ptr_, pool_size_);
        // 初始化空闲块链表：整个池作为初始空闲块
        free_blocks_.emplace(pool_ptr_, pool_size_);
    }

    ~CUDAMemoryPool() {
        cudaFree(pool_ptr_); // 释放整个池
    }

    // 分配显存（按对齐大小）
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        size = align_size(size);

        // 查找第一个足够大的空闲块
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= size) {
                void* ptr = it->first;
                size_t remaining = it->second - size;

                // 分割块：更新空闲链表
                free_blocks_.erase(it);
                if (remaining > 0) {
                    free_blocks_.emplace(static_cast<char*>(ptr) + size, remaining);
                }

                // 记录已分配块
                allocated_blocks_[ptr] = size;
                return ptr;
            }
        }

        // 池中无足够空间，返回 nullptr（实际可扩展池或抛出异常）
        return nullptr;
    }

    // 释放显存（标记为空闲块，合并相邻块）
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            throw std::runtime_error("Invalid pointer freed");
        }

        size_t size = it->second;
        allocated_blocks_.erase(it);

        // 插入空闲块并尝试合并相邻块
        auto inserted = free_blocks_.emplace(ptr, size).first;
        auto prev = free_blocks_.end();
        auto next = free_blocks_.end();

        if (inserted != free_blocks_.begin()) {
            prev = std::prev(inserted);
            if (static_cast<char*>(prev->first) + prev->second == inserted->first) {
                // 合并前一块
                prev->second += inserted->second;
                free_blocks_.erase(inserted);
                inserted = prev;
            }
        }

        next = std::next(inserted);
        if (next != free_blocks_.end() && static_cast<char*>(inserted->first) + inserted->second == next->first) {
            // 合并后一块
            inserted->second += next->second;
            free_blocks_.erase(next);
        }
    }

private:
    // 对齐内存大小（例如 256 字节对齐）
    size_t align_size(size_t size) {
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }

    void* pool_ptr_ = nullptr;         // 显存池指针
    size_t pool_size_;                  // 池总大小
    size_t alignment_;                  // 对齐要求
    std::mutex mutex_;                  // 线程安全锁

    // 空闲块链表：<起始地址, 块大小>
    std::unordered_map<void*, size_t> free_blocks_;
    // 已分配块记录：<起始地址, 块大小>
    std::unordered_map<void*, size_t> allocated_blocks_;
};