#include "cuda_memory_pool.cuh"
#include "cuda_error_check.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

CudaMemoryPool::CudaMemoryPool(size_t pool_size)
    : pool_ptr_(nullptr), pool_size_(pool_size), current_offset_(0) {
    if (pool_size_ > 0) {
        CUDA_CHECK(cudaMalloc(&pool_ptr_, pool_size_));
    }
}

CudaMemoryPool::~CudaMemoryPool() {
    if (pool_ptr_) {
        // It's good practice to check the error, even in a destructor.
        cudaFree(pool_ptr_);
    }
}

void* CudaMemoryPool::allocate(size_t size) {
    // Align to 128 bytes for performance
    const size_t alignment = 128;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    if (current_offset_ + aligned_size > pool_size_) {
        throw std::bad_alloc();
    }

    void* ptr = static_cast<char*>(pool_ptr_) + current_offset_;
    current_offset_ += aligned_size;
    return ptr;
}

void CudaMemoryPool::reset() {
    current_offset_ = 0;
}

size_t CudaMemoryPool::get_usage() const {
    return current_offset_;
}