#include "svm_cuda_optimized.cuh"
#include <iostream>
#include <algorithm>

CudaMemoryPool::CudaMemoryPool(size_t initial_size) 
    : head_(nullptr), total_allocated_(0), pool_size_(initial_size) {
    
    // Allocate initial pool
    void* pool_ptr;
    CUDA_CHECK(cudaMalloc(&pool_ptr, pool_size_));
    
    head_ = new MemoryBlock{pool_ptr, pool_size_, true, nullptr};
}

CudaMemoryPool::~CudaMemoryPool() {
    MemoryBlock* current = head_;
    while (current) {
        if (current == head_) {
            CUDA_CHECK(cudaFree(current->ptr));
        }
        MemoryBlock* next = current->next;
        delete current;
        current = next;
    }
}

void* CudaMemoryPool::allocate(size_t size) {
    // Align to 256 bytes for optimal memory access
    size = (size + 255) & ~255;
    
    MemoryBlock* current = head_;
    while (current) {
        if (current->is_free && current->size >= size) {
            current->is_free = false;
            
            // Split block if significantly larger
            if (current->size > size + 1024) {
                MemoryBlock* new_block = new MemoryBlock{
                    static_cast<char*>(current->ptr) + size,
                    current->size - size,
                    true,
                    current->next
                };
                current->next = new_block;
                current->size = size;
            }
            
            total_allocated_ += size;
            return current->ptr;
        }
        current = current->next;
    }
    
    // No suitable block found, allocate new memory
    void* new_ptr;
    CUDA_CHECK(cudaMalloc(&new_ptr, size));
    
    MemoryBlock* new_block = new MemoryBlock{new_ptr, size, false, head_};
    head_ = new_block;
    total_allocated_ += size;
    
    return new_ptr;
}

void CudaMemoryPool::deallocate(void* ptr) {
    MemoryBlock* current = head_;
    while (current) {
        if (current->ptr == ptr) {
            current->is_free = true;
            total_allocated_ -= current->size;
            
            // Coalesce adjacent free blocks
            MemoryBlock* next = current->next;
            while (next && next->is_free && 
                   static_cast<char*>(current->ptr) + current->size == next->ptr) {
                current->size += next->size;
                current->next = next->next;
                delete next;
                next = current->next;
            }
            break;
        }
        current = current->next;
    }
}

void CudaMemoryPool::reset() {
    MemoryBlock* current = head_;
    while (current) {
        current->is_free = true;
        current = current->next;
    }
    total_allocated_ = 0;
}
