#pragma once

#include <cstddef> // for size_t

// A simple, non-thread-safe memory pool for CUDA allocations.
// Reduces cudaMalloc/cudaFree overhead for frequent, variable-sized allocations.
class CudaMemoryPool {
public:
    // Allocates a large chunk of memory from the GPU upon construction.
    explicit CudaMemoryPool(size_t pool_size);
    ~CudaMemoryPool();

    // Allocate a block of memory from the pool.
    void* allocate(size_t size);

    // Resets the pool, making all allocated memory available again without freeing it.
    void reset();

    // Returns the currently used memory in the pool.
    size_t get_usage() const;

private:
    void* pool_ptr_;
    size_t pool_size_;
    size_t current_offset_;
};