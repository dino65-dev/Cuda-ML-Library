#include "svm_cuda_optimized.cuh"

StreamingKernelCache::StreamingKernelCache(int max_size, CudaMemoryPool* pool)
    : max_cache_size_(max_size), cache_size_(0), memory_pool_(pool) {
    
    cache_buffer_ = static_cast<float*>(
        memory_pool_->allocate(max_cache_size_ * sizeof(float))
    );
    cache_indices_ = static_cast<int*>(
        memory_pool_->allocate(max_cache_size_ * sizeof(int))
    );
    
    // Initialize cache indices to -1 (invalid)
    CUDA_CHECK(cudaMemset(cache_indices_, -1, max_cache_size_ * sizeof(int)));
}

StreamingKernelCache::~StreamingKernelCache() {
    if (cache_buffer_) memory_pool_->deallocate(cache_buffer_);
    if (cache_indices_) memory_pool_->deallocate(cache_indices_);
}

float* StreamingKernelCache::get_kernel_row(int i, const float* X, int n_samples, 
                                          int n_features, const SVMParams& params) {
    // Check if row is already cached
    for (int j = 0; j < cache_size_; j++) {
        int cached_idx;
        CUDA_CHECK(cudaMemcpy(&cached_idx, &cache_indices_[j], sizeof(int), 
                             cudaMemcpyDeviceToHost));
        if (cached_idx == i) {
            return &cache_buffer_[j * n_samples];
        }
    }
    
    // Row not cached, compute and store
    int cache_pos = cache_size_ % max_cache_size_;
    float* row_ptr = &cache_buffer_[cache_pos * n_samples];
    
    // Compute kernel row on GPU
    dim3 block_size(256);
    dim3 grid_size((n_samples + block_size.x - 1) / block_size.x);
    
    switch (params.kernel_type) {
        case KernelType::RBF:
            optimized_rbf_kernel_streaming<<<grid_size, block_size>>>(
                &X[i * n_features], X, row_ptr,
                1, n_samples, n_features, params.gamma, 0
            );
            break;
        // Add other kernel types...
    }
    
    // Update cache metadata
    CUDA_CHECK(cudaMemcpy(&cache_indices_[cache_pos], &i, sizeof(int), 
                         cudaMemcpyHostToDevice));
    
    if (cache_size_ < max_cache_size_) cache_size_++;
    
    return row_ptr;
}

void StreamingKernelCache::invalidate() {
    CUDA_CHECK(cudaMemset(cache_indices_, -1, max_cache_size_ * sizeof(int)));
    cache_size_ = 0;
}
