#include "svm_cuda_optimized.cuh"

OptimizedCudaSVM::OptimizedCudaSVM(const SVMParams& params) 
    : params_(params), n_samples_(0), n_features_(0), n_sv_(0), bias_(0.0f) {
    
    // Initialize CUDA resources
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUDA_CHECK(curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT));
    
    // Create CUDA streams for overlapped execution
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUDA_CHECK(cudaStreamCreate(&memory_stream_));
    
    // Initialize memory pool (1GB initial size, adjust based on GPU memory)
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t pool_size = min(free_mem * 0.8, (size_t)(2ULL * 1024 * 1024 * 1024)); // 80% of free or 2GB max
    
    memory_pool_ = std::make_unique<CudaMemoryPool>(pool_size);
    
    // Create performance monitoring events
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
    
    // Initialize pointers
    h_X_pinned_ = nullptr;
    h_y_pinned_ = nullptr;
    h_predictions_pinned_ = nullptr;
    d_X_ = nullptr;
    d_y_ = nullptr;
    d_alpha_ = nullptr;
    d_gradient_ = nullptr;
    d_X_half_ = nullptr;
    d_working_set_ = nullptr;
}

OptimizedCudaSVM::~OptimizedCudaSVM() {
    cleanup_optimized_memory();
    
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(memory_stream_);
    cublasDestroy(cublas_handle_);
    curandDestroyGenerator(curand_gen_);
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void OptimizedCudaSVM::initialize_optimized_memory(int n_samples, int n_features) {
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Allocate pinned host memory for faster transfers
    CUDA_CHECK(cudaMallocHost(&h_X_pinned_, n_samples * n_features * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y_pinned_, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_predictions_pinned_, n_samples * sizeof(float)));
    
    // Allocate device memory using memory pool
    d_X_ = static_cast<float*>(memory_pool_->allocate(n_samples * n_features * sizeof(float)));
    d_y_ = static_cast<float*>(memory_pool_->allocate(n_samples * sizeof(float)));
    d_alpha_ = static_cast<float*>(memory_pool_->allocate(n_samples * sizeof(float)));
    d_gradient_ = static_cast<float*>(memory_pool_->allocate(n_samples * sizeof(float)));
    
    // Allocate half-precision version for memory efficiency
    d_X_half_ = static_cast<__half*>(memory_pool_->allocate(n_samples * n_features * sizeof(__half)));
    
    // Working set for SMO algorithm
    working_set_size_ = min(1024, n_samples / 10); // Adaptive working set size
    d_working_set_ = static_cast<int*>(memory_pool_->allocate(working_set_size_ * sizeof(int)));
    
    // Initialize kernel cache
    int cache_size = min(n_samples / 4, 10000); // Adaptive cache size
    kernel_cache_ = std::make_unique<StreamingKernelCache>(cache_size, memory_pool_.get());
    
    // Initialize alpha to zeros
    CUDA_CHECK(cudaMemsetAsync(d_alpha_, 0, n_samples * sizeof(float), compute_stream_));
}

void OptimizedCudaSVM::cleanup_optimized_memory() {
    if (h_X_pinned_) CUDA_CHECK(cudaFreeHost(h_X_pinned_));
    if (h_y_pinned_) CUDA_CHECK(cudaFreeHost(h_y_pinned_));
    if (h_predictions_pinned_) CUDA_CHECK(cudaFreeHost(h_predictions_pinned_));
    
    // Memory pool handles device memory cleanup
    memory_pool_->reset();
}

void OptimizedCudaSVM::fit(const float* X, const float* y, int n_samples, int n_features) {
    CUDA_CHECK(cudaEventRecord(start_event_, compute_stream_));
    
    initialize_optimized_memory(n_samples, n_features);
    
    // Copy data to pinned memory first
    memcpy(h_X_pinned_, X, n_samples * n_features * sizeof(float));
    memcpy(h_y_pinned_, y, n_samples * sizeof(float));
    
    // Asynchronous memory transfer
    CUDA_CHECK(cudaMemcpyAsync(d_X_, h_X_pinned_, n_samples * n_features * sizeof(float), 
                              cudaMemcpyHostToDevice, memory_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_y_, h_y_pinned_, n_samples * sizeof(float), 
                              cudaMemcpyHostToDevice, memory_stream_));
    
    // Convert to half precision for memory efficiency
    dim3 block_size(256);
    dim3 grid_size((n_samples * n_features + block_size.x - 1) / block_size.x);
    
    // Wait for memory transfer to complete
    CUDA_CHECK(cudaStreamSynchronize(memory_stream_));
    
    // Convert to half precision asynchronously
    // (Implementation would include a kernel to convert float to __half)
    
    // Run optimized SMO algorithm
    optimized_smo_algorithm();
    
    // Extract support vectors (implementation similar to original but optimized)
    // ... (support vector extraction code)
    
    CUDA_CHECK(cudaEventRecord(stop_event_, compute_stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
}

void OptimizedCudaSVM::optimized_smo_algorithm() {
    const int max_iterations = params_.max_iter;
    float tolerance = params_.tolerance;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Adaptive working set selection
        adaptive_working_set_selection();
        
        // Mixed precision kernel computation
        mixed_precision_kernel_computation();
        
        // Update gradients using streaming computation
        // (Implementation details...)
        
        // Check convergence with reduced host-device synchronization
        if (iter % 10 == 0) {
            // Convergence check
            // (Implementation details...)
        }
    }
}

void OptimizedCudaSVM::predict_batch_async(const float* X, float* predictions, 
                                         int n_samples, int n_features) {
    // Asynchronous batch prediction for high throughput
    float* d_X_test = static_cast<float*>(memory_pool_->allocate(n_samples * n_features * sizeof(float)));
    float* d_predictions = static_cast<float*>(memory_pool_->allocate(n_samples * sizeof(float)));
    
    // Asynchronous memory transfers
    CUDA_CHECK(cudaMemcpyAsync(d_X_test, X, n_samples * n_features * sizeof(float), 
                              cudaMemcpyHostToDevice, memory_stream_));
    
    // Launch prediction kernel
    dim3 block_size(256);
    dim3 grid_size((n_samples + block_size.x - 1) / block_size.x);
    
    optimized_prediction_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
        thrust::raw_pointer_cast(sv_X_.data()),
        thrust::raw_pointer_cast(sv_alpha_.data()),
        d_X_test, d_predictions,
        n_sv_, n_samples, n_features,
        bias_, params_.kernel_type,
        params_.gamma, params_.coef0, params_.degree
    );
    
    // Asynchronous result transfer
    CUDA_CHECK(cudaMemcpyAsync(predictions, d_predictions, n_samples * sizeof(float), 
                              cudaMemcpyDeviceToHost, compute_stream_));
    
    // Clean up temporary memory
    memory_pool_->deallocate(d_X_test);
    memory_pool_->deallocate(d_predictions);
}

float OptimizedCudaSVM::get_last_training_time() const {
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
    return elapsed_time / 1000.0f; // Convert to seconds
}

size_t OptimizedCudaSVM::get_memory_usage() const {
    return memory_pool_->get_usage();
}
