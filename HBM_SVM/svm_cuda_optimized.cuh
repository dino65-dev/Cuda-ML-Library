#ifndef SVM_CUDA_OPTIMIZED_CUH
#define SVM_CUDA_OPTIMIZED_CUH

#ifndef USE_CUDA
#define USE_CUDA 1  // Default to CUDA unless explicitly disabled
#endif

#if USE_CUDA == 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <memory>
#include <vector>

// SVM Parameters structure (compatible with both CUDA and CPU)
struct SVMParams {
    int svm_type = 0;        // 0=C_SVC, 1=NU_SVC, 2=EPSILON_SVR, 3=NU_SVR
    int kernel_type = 1;     // 0=LINEAR, 1=RBF, 2=POLYNOMIAL, 3=SIGMOID
    float C = 1.0f;
    float epsilon = 0.1f;
    float gamma = 0.1f;
    float coef0 = 0.0f;
    int degree = 3;
    float nu = 0.5f;
    float tolerance = 1e-3f;
    int max_iter = 1000;
    bool shrinking = true;
    bool probability = false;
};

#if USE_CUDA == 1
// Memory pool for efficient GPU memory management
class CudaMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_free;
        MemoryBlock* next;
    };
    
    MemoryBlock* head_;
    size_t total_allocated_;
    size_t pool_size_;
    
public:
    CudaMemoryPool(size_t initial_size = 1024 * 1024 * 1024); // 1GB default
    ~CudaMemoryPool();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();
    size_t get_usage() const { return total_allocated_; }
};

// Streaming kernel cache for memory efficiency
class StreamingKernelCache {
private:
    float* cache_buffer_;
    int* cache_indices_;
    int cache_size_;
    int max_cache_size_;
    CudaMemoryPool* memory_pool_;
    
public:
    StreamingKernelCache(int max_size, CudaMemoryPool* pool);
    ~StreamingKernelCache();
    
    float* get_kernel_row(int i, const float* X, int n_samples, int n_features, 
                         const SVMParams& params);
    void invalidate();
};
#else
// CPU fallback declarations
class CudaMemoryPool {
public:
    CudaMemoryPool(size_t initial_size = 0);
    ~CudaMemoryPool();
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();
    size_t get_usage() const;
};
#endif

// Enhanced SVM class with optimizations (works with both CUDA and CPU)
class OptimizedCudaSVM {
private:
#if USE_CUDA == 1
    SVMParams params_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;
    
    // CUDA streams for overlapped execution
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    
    // Memory pool
    std::unique_ptr<CudaMemoryPool> memory_pool_;
    std::unique_ptr<StreamingKernelCache> kernel_cache_;
    
    // Pinned host memory for faster transfers
    float* h_X_pinned_;
    float* h_y_pinned_;
    float* h_predictions_pinned_;
    
    // Device memory (managed by memory pool)
    float* d_X_;
    float* d_y_;
    float* d_alpha_;
    float* d_gradient_;
    __half* d_X_half_;  // Half precision for memory efficiency
    
    // Working set management
    int* d_working_set_;
    int working_set_size_;
    
    // Problem dimensions
    int n_samples_;
    int n_features_;
    int n_sv_;
    float bias_;
    
    // Support vectors (optimized storage)
    thrust::device_vector<float> sv_X_;
    thrust::device_vector<float> sv_alpha_;
    thrust::device_vector<int> sv_indices_;
    
    // Performance monitoring
    cudaEvent_t start_event_, stop_event_;
#else
    // CPU fallback implementation (forward declaration)
    void* cpu_impl_;
#endif

public:
    OptimizedCudaSVM(const SVMParams& params);
    ~OptimizedCudaSVM();
    
    void fit(const float* X, const float* y, int n_samples, int n_features);
    void predict(const float* X, float* predictions, int n_samples, int n_features);
    void predict_batch_async(const float* X, float* predictions, 
                           int n_samples, int n_features);
    
    // Performance metrics
    float get_training_time() const;
    size_t get_memory_usage() const;

private:
#if USE_CUDA == 1
    void initialize_optimized_memory(int n_samples, int n_features);
    void cleanup_optimized_memory();
    void optimized_smo_algorithm();
    void adaptive_working_set_selection();
    void mixed_precision_kernel_computation();
    void compute_kernel_streaming(int row_start, int row_end);
#endif
};

#if USE_CUDA == 1
// Optimized CUDA kernels (only available when CUDA is enabled)
__global__ void optimized_rbf_kernel_streaming(
    const float* X1, const float* X2, float* K,
    int n1, int n2, int n_features, float gamma,
    int row_offset
);

__global__ void half_precision_kernel_computation(
    const __half* X1, const __half* X2, float* K,
    int n1, int n2, int n_features, float gamma
);

__global__ void adaptive_gradient_update(
    const float* kernel_row, const float* y, const float* alpha,
    float* gradient, int n_samples, int row_idx, float C
);

__global__ void vectorized_working_set_selection(
    const float* gradient, const float* y, const float* alpha,
    int* working_set, float* scores, int n_samples, float C, float tolerance
);

__global__ void optimized_prediction_kernel(
    const float* sv_X, const float* sv_alpha, const float* X_test,
    float* predictions, int n_sv, int n_test, int n_features,
    float bias, int kernel_type, float gamma, float coef0, int degree
);

// Memory-efficient matrix operations
template<typename T>
__global__ void blocked_matrix_multiply(
    const T* A, const T* B, T* C,
    int M, int N, int K, int block_size
);
#endif // USE_CUDA == 1

#endif // SVM_CUDA_OPTIMIZED_CUH
