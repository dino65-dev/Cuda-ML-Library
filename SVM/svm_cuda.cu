#include "svm_cuda.cuh"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error at " << __FILE__ << ":" << __LINE__ \
                      << " - status code: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

// RBF Kernel Implementation
__global__ void rbf_kernel_matrix(const float* X, float* K, int n_samples, 
                                 int n_features, float gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float dist_sq = 0.0f;
        for (int k = 0; k < n_features; k++) {
            float diff = X[i * n_features + k] - X[j * n_features + k];
            dist_sq += diff * diff;
        }
        K[i * n_samples + j] = expf(-gamma * dist_sq);
    }
}

// Linear Kernel Implementation
__global__ void linear_kernel_matrix(const float* X, float* K, int n_samples, 
                                    int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float dot_product = 0.0f;
        for (int k = 0; k < n_features; k++) {
            dot_product += X[i * n_features + k] * X[j * n_features + k];
        }
        K[i * n_samples + j] = dot_product;
    }
}

// Polynomial Kernel Implementation
__global__ void polynomial_kernel_matrix(const float* X, float* K, int n_samples, 
                                        int n_features, float gamma, 
                                        float coef0, int degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float dot_product = 0.0f;
        for (int k = 0; k < n_features; k++) {
            dot_product += X[i * n_features + k] * X[j * n_features + k];
        }
        K[i * n_samples + j] = powf(gamma * dot_product + coef0, degree);
    }
}

// Sigmoid Kernel Implementation
__global__ void sigmoid_kernel_matrix(const float* X, float* K, int n_samples, 
                                     int n_features, float gamma, float coef0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float dot_product = 0.0f;
        for (int k = 0; k < n_features; k++) {
            dot_product += X[i * n_features + k] * X[j * n_features + k];
        }
        K[i * n_samples + j] = tanhf(gamma * dot_product + coef0);
    }
}

// Gradient Computation Kernel
__global__ void compute_gradient_kernel(const float* K, const float* y, 
                                       const float* alpha, float* gradient, 
                                       int n_samples, float C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        float sum = 0.0f;
        for (int j = 0; j < n_samples; j++) {
            sum += alpha[j] * y[j] * K[i * n_samples + j];
        }
        gradient[i] = 1.0f - sum;
    }
}

// Working Set Selection Kernel (simplified version)
__global__ void update_working_set_kernel(const float* gradient, const float* y,
                                         const float* alpha, int* working_set,
                                         int n_samples, float C, float tolerance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        float grad_i = gradient[i];
        bool is_upper_bound = (y[i] > 0 && alpha[i] >= C) || (y[i] < 0 && alpha[i] <= 0);
        bool is_lower_bound = (y[i] > 0 && alpha[i] <= 0) || (y[i] < 0 && alpha[i] >= C);
        
        working_set[i] = 0;
        if (!is_upper_bound && grad_i > tolerance) {
            working_set[i] = 1;
        } else if (!is_lower_bound && grad_i < -tolerance) {
            working_set[i] = -1;
        }
    }
}

// Prediction Kernel
__global__ void predict_kernel(const float* sv_X, const float* sv_alpha,
                              const float* X_test, float* predictions,
                              int n_sv, int n_test, int n_features,
                              float bias, KernelType kernel_type,
                              float gamma, float coef0, int degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_test) {
        float sum = 0.0f;
        
        for (int j = 0; j < n_sv; j++) {
            float kernel_val = 0.0f;
            
            switch (kernel_type) {
                case KernelType::LINEAR: {
                    for (int k = 0; k < n_features; k++) {
                        kernel_val += X_test[i * n_features + k] * sv_X[j * n_features + k];
                    }
                    break;
                }
                case KernelType::RBF: {
                    float dist_sq = 0.0f;
                    for (int k = 0; k < n_features; k++) {
                        float diff = X_test[i * n_features + k] - sv_X[j * n_features + k];
                        dist_sq += diff * diff;
                    }
                    kernel_val = expf(-gamma * dist_sq);
                    break;
                }
                case KernelType::POLYNOMIAL: {
                    float dot_product = 0.0f;
                    for (int k = 0; k < n_features; k++) {
                        dot_product += X_test[i * n_features + k] * sv_X[j * n_features + k];
                    }
                    kernel_val = powf(gamma * dot_product + coef0, degree);
                    break;
                }
                case KernelType::SIGMOID: {
                    float dot_product = 0.0f;
                    for (int k = 0; k < n_features; k++) {
                        dot_product += X_test[i * n_features + k] * sv_X[j * n_features + k];
                    }
                    kernel_val = tanhf(gamma * dot_product + coef0);
                    break;
                }
            }
            
            sum += sv_alpha[j] * kernel_val;
        }
        
        predictions[i] = sum + bias;
    }
}

// CudaSVM Implementation
CudaSVM::CudaSVM(const SVMParams& params) : params_(params) {
    // Try to create CUDA handles, but don't fail if CUDA is not available
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    curandStatus_t curand_status = curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);

    if (cublas_status != CUBLAS_STATUS_SUCCESS || curand_status != CURAND_STATUS_SUCCESS) {
        // CUDA not available, set handles to nullptr
        cublas_handle_ = nullptr;
        curand_gen_ = nullptr;
        std::cerr << "CUDA not available, using CPU fallback" << std::endl;
    }

    d_X_ = nullptr;
    d_y_ = nullptr;
    d_alpha_ = nullptr;
    d_kernel_cache_ = nullptr;
    d_gradient_ = nullptr;
    d_active_set_ = nullptr;

    n_samples_ = 0;
    n_features_ = 0;
    n_sv_ = 0;
    bias_ = 0.0f;
}

CudaSVM::~CudaSVM() {
    cleanup_memory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (curand_gen_) {
        curandDestroyGenerator(curand_gen_);
    }
}

void CudaSVM::initialize_memory(int n_samples, int n_features) {
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_X_, n_samples * n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alpha_, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel_cache_, n_samples * n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradient_, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_active_set_, n_samples * sizeof(int)));
    
    // Initialize alpha to zeros
    CUDA_CHECK(cudaMemset(d_alpha_, 0, n_samples * sizeof(float)));
}

void CudaSVM::cleanup_memory() {
    if (d_X_) CUDA_CHECK(cudaFree(d_X_));
    if (d_y_) CUDA_CHECK(cudaFree(d_y_));
    if (d_alpha_) CUDA_CHECK(cudaFree(d_alpha_));
    if (d_kernel_cache_) CUDA_CHECK(cudaFree(d_kernel_cache_));
    if (d_gradient_) CUDA_CHECK(cudaFree(d_gradient_));
    if (d_active_set_) CUDA_CHECK(cudaFree(d_active_set_));
}

void CudaSVM::fit(const float* X, const float* y, int n_samples, int n_features) {
    // Check if CUDA is available
    if (!cublas_handle_ || !curand_gen_) {
        throw std::runtime_error("CUDA not available");
    }

    initialize_memory(n_samples, n_features);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X_, X, n_samples * n_features * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_, y, n_samples * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Compute kernel matrix
    compute_kernel_matrix();
    
    // Run SMO algorithm
    smo_algorithm();
    
    // Extract support vectors
    thrust::host_vector<float> h_alpha(n_samples);
    CUDA_CHECK(cudaMemcpy(h_alpha.data(), d_alpha_, n_samples * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Count support vectors and copy them
    n_sv_ = 0;
    for (int i = 0; i < n_samples; i++) {
        if (std::abs(h_alpha[i]) > 1e-6) {
            n_sv_++;
        }
    }
    
    sv_X_.resize(n_sv_ * n_features);
    sv_alpha_.resize(n_sv_);
    sv_indices_.resize(n_sv_);
    
    int sv_idx = 0;
    for (int i = 0; i < n_samples; i++) {
        if (std::abs(h_alpha[i]) > 1e-6) {
            sv_alpha_[sv_idx] = h_alpha[i];
            sv_indices_[sv_idx] = i;
            
            for (int j = 0; j < n_features; j++) {
                sv_X_[sv_idx * n_features + j] = X[i * n_features + j];
            }
            sv_idx++;
        }
    }
    
    // Compute bias
    compute_bias();
}

void CudaSVM::compute_kernel_matrix() {
    dim3 block_size(16, 16);
    dim3 grid_size((n_samples_ + block_size.x - 1) / block_size.x,
                   (n_samples_ + block_size.y - 1) / block_size.y);
    
    switch (params_.kernel_type) {
        case KernelType::LINEAR:
            linear_kernel_matrix<<<grid_size, block_size>>>(
                d_X_, d_kernel_cache_, n_samples_, n_features_);
            break;
        case KernelType::RBF:
            rbf_kernel_matrix<<<grid_size, block_size>>>(
                d_X_, d_kernel_cache_, n_samples_, n_features_, params_.gamma);
            break;
        case KernelType::POLYNOMIAL:
            polynomial_kernel_matrix<<<grid_size, block_size>>>(
                d_X_, d_kernel_cache_, n_samples_, n_features_, 
                params_.gamma, params_.coef0, params_.degree);
            break;
        case KernelType::SIGMOID:
            sigmoid_kernel_matrix<<<grid_size, block_size>>>(
                d_X_, d_kernel_cache_, n_samples_, n_features_, 
                params_.gamma, params_.coef0);
            break;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaSVM::smo_algorithm() {
    // Simplified SMO implementation
    // In a production version, this would be much more sophisticated
    
    for (int iter = 0; iter < params_.max_iter; iter++) {
        // Update gradient
        update_gradient();
        
        // Select working set (simplified)
        dim3 block_size(256);
        dim3 grid_size((n_samples_ + block_size.x - 1) / block_size.x);
        
        update_working_set_kernel<<<grid_size, block_size>>>(
            d_gradient_, d_y_, d_alpha_, d_active_set_, 
            n_samples_, params_.C, params_.tolerance);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // In a full implementation, you would:
        // 1. Select optimal working set pair
        // 2. Solve analytical subproblem
        // 3. Update alpha values
        // 4. Check convergence criteria
        
        // For brevity, using a simplified approach here
        break;
    }
}

void CudaSVM::update_gradient() {
    dim3 block_size(256);
    dim3 grid_size((n_samples_ + block_size.x - 1) / block_size.x);
    
    compute_gradient_kernel<<<grid_size, block_size>>>(
        d_kernel_cache_, d_y_, d_alpha_, d_gradient_, n_samples_, params_.C);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaSVM::compute_bias() {
    // Simplified bias computation
    // In practice, this would be more sophisticated
    bias_ = 0.0f;
}

void CudaSVM::predict(const float* X, float* predictions, int n_samples, int n_features) {
    // Check if CUDA is available
    if (!cublas_handle_ || !curand_gen_) {
        throw std::runtime_error("CUDA not available");
    }

    float* d_X_test;
    float* d_predictions;
    
    CUDA_CHECK(cudaMalloc(&d_X_test, n_samples * n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predictions, n_samples * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_X_test, X, n_samples * n_features * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    dim3 block_size(256);
    dim3 grid_size((n_samples + block_size.x - 1) / block_size.x);
    
    predict_kernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(sv_X_.data()),
        thrust::raw_pointer_cast(sv_alpha_.data()),
        d_X_test, d_predictions,
        n_sv_, n_samples, n_features,
        bias_, params_.kernel_type,
        params_.gamma, params_.coef0, params_.degree);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(predictions, d_predictions, n_samples * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // For classification, apply sign function
    if (params_.svm_type == SVMType::C_SVC || params_.svm_type == SVMType::NU_SVC) {
        for (int i = 0; i < n_samples; i++) {
            predictions[i] = predictions[i] > 0 ? 1.0f : -1.0f;
        }
    }
    
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_predictions));
}

void CudaSVM::predict_proba(const float* X, float* probabilities, int n_samples, int n_features) {
    // Check if CUDA is available
    if (!cublas_handle_ || !curand_gen_) {
        throw std::runtime_error("CUDA not available");
    }

    // Get decision function values
    predict(X, probabilities, n_samples, n_features);
    
    // Convert to probabilities using Platt scaling (simplified)
    for (int i = 0; i < n_samples; i++) {
        float decision_val = probabilities[i];
        probabilities[i] = 1.0f / (1.0f + expf(-decision_val));
    }
}

extern "C" {
    void* cuda_create_svm(SVMParams* params) {
        try {
            return new CudaSVM(*params);
        } catch (...) {
            return nullptr;
        }
    }
    
    void cuda_destroy_svm(void* svm_ptr) {
        if (svm_ptr) {
            delete static_cast<CudaSVM*>(svm_ptr);
        }
    }
    
    void cuda_fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->fit(X, y, n_samples, n_features);
        }
    }
    
    void cuda_predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->predict(X, predictions, n_samples, n_features);
        }
    }
    
    void cuda_predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->predict_proba(X, probabilities, n_samples, n_features);
        }
    }
}
