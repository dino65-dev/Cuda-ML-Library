#include "svm_cuda_optimized.cuh"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>

#if USE_CUDA == 0

// CPU fallback implementations when CUDA is not available

// Simple CPU memory pool for fallback
class CpuMemoryPool {
private:
    std::vector<std::unique_ptr<std::vector<float>>> allocations;
    size_t total_allocated;
    
public:
    CpuMemoryPool(size_t initial_size = 1024 * 1024 * 1024) : total_allocated(0) {
        // Reserve some capacity
        allocations.reserve(100);
    }
    
    ~CpuMemoryPool() = default;
    
    void* allocate(size_t size) {
        size_t count = size / sizeof(float);
        auto ptr = std::make_unique<std::vector<float>>(count);
        void* result = ptr->data();
        total_allocated += size;
        allocations.push_back(std::move(ptr));
        return result;
    }
    
    void deallocate(void* ptr) {
        // For simplicity, we don't actually deallocate in CPU fallback
        // Real implementation would track and free memory
    }
    
    void reset() {
        allocations.clear();
        total_allocated = 0;
    }
    
    size_t get_usage() const { return total_allocated; }
};

// CPU fallback SVM implementation
class CpuSVM {
private:
    SVMParams params;
    std::vector<float> support_vectors;
    std::vector<float> alpha;
    std::vector<int> support_indices;
    float bias;
    int n_support;
    int n_features;
    bool is_fitted;
    float training_time;
    size_t memory_usage;
    CpuMemoryPool memory_pool;
    
    float rbf_kernel(const float* x1, const float* x2, int n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_features; i++) {
            float diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return expf(-params.gamma * sum);
    }
    
    float linear_kernel(const float* x1, const float* x2, int n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_features; i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }
    
    float poly_kernel(const float* x1, const float* x2, int n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_features; i++) {
            sum += x1[i] * x2[i];
        }
        return powf(params.gamma * sum + params.coef0, params.degree);
    }
    
    float sigmoid_kernel(const float* x1, const float* x2, int n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_features; i++) {
            sum += x1[i] * x2[i];
        }
        return tanhf(params.gamma * sum + params.coef0);
    }
    
    float compute_kernel(const float* x1, const float* x2, int n_features) {
        switch (params.kernel_type) {
            case 0: return linear_kernel(x1, x2, n_features);
            case 1: return rbf_kernel(x1, x2, n_features);
            case 2: return poly_kernel(x1, x2, n_features);
            case 3: return sigmoid_kernel(x1, x2, n_features);
            default: return rbf_kernel(x1, x2, n_features);
        }
    }
    
public:
    CpuSVM(const SVMParams& p) : params(p), bias(0.0f), n_support(0), 
                                 n_features(0), is_fitted(false), 
                                 training_time(0.0f), memory_usage(0) {}
    
    ~CpuSVM() = default;
    
    void fit(const float* X, const float* y, int n_samples, int n_features_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        this->n_features = n_features_;
        
        // Simple SMO-like algorithm for CPU fallback
        // This is a simplified implementation for demonstration
        
        // Initialize alpha values
        std::vector<float> alpha_vec(n_samples, 0.0f);
        
        // Simple training loop (simplified SMO)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n_samples - 1);
        
        for (int iter = 0; iter < std::min(params.max_iter, 100); iter++) {
            bool changed = false;
            
            for (int i = 0; i < std::min(n_samples, 50); i++) {
                int idx = dis(gen);
                
                // Compute decision value
                float decision = bias;
                for (int j = 0; j < n_samples; j++) {
                    if (alpha_vec[j] > 0) {
                        float kernel_val = compute_kernel(&X[idx * n_features_], 
                                                        &X[j * n_features_], n_features_);
                        decision += alpha_vec[j] * y[j] * kernel_val;
                    }
                }
                
                float error = decision - y[idx];
                
                // Simple alpha update
                if (abs(error) > params.tolerance) {
                    float old_alpha = alpha_vec[idx];
                    alpha_vec[idx] = std::max(0.0f, std::min(params.C, 
                                    alpha_vec[idx] - 0.01f * error * y[idx]));
                    
                    if (abs(alpha_vec[idx] - old_alpha) > 1e-5) {
                        changed = true;
                    }
                }
            }
            
            if (!changed) break;
        }
        
        // Extract support vectors
        support_vectors.clear();
        alpha.clear();
        support_indices.clear();
        
        for (int i = 0; i < n_samples; i++) {
            if (alpha_vec[i] > 1e-5) {
                support_indices.push_back(i);
                alpha.push_back(alpha_vec[i]);
                for (int j = 0; j < n_features_; j++) {
                    support_vectors.push_back(X[i * n_features_ + j]);
                }
            }
        }
        
        n_support = support_indices.size();
        
        // Compute bias (simplified)
        bias = 0.0f;
        if (n_support > 0) {
            for (int i = 0; i < std::min(n_support, 10); i++) {
                float sum = 0.0f;
                for (int j = 0; j < n_support; j++) {
                    float kernel_val = compute_kernel(&support_vectors[i * n_features_], 
                                                    &support_vectors[j * n_features_], n_features_);
                    sum += alpha[j] * y[support_indices[j]] * kernel_val;
                }
                bias += y[support_indices[i]] - sum;
            }
            bias /= std::min(n_support, 10);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        training_time = std::chrono::duration<float>(end_time - start_time).count();
        memory_usage = (support_vectors.size() + alpha.size()) * sizeof(float);
        
        is_fitted = true;
    }
    
    void predict(const float* X, float* predictions, int n_samples, int n_features_) {
        if (!is_fitted) {
            for (int i = 0; i < n_samples; i++) {
                predictions[i] = 0.0f;
            }
            return;
        }
        
        for (int i = 0; i < n_samples; i++) {
            float sum = bias;
            for (int j = 0; j < n_support; j++) {
                float kernel_val = compute_kernel(&X[i * n_features_], 
                                                &support_vectors[j * n_features_], n_features_);
                sum += alpha[j] * kernel_val;
            }
            
            // Apply appropriate output format
            if (params.svm_type == 0 || params.svm_type == 1) {
                // Classification
                predictions[i] = (sum > 0) ? 1.0f : -1.0f;
            } else {
                // Regression
                predictions[i] = sum;
            }
        }
    }
    
    void predict_batch_async(const float* X, float* predictions, int n_samples, int n_features_) {
        // For CPU fallback, async is the same as regular predict
        predict(X, predictions, n_samples, n_features_);
    }
    
    float get_training_time() const { return training_time; }
    size_t get_memory_usage() const { return memory_usage; }
};

// CPU implementations for the CUDA functions
CudaMemoryPool::CudaMemoryPool(size_t initial_size) {
    // CPU fallback doesn't need member initialization
}

CudaMemoryPool::~CudaMemoryPool() {
    // Cleanup for CPU fallback
}

void* CudaMemoryPool::allocate(size_t size) {
    return malloc(size);
}

void CudaMemoryPool::deallocate(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

void CudaMemoryPool::reset() {
    // Nothing to reset for simple malloc/free
}

size_t CudaMemoryPool::get_usage() const {
    return 0; // Can't track in simple implementation
}

// OptimizedCudaSVM CPU implementation via delegation to CpuSVM
OptimizedCudaSVM::OptimizedCudaSVM(const SVMParams& params) {
#if USE_CUDA == 0
    cpu_impl_ = new CpuSVM(params);
#endif
}

OptimizedCudaSVM::~OptimizedCudaSVM() {
#if USE_CUDA == 0
    delete static_cast<CpuSVM*>(cpu_impl_);
#endif
}

void OptimizedCudaSVM::fit(const float* X, const float* y, int n_samples, int n_features) {
#if USE_CUDA == 0
    static_cast<CpuSVM*>(cpu_impl_)->fit(X, y, n_samples, n_features);
#endif
}

void OptimizedCudaSVM::predict(const float* X, float* predictions, int n_samples, int n_features) {
#if USE_CUDA == 0
    static_cast<CpuSVM*>(cpu_impl_)->predict(X, predictions, n_samples, n_features);
#endif
}

void OptimizedCudaSVM::predict_batch_async(const float* X, float* predictions, int n_samples, int n_features) {
#if USE_CUDA == 0
    static_cast<CpuSVM*>(cpu_impl_)->predict_batch_async(X, predictions, n_samples, n_features);
#endif
}

float OptimizedCudaSVM::get_training_time() const {
#if USE_CUDA == 0
    return static_cast<CpuSVM*>(cpu_impl_)->get_training_time();
#else
    return 0.0f;
#endif
}

size_t OptimizedCudaSVM::get_memory_usage() const {
#if USE_CUDA == 0
    return static_cast<CpuSVM*>(cpu_impl_)->get_memory_usage();
#else
    return 0;
#endif
}

#endif // USE_CUDA == 0