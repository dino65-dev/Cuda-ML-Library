#ifndef SVM_CUDA_CUH
#define SVM_CUDA_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <memory>
#include "svm_bridge.h"  // Include common definitions

class CudaSVM {
private:
    SVMParams params_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;
    
    // Device memory pointers
    float* d_X_;
    float* d_y_;
    float* d_alpha_;
    float* d_kernel_cache_;
    float* d_gradient_;
    int* d_active_set_;
    
    // Host data
    int n_samples_;
    int n_features_;
    int n_sv_;
    float bias_;
    
    // Support vectors
    thrust::device_vector<float> sv_X_;
    thrust::device_vector<float> sv_alpha_;
    thrust::device_vector<int> sv_indices_;

public:
    CudaSVM(const SVMParams& params);
    ~CudaSVM();
    
    void fit(const float* X, const float* y, int n_samples, int n_features);
    void predict(const float* X, float* predictions, int n_samples, int n_features);
    void predict_proba(const float* X, float* probabilities, int n_samples, int n_features);
    
    float get_bias() const { return bias_; }
    int get_n_support_vectors() const { return n_sv_; }
    
private:
    void initialize_memory(int n_samples, int n_features);
    void cleanup_memory();
    void smo_algorithm();
    void compute_kernel_matrix();
    void update_gradient();
    bool select_working_set(int& i, int& j);
    void update_alpha_pair(int i, int j);
    void compute_bias();
    float kernel_function(const float* xi, const float* xj, int n_features);
};

// CUDA kernel declarations
__global__ void rbf_kernel_matrix(const float* X, float* K, int n_samples, 
                                 int n_features, float gamma);

__global__ void linear_kernel_matrix(const float* X, float* K, int n_samples, 
                                    int n_features);

__global__ void polynomial_kernel_matrix(const float* X, float* K, int n_samples, 
                                        int n_features, float gamma, 
                                        float coef0, int degree);

__global__ void sigmoid_kernel_matrix(const float* X, float* K, int n_samples, 
                                     int n_features, float gamma, float coef0);

__global__ void compute_gradient_kernel(const float* K, const float* y, 
                                       const float* alpha, float* gradient, 
                                       int n_samples, float C);

__global__ void update_working_set_kernel(const float* gradient, const float* y,
                                         const float* alpha, int* working_set,
                                         int n_samples, float C, float tolerance);

__global__ void predict_kernel(const float* sv_X, const float* sv_alpha,
                              const float* X_test, float* predictions,
                              int n_sv, int n_test, int n_features,
                              float bias, KernelType kernel_type,
                              float gamma, float coef0, int degree);

#endif // SVM_CUDA_CUH
