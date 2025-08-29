#include "random_forest_cuda_optimized.cuh"
#include "cuda_memory_pool.cuh" // Assumed from SVM implementation
#include "cuda_error_check.cuh" // Assumed from SVM implementation
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

// Forward declare kernels from rf_kernels.cu
void launch_build_trees_kernel(
    TreeNode* d_trees, const float* d_X, const int* d_y,
    int n_samples, int n_features, int n_estimators, int max_depth,
    int max_nodes_per_tree, int max_features, int n_classes,
    cudaStream_t stream);

void launch_predict_kernel(
    const TreeNode* d_trees, const float* d_X_test, int* d_predictions,
    int n_samples, int n_features, int n_estimators, int max_nodes_per_tree,
    int n_classes, cudaStream_t stream);


static std::string last_error_msg;

OptimizedCudaRandomForest::OptimizedCudaRandomForest(const RFParams& params)
    : params_(params), n_samples_(0), n_features_(0), d_trees_(nullptr) {

    try {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t pool_size = std::min((size_t)(free_mem * 0.8), (size_t)(2ULL * 1024 * 1024 * 1024));
        memory_pool_ = std::make_unique<CudaMemoryPool>(pool_size);

        max_nodes_per_tree_ = (1 << (params_.max_depth + 1)) - 1;

    } catch (const std::runtime_error& e) {
        last_error_msg = "RF Initialization failed: " + std::string(e.what());
        throw;
    }
}

OptimizedCudaRandomForest::~OptimizedCudaRandomForest() {
    cleanup_memory();
    cudaStreamDestroy(stream_);
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void OptimizedCudaRandomForest::initialize_memory(int n_samples, int n_features) {
    n_samples_ = n_samples;
    n_features_ = n_features;
    
    // Reset pool to clear memory from previous fits
    memory_pool_->reset();

    size_t trees_size = (size_t)params_.n_estimators * max_nodes_per_tree_ * sizeof(TreeNode);
    d_trees_ = static_cast<TreeNode*>(memory_pool_->allocate(trees_size));
    CUDA_CHECK(cudaMemsetAsync(d_trees_, 0, trees_size, stream_));
}

void OptimizedCudaRandomForest::cleanup_memory() {
    memory_pool_->reset();
    d_trees_ = nullptr;
}

void OptimizedCudaRandomForest::fit(const float* X, const int* y, int n_samples, int n_features) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    initialize_memory(n_samples, n_features);

    // Allocate and transfer training data
    float* d_X = static_cast<float*>(memory_pool_->allocate(n_samples * n_features * sizeof(float)));
    int* d_y = static_cast<int*>(memory_pool_->allocate(n_samples * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_X, X, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y, n_samples * sizeof(int), cudaMemcpyHostToDevice, stream_));

    // Launch the main tree building kernel
    launch_build_trees_kernel(
        d_trees_, d_X, d_y,
        n_samples, n_features, params_.n_estimators, params_.max_depth,
        max_nodes_per_tree_, params_.max_features, params_.n_classes,
        stream_
    );

    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
}

void OptimizedCudaRandomForest::predict(const float* X, int* predictions, int n_samples, int n_features) {
    if (!d_trees_) {
        throw std::runtime_error("Model must be fitted before prediction.");
    }

    float* d_X_test = static_cast<float*>(memory_pool_->allocate(n_samples * n_features * sizeof(float)));
    int* d_predictions = static_cast<int*>(memory_pool_->allocate(n_samples * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_X_test, X, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice, stream_));

    launch_predict_kernel(
        d_trees_, d_X_test, d_predictions,
        n_samples, n_features, params_.n_estimators, max_nodes_per_tree_,
        params_.n_classes, stream_
    );

    CUDA_CHECK(cudaMemcpyAsync(predictions, d_predictions, n_samples * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // The memory for d_X_test and d_predictions will be freed on the next fit() or destruction
}

float OptimizedCudaRandomForest::get_training_time() const {
    float elapsed_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
    return elapsed_time / 1000.0f; // Convert ms to seconds
}

size_t OptimizedCudaRandomForest::get_memory_usage() const {
    return memory_pool_->get_usage();
}

// C-style API implementation
extern "C" {
    void* create_optimized_rf(const RFParams* params) {
        try {
            return new OptimizedCudaRandomForest(*params);
        } catch (const std::exception& e) {
            last_error_msg = e.what();
            return nullptr;
        }
    }

    void destroy_optimized_rf(void* rf_ptr) {
        delete static_cast<OptimizedCudaRandomForest*>(rf_ptr);
    }

    int fit_optimized_rf(void* rf_ptr, const float* X, const int* y, int n_samples, int n_features) {
        try {
            static_cast<OptimizedCudaRandomForest*>(rf_ptr)->fit(X, y, n_samples, n_features);
            return 0; // Success
        } catch (const std::exception& e) {
            last_error_msg = e.what();
            return 1; // Failure
        }
    }

    int predict_optimized_rf(void* rf_ptr, const float* X, int* predictions, int n_samples, int n_features) {
        try {
            static_cast<OptimizedCudaRandomForest*>(rf_ptr)->predict(X, predictions, n_samples, n_features);
            return 0; // Success
        } catch (const std::exception& e) {
            last_error_msg = e.what();
            return 1; // Failure
        }
    }

    float get_rf_training_time(void* rf_ptr) { return static_cast<OptimizedCudaRandomForest*>(rf_ptr)->get_training_time(); }
    size_t get_rf_memory_usage(void* rf_ptr) { return static_cast<OptimizedCudaRandomForest*>(rf_ptr)->get_memory_usage(); }
    const char* get_rf_last_error() { return last_error_msg.c_str(); }
    void clear_rf_error() { last_error_msg.clear(); }
}