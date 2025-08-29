#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <curand.h>

class CudaMemoryPool; // Assumed to exist from SVM implementation

// Parameters for the Random Forest model, matching Python wrapper
struct RFParams {
    int n_estimators;
    int max_depth;
    int min_samples_split;
    int max_features;
    int n_classes;
    bool bootstrap;
    unsigned long long seed;
    int svm_type; // Placeholder for future compatibility
};

// Represents a single node in a decision tree on the GPU
struct TreeNode {
    int feature_index;      // Feature used for splitting
    float threshold;        // Threshold value for the split
    int left_child;         // Index of the left child node in the tree's node array
    int right_child;        // Index of the right child node
    int class_prediction;   // Predicted class if this is a leaf node (-1 if not a leaf)
};

// C-style interface for Python ctypes wrapper
extern "C" {
    void* create_optimized_rf(const RFParams* params);
    void destroy_optimized_rf(void* rf_ptr);
    int fit_optimized_rf(void* rf_ptr, const float* X, const int* y, int n_samples, int n_features);
    int predict_optimized_rf(void* rf_ptr, const float* X, int* predictions, int n_samples, int n_features);
    float get_rf_training_time(void* rf_ptr);
    size_t get_rf_memory_usage(void* rf_ptr);
    const char* get_rf_last_error();
    void clear_rf_error();
}

// Main C++ class for the CUDA-accelerated Random Forest
class OptimizedCudaRandomForest {
public:
    explicit OptimizedCudaRandomForest(const RFParams& params);
    ~OptimizedCudaRandomForest();

    void fit(const float* X, const int* y, int n_samples, int n_features);
    void predict(const float* X, int* predictions, int n_samples, int n_features);

    float get_training_time() const;
    size_t get_memory_usage() const;

private:
    void initialize_memory(int n_samples, int n_features);
    void cleanup_memory();

    RFParams params_;
    int n_samples_;
    int n_features_;
    int max_nodes_per_tree_;

    // CUDA resources
    cudaStream_t stream_;
    cudaEvent_t start_event_, stop_event_;
    std::unique_ptr<CudaMemoryPool> memory_pool_;

    // Device data (managed by memory_pool_)
    TreeNode* d_trees_; // All trees stored contiguously
};