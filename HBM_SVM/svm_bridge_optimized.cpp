#include "svm_cuda_optimized.cuh"
#include <iostream>
#include <exception>

// Global error handling
static std::string last_error = "";

extern "C" {
    // Error handling functions
    const char* get_last_error() {
        return last_error.c_str();
    }
    
    void clear_error() {
        last_error.clear();
    }
    
    // SVM management functions
    void* create_optimized_svm(SVMParams* params) {
        try {
            if (!params) {
                last_error = "NULL parameters provided";
                return nullptr;
            }
            
            OptimizedCudaSVM* svm = new OptimizedCudaSVM(*params);
            return static_cast<void*>(svm);
        } catch (const std::exception& e) {
            last_error = "Failed to create SVM: " + std::string(e.what());
            return nullptr;
        } catch (...) {
            last_error = "Unknown error occurred while creating SVM";
            return nullptr;
        }
    }
    
    void destroy_optimized_svm(void* svm_ptr) {
        try {
            if (svm_ptr) {
                OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
                delete svm;
            }
        } catch (...) {
            last_error = "Error occurred while destroying SVM";
        }
    }
    
    int fit_optimized_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        try {
            if (!svm_ptr || !X || !y || n_samples <= 0 || n_features <= 0) {
                last_error = "Invalid parameters for fit operation";
                return -1;
            }
            
            OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
            svm->fit(X, y, n_samples, n_features);
            return 0; // Success
        } catch (const std::exception& e) {
            last_error = "Fit operation failed: " + std::string(e.what());
            return -1;
        } catch (...) {
            last_error = "Unknown error during fit operation";
            return -1;
        }
    }
    
    int predict_optimized_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        try {
            if (!svm_ptr || !X || !predictions || n_samples <= 0 || n_features <= 0) {
                last_error = "Invalid parameters for predict operation";
                return -1;
            }
            
            OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
            svm->predict(X, predictions, n_samples, n_features);
            return 0; // Success
        } catch (const std::exception& e) {
            last_error = "Predict operation failed: " + std::string(e.what());
            return -1;
        } catch (...) {
            last_error = "Unknown error during predict operation";
            return -1;
        }
    }
    
    int predict_batch_async_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        try {
            if (!svm_ptr || !X || !predictions || n_samples <= 0 || n_features <= 0) {
                last_error = "Invalid parameters for async predict operation";
                return -1;
            }
            
            OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
            svm->predict_batch_async(X, predictions, n_samples, n_features);
            return 0; // Success
        } catch (const std::exception& e) {
            last_error = "Async predict operation failed: " + std::string(e.what());
            return -1;
        } catch (...) {
            last_error = "Unknown error during async predict operation";
            return -1;
        }
    }
    
    float get_training_time(void* svm_ptr) {
        try {
            if (!svm_ptr) return -1.0f;
            OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
            return svm->get_last_training_time();
        } catch (...) {
            return -1.0f;
        }
    }
    
    size_t get_memory_usage(void* svm_ptr) {
        try {
            if (!svm_ptr) return 0;
            OptimizedCudaSVM* svm = static_cast<OptimizedCudaSVM*>(svm_ptr);
            return svm->get_memory_usage();
        } catch (...) {
            return 0;
        }
    }
    
    // Utility functions
    int check_cuda_available() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess && device_count > 0) ? 1 : 0;
    }
    
    void get_gpu_info(int* device_count, char* device_name, int name_length) {
        try {
            cudaGetDeviceCount(device_count);
            if (*device_count > 0) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                strncpy(device_name, prop.name, name_length - 1);
                device_name[name_length - 1] = '\0';
            }
        } catch (...) {
            *device_count = 0;
            if (name_length > 0) device_name[0] = '\0';
        }
    }
}
