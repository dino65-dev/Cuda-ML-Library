#include "svm_cuda.cuh"

extern "C" {
    void* create_svm(SVMParams* params) {
        try {
            return new CudaSVM(*params);
        } catch (...) {
            return nullptr;
        }
    }
    
    void destroy_svm(void* svm_ptr) {
        if (svm_ptr) {
            delete static_cast<CudaSVM*>(svm_ptr);
        }
    }
    
    void fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->fit(X, y, n_samples, n_features);
        }
    }
    
    void predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->predict(X, predictions, n_samples, n_features);
        }
    }
    
    void predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVM*>(svm_ptr)->predict_proba(X, probabilities, n_samples, n_features);
        }
    }
}
