// Don't include svm_cuda.cuh directly to avoid compilation errors

struct SVMParams {
    int svm_type;
    int kernel_type;
    float C;
    float epsilon;
    float gamma;
    float coef0;
    int degree;
    float nu;
    float tolerance;
    int max_iter;
    bool shrinking;
    bool probability;
};

// These functions will be implemented in the CUDA file
extern void* cuda_create_svm(SVMParams* params);
extern void cuda_destroy_svm(void* svm_ptr);
extern void cuda_fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features);
extern void cuda_predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features);
extern void cuda_predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features);

extern "C" {
    void* create_svm(SVMParams* params) {
        return cuda_create_svm(params);
    }
    
    void destroy_svm(void* svm_ptr) {
        cuda_destroy_svm(svm_ptr);
    }
    
    void fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        cuda_fit_svm(svm_ptr, X, y, n_samples, n_features);
    }
    
    void predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        cuda_predict_svm(svm_ptr, X, predictions, n_samples, n_features);
    }
    
    void predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features) {
        cuda_predict_proba_svm(svm_ptr, X, probabilities, n_samples, n_features);
    }
}