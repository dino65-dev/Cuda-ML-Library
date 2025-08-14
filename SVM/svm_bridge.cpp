
// Add forward declaration:

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

// Forward declare the class
class CudaSVM;

extern "C" {
    void* create_svm(SVMParams* params);
    void destroy_svm(void* svm_ptr);
    void fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features);
    void predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features);
    void predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features);
}