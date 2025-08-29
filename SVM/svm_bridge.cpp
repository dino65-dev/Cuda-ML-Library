#include <cmath>

extern "C" {
    // Struct definition (matching the Python ctypes structure)
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

    // Simple mock implementation for testing
    // This will work without CUDA complications
    struct SimpleSVM {
        SVMParams params;
        float* support_vectors;
        float* alpha;
        float bias;
        int n_support;
        int n_features;
        bool is_fitted;
        
        SimpleSVM(const SVMParams& p) : params(p), support_vectors(nullptr), 
                                      alpha(nullptr), bias(0.0f), n_support(0), 
                                      n_features(0), is_fitted(false) {}
        
        ~SimpleSVM() {
            if (support_vectors) delete[] support_vectors;
            if (alpha) delete[] alpha;
        }
        
        void fit(const float* X, const float* y, int n_samples, int n_features) {
            this->n_features = n_features;
            // Simple mock: store a subset as support vectors
            n_support = (n_samples < 100) ? n_samples : 100;
            
            support_vectors = new float[n_support * n_features];
            alpha = new float[n_support];
            
            // Copy first n_support samples as support vectors
            for (int i = 0; i < n_support; i++) {
                for (int j = 0; j < n_features; j++) {
                    support_vectors[i * n_features + j] = X[i * n_features + j];
                }
                alpha[i] = 1.0f; // Mock alpha values
            }
            
            bias = 0.1f; // Mock bias
            is_fitted = true;
        }
        
        void predict(const float* X, float* predictions, int n_samples, int n_features) {
            if (!is_fitted) return;
            
            // Simple linear prediction for testing
            for (int i = 0; i < n_samples; i++) {
                float sum = bias;
                for (int sv = 0; sv < n_support && sv < 10; sv++) {
                    float dot_product = 0.0f;
                    for (int f = 0; f < n_features; f++) {
                        dot_product += X[i * n_features + f] * support_vectors[sv * n_features + f];
                    }
                    sum += alpha[sv] * dot_product * 0.01f; // Scale down for stability
                }
                
                // Apply sign for classification
                if (params.svm_type == 0 || params.svm_type == 1) {
                    predictions[i] = (sum > 0) ? 1.0f : -1.0f;
                } else {
                    predictions[i] = sum;
                }
            }
        }
        
        void predict_proba(const float* X, float* probabilities, int n_samples, int n_features) {
            if (!is_fitted) return;
            
            // Get decision values first
            predict(X, probabilities, n_samples, n_features);
            
            // Convert to probabilities using sigmoid
            for (int i = 0; i < n_samples; i++) {
                float decision = probabilities[i];
                probabilities[i] = 1.0f / (1.0f + expf(-decision));
            }
        }
    };

    // Exported C functions
    void* create_svm(SVMParams* params) {
        try {
            return new SimpleSVM(*params);
        } catch (...) {
            return nullptr;
        }
    }
    
    void destroy_svm(void* svm_ptr) {
        if (svm_ptr) {
            delete static_cast<SimpleSVM*>(svm_ptr);
        }
    }
    
    void fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<SimpleSVM*>(svm_ptr)->fit(X, y, n_samples, n_features);
        }
    }
    
    void predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<SimpleSVM*>(svm_ptr)->predict(X, predictions, n_samples, n_features);
        }
    }
    
    void predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<SimpleSVM*>(svm_ptr)->predict_proba(X, probabilities, n_samples, n_features);
        }
    }
}