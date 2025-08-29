#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include "svm_cuda.cuh"

extern "C" {
    // Use the SVMParams from the CUDA header
    using SVMParams = ::SVMParams;

    // Enhanced CUDA SVM wrapper with proper SMO algorithm
    struct CudaSVMWrapper {
        ::SVMParams cuda_params;  // CUDA SVM parameters
        std::unique_ptr<CudaSVM> cuda_svm;
        bool is_fitted;

        CudaSVMWrapper(const ::SVMParams& p) : is_fitted(false) {
            // Convert Python-style params to CUDA SVM params
            cuda_params.svm_type = static_cast<SVMType>(p.svm_type);
            cuda_params.kernel_type = static_cast<KernelType>(p.kernel_type);
            cuda_params.C = p.C;
            cuda_params.epsilon = p.epsilon;
            cuda_params.gamma = p.gamma;
            cuda_params.coef0 = p.coef0;
            cuda_params.degree = p.degree;
            cuda_params.nu = p.nu;
            cuda_params.tolerance = p.tolerance;
            cuda_params.max_iter = p.max_iter;
            cuda_params.shrinking = p.shrinking;
            cuda_params.probability = p.probability;

            try {
                cuda_svm = std::make_unique<CudaSVM>(cuda_params);
            } catch (const std::exception& e) {
                std::cerr << "CUDA SVM initialization failed, falling back to CPU: " << e.what() << std::endl;
                cuda_svm = nullptr; // Will use CPU fallback in methods
            }
        }

        void fit(const float* X, const float* y, int n_samples, int n_features) {
            if (cuda_svm) {
                try {
                    cuda_svm->fit(X, y, n_samples, n_features);
                    is_fitted = true;
                } catch (const std::exception& e) {
                    std::cerr << "CUDA SVM fit error, falling back to CPU: " << e.what() << std::endl;
                    cpu_fallback_fit(X, y, n_samples, n_features);
                }
            } else {
                cpu_fallback_fit(X, y, n_samples, n_features);
            }
        }

        void predict(const float* X, float* predictions, int n_samples, int n_features) {
            if (!is_fitted) return;

            if (cuda_svm) {
                try {
                    cuda_svm->predict(X, predictions, n_samples, n_features);
                } catch (const std::exception& e) {
                    std::cerr << "CUDA SVM predict error, falling back to CPU: " << e.what() << std::endl;
                    cpu_fallback_predict(X, predictions, n_samples, n_features);
                }
            } else {
                cpu_fallback_predict(X, predictions, n_samples, n_features);
            }
        }

        void predict_proba(const float* X, float* probabilities, int n_samples, int n_features) {
            if (!is_fitted) return;

            if (cuda_svm) {
                try {
                    cuda_svm->predict_proba(X, probabilities, n_samples, n_features);
                } catch (const std::exception& e) {
                    std::cerr << "CUDA SVM predict_proba error, falling back to CPU: " << e.what() << std::endl;
                    cpu_fallback_predict_proba(X, probabilities, n_samples, n_features);
                }
            } else {
                cpu_fallback_predict_proba(X, probabilities, n_samples, n_features);
            }
        }

    private:
        // Enhanced CPU SVM implementation with SMO algorithm
        std::vector<float> X_train_;
        std::vector<float> y_train_;
        std::vector<float> alphas_;
        std::vector<float> error_cache_;
        float bias_ = 0.0f;
        int n_samples_ = 0;
        int n_features_ = 0;
        bool is_fitted_ = false;

        // SMO algorithm parameters
        float tolerance_ = 1e-3f;
        float epsilon_ = 1e-5f;
        int max_passes_ = 10;

        // Kernel functions
        float kernel_function(const float* x1, const float* x2, int n_features) {
            switch (cuda_params.kernel_type) {
                case KernelType::LINEAR:
                    return linear_kernel(x1, x2, n_features);
                case KernelType::RBF:
                    return rbf_kernel(x1, x2, n_features, cuda_params.gamma);
                case KernelType::POLYNOMIAL:
                    return polynomial_kernel(x1, x2, n_features, cuda_params.gamma, cuda_params.coef0, cuda_params.degree);
                case KernelType::SIGMOID:
                    return sigmoid_kernel(x1, x2, n_features, cuda_params.gamma, cuda_params.coef0);
                default:
                    return rbf_kernel(x1, x2, n_features, cuda_params.gamma);
            }
        }

        float linear_kernel(const float* x1, const float* x2, int n_features) {
            float sum = 0.0f;
            for (int i = 0; i < n_features; i++) {
                sum += x1[i] * x2[i];
            }
            return sum;
        }

        float rbf_kernel(const float* x1, const float* x2, int n_features, float gamma) {
            float sum = 0.0f;
            for (int i = 0; i < n_features; i++) {
                float diff = x1[i] - x2[i];
                sum += diff * diff;
            }
            return expf(-gamma * sum);
        }

        float polynomial_kernel(const float* x1, const float* x2, int n_features, float gamma, float coef0, int degree) {
            float sum = 0.0f;
            for (int i = 0; i < n_features; i++) {
                sum += x1[i] * x2[i];
            }
            return powf(gamma * sum + coef0, degree);
        }

        float sigmoid_kernel(const float* x1, const float* x2, int n_features, float gamma, float coef0) {
            float sum = 0.0f;
            for (int i = 0; i < n_features; i++) {
                sum += x1[i] * x2[i];
            }
            return tanhf(gamma * sum + coef0);
        }

        // Decision function
        float decision_function(int sample_idx) {
            float sum = bias_;
            for (int i = 0; i < n_samples_; i++) {
                if (alphas_[i] > epsilon_) {
                    float kernel_val = kernel_function(&X_train_[sample_idx * n_features_],
                                                     &X_train_[i * n_features_], n_features_);
                    sum += alphas_[i] * y_train_[i] * kernel_val;
                }
            }
            return sum;
        }

        // Simplified SMO algorithm for demonstration
        void smo_algorithm() {
            alphas_.resize(n_samples_, 0.0f);
            error_cache_.resize(n_samples_, 0.0f);

            int num_changed = 0;
            bool examine_all = true;
            int passes = 0;

            while ((num_changed > 0 || examine_all) && passes < cuda_params.max_iter) {
                num_changed = 0;

                if (examine_all) {
                    // Examine all examples
                    for (int i = 0; i < n_samples_; i++) {
                        if (examine_example(i)) {
                            num_changed++;
                        }
                    }
                } else {
                    // Examine examples where alpha is not 0 or C
                    for (int i = 0; i < n_samples_; i++) {
                        if ((alphas_[i] > epsilon_ && alphas_[i] < cuda_params.C - epsilon_) ||
                            (alphas_[i] < -epsilon_ && alphas_[i] > -cuda_params.C + epsilon_)) {
                            if (examine_example(i)) {
                                num_changed++;
                            }
                        }
                    }
                }

                if (examine_all) {
                    examine_all = false;
                } else if (num_changed == 0) {
                    examine_all = true;
                }

                passes++;
            }

            // Calculate final bias
            calculate_bias();
        }

        bool examine_example(int i2) {
            float y2 = y_train_[i2];
            float alph2 = alphas_[i2];
            float E2 = (alph2 > epsilon_ && alph2 < cuda_params.C - epsilon_) ?
                      error_cache_[i2] : decision_function(i2) - y2;

            float r2 = E2 * y2;

            if ((r2 < -tolerance_ && alph2 < cuda_params.C - epsilon_) ||
                (r2 > tolerance_ && alph2 > epsilon_)) {

                // Simple second choice heuristic
                int i1 = find_second_choice(i2, E2);
                if (i1 >= 0 && take_step(i1, i2)) {
                    return true;
                }

                // Loop over all possible i1
                for (int i = 0; i < n_samples_; i++) {
                    if (take_step(i, i2)) {
                        return true;
                    }
                }
            }

            return false;
        }

        int find_second_choice(int i2, float E2) {
            int i1 = -1;
            float max_delta = 0.0f;

            for (int i = 0; i < n_samples_; i++) {
                if (alphas_[i] > epsilon_ && alphas_[i] < cuda_params.C - epsilon_) {
                    float E1 = error_cache_[i];
                    float delta = fabsf(E1 - E2);
                    if (delta > max_delta) {
                        max_delta = delta;
                        i1 = i;
                    }
                }
            }

            return i1;
        }

        bool take_step(int i1, int i2) {
            if (i1 == i2) return false;

            float alph1 = alphas_[i1];
            float alph2 = alphas_[i2];
            float y1 = y_train_[i1];
            float y2 = y_train_[i2];
            float E1 = (alph1 > epsilon_ && alph1 < cuda_params.C - epsilon_) ?
                      error_cache_[i1] : decision_function(i1) - y1;
            float E2 = (alph2 > epsilon_ && alph2 < cuda_params.C - epsilon_) ?
                      error_cache_[i2] : decision_function(i2) - y2;

            float s = y1 * y2;

            // Compute L and H
            float L, H;
            if (y1 != y2) {
                L = std::max(0.0f, alph2 - alph1);
                H = std::min(cuda_params.C, cuda_params.C + alph2 - alph1);
            } else {
                L = std::max(0.0f, alph1 + alph2 - cuda_params.C);
                H = std::min(cuda_params.C, alph1 + alph2);
            }

            if (L >= H) return false;

            // Compute eta
            float k11 = kernel_function(&X_train_[i1 * n_features_], &X_train_[i1 * n_features_], n_features_);
            float k12 = kernel_function(&X_train_[i1 * n_features_], &X_train_[i2 * n_features_], n_features_);
            float k22 = kernel_function(&X_train_[i2 * n_features_], &X_train_[i2 * n_features_], n_features_);
            float eta = 2.0f * k12 - k11 - k22;

            float a2;
            if (eta < 0.0f) {
                a2 = alph2 - y2 * (E1 - E2) / eta;
                if (a2 < L) a2 = L;
                else if (a2 > H) a2 = H;
            } else {
                // Use middle of range
                a2 = (L + H) / 2.0f;
            }

            if (fabsf(a2 - alph2) < epsilon_ * (a2 + alph2 + epsilon_)) return false;

            // Update alpha2
            float a1 = alph1 + s * (alph2 - a2);

            // Update model
            alphas_[i1] = a1;
            alphas_[i2] = a2;

            // Update error cache
            for (int i = 0; i < n_samples_; i++) {
                if (alphas_[i] > epsilon_ && alphas_[i] < cuda_params.C - epsilon_) {
                    error_cache_[i] += y1 * (a1 - alph1) *
                                     kernel_function(&X_train_[i * n_features_], &X_train_[i1 * n_features_], n_features_) +
                                     y2 * (a2 - alph2) *
                                     kernel_function(&X_train_[i * n_features_], &X_train_[i2 * n_features_], n_features_);
                }
            }

            error_cache_[i1] = 0.0f;
            error_cache_[i2] = 0.0f;

            return true;
        }

        void calculate_bias() {
            float b_temp = 0.0f;
            int count = 0;

            for (int i = 0; i < n_samples_; i++) {
                if (alphas_[i] > epsilon_ && alphas_[i] < cuda_params.C - epsilon_) {
                    float sum = 0.0f;
                    for (int j = 0; j < n_samples_; j++) {
                        if (alphas_[j] > epsilon_) {
                            sum += alphas_[j] * y_train_[j] *
                                   kernel_function(&X_train_[i * n_features_], &X_train_[j * n_features_], n_features_);
                        }
                    }
                    b_temp += y_train_[i] - sum;
                    count++;
                }
            }

            if (count > 0) {
                bias_ = b_temp / count;
            } else {
                bias_ = 0.0f; // Default bias
            }
        }

        void cpu_fallback_fit(const float* X, const float* y, int n_samples, int n_features) {
            n_samples_ = n_samples;
            n_features_ = n_features;

            // Store training data
            X_train_.resize(n_samples * n_features);
            y_train_.resize(n_samples);
            std::copy(X, X + n_samples * n_features, X_train_.begin());
            std::copy(y, y + n_samples, y_train_.begin());

            // Initialize parameters
            tolerance_ = cuda_params.tolerance;
            max_passes_ = cuda_params.max_iter / 10; // Adjust for SMO passes

            // Run SMO algorithm
            smo_algorithm();

            is_fitted_ = true;
            is_fitted = true; // For backward compatibility
        }

        void cpu_fallback_predict(const float* X, float* predictions, int n_samples, int n_features) {
            for (int i = 0; i < n_samples; i++) {
                float sum = bias_;
                for (int j = 0; j < n_samples_; j++) {
                    if (alphas_[j] > epsilon_) {
                        float kernel_val = kernel_function(&X[i * n_features], &X_train_[j * n_features_], n_features);
                        sum += alphas_[j] * y_train_[j] * kernel_val;
                    }
                }

                // Apply different logic based on SVM type
                if (cuda_params.svm_type == SVMType::C_SVC || cuda_params.svm_type == SVMType::NU_SVC) {
                    // Classification: apply sign function
                    predictions[i] = (sum > 0) ? 1.0f : -1.0f;
                } else {
                    // Regression: return raw decision value
                    predictions[i] = sum;
                }
            }
        }

        void cpu_fallback_predict_proba(const float* X, float* probabilities, int n_samples, int n_features) {
            // Get decision values first
            cpu_fallback_predict(X, probabilities, n_samples, n_features);

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
            return new CudaSVMWrapper(*params);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create CUDA SVM: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void destroy_svm(void* svm_ptr) {
        if (svm_ptr) {
            delete static_cast<CudaSVMWrapper*>(svm_ptr);
        }
    }

    void fit_svm(void* svm_ptr, float* X, float* y, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVMWrapper*>(svm_ptr)->fit(X, y, n_samples, n_features);
        }
    }

    void predict_svm(void* svm_ptr, float* X, float* predictions, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVMWrapper*>(svm_ptr)->predict(X, predictions, n_samples, n_features);
        }
    }

    void predict_proba_svm(void* svm_ptr, float* X, float* probabilities, int n_samples, int n_features) {
        if (svm_ptr) {
            static_cast<CudaSVMWrapper*>(svm_ptr)->predict_proba(X, probabilities, n_samples, n_features);
        }
    }
}