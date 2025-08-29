#ifndef SVM_BRIDGE_H
#define SVM_BRIDGE_H

// Forward declarations and basic types for the SVM bridge
// This header avoids CUDA dependencies for g++ compilation

enum class KernelType {
    LINEAR = 0,
    RBF = 1,
    POLYNOMIAL = 2,
    SIGMOID = 3
};

enum class SVMType {
    C_SVC = 0,    // Classification
    NU_SVC = 1,   // Nu-Classification
    EPSILON_SVR = 2,  // Regression
    NU_SVR = 3    // Nu-Regression
};

struct SVMParams {
    SVMType svm_type = SVMType::C_SVC;
    KernelType kernel_type = KernelType::RBF;
    float C = 1.0f;
    float epsilon = 0.1f;
    float gamma = 0.1f;
    float coef0 = 0.0f;
    int degree = 3;
    float nu = 0.5f;
    float tolerance = 1e-3f;
    int max_iter = 1000;
    bool shrinking = true;
    bool probability = false;
};

#endif // SVM_BRIDGE_H
