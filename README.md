# CUDA ML Library

A high-performance CUDA-accelerated Machine Learning library with automatic CPU fallback support, featuring optimized Support Vector Machine implementations for both classification and regression tasks.

## 🚀 Features

- **GPU Acceleration**: Full CUDA support for NVIDIA GPUs with Compute Capability 7.0+
- **Automatic CPU Fallback**: Seamless fallback to optimized CPU implementation when CUDA is unavailable
- **Cross-Platform Compatibility**: Linux, Windows, and macOS support
- **Multiple SVM Types**: Classification (C-SVC, Nu-SVC) and Regression (Epsilon-SVR, Nu-SVR)
- **Multiple Kernel Functions**: Linear, RBF, Polynomial, and Sigmoid kernels
- **Advanced Algorithms**: SMO (Sequential Minimal Optimization) algorithm implementation
- **Memory Optimization**: Efficient GPU memory management with pooling
- **Easy Integration**: Scikit-learn compatible API

## 📋 System Requirements

### Hardware Requirements
- **GPU (Optional)**: NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX 20 series, GTX 1650+, Tesla V100+)
- **CPU (Required)**: Any modern x86_64 processor
- **RAM**: 4GB+ system memory (8GB+ recommended for large datasets)

### Software Requirements
- **CUDA Toolkit** (Optional): Version 12.0+ for GPU acceleration
- **Python**: 3.8+
- **Dependencies**: numpy ≥1.19.0, scikit-learn ≥1.0.0

### Supported Environments
- **GPU-Accelerated**: Systems with CUDA-capable NVIDIA GPUs
- **CPU-Only**: Any system (automatic fallback when CUDA unavailable)
- **Cloud Platforms**: Google Colab, AWS, Azure, etc.
- **Cross-Platform**: Linux, Windows, macOS

## 🛠️ Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install cuda-ml-library
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/dino65-dev/Cuda_ML_Library.git
cd Cuda_ML_Library

# Install dependencies
pip install numpy scikit-learn

# Build the CUDA library
cd SVM
make clean
make

# Install the package
cd ..
pip install -e .
```

The build process will:
- Auto-detect CUDA availability and GPU architecture
- Compile CUDA kernels when GPU is available
- Create CPU fallback implementation when CUDA is unavailable
- Generate optimized shared libraries with universal compatibility

## 🚀 Quick Start

### Classification Example

```python
from SVM.cuda_svm import CudaSVC
import numpy as np

# Generate sample data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create and train the model (automatically uses CUDA if available)
svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale')
svc.fit(X, y)

# Make predictions
predictions = svc.predict(X_test)
probabilities = svc.predict_proba(X_test)  # If probability=True

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
```

### Regression Example

```python
from SVM.cuda_svm import CudaSVR
import numpy as np

# Generate sample data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)

# Create and train the model
svr = CudaSVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='auto')
svr.fit(X, y)

# Make predictions
predictions = svr.predict(X_test)

print(f"R² Score: {r2_score(y_test, predictions)}")
```

## 📚 API Reference

### CudaSVC (Classification)

```python
CudaSVC(
    svm_type='c_svc',     # 'c_svc' or 'nu_svc'
    kernel='rbf',         # 'linear', 'rbf', 'poly', 'sigmoid'
    C=1.0,               # Regularization parameter
    gamma='scale',        # Kernel coefficient
    coef0=0.0,           # Independent term for poly/sigmoid
    degree=3,            # Degree for polynomial kernel
    nu=0.5,              # Nu parameter for nu-SVM
    tolerance=1e-3,      # Tolerance for stopping criterion
    max_iter=1000,       # Maximum iterations
    shrinking=True,      # Use shrinking heuristic
    probability=False    # Enable probability estimates
)
```

### CudaSVR (Regression)

```python
CudaSVR(
    svm_type='epsilon_svr',  # 'epsilon_svr' or 'nu_svr'
    kernel='rbf',            # 'linear', 'rbf', 'poly', 'sigmoid'
    C=1.0,                   # Regularization parameter
    epsilon=0.1,             # Epsilon for epsilon-SVR
    gamma='scale',           # Kernel coefficient
    coef0=0.0,              # Independent term
    degree=3,               # Polynomial degree
    nu=0.5,                 # Nu parameter
    tolerance=1e-3,         # Stopping tolerance
    max_iter=1000          # Maximum iterations
)
```

## 🔧 Advanced Usage

### Hardware Detection

```python
from SVM.cuda_svm import CudaSVC

# The library automatically detects and uses available hardware
svc = CudaSVC()
print("CUDA SVM initialized successfully")

# Hardware detection and optimization happen automatically
svc.fit(X_train, y_train)
```

### Kernel Customization

```python
# RBF Kernel with custom gamma
svc_rbf = CudaSVC(kernel='rbf', gamma=0.001)

# Polynomial Kernel
svc_poly = CudaSVC(kernel='poly', degree=4, coef0=1.0, gamma='auto')

# Linear Kernel (fastest)
svc_linear = CudaSVC(kernel='linear')

# Sigmoid Kernel
svc_sigmoid = CudaSVC(kernel='sigmoid', gamma='scale', coef0=0.0)
```

## ⚠️ Important Notes

### Current Status

- **SVM**: Fully functional and ready for production use
- **HBM_SVM**: Currently in development and may not be fully functional yet

**Please use the standard SVM implementation for all production workloads.**

### Performance Tips

1. **GPU Memory**: Ensure sufficient GPU memory for large datasets
2. **Batch Processing**: For very large datasets, consider batch processing
3. **Kernel Selection**: Linear kernels are fastest, RBF kernels offer good accuracy
4. **Parameter Tuning**: Use cross-validation for optimal parameter selection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: [https://github.com/dino65-dev/Cuda_ML_Library](https://github.com/dino65-dev/Cuda_ML_Library)
- **Issues**: [https://github.com/dino65-dev/Cuda_ML_Library/issues](https://github.com/dino65-dev/Cuda_ML_Library/issues)
- **Documentation**: [Usage Examples](./Usage/)

## 📊 Version

Current Version: **0.1.0**

---

**Made with ❤️ by [dino65-dev](https://github.com/dino65-dev)**
