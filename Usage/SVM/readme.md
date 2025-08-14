# CUDA SVM Usage Guide

This directory contains examples and documentation for using the CUDA-accelerated Support Vector Machine (SVM) implementation.

## 🚀 Quick Start

```python
from SVM.cuda_svm import CudaSVC, CudaSVR
import numpy as np

# Classification
svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale')
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)

# Regression  
svr = CudaSVR(C=1.0, epsilon=0.1, kernel='rbf')
svr.fit(X_train, y_train)
predictions = svr.predict(X_test)
```

## 📋 GPU Requirements

### Minimum Requirements
- **NVIDIA GPU**: CUDA Compute Capability 7.0+ (RTX 20 series, GTX 1650+, Tesla V100+)
- **CUDA Toolkit**: Version 12.0 or higher
- **GPU Memory**: Minimum 2GB VRAM (4GB+ recommended)
- **NVIDIA Drivers**: Latest stable version (470.x or newer)

### Recommended Specifications
- **GPU**: RTX 3070/4060 or better
- **CUDA Cores**: 2048+ CUDA cores
- **GPU Memory**: 8GB+ VRAM for large datasets
- **System RAM**: 8GB+ (for data preprocessing)

### Supported Architectures
- **Ampere**: RTX 30/40 series (sm_86, sm_89)
- **Turing**: RTX 20 series, GTX 16 series (sm_75)
- **Volta**: Tesla V100, Titan V (sm_70)
- **Pascal**: GTX 10 series (sm_60, sm_61) - Limited support

## 🛠️ Installation & Setup

### 1. Build the CUDA Library

```bash
cd /path/to/Cuda-ML-Library/SVM
make clean
make
```

The build process will:
- Auto-detect your GPU architecture
- Compile CUDA kernels optimized for your hardware
- Create `libcuda_svm.so` shared library

### 2. Install Dependencies

```bash
pip install numpy scikit-learn
```

### 3. Verify Installation

```bash
python -c "from SVM.cuda_svm import CudaSVC; print('CUDA SVM imported successfully!')"
```

## 📊 Usage Examples

### Basic Classification

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SVM.cuda_svm import CudaSVC

# Generate sample data
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to -1, 1 format
y_train_svm = np.where(y_train == 0, -1, 1)
y_test_svm = np.where(y_test == 0, -1, 1)

# Train CUDA SVM
svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
svc.fit(X_train_scaled, y_train_svm)

# Make predictions
predictions = svc.predict(X_test_scaled)
probabilities = svc.predict_proba(X_test_scaled)
accuracy = svc.score(X_test_scaled, y_test_svm)

print(f"Accuracy: {accuracy:.4f}")
```

### Basic Regression

```python
from sklearn.datasets import make_regression
from SVM.cuda_svm import CudaSVR

# Generate regression data
X, y = make_regression(n_samples=5000, n_features=15, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Train CUDA SVR
svr = CudaSVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')
svr.fit(X_train_scaled, y_train_scaled)

# Evaluate
r2_score = svr.score(X_test_scaled, y_test_scaled)
print(f"R² Score: {r2_score:.4f}")
```

## ⚙️ Parameters

### CudaSVC (Classification)
- `C` (float, default=1.0): Regularization parameter
- `kernel` (str, default='rbf'): Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- `gamma` (str/float, default='scale'): Kernel coefficient ('scale', 'auto', or numeric value)
- `probability` (bool, default=False): Enable probability estimates
- `tolerance` (float, default=1e-3): Tolerance for stopping criterion
- `max_iter` (int, default=1000): Maximum iterations

### CudaSVR (Regression)
- `C` (float, default=1.0): Regularization parameter
- `epsilon` (float, default=0.1): Epsilon-tube parameter
- `kernel` (str, default='rbf'): Kernel type
- `gamma` (str/float, default='scale'): Kernel coefficient
- `tolerance` (float, default=1e-3): Tolerance for stopping criterion
- `max_iter` (int, default=1000): Maximum iterations

### Kernel Options
- **Linear**: `K(x,y) = x·y`
- **RBF**: `K(x,y) = exp(-γ||x-y||²)`
- **Polynomial**: `K(x,y) = (γx·y + r)^d`
- **Sigmoid**: `K(x,y) = tanh(γx·y + r)`

## 🚀 Performance Optimization

### Data Preprocessing
```python
# Always standardize your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# For classification, use -1/+1 labels
y_svm = np.where(y == 0, -1, 1)
```

### Memory Management
```python
# For large datasets, process in batches
batch_size = 10000
predictions = []
for i in range(0, len(X_test), batch_size):
    batch_pred = svc.predict(X_test[i:i+batch_size])
    predictions.extend(batch_pred)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

class CudaSVMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
    
    def fit(self, X, y):
        self.svm_ = CudaSVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        self.svm_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.svm_.predict(X)
    
    def score(self, X, y):
        return self.svm_.score(X, y)

# Grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

grid_search = GridSearchCV(CudaSVMWrapper(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_svm)
print(f"Best parameters: {grid_search.best_params_}")
```

## 📈 Performance Benchmarks

### Typical Speedup vs scikit-learn
- **Small datasets** (< 1K samples): 2-5x faster
- **Medium datasets** (1K-10K samples): 5-15x faster  
- **Large datasets** (10K+ samples): 10-50x faster

*Actual speedup depends on dataset size, features, kernel, and GPU specifications.*

## 🔧 Troubleshooting

### Common Issues

**"CUDA SVM library not found"**
```bash
cd SVM/
make clean && make
```

**"CUDA error" during execution**
- Check GPU memory usage: `nvidia-smi`
- Reduce batch size or dataset size
- Ensure CUDA drivers are up to date

**Poor performance compared to CPU**
- Small datasets may not benefit from GPU acceleration
- Ensure data is preprocessed (standardized)
- Try different kernel parameters

**Out of memory errors**
- Reduce dataset size or process in batches
- Use data types with lower precision (float32)
- Check available GPU memory: `nvidia-smi`

### Performance Monitoring
```python
import time

# Benchmark your SVM
start_time = time.time()
svc.fit(X_train, y_train)
fit_time = time.time() - start_time

start_time = time.time()
predictions = svc.predict(X_test)
predict_time = time.time() - start_time

print(f"Training time: {fit_time:.2f}s")
print(f"Prediction time: {predict_time:.2f}s")
```

## 📚 Additional Examples

See [`svm_usage.py`](svm_usage.py) for comprehensive examples including:
- Performance comparison with scikit-learn
- Hyperparameter tuning with GridSearchCV
- Advanced usage patterns
- Error handling and best practices

## 🤝 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the main project README
3. Submit issues to the project repository

---

**Note**: This implementation requires a CUDA-capable GPU. For CPU-only environments, use scikit-learn's SVM implementation.