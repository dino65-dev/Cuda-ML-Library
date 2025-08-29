# CUDA-Accelerated Random Forest

This directory contains a high-performance CUDA-accelerated Random Forest implementation with automatic CPU fallback.

## Features

- **CUDA Acceleration**: GPU-accelerated training and prediction when CUDA is available
- **CPU Fallback**: Automatic fallback to scikit-learn's RandomForestClassifier when CUDA is not available
- **Scikit-learn Compatible**: Drop-in replacement with the same API as scikit-learn
- **Memory Pool**: Efficient GPU memory management with custom memory pool
- **Streaming**: Asynchronous CUDA operations for better performance

## Files

- `cuda_rf_optimized.py` - Main Python wrapper with CUDA/CPU fallback
- `random_forest_cuda_optimized.cu` - CUDA implementation
- `random_forest_cuda_optimized.cuh` - CUDA header
- `rf_kernels.cu` - CUDA kernels for tree building and prediction
- `cuda_memory_pool.cu` - Memory pool implementation
- `cuda_memory_pool.cuh` - Memory pool header
- `cuda_error_check.cuh` - CUDA error checking macros
- `Makefile` - Build configuration

## Usage

```python
from cuda_rf_optimized import OptimizedCudaRFClassifier

# Create classifier
rf = OptimizedCudaRFClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Fit the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Get performance metrics
metrics = rf.get_performance_metrics()
print(f"Training time: {metrics['training_time_s']:.2f}s")
print(f"Backend: {metrics['backend']}")
```

## Building the CUDA Library

To build the CUDA library (requires CUDA toolkit):

```bash
cd Random_forest
make clean && make
```

This will create `libcuda_rf_optimized.so` which the Python wrapper will automatically detect and use.

## Requirements

- Python 3.6+
- NumPy
- Scikit-learn (for CPU fallback)
- CUDA toolkit (optional, for GPU acceleration)

## Notes

- If CUDA is not available, the implementation automatically falls back to CPU using scikit-learn
- The CUDA implementation includes a simplified tree building kernel for demonstration
- For production use, consider using more sophisticated tree building algorithms