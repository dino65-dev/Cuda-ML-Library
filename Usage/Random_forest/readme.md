# High Bandwidth Memory (HBM) Optimized CUDA Random Forest

This directory contains the advanced, HBM-optimized version of the CUDA Random Forest implementation, designed for maximum performance on modern GPU architectures.

## 🎯 Key Optimizations

The Random Forest algorithm, particularly the tree-building process, is computationally intensive. This implementation leverages the GPU to parallelize this task effectively.

1.  **Parallel Tree Construction**: Each tree in the forest is built by a dedicated CUDA block, allowing for massive parallelism across the `n_estimators`.
2.  **Optimized Prediction**: The prediction phase is fully parallel, with each thread independently processing a sample and traversing all trees.
3.  **Efficient Memory Management**: Utilizes the same `CudaMemoryPool` as the HBM SVM to reduce `cudaMalloc` overhead and manage GPU memory efficiently.
4.  **On-the-Fly Random Number Generation**: Uses `cuRAND` on the device for bootstrap sampling and feature selection, avoiding costly data transfers.

### Performance Gains

- **10-100x faster** training compared to CPU-based `scikit-learn` for large datasets.
- **Superior scaling** with `n_estimators` and `n_samples`.
- **Low-latency inference** for real-time prediction tasks.

## 🚀 Quick Start

### Prerequisites

**GPU Requirements:**
- **NVIDIA GPU**: CUDA Compute Capability 6.0+ (Pascal or newer).
- **CUDA Toolkit**: Version 11.0 or higher.
- **GPU Memory**: 4GB+ VRAM recommended for large datasets.

### Build the Optimized Library

You will need to create a `Makefile` in the `HBM_RandomForest` directory to compile `random_forest_cuda_optimized.cu` and `rf_kernels.cu` into a shared library (`libcuda_rf_optimized.so`).

```bash
# Navigate to the implementation directory
cd /path/to/Cuda-ML-Library/HBM_RandomForest

# Compile the code (assuming a Makefile is present)
make
```

### Basic Usage

```python
from RandomForest.cuda_rf_optimized import OptimizedCudaRFClassifier
from sklearn.datasets import make_classification

# 1. Generate data
X, y = make_classification(n_samples=10000, n_features=50, random_state=42)

# 2. Create and train the model
clf = OptimizedCudaRFClassifier(
    n_estimators=150,
        max_depth=15,
            verbose=True,
                random_state=42
                )
                clf.fit(X, y)

                # 3. Make predictions
                predictions = clf.predict(X)

                # 4. Evaluate
                accuracy = clf.score(X, y)
                print(f"Model Accuracy: {accuracy:.4f}")

                # 5. Get performance metrics
                metrics = clf.get_performance_metrics()
                print(f"Training Time: {metrics['training_time_s']:.2f}s")
```

## ⚙️ Parameters

### `OptimizedCudaRFClassifier`
- `n_estimators` (int, default=100): The number of trees in the forest.
- `max_depth` (int, default=10): The maximum depth of each tree.
- `min_samples_split` (int, default=2): The minimum number of samples required to split an internal node (Note: currently a placeholder in the simplified kernel).
- `max_features` (str, default='sqrt'): The number of features to consider when looking for the best split.
  - `'sqrt'`: `max_features=sqrt(n_features)`
  - `'log2'`: `max_features=log2(n_features)`
  - `int`: `max_features=max_features`
  - `float`: `max_features=int(max_features * n_features)`
- `bootstrap` (bool, default=True): Whether bootstrap samples are used when building trees (Note: enabled by default in the kernel concept).
- `random_state` (int, default=None): Seed for the random number generator for reproducibility.
- `verbose` (bool, default=False): Enable verbose output during training.

## 📈 Performance and Usage

### When to Use HBM Random Forest

✅ **Use HBM RF when:**
- Working with **large datasets** (>10,000 samples, >50 features).
- `n_estimators` is high (e.g., >100).
- Training time is a critical bottleneck.
- You have a CUDA-capable NVIDIA GPU.

❌ **Consider `scikit-learn` when:**
- Working with very small datasets where GPU overhead may negate benefits.
- You do not have a compatible NVIDIA GPU.

## 🔍 Monitoring and Debugging

### Error Handling
The Python wrapper will raise a `CudaRFError` if any issue occurs in the C++/CUDA backend, providing a descriptive error message.

```python
from RandomForest.cuda_rf_optimized import CudaRFError

try:
    clf.fit(X, y)
except CudaRFError as e:
    print(f"A GPU-specific error occurred: {e}")
```

### Complete Example

See `hbm_rf_usage.py` for a comprehensive example that includes performance comparison with `scikit-learn`.