# CUDA ML Library

This repository contains CUDA-accelerated machine learning implementations with automatic CPU fallback.

## Available Implementations

### Random Forest
- **Location**: `Random_forest/`
- **Status**: ✅ Fully functional with CUDA acceleration and CPU fallback
- **Features**: GPU-accelerated Random Forest with automatic fallback to scikit-learn

### SVM (Support Vector Machine)
- **Location**: `SVM/` and `HBM_SVM/`
- **Status**: ⚠️ In development
- **Note**: HBM SVM may not be fully functional yet

## Quick Start

### Random Forest

```bash
cd Usage/Random_forest
python hbm_rf_usage.py
```

This will automatically use CUDA if available, or fallback to CPU implementation.

## Building

Each implementation has its own build system. See the respective directories for build instructions.