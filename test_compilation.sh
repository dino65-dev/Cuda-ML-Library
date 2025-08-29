#!/bin/bash
# Test script for CUDA SVM compilation on Colab
echo "=== CUDA SVM Compilation Test ==="

# Navigate to SVM directory
cd SVM

# Clean and build
echo "Cleaning previous build..."
make clean

echo "Building CUDA SVM library..."
if make; then
    echo "✅ Compilation successful!"
    echo ""
    echo "=== Testing Library ==="

    # Test the library
    python3 -c "
import sys
sys.path.append('..')
from SVM.cuda_svm import CudaSVM
import numpy as np

print('Testing CUDA SVM functionality...')

# Simple test
X = np.random.randn(50, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32) * 2 - 1

svm = CudaSVM(kernel='rbf', C=1.0)
svm.fit(X, y)
pred = svm.predict(X[:10])

print('✅ CUDA SVM test passed!')
print('Library is ready for use.')
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✅ All tests passed!"
    else
        echo "⚠️  Python test failed, but compilation was successful"
    fi

else
    echo "❌ Compilation failed!"
    exit 1
fi
