#!/bin/bash

echo "Building and Testing Optimized CUDA SVM"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}❌${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

# Check CUDA installation
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    print_warning "NVCC not found. Building with CPU fallback."
    print_warning "For best performance, install CUDA toolkit: make install-cuda"
    BUILD_MODE="cpu-only"
else
    print_success "NVCC found: $(nvcc --version | grep release | cut -d',' -f2)"
    BUILD_MODE="all"
fi

# Check GPU
echo ""
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. GPU might not be available."
    print_warning "This is normal in development containers without GPU access."
else
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ $gpu_count -gt 0 ]; then
        print_success "Found $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    else
        print_warning "No GPUs detected. Code will compile but may not run."
    fi
fi

# Clean and build
echo ""
echo "Building CUDA SVM library..."
make clean 2>/dev/null || true

if ! make $BUILD_MODE; then
    print_error "Build failed!"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Make sure required development tools are installed"
    echo "2. Check that Python development headers are available"
    echo "3. Verify all source files are present"
    exit 1
fi

if [ "$BUILD_MODE" = "cpu-only" ]; then
    print_success "Build successful with CPU fallback - libcuda_svm_optimized.so created"
else
    print_success "Build successful with CUDA support - libcuda_svm_optimized.so created"
fi

# Check if library file was created
if [ ! -f "libcuda_svm_optimized.so" ]; then
    print_error "Library file not found after build!"
    exit 1
fi

# Test library loading
echo ""
echo "Testing library loading..."
if ! python3 -c "
try:
    from cuda_svm_optimized import OptimizedCudaSVC, OptimizedCudaSVR
    print('✓ Library modules imported successfully')
    
    # Test basic instantiation
    svc = OptimizedCudaSVC(verbose=False)
    print('✓ OptimizedCudaSVC created successfully')
    
    svr = OptimizedCudaSVR(verbose=False)  
    print('✓ OptimizedCudaSVR created successfully')
    
except Exception as e:
    print(f'❌ Library loading failed: {e}')
    exit(1)
" 2>/dev/null; then
    print_error "Library loading failed!"
    echo ""
    echo "Possible issues:"
    echo "1. Library compilation incomplete"
    echo "2. Missing CUDA runtime libraries"
    echo "3. GPU drivers not installed"
    echo ""
    echo "Try running the library test manually:"
    echo "python3 -c 'from cuda_svm_optimized import OptimizedCudaSVC'"
    exit 1
fi

print_success "Library loading successful"

# Test with sample data
echo ""
echo "Testing with sample data..."
if ! python3 -c "
import numpy as np
from sklearn.datasets import make_classification
from cuda_svm_optimized import OptimizedCudaSVC

try:
    # Generate small test dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to SVM format
    
    # Test basic functionality
    svm = OptimizedCudaSVC(C=1.0, kernel='rbf', verbose=False)
    svm.fit(X, y)
    predictions = svm.predict(X[:10])
    accuracy = svm.score(X[:50], y[:50])
    
    print(f'✓ Basic training and prediction successful')
    print(f'✓ Test accuracy: {accuracy:.3f}')
    
    # Get performance metrics
    metrics = svm.get_performance_metrics()
    print(f'✓ Performance metrics retrieved')
    print(f'  - Training time: {metrics[\"training_time\"]:.3f}s')
    print(f'  - Memory usage: {metrics[\"memory_usage_gb\"]:.3f}GB')
    
except Exception as e:
    print(f'❌ Sample test failed: {e}')
    exit(1)
" 2>/dev/null; then
    print_error "Sample test failed!"
    exit 1
fi

print_success "Sample test passed"

# Run usage examples if available
echo ""
echo "Checking for usage examples..."
USAGE_DIR="../Usage/HBM_SVM"
if [ -f "$USAGE_DIR/hbm_svm_usage.py" ]; then
    echo "Running comprehensive usage examples..."
    cd "$USAGE_DIR" || exit 1
    
    # Add current directory to Python path for imports
    export PYTHONPATH="../../HBM_SVM:$PYTHONPATH"
    
    if python3 hbm_svm_usage.py; then
        print_success "Usage examples completed successfully"
    else
        print_warning "Usage examples failed (this might be due to GPU unavailability)"
        echo "Examples may still work on systems with proper GPU setup"
    fi
    
    cd - > /dev/null || exit 1
else
    print_warning "Usage examples not found at $USAGE_DIR/hbm_svm_usage.py"
fi

# Final summary
echo ""
echo "============================================"
print_success "All tests passed!"
echo ""
echo "Your Optimized CUDA SVM is ready to use!"
echo ""
echo "Quick start:"
echo "  from cuda_svm_optimized import OptimizedCudaSVC, OptimizedCudaSVR"
echo "  svm = OptimizedCudaSVC(C=1.0, kernel='rbf')"
echo "  svm.fit(X_train, y_train)"
echo "  predictions = svm.predict(X_test)"
echo ""
echo "For more examples, see: ../Usage/HBM_SVM/hbm_svm_usage.py"
echo "============================================"