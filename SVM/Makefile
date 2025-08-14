NVCC = nvcc
CXX = g++
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Detect CUDA architecture automatically or provide fallback
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | sed 's/\.//g')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = 75
endif

CUDA_FLAGS = -O3 -arch=sm_$(CUDA_ARCH) -std=c++14 -Xcompiler -fPIC --extended-lambda
CXX_FLAGS = -O3 -fPIC -std=c++14

INCLUDES = -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I/usr/local/cuda/include
LIBS = -lm

TARGET = libcuda_svm.so
CPP_SOURCES = svm_bridge.cpp

all: $(TARGET)

# Build C++ only version for testing
$(TARGET): $(CPP_SOURCES)
    @echo "Building SVM library (C++ version)..."
    $(CXX) $(CXX_FLAGS) $(INCLUDES) -shared -o $@ $^ $(LIBS)
    @echo "Build completed successfully!"

clean:
    @echo "Cleaning build files..."
    rm -f $(TARGET) *.o

check-cuda:
    @echo "Checking CUDA installation..."
    @which nvcc >/dev/null 2>&1 || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
    @echo "Detected CUDA architecture: sm_$(CUDA_ARCH)"
    @nvcc --version
    @echo "CUDA check passed."

.PHONY: all clean check-cuda