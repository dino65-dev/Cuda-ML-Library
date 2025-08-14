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
LIBS = -lcublas -lcurand -L/usr/local/cuda/lib64

TARGET = libcuda_svm.so
CUDA_SOURCES = svm_cuda.cu
CPP_SOURCES = svm_bridge.cpp

all: $(TARGET)

$(TARGET): $(CUDA_SOURCES) $(CPP_SOURCES)
	@echo "Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@echo "Building CUDA SVM library..."
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -shared -o $@ $^ $(LIBS)
	@echo "Build completed successfully!"

clean:
	rm -f $(TARGET)

check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version
	@echo "Checking GPU..."
	@nvidia-smi --query-gpu=name,compute_cap --format=csv 2>/dev/null || echo "WARNING: nvidia-smi not available (no GPU or driver)"

install-cuda:
	@echo "Installing CUDA toolkit..."
	@echo "This requires sudo access and internet connection"
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get install -y cuda-toolkit-12-6
	@echo "Please restart your shell and run 'source ~/.bashrc' then 'make check-cuda'"

.PHONY: all clean check-cuda install-cuda