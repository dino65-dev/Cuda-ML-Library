# Makefile for Optimized CUDA SVM with HBM support
NVCC = nvcc
CXX = g++
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Detect CUDA architecture automatically or provide fallback
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | sed 's/\.//g')
ifeq ($(CUDA_ARCH),)
	CUDA_ARCH = 75  # Default fallback for modern GPUs
endif
GPU_ARCH = -arch=sm_$(CUDA_ARCH)

# Compiler flags for optimized build
NVCC_FLAGS = -O3 $(GPU_ARCH) -std=c++14 -Xcompiler -fPIC --expt-relaxed-constexpr --extended-lambda
CXX_FLAGS = -O3 -fPIC -std=c++14 -Wall

# Include paths
INCLUDES = -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I/usr/local/cuda/include -I.

# Library paths and libraries
CUDA_LIB_PATH = /usr/local/cuda/lib64
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcurand

# Target
TARGET = libcuda_svm_optimized.so

# Source files
CUDA_SOURCES = svm_cuda_optimized.cu memory_pool.cu streaming_cache.cu optimized_kernels.cu
CPP_SOURCES = svm_bridge_optimized.cpp
HEADERS = svm_cuda_optimized.cuh

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

.PHONY: all clean install test check-cuda gpu-info debug

all: check-cuda $(TARGET)

check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc >/dev/null 2>&1 || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
	@echo "Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@nvcc --version
	@echo "CUDA check passed."

# Compile CUDA source files
%.o: %.cu $(HEADERS)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# Compile C++ source files  
%.o: %.cpp $(HEADERS)
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Link everything into shared library
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	@echo "Linking optimized CUDA SVM library..."
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ $(LIBS)
	@echo "Build completed successfully! Created $(TARGET)"

# Test compilation and library loading
test: $(TARGET)
	@echo "Testing library loading..."
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); print('Optimized CUDA SVM library loaded successfully!')"

# Install to local Python environment
install: $(TARGET)
	@echo "Installing optimized CUDA SVM library..."
	@# Copy shared library to current directory (where Python can find it)
	@cp $(TARGET) ../SVM/libcuda_svm_optimized.so 2>/dev/null || echo "Optional: Copy to SVM directory failed"
	@# Copy Python module
	@PYTHON_SITE=$$(python3 -c "import site; print(site.getsitepackages()[0])") && \
	 (cp cuda_svm_optimized.py "$$PYTHON_SITE/" || \
	  cp cuda_svm_optimized.py "$$(python3 -c "import site; print(site.getusersitepackages())")/" || \
	  echo "Please manually copy cuda_svm_optimized.py to your Python path")
	@echo "Installation completed!"

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -f *.o $(TARGET)
	@echo "Clean completed!"

# Debug build with debugging symbols
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG  
debug: clean $(TARGET)
	@echo "Debug build completed!"

# Show GPU information
gpu-info:
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available"
	@echo "=== CUDA Information ==="  
	@nvcc --version 2>/dev/null || echo "nvcc not available"
	@echo "=== Detected Architecture ==="
	@echo "Building for: sm_$(CUDA_ARCH)"

# Performance benchmark target
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	python3 -c "from cuda_svm_optimized import benchmark_performance; benchmark_performance()" 2>/dev/null || \
	echo "Benchmark requires cuda_svm_optimized.py with benchmark_performance() function"

# Memory test
memory-test: $(TARGET)
	@echo "Testing memory management..."
	python3 -c "from cuda_svm_optimized import test_memory_pool; test_memory_pool()" 2>/dev/null || \
	echo "Memory test requires cuda_svm_optimized.py with test_memory_pool() function"

# Cuda install
install-cuda:
	@echo "Installing CUDA toolkit..."
	@echo "This requires sudo access and internet connection"
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get install -y cuda-toolkit-12-6
	@echo "CUDA installation completed!"
	@echo "Please run: source ~/.bashrc"
	@echo "Then try: make check-cuda"

.PHONY: all clean install test check-cuda gpu-info debug install-cuda