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

# Object files
CUDA_OBJ = svm_cuda.o
CPP_OBJ = svm_bridge.o

all: $(TARGET)

# Compile CUDA source to object file
$(CUDA_OBJ): $(CUDA_SOURCES)
	@echo "Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@echo "Compiling CUDA source..."
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -dc $< -o $@

# Compile C++ source to object file
$(CPP_OBJ): $(CPP_SOURCES)
	@echo "Compiling C++ bridge..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Link everything into shared library
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	@echo "Linking CUDA SVM library..."
	$(NVCC) $(CUDA_FLAGS) -shared -o $@ $^ $(LIBS)
	@echo "Build completed successfully!"

clean:
	@echo "Cleaning build files..."
	rm -f $(TARGET) $(CUDA_OBJ) $(CPP_OBJ)

check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc >/dev/null 2>&1 || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
	@echo "Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@nvcc --version
	@echo "CUDA check passed."

install-cuda:
	@echo "Installing CUDA toolkit..."
	@echo "This requires sudo access and internet connection"
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get install -y cuda-toolkit-12-6
	@echo "Please restart your shell and run 'source ~/.bashrc' then 'make check-cuda'"

.PHONY: all clean check-cuda install-cuda