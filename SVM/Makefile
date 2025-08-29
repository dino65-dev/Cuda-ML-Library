NVCC = nvcc
CXX = g++
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Check if CUDA is available
CUDA_AVAILABLE := $(shell which nvcc >/dev/null 2>&1 && echo "yes" || echo "no")

# Detect CUDA architecture automatically or provide fallback
ifeq ($(CUDA_AVAILABLE),yes)
	CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | sed 's/\.//g')
	ifeq ($(CUDA_ARCH),)
		CUDA_ARCH = 75
	endif
	CUDA_FLAGS = -O3 -arch=sm_$(CUDA_ARCH) -std=c++14 -Xcompiler -fPIC --extended-lambda
	INCLUDES = -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I/usr/local/cuda/include
	LIBS = -lcublas -lcurand -L/usr/local/cuda/lib64
	USE_CUDA = 1
else
	CXX_FLAGS = -O3 -fPIC -std=c++14 -DUSE_CUDA=0
	INCLUDES = -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
	LIBS = -lm
	USE_CUDA = 0
endif

TARGET = libcuda_svm.so

ifeq ($(USE_CUDA),1)
	CUDA_SOURCES = svm_cuda.cu
	CPP_SOURCES = svm_bridge.cpp
	CUDA_OBJ = svm_cuda.o
	CPP_OBJ = svm_bridge.o
else
	CUDA_SOURCES = 
	CPP_SOURCES = svm_bridge.cpp
	CUDA_OBJ = 
	CPP_OBJ = svm_bridge.o
endif

all: $(TARGET)

# Compile CUDA source to object file (only if CUDA available)
ifeq ($(USE_CUDA),1)
$(CUDA_OBJ): $(CUDA_SOURCES)
	@echo "✓ Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@echo "Compiling CUDA source..."
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -dc $< -o $@
endif

# Compile C++ source to object file
$(CPP_OBJ): $(CPP_SOURCES)
ifeq ($(USE_CUDA),1)
	@echo "Compiling C++ bridge..."
	$(CXX) -O3 -fPIC -std=c++14 -DUSE_CUDA=1 $(INCLUDES) -c $< -o $@
else
	@echo "Compiling C++ bridge (CPU only)..."
	$(CXX) -O3 -fPIC -std=c++14 -DUSE_CUDA=0 $(INCLUDES) -c $< -o $@
endif

# Link everything into shared library
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	@echo "Linking SVM library..."
ifeq ($(USE_CUDA),1)
	$(NVCC) $(CUDA_FLAGS) -shared -o $@ $^ $(LIBS)
	@echo "✓ Build completed successfully with CUDA support!"
else
	$(CXX) -O3 -fPIC -std=c++14 -shared -o $@ $^ $(LIBS)
	@echo "✓ Build completed successfully with CPU fallback!"
endif

cpu-only:
	@echo "Building CPU-only version..."
	$(MAKE) USE_CUDA=0

clean:
	@echo "Cleaning build files..."
	rm -f $(TARGET) $(CUDA_OBJ) $(CPP_OBJ)

check-cuda:
	@echo "Checking CUDA installation..."
ifeq ($(CUDA_AVAILABLE),yes)
	@echo "✓ NVCC found: $(shell nvcc --version | grep release | cut -d',' -f2)"
	@echo "✓ Detected CUDA architecture: sm_$(CUDA_ARCH)"
	@echo "✓ CUDA check passed."
else
	@echo "⚠ NVCC not found. Building CPU-only version."
	@echo "  Install CUDA toolkit for GPU acceleration:"
	@echo "  Run: make install-cuda"
endif

install-cuda:
	@echo "Installing CUDA toolkit..."
	@echo "This requires sudo access and internet connection"
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get install -y cuda-toolkit-12-6
	@echo "Please restart your shell and run 'source ~/.bashrc' then 'make check-cuda'"

.PHONY: all clean check-cuda install-cuda cpu-only