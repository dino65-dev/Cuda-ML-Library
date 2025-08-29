# Makefile for the HBM_RandomForest CUDA module

# Compiler
NVCC = nvcc

# Common flags for compilation and linking
NVCC_FLAGS = -O3 -Xcompiler -fPIC

# Specify include directories. Assumes a 'common' directory at the same level as 'HBM_RandomForest'
# This is required for "common/cuda_memory_pool.cuh" and "common/cuda_error_check.cuh"
INCLUDES = -I.

# Specify CUDA architecture(s) to support.
# Pascal (6.0), Volta (7.0), Turing (7.5), Ampere (8.6). Add more as needed for your GPU.
ARCH_FLAGS = -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86

# Flags for compiling to object files (.cu -> .o)
COMPILE_FLAGS = -c $(NVCC_FLAGS) $(ARCH_FLAGS) $(INCLUDES)

# Flags for linking object files into a shared library
LINK_FLAGS = --shared $(NVCC_FLAGS) $(ARCH_FLAGS)

# --- Target Definition ---

# Final shared library that the Python wrapper will load
TARGET = libcuda_rf_optimized.so

# C++/CUDA source files
SRCS = random_forest_cuda_optimized.cu rf_kernels.cu

# Object files derived from source files
OBJS = $(SRCS:.cu=.o)

# Common object files from the shared directory
COMMON_OBJS = cuda_memory_pool.o

# --- Rules ---

# Default target: build the shared library
all: $(TARGET)

# Rule to link the shared library from object files
$(TARGET): $(OBJS) $(COMMON_OBJS)
	@echo "Linking shared library: $@"
	$(NVCC) $(LINK_FLAGS) -o $@ $^

# Rule to build common object files. This makes the build self-contained.
cuda_memory_pool.o: cuda_memory_pool.cu cuda_memory_pool.cuh cuda_error_check.cuh
	@echo "Compiling common dependency: $<"
	$(NVCC) $(COMPILE_FLAGS) -o $@ $<

# Specific rule to compile random_forest_cuda_optimized.cu into random_forest_cuda_optimized.o
random_forest_cuda_optimized.o: random_forest_cuda_optimized.cu random_forest_cuda_optimized.cuh cuda_memory_pool.cuh cuda_error_check.cuh
	@echo "Compiling: $< -> $@"
	$(NVCC) $(COMPILE_FLAGS) -o $@ $<

# Specific rule to compile rf_kernels.cu into rf_kernels.o
rf_kernels.o: rf_kernels.cu random_forest_cuda_optimized.cuh cuda_memory_pool.cuh cuda_error_check.cuh
	@echo "Compiling: $< -> $@"
	$(NVCC) $(COMPILE_FLAGS) -o $@ $<

# Clean up build artifacts
clean:
	@echo "Cleaning up build files..."
	rm -f $(OBJS) $(COMMON_OBJS) $(TARGET)

# Phony targets do not represent files
.PHONY: all clean