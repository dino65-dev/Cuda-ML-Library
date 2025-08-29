#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <stdexcept>

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CURAND error"); \
        } \
    } while(0)