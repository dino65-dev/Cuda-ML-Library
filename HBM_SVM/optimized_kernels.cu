#include "svm_cuda_optimized.cuh"

// Optimized RBF kernel with streaming computation
__global__ void optimized_rbf_kernel_streaming(
    const float* X1, const float* X2, float* K,
    int n1, int n2, int n_features, float gamma,
    int row_offset) {
    
    __shared__ float shared_X1[32 * 16];  // Shared memory for X1
    __shared__ float shared_X2[32 * 16];  // Shared memory for X2
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_idx = bid * blockDim.x + tid;
    
    if (global_idx >= n2) return;
    
    // Load data into shared memory with coalesced access
    for (int feature_block = 0; feature_block < n_features; feature_block += 16) {
        int features_to_load = min(16, n_features - feature_block);
        
        // Load X1 features to shared memory
        if (tid < features_to_load) {
            shared_X1[tid] = X1[feature_block + tid];
        }
        
        // Load X2 features to shared memory
        if (tid < features_to_load && global_idx < n2) {
            shared_X2[tid] = X2[global_idx * n_features + feature_block + tid];
        }
        
        __syncthreads();
        
        // Compute partial distance
        float partial_dist = 0.0f;
        for (int i = 0; i < features_to_load; i++) {
            float diff = shared_X1[i] - shared_X2[i];
            partial_dist += diff * diff;
        }
        
        if (feature_block == 0) {
            K[global_idx + row_offset * n2] = partial_dist;
        } else {
            K[global_idx + row_offset * n2] += partial_dist;
        }
        
        __syncthreads();
    }
    
    // Apply RBF transformation
    if (global_idx < n2) {
        K[global_idx + row_offset * n2] = __expf(-gamma * K[global_idx + row_offset * n2]);
    }
}

// Half-precision kernel computation for memory efficiency
__global__ void half_precision_kernel_computation(
    const __half* X1, const __half* X2, float* K,
    int n1, int n2, int n_features, float gamma) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n1 || j >= n2) return;
    
    __half2 dist_sq = __float2half2_rn(0.0f);
    
    // Vectorized computation using half2
    for (int k = 0; k < n_features; k += 2) {
        __half2 x1_vec = reinterpret_cast<const __half2*>(&X1[i * n_features + k])[0];
        __half2 x2_vec = reinterpret_cast<const __half2*>(&X2[j * n_features + k])[0];
        
        __half2 diff = __hsub2(x1_vec, x2_vec);
        dist_sq = __hfma2(diff, diff, dist_sq);
    }
    
    // Convert back to float for final computation
    float dist_float = __half2float(__hadd(__low2half(dist_sq), __high2half(dist_sq)));
    K[i * n2 + j] = __expf(-gamma * dist_float);
}

// Vectorized working set selection with reduction
__global__ void vectorized_working_set_selection(
    const float* gradient, const float* y, const float* alpha,
    int* working_set, float* scores, int n_samples, float C, float tolerance) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_idx = bid * blockDim.x + tid;
    
    __shared__ float shared_grad[256];
    __shared__ float shared_scores[256];
    __shared__ int shared_indices[256];
    
    // Load gradient values
    if (global_idx < n_samples) {
        shared_grad[tid] = gradient[global_idx];
        shared_indices[tid] = global_idx;
        
        // Compute selection score
        float alpha_val = alpha[global_idx];
        float y_val = y[global_idx];
        
        bool is_upper_bound = (y_val > 0 && alpha_val >= C) || (y_val < 0 && alpha_val <= 0);
        bool is_lower_bound = (y_val > 0 && alpha_val <= 0) || (y_val < 0 && alpha_val >= C);
        
        if (!is_upper_bound && shared_grad[tid] > tolerance) {
            shared_scores[tid] = shared_grad[tid];
        } else if (!is_lower_bound && shared_grad[tid] < -tolerance) {
            shared_scores[tid] = -shared_grad[tid];
        } else {
            shared_scores[tid] = 0.0f;
        }
    } else {
        shared_scores[tid] = 0.0f;
        shared_indices[tid] = -1;
    }
    
    __syncthreads();
    
    // Find maximum score in block using reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < blockDim.x) {
            if (shared_scores[tid + stride] > shared_scores[tid]) {
                shared_scores[tid] = shared_scores[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write block maximum to global memory
    if (tid == 0) {
        scores[bid] = shared_scores[0];
        working_set[bid] = shared_indices[0];
    }
}

// Optimized prediction with memory coalescing
__global__ void optimized_prediction_kernel(
    const float* sv_X, const float* sv_alpha, const float* X_test,
    float* predictions, int n_sv, int n_test, int n_features,
    float bias, KernelType kernel_type, float gamma, float coef0, int degree) {
    
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (test_idx >= n_test) return;
    
    float sum = 0.0f;
    
    // Process support vectors in blocks for better cache utilization
    for (int sv_block = 0; sv_block < n_sv; sv_block += 32) {
        int sv_end = min(sv_block + 32, n_sv);
        
        for (int sv_idx = sv_block; sv_idx < sv_end; sv_idx++) {
            float kernel_val = 0.0f;
            
            // Compute kernel value with vectorized operations
            switch (kernel_type) {
                case KernelType::RBF: {
                    float dist_sq = 0.0f;
                    
                    // Unroll loop for better performance
                    #pragma unroll 4
                    for (int k = 0; k < n_features; k++) {
                        float diff = X_test[test_idx * n_features + k] - sv_X[sv_idx * n_features + k];
                        dist_sq += diff * diff;
                    }
                    kernel_val = __expf(-gamma * dist_sq);
                    break;
                }
                case KernelType::LINEAR: {
                    #pragma unroll 4
                    for (int k = 0; k < n_features; k++) {
                        kernel_val += X_test[test_idx * n_features + k] * sv_X[sv_idx * n_features + k];
                    }
                    break;
                }
            }
            
            sum += sv_alpha[sv_idx] * kernel_val;
        }
    }
    
    predictions[test_idx] = sum + bias;
}

// Template for blocked matrix multiplication
template<typename T>
__global__ void blocked_matrix_multiply(
    const T* A, const T* B, T* C,
    int M, int N, int K, int block_size) {
    
    __shared__ T shared_A[32][32];
    __shared__ T shared_B[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = static_cast<T>(0);
    
    for (int tile = 0; tile < (K + block_size - 1) / block_size; tile++) {
        // Load tiles into shared memory
        if (row < M && tile * block_size + threadIdx.x < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * block_size + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        
        if (col < N && tile * block_size + threadIdx.y < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * block_size + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < block_size; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
