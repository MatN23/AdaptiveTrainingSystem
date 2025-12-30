// Copyright (c) 2025 MatN23. All rights reserved.
// Licensed under the Custom License below.
// fused_grad_clip.cu - OPTIMIZED VERSION
// Fully async gradient norm computation + clipping with vectorization
//
// Compile with:
// nvcc -O3 -arch=sm_80 --use_fast_math --ptxas-options=-v -lineinfo \
//      --compiler-options '-fPIC' -shared fused_grad_clip.cu -o fused_grad_clip.so

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// Warp reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel with vectorized loads
__global__ void compute_grad_norm_squared_kernel(
    float** __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    float* __restrict__ global_norm_sq,
    int num_tensors
) {
    int tensor_idx = blockIdx.x;
    if (tensor_idx >= num_tensors) return;
    
    float* grad = grad_ptrs[tensor_idx];
    int size = grad_sizes[tensor_idx];
    
    float sum_sq = 0.0f;
    
    // Vectorized loads (4 floats at a time)
    int vec_end = (size / 4) * 4;
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(&grad[i]);
        sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    // Handle remainder
    for (int i = vec_end + threadIdx.x; i < size; i += blockDim.x) {
        float val = grad[i];
        sum_sq += val * val;
    }
    
    // Warp reduce
    sum_sq = warp_reduce_sum(sum_sq);
    
    // Block reduce
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();
    
    if (wid == 0) {
        sum_sq = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        
        if (threadIdx.x == 0) {
            atomicAdd(global_norm_sq, sum_sq);
        }
    }
}

// Optimized clipping kernel with vectorization
__global__ void clip_gradients_kernel(
    float** __restrict__ grad_ptrs,
    const int* __restrict__ grad_sizes,
    const float* __restrict__ total_norm_device,
    float max_norm,
    int num_tensors
) {
    int tensor_idx = blockIdx.x;
    if (tensor_idx >= num_tensors) return;
    
    float* grad = grad_ptrs[tensor_idx];
    int size = grad_sizes[tensor_idx];
    
    float total_norm = *total_norm_device;
    float clip_coef = max_norm / (total_norm + 1e-6f);
    
    if (clip_coef >= 1.0f) return;  // No clipping needed
    
    // Vectorized clipping
    int vec_end = (size / 4) * 4;
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<float4*>(&grad[i]);
        vals.x *= clip_coef;
        vals.y *= clip_coef;
        vals.z *= clip_coef;
        vals.w *= clip_coef;
        *reinterpret_cast<float4*>(&grad[i]) = vals;
    }
    
    // Handle remainder
    for (int i = vec_end + threadIdx.x; i < size; i += blockDim.x) {
        grad[i] *= clip_coef;
    }
}

__global__ void sqrt_kernel(float* norm_sq, float* norm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *norm = sqrtf(*norm_sq);
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" {

float fused_grad_clip_launcher(
    float** grad_ptrs_device,
    int* grad_sizes_device,
    int num_tensors,
    float max_norm,
    cudaStream_t stream
) {
    // Pinned memory for async transfer
    static float* norm_pinned = nullptr;
    if (norm_pinned == nullptr) {
        CUDA_CHECK(cudaHostAlloc(&norm_pinned, sizeof(float), cudaHostAllocDefault));
    }
    
    // Device memory
    float* norm_sq_device;
    float* norm_device;
    CUDA_CHECK(cudaMallocAsync(&norm_sq_device, sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&norm_device, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(norm_sq_device, 0, sizeof(float), stream));
    
    // Compute norm^2 with optimal thread count
    int threads = 256;
    int blocks = num_tensors;
    
    compute_grad_norm_squared_kernel<<<blocks, threads, 0, stream>>>(
        grad_ptrs_device, grad_sizes_device, norm_sq_device, num_tensors
    );
    
    // Compute sqrt on GPU
    sqrt_kernel<<<1, 1, 0, stream>>>(norm_sq_device, norm_device);
    
    // Clip gradients
    clip_gradients_kernel<<<blocks, threads, 0, stream>>>(
        grad_ptrs_device, grad_sizes_device, norm_device, max_norm, num_tensors
    );
    
    // Async copy to host
    CUDA_CHECK(cudaMemcpyAsync(norm_pinned, norm_device, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    
    // Synchronize only at the end
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float total_norm = *norm_pinned;
    
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(norm_sq_device, stream));
    CUDA_CHECK(cudaFreeAsync(norm_device, stream));
    
    return total_norm;
}

}  // extern "C"