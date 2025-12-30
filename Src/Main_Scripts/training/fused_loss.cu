// Copyright (c) 2025 MatN23. All rights reserved.
// Licensed under the Custom License below.
// fused_loss.cu
// Fused Cross Entropy + Accuracy computation
//
// Compile with:
// nvcc -O3 -arch=sm_80 --use_fast_math --ptxas-options=-v -lineinfo \
//      --compiler-options '-fPIC' -shared fused_loss.cu -o fused_loss.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// Warp reduction primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Optimized argmax using warp primitives
__device__ __forceinline__ void warp_reduce_argmax(float &val, int &idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Optimized fused kernel with vectorized loads
__global__ void fused_cross_entropy_accuracy_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const int64_t pad_token_id,
    float* __restrict__ loss_out,
    float* __restrict__ accuracy_out,
    int64_t* __restrict__ valid_tokens_out,
    const int total_tokens,
    const int vocab_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int64_t label = labels[token_idx];
    
    // Early exit for invalid tokens
    if (label == pad_token_id || label < 0 || label >= vocab_size) return;
    
    const float* logit_row = logits + token_idx * vocab_size;
    
    // Step 1: Find max logit (vectorized when possible)
    float max_logit = -FLT_MAX;
    
    // Vectorized load for aligned data (4 floats at a time)
    int vec_end = (vocab_size / 4) * 4;
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(&logit_row[i]);
        max_logit = fmaxf(max_logit, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    
    // Handle remainder
    for (int i = vec_end + threadIdx.x; i < vocab_size; i += blockDim.x) {
        max_logit = fmaxf(max_logit, logit_row[i]);
    }
    
    // Reduce max across warps
    max_logit = warp_reduce_max(max_logit);
    
    __shared__ float warp_maxes[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    if (lane == 0) warp_maxes[wid] = max_logit;
    __syncthreads();
    
    if (wid == 0) {
        max_logit = (threadIdx.x < (blockDim.x >> 5)) ? warp_maxes[lane] : -FLT_MAX;
        max_logit = warp_reduce_max(max_logit);
        if (threadIdx.x == 0) warp_maxes[0] = max_logit;
    }
    __syncthreads();
    max_logit = warp_maxes[0];
    
    // Step 2: Compute sum of exp (vectorized)
    float sum_exp = 0.0f;
    
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(&logit_row[i]);
        sum_exp += expf(vals.x - max_logit) + expf(vals.y - max_logit) + 
                   expf(vals.z - max_logit) + expf(vals.w - max_logit);
    }
    
    for (int i = vec_end + threadIdx.x; i < vocab_size; i += blockDim.x) {
        sum_exp += expf(logit_row[i] - max_logit);
    }
    
    sum_exp = warp_reduce_sum(sum_exp);
    
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[wid] = sum_exp;
    __syncthreads();
    
    if (wid == 0) {
        sum_exp = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
        if (threadIdx.x == 0) warp_sums[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = warp_sums[0];
    
    // Step 3: Compute loss
    float label_logit = logit_row[label];
    float log_prob = (label_logit - max_logit) - __logf(sum_exp + 1e-10f);
    float token_loss = -log_prob;
    
    // Clamp loss to prevent NaN propagation
    token_loss = fminf(token_loss, 100.0f);
    
    // Step 4: Argmax for accuracy (optimized with warp primitives)
    float my_max = -FLT_MAX;
    int my_idx = 0;
    
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(&logit_row[i]);
        
        if (vals.x > my_max) { my_max = vals.x; my_idx = i; }
        if (vals.y > my_max) { my_max = vals.y; my_idx = i + 1; }
        if (vals.z > my_max) { my_max = vals.z; my_idx = i + 2; }
        if (vals.w > my_max) { my_max = vals.w; my_idx = i + 3; }
    }
    
    for (int i = vec_end + threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = logit_row[i];
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }
    
    // Warp-level argmax
    warp_reduce_argmax(my_max, my_idx);
    
    __shared__ float shared_vals[32];
    __shared__ int shared_idxs[32];
    
    if (lane == 0) {
        shared_vals[wid] = my_max;
        shared_idxs[wid] = my_idx;
    }
    __syncthreads();
    
    if (wid == 0) {
        my_max = (threadIdx.x < (blockDim.x >> 5)) ? shared_vals[lane] : -FLT_MAX;
        my_idx = (threadIdx.x < (blockDim.x >> 5)) ? shared_idxs[lane] : 0;
        warp_reduce_argmax(my_max, my_idx);
        
        if (threadIdx.x == 0) {
            shared_idxs[0] = my_idx;
        }
    }
    __syncthreads();
    
    int predicted = shared_idxs[0];
    int correct = (predicted == (int)label) ? 1 : 0;
    
    // Step 5: Atomic accumulation (only thread 0)
    if (threadIdx.x == 0) {
        atomicAdd(loss_out, token_loss);
        atomicAdd(accuracy_out, (float)correct);
        atomicAdd((unsigned long long*)valid_tokens_out, 1ULL);
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

void fused_cross_entropy_accuracy_launcher(
    const float* logits,
    const int64_t* labels,
    int64_t pad_token_id,
    float* loss_out,
    float* accuracy_out,
    int64_t* valid_tokens_out,
    int total_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemsetAsync(loss_out, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(accuracy_out, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(valid_tokens_out, 0, sizeof(int64_t), stream));
    
    // Use 256 threads for optimal occupancy
    int threads = 256;
    int blocks = total_tokens;
    
    fused_cross_entropy_accuracy_kernel<<<blocks, threads, 0, stream>>>(
        logits, labels, pad_token_id,
        loss_out, accuracy_out, valid_tokens_out,
        total_tokens, vocab_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"