// Copyright (c) 2025 MatN23. All rights reserved.
// GOD-TIER fused_loss.cu - TRUE SINGLE-PASS with online softmax
// 
// Uses numerically stable online algorithm to compute max, sum, and argmax
// in ONE PASS through the vocabulary!
//
// Compile with:
// nvcc -O3 -arch=sm_75 --use_fast_math --ptxas-options=-v \
//      --compiler-options '-fPIC' -shared fused_loss.cu -o fused_loss.so

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level reduction for online softmax state
struct SoftmaxState {
    float m;  // running max
    float d;  // running sum of exp
    float argmax_val;
    int argmax_idx;
};

__device__ __forceinline__ SoftmaxState warp_reduce_softmax_state(SoftmaxState state) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_m = __shfl_xor_sync(FULL_MASK, state.m, offset);
        float other_d = __shfl_xor_sync(FULL_MASK, state.d, offset);
        float other_argmax_val = __shfl_xor_sync(FULL_MASK, state.argmax_val, offset);
        int other_argmax_idx = __shfl_xor_sync(FULL_MASK, state.argmax_idx, offset);
        
        // Merge two softmax states
        float m_new = fmaxf(state.m, other_m);
        state.d = state.d * __expf(state.m - m_new) + other_d * __expf(other_m - m_new);
        state.m = m_new;
        
        // Merge argmax
        if (other_argmax_val > state.argmax_val) {
            state.argmax_val = other_argmax_val;
            state.argmax_idx = other_argmax_idx;
        }
    }
    return state;
}

// SINGLE-PASS kernel: computes everything in ONE traversal!
// Uses online softmax algorithm (numerically stable)
__global__ void __launch_bounds__(256, 4)
fused_cross_entropy_single_pass(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const int64_t pad_token_id,
    float* __restrict__ loss_out,
    float* __restrict__ accuracy_out,
    int64_t* __restrict__ valid_tokens_out,
    const int total_tokens,
    const int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const int64_t label = labels[token_idx];
    
    // Skip padding
    if (label == pad_token_id || label < 0 || label >= vocab_size) {
        return;
    }
    
    const float* logit_row = logits + (size_t)token_idx * vocab_size;
    
    // =========================================================================
    // SINGLE PASS: Online softmax + argmax together!
    // =========================================================================
    SoftmaxState state;
    state.m = -FLT_MAX;
    state.d = 0.0f;
    state.argmax_val = -FLT_MAX;
    state.argmax_idx = 0;
    
    float label_logit = 0.0f;
    
    // Vectorized processing - process 4 elements at a time
    const int vec_end = (vocab_size / 4) * 4;
    
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = __ldg(reinterpret_cast<const float4*>(&logit_row[i]));
        
        // Process each value with online softmax
        float values[4] = {vals.x, vals.y, vals.z, vals.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = values[j];
            int idx = i + j;
            
            // Online softmax update
            float m_new = fmaxf(state.m, x);
            state.d = state.d * __expf(state.m - m_new) + __expf(x - m_new);
            state.m = m_new;
            
            // Track argmax
            if (x > state.argmax_val) {
                state.argmax_val = x;
                state.argmax_idx = idx;
            }
            
            // Store label logit if we hit it
            if (idx == label) {
                label_logit = x;
            }
        }
    }
    
    // Handle remainder elements
    for (int i = vec_end + threadIdx.x; i < vocab_size; i += blockDim.x) {
        float x = __ldg(&logit_row[i]);
        
        // Online softmax update
        float m_new = fmaxf(state.m, x);
        state.d = state.d * __expf(state.m - m_new) + __expf(x - m_new);
        state.m = m_new;
        
        // Track argmax
        if (x > state.argmax_val) {
            state.argmax_val = x;
            state.argmax_idx = i;
        }
        
        // Store label logit if we hit it
        if (i == label) {
            label_logit = x;
        }
    }
    
    // =========================================================================
    // Reduce across warp
    // =========================================================================
    state = warp_reduce_softmax_state(state);
    
    // =========================================================================
    // Reduce across block using shared memory
    // =========================================================================
    __shared__ float smem_m[32];
    __shared__ float smem_d[32];
    __shared__ float smem_argmax_val[32];
    __shared__ int smem_argmax_idx[32];
    __shared__ float smem_label_logit[256];
    
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    
    // Store warp results
    if (lane == 0) {
        smem_m[wid] = state.m;
        smem_d[wid] = state.d;
        smem_argmax_val[wid] = state.argmax_val;
        smem_argmax_idx[wid] = state.argmax_idx;
    }
    smem_label_logit[threadIdx.x] = label_logit;
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        // Load from shared memory
        if (lane < (blockDim.x >> 5)) {
            state.m = smem_m[lane];
            state.d = smem_d[lane];
            state.argmax_val = smem_argmax_val[lane];
            state.argmax_idx = smem_argmax_idx[lane];
        } else {
            state.m = -FLT_MAX;
            state.d = 0.0f;
            state.argmax_val = -FLT_MAX;
            state.argmax_idx = 0;
        }
        
        // Reduce within first warp
        state = warp_reduce_softmax_state(state);
        
        if (lane == 0) {
            smem_m[0] = state.m;
            smem_d[0] = state.d;
            smem_argmax_idx[0] = state.argmax_idx;
        }
    }
    __syncthreads();
    
    // =========================================================================
    // Compute final loss and accuracy
    // =========================================================================
    if (threadIdx.x == 0) {
        const float max_logit = smem_m[0];
        const float sum_exp = smem_d[0];
        const int pred_idx = smem_argmax_idx[0];
        
        // Find label_logit from shared memory
        float final_label_logit = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            final_label_logit = fmaxf(final_label_logit, smem_label_logit[i]);
        }
        
        // If we didn't find it, load it directly
        if (final_label_logit == 0.0f) {
            final_label_logit = logit_row[label];
        }
        
        // Compute loss: -log(softmax[label])
        // softmax[label] = exp(label_logit - max) / sum_exp
        // loss = -(label_logit - max - log(sum_exp))
        const float loss = -((final_label_logit - max_logit) - __logf(sum_exp + 1e-10f));
        const float accuracy = (pred_idx == (int)label) ? 1.0f : 0.0f;
        
        // Atomic updates
        atomicAdd(loss_out, fminf(loss, 100.0f));
        atomicAdd(accuracy_out, accuracy);
        atomicAdd((unsigned long long*)valid_tokens_out, 1ULL);
    }
}

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
    cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
    cudaMemsetAsync(accuracy_out, 0, sizeof(float), stream);
    cudaMemsetAsync(valid_tokens_out, 0, sizeof(int64_t), stream);
    
    // One block per token
    const int num_blocks = total_tokens;
    const int threads = 256;
    
    fused_cross_entropy_single_pass<<<num_blocks, threads, 0, stream>>>(
        logits, labels, pad_token_id,
        loss_out, accuracy_out, valid_tokens_out,
        total_tokens, vocab_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}

}  // extern "C"