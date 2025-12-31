// Copyright (c) 2025 MatN23. All rights reserved.
// ULTRA-OPTIMIZED fused_loss.cu - Back to V4 + Better Optimizations
// 
// Revert persistent kernel, add instead:
// - Wider SIMD (8 elements)
// - Branchless operations
// - Better register usage
// - Manual loop unrolling
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
        
        // Merge argmax (branchless)
        int other_greater = (other_argmax_val > state.argmax_val);
        state.argmax_val = other_greater ? other_argmax_val : state.argmax_val;
        state.argmax_idx = other_greater ? other_argmax_idx : state.argmax_idx;
    }
    return state;
}

// OPTIMIZED: Single-pass with 8-wide SIMD and branchless operations
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
    // SINGLE PASS: Online softmax + argmax with 8-wide SIMD
    // =========================================================================
    SoftmaxState state;
    state.m = -FLT_MAX;
    state.d = 0.0f;
    state.argmax_val = -FLT_MAX;
    state.argmax_idx = 0;
    
    float label_logit = 0.0f;
    
    // Process 8 elements at a time for better ILP
    const int vec8_end = (vocab_size / 8) * 8;
    
    for (int i = threadIdx.x * 8; i < vec8_end; i += blockDim.x * 8) {
        // Load 8 floats (2x float4)
        float4 vals1 = __ldg(reinterpret_cast<const float4*>(&logit_row[i]));
        float4 vals2 = __ldg(reinterpret_cast<const float4*>(&logit_row[i + 4]));
        
        // Manually unroll for 8 elements (better than loop)
        float x;
        int idx;
        int is_label, is_greater;
        
        // Element 0
        x = vals1.x; idx = i;
        float m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 1
        x = vals1.y; idx = i + 1;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 2
        x = vals1.z; idx = i + 2;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 3
        x = vals1.w; idx = i + 3;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 4
        x = vals2.x; idx = i + 4;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 5
        x = vals2.y; idx = i + 5;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 6
        x = vals2.z; idx = i + 6;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        // Element 7
        x = vals2.w; idx = i + 7;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
    }
    
    // Handle remainder (4 elements at a time)
    const int vec4_end = (vocab_size / 4) * 4;
    for (int i = vec8_end + threadIdx.x * 4; i < vec4_end; i += blockDim.x * 4) {
        float4 vals = __ldg(reinterpret_cast<const float4*>(&logit_row[i]));
        
        float x; int idx; int is_label, is_greater;
        
        x = vals.x; idx = i;
        float m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        x = vals.y; idx = i + 1;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        x = vals.z; idx = i + 2;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
        
        x = vals.w; idx = i + 3;
        m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? idx : state.argmax_idx;
        is_label = (idx == label);
        label_logit = is_label ? x : label_logit;
    }
    
    // Final scalar remainder
    for (int i = vec4_end + threadIdx.x; i < vocab_size; i += blockDim.x) {
        float x = __ldg(&logit_row[i]);
        
        float m_old = state.m;
        state.m = fmaxf(state.m, x);
        state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);
        
        int is_greater = (x > state.argmax_val);
        state.argmax_val = is_greater ? x : state.argmax_val;
        state.argmax_idx = is_greater ? i : state.argmax_idx;
        
        int is_label = (i == label);
        label_logit = is_label ? x : label_logit;
    }
    
    // =========================================================================
    // Warp-level reduction
    // =========================================================================
    state = warp_reduce_softmax_state(state);
    
    __shared__ float smem_m[32];
    __shared__ float smem_d[32];
    __shared__ float smem_argmax_val[32];
    __shared__ int smem_argmax_idx[32];
    __shared__ float smem_label_logit[256];
    
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    
    if (lane == 0) {
        smem_m[wid] = state.m;
        smem_d[wid] = state.d;
        smem_argmax_val[wid] = state.argmax_val;
        smem_argmax_idx[wid] = state.argmax_idx;
    }
    smem_label_logit[threadIdx.x] = label_logit;
    __syncthreads();
    
    // =========================================================================
    // Block-level reduction
    // =========================================================================
    if (wid == 0) {
        if (lane < 8) {  // Only 8 warps
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
        
        // Find label_logit
        float final_label_logit = smem_label_logit[0];
        #pragma unroll
        for (int i = 1; i < 256; i++) {
            final_label_logit = fmaxf(final_label_logit, smem_label_logit[i]);
        }
        
        if (final_label_logit == 0.0f) {
            final_label_logit = logit_row[label];
        }
        
        const float loss = -((final_label_logit - max_logit) - __logf(sum_exp + 1e-10f));
        const float accuracy = (pred_idx == (int)label) ? 1.0f : 0.0f;
        
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
    
    // One block per token (this works best for this workload!)
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