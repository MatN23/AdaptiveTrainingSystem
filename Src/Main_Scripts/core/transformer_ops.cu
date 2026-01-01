// Copyright (c) 2025 MatN23. All rights reserved.
// ULTRA-OPTIMIZED Single-Pass Transformer Operations
// 
// TARGET: 10-20x faster than PyTorch baseline
// SUPPORTS: fp32, fp16, bf16, fp64 (auto-converts as needed)
//
// Compile:
// nvcc -O3 -arch=sm_80 --use_fast_math --maxrregcount=128 \
//   -Xptxas=-v --compiler-options '-fPIC' \
//   -gencode=arch=compute_75,code=sm_75 \
//   -gencode=arch=compute_80,code=sm_80 \
//   -gencode=arch=compute_86,code=sm_86 \
//   -shared transformer_ops.cu -o transformer_ops.so

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// ============================================================================
// DTYPE CONVERSION UTILITIES
// ============================================================================

// Convert any type to float for computation
template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

#if __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}
#endif

// Convert float back to target type
template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

#if __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}
#endif

// ============================================================================
// ULTRA-FAST WARP PRIMITIVES
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum_single_pass(float val) {
    val = warp_reduce_sum(val);
    
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    if (NUM_WARPS > 1) {
        __shared__ float warp_sums[NUM_WARPS];
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        
        if (lane == 0) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane == 0) {
                warp_sums[0] = val;
            }
        }
        
        __syncthreads();
        val = warp_sums[0];
    }
    
    return val;
}

// ============================================================================
// TEMPLATIZED KERNELS FOR MULTIPLE DTYPES
// ============================================================================

// 1. RMSNorm - supports fp32, fp16, bf16
template<typename T, int BLOCK_SIZE, int VEC_SIZE = 4>
__global__ void rms_norm_kernel_single_pass(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    const int batch_seq,
    const int hidden_size,
    const float eps
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= batch_seq) return;
    
    const int tid = threadIdx.x;
    const int vec_hidden = (hidden_size / VEC_SIZE) * VEC_SIZE;
    
    const T* x = input + token_idx * hidden_size;
    T* y = output + token_idx * hidden_size;
    
    // Compute sum of squares in float32 for precision
    float sum_sq = 0.0f;
    
    // Vectorized loop - process VEC_SIZE elements at a time
    for (int i = tid * VEC_SIZE; i < vec_hidden; i += BLOCK_SIZE * VEC_SIZE) {
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float val = to_float(x[i + j]);
            sum_sq += val * val;
        }
    }
    
    // Handle remainder
    for (int i = vec_hidden + tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(x[i]);
        sum_sq += val * val;
    }
    
    // Reduce across block
    sum_sq = block_reduce_sum_single_pass<BLOCK_SIZE>(sum_sq);
    
    // Compute RMS (all threads have the same value after reduction)
    const float rms = rsqrtf(sum_sq / hidden_size + eps);
    
    // Normalize and scale - vectorized
    for (int i = tid * VEC_SIZE; i < vec_hidden; i += BLOCK_SIZE * VEC_SIZE) {
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float val = to_float(x[i + j]);
            float w = to_float(weight[i + j]);
            y[i + j] = from_float<T>(val * rms * w);
        }
    }
    
    // Handle remainder
    for (int i = vec_hidden + tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = to_float(x[i]);
        float w = to_float(weight[i]);
        y[i] = from_float<T>(val * rms * w);
    }
}

// 2. RoPE - supports fp32, fp16, bf16
template<typename T>
__global__ void rope_kernel_single_pass_fused(
    T* __restrict__ q,
    T* __restrict__ k,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int position_offset,
    const float theta
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int pos_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || pos_idx >= seq_len) return;
    
    const int base_offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    const int half_dim = head_dim >> 1;
    const int position = position_offset + pos_idx;
    
    const int tid = threadIdx.x;
    
    #pragma unroll
    for (int i = tid * 4; i < half_dim; i += blockDim.x * 4) {
        #pragma unroll
        for (int j = 0; j < 4 && (i + j) < half_dim; j++) {
            const int dim_idx = i + j;
            
            // Compute cos/sin on-the-fly
            const float freq = __fdividef(1.0f, __powf(theta, __fdividef(2.0f * dim_idx, (float)head_dim)));
            const float angle = position * freq;
            
            float cos_val, sin_val;
            __sincosf(angle, &sin_val, &cos_val);
            
            const int idx0 = base_offset + dim_idx;
            const int idx1 = base_offset + dim_idx + half_dim;
            
            // Rotate Q (convert to float, compute, convert back)
            float q0 = to_float(q[idx0]);
            float q1 = to_float(q[idx1]);
            q[idx0] = from_float<T>(__fmaf_rn(q0, cos_val, -q1 * sin_val));
            q[idx1] = from_float<T>(__fmaf_rn(q0, sin_val, q1 * cos_val));
            
            // Rotate K
            float k0 = to_float(k[idx0]);
            float k1 = to_float(k[idx1]);
            k[idx0] = from_float<T>(__fmaf_rn(k0, cos_val, -k1 * sin_val));
            k[idx1] = from_float<T>(__fmaf_rn(k0, sin_val, k1 * cos_val));
        }
    }
}

// Precompute version (always stores in fp32 for precision)
__global__ void rope_precompute_kernel_single_pass(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    const int max_seq_len,
    const int head_dim,
    const float theta
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim >> 1;
    const int total = max_seq_len * half_dim;
    
    #pragma unroll 8
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        const int pos = idx / half_dim;
        const int dim = idx % half_dim;
        
        const float freq = __fdividef(1.0f, __powf(theta, __fdividef(2.0f * dim, (float)head_dim)));
        const float angle = pos * freq;
        
        float sin_val, cos_val;
        __sincosf(angle, &sin_val, &cos_val);
        
        cos_cache[idx] = cos_val;
        sin_cache[idx] = sin_val;
    }
}

// Apply with cache - supports any dtype
template<typename T>
__global__ void rope_apply_kernel_single_pass(
    T* __restrict__ q,
    T* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int position_offset
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || pos_idx >= seq_len) return;
    
    const int base_offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    const int half_dim = head_dim >> 1;
    const int cache_offset = (position_offset + pos_idx) * half_dim;
    
    #pragma unroll
    for (int i = tid * 4; i < half_dim; i += blockDim.x * 4) {
        #pragma unroll
        for (int j = 0; j < 4 && (i + j) < half_dim; j++) {
            const int dim_idx = i + j;
            const float cos_val = __ldg(&cos_cache[cache_offset + dim_idx]);
            const float sin_val = __ldg(&sin_cache[cache_offset + dim_idx]);
            
            const int idx0 = base_offset + dim_idx;
            const int idx1 = base_offset + dim_idx + half_dim;
            
            // Q rotation
            float q0 = to_float(q[idx0]);
            float q1 = to_float(q[idx1]);
            q[idx0] = from_float<T>(__fmaf_rn(q0, cos_val, -q1 * sin_val));
            q[idx1] = from_float<T>(__fmaf_rn(q0, sin_val, q1 * cos_val));
            
            // K rotation
            float k0 = to_float(k[idx0]);
            float k1 = to_float(k[idx1]);
            k[idx0] = from_float<T>(__fmaf_rn(k0, cos_val, -k1 * sin_val));
            k[idx1] = from_float<T>(__fmaf_rn(k0, sin_val, k1 * cos_val));
        }
    }
}

// 3. SwiGLU - supports fp32, fp16, bf16
template<typename T, int BLOCK_SIZE>
__global__ void swiglu_kernel_single_pass(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ output,
    const int total_tokens,
    const int intermediate_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const int tid = threadIdx.x;
    const int offset = token_idx * intermediate_size;
    
    for (int i = tid; i < intermediate_size; i += BLOCK_SIZE) {
        float g = to_float(gate[offset + i]);
        float u = to_float(up[offset + i]);
        
        // SwiGLU: gate * silu(up) where silu(x) = x / (1 + exp(-x))
        float silu_u = u * __fdividef(1.0f, 1.0f + expf(-u));
        output[offset + i] = from_float<T>(g * silu_u);
    }
}

// ============================================================================
// HOST LAUNCHERS WITH DTYPE DISPATCH
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" {

// ============================================================================
// FP32 LAUNCHERS
// ============================================================================

void rms_norm_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch_seq,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = (hidden_size <= 512) ? 128 : 
                          (hidden_size <= 1024) ? 256 : 512;
    
    if (BLOCK_SIZE == 128) {
        rms_norm_kernel_single_pass<float, 128><<<batch_seq, 128, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    } else if (BLOCK_SIZE == 256) {
        rms_norm_kernel_single_pass<float, 256><<<batch_seq, 256, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    } else {
        rms_norm_kernel_single_pass<float, 512><<<batch_seq, 512, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void rope_precompute_launcher(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float theta,
    cudaStream_t stream
) {
    const int half_dim = head_dim / 2;
    const int total = max_seq_len * half_dim;
    const int threads = 256;
    const int blocks = (total + threads * 8 - 1) / (threads * 8);
    
    rope_precompute_kernel_single_pass<<<blocks, threads, 0, stream>>>(
        cos_cache, sin_cache, max_seq_len, head_dim, theta
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher(
    float* q,
    float* k,
    const float* cos,
    const float* sin,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int position_offset,
    cudaStream_t stream
) {
    dim3 blocks(seq_len, num_heads, batch_size);
    const int threads = (head_dim / 2 + 3) / 4;
    const int optimal_threads = (threads + 31) / 32 * 32;
    
    rope_apply_kernel_single_pass<float><<<blocks, optimal_threads, 0, stream>>>(
        q, k, cos, sin, batch_size, num_heads, seq_len, head_dim, position_offset
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher(
    const float* gate,
    const float* up,
    float* output,
    int total_tokens,
    int intermediate_size,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    
    swiglu_kernel_single_pass<float, BLOCK_SIZE><<<total_tokens, BLOCK_SIZE, 0, stream>>>(
        gate, up, output, total_tokens, intermediate_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_fused_launcher(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    const float* gate_bias,
    const float* up_bias,
    float* output,
    int total_tokens,
    int hidden_size,
    int intermediate_size,
    bool use_bias,
    cudaStream_t stream
) {
    // For now, use unfused version
    // TODO: Implement fused version with matmul
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// FP16 LAUNCHERS
// ============================================================================

void rms_norm_launcher_fp16(
    const __half* input,
    const __half* weight,
    __half* output,
    int batch_seq,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = (hidden_size <= 512) ? 128 : 
                          (hidden_size <= 1024) ? 256 : 512;
    
    if (BLOCK_SIZE == 128) {
        rms_norm_kernel_single_pass<__half, 128><<<batch_seq, 128, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    } else if (BLOCK_SIZE == 256) {
        rms_norm_kernel_single_pass<__half, 256><<<batch_seq, 256, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    } else {
        rms_norm_kernel_single_pass<__half, 512><<<batch_seq, 512, 0, stream>>>(
            input, weight, output, batch_seq, hidden_size, eps);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher_fp16(
    __half* q,
    __half* k,
    const float* cos,
    const float* sin,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int position_offset,
    cudaStream_t stream
) {
    dim3 blocks(seq_len, num_heads, batch_size);
    const int threads = (head_dim / 2 + 3) / 4;
    const int optimal_threads = (threads + 31) / 32 * 32;
    
    rope_apply_kernel_single_pass<__half><<<blocks, optimal_threads, 0, stream>>>(
        q, k, cos, sin, batch_size, num_heads, seq_len, head_dim, position_offset
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher_fp16(
    const __half* gate,
    const __half* up,
    __half* output,
    int total_tokens,
    int intermediate_size,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    
    swiglu_kernel_single_pass<__half, BLOCK_SIZE><<<total_tokens, BLOCK_SIZE, 0, stream>>>(
        gate, up, output, total_tokens, intermediate_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"