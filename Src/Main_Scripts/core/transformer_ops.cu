// Copyright (c) 2025 MatN23. All rights reserved.
// ULTRA-OPTIMIZED Single-Pass Transformer Operations
//
// TARGET: 10-20x faster than PyTorch baseline
// Compile:
// nvcc -O3 -arch=sm_75 --use_fast_math --maxrregcount=128 \
//   -Xptxas=-v --compiler-options '-fPIC' \
//   -gencode=arch=compute_75,code=sm_75 \
//   -gencode=arch=compute_80,code=sm_80 \
//   -gencode=arch=compute_86,code=sm_86 \
//   -shared transformer_ops.cu -o transformer_ops.so

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// DTYPE CONVERSION
// ============================================================================

template <typename T> __device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <> __device__ __forceinline__ float to_float<__half>(__half val) {
  return __half2float(val);
}

#if __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
#endif

template <typename T> __device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

template <> __device__ __forceinline__ __half from_float<__half>(float val) {
  return __float2half(val);
}

#if __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}
#endif

// ============================================================================
// WARP PRIMITIVES
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
  val = warp_reduce_sum(val);

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  if (NUM_WARPS > 1) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0)
      warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
      val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane == 0)
        warp_sums[0] = val;
    }
    __syncthreads();
    val = warp_sums[0];
  }

  return val;
}

// ============================================================================
// RMSNORM - FIXED ACCURACY + OPTIMIZED
// ============================================================================

template <typename T, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
    rms_norm_kernel_ultra(const T *__restrict__ input,
                          const T *__restrict__ weight, T *__restrict__ output,
                          const int batch_seq, const int hidden_size,
                          const float eps) {
  const int token_idx = blockIdx.x;
  if (token_idx >= batch_seq)
    return;

  const int tid = threadIdx.x;
  const T *x = input + (int64_t)token_idx * hidden_size;
  T *y = output + (int64_t)token_idx * hidden_size;

  // Compute sum of squares with higher precision
  float sum_sq = 0.0f;

  // Process 4 elements per thread for vectorization
  for (int i = tid * 4; i < hidden_size; i += BLOCK_SIZE * 4) {
#pragma unroll
    for (int j = 0; j < 4 && (i + j) < hidden_size; j++) {
      float val = to_float(x[i + j]);
      sum_sq = __fmaf_rn(val, val, sum_sq);
    }
  }

  // Reduce across block
  sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);

  // Compute RMS normalization factor
  const float mean_sq = sum_sq / float(hidden_size);
  const float rms_norm = rsqrtf(mean_sq + eps);

  // Normalize and scale
  for (int i = tid * 4; i < hidden_size; i += BLOCK_SIZE * 4) {
#pragma unroll
    for (int j = 0; j < 4 && (i + j) < hidden_size; j++) {
      int idx = i + j;
      float val = to_float(x[idx]);
      float w = to_float(weight[idx]);
      y[idx] = from_float<T>(val * rms_norm * w);
    }
  }
}

// ============================================================================
// ROPE - OPTIMIZED WITH CACHE
// ============================================================================

__global__ void __launch_bounds__(256, 4)
    rope_precompute_ultra(float *__restrict__ cos_cache,
                          float *__restrict__ sin_cache, const int max_seq_len,
                          const int head_dim, const float theta_base) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim >> 1;
  const int total = max_seq_len * half_dim;

  for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
    const int pos = i / half_dim;
    const int dim = i % half_dim;

    const float freq = 1.0f / powf(theta_base, (2.0f * dim) / float(head_dim));
    const float angle = pos * freq;

    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    cos_cache[i] = cos_val;
    sin_cache[i] = sin_val;
  }
}

template <typename T>
__global__ void __launch_bounds__(128, 4)
    rope_apply_ultra(T *__restrict__ q, T *__restrict__ k,
                     const float *__restrict__ cos_cache,
                     const float *__restrict__ sin_cache, const int batch_size,
                     const int num_heads, const int seq_len, const int head_dim,
                     const int position_offset) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int pos_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || pos_idx >= seq_len)
    return;

  const int64_t base_offset =
      ((int64_t)(batch_idx * num_heads + head_idx) * seq_len + pos_idx) *
      head_dim;
  const int half_dim = head_dim >> 1;
  const int64_t cache_offset = (int64_t)(position_offset + pos_idx) * half_dim;

  // Process pairs of dimensions
  for (int i = tid; i < half_dim; i += blockDim.x) {
    const float cos_val = cos_cache[cache_offset + i];
    const float sin_val = sin_cache[cache_offset + i];

    const int idx0 = base_offset + i;
    const int idx1 = base_offset + half_dim + i;

    // Rotate Q
    float q0 = to_float(q[idx0]);
    float q1 = to_float(q[idx1]);
    float q_rot0 = q0 * cos_val - q1 * sin_val;
    float q_rot1 = q0 * sin_val + q1 * cos_val;
    q[idx0] = from_float<T>(q_rot0);
    q[idx1] = from_float<T>(q_rot1);

    // Rotate K
    float k0 = to_float(k[idx0]);
    float k1 = to_float(k[idx1]);
    float k_rot0 = k0 * cos_val - k1 * sin_val;
    float k_rot1 = k0 * sin_val + k1 * cos_val;
    k[idx0] = from_float<T>(k_rot0);
    k[idx1] = from_float<T>(k_rot1);
  }
}

// ============================================================================
// SWIGLU - OPTIMIZED
// ============================================================================

template <typename T, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
    swiglu_kernel_ultra(const T *__restrict__ gate, const T *__restrict__ up,
                        T *__restrict__ output, const int total_tokens,
                        const int intermediate_size) {
  const int token_idx = blockIdx.x;
  if (token_idx >= total_tokens)
    return;

  const int tid = threadIdx.x;
  const int64_t offset = (int64_t)token_idx * intermediate_size;

  // Process 4 elements per thread
  for (int i = tid * 4; i < intermediate_size; i += BLOCK_SIZE * 4) {
#pragma unroll
    for (int j = 0; j < 4 && (i + j) < intermediate_size; j++) {
      const int idx = offset + i + j;
      float g = to_float(gate[idx]);
      float u = to_float(up[idx]);

      // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
      float silu_u = u / (1.0f + expf(-u));

      output[idx] = from_float<T>(g * silu_u);
    }
  }
}

// ============================================================================
// HOST LAUNCHERS
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

extern "C" {

void rms_norm_launcher(const float *input, const float *weight, float *output,
                       int batch_seq, int hidden_size, float eps,
                       cudaStream_t stream) {
  const int threads = 256;
  rms_norm_kernel_ultra<float, 256><<<batch_seq, threads, 0, stream>>>(
      input, weight, output, batch_seq, hidden_size, eps);
  CUDA_CHECK(cudaGetLastError());
}

void rms_norm_launcher_fp16(const __half *input, const __half *weight,
                            __half *output, int batch_seq, int hidden_size,
                            float eps, cudaStream_t stream) {
  const int threads = 256;
  rms_norm_kernel_ultra<__half, 256><<<batch_seq, threads, 0, stream>>>(
      input, weight, output, batch_seq, hidden_size, eps);
  CUDA_CHECK(cudaGetLastError());
}

void rms_norm_launcher_bf16(const __nv_bfloat16 *input,
                            const __nv_bfloat16 *weight, __nv_bfloat16 *output,
                            int batch_seq, int hidden_size, float eps,
                            cudaStream_t stream) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int threads = 256;
  rms_norm_kernel_ultra<__nv_bfloat16, 256><<<batch_seq, threads, 0, stream>>>(
      input, weight, output, batch_seq, hidden_size, eps);
#endif
  CUDA_CHECK(cudaGetLastError());
}

void rope_precompute_launcher(float *cos_cache, float *sin_cache,
                              int max_seq_len, int head_dim, float theta,
                              cudaStream_t stream) {
  const int total = max_seq_len * (head_dim / 2);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  rope_precompute_ultra<<<blocks, threads, 0, stream>>>(
      cos_cache, sin_cache, max_seq_len, head_dim, theta);
  CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher(float *q, float *k, const float *cos, const float *sin,
                         int batch_size, int num_heads, int seq_len,
                         int head_dim, int position_offset,
                         cudaStream_t stream) {
  dim3 blocks(seq_len, num_heads, batch_size);
  rope_apply_ultra<float>
      <<<blocks, 128, 0, stream>>>(q, k, cos, sin, batch_size, num_heads,
                                   seq_len, head_dim, position_offset);
  CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher_fp16(__half *q, __half *k, const float *cos,
                              const float *sin, int batch_size, int num_heads,
                              int seq_len, int head_dim, int position_offset,
                              cudaStream_t stream) {
  dim3 blocks(seq_len, num_heads, batch_size);
  rope_apply_ultra<__half>
      <<<blocks, 128, 0, stream>>>(q, k, cos, sin, batch_size, num_heads,
                                   seq_len, head_dim, position_offset);
  CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher_bf16(__nv_bfloat16 *q, __nv_bfloat16 *k,
                              const float *cos, const float *sin,
                              int batch_size, int num_heads, int seq_len,
                              int head_dim, int position_offset,
                              cudaStream_t stream) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  dim3 blocks(seq_len, num_heads, batch_size);
  rope_apply_ultra<__nv_bfloat16>
      <<<blocks, 128, 0, stream>>>(q, k, cos, sin, batch_size, num_heads,
                                   seq_len, head_dim, position_offset);
#endif
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher(const float *gate, const float *up, float *output,
                     int total_tokens, int intermediate_size,
                     cudaStream_t stream) {
  swiglu_kernel_ultra<float, 256><<<total_tokens, 256, 0, stream>>>(
      gate, up, output, total_tokens, intermediate_size);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher_fp16(const __half *gate, const __half *up, __half *output,
                          int total_tokens, int intermediate_size,
                          cudaStream_t stream) {
  swiglu_kernel_ultra<__half, 256><<<total_tokens, 256, 0, stream>>>(
      gate, up, output, total_tokens, intermediate_size);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher_bf16(const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
                          __nv_bfloat16 *output, int total_tokens,
                          int intermediate_size, cudaStream_t stream) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  swiglu_kernel_ultra<__nv_bfloat16, 256><<<total_tokens, 256, 0, stream>>>(
      gate, up, output, total_tokens, intermediate_size);
#endif
  CUDA_CHECK(cudaGetLastError());
}
}