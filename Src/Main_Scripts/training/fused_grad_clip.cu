// Copyright (c) 2025 MatN23. All rights reserved.
// FIXED: Proper grid synchronization using two-kernel approach
//
// Compile with:
// nvcc -O3 -arch=sm_75 --use_fast_math --ptxas-options=-v \
//      --compiler-options '-fPIC' -shared fused_grad_clip.cu -o
//      fused_grad_clip.so

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}

// ============================================================================
// KERNEL 1: Compute gradient norm
// ============================================================================
__global__ void __launch_bounds__(256, 8)
    compute_grad_norm_kernel(float **__restrict__ grad_ptrs,
                             const int *__restrict__ grad_sizes,
                             const int num_tensors,
                             float *__restrict__ norm_out) {
  __shared__ float smem[32];

  float thread_norm_sq = 0.0f;

  // Each thread processes elements from all tensors
  for (int tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
    float *grad = grad_ptrs[tensor_idx];
    const int64_t size = grad_sizes[tensor_idx];

    // Vectorized processing
    const int64_t vec_size = size / 4;
    const int64_t vec_start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t vec_stride = (int64_t)gridDim.x * blockDim.x;

    // Process float4 chunks
    for (int64_t i = vec_start; i < vec_size; i += vec_stride) {
      float4 vals = __ldg(reinterpret_cast<const float4 *>(&grad[i * 4]));
      thread_norm_sq +=
          vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }

    // Handle remainder
    const int64_t remainder_start = vec_size * 4;
    for (int64_t i =
             remainder_start + threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
         i < size; i += (int64_t)gridDim.x * blockDim.x) {
      float val = __ldg(&grad[i]);
      thread_norm_sq += val * val;
    }
  }

  // Warp reduce
  thread_norm_sq = warp_reduce_sum(thread_norm_sq);

  // Block reduce
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  if (lane == 0)
    smem[wid] = thread_norm_sq;
  __syncthreads();

  float block_norm_sq = 0.0f;
  if (wid == 0) {
    thread_norm_sq = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    block_norm_sq = warp_reduce_sum(thread_norm_sq);
  }

  // Grid reduce using atomics (once per block)
  if (threadIdx.x == 0) {
    atomicAdd(norm_out, block_norm_sq);
  }
}

// ============================================================================
// KERNEL 2: Apply gradient clipping
// ============================================================================
__global__ void __launch_bounds__(256, 8)
    apply_grad_clip_kernel(float **__restrict__ grad_ptrs,
                           const int *__restrict__ grad_sizes,
                           const int num_tensors, const float clip_coef) {
  // Early exit if no clipping needed
  if (clip_coef >= 0.999f)
    return;

  // Apply clipping (vectorized)
  for (int tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
    float *grad = grad_ptrs[tensor_idx];
    const int64_t size = grad_sizes[tensor_idx];

    const int64_t vec_size = size / 4;
    const int64_t vec_start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t vec_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = vec_start; i < vec_size; i += vec_stride) {
      float4 *vec_ptr = reinterpret_cast<float4 *>(grad);
      float4 vals = vec_ptr[i];
      vals.x *= clip_coef;
      vals.y *= clip_coef;
      vals.z *= clip_coef;
      vals.w *= clip_coef;
      vec_ptr[i] = vals;
    }

    const int64_t remainder_start = vec_size * 4;
    for (int64_t i =
             remainder_start + threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
         i < size; i += (int64_t)gridDim.x * blockDim.x) {
      grad[i] *= clip_coef;
    }
  }
}

// ============================================================================
// HOST LAUNCHER - TWO KERNEL APPROACH
// ============================================================================
extern "C" {

float fused_grad_clip_launcher(float **grad_ptrs_device, int *grad_sizes_device,
                               int num_tensors, float max_norm,
                               cudaStream_t stream) {
  // Allocate output buffer
  float *norm_buffer;
  cudaMallocAsync(&norm_buffer, sizeof(float), stream);
  cudaMemsetAsync(norm_buffer, 0, sizeof(float), stream);

  const int threads = 256;
  const int blocks = 256;

  // KERNEL 1: Compute norm (ALL blocks must finish)
  compute_grad_norm_kernel<<<blocks, threads, 0, stream>>>(
      grad_ptrs_device, grad_sizes_device, num_tensors, norm_buffer);

  // Read back the norm squared (IMPLICIT SYNC - kernel must finish!)
  float total_norm_sq;
  cudaMemcpyAsync(&total_norm_sq, norm_buffer, sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); // ✅ CRITICAL: Wait for norm computation

  const float total_norm = sqrtf(total_norm_sq);
  const float clip_coef = fminf(max_norm / (total_norm + 1e-6f), 1.0f);

  // KERNEL 2: Apply clipping with computed coefficient
  apply_grad_clip_kernel<<<blocks, threads, 0, stream>>>(
      grad_ptrs_device, grad_sizes_device, num_tensors, clip_coef);

  cudaFreeAsync(norm_buffer, stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }

  return total_norm;
}

} // extern "C"