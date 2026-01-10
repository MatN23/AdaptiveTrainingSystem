// Copyright (c) 2025 MatN23. All rights reserved.
//
// Optimized Fused Gradient Clipping Kernel (STABLE VERSION)
// - Uses Cooperative Groups for single-kernel grid synchronization
// - Double-precision accumulation for gradient norms
// - Explicit error checking for occupancy and cooperative launches

#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Set to 1 to enable debug printf, 0 to disable
#define GRAD_CLIP_DEBUG 0

__device__ __forceinline__ double block_reduce_sum_double(double val) {
  __shared__ double shared_smem[32];

  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }

  if (lane == 0)
    shared_smem[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared_smem[lane] : 0.0;
  if (wid == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
  }

  return val;
}

__global__ void fused_grad_clip_kernel(float **grad_ptrs, const int *grad_sizes,
                                       const int num_tensors,
                                       const float max_norm,
                                       double *norm_out_double,
                                       float *clip_coef_out,
                                       float *final_norm_out) {

  cg::grid_group grid = cg::this_grid();

  // PASS 1: Compute total norm squared
  double thread_norm_sq = 0.0;
  for (int t = 0; t < num_tensors; t++) {
    float *grad = grad_ptrs[t];
    const int size = grad_sizes[t];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
      float val = grad[i];
      thread_norm_sq += (double)val * val;
    }
  }

  double block_sum = block_reduce_sum_double(thread_norm_sq);
  if (threadIdx.x == 0) {
    atomicAdd(norm_out_double, block_sum);
  }

  // CRITICAL FIX: Ensure atomicAdd writes are visible to all threads before
  // sync
  __threadfence();

  grid.sync();

  // PASS 2: Compute clip coefficient
  if (grid.thread_rank() == 0) {
    double accumulated_norm_sq = *norm_out_double;
    float total_norm = sqrtf((float)accumulated_norm_sq);
#if GRAD_CLIP_DEBUG
    printf("DEBUG GRAD_CLIP: norm_sq=%.6f, total_norm=%.6f, max_norm=%.6f\n",
           accumulated_norm_sq, total_norm, max_norm);
#endif
    float coef = fminf(max_norm / (total_norm + 1e-6f), 1.0f);
    *clip_coef_out = coef;
    *final_norm_out = total_norm;
  }

  grid.sync();

  float clip_coef = *clip_coef_out;
  if (clip_coef >= 0.999f)
    return;

  // PASS 3: Apply Clipping
  for (int t = 0; t < num_tensors; t++) {
    float *grad = grad_ptrs[t];
    const int size = grad_sizes[t];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
      grad[i] *= clip_coef;
    }
  }
}

extern "C" {

void fused_grad_clip_launcher(float **grad_ptrs_device, int *grad_sizes_device,
                              int num_tensors, float max_norm,
                              float *norm_buffer, cudaStream_t stream) {

  if (norm_buffer == nullptr)
    return;

  cudaMemsetAsync(norm_buffer, 0, 16, stream);

  double *norm_out_double = reinterpret_cast<double *>(norm_buffer);
  float *clip_coef_out = norm_buffer + 2;
  float *final_norm_out = norm_buffer + 3;

  static int cached_num_blocks = -1;
  static int cached_num_sms = -1;

  if (cached_num_blocks == -1) {
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, fused_grad_clip_kernel, 256, 0);

    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&cached_num_sms, cudaDevAttrMultiProcessorCount,
                           device);

    // Optimal grid size to fill the GPU
    cached_num_blocks = blocks_per_sm * cached_num_sms;

    // Optional: Safety limit for very large GPUs
    if (cached_num_blocks > 2048)
      cached_num_blocks = 2048;
  }

  int num_blocks = cached_num_blocks;

  void *kernel_args[] = {&grad_ptrs_device, &grad_sizes_device, &num_tensors,
                         &max_norm,         &norm_out_double,   &clip_coef_out,
                         &final_norm_out};

  dim3 gridDim(num_blocks);
  dim3 blockDim(256);

  cudaError_t err =
      cudaLaunchCooperativeKernel((void *)fused_grad_clip_kernel, gridDim,
                                  blockDim, kernel_args, 0, stream);

  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Cooperative Launch Failed: %s\n",
            cudaGetErrorString(err));
  }
}

} // extern "C"