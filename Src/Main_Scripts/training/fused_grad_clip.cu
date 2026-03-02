// Copyright (c) 2025 MatN23. All rights reserved.
// Licensed under the Custom License below.

#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define GRAD_CLIP_DEBUG 1  // Set to 0 to disable debug prints

// Warp reduce sum for double
__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

// Block reduce sum (32 threads per warp assumed)
__device__ double block_reduce_sum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void fused_grad_clip_kernel(
    float **grad_ptrs, const int *grad_sizes, int num_tensors,
    float max_norm, double *norm_out, float *clip_coef_out, float *final_norm_out)
{
    cg::grid_group grid = cg::this_grid();

    // PASS 1: compute total squared norm
    double thread_norm_sq = 0.0;
    for (int t = 0; t < num_tensors; t++) {
        float *grad = grad_ptrs[t];
        int size = grad_sizes[t];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
            thread_norm_sq += static_cast<double>(grad[i]) * grad[i];
    }

    double block_sum = block_reduce_sum(thread_norm_sq);
    if (threadIdx.x == 0) atomicAdd(norm_out, block_sum);

    __threadfence();
    grid.sync();

    // PASS 2: compute clip coefficient
    if (grid.thread_rank() == 0) {
        double total_norm_sq = *norm_out;
        float total_norm = sqrtf(static_cast<float>(total_norm_sq));
        *final_norm_out = total_norm;

        // Safeguard tiny norms for FP16
        float coef = (total_norm < 1e-6f) ? 1.0f : fminf(max_norm / total_norm, 1.0f);
        *clip_coef_out = coef;

#if GRAD_CLIP_DEBUG
        printf("[DEBUG] total_norm=%.6f, clip_coef=%.6f, max_norm=%.6f\n",
               total_norm, coef, max_norm);
#endif
    }

    grid.sync();

    // PASS 3: apply clipping
    float coef = *clip_coef_out;
    if (coef >= 0.999f) return; // almost no clipping needed

    for (int t = 0; t < num_tensors; t++) {
        float *grad = grad_ptrs[t];
        int size = grad_sizes[t];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
            grad[i] *= coef;
    }
}

extern "C" {

void fused_grad_clip_launcher(
    float **grad_ptrs_device, int *grad_sizes_device,
    int num_tensors, float max_norm,
    double *norm_buffer, cudaStream_t stream)
{
    if (!norm_buffer) return;

    float *clip_coef_out = reinterpret_cast<float*>(norm_buffer + 1);
    float *final_norm_out = reinterpret_cast<float*>(norm_buffer + 2);

    // Clear norm
    cudaMemsetAsync(norm_buffer, 0, sizeof(double), stream);

    int blocks_per_sm, num_sms, cached_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fused_grad_clip_kernel, 256, 0);
    int device; cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    cached_blocks = min(blocks_per_sm * num_sms, 2048);

    dim3 grid(cached_blocks);
    dim3 block(256);

    void *args[] = {
        &grad_ptrs_device, &grad_sizes_device, &num_tensors, &max_norm,
        &norm_buffer, &clip_coef_out, &final_norm_out
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)fused_grad_clip_kernel, grid, block, args, 0, stream
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Launch Failed: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"