// fused_grad_clip.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ __forceinline__ double warp_reduce_sum_double(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

template<typename scalar_t>
__global__ void fused_grad_clip_kernel(
    scalar_t **grad_ptrs,
    const int *grad_sizes,
    const int num_tensors,
    const float max_norm,
    double *norm_out_double,
    float *clip_coef_out,
    float *final_norm_out
) {
    cg::grid_group grid = cg::this_grid();
    double thread_norm_sq = 0.0;

    for (int t = 0; t < num_tensors; t++) {
        scalar_t *grad = grad_ptrs[t];
        int size = grad_sizes[t];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
            thread_norm_sq += static_cast<double>(grad[i]) * static_cast<double>(grad[i]);
        }
    }

    __shared__ double shared_smem[WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;

    double block_sum = warp_reduce_sum_double(thread_norm_sq);
    if (lane == 0) shared_smem[wid] = block_sum;
    __syncthreads();

    if (wid == 0) {
        double val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_smem[lane] : 0.0;
        val = warp_reduce_sum_double(val);
        if (lane == 0) atomicAdd(norm_out_double, val);
    }

    __threadfence();
    grid.sync();

    if (grid.thread_rank() == 0) {
        double accumulated_norm_sq = *norm_out_double;
        float total_norm = sqrtf(static_cast<float>(accumulated_norm_sq + 1e-16f));
        float coef = fminf(max_norm / (total_norm + 1e-6f), 1.0f);
        *clip_coef_out = coef;
        *final_norm_out = total_norm;
    }
    grid.sync();

    float clip_coef = *clip_coef_out;
    if (clip_coef >= 0.999f) return;

    for (int t = 0; t < num_tensors; t++) {
        scalar_t *grad = grad_ptrs[t];
        int size = grad_sizes[t];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
            grad[i] = static_cast<scalar_t>(static_cast<float>(grad[i]) * clip_coef);
        }
    }
}

extern "C" {

void fused_grad_clip_launcher_fp32(
    float **grad_ptrs_device,
    int *grad_sizes_device,
    int num_tensors,
    float max_norm,
    float *norm_buffer,
    cudaStream_t stream
) {
    if (!norm_buffer) return;

    cudaMemsetAsync(norm_buffer, 0, 16, stream);
    double *norm_out_double = reinterpret_cast<double *>(norm_buffer);
    float *clip_coef_out = norm_buffer + 2;
    float *final_norm_out = norm_buffer + 3;

    int num_sms, blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fused_grad_clip_kernel<float>, 256, 0);
    int num_blocks = min(blocks_per_sm * num_sms, 1024);

    fused_grad_clip_kernel<float><<<num_blocks, 256, 0, stream>>>(
        grad_ptrs_device,
        grad_sizes_device,
        num_tensors,
        max_norm,
        norm_out_double,
        clip_coef_out,
        final_norm_out
    );
}

void fused_grad_clip_launcher_fp16(
    __half **grad_ptrs_device,
    int *grad_sizes_device,
    int num_tensors,
    float max_norm,
    float *norm_buffer,
    cudaStream_t stream
) {
    if (!norm_buffer) return;

    cudaMemsetAsync(norm_buffer, 0, 16, stream);
    double *norm_out_double = reinterpret_cast<double *>(norm_buffer);
    float *clip_coef_out = norm_buffer + 2;
    float *final_norm_out = norm_buffer + 3;

    int num_sms, blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fused_grad_clip_kernel<__half>, 256, 0);
    int num_blocks = min(blocks_per_sm * num_sms, 1024);

    fused_grad_clip_kernel<__half><<<num_blocks, 256, 0, stream>>>(
        grad_ptrs_device,
        grad_sizes_device,
        num_tensors,
        max_norm,
        norm_out_double,
        clip_coef_out,
        final_norm_out
    );
}

} // extern "C"