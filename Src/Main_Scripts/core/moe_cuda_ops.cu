// Copyright (c) 2025 MatN23. All rights reserved.
// FIXED: Proper error handling, dtype matching, and device guards

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <limits>

#define WARP_SIZE 32

// CRITICAL: Throw exceptions on CUDA errors instead of silent failures
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + \
                                   cudaGetErrorString(err)); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel execution failed: ") + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

// Tensor validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// =============================================================================
// KERNEL: Top-K Gating
// =============================================================================

__global__ void topk_gating_kernel_optimized(
    const float* __restrict__ gate_logits,
    int64_t* __restrict__ top_k_indices,      // FIXED: int64 not int32
    float* __restrict__ top_k_weights,
    const int num_tokens,
    const int num_experts,
    const int k,
    const float temperature
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int token_idx = warp_id;
    
    if (token_idx >= num_tokens) return;
    
    const float* token_logits = gate_logits + token_idx * num_experts;
    
    float local_vals[4];
    int local_idxs[4];
    
    #pragma unroll
    for (int i = 0; i < k; i++) {
        local_vals[i] = -INFINITY;
        local_idxs[i] = -1;
    }
    
    for (int base = lane_id; base < num_experts; base += WARP_SIZE) {
        float val = token_logits[base] / temperature;
        int expert_id = base;
        
        if (k == 2) {
            if (val > local_vals[0]) {
                local_vals[1] = local_vals[0];
                local_idxs[1] = local_idxs[0];
                local_vals[0] = val;
                local_idxs[0] = expert_id;
            } else if (val > local_vals[1]) {
                local_vals[1] = val;
                local_idxs[1] = expert_id;
            }
        } else {
            for (int i = 0; i < k; i++) {
                if (val > local_vals[i]) {
                    for (int j = k - 1; j > i; j--) {
                        local_vals[j] = local_vals[j - 1];
                        local_idxs[j] = local_idxs[j - 1];
                    }
                    local_vals[i] = val;
                    local_idxs[i] = expert_id;
                    break;
                }
            }
        }
    }
    
    for (int i = 0; i < k; i++) {
        float best_val = local_vals[i];
        int best_idx = local_idxs[i];
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        
        best_val = __shfl_sync(0xffffffff, best_val, 0);
        best_idx = __shfl_sync(0xffffffff, best_idx, 0);
        
        if (best_idx == local_idxs[i]) {
            local_vals[i] = -INFINITY;
        }
        
        if (lane_id == 0) {
            local_vals[i] = best_val;
            local_idxs[i] = best_idx;
        }
    }
    
    if (lane_id == 0) {
        float max_logit = local_vals[0];
        #pragma unroll
        for (int i = 1; i < k; i++) {
            max_logit = fmaxf(max_logit, local_vals[i]);
        }
        
        float sum_exp = 0.0f;
        #pragma unroll
        for (int i = 0; i < k; i++) {
            float exp_val = expf(local_vals[i] - max_logit);
            local_vals[i] = exp_val;
            sum_exp += exp_val;
        }
        
        float inv_sum = 1.0f / sum_exp;
        int64_t* out_indices = top_k_indices + token_idx * k;  // FIXED: int64
        float* out_weights = top_k_weights + token_idx * k;
        
        #pragma unroll
        for (int i = 0; i < k; i++) {
            out_weights[i] = local_vals[i] * inv_sum;
            out_indices[i] = local_idxs[i];  // Implicit cast to int64
        }
    }
}

// =============================================================================
// KERNEL: Dispatch Tokens
// =============================================================================

__global__ void dispatch_tokens_kernel_optimized(
    const float* __restrict__ tokens,
    const int64_t* __restrict__ top_k_indices,  // FIXED: int64
    int* __restrict__ expert_positions,
    float* __restrict__ expert_inputs,
    int64_t* __restrict__ token_map,            // FIXED: int64
    const int num_tokens,
    const int num_experts,
    const int k,
    const int hidden_dim,
    const int capacity
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    const float* token_data = tokens + token_idx * hidden_dim;
    const int64_t* token_experts = top_k_indices + token_idx * k;
    
    __shared__ int shared_positions[8];
    
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            int expert_id = (int)token_experts[i];  // Cast int64 to int
            if (expert_id >= 0 && expert_id < num_experts) {
                shared_positions[i] = atomicAdd(&expert_positions[expert_id], 1);
            } else {
                shared_positions[i] = -1;
            }
        }
    }
    __syncthreads();
    
    for (int i = 0; i < k; i++) {
        int expert_id = (int)token_experts[i];
        int pos = shared_positions[i];
        
        if (pos < 0 || pos >= capacity || expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        
        if (tid == 0) {
            token_map[expert_id * capacity + pos] = token_idx * k + i;
        }
        
        float* expert_input = expert_inputs + (expert_id * capacity + pos) * hidden_dim;
        
        if (hidden_dim % 4 == 0 && ((size_t)token_data % 16 == 0)) {
            const int vec_dim = hidden_dim / 4;
            for (int d = tid; d < vec_dim; d += blockDim.x) {
                reinterpret_cast<float4*>(expert_input)[d] = 
                    reinterpret_cast<const float4*>(token_data)[d];
            }
        } else {
            for (int d = tid; d < hidden_dim; d += blockDim.x) {
                expert_input[d] = token_data[d];
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// KERNEL: Combine Expert Outputs
// =============================================================================

__global__ void combine_expert_outputs_kernel_optimized(
    const float* __restrict__ expert_outputs,
    const int64_t* __restrict__ token_map,     // FIXED: int64
    const float* __restrict__ top_k_weights,
    float* __restrict__ combined_output,
    const int num_experts,
    const int capacity,
    const int hidden_dim,
    const int num_tokens,
    const int k
) {
    const int expert_id = blockIdx.x;
    const int pos = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (expert_id >= num_experts || pos >= capacity) return;
    
    int64_t token_weight_idx = token_map[expert_id * capacity + pos];
    if (token_weight_idx < 0) return;
    
    int token_idx = (int)(token_weight_idx / k);
    if (token_idx >= num_tokens) return;
    
    float weight = top_k_weights[token_weight_idx];
    
    const float* expert_out = expert_outputs + (expert_id * capacity + pos) * hidden_dim;
    float* output = combined_output + token_idx * hidden_dim;
    
    if (hidden_dim % 4 == 0) {
        for (int d = tid; d < hidden_dim / 4; d += blockDim.x) {
            float4 vec = reinterpret_cast<const float4*>(expert_out)[d];
            vec.x *= weight;
            vec.y *= weight;
            vec.z *= weight;
            vec.w *= weight;
            
            atomicAdd(&output[d * 4 + 0], vec.x);
            atomicAdd(&output[d * 4 + 1], vec.y);
            atomicAdd(&output[d * 4 + 2], vec.z);
            atomicAdd(&output[d * 4 + 3], vec.w);
        }
    } else {
        for (int d = tid; d < hidden_dim; d += blockDim.x) {
            atomicAdd(&output[d], weight * expert_out[d]);
        }
    }
}

// =============================================================================
// C++ INTERFACE FUNCTIONS
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor> topk_gating_cuda(
    torch::Tensor gate_logits,
    int64_t k,
    double temperature
) {
    CHECK_INPUT(gate_logits);
    TORCH_CHECK(gate_logits.dim() == 2, "gate_logits must be 2D");
    TORCH_CHECK(gate_logits.dtype() == torch::kFloat32, "gate_logits must be float32");
    TORCH_CHECK(k > 0 && k <= 8, "k must be in [1, 8]");
    
    const int64_t num_tokens = gate_logits.size(0);
    const int64_t num_experts = gate_logits.size(1);
    
    TORCH_CHECK(num_tokens > 0 && num_experts > 0, "Invalid dimensions");
    
    // FIXED: Use device guard to handle multi-GPU
    c10::cuda::CUDAGuard device_guard(gate_logits.device());
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(gate_logits.device());
    
    auto top_k_weights = torch::empty({num_tokens, k}, options);
    
    // FIXED: Use int64 (PyTorch default) not int32
    auto top_k_indices = torch::empty({num_tokens, k}, options.dtype(torch::kInt64));
    
    // FIXED: Use current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int warps_per_block = 8;
    const int threads = warps_per_block * WARP_SIZE;
    const int blocks = std::max(1, (int)((num_tokens + warps_per_block - 1) / warps_per_block));
    
    topk_gating_kernel_optimized<<<blocks, threads, 0, stream>>>(
        gate_logits.data_ptr<float>(),
        top_k_indices.data_ptr<int64_t>(),  // FIXED: int64
        top_k_weights.data_ptr<float>(),
        (int)num_tokens,
        (int)num_experts,
        (int)k,
        (float)temperature
    );
    
    CUDA_CHECK_KERNEL();  // FIXED: Actually check for errors
    
    return std::make_tuple(top_k_indices, top_k_weights);
}

std::tuple<torch::Tensor, torch::Tensor> dispatch_tokens_cuda(
    torch::Tensor tokens,
    torch::Tensor top_k_indices,
    int64_t num_experts,
    int64_t capacity
) {
    CHECK_INPUT(tokens);
    CHECK_INPUT(top_k_indices);
    
    TORCH_CHECK(tokens.dim() == 2, "tokens must be 2D");
    TORCH_CHECK(top_k_indices.dim() == 2, "top_k_indices must be 2D");
    TORCH_CHECK(tokens.dtype() == torch::kFloat32, "tokens must be float32");
    TORCH_CHECK(top_k_indices.dtype() == torch::kInt64, "top_k_indices must be int64");
    
    const int64_t num_tokens = tokens.size(0);
    const int64_t hidden_dim = tokens.size(1);
    const int64_t k = top_k_indices.size(1);
    
    TORCH_CHECK(num_tokens == top_k_indices.size(0), "Shape mismatch");
    TORCH_CHECK(num_experts > 0 && capacity > 0, "Invalid capacity");
    
    c10::cuda::CUDAGuard device_guard(tokens.device());
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(tokens.device());
    
    auto expert_inputs = torch::zeros({num_experts, capacity, hidden_dim}, options);
    
    // FIXED: token_map uses int64
    auto token_map = torch::full({num_experts, capacity}, -1, options.dtype(torch::kInt64));
    auto expert_positions = torch::zeros({num_experts}, options.dtype(torch::kInt32));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int threads = 256;
    const int blocks = std::max(1, (int)num_tokens);
    
    dispatch_tokens_kernel_optimized<<<blocks, threads, 0, stream>>>(
        tokens.data_ptr<float>(),
        top_k_indices.data_ptr<int64_t>(),  // FIXED: int64
        expert_positions.data_ptr<int>(),
        expert_inputs.data_ptr<float>(),
        token_map.data_ptr<int64_t>(),      // FIXED: int64
        (int)num_tokens,
        (int)num_experts,
        (int)k,
        (int)hidden_dim,
        (int)capacity
    );
    
    CUDA_CHECK_KERNEL();
    
    return std::make_tuple(expert_inputs, token_map);
}

torch::Tensor combine_expert_outputs_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor token_map,
    torch::Tensor top_k_weights,
    int64_t num_tokens,
    int64_t k
) {
    CHECK_INPUT(expert_outputs);
    CHECK_INPUT(token_map);
    CHECK_INPUT(top_k_weights);
    
    TORCH_CHECK(expert_outputs.dim() == 3, "expert_outputs must be 3D");
    TORCH_CHECK(token_map.dim() == 2, "token_map must be 2D");
    TORCH_CHECK(expert_outputs.dtype() == torch::kFloat32, "expert_outputs must be float32");
    TORCH_CHECK(token_map.dtype() == torch::kInt64, "token_map must be int64");
    
    const int64_t num_experts = expert_outputs.size(0);
    const int64_t capacity = expert_outputs.size(1);
    const int64_t hidden_dim = expert_outputs.size(2);
    
    c10::cuda::CUDAGuard device_guard(expert_outputs.device());
    
    auto combined = torch::zeros({num_tokens, hidden_dim}, 
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(expert_outputs.device()));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    dim3 grid((int)num_experts, (int)capacity);
    const int threads = 256;
    
    // Flatten weights if needed
    torch::Tensor weights_flat = top_k_weights.dim() == 2 ? 
        top_k_weights.flatten() : top_k_weights;
    
    combine_expert_outputs_kernel_optimized<<<grid, threads, 0, stream>>>(
        expert_outputs.data_ptr<float>(),
        token_map.data_ptr<int64_t>(),  // FIXED: int64
        weights_flat.data_ptr<float>(),
        combined.data_ptr<float>(),
        (int)num_experts,
        (int)capacity,
        (int)hidden_dim,
        (int)num_tokens,
        (int)k
    );
    
    CUDA_CHECK_KERNEL();
    
    return combined;
}

// Stub implementations
torch::Tensor compute_expert_capacity_cuda(torch::Tensor top_k_indices, int64_t num_experts) {
    return torch::zeros({num_experts}, top_k_indices.options().dtype(torch::kInt32));
}

torch::Tensor compute_load_balancing_loss_cuda(torch::Tensor gate_probs, torch::Tensor top_k_indices) {
    return torch::zeros({1}, gate_probs.options());
}

// =============================================================================
// PYBIND11 MODULE
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_gating", &topk_gating_cuda, "Top-K gating (CUDA)");
    m.def("dispatch_tokens", &dispatch_tokens_cuda, "Token dispatch (CUDA)");
    m.def("combine_expert_outputs", &combine_expert_outputs_cuda, "Combine outputs (CUDA)");
    m.def("compute_expert_capacity", &compute_expert_capacity_cuda, "Expert capacity (CUDA)");
    m.def("compute_load_balancing_loss", &compute_load_balancing_loss_cuda, "Load balance loss (CUDA)");
}