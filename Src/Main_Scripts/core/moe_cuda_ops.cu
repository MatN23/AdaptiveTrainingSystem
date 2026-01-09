// Copyright (c) 2025 MatN23. All rights reserved.
// ULTRA-OPTIMIZED Single-Pass MoE Operations
//
// TARGET: 5-10x faster than PyTorch baseline
// KEY: Single-pass top-k, fused dispatch+combine, zero redundancy
//
// Compile:
// nvcc -O3 -arch=sm_80 --use_fast_math --maxrregcount=96 \
//   -Xptxas=-v --compiler-options '-fPIC' \
//   -gencode=arch=compute_75,code=sm_75 \
//   -gencode=arch=compute_80,code=sm_80 \
//   -gencode=arch=compute_86,code=sm_86 \
//   -shared moe_cuda_ops.cu -o moe_cuda_ops.so

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE 32
#define MAX_K 8

// Error checking with exceptions
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    TORCH_CHECK(err == cudaSuccess,                                            \
                "CUDA kernel launch failed: ", cudaGetErrorString(err));       \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

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

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

// ============================================================================
// ============================================================================
// OPTIMIZED TOP-K GATING - WARP-PER-TOKEN FOR COALESCED ACCESS
// ============================================================================

// Each warp handles one token completely.
// Threads in a warp load experts in parallel -> PERFECT COALESCING.
template <int K>
__global__ void topk_gating_kernel_warp_parallel(
    const float *__restrict__ gate_logits, int64_t *__restrict__ top_k_indices,
    float *__restrict__ top_k_weights, const int num_tokens,
    const int num_experts, const float temperature) {

  const int token_idx =
      blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
  const int lane = threadIdx.x % WARP_SIZE;

  if (token_idx >= num_tokens)
    return;

  const float *token_logits = gate_logits + (int64_t)token_idx * num_experts;

  // Each thread handles a subset of experts
  float local_best_vals[MAX_K];
  int local_best_idxs[MAX_K];
#pragma unroll
  for (int i = 0; i < K; i++) {
    local_best_vals[i] = -INFINITY;
    local_best_idxs[i] = -1;
  }

  // 1. Cooperative load + initial top-K
  for (int expert_id = lane; expert_id < num_experts; expert_id += WARP_SIZE) {
    float val = token_logits[expert_id] / temperature;

    // Local insertion check
    if (val > local_best_vals[K - 1]) {
      int pos = K - 1;
      while (pos > 0 && val > local_best_vals[pos - 1]) {
        local_best_vals[pos] = local_best_vals[pos - 1];
        local_best_idxs[pos] = local_best_idxs[pos - 1];
        pos--;
      }
      local_best_vals[pos] = val;
      local_best_idxs[pos] = expert_id;
    }
  }

  // 2. Warp-level merge of top-K
  // We use shared memory or shuffle. Shuffle is faster for K<=8.
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    // Merge best values from 'offset' steps away
#pragma unroll
    for (int i = 0; i < K; i++) {
      float other_val = __shfl_xor_sync(0xffffffff, local_best_vals[i], offset);
      int other_idx = __shfl_xor_sync(0xffffffff, local_best_idxs[i], offset);

      // Merge other_val into local_best
      if (other_val > local_best_vals[K - 1]) {
        // Skip if we already have this index (avoid duplicates)
        bool exists = false;
#pragma unroll
        for (int j = 0; j < K; j++)
          if (local_best_idxs[j] == other_idx)
            exists = true;

        if (!exists) {
          int pos = K - 1;
          while (pos > 0 && other_val > local_best_vals[pos - 1]) {
            local_best_vals[pos] = local_best_vals[pos - 1];
            local_best_idxs[pos] = local_best_idxs[pos - 1];
            pos--;
          }
          local_best_vals[pos] = other_val;
          local_best_idxs[pos] = other_idx;
        }
      }
    }
  }

  // 3. Softmax normalization (Warp-parallel)
  float max_logit = local_best_vals[0]; // Already synchronized across warp

  float sum_exp = 0.0f;
#pragma unroll
  for (int i = 0; i < K; i++) {
    local_best_vals[i] = expf(local_best_vals[i] - max_logit);
    sum_exp += local_best_vals[i];
  }

  // 4. Write output (thread 0 only per token)
  if (lane == 0) {
    const float inv_sum = 1.0f / (sum_exp + 1e-9f);
    int64_t *out_indices = top_k_indices + (int64_t)token_idx * K;
    float *out_weights = top_k_weights + (int64_t)token_idx * K;

#pragma unroll
    for (int i = 0; i < K; i++) {
      out_weights[i] = local_best_vals[i] * inv_sum;
      out_indices[i] = (int64_t)local_best_idxs[i];
    }
  }
}

// ============================================================================
// ULTRA-OPTIMIZED DISPATCH - SINGLE PASS WITH VECTORIZATION
// ============================================================================

// SINGLE PASS: Dispatch tokens with coalesced writes
template <int K, int VEC_SIZE = 4>
__global__ void dispatch_tokens_kernel_single_pass(
    const float *__restrict__ tokens, const int64_t *__restrict__ top_k_indices,
    int *__restrict__ expert_positions, float *__restrict__ expert_inputs,
    int64_t *__restrict__ token_map, const int num_tokens,
    const int num_experts, const int hidden_dim, const int capacity) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (token_idx >= num_tokens)
    return;

  const float *token_data = tokens + (int64_t)token_idx * hidden_dim;
  const int64_t *token_experts = top_k_indices + (int64_t)token_idx * K;

  // Shared memory for positions (minimal usage)
  __shared__ int shared_pos[MAX_K];

  // SINGLE PASS: Get positions atomically (only thread 0)
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < K; i++) {
      int expert_id = (int)token_experts[i];
      if (expert_id >= 0 && expert_id < num_experts) {
        shared_pos[i] = atomicAdd(&expert_positions[expert_id], 1);
      } else {
        shared_pos[i] = -1;
      }
    }
  }
  __syncthreads();

  // SINGLE PASS: Copy token to all K experts with vectorized writes
  const int vec_hidden = (hidden_dim / VEC_SIZE) * VEC_SIZE;
  const float4 *token_vec = reinterpret_cast<const float4 *>(token_data);

#pragma unroll
  for (int i = 0; i < K; i++) {
    int expert_id = (int)token_experts[i];
    int pos = shared_pos[i];

    if (pos < 0 || pos >= capacity || expert_id < 0 ||
        expert_id >= num_experts) {
      continue;
    }

    // Write token map (thread 0 only)
    if (tid == 0) {
      token_map[(int64_t)expert_id * capacity + pos] =
          (int64_t)token_idx * K + i;
    }

    // Vectorized copy (all threads)
    float *expert_input =
        expert_inputs + (int64_t)(expert_id * capacity + pos) * hidden_dim;
    float4 *expert_vec = reinterpret_cast<float4 *>(expert_input);

#pragma unroll 4
    for (int d = tid; d < vec_hidden / VEC_SIZE; d += blockDim.x) {
      expert_vec[d] = __ldg(&token_vec[d]);
    }

    // Handle remainder
    for (int d = vec_hidden + tid; d < hidden_dim; d += blockDim.x) {
      expert_input[d] = __ldg(&token_data[d]);
    }
  }
}

// ============================================================================
// ULTRA-OPTIMIZED COMBINE - SINGLE PASS WITH ATOMIC REDUCTION
// ============================================================================

// SINGLE PASS: Combine expert outputs with vectorized atomic adds
template <int VEC_SIZE = 4>
__global__ void combine_expert_outputs_kernel_single_pass(
    const float *__restrict__ expert_outputs,
    const int64_t *__restrict__ token_map,
    const float *__restrict__ top_k_weights,
    float *__restrict__ combined_output, const int num_experts,
    const int capacity, const int hidden_dim, const int num_tokens,
    const int k) {
  const int expert_id = blockIdx.x;
  const int pos = blockIdx.y;
  const int tid = threadIdx.x;

  if (expert_id >= num_experts || pos >= capacity)
    return;

  // Load token info
  const int64_t token_weight_idx =
      token_map[(int64_t)expert_id * capacity + pos];
  if (token_weight_idx < 0)
    return;

  const int token_idx = (int)(token_weight_idx / k);
  if (token_idx >= num_tokens)
    return;

  const float weight = __ldg(&top_k_weights[token_weight_idx]);

  // SINGLE PASS: Vectorized weighted addition with atomic adds
  const int64_t expert_offset =
      (int64_t)(expert_id * capacity + pos) * hidden_dim;
  const float *expert_out = expert_outputs + expert_offset;
  float *output = combined_output + (int64_t)token_idx * hidden_dim;

  const int vec_hidden = (hidden_dim / VEC_SIZE) * VEC_SIZE;
  const float4 *expert_vec = reinterpret_cast<const float4 *>(expert_out);

#pragma unroll 4
  for (int d = tid; d < vec_hidden / VEC_SIZE; d += blockDim.x) {
    float4 val = __ldg(&expert_vec[d]);

    // Scale by weight
    val.x *= weight;
    val.y *= weight;
    val.z *= weight;
    val.w *= weight;

    // Atomic add to output
    atomicAdd(&output[d * VEC_SIZE + 0], val.x);
    atomicAdd(&output[d * VEC_SIZE + 1], val.y);
    atomicAdd(&output[d * VEC_SIZE + 2], val.z);
    atomicAdd(&output[d * VEC_SIZE + 3], val.w);
  }

  // Handle remainder
  for (int d = vec_hidden + tid; d < hidden_dim; d += blockDim.x) {
    atomicAdd(&output[d], weight * __ldg(&expert_out[d]));
  }
}

// ============================================================================
// C++ INTERFACE
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
topk_gating_cuda(torch::Tensor gate_logits, int64_t k, double temperature) {
  CHECK_INPUT(gate_logits);
  TORCH_CHECK(gate_logits.dim() == 2, "gate_logits must be 2D");
  TORCH_CHECK(gate_logits.dtype() == torch::kFloat32,
              "gate_logits must be float32");
  TORCH_CHECK(k > 0 && k <= MAX_K, "k must be in [1, ", MAX_K, "]");

  const int64_t num_tokens = gate_logits.size(0);
  const int64_t num_experts = gate_logits.size(1);

  c10::cuda::CUDAGuard device_guard(gate_logits.device());

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(gate_logits.device());

  auto top_k_weights = torch::empty({num_tokens, k}, options);
  auto top_k_indices =
      torch::empty({num_tokens, k}, options.dtype(torch::kInt64));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Launch config: 8 warps (256 threads) per block
  const int threads = 256;
  const int blocks = (num_tokens * WARP_SIZE + threads - 1) / threads;

  // Dispatch based on K (template specialization for optimal performance)
  if (k == 2) {
    topk_gating_kernel_warp_parallel<2><<<blocks, threads, 0, stream>>>(
        gate_logits.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        top_k_weights.data_ptr<float>(), (int)num_tokens, (int)num_experts,
        (float)temperature);
  } else if (k == 4) {
    topk_gating_kernel_warp_parallel<4><<<blocks, threads, 0, stream>>>(
        gate_logits.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        top_k_weights.data_ptr<float>(), (int)num_tokens, (int)num_experts,
        (float)temperature);
  } else if (k == 8) {
    topk_gating_kernel_warp_parallel<8><<<blocks, threads, 0, stream>>>(
        gate_logits.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        top_k_weights.data_ptr<float>(), (int)num_tokens, (int)num_experts,
        (float)temperature);
  } else {
    // Fallback for arbitrary k
    topk_gating_kernel_warp_parallel<MAX_K><<<blocks, threads, 0, stream>>>(
        gate_logits.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        top_k_weights.data_ptr<float>(), (int)num_tokens, (int)num_experts,
        (float)temperature);
  }

  CUDA_CHECK_KERNEL();

  return std::make_tuple(top_k_indices, top_k_weights);
}

std::tuple<torch::Tensor, torch::Tensor>
dispatch_tokens_cuda(torch::Tensor tokens, torch::Tensor top_k_indices,
                     int64_t num_experts, int64_t capacity) {
  CHECK_INPUT(tokens);
  CHECK_INPUT(top_k_indices);

  TORCH_CHECK(tokens.dim() == 2, "tokens must be 2D");
  TORCH_CHECK(top_k_indices.dim() == 2, "top_k_indices must be 2D");
  TORCH_CHECK(tokens.dtype() == torch::kFloat32, "tokens must be float32");
  TORCH_CHECK(top_k_indices.dtype() == torch::kInt64,
              "top_k_indices must be int64");

  const int64_t num_tokens = tokens.size(0);
  const int64_t hidden_dim = tokens.size(1);
  const int64_t k = top_k_indices.size(1);

  c10::cuda::CUDAGuard device_guard(tokens.device());

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(tokens.device());

  auto expert_inputs =
      torch::zeros({num_experts, capacity, hidden_dim}, options);
  auto token_map =
      torch::full({num_experts, capacity}, -1, options.dtype(torch::kInt64));
  auto expert_positions =
      torch::zeros({num_experts}, options.dtype(torch::kInt32));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int threads = 256;
  const int blocks = (int)num_tokens;

  // Dispatch based on K for optimal performance
  if (k == 2) {
    dispatch_tokens_kernel_single_pass<2><<<blocks, threads, 0, stream>>>(
        tokens.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        expert_positions.data_ptr<int>(), expert_inputs.data_ptr<float>(),
        token_map.data_ptr<int64_t>(), (int)num_tokens, (int)num_experts,
        (int)hidden_dim, (int)capacity);
  } else if (k == 4) {
    dispatch_tokens_kernel_single_pass<4><<<blocks, threads, 0, stream>>>(
        tokens.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        expert_positions.data_ptr<int>(), expert_inputs.data_ptr<float>(),
        token_map.data_ptr<int64_t>(), (int)num_tokens, (int)num_experts,
        (int)hidden_dim, (int)capacity);
  } else {
    dispatch_tokens_kernel_single_pass<MAX_K><<<blocks, threads, 0, stream>>>(
        tokens.data_ptr<float>(), top_k_indices.data_ptr<int64_t>(),
        expert_positions.data_ptr<int>(), expert_inputs.data_ptr<float>(),
        token_map.data_ptr<int64_t>(), (int)num_tokens, (int)num_experts,
        (int)hidden_dim, (int)capacity);
  }

  CUDA_CHECK_KERNEL();

  return std::make_tuple(expert_inputs, token_map);
}

torch::Tensor combine_expert_outputs_cuda(torch::Tensor expert_outputs,
                                          torch::Tensor token_map,
                                          torch::Tensor top_k_weights,
                                          int64_t num_tokens, int64_t k) {
  CHECK_INPUT(expert_outputs);
  CHECK_INPUT(token_map);
  CHECK_INPUT(top_k_weights);

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

  torch::Tensor weights_flat =
      top_k_weights.dim() == 2 ? top_k_weights.flatten() : top_k_weights;

  combine_expert_outputs_kernel_single_pass<4><<<grid, threads, 0, stream>>>(
      expert_outputs.data_ptr<float>(), token_map.data_ptr<int64_t>(),
      weights_flat.data_ptr<float>(), combined.data_ptr<float>(),
      (int)num_experts, (int)capacity, (int)hidden_dim, (int)num_tokens,
      (int)k);

  CUDA_CHECK_KERNEL();

  return combined;
}

// Stub implementations
torch::Tensor compute_expert_capacity_cuda(torch::Tensor top_k_indices,
                                           int64_t num_experts) {
  return torch::zeros({num_experts},
                      top_k_indices.options().dtype(torch::kInt32));
}

torch::Tensor compute_load_balancing_loss_cuda(torch::Tensor gate_probs,
                                               torch::Tensor top_k_indices) {
  return torch::zeros({1}, gate_probs.options());
}

// ============================================================================
// PYBIND11 MODULE
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_gating", &topk_gating_cuda, "Top-K gating (CUDA) - Single Pass");
  m.def("dispatch_tokens", &dispatch_tokens_cuda,
        "Token dispatch (CUDA) - Single Pass");
  m.def("combine_expert_outputs", &combine_expert_outputs_cuda,
        "Combine outputs (CUDA) - Single Pass");
  m.def("compute_expert_capacity", &compute_expert_capacity_cuda,
        "Expert capacity (CUDA)");
  m.def("compute_load_balancing_loss", &compute_load_balancing_loss_cuda,
        "Load balance loss (CUDA)");
}