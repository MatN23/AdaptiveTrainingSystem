# Copyright (c) 2025 MatN23. All rights reserved.
// Copyright (c) 2025 MatN23. All rights reserved.
// FIXED VERSION - Corrected label_logit retrieval bug
//
// BUG FIXED: Label logit was being lost when multiple threads processed
// different parts of the vocabulary. Now uses atomic flag to find correct
// value.
//
// Compile with:
// nvcc -O3 -arch=sm_75 --use_fast_math --ptxas-options=-v \
//      --compiler-options '-fPIC' -shared fused_loss.cu -o fused_loss.so

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level reduction for online softmax state
struct SoftmaxState {
  float m; // running max
  float d; // running sum of exp
  float argmax_val;
  int argmax_idx;
};

__device__ __forceinline__ SoftmaxState
warp_reduce_softmax_state(SoftmaxState state) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_m = __shfl_xor_sync(FULL_MASK, state.m, offset);
    float other_d = __shfl_xor_sync(FULL_MASK, state.d, offset);
    float other_argmax_val =
        __shfl_xor_sync(FULL_MASK, state.argmax_val, offset);
    int other_argmax_idx = __shfl_xor_sync(FULL_MASK, state.argmax_idx, offset);

    // Merge two softmax states
    float m_new = fmaxf(state.m, other_m);
    state.d =
        state.d * __expf(state.m - m_new) + other_d * __expf(other_m - m_new);
    state.m = m_new;

    // Merge argmax (branchless)
    int other_greater = (other_argmax_val > state.argmax_val);
    state.argmax_val = other_greater ? other_argmax_val : state.argmax_val;
    state.argmax_idx = other_greater ? other_argmax_idx : state.argmax_idx;
  }
  return state;
}

// FIXED: Single-pass with corrected label_logit retrieval
__global__ void __launch_bounds__(256, 4) fused_cross_entropy_single_pass(
    const float *__restrict__ logits, const int64_t *__restrict__ labels,
    const float *__restrict__ loss_weights, const int64_t pad_token_id,
    float *__restrict__ loss_out, float *__restrict__ accuracy_out,
    int64_t *__restrict__ valid_tokens_out,
    float *__restrict__ total_weight_out, const int total_tokens,
    const int vocab_size) {
  const int token_idx = blockIdx.x;
  if (token_idx >= total_tokens)
    return;

  const int64_t label = labels[token_idx];

  const float *logit_row = logits + (int64_t)token_idx * vocab_size;
  const float weight =
      (loss_weights != nullptr) ? loss_weights[token_idx] : 1.0f;

  // Skip padding
  if (label == pad_token_id || label < 0 || label >= vocab_size) {
    return;
  }

  // =========================================================================
  // COALESCED VECTORIZED ACCESS
  // =========================================================================
  SoftmaxState state;
  state.m = -FLT_MAX;
  state.d = 0.0f;
  state.argmax_val = -FLT_MAX;
  state.argmax_idx = 0;

  float label_logit = 0.0f;
  bool found_label = false; //  FIX: Track if we found the label

  // Reinterpret row as float4 vectors for perfect coalescing
  const int num_vecs = vocab_size / 4;
  const float4 *logit_row_vec = reinterpret_cast<const float4 *>(logit_row);

  // Each thread processes multiple vectors using grid-stride
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 vals = __ldg(&logit_row_vec[i]);

    // Process each element in float4
    float elems[4] = {vals.x, vals.y, vals.z, vals.w};

#pragma unroll
    for (int j = 0; j < 4; j++) {
      float x = elems[j];
      int idx = i * 4 + j;

      float m_old = state.m;
      state.m = fmaxf(state.m, x);
      state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);

      if (x > state.argmax_val) {
        state.argmax_val = x;
        state.argmax_idx = idx;
      }

      //  FIX: Mark that we found the label
      if (idx == (int)label) {
        label_logit = x;
        found_label = true;
      }
    }
  }

  // Final scalar remainder
  for (int i = num_vecs * 4 + threadIdx.x; i < vocab_size; i += blockDim.x) {
    float x = __ldg(&logit_row[i]);

    float m_old = state.m;
    state.m = fmaxf(state.m, x);
    state.d = state.d * __expf(m_old - state.m) + __expf(x - state.m);

    if (x > state.argmax_val) {
      state.argmax_val = x;
      state.argmax_idx = i;
    }

    //  FIX: Mark that we found the label
    if (i == (int)label) {
      label_logit = x;
      found_label = true;
    }
  }

  // =========================================================================
  // Warp-level reduction
  // =========================================================================
  state = warp_reduce_softmax_state(state);

  __shared__ float smem_m[32];
  __shared__ float smem_d[32];
  __shared__ float smem_argmax_val[32];
  __shared__ int smem_argmax_idx[32];
  __shared__ float smem_label_logit_value; //  BULLETPROOF: Single value
  __shared__ int smem_label_found_flag;    //  BULLETPROOF: Atomic flag

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  // Initialize shared memory ONCE
  if (threadIdx.x == 0) {
    smem_label_logit_value = 0.0f;
    smem_label_found_flag = 0;
  }
  __syncthreads();

  // If this thread found the label, atomically write it
  if (found_label) {
    atomicExch(&smem_label_logit_value, label_logit);
    atomicExch(&smem_label_found_flag, 1);
  }

  if (lane == 0) {
    smem_m[wid] = state.m;
    smem_d[wid] = state.d;
    smem_argmax_val[wid] = state.argmax_val;
    smem_argmax_idx[wid] = state.argmax_idx;
  }
  __syncthreads();

  // =========================================================================
  // Block-level reduction
  // =========================================================================
  if (wid == 0) {
    if (lane < 8) { // Only 8 warps
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

    //  BULLETPROOF: Read atomic flag
    float final_label_logit;
    if (smem_label_found_flag == 1) {
      final_label_logit = smem_label_logit_value;
    } else {
      // Fallback: directly read from global memory
      final_label_logit = __ldg(&logit_row[label]);
    }

    // Compute loss using numerically stable formula
    const float loss =
        -((final_label_logit - max_logit) - __logf(sum_exp + 1e-10f));

    // Clamp loss to prevent overflow
    const float clamped_loss = fminf(loss, 100.0f);
    const float weighted_loss = weight * clamped_loss;
    const float accuracy = (pred_idx == (int)label) ? 1.0f : 0.0f;

    atomicAdd(loss_out, weighted_loss);
    atomicAdd(accuracy_out, accuracy);
    atomicAdd((unsigned long long *)valid_tokens_out, 1ULL);
    if (total_weight_out != nullptr) {
      atomicAdd(total_weight_out, weight);
    }
  }
}

extern "C" {

void fused_cross_entropy_accuracy_launcher(
    const float *logits, const int64_t *labels, const float *loss_weights,
    int64_t pad_token_id, float *loss_out, float *accuracy_out,
    int64_t *valid_tokens_out, float *total_weight_out, int total_tokens,
    int vocab_size, cudaStream_t stream) {
  cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
  cudaMemsetAsync(accuracy_out, 0, sizeof(float), stream);
  cudaMemsetAsync(valid_tokens_out, 0, sizeof(int64_t), stream);
  if (total_weight_out != nullptr) {
    cudaMemsetAsync(total_weight_out, 0, sizeof(float), stream);
  }

  // One block per token
  const int num_blocks = total_tokens;
  const int threads = 256;

  fused_cross_entropy_single_pass<<<num_blocks, threads, 0, stream>>>(
      logits, labels, loss_weights, pad_token_id, loss_out, accuracy_out,
      valid_tokens_out, total_weight_out, total_tokens, vocab_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
  }
}

} // extern "C"

// ============================================================================
// BACKWARD KERNEL: Computes gradient of cross-entropy loss w.r.t. logits
// grad_logits[i] = (softmax[i] - 1{i == label}) * grad_output * weight / N
// ============================================================================

__global__ void __launch_bounds__(256, 4) fused_cross_entropy_backward(
    const float *__restrict__ logits, const int64_t *__restrict__ labels,
    const float *__restrict__ loss_weights, const int64_t pad_token_id,
    const float grad_output,      // Scalar upstream gradient (typically 1.0)
    const float inv_valid_tokens, // 1.0 / valid_token_count for mean reduction
    float *__restrict__ grad_logits, // Output: gradient w.r.t. logits
    const int total_tokens, const int vocab_size) {

  const int token_idx = blockIdx.x;
  if (token_idx >= total_tokens)
    return;

  const int64_t label = labels[token_idx];
  const float weight =
      (loss_weights != nullptr) ? loss_weights[token_idx] : 1.0f;

  // Output row
  float *grad_row = grad_logits + (int64_t)token_idx * vocab_size;

  // Skip padding - zero gradient
  if (label == pad_token_id || label < 0 || label >= vocab_size) {
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      grad_row[i] = 0.0f;
    }
    return;
  }

  const float *logit_row = logits + (int64_t)token_idx * vocab_size;

  // ===========================================================================
  // Pass 1: Find max logit for numerical stability (parallel reduction)
  // ===========================================================================
  float local_max = -FLT_MAX;
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    local_max = fmaxf(local_max, __ldg(&logit_row[i]));
  }

// Warp reduction for max
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(FULL_MASK, local_max, offset));
  }

  __shared__ float smem_max[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

  if (lane == 0)
    smem_max[wid] = local_max;
  __syncthreads();

  if (wid == 0) {
    local_max = (lane < 8) ? smem_max[lane] : -FLT_MAX;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_max =
          fmaxf(local_max, __shfl_xor_sync(FULL_MASK, local_max, offset));
    }
    if (lane == 0)
      smem_max[0] = local_max;
  }
  __syncthreads();

  const float max_logit = smem_max[0];

  // ===========================================================================
  // Pass 2: Compute sum of exp(logit - max) for softmax denominator
  // ===========================================================================
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    local_sum += __expf(__ldg(&logit_row[i]) - max_logit);
  }

// Warp reduction for sum
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(FULL_MASK, local_sum, offset);
  }

  __shared__ float smem_sum[32];
  if (lane == 0)
    smem_sum[wid] = local_sum;
  __syncthreads();

  if (wid == 0) {
    local_sum = (lane < 8) ? smem_sum[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_sum += __shfl_xor_sync(FULL_MASK, local_sum, offset);
    }
    if (lane == 0)
      smem_sum[0] = local_sum;
  }
  __syncthreads();

  const float sum_exp = smem_sum[0];
  const float inv_sum = 1.0f / (sum_exp + 1e-10f);

  // ===========================================================================
  // Pass 3: Compute gradient: grad = (softmax - one_hot) * weight * grad_out /
  // N
  // ===========================================================================
  const float scale = grad_output * weight * inv_valid_tokens;

  // Vectorized writes for perfect coalescing
  const int num_vecs = vocab_size / 4;
  float4 *grad_row_vec = reinterpret_cast<float4 *>(grad_row);

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    float4 logits_v = __ldg(reinterpret_cast<const float4 *>(logit_row) + i);
    float4 grad_v;

    int base_idx = i * 4;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float logit_val = (&logits_v.x)[j];
      float softmax_val = __expf(logit_val - max_logit) * inv_sum;
      float one_hot = ((base_idx + j) == (int)label) ? 1.0f : 0.0f;
      (&grad_v.x)[j] = (softmax_val - one_hot) * scale;
    }

    grad_row_vec[i] = grad_v;
  }

  // Handle remainder
  for (int i = num_vecs * 4 + threadIdx.x; i < vocab_size; i += blockDim.x) {
    float logit_val = __ldg(&logit_row[i]);
    float softmax_val = __expf(logit_val - max_logit) * inv_sum;
    float one_hot = (i == (int)label) ? 1.0f : 0.0f;
    grad_row[i] = (softmax_val - one_hot) * scale;
  }
}

extern "C" {

void fused_cross_entropy_backward_launcher(
    const float *logits, const int64_t *labels, const float *loss_weights,
    int64_t pad_token_id, float grad_output, float inv_valid_tokens,
    float *grad_logits, int total_tokens, int vocab_size, cudaStream_t stream) {

  const int num_blocks = total_tokens;
  const int threads = 256;

  fused_cross_entropy_backward<<<num_blocks, threads, 0, stream>>>(
      logits, labels, loss_weights, pad_token_id, grad_output, inv_valid_tokens,
      grad_logits, total_tokens, vocab_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA backward kernel error: %s\n",
            cudaGetErrorString(err));
  }
}

} // extern "C"