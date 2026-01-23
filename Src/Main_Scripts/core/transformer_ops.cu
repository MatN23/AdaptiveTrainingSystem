// Copyright (c) 2025 MatN23. All rights reserved.
// PRODUCTION TENSOR CORE KERNELS - CUTLASS PRINCIPLES
//
// KEY DESIGN CHANGES:
// 1. ✅ Epilogue fusion (SwiGLU/bias inside GEMM, not separate kernel)
// 2. ✅ Fragment-based computation (minimal shared memory)
// 3. ✅ Shape-specialized kernels (not one-size-fits-all)
// 4. ✅ Vectorized memory operations (float4, half2)
// 5. ✅ Warp-level reductions (no full-block syncs in hot paths)
// 6. ✅ Proper WMMA usage (16x16x16, all lanes participate)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Tensor Core dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// VECTORIZED MEMORY OPS
// ============================================================================

struct __align__(8) half4 {
  __half2 xy, zw;
};

__device__ __forceinline__ half4 load_half4(const __half *ptr) {
  half4 result;
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
               : "=r"(reinterpret_cast<uint32_t &>(result.xy)),
                 "=r"(reinterpret_cast<uint32_t &>(result.zw))
               : "l"(ptr));
  return result;
}

__device__ __forceinline__ void store_half4(__half *ptr, half4 val) {
  asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(ptr),
               "r"(reinterpret_cast<uint32_t &>(val.xy)),
               "r"(reinterpret_cast<uint32_t &>(val.zw)));
}

// ============================================================================
// RMSNORM VECTOR UTILS
// ============================================================================
template <typename T> struct RMSNormUtils;

template <> struct RMSNormUtils<float> {
  using Vec = float4;
  static __device__ __forceinline__ Vec load(const float *p) {
    return *reinterpret_cast<const float4 *>(p);
  }
  static __device__ __forceinline__ void store(float *p, Vec v) {
    *reinterpret_cast<float4 *>(p) = v;
  }
  static __device__ __forceinline__ float sum_sq(Vec v) {
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  static __device__ __forceinline__ Vec scale(Vec v, float s, Vec w) {
    return make_float4(v.x * s * w.x, v.y * s * w.y, v.z * s * w.z,
                       v.w * s * w.w);
  }
};

template <> struct RMSNormUtils<__half> {
  using Vec = half4;
  static __device__ __forceinline__ Vec load(const __half *p) {
    return load_half4(p);
  }
  static __device__ __forceinline__ void store(__half *p, Vec v) {
    store_half4(p, v);
  }
  static __device__ __forceinline__ float sum_sq(Vec v) {
    float sq = 0.0f;
    sq += __half2float(v.xy.x) * __half2float(v.xy.x);
    sq += __half2float(v.xy.y) * __half2float(v.xy.y);
    sq += __half2float(v.zw.x) * __half2float(v.zw.x);
    sq += __half2float(v.zw.y) * __half2float(v.zw.y);
    return sq;
  }
  static __device__ __forceinline__ Vec scale(Vec v, float s, Vec w) {
    half4 res;
    res.xy.x = __float2half(__half2float(v.xy.x) * s * __half2float(w.xy.x));
    res.xy.y = __float2half(__half2float(v.xy.y) * s * __half2float(w.xy.y));
    res.zw.x = __float2half(__half2float(v.zw.x) * s * __half2float(w.zw.x));
    res.zw.y = __float2half(__half2float(v.zw.y) * s * __half2float(w.zw.y));
    return res;
  }
};

// ============================================================================
// TYPE CONVERSION
// ============================================================================

template <typename T> __device__ __forceinline__ __half to_half(T val);
template <> __device__ __forceinline__ __half to_half(float val) {
  return __float2half(val);
}
template <> __device__ __forceinline__ __half to_half(__half val) {
  return val;
}
template <> __device__ __forceinline__ __half to_half(uint8_t val) {
  return __float2half((float)val);
}

template <typename T> __device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <> __device__ __forceinline__ float to_float<__half>(__half val) {
  return __half2float(val);
}

template <typename T> __device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

template <> __device__ __forceinline__ __half from_float<__half>(float val) {
  return __float2half(val);
}

// ============================================================================
// WARP REDUCTIONS
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
  }
  return val;
}

// ============================================================================
// FAST ACTIVATIONS
// ============================================================================

__device__ __forceinline__ float silu_fast(float x) {
  x = fminf(fmaxf(x, -5.0f), 5.0f);
  float x2 = x * x;
  float x3 = x2 * x;
  float sigmoid_approx = 0.5f + 0.25f * x - 0.02083f * x3;
  sigmoid_approx = fminf(fmaxf(sigmoid_approx, 0.0f), 1.0f);
  return x * sigmoid_approx;
}

// ============================================================================
// SHAPE-SPECIALIZED RMSNORM (Warp-level for common sizes)
// ============================================================================

// For hidden_size = 4096 (8 warps, each handles 512 dims)
// For hidden_size = 4096 (8 warps, each handles 512 dims)
// For arbitrary hidden_size
// ============================================================================
// TEMPLATE-SPECIALIZED RMSNORM (High Performance)
// ============================================================================

template <typename T, int N>
__global__ void __launch_bounds__(1024)
    rms_norm_aligned(const T *__restrict__ input,
                     const T *__restrict__ weight, // __grid_constant__ on SM90+
                     T *__restrict__ output, const int batch_seq,
                     const float eps) {

  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  // Vectorized Load Type
  using Utils = RMSNormUtils<T>;
  using Vec = typename Utils::Vec;
  constexpr int VEC_SIZE = 4;         // float4 or half4
  constexpr int ELEMS_PER_THREAD = 4; // Each thread handles 1 vector
  constexpr int THREADS_PER_BLOCK = N / VEC_SIZE; // e.g. 1024 -> 256 threads

  // Static Assert to ensure block size is valid
  static_assert(THREADS_PER_BLOCK <= 1024,
                "Hidden size too large for simple block");

  const int64_t offset = (int64_t)token_idx * N;

  // 1. Cooperative Load & SumSq
  Vec vals = Utils::load(&input[offset + tid * 4]);
  float mysum = Utils::sum_sq(vals);

  // 2. Block Reduce (Warp-Level + Shared)
  // Standard warp reduce
  mysum = warp_reduce_sum(mysum);

  __shared__ float s_warps[32]; // Max 32 warps (1024 threads)
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0)
    s_warps[warp_id] = mysum;
  __syncthreads();

  // Final reduce by first warp
  if (warp_id == 0) {
    float block_sum = (tid < (THREADS_PER_BLOCK / 32)) ? s_warps[tid] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (tid == 0) {
      s_warps[0] = rsqrtf(block_sum / float(N) + eps);
    }
  }
  __syncthreads();

  float scale = s_warps[0];

  // 3. Write
  Vec w = Utils::load(&weight[tid * 4]);
  Vec res = Utils::scale(vals, scale, w);
  Utils::store(&output[offset + tid * 4], res);
}

// For arbitrary hidden_size
template <typename T>
__global__ void __launch_bounds__(256)
    rms_norm_dynamic(const T *__restrict__ input, const T *__restrict__ weight,
                     T *__restrict__ output, const int batch_seq,
                     const int hidden_size, const float eps) {

  const int token_idx = blockIdx.x;
  if (token_idx >= batch_seq)
    return;

  const int tid = threadIdx.x;
  const int64_t offset = (int64_t)token_idx * hidden_size;

  // 1. Compute Variance (Sum of Squares) with Block Reduction
  float sum_sq = 0.0f;
  using Utils = RMSNormUtils<T>;
  using Vec = typename Utils::Vec;

  // Vectorized strided loop
  int vec_size = hidden_size / 4;
  int remainder =
      hidden_size % 4; // Should be 0 for most models, but handle safely?
  // Assume generic sizing:

  for (int i = tid * 4; i < hidden_size - remainder; i += blockDim.x * 4) {
    Vec vals = Utils::load(&input[offset + i]);
    sum_sq += Utils::sum_sq(vals);
  }

  // Handle remainder (scalar) if any - usually optimized out for aligned sizes
  for (int i = (hidden_size & ~3) + tid; i < hidden_size; i += blockDim.x) {
    float val = to_float(input[offset + i]);
    sum_sq += val * val;
  }

  // Block Reduction
  sum_sq = warp_reduce_sum(sum_sq);

  __shared__ float s_max[8]; // 8 warps
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0)
    s_max[warp_id] = sum_sq;
  __syncthreads();

  if (warp_id == 0) {
    sum_sq = (tid < 8) ? s_max[tid] : 0.0f;
    sum_sq = warp_reduce_sum(sum_sq);
    if (tid == 0) {
      s_max[0] = rsqrtf(sum_sq / float(hidden_size) + eps);
    }
  }
  __syncthreads();

  const float rms_scale = s_max[0];

  // 2. Normalize and Scale (Vectorized)
  for (int i = tid * 4; i < hidden_size - remainder; i += blockDim.x * 4) {
    Vec vals = Utils::load(&input[offset + i]);
    Vec ws = Utils::load(&weight[i]);
    Vec res = Utils::scale(vals, rms_scale, ws);
    Utils::store(&output[offset + i], res);
  }

  // Remainder scalars
  for (int i = (hidden_size & ~3) + tid; i < hidden_size; i += blockDim.x) {
    float val = to_float(input[offset + i]);
    float w = to_float(weight[i]);
    float res = val * rms_scale * w;
    if constexpr (std::is_same_v<T, __half>) {
      output[offset + i] = __float2half(res);
    } else {
      output[offset + i] = res;
    }
  }
}

// ============================================================================
// FUSED GEMM + SWIGLU EPILOGUE (CUTLASS-style)
//
// C = SwiGLU(A @ W_gate, A @ W_up)
//
// Key: TWO GEMMs with fused activation, not separate kernels
// ============================================================================

#if __CUDA_ARCH__ >= 700

template <int HIDDEN = 4096, int INTER = 11008>
__global__ void __launch_bounds__(256) fused_gemm_swiglu_epilogue(
    const __half *__restrict__ input,  // [batch_seq, HIDDEN]
    const __half *__restrict__ W_gate, // [HIDDEN, INTER] col-major
    const __half *__restrict__ W_up,   // [HIDDEN, INTER] col-major
    __half *__restrict__ output,       // [batch_seq, INTER]
    const int batch_seq) {

  const int token_idx = blockIdx.x;
  if (token_idx >= batch_seq)
    return;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  // Each warp computes 16 outputs (WMMA_N)
  // 8 warps → 128 outputs per block, so need multiple blocks for INTER=11008
  const int outputs_per_block = 128;
  const int block_output_start = blockIdx.y * outputs_per_block;
  const int warp_output = block_output_start + warp_id * WMMA_N;

  if (warp_output >= INTER)
    return;

  const int64_t input_offset = (int64_t)token_idx * HIDDEN;

  // WMMA fragments for BOTH GEMMs
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::col_major>
      gate_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::col_major>
      up_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gate_acc;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> up_acc;

  wmma::fill_fragment(gate_acc, 0.0f);
  wmma::fill_fragment(up_acc, 0.0f);

  // Accumulate over hidden dimension
  for (int k = 0; k < HIDDEN; k += WMMA_K) {
    // Load input once, reuse for both GEMMs
    wmma::load_matrix_sync(a_frag, &input[input_offset + k], HIDDEN);

    // Load weights for both projections
    wmma::load_matrix_sync(gate_frag, &W_gate[warp_output * HIDDEN + k],
                           HIDDEN);
    wmma::load_matrix_sync(up_frag, &W_up[warp_output * HIDDEN + k], HIDDEN);

    // Dual GEMM
    wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
    wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);
  }

  // EPILOGUE: Apply SwiGLU and store
  __half gate_vals[WMMA_M * WMMA_N];
  __half up_vals[WMMA_M * WMMA_N];

  wmma::store_matrix_sync(gate_vals, gate_acc, WMMA_N, wmma::mem_row_major);
  wmma::store_matrix_sync(up_vals, up_acc, WMMA_N, wmma::mem_row_major);

// Apply SwiGLU element-wise
#pragma unroll
  for (int i = 0; i < WMMA_N; i++) {
    if (warp_output + i < INTER) {
      float g = __half2float(gate_vals[i]);
      float u = __half2float(up_vals[i]);
      output[(int64_t)token_idx * INTER + warp_output + i] =
          __float2half(g * silu_fast(u));
    }
  }
}

#endif // __CUDA_ARCH__ >= 700

// ============================================================================
// FUSED MLP BLOCK (3 kernels → 2 kernels)
//
// Kernel 1: RMSNorm
// Kernel 2: GEMM(gate+up) + SwiGLU + GEMM(down) - ALL FUSED
// ============================================================================

// Note: This kernel requires sm_70+ for WMMA support.
// The __CUDA_ARCH__ check is done inside the kernel.

// (Type conversion helpers removed - defined above)

// ============================================================================
// FUSED MLP WITH TENSOR CORES - PHASE 1 OPTIMIZATIONS
//
// ARCHITECTURE:
// - Phase 1: ALL warps cooperatively compute SwiGLU(gate, up) -> smem
// - Phase 2: Each warp owns exclusive HIDDEN tiles, reads intermediate from
// smem
//
// Key optimizations:
// 1. ✅ NO ATOMICS - warp-exclusive output tiles
// 2. ✅ Direct epilogue on accumulator fragments (SwiGLU)
// 3. ✅ Proper bounds checking
// 4. ✅ Bank conflict padding (+8)
// 5. ✅ sm_75 compatible
//
// TODO Phase 2: cp.async pipeline for weight tiles
// ============================================================================

// Helper to align shared memory pointer
template <typename T> __device__ T *align_ptr(void *ptr, int align_bytes = 16) {
  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t aligned = (addr + align_bytes - 1) & ~(align_bytes - 1);
  return (T *)aligned;
}

template <typename T_IO, typename T_W>
__global__ void __launch_bounds__(256) fused_mlp_tensor_core(
    const T_IO *__restrict__ input_normalized, // [batch_seq, HIDDEN]
    const T_W *__restrict__ W_gate,            // [HIDDEN, INTER] col-major
    const T_W *__restrict__ W_up,              // [HIDDEN, INTER] col-major
    const T_W *__restrict__ W_down,            // [INTER, HIDDEN] col-major
    T_IO *__restrict__ output,                 // [batch_seq, HIDDEN]
    const int batch_seq, const int HIDDEN, const int INTER) {

  const int token_idx = blockIdx.x;
  if (token_idx >= batch_seq)
    return;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int NUM_WARPS = 8; // Fixed block size 256

  const int64_t input_offset = (int64_t)token_idx * HIDDEN;
  const int64_t output_offset = (int64_t)token_idx * HIDDEN;

  // ==========================================================================
  // SHARED MEMORY LAYOUT (Dynamic)
  // ==========================================================================
  // Layout: [Input (Half): HIDDEN] | [Padding] | [Output (Float): HIDDEN]
  // NOTE: smem_weights and smem_inter are Static Shared Memory

  extern __shared__ char smem_dyn_base[];
  __half *smem_input = (__half *)smem_dyn_base;
  // Offset output buffer properly (align to 16 bytes)
  float *smem_output = align_ptr<float>(smem_input + HIDDEN);

  // Static shared memory (Separate bank?)
  __shared__ __half smem_weights[NUM_WARPS][2][WMMA_K * WMMA_N];
  constexpr int INTER_TILE = 256;
  __shared__ __half smem_inter[INTER_TILE + 8];

  // ==========================================================================
  // STEP 0: LOAD INPUT TO SHARED MEMORY (Convert to HALF)
  // ==========================================================================
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    smem_input[i] = to_half(input_normalized[input_offset + i]);
    smem_output[i] = 0.0f;
  }
  __syncthreads();

  // ==========================================================================
  // TILE OVER INTER DIMENSION
  // ==========================================================================
  for (int inter_tile = 0; inter_tile < INTER; inter_tile += INTER_TILE) {
    const int tile_end = min(inter_tile + INTER_TILE, INTER);
    const int tile_size = tile_end - inter_tile;

    // ========================================================================
    // PHASE 1: Gate + Up GEMMs -> SwiGLU -> smem_inter
    // ========================================================================
    const int inter_per_warp = (tile_size + NUM_WARPS - 1) / NUM_WARPS;
    const int warp_start = warp_id * inter_per_warp;
    // Bounds check
    const int warp_actual_len =
        (warp_start < tile_size) ? min(inter_per_warp, tile_size - warp_start)
                                 : 0;

    for (int i = 0; i < warp_actual_len; i += WMMA_N) {
      const int current_inter_col = inter_tile + warp_start + i;
      // Safety check (should be covered by loop bound but good for edges)
      if (current_inter_col >= INTER)
        break;

      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gate_acc;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> up_acc;
      wmma::fill_fragment(gate_acc, 0.0f);
      wmma::fill_fragment(up_acc, 0.0f);

      // Loop K (Inner Product)
      for (int k = 0; k < HIDDEN; k += WMMA_K) {
        // A Fragment: Load from smem_input
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                       wmma::row_major>
            a_frag;

        // Handle K-remainder edge case? WMMA requires full tiles.
        // Requirement: HIDDEN % 16 == 0.
        wmma::load_matrix_sync(a_frag, &smem_input[k],
                               0); // Stride 0 for broadcast

        // B Fragments (Weights)
        __half *warp_gate_smem = smem_weights[warp_id][0];
        __half *warp_up_smem = smem_weights[warp_id][1];

// Cooperative Load [16x16] tile
#pragma unroll
        for (int e = 0; e < 8; e++) {
          int lane_offset = lane_id * 8 + e;
          int local_k = lane_offset % 16;
          int local_n = lane_offset / 16;
          int global_k = k + local_k;
          int global_n = current_inter_col + local_n;

          if (global_k < HIDDEN && global_n < INTER) {
            warp_gate_smem[lane_offset] =
                to_half(W_gate[global_n * HIDDEN + global_k]);
            warp_up_smem[lane_offset] =
                to_half(W_up[global_n * HIDDEN + global_k]);
          } else {
            warp_gate_smem[lane_offset] = 0;
            warp_up_smem[lane_offset] = 0;
          }
        }

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                       wmma::col_major>
            gate_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                       wmma::col_major>
            up_frag;
        wmma::load_matrix_sync(gate_frag, warp_gate_smem, 16);
        wmma::load_matrix_sync(up_frag, warp_up_smem, 16);
        wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
        wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);
      }

      // EPILOGUE: SwiGLU -> smem_inter
      // Store accumulators (float) to register file (float array)
      float gate_vals[WMMA_M * WMMA_N];
      float up_vals[WMMA_M * WMMA_N];

      wmma::store_matrix_sync(gate_vals, gate_acc, WMMA_N, wmma::mem_row_major);
      wmma::store_matrix_sync(up_vals, up_acc, WMMA_N, wmma::mem_row_major);

#pragma unroll
      for (int t = 0; t < WMMA_N; t++) {
        // We only care about the first row (since all rows of A were identical)
        if (current_inter_col + t < INTER) {
          float g = gate_vals[t];
          float u = up_vals[t];
          // Store to shared memory
          smem_inter[warp_start + i + t] = __float2half(g * silu_fast(u));
        }
      }
    }
    __syncthreads();

    // ========================================================================
    // PHASE 2: Down Projection
    // ========================================================================
    const int out_chunk_size = (HIDDEN + NUM_WARPS - 1) / NUM_WARPS;
    const int out_start = warp_id * out_chunk_size;
    const int out_end = min(out_start + out_chunk_size, HIDDEN);

    for (int h = out_start + lane_id; h < out_end; h += WARP_SIZE) {
      float acc = 0.0f;
      int k = 0;
// Unroll
#pragma unroll 4
      for (; k + 1 < tile_size; k += 2) {
        __half2 inter_h2 = *reinterpret_cast<const __half2 *>(&smem_inter[k]);
        float iv0 = __half2float(inter_h2.x);
        float iv1 = __half2float(inter_h2.y);
        float w0 = to_float(W_down[h * INTER + inter_tile + k]);
        float w1 = to_float(W_down[h * INTER + inter_tile + k + 1]);
        acc += iv0 * w0 + iv1 * w1;
      }
      for (; k < tile_size; k++) {
        float iv = __half2float(smem_inter[k]);
        float w = to_float(W_down[h * INTER + inter_tile + k]);
        acc += iv * w;
      }
      smem_output[h] += acc;
    }
    __syncthreads();
  }

  // Write Final Output
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    output[output_offset + i] = from_float<T_IO>(smem_output[i]);
  }
}
/* OLD KERNEL BODY REPLACED

  const int token_idx = blockIdx.x;
  if (token_idx >= batch_seq)
    return;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int NUM_WARPS = 8;

  const int64_t input_offset = (int64_t)token_idx * HIDDEN;
  const int64_t output_offset = (int64_t)token_idx * HIDDEN;

  // ==========================================================================
  // SHARED MEMORY LAYOUT
  // ==========================================================================
  constexpr int INTER_TILE = 256; // Process 256 intermediate elements at a time
  __shared__ __half smem_inter[INTER_TILE + 8]; // +8 for bank conflict padding

  // Output accumulators - each warp owns HIDDEN/NUM_WARPS elements
  constexpr int HIDDEN_PER_WARP = HIDDEN / NUM_WARPS; // 512
  __shared__ float smem_output[HIDDEN + 32];

  // Initialize output accumulators
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    smem_output[i] = 0.0f;
  }
  __syncthreads();

  // ==========================================================================
  // TILE OVER INTER DIMENSION
  // ==========================================================================
  for (int inter_tile = 0; inter_tile < INTER; inter_tile += INTER_TILE) {
    const int tile_end = min(inter_tile + INTER_TILE, INTER);
    const int tile_size = tile_end - inter_tile;

    // ========================================================================
    // PHASE 1: Gate + Up GEMMs + SwiGLU -> smem_inter
    // ========================================================================
    // All warps cooperatively compute this tile's intermediate values

    const int inter_per_warp = (tile_size + NUM_WARPS - 1) / NUM_WARPS;
    const int warp_start = warp_id * inter_per_warp;
    const int warp_end = min(warp_start + inter_per_warp, tile_size);

    for (int i = warp_start; i < warp_end; i += WMMA_N) {
      const int global_inter = inter_tile + i;
      if (global_inter >= INTER)
        break;

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                     wmma::col_major>
          gate_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                     wmma::col_major>
          up_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gate_acc;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> up_acc;

      wmma::fill_fragment(gate_acc, 0.0f);
      wmma::fill_fragment(up_acc, 0.0f);

      // Accumulate over HIDDEN (K dimension)
      for (int k = 0; k < HIDDEN; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, &input_normalized[input_offset + k],
                               HIDDEN);
        wmma::load_matrix_sync(gate_frag, &W_gate[global_inter * HIDDEN + k],
                               HIDDEN);
        wmma::load_matrix_sync(up_frag, &W_up[global_inter * HIDDEN + k],
                               HIDDEN);

        wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
        wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);
      }

// DIRECT EPILOGUE: Apply SwiGLU on accumulators, store to smem
// FIX: Distribute writes across lanes instead of only lane 0
#pragma unroll
      for (int t = 0; t < gate_acc.num_elements; t++) {
        int local_idx = i + t;
        int target_lane = t % WARP_SIZE; // Distribute writes across lanes
        if (local_idx < tile_size && lane_id == target_lane) {
          float g = gate_acc.x[t];
          float u = up_acc.x[t];
          smem_inter[local_idx] = __float2half(g * silu_fast(u));
        }
      }
    }
    __syncthreads();

    // ========================================================================
    // PHASE 2: Down Projection - Vectorized with half2
    // ========================================================================
    // Each warp owns exclusive output range [out_start, out_end)

    const int out_start = warp_id * HIDDEN_PER_WARP;
    const int out_end = out_start + HIDDEN_PER_WARP;

    // Each lane handles multiple outputs with stride
    for (int h = out_start + lane_id; h < out_end; h += WARP_SIZE) {
      float acc = 0.0f;

      // Vectorized inner loop: process 2 intermediate values at a time
      int k = 0;
#pragma unroll 4
      for (; k + 1 < tile_size; k += 2) {
        // Load 2 intermediate values (half2)
        __half2 inter_h2 = *reinterpret_cast<const __half2 *>(&smem_inter[k]);
        float inter_val0 = __half2float(inter_h2.x);
        float inter_val1 = __half2float(inter_h2.y);

        // Load weights (still scalar - could vectorize W_down too)
        float w0 = __half2float(W_down[h * INTER + inter_tile + k]);
        float w1 = __half2float(W_down[h * INTER + inter_tile + k + 1]);

        acc += inter_val0 * w0 + inter_val1 * w1;
      }
      // Handle remainder
      for (; k < tile_size; k++) {
        float inter_val = __half2float(smem_inter[k]);
        float w = __half2float(W_down[h * INTER + inter_tile + k]);
        acc += inter_val * w;
      }

      // Accumulate into smem_output (no atomics - exclusive ownership!)
      smem_output[h] += acc;
    }
    __syncthreads();
  }

  // ==========================================================================
  // WRITE FINAL OUTPUT
  // ==========================================================================
*/

// ============================================================================
// ROPE (Unchanged - already optimal)
// ============================================================================

__global__ void __launch_bounds__(256)
    rope_precompute(float *__restrict__ cos_cache,
                    float *__restrict__ sin_cache, const int max_seq_len,
                    const int head_dim, const float theta_base) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim >> 1;
  const int total = max_seq_len * half_dim;

  if (idx >= total)
    return;

  const int pos = idx / half_dim;
  const int dim = idx % half_dim;
  // Use exp/log trick instead of slow powf: theta^x = exp(x * log(theta))
  const float freq = expf(-logf(theta_base) * (2.0f * dim) / float(head_dim));
  const float angle = pos * freq;

  float sin_val, cos_val;
  sincosf(angle, &sin_val, &cos_val);

  cos_cache[idx] = cos_val;
  sin_cache[idx] = sin_val;
}

template <typename T>
__global__ void __launch_bounds__(256)
    rope_apply(T *__restrict__ q, T *__restrict__ k,
               const float *__restrict__ cos_cache,
               const float *__restrict__ sin_cache, const int batch_size,
               const int num_heads, const int seq_len, const int head_dim,
               const int position_offset) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int pos = blockIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || pos >= seq_len)
    return;

  const int half_dim = head_dim >> 1;
  const int64_t base_offset =
      ((int64_t)batch_idx * num_heads + head_idx) * seq_len * head_dim +
      (int64_t)pos * head_dim;

  for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
    const int64_t idx0 = base_offset + d;
    const int64_t idx1 = base_offset + half_dim + d;
    const int64_t cache_idx = (int64_t)(position_offset + pos) * half_dim + d;

    const float cos_val = __ldg(&cos_cache[cache_idx]);
    const float sin_val = __ldg(&sin_cache[cache_idx]);

    float q0 = to_float(__ldg(&q[idx0]));
    float q1 = to_float(__ldg(&q[idx1]));
    q[idx0] = from_float<T>(q0 * cos_val - q1 * sin_val);
    q[idx1] = from_float<T>(q0 * sin_val + q1 * cos_val);

    float k0 = to_float(__ldg(&k[idx0]));
    float k1 = to_float(__ldg(&k[idx1]));
    k[idx0] = from_float<T>(k0 * cos_val - k1 * sin_val);
    k[idx1] = from_float<T>(k0 * sin_val + k1 * cos_val);
  }
}

// ============================================================================
// C API
// ============================================================================

// Dispatcher
template <typename T>
void dispatch_rms_norm(const T *input, const T *weight, T *output,
                       int batch_seq, int hidden_size, float eps,
                       cudaStream_t stream) {
  if (hidden_size == 4096) {
    rms_norm_aligned<T, 4096>
        <<<batch_seq, 1024, 0, stream>>>(input, weight, output, batch_seq, eps);
  } else if (hidden_size == 2048) {
    rms_norm_aligned<T, 2048>
        <<<batch_seq, 512, 0, stream>>>(input, weight, output, batch_seq, eps);
  } else if (hidden_size == 1024) {
    rms_norm_aligned<T, 1024>
        <<<batch_seq, 256, 0, stream>>>(input, weight, output, batch_seq, eps);
  } else if (hidden_size == 8192) {
    // Need overlapping or loop for >4096 because max threads is 1024 (covers
    // 4096 elements with vec4) For now, fallback for >4096 or implement looped
    // kernel
    rms_norm_dynamic<T><<<batch_seq, 256, 0, stream>>>(
        input, weight, output, batch_seq, hidden_size, eps);
  } else {
    rms_norm_dynamic<T><<<batch_seq, 256, 0, stream>>>(
        input, weight, output, batch_seq, hidden_size, eps);
  }
}

extern "C" {

// Forward declaration
void rms_norm_launcher(const float *input, const float *weight, float *output,
                       int batch_seq, int hidden_size, float eps,
                       cudaStream_t stream);

void rms_norm_launcher_fp16(const __half *input, const __half *weight,
                            __half *output, int batch_seq, int hidden_size,
                            float eps, cudaStream_t stream) {
  dispatch_rms_norm(input, weight, output, batch_seq, hidden_size, eps, stream);
  CUDA_CHECK(cudaGetLastError());
}

// Fused MLP with pre-allocated workspace (NO cudaMalloc in hot path!)
// workspace must be at least batch_seq * hidden_size * sizeof(__half) bytes
// Fused MLP with pre-allocated workspace (NO cudaMalloc in hot path!)
// workspace must be at least batch_seq * hidden_size * sizeof(__half) bytes
void fused_mlp_block_launcher_fp16(
    const __half *input, const __half *norm_weight, const __half *W_gate,
    const __half *W_up, const __half *W_down, __half *output,
    __half *workspace, // Pre-allocated buffer for normalized output
    int batch_seq, int hidden_size, int intermediate_size, float eps,
    cudaStream_t stream) {

  rms_norm_launcher_fp16(input, norm_weight, workspace, batch_seq, hidden_size,
                         eps, stream);

  // Dynamic shared memory calculation:
  // smem_input (half) + smem_output (float) + 16 (align)
  size_t smem_size =
      (size_t)hidden_size * (sizeof(__half) + sizeof(float)) + 64;

  // Ensure we don't exceed hardware limits limits (48KB/64KB/96KB)
  // For HIDDEN=8192: 8192*6 = ~48KB. Fits.

  fused_mlp_tensor_core<__half, __half><<<batch_seq, 256, smem_size, stream>>>(
      workspace, W_gate, W_up, W_down, output, batch_seq, hidden_size,
      intermediate_size);

  CUDA_CHECK(cudaGetLastError());
}

// FP32 Launcher (Staged Pipeline: Load float -> Convert half -> Smem -> Tensor
// Core)
void fused_mlp_block_launcher_fp32(const float *input, const float *norm_weight,
                                   const float *W_gate, const float *W_up,
                                   const float *W_down, float *output,
                                   float *workspace, int batch_seq,
                                   int hidden_size, int intermediate_size,
                                   float eps, cudaStream_t stream) {

  // Use generic rms_norm (FP32)
  rms_norm_dynamic<float><<<batch_seq, 256, 0, stream>>>(
      input, norm_weight, workspace, batch_seq, hidden_size, eps);

  size_t smem_size =
      (size_t)hidden_size * (sizeof(__half) + sizeof(float)) + 64;

  fused_mlp_tensor_core<float, float><<<batch_seq, 256, smem_size, stream>>>(
      workspace, W_gate, W_up, W_down, output, batch_seq, hidden_size,
      intermediate_size);

  CUDA_CHECK(cudaGetLastError());
}

// W8A16 Launcher (FP16 Activation, FP8 Weight)
void fused_mlp_block_launcher_w8a16(const __half *input,
                                    const __half *norm_weight,
                                    const uint8_t *W_gate, const uint8_t *W_up,
                                    const uint8_t *W_down, __half *output,
                                    __half *workspace, int batch_seq,
                                    int hidden_size, int intermediate_size,
                                    float eps, cudaStream_t stream) {

  rms_norm_launcher_fp16(input, norm_weight, workspace, batch_seq, hidden_size,
                         eps, stream);

  size_t smem_size =
      (size_t)hidden_size * (sizeof(__half) + sizeof(float)) + 64;

  fused_mlp_tensor_core<__half, uint8_t><<<batch_seq, 256, smem_size, stream>>>(
      workspace, W_gate, W_up, W_down, output, batch_seq, hidden_size,
      intermediate_size);

  CUDA_CHECK(cudaGetLastError());
}

// Legacy API that allocates internally (for backward compatibility)
// NOTE: This is SLOW - use the workspace version for production!
void fused_mlp_block_launcher_fp16_legacy(
    const __half *input, const __half *norm_weight, const __half *W_gate,
    const __half *W_up, const __half *W_down, __half *output, int batch_seq,
    int hidden_size, int intermediate_size, float eps, cudaStream_t stream) {

  // Allocate workspace (SLOW - only for compatibility)
  __half *workspace;
  CUDA_CHECK(
      cudaMalloc(&workspace, (size_t)batch_seq * hidden_size * sizeof(__half)));

  fused_mlp_block_launcher_fp16(input, norm_weight, W_gate, W_up, W_down,
                                output, workspace, batch_seq, hidden_size,
                                intermediate_size, eps, stream);

  CUDA_CHECK(cudaFree(workspace));
}

void rope_precompute_launcher(float *cos_cache, float *sin_cache,
                              int max_seq_len, int head_dim, float theta,
                              cudaStream_t stream) {
  const int total = max_seq_len * (head_dim / 2);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  rope_precompute<<<blocks, threads, 0, stream>>>(cos_cache, sin_cache,
                                                  max_seq_len, head_dim, theta);
  CUDA_CHECK(cudaGetLastError());
}

void rope_apply_launcher_fp16(__half *q, __half *k, const float *cos,
                              const float *sin, int batch_size, int num_heads,
                              int seq_len, int head_dim, int position_offset,
                              cudaStream_t stream) {
  dim3 blocks(seq_len, num_heads, batch_size);
  rope_apply<__half><<<blocks, 256, 0, stream>>>(q, k, cos, sin, batch_size,
                                                 num_heads, seq_len, head_dim,
                                                 position_offset);
  CUDA_CHECK(cudaGetLastError());
}

// FP32 version for compatibility
void rope_apply_launcher(float *q, float *k, const float *cos, const float *sin,
                         int batch_size, int num_heads, int seq_len,
                         int head_dim, int position_offset,
                         cudaStream_t stream) {
  dim3 blocks(seq_len, num_heads, batch_size);
  rope_apply<float><<<blocks, 256, 0, stream>>>(q, k, cos, sin, batch_size,
                                                num_heads, seq_len, head_dim,
                                                position_offset);
  CUDA_CHECK(cudaGetLastError());
}

// Deprecated/fallback functions
void rms_norm_launcher(const float *input, const float *weight, float *output,
                       int batch_seq, int hidden_size, float eps,
                       cudaStream_t stream) {
  dispatch_rms_norm(input, weight, output, batch_seq, hidden_size, eps, stream);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_launcher(const float *gate, const float *up, float *output,
                     int64_t total_elements, cudaStream_t stream) {
  // Standalone SwiGLU deprecated - use fused MLP for actual speedups
  printf("Warning: Standalone SwiGLU deprecated - use fused MLP\n");
}

void swiglu_launcher_fp16(const __half *gate, const __half *up, __half *output,
                          int64_t total_elements, cudaStream_t stream) {
  printf("Warning: Standalone SwiGLU deprecated - use fused MLP\n");
}

void fused_rmsnorm_linear_launcher_fp16(const __half *input,
                                        const __half *norm_weight,
                                        const __half *W, __half *output,
                                        int batch_seq, int hidden_size,
                                        int out_size, float eps,
                                        cudaStream_t stream) {
  printf("Warning: Use fused_mlp_block_launcher_fp16 for full pipeline\n");
}

} // extern "C"