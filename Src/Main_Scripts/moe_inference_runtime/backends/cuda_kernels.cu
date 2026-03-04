// backends/cuda_kernels.cu
// CUDA kernels for NVIDIA GPU acceleration

#ifdef USE_CUDA

#include "../core/kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

// Global cuBLAS handle
static cublasHandle_t g_cublas_handle = nullptr;
static float* g_expert_gate_tmp = nullptr;
static float* g_expert_up_tmp = nullptr;
static int g_expert_tmp_capacity = 0;

// Initialize CUDA
__attribute__((constructor))
static void init_cuda() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 0) {
        cudaSetDevice(0);
        cublasCreate(&g_cublas_handle);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("✓ CUDA initialized: %s\n", prop.name);
    }
}

// Cleanup
__attribute__((destructor))
static void cleanup_cuda() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
    }
    if (g_expert_gate_tmp) {
        cudaFree(g_expert_gate_tmp);
        g_expert_gate_tmp = nullptr;
    }
    if (g_expert_up_tmp) {
        cudaFree(g_expert_up_tmp);
        g_expert_up_tmp = nullptr;
    }
    g_expert_tmp_capacity = 0;
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

// RMS Normalization kernel
__global__ void rms_norm_kernel(
    const float* x,
    const float* weight,
    float* out,
    int n,
    int dim,
    float eps
) {
    int idx = blockIdx.x;
    if (idx >= n) return;
    
    const float* xi = x + idx * dim;
    float* yi = out + idx * dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = xi[i];
        sum_sq += val * val;
    }
    
    // Block reduce
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared_sum[0] / dim + eps);
    float scale = 1.0f / rms;
    
    // Apply normalization
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        yi[i] = xi[i] * scale * weight[i];
    }
}

// SiLU activation kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        out[idx] = xi / (1.0f + expf(-xi));
    }
}

// Element-wise multiplication kernel
__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Add in-place kernel
__global__ void add_inplace_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

// Scaled add kernel
__global__ void add_scaled_kernel(float* dst, const float* src, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx] * scale;
    }
}

// RoPE kernel
__global__ void rope_kernel(
    float* q,
    float* k,
    const float* cos_cache,
    const float* sin_cache,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int pos_offset
) {
    int h = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    
    if (d >= head_dim / 2) return;
    
    int pos = pos_offset + s;
    float cos_val = cos_cache[pos * head_dim + d];
    float sin_val = sin_cache[pos * head_dim + d];
    
    // Apply to query
    if (h < num_heads) {
        int idx = (h * seq_len + s) * head_dim;
        float q_real = q[idx + d];
        float q_imag = q[idx + d + head_dim / 2];
        q[idx + d] = q_real * cos_val - q_imag * sin_val;
        q[idx + d + head_dim / 2] = q_real * sin_val + q_imag * cos_val;
    }
    
    // Apply to key
    if (h < num_kv_heads) {
        int idx = (h * seq_len + s) * head_dim;
        float k_real = k[idx + d];
        float k_imag = k[idx + d + head_dim / 2];
        k[idx + d] = k_real * cos_val - k_imag * sin_val;
        k[idx + d + head_dim / 2] = k_real * sin_val + k_imag * cos_val;
    }
}

// Attention softmax kernel
__global__ void attention_softmax_kernel(
    float* scores,
    int seq_len,
    int kv_seq_len,
    int cache_pos
) {
    int s = blockIdx.x;
    int tid = threadIdx.x;
    if (s >= seq_len) return;

    int valid_len = cache_pos + s + 1;
    if (valid_len > kv_seq_len) valid_len = kv_seq_len;

    float* row = scores + s * kv_seq_len;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    float local_max = -FLT_MAX;
    for (int t = tid; t < valid_len; t += blockDim.x) {
        local_max = fmaxf(local_max, row[t]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int t = tid; t < valid_len; t += blockDim.x) {
        float ex = expf(row[t] - max_val);
        row[t] = ex;
        local_sum += ex;
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / (ssum[0] + 1e-9f);

    for (int t = tid; t < valid_len; t += blockDim.x) {
        row[t] *= inv_sum;
    }
    for (int t = valid_len + tid; t < kv_seq_len; t += blockDim.x) {
        row[t] = 0.0f;
    }
}

__global__ void moe_softmax_topk_kernel(
    const float* logits,
    int32_t* expert_ids,
    float* expert_weights,
    int seq_len,
    int num_experts,
    int top_k
) {
    constexpr int MAX_TOP_K = 32;
    int token_idx = blockIdx.x;
    if (token_idx >= seq_len || threadIdx.x != 0) return;

    if (top_k > MAX_TOP_K) top_k = MAX_TOP_K;
    const float* row = logits + token_idx * num_experts;

    float max_val = -FLT_MAX;
    for (int e = 0; e < num_experts; ++e) {
        max_val = fmaxf(max_val, row[e]);
    }

    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        sum_exp += expf(row[e] - max_val);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    float top_vals[MAX_TOP_K];
    int top_idxs[MAX_TOP_K];
    for (int i = 0; i < top_k; ++i) {
        top_vals[i] = -1.0f;
        top_idxs[i] = -1;
    }

    for (int e = 0; e < num_experts; ++e) {
        float p = expf(row[e] - max_val) * inv_sum;
        if (p <= top_vals[top_k - 1]) continue;
        top_vals[top_k - 1] = p;
        top_idxs[top_k - 1] = e;
        for (int i = top_k - 2; i >= 0; --i) {
            if (top_vals[i + 1] <= top_vals[i]) break;
            float tmp_v = top_vals[i];
            int tmp_i = top_idxs[i];
            top_vals[i] = top_vals[i + 1];
            top_idxs[i] = top_idxs[i + 1];
            top_vals[i + 1] = tmp_v;
            top_idxs[i + 1] = tmp_i;
        }
    }

    float sum_top = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        sum_top += fmaxf(top_vals[i], 0.0f);
    }
    float inv_top = 1.0f / (sum_top + 1e-9f);

    int base = token_idx * top_k;
    for (int i = 0; i < top_k; ++i) {
        expert_ids[base + i] = top_idxs[i];
        expert_weights[base + i] = fmaxf(top_vals[i], 0.0f) * inv_top;
    }
}

// MoE routing kernel (compute gating logits)
__global__ void moe_gating_kernel(
    const float* x,
    const float* gate_weight,
    float* logits,
    int seq_len,
    int hidden_size,
    int num_experts
) {
    int t = blockIdx.x;
    int e = blockIdx.y;
    
    if (t >= seq_len || e >= num_experts) return;
    
    const float* xi = x + t * hidden_size;
    const float* gate_e = gate_weight + e * hidden_size;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += xi[i] * gate_e[i];
    }
    
    // Block reduce
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        logits[t * num_experts + e] = shared_sum[0];
    }
}

// MoD routing kernel (compute token importance scores)
__global__ void mod_scoring_kernel(
    const float* x,
    const float* router_weight,
    float* scores,
    int seq_len,
    int hidden_size
) {
    int t = blockIdx.x;
    if (t >= seq_len) return;
    
    const float* xi = x + t * hidden_size;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += xi[i] * router_weight[i];
    }
    
    // Block reduce
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        // Sigmoid activation
        float logit = shared_sum[0];
        scores[t] = 1.0f / (1.0f + expf(-logit));
    }
}

// ============================================================================
// KERNEL INTERFACE IMPLEMENTATIONS
// ============================================================================

namespace kernels {

void rms_norm(const float* x, const float* weight, float* out,
              int32_t n, int32_t dim, float eps) {
    dim3 blocks(n);
    dim3 threads(256);
    rms_norm_kernel<<<blocks, threads>>>(x, weight, out, n, dim, eps);
}

void matmul(const float* a, const float* b, float* c,
            int32_t m, int32_t k, int32_t n) {
    // Use cuBLAS for GEMM: C = A @ B^T
    // A: [m, k], B: [n, k], C: [m, n]
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major, so we compute: C^T = B @ A^T
    cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N,    // B is not transposed (but stored as B^T)
        CUBLAS_OP_N,    // A is not transposed
        n, m, k,        // Dimensions
        &alpha,
        b, n,           // B matrix
        a, k,           // A matrix
        &beta,
        c, n            // C matrix
    );
}

void silu(const float* x, float* out, int32_t n) {
    int blocks = (n + 255) / 256;
    silu_kernel<<<blocks, 256>>>(x, out, n);
}

void mul(const float* a, const float* b, float* out, int32_t n) {
    int blocks = (n + 255) / 256;
    mul_kernel<<<blocks, 256>>>(a, b, out, n);
}

void add_inplace(float* dst, const float* src, int32_t n) {
    int blocks = (n + 255) / 256;
    add_inplace_kernel<<<blocks, 256>>>(dst, src, n);
}

void add_scaled(float* dst, const float* src, float scale, int32_t n) {
    int blocks = (n + 255) / 256;
    add_scaled_kernel<<<blocks, 256>>>(dst, src, scale, n);
}

void apply_rope(float* q, float* k,
                const float* cos_cache, const float* sin_cache,
                int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
                int32_t head_dim, int32_t pos_offset) {
    dim3 blocks(std::max(num_heads, num_kv_heads), seq_len);
    dim3 threads(head_dim / 2);
    
    rope_kernel<<<blocks, threads>>>(
        q, k, cos_cache, sin_cache,
        seq_len, num_heads, num_kv_heads, head_dim, pos_offset
    );
}

void attention(const float* q, const float* k, const float* v, float* out,
               float* kv_cache_k, float* kv_cache_v,
               int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
               int32_t head_dim, int32_t cache_pos) {
    // Simplified CUDA attention implementation
    // For production, use cuDNN or Flash Attention CUDA kernels
    
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_seq_len = cache_pos + seq_len;
    
    // Update KV cache (memcpy on device)
    for (int h = 0; h < num_kv_heads; ++h) {
        cudaMemcpyAsync(
            kv_cache_k + (h * kv_seq_len + cache_pos) * head_dim,
            k + h * seq_len * head_dim,
            seq_len * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            0
        );
        cudaMemcpyAsync(
            kv_cache_v + (h * kv_seq_len + cache_pos) * head_dim,
            v + h * seq_len * head_dim,
            seq_len * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            0
        );
    }
    
    // Attention computation (simplified - use cuBLAS for Q @ K^T)
    int heads_per_kv = num_heads / num_kv_heads;
    
    float* scores = nullptr;
    cudaMalloc(&scores, seq_len * kv_seq_len * sizeof(float));

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / heads_per_kv;

        // Q @ K^T using cuBLAS
        const float alpha = scale;
        const float beta = 0.0f;
        
        cublasSgemm(
            g_cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            kv_seq_len, seq_len, head_dim,
            &alpha,
            kv_cache_k + kv_h * kv_seq_len * head_dim, head_dim,
            q + h * seq_len * head_dim, head_dim,
            &beta,
            scores, kv_seq_len
        );

        // Softmax
        attention_softmax_kernel<<<seq_len, 256>>>(scores, seq_len, kv_seq_len, cache_pos);

        // Scores @ V using cuBLAS
        const float one = 1.0f;
        const float zero = 0.0f;
        
        cublasSgemm(
            g_cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim, seq_len, kv_seq_len,
            &one,
            kv_cache_v + kv_h * kv_seq_len * head_dim, head_dim,
            scores, kv_seq_len,
            &zero,
            out + h * seq_len * head_dim, head_dim
        );
    }

    cudaFree(scores);
}

void moe_route(const float* x, const float* gate_weight,
               int32_t* expert_ids, float* expert_weights,
               int32_t seq_len, int32_t hidden_size,
               int32_t num_experts, int32_t top_k) {
    if (top_k <= 0 || seq_len <= 0) {
        return;
    }

    // Allocate device memory for logits
    float* d_logits;
    cudaMalloc(&d_logits, seq_len * num_experts * sizeof(float));

    int32_t* d_expert_ids;
    float* d_expert_weights;
    cudaMalloc(&d_expert_ids, seq_len * top_k * sizeof(int32_t));
    cudaMalloc(&d_expert_weights, seq_len * top_k * sizeof(float));
    
    // Compute gating logits
    dim3 blocks(seq_len, num_experts);
    dim3 threads(256);
    moe_gating_kernel<<<blocks, threads>>>(
        x, gate_weight, d_logits, seq_len, hidden_size, num_experts
    );

    moe_softmax_topk_kernel<<<seq_len, 1>>>(
        d_logits, d_expert_ids, d_expert_weights, seq_len, num_experts, top_k
    );

    cudaMemcpy(expert_ids, d_expert_ids, seq_len * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(expert_weights, d_expert_weights, seq_len * top_k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_logits);
    cudaFree(d_expert_ids);
    cudaFree(d_expert_weights);
}

void mod_route(const float* x, const float* router_weight,
               float* scores, int32_t seq_len, int32_t hidden_size) {
    float* d_scores = nullptr;
    cudaMalloc(&d_scores, seq_len * sizeof(float));
    mod_scoring_kernel<<<seq_len, 256>>>(x, router_weight, d_scores, seq_len, hidden_size);
    cudaMemcpy(scores, d_scores, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_scores);
}

void expert_forward(const float* x, float* out,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int32_t hidden_size, int32_t intermediate_size) {
    if (g_expert_tmp_capacity < intermediate_size) {
        if (g_expert_gate_tmp) cudaFree(g_expert_gate_tmp);
        if (g_expert_up_tmp) cudaFree(g_expert_up_tmp);
        cudaMalloc(&g_expert_gate_tmp, intermediate_size * sizeof(float));
        cudaMalloc(&g_expert_up_tmp, intermediate_size * sizeof(float));
        g_expert_tmp_capacity = intermediate_size;
    }

    float* gate_out = g_expert_gate_tmp;
    float* up_out = g_expert_up_tmp;
    
    // Gate projection: gate_out = x @ gate_w^T
    const float one = 1.0f;
    const float zero = 0.0f;
    
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T,
                hidden_size, intermediate_size, &one,
                gate_w, hidden_size, x, 1, &zero, gate_out, 1);
    
    // Up projection: up_out = x @ up_w^T
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T,
                hidden_size, intermediate_size, &one,
                up_w, hidden_size, x, 1, &zero, up_out, 1);
    
    // SwiGLU: gate_out = silu(gate_out) * up_out
    silu_kernel<<<(intermediate_size + 255) / 256, 256>>>(gate_out, gate_out, intermediate_size);
    mul_kernel<<<(intermediate_size + 255) / 256, 256>>>(gate_out, up_out, gate_out, intermediate_size);
    
    // Down projection: out = gate_out @ down_w^T
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T,
                intermediate_size, hidden_size, &one,
                down_w, intermediate_size, gate_out, 1, &zero, out, 1);
    
}

int32_t sample(const float* logits, int32_t vocab_size, float temperature) {
    std::vector<float> h_logits(vocab_size);
#if CUDART_VERSION >= 10000
    cudaPointerAttributes attr;
    cudaError_t attr_status = cudaPointerGetAttributes(&attr, logits);
    bool is_device_ptr = false;
    if (attr_status == cudaSuccess) {
#if CUDART_VERSION >= 11000
        is_device_ptr = (attr.type == cudaMemoryTypeDevice);
#else
        is_device_ptr = (attr.memoryType == cudaMemoryTypeDevice);
#endif
    } else {
        // Clear the sticky error raised by querying a host pointer.
        (void)cudaGetLastError();
    }
#else
    bool is_device_ptr = true;
#endif

    if (is_device_ptr) {
        cudaMemcpy(h_logits.data(), logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(h_logits.data(), logits, vocab_size * sizeof(float));
    }

    // Apply temperature and softmax
    float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
    float sum_exp = 0.0f;
    
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = expf((h_logits[i] - max_logit) / temperature);
        sum_exp += h_logits[i];
    }
    
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] /= sum_exp;
    }
    
    // Sample
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += h_logits[i];
        if (r < cumsum) {
            return i;
        }
    }

    return vocab_size - 1;
}

} // namespace kernels

#endif // USE_CUDA
