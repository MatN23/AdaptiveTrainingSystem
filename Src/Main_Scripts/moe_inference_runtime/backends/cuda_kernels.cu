// backends/cuda_kernels.cu
// CUDA kernels for NVIDIA GPU acceleration

#ifdef USE_CUDA

#include "../core/kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>

// Global cuBLAS handle
static cublasHandle_t g_cublas_handle = nullptr;

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
    if (s >= seq_len) return;
    
    float* row = scores + s * kv_seq_len;
    
    // Find max (for numerical stability)
    float max_val = -1e9f;
    for (int t = 0; t <= cache_pos + s; ++t) {
        max_val = fmaxf(max_val, row[t]);
    }
    
    // Exp and sum
    float sum_exp = 0.0f;
    for (int t = 0; t <= cache_pos + s; ++t) {
        row[t] = expf(row[t] - max_val);
        sum_exp += row[t];
    }
    
    // Normalize
    for (int t = 0; t <= cache_pos + s; ++t) {
        row[t] /= sum_exp;
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
    cudaDeviceSynchronize();
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
    
    cudaDeviceSynchronize();
}

void silu(const float* x, float* out, int32_t n) {
    int blocks = (n + 255) / 256;
    silu_kernel<<<blocks, 256>>>(x, out, n);
    cudaDeviceSynchronize();
}

void mul(const float* a, const float* b, float* out, int32_t n) {
    int blocks = (n + 255) / 256;
    mul_kernel<<<blocks, 256>>>(a, b, out, n);
    cudaDeviceSynchronize();
}

void add_inplace(float* dst, const float* src, int32_t n) {
    int blocks = (n + 255) / 256;
    add_inplace_kernel<<<blocks, 256>>>(dst, src, n);
    cudaDeviceSynchronize();
}

void add_scaled(float* dst, const float* src, float scale, int32_t n) {
    int blocks = (n + 255) / 256;
    add_scaled_kernel<<<blocks, 256>>>(dst, src, scale, n);
    cudaDeviceSynchronize();
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
    
    cudaDeviceSynchronize();
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
        cudaMemcpy(
            kv_cache_k + (h * kv_seq_len + cache_pos) * head_dim,
            k + h * seq_len * head_dim,
            seq_len * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            kv_cache_v + (h * kv_seq_len + cache_pos) * head_dim,
            v + h * seq_len * head_dim,
            seq_len * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
    }
    
    // Attention computation (simplified - use cuBLAS for Q @ K^T)
    int heads_per_kv = num_heads / num_kv_heads;
    
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / heads_per_kv;
        
        // Allocate scores
        float* scores;
        cudaMalloc(&scores, seq_len * kv_seq_len * sizeof(float));
        
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
        attention_softmax_kernel<<<seq_len, 1>>>(scores, seq_len, kv_seq_len, cache_pos);
        
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
        
        cudaFree(scores);
    }
    
    cudaDeviceSynchronize();
}

void moe_route(const float* x, const float* gate_weight,
               int32_t* expert_ids, float* expert_weights,
               int32_t seq_len, int32_t hidden_size,
               int32_t num_experts, int32_t top_k) {
    // Allocate device memory for logits
    float* d_logits;
    cudaMalloc(&d_logits, seq_len * num_experts * sizeof(float));
    
    // Compute gating logits
    dim3 blocks(seq_len, num_experts);
    dim3 threads(256);
    moe_gating_kernel<<<blocks, threads>>>(
        x, gate_weight, d_logits, seq_len, hidden_size, num_experts
    );
    
    // Copy logits back and do top-k selection on CPU
    // (For production, implement top-k on GPU)
    float* h_logits = new float[seq_len * num_experts];
    cudaMemcpy(h_logits, d_logits, seq_len * num_experts * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int t = 0; t < seq_len; ++t) {
        float* logits = h_logits + t * num_experts;
        
        // Softmax
        float max_logit = *std::max_element(logits, logits + num_experts);
        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            logits[e] = expf(logits[e] - max_logit);
            sum_exp += logits[e];
        }
        for (int e = 0; e < num_experts; ++e) {
            logits[e] /= sum_exp;
        }
        
        // Top-k selection
        std::vector<int> indices(num_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        
        // Renormalize
        float sum_top_k = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum_top_k += logits[indices[k]];
        }
        
        for (int k = 0; k < top_k; ++k) {
            expert_ids[t * top_k + k] = indices[k];
            expert_weights[t * top_k + k] = logits[indices[k]] / sum_top_k;
        }
    }
    
    delete[] h_logits;
    cudaFree(d_logits);
}

void mod_route(const float* x, const float* router_weight,
               float* scores, int32_t seq_len, int32_t hidden_size) {
    mod_scoring_kernel<<<seq_len, 256>>>(x, router_weight, scores, seq_len, hidden_size);
    cudaDeviceSynchronize();
}

void expert_forward(const float* x, float* out,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int32_t hidden_size, int32_t intermediate_size) {
    // Allocate temporary buffers
    float *gate_out, *up_out;
    cudaMalloc(&gate_out, intermediate_size * sizeof(float));
    cudaMalloc(&up_out, intermediate_size * sizeof(float));
    
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
    
    cudaFree(gate_out);
    cudaFree(up_out);
    cudaDeviceSynchronize();
}

int32_t sample(const float* logits, int32_t vocab_size, float temperature) {
    // Copy logits to host and sample on CPU
    float* h_logits = new float[vocab_size];
    cudaMemcpy(h_logits, logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Apply temperature and softmax
    float max_logit = *std::max_element(h_logits, h_logits + vocab_size);
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
            delete[] h_logits;
            return i;
        }
    }
    
    delete[] h_logits;
    return vocab_size - 1;
}

} // namespace kernels

#endif // USE_CUDA