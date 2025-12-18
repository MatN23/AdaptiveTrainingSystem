// backends/mps_kernels.mm
// Apple Metal Performance Shaders backend for MPS

#ifdef USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../core/kernels.hpp"
#include <vector>

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

// Initialize Metal
__attribute__((constructor))
static void init_metal() {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device) {
            g_queue = [g_device newCommandQueue];
            NSLog(@"✓ Metal initialized: %@", [g_device name]);
        }
    }
}

// Helper: Create buffer
static id<MTLBuffer> create_buffer(const float* data, size_t count) {
    size_t bytes = count * sizeof(float);
    if (data) {
        return [g_device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    } else {
        return [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
}

namespace kernels {

// ============================================================================
// BASIC OPS
// ============================================================================

void rms_norm(const float* x, const float* weight, float* out,
              int32_t n, int32_t dim, float eps) {
    @autoreleasepool {
        id<MTLBuffer> x_buf = create_buffer(x, n * dim);
        id<MTLBuffer> w_buf = create_buffer(weight, dim);
        id<MTLBuffer> out_buf = create_buffer(nullptr, n * dim);
        
        // Use MPS for normalization (approximate with instance norm)
        MPSImageNormalizationMeanAndVariance* norm = 
            [[MPSImageNormalizationMeanAndVariance alloc] 
                initWithDevice:g_device];
        
        // Fall back to CPU for now (MPS doesn't have RMSNorm)
        // TODO: Write custom Metal shader
        for (int32_t i = 0; i < n; ++i) {
            const float* xi = x + i * dim;
            float* yi = out + i * dim;
            
            float sum_sq = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                sum_sq += xi[j] * xi[j];
            }
            
            float rms = sqrtf(sum_sq / dim + eps);
            float scale = 1.0f / rms;
            
            for (int32_t j = 0; j < dim; ++j) {
                yi[j] = xi[j] * scale * weight[j];
            }
        }
    }
}

void matmul(const float* a, const float* b, float* c,
            int32_t m, int32_t k, int32_t n) {
    @autoreleasepool {
        // Use MPS matrix multiplication
        id<MTLBuffer> a_buf = create_buffer(a, m * k);
        id<MTLBuffer> b_buf = create_buffer(b, n * k);
        id<MTLBuffer> c_buf = create_buffer(nullptr, m * n);
        
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:k columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        
        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:desc_c];
        
        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] 
            initWithDevice:g_device transposeLeft:NO transposeRight:YES
            resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
        
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];
        
        // Copy result back
        memcpy(c, [c_buf contents], m * n * sizeof(float));
    }
}

void silu(const float* x, float* out, int32_t n) {
    // SiLU: x / (1 + exp(-x))
    for (int32_t i = 0; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + expf(-xi));
    }
}

void mul(const float* a, const float* b, float* out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void add_inplace(float* dst, const float* src, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void add_scaled(float* dst, const float* src, float scale, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
}

// ============================================================================
// ROPE (CPU fallback for now)
// ============================================================================

void apply_rope(float* q, float* k,
                const float* cos_cache, const float* sin_cache,
                int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
                int32_t head_dim, int32_t pos_offset) {
    int32_t half_dim = head_dim / 2;
    
    for (int32_t h = 0; h < num_heads; ++h) {
        for (int32_t s = 0; s < seq_len; ++s) {
            int32_t pos = pos_offset + s;
            const float* cos = cos_cache + pos * head_dim;
            const float* sin = sin_cache + pos * head_dim;
            
            float* qi = q + (h * seq_len + s) * head_dim;
            
            for (int32_t d = 0; d < half_dim; ++d) {
                float q_real = qi[d];
                float q_imag = qi[d + half_dim];
                qi[d] = q_real * cos[d] - q_imag * sin[d];
                qi[d + half_dim] = q_real * sin[d] + q_imag * cos[d];
            }
        }
    }
    
    for (int32_t h = 0; h < num_kv_heads; ++h) {
        for (int32_t s = 0; s < seq_len; ++s) {
            int32_t pos = pos_offset + s;
            const float* cos = cos_cache + pos * head_dim;
            const float* sin = sin_cache + pos * head_dim;
            
            float* ki = k + (h * seq_len + s) * head_dim;
            
            for (int32_t d = 0; d < half_dim; ++d) {
                float k_real = ki[d];
                float k_imag = ki[d + half_dim];
                ki[d] = k_real * cos[d] - k_imag * sin[d];
                ki[d + half_dim] = k_real * sin[d] + k_imag * cos[d];
            }
        }
    }
}

// ============================================================================
// ATTENTION (CPU fallback - MPS attention is complex)
// ============================================================================

void attention(const float* q, const float* k, const float* v, float* out,
               float* kv_cache_k, float* kv_cache_v,
               int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
               int32_t head_dim, int32_t cache_pos) {
    // Use CPU implementation for now
    // TODO: Optimize with Metal shaders
    
    float scale = 1.0f / sqrtf((float)head_dim);
    int32_t kv_seq_len = cache_pos + seq_len;
    
    // Update cache
    for (int32_t h = 0; h < num_kv_heads; ++h) {
        for (int32_t s = 0; s < seq_len; ++s) {
            int32_t cache_idx = (h * kv_seq_len + cache_pos + s) * head_dim;
            int32_t input_idx = (h * seq_len + s) * head_dim;
            memcpy(kv_cache_k + cache_idx, k + input_idx, head_dim * sizeof(float));
            memcpy(kv_cache_v + cache_idx, v + input_idx, head_dim * sizeof(float));
        }
    }
    
    // Attention computation (simplified)
    int32_t heads_per_kv = num_heads / num_kv_heads;
    
    for (int32_t h = 0; h < num_heads; ++h) {
        int32_t kv_h = h / heads_per_kv;
        
        for (int32_t s = 0; s < seq_len; ++s) {
            const float* qi = q + (h * seq_len + s) * head_dim;
            float* outi = out + (h * seq_len + s) * head_dim;
            memset(outi, 0, head_dim * sizeof(float));
            
            float sum_weights = 0.0f;
            for (int32_t t = 0; t <= cache_pos + s; ++t) {
                const float* ki = kv_cache_k + (kv_h * kv_seq_len + t) * head_dim;
                const float* vi = kv_cache_v + (kv_h * kv_seq_len + t) * head_dim;
                
                float score = 0.0f;
                for (int32_t d = 0; d < head_dim; ++d) {
                    score += qi[d] * ki[d];
                }
                score *= scale;
                score = expf(score);
                sum_weights += score;
                
                for (int32_t d = 0; d < head_dim; ++d) {
                    outi[d] += score * vi[d];
                }
            }
            
            for (int32_t d = 0; d < head_dim; ++d) {
                outi[d] /= sum_weights;
            }
        }
    }
}

// ============================================================================
// MOE/MOD (CPU fallback)
// ============================================================================

void moe_route(const float* x, const float* gate_weight,
               int32_t* expert_ids, float* expert_weights,
               int32_t seq_len, int32_t hidden_size,
               int32_t num_experts, int32_t top_k) {
    // TODO: Optimize with Metal
    for (int32_t t = 0; t < seq_len; ++t) {
        const float* xi = x + t * hidden_size;
        std::vector<float> logits(num_experts);
        
        for (int32_t e = 0; e < num_experts; ++e) {
            float sum = 0.0f;
            for (int32_t i = 0; i < hidden_size; ++i) {
                sum += xi[i] * gate_weight[e * hidden_size + i];
            }
            logits[e] = sum;
        }
        
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (auto& l : logits) {
            l = expf(l - max_logit);
            sum_exp += l;
        }
        for (auto& l : logits) l /= sum_exp;
        
        std::vector<int32_t> indices(num_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        
        float sum_top = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum_top += logits[indices[k]];
        }
        
        for (int k = 0; k < top_k; ++k) {
            expert_ids[t * top_k + k] = indices[k];
            expert_weights[t * top_k + k] = logits[indices[k]] / sum_top;
        }
    }
}

void mod_route(const float* x, const float* router_weight,
               float* scores, int32_t seq_len, int32_t hidden_size) {
    for (int32_t t = 0; t < seq_len; ++t) {
        float sum = 0.0f;
        for (int32_t i = 0; i < hidden_size; ++i) {
            sum += x[t * hidden_size + i] * router_weight[i];
        }
        scores[t] = 1.0f / (1.0f + expf(-sum));
    }
}

void expert_forward(const float* x, float* out,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int32_t hidden_size, int32_t intermediate_size) {
    std::vector<float> gate_out(intermediate_size);
    std::vector<float> up_out(intermediate_size);
    
    for (int32_t i = 0; i < intermediate_size; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int32_t j = 0; j < hidden_size; ++j) {
            g += x[j] * gate_w[i * hidden_size + j];
            u += x[j] * up_w[i * hidden_size + j];
        }
        gate_out[i] = (g / (1.0f + expf(-g))) * u;
    }
    
    for (int32_t i = 0; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (int32_t j = 0; j < intermediate_size; ++j) {
            sum += gate_out[j] * down_w[i * intermediate_size + j];
        }
        out[i] = sum;
    }
}

int32_t sample(const float* logits, int32_t vocab_size, float temperature) {
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    
    return vocab_size - 1;
}

} // namespace kernels

#endif // USE_MPS