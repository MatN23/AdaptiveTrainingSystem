// backends/cpu_kernels.cpp
// CPU kernels with SIMD (AVX2/NEON) and threading

#include "../core/kernels.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#ifdef USE_AVX2
#include <immintrin.h>
#define SIMD_WIDTH 8
#elif defined(USE_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4
#else
#define SIMD_WIDTH 1
#endif

namespace kernels {

// ============================================================================
// SIMD HELPERS
// ============================================================================

inline float dot_product(const float* a, const float* b, int32_t n) {
    float sum = 0.0f;
    int32_t i = 0;
    
#ifdef USE_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, hi);
    sum = _mm_cvtss_f32(lo);
    
#elif defined(USE_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }
    
    float32x2_t s = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    sum = vget_lane_f32(vpadd_f32(s, s), 0);
#endif
    
    // Scalar remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

// ============================================================================
// BASIC OPS
// ============================================================================

void rms_norm(const float* x, const float* weight, float* out,
              int32_t n, int32_t dim, float eps) {
    for (int32_t i = 0; i < n; ++i) {
        const float* xi = x + i * dim;
        float* yi = out + i * dim;
        
        // Compute RMS
        float sum_sq = 0.0f;
        for (int32_t j = 0; j < dim; ++j) {
            sum_sq += xi[j] * xi[j];
        }
        
        float rms = std::sqrt(sum_sq / dim + eps);
        float scale = 1.0f / rms;
        
        // Normalize and scale
        for (int32_t j = 0; j < dim; ++j) {
            yi[j] = xi[j] * scale * weight[j];
        }
    }
}

void matmul(const float* a, const float* b, float* c,
            int32_t m, int32_t k, int32_t n) {
    // c = a @ b^T
    // a: [m, k], b: [n, k], c: [m, n]
    
    int32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    
    auto worker = [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; ++i) {
            for (int32_t j = 0; j < n; ++j) {
                c[i * n + j] = dot_product(a + i * k, b + j * k, k);
            }
        }
    };
    
    int32_t chunk = (m + num_threads - 1) / num_threads;
    for (int32_t t = 0; t < num_threads; ++t) {
        int32_t start = t * chunk;
        int32_t end = std::min(start + chunk, m);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    
    for (auto& t : threads) t.join();
}

void silu(const float* x, float* out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
}

void mul(const float* a, const float* b, float* out, int32_t n) {
    int32_t i = 0;
    
#ifdef USE_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#elif defined(USE_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#endif
    
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void add_inplace(float* dst, const float* src, int32_t n) {
    int32_t i = 0;
    
#ifdef USE_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 vs = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(vd, vs));
    }
#elif defined(USE_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t vd = vld1q_f32(dst + i);
        float32x4_t vs = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(vd, vs));
    }
#endif
    
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

void add_scaled(float* dst, const float* src, float scale, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
}

// ============================================================================
// ROPE
// ============================================================================

void apply_rope(float* q, float* k,
                const float* cos_cache, const float* sin_cache,
                int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
                int32_t head_dim, int32_t pos_offset) {
    int32_t half_dim = head_dim / 2;
    
    // Apply to queries
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
    
    // Apply to keys
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
// ATTENTION
// ============================================================================

void attention(const float* q, const float* k, const float* v, float* out,
               float* kv_cache_k, float* kv_cache_v,
               int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
               int32_t head_dim, int32_t cache_pos) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int32_t kv_seq_len = cache_pos + seq_len;
    
    // Update KV cache
    for (int32_t h = 0; h < num_kv_heads; ++h) {
        for (int32_t s = 0; s < seq_len; ++s) {
            int32_t cache_idx = (h * kv_seq_len + cache_pos + s) * head_dim;
            int32_t input_idx = (h * seq_len + s) * head_dim;
            std::memcpy(kv_cache_k + cache_idx, k + input_idx, head_dim * sizeof(float));
            std::memcpy(kv_cache_v + cache_idx, v + input_idx, head_dim * sizeof(float));
        }
    }
    
    // Attention per head
    int32_t heads_per_kv = num_heads / num_kv_heads;
    
    for (int32_t h = 0; h < num_heads; ++h) {
        int32_t kv_h = h / heads_per_kv;
        
        for (int32_t s = 0; s < seq_len; ++s) {
            const float* qi = q + (h * seq_len + s) * head_dim;
            
            // Compute attention scores
            std::vector<float> scores(kv_seq_len);
            for (int32_t t = 0; t < kv_seq_len; ++t) {
                // Causal mask
                if (t > cache_pos + s) {
                    scores[t] = -1e9f;
                } else {
                    const float* ki = kv_cache_k + (kv_h * kv_seq_len + t) * head_dim;
                    scores[t] = dot_product(qi, ki, head_dim) * scale;
                }
            }
            
            // Softmax
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (auto& sc : scores) {
                sc = std::exp(sc - max_score);
                sum_exp += sc;
            }
            for (auto& sc : scores) sc /= sum_exp;
            
            // Weighted sum of values
            float* outi = out + (h * seq_len + s) * head_dim;
            std::memset(outi, 0, head_dim * sizeof(float));
            
            for (int32_t t = 0; t < kv_seq_len; ++t) {
                const float* vi = kv_cache_v + (kv_h * kv_seq_len + t) * head_dim;
                float w = scores[t];
                for (int32_t d = 0; d < head_dim; ++d) {
                    outi[d] += w * vi[d];
                }
            }
        }
    }
}

// ============================================================================
// MOE ROUTING
// ============================================================================

void moe_route(const float* x, const float* gate_weight,
               int32_t* expert_ids, float* expert_weights,
               int32_t seq_len, int32_t hidden_size,
               int32_t num_experts, int32_t top_k) {
    for (int32_t t = 0; t < seq_len; ++t) {
        const float* xi = x + t * hidden_size;
        
        // Compute logits
        std::vector<float> logits(num_experts);
        for (int32_t e = 0; e < num_experts; ++e) {
            logits[e] = dot_product(xi, gate_weight + e * hidden_size, hidden_size);
        }
        
        // Softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (auto& l : logits) {
            l = std::exp(l - max_logit);
            sum_exp += l;
        }
        for (auto& l : logits) l /= sum_exp;
        
        // Top-k
        std::vector<int32_t> indices(num_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        
        // Renormalize top-k
        float sum_top_k = 0.0f;
        for (int32_t k = 0; k < top_k; ++k) {
            sum_top_k += logits[indices[k]];
        }
        
        for (int32_t k = 0; k < top_k; ++k) {
            expert_ids[t * top_k + k] = indices[k];
            expert_weights[t * top_k + k] = logits[indices[k]] / sum_top_k;
        }
    }
}

// ============================================================================
// MOD ROUTING
// ============================================================================

void mod_route(const float* x, const float* router_weight,
               float* scores, int32_t seq_len, int32_t hidden_size) {
    for (int32_t t = 0; t < seq_len; ++t) {
        scores[t] = dot_product(x + t * hidden_size, router_weight, hidden_size);
        scores[t] = 1.0f / (1.0f + std::exp(-scores[t]));  // Sigmoid
    }
}

// ============================================================================
// EXPERT FORWARD (SwiGLU)
// ============================================================================

void expert_forward(const float* x, float* out,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int32_t hidden_size, int32_t intermediate_size) {
    // Gate projection
    std::vector<float> gate_out(intermediate_size);
    for (int32_t i = 0; i < intermediate_size; ++i) {
        gate_out[i] = dot_product(x, gate_w + i * hidden_size, hidden_size);
    }
    
    // Up projection
    std::vector<float> up_out(intermediate_size);
    for (int32_t i = 0; i < intermediate_size; ++i) {
        up_out[i] = dot_product(x, up_w + i * hidden_size, hidden_size);
    }
    
    // SwiGLU: silu(gate) * up
    for (int32_t i = 0; i < intermediate_size; ++i) {
        float g = gate_out[i];
        gate_out[i] = (g / (1.0f + std::exp(-g))) * up_out[i];
    }
    
    // Down projection
    for (int32_t i = 0; i < hidden_size; ++i) {
        out[i] = dot_product(gate_out.data(), down_w + i * intermediate_size, intermediate_size);
    }
}

// ============================================================================
// SAMPLING
// ============================================================================

int32_t sample(const float* logits, int32_t vocab_size, float temperature) {
    // Apply temperature
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    for (auto& p : probs) p /= sum_exp;
    
    // Sample
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    
    return dist(gen);
}

} // namespace kernels