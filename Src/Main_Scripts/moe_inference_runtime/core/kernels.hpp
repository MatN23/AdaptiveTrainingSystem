// core/kernels.hpp
// Backend kernel interface - implemented by CPU/MPS/CUDA

#pragma once
#include <cstdint>

namespace kernels {

// Basic ops
void rms_norm(const float* x, const float* weight, float* out, 
              int32_t n, int32_t dim, float eps);

void matmul(const float* a, const float* b, float* c,
            int32_t m, int32_t k, int32_t n);

void silu(const float* x, float* out, int32_t n);

void mul(const float* a, const float* b, float* out, int32_t n);

void add_inplace(float* dst, const float* src, int32_t n);

void add_scaled(float* dst, const float* src, float scale, int32_t n);

// RoPE
void apply_rope(float* q, float* k, 
                const float* cos_cache, const float* sin_cache,
                int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
                int32_t head_dim, int32_t pos_offset);

// Attention
void attention(const float* q, const float* k, const float* v, float* out,
               float* kv_cache_k, float* kv_cache_v,
               int32_t seq_len, int32_t num_heads, int32_t num_kv_heads,
               int32_t head_dim, int32_t cache_pos);

// MoE routing
void moe_route(const float* x, const float* gate_weight,
               int32_t* expert_ids, float* expert_weights,
               int32_t seq_len, int32_t hidden_size, 
               int32_t num_experts, int32_t top_k);

// MoD routing
void mod_route(const float* x, const float* router_weight,
               float* scores, int32_t seq_len, int32_t hidden_size);

// Expert forward (SwiGLU)
void expert_forward(const float* x, float* out,
                    const float* gate_w, const float* up_w, const float* down_w,
                    int32_t hidden_size, int32_t intermediate_size);

// Sampling
int32_t sample(const float* logits, int32_t vocab_size, float temperature);

} // namespace kernels