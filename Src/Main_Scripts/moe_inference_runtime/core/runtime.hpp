// core/runtime.hpp
// Minimal MoE+MoD Inference Runtime - Main API

#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// ============================================================================
// TENSOR
// ============================================================================

struct Tensor {
    float* data = nullptr;
    std::vector<int64_t> shape;
    bool owns_memory = false;
    
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    
    ~Tensor() {
        if (owns_memory && data) {
#ifdef USE_CUDA
            cudaFree(data);
#else
            delete[] static_cast<float*>(data);
#endif
        }
    }
};

// ============================================================================
// MODEL CONFIG (frozen at export)
// ============================================================================

struct ModelConfig {
    // Architecture
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t intermediate_size;
    int32_t max_seq_len;
    
    // MoE
    bool use_moe;
    int32_t num_experts;
    int32_t moe_top_k;
    std::vector<bool> moe_layers;  // Which layers use MoE
    
    // MoD
    bool use_mod;
    float mod_capacity;
    std::vector<bool> mod_layers;  // Which layers use MoD
    
    // RoPE
    float rope_theta;
};

// ============================================================================
// MODEL WEIGHTS (frozen)
// ============================================================================

struct ModelWeights {
    // Embeddings
    Tensor embed_tokens;      // [vocab_size, hidden_size]
    
    // Per-layer weights
    struct Layer {
        // Attention
        Tensor attn_q, attn_k, attn_v, attn_o;
        Tensor attn_norm_weight;
        
        // FFN (dense or MoE)
        bool is_moe;
        
        // Dense FFN
        Tensor ffn_gate, ffn_up, ffn_down;
        
        // MoE
        Tensor moe_gate_weight;
        std::vector<Tensor> expert_gate;  // Per expert
        std::vector<Tensor> expert_up;
        std::vector<Tensor> expert_down;
        
        // MoD router
        Tensor mod_router_weight;
        float mod_threshold;
        
        Tensor ffn_norm_weight;
    };
    
    std::vector<Layer> layers;
    
    // Final
    Tensor final_norm_weight;
    Tensor lm_head;           // [vocab_size, hidden_size] (or tied to embed)
    
    // RoPE cache
    Tensor rope_cos;          // [max_seq_len, head_dim]
    Tensor rope_sin;
};

// ============================================================================
// INFERENCE ENGINE
// ============================================================================

class InferenceEngine {
private:
    ModelConfig config_;
    ModelWeights weights_;
    
    // KV cache for generation
    std::vector<Tensor> kv_cache_k_;
    std::vector<Tensor> kv_cache_v_;
    int32_t cache_pos_ = 0;
    
    // Temporary buffers
    std::vector<Tensor> buffers_;
    
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine();
    
    // Forward pass
    void forward(
        const int32_t* input_ids,
        int32_t batch_size,
        int32_t seq_len,
        float* logits_out
    );
    
    // Generation
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt,
        int32_t max_tokens,
        float temperature = 1.0f
    );
    
    void reset_cache();
    
private:
    void load_model(const std::string& path);
    void init_buffers();
    
    // Layer operations
    void embedding(const int32_t* ids, int32_t n, Tensor& out);
    void rms_norm(const Tensor& x, const Tensor& w, Tensor& out);
    void linear(const Tensor& x, const Tensor& w, Tensor& out);
    void attention_layer(int32_t layer_idx, const Tensor& x, Tensor& out);
    void ffn_layer(int32_t layer_idx, const Tensor& x, Tensor& out);
    void moe_layer(int32_t layer_idx, const Tensor& x, Tensor& out);
    void mod_layer(int32_t layer_idx, const Tensor& x, Tensor& out);
};

// ============================================================================
// MODEL FILE FORMAT
// ============================================================================

struct ModelHeader {
    uint32_t magic = 0x4D4F4549;  // "MOEI"
    uint32_t version = 1;
    ModelConfig config;
};

// File layout:
// [Header] [Weights...] [RoPE cache]
// All weights stored as float32 arrays