// core/runtime.cpp
// Model loading and execution orchestration

#include "runtime.hpp"
#include "kernels.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <random>

// ============================================================================
// MODEL LOADING
// ============================================================================

InferenceEngine::InferenceEngine(const std::string& model_path) {
    load_model(model_path);
    init_buffers();
}

InferenceEngine::~InferenceEngine() {
    // Cleanup handled by Tensor destructors
}

void InferenceEngine::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model: " + path);
    }
    
    // Read header
    ModelHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(ModelHeader));
    
    if (header.magic != 0x4D4F4549) {
        throw std::runtime_error("Invalid model file");
    }
    
    config_ = header.config;
    
#ifdef USE_CUDA
    std::cout << "✓ Loaded config (CUDA mode): " << config_.num_layers << " layers, "
              << config_.hidden_size << " hidden" << std::endl;
#elif defined(USE_MPS)
    std::cout << "✓ Loaded config (MPS mode): " << config_.num_layers << " layers, "
              << config_.hidden_size << " hidden" << std::endl;
#else
    std::cout << "✓ Loaded config (CPU mode): " << config_.num_layers << " layers, "
              << config_.hidden_size << " hidden" << std::endl;
#endif
    
    // Helper to read tensor
    auto read_tensor = [&](std::vector<int64_t> shape) {
        Tensor t;
        t.shape = shape;
        int64_t n = t.numel();
        
#ifdef USE_CUDA
        // Allocate on GPU
        float* h_data = new float[n];
        file.read(reinterpret_cast<char*>(h_data), n * sizeof(float));
        
        cudaMalloc(&t.data, n * sizeof(float));
        cudaMemcpy(t.data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_data;
        
        t.owns_memory = true;
#else
        // CPU/MPS: allocate on host
        t.data = new float[n];
        t.owns_memory = true;
        file.read(reinterpret_cast<char*>(t.data), n * sizeof(float));
#endif
        
        return t;
    };
    
    // Load embeddings
    weights_.embed_tokens = read_tensor({config_.vocab_size, config_.hidden_size});
    
    // Load layers
    weights_.layers.resize(config_.num_layers);
    int32_t head_dim = config_.hidden_size / config_.num_heads;
    
    for (int32_t i = 0; i < config_.num_layers; ++i) {
        auto& layer = weights_.layers[i];
        
        // Attention
        layer.attn_q = read_tensor({config_.hidden_size, config_.hidden_size});
        layer.attn_k = read_tensor({config_.hidden_size, config_.num_kv_heads * head_dim});
        layer.attn_v = read_tensor({config_.hidden_size, config_.num_kv_heads * head_dim});
        layer.attn_o = read_tensor({config_.hidden_size, config_.hidden_size});
        layer.attn_norm_weight = read_tensor({config_.hidden_size});
        
        // FFN
        layer.is_moe = config_.moe_layers[i];
        
        if (layer.is_moe) {
            // MoE layer
            layer.moe_gate_weight = read_tensor({config_.hidden_size, config_.num_experts});
            
            layer.expert_gate.resize(config_.num_experts);
            layer.expert_up.resize(config_.num_experts);
            layer.expert_down.resize(config_.num_experts);
            
            for (int32_t e = 0; e < config_.num_experts; ++e) {
                layer.expert_gate[e] = read_tensor({config_.hidden_size, config_.intermediate_size});
                layer.expert_up[e] = read_tensor({config_.hidden_size, config_.intermediate_size});
                layer.expert_down[e] = read_tensor({config_.intermediate_size, config_.hidden_size});
            }
        } else {
            // Dense FFN
            layer.ffn_gate = read_tensor({config_.hidden_size, config_.intermediate_size});
            layer.ffn_up = read_tensor({config_.hidden_size, config_.intermediate_size});
            layer.ffn_down = read_tensor({config_.intermediate_size, config_.hidden_size});
            
            // MoD router if enabled
            if (config_.mod_layers[i]) {
                layer.mod_router_weight = read_tensor({config_.hidden_size, 1});
                file.read(reinterpret_cast<char*>(&layer.mod_threshold), sizeof(float));
            }
        }
        
        layer.ffn_norm_weight = read_tensor({config_.hidden_size});
    }
    
    // Final
    weights_.final_norm_weight = read_tensor({config_.hidden_size});
    weights_.lm_head = read_tensor({config_.vocab_size, config_.hidden_size});
    
    // RoPE cache
    weights_.rope_cos = read_tensor({config_.max_seq_len, head_dim});
    weights_.rope_sin = read_tensor({config_.max_seq_len, head_dim});
    
    std::cout << "✓ Model loaded: " << path << std::endl;
}

void InferenceEngine::init_buffers() {
    // Pre-allocate activation buffers
    buffers_.resize(10);  // Generous buffer count
    
    int32_t max_batch_seq = config_.max_seq_len;
    int32_t hidden = config_.hidden_size;
    int32_t intermediate = config_.intermediate_size;
    
    // Common buffer shapes
    for (int i = 0; i < 5; ++i) {
        buffers_[i].shape = {1, max_batch_seq, hidden};
        size_t bytes = max_batch_seq * hidden * sizeof(float);
        
#ifdef USE_CUDA
        cudaMalloc(&buffers_[i].data, bytes);
#else
        buffers_[i].data = new float[max_batch_seq * hidden];
#endif
        buffers_[i].owns_memory = true;
    }
    
    // Intermediate buffers for FFN
    for (int i = 5; i < 10; ++i) {
        buffers_[i].shape = {1, max_batch_seq, intermediate};
        size_t bytes = max_batch_seq * intermediate * sizeof(float);
        
#ifdef USE_CUDA
        cudaMalloc(&buffers_[i].data, bytes);
#else
        buffers_[i].data = new float[max_batch_seq * intermediate];
#endif
        buffers_[i].owns_memory = true;
    }
    
    // KV cache
    int32_t head_dim = hidden / config_.num_heads;
    kv_cache_k_.resize(config_.num_layers);
    kv_cache_v_.resize(config_.num_layers);
    
    for (int32_t i = 0; i < config_.num_layers; ++i) {
        kv_cache_k_[i].shape = {1, config_.num_kv_heads, config_.max_seq_len, head_dim};
        size_t bytes = config_.num_kv_heads * config_.max_seq_len * head_dim * sizeof(float);
        
#ifdef USE_CUDA
        cudaMalloc(&kv_cache_k_[i].data, bytes);
        cudaMalloc(&kv_cache_v_[i].data, bytes);
#else
        kv_cache_k_[i].data = new float[config_.num_kv_heads * config_.max_seq_len * head_dim];
        kv_cache_v_[i].data = new float[config_.num_kv_heads * config_.max_seq_len * head_dim];
#endif
        
        kv_cache_k_[i].owns_memory = true;
        
        kv_cache_v_[i].shape = {1, config_.num_kv_heads, config_.max_seq_len, head_dim};
        kv_cache_v_[i].owns_memory = true;
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

void InferenceEngine::forward(
    const int32_t* input_ids,
    int32_t batch_size,
    int32_t seq_len,
    float* logits_out
) {
    // Simple single-batch inference for now
    assert(batch_size == 1);
    
    Tensor& x = buffers_[0];
    Tensor& residual = buffers_[1];
    Tensor& attn_out = buffers_[2];
    Tensor& ffn_out = buffers_[3];
    
    // Embedding
    embedding(input_ids, seq_len, x);
    
    // Transformer layers
    for (int32_t i = 0; i < config_.num_layers; ++i) {
        // Save residual
        std::memcpy(residual.data, x.data, seq_len * config_.hidden_size * sizeof(float));
        
        // Attention
        attention_layer(i, x, attn_out);
        
        // Residual
        kernels::add_inplace(x.data, attn_out.data, seq_len * config_.hidden_size);
        
        // Save residual again
        std::memcpy(residual.data, x.data, seq_len * config_.hidden_size * sizeof(float));
        
        // FFN (MoE or dense with MoD)
        if (weights_.layers[i].is_moe) {
            moe_layer(i, x, ffn_out);
        } else if (config_.mod_layers[i]) {
            mod_layer(i, x, ffn_out);
        } else {
            ffn_layer(i, x, ffn_out);
        }
        
        // Residual
        kernels::add_inplace(x.data, ffn_out.data, seq_len * config_.hidden_size);
    }
    
    // Final norm
    rms_norm(x, weights_.final_norm_weight, x);
    
    // LM head
    linear(x, weights_.lm_head, x);
    
    // Copy logits
    std::memcpy(logits_out, x.data, seq_len * config_.vocab_size * sizeof(float));
}

// ============================================================================
// LAYER OPERATIONS (delegated to backend kernels)
// ============================================================================

void InferenceEngine::embedding(const int32_t* ids, int32_t n, Tensor& out) {
#ifdef USE_CUDA
    // Copy input IDs to device
    int32_t* d_ids;
    cudaMalloc(&d_ids, n * sizeof(int32_t));
    cudaMemcpy(d_ids, ids, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // Gather embeddings on GPU (simplified - use a kernel for production)
    float* h_out = new float[n * config_.hidden_size];
    float* embed_cpu = new float[weights_.embed_tokens.numel()];
    cudaMemcpy(embed_cpu, weights_.embed_tokens.data, 
               weights_.embed_tokens.numel() * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int32_t i = 0; i < n; ++i) {
        int32_t id = ids[i];
        std::memcpy(h_out + i * config_.hidden_size,
                    embed_cpu + id * config_.hidden_size,
                    config_.hidden_size * sizeof(float));
    }
    
    cudaMemcpy(out.data, h_out, n * config_.hidden_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    delete[] h_out;
    delete[] embed_cpu;
    cudaFree(d_ids);
#else
    // CPU/MPS path
    for (int32_t i = 0; i < n; ++i) {
        int32_t id = ids[i];
        float* src = static_cast<float*>(weights_.embed_tokens.data) + id * config_.hidden_size;
        float* dst = static_cast<float*>(out.data) + i * config_.hidden_size;
        std::memcpy(dst, src, config_.hidden_size * sizeof(float));
    }
#endif
}

void InferenceEngine::rms_norm(const Tensor& x, const Tensor& w, Tensor& out) {
    kernels::rms_norm(
        static_cast<const float*>(x.data), 
        static_cast<const float*>(w.data), 
        static_cast<float*>(out.data), 
        x.shape[1], config_.hidden_size, 1e-6f
    );
}

void InferenceEngine::linear(const Tensor& x, const Tensor& w, Tensor& out) {
    int32_t seq_len = x.shape[1];
    int32_t in_features = x.shape[2];
    int32_t out_features = w.shape[0];
    
    kernels::matmul(
        static_cast<const float*>(x.data), 
        static_cast<const float*>(w.data), 
        static_cast<float*>(out.data),
        seq_len, in_features, out_features
    );
}

void InferenceEngine::attention_layer(int32_t layer_idx, const Tensor& x, Tensor& out) {
    auto& layer = weights_.layers[layer_idx];
    
    // Simplified attention - full implementation in kernels
    Tensor& normed = buffers_[4];
    rms_norm(x, layer.attn_norm_weight, normed);
    
    // Q, K, V projections
    Tensor q, k, v;
    linear(normed, layer.attn_q, q);
    linear(normed, layer.attn_k, k);
    linear(normed, layer.attn_v, v);
    
    // Apply RoPE
    kernels::apply_rope(q.data, k.data, weights_.rope_cos.data, weights_.rope_sin.data,
                        x.shape[1], config_.num_heads, config_.num_kv_heads, 
                        config_.hidden_size / config_.num_heads, cache_pos_);
    
    // Attention + KV cache
    kernels::attention(q.data, k.data, v.data, out.data,
                       kv_cache_k_[layer_idx].data, kv_cache_v_[layer_idx].data,
                       x.shape[1], config_.num_heads, config_.num_kv_heads,
                       config_.hidden_size / config_.num_heads, cache_pos_);
    
    // Output projection
    linear(out, layer.attn_o, out);
}

void InferenceEngine::ffn_layer(int32_t layer_idx, const Tensor& x, Tensor& out) {
    auto& layer = weights_.layers[layer_idx];
    
    Tensor& normed = buffers_[4];
    Tensor& gate_out = buffers_[5];
    Tensor& up_out = buffers_[6];
    
    rms_norm(x, layer.ffn_norm_weight, normed);
    
    // SwiGLU: down(silu(gate(x)) * up(x))
    linear(normed, layer.ffn_gate, gate_out);
    linear(normed, layer.ffn_up, up_out);
    
    kernels::silu(gate_out.data, gate_out.data, 
                  x.shape[1] * config_.intermediate_size);
    
    kernels::mul(gate_out.data, up_out.data, gate_out.data,
                 x.shape[1] * config_.intermediate_size);
    
    linear(gate_out, layer.ffn_down, out);
}

void InferenceEngine::moe_layer(int32_t layer_idx, const Tensor& x, Tensor& out) {
    auto& layer = weights_.layers[layer_idx];
    int32_t seq_len = x.shape[1];
    
    Tensor& normed = buffers_[4];
    rms_norm(x, layer.ffn_norm_weight, normed);
    
    // Route tokens to experts
    std::vector<int32_t> expert_ids(seq_len * config_.moe_top_k);
    std::vector<float> expert_weights(seq_len * config_.moe_top_k);
    
    kernels::moe_route(normed.data, layer.moe_gate_weight.data,
                       expert_ids.data(), expert_weights.data(),
                       seq_len, config_.hidden_size, config_.num_experts, config_.moe_top_k);
    
    // Execute experts
    std::memset(out.data, 0, seq_len * config_.hidden_size * sizeof(float));
    
    for (int32_t t = 0; t < seq_len; ++t) {
        for (int32_t k = 0; k < config_.moe_top_k; ++k) {
            int32_t expert_id = expert_ids[t * config_.moe_top_k + k];
            float weight = expert_weights[t * config_.moe_top_k + k];
            
            float* token_in = normed.data + t * config_.hidden_size;
            float* token_out = buffers_[5].data;
            
            // SwiGLU through expert
            kernels::expert_forward(token_in, token_out,
                                    layer.expert_gate[expert_id].data,
                                    layer.expert_up[expert_id].data,
                                    layer.expert_down[expert_id].data,
                                    config_.hidden_size, config_.intermediate_size);
            
            // Accumulate weighted output
            kernels::add_scaled(out.data + t * config_.hidden_size, token_out,
                                weight, config_.hidden_size);
        }
    }
}

void InferenceEngine::mod_layer(int32_t layer_idx, const Tensor& x, Tensor& out) {
    auto& layer = weights_.layers[layer_idx];
    int32_t seq_len = x.shape[1];
    
    Tensor& normed = buffers_[4];
    rms_norm(x, layer.ffn_norm_weight, normed);
    
    // Compute token importance scores
    std::vector<float> scores(seq_len);
    kernels::mod_route(normed.data, layer.mod_router_weight.data,
                       scores.data(), seq_len, config_.hidden_size);
    
    // Select tokens above threshold
    std::vector<int32_t> compute_mask(seq_len);
    int32_t num_compute = 0;
    
    for (int32_t t = 0; t < seq_len; ++t) {
        if (scores[t] >= layer.mod_threshold) {
            compute_mask[num_compute++] = t;
        }
    }
    
    // Execute FFN only on selected tokens
    std::memset(out.data, 0, seq_len * config_.hidden_size * sizeof(float));
    
    for (int32_t i = 0; i < num_compute; ++i) {
        int32_t t = compute_mask[i];
        float* token_in = normed.data + t * config_.hidden_size;
        float* token_out = out.data + t * config_.hidden_size;
        
        kernels::expert_forward(token_in, token_out,
                                layer.ffn_gate.data,
                                layer.ffn_up.data,
                                layer.ffn_down.data,
                                config_.hidden_size, config_.intermediate_size);
    }
}

// ============================================================================
// GENERATION
// ============================================================================

std::vector<int32_t> InferenceEngine::generate(
    const std::vector<int32_t>& prompt,
    int32_t max_tokens,
    float temperature
) {
    std::vector<int32_t> output = prompt;
    std::vector<float> logits(config_.vocab_size);
    
    reset_cache();
    
    for (int32_t i = 0; i < max_tokens; ++i) {
        // Forward pass
        forward(output.data(), 1, output.size(), logits.data());
        
        // Sample next token
        int32_t next_token = kernels::sample(logits.data(), config_.vocab_size, temperature);
        output.push_back(next_token);
        
        cache_pos_++;
        
        // Stop on EOS
        if (next_token == 0) break;
    }
    
    return output;
}

void InferenceEngine::reset_cache() {
    cache_pos_ = 0;
    for (auto& cache : kv_cache_k_) {
        std::memset(cache.data, 0, cache.numel() * sizeof(float));
    }
    for (auto& cache : kv_cache_v_) {
        std::memset(cache.data, 0, cache.numel() * sizeof(float));
    }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.bin> <prompt>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string prompt = argv[2];
    
    try {
#ifdef USE_CUDA
        std::cout << "🚀 CUDA Inference Mode" << std::endl;
#elif defined(USE_MPS)
        std::cout << "🚀 Apple MPS Inference Mode" << std::endl;
#else
        std::cout << "🚀 CPU Inference Mode" << std::endl;
#endif
        
        InferenceEngine engine(model_path);
        
        // Tokenize prompt (simplified - use proper tokenizer)
        std::vector<int32_t> input_ids = {1, 2, 3};  // Dummy tokens
        
        std::cout << "Generating..." << std::endl;
        auto output = engine.generate(input_ids, 100, 1.0f);
        
        std::cout << "Generated " << output.size() << " tokens" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}