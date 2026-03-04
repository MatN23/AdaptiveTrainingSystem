// core/runtime.cpp
// Model loading and execution orchestration

#include "runtime.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace {

inline void tensor_copy(float* dst, const float* src, size_t count) {
#ifdef USE_CUDA
  cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice);
#else
  std::memcpy(dst, src, count * sizeof(float));
#endif
}

inline void tensor_zero(float* dst, size_t count) {
#ifdef USE_CUDA
  cudaMemset(dst, 0, count * sizeof(float));
#else
  std::memset(dst, 0, count * sizeof(float));
#endif
}

inline void tensor_to_host(float* dst_host, const float* src, size_t count) {
#ifdef USE_CUDA
  cudaMemcpy(dst_host, src, count * sizeof(float), cudaMemcpyDeviceToHost);
#else
  std::memcpy(dst_host, src, count * sizeof(float));
#endif
}

} // namespace

// ============================================================================
// MODEL LOADING
// ============================================================================

InferenceEngine::InferenceEngine(const std::string &model_path) {
  load_model(model_path);
  init_buffers();
}

InferenceEngine::~InferenceEngine() {
  // Cleanup handled by Tensor destructors
}

void InferenceEngine::load_model(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open model: " + path);
  }

  // Read magic and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));

  if (magic != 0x4D4F4549) {
    throw std::runtime_error("Invalid model file (bad magic)");
  }

  if (version != 1) {
    throw std::runtime_error("Unsupported model version");
  }

  // Read config fields individually (not as a block!)
  file.read(reinterpret_cast<char *>(&config_.vocab_size), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.hidden_size), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.num_layers), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.num_heads), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.num_kv_heads), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.intermediate_size),
            sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.max_seq_len), sizeof(int32_t));

  file.read(reinterpret_cast<char *>(&config_.use_moe), sizeof(bool));
  file.read(reinterpret_cast<char *>(&config_.num_experts), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&config_.moe_top_k), sizeof(int32_t));

  // Read moe_layers vector
  int32_t moe_layers_size;
  file.read(reinterpret_cast<char *>(&moe_layers_size), sizeof(int32_t));
  config_.moe_layers.resize(moe_layers_size);
  for (int32_t i = 0; i < moe_layers_size; ++i) {
    bool val;
    file.read(reinterpret_cast<char *>(&val), sizeof(bool));
    config_.moe_layers[i] = val;
  }

  file.read(reinterpret_cast<char *>(&config_.use_mod), sizeof(bool));
  file.read(reinterpret_cast<char *>(&config_.mod_capacity), sizeof(float));

  // Read mod_layers vector
  int32_t mod_layers_size;
  file.read(reinterpret_cast<char *>(&mod_layers_size), sizeof(int32_t));
  config_.mod_layers.resize(mod_layers_size);
  for (int32_t i = 0; i < mod_layers_size; ++i) {
    bool val;
    file.read(reinterpret_cast<char *>(&val), sizeof(bool));
    config_.mod_layers[i] = val;
  }

  file.read(reinterpret_cast<char *>(&config_.rope_theta), sizeof(float));

#ifdef USE_CUDA
  std::cout << "✓ Loaded config (CUDA mode): " << config_.num_layers
            << " layers, " << config_.hidden_size << " hidden" << std::endl;
#elif defined(USE_MPS)
  std::cout << "✓ Loaded config (MPS mode): " << config_.num_layers
            << " layers, " << config_.hidden_size << " hidden" << std::endl;
#else
  std::cout << "✓ Loaded config (CPU mode): " << config_.num_layers
            << " layers, " << config_.hidden_size << " hidden" << std::endl;
#endif

  // Helper to read tensor
  auto read_tensor = [&](std::vector<int64_t> shape) {
    Tensor t;
    t.shape = shape;
    int64_t n = t.numel();

#ifdef USE_CUDA
    // Allocate on GPU
    float *h_data = new float[n];
    file.read(reinterpret_cast<char *>(h_data), n * sizeof(float));

    cudaMalloc(&t.data, n * sizeof(float));
    cudaMemcpy(t.data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_data;

    t.owns_memory = true;
#else
    // CPU/MPS: allocate on host
    t.data = new float[n];
    t.owns_memory = true;
    file.read(reinterpret_cast<char *>(t.data), n * sizeof(float));
#endif

    return t;
  };

  // Load embeddings
  weights_.embed_tokens =
      read_tensor({config_.vocab_size, config_.hidden_size});

  // Load layers
  weights_.layers.resize(config_.num_layers);
  int32_t head_dim = config_.hidden_size / config_.num_heads;

  for (int32_t i = 0; i < config_.num_layers; ++i) {
    auto &layer = weights_.layers[i];

    // Attention
    layer.attn_q = read_tensor({config_.hidden_size, config_.hidden_size});
    layer.attn_k =
        read_tensor({config_.hidden_size, config_.num_kv_heads * head_dim});
    layer.attn_v =
        read_tensor({config_.hidden_size, config_.num_kv_heads * head_dim});
    layer.attn_o = read_tensor({config_.hidden_size, config_.hidden_size});
    layer.attn_norm_weight = read_tensor({config_.hidden_size});

    // FFN
    layer.is_moe = config_.moe_layers[i];

    if (layer.is_moe) {
      // MoE layer
      layer.moe_gate_weight =
          read_tensor({config_.hidden_size, config_.num_experts});

      layer.expert_gate.resize(config_.num_experts);
      layer.expert_up.resize(config_.num_experts);
      layer.expert_down.resize(config_.num_experts);

      for (int32_t e = 0; e < config_.num_experts; ++e) {
        layer.expert_gate[e] =
            read_tensor({config_.hidden_size, config_.intermediate_size});
        layer.expert_up[e] =
            read_tensor({config_.hidden_size, config_.intermediate_size});
        layer.expert_down[e] =
            read_tensor({config_.intermediate_size, config_.hidden_size});
      }
    } else {
      // Dense FFN
      layer.ffn_gate =
          read_tensor({config_.hidden_size, config_.intermediate_size});
      layer.ffn_up =
          read_tensor({config_.hidden_size, config_.intermediate_size});
      layer.ffn_down =
          read_tensor({config_.intermediate_size, config_.hidden_size});

      // MoD router if enabled
      if (config_.mod_layers[i]) {
        layer.mod_router_weight = read_tensor({config_.hidden_size, 1});
        file.read(reinterpret_cast<char *>(&layer.mod_threshold),
                  sizeof(float));
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
  buffers_.resize(12); // Increased for q,k,v buffers

  int32_t max_batch_seq = config_.max_seq_len;
  int32_t hidden = config_.hidden_size;
  int32_t intermediate = config_.intermediate_size;

  // Common buffer shapes (x, residual, attn_out, ffn_out, normed, q, k, v)
  for (int i = 0; i < 8; ++i) {
    buffers_[i].shape = {1, max_batch_seq, hidden};
    size_t bytes = max_batch_seq * hidden * sizeof(float);

#ifdef USE_CUDA
    cudaMalloc(&buffers_[i].data, bytes);
#else
    buffers_[i].data = new float[max_batch_seq * hidden];
#endif
    buffers_[i].owns_memory = true;
  }

  // Intermediate buffers for FFN (gate_out, up_out, temp, temp2)
  for (int i = 8; i < 12; ++i) {
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
    kv_cache_k_[i].shape = {1, config_.num_kv_heads, config_.max_seq_len,
                            head_dim};
    size_t bytes =
        config_.num_kv_heads * config_.max_seq_len * head_dim * sizeof(float);

#ifdef USE_CUDA
    cudaMalloc(&kv_cache_k_[i].data, bytes);
    cudaMalloc(&kv_cache_v_[i].data, bytes);
#else
    kv_cache_k_[i].data =
        new float[config_.num_kv_heads * config_.max_seq_len * head_dim];
    kv_cache_v_[i].data =
        new float[config_.num_kv_heads * config_.max_seq_len * head_dim];
#endif

    kv_cache_k_[i].owns_memory = true;

    kv_cache_v_[i].shape = {1, config_.num_kv_heads, config_.max_seq_len,
                            head_dim};
    kv_cache_v_[i].owns_memory = true;
  }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

void InferenceEngine::forward(const int32_t *input_ids, int32_t batch_size,
                              int32_t seq_len, float *logits_out) {
  // Simple single-batch inference for now
  assert(batch_size == 1);

  // Update buffer shapes to match current sequence length
  // This is CRITICAL: otherwise layers process garbage data up to max_seq_len
  for (auto &buf : buffers_) {
    if (buf.shape.size() >= 2) {
      buf.shape[1] = seq_len;
    }
  }

  Tensor &x = buffers_[0];
  Tensor &residual = buffers_[1];
  Tensor &attn_out = buffers_[2];
  Tensor &ffn_out = buffers_[3];

  // Embedding
  embedding(input_ids, seq_len, x);

  // Transformer layers
  for (int32_t i = 0; i < config_.num_layers; ++i) {
    // Save residual
    tensor_copy(residual.data, x.data, seq_len * config_.hidden_size);

    // Attention
    attention_layer(i, x, attn_out);

    // Residual
    kernels::add_inplace(x.data, attn_out.data, seq_len * config_.hidden_size);

    // Save residual again
    tensor_copy(residual.data, x.data, seq_len * config_.hidden_size);

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
  tensor_to_host(logits_out, x.data, seq_len * config_.vocab_size);
}

// ============================================================================
// LAYER OPERATIONS (delegated to backend kernels)
// ============================================================================

void InferenceEngine::embedding(const int32_t *ids, int32_t n, Tensor &out) {
#ifdef USE_CUDA
  // Gather directly on device via row-wise D2D copies.
  const float *embed = static_cast<const float *>(weights_.embed_tokens.data);
  float *dst = static_cast<float *>(out.data);
  for (int32_t i = 0; i < n; ++i) {
    int32_t id = ids[i];
    cudaMemcpy(dst + (int64_t)i * config_.hidden_size,
               embed + (int64_t)id * config_.hidden_size,
               config_.hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
  }
#else
  // CPU/MPS path
  for (int32_t i = 0; i < n; ++i) {
    int32_t id = ids[i];
    float *src = static_cast<float *>(weights_.embed_tokens.data) +
                 id * config_.hidden_size;
    float *dst = static_cast<float *>(out.data) + i * config_.hidden_size;
    std::memcpy(dst, src, config_.hidden_size * sizeof(float));
  }
#endif
}

void InferenceEngine::rms_norm(const Tensor &x, const Tensor &w, Tensor &out) {
  kernels::rms_norm(
      static_cast<const float *>(x.data), static_cast<const float *>(w.data),
      static_cast<float *>(out.data), x.shape[1], config_.hidden_size, 1e-6f);
}

void InferenceEngine::linear(const Tensor &x, const Tensor &w, Tensor &out) {
  int32_t seq_len = x.shape[1];
  int32_t in_features = x.shape[2];
  int32_t out_features = w.shape[0];

  kernels::matmul(
      static_cast<const float *>(x.data), static_cast<const float *>(w.data),
      static_cast<float *>(out.data), seq_len, in_features, out_features);
}

void InferenceEngine::attention_layer(int32_t layer_idx, const Tensor &x,
                                      Tensor &out) {
  auto &layer = weights_.layers[layer_idx];

  // Use pre-allocated buffers
  Tensor &normed = buffers_[4];
  Tensor &q = buffers_[5];
  Tensor &k = buffers_[6];
  Tensor &v = buffers_[7];

  rms_norm(x, layer.attn_norm_weight, normed);

  // Q, K, V projections
  linear(normed, layer.attn_q, q);
  linear(normed, layer.attn_k, k);
  linear(normed, layer.attn_v, v);

  // Apply RoPE
  int32_t head_dim = config_.hidden_size / config_.num_heads;
  kernels::apply_rope(
      q.data, k.data, static_cast<const float *>(weights_.rope_cos.data),
      static_cast<const float *>(weights_.rope_sin.data), x.shape[1],
      config_.num_heads, config_.num_kv_heads, head_dim, cache_pos_);

  // Attention + KV cache
  kernels::attention(q.data, k.data, v.data, out.data,
                     kv_cache_k_[layer_idx].data, kv_cache_v_[layer_idx].data,
                     x.shape[1], config_.num_heads, config_.num_kv_heads,
                     head_dim, cache_pos_);

  // Output projection
  linear(out, layer.attn_o, out);
}

void InferenceEngine::ffn_layer(int32_t layer_idx, const Tensor &x,
                                Tensor &out) {
  auto &layer = weights_.layers[layer_idx];

  Tensor &normed = buffers_[4];
  Tensor &gate_out = buffers_[8];
  Tensor &up_out = buffers_[9];

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

void InferenceEngine::moe_layer(int32_t layer_idx, const Tensor &x,
                                Tensor &out) {
  auto &layer = weights_.layers[layer_idx];
  int32_t seq_len = x.shape[1];

  Tensor &normed = buffers_[4];
  rms_norm(x, layer.ffn_norm_weight, normed);

  // Route tokens to experts
  std::vector<int32_t> expert_ids(seq_len * config_.moe_top_k);
  std::vector<float> expert_weights(seq_len * config_.moe_top_k);

  kernels::moe_route(normed.data, layer.moe_gate_weight.data, expert_ids.data(),
                     expert_weights.data(), seq_len, config_.hidden_size,
                     config_.num_experts, config_.moe_top_k);

  // Execute experts
  tensor_zero(out.data, seq_len * config_.hidden_size);

  // Use buffer for expert output
  Tensor &expert_out = buffers_[8];

  for (int32_t t = 0; t < seq_len; ++t) {
    for (int32_t k = 0; k < config_.moe_top_k; ++k) {
      int32_t expert_id = expert_ids[t * config_.moe_top_k + k];
      float weight = expert_weights[t * config_.moe_top_k + k];

      float *token_in = normed.data + t * config_.hidden_size;

      // SwiGLU through expert
      kernels::expert_forward(
          token_in, expert_out.data, layer.expert_gate[expert_id].data,
          layer.expert_up[expert_id].data, layer.expert_down[expert_id].data,
          config_.hidden_size, config_.intermediate_size);

      // Accumulate weighted output
      kernels::add_scaled(out.data + t * config_.hidden_size, expert_out.data,
                          weight, config_.hidden_size);
    }
  }
}

void InferenceEngine::mod_layer(int32_t layer_idx, const Tensor &x,
                                Tensor &out) {
  auto &layer = weights_.layers[layer_idx];
  int32_t seq_len = x.shape[1];

  Tensor &normed = buffers_[4];
  rms_norm(x, layer.ffn_norm_weight, normed);

  // Compute token importance scores
  std::vector<float> scores(seq_len);
  kernels::mod_route(normed.data, layer.mod_router_weight.data, scores.data(),
                     seq_len, config_.hidden_size);

  // Select tokens above threshold
  std::vector<int32_t> compute_mask(seq_len);
  int32_t num_compute = 0;

  for (int32_t t = 0; t < seq_len; ++t) {
    if (scores[t] >= layer.mod_threshold) {
      compute_mask[num_compute++] = t;
    }
  }

  // Execute FFN only on selected tokens
  tensor_zero(out.data, seq_len * config_.hidden_size);

  for (int32_t i = 0; i < num_compute; ++i) {
    int32_t t = compute_mask[i];
    float *token_in = normed.data + t * config_.hidden_size;
    float *token_out = out.data + t * config_.hidden_size;

    kernels::expert_forward(token_in, token_out, layer.ffn_gate.data,
                            layer.ffn_up.data, layer.ffn_down.data,
                            config_.hidden_size, config_.intermediate_size);
  }
}

// ============================================================================
// GENERATION
// ============================================================================

std::vector<int32_t>
InferenceEngine::generate(const std::vector<int32_t> &prompt,
                          int32_t max_tokens, float temperature) {
  std::vector<int32_t> output = prompt;

  // Allocate buffer for all positions' logits
  std::vector<float> all_logits(config_.max_seq_len * config_.vocab_size);
  std::vector<float> last_logits(config_.vocab_size);

  reset_cache();

  for (int32_t i = 0; i < max_tokens; ++i) {
    // Forward pass - outputs logits for all positions
    forward(output.data(), 1, output.size(), all_logits.data());

    // Extract logits for the LAST token position only
    int32_t last_pos = output.size() - 1;
    std::memcpy(last_logits.data(),
                all_logits.data() + last_pos * config_.vocab_size,
                config_.vocab_size * sizeof(float));

    // Sample next token from the LAST position's logits
    int32_t next_token =
        kernels::sample(last_logits.data(), config_.vocab_size, temperature);
    output.push_back(next_token);

    cache_pos_++;

    // Stop on EOS tokens (tiktoken gpt-4: 100257, gpt-2: 50256, common: 0, 1,
    // 2)
    if (next_token == 100257 || next_token == 50256 || next_token == 0 ||
        next_token == 1 || next_token == 2) {
      break;
    }
  }

  return output;
}

void InferenceEngine::reset_cache() {
  cache_pos_ = 0;
  for (auto &cache : kv_cache_k_) {
    tensor_zero(cache.data, cache.numel());
  }
  for (auto &cache : kv_cache_v_) {
    tensor_zero(cache.data, cache.numel());
  }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

std::vector<int32_t> read_token_file(const std::string &path) {
  std::vector<int32_t> tokens;
  std::ifstream file(path);

  if (!file) {
    std::cerr << "Warning: Could not open token file: " << path << std::endl;
    return {1, 2, 3}; // Fallback to dummy tokens
  }

  std::string content;
  std::getline(file, content);

  // Parse comma-separated integers
  std::stringstream ss(content);
  std::string token_str;

  while (std::getline(ss, token_str, ',')) {
    try {
      // Trim whitespace
      token_str.erase(0, token_str.find_first_not_of(" \t\n\r"));
      token_str.erase(token_str.find_last_not_of(" \t\n\r") + 1);

      if (!token_str.empty()) {
        int32_t token = std::stoi(token_str);
        tokens.push_back(token);
      }
    } catch (...) {
      std::cerr << "Warning: Failed to parse token: '" << token_str << "'"
                << std::endl;
    }
  }

  if (tokens.empty()) {
    std::cerr << "Warning: No valid tokens found, using dummy tokens"
              << std::endl;
    return {1, 2, 3};
  }

  return tokens;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <model.bin> [tokens_file|prompt]"
              << std::endl;
    std::cout << "\nModes:" << std::endl;
    std::cout << "  Interactive: " << argv[0] << " model.bin" << std::endl;
    std::cout << "  Token file:  " << argv[0] << " model.bin tokens.txt"
              << std::endl;
    std::cout << "  Text prompt: " << argv[0] << " model.bin \"Hello!\""
              << std::endl;
    std::cout
        << "\nNote: For proper tokenization, use chat_integrated.py wrapper"
        << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  bool interactive = (argc == 2);

  try {
#ifdef USE_CUDA
    std::cout << "🚀 CUDA Inference Mode" << std::endl;
#elif defined(USE_MPS)
    std::cout << "🚀 Apple MPS Inference Mode" << std::endl;
#else
    std::cout << "🚀 CPU Inference Mode" << std::endl;
#endif

    InferenceEngine engine(model_path);

    if (interactive) {
      // Interactive chat mode
      std::cout << "\n💬 Interactive Chat Mode" << std::endl;
      std::cout << "Type token file path (or 'quit' to exit)\n" << std::endl;

      std::string line;
      while (true) {
        std::cout << "Token file: ";
        std::getline(std::cin, line);

        if (line.empty())
          continue;
        if (line == "quit" || line == "exit" || line == "q")
          break;

        // Read tokens from file
        std::vector<int32_t> input_ids = read_token_file(line);

        std::cout << "Loaded " << input_ids.size() << " tokens" << std::endl;
        std::cout << "Generating..." << std::endl;

        auto output = engine.generate(input_ids, 100, 1.0f);

        // Print token IDs for Python to decode
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < output.size(); ++i) {
          std::cout << output[i];
          if (i < output.size() - 1)
            std::cout << ", ";
        }
        std::cout << std::endl << std::endl;
      }

      std::cout << "Goodbye!" << std::endl;

    } else {
      // Single prompt/file mode
      std::string input_arg = argv[2];
      std::vector<int32_t> input_ids;

      // Check if it's a file or text prompt
      std::ifstream test_file(input_arg);
      if (test_file.good()) {
        test_file.close();
        // It's a file
        input_ids = read_token_file(input_arg);
        std::cout << "Loaded " << input_ids.size() << " tokens from file"
                  << std::endl;
      } else {
        // It's a text prompt - use dummy tokens
        std::cout << "Text prompt (no tokenizer): using dummy tokens"
                  << std::endl;
        std::cout
            << "⚠️  For proper tokenization, use: python chat_integrated.py"
            << std::endl;
        input_ids = {1, 2, 3};
      }

      std::cout << "Generating..." << std::endl;

      auto output = engine.generate(input_ids, 100, 1.0f);

      std::cout << "\n✓ Generated " << output.size() << " tokens" << std::endl;
      std::cout << "\nToken IDs: ";
      for (size_t i = 0; i < output.size(); ++i) {
        std::cout << output[i];
        if (i < output.size() - 1)
          std::cout << ", ";
        if (i > 0 && i % 20 == 0)
          std::cout << "\n           ";
      }
      std::cout << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
