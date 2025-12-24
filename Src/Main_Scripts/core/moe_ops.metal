#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MoE OPERATIONS - 2-4x FASTER
// ============================================================================

// ============================================================================
// 1. TOP-K GATING - Optimized for Metal
// ============================================================================

kernel void topk_gating(
    device const float* gate_logits [[buffer(0)]],
    device int* top_k_indices [[buffer(1)]],
    device float* top_k_weights [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& temperature [[buffer(6)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (token_idx >= num_tokens) return;
    
    device const float* token_logits = gate_logits + token_idx * num_experts;
    
    // Each thread maintains local top-k
    threadgroup float shared_vals[256];
    threadgroup int shared_idxs[256];
    
    float local_vals[4];
    int local_idxs[4];
    
    // Initialize
    for (uint i = 0; i < k && i < 4; i++) {
        local_vals[i] = -INFINITY;
        local_idxs[i] = -1;
    }
    
    // Each thread processes subset of experts
    for (uint expert_id = tid; expert_id < num_experts; expert_id += 256) {
        float val = token_logits[expert_id] / temperature;
        
        // Insert into local top-k (optimized for k=2)
        if (k == 2) {
            if (val > local_vals[0]) {
                local_vals[1] = local_vals[0];
                local_idxs[1] = local_idxs[0];
                local_vals[0] = val;
                local_idxs[0] = expert_id;
            } else if (val > local_vals[1]) {
                local_vals[1] = val;
                local_idxs[1] = expert_id;
            }
        } else {
            // General k
            for (uint i = 0; i < k; i++) {
                if (val > local_vals[i]) {
                    // Shift down
                    for (uint j = k - 1; j > i; j--) {
                        local_vals[j] = local_vals[j - 1];
                        local_idxs[j] = local_idxs[j - 1];
                    }
                    local_vals[i] = val;
                    local_idxs[i] = expert_id;
                    break;
                }
            }
        }
    }
    
    // Store local results to shared memory
    for (uint i = 0; i < k; i++) {
        shared_vals[tid * 4 + i] = local_vals[i];
        shared_idxs[tid * 4 + i] = local_idxs[i];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Thread 0 merges all results
    if (tid == 0) {
        float final_vals[4];
        int final_idxs[4];
        
        for (uint i = 0; i < k; i++) {
            final_vals[i] = -INFINITY;
            final_idxs[i] = -1;
        }
        
        // Merge from all threads
        for (uint t = 0; t < 256; t++) {
            for (uint i = 0; i < k; i++) {
                float val = shared_vals[t * 4 + i];
                int idx = shared_idxs[t * 4 + i];
                
                if (idx == -1) continue;
                
                for (uint j = 0; j < k; j++) {
                    if (val > final_vals[j]) {
                        for (uint m = k - 1; m > j; m--) {
                            final_vals[m] = final_vals[m - 1];
                            final_idxs[m] = final_idxs[m - 1];
                        }
                        final_vals[j] = val;
                        final_idxs[j] = idx;
                        break;
                    }
                }
            }
        }
        
        // Compute softmax with numerical stability
        float max_logit = final_vals[0];
        float sum_exp = 0.0f;
        
        for (uint i = 0; i < k; i++) {
            float exp_val = exp(final_vals[i] - max_logit);
            final_vals[i] = exp_val;
            sum_exp += exp_val;
        }
        
        float inv_sum = 1.0f / sum_exp;
        
        // Write results
        device int* out_indices = top_k_indices + token_idx * k;
        device float* out_weights = top_k_weights + token_idx * k;
        
        for (uint i = 0; i < k; i++) {
            out_weights[i] = final_vals[i] * inv_sum;
            out_indices[i] = final_idxs[i];
        }
    }
}

// ============================================================================
// 2. DISPATCH TOKENS - Optimized memory access
// ============================================================================

kernel void dispatch_tokens(
    device const float* tokens [[buffer(0)]],
    device const int* top_k_indices [[buffer(1)]],
    device atomic_int* expert_positions [[buffer(2)]],
    device float* expert_inputs [[buffer(3)]],
    device int* token_map [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& capacity [[buffer(9)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (token_idx >= num_tokens) return;
    
    device const float* token_data = tokens + token_idx * hidden_dim;
    device const int* token_experts = top_k_indices + token_idx * k;
    
    threadgroup int shared_positions[8];
    
    // Thread 0 atomically reserves positions
    if (tid == 0) {
        for (uint i = 0; i < k; i++) {
            int expert_id = token_experts[i];
            if (expert_id >= 0 && expert_id < num_experts) {
                shared_positions[i] = atomic_fetch_add_explicit(
                    &expert_positions[expert_id], 1, 
                    memory_order_relaxed
                );
            } else {
                shared_positions[i] = -1;
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // All threads cooperate to copy data
    for (uint i = 0; i < k; i++) {
        int expert_id = token_experts[i];
        int pos = shared_positions[i];
        
        if (pos < 0 || pos >= capacity || expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        
        // Thread 0 updates token map
        if (tid == 0) {
            token_map[expert_id * capacity + pos] = token_idx * k + i;
        }
        
        // All threads copy data (vectorized)
        device float* expert_input = expert_inputs + (expert_id * capacity + pos) * hidden_dim;
        
        uint vec_dim = hidden_dim / 4;
        for (uint d = tid; d < vec_dim; d += 256) {
            ((device float4*)expert_input)[d] = ((device float4*)token_data)[d];
        }
        
        // Handle remainder
        for (uint d = vec_dim * 4 + tid; d < hidden_dim; d += 256) {
            expert_input[d] = token_data[d];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// 3. COMBINE EXPERT OUTPUTS - Optimized atomic operations
// ============================================================================

kernel void combine_expert_outputs(
    device const float* expert_outputs [[buffer(0)]],
    device const int* token_map [[buffer(1)]],
    device const float* top_k_weights [[buffer(2)]],
    device atomic_float* combined_output [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant uint& num_tokens [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint expert_id = gid.x;
    uint pos = gid.y;
    
    if (expert_id >= num_experts || pos >= capacity) return;
    
    int token_weight_idx = token_map[expert_id * capacity + pos];
    if (token_weight_idx < 0) return;
    
    int token_idx = token_weight_idx / k;
    if (token_idx >= num_tokens) return;
    
    float weight = top_k_weights[token_weight_idx];
    
    device const float* expert_out = expert_outputs + (expert_id * capacity + pos) * hidden_dim;
    
    // Vectorized atomic accumulation
    uint vec_dim = hidden_dim / 4;
    for (uint d = tid; d < vec_dim; d += 256) {
        float4 vec = ((device float4*)expert_out)[d];
        vec *= weight;
        
        // Metal doesn't have atomic float4, so we do component-wise
        atomic_fetch_add_explicit(&combined_output[token_idx * hidden_dim + d * 4 + 0], 
                                   vec.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&combined_output[token_idx * hidden_dim + d * 4 + 1], 
                                   vec.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&combined_output[token_idx * hidden_dim + d * 4 + 2], 
                                   vec.z, memory_order_relaxed);
        atomic_fetch_add_explicit(&combined_output[token_idx * hidden_dim + d * 4 + 3], 
                                   vec.w, memory_order_relaxed);
    }
    
    // Handle remainder
    for (uint d = vec_dim * 4 + tid; d < hidden_dim; d += 256) {
        float val = weight * expert_out[d];
        atomic_fetch_add_explicit(&combined_output[token_idx * hidden_dim + d], 
                                   val, memory_order_relaxed);
    }
}