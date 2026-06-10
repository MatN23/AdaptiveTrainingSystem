# Copyright (c) 2025 MatN23. All rights reserved.

import pytest
import torch
from Main_Scripts.core.unified_ops import (
    PyTorchRMSNorm,
    PyTorchRoPE,
    PyTorchSwiGLU,
    PyTorchMoEOps
)

class TestPyTorchFallbacks:
    """Test the PyTorch fallback implementations in unified_ops."""

    def test_pytorch_rmsnorm(self):
        norm = PyTorchRMSNorm(128)
        x = torch.randn(2, 64, 128)
        y = norm(x)
        assert y.shape == x.shape
        # Check that norm of output is close to sqrt(dim)
        # RMSNorm(x) = (x / RMS(x)) * weight
        # RMS(x) = sqrt(mean(x^2))
        # y^2 = (x^2 / mean(x^2)) * weight^2
        # mean(y^2) = mean(x^2) / mean(x^2) * weight^2 = weight^2
        assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 64), atol=1e-5)

    def test_pytorch_rope(self):
        rope = PyTorchRoPE(dim=32, max_seq_len=64)
        cos, sin = rope(seq_len=32, device=torch.device('cpu'))
        assert cos.shape == (32, 32)
        assert sin.shape == (32, 32)

    def test_pytorch_swiglu(self):
        swiglu = PyTorchSwiGLU(hidden_size=128, intermediate_size=512)
        x = torch.randn(2, 64, 128)
        y = swiglu(x)
        assert y.shape == (2, 64, 128)

    def test_pytorch_moe_ops(self):
        # Test topk_gating
        gate_logits = torch.randn(2, 10, 8) # batch, seq, experts
        indices, weights = PyTorchMoEOps.topk_gating(gate_logits, k=2)
        assert indices.shape == (2, 10, 2)
        assert weights.shape == (2, 10, 2)
        assert torch.allclose(weights.sum(-1), torch.ones(2, 10))

        # Test dispatch and combine (minimal)
        tokens = torch.randn(4, 16) # 4 tokens, 16 hidden
        indices = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 3]]) # 4 tokens, k=2
        num_experts = 4
        capacity = 4
        
        expert_inputs, token_map = PyTorchMoEOps.dispatch_tokens(
            tokens, indices, num_experts, capacity
        )
        assert expert_inputs.shape == (num_experts, capacity, 16)
        
        # Mock expert outputs (just return inputs)
        expert_outputs = expert_inputs
        
        combined = PyTorchMoEOps.combine_expert_outputs(
            expert_outputs, token_map, torch.ones(4, 2), 4, 2
        )
        assert combined.shape == (4, 16)
