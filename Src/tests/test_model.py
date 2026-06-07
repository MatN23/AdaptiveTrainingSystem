import pytest
import torch
import sys
from pathlib import Path

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from Main_Scripts.core.model import (
    RMSNorm, RotaryEmbedding, apply_rotary_pos_emb,
    SwiGLUExpert, MoEFFNLayer, DenseGroupedQueryAttention,
    DeepSeekTransformer, DeepSeekConfig
)


class TestRMSNorm:
    """Test RMSNorm layer."""
    
    def test_forward(self, mock_config):
        norm = RMSNorm(mock_config.hidden_size, mock_config.rms_norm_eps)
        x = torch.randn(2, 10, mock_config.hidden_size)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_numerical_stability(self):
        """Test RMSNorm with extreme values."""
        norm = RMSNorm(128)
        
        # Very small values
        x_small = torch.randn(2, 10, 128) * 1e-6
        output_small = norm(x_small)
        assert not torch.isnan(output_small).any()
        
        # Very large values
        x_large = torch.randn(2, 10, 128) * 1e6
        output_large = norm(x_large)
        assert not torch.isnan(output_large).any()


class TestRotaryEmbedding:
    """Test Rotary Position Embeddings."""
    
    def test_initialization(self, mock_config):
        head_dim = mock_config.hidden_size // mock_config.num_heads
        rope = RotaryEmbedding(head_dim, max_seq_len=128)
        
        assert rope.dim == head_dim
        assert rope.max_seq_len == 128
    
    def test_forward(self, mock_config):
        head_dim = mock_config.hidden_size // mock_config.num_heads
        rope = RotaryEmbedding(head_dim, max_seq_len=128)
        
        cos, sin = rope(seq_len=64, device=torch.device('cpu'))
        
        assert cos.shape == (64, head_dim)
        assert sin.shape == (64, head_dim)
        assert not torch.isnan(cos).any()
    
    def test_cache_extension(self, mock_config):
        """Test automatic cache extension for longer sequences."""
        head_dim = mock_config.hidden_size // mock_config.num_heads
        rope = RotaryEmbedding(head_dim, max_seq_len=64)
        
        # Request longer sequence
        cos, sin = rope(seq_len=128, device=torch.device('cpu'))
        
        assert cos.shape == (128, head_dim)
        assert rope.max_seq_len >= 128


class TestAttention:
    """Test attention mechanisms."""
    
    def test_attention_forward(self, mock_config):
        attn = DenseGroupedQueryAttention(mock_config)
        x = torch.randn(2, 10, mock_config.hidden_size)
        
        output = attn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_with_mask(self, mock_config):
        attn = DenseGroupedQueryAttention(mock_config)
        x = torch.randn(2, 10, mock_config.hidden_size)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask last 5 tokens
        
        output = attn(x, attention_mask=mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestMoE:
    """Test Mixture of Experts."""
    
    def test_swiglu_expert(self, mock_config):
        expert = SwiGLUExpert(mock_config)
        x = torch.randn(2, 10, mock_config.hidden_size)
        
        output = expert(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_moe_layer(self, mock_config):
        mock_config.use_moe = True
        moe_layer = MoEFFNLayer(mock_config)
        
        x = torch.randn(2, 10, mock_config.hidden_size)
        output, aux_loss = moe_layer(x)
        
        assert output.shape == x.shape
        assert aux_loss is not None
        assert not torch.isnan(output).any()
        assert aux_loss.item() >= 0


class TestFullModel:
    """Test complete model."""
    
    def test_model_forward(self, small_model):
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        output = small_model(input_ids)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        assert logits.shape == (batch_size, seq_len, 1000)
        assert not torch.isnan(logits).any()
    
    def test_model_backward(self, small_model):
        """Test backward pass works - with diagnostics."""
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        
        # Ensure model is in training mode
        small_model.train()
        
        output = small_model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 1000),
            labels.view(-1)
        )
        
        loss.backward()
        
        #  DIAGNOSTIC: Find parameters with None gradients
        params_without_grad = []
        for name, param in small_model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append({
                    'name': name,
                    'shape': param.shape,
                    'layer': name.split('.')[0:3]  # Get first 3 levels
                })
        
        # Print diagnostics if any params have None grad
        if params_without_grad:
            print("\n" + "="*70)
            print(" PARAMETERS WITH MISSING GRADIENTS:")
            print("="*70)
            for info in params_without_grad:
                print(f"  Name: {info['name']}")
                print(f"  Shape: {info['shape']}")
                print(f"  Layer hierarchy: {' -> '.join(info['layer'])}")
                print("-"*70)
            
            # Additional context
            print("\n GRADIENT STATISTICS:")
            total_params = sum(1 for p in small_model.parameters() if p.requires_grad)
            params_with_grad = sum(1 for p in small_model.parameters() 
                                if p.requires_grad and p.grad is not None)
            print(f"  Total parameters: {total_params}")
            print(f"  Parameters with gradients: {params_with_grad}")
            print(f"  Parameters without gradients: {len(params_without_grad)}")
            print("="*70 + "\n")
        
        # Original assertion with better error message
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"Parameter '{name}' (shape {param.shape}) has no gradient!"