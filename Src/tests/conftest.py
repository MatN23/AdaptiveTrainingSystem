# Copyright (c) 2025 MatN23. All rights reserved.
# ============================================================================
# tests/conftest.py - Shared Fixtures
# ============================================================================

import pytest
import torch
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import Mock

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class MockConfig:
    """Mock configuration for testing."""
    # Model architecture
    vocab_size = 1000
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    intermediate_size = 512
    seq_length = 64
    
    # Training parameters
    batch_size = 2
    num_epochs = 2
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 0.01
    max_grad_norm = 1.0
    gradient_accumulation_steps = 2
    save_total_limit = 3
    
    # Precision
    precision = 'fp32'
    inference_precision = 'fp32'
    
    # MoE/MoD
    use_moe = False
    use_mod = False
    num_experts = 4
    moe_top_k = 2
    capacity_factor = 1.25
    mod_capacity_factor = 0.5
    
    # Training configuration
    dropout = 0.0
    rms_norm_eps = 1e-6
    rope_theta = 10000.0
    init_std = 0.02
    use_stable_embedding = True
    tie_word_embeddings = True
    gradient_checkpointing = False
    
    # Quantization
    quantization_method = None
    quantization_bits = None
    
    # CUDA & Compilation
    use_cuda = False  # Default to False for tests unless specifically needed
    compile_mode = 'default'
    compile_backend = 'inductor'
    
    # DeepSpeed
    use_deepspeed = False
    cpu_offload = False
    zero_stage = 0
    
    # Logging
    experiment_name = "test_experiment"
    log_level = "INFO"
    log_every_n_steps = 10
    
    # Scheduler
    use_lr_scheduler = True
    lr_scheduler = "cosine"
    warmup_ratio = 0.1
    
    # Paths
    train_data_path = "test_train.jsonl"
    eval_data_path = "test_eval.jsonl"
    
    # Adaptive LR
    enable_adaptive_lr = True
    allow_scheduler_override = True
    
    # Experiment directory (will be set by tests if needed)
    experiment_dir = None
    
    # Custom Kernel Config
    cuda_optimization_params = {
        'use_fused_rmsnorm': False,
        'use_fused_rope': False,
        'use_fused_swiglu': False,
        'use_fused_moe': False,
        'use_fused_loss': False,
        'use_fused_grad_clip': False
    }
    
    def validate(self):
        pass
    
    def save(self, path):
        pass


class MockHFTokenizer:
    """Mock HuggingFace tokenizer that's nested inside ConversationTokenizer."""
    def encode(self, text, add_special_tokens=False, return_tensors=None):
        """Encode text to token IDs."""
        if isinstance(text, list):
            # Batch encoding
            return [list(range(min(len(t), 50))) for t in text]
        else:
            # Single text encoding
            return list(range(min(len(text), 50)))
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        return "decoded text"


class MockTokenizer:
    """Mock ConversationTokenizer that wraps a HuggingFace tokenizer."""
    vocab_size = 1000
    pad_token_id = 0
    
    def __init__(self):
        # Nested tokenizer (HuggingFace tokenizer)
        self.tokenizer = MockHFTokenizer()
    
    special_tokens = {
        "<|im_start|>": 1000,
        "<|im_end|>": 1001,
        "<|user|>": 1002,
        "<|assistant|>": 1003,
        "<|system|>": 1004,
    }
    
    def encode_conversation(self, conversation):
        return [1, 2, 3, 4, 5] * 10
    
    def encode(self, text):
        return list(range(min(len(text), 50)))
    
    def decode(self, token_ids):
        return "decoded text"
    
    def get_role_token(self, role):
        return self.special_tokens.get(f"<|{role}|>", 1002)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return MockConfig()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    return MockTokenizer()


@pytest.fixture
def sample_conversation_data(temp_dir):
    """Create sample conversation dataset."""
    data = []
    for i in range(10):
        data.append({
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."}
            ]
        })
    
    data_path = temp_dir / "test_data.jsonl"
    with open(data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return str(data_path)


@pytest.fixture
def sample_base_training_data(temp_dir):
    """Create sample base training dataset."""
    data_path = temp_dir / "test_base.txt"
    with open(data_path, 'w') as f:
        for i in range(10):
            f.write(f"This is sample text number {i}. " * 20 + "\n")
    
    return str(data_path)


@pytest.fixture
def small_model(mock_config):
    """Create small model for testing."""
    try:
        from Main_Scripts.core.model import DeepSeekTransformer, DeepSeekConfig
        
        model_config = DeepSeekConfig(
            vocab_size=mock_config.vocab_size,
            hidden_size=mock_config.hidden_size,
            num_layers=mock_config.num_layers,
            num_heads=mock_config.num_heads,
            num_kv_heads=mock_config.num_kv_heads,
            seq_length=mock_config.seq_length
        )
        
        return DeepSeekTransformer(model_config)
    except ImportError:
        pytest.skip("Model module not available")


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return Mock()