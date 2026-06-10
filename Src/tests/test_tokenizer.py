# Copyright (c) 2025 MatN23. All rights reserved.
import pytest
import torch
import sys
from pathlib import Path

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class TestConversationTokenizer:
    """Test ConversationTokenizer."""
    
    def test_initialization(self):
        """Test tokenizer can be initialized."""
        try:
            from Main_Scripts.core.tokenizer import ConversationTokenizer
            
            tokenizer = ConversationTokenizer(model_name="gpt2")
            
            assert tokenizer.vocab_size > 0
            assert tokenizer.pad_token_id is not None
            assert len(tokenizer.special_tokens) > 0
        except ImportError:
            pytest.skip("Tokenizer module not available")
    
    def test_encode_conversation(self, mock_tokenizer):
        """Test conversation encoding."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        
        tokens = mock_tokenizer.encode_conversation(conversation)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_decode_tokens(self, mock_tokenizer):
        """Test token decoding."""
        tokens = [1, 2, 3, 4, 5]
        
        text = mock_tokenizer.decode(tokens)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_special_tokens(self, mock_tokenizer):
        """Test special tokens are defined."""
        assert "<|im_start|>" in mock_tokenizer.special_tokens
        assert "<|im_end|>" in mock_tokenizer.special_tokens
        assert "<|user|>" in mock_tokenizer.special_tokens
        assert "<|assistant|>" in mock_tokenizer.special_tokens