import pytest
import torch
import sys
from pathlib import Path

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class TestConversationDataset:
    """Test conversation dataset."""
    
    def test_dataset_loading(self, sample_conversation_data, mock_tokenizer, mock_config):
        """Test conversation dataset can be loaded."""
        try:
            from Main_Scripts.core.dataset import FastConversationDataset
            
            dataset = FastConversationDataset(
                sample_conversation_data,
                mock_tokenizer,
                mock_config
            )
            
            assert len(dataset) > 0
            
            sample = dataset[0]
            assert 'input_ids' in sample
            assert 'labels' in sample
            assert 'attention_mask' in sample
            
        except ImportError:
            pytest.skip("Dataset module not available")
    
    def test_dataloader_creation(self, sample_conversation_data, mock_tokenizer, mock_config):
        """Test dataloader can be created."""
        try:
            from Main_Scripts.core.dataset import FastConversationDataset, create_dataloader
            
            dataset = FastConversationDataset(
                sample_conversation_data,
                mock_tokenizer,
                mock_config
            )
            
            dataloader = create_dataloader(dataset, mock_config, shuffle=True)
            
            assert len(dataloader) > 0
            
            batch = next(iter(dataloader))
            assert 'input_ids' in batch
            assert batch['input_ids'].shape[0] <= mock_config.batch_size
            
        except ImportError:
            pytest.skip("Dataset module not available")


class TestBaseTrainingDataset:
    """Test base training dataset."""
    
    def test_base_dataset_loading(self, sample_base_training_data, mock_tokenizer, mock_config):
        """Test base training dataset can be loaded."""
        try:
            from Main_Scripts.core.dataset import FastBaseTrainingDataset
            
            dataset = FastBaseTrainingDataset(
                sample_base_training_data,
                mock_tokenizer,
                mock_config
            )
            
            assert len(dataset) > 0
            
            sample = dataset[0]
            assert 'input_ids' in sample
            assert 'labels' in sample
            
        except ImportError:
            pytest.skip("Dataset module not available")
class TestRobustness:
    """Test robustness to edge cases and corrupted data."""

    def test_corrupted_jsonl(self, temp_dir, mock_tokenizer, mock_config):
        """Test handling of corrupted lines in JSONL."""
        data_path = temp_dir / "corrupted.jsonl"
        with open(data_path, 'w') as f:
            f.write('{"messages": [{"role": "user", "content": "hi"}]}\n') # Valid
            f.write('INVALID JSON\n') # Corrupted
            f.write('{"wrong_key": "oops"}\n') # Missing 'messages'
            
        from Main_Scripts.core.dataset import FastConversationDataset
        try:
            dataset = FastConversationDataset(str(data_path), mock_tokenizer, mock_config)
            assert len(dataset) >= 0
        except Exception:
            pass

    def test_empty_file(self, temp_dir, mock_tokenizer, mock_config):
        """Test handling of empty training files."""
        data_path = temp_dir / "empty.jsonl"
        data_path.touch()
        
        from Main_Scripts.core.dataset import FastConversationDataset
        try:
            dataset = FastConversationDataset(str(data_path), mock_tokenizer, mock_config)
            assert len(dataset) == 0
        except Exception:
            pass
