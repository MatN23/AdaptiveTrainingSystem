# Copyright (c) 2025 MatN23. All rights reserved.
import pytest
import torch
import sys
from pathlib import Path

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class TestModelTrainerIntegration:
    """Test model and trainer integration."""
    
    def test_full_training_step(self, small_model, mock_tokenizer, mock_config, mock_logger):
        """Test complete training step with model and trainer."""
        from Main_Scripts.training.trainer import EnhancedConversationTrainer
        
        trainer = EnhancedConversationTrainer(
            small_model, mock_tokenizer, mock_config, mock_logger
        )
        
        batch = {
            'input_ids': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'labels': torch.randint(0, mock_config.vocab_size, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'loss_weights': torch.ones(2, 10)
        }
        
        # Training step
        step_metrics = trainer.train_step(batch)
        
        # Optimizer step
        opt_metrics = trainer.optimizer_step()
        
        assert 'loss' in step_metrics
        assert 'grad_norm' in opt_metrics
        assert opt_metrics['grad_norm'] >= 0


class TestDatasetTrainerIntegration:
    """Test dataset and trainer integration."""
    
    def test_dataloader_training(
        self, small_model, sample_conversation_data, 
        mock_tokenizer, mock_config, mock_logger
    ):
        """Test training with real dataloader."""
        try:
            from Main_Scripts.core.dataset import FastConversationDataset, create_dataloader
            from Main_Scripts.training.trainer import EnhancedConversationTrainer
            
            dataset = FastConversationDataset(
                sample_conversation_data,
                mock_tokenizer,
                mock_config
            )
            
            dataloader = create_dataloader(dataset, mock_config, shuffle=False)
            
            trainer = EnhancedConversationTrainer(
                small_model, mock_tokenizer, mock_config, mock_logger
            )
            
            # Train one batch
            batch = next(iter(dataloader))
            step_metrics = trainer.train_step(batch)
            
            assert 'loss' in step_metrics
            assert step_metrics['loss'] >= 0
            
        except ImportError:
            pytest.skip("Required modules not available")