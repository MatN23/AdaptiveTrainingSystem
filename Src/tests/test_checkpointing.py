# Copyright (c) 2025 MatN23. All rights reserved.

import pytest
import torch
import os
import shutil
import json
from pathlib import Path
from Main_Scripts.training.checkpoint import CheckpointManager

class TestCheckpointing:
    """Test suite for CheckpointManager."""

    @pytest.fixture
    def checkpoint_manager(self, mock_config, temp_dir):
        """Create a CheckpointManager instance for testing."""
        # Update mock_config with necessary attributes for CheckpointManager
        mock_config.save_total_limit = 3
        return CheckpointManager(mock_config, checkpoint_dir=str(temp_dir / "checkpoints"))

    @pytest.fixture
    def mock_model_opt_sched(self, mock_config, small_model):
        """Create mock model, optimizer, and scheduler."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=mock_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return small_model, optimizer, scheduler

    def test_initialization(self, checkpoint_manager, temp_dir, mock_config):
        """Test CheckpointManager initialization."""
        assert checkpoint_manager.checkpoint_dir.exists()
        assert checkpoint_manager.experiment_dir.exists()
        assert checkpoint_manager.experiment_dir.name == mock_config.experiment_name

    def test_save_and_load_checkpoint(self, checkpoint_manager, mock_model_opt_sched):
        """Test basic saving and loading functionality."""
        model, optimizer, scheduler = mock_model_opt_sched
        metrics = {'train_losses': [0.5], 'eval_losses': [0.4]}
        
        # Save checkpoint
        path = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, global_step=100, 
            current_epoch=1, metrics=metrics
        )
        assert os.path.exists(path)
        
        # Load checkpoint
        new_model, new_opt, new_sched = mock_model_opt_sched
        epoch = checkpoint_manager.load_checkpoint(path, new_model, new_opt, new_sched)
        
        assert epoch == 1
        assert len(checkpoint_manager.checkpoint_history) == 1
        assert checkpoint_manager.checkpoint_history[0]['step'] == 100

    def test_best_checkpoint_handling(self, checkpoint_manager, mock_model_opt_sched):
        """Test tracking and symlinking of the best checkpoint."""
        model, optimizer, scheduler = mock_model_opt_sched
        
        # Save first checkpoint
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, global_step=100, 
            current_epoch=1, metrics={'eval_losses': [0.5]}
        )
        
        # Save second (better) checkpoint
        path2 = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, global_step=200, 
            current_epoch=2, metrics={'eval_losses': [0.3]}
        )
        
        assert checkpoint_manager.best_checkpoint_path == path2
        best_link = checkpoint_manager.experiment_dir / "best_checkpoint.pt"
        assert best_link.exists()
        assert best_link.is_symlink()

    def test_checkpoint_cleanup(self, checkpoint_manager, mock_model_opt_sched):
        """Test that old checkpoints are correctly cleaned up."""
        model, optimizer, scheduler = mock_model_opt_sched
        checkpoint_manager.config.save_total_limit = 2
        
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, global_step=i*100, 
                current_epoch=i, metrics={}
            )
            
        # Should only have 2 checkpoints left (plus history update)
        # Note: the history might be larger if we don't prune it, but CheckpointManager._cleanup_old_checkpoints
        # DOES prune the history list in memory and on disk.
        assert len(checkpoint_manager.checkpoint_history) <= 2
        
        # Verify only 2 files remain in the directory (excluding history and best link)
        files = [f for f in checkpoint_manager.experiment_dir.glob("checkpoint_*.pt")]
        assert len(files) <= 2

    def test_compatibility_validation(self, checkpoint_manager, mock_model_opt_sched, mock_config):
        """Test architecture compatibility checks during loading."""
        model, optimizer, scheduler = mock_model_opt_sched
        path = checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 0, 0, {})
        
        # Force a mismatch in config
        checkpoint_manager.config.hidden_size = 999
        
        with pytest.raises(ValueError, match="Model architecture mismatch"):
            checkpoint_manager.load_checkpoint(path, model)

    def test_backup_and_emergency_save(self, checkpoint_manager, mock_model_opt_sched):
        """Test backup creation and emergency save."""
        model, optimizer, scheduler = mock_model_opt_sched
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, 100, 1, {})
        
        # Test emergency save
        e_path = checkpoint_manager.emergency_save(model, optimizer, scheduler, 100, 1, {})
        assert "emergency" in e_path
        
        # Test backup
        backup_path = checkpoint_manager.create_backup()
        assert os.path.exists(backup_path)
        assert os.path.exists(os.path.join(backup_path, checkpoint_manager.config.experiment_name))
        assert os.path.exists(os.path.join(backup_path, "backup_metadata.json"))

    def test_load_keywords(self, checkpoint_manager, mock_model_opt_sched):
        """Test loading using 'latest' and 'best' keywords."""
        model, optimizer, scheduler = mock_model_opt_sched
        
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, global_step=100, 
            current_epoch=1, metrics={'eval_losses': [0.1]}, is_best=True
        )
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, global_step=200, 
            current_epoch=2, metrics={'eval_losses': [0.5]}
        )
        
        # Test 'latest'
        epoch_latest = checkpoint_manager.load_checkpoint("latest", model)
        assert epoch_latest == 2
        
        # Test 'best'
        epoch_best = checkpoint_manager.load_checkpoint("best", model)
        assert epoch_best == 1
