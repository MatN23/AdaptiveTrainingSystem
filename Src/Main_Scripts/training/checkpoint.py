# Licensed under the Custom License below.

import os
import json
import logging
import shutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


class CheckpointManager:
    """Enhanced checkpoint management with versioning and automatic cleanup."""
    
    def __init__(self, config, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment-specific directory
        self.experiment_dir = self.checkpoint_dir / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoint_history = []
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')
        
        self._load_checkpoint_history()
        
        logging.info(f"CheckpointManager initialized: {self.experiment_dir}")
    
    def save_checkpoint(self, model, optimizer, scheduler, global_step: int, 
                       current_epoch: int, metrics: Dict[str, Any], 
                       suffix: str = None, is_best: bool = False) -> str:
        """Save checkpoint with comprehensive state and metadata."""
        
        # Generate checkpoint filename
        if suffix:
            checkpoint_name = f"checkpoint_{suffix}.pt"
        else:
            checkpoint_name = f"checkpoint_epoch_{current_epoch:03d}_step_{global_step:06d}.pt"
        
        checkpoint_path = self.experiment_dir / checkpoint_name
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'global_step': global_step,
                'current_epoch': current_epoch,
                'metrics': metrics,
                'config': self.config.__dict__,
                'model_config': {
                    'vocab_size': self.config.vocab_size,
                    'hidden_size': self.config.hidden_size,
                    'num_layers': self.config.num_layers,
                    'num_heads': self.config.num_heads,
                    'num_kv_heads': self.config.num_kv_heads,
                    'seq_length': self.config.seq_length,
                    'intermediate_size': self.config.intermediate_size,
                },
                'save_time': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update checkpoint history
            checkpoint_info = {
                'path': str(checkpoint_path),
                'epoch': current_epoch,
                'step': global_step,
                'save_time': checkpoint_data['save_time'],
                'is_best': is_best,
                'eval_loss': metrics.get('eval_losses', [float('inf')])[-1] if metrics.get('eval_losses') else float('inf')
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # Update best checkpoint if needed
            current_eval_loss = checkpoint_info['eval_loss']
            if is_best or current_eval_loss < self.best_metric_value:
                self.best_metric_value = current_eval_loss
                self.best_checkpoint_path = str(checkpoint_path)
                
                best_link = self.experiment_dir / "best_checkpoint.pt"
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(checkpoint_path.name)
                
                logging.info(f"New best checkpoint saved: {checkpoint_path} (eval_loss: {current_eval_loss:.6f})")
            
            self._save_checkpoint_history()
            
            # Clean up old checkpoints if needed
            self._cleanup_old_checkpoints()
            
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, 
                       scheduler=None, strict: bool = True) -> int:
        """Load checkpoint and restore training state."""
        
        checkpoint_path = Path(checkpoint_path)
        
        # Handle special keywords
        if checkpoint_path.name == "latest":
            checkpoint_path = self.get_latest_checkpoint()
        elif checkpoint_path.name == "best":
            checkpoint_path = self.get_best_checkpoint()
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            
            device = next(model.parameters()).device
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            self._validate_checkpoint_compatibility(checkpoint_data)
            
            if strict:
                model.load_state_dict(checkpoint_data['model_state_dict'])
            else:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data['model_state_dict'], strict=False
                )
                if missing_keys:
                    logging.warning(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                except Exception as e:
                    logging.warning(f"Failed to load optimizer state: {e}")
            
            if scheduler and 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict']:
                try:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                except Exception as e:
                    logging.warning(f"Failed to load scheduler state: {e}")
            
            # Extract training state
            current_epoch = checkpoint_data.get('current_epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            
            logging.info(f"Checkpoint loaded successfully. Resuming from epoch {current_epoch}, step {global_step}")
            
            return current_epoch
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        # Find the checkpoint with the highest step count
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['step'])
        return Path(latest_checkpoint['path'])
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        if self.best_checkpoint_path:
            return Path(self.best_checkpoint_path)
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their metadata."""
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """Delete a specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                
                # Remove from history
                self.checkpoint_history = [
                    cp for cp in self.checkpoint_history 
                    if cp['path'] != str(checkpoint_path)
                ]
                self._save_checkpoint_history()
                
                logging.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
        except Exception as e:
            logging.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
        
        return False
    
    def create_backup(self, backup_dir: str = None) -> str:
        """Create a backup of all checkpoints."""
        if backup_dir is None:
            backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path("backups") / backup_dir
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy all checkpoints
            shutil.copytree(self.experiment_dir, backup_path / self.config.experiment_name)
            
            backup_metadata = {
                'backup_time': datetime.now().isoformat(),
                'experiment_name': self.config.experiment_name,
                'source_path': str(self.experiment_dir),
                'checkpoint_count': len(self.checkpoint_history),
                'best_checkpoint': self.best_checkpoint_path
            }
            
            with open(backup_path / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logging.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            raise
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_file = self.experiment_dir / "checkpoint_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_history = data.get('checkpoints', [])
                    self.best_checkpoint_path = data.get('best_checkpoint_path')
                    self.best_metric_value = data.get('best_metric_value', float('inf'))
            except Exception as e:
                logging.warning(f"Failed to load checkpoint history: {e}")
                self.checkpoint_history = []
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_file = self.experiment_dir / "checkpoint_history.json"
        
        try:
            history_data = {
                'checkpoints': self.checkpoint_history,
                'best_checkpoint_path': self.best_checkpoint_path,
                'best_metric_value': self.best_metric_value,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save checkpoint history: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints based on save_total_limit."""
        if self.config.save_total_limit <= 0:
            return
        
        # Get regular checkpoints (not best or emergency)
        regular_checkpoints = [
            cp for cp in self.checkpoint_history 
            if not cp.get('is_best', False) and 'emergency' not in cp['path']
        ]
        
        # Sort by step (newest first)
        regular_checkpoints.sort(key=lambda x: x['step'], reverse=True)
        
        # Remove excess checkpoints
        checkpoints_to_remove = regular_checkpoints[self.config.save_total_limit:]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    logging.debug(f"Cleaned up old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logging.warning(f"Failed to clean up {checkpoint_path}: {e}")
        
        # Update history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history 
            if cp not in checkpoints_to_remove
        ]
    
    def _validate_checkpoint_compatibility(self, checkpoint_data: Dict[str, Any]):
        """Validate that checkpoint is compatible with current configuration."""
        if 'model_config' not in checkpoint_data:
            logging.warning("Checkpoint missing model configuration")
            return
        
        checkpoint_config = checkpoint_data['model_config']
        current_config = {
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'seq_length': self.config.seq_length,
        }
        
        critical_params = ['vocab_size', 'hidden_size', 'num_layers', 'num_heads']
        
        for param in critical_params:
            if param in checkpoint_config and param in current_config:
                if checkpoint_config[param] != current_config[param]:
                    raise ValueError(
                        f"Model architecture mismatch: {param} "
                        f"checkpoint={checkpoint_config[param]} vs current={current_config[param]}"
                    )
    
    def get_resume_path(self) -> Optional[str]:
        """Get the best path to resume training from."""
        # Try to find the latest checkpoint first
        latest = self.get_latest_checkpoint()
        if latest and latest.exists():
            return str(latest)
        
        # Fall back to best checkpoint
        best = self.get_best_checkpoint()
        if best and best.exists():
            return str(best)
        
        return None
    
    def emergency_save(self, model, optimizer, scheduler, global_step: int, 
                      current_epoch: int, metrics: Dict[str, Any]) -> str:
        """Emergency checkpoint save (e.g., on interruption)."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.save_checkpoint(
            model, optimizer, scheduler, global_step, current_epoch, metrics,
            suffix=f"emergency_{timestamp}"
        )