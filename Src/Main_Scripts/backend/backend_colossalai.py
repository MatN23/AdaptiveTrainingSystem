# Drop-in Colossal-AI replacement for DeepSpeed

import os
import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

# Structure: ColossalAI/colossalai/__init__.py
COLOSSALAI_PATH = Path(__file__).parent.parent / "ColossalAI"
if str(COLOSSALAI_PATH) not in sys.path:
    sys.path.insert(0, str(COLOSSALAI_PATH))

# Try importing ColossalAI
try:
    import colossalai  # This now finds ColossalAI/colossalai/
    from colossalai.booster import Booster
    from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin
    from colossalai.cluster import DistCoordinator
    from colossalai.nn.optimizer import HybridAdam
    COLOSSALAI_AVAILABLE = True
    logging.info(" ColossalAI loaded from local repo")
except ImportError as e:
    COLOSSALAI_AVAILABLE = False
    logging.warning(f" ColossalAI not available: {e}")

COLOSSALAI_AVAILABLE = True


class ColossalAIEngine:
    """
    Drop-in replacement for DeepSpeed engine.
    Mimics DeepSpeed API so your existing code works unchanged.
    """
    
    def __init__(self, model, optimizer, config):
        self.config = config
        self.coordinator = DistCoordinator()
        
        # Extract settings from config
        zero_stage = getattr(config, 'zero_stage', 2)
        precision = getattr(config, 'precision', 'fp16')
        cpu_offload = getattr(config, 'cpu_offload', False)
        max_norm = getattr(config, 'max_grad_norm', 1.0)
        
        if zero_stage == 3:
            plugin = GeminiPlugin(
                placement_policy='cpu' if cpu_offload else 'cuda',
                precision=precision,
                max_norm=max_norm,
            )
        else:
            plugin = LowLevelZeroPlugin(
                stage=zero_stage,
                precision=precision,
                max_norm=max_norm,
            )
        
        self.booster = Booster(plugin=plugin)
        
        # Wrap model and optimizer
        self.module, self.optimizer, _, _, self.scheduler = self.booster.boost(
            model=model,
            optimizer=optimizer,
        )
        
        # Metadata
        self.world_size = self.coordinator.world_size
        self.local_rank = self.coordinator.rank
        
        logging.info(f"ColossalAI initialized: ZeRO-{zero_stage}, {precision}, Rank {self.local_rank}/{self.world_size}")
    
    def backward(self, loss):
        """DeepSpeed: engine.backward(loss)"""
        self.booster.backward(loss, self.optimizer)
    
    def step(self):
        """DeepSpeed: engine.step()"""
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
    
    def save_checkpoint(self, save_dir, tag=None):
        """DeepSpeed: engine.save_checkpoint(save_dir, tag)"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = f"_{tag}" if tag else ""
        self.booster.save_model(self.module, str(save_dir / f"model{suffix}.pt"))
        self.booster.save_optimizer(self.optimizer, str(save_dir / f"optimizer{suffix}.pt"))
        
        logging.info(f"Checkpoint saved: {save_dir}")
        return str(save_dir)
    
    def load_checkpoint(self, load_dir, tag=None):
        """DeepSpeed: engine.load_checkpoint(load_dir, tag)"""
        load_dir = Path(load_dir)
        suffix = f"_{tag}" if tag else ""
        
        model_path = load_dir / f"model{suffix}.pt"
        optimizer_path = load_dir / f"optimizer{suffix}.pt"
        
        if model_path.exists():
            self.booster.load_model(self.module, str(model_path))
        if optimizer_path.exists():
            self.booster.load_optimizer(self.optimizer, str(optimizer_path))
        
        logging.info(f"Checkpoint loaded: {load_dir}")
    
    def get_lr(self):
        """DeepSpeed: engine.get_lr()"""
        if self.scheduler:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_global_grad_norm(self):
        """DeepSpeed: engine.get_global_grad_norm()"""
        total_norm = 0.0
        for p in self.module.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def eval(self):
        """Set model to eval mode"""
        self.module.eval()
    
    def train(self):
        """Set model to train mode"""
        self.module.train()
    
    def is_main_process(self):
        """Check if rank 0"""
        return self.local_rank == 0


def create_colossalai_backend(model, config, expert_registry=None):
    """
    Factory function - drop-in replacement for create_deepspeed_backend()
    
    Usage in your code:
        # OLD: from backend.backend_deepspeed import create_deepspeed_backend
        # NEW: from backend.backend_colossalai import create_colossalai_backend
        
        # OLD: backend = create_deepspeed_backend(model, config)
        # NEW: backend = create_colossalai_backend(model, config)
    """
    
    optimizer = HybridAdam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=getattr(config, 'weight_decay', 0.01)
    )
    
    engine = ColossalAIEngine(model, optimizer, config)
    
    return engine