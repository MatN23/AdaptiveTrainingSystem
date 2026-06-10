# Copyright (c) 2025 MatN23. All rights reserved.
"""
PyTorch FSDP Backend Wrapper
Provides unified API for FSDP training that matches PyTorch interface.
"""

import os
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        StateDictType,
        FullStateDictConfig,
        FullOptimStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    from torch.amp import autocast, GradScaler
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


class FSDPBackend:
    """
    Wrapper for PyTorch FSDP that provides a unified interface.
    
    This allows trainer code to call:
    - model(input) for forward pass
    - model.backward(loss) for backprop
    - model.step() for optimizer step
    - model.zero_grad() for gradient zeroing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        model_parameters: Optional[list] = None
    ):
        """
        Initialize FSDP backend.
        
        Args:
            model: PyTorch model to wrap
            config: Configuration object with FSDP settings
            model_parameters: Optional model parameters (unused, for API compatibility)
        """
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP is not available. Requires PyTorch >= 2.0")
        
        self.config = config
        self.original_model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._init_distributed()
        
        self.mixed_precision_policy = self._create_mixed_precision_policy()
        
        # Wrap model with FSDP
        self.model = self._wrap_model_with_fsdp(model)
        
        self.optimizer = self._create_optimizer()
        
        self.scheduler = None  # Will be set up when total_steps is known
        
        self.use_amp = self._should_use_grad_scaler()
        self.scaler = GradScaler() if self.use_amp else None
        
        self.global_step = 0
        
        logging.info(f"FSDP backend initialized:")
        logging.info(f"  World size: {self.world_size}")
        logging.info(f"  Local rank: {self.local_rank}")
        logging.info(f"  Sharding strategy: {getattr(config, 'fsdp_sharding_strategy', 'FULL_SHARD')}")
    
    def _init_distributed(self):
        """Initialize distributed training if not already initialized."""
        if not torch.distributed.is_initialized():
            # Get environment variables
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            if world_size > 1:
                torch.distributed.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://'
                )
                torch.cuda.set_device(local_rank)
                logging.info(f"Initialized distributed training: rank={local_rank}, world_size={world_size}")
    
    @property
    def world_size(self) -> int:
        """Get world size."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1
    
    @property
    def local_rank(self) -> int:
        """Get local rank."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0
    
    def _create_mixed_precision_policy(self):
        """Create mixed precision policy based on configuration."""
        precision = getattr(self.config, 'precision', 'fp32')
        
        if precision in ['fp16', 'mixed_fp16']:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif precision in ['bf16', 'mixed_bf16']:
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            return None  # FP32
    
    def _wrap_model_with_fsdp(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP."""
        # Determine sharding strategy
        sharding_strategy_name = getattr(self.config, 'fsdp_sharding_strategy', 'FULL_SHARD')
        sharding_strategy_map = {
            'FULL_SHARD': ShardingStrategy.FULL_SHARD,
            'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
            'NO_SHARD': ShardingStrategy.NO_SHARD,
            'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = sharding_strategy_map.get(
            sharding_strategy_name,
            ShardingStrategy.FULL_SHARD
        )
        
        # Auto wrap policy - wrap layers above this parameter count
        auto_wrap_threshold = getattr(self.config, 'fsdp_auto_wrap_threshold', 1e8)
        auto_wrap_policy = size_based_auto_wrap_policy if auto_wrap_threshold > 0 else None
        
        # CPU offloading
        cpu_offload = getattr(self.config, 'cpu_offload', False)
        
        fsdp_kwargs = {
            'sharding_strategy': sharding_strategy,
            'mixed_precision': self.mixed_precision_policy,
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'device_id': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'limit_all_gathers': True,
            'use_orig_params': True,  # Important for optimizer compatibility
        }
        
        if auto_wrap_policy:
            fsdp_kwargs['auto_wrap_policy'] = lambda module, recurse, nonwrapped_numel: (
                recurse(module) if recurse else nonwrapped_numel >= auto_wrap_threshold
            )
        
        if cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)
        
        # Wrap model
        fsdp_model = FSDP(model, **fsdp_kwargs)
        
        # Apply activation checkpointing if enabled
        if getattr(self.config, 'gradient_checkpointing', False):
            self._apply_activation_checkpointing(fsdp_model)
        
        return fsdp_model
    
    def _apply_activation_checkpointing(self, model: FSDP):
        """Apply activation checkpointing to reduce memory usage."""
        try:
            def check_fn(submodule):
                # Checkpoint transformer blocks or similar large modules
                return isinstance(submodule, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))
            
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
                    module,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=check_fn
            )
            logging.info("Activation checkpointing applied")
        except Exception as e:
            logging.warning(f"Could not apply activation checkpointing: {e}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': getattr(self.config, 'weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        try:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=torch.cuda.is_available()
            )
        except Exception:
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def _should_use_grad_scaler(self) -> bool:
        """Determine if gradient scaling is needed."""
        precision = getattr(self.config, 'precision', 'fp32')
        return precision in ['fp16', 'mixed_fp16']
    
    def setup_scheduler(self, total_steps: int):
        """
        Setup learning rate scheduler.
        
        Args:
            total_steps: Total number of training steps
        """
        if not getattr(self.config, 'use_lr_scheduler', True):
            self.scheduler = None
            return
        
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        lr_scheduler_type = getattr(self.config, 'lr_scheduler', 'cosine')
        
        if lr_scheduler_type == "cosine":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
                    min_lr_ratio = getattr(self.config, 'min_lr', self.config.learning_rate * 0.1) / self.config.learning_rate
                    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        elif lr_scheduler_type == "linear":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    progress = (current_step - warmup_steps) / max(1, (total_steps - warmup_steps))
                    min_lr_ratio = getattr(self.config, 'min_lr', 0) / self.config.learning_rate
                    return max(min_lr_ratio, 1.0 - progress)
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        elif lr_scheduler_type == "constant":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    return 1.0
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        logging.info(f"Learning rate scheduler initialized: {lr_scheduler_type}")
    
    # UNIFIED API - Forward Pass
    
    def __call__(self, *args, **kwargs):
        """Forward pass - matches PyTorch model interface."""
        return self.model(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Explicit forward pass."""
        return self.model(*args, **kwargs)
    
    # UNIFIED API - Training Methods
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass - matches PyTorch loss.backward() interface.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """
        Optimizer step - matches PyTorch optimizer.step() interface.
        Updates model parameters and learning rate.
        """
        # Unscale gradients if using AMP
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients - matches PyTorch optimizer.zero_grad() interface.
        
        Args:
            set_to_none: Whether to set gradients to None
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    # UNIFIED API - Model State
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()
    
    def state_dict(self):
        """Get model state dict."""
        # FSDP requires special handling for state dict
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            return self.model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict."""
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            self.model.load_state_dict(state_dict)
    
    # UNIFIED API - Learning Rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]['lr']
    
    def get_last_lr(self):
        """Get last learning rate (scheduler compatibility)."""
        return [self.get_lr()]
    
    # UNIFIED API - Gradient Information
    
    def get_global_grad_norm(self) -> float:
        """Get global gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    # UNIFIED API - Checkpointing
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        global_step: int,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint using FSDP's checkpoint system.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
            global_step: Current global step
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only save on rank 0
        if self.is_main_process():
            # Get full state dict
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                model_state = self.model.state_dict()
            
            # Get optimizer state
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                optim_state = FSDP.optim_state_dict(self.model, self.optimizer)
            
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optim_state,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'global_step': global_step,
                'epoch': epoch,
                'config': self.config
            }
            
            if additional_state is not None:
                checkpoint_data.update(additional_state)
            
            torch.save(checkpoint_data, checkpoint_path)
            logging.info(f"FSDP checkpoint saved: {checkpoint_path}")
        
        # Wait for all processes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint using FSDP's checkpoint system.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer_states: Whether to load optimizer states
            load_lr_scheduler_states: Whether to load LR scheduler states
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer_states and 'optimizer_state_dict' in checkpoint:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            ):
                optim_state = FSDP.optim_state_dict_to_load(
                    self.model, self.optimizer, checkpoint['optimizer_state_dict']
                )
                self.optimizer.load_state_dict(optim_state)
        
        if load_lr_scheduler_states and self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0)
        }
        
        logging.info(f"FSDP checkpoint loaded: {checkpoint_path}")
        return metadata
    
    # UNIFIED API - Device Management
    
    def to(self, device):
        """Device movement (no-op for FSDP, handles internally)."""
        return self
    
    def cuda(self):
        """Move to CUDA (no-op for FSDP)."""
        return self
    
    def cpu(self):
        """Move to CPU (no-op for FSDP)."""
        return self
    
    # Backend-Specific Properties
    
    @property
    def module(self):
        """Access underlying model module."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "fsdp"
    
    # Utility Methods
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.local_rank == 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {}
    
    def get_autocast_context(self):
        """Get autocast context for mixed precision."""
        precision = getattr(self.config, 'precision', 'fp32')
        
        if precision in ['fp16', 'mixed_fp16']:
            return autocast('cuda', dtype=torch.float16, enabled=True)
        elif precision in ['bf16', 'mixed_bf16']:
            return autocast('cuda', dtype=torch.bfloat16, enabled=True)
        else:
            return nullcontext()
    
    def __repr__(self):
        return (f"FSDPBackend(world_size={self.world_size}, "
                f"local_rank={self.local_rank}, "
                f"sharding_strategy={getattr(self.config, 'fsdp_sharding_strategy', 'FULL_SHARD')})")


# Factory Function

def create_fsdp_backend(
    model: nn.Module,
    config: Any,
    model_parameters: Optional[list] = None
) -> FSDPBackend:
    """
    Factory function to create FSDP backend.
    
    Args:
        model: PyTorch model
        config: Configuration object
        model_parameters: Optional model parameters (unused, for API compatibility)
        
    Returns:
        FSDPBackend instance
    """
    return FSDPBackend(model, config, model_parameters)