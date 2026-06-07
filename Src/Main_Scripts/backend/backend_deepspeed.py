"""
DeepSpeed Backend Wrapper
Provides unified API for DeepSpeed training that matches PyTorch interface.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedEngine = object


class DeepSpeedBackend:
    """
    Wrapper for DeepSpeed that provides a PyTorch-like interface.
    
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
        Initialize DeepSpeed backend.
        
        Args:
            model: PyTorch model to wrap
            config: Configuration object with DeepSpeed settings
            model_parameters: Optional model parameters (will use model.parameters() if None)
        """
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed is not available. Install with: pip install deepspeed")
        
        self.config = config
        self.original_model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ds_config = self._create_deepspeed_config()
        
        if model_parameters is None:
            model_parameters = model.parameters()
        
        self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model_parameters
        )
        
        self.global_step = 0
        
        logging.info(f"DeepSpeed backend initialized:")
        logging.info(f"  World size: {self.engine.world_size}")
        logging.info(f"  Local rank: {self.engine.local_rank}")
        logging.info(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'disabled')}")
    
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration from trainer config."""
        micro_batch_size = getattr(self.config, 'batch_size', 1)
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
        
        # Determine precision settings
        precision = getattr(self.config, 'precision', 'fp32')
        use_fp16 = precision in ['fp16', 'mixed_fp16']
        use_bf16 = precision in ['bf16', 'mixed_bf16']
        
        ds_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            
            "fp16": {
                "enabled": use_fp16,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            "bf16": {
                "enabled": use_bf16
            },
            
            "gradient_clipping": getattr(self.config, 'max_grad_norm', 1.0),
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": getattr(self.config, 'weight_decay', 0.01)
                }
            },
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": getattr(self.config, 'min_lr', 1e-6),
                    "warmup_max_lr": self.config.learning_rate,
                    "warmup_num_steps": 100  # Will be updated when steps_per_epoch is known
                }
            },
            
            "zero_optimization": {
                "stage": getattr(self.config, 'zero_stage', 2),
                "allgather_partitions": True,
                "allgather_bucket_size": int(5e8),
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": int(5e8),
                "contiguous_gradients": True
            },
            
            "steps_per_print": 1,
            "wall_clock_breakdown": False,
            "dump_state": False
        }
        
        # Add CPU offloading if configured
        if getattr(self.config, 'cpu_offload', False):
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "nvme_path": getattr(self.config, 'nvme_path', None),
                "buffer_count": 4,
                "pin_memory": True,
                "pipeline_read": True,
                "pipeline_write": True,
                "fast_init": False
            }
            
            if getattr(self.config, 'cpu_offload_parameters', False):
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "nvme_path": getattr(self.config, 'nvme_path', None),
                    "buffer_count": 5,
                    "buffer_size": 100000000.0,
                    "max_in_cpu": 1000000000.0,
                    "pin_memory": True
                }
        
        # Add MoE configuration if enabled
        if hasattr(self.config, 'use_moe') and self.config.use_moe:
            num_experts = getattr(self.config, 'num_experts', 8)
            expert_parallel_size = self._calculate_expert_parallel_size(world_size, num_experts)
            
            ds_config["moe"] = {
                "enabled": True,
                "num_experts": num_experts,
                "expert_parallel_size": expert_parallel_size,
                "top_k": getattr(self.config, 'moe_top_k', 2),
                "capacity_factor": getattr(self.config, 'capacity_factor', 1.25),
                "eval_capacity_factor": 3.2,
                "min_capacity": 16,
                "use_residual": True,
                "load_balance_loss_coef": getattr(self.config, 'load_balancing_weight', 0.08),
                "load_balance_type": "aux_loss",
                "router_jitter_noise": 0.01,
                "enable_expert_tensor_parallelism": True,
                "all_to_all_dispatch": True,
                "overlap_alltoall": True,
                "comm_dtype": "fp16" if use_fp16 else "bf16",
                "pad_expert_input_to_capacity": True,
                "enable_expert_weight_parallelism": True,
                "moe_param_group": True,
                "expert_placement_policy": "balanced",
                "use_tutel": False,
            }
        
        return ds_config
    
    def _calculate_expert_parallel_size(self, world_size: int, num_experts: int) -> int:
        """Calculate optimal expert parallel size."""
        possible_sizes = []
        for i in range(1, world_size + 1):
            if world_size % i == 0:
                experts_per_group = num_experts // i
                if experts_per_group >= 1:
                    possible_sizes.append(i)
        
        return possible_sizes[-1] if possible_sizes else 1
    
    # UNIFIED API - Forward Pass
    
    def __call__(self, *args, **kwargs):
        """Forward pass - matches PyTorch model interface."""
        return self.engine(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Explicit forward pass."""
        return self.engine(*args, **kwargs)
    
    # UNIFIED API - Training Methods
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass - matches PyTorch loss.backward() interface.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        self.engine.backward(loss)
    
    def step(self):
        """
        Optimizer step - matches PyTorch optimizer.step() interface.
        Updates model parameters and learning rate.
        """
        self.engine.step()
        self.global_step += 1
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients - matches PyTorch optimizer.zero_grad() interface.
        
        Args:
            set_to_none: Whether to set gradients to None (ignored for DeepSpeed)
        """
        # DeepSpeed handles gradient zeroing internally
        pass
    
    # UNIFIED API - Model State
    
    def train(self):
        """Set model to training mode."""
        self.engine.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.engine.eval()
    
    def parameters(self):
        """Get model parameters."""
        return self.engine.parameters()
    
    def named_parameters(self):
        """Get named model parameters."""
        return self.engine.named_parameters()
    
    def state_dict(self):
        """Get model state dict."""
        return self.engine.module.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict."""
        self.engine.module.load_state_dict(state_dict)
    
    # UNIFIED API - Learning Rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        try:
            if hasattr(self.engine, 'get_lr') and callable(self.engine.get_lr):
                lr_list = self.engine.get_lr()
                if lr_list and len(lr_list) > 0:
                    return lr_list[0]
        except Exception:
            pass
        return self.config.learning_rate
    
    def get_last_lr(self):
        """Get last learning rate (scheduler compatibility)."""
        return [self.get_lr()]
    
    # UNIFIED API - Gradient Information
    
    def get_global_grad_norm(self) -> float:
        """Get global gradient norm."""
        try:
            if hasattr(self.engine, 'get_global_grad_norm'):
                norm = self.engine.get_global_grad_norm()
                if norm is not None:
                    return float(norm)
        except Exception:
            pass
        return 0.0
    
    # UNIFIED API - Checkpointing
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        global_step: int,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint using DeepSpeed's checkpoint system.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
            global_step: Current global step
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}_step_{global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.engine.save_checkpoint(str(checkpoint_path))
        
        if additional_state is not None:
            metadata_path = checkpoint_path / "metadata.json"
            metadata = {
                'epoch': epoch,
                'global_step': global_step,
                **additional_state
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"DeepSpeed checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint using DeepSpeed's checkpoint system.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            load_optimizer_states: Whether to load optimizer states
            load_lr_scheduler_states: Whether to load LR scheduler states
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        _, client_state = self.engine.load_checkpoint(
            str(checkpoint_path),
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states
        )
        
        metadata_path = checkpoint_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logging.info(f"DeepSpeed checkpoint loaded: {checkpoint_path}")
        return metadata
    
    # UNIFIED API - Device Management
    
    def to(self, device):
        """Device movement (no-op for DeepSpeed, handles internally)."""
        return self
    
    def cuda(self):
        """Move to CUDA (no-op for DeepSpeed)."""
        return self
    
    def cpu(self):
        """Move to CPU (no-op for DeepSpeed)."""
        return self
    
    # Backend-Specific Properties
    
    @property
    def module(self):
        """Access underlying model module."""
        return self.engine.module
    
    @property
    def world_size(self) -> int:
        """Get world size."""
        return self.engine.world_size
    
    @property
    def local_rank(self) -> int:
        """Get local rank."""
        return self.engine.local_rank
    
    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "deepspeed"
    
    # Utility Methods
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.engine.local_rank == 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {}
    
    def estimate_memory_usage(self):
        """Estimate memory usage for ZeRO-3."""
        if hasattr(self.config, 'zero_stage') and self.config.zero_stage == 3:
            try:
                estimate_zero3_model_states_mem_needs_all_live(
                    self.engine.module,
                    num_gpus_per_node=torch.cuda.device_count(),
                    num_nodes=1
                )
            except Exception as e:
                logging.warning(f"Could not estimate ZeRO-3 memory: {e}")
    
    def __repr__(self):
        return (f"DeepSpeedBackend(world_size={self.world_size}, "
                f"local_rank={self.local_rank}, "
                f"zero_stage={getattr(self.config, 'zero_stage', 'N/A')})")


# Factory Function

def create_deepspeed_backend(
    model: nn.Module,
    config: Any,
    model_parameters: Optional[list] = None
) -> DeepSpeedBackend:
    """
    Factory function to create DeepSpeed backend.
    
    Args:
        model: PyTorch model
        config: Configuration object
        model_parameters: Optional model parameters
        
    Returns:
        DeepSpeedBackend instance
    """
    return DeepSpeedBackend(model, config, model_parameters)