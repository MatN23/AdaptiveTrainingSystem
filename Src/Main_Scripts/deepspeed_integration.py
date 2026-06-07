# deepspeed_integration.py
"""
Integration layer between your training system and DeepSpeed backend.
Place this in your backend/ directory.
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path

from deepspeed_backend.zero_stage_manager import ZeROStageManager, ZeROConfig, create_zero_manager
from deepspeed_backend.optimizer_wrappers import create_zero_optimizer
from deepspeed_backend.scheduler_wrappers import create_scheduler
from deepspeed_backend.offload import OffloadConfig, create_offload_manager
from deepspeed_backend.checkpointing import ZeROCheckpoint


class DeepSpeedIntegration:
    """
    Integrates your DeepSpeed remake with the existing training system.
    """
    
    def __init__(self, config, model, expert_registry=None):
        self.config = config
        self.model = model
        self.expert_registry = expert_registry or {}
        
        self.zero_manager = None
        self.optimizer = None
        self.scheduler = None
        self.offload_manager = None
        self.checkpoint_manager = None
        
        self._setup_deepspeed()
    
    def _setup_deepspeed(self):
        """Setup all DeepSpeed components"""
        
        # 1. Setup offloading if requested
        if getattr(self.config, 'cpu_offload', False):
            offload_config = OffloadConfig(
                device='cpu' if not getattr(self.config, 'nvme_path', None) else 'nvme',
                offload_optimizer_states=True,
                offload_parameters=getattr(self.config, 'cpu_offload_parameters', False),
                nvme_path=getattr(self.config, 'nvme_path', None),
                pin_memory=True,
                async_offload=True,
            )
            self.offload_manager = create_offload_manager(offload_config, self.expert_registry)
            print(f" Offload manager initialized: {offload_config.device}")
        
        # 2. Create ZeRO-compatible optimizer
        zero_stage = getattr(self.config, 'zero_stage', 2)
        
        self.optimizer = create_zero_optimizer(
            optimizer_name='adamw',
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            zero_stage=zero_stage,
            partition_rank=0,  # Will be set by distributed init
            world_size=1,      # Will be set by distributed init
            expert_registry=self.expert_registry,
            gradient_clipping=getattr(self.config, 'max_grad_norm', None),
            offload_manager=self.offload_manager,
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
        )
        print(f" ZeRO-{zero_stage} optimizer initialized")
        
        # 3. Create ZeRO stage manager
        zero_config = ZeROConfig(
            stage=zero_stage,
            overlap_comm=True,
            cpu_offload=getattr(self.config, 'cpu_offload', False),
            nvme_offload=bool(getattr(self.config, 'nvme_path', None)),
            partition_size=int(1e9),
        )
        
        self.zero_manager = create_zero_manager(
            model=self.model,
            optimizer=self.optimizer,
            config=zero_config,
            expert_registry=self.expert_registry,
        )
        print(f" ZeRO stage manager initialized")
        
        # 4. Create scheduler
        total_steps = self._calculate_total_steps()
        warmup_steps = int(total_steps * getattr(self.config, 'warmup_ratio', 0.05))
        
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_name=getattr(self.config, 'lr_scheduler', 'cosine'),
            warmup_steps=warmup_steps,
            max_steps=total_steps,
            expert_registry=self.expert_registry,
            min_lr=getattr(self.config, 'min_lr', 0.0),
        )
        print(f" Scheduler initialized: {warmup_steps} warmup steps, {total_steps} total steps")
        
        # 5. Setup checkpoint manager
        checkpoint_dir = Path(f"experiments/{self.config.experiment_name}/checkpoints")
        self.checkpoint_manager = ZeROCheckpoint(
            checkpoint_dir=str(checkpoint_dir),
            zero_stage=zero_stage,
            expert_registry=self.expert_registry,
            save_optimizer=True,
            save_scheduler=True,
        )
        print(f" Checkpoint manager initialized: {checkpoint_dir}")
    
    def _calculate_total_steps(self):
        """Calculate total training steps"""
        # This should match your training calculation
        batch_size = self.config.batch_size
        grad_accum = getattr(self.config, 'gradient_accumulation_steps', 1)
        num_epochs = self.config.num_epochs
        
        # Estimate dataset size (you'll need to pass this in)
        dataset_size = getattr(self.config, 'dataset_size', 10000)
        
        steps_per_epoch = dataset_size // (batch_size * grad_accum)
        total_steps = steps_per_epoch * num_epochs
        
        return max(total_steps, 1000)  # Minimum 1000 steps
    
    def optimizer_step(self):
        """Execute optimizer step with ZeRO awareness"""
        self.zero_manager.step()
    
    def scheduler_step(self):
        """Execute scheduler step"""
        if self.scheduler:
            self.scheduler.step()
    
    def save_checkpoint(self, step, epoch, metadata=None):
        """Save checkpoint using ZeRO checkpoint manager"""
        return self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            epoch=epoch,
            metadata=metadata,
        )
    
    def load_checkpoint(self, checkpoint_path, strict=True):
        """Load checkpoint using ZeRO checkpoint manager"""
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=strict,
        )
    
    def get_memory_stats(self):
        """Get memory statistics"""
        stats = self.zero_manager.get_memory_stats()
        
        if self.offload_manager:
            if hasattr(self.offload_manager, 'get_memory_savings'):
                stats['offload'] = self.offload_manager.get_memory_savings()
            elif hasattr(self.offload_manager, 'get_stats'):
                stats['offload'] = self.offload_manager.get_stats()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.zero_manager:
            self.zero_manager.cleanup()
        
        if self.offload_manager and hasattr(self.offload_manager, 'cleanup'):
            self.offload_manager.cleanup()


def integrate_with_trainer(trainer, config, model, expert_registry=None):
    """
    Integrate DeepSpeed with your existing trainer.
    
    Usage in Main.py:
        from backend.deepspeed_integration import integrate_with_trainer
        
        # After creating trainer
        if config.use_deepspeed:
            deepspeed_integration = integrate_with_trainer(
                trainer=orchestrator.trainer,
                config=config,
                model=model,
                expert_registry={}
            )
    """
    integration = DeepSpeedIntegration(config, model, expert_registry)
    
    # Replace trainer's optimizer and scheduler
    trainer.optimizer = integration.optimizer
    trainer.scheduler = integration.scheduler
    
    trainer.deepspeed_integration = integration
    trainer.use_deepspeed = True
    
    # Monkey-patch optimizer step
    original_optimizer_step = trainer.optimizer.step
    def wrapped_optimizer_step(*args, **kwargs):
        integration.optimizer_step()
    trainer.optimizer.step = wrapped_optimizer_step
    
    # Monkey-patch scheduler step
    if trainer.scheduler:
        original_scheduler_step = trainer.scheduler.step
        def wrapped_scheduler_step(*args, **kwargs):
            integration.scheduler_step()
        trainer.scheduler.step = wrapped_scheduler_step
    
    print(" DeepSpeed integration completed")
    return integration