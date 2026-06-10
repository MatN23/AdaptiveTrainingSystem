# Licensed under the Custom License below.

import yaml
import os
import psutil
import torch
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class Config:
    """Enhanced configuration with validation and serialization."""
    # Model architecture
    vocab_size: int = 100352
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    seq_length: int = 1024
    intermediate_size: Optional[int] = None  # Auto-calculated if None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    
    # Training parameters
    batch_size: int = 2
    micro_batch_size: Optional[int] = None  # Auto-calculated if None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.15
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 1000
    precision: str = "auto"
    inference_precision: str = "auto"
    compile: bool = True
    compile_mode: str = "default"             # default, reduce-overhead, max-autotune
    
    # Data parameters
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_workers: int = 2
    assistant_loss_weight: float = 1.5
    max_conversations_per_file: int = 10000
    streaming_threshold_gb: float = 10.0
    prefetch_factor: int = 4
    pin_memory: bool = True
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    # Stability and optimization
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stable_embedding: bool = True
    gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_triton_fp8: bool = False
    
    # CUDA Optimization toggles
    use_cuda: str = "auto"                    # auto, True, False
    use_fused_rmsnorm: bool = False
    use_fused_rope: bool = True
    use_fused_swiglu: bool = True
    use_fused_moe: bool = True
    use_fused_loss: bool = True
    use_fused_grad_clip: bool = True
    validate_moe_cuda_indices: bool = False
    force_dense_expert_grads: bool = False
    routing_stats_update_interval: int = 64
    
    # MoE parameters
    use_moe: bool = True
    use_mod: bool = True
    mod_routing_stats_update_interval: int = 64
    num_experts: int = 8
    moe_top_k: int = 1
    capacity_factor: float = 1.5
    load_balancing_weight: float = 0.001
    expert_parallel_size: Optional[int] = None
    
    # DeepSpeed parameters
    use_deepspeed: bool = True
    zero_stage: int = 0
    cpu_offload: bool = False
    cpu_offload_optimizer: bool = False
    cpu_offload_parameters: bool = False
    aggressive_cpu_offload: bool = False
    nvme_path: Optional[str] = None
    nvme_offload_optimizer: bool = False
    nvme_offload_parameters: bool = False
    gradient_compression: bool = False
    communication_backend: str = "nccl"
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    allgather_bucket_size: int = 500000000
    reduce_bucket_size: int = 500000000
    
    # Production settings
    experiment_name: Optional[str] = None
    seed: int = 42
    log_level: str = "INFO"
    save_total_limit: int = 5
    early_stopping_patience: Optional[int] = None
    min_lr: float = 1e-6
    lr_scheduler: str = "cosine"
    use_lr_scheduler: bool = True
    
    # Monitoring and fault tolerance
    health_check_interval: int = 100
    auto_resume: bool = True
    backup_every_n_hours: int = 6
    max_retries: int = 3
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Adaptive Learning Rate Control
    enable_adaptive_lr: bool = True           # Master switch for adaptive LR adjustments
    allow_scheduler_override: bool = True     # Allow orchestrator to override scheduler
    min_override_threshold: float = 0.2       # Only override if change > 20%
    emergency_override_enabled: bool = True   # Always allow emergency LR reductions
    log_lr_decisions: bool = True             # Log all LR decision making
    
    # Advanced precision settings
    auto_tune_precision: bool = True
    precision_target: str = "balanced"
    dynamic_precision: bool = False
    tf32_enabled: Optional[bool] = None
    fp16_loss_scale: float = 65536.0
    bf16_enabled: bool = True
    
    # Memory optimization
    max_memory_usage: float = 0.9
    memory_cleanup_interval: int = 1000
    enable_cpu_adam: bool = False
    partition_activations: bool = False
    
    # Multi-node settings
    master_addr: Optional[str] = None
    master_port: int = 29500
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    
    # Data processing
    data_cache_dir: str = "data/cache"
    tokenizer_cache_dir: str = "tokenizers/cache"
    max_seq_length_percentile: float = 0.95
    
    # Checkpointing enhancements
    save_optimizer_states: bool = True
    checkpoint_compression: bool = True
    async_save: bool = True
    universal_checkpoint: bool = True
    
    # Performance profiling
    profile_memory: bool = False
    profile_communication: bool = False
    log_throughput: bool = True
    
    # Device-specific flags (internal use)
    _batch_size_set: bool = field(default=False, init=False, repr=False)
    _device_optimizations_applied: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        self.validate()
        self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure settings based on hardware and model size."""
        if self.experiment_name is None:
            self.experiment_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure vocab_size is divisible by 64 for efficiency
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size + 63) // 64) * 64
        
        # Auto-calculate intermediate_size if not specified (SwiGLU pattern)
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 8 / 3)
            # Round to nearest multiple of 64 for efficiency
            self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64
        
        # Auto-calculate micro_batch_size for DeepSpeed
        if self.micro_batch_size is None:
            self.micro_batch_size = max(1, self.batch_size // max(1, torch.cuda.device_count()))
        
        # Auto-configure expert parallelism
        if self.use_moe and self.expert_parallel_size is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.expert_parallel_size = min(self.num_experts, num_gpus)
        
        # Auto-configure DeepSpeed ZeRO stage based on model size
        if self.zero_stage == 0:
            estimated_params = self._estimate_parameters()
            gpu_memory_gb = self._get_gpu_memory_gb()
            
            if estimated_params > 50e9 or (gpu_memory_gb and estimated_params * 2 > gpu_memory_gb * 1e9):
                self.zero_stage = 3
                self.cpu_offload = True
            elif estimated_params > 10e9:
                self.zero_stage = 3
            elif estimated_params > 1e9:
                self.zero_stage = 2
            else:
                self.zero_stage = 1
        
        # Auto-enable aggressive optimizations for large models
        if self._estimate_parameters() > 50e9:
            self.gradient_compression = True
            self.cpu_offload_optimizer = True
            self.cpu_offload_parameters = True
        if self._estimate_parameters() > 1e9:
            self.enable_cpu_adam = True
        
        # Auto-configure precision based on hardware
        if self.precision == "auto":
            self.precision = self._auto_select_precision()
        
        if self.inference_precision == "auto":
            self.inference_precision = self._auto_select_precision(for_inference=True)
        
        # Auto-enable TF32 on Ampere+ GPUs
        if self.tf32_enabled is None:
            self.tf32_enabled = self._supports_tf32()
        
        # Calculate effective batch size
        world_size = self.world_size or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps * world_size
        
        # Auto-configure data loading
        if self.num_workers == 2:
            self.num_workers = min(os.cpu_count() or 4, 16)
        
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tokenizer_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        embed_params = self.vocab_size * self.hidden_size * 2
        attention_params_per_layer = (
            4 * self.hidden_size * self.hidden_size +
            self.hidden_size
        )
        
        if self.use_moe:
            ff_params_per_layer = (
                self.num_experts * 3 * self.hidden_size * self.intermediate_size +
                self.hidden_size * self.num_experts +
                self.hidden_size
            )
        else:
            ff_params_per_layer = (
                3 * self.hidden_size * self.intermediate_size +
                self.hidden_size
            )
        
        total_params = (
            embed_params + 
            self.num_layers * (attention_params_per_layer + ff_params_per_layer)
        )
        
        return total_params
    
    def _get_gpu_memory_gb(self) -> Optional[float]:
        """Get available GPU memory in GB."""
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3)
            except:
                return None
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS uses unified memory - return system memory
            try:
                mem = psutil.virtual_memory()
                return mem.total / (1024**3)
            except:
                return None
        return None
    
    def _auto_select_precision(self, for_inference: bool = False) -> str:
        """Auto-select precision based on hardware capabilities."""
        # MPS support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS best supports fp16 and fp32, bf16 support is limited
            return "fp16"  # FP16 for both training and inference on MPS
        
        # CPU fallback
        if not torch.cuda.is_available():
            return "fp32"
        
        try:
            compute_capability = torch.cuda.get_device_capability()[0]
            
            if compute_capability >= 8:  # Ampere and newer
                return "mixed_bf16" if not for_inference else "bf16"
            elif compute_capability >= 7:  # Volta and Turing
                return "mixed_fp16" if not for_inference else "fp16"
            else:  # Older GPUs
                return "fp32"
        except:
            return "fp32"
    
    def _supports_tf32(self) -> bool:
        """Check if current GPU supports TF32."""
        if not torch.cuda.is_available():
            return False
        
        try:
            # TF32 is supported on Ampere (compute capability 8.0) and newer
            compute_capability = torch.cuda.get_device_capability()[0]
            return compute_capability >= 8
        except:
            return False
    
    def apply_device_optimizations(self, device_type: str = None):
        """Apply device-specific optimizations to configuration."""
        if self._device_optimizations_applied:
            return  # Already applied
        
        if device_type is None:
            if torch.cuda.is_available():
                device_type = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = 'mps'
            else:
                device_type = 'cpu'
        
        print(f"Applying {device_type.upper()} optimizations...")
        
        if device_type == 'mps':
            # MPS-specific optimizations
            self.use_deepspeed = False
            self.use_flash_attention = False
            self.compile = False  # May have issues on MPS
            
            # Precision settings
            if self.precision == 'auto':
                self.precision = 'fp16'
            elif self.precision in ['bf16', 'mixed_bf16']:
                print(f"Warning: BF16 not fully supported on MPS, switching to FP16")
                self.precision = 'fp16'
            
            if self.inference_precision == 'auto':
                self.inference_precision = 'fp16'
            elif self.inference_precision in ['bf16', 'mixed_bf16']:
                self.inference_precision = 'fp16'
            
            # Memory and performance settings
            self.num_workers = 0  # MPS works best with main process data loading
            self.pin_memory = False
            
            # Reduce batch size if not explicitly set
            if not self._batch_size_set:
                if self.batch_size > 4:
                    print(f"Reducing batch size from {self.batch_size} to 4 for MPS")
                    self.batch_size = 4
            
            # Increase gradient accumulation to maintain effective batch size
            if self.gradient_accumulation_steps < 8:
                self.gradient_accumulation_steps = 8
            
            print("MPS optimizations applied:")
            print(f"  - Precision: {self.precision}")
            print(f"  - DeepSpeed: Disabled")
            print(f"  - Flash Attention: Disabled")
            print(f"  - Model Compile: Disabled")
            print(f"  - Batch Size: {self.batch_size}")
            print(f"  - Gradient Accumulation: {self.gradient_accumulation_steps}")
        
        elif device_type == 'cuda':
            # CUDA optimizations
            if self.precision == 'auto':
                self.precision = self._auto_select_precision()
            if self.inference_precision == 'auto':
                self.inference_precision = self._auto_select_precision(for_inference=True)
            
            # Enable TF32 if supported
            if self.tf32_enabled is None:
                self.tf32_enabled = self._supports_tf32()
            
            # CUDA-specific settings
            self.pin_memory = True
            
            print("CUDA optimizations applied:")
            print(f"  - Precision: {self.precision}")
            print(f"  - TF32: {self.tf32_enabled}")
            print(f"  - DeepSpeed: {self.use_deepspeed}")
            print(f"  - Flash Attention: {self.use_flash_attention}")
        
        else:  # CPU
            # CPU optimizations
            self.precision = 'fp32'
            self.inference_precision = 'fp32'
            self.use_deepspeed = False
            self.use_flash_attention = False
            self.compile = False
            self.pin_memory = False
            
            # Smaller batch size for CPU
            if not self._batch_size_set and self.batch_size > 2:
                print(f"Reducing batch size from {self.batch_size} to 2 for CPU")
                self.batch_size = 2
            
            print("CPU optimizations applied:")
            print(f"  - Precision: {self.precision}")
            print(f"  - Batch Size: {self.batch_size}")
            print(f"  - Warning: CPU training is significantly slower")
        
        self._device_optimizations_applied = True
    
    def get_device_compatibility_report(self) -> Dict[str, Any]:
        """Generate a compatibility report for current configuration and device."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config_name': self.experiment_name,
            'device_detection': {},
            'compatibility_issues': [],
            'recommendations': [],
            'feature_support': {},
        }
        
        # Detect devices
        report['device_detection'] = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'primary_device': 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
        }
        
        primary_device = report['device_detection']['primary_device']
        
        if primary_device == 'mps':
            report['feature_support'] = {
                'deepspeed': False,
                'flash_attention': False,
                'fp16': True,
                'bf16': False,
                'model_compile': 'limited',
                'gradient_checkpointing': True,
            }
            
            if self.use_deepspeed:
                report['compatibility_issues'].append("DeepSpeed is not supported on MPS")
                report['recommendations'].append("Set use_deepspeed=False")
            
            if self.use_flash_attention:
                report['compatibility_issues'].append("Flash Attention is not supported on MPS")
                report['recommendations'].append("Set use_flash_attention=False")
            
            if self.precision in ['bf16', 'mixed_bf16']:
                report['compatibility_issues'].append("BF16 has limited support on MPS")
                report['recommendations'].append("Use FP16 or FP32 instead")
            
            if self.batch_size > 4:
                report['recommendations'].append(f"Consider reducing batch size from {self.batch_size} to 4 or lower")
            
            # MPS memory check
            try:
                mem = psutil.virtual_memory()
                mem_gb = mem.total / (1024**3)
                if mem_gb < 16:
                    report['recommendations'].append(f"System has {mem_gb:.1f}GB RAM. 16GB+ recommended for MPS training")
            except:
                pass
        
        elif primary_device == 'cuda':
            report['feature_support'] = {
                'deepspeed': True,
                'flash_attention': True,
                'fp16': True,
                'bf16': True,
                'model_compile': True,
                'gradient_checkpointing': True,
            }
            
            try:
                compute_cap = torch.cuda.get_device_capability()[0]
                if compute_cap < 7 and self.precision in ['fp16', 'mixed_fp16']:
                    report['compatibility_issues'].append(f"GPU compute capability {compute_cap} may have limited FP16 support")
                    report['recommendations'].append("Consider using FP32 on older GPUs")
                
                if compute_cap < 8 and self.precision in ['bf16', 'mixed_bf16']:
                    report['compatibility_issues'].append(f"GPU compute capability {compute_cap} does not support BF16")
                    report['recommendations'].append("Use FP16 or FP32 instead")
            except:
                pass
        
        else:  # CPU
            report['feature_support'] = {
                'deepspeed': False,
                'flash_attention': False,
                'fp16': False,
                'bf16': 'limited',
                'model_compile': False,
                'gradient_checkpointing': True,
            }
            
            report['recommendations'].append("CPU training is slow - consider using GPU for production")
            if self.precision != 'fp32':
                report['recommendations'].append("Use FP32 precision on CPU")
        
        return report
    
    def get_active_parameters(self) -> int:
        """Get number of active parameters (important for MoE models)."""
        if not self.use_moe:
            return self._estimate_parameters()
        
        total_params = self._estimate_parameters()
        embed_params = self.vocab_size * self.hidden_size * 2
        attention_params = self.num_layers * (4 * self.hidden_size * self.hidden_size + self.hidden_size)
        routing_params = self.num_layers * self.hidden_size * self.num_experts
        norm_params = self.num_layers * self.hidden_size
        
        non_expert_params = embed_params + attention_params + routing_params + norm_params
        active_expert_params = self.num_layers * self.moe_top_k * 3 * self.hidden_size * self.intermediate_size
        
        return int(non_expert_params + active_expert_params)
    
    def validate(self):
        """Enhanced validation with better error messages."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        
        valid_precisions = ["fp32", "fp16", "bf16", "mixed_fp16", "mixed_bf16", "tf32", "auto"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.precision}. Valid options: {valid_precisions}")
        
        if self.inference_precision not in valid_precisions + ["dynamic", "int8"]:
            raise ValueError(f"Invalid inference_precision: {self.inference_precision}")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError("Warmup ratio must be between 0 and 1")
        
        if self.use_moe:
            valid_experts = [8, 16, 32, 64]
            if self.num_experts not in valid_experts:
                raise ValueError(f"Invalid num_experts={self.num_experts}. Choose from {valid_experts}.")
            
            if self.moe_top_k not in [1, 2]:
                raise ValueError("moe_top_k should be 1 or 2 (1 recommended for 8x pattern)")
            
            if self.moe_top_k > self.num_experts:
                raise ValueError("moe_top_k cannot exceed num_experts")
            
            if self.capacity_factor < 1.0:
                raise ValueError("capacity_factor must be at least 1.0")
            
            if self.expert_parallel_size and self.expert_parallel_size > self.num_experts:
                raise ValueError("expert_parallel_size cannot exceed num_experts")
        
        if self.use_deepspeed:
            if self.zero_stage not in [0, 1, 2, 3]:
                raise ValueError("zero_stage must be 0 (auto), 1, 2, or 3")
            
            if self.nvme_path and not Path(self.nvme_path).exists():
                raise ValueError(f"NVMe path does not exist: {self.nvme_path}")
        
        if not (0.1 <= self.max_memory_usage <= 1.0):
            raise ValueError("max_memory_usage must be between 0.1 and 1.0")
        
        if self.streaming_threshold_gb <= 0:
            raise ValueError("streaming_threshold_gb must be positive")
    
    def get_memory_estimate_gb(self) -> Dict[str, float]:
        """Estimate memory usage in GB for different components."""
        params = self._estimate_parameters()
        active_params = self.get_active_parameters()
        
        # Parameter memory (in bytes)
        if self.precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"]:
            param_bytes = params * 2
        else:
            param_bytes = params * 4
        
        grad_bytes = param_bytes
        optimizer_bytes = params * 4 * 2  # Adam: 2x parameters
        activation_bytes = (
            self.batch_size * self.seq_length * self.hidden_size * self.num_layers * 4
        )
        
        # Convert to GB
        param_gb = param_bytes / (1024**3)
        grad_gb = grad_bytes / (1024**3)
        optimizer_gb = optimizer_bytes / (1024**3)
        activation_gb = activation_bytes / (1024**3)
        
        total_gb = param_gb + grad_gb + optimizer_gb + activation_gb
        
        # Apply DeepSpeed optimizations
        if self.use_deepspeed:
            if self.zero_stage >= 2:
                grad_gb = grad_gb / max(1, torch.cuda.device_count())
            if self.zero_stage >= 3:
                param_gb = param_gb / max(1, torch.cuda.device_count())
            if self.cpu_offload_optimizer:
                optimizer_gb = 0
        
        return {
            "parameters": param_gb,
            "gradients": grad_gb, 
            "optimizer": optimizer_gb,
            "activations": activation_gb,
            "total": param_gb + grad_gb + optimizer_gb + activation_gb,
            "active_parameters": active_params,
            "total_parameters": params
        }
    
    def save(self, path: str):
        """Save configuration to file with metadata."""
        config_dict = asdict(self)
        
        # Remove internal fields
        config_dict.pop('_batch_size_set', None)
        config_dict.pop('_device_optimizations_applied', None)
        
        config_dict["_metadata"] = {
            "created": datetime.now().isoformat(),
            "lumina_version": "2.0",
            "estimated_parameters": self._estimate_parameters(),
            "active_parameters": self.get_active_parameters(),
            "memory_estimate": self.get_memory_estimate_gb()
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove metadata if present
        config_dict.pop("_metadata", None)
        config_dict.pop("_batch_size_set", None)
        config_dict.pop("_device_optimizations_applied", None)
        
        return cls(**config_dict)
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        ds_config = {
            "train_batch_size": self.effective_batch_size,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": getattr(self, 'max_grad_norm', 1.0),
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8
                }
            },
            
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": int(self.warmup_ratio * 10000),
                    "total_num_steps": 10000
                }
            },
            
            "communication_data_type": "fp32",
            "gradient_compression": {
                "enabled": self.gradient_compression
            },
            
            "wall_clock_breakdown": False,
            "memory_breakdown": self.profile_memory
        }
        
        # Precision settings
        if self.precision in ["fp16", "mixed_fp16"]:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": self.fp16_loss_scale,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "auto_cast": False
            }
        elif self.precision in ["bf16", "mixed_bf16"]:
            ds_config["bf16"] = {
                "enabled": True
            }
        
        # ZeRO configuration
        if self.zero_stage > 0:
            zero_config = {
                "stage": self.zero_stage,
                "overlap_comm": self.overlap_comm,
                "contiguous_gradients": self.contiguous_gradients,
                "sub_group_size": 1000000000,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": self.allgather_bucket_size,
                "stage3_prefetch_bucket_size": 50000000,
                "stage3_param_persistence_threshold": 100000,
                "stage3_max_live_parameters": 1000000000,
                "stage3_max_reuse_distance": 1000000000
            }
            
            if self.cpu_offload:
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            if self.cpu_offload_parameters:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            if self.nvme_path:
                if self.nvme_offload_optimizer:
                    zero_config["offload_optimizer"] = {
                        "device": "nvme",
                        "nvme_path": self.nvme_path,
                        "pin_memory": True
                    }
                if self.nvme_offload_parameters:
                    zero_config["offload_param"] = {
                        "device": "nvme", 
                        "nvme_path": self.nvme_path,
                        "pin_memory": True
                    }
            
            ds_config["zero_optimization"] = zero_config
        # Activation checkpointing
        if self.gradient_checkpointing:
            ds_config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            }
        
        return ds_config


class ConfigPresets:
    """Updated configuration presets following 8x MoE pattern with MPS support."""
    
    @staticmethod
    def debug() -> Config:
        """Minimal config for debugging and testing."""
        return Config(
            # Tiny model for fast iteration
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            seq_length=256,
            intermediate_size=256,
            
            # Fast training settings
            batch_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=0,
            
            # MoE settings - 8x pattern but smaller
            use_moe=True,
            num_experts=32,
            moe_top_k=2,
            capacity_factor=1.1,
            load_balancing_weight=0.005,
            expert_parallel_size=2,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=1,
            cpu_offload=False,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler="cosine",
            gradient_checkpointing=False,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.7,
            streaming_threshold_gb=1.0
        )
    
    def debug_300m() -> Config:
        """Minimal config for debugging and testing."""
        return Config(
            # Tiny model for fast iteration
            hidden_size=786,
            num_layers=6,
            num_heads=4,
            num_kv_heads=4,
            seq_length=512,
            intermediate_size=768,
            
            # Fast training settings
            batch_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=1,
            learning_rate=5e-5,
            weight_decay=0.01,
            eval_every_n_batches=50,
            save_every_n_batches=100,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=0,
            
            # MoE settings - 8x pattern but smaller
            use_moe=True,
            num_experts=8,
            moe_top_k=2,
            capacity_factor=1.1,
            load_balancing_weight=0.005,
            expert_parallel_size=2,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=1,
            cpu_offload=False,
            
            # Monitoring and stability
            experiment_name="debug_run",
            log_level="DEBUG",
            health_check_interval=10,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler="cosine",
            gradient_checkpointing=False,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.7,
            streaming_threshold_gb=1.0
        )
    
    @staticmethod
    def moe_stress_test() -> Config:
        """Config designed specifically to benchmark MoE CUDA vs PyTorch routing."""
        return Config(
            # Bigger than debug, but nowhere near full model scale
            hidden_size=768,              # routing starts to matter here
            num_layers=6,                 # gives depth + stable timing
            num_heads=8,
            num_kv_heads=2,
            seq_length=256,

            # MoE-heavy FFN (forces dispatch + combine to be real work)
            intermediate_size=4096,

            # Enough tokens to stress routing, but still fast
            batch_size=4,
            micro_batch_size=1,
            gradient_accumulation_steps=1,

            # MoE settings  THIS is what makes the CUDA path shine
            use_moe=True,
            num_experts=32,               # enough to make routing nontrivial
            moe_top_k=2,                  # makes dispatch 2 heavier
            capacity_factor=1.25,         # realistic routing load
            load_balancing_weight=0.01,
            expert_parallel_size=4,       # uses EP kernels if you have them

            # Performance toggles
            compile=True,
            use_flash_attention=True,
            gradient_checkpointing=False, # keep timing consistent

            # Precision + stability
            precision="auto",
            inference_precision="auto",
            auto_tune_precision=False,
            dynamic_precision=False,

            # Training loop params (mostly irrelevant for benchmarking)
            num_epochs=1,
            learning_rate=2e-4,
            weight_decay=0.01,

            # Monitoring
            experiment_name="moe_stress_test",
            log_level="INFO",
            eval_every_n_batches=200,
            save_every_n_batches=500,
            save_total_limit=1,

            # Infra
            use_deepspeed=False,          # avoids DS overhead in timings
            zero_stage=0,
            cpu_offload=False,

            max_memory_usage=0.9,
            streaming_threshold_gb=1.5,
        )

    
    @staticmethod
    def debug_200m() -> Config:
        """~200M hybrid MoE model tuned for T4 GPUs."""
        return Config(
            # Model size
            hidden_size=640,
            num_layers=12,
            num_heads=8,
            num_kv_heads=8,
            seq_length=512,
            intermediate_size=2560,

            # Training
            batch_size=4,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            num_epochs=3,
            learning_rate=3e-5,
            weight_decay=0.01,
            eval_every_n_batches=100,
            save_every_n_batches=250,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=2,

            # MoE
            use_moe=False,
            use_mod=True,
            num_experts=32,
            moe_top_k=2,
            capacity_factor=1.2,
            load_balancing_weight=0.01,
            expert_parallel_size=1,

            use_deepspeed=True,
            zero_stage=1,
            cpu_offload=False,

            # Monitoring and stability
            experiment_name="debug_200m",
            log_level="INFO",
            health_check_interval=15,
            save_total_limit=3,
            early_stopping_patience=None,
            max_retries=1,
            lr_scheduler="cosine",
            gradient_checkpointing=True,

            # Precision tuning
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,

            # Memory and performance
            max_memory_usage=0.8,
            streaming_threshold_gb=1.0
        )
    
    @staticmethod
    def b1() -> Config:
        """1B active parameter model (8x1B = 8B total)."""
        return Config(
            # Model architecture for ~1B active parameters
            hidden_size=1908,
            num_layers=31,
            num_heads=12,
            num_kv_heads=4,
            seq_length=2048,
            intermediate_size=None,
            
            # Training settings
            batch_size=8,
            micro_batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=3,
            learning_rate=3e-4,
            weight_decay=0.01,
            eval_every_n_batches=500,
            save_every_n_batches=1000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=4,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.01,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=2,
            cpu_offload=False,
            gradient_compression=False,
            
            # Production settings
            experiment_name="b1_8x1b",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=5,
            early_stopping_patience=5,
            backup_every_n_hours=12,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.85,
            streaming_threshold_gb=5.0
        )
    
    @staticmethod
    def b7() -> Config:
        """7B active parameter model (8x7B = 56B total) - Mixtral-style."""
        return Config(
            # Model architecture for ~7B active parameters
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            seq_length=4096,
            intermediate_size=None,
            
            # Training settings
            batch_size=16,
            micro_batch_size=1,
            gradient_accumulation_steps=8,
            num_epochs=3,
            learning_rate=1e-4,
            weight_decay=0.01,
            eval_every_n_batches=1000,
            save_every_n_batches=2000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=8,
            
            # MoE settings - 8x pattern like Mixtral
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.01,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=0,
            cpu_offload=False,
            gradient_compression=False,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b7_8x7b_mixtral",
            log_level="INFO",
            health_check_interval=100,
            save_total_limit=10,
            early_stopping_patience=10,
            backup_every_n_hours=6,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            
            # Precision settings
            auto_tune_precision=True,
            precision_target="balanced",
            dynamic_precision=False,
            
            # Memory settings
            max_memory_usage=0.9,
            streaming_threshold_gb=10.0,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=False,
            log_throughput=True
        )
    
    @staticmethod
    def b14() -> Config:
        """14B active parameter model (8x14B = 112B total)."""
        return Config(
            # Model architecture for ~14B active parameters
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            seq_length=4096,
            intermediate_size=None,
            
            # Training settings
            batch_size=32,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            num_epochs=2,
            learning_rate=8e-5,
            weight_decay=0.01,
            eval_every_n_batches=2000,
            save_every_n_batches=5000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=8,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.015,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=0,
            cpu_offload=False,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b14_8x14b",
            log_level="INFO",
            health_check_interval=200,
            save_total_limit=15,
            early_stopping_patience=15,
            backup_every_n_hours=4,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.05,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=None,
            
            # Memory settings
            max_memory_usage=0.9,
            streaming_threshold_gb=25.0,
            enable_cpu_adam=False,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            log_throughput=True
        )
    
    @staticmethod
    def b30() -> Config:
        """30B active parameter model (8x30B = 240B total)."""
        return Config(
            # Model architecture for ~30B active parameters
            hidden_size=6656,
            num_layers=48,
            num_heads=52,
            num_kv_heads=13,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=64,
            micro_batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=2,
            learning_rate=6e-5,
            weight_decay=0.01,
            eval_every_n_batches=3000,
            save_every_n_batches=10000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=12,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.02,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b30_8x30b",
            log_level="INFO",
            health_check_interval=300,
            save_total_limit=20,
            early_stopping_patience=20,
            backup_every_n_hours=3,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.03,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=50.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b50() -> Config:
        """50B active parameter model (8x50B = 400B total)."""
        return Config(
            # Model architecture for ~50B active parameters
            hidden_size=8192,
            num_layers=56,
            num_heads=64,
            num_kv_heads=16,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=128,
            micro_batch_size=1,
            gradient_accumulation_steps=64,
            num_epochs=2,
            learning_rate=4e-5,
            weight_decay=0.01,
            eval_every_n_batches=5000,
            save_every_n_batches=15000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.02,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            
            # Production settings
            experiment_name="b50_8x50b",
            log_level="INFO",
            health_check_interval=500,
            save_total_limit=25,
            early_stopping_patience=25,
            backup_every_n_hours=2,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.02,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=100.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b75() -> Config:
        """75B active parameter model (8x75B = 600B total)."""
        return Config(
            # Model architecture for ~75B active parameters
            hidden_size=10240,
            num_layers=64,
            num_heads=80,
            num_kv_heads=20,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=256,
            micro_batch_size=1,
            gradient_accumulation_steps=128,
            num_epochs=1,
            learning_rate=3e-5,
            weight_decay=0.01,
            eval_every_n_batches=8000,
            save_every_n_batches=20000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.025,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            nvme_offload_optimizer=False,
            nvme_offload_parameters=False,
            
            # Production settings
            experiment_name="b75_8x75b",
            log_level="INFO",
            health_check_interval=800,
            save_total_limit=30,
            early_stopping_patience=30,
            backup_every_n_hours=2,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.02,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=150.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b100() -> Config:
        """100B active parameter model (8x100B = 800B total)."""
        return Config(
            # Model architecture for ~100B active parameters
            hidden_size=12288,
            num_layers=72,
            num_heads=96,
            num_kv_heads=24,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=512,
            micro_batch_size=1,
            gradient_accumulation_steps=256,
            num_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_every_n_batches=10000,
            save_every_n_batches=25000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.025,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            nvme_offload_optimizer=False,
            nvme_offload_parameters=False,
            
            # Production settings
            experiment_name="b100_8x100b",
            log_level="INFO",
            health_check_interval=1000,
            save_total_limit=40,
            early_stopping_patience=40,
            backup_every_n_hours=1,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.01,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=200.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b200() -> Config:
        """200B active parameter model (8x200B = 1.6T total)."""
        return Config(
            # Model architecture for ~200B active parameters
            hidden_size=16384,
            num_layers=88,
            num_heads=128,
            num_kv_heads=32,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=1024,
            micro_batch_size=1,
            gradient_accumulation_steps=512,
            num_epochs=1,
            learning_rate=1.5e-5,
            weight_decay=0.01,
            eval_every_n_batches=15000,
            save_every_n_batches=30000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.03,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            nvme_offload_optimizer=False,
            nvme_offload_parameters=False,
            
            # Production settings
            experiment_name="b200_8x200b",
            log_level="INFO",
            health_check_interval=2000,
            save_total_limit=50,
            early_stopping_patience=50,
            backup_every_n_hours=1,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.01,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=400.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def b300() -> Config:
        """300B active parameter model (8x300B = 2.4T total)."""
        return Config(
            # Model architecture for ~300B active parameters
            hidden_size=20480,
            num_layers=96,
            num_heads=160,
            num_kv_heads=40,
            seq_length=8192,
            intermediate_size=None,
            
            # Training settings
            batch_size=2048,
            micro_batch_size=1,
            gradient_accumulation_steps=1024,
            num_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            eval_every_n_batches=20000,
            save_every_n_batches=40000,
            precision="auto",
            inference_precision="auto",
            compile=True,
            num_workers=16,
            
            # MoE settings - 8x pattern
            use_moe=False,
            use_mod=True,
            num_experts=8,
            moe_top_k=1,
            capacity_factor=1.25,
            load_balancing_weight=0.03,
            
            # DeepSpeed settings
            use_deepspeed=True,
            zero_stage=3,
            cpu_offload=True,
            cpu_offload_optimizer=True,
            cpu_offload_parameters=True,
            aggressive_cpu_offload=True,
            gradient_compression=True,
            overlap_comm=True,
            contiguous_gradients=True,
            nvme_offload_optimizer=False,
            nvme_offload_parameters=False,
            
            # Production settings
            experiment_name="b300_8x300b",
            log_level="INFO",
            health_check_interval=3000,
            save_total_limit=60,
            early_stopping_patience=60,
            backup_every_n_hours=1,
            lr_scheduler="cosine",
            gradient_checkpointing=True,
            use_flash_attention=True,
            warmup_ratio=0.01,
            
            # Precision settings
            auto_tune_precision=False,
            precision_target="quality",
            dynamic_precision=False,
            tf32_enabled=True,
            
            # Memory settings
            max_memory_usage=0.95,
            streaming_threshold_gb=600.0,
            enable_cpu_adam=True,
            partition_activations=True,
            
            # Monitoring
            enable_wandb=False,
            profile_memory=True,
            profile_communication=True,
            log_throughput=True
        )
    
    @staticmethod
    def get_preset_info() -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all available presets."""
        return {
            "debug": {
                "description": "Minimal configuration for debugging and testing",
                "use_case": "Development and debugging",
                "active_params": "~500K",
                "total_params": "~4M (8x500K)",
                "memory_usage": "Very Low (~10MB)",
                "training_speed": "Very Fast",
                "precision": "Auto-detected",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-1",
                "hardware": "Any GPU/MPS/CPU",
                "mps_compatible": True
            },
            "debug_200m": {
                "description": "200M model for quick testing",
                "use_case": "Quick prototyping",
                "active_params": "~200M",
                "total_params": "~6B",
                "memory_usage": "Low (~2GB)",
                "training_speed": "Fast",
                "precision": "Auto-detected",
                "moe_pattern": "32x experts, top-2 routing",
                "deepspeed": "ZeRO-1",
                "hardware": "T4/MPS/Mid-range GPU",
                "mps_compatible": True
            },
            "b1": {
                "description": "1B active parameter model (8x1B total)",
                "use_case": "Resource-constrained environments, experimentation",
                "active_params": "~1B",
                "total_params": "~8B (8x1B)",
                "memory_usage": "Medium (~4-8GB)",
                "training_speed": "Fast",
                "precision": "Auto-detected (fp16/bf16)",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-2 with auto-optimization",
                "hardware": "RTX 3090/4090, A6000, M1/M2/M3 Max",
                "mps_compatible": True
            },
            "b7": {
                "description": "7B active parameter model (8x7B total) - Mixtral-style",
                "use_case": "General purpose training, production applications",
                "active_params": "~7B",
                "total_params": "~56B (8x7B)",
                "memory_usage": "High (~28GB)",
                "training_speed": "Medium",
                "precision": "Auto-detected mixed precision",
                "moe_pattern": "8x experts, top-1 routing (Mixtral pattern)",
                "deepspeed": "Auto ZeRO stage with CPU offload",
                "hardware": "A100-40GB/80GB, M2/M3 Ultra (with optimizations)",
                "mps_compatible": True
            },
            "b14": {
                "description": "14B active parameter model (8x14B total)",
                "use_case": "High-performance applications, research",
                "active_params": "~14B",
                "total_params": "~112B (8x14B)",
                "memory_usage": "Very High (~56GB)",
                "training_speed": "Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with CPU offload",
                "hardware": "Multiple A100-80GB or H100",
                "mps_compatible": False
            },
            "b30": {
                "description": "30B active parameter model (8x30B total)",
                "use_case": "Advanced research, large-scale applications",
                "active_params": "~30B",
                "total_params": "~240B (8x30B)",
                "memory_usage": "Extreme (~120GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU offload",
                "hardware": "Multiple A100-80GB or H100 (8+ GPUs)",
                "mps_compatible": False
            },
            "b50": {
                "description": "50B active parameter model (8x50B total)",
                "use_case": "Enterprise-scale research and applications",
                "active_params": "~50B",
                "total_params": "~400B (8x50B)",
                "memory_usage": "Massive (~200GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU offload",
                "hardware": "Multiple H100 (16+ GPUs)",
                "mps_compatible": False
            },
            "b75": {
                "description": "75B active parameter model (8x75B total)",
                "use_case": "Frontier research, state-of-the-art performance",
                "active_params": "~75B",
                "total_params": "~600B (8x75B)",
                "memory_usage": "Massive (~300GB)",
                "training_speed": "Very Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU offload",
                "hardware": "Large H100 cluster (32+ GPUs)",
                "mps_compatible": False
            },
            "b100": {
                "description": "100B active parameter model (8x100B total)",
                "use_case": "Cutting-edge research, flagship models",
                "active_params": "~100B",
                "total_params": "~800B (8x100B)",
                "memory_usage": "Massive (~400GB)",
                "training_speed": "Extremely Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU offload",
                "hardware": "Large H100 cluster (64+ GPUs)",
                "mps_compatible": False
            },
            "b200": {
                "description": "200B active parameter model (8x200B total)",
                "use_case": "Frontier research, extremely large-scale models",
                "active_params": "~200B",
                "total_params": "~1.6T (8x200B)",
                "memory_usage": "Extreme (~800GB)",
                "training_speed": "Extremely Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU/NVMe offload",
                "hardware": "Large H100/H200 cluster (128+ GPUs)",
                "mps_compatible": False
            },
            "b300": {
                "description": "300B active parameter model (8x300B total)",
                "use_case": "Ultra-scale research, breaking boundaries",
                "active_params": "~300B",
                "total_params": "~2.4T (8x300B)",
                "memory_usage": "Extreme (~1.2TB)",
                "training_speed": "Extremely Slow",
                "precision": "Mixed BF16 with TF32",
                "moe_pattern": "8x experts, top-1 routing",
                "deepspeed": "ZeRO-3 with aggressive CPU/NVMe offload",
                "hardware": "Massive H100/H200 cluster (256+ GPUs)",
                "mps_compatible": False
            }
        }
    
    @staticmethod
    def get_mps_compatible_presets() -> List[str]:
        """Get list of MPS-compatible presets."""
        return ['debug', 'debug_200m', 'b1', 'b7']
    
    @staticmethod
    def compare_presets() -> Dict[str, Any]:
        """Compare all presets across key dimensions with 8x MoE considerations."""
        presets = {
            'debug': ConfigPresets.debug(),
            'debug_200m': ConfigPresets.debug_200m(),
            'b1': ConfigPresets.b1(),
            'b7': ConfigPresets.b7(),
            'b14': ConfigPresets.b14(),
            'b30': ConfigPresets.b30(),
            'b50': ConfigPresets.b50(),
            'b75': ConfigPresets.b75(),
            'b100': ConfigPresets.b100(),
            'b200': ConfigPresets.b200(),
            'b300': ConfigPresets.b300(),
        }
        
        comparison = {}
        
        for name, config in presets.items():
            memory_est = config.get_memory_estimate_gb()
            
            comparison[name] = {
                'active_parameters': memory_est['active_parameters'],
                'total_parameters': memory_est['total_parameters'], 
                'parameter_efficiency': f"{memory_est['active_parameters'] / memory_est['total_parameters']:.1%}",
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'sequence_length': config.seq_length,
                'micro_batch_size': config.micro_batch_size,
                'effective_batch_size': config.effective_batch_size,
                'learning_rate': config.learning_rate,
                'precision': config.precision,
                'inference_precision': config.inference_precision,
                'moe_pattern': f"{config.num_experts}x experts, top-{config.moe_top_k}",
                'capacity_factor': config.capacity_factor,
                'load_balancing_weight': config.load_balancing_weight,
                'deepspeed_stage': f"ZeRO-{config.zero_stage}" if config.zero_stage > 0 else "Auto",
                'cpu_offload': config.cpu_offload,
                'flash_attention': config.use_flash_attention,
                'estimated_memory_gb': memory_est['total'],
                'streaming_threshold_gb': config.streaming_threshold_gb,
                'mps_compatible': name in ConfigPresets.get_mps_compatible_presets()
            }
        
        return comparison


class ConfigManager:
    """Enhanced configuration management with hardware optimization and validation."""
    
    @staticmethod
    def create_config(
        preset: str = "b7",
        overrides: Optional[Dict[str, Any]] = None,
        optimize_for_hardware: bool = True,
        target_memory_usage: float = 0.9
    ) -> Config:
        """Create a configuration with optional overrides and hardware optimization."""
        
        # Get base configuration
        preset_methods = {
            'debug': ConfigPresets.debug,
            'debug_200m': ConfigPresets.debug_200m,
            'b1': ConfigPresets.b1,
            'b7': ConfigPresets.b7,
            'b14': ConfigPresets.b14,
            'b30': ConfigPresets.b30,
            'b50': ConfigPresets.b50,
            'b75': ConfigPresets.b75,
            'b100': ConfigPresets.b100,
            'b200': ConfigPresets.b200,
            'b300': ConfigPresets.b300,
        }
        
        if preset in preset_methods:
            config = preset_methods[preset]()
        else:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}")
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    # Track if batch size was explicitly set
                    if key == 'batch_size':
                        config._batch_size_set = True
                else:
                    print(f"Warning: Unknown config parameter '{key}' ignored")
        
        # Hardware optimization
        if optimize_for_hardware:
            config = ConfigManager.optimize_for_hardware(config, target_memory_usage)
        
        return config
    
    @staticmethod
    def optimize_for_hardware(config: Config, target_memory_usage: float = 0.9) -> Config:
        """Optimize configuration based on available hardware."""
        
        # Detect primary device
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
        
        print(f"Optimizing for {device_type.upper()} hardware...")
        
        # Apply device-specific optimizations
        config.apply_device_optimizations(device_type)
        
        memory_est = config.get_memory_estimate_gb()
        required_memory = memory_est['total']
        
        # Get available memory
        if device_type == 'cuda':
            gpu_count = torch.cuda.device_count()
            gpu_memory_gb = config._get_gpu_memory_gb() or 8
            total_memory_gb = gpu_memory_gb * gpu_count
            
            print(f"Hardware detected: {gpu_count} GPUs, {gpu_memory_gb:.1f}GB each")
        elif device_type == 'mps':
            try:
                mem = psutil.virtual_memory()
                total_memory_gb = mem.total / (1024**3)
            except:
                total_memory_gb = 16.0
            
            print(f"Hardware detected: Apple Silicon with {total_memory_gb:.1f}GB unified memory")
        else:
            try:
                mem = psutil.virtual_memory()
                total_memory_gb = mem.total / (1024**3)
            except:
                total_memory_gb = 8.0
            
            print(f"Hardware detected: CPU with {total_memory_gb:.1f}GB memory")
        
        print(f"Required memory: {required_memory:.1f}GB, Available: {total_memory_gb:.1f}GB")
        
        # Enable CPU offloading if needed (not applicable for MPS)
        if device_type == 'cuda' and required_memory > total_memory_gb * target_memory_usage:
            print("Enabling CPU offloading due to memory constraints")
            config.cpu_offload = True
            config.cpu_offload_optimizer = True
            if required_memory > total_memory_gb * 1.5:
                config.cpu_offload_parameters = True
                config.aggressive_cpu_offload = True
        
        # Adjust ZeRO stage for CUDA (not applicable for MPS)
        if device_type == 'cuda' and config.zero_stage == 0:
            if required_memory > total_memory_gb * 2:
                config.zero_stage = 3
            elif required_memory > total_memory_gb:
                config.zero_stage = 2
            else:
                config.zero_stage = 1
        
        # Enable gradient compression for multi-GPU setups
        if device_type == 'cuda' and torch.cuda.device_count() > 4:
            config.gradient_compression = True
        
        # Optimize expert parallelism for MoE
        if config.use_moe and config.expert_parallel_size is None:
            if device_type == 'cuda':
                gpu_count = torch.cuda.device_count()
                config.expert_parallel_size = min(config.num_experts, gpu_count)
            else:
                config.expert_parallel_size = 1
        
        print(f"Optimized config: Precision={config.precision}, CPU offload={config.cpu_offload}")
        
        return config
    
    @staticmethod
    def validate_config(config: Config, strict: bool = False) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        try:
            config.validate()
        except ValueError as e:
            if strict:
                raise
            warnings.append(f"Validation error: {e}")
        
        # Hardware compatibility checks
        if torch.cuda.is_available():
            gpu_memory_gb = config._get_gpu_memory_gb()
            memory_est = config.get_memory_estimate_gb()
            
            if gpu_memory_gb and memory_est['total'] > gpu_memory_gb * 0.9:
                warnings.append(f"High memory usage: {memory_est['total']:.1f}GB > {gpu_memory_gb * 0.9:.1f}GB")
                warnings.append("Consider enabling CPU offloading or reducing batch size")
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS-specific validation
            if config.use_deepspeed:
                warnings.append("DeepSpeed is not supported on MPS - will be disabled")
            if config.use_flash_attention:
                warnings.append("Flash Attention is not supported on MPS - will be disabled")
            if config.precision in ['bf16', 'mixed_bf16']:
                warnings.append("BF16 has limited support on MPS - recommend FP16 or FP32")
            
            try:
                mem = psutil.virtual_memory()
                mem_gb = mem.total / (1024**3)
                memory_est = config.get_memory_estimate_gb()
                
                if memory_est['total'] > mem_gb * 0.7:
                    warnings.append(f"High memory usage on MPS: {memory_est['total']:.1f}GB")
                    warnings.append("Consider reducing batch size or model size")
            except:
                pass
        
        # MoE-specific validations
        if config.use_moe:
            if config.num_experts != 8:
                warnings.append("Non-standard expert count: 8 experts recommended for consistency")
            
            if config.moe_top_k != 1:
                warnings.append("Top-1 routing recommended for 8x MoE pattern efficiency")
        
        # DeepSpeed compatibility
        if config.use_deepspeed and config.zero_stage > 2 and not config.cpu_offload:
            warnings.append("ZeRO-3 typically requires CPU offloading for optimal performance")
        
        return warnings
    
    @staticmethod 
    def save_config_with_metadata(config: Config, path: str):
        """Save configuration with comprehensive metadata."""
        config_dict = asdict(config)
        
        # Remove internal fields
        config_dict.pop('_batch_size_set', None)
        config_dict.pop('_device_optimizations_applied', None)
        
        # Add comprehensive metadata
        config_dict["_metadata"] = {
            "created": datetime.now().isoformat(),
            "lumina_version": "2.0",
            "framework": "PyTorch + DeepSpeed",
            "moe_pattern": f"{config.num_experts}x experts, top-{config.moe_top_k} routing",
            "estimated_parameters": {
                "total": config._estimate_parameters(),
                "active": config.get_active_parameters(),
                "efficiency": f"{config.get_active_parameters() / config._estimate_parameters():.1%}"
            },
            "memory_estimate_gb": config.get_memory_estimate_gb(),
            "hardware_requirements": {
                "min_gpu_memory_gb": config.get_memory_estimate_gb()['total'] / max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1),
                "recommended_gpus": max(1, int(config.get_memory_estimate_gb()['total'] / 40)),
                "supports_mps": config.experiment_name in ConfigPresets.get_mps_compatible_presets(),
                "supports_cpu_offload": config.cpu_offload,
            },
            "training_characteristics": {
                "effective_batch_size": config.effective_batch_size,
                "gradient_accumulation": config.gradient_accumulation_steps,
                "precision": config.precision,
                "deepspeed_stage": f"ZeRO-{config.zero_stage}" if config.zero_stage > 0 else "Auto"
            }
        }
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"Configuration saved to {path}")
        print(f"Active parameters: {config.get_active_parameters():,}")
        print(f"Total parameters: {config._estimate_parameters():,}")
        print(f"Parameter efficiency: {config.get_active_parameters() / config._estimate_parameters():.1%}")