# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import logging
import traceback
import psutil
import gc
import json
import time
import math
import signal
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import warnings
import builtins
import sys

builtins.exit = sys.exit
builtins.quit = sys.exit  # just in case some code calls quit()


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utility modules
try:
    from utils.data_processing import process_oasst_data, validate_data_comprehensive, create_sample_data
    from utils.environment import validate_environment, estimate_training_time, get_system_info
    from utils.reporting import create_data_summary_report, create_training_report
    UTILS_AVAILABLE = True
    print("✓ Utils modules loaded successfully")
except ImportError as e:
    UTILS_AVAILABLE = False
    print(f"⚠ Utils modules not available: {e}")

try:
    from backend.backend_deepspeed import create_deepspeed_backend
    from backend.backend_fsdp import create_fsdp_backend
    BACKEND_DEEPSPEED_AVAILABLE = True
    BACKEND_FSDP_AVAILABLE = True
except ImportError:
    BACKEND_DEEPSPEED_AVAILABLE = False
    BACKEND_FSDP_AVAILABLE = False

try:
    from backend.backend_colossalai import create_colossalai_backend
    BACKEND_COLOSSALAI_AVAILABLE = True
except ImportError:
    BACKEND_COLOSSALAI_AVAILABLE = False

try:
    from deepspeed_integration import integrate_with_trainer
    DEEPSPEED_REMAKE_AVAILABLE = True
    print("✓ DeepSpeed remake available")
except ImportError:
    DEEPSPEED_REMAKE_AVAILABLE = False
    print("⚠ DeepSpeed remake not available")

try:
    from training.chinchilla_scaler import EnhancedChinchillaScaler
    CHINCHILLA_SCALER_AVAILABLE = True
    print("✓ Chinchilla scaler available")
except ImportError:
    CHINCHILLA_SCALER_AVAILABLE = False
    print("⚠ Chinchilla scaler not available")

# DeepSpeed imports
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logging.warning("DeepSpeed not available")

# Import our modules with fallbacks
try:
    # Direct import from config.config_manager
    from config.config_manager import Config, ConfigPresets
    print("✓ Config loaded from config.config_manager")
except ImportError as e:
    print(f"Failed to import from config.config_manager: {e}")
    try:
        # Try adding config to path and importing directly
        import sys
        from pathlib import Path
        config_dir = Path(__file__).parent / 'config'
        if str(config_dir) not in sys.path:
            sys.path.insert(0, str(config_dir))
        from config_manager import Config, ConfigPresets
        print("✓ Config loaded from config_manager (direct)")
    except ImportError as e2:
        print("ERROR: Could not import config classes")
        print(f"  Primary error: {e}")
        print(f"  Fallback error: {e2}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Script location: {Path(__file__).parent}")
        config_path = Path(__file__).parent / 'config' / 'config_manager.py'
        print(f"  Config file exists: {config_path.exists()}")
        if config_path.exists():
            print(f"  Config file path: {config_path}")
        print(f"  Python path (first 3 entries): {sys.path[:3]}")
        
        # Check if yaml is installed
        try:
            import yaml
            print("  PyYAML is installed")
        except ImportError:
            print("  ERROR: PyYAML is NOT installed!")
            print("  Install with: pip install pyyaml")
        
        sys.exit(1)

try:
    from core.tokenizer import ConversationTokenizer
    from core.model import DeepSeekTransformer, DeepSeekConfig
    from core.dataset import ConversationDataset, create_dataloader
    from core.triton_ops import replace_linear_with_fp8, is_triton_available
    print("✓ Core modules loaded")
except ImportError as e:
    print(f"ERROR: Could not import core modules: {e}")
    sys.exit(1)

# Import training infrastructure (orchestrator, trainer, checkpoint)
TRAINING_INFRASTRUCTURE_AVAILABLE = False
try:
    from training.orchestrator import AdaptiveTrainingOrchestrator
    from training.trainer import EnhancedConversationTrainer
    from training.checkpoint import CheckpointManager
    TRAINING_INFRASTRUCTURE_AVAILABLE = True
    print("✓ Advanced training infrastructure available (Orchestrator + Trainer + Checkpoint)")
except ImportError:
    try:
        from orchestrator import AdaptiveTrainingOrchestrator
        from trainer import EnhancedConversationTrainer
        from checkpoint import CheckpointManager
        TRAINING_INFRASTRUCTURE_AVAILABLE = True
        print("✓ Advanced training infrastructure available (Orchestrator + Trainer + Checkpoint)")
    except ImportError:
        print("⚠ Advanced training infrastructure not available - will use fallback")

os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'


def validate_data_paths(data_params: dict) -> bool:
    """
    Validate all data paths before training starts.
    Returns True if all paths are valid, False otherwise.
    """
    print("\n" + "="*80)
    print("VALIDATING DATA PATHS")
    print("="*80)
    
    training_mode = data_params.get('training_mode', 'finetuning_only')
    print(f"Training Mode: {training_mode}\n")
    
    all_valid = True
    checked_paths = []
    
    # Check base training paths
    if training_mode in ['base_only', 'hybrid', 'interleaved']:
        base_paths = data_params.get('base_training_paths', [])
        if base_paths:
            print("Base Training Paths:")
            for i, path_str in enumerate(base_paths, 1):
                path = Path(path_str)
                
                # Check if it's a directory (ERROR)
                if path.is_dir():
                    print(f"  ✗ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                    all_valid = False
                # Check if file exists
                elif path.exists() and path.is_file():
                    size = path.stat().st_size / (1024*1024)
                    print(f"  ✓ [{i}] {path_str} ({size:.2f} MB)")
                    checked_paths.append(str(path))
                else:
                    print(f"  ✗ [{i}] {path_str} - File not found!")
                    all_valid = False
    
    # Check fine-tuning paths
    if training_mode in ['finetuning_only', 'hybrid', 'interleaved']:
        ft_paths = data_params.get('finetuning_paths', [])
        if ft_paths:
            print("\nFine-tuning Paths:")
            for i, path_str in enumerate(ft_paths, 1):
                path = Path(path_str)
                
                # Check if it's a directory (ERROR)
                if path.is_dir():
                    print(f"  ✗ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                    all_valid = False
                # Check if file exists
                elif path.exists() and path.is_file():
                    size = path.stat().st_size / (1024*1024)
                    print(f"  ✓ [{i}] {path_str} ({size:.2f} MB)")
                    checked_paths.append(str(path))
                else:
                    print(f"  ✗ [{i}] {path_str} - File not found!")
                    all_valid = False
        else:
            print("\n✗ No fine-tuning paths specified for finetuning_only mode!")
            all_valid = False
    
    # Check eval paths
    eval_paths = []
    if training_mode == 'base_only':
        eval_paths = data_params.get('base_eval_paths', [])
    elif training_mode == 'finetuning_only':
        eval_paths = data_params.get('finetuning_eval_paths', [])
    
    if eval_paths:
        print("\nEvaluation Paths:")
        for i, path_str in enumerate(eval_paths, 1):
            path = Path(path_str)
            
            if path.is_dir():
                print(f"  ✗ [{i}] {path_str} - ERROR: This is a directory, not a file!")
                all_valid = False
            elif path.exists() and path.is_file():
                size = path.stat().st_size / (1024*1024)
                print(f"  ✓ [{i}] {path_str} ({size:.2f} MB)")
            else:
                print(f"  ⚠️ [{i}] {path_str} - File not found (will use training data)")
    
    # Summary
    print("\n" + "="*80)
    if all_valid and checked_paths:
        print(f"✓ VALIDATION PASSED - All {len(checked_paths)} file(s) are valid")
        print("="*80 + "\n")
        return True
    else:
        print("✗ VALIDATION FAILED - Please fix the issues above")
        print("="*80 + "\n")
        return False

def setup_router_finetuning(orchestrator, router_params):
    """
    Optional: Allow router to be fine-tuned during main training
    
    Add this after orchestrator initialization
    """
    if not router_params.get('finetune_router', False):
        return orchestrator
    
    print("\n" + "="*80)
    print("ENABLING ROUTER FINE-TUNING")
    print("="*80)
    
    # Collect router parameters
    router_params_list = []
    for name, param in orchestrator.model.named_parameters():
        if 'gate' in name.lower() or 'router' in name.lower():
            param.requires_grad = True  # Enable gradients
            router_params_list.append(param)
    
    print(f"✓ Found {len(router_params_list)} router parameters")
    print(f"  Total router params: {sum(p.numel() for p in router_params_list):,}")
    
    # Create separate optimizer for router (lower LR)
    from torch.optim import AdamW
    router_optimizer = AdamW(
        router_params_list,
        lr=router_params.get('router_learning_rate', 1e-5),
        weight_decay=0.01
    )
    
    # Store in orchestrator
    orchestrator.router_optimizer = router_optimizer
    
    print(f"✓ Router optimizer created (LR: {router_params.get('router_learning_rate', 1e-5):.2e})")
    print("  Router will be fine-tuned during training")
    print("="*80 + "\n")
    
    return orchestrator

def load_pretrained_router(model, router_checkpoint_path: str):
    """
    Load a pre-trained router model and replace gates.
    FIXED VERSION: Creates separate adapter instances per layer.
    """
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED ROUTER MODEL")
    print("="*80)
    
    checkpoint_path = Path(router_checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Router checkpoint not found: {checkpoint_path}")
        print("   Continuing with standard gates...")
        return model
    
    print(f"📁 Loading router from: {checkpoint_path}")
    
    try:
        # Load checkpoint to verify
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n✅ Router checkpoint loaded successfully")
        print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Validation accuracy: {checkpoint.get('metrics', {}).get('top1_acc', 'unknown')}")
        
        # Import adapter
        from router_training_system import RouterGateAdapter
        
        print(f"\n🔄 Replacing gates with trained router...")
        num_replaced = 0
        
        device = next(model.parameters()).device
        
        for name, module in model.named_modules():
            if hasattr(module, 'gate'):
                # Check if this is an MoE layer
                if 'moe' in name.lower() or 'expert' in name.lower():
                    # ✅ FIX: Create NEW adapter instance for each layer
                    adapter = RouterGateAdapter(
                        router_checkpoint_path=str(checkpoint_path),
                        device=str(device)
                    )
                    
                    # Replace gate
                    old_gate_params = sum(p.numel() for p in module.gate.parameters())
                    module.gate = adapter
                    new_gate_params = sum(p.numel() for p in module.gate.parameters())
                    
                    num_replaced += 1
                    
                    print(f"  ✅ Replaced gate in layer: {name}")
                    print(f"    Old gate params: {old_gate_params:,}")
                    print(f"    New router params: {new_gate_params:,}")
        
        if num_replaced == 0:
            print(f"\n⚠️  WARNING: No gates found to replace!")
            print(f"     Model might not have MoE layers or gates not accessible")
            return model
        
        print(f"\n✅ ROUTER INTEGRATION COMPLETE")
        print(f"   Replaced {num_replaced} gate(s) with trained router")
        print(f"   Model now uses learned routing decisions")
        print("="*80 + "\n")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error loading router: {e}")
        traceback.print_exc()
        print("\n⚠️  Continuing with standard gates...")
        return model
    
def validate_mps_compatibility(config) -> Tuple[bool, List[str]]:
    """
    Validate configuration for MPS (Apple Silicon) compatibility.
    
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check if we're actually on MPS
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return True, []  # Not MPS, no issues
    
    # Check DeepSpeed (not supported on MPS)
    if getattr(config, 'use_deepspeed', False):
        issues.append("DeepSpeed is not supported on MPS. Set use_deepspeed=False")
    
    # Check Flash Attention (not supported on MPS)
    if getattr(config, 'use_flash_attention', True):
        issues.append("Flash Attention is not supported on MPS. Set use_flash_attention=False")
    
    # Check precision (BF16 has limited support)
    precision = getattr(config, 'precision', 'fp32')
    if precision in ['bf16', 'mixed_bf16']:
        issues.append("BF16 precision has limited support on MPS. Use 'fp16' or 'fp32' instead")
    
    # Check model compilation (can be problematic)
    if getattr(config, 'compile', False):
        issues.append("Model compilation may cause issues on MPS. Consider setting compile=False")
    
    # Check batch size (MPS uses unified memory)
    batch_size = getattr(config, 'batch_size', 1)
    if batch_size > 8:
        issues.append(f"Large batch size ({batch_size}) may cause memory issues on MPS. Consider starting with batch_size=2-4")
    
    is_compatible = len(issues) == 0
    return is_compatible, issues


def wrap_orchestrator_with_oom_protection(orchestrator, train_dataset, eval_dataset):
    """
    Wrap the orchestrator's adaptive training with OOM protection.
    
    This function catches OOM errors from the orchestrator and automatically
    adjusts batch size, then recreates the orchestrator with new settings.
    """
    original_batch_size = orchestrator.config.batch_size
    original_grad_accum = getattr(orchestrator.config, 'gradient_accumulation_steps', 1)
    min_batch_size = 1
    
    attempt = 0
    max_attempts = 10  # Prevent infinite loops
    
    print(f"\n{'='*80}")
    print(f"ADAPTIVE TRAINING WITH OOM PROTECTION")
    print(f"{'='*80}")
    print(f"Initial configuration:")
    print(f"  Batch size: {original_batch_size}")
    print(f"  Gradient accumulation: {original_grad_accum}")
    print(f"  Effective batch size: {original_batch_size * original_grad_accum}")
    print(f"  Max attempts: {max_attempts}")
    
    while orchestrator.config.batch_size >= min_batch_size and attempt < max_attempts:
        attempt += 1
        
        try:
            print(f"\n{'='*80}")
            print(f"TRAINING ATTEMPT {attempt}/{max_attempts}")
            print(f"{'='*80}")
            print(f"Current configuration:")
            print(f"  Batch size: {orchestrator.config.batch_size}")
            print(f"  Gradient accumulation: {orchestrator.config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {orchestrator.config.batch_size * orchestrator.config.gradient_accumulation_steps}")
            
            # Run adaptive training
            orchestrator.run_adaptive_training()

            # Verify scheduler
            print("\n" + "="*80)
            print("🔍 SCHEDULER VERIFICATION")
            print("="*80)

            if hasattr(orchestrator.trainer, 'scheduler'):
                if orchestrator.trainer.scheduler is not None:
                    scheduler_type = type(orchestrator.trainer.scheduler).__name__
                    print(f"✅ Scheduler found: {scheduler_type}")

                    try:
                        # DeepSpeed schedulers work differently
                        if hasattr(orchestrator.trainer, 'use_deepspeed') and orchestrator.trainer.use_deepspeed:
                            print(f"✅ Using DeepSpeed scheduler (managed internally)")
                            print(f"   Initial LR: {orchestrator.config.learning_rate:.2e}")
                            print(f"   Scheduler type: WarmupLR")
                            print(f"   Warmup steps: ~{int(getattr(orchestrator, 'steps_per_epoch', 100) * 0.05)}")
                        else:
                            # Standard PyTorch scheduler verification
                            initial_lr = orchestrator.trainer.scheduler.get_last_lr()[0]
                            base_lrs = orchestrator.trainer.scheduler.base_lrs
                            print(f"✅ Initial LR: {initial_lr:.2e}")
                            print(f"✅ Base LRs: {[f'{lr:.2e}' for lr in base_lrs]}")
                            print(f"✅ Config LR: {orchestrator.config.learning_rate:.2e}")

                            # Verify they match
                            if abs(initial_lr - orchestrator.config.learning_rate) > 1e-9:
                                print(f"⚠️ WARNING: Scheduler LR doesn't match config LR!")
                            else:
                                print(f"✅ Scheduler LR matches config")

                    except AttributeError as e:
                        print(f"ℹ️ Scheduler state not fully accessible (normal for DeepSpeed): {e}")
                    except Exception as e:
                        print(f"⚠️ Could not fully verify scheduler: {e}")
                else:
                    print("⚠️ Scheduler is None")
                    print(f"   use_lr_scheduler: {getattr(orchestrator.config, 'use_lr_scheduler', 'not set')}")
                    print(f"   LR will remain constant at: {orchestrator.config.learning_rate:.2e}")
            else:
                print("❌ Trainer has no scheduler attribute!")

            print("="*80 + "\n")
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"])
            
            if is_oom:
                print(f"\n{'='*80}")
                print(f"⚠ OOM ERROR DETECTED IN ORCHESTRATOR (Attempt {attempt})")
                print(f"{'='*80}")
                print(f"Error message: {str(e)[:300]}")
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("✓ Cleared CUDA cache")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    print("✓ Cleared MPS cache")
                
                # Force garbage collection
                import gc
                gc.collect()
                print("✓ Ran garbage collection")
                
                # Adaptive strategy for batch size reduction
                current_batch = orchestrator.config.batch_size
                current_grad_accum = orchestrator.config.gradient_accumulation_steps
                
                # Strategy 1: If gradient accumulation is reasonable, increase it and reduce batch size
                if current_grad_accum < 16 and current_batch > 1:
                    new_batch_size = max(1, current_batch // 2)
                    new_grad_accum = min(32, current_grad_accum * 2)
                    strategy = "Halve batch size, double gradient accumulation"
                
                # Strategy 2: Just reduce batch size
                elif current_batch > 1:
                    new_batch_size = max(1, current_batch // 2)
                    new_grad_accum = current_grad_accum
                    strategy = "Halve batch size only"
                
                # Strategy 3: Can't reduce further
                else:
                    print(f"\n{'='*80}")
                    print(f"✗ CANNOT REDUCE BATCH SIZE FURTHER")
                    print(f"{'='*80}")
                    print(f"Current batch size: {current_batch}")
                    print(f"Current gradient accumulation: {current_grad_accum}")
                    print(f"\nSuggestions to reduce memory usage:")
                    print(f"  1. Reduce sequence length (current: {orchestrator.config.seq_length})")
                    print(f"  2. Reduce model size (layers: {orchestrator.config.num_layers}, hidden: {orchestrator.config.hidden_size})")
                    print(f"  3. Enable gradient checkpointing: config.gradient_checkpointing = True")
                    print(f"  4. Use lower precision: config.precision = 'fp16' or 'bf16'")
                    print(f"  5. Enable CPU offloading: config.cpu_offload = True")
                    print(f"  6. Use DeepSpeed ZeRO: config.use_deepspeed = True, config.zero_stage = 3")
                    if hasattr(orchestrator.config, 'use_moe') and orchestrator.config.use_moe:
                        print(f"  7. Reduce number of experts: config.num_experts (current: {orchestrator.config.num_experts})")
                    raise
                
                print(f"\n{'='*80}")
                print(f"APPLYING RECOVERY STRATEGY: {strategy}")
                print(f"{'='*80}")
                print(f"  Previous batch size: {current_batch}")
                print(f"  Previous gradient accumulation: {current_grad_accum}")
                print(f"  Previous effective batch: {current_batch * current_grad_accum}")
                print(f"  →")
                print(f"  New batch size: {new_batch_size}")
                print(f"  New gradient accumulation: {new_grad_accum}")
                print(f"  New effective batch: {new_batch_size * new_grad_accum}")
                
                # Update configuration
                orchestrator.config.batch_size = new_batch_size
                orchestrator.config.gradient_accumulation_steps = new_grad_accum
                
                # Recreate orchestrator with new configuration
                print(f"\nRecreating orchestrator with adjusted configuration...")
                try:
                    # Import here to avoid circular dependency
                    from training.orchestrator import AdaptiveTrainingOrchestrator
                    orchestrator = AdaptiveTrainingOrchestrator(orchestrator.config)
                    print(f"✓ Orchestrator recreated successfully")
                except ImportError:
                    from orchestrator import AdaptiveTrainingOrchestrator
                    orchestrator = AdaptiveTrainingOrchestrator(orchestrator.config)
                    print(f"✓ Orchestrator recreated successfully")
                
                print(f"\nPreparing to retry training (attempt {attempt + 1}/{max_attempts})...")
                
            else:
                # Not an OOM error, re-raise
                print(f"\n{'='*80}")
                print(f"✗ NON-OOM ERROR DETECTED")
                print(f"{'='*80}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)[:500]}")
                raise
        
        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print(f"TRAINING INTERRUPTED BY USER")
            print(f"{'='*80}")
            raise
        
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ UNEXPECTED ERROR")
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)[:500]}")
            raise
    
    # Check if we exhausted attempts
    if attempt >= max_attempts:
        print(f"\n{'='*80}")
        print(f"✗ MAXIMUM ATTEMPTS REACHED")
        print(f"{'='*80}")
        print(f"Training failed after {max_attempts} attempts")
        raise RuntimeError(f"Could not complete training after {max_attempts} OOM recovery attempts")
    
    # Report final configuration
    final_batch = orchestrator.config.batch_size
    final_grad_accum = orchestrator.config.gradient_accumulation_steps
    
    if final_batch != original_batch_size or final_grad_accum != original_grad_accum:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED WITH ADJUSTED CONFIGURATION")
        print(f"{'='*80}")
        print(f"Original → Final:")
        print(f"  Batch size: {original_batch_size} → {final_batch}")
        print(f"  Gradient accumulation: {original_grad_accum} → {final_grad_accum}")
        print(f"  Effective batch size: {original_batch_size * original_grad_accum} → {final_batch * final_grad_accum}")
        
        # Save optimal configuration
        try:
            optimal_config = {
                'batch_size': final_batch,
                'gradient_accumulation_steps': final_grad_accum,
                'effective_batch_size': final_batch * final_grad_accum,
                'original_batch_size': original_batch_size,
                'original_gradient_accumulation': original_grad_accum,
                'attempts_needed': attempt,
                'device': str(orchestrator.trainer.device) if hasattr(orchestrator, 'trainer') else 'unknown'
            }
            
            optimal_path = Path("optimal_batch_config.json")
            with open(optimal_path, 'w') as f:
                json.dump(optimal_config, f, indent=2)
            print(f"\n✓ Saved optimal configuration to {optimal_path}")
            print(f"  Use these settings for future runs to avoid OOM errors")
        except Exception as e:
            print(f"⚠ Could not save optimal configuration: {e}")
    
    return orchestrator


def validate_precision_support(precision: str, device: torch.device) -> Tuple[bool, str]:
    """Validate if the requested precision is supported by the hardware."""
    if precision in ['fp32', 'float32']:
        return True, ""
    
    if device.type == 'cpu':
        if precision in ['fp16', 'mixed_fp16']:
            return False, f"FP16 precision '{precision}' is not supported on CPU. Use 'fp32' or 'bf16' instead."
        elif precision in ['bf16', 'mixed_bf16']:
            try:
                test_tensor = torch.randn(2, 2, dtype=torch.bfloat16)
                _ = test_tensor + test_tensor
                return True, ""
            except:
                return False, f"BF16 precision '{precision}' is not supported on this CPU. Use 'fp32' instead."
    
    if device.type == 'mps':
        # MPS supports FP16 and FP32, but BF16 support is limited
        if precision in ['fp16', 'mixed_fp16']:
            return True, ""
        elif precision in ['bf16', 'mixed_bf16']:
            return False, f"BF16 precision '{precision}' is not fully supported on MPS. Use 'fp16' or 'fp32' instead."
        return True, ""
    
    if device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        
        if precision in ['fp16', 'mixed_fp16']:
            if major > 5 or (major == 5 and minor >= 3):
                return True, ""
            else:
                return False, f"FP16 precision '{precision}' requires compute capability >= 5.3. " \
                             f"Your GPU has {major}.{minor}. Use 'fp32' instead."
        
        elif precision in ['bf16', 'mixed_bf16']:
            if major >= 8:
                return True, ""
            else:
                return False, f"BF16 precision '{precision}' requires compute capability >= 8.0 (Ampere GPU or newer). " \
                             f"Your GPU has {major}.{minor}. Use 'fp16' or 'fp32' instead."
    
    return True, ""


def config_to_deepseek_config(config: Config):
    """
    Convert training Config to DeepSeekConfig.
    
    Args:
        config: Training configuration object
        
    Returns:
        DeepSeekConfig object
    """
    return DeepSeekConfig(
        vocab_size=getattr(config, 'vocab_size', 50257),
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 12),
        num_heads=getattr(config, 'num_heads', 12),
        num_kv_heads=getattr(config, 'num_kv_heads', None),
        intermediate_size=getattr(config, 'intermediate_size', None),
        seq_length=getattr(config, 'seq_length', 2048),
        dropout=getattr(config, 'dropout', 0.0),
        rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        init_std=getattr(config, 'init_std', 0.02),
        use_stable_embedding=getattr(config, 'use_stable_embedding', True),
        tie_word_embeddings=getattr(config, 'tie_word_embeddings', True),
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        use_moe=getattr(config, 'use_moe', False),
        num_experts=getattr(config, 'num_experts', 8),
        moe_top_k=getattr(config, 'moe_top_k', 2),
        capacity_factor=getattr(config, 'capacity_factor', 1.25),
        load_balancing_weight=getattr(config, 'load_balancing_weight', 0.01),
    )


def print_banner(title: str, width: int = 80):
    """Print a formatted banner."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def print_system_diagnostics():
    """Print comprehensive system diagnostics using utils."""
    print_banner("SYSTEM DIAGNOSTICS & ENVIRONMENT VALIDATION")
    
    if UTILS_AVAILABLE:
        system_info = get_system_info()
        
        print_section("Hardware Information")
        print(f"  Python Version: {system_info.get('python_version', 'Unknown')[:70]}")
        print(f"  PyTorch Version: {system_info.get('pytorch_version', 'Unknown')}")
        print(f"  CUDA Available: {system_info.get('cuda_available', False)}")
        print(f"  MPS Available: {system_info.get('mps_available', False)}")
        
        # Display primary acceleration method
        if system_info.get('cuda_available'):
            print(f"  Primary Accelerator: NVIDIA CUDA")
        elif system_info.get('mps_available'):
            print(f"  Primary Accelerator: Apple Silicon (MPS)")
        else:
            print(f"  Primary Accelerator: CPU Only")
        
        # CUDA-specific information
        if system_info.get('cuda_available'):
            print(f"  CUDA Version: {system_info.get('cuda_version', 'Unknown')}")
            print(f"  GPU Count: {system_info.get('gpu_count', 0)}")
            print(f"  GPU Model: {system_info.get('gpu_name', 'Unknown')}")
            print(f"  GPU Memory: {system_info.get('gpu_memory_gb', 0):.2f} GB")
            
            # Additional GPU info
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    compute_cap = torch.cuda.get_device_capability(i)
                    print(f"  GPU {i} Details:")
                    print(f"    Name: {props.name}")
                    print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                    print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
                    print(f"    Multi-Processors: {props.multi_processor_count}")
        
        # MPS-specific information
        if system_info.get('mps_available'):
            print(f"  MPS Device: {system_info.get('device_type', 'Apple Silicon')}")
            print(f"  MPS Backend: {system_info.get('mps_backend', 'Metal Performance Shaders')}")
            print(f"  Unified Memory Architecture: Yes")
            
            # Check PyTorch version for MPS compatibility
            pytorch_version = system_info.get('pytorch_version', '0.0.0')
            try:
                major, minor = pytorch_version.split('.')[:2]
                version_num = float(f"{major}.{minor}")
                if version_num >= 2.0:
                    print(f"  MPS Support Level: Full (PyTorch >= 2.0)")
                elif version_num >= 1.12:
                    print(f"  MPS Support Level: Beta (PyTorch >= 1.12)")
                else:
                    print(f"  MPS Support Level: Not Available (Need PyTorch >= 1.12)")
            except:
                print(f"  MPS Support Level: Unknown")
            
            # MPS capabilities and limitations
            print(f"  MPS Capabilities:")
            print(f"    - FP32 Training: ✓ Supported")
            print(f"    - FP16 Training: ✓ Supported")
            print(f"    - BF16 Training: ✗ Limited Support (use FP16 instead)")
            print(f"    - Flash Attention: ✗ Not Supported (CUDA only)")
            print(f"    - DeepSpeed: ✗ Not Supported (CUDA only)")
            print(f"    - Gradient Checkpointing: ✓ Supported")
            print(f"    - Mixed Precision: ✓ Supported (FP16)")
        
        print_section("System Resources")
        if system_info.get('system_memory_gb'):
            total_mem = system_info.get('system_memory_gb', 0)
            avail_mem = system_info.get('available_memory_gb', 0)
            used_mem = total_mem - avail_mem
            mem_percent = (used_mem / total_mem * 100) if total_mem > 0 else 0
            print(f"  Total System Memory: {total_mem:.2f} GB")
            print(f"  Available Memory: {avail_mem:.2f} GB")
            print(f"  Used Memory: {used_mem:.2f} GB ({mem_percent:.1f}%)")
            
            # Add MPS-specific memory note
            if system_info.get('mps_available'):
                print(f"  Note: MPS uses unified memory (shared with system)")
                recommended_mem = 16.0  # GB
                if total_mem < recommended_mem:
                    print(f"  ⚠️ Warning: {recommended_mem:.0f}GB+ recommended for MPS training")
        
        if system_info.get('cpu_count'):
            print(f"  CPU Cores: {system_info.get('cpu_count', 0)}")
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    print(f"  CPU Frequency: {cpu_freq.current:.0f} MHz")
                cpu_percent = psutil.cpu_percent(interval=0.1)
                print(f"  CPU Usage: {cpu_percent:.1f}%")
            except:
                pass
        
        if system_info.get('disk_space_gb'):
            disk_free = system_info.get('disk_space_gb', 0)
            try:
                disk_usage = shutil.disk_usage(".")
                disk_total = disk_usage.total / 1e9
                disk_used = disk_usage.used / 1e9
                disk_percent = (disk_used / disk_total * 100) if disk_total > 0 else 0
                print(f"  Total Disk Space: {disk_total:.2f} GB")
                print(f"  Used Disk Space: {disk_used:.2f} GB ({disk_percent:.1f}%)")
                print(f"  Free Disk Space: {disk_free:.2f} GB")
            except:
                print(f"  Free Disk Space: {disk_free:.2f} GB")
        
        print_section("Environment Validation")
        issues = validate_environment()
        if issues:
            print("  Detected Issues:")
            for i, issue in enumerate(issues, 1):
                # Color code issues
                if "MPS" in issue or "Apple Silicon" in issue:
                    symbol = "ℹ️ "
                elif "insufficient" in issue.lower() or "low" in issue.lower():
                    symbol = "⚠️ "
                elif "slow" in issue.lower() or "not available" in issue.lower():
                    symbol = "⚠️ "
                else:
                    symbol = "   "
                print(f"    {symbol}{i}. {issue}")
        else:
            print("  ✓ All environment checks passed successfully")
        
        # Check for required libraries
        print_section("Library Availability")
        libraries = {
            'DeepSpeed': DEEPSPEED_AVAILABLE,
            'Training Infrastructure': TRAINING_INFRASTRUCTURE_AVAILABLE,
            'Utils': UTILS_AVAILABLE,
        }
        
        # Add MPS-specific library notes
        if system_info.get('mps_available'):
            libraries['Flash Attention (CUDA only)'] = False
            libraries['MPS Fallback'] = True  # Always available with MPS
        
        for lib, available in libraries.items():
            status = "Available" if available else "Not Available"
            symbol = "✓" if available else "✗"
            
            # Special handling for expected unavailability on MPS
            if system_info.get('mps_available') and 'CUDA only' in lib:
                symbol = "ℹ️ "
                status = "Not Available (Expected on MPS)"
            
            print(f"  {symbol} {lib}: {status}")
        
        # Device-specific recommendations
        print_section("Hardware-Specific Recommendations")
        
        if system_info.get('mps_available'):
            print("  Apple Silicon (MPS) Detected:")
            print("    1. Use FP16 precision for optimal performance")
            print("    2. Disable DeepSpeed (not supported)")
            print("    3. Disable Flash Attention (CUDA only)")
            print("    4. Start with smaller batch sizes")
            print("    5. Monitor unified memory usage")
            print("    6. Consider disabling model compilation initially")
            print("    7. Enable MPS fallback for unsupported ops")
            print("")
            print("  Recommended Configuration:")
            print("    config.precision = 'fp16'")
            print("    config.use_deepspeed = False")
            print("    config.use_flash_attention = False")
            print("    config.compile = False")
            print("    config.batch_size = 2  # Start small")
            
        elif system_info.get('cuda_available'):
            gpu_memory = system_info.get('gpu_memory_gb', 0)
            print("  NVIDIA CUDA Detected:")
            
            if gpu_memory >= 80:
                print("    - High-end GPU detected (80GB+)")
                print("    - Can train large models with full features")
                print("    - Flash Attention recommended")
                print("    - DeepSpeed ZeRO-3 for very large models")
            elif gpu_memory >= 40:
                print("    - Professional GPU detected (40GB+)")
                print("    - Can train medium to large models")
                print("    - Flash Attention recommended")
                print("    - Consider DeepSpeed ZeRO-2/3")
            elif gpu_memory >= 16:
                print("    - Consumer high-end GPU detected (16GB+)")
                print("    - Can train small to medium models")
                print("    - Flash Attention recommended")
                print("    - Consider gradient checkpointing")
            elif gpu_memory >= 8:
                print("    - Consumer mid-range GPU detected (8GB+)")
                print("    - Limited to small models")
                print("    - Gradient checkpointing essential")
                print("    - Small batch sizes required")
            else:
                print("    - Limited GPU memory detected")
                print("    - Consider CPU training or upgrading GPU")
            
            # Check compute capability
            if torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability(0)
                major = compute_cap[0]
                
                if major >= 8:
                    print("    - Ampere or newer: BF16 supported")
                    print("    - Recommended precision: mixed_bf16")
                elif major >= 7:
                    print("    - Volta/Turing: FP16 supported")
                    print("    - Recommended precision: mixed_fp16")
                else:
                    print("    - Older architecture: FP32 only")
                    print("    - Recommended precision: fp32")
        else:
            print("  CPU Only Detected:")
            print("    - Training will be significantly slower")
            print("    - Recommended for debugging only")
            print("    - Consider cloud GPU instances for production")
            print("    - Use smaller models and batch sizes")
        
        # Memory recommendations
        print_section("Memory Recommendations")
        if system_info.get('mps_available'):
            total_mem = system_info.get('system_memory_gb', 0)
            if total_mem >= 32:
                print("  ✓ Excellent memory (32GB+) for MPS training")
            elif total_mem >= 16:
                print("  ✓ Good memory (16GB+) for MPS training")
            elif total_mem >= 8:
                print("  ⚠️ Limited memory (8GB) - use small models only")
            else:
                print("  ⚠️ Insufficient memory (<8GB) - not recommended")
        elif system_info.get('cuda_available'):
            gpu_mem = system_info.get('gpu_memory_gb', 0)
            sys_mem = system_info.get('system_memory_gb', 0)
            print(f"  GPU Memory: {gpu_mem:.1f}GB")
            print(f"  System Memory: {sys_mem:.1f}GB")
            if sys_mem < 16:
                print("  ⚠️ Consider 16GB+ system RAM for optimal performance")
        
    else:
        print("  Utils not available - showing basic diagnostics only")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  Device: Apple Silicon with MPS")
            try:
                import psutil
                total_mem = psutil.virtual_memory().total / 1e9
                print(f"  Unified Memory: {total_mem:.2f} GB")
            except:
                print(f"  Unified Memory: Unknown")
    
    print("="*80)


def prepare_and_validate_data(config, tokenizer):
    """
    Prepare and validate training data with comprehensive reporting.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        
    Returns:
        Dictionary containing dataset statistics
    """
    print_banner("DATA PREPARATION & VALIDATION")
    
    train_data_path = Path(config.train_data_path)
    eval_data_path = Path(config.eval_data_path)
    
    # Process OASST data if raw data is available
    if UTILS_AVAILABLE and hasattr(config, 'raw_oasst_path'):
        raw_path = Path(config.raw_oasst_path)
        if raw_path.exists() and not train_data_path.exists():
            print_section("Processing Raw OASST Data")
            print(f"  Input File: {raw_path}")
            print(f"  Output File: {train_data_path}")
            print(f"  Max Conversations: {config.max_conversations_per_file}")
            
            processing_start = time.time()
            num_processed = process_oasst_data(
                str(raw_path),
                str(train_data_path),
                max_conversations=config.max_conversations_per_file
            )
            processing_time = time.time() - processing_start
            
            print(f"  Processed {num_processed:,} conversations in {processing_time:.2f} seconds")
            print(f"  Processing rate: {num_processed/processing_time:.0f} conversations/sec")
    
    # Create sample data if needed
    if not train_data_path.exists():
        print_section("Creating Sample Data")
        if UTILS_AVAILABLE:
            print(f"  Training data not found: {train_data_path}")
            print(f"  Creating sample data for testing...")
            num_samples = 200
            create_sample_data(str(train_data_path), num_conversations=num_samples)
            print(f"  Created {num_samples} sample conversations")
        else:
            raise FileNotFoundError(f"Training data not found: {train_data_path}")
    
    # Validate data
    datasets_info = {}
    
    if UTILS_AVAILABLE:
        print_section("Validating Training Data")
        validation_start = time.time()
        train_stats = validate_data_comprehensive(str(train_data_path), tokenizer, max_check=5000)
        validation_time = time.time() - validation_start
        
        print(f"\n  File Information:")
        print(f"    Path: {train_data_path}")
        print(f"    Size: {train_stats['file_info'].get('size_mb', 0):.2f} MB")
        print(f"    Last Modified: {train_stats['file_info'].get('modified', 'Unknown')}")
        print(f"    Validation Time: {validation_time:.2f} seconds")
        
        print(f"\n  Conversation Statistics:")
        conv_stats = train_stats['conversation_stats']
        print(f"    Total Lines: {conv_stats.get('total_lines', 0):,}")
        print(f"    Valid Conversations: {conv_stats.get('valid_conversations', 0):,}")
        print(f"    Invalid Conversations: {conv_stats.get('invalid_conversations', 0):,}")
        print(f"    Success Rate: {train_stats['quality_metrics'].get('success_rate', 0):.2%}")
        print(f"    Avg Messages/Conversation: {conv_stats.get('avg_messages_per_conversation', 0):.1f}")
        print(f"    Max Messages: {conv_stats.get('max_messages', 0)}")
        print(f"    Min Messages: {conv_stats.get('min_messages', 0)}")
        
        print(f"\n  Token Statistics:")
        token_stats = train_stats['token_stats']
        print(f"    Average Tokens: {token_stats.get('avg_tokens', 0):.0f}")
        print(f"    Median Tokens: {token_stats.get('median_tokens', 0):.0f}")
        print(f"    Max Tokens: {token_stats.get('max_tokens', 0):,}")
        print(f"    Min Tokens: {token_stats.get('min_tokens', 0):,}")
        print(f"    Std Dev Tokens: {token_stats.get('std_tokens', 0):.1f}")
        
        print(f"\n  Role Distribution:")
        role_dist = conv_stats.get('role_distribution', {})
        total_messages = sum(role_dist.values())
        for role, count in sorted(role_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_messages * 100) if total_messages > 0 else 0
            print(f"    {role.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n  Quality Metrics:")
        quality = train_stats['quality_metrics']
        print(f"    Error Rate: {quality.get('error_rate', 0):.2%}")
        print(f"    Total Quality Issues: {quality.get('total_quality_issues', 0)}")
        
        issues = quality.get('quality_issues_sample', [])
        if issues:
            print(f"    Sample Issues (first 5):")
            for i, issue in enumerate(issues[:5], 1):
                print(f"      {i}. {issue}")
        
        datasets_info['train'] = train_stats
        
        # Validate eval data if different
        if eval_data_path.exists() and eval_data_path != train_data_path:
            print_section("Validating Evaluation Data")
            eval_stats = validate_data_comprehensive(str(eval_data_path), tokenizer, max_check=2000)
            print(f"  Eval Conversations: {eval_stats['conversation_stats'].get('valid_conversations', 0):,}")
            print(f"  Eval Success Rate: {eval_stats['quality_metrics'].get('success_rate', 0):.2%}")
            datasets_info['eval'] = eval_stats
        
        # Generate comprehensive data report
        if getattr(config, 'generate_data_reports', True):
            print_section("Generating Data Summary Report")
            report_paths = [str(train_data_path)]
            if eval_data_path.exists() and eval_data_path != train_data_path:
                report_paths.append(str(eval_data_path))
            
            report_file = "data_summary_report.html"
            create_data_summary_report(report_paths, tokenizer, report_file)
            print(f"  HTML report saved: {report_file}")
    else:
        print("  Utils not available - skipping detailed validation")
        print(f"  Training data exists: {train_data_path.exists()}")
    
    print("="*80)
    return datasets_info


def estimate_and_display_training_time(config, dataset_size: int):
    """
    Estimate and display training time using utils.
    
    Args:
        config: Training configuration
        dataset_size: Number of samples in dataset
    """
    if not UTILS_AVAILABLE:
        print("  Utils not available - skipping estimation")
        return
    
    print_section("Time Estimates")
    
    try:
        estimates = estimate_training_time(config, dataset_size)
        
        # Display basic estimates
        print(f"  Total Tokens to Process: {estimates['total_tokens']:,}")
        print(f"  Estimated Throughput: {estimates['tokens_per_second']:,.0f} tokens/second")
        print(f"  Device Type: {estimates['device_type']}")
        
        # Time estimates
        print(f"\n  Estimated Training Time:")
        hours = estimates['estimated_hours']
        
        # Convert to human-readable format
        if hours < 0.1:
            human_readable = f"{hours * 60:.1f} minutes"
        elif hours < 1:
            human_readable = f"{hours * 60:.0f} minutes"
        elif hours < 24:
            human_readable = f"{hours:.1f} hours"
        else:
            days = int(hours // 24)
            remaining_hours = hours % 24
            human_readable = f"{days} days, {remaining_hours:.1f} hours"
        
        print(f"    Total Hours: {hours:.2f}")
        print(f"    Total Days: {estimates['estimated_days']:.2f}")
        print(f"    Human Readable: {human_readable}")
        
        # Memory estimates
        print(f"\n  Memory Estimates:")
        print(f"    Estimated Memory Needed: {estimates['estimated_memory_needed_gb']:.2f} GB")
        print(f"    Memory Utilization: {estimates['memory_utilization']:.1%}")
        
        if estimates.get('memory_warning'):
            print(f"    ⚠️ WARNING: High memory utilization expected!")
            print(f"       Consider reducing batch size or enabling gradient checkpointing")
        else:
            print(f"    ✓ Memory utilization within safe limits")
        
        # Model info
        print(f"\n  Model Information:")
        print(f"    Total Parameters: {estimates['model_parameters']:,}")
        
        # Device-specific info
        print(f"\n  Device-Specific Details:")
        if estimates['device_type'] == 'cuda':
            print(f"    CUDA acceleration enabled")
            print(f"    Recommended precision: mixed_fp16 or mixed_bf16")
        elif estimates['device_type'] == 'mps':
            print(f"    Apple Silicon (MPS) acceleration")
            print(f"    Recommended precision: fp16")
            print(f"    Note: Some operations may fall back to CPU")
        else:
            print(f"    CPU only - training will be slow")
            print(f"    Consider using a GPU for production training")
        
        # Optimization recommendations
        print(f"\n  Optimization Recommendations:")
        
        if estimates['memory_utilization'] > 0.85:
            print(f"    Memory Optimization:")
            print(f"      - Reduce batch size")
            print(f"      - Enable gradient checkpointing")
            print(f"      - Use gradient accumulation")
            print(f"      - Consider mixed precision training")
        
        if estimates['estimated_hours'] > 48:
            print(f"    Training Time Optimization:")
            print(f"      - Consider using multiple GPUs")
            print(f"      - Enable mixed precision training")
            print(f"      - Use DeepSpeed ZeRO optimization")
            print(f"      - Reduce dataset size for initial experiments")
        
        if estimates['estimated_hours'] < 1:
            print(f"    ✓ Training time is reasonable")
        elif estimates['estimated_hours'] < 12:
            print(f"    Training time is moderate")
        else:
            print(f"    Training will take significant time - ensure stable environment")
        
        # Training schedule recommendation
        print(f"\n  Recommended Training Schedule:")
        if hours < 2:
            print(f"    Single session training recommended")
        elif hours < 12:
            print(f"    Can complete in one work session")
            print(f"    Ensure checkpoint saving every 1-2 hours")
        elif hours < 24:
            print(f"    Day-long training session")
            print(f"    Enable checkpoint saving every hour")
            print(f"    Monitor progress regularly")
        else:
            days = int(hours // 24) + 1
            print(f"    Multi-day training ({days} days)")
            print(f"    Enable checkpoint saving every 2-4 hours")
            print(f"    Set up monitoring and alerting")
            print(f"    Consider cloud training for reliability")
    
    except Exception as e:
        print(f"  Error estimating training time: {e}")        
        traceback.print_exc()
        print("  Continuing without time estimates...")


def setup_signal_handlers(orchestrator):
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        orchestrator: Training orchestrator instance
    """
    def signal_handler(sig, frame):
        print("\n" + "="*80)
        print("RECEIVED INTERRUPT SIGNAL - INITIATING GRACEFUL SHUTDOWN")
        print("="*80)
        print("Saving current state...")
        
        try:
            if orchestrator:
                orchestrator._save_meta_learning_state()
                print("Meta-learning state saved")
        except Exception as e:
            print(f"Error saving state: {e}")
        
        print("Cleanup complete. Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_and_setup_experiment(config) -> Path:
    """
    Validate configuration and setup experiment directory.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to experiment directory
    """
    print_section("Experiment Setup")
    
    experiment_dir = Path(f"experiments/{config.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Experiment Directory: {experiment_dir}")
    print(f"  Experiment Name: {config.experiment_name}")
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'reports', 'metrics']
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(exist_ok=True)
    print(f"  Created subdirectories: {', '.join(subdirs)}")
    
    # Save configuration
    config_path = experiment_dir / "config.yaml"
    config.save(str(config_path))
    print(f"  Configuration saved: {config_path}")
    
    # Save configuration as JSON for easier parsing
    config_json_path = experiment_dir / "config.json"
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(config_json_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  Configuration (JSON) saved: {config_json_path}")
    
    return experiment_dir


def save_experiment_metadata(experiment_dir: Path, config, model, datasets_info):
    """
    Save comprehensive experiment metadata.
    
    Args:
        experiment_dir: Path to experiment directory
        config: Training configuration
        model: Model instance
        datasets_info: Dataset statistics
    """
    print_section("Saving Experiment Metadata")
    
    metadata = {
        'experiment_name': config.experiment_name,
        'created_at': datetime.now().isoformat(),
        'training_config': {
            'epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'precision': config.precision,
            'gradient_accumulation': config.gradient_accumulation_steps,
        },
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'seq_length': config.seq_length,
            'use_moe': config.use_moe,
            'num_experts': config.num_experts if config.use_moe else None,
        }
    }
    
    # Add model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['model_parameters'] = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }
    
    # Save system info
    if UTILS_AVAILABLE:
        system_info_path = experiment_dir / "system_info.json"
        with open(system_info_path, 'w') as f:
            json.dump(get_system_info(), f, indent=2, default=str)
        print(f"  System info saved: {system_info_path}")
        
        # Save dataset statistics
        if datasets_info:
            data_stats_path = experiment_dir / "data_statistics.json"
            with open(data_stats_path, 'w') as f:
                json.dump(datasets_info, f, indent=2, default=str)
            print(f"  Dataset statistics saved: {data_stats_path}")
    
    # Save metadata
    metadata_path = experiment_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Experiment metadata saved: {metadata_path}")

def load_checkpoint_for_continuation(checkpoint_path: str, orchestrator) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state - FIXED VERSION
    """
    print("\n" + "="*80)
    print("LOADING CHECKPOINT FOR CONTINUATION - FIXED VERSION")
    print("="*80)
    
    checkpoint_path = Path(checkpoint_path)
    
    # Find checkpoint file
    if checkpoint_path.is_dir():
        checkpoints = sorted(checkpoint_path.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_file = checkpoints[-1]  # Latest checkpoint
        print(f"Found {len(checkpoints)} checkpoints, using latest: {checkpoint_file.name}")
    else:
        checkpoint_file = checkpoint_path
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    print(f"\nLoading from: {checkpoint_file}")
    print(f"File size: {checkpoint_file.stat().st_size / 1e6:.2f} MB")
    
    try:
        # Try with weights_only=False for older checkpoints
        print("🔄 Attempting to load with weights_only=False...")
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        print("✅ Loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint with weights_only=False: {e}")
        print("🔄 Attempting to load with safe globals...")
        try:
            # Try with safe globals
            import torch.serialization
            torch.serialization.add_safe_globals(['config.config_manager.Config'])
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print("✅ Loaded checkpoint with safe globals")
        except Exception as e2:
            print(f"❌ Could not load checkpoint: {e2}")
            print("Starting training from scratch...")
            raise
    
    # CRITICAL FIX: Direct state restoration
    print("\n🔄 RESTORING TRAINING STATE:")
    
    # Restore model state
    if 'model_state_dict' in checkpoint:
        orchestrator.model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model state restored")
    else:
        print("❌ No model_state_dict found in checkpoint")
    
    # Restore optimizer state
    if 'optimizer_state_dict' in checkpoint and hasattr(orchestrator, 'trainer'):
        if orchestrator.trainer and hasattr(orchestrator.trainer, 'optimizer'):
            orchestrator.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state restored")
        else:
            print("⚠️  Optimizer found but trainer not ready")
    else:
        print("⚠️  No optimizer_state_dict found in checkpoint")
    
    # Restore scheduler state
    if 'scheduler_state_dict' in checkpoint and hasattr(orchestrator, 'trainer'):
        if orchestrator.trainer and hasattr(orchestrator.trainer, 'scheduler') and orchestrator.trainer.scheduler:
            orchestrator.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Scheduler state restored")
        else:
            print("⚠️  Scheduler found but trainer scheduler not ready")
    else:
        print("⚠️  No scheduler_state_dict found in checkpoint")
    
    # Restore training progress
    info = {
        'start_epoch': checkpoint.get('epoch', 0) + 1,  # Start from NEXT epoch
        'global_step': checkpoint.get('global_step', 0),
        'best_loss': checkpoint.get('best_loss', float('inf')),
        'checkpoint_path': str(checkpoint_file)
    }
    
    # CRITICAL: Set orchestrator state directly
    orchestrator.start_epoch = info['start_epoch']
    orchestrator.global_step = info['global_step']
    orchestrator.best_loss = info['best_loss']
    
    print(f"\n✅ CHECKPOINT LOADED SUCCESSFULLY:")
    print(f"  Epoch: {checkpoint.get('epoch', 0)} → Continuing from epoch {info['start_epoch']}")
    print(f"  Global Step: {info['global_step']}")
    print(f"  Best Loss: {info['best_loss']:.4f}")
    print("="*80)
    
    return info


def setup_multi_dataset_training(config, data_params):
    """
    Setup training with multi-dataset support.
    
    This replaces the direct data_path usage with MultiDatasetManager.
    """
    from core.dataset import MultiDatasetManager
    
    print_section("Multi-Dataset Setup")
    
    # Initialize dataset manager
    dataset_manager = MultiDatasetManager(data_params)
    
    # Validate all datasets
    if data_params.get('validate_datasets', True):
        print("Validating datasets...")
        validation_stats = dataset_manager.validate_all_datasets()
        
        if validation_stats.get('validation_errors'):
            print("Validation errors found:")
            for error in validation_stats['validation_errors']:
                print(f"  ✗ {error}")
        
        print(f"\nTraining datasets: {len(validation_stats.get('train_datasets', []))}")
        for ds in validation_stats.get('train_datasets', []):
            print(f"  - {Path(ds['path']).name}: {ds['conversations']:,} conversations ({ds['size_mb']:.1f} MB)")
        
        print(f"\nEvaluation datasets: {len(validation_stats.get('eval_datasets', []))}")
        for ds in validation_stats.get('eval_datasets', []):
            print(f"  - {Path(ds['path']).name}: {ds['conversations']:,} conversations ({ds['size_mb']:.1f} MB)")
        
        print(f"\nTotal training conversations: {validation_stats.get('total_train_conversations', 0):,}")
        print(f"Total evaluation conversations: {validation_stats.get('total_eval_conversations', 0):,}")
    
    # Prepare combined datasets
    print("\nPreparing combined training data...")
    train_data_path = dataset_manager.prepare_training_data(
        force_rebuild=data_params.get('force_rebuild', False)
    )
    print(f"Training data ready: {train_data_path}")
    
    eval_data_path = None
    if dataset_manager.eval_paths:
        print("\nPreparing combined evaluation data...")
        eval_data_path = dataset_manager.prepare_evaluation_data()
        print(f"Evaluation data ready: {eval_data_path}")
    
    # Update config with final paths
    config.train_data_path = train_data_path
    config.eval_data_path = eval_data_path if eval_data_path else train_data_path
    
    return train_data_path, eval_data_path


def auto_adjust_epochs_chinchilla(config, model, dataset):
    """
    Chinchilla-style automatic epoch adjustment based on dataset size and model parameters.
    
    Formula: N_opt ≈ 20 * P (where P = model parameters in billions)
    
    Args:
        config: Training configuration object
        model: Model instance (to count parameters)
        dataset: Dataset instance (to estimate tokens)
        
    Returns:
        Updated config with adjusted num_epochs
    """
    print("\n" + "="*80)
    print("🧠 CHINCHILLA-STYLE EPOCH SCALING")
    print("="*80)
    
    # Get model parameter count
    try:
        P = sum(p.numel() for p in model.parameters())
        P_billions = P / 1e9
        print(f"📊 Model Parameters: {P:,} ({P_billions:.2f}B)")
    except Exception as e:
        print(f"⚠️ Could not count model parameters: {e}")
        print("   Skipping auto-epoch scaling")
        return config
    
    # Estimate dataset tokens
    try:
        # Method 1: Try to get from dataset directly
        if hasattr(dataset, 'total_tokens'):
            dataset_tokens = dataset.total_tokens
        # Method 2: Estimate from dataset size and sequence length
        elif hasattr(dataset, '__len__'):
            num_samples = len(dataset)
            seq_length = getattr(config, 'seq_length', 2048)
            dataset_tokens = num_samples * seq_length
        else:
            print("⚠️ Could not estimate dataset size")
            print("   Skipping auto-epoch scaling")
            return config
        
        print(f"📚 Dataset Tokens: {dataset_tokens:,} ({dataset_tokens/1e9:.2f}B)")
    except Exception as e:
        print(f"⚠️ Could not estimate dataset tokens: {e}")
        print("   Skipping auto-epoch scaling")
        return config
    
    # Chinchilla scaling: N_opt ≈ 20 * P (20 tokens per parameter)
    N_opt = int(20 * P)  # Optimal total tokens to see
    print(f"Chinchilla Optimal Tokens: {N_opt:,} ({N_opt/1e9:.2f}B)")
    
    # Calculate needed epochs
    if dataset_tokens <= 0:
        print("⚠️ Invalid dataset token count")
        return config
    
    tokens_per_epoch = dataset_tokens
    optimal_epochs = max(1, round(N_opt / tokens_per_epoch))
    
    # Get current epoch setting
    old_epochs = getattr(config, 'num_epochs', 3)
    
    # Apply constraints
    min_epochs = getattr(config, 'min_auto_epochs', 1)
    max_epochs = getattr(config, 'max_auto_epochs', 50)
    final_epochs = max(min_epochs, min(optimal_epochs, max_epochs))
    
    print(f"\n📈 Epoch Calculation:")
    print(f"   Tokens per epoch: {tokens_per_epoch:,}")
    print(f"   Optimal epochs (unconstrained): {optimal_epochs}")
    print(f"   Epoch constraints: {min_epochs} - {max_epochs}")
    print(f"   Original config: {old_epochs} epochs")
    print(f"   ➡️ Adjusted to: {final_epochs} epochs")
    
    # Calculate total training tokens
    total_tokens = tokens_per_epoch * final_epochs
    chinchilla_ratio = (total_tokens / N_opt) * 100
    
    print(f"\n📢 Training Token Budget:")
    print(f"   Total tokens (new): {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"   Chinchilla target: {N_opt:,} ({N_opt/1e9:.2f}B)")
    print(f"   Coverage: {chinchilla_ratio:.1f}% of optimal")
    
    if chinchilla_ratio < 50:
        print(f"   ⚠️ WARNING: Significantly under Chinchilla recommendation")
        print(f"      Consider increasing max_auto_epochs or dataset size")
    elif chinchilla_ratio > 150:
        print(f"   ⚠️ WARNING: Exceeding Chinchilla recommendation")
        print(f"      May lead to overfitting - consider early stopping")
    else:
        print(f"   ✓ Within reasonable range of Chinchilla scaling")
    
    # Update config
    config.num_epochs = final_epochs
    
    print("="*80 + "\n")
    
    return config


def main():
    """Main training function with advanced features and comprehensive logging."""
    
    # ========================================================================
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS
    # ========================================================================
    
    # Base model configuration
    config_choice = 'debug'  # Options: 'debug', 'lean_8gb', 'debug_200m', 'b1', 'b7', 'b14', 'b200', 'b300'
    
    # Auto-detect low memory environments (like 8GB Macs)
    try:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        if total_mem_gb < 9.0 and config_choice == 'debug':
            print(f"  ⚠️  LOW MEMORY DETECTED: {total_mem_gb:.1f}GB RAM")
            print(f"  Automatically switching to 'lean_8gb' preset for stability")
            config_choice = 'lean_8gb'
    except:
        pass
    
    # Training mode selection
    use_adaptive_training = TRAINING_INFRASTRUCTURE_AVAILABLE  # Orchestrator with AI-driven optimization

    # ========================================================================
    # 📊 STANDARD TRAINING PARAMETERS
    # ========================================================================
    training_params = {
        'use_moe': True,
        'use_mod': True,
        'use_cuda_moe': True,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'use_lr_scheduler': True,
        'lr_scheduler': "cosine",  # cosine, constant, or linear
        'warmup_ratio': 0.02,
        'batch_size': 30,
        'gradient_accumulation_steps': 8,
        
        'precision': "fp16",
        'inference_precision': "fp16",
        'num_experts': 8,
        'moe_top_k': 2,
        'compile': True,
        
        'max_memory_usage': 0.95,
        'save_every_n_batches': 1000,
        'eval_every_n_batches': 500,
        'use_flash_attention': True,
        'gradient_checkpointing': True,

        'num_workers': 0,
        'save_total_limit': 5,
        'weight_decay': 0.01,
    }

    # ========================================================================
    # 1. ADAPTIVE INTELLIGENCE PARAMETERS
    # ========================================================================
    adaptive_intelligence_params = {
        # Meta-learning confidence thresholds
        'meta_confidence_soft': 0.70,       # Start suggesting changes
        'meta_confidence_medium': 0.80,     # Make moderate changes
        'meta_confidence_hard': 0.90,       # Make aggressive changes
        'meta_confidence_critical': 0.95,   # Emergency interventions
        
        # Decision memory
        'strategy_memory_size': 20,         # Remember top N strategies
        'learning_transfer_weight': 0.8,    # Trust past learnings (0-1)
        
        # Risk tolerance
        'adaptive_risk_tolerance': 'balanced',  # 'conservative', 'balanced', 'aggressive'
        'exploration_rate': 0.15,           # Try new strategies 15% of time
    }
    
    # ========================================================================
    # 2. DYNAMIC ARCHITECTURE PARAMETERS
    # ========================================================================
    dynamic_architecture_params = {
        # MoE Expert Management
        'dynamic_expert_management': True,
        'expert_growth_threshold': 0.85,    # Add expert when utilization > 85%
        'expert_prune_threshold': 0.15,     # Remove when utilization < 15%
        'max_experts_per_layer': 16,        # Safety limit
        'min_experts_per_layer': 4,         # Minimum experts
        
        # MoD Adaptive Capacity
        'mod_capacity_adaptation': True,
        'mod_early_training_aggr': 0.7,     # More computation early
        'mod_mid_training_aggr': 0.5,       # Balanced
        'mod_late_training_aggr': 0.3,      # Aggressive savings late
        
        # Layer-specific routing
        'per_layer_routing_config': True,
        'attention_heavy_layers': [0, 1, 2, -3, -2, -1],  # More compute here
    }
    
    # ========================================================================
    # 3. PREDICTIVE OPTIMIZATION PARAMETERS
    # ========================================================================
    predictive_optimization_params = {
        # Convergence prediction
        'convergence_prediction_horizon': 1000,  # Steps to look ahead
        'plateau_detection_window': 200,         # Steps for plateau detection
        'divergence_early_warning': 50,          # Early warning steps
        
        # Resource forecasting
        'memory_trend_analysis': True,
        'oom_prediction_confidence': 0.85,
        'throughput_optimization_mode': True,
        
        # Smart checkpointing
        'importance_based_checkpointing': True,
        'checkpoint_quality_metric': 'loss_gradient',  # 'loss_gradient' or 'convergence_rate'
    }
    
    # ========================================================================
    # 4. QUALITY-AWARE TRAINING PARAMETERS
    # ========================================================================
    quality_aware_training_params = {
        # Loss landscape awareness
        'loss_smoothness_threshold': 0.01,      # Maximum acceptable noise
        'gradient_health_monitoring': True,
        'curvature_aware_training': True,       # Adjust for loss landscape
        
        # Output quality guards
        'max_perplexity_spike': 2.0,            # Reject if perplexity doubles
        'min_quality_improvement': 0.001,       # Min improvement per epoch
        
        # Knowledge retention
        'catastrophic_forgetting_threshold': 0.15,  # Max base knowledge loss
        'knowledge_preservation_strength': 0.3,     # How strongly to preserve
    }
    
    # ========================================================================
    # ⚡ 5. HARDWARE-AWARE OPTIMIZATION PARAMETERS
    # ========================================================================
    hardware_aware_optimization_params = {
        # GPU architecture specific
        'hardware_optimization_level': 'aggressive',  # 'minimal', 'balanced', 'aggressive'
        'tensor_core_optimization': 'aggressive',
        'memory_bus_utilization_target': 0.85,
        
        # Multi-GPU optimization
        'communication_overlap_aggressiveness': 0.8,
        'gradient_sync_strategy': 'adaptive',    # 'smart', 'eager', 'lazy'
        
        # Power efficiency
        'power_efficiency_mode': False,
        'thermal_throttling_avoidance': True,
    }
    
    # ========================================================================
    # 6. DATA INTELLIGENCE PARAMETERS
    # ========================================================================
    data_intelligence_params = {
        # Dynamic data sampling
        'difficulty_based_sampling': True,
        'curriculum_learning_aggressiveness': 0.7,
        'hard_example_mining_threshold': 0.1,
        
        # Quality filtering
        'automatic_data_cleaning': True,
        'data_quality_threshold': 0.85,
        'diversity_penalty': 0.1,               # Penalize repetitive patterns
        
        # Adaptive batch construction
        'sequence_length_optimization': True,
        'similarity_aware_batching': True,      # Batch similar lengths together
    }
    
    # ========================================================================
    # 7. SAFETY & ROBUSTNESS PARAMETERS
    # ========================================================================
    safety_robustness_params = {
        # Training stability
        'maximum_acceptable_instability': 0.05,
        'recovery_aggressiveness': 0.8,
        'emergency_rollback_depth': 500,        # Steps to rollback
        
        # Output safety
        'toxicity_monitoring': True,
        'bias_detection_sensitivity': 0.7,
        'factuality_guards': True,
        
        # Model health
        'max_weight_norm': 10.0,
        'max_gradient_norm': 5.0,
    }
    
    # ========================================================================
    # 8. MULTI-OBJECTIVE OPTIMIZATION PARAMETERS
    # ========================================================================
    multi_objective_optimization_params = {
        # Trade-off management
        'speed_quality_tradeoff': 0.5,          # 0 = max speed, 1 = max quality
        'memory_performance_balance': 0.6,      # 0 = min memory, 1 = max perf
        
        # Objective weights
        'primary_objective': 'quality',         # 'quality', 'speed', 'stability', 'efficiency'
        'secondary_objective_speed': 0.3,
        'secondary_objective_memory': 0.3,
        'secondary_objective_quality': 0.4,
        
        # Constraint handling
        'hard_constraints_memory': True,
        'hard_constraints_time': True,
        'soft_constraints_quality': True,
        'soft_constraints_stability': True,
    }
    
    # ========================================================================
    # 9. ADAPTIVE LR PARAMETERS
    # ========================================================================
    adaptive_lr_params = {
        'enable_adaptive_lr': True,
        'allow_scheduler_override': True,
        'min_override_threshold': 0.2,
        'emergency_override_enabled': True,
        'log_lr_decisions': True,
    }
    
    # ========================================================================
    # 10. DATA CONFIGURATION
    # ========================================================================
    data_params = {
        # Base training paths (pre-training on raw text - .txt and .jsonl)
        'base_training_paths': [
            'datasets/wikipedia_1.txt',
            'datasets/wikipedia_2.txt',
            'datasets/stackoverflow_1.txt',
            'datasets/stackoverflow_2.txt',
            'datasets/gutenberg_1.txt',
            'datasets/arxiv_1.txt',
            'datasets/arxiv_2.txt',
            'datasets/pubmed_1.txt',
            'datasets/pubmed_2.txt',
            'datasets/pubmed_3.txt',
            'datasets/pubmed_4.txt',  
            'datasets/openwebtext_1.txt',     
            'datasets/ccnews_1.txt',
        ],

        'base_eval_paths': [  # Only .jsonl
            'datasets/oasst1_validation.jsonl',
        ],

        # Fine-tuning paths (instruction tuning - only .jsonl)
        'finetuning_paths': [
            'datasets/oasst1_train.jsonl',
            'datasets/oasst1_train_part2.jsonl',
            'datasets/oasst1_train_part3.jsonl',
        ],

        'finetuning_eval_paths': [
            'datasets/oasst1_validation.jsonl',
        ],

        # Training mode
        'training_mode': 'finetuning_only',  # 'base_only', 'finetuning_only', 'hybrid', 'interleaved'
        'base_finetuning_ratio': 0.7,  # For interleaved mode: 70% base, 30% fine-tuning

        # Dataset processing
        'max_conversations_per_dataset': None,
        'validate_datasets': True,
        'cache_combined_dataset': True,
        'streaming_threshold_gb': 10.0,  # Use streaming for files > 10GB
    }
    
    # ========================================================================
    # 11. DEEPSPEED & QUANTIZATION CONFIGURATION
    # ========================================================================
    deepspeed_params = {
        'use_deepspeed': False,
        'cpu_offload': True,
        'cpu_offload_optimizer': True,
        'cpu_offload_parameters': True,
        'zero_stage': 3,
        'nvme_path': None,
        'max_grad_norm': 1.0,
    }
    
    quantization_params = {
        'quantization_method': None,  # Options: None, 'bnb', 'gptq', 'quanto'
        'quantization_bits': None,    # Options: None, 4, 8
    }
    
    # ========================================================================
    # 💾 12. CHECKPOINT PARAMETERS
    # ========================================================================
    checkpoint_params = {
        'resume_from_checkpoint': None,  # Set to 'checkpoint_final_597.pt' to resume
        'resume_training': False,        # Set to True to resume
        'reset_optimizer': False,       
        'reset_scheduler': False,       
    }
    
    # Fixed: Handle None checkpoint path safely
    checkpoint_file = checkpoint_params.get('resume_from_checkpoint')
    if checkpoint_file:
        checkpoint_path = Path(checkpoint_file)
        if checkpoint_params['resume_training'] and not checkpoint_path.exists():
            print(f"⚠️ WARNING: Checkpoint not found: {checkpoint_path}")
            print("   Starting training from scratch instead")
            checkpoint_params['resume_training'] = False
    elif checkpoint_params['resume_training']:
        # resume_training is True but no checkpoint specified
        print(f"⚠️ WARNING: resume_training=True but no checkpoint specified")
        print("   Starting training from scratch instead")
        checkpoint_params['resume_training'] = False
    
    # ========================================================================
    # 13. MONITORING & LOGGING PARAMETERS
    # ========================================================================
    monitoring_params = {
        'verbosity': 'normal',  #'silent', 'minimal', 'normal', 'detailed', 'debug', 'trace'
        'log_level': "INFO",
        'experiment_name': f'Enhanced_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'early_stopping_patience': 5,
        'backup_every_n_hours': 12,
        'enable_wandb': True,
        'wandb_project': 'deepseek-moe-training',
        'wandb_entity': 'matiasnhmb',
        'health_check_interval': 50,
        'log_every_n_steps': 50,
        'adaptive_log_frequency': 100,
    }
    
    # ========================================================================
    # 14. ADVANCED FEATURES CONFIGURATION
    # ========================================================================
    advanced_features = {
        'enable_data_validation': True,
        'generate_data_reports': True,
        'estimate_training_time': True,
        'generate_training_reports': True,
        'auto_tune_batch_size': False,
        'continuous_checkpointing': True,
        'enable_profiling': False,
        'save_optimizer_states': True,
    }
    
    # ========================================================================
    # 15. CHINCHILLA SCALING PARAMETERS
    # ========================================================================
    chinchilla_params = {
        'auto_epoch_scaling': True,
        'min_auto_epochs': 1,
        'max_auto_epochs': 50,
        'chinchilla_multiplier': 20,
        'enable_loss_landscape': True,
        'enable_compute_efficiency': True,
        'enable_adaptive_curriculum': True,
        'enable_early_stopping': True,
        'plateau_patience': 5,
        'efficiency_decline_threshold': 0.3,
        'convergence_threshold': 0.85,
        'enable_memory_aware_scaling': True,
        'quality_aware_adjustment': True,
    }

    # ========================================================================
    # 16. BACKEND PARAMS
    # ========================================================================
    backend_params = {
        'backend': 'fsdp',  # Options: 'fsdp', 'deepspeed', 'colossalai', 'pytorch' colossalai not recommended
        
        # FSDP specific
        'use_fsdp': False,
        'fsdp_sharding_strategy': 'FULL_SHARD',
        'fsdp_auto_wrap_threshold': 1e8,
        
        # Original DeepSpeed
        'use_deepspeed': False,
        'zero_stage': 3,
        
    }

    # ========================================================================
    # 17. ROUTER MODEL TRAINING PARAMETERS
    # ========================================================================
    router_training_params = {
        'enable_router_training': False,  # Set to True when ready
        
        # Data collection
        'collection_batches': 1000,  # Collect from first 1000 batches
        
        # Router architecture (14M params)
        'router_hidden': 256,
        'router_layers': 4,
        'router_heads': 4,
        
        # Training
        'router_epochs': 10,
        'router_batch_size': 32,
        'router_learning_rate': 5e-4,
        
        # Loss weights
        'classification_weight': 1.0,
        'distribution_weight': 0.5,
        'load_balance_weight': 0.01,
    }

    # ========================================================================
    # 18. PRE-TRAINED ROUTER PARAMETERS
    # ========================================================================
    pretrained_router_params = {
        'use_pretrained_router': False,  # ← Set to True!
        'router_checkpoint_path': 'router_training/best_router_model.pt',
        
        # Optional fine-tuning
        'finetune_router': False,  # Keep frozen or fine-tune?
        'router_learning_rate': 1e-5,  # If fine-tuning
    }
    # ========================================================================
    # END CONFIGURATION SECTION
    # ========================================================================

    # Validate data paths FIRST
    if not validate_data_paths(data_params):
        print("\n✗ Data path validation failed. Cannot continue.")
        print("Please check your file paths in the data_params configuration.\n")
        return 1
    
    # Check MPS compatibility if applicable
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Will validate after config is created
        is_mps = True
    else:
        is_mps = False
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, monitoring_params['log_level']),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    print_banner("DEEPSEEK MOE TRANSFORMER - TRAINING SYSTEM")
    print(f"Experiment: {monitoring_params['experiment_name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Adaptive AI-Driven' if use_adaptive_training else 'Standard'} Training")
    print("")
    print("ENHANCED FEATURES ENABLED:")
    print("  ✓ Adaptive Intelligence")
    print("  ✓ Dynamic Architecture")
    print("  ✓ Predictive Optimization")
    print("  ✓ Quality-Aware Training")
    print("  ✓ Hardware-Aware Optimization")
    print("  ✓ Data Intelligence")
    print("  ✓ Safety & Robustness")
    print("  ✓ Multi-Objective Optimization")
    print("")
    print(f"System Status:")
    print(f"  Training Infrastructure: {'Available' if TRAINING_INFRASTRUCTURE_AVAILABLE else 'Not Available'}")
    print(f"  Utils Modules: {'Available' if UTILS_AVAILABLE else 'Not Available'}")
    print(f"  DeepSpeed: {'Available' if DEEPSPEED_AVAILABLE else 'Not Available'}")
    print(f"  CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}")
    if torch.cuda.is_available():
        print(f"  GPU Count: {torch.cuda.device_count()}")
    print("="*80)
    
    try:
        # Step 1: System diagnostics
        print_banner("STEP 1: SYSTEM DIAGNOSTICS")
        print_system_diagnostics()
        
        # Step 2: Create base configuration
        print_banner("STEP 2: CREATING ENHANCED CONFIGURATION")
        print(f"Loading configuration preset: {config_choice}")
        
        if hasattr(ConfigPresets, config_choice):
            config = getattr(ConfigPresets, config_choice)()
            print(f"Base configuration loaded successfully")
        else:
            raise ValueError(f"Unknown config preset: {config_choice}")

        # FORCE REMOVE ANY LIMITS
        config.limit_train_batches = None
        config.max_train_steps = None  
        config.fast_dev_run = False

        print("="*80)
        print("DEBUG: FORCING FULL DATASET TRAINING")
        print("="*80)   
        
        # Apply all parameter overrides
        all_params = {
            **training_params,
            **adaptive_intelligence_params,
            **dynamic_architecture_params,
            **predictive_optimization_params,
            **quality_aware_training_params,
            **hardware_aware_optimization_params,
            **data_intelligence_params,
            **safety_robustness_params,
            **multi_objective_optimization_params,
            **adaptive_lr_params,
            **data_params, 
            **deepspeed_params, 
            **checkpoint_params,
            **quantization_params,
            **monitoring_params,
            **advanced_features,
            **chinchilla_params,
            **backend_params,
        }
        
        print_section("Applying Enhanced Parameter Overrides")
        print(f"Total parameters to configure: {len(all_params)}")
        
        # Group parameters by category for better output
        param_categories = {
            'Adaptive Intelligence': adaptive_intelligence_params,
            'Dynamic Architecture': dynamic_architecture_params,
            'Predictive Optimization': predictive_optimization_params,
            'Quality-Aware Training': quality_aware_training_params,
            'Hardware Optimization': hardware_aware_optimization_params,
            'Data Intelligence': data_intelligence_params,
            'Safety & Robustness': safety_robustness_params,
            'Multi-Objective': multi_objective_optimization_params,
        }
        
        for category, params in param_categories.items():
            print(f"\n{category}:")
            configured_count = 0
            for key, value in params.items():
                setattr(config, key, value)
                configured_count += 1
            print(f"  ✓ Configured {configured_count} parameters")
        
        # Apply remaining parameters silently
        override_count = 0
        for key, value in all_params.items():
            if value is not None:
                old_value = getattr(config, key, None)
                setattr(config, key, value)
                if old_value is not None and old_value != value and key not in ['raw_oasst_path']:
                    override_count += 1
        
        print(f"\n✓ Applied {override_count} additional configuration overrides")
        print(f"✓ Total enhanced parameters: {len(all_params)}")
        
        # Validate MPS compatibility if applicable
        if is_mps:
            print_banner("MPS COMPATIBILITY CHECK")
            is_compatible, issues = validate_mps_compatibility(config)
            
            if not is_compatible:
                print("MPS Compatibility Issues Detected:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
                
                print("\nApplying automatic fixes...")
                config.apply_device_optimizations('mps')
                
                # Re-validate
                is_compatible, remaining_issues = validate_mps_compatibility(config)
                if remaining_issues:
                    print("\nRemaining issues after auto-fix:")
                    for issue in remaining_issues:
                        print(f" {issue}")
            else:
                print("✓ Configuration is MPS compatible")
        
        # Step 3: Validate precision support
        print_banner("STEP 3: VALIDATING PRECISION SUPPORT")
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if is_mps else 'cpu'))
        training_precision = training_params.get('precision', 'fp32')
        inference_precision = training_params.get('inference_precision', 'fp16')
        
        print(f"Device: {device}")
        print(f"Training Precision: {training_precision}")
        print(f"Inference Precision: {inference_precision}")
        
        # Validate training precision
        is_supported, error_msg = validate_precision_support(training_precision, device)
        if not is_supported:
            print(f"\nERROR: Precision Validation Failed")
            print(f"  {error_msg}")
            print(f"\nTraining cannot continue with unsupported precision.")
            return 1
        else:
            print(f"✓ Training precision validated successfully")
        
        # Validate inference precision
        is_supported, error_msg = validate_precision_support(inference_precision, device)
        if not is_supported:
            print(f"\nInference precision not supported")
            print(f"  {error_msg}")
            print(f"  Using training precision for inference instead")
            config.inference_precision = training_precision
        else:
            print(f"✓ Inference precision validated successfully")
        
        # Step 4: Validate configuration
        print_banner("STEP 4: VALIDATING ENHANCED CONFIGURATION")
        try:
            config.validate()
            print("✓ Configuration validation passed")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Sequence length: {config.seq_length}")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Epochs: {config.num_epochs}")
            print(f"  Risk Tolerance: {config.adaptive_risk_tolerance}")
            print(f"  Hardware Optimization: {config.hardware_optimization_level}")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return 1
        
        # Step 5: Initialize tokenizer
        print_banner("STEP 5: INITIALIZING TOKENIZER")
        tokenizer = ConversationTokenizer()
        config.vocab_size = tokenizer.vocab_size
        print(f"✓ Tokenizer initialized successfully")
        print(f"  Vocabulary size: {config.vocab_size:,}")
        print(f"  Special tokens: {len(tokenizer.special_tokens) if hasattr(tokenizer, 'special_tokens') else 'N/A'}")
        
        # Step 6: Data preparation and validation (ADVANCED)
        datasets_info = None
        if advanced_features.get('enable_data_validation'):
            datasets_info = prepare_and_validate_data(config, tokenizer)
        else:
            print_banner("STEP 6: DATA PREPARATION (BASIC)")
            print("Skipping advanced data validation")
        
        # Step 7: Setup datasets with data intelligence
        print_banner("STEP 7: SETTING UP DATASETS WITH DATA INTELLIGENCE")
        
        from core.dataset import HybridDatasetManager, setup_datasets

        # Transfer data params to config attributes
        config.base_training_paths = data_params.get('base_training_paths', [])
        config.base_eval_paths = data_params.get('base_eval_paths', [])
        config.finetuning_paths = data_params.get('finetuning_paths', [])
        config.finetuning_eval_paths = data_params.get('finetuning_eval_paths', [])
        config.training_mode = data_params.get('training_mode', 'finetuning_only')
        config.data_cache_dir = data_params.get('data_cache_dir', 'data/cache')
        config.streaming_threshold_gb = data_params.get('streaming_threshold_gb', 10.0)
        config.base_finetuning_ratio = data_params.get('base_finetuning_ratio', 0.5)
        config.max_conversations_per_dataset = data_params.get('max_conversations_per_dataset', None)

        print(f"\nTraining mode: {config.training_mode}")
        print(f"Data Intelligence Features:")
        print(f"  Difficulty-based sampling: {config.difficulty_based_sampling}")
        print(f"  Curriculum learning aggressiveness: {config.curriculum_learning_aggressiveness}")
        print(f"  Automatic data cleaning: {config.automatic_data_cleaning}")
        print(f"  Quality threshold: {config.data_quality_threshold}")
        print(f"  Sequence length optimization: {config.sequence_length_optimization}")

        try:
            train_dataset, eval_dataset = setup_datasets(config, tokenizer)

            print(f"\n✓ Datasets loaded successfully!")
            print(f"  Training dataset: {len(train_dataset):,} samples")
            if eval_dataset != train_dataset:
                print(f"  Evaluation dataset: {len(eval_dataset):,} samples")
            else:
                print(f"  Using training data for evaluation")

        except Exception as e:
            print(f"\n✗ Dataset loading failed: {e}")
            traceback.print_exc()
            return 1

        # Display dataset statistics
        print("\n" + "="*80)
        print("DATASET & TRAINING CALCULATION VERIFICATION")
        print("="*80)
        print(f"Dataset size: {len(train_dataset):,} conversations")
        print(f"Batch size: {config.batch_size}")
        print(f"Expected batches per epoch: {len(train_dataset) / config.batch_size:.2f}")
        print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Expected optimizer steps per epoch: {len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps):.2f}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Total epochs: {config.num_epochs}")
        print(f"Total optimizer steps for training: {(len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs:.0f}")
        print("="*80 + "\n")
        
        # Step 8: Estimate training time (ADVANCED)
        print_banner("STEP 8: ESTIMATE TRAINING TIME")
        if advanced_features.get('estimate_training_time'):
            estimate_and_display_training_time(config, len(train_dataset))
        
        # Step 9: Initialize model with FSDP backend
        print_banner("STEP 9: INITIALIZING MODEL WITH FSDP BACKEND")
        print("Creating model configuration...")
        model_config = config_to_deepseek_config(config)

        print("Initializing base model architecture...")
        base_model = DeepSeekTransformer(model_config)

        # Apply FP16-safe initialization
        def init_weights_for_fp16(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        base_model.apply(init_weights_for_fp16)
        print("✓ Applied FP16-safe weight initialization")

        # Apply Triton FP8 Emulation (Optional)
        if getattr(config, 'use_triton_fp8', False):
            print_section("Applying Triton FP8 Emulation")
            if is_triton_available():
                # Replace Linear layers (except heads/embeddings usually, but replace_linear handles nn.Linear)
                # Typically we might want to skip the output head or embeddings?
                # nn.Embedding is not nn.Linear. Output head probably is.
                # For now, replace all.
                replace_linear_with_fp8(base_model)
                print("✓ Replaced Linear layers with TritonFP8Linear")
            else:
                print("⚠️ Triton not available - skipping FP8 emulation")

        # Determine backend and wrap model
        backend_choice = backend_params.get('backend', 'pytorch')
        use_fsdp = backend_params.get('use_fsdp', False)
        use_deepspeed = backend_params.get('use_deepspeed', False)

        use_deepspeed_remake = backend_params.get('use_deepspeed_remake', False)
        deepspeed_integration_needed = False
        expert_registry = {}

        # ----- SANITIZE CONFLICTS BETWEEN DEEPSPEED <-> FSDP FLAGS -----
        # Prevent deepspeed params from accidentally affecting FSDP runs and vice-versa.
        if backend_choice == 'fsdp' or use_fsdp:
            if getattr(config, 'use_deepspeed', False):
                logging.warning("Config sets use_deepspeed=True but backend_choice is FSDP — disabling use_deepspeed to avoid conflicts")
                config.use_deepspeed = False
            # also clear DeepSpeed-specific attrs that some code may check
            for attr in ['zero_stage', 'cpu_offload', 'nvme_path']:
                if hasattr(config, attr):
                    try:
                        setattr(config, attr, None if attr.endswith('path') else False)
                    except Exception:
                        pass

        # If DeepSpeed is explicitly selected, clear FSDP-specific flags
        if backend_choice == 'deepspeed_remake' or use_deepspeed_remake:
            if getattr(config, 'use_deepspeed', False):
                logging.warning("Disabling original DeepSpeed to use DeepSpeed remake")
                config.use_deepspeed = False
            if getattr(config, 'use_fsdp', False):
                logging.warning("Disabling FSDP to use DeepSpeed remake")
                config.use_fsdp = False
            
            for attr in ['fsdp_sharding_strategy', 'fsdp_auto_wrap_threshold', 'zero_stage']:
                if hasattr(config, attr):
                    setattr(config, attr, None)

        # ---------------------------------------------------------------
        print(f"\nBackend Selection:")
        print(f"  Requested backend: {backend_choice}")
        print(f"  FSDP available: {BACKEND_FSDP_AVAILABLE}")
        print(f"  DeepSpeed available: {BACKEND_DEEPSPEED_AVAILABLE}")
        print(f"  DeepSpeed Remake available: {DEEPSPEED_REMAKE_AVAILABLE}")

        # Check if we're in a distributed environment
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_distributed = world_size > 1

        print(f"  Distributed Environment: {'Yes' if is_distributed else 'No'}")
        print(f"  World Size: {world_size}")
        print(f"  Local Rank: {local_rank}")
        if (backend_choice == 'deepspeed_remake' or use_deepspeed_remake) and DEEPSPEED_REMAKE_AVAILABLE:
            print("\n" + "="*80)
            print("INITIALIZING DEEPSPEED REMAKE BACKEND")
            print("="*80)
            
            remake_config = backend_params.get('deepspeed_remake_config', {})
            for key, value in remake_config.items():
                setattr(config, key, value)
            
            for key, value in backend_params.items():
                if not hasattr(config, key):
                    setattr(config, key, value)
            
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 
                                    ('mps' if is_mps else 'cpu'))
                model = base_model.to(device)
                
                print(f"Model moved to {device}")
                print("\nDeepSpeed Remake Configuration:")
                print(f"  ZeRO Stage: {config.zero_stage}")
                print(f"  CPU Offload: {config.cpu_offload}")
                print(f"  CPU Offload Parameters: {getattr(config, 'cpu_offload_parameters', False)}")
                print(f"  NVMe Path: {getattr(config, 'nvme_path', None) or 'None'}")
                print(f"  Overlap Communication: {getattr(config, 'overlap_comm', True)}")
                print(f"  Gradient Clipping: {getattr(config, 'gradient_clipping', 1.0)}")
                
                if config.use_moe:
                    print(f"\nBuilding expert registry for MoE model...")
                    for name, module in model.named_modules():
                        if 'expert' in name.lower() or 'moe' in name.lower():
                            expert_registry[name] = module
                    print(f"  Found {len(expert_registry)} expert modules")
                
                print("\nDeepSpeed Remake will be integrated after orchestrator creation...")
                
                using_backend = "DeepSpeed Remake (Pending)"
                deepspeed_integration_needed = True
                
            except Exception as e:
                print(f"✗ DeepSpeed Remake initialization failed: {e}")
                traceback.print_exc()
                print("Falling back to PyTorch backend...")
                model = base_model
                device = torch.device('cuda' if torch.cuda.is_available() else 
                                    ('mps' if is_mps else 'cpu'))
                model = model.to(device)
                using_backend = "PyTorch (DeepSpeed Remake fallback)"
                deepspeed_integration_needed = False

        # Priority: DeepSpeed Remake > FSDP > Original DeepSpeed > PyTorch
        elif (backend_choice == 'fsdp' or use_fsdp) and BACKEND_FSDP_AVAILABLE and is_distributed:
            print("\n" + "="*80)
            print("INITIALIZING FSDP BACKEND")
            print("="*80)
            print("\n⚠️  FSDP requires distributed environment (multi-GPU)")
            print("   Single GPU detected - falling back to PyTorch backend")
            backend_choice = 'pytorch'
            use_fsdp = False
            
            # Transfer backend params to config
            for key, value in backend_params.items():
                if not hasattr(config, key):
                    setattr(config, key, value)
            
            try:
                model = create_fsdp_backend(base_model, config)
                
                print("✓ FSDP backend initialized successfully")
                print(f"\nFSDP Configuration:")
                print(f"  Backend: {model.backend_name}")
                print(f"  Sharding Strategy: {getattr(config, 'fsdp_sharding_strategy', 'FULL_SHARD')}")
                print(f"  World Size: {model.world_size}")
                print(f"  Local Rank: {model.local_rank}")
                print(f"  Mixed Precision: {config.precision}")
                print(f"  CPU Offload: {getattr(config, 'cpu_offload', False)}")
                print(f"  Auto Wrap Threshold: {getattr(config, 'fsdp_auto_wrap_threshold', 1e8):.0e} params")
                print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
                
                # Setup scheduler after FSDP initialization
                steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
                total_steps = steps_per_epoch * config.num_epochs
                model.setup_scheduler(total_steps)
                print(f"  Scheduler: {'Configured' if model.scheduler else 'None'}")
                
                using_backend = "FSDP"
                
            except Exception as e:
                print(f"✗ FSDP initialization failed: {e}")
                print("Falling back to PyTorch backend...")
                model = base_model
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if is_mps else 'cpu'))
                model = model.to(device)
                using_backend = "PyTorch (FSDP fallback)"
        elif backend_choice == 'colossalai':
            print("\n" + "="*80)
            print("INITIALIZING COLOSSAL-AI BACKEND")
            print("="*80)
            
            try:
                model = create_colossalai_backend(base_model, config)
                using_backend = "Colossal-AI"
                print("✓ Colossal-AI backend initialized successfully")
            except Exception as e:
                print(f"✗ Colossal-AI initialization failed: {e}")
                print("Falling back to PyTorch backend...")
                model = base_model
                device = torch.device('cuda' if torch.cuda.is_available() else 
                                    ('mps' if is_mps else 'cpu'))
                model = model.to(device)
                using_backend = "PyTorch (Colossal-AI fallback)"

        elif (backend_choice == 'deepspeed' or use_deepspeed) and BACKEND_DEEPSPEED_AVAILABLE:
            print("\n" + "="*80)
            print("INITIALIZING DEEPSPEED BACKEND")
            print("="*80)
            
            # Transfer backend params to config
            for key, value in backend_params.items():
                if not hasattr(config, key):
                    setattr(config, key, value)
            
            try:
                model = create_deepspeed_backend(base_model, config)
                
                print("✓ DeepSpeed backend initialized successfully")
                print(f"\nDeepSpeed Configuration:")
                print(f"  Backend: {model.backend_name}")
                print(f"  ZeRO Stage: {getattr(config, 'zero_stage', 2)}")
                print(f"  World Size: {model.world_size}")
                print(f"  Local Rank: {model.local_rank}")
                print(f"  Mixed Precision: {config.precision}")
                print(f"  CPU Offload: {getattr(config, 'cpu_offload', False)}")
                print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
                
                using_backend = "DeepSpeed"
                
            except Exception as e:
                print(f"✗ DeepSpeed initialization failed: {e}")
                print("Falling back to PyTorch backend...")
                model = base_model
                device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if is_mps else 'cpu'))
                model = model.to(device)
                using_backend = "PyTorch (DeepSpeed fallback)"

        else:
            print("\n" + "="*80)
            print("USING STANDARD PYTORCH BACKEND")
            print("="*80)
            
            model = base_model
            device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if is_mps else 'cpu'))
            model = model.to(device)
            
            print(f"✓ Model moved to device: {device}")
            using_backend = "PyTorch"

        # Model compilation (only for PyTorch backend, not for FSDP/DeepSpeed)
        if using_backend == "PyTorch" and torch.__version__ >= "2.0" and not is_mps and config.compile:
            print("\nCompiling model with torch.compile...")
            try:
                model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=True,
                    dynamic=False
                )
                print("✓ Model compiled successfully")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
                print("Continuing without compilation...")

        # Calculate and display model statistics
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

        print("\n" + "="*80)
        print("MODEL STATISTICS")
        print("="*80)
        print(f"Backend: {using_backend}")
        print(f"\nArchitecture: DeepSeek Transformer")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Attention Heads: {config.num_heads}")
        print(f"  Sequence Length: {config.seq_length}")
        print(f"  Vocab Size: {config.vocab_size:,}")

        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {total_params - trainable_params:,}")

        print(f"\nModel Size Estimates:")
        print(f"  FP32: {total_params * 4 / 1e9:.2f} GB")
        print(f"  FP16: {total_params * 2 / 1e9:.2f} GB")
        print(f"  BF16: {total_params * 2 / 1e9:.2f} GB")

        if config.use_moe:
            print(f"\nMixture of Experts Configuration:")
            print(f"  Number of Experts: {config.num_experts}")
            print(f"  Top-K Routing: {config.moe_top_k}")
            print(f"  Capacity Factor: {config.capacity_factor}")
            print(f"  Load Balancing Weight: {config.load_balancing_weight}")
            
            if config.dynamic_expert_management:
                print(f"\nDynamic Architecture Features:")
                print(f"  Dynamic Expert Management: Enabled")
                print(f"  Expert Growth Threshold: {config.expert_growth_threshold}")
                print(f"  Expert Prune Threshold: {config.expert_prune_threshold}")
                print(f"  Max Experts Per Layer: {config.max_experts_per_layer}")
                print(f"  Min Experts Per Layer: {config.min_experts_per_layer}")

        if using_backend in ["FSDP", "DeepSpeed"]:
            print(f"\nDistributed Training:")
            print(f"  World Size: {model.world_size}")
            print(f"  Local Rank: {model.local_rank}")
            print(f"  Main Process: {model.is_main_process()}")
            
            memory_stats = model.get_memory_stats()
            if memory_stats:
                print(f"\nGPU Memory (Current Process):")
                print(f"  Allocated: {memory_stats.get('allocated_gb', 0):.2f} GB")
                print(f"  Reserved: {memory_stats.get('reserved_gb', 0):.2f} GB")
                print(f"  Max Allocated: {memory_stats.get('max_allocated_gb', 0):.2f} GB")

        print("="*80 + "\n")


        # Step 9.5: Auto-adjust epochs using Chinchilla scaling
        if getattr(config, 'auto_epoch_scaling', False):
            if CHINCHILLA_SCALER_AVAILABLE:
                print_banner("STEP 9.5: CHINCHILLA EPOCH SCALING")               
            else:
                print("⚠️ Chinchilla scaler not available, skipping auto-scaling")

        # ============================================================
        # Step 9.6 - TRAIN Router Model (BEFORE loading pretrained)
        # ============================================================
        if router_training_params.get('enable_router_training', False):
            print_banner("STEP 9.6: TRAINING ROUTER MODEL")
            
            from router_training_system import train_router_model
            
            # Prepare router config
            router_config = {
                'hidden_size': config.hidden_size,
                'num_experts': config.num_experts,
                'top_k': config.moe_top_k,
                **router_training_params  # Merge all router params
            }
            
            # Create a dataloader for router training
            from torch.utils.data import DataLoader
            
            router_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,  # Keep order for routing data collection
                num_workers=config.num_workers,
                collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
            )
            
            # Train the router
            try:
                trained_router = train_router_model(
                    existing_model=model,
                    train_dataloader=router_dataloader,
                    config=router_config,
                    output_dir="router_training"
                )
                
                print("\n✅ Router training complete!")
                print(f"   Best model saved to: router_training/best_router_model.pt")
                print(f"   You can now use this router by setting:")
                print(f"     pretrained_router_params['use_pretrained_router'] = True")
                
            except Exception as e:
                print(f"\n❌ Router training failed: {e}")
                traceback.print_exc()
                print("Continuing with standard gates...")

        # ============================================================
        # Step 9.7 - Load Pre-trained Router (AFTER optional training)
        # ============================================================
        if pretrained_router_params['use_pretrained_router']:
            print_banner("STEP 9.7: LOADING PRE-TRAINED ROUTER MODEL")
            model = load_pretrained_router(
                model,
                pretrained_router_params['router_checkpoint_path']
            )
            
            # Optional: Verify it loaded correctly
            from pathlib import Path
            if Path(pretrained_router_params['router_checkpoint_path']).exists():
                print("\n✓ Router model integration complete")
                print("  Model will use learned routing decisions during training")


        # ============================================================
        # Step 9.8 - Load Pre-trained Router (AFTER training)
        # ============================================================
        if pretrained_router_params['use_pretrained_router']:
            print_banner("STEP 9.7: LOADING PRE-TRAINED ROUTER MODEL")
            model = load_pretrained_router(
                model,
                pretrained_router_params['router_checkpoint_path']
            )
        
        # Step 10: Initialize enhanced training system
        print_banner("STEP 10: INITIALIZING ENHANCED TRAINING SYSTEM")
        
        orchestrator = None
        
        if use_adaptive_training and TRAINING_INFRASTRUCTURE_AVAILABLE:
            print("Initializing Enhanced Adaptive Training Orchestrator")
            print("\nENHANCED FEATURES ACTIVE:")
            print("  ✓ AI-driven hyperparameter optimization")
            print("  ✓ Adaptive intelligence with meta-learning")
            print("  ✓ Real-time performance monitoring")
            print("  ✓ Dynamic architecture optimization")
            print("  ✓ Predictive resource management")
            print("  ✓ Quality-aware training guards")
            print("  ✓ Hardware-specific optimizations")
            print("  ✓ Intelligent data sampling")
            print("  ✓ Multi-objective optimization")
            print("  ✓ Advanced safety & robustness")
            print("  ✓ Automatic recovery from failures")
            print("  ✓ Performance profiling and analysis")
            
            try:
                orchestrator = AdaptiveTrainingOrchestrator(config)
                print("DEBUG: Orchestrator created")
                
                orchestrator.initialize_training()

                # INLINE TEST - Cannot be skipped
                print("\n🔍 INLINE ADAPTIVE TEST")
                print(f"Trainer type: {type(orchestrator.trainer).__name__}")
                print(f"Has adjust_learning_rate: {hasattr(orchestrator.trainer, 'adjust_learning_rate')}")
                print(f"Has get_current_metrics: {hasattr(orchestrator.trainer, 'get_current_metrics')}")
                print(f"Has _monitoring_queue: {hasattr(orchestrator.trainer, '_monitoring_queue')}")
                print(f"Scheduler exists: {orchestrator.trainer.scheduler is not None if hasattr(orchestrator.trainer, 'scheduler') else 'No attr'}")
                print()
                print("DEBUG: Training initialized")
                
                print("\n" + "="*80)
                print("STEP 10.9: TESTING ADAPTIVE PIPELINE".center(80))
                print("\n✓ Orchestrator initialized successfully")
                print("\n" + "="*80)
                print("🔍 TRAINER VERIFICATION")
                print("="*80)

                # Initialize training system
                orchestrator.initialize_training()

                # STEP 10.9: TEST ADAPTIVE PIPELINE
                print_banner("STEP 10.9: TESTING ADAPTIVE PIPELINE")
                
                logging.info("Testing adaptive training pipeline...")
                
                # Test 1: get_current_metrics()
                try:
                    test_metrics = orchestrator.trainer.get_current_metrics()
                    logging.info(f"✅ Test 1 PASSED: get_current_metrics() works")
                    logging.info(f"   Returned: {type(test_metrics).__name__}")
                except Exception as e:
                    logging.error(f"❌ Test 1 FAILED: get_current_metrics() error: {e}")
                
                # Test 2: adjust_learning_rate()
                try:
                    original_lr = orchestrator.trainer.optimizer.param_groups[0]['lr']
                    orchestrator.trainer.adjust_learning_rate(1e-5, grace_period=5)
                    new_lr = orchestrator.trainer.optimizer.param_groups[0]['lr']
                    
                    if abs(new_lr - 1e-5) < 1e-9:
                        logging.info(f"✅ Test 2 PASSED: adjust_learning_rate() works")
                        logging.info(f"   LR changed: {original_lr:.2e} → {new_lr:.2e}")
                        # Restore original LR
                        orchestrator.trainer.adjust_learning_rate(original_lr, grace_period=0)
                    else:
                        logging.error(f"❌ Test 2 FAILED: LR not changed correctly")
                        logging.error(f"   Expected: 1e-5, Got: {new_lr:.2e}")
                except Exception as e:
                    logging.error(f"❌ Test 2 FAILED: adjust_learning_rate() error: {e}")
                
                # Test 3: Monitoring queue
                try:
                    has_queue = hasattr(orchestrator.trainer, '_monitoring_queue')
                    if has_queue:
                        logging.info(f"✅ Test 3 PASSED: Trainer has monitoring queue")
                    else:
                        logging.error(f"❌ Test 3 FAILED: Trainer missing _monitoring_queue")
                except Exception as e:
                    logging.error(f"❌ Test 3 FAILED: Queue check error: {e}")
                
                logging.info("\n" + "="*80)
                logging.info("ADAPTIVE PIPELINE TEST COMPLETE")
                logging.info("="*80 + "\n")

                # Verify trainer is real
                trainer_type = type(orchestrator.trainer).__name__
                print(f"Trainer type: {trainer_type}")

                if trainer_type == 'AdaptiveTrainer':
                    print("✗ CRITICAL ERROR: Using fallback trainer!")
                    print("   Real EnhancedConversationTrainer failed to load")
                    print("   Training will NOT work!")
                    sys.exit(1)

                # Verify train method exists and is callable
                if not hasattr(orchestrator.trainer, 'train'):
                    print("✗ CRITICAL ERROR: Trainer has no train method!")
                    sys.exit(1)

                if not callable(orchestrator.trainer.train):
                    print("✗ CRITICAL ERROR: Trainer.train is not callable!")
                    sys.exit(1)

                print("✓ Trainer verification passed")
                print(f"✓ Trainer class: {orchestrator.trainer.__class__.__module__}.{trainer_type}")
                print("="*80 + "\n")

                # Integrate Chinchilla scaler if enabled
                if getattr(config, 'auto_epoch_scaling', False) and CHINCHILLA_SCALER_AVAILABLE:
                    print_banner("INTEGRATING ENHANCED CHINCHILLA SCALER")
                    scaler = EnhancedChinchillaScaler(config, model, train_dataset)
                    config.num_epochs = scaler.get_optimal_epochs()
                    orchestrator.trainer.chinchilla_scaler = scaler
                    print(f"✓ Chinchilla scaler attached to trainer")
                    print(f"✓ Optimal epochs: {config.num_epochs}")
                    print("="*80 + "\n")
                if deepspeed_integration_needed and DEEPSPEED_REMAKE_AVAILABLE:
                    print_banner("INTEGRATING DEEPSPEED REMAKE WITH TRAINER")
                    
                    try:
                        deepspeed_integration = integrate_with_trainer(
                            trainer=orchestrator.trainer,  # ✅ The actual trainer object
                            config=config,
                            model=model,
                            expert_registry=expert_registry if config.use_moe else None
                        )
                        
                        orchestrator.deepspeed_integration = deepspeed_integration
                        
                        print("✓ DeepSpeed Remake integration complete")
                        print(f"  Optimizer type: {type(orchestrator.trainer.optimizer).__name__}")
                        print(f"  Scheduler type: {type(orchestrator.trainer.scheduler).__name__ if orchestrator.trainer.scheduler else 'None'}")
                        print(f"  ZeRO Stage: {config.zero_stage}")
                        print(f"  Offloading: {'Enabled' if deepspeed_integration.offload_manager else 'Disabled'}")
                        print("="*80 + "\n")
                        
                        using_backend = "DeepSpeed Remake (Integrated)"
                        
                    except Exception as e:
                        print(f"✗ DeepSpeed Remake integration failed: {e}")
                        traceback.print_exc()
                        print("Continuing with standard optimizer...")
            except Exception as e:
                print(f"\nERROR: Failed to initialize orchestrator: {e}")                
                traceback.print_exc()
                return 1
            
        else:
            print("ERROR: Advanced training infrastructure required but not available")
            print("Please ensure the following modules are available:")
            print("  - training.orchestrator")
            print("  - training.trainer")
            print("  - training.checkpoint")
            return 1
        
        # After Chinchilla integration in Step 10:
        if getattr(config, 'auto_epoch_scaling', False):
            print("\n" + "="*80)
            print("CHINCHILLA SYSTEM VERIFICATION")
            print("="*80)
            print(f"Original epochs in config: {training_params['num_epochs']}")
            print(f"Final epochs after Chinchilla: {config.num_epochs}")
            print(f"Chinchilla scaler attached: {hasattr(orchestrator.trainer, 'chinchilla_scaler')}")
            if hasattr(orchestrator.trainer, 'chinchilla_scaler'):
                scaler = orchestrator.trainer.chinchilla_scaler
                print(f"Scaler type: {type(scaler).__name__}")
                print(f"Optimal epochs calculated: {scaler.get_optimal_epochs()}")
            print("="*80)

        # Step 10.5: Checkpoint Resumption
        print_banner("STEP 10.5: CHECKPOINT MANAGEMENT")

        if checkpoint_params.get('resume_training', False) and checkpoint_params.get('resume_from_checkpoint'):
            checkpoint_path = checkpoint_params['resume_from_checkpoint']
            print(f"📄 Loading checkpoint: {checkpoint_path}")

            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                print("✓ Checkpoint loaded successfully!")
                print(f"Checkpoint info:")
                print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"   - Global step: {checkpoint.get('global_step', 'unknown')}")
                print(f"   - Loss: {checkpoint.get('loss', 'unknown')}")

                # Load model state
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Model weights loaded")

                # Load optimizer state if available and not reset
                if not checkpoint_params.get('reset_optimizer', False) and 'optimizer_state_dict' in checkpoint:
                    if hasattr(orchestrator.trainer, 'optimizer'):
                        orchestrator.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("✓ Optimizer state loaded")

                # Load scheduler state if available and not reset
                if not checkpoint_params.get('reset_scheduler', False) and 'scheduler_state_dict' in checkpoint:
                    if hasattr(orchestrator.trainer, 'scheduler') and orchestrator.trainer.scheduler:
                        orchestrator.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("✓ Scheduler state loaded")

                # Set training state
                start_epoch = checkpoint.get('epoch', 0) + 1
                global_step = checkpoint.get('global_step', 0)
                best_loss = checkpoint.get('best_loss', float('inf'))

                # Update orchestrator directly
                orchestrator.start_epoch = start_epoch
                orchestrator.global_step = global_step
                orchestrator.best_loss = best_loss

                print(f"\nTraining will continue from:")
                print(f"   - Epoch: {start_epoch}")
                print(f"   - Global step: {global_step}")
                print(f"   - Best loss: {best_loss:.4f}")

            except Exception as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("Starting training from scratch...")
                checkpoint_params['resume_training'] = False
        else:
            print("Starting fresh training session")
            print("✓ No checkpoint to resume from")

        # Verify scheduler
        print("\n" + "="*80)
        print("SCHEDULER VERIFICATION")
        print("="*80)

        if hasattr(orchestrator.trainer, 'scheduler'):
            if orchestrator.trainer.scheduler is not None:
                scheduler_type = type(orchestrator.trainer.scheduler).__name__
                print(f"✓ Scheduler found: {scheduler_type}")

                try:
                    initial_lr = orchestrator.trainer.scheduler.get_last_lr()[0]
                    base_lrs = orchestrator.trainer.scheduler.base_lrs
                    print(f"✓ Initial LR: {initial_lr:.2e}")
                    print(f"✓ Base LRs: {[f'{lr:.2e}' for lr in base_lrs]}")
                    print(f"✓ Config LR: {orchestrator.config.learning_rate:.2e}")

                    # Verify they match
                    if abs(initial_lr - orchestrator.config.learning_rate) > 1e-9:
                        print(f"WARNING: Scheduler LR doesn't match config LR!")
                    else:
                        print(f"✓ Scheduler LR matches config")

                except Exception as e:
                    print(f"Could not read scheduler state: {e}")
            else:
                print("Scheduler is None")
                print(f"   use_lr_scheduler: {getattr(orchestrator.config, 'use_lr_scheduler', 'not set')}")
                print(f"   LR will remain constant at: {orchestrator.config.learning_rate:.2e}")
        else:
            print("✗ Trainer has no scheduler attribute!")

        print("="*80 + "\n")

        # 🔥 FIX: Setup scheduler NOW with dataset info
        print("\n" + "="*80)
        print("STEP 10.8: SETTING UP LEARNING RATE SCHEDULER".center(80))
        print("="*80)
        
        if orchestrator.trainer and not orchestrator.use_deepspeed:
            # Calculate total steps
            gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
            batches_per_epoch = len(train_dataset) // config.batch_size
            steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
            total_steps = steps_per_epoch * config.num_epochs
            
            print(f"Scheduler Configuration:")
            print(f"  Batches per epoch: {batches_per_epoch}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total epochs: {config.num_epochs}")
            print(f"  Total steps: {total_steps}")
            print(f"  Scheduler type: {config.lr_scheduler}")
            
            # Setup the scheduler
            orchestrator.trainer._setup_scheduler(total_steps)
            
            # Verify it worked
            if orchestrator.trainer.scheduler is not None:
                print(f"✅ Scheduler created: {type(orchestrator.trainer.scheduler).__name__}")
                try:
                    initial_lr = orchestrator.trainer.scheduler.get_last_lr()[0]
                    print(f"✅ Initial LR: {initial_lr:.2e}")
                except:
                    print(f"✅ Scheduler ready (LR: {config.learning_rate:.2e})")
            else:
                print(f"❌ WARNING: Scheduler is still None!")
                print(f"   Adaptive training will work but scheduler won't!")
        else:
            print("Skipping scheduler setup (DeepSpeed handles it)")
        
        print("="*80 + "\n")
        
        # Step 11: Setup signal handlers
        print_banner("STEP 11: SETTING UP SIGNAL HANDLERS")
        setup_signal_handlers(orchestrator)
        print("✓ Signal handlers configured for graceful shutdown")
        print("  SIGINT (Ctrl+C): Save state and exit")
        print("  SIGTERM: Save state and exit")
        
        # Step 12: Create experiment directory and save metadata
        print_banner("STEP 12: EXPERIMENT SETUP")
        experiment_dir = validate_and_setup_experiment(config)
        
        # Save enhanced parameters summary
        print_section("Saving Enhanced Parameters")
        enhanced_params_path = experiment_dir / "enhanced_parameters.json"
        enhanced_summary = {
            'adaptive_intelligence': adaptive_intelligence_params,
            'dynamic_architecture': dynamic_architecture_params,
            'predictive_optimization': predictive_optimization_params,
            'quality_aware_training': quality_aware_training_params,
            'hardware_optimization': hardware_aware_optimization_params,
            'data_intelligence': data_intelligence_params,
            'safety_robustness': safety_robustness_params,
            'multi_objective': multi_objective_optimization_params,
        }
        with open(enhanced_params_path, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        print(f"✓ Enhanced parameters saved: {enhanced_params_path}")
        
        save_experiment_metadata(experiment_dir, config, model, datasets_info)
        
        # Step 13: Display enhanced training configuration summary
        print_banner("STEP 13: ENHANCED TRAINING CONFIGURATION SUMMARY")
        
        print_section("Model Configuration")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Attention Heads: {config.num_heads}")
        print(f"  Sequence Length: {config.seq_length}")
        print(f"  MoE: {'Enabled' if config.use_moe else 'Disabled'}")
        
        print_section("Enhanced Training Configuration")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Min Learning Rate: {config.min_lr}")
        print(f"  LR Scheduler: {config.lr_scheduler}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  Precision: {config.precision}")
        print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
        
        print_section("Adaptive Intelligence Features")
        print(f"  Risk Tolerance: {config.adaptive_risk_tolerance}")
        print(f"  Meta-Learning Transfer Weight: {config.learning_transfer_weight}")
        print(f"  Strategy Memory Size: {config.strategy_memory_size}")
        print(f"  Exploration Rate: {config.exploration_rate}")
        print(f"  Confidence Thresholds:")
        print(f"    - Soft: {config.meta_confidence_soft}")
        print(f"    - Medium: {config.meta_confidence_medium}")
        print(f"    - Hard: {config.meta_confidence_hard}")
        print(f"    - Critical: {config.meta_confidence_critical}")
        
        print_section("Dynamic Architecture Features")
        if config.use_moe:
            print(f"  Dynamic Expert Management: {config.dynamic_expert_management}")
            print(f"  Expert Growth Threshold: {config.expert_growth_threshold}")
            print(f"  Expert Prune Threshold: {config.expert_prune_threshold}")
            print(f"  Max Experts Per Layer: {config.max_experts_per_layer}")
            print(f"  Min Experts Per Layer: {config.min_experts_per_layer}")
        print(f"  MoD Capacity Adaptation: {config.mod_capacity_adaptation}")
        print(f"  Per-Layer Routing Config: {config.per_layer_routing_config}")
        
        print_section("Predictive Optimization Features")
        print(f"  Convergence Prediction Horizon: {config.convergence_prediction_horizon} steps")
        print(f"  Plateau Detection Window: {config.plateau_detection_window} steps")
        print(f"  Divergence Early Warning: {config.divergence_early_warning} steps")
        print(f"  Memory Trend Analysis: {config.memory_trend_analysis}")
        print(f"  OOM Prediction Confidence: {config.oom_prediction_confidence}")
        print(f"  Importance-Based Checkpointing: {config.importance_based_checkpointing}")
        
        print_section("Quality & Safety Features")
        print(f"  Loss Smoothness Threshold: {config.loss_smoothness_threshold}")
        print(f"  Gradient Health Monitoring: {config.gradient_health_monitoring}")
        print(f"  Curvature-Aware Training: {config.curvature_aware_training}")
        print(f"  Max Perplexity Spike: {config.max_perplexity_spike}")
        print(f"  Min Quality Improvement: {config.min_quality_improvement}")
        print(f"  Catastrophic Forgetting Threshold: {config.catastrophic_forgetting_threshold}")
        print(f"  Toxicity Monitoring: {config.toxicity_monitoring}")
        print(f"  Max Weight Norm: {config.max_weight_norm}")
        print(f"  Max Gradient Norm: {config.max_gradient_norm}")
        
        print_section("Hardware Optimization Features")
        print(f"  Hardware Optimization Level: {config.hardware_optimization_level}")
        print(f"  Tensor Core Optimization: {config.tensor_core_optimization}")
        print(f"  Memory Bus Utilization Target: {config.memory_bus_utilization_target}")
        print(f"  Communication Overlap Aggressiveness: {config.communication_overlap_aggressiveness}")
        print(f"  Gradient Sync Strategy: {config.gradient_sync_strategy}")
        print(f"  Thermal Throttling Avoidance: {config.thermal_throttling_avoidance}")
        
        print_section("Data Intelligence Features")
        print(f"  Difficulty-Based Sampling: {config.difficulty_based_sampling}")
        print(f"  Curriculum Learning Aggressiveness: {config.curriculum_learning_aggressiveness}")
        print(f"  Hard Example Mining Threshold: {config.hard_example_mining_threshold}")
        print(f"  Automatic Data Cleaning: {config.automatic_data_cleaning}")
        print(f"  Data Quality Threshold: {config.data_quality_threshold}")
        print(f"  Diversity Penalty: {config.diversity_penalty}")
        print(f"  Sequence Length Optimization: {config.sequence_length_optimization}")
        print(f"  Similarity-Aware Batching: {config.similarity_aware_batching}")
        
        print_section("Multi-Objective Optimization")
        print(f"  Primary Objective: {config.primary_objective}")
        print(f"  Speed-Quality Tradeoff: {config.speed_quality_tradeoff}")
        print(f"  Memory-Performance Balance: {config.memory_performance_balance}")
        print(f"  Secondary Objective Weights:")
        print(f"    - Speed: {config.secondary_objective_speed}")
        print(f"    - Memory: {config.secondary_objective_memory}")
        print(f"    - Quality: {config.secondary_objective_quality}")
        
        print_section("Optimization Configuration")
        if quantization_params.get('quantization_method'):
            print(f"  Quantization: {quantization_params['quantization_method']} ({quantization_params['quantization_bits']}-bit)")
        else:
            print(f"  Quantization: Disabled")
        print(f"  Flash Attention: {config.use_flash_attention}")
        print(f"  Model Compilation: {config.compile}")
        
        print_section("Data Configuration")
        print(f"  Training Samples: {len(train_dataset):,}")
        print(f"  Evaluation Samples: {len(eval_dataset):,}")
        print(f"  Steps per Epoch: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)}")
        print(f"  Total Training Steps: {(len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs}")
        
        # Step 14: Start enhanced training
        print_banner("STEP 14: STARTING ENHANCED ADAPTIVE TRAINING")
        
        # Verify adaptive monitoring pipeline
        print("\n" + "="*80)
        print("VERIFYING ADAPTIVE MONITORING PIPELINE")
        print("="*80)
        
        # Check monitoring thread
        monitoring_active = orchestrator.monitoring_thread and orchestrator.monitoring_thread.is_alive()
        print(f"Monitoring thread: {'✅ Active' if monitoring_active else '❌ Not running'}")
        
        # Check monitoring queue
        queue_exists = orchestrator.monitoring_queue is not None
        print(f"Monitoring queue: {'✅ Connected' if queue_exists else '❌ Missing'}")
        if queue_exists:
            print(f"  Queue size: {orchestrator.monitoring_queue.qsize()}/{orchestrator.monitoring_queue.maxsize}")
        
        # Check trainer enhancements
        has_queue = hasattr(orchestrator.trainer, '_monitoring_queue')
        has_metrics = hasattr(orchestrator.trainer, 'get_current_metrics')
        print(f"Trainer monitoring injection: {'✅ Complete' if has_queue and has_metrics else '❌ Incomplete'}")
        
        # Test the pipeline
        print("\n🧪 Testing metric collection pipeline...")
        try:
            test_metric = orchestrator.trainer.get_current_metrics()
            print(f"  ✅ get_current_metrics() → {type(test_metric).__name__}")
            
            orchestrator.monitoring_queue.put(test_metric, block=False)
            print(f"  ✅ Queue.put() → Success")
            
            retrieved = orchestrator.monitoring_queue.get(timeout=0.1)
            print(f"  ✅ Queue.get() → {type(retrieved).__name__}")
            print(f"\n✅ PIPELINE TEST PASSED - Adaptive monitoring is functional!")
            
        except Exception as e:
            print(f"  ❌ PIPELINE TEST FAILED: {e}")
            print(f"  ⚠️  Adaptive features will be limited!")
            traceback.print_exc()
        
        print("="*80 + "\n")
        
        # Original training start messages
        print(f"Training begins at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment directory: {experiment_dir}")
        print("")
        print("All enhanced features are active and monitoring training")
        print("Press Ctrl+C at any time to gracefully stop training and save progress")
        print("="*80)
        
        training_start_time = time.time()
        
        # Run adaptive training with OOM protection
        try:
            orchestrator = wrap_orchestrator_with_oom_protection(
                orchestrator, 
                train_dataset, 
                eval_dataset
            )
        except KeyboardInterrupt:
            print("\n" + "="*80)
            print("TRAINING INTERRUPTED BY USER")
            print("="*80)
            raise
        except Exception as e:
            print("\n" + "="*80)
            print("TRAINING ERROR")
            print("="*80)
            print(f"Error: {e}")
            traceback.print_exc()
            raise
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        total_training_hours = total_training_time / 3600

        # Step 14.5: Train Router Model (OPTIONAL)
        if router_training_params.get('enable_router_training', False):
            print_banner("STEP 14.5: TRAINING ROUTER MODEL")
            
            from router_training_system import train_router_model, replace_gates_with_router
            
            router_config = {
                'hidden_size': config.hidden_size,
                'num_experts': config.num_experts,
                'top_k': config.moe_top_k,
                **router_training_params
            }
            
            # Train the router
            trained_router = train_router_model(
                existing_model=model,
                train_dataloader=train_dataloader,
                config=router_config,
                output_dir="router_training"
            )
            
            # Replace gates with trained router
            model = replace_gates_with_router(
                model,
                router_model_path="router_training/best_router_model.pt"
            )
            
            print("✓ Router model trained and integrated!")
            print("  Model now uses learned routing instead of simple gates")
        
        # Step 15: Training completion summary
        print_banner("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        
        print_section("Training Summary")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Started: {datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Completed: {datetime.fromtimestamp(training_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Time: {total_training_hours:.2f} hours ({total_training_time/60:.1f} minutes)")
        
        # Get enhanced adaptive training status
        status = orchestrator.get_adaptive_status()
        
        print_section("Enhanced Adaptive Training Statistics")
        print(f"  Adaptive Decisions Made: {status.get('adaptive_decisions_made', 'N/A')}")
        print(f"  Metrics Collected: {status.get('metrics_collected', 'N/A')}")
        print(f"  Meta-Learning Runs: {status.get('meta_learning_runs', 'N/A')}")
        print(f"  Checkpoints Saved: {status.get('checkpoints_saved', 'N/A')}")
        print(f"  Expert Adjustments: {status.get('expert_adjustments', 0)}")
        print(f"  Quality Interventions: {status.get('quality_interventions', 0)}")
        print(f"  Memory Optimizations: {status.get('memory_optimizations', 0)}")
        print(f"  LR Adjustments: {status.get('lr_adjustments', 0)}")
        
        # Step 16: Generate final reports (ADVANCED)
        if advanced_features.get('generate_training_reports') and UTILS_AVAILABLE:
            print_banner("STEP 16: GENERATING TRAINING REPORTS")
            try:
                create_training_report(str(experiment_dir))
                print(f"✓ Training report generated: {experiment_dir}/training_report.html")
            except Exception as e:
                print(f"Could not generate training report: {e}")
        
        # Step 17: Save final summary
        print_section("Saving Final Summary")
        
        summary = {
            'experiment_name': config.experiment_name,
            'training_completed': datetime.now().isoformat(),
            'training_start_time': datetime.fromtimestamp(training_start_time).isoformat(),
            'training_end_time': datetime.fromtimestamp(training_end_time).isoformat(),
            'total_training_time_seconds': total_training_time,
            'total_training_time_hours': total_training_hours,
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': total_params - trainable_params,
            },
            'enhanced_features': {
                'adaptive_intelligence': True,
                'dynamic_architecture': True,
                'predictive_optimization': True,
                'quality_aware_training': True,
                'hardware_optimization': True,
                'data_intelligence': True,
                'safety_robustness': True,
                'multi_objective': True,
            },
            'configuration': {
                'model_preset': config_choice,
                'moe_enabled': config.use_moe,
                'num_experts': config.num_experts if config.use_moe else None,
                'quantization': quantization_params.get('quantization_method'),
                'precision': config.precision,
                'training_mode': 'enhanced_adaptive',
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'epochs': config.num_epochs,
                'risk_tolerance': config.adaptive_risk_tolerance,
                'hardware_optimization_level': config.hardware_optimization_level,
                'primary_objective': config.primary_objective,
            },
            'dataset_info': {
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
            },
            'adaptive_training_status': status,
            'advanced_features_used': {k: v for k, v in advanced_features.items() if v},
        }
        
        summary_path = experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Training summary saved: {summary_path}")
        
        # Final success message
        print_banner("✨ ALL OPERATIONS COMPLETED SUCCESSFULLY ✨")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Total execution time: {(time.time() - training_start_time)/3600:.2f} hours")
        print("")
        print("🚀 Enhanced DeepSeek MoE Training System V2.0")
        print("   With Advanced AI-Driven Optimization")
        print("")
        print("Enhanced features utilized:")
        print("  ✓ Adaptive Intelligence")
        print("  ✓ Dynamic Architecture")
        print("  ✓ Predictive Optimization")
        print("  ✓ Quality-Aware Training")
        print("  ✓ Hardware-Aware Optimization")
        print("  ✓ Data Intelligence")
        print("  ✓ Safety & Robustness")
        print("  ✓ Multi-Objective Optimization")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("TRAINING INTERRUPTED BY USER")
        print("="*80)
        print("Saving emergency state...")
        
        if 'orchestrator' in locals() and orchestrator:
            try:
                orchestrator._save_meta_learning_state()
                print("✓ Meta-learning state saved successfully")
            except Exception as e:
                print(f"Error saving state: {e}")
        
        print("Graceful shutdown complete")
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("TRAINING FAILED WITH ERROR")
        print("="*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\nFull Traceback:")
        traceback.print_exc()
        print("="*80)
        return 1
        
    finally:
        print("\n" + "="*80)
        print("CLEANUP AND RESOURCE RELEASE")
        print("="*80)
        
        # Cleanup orchestrator
        if 'orchestrator' in locals() and orchestrator:
            try:
                print("Cleaning up orchestrator...")
                orchestrator.cleanup()
                print("✓ Orchestrator cleanup complete")
            except Exception as e:
                print(f"Orchestrator cleanup error: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            print("Clearing GPU memory cache...")
            torch.cuda.empty_cache()
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  GPU Memory Allocated: {memory_allocated:.2f} GB")
                print(f"  GPU Memory Reserved: {memory_reserved:.2f} GB")
            except:
                pass
        elif is_mps:
            print("Clearing MPS memory cache...")
            torch.mps.empty_cache()
        
        # Run garbage collection
        print("Running garbage collection...")
        gc.collect()
        
        # Final system status
        if UTILS_AVAILABLE:
            try:
                final_info = get_system_info()
                if final_info.get('available_memory_gb'):
                    print(f"  Available System Memory: {final_info.get('available_memory_gb', 0):.2f} GB")
            except:
                pass
        
        print("✓ Cleanup complete")
        print("="*80)


if __name__ == "__main__":
    exit(main())