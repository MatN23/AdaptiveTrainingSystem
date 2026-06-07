# Licensed under the Custom License below.

import os
import shutil
import torch
import platform
from typing import List, Dict, Any, Optional


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information including MPS support."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'python_version': None,
        'system_memory_gb': None,
        'disk_space_gb': None,
        'platform': platform.system(),
        'platform_release': platform.release(),
        'processor': platform.processor(),
    }
    
    # Python version
    import sys
    info['python_version'] = sys.version
    
    # CUDA information
    if torch.cuda.is_available():
        try:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Detailed GPU info for first device
            props = torch.cuda.get_device_properties(0)
            compute_cap = torch.cuda.get_device_capability(0)
            info['gpu_compute_capability'] = f"{compute_cap[0]}.{compute_cap[1]}"
            info['gpu_multi_processors'] = props.multi_processor_count
            info['gpu_max_threads_per_block'] = props.max_threads_per_block
            info['gpu_max_threads_per_mp'] = props.max_threads_per_multi_processor
        except Exception as e:
            info['cuda_error'] = str(e)
    
    # MPS information
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device_type'] = 'Apple Silicon (MPS)'
        info['mps_backend'] = 'Metal Performance Shaders'
        info['unified_memory'] = True
        
        # Detect specific Apple chip if possible
        try:
            if platform.system() == 'Darwin':
                import subprocess
                chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                info['processor_detail'] = chip_info
                
                # Detect M1/M2/M3 series
                if 'Apple' in chip_info:
                    if 'M1' in chip_info:
                        info['apple_silicon_generation'] = 'M1'
                    elif 'M2' in chip_info:
                        info['apple_silicon_generation'] = 'M2'
                    elif 'M3' in chip_info:
                        info['apple_silicon_generation'] = 'M3'
                    
                    # Detect Pro/Max/Ultra variants
                    if 'Pro' in chip_info:
                        info['apple_silicon_variant'] = 'Pro'
                    elif 'Max' in chip_info:
                        info['apple_silicon_variant'] = 'Max'
                    elif 'Ultra' in chip_info:
                        info['apple_silicon_variant'] = 'Ultra'
        except:
            pass
        
        pytorch_version = info['pytorch_version']
        try:
            major, minor = pytorch_version.split('.')[:2]
            version_num = float(f"{major}.{minor}")
            if version_num >= 2.0:
                info['mps_support_level'] = 'full'
            elif version_num >= 1.12:
                info['mps_support_level'] = 'beta'
            else:
                info['mps_support_level'] = 'none'
        except:
            info['mps_support_level'] = 'unknown'
        
        info['mps_capabilities'] = {
            'fp32': True,
            'fp16': True,
            'bf16': False,  # Limited support
            'flash_attention': False,
            'deepspeed': False,
            'gradient_checkpointing': True,
            'mixed_precision': True,
        }
    
    # System memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['system_memory_gb'] = memory.total / 1e9
        info['available_memory_gb'] = memory.available / 1e9
        info['memory_percent_used'] = memory.percent
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu_freq_current_mhz'] = cpu_freq.current
                info['cpu_freq_min_mhz'] = cpu_freq.min
                info['cpu_freq_max_mhz'] = cpu_freq.max
        except:
            pass
        
        # CPU usage
        try:
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        except:
            pass
    except ImportError:
        info['psutil_available'] = False
    
    # Disk space
    try:
        disk_usage = shutil.disk_usage(".")
        info['disk_space_gb'] = disk_usage.free / 1e9
        info['disk_total_gb'] = disk_usage.total / 1e9
        info['disk_used_gb'] = disk_usage.used / 1e9
        info['disk_percent_used'] = (disk_usage.used / disk_usage.total) * 100
    except Exception:
        pass
    
    return info


def validate_environment() -> List[str]:
    """Validate the training environment with MPS support."""
    issues = []
    
    torch_version = torch.__version__
    try:
        major, minor = torch_version.split('.')[:2]
        version_num = float(f"{major}.{minor}")
        
        if version_num < 1.12:
            issues.append(f"PyTorch version {torch_version} is old, recommend >= 1.12 for MPS or >= 2.0 for full support")
        elif version_num < 2.0:
            issues.append(f"PyTorch version {torch_version} has beta MPS support, recommend >= 2.0 for stable MPS")
    except:
        issues.append(f"Could not parse PyTorch version: {torch_version}")
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if not has_cuda and not has_mps:
        issues.append("No GPU acceleration available (neither CUDA nor MPS), training will be slow on CPU")
    elif has_mps:
        # MPS-specific checks
        issues.append("INFO: Using MPS (Apple Silicon) - some features have limited support compared to CUDA")
        issues.append("INFO: DeepSpeed and Flash Attention are not available on MPS")
        issues.append("INFO: Use FP16 or FP32 precision (BF16 has limited support on MPS)")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / 1e9
            
            if memory_gb < 8:
                issues.append(f"System memory {memory_gb:.1f}GB is insufficient for MPS training (8GB+ required)")
            elif memory_gb < 16:
                issues.append(f"System memory {memory_gb:.1f}GB may be limiting for MPS training (16GB+ recommended)")
        except ImportError:
            issues.append("psutil not available, cannot check system memory for MPS")
    elif has_cuda:
        # CUDA-specific checks
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:
                issues.append(f"GPU memory {gpu_memory:.1f}GB may be insufficient (8GB+ recommended)")
            elif gpu_memory < 16:
                issues.append(f"GPU memory {gpu_memory:.1f}GB is moderate (16GB+ recommended for larger models)")
        except:
            issues.append("Could not check GPU memory")
    
    try:
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / 1e9
        if free_gb < 10:
            issues.append(f"Low disk space: {free_gb:.1f}GB free (10GB+ recommended)")
        elif free_gb < 50:
            issues.append(f"Moderate disk space: {free_gb:.1f}GB free (50GB+ recommended for checkpoints)")
    except:
        issues.append("Could not check disk space")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1e9
        
        if available_gb < 4:
            issues.append(f"Low available system memory: {available_gb:.1f}GB (4GB+ recommended)")
        elif available_gb < 8:
            issues.append(f"Moderate available system memory: {available_gb:.1f}GB (8GB+ recommended)")
    except ImportError:
        issues.append("psutil not available, cannot check system memory")
    
    try:
        import numpy
    except ImportError:
        issues.append("NumPy not available")
    
    try:
        import tiktoken
    except ImportError:
        issues.append("tiktoken not available for tokenization")
    
    # Platform-specific checks
    if platform.system() == 'Darwin' and not has_mps:
        try:
            major, minor = torch_version.split('.')[:2]
            version_num = float(f"{major}.{minor}")
            if version_num >= 1.12:
                issues.append("Running on macOS but MPS not available - check PyTorch installation")
        except:
            pass
    
    return issues


def estimate_training_time(config, dataset_size: int) -> Dict[str, float]:
    """Estimate training time and resource requirements with MPS support."""
    from core.model import estimate_parameters
    
    # Model parameter estimation
    params = estimate_parameters(config)
    
    # Rough estimates based on empirical data
    tokens_per_sample = config.seq_length
    total_tokens = dataset_size * tokens_per_sample * config.num_epochs
    
    # Device detection
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # GPU/MPS estimates
    if has_cuda:
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / 1e9
        device_type = 'cuda'
        
        # Memory requirements (rough estimate)
        model_memory = params * 4 / 1e9  # 4 bytes per parameter (fp32)
        optimizer_memory = params * 8 / 1e9  # Adam needs ~8 bytes per parameter
        activation_memory = config.batch_size * config.seq_length * config.hidden_size * 4 / 1e9
        total_memory_needed = model_memory + optimizer_memory + activation_memory
        
        # Throughput estimates (very rough)
        if "A100" in gpu_props.name:
            tokens_per_sec = 50000
        elif "H100" in gpu_props.name:
            tokens_per_sec = 80000
        elif "V100" in gpu_props.name:
            tokens_per_sec = 25000
        elif "T4" in gpu_props.name:
            tokens_per_sec = 10000
        elif "RTX 4090" in gpu_props.name:
            tokens_per_sec = 35000
        elif "RTX 3090" in gpu_props.name:
            tokens_per_sec = 25000
        else:
            tokens_per_sec = 5000  # Conservative estimate
        
        # Adjust for precision
        precision = getattr(config, 'precision', 'fp32')
        if precision in ["fp16", "bf16", "mixed_fp16", "mixed_bf16"]:
            tokens_per_sec *= 1.5
            total_memory_needed *= 0.6
        
        memory_utilization = min(total_memory_needed / gpu_memory_gb, 1.0)
        
    elif has_mps:
        device_type = 'mps'
        
        # Get system memory (unified architecture)
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_memory_gb = memory.total / 1e9
        except:
            system_memory_gb = 16.0  # Assume reasonable default
        
        # Memory requirements for MPS
        model_memory = params * 4 / 1e9  # 4 bytes per parameter (fp32)
        optimizer_memory = params * 8 / 1e9  # Adam needs ~8 bytes per parameter
        activation_memory = config.batch_size * config.seq_length * config.hidden_size * 4 / 1e9
        total_memory_needed = model_memory + optimizer_memory + activation_memory
        
        # MPS throughput estimates (conservative, varies by chip)
        # M1: ~3000-5000 tokens/sec
        # M1 Pro/Max: ~5000-8000 tokens/sec
        # M2: ~4000-6000 tokens/sec
        # M2 Pro/Max: ~6000-10000 tokens/sec
        # M3: ~5000-7000 tokens/sec
        # M3 Pro/Max: ~7000-12000 tokens/sec
        
        try:
            system_info = get_system_info()
            silicon_gen = system_info.get('apple_silicon_generation', 'M1')
            silicon_var = system_info.get('apple_silicon_variant', '')
            
            if silicon_gen == 'M3':
                if silicon_var == 'Max':
                    tokens_per_sec = 12000
                elif silicon_var == 'Pro':
                    tokens_per_sec = 9000
                else:
                    tokens_per_sec = 6000
            elif silicon_gen == 'M2':
                if silicon_var == 'Max':
                    tokens_per_sec = 10000
                elif silicon_var == 'Pro':
                    tokens_per_sec = 7500
                else:
                    tokens_per_sec = 5000
            else:  # M1 or unknown
                if silicon_var == 'Max':
                    tokens_per_sec = 8000
                elif silicon_var == 'Pro':
                    tokens_per_sec = 6000
                else:
                    tokens_per_sec = 4000
        except:
            tokens_per_sec = 5000  # Conservative default for MPS
        
        # Adjust for precision on MPS
        precision = getattr(config, 'precision', 'fp32')
        if precision in ["fp16", "mixed_fp16"]:
            tokens_per_sec *= 1.3  # Less speedup than CUDA
            total_memory_needed *= 0.6
        
        memory_utilization = min(total_memory_needed / system_memory_gb, 1.0)
        
    else:
        # CPU estimates (much slower)
        device_type = 'cpu'
        tokens_per_sec = 100
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_memory_gb = memory.total / 1e9
        except:
            system_memory_gb = 8.0  # Assume reasonable default
        
        model_memory = params * 4 / 1e9
        optimizer_memory = params * 8 / 1e9
        activation_memory = config.batch_size * config.seq_length * config.hidden_size * 4 / 1e9
        total_memory_needed = model_memory + optimizer_memory + activation_memory
        
        memory_utilization = min(total_memory_needed / system_memory_gb, 1.0)
    
    estimated_time_hours = total_tokens / tokens_per_sec / 3600
    
    return {
        'estimated_hours': estimated_time_hours,
        'estimated_days': estimated_time_hours / 24,
        'total_tokens': total_tokens,
        'tokens_per_second': tokens_per_sec,
        'memory_utilization': memory_utilization,
        'memory_warning': memory_utilization > 0.9,
        'device_type': device_type,
        'estimated_memory_needed_gb': total_memory_needed,
        'model_parameters': params,
    }


def get_optimal_device() -> torch.device:
    """Get the optimal device for training with priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get detailed information about a specific device."""
    if device is None:
        device = get_optimal_device()
    
    info = {
        'device_type': device.type,
        'device_str': str(device),
    }
    
    if device.type == 'cuda':
        info.update({
            'device_name': torch.cuda.get_device_name(device),
            'device_index': device.index or 0,
            'total_memory_gb': torch.cuda.get_device_properties(device).total_memory / 1e9,
            'compute_capability': torch.cuda.get_device_capability(device),
        })
    elif device.type == 'mps':
        system_info = get_system_info()
        info.update({
            'device_name': system_info.get('device_type', 'Apple Silicon'),
            'unified_memory': True,
            'system_memory_gb': system_info.get('system_memory_gb', 0),
            'mps_support_level': system_info.get('mps_support_level', 'unknown'),
        })
    else:  # CPU
        system_info = get_system_info()
        info.update({
            'device_name': 'CPU',
            'cpu_count': system_info.get('cpu_count', 0),
            'system_memory_gb': system_info.get('system_memory_gb', 0),
        })
    
    return info


def check_mps_compatibility() -> Dict[str, Any]:
    """Check MPS compatibility and return detailed status."""
    result = {
        'available': False,
        'pytorch_version': torch.__version__,
        'platform': platform.system(),
        'issues': [],
        'recommendations': [],
    }
    
    if platform.system() != 'Darwin':
        result['issues'].append("MPS is only available on macOS")
        return result
    
    try:
        major, minor = torch.__version__.split('.')[:2]
        version_num = float(f"{major}.{minor}")
        
        if version_num < 1.12:
            result['issues'].append(f"PyTorch {torch.__version__} does not support MPS (need >= 1.12)")
            result['recommendations'].append("Upgrade PyTorch: pip install --upgrade torch torchvision torchaudio")
            return result
        elif version_num < 2.0:
            result['available'] = True
            result['support_level'] = 'beta'
            result['recommendations'].append("Consider upgrading to PyTorch 2.0+ for stable MPS support")
        else:
            result['available'] = True
            result['support_level'] = 'full'
    except:
        result['issues'].append("Could not parse PyTorch version")
        return result
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        result['available'] = False
        result['issues'].append("MPS backend not available")
        result['recommendations'].append("Ensure you have PyTorch built with MPS support")
        return result
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        result['system_memory_gb'] = memory_gb
        
        if memory_gb < 8:
            result['issues'].append(f"Limited system memory ({memory_gb:.1f}GB)")
            result['recommendations'].append("8GB+ RAM recommended for MPS training")
        elif memory_gb < 16:
            result['recommendations'].append("16GB+ RAM recommended for optimal MPS training")
    except ImportError:
        result['recommendations'].append("Install psutil to check system memory: pip install psutil")
    
    # Capability checks
    result['capabilities'] = {
        'fp32_training': True,
        'fp16_training': True,
        'bf16_training': False,
        'mixed_precision': True,
        'gradient_checkpointing': True,
        'flash_attention': False,
        'deepspeed': False,
    }
    
    return result


def get_recommended_config_for_device(device_type: str = None) -> Dict[str, Any]:
    """Get recommended configuration for a specific device type."""
    if device_type is None:
        device_type = get_optimal_device().type
    
    if device_type == 'mps':
        return {
            'precision': 'fp16',
            'use_deepspeed': False,
            'use_flash_attention': False,
            'compile': False,
            'gradient_checkpointing': True,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
            'num_workers': 0,  # MPS works best with main process loading
            'pin_memory': False,
            'notes': [
                'MPS uses unified memory architecture',
                'FP16 provides good balance of speed and stability',
                'Start with small batch sizes and scale up',
                'Some operations may fall back to CPU',
            ]
        }
    elif device_type == 'cuda':
        return {
            'precision': 'auto',  # Will auto-detect based on GPU
            'use_deepspeed': True,
            'use_flash_attention': True,
            'compile': True,
            'gradient_checkpointing': True,
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'num_workers': 4,
            'pin_memory': True,
            'notes': [
                'CUDA supports all features',
                'Precision will be auto-detected based on GPU capability',
                'DeepSpeed and Flash Attention available',
            ]
        }
    else:  # CPU
        return {
            'precision': 'fp32',
            'use_deepspeed': False,
            'use_flash_attention': False,
            'compile': False,
            'gradient_checkpointing': False,
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'num_workers': 2,
            'pin_memory': False,
            'notes': [
                'CPU training is significantly slower',
                'Recommended for debugging only',
                'Consider cloud GPU for production training',
            ]
        }