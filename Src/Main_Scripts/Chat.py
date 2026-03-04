# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import sys
import json
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.model import DeepSeekTransformer, DeepSeekConfig
from core.tokenizer import ConversationTokenizer
from config.config_manager import Config

# Disable logging for cleaner chat interface
logging.basicConfig(level=logging.ERROR)


# ============================================================================
# HARDCODED CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

class HardcodedConfig:
    """All configuration in one place - edit these values as needed"""
    
    # CHECKPOINT SETTINGS
    # Set to None for auto-detection, or specify exact path
    CHECKPOINT_PATH = "checkpoints/final.pt"  # e.g., "checkpoints/best_model.pt" or "checkpoints/epoch_3/"
    
    # GENERATION SETTINGS
    MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    DEFAULT_REPETITION_PENALTY = 1.1
    
    # DISPLAY SETTINGS
    STREAM_OUTPUT = True  # Show tokens as they're generated
    SHOW_TIMING = True
    SHOW_TOKEN_COUNT = True
    
    # CONVERSATION SETTINGS
    MAX_HISTORY_MESSAGES = 20  # Keep last N messages in context
    CONTEXT_WINDOW = 2048  # Maximum context length
    DEFAULT_MODE = 'standard'  # 'standard', 'creative', 'precise', 'analytical'
    DEFAULT_SYSTEM_PROMPT = None  # Set to string for default system prompt
    
    # CONVERSATION MODES (temperature, top_p, top_k, repetition_penalty)
    MODES = {
        'standard': {
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1
        },
        'creative': {
            'temperature': 1.2,
            'top_p': 0.95,
            'top_k': 100,
            'repetition_penalty': 1.05
        },
        'precise': {
            'temperature': 0.3,
            'top_p': 0.7,
            'top_k': 20,
            'repetition_penalty': 1.2
        },
        'analytical': {
            'temperature': 0.5,
            'top_p': 0.8,
            'top_k': 30,
            'repetition_penalty': 1.15
        }
    }
    
    # STOP TOKENS
    STOP_TOKENS = [
        '<|im_end|>',
        '<|endoftext|>',
        '<|user|>',
        '\n\nUser:',
        '\n\nHuman:',
    ]
    
    # FILE PATHS
    CONVERSATIONS_DIR = "conversations"  # Where to save chat logs
    
    # DEVICE SETTINGS
    # Set to 'cuda', 'mps', 'cpu', or None for auto-detection
    FORCE_DEVICE = None


# ============================================================================
# SESSION TRACKING
# ============================================================================

@dataclass  
class SessionStats:
    """Track session statistics"""
    start_time: datetime = field(default_factory=datetime.now)
    messages_sent: int = 0
    messages_received: int = 0
    total_tokens_generated: int = 0
    total_generation_time: float = 0.0
    
    def tokens_per_second(self) -> float:
        if self.total_generation_time > 0:
            return self.total_tokens_generated / self.total_generation_time
        return 0.0
    
    def avg_response_time(self) -> float:
        if self.messages_received > 0:
            return self.total_generation_time / self.messages_received
        return 0.0


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_checkpoint_smart(checkpoint_path: str, device: torch.device) -> Tuple[Dict, Optional[Dict]]:
    """Load checkpoint with support for single files and ZeRO shards"""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"📂 Loading checkpoint: {checkpoint_path.name}")
    
    # Case 1: Directory with ZeRO shards
    if checkpoint_path.is_dir():
        print("   Detected ZeRO checkpoint directory")
        return load_zero_shards(checkpoint_path, device)
    
    # Case 2: Single checkpoint file
    elif checkpoint_path.is_file():
        print("   Loading single checkpoint file")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model state dict
        if isinstance(checkpoint, dict):
            for key in ['module', 'model_state_dict', 'state_dict', 'model']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    config_dict = checkpoint.get('config', None)
                    return state_dict, config_dict
            return checkpoint, None
        else:
            return checkpoint, None
    
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_zero_shards(checkpoint_dir: Path, device: torch.device) -> Tuple[Dict, Optional[Dict]]:
    """Load and merge ZeRO checkpoint shards"""
    
    # Find shard files
    shard_patterns = [
        "*model_states.pt",
        "*mp_rank_*.pt",
        "zero_pp_rank_*_mp_rank_*.pt",
        "pytorch_model*.pt"
    ]
    
    shard_files = []
    for pattern in shard_patterns:
        files = list(checkpoint_dir.glob(pattern))
        if files:
            shard_files.extend(files)
            break
    
    if not shard_files:
        raise FileNotFoundError(f"No checkpoint shards found in {checkpoint_dir}")
    
    print(f"   Found {len(shard_files)} shard(s)")
    
    # Load and merge
    merged_state_dict = {}
    config_dict = None
    
    for i, shard_file in enumerate(sorted(shard_files)):
        print(f"   Loading shard {i+1}/{len(shard_files)}: {shard_file.name}")
        
        shard_data = torch.load(shard_file, map_location=device, weights_only=False)
        
        # Extract state dict
        if isinstance(shard_data, dict):
            if 'module' in shard_data:
                shard_state = shard_data['module']
            elif 'model_state_dict' in shard_data:
                shard_state = shard_data['model_state_dict']
            elif 'state_dict' in shard_data:
                shard_state = shard_data['state_dict']
            else:
                shard_state = shard_data
            
            if config_dict is None and 'config' in shard_data:
                config_dict = shard_data['config']
        else:
            shard_state = shard_data
        
        # Merge parameters
        for key, tensor in shard_state.items():
            merged_state_dict[key] = tensor
    
    print(f"   ✓ Merged {len(merged_state_dict)} parameters")
    return merged_state_dict, config_dict


def infer_config_from_state_dict(state_dict: Dict) -> DeepSeekConfig:
    """Infer model configuration from state dict"""
    
    print("🔍 Inferring configuration from checkpoint...")
    
    config_params = {}
    
    # Find vocab size and hidden size
    for key, tensor in state_dict.items():
        if 'embed_tokens.weight' in key and tensor.dim() == 2:
            config_params['vocab_size'] = tensor.shape[0]
            config_params['hidden_size'] = tensor.shape[1]
            print(f"   vocab_size={tensor.shape[0]}, hidden_size={tensor.shape[1]}")
            break
    
    # Count layers
    max_layer_idx = -1
    for key in state_dict.keys():
        if 'layers.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        max_layer_idx = max(max_layer_idx, layer_idx)
                    except ValueError:
                        continue
    
    if max_layer_idx >= 0:
        config_params['num_layers'] = max_layer_idx + 1
        print(f"   num_layers={config_params['num_layers']}")
    
    # Infer attention heads
    hidden_size = config_params.get('hidden_size', 768)
    for key, tensor in state_dict.items():
        if 'self_attn.q_proj.weight' in key and tensor.dim() == 2:
            for heads in [8, 12, 16, 32, 64]:
                if hidden_size % heads == 0:
                    config_params['num_heads'] = heads
                    config_params['num_kv_heads'] = max(1, heads // 4)
                    print(f"   num_heads={heads}, num_kv_heads={config_params['num_kv_heads']}")
                    break
            break
    
    # Infer intermediate size
    for key, tensor in state_dict.items():
        if 'gate_up_proj.weight' in key and tensor.dim() == 2:
            config_params['intermediate_size'] = tensor.shape[0] // 2
            print(f"   intermediate_size={config_params['intermediate_size']}")
            break
    
    # Apply defaults matching DeepSeekConfig from model.py
    defaults = {
        'vocab_size': 50304,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'num_kv_heads': 4,
        'seq_length': 2048,
        'intermediate_size': None,
        'dropout': 0.0,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'use_flash_attention': True,
        'gradient_checkpointing': False,
        'use_moe': False,
        'use_mod': False,
        'use_stable_embedding': True,
        'tie_word_embeddings': True,
        'init_std': 0.02,
    }
    
    for key, default_value in defaults.items():
        if key not in config_params:
            config_params[key] = default_value
    
    if config_params['intermediate_size'] is None:
        config_params['intermediate_size'] = 4 * config_params['hidden_size']
    
    return DeepSeekConfig(**config_params)


def find_latest_checkpoint() -> Optional[Path]:
    """Find the most recent checkpoint automatically"""
    
    print("🔍 Searching for checkpoints...")
    
    search_dirs = [
        Path("checkpoints"),
        Path("experiments"),
        Path("models"),
        Path("outputs"),
        Path("."),
    ]
    
    candidates = []
    
    for directory in search_dirs:
        if not directory.exists():
            continue
        
        # Look for checkpoint directories (ZeRO shards)
        for subdir in directory.iterdir():
            if subdir.is_dir():
                if list(subdir.glob("*model_states.pt")):
                    candidates.append((subdir.stat().st_mtime, subdir))
        
        # Look for checkpoint files
        for pattern in ["*model_states.pt", "best_*.pt", "checkpoint_*.pt", "*.pt"]:
            for file in directory.glob(pattern):
                if file.is_file():
                    candidates.append((file.stat().st_mtime, file))
    
    if not candidates:
        return None
    
    # Return most recent
    candidates.sort(reverse=True, key=lambda x: x[0])
    found = candidates[0][1]
    print(f"   ✓ Found: {found}")
    return found


# ============================================================================
# GENERATION ENGINE
# ============================================================================

class GenerationEngine:
    """Handle token generation with streaming support"""
    
    def __init__(self, model: DeepSeekTransformer, tokenizer: ConversationTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stream: bool = True
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate tokens with optional streaming"""
        
        start_time = time.time()
        generated_tokens = []
        
        # Limit prompt to fit context window and preallocate generation buffer.
        max_prompt_len = max(1, HardcodedConfig.CONTEXT_WINDOW - max_new_tokens)
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[-max_prompt_len:]

        total_capacity = min(HardcodedConfig.CONTEXT_WINDOW, len(prompt_tokens) + max_new_tokens)
        input_buffer = torch.empty((1, total_capacity), dtype=torch.long, device=self.device)
        if prompt_tokens:
            input_buffer[0, :len(prompt_tokens)] = torch.tensor(
                prompt_tokens,
                dtype=torch.long,
                device=self.device
            )
        cur_len = len(prompt_tokens)
        
        with torch.no_grad():
            if cur_len == 0:
                return [], {
                    'tokens_generated': 0,
                    'generation_time': 0.0,
                    'tokens_per_second': 0.0
                }

            special_stop_ids = set()
            text_stop_markers = []
            for stop in HardcodedConfig.STOP_TOKENS:
                if stop in self.tokenizer.special_tokens:
                    special_stop_ids.add(self.tokenizer.special_tokens[stop])
                else:
                    text_stop_markers.append(stop)

            # Prime KV cache with the full prompt once.
            outputs = self.model(
                input_buffer[:, :cur_len],
                return_aux_loss=False,
                use_cache=True
            )
            if isinstance(outputs, tuple):
                logits = outputs[0]
                past_key_values = outputs[-1]
            else:
                logits = outputs
                past_key_values = None

            for step in range(max_new_tokens):
                # Get next token logits
                next_token_logits = logits[0, -1, :].float()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_tokens:
                    for token_id in set(generated_tokens[-50:]):
                        if token_id < len(next_token_logits):
                            if next_token_logits[token_id] < 0:
                                next_token_logits[token_id] *= repetition_penalty
                            else:
                                next_token_logits[token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / max(temperature, 0.01)
                
                # Apply top-k
                if top_k > 0:
                    top_k_actual = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, top_k_actual)
                    next_token_logits[next_token_logits < values[-1]] = -float('inf')
                
                # Apply top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove.sum() > 1:
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check stop conditions
                if next_token == 0:
                    break

                if next_token in special_stop_ids:
                    break

                if text_stop_markers:
                    try:
                        decoded = self.tokenizer.decode([next_token])
                        if any(stop in decoded for stop in text_stop_markers):
                            break
                    except:
                        pass
                
                generated_tokens.append(next_token)
                
                # Stream output
                if stream:
                    try:
                        token_text = self.tokenizer.decode([next_token])
                        print(token_text, end='', flush=True)
                    except:
                        pass
                
                if cur_len >= total_capacity:
                    break

                # Append token to preallocated buffer.
                input_buffer[0, cur_len] = next_token
                cur_len += 1

                # Decode one token at a time with KV cache.
                outputs = self.model(
                    input_buffer[:, cur_len - 1:cur_len],
                    return_aux_loss=False,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    past_key_values = outputs[-1]
                else:
                    logits = outputs
                    past_key_values = None
        
        generation_time = time.time() - start_time
        
        metrics = {
            'tokens_generated': len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0
        }
        
        return generated_tokens, metrics


# ============================================================================
# CHAT INTERFACE
# ============================================================================

class ChatInterface:
    """Main chat interface"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.stats = SessionStats()
        
        # Conversation state
        self.conversation_history = []
        self.system_prompt = HardcodedConfig.DEFAULT_SYSTEM_PROMPT
        self.current_mode = HardcodedConfig.DEFAULT_MODE
        
        # Initialize components
        print("\n" + "="*70)
        print(" "*20 + "CHAT INTERFACE")
        print("="*70 + "\n")
        
        self._load_model()
        self._initialize_tokenizer()
        self._initialize_generator()
        
        print("\n" + "="*70)
        print(" "*25 + "READY!")
        print("="*70 + "\n")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        
        if HardcodedConfig.FORCE_DEVICE:
            device = torch.device(HardcodedConfig.FORCE_DEVICE)
            print(f"🖥️  Using forced device: {HardcodedConfig.FORCE_DEVICE}")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print("⚠️  Using CPU (will be slow)")
        
        return device
    
    def _load_model(self):
        """Load model from checkpoint"""
        
        # Find checkpoint
        if HardcodedConfig.CHECKPOINT_PATH:
            checkpoint_path = Path(HardcodedConfig.CHECKPOINT_PATH)
        else:
            checkpoint_path = find_latest_checkpoint()
            if not checkpoint_path:
                raise FileNotFoundError("No checkpoint found! Set CHECKPOINT_PATH in HardcodedConfig")
        
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        state_dict, config_dict = load_checkpoint_smart(str(checkpoint_path), self.device)
        
        # Get configuration
        if config_dict:
            print("✓ Using configuration from checkpoint")
            if isinstance(config_dict, dict):
                self.model_config = DeepSeekConfig(**config_dict)
            else:
                self.model_config = config_dict
        else:
            self.model_config = infer_config_from_state_dict(state_dict)
        
        # Create model
        print("\n🏗️  Initializing model...")
        self.model = DeepSeekTransformer(self.model_config).to(self.device)
        
        # Load weights
        print("📥 Loading model weights...")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"   ⚠️  {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"   ⚠️  {len(unexpected_keys)} unexpected keys")
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✓ Model loaded successfully!")
        print(f"   Parameters: {total_params:,}")
        print(f"   Hidden size: {self.model_config.hidden_size}")
        print(f"   Layers: {self.model_config.num_layers}")
        print(f"   Vocab: {self.model_config.vocab_size:,}")
        
        self.model.eval()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer"""
        print("\n📝 Initializing tokenizer...")
        
        self.tokenizer = ConversationTokenizer(
            model_name="gpt-4",
            max_context_length=self.model_config.seq_length,
            enable_caching=True,
            thread_safe=False,
            validation_level="moderate"
        )
        print(f"   ✓ Vocab size: {self.tokenizer.vocab_size:,}")
    
    def _initialize_generator(self):
        """Initialize generation engine"""
        print("⚙️  Initializing generation engine...")
        
        self.generator = GenerationEngine(self.model, self.tokenizer, self.device)
        print("   ✓ Ready to generate")
    
    def _format_prompt(self, user_message: str) -> str:
        """Format conversation into prompt"""
        
        parts = []
        
        if self.system_prompt:
            parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
        
        history = self.conversation_history[-HardcodedConfig.MAX_HISTORY_MESSAGES:]
        
        for msg in history:
            role = msg['role']
            content = msg['content']
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)
    
    def _generate_response(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response to user message"""
        
        # Format prompt
        prompt = self._format_prompt(user_message)
        
        # Tokenize
        try:
            prompt_tokens = self.tokenizer.tokenizer.encode(prompt)
        except Exception as e:
            print(f"\n✗ Tokenization error: {e}")
            return "Sorry, I couldn't process your message.", {}
        
        # Get mode parameters
        mode_params = HardcodedConfig.MODES[self.current_mode]
        
        # Generate
        print("Assistant: ", end='', flush=True)
        
        generated_tokens, metrics = self.generator.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=HardcodedConfig.MAX_NEW_TOKENS,
            temperature=mode_params['temperature'],
            top_p=mode_params['top_p'],
            top_k=mode_params['top_k'],
            repetition_penalty=mode_params['repetition_penalty'],
            stream=HardcodedConfig.STREAM_OUTPUT
        )
        
        # Decode
        try:
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            response = response.strip()
        except Exception as e:
            print(f"\n✗ Decoding error: {e}")
            response = "[Decoding error]"
        
        # Update history
        self.conversation_history.append({'role': 'user', 'content': user_message})
        self.conversation_history.append({'role': 'assistant', 'content': response})
        
        # Update stats
        self.stats.messages_sent += 1
        self.stats.messages_received += 1
        self.stats.total_tokens_generated += metrics['tokens_generated']
        self.stats.total_generation_time += metrics['generation_time']
        
        return response, metrics
    
    def _print_header(self):
        """Print chat header"""
        print("\n" + "="*70)
        print(" "*15 + "DEEPSEEK TRANSFORMER CHAT")
        print("="*70)
        print(f"Model: {self.model_config.num_layers}L-{self.model_config.hidden_size}H")
        print(f"Device: {self.device}")
        print(f"Mode: {self.current_mode}")
        print(f"Checkpoint: {self.checkpoint_path.name}")
        print("="*70)
        print("\n💡 Commands: /help /quit /clear /stats /mode /save /system")
        print("="*70 + "\n")
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands"""
        
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ['/quit', '/exit', '/q']:
            return False
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/clear':
            self.conversation_history.clear()
            print("✓ Conversation history cleared\n")
        
        elif cmd == '/stats':
            self._show_stats()
        
        elif cmd == '/mode':
            self._set_mode(args)
        
        elif cmd == '/system':
            self._set_system_prompt(args)
        
        elif cmd == '/save':
            self._save_conversation(args)
        
        elif cmd == '/config':
            self._show_config()
        
        else:
            print(f"✗ Unknown command: {cmd}")
            print("Type /help for available commands\n")
        
        return True
    
    def _show_help(self):
        """Show help"""
        print("\n" + "="*70)
        print("AVAILABLE COMMANDS")
        print("="*70)
        print("/help              - Show this help message")
        print("/quit, /exit, /q   - Exit the chat")
        print("/clear             - Clear conversation history")
        print("/stats             - Show session statistics")
        print("/mode <mode>       - Change conversation mode")
        print("                     Modes: standard, creative, precise, analytical")
        print("/system <text>     - Set system prompt (or clear if empty)")
        print("/save [name]       - Save conversation to file")
        print("/config            - Show current configuration")
        print("="*70 + "\n")
    
    def _show_stats(self):
        """Show session statistics"""
        duration = datetime.now() - self.stats.start_time
        
        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Duration: {duration}")
        print(f"Messages exchanged: {self.stats.messages_sent}")
        print(f"Total tokens generated: {self.stats.total_tokens_generated:,}")
        print(f"Average tokens/sec: {self.stats.tokens_per_second():.1f}")
        print(f"Average response time: {self.stats.avg_response_time():.2f}s")
        print(f"Current mode: {self.current_mode}")
        print(f"History length: {len(self.conversation_history)} messages")
        print("="*70 + "\n")
    
    def _set_mode(self, mode: str):
        """Set conversation mode"""
        mode = mode.strip().lower()
        
        if mode in HardcodedConfig.MODES:
            self.current_mode = mode
            params = HardcodedConfig.MODES[mode]
            print(f"\n✓ Mode set to: {mode}")
            print(f"   Temperature: {params['temperature']}")
            print(f"   Top-p: {params['top_p']}")
            print(f"   Top-k: {params['top_k']}")
            print(f"   Repetition penalty: {params['repetition_penalty']}\n")
        else:
            print(f"\n✗ Invalid mode: {mode}")
            print(f"Available modes: {', '.join(HardcodedConfig.MODES.keys())}\n")
    
    def _set_system_prompt(self, prompt: str):
        """Set system prompt"""
        if prompt.strip():
            self.system_prompt = prompt.strip()
            print(f"\n✓ System prompt set: {prompt[:60]}...\n")
        else:
            self.system_prompt = None
            print("\n✓ System prompt cleared\n")
    
    def _save_conversation(self, name: str):
        """Save conversation to file"""
        if not self.conversation_history:
            print("\n✗ No conversation to save\n")
            return
        
        # Create save directory
        save_dir = Path(HardcodedConfig.CONVERSATIONS_DIR)
        save_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name.strip():
            filename = f"{name.strip()}_{timestamp}.json"
        else:
            filename = f"chat_{timestamp}.json"
        
        filepath = save_dir / filename
        
        # Save data
        data = {
            'timestamp': timestamp,
            'model': str(self.checkpoint_path),
            'mode': self.current_mode,
            'system_prompt': self.system_prompt,
            'conversation': self.conversation_history,
            'stats': {
                'messages': len(self.conversation_history),
                'tokens_generated': self.stats.total_tokens_generated
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Conversation saved: {filepath}\n")
        except Exception as e:
            print(f"\n✗ Failed to save: {e}\n")
    
    def _show_config(self):
        """Show current configuration"""
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        print(f"Model: {self.model_config.hidden_size}d, {self.model_config.num_layers} layers")
        print(f"Vocab size: {self.model_config.vocab_size:,}")
        print(f"Context window: {HardcodedConfig.CONTEXT_WINDOW}")
        print(f"Max new tokens: {HardcodedConfig.MAX_NEW_TOKENS}")
        print(f"Stream output: {HardcodedConfig.STREAM_OUTPUT}")
        print(f"Current mode: {self.current_mode}")
        
        mode_params = HardcodedConfig.MODES[self.current_mode]
        print(f"\nMode parameters:")
        for key, value in mode_params.items():
            print(f"  {key}: {value}")
        
        print("="*70 + "\n")
    
    def run(self):
        """Run the chat interface"""
        
        self._print_header()
        
        # Setup signal handler for graceful exit
        def signal_handler(signum, frame):
            print("\n\n" + "="*70)
            print("CHAT SESSION ENDED")
            print("="*70)
            self._show_stats()
            print("👋 Goodbye!\n")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                # Get user input
                try:
                    mode_indicator = self.current_mode[0].upper()
                    user_input = input(f"\n[{mode_indicator}] You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n" + "="*70)
                    print("CHAT SESSION ENDED")
                    print("="*70)
                    self._show_stats()
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # Generate response
                try:
                    response, metrics = self._generate_response(user_input)
                    
                    # If not streaming, print response
                    if not HardcodedConfig.STREAM_OUTPUT:
                        print(f"Assistant: {response}")
                    
                    # Print metrics if enabled
                    if HardcodedConfig.SHOW_TIMING or HardcodedConfig.SHOW_TOKEN_COUNT:
                        info_parts = []
                        
                        if HardcodedConfig.SHOW_TIMING:
                            info_parts.append(f"⏱ {metrics['generation_time']:.2f}s")
                        
                        if HardcodedConfig.SHOW_TOKEN_COUNT:
                            info_parts.append(f"🔢 {metrics['tokens_generated']} tokens")
                            info_parts.append(f"⚡ {metrics['tokens_per_second']:.1f} tok/s")
                        
                        if info_parts:
                            print(f"\n   [{' | '.join(info_parts)}]")
                    
                    # Add newline for readability
                    if HardcodedConfig.STREAM_OUTPUT:
                        print()
                
                except Exception as e:
                    print(f"\n✗ Generation error: {e}")
                    import traceback
                    traceback.print_exc()
        
        finally:
            print("\n👋 Goodbye!\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - fully hardcoded, no arguments needed"""
    
    print("\n" + "="*70)
    print(" "*10 + "DEEPSEEK TRANSFORMER CHAT INTERFACE")
    print(" "*20 + "Fully Hardcoded Edition")
    print("="*70)
    print("\n📋 Configuration can be modified in HardcodedConfig class")
    print("🚀 Just run this script - no arguments needed!\n")
    
    try:
        # Initialize and run chat
        chat = ChatInterface()
        chat.run()
        return 0
        
    except KeyboardInterrupt:
        print("\n\n💬 Chat interrupted by user\n")
        return 0
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\n💡 Tips:")
        print("   1. Train a model first")
        print("   2. Set CHECKPOINT_PATH in HardcodedConfig class")
        print("   3. Make sure checkpoint files exist\n")
        return 1
    
    except Exception as e:
        print(f"\n✗ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
