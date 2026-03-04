# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import tiktoken
import time
from typing import List, Tuple, Union

class TokenizationMode(Enum):
    """Tokenization modes for different use cases."""
    STANDARD = "standard"
    COMPACT = "compact"  # Removes extra whitespace
    PRESERVE_FORMATTING = "preserve_formatting"  # Keeps all formatting
    CHAT_OPTIMIZED = "chat_optimized"  # Optimized for chat conversations


@dataclass
class TokenizationStats:
    """Statistics from tokenization process."""
    total_tokens: int
    special_tokens: int
    content_tokens: int
    message_count: int
    avg_tokens_per_message: float
    encoding_time_ms: float
    warnings: List[str]


class ConversationTokenizer:
    """Production tokenizer with enhanced error handling and validation."""
    
    # Thread-safe tokenizer cache
    _tokenizer_cache = {}
    _cache_lock = threading.Lock()
    
    # Precompiled regex patterns for efficiency
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    _NEWLINE_PATTERN = re.compile(r'\n+')
    _ROLE_PATTERN = re.compile(r'^(user|prompter|assistant|system|human|ai|bot)$', re.IGNORECASE)
    
    def __init__(self, 
                 model_name: str = "gpt-4",  # Changed default from gpt2 to gpt-4
                 max_context_length: int = 8192,  # Updated for GPT-4 context length
                 enable_caching: bool = True,
                 thread_safe: bool = True,
                 validation_level: str = "strict"):
        """
        Initialize enhanced tokenizer.
        
        Args:
            model_name: Tokenizer model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
            max_context_length: Maximum context length for truncation
            enable_caching: Enable LRU caching for repeated tokenizations
            thread_safe: Enable thread-safe operations
            validation_level: 'strict', 'moderate', or 'permissive'
        """

        self.model_name = model_name
        self.max_context_length = max_context_length
        self.enable_caching = enable_caching
        self.thread_safe = thread_safe
        self.validation_level = validation_level
        
        # Initialize tokenizer with caching
        self.tokenizer = self._get_or_create_tokenizer(model_name)
        self.base_vocab_size = self.tokenizer.n_vocab
        
        # Enhanced special tokens with metadata
        self.special_tokens = {
            "<|im_start|>": self.base_vocab_size,
            "<|im_end|>": self.base_vocab_size + 1,
            "<|user|>": self.base_vocab_size + 2,
            "<|assistant|>": self.base_vocab_size + 3,
            "<|system|>": self.base_vocab_size + 4,
            "<|human|>": self.base_vocab_size + 5,  # Additional role support
            "<|ai|>": self.base_vocab_size + 6,
            "<|bot|>": self.base_vocab_size + 7,
            "<|thought|>": self.base_vocab_size + 8,  # For reasoning traces
            "<|tool|>": self.base_vocab_size + 9,    # For tool usage
            "<|error|>": self.base_vocab_size + 10,  # For error handling
            "<|truncated|>": self.base_vocab_size + 11,  # Truncation marker
            "<|endoftext|>": self.base_vocab_size + 12, 
        }
        
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)
        self._reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # CRITICAL FIX: Add pad_token_id property
        self.pad_token_id = 0  # Using 0 as pad token ID (common convention)
        
        # Pad vocab size to be efficient for modern hardware
        alignment = 128  # Better for modern GPUs
        if self.vocab_size % alignment != 0:
            self.vocab_size = ((self.vocab_size + alignment - 1) // alignment) * alignment
        
        # Role mapping with aliases
        self._role_mapping = {
            'user': self.special_tokens["<|user|>"],
            'prompter': self.special_tokens["<|user|>"],
            'human': self.special_tokens["<|human|>"],
            'assistant': self.special_tokens["<|assistant|>"],
            'ai': self.special_tokens["<|ai|>"],
            'bot': self.special_tokens["<|bot|>"],
            'system': self.special_tokens["<|system|>"],
            'thought': self.special_tokens["<|thought|>"],
            'tool': self.special_tokens["<|tool|>"],
        }
        
        # Statistics tracking
        self.stats = {
            'total_conversations_processed': 0,
            'total_tokens_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'encoding_errors': 0,
        }
        
        # Thread safety
        if self.thread_safe:
            self._lock = threading.RLock()
        
        logging.info(f"Enhanced tokenizer initialized:")
        logging.info(f"  Model: {model_name}")
        logging.info(f"  Vocab size: {self.vocab_size:,}")
        logging.info(f"  Max context: {max_context_length:,}")
        logging.info(f"  Special tokens: {len(self.special_tokens)}")
        logging.info(f"  Pad token ID: {self.pad_token_id}")
    
    @classmethod
    def _get_or_create_tokenizer(cls, model_name: str):
        """Thread-safe tokenizer creation with caching."""
        with cls._cache_lock:
            if model_name not in cls._tokenizer_cache:
                try:
                    # Try to get encoding for the specific model
                    if model_name in ["gpt-4", "gpt-4-turbo", "gpt-4-32k"]:
                        # GPT-4 models use cl100k_base encoding
                        cls._tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
                    elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
                        # GPT-3.5-turbo also uses cl100k_base
                        cls._tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
                    else:
                        # Try to get encoding by model name, fallback to cl100k_base for newer models
                        try:
                            cls._tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
                        except KeyError:
                            logging.warning(f"Model {model_name} not found, using cl100k_base encoding")
                            cls._tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
                    
                    logging.debug(f"Created new tokenizer for {model_name}")
                except Exception as e:
                    logging.error(f"Failed to load tokenizer {model_name}: {e}")
                    # Fallback to gpt2 if everything fails
                    if model_name != "gpt2":
                        logging.warning(f"Falling back to gpt2 tokenizer")
                        cls._tokenizer_cache[model_name] = tiktoken.get_encoding("gpt2")
                    else:
                        raise
            return cls._tokenizer_cache[model_name]
    
    def _validate_conversation(self, conversation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Enhanced conversation validation with detailed error reporting."""
        warnings = []
        
        if not isinstance(conversation, dict):
            return False, ["Conversation must be a dictionary"]
        
        messages = conversation.get('messages', [])
        if not messages:
            warnings.append("Empty conversation")
            return True, warnings
        
        if not isinstance(messages, list):
            return False, ["Messages must be a list"]
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                warnings.append(f"Message {i} is not a dictionary")
                continue
            
            role = message.get('role', '').strip().lower()
            content = message.get('content', '')
            
            # Role validation
            if not role:
                warnings.append(f"Message {i} missing role")
            elif not self._ROLE_PATTERN.match(role):
                if self.validation_level == "strict":
                    warnings.append(f"Message {i} has invalid role: '{role}'")
            
            # Content validation
            if not content and self.validation_level in ["strict", "moderate"]:
                warnings.append(f"Message {i} has empty content")
            
            if isinstance(content, str) and len(content.strip()) > 50000:
                warnings.append(f"Message {i} content is very long ({len(content)} chars)")
        
        return True, warnings
    
    def _preprocess_content(self, content: str, mode: TokenizationMode = TokenizationMode.STANDARD) -> str:
        """Enhanced content preprocessing with multiple modes."""
        if not isinstance(content, str):
            content = str(content)
        
        if mode == TokenizationMode.COMPACT:
            # Aggressive whitespace normalization
            content = self._WHITESPACE_PATTERN.sub(' ', content)
            content = self._NEWLINE_PATTERN.sub('\n', content)
            content = content.strip()
        elif mode == TokenizationMode.PRESERVE_FORMATTING:
            # Minimal processing
            pass
        elif mode == TokenizationMode.CHAT_OPTIMIZED:
            # Optimized for chat: normalize whitespace but preserve structure
            content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces/tabs
            content = re.sub(r'\n{3,}', '\n\n', content)  # Limit consecutive newlines
            content = content.strip()
        else:  # STANDARD
            content = content.strip()
        
        return content
    
    @lru_cache(maxsize=1024 if True else 0)  # Controlled by enable_caching
    def _cached_encode(self, content: str) -> Tuple[List[int], bool]:
        """Cached content encoding with error handling."""
        try:
            tokens = self.tokenizer.encode(
                content,
                allowed_special={'<|endoftext|>'}
            )

            return tokens, True
        except Exception as e:
            logging.warning(f"Encoding error for content (len={len(content)}): {e}")
            # Attempt recovery with cleaned content
            try:
                cleaned = ''.join(c for c in content if ord(c) < 65536)  # Remove high unicode
                tokens = self.tokenizer.encode(cleaned)
                return tokens, False  # Indicate content was modified
            except:
                return [], False
    
    def encode_conversation(self, 
                          conversation: Dict[str, Any],
                          mode: TokenizationMode = TokenizationMode.STANDARD,
                          max_length: Optional[int] = None,
                          truncation_strategy: str = "end",
                          return_stats: bool = False) -> Union[List[int], Tuple[List[int], TokenizationStats]]:
        """
        Enhanced conversation encoding with comprehensive options.
        
        Args:
            conversation: Conversation dictionary with messages
            mode: Tokenization mode
            max_length: Maximum token length (overrides instance setting)
            truncation_strategy: 'start', 'end', or 'middle'
            return_stats: Whether to return detailed statistics
        """
        import time
        start_time = time.perf_counter()

        # Keep expensive tokenization work outside global lock.
        return self._encode_conversation_impl(
            conversation, mode, max_length, truncation_strategy, return_stats, start_time
        )
    
    def _apply_stats_delta(self, deltas: Dict[str, int]) -> None:
        if not deltas:
            return
        if self.thread_safe:
            with self._lock:
                for key, value in deltas.items():
                    self.stats[key] = self.stats.get(key, 0) + value
        else:
            for key, value in deltas.items():
                self.stats[key] = self.stats.get(key, 0) + value
    
    def _encode_message_tokens(self, message: Dict[str, Any], mode: TokenizationMode) -> List[int]:
        """Encode a single message into special-token-wrapped token IDs."""
        if not isinstance(message, dict):
            return []

        role = str(message.get('role', '')).strip().lower()
        content = message.get('content', '')

        if not content and self.validation_level == "strict":
            return []

        if role not in self._role_mapping:
            role = 'user'

        output_tokens = [
            self.special_tokens["<|im_start|>"],
            self._role_mapping[role],
        ]

        if content:
            processed_content = self._preprocess_content(str(content), mode)
            if processed_content:
                if self.enable_caching:
                    content_tokens, _ = self._cached_encode(processed_content)
                else:
                    content_tokens = self.tokenizer.encode(
                        processed_content,
                        allowed_special={'<|endoftext|>'}
                    )
                output_tokens.extend(content_tokens)

        output_tokens.append(self.special_tokens["<|im_end|>"])
        return output_tokens
    
    def _encode_conversation_impl(self, conversation, mode, max_length, truncation_strategy, return_stats, start_time):
        """Internal implementation of conversation encoding."""
        # Validation
        is_valid, warnings = self._validate_conversation(conversation)
        if not is_valid and self.validation_level == "strict":
            self._apply_stats_delta({'validation_errors': 1})
            if return_stats:
                stats = TokenizationStats(0, 0, 0, 0, 0.0, 0.0, warnings)
                return [], stats
            return []
        
        try:
            tokens = []
            messages = conversation.get('messages', [])
            content_tokens = 0
            special_tokens = 0
            processed_messages = 0
            
            if not messages:
                if return_stats:
                    end_time = time.perf_counter()
                    stats = TokenizationStats(0, 0, 0, 0, 0.0, (end_time - start_time) * 1000, warnings)
                    return [], stats
                return []
            
            for message in messages:
                if not isinstance(message, dict):
                    continue
                
                role = message.get('role', '').strip().lower()
                content = message.get('content', '')
                
                if not content and self.validation_level == "strict":
                    continue
                
                # Normalize role
                if role not in self._role_mapping:
                    role = 'user'  # Safe fallback
                
                # Start message
                tokens.append(self.special_tokens["<|im_start|>"])
                special_tokens += 1
                
                # Add role
                role_token = self._role_mapping[role]
                tokens.append(role_token)
                special_tokens += 1
                
                # Process and encode content
                if content:
                    processed_content = self._preprocess_content(str(content), mode)
                    if processed_content:
                        if self.enable_caching:
                            content_token_list, encoding_success = self._cached_encode(processed_content)
                        else:
                            try:
                                content_token_list = self.tokenizer.encode(
                                    processed_content,
                                    allowed_special={'<|endoftext|>'}
                                )

                                encoding_success = True
                            except Exception as e:
                                logging.warning(f"Content encoding failed: {e}")
                                content_token_list = []
                                encoding_success = False
                                self._apply_stats_delta({'encoding_errors': 1})
                        
                        if content_token_list:
                            tokens.extend(content_token_list)
                            content_tokens += len(content_token_list)
                
                # End message
                tokens.append(self.special_tokens["<|im_end|>"])
                special_tokens += 1
                processed_messages += 1
            
            # Apply length constraints
            max_len = max_length or self.max_context_length
            if len(tokens) > max_len:
                tokens = self._apply_truncation(tokens, max_len, truncation_strategy)
                warnings.append(f"Conversation truncated from {len(tokens)} to {max_len} tokens")
            
            # Update statistics
            self._apply_stats_delta({
                'total_conversations_processed': 1,
                'total_tokens_generated': len(tokens)
            })
            
            if return_stats:
                end_time = time.perf_counter()
                avg_tokens = len(tokens) / max(1, processed_messages)
                stats = TokenizationStats(
                    total_tokens=len(tokens),
                    special_tokens=special_tokens,
                    content_tokens=content_tokens,
                    message_count=processed_messages,
                    avg_tokens_per_message=avg_tokens,
                    encoding_time_ms=(end_time - start_time) * 1000,
                    warnings=warnings
                )
                return tokens, stats
            
            return tokens
            
        except Exception as e:
            logging.error(f"Conversation encoding failed: {e}")
            self._apply_stats_delta({'encoding_errors': 1})
            if return_stats:
                end_time = time.perf_counter()
                stats = TokenizationStats(0, 0, 0, 0, 0.0, (end_time - start_time) * 1000, [str(e)])
                return [], stats
            return []
    
    def _apply_truncation(self, tokens: List[int], max_length: int, strategy: str) -> List[int]:
        """Apply truncation strategy to token sequence."""
        if len(tokens) <= max_length:
            return tokens
        
        # Reserve space for truncation marker
        effective_max = max_length - 1
        
        if strategy == "start":
            # Keep the end, truncate the beginning
            truncated = tokens[-effective_max:]
        elif strategy == "middle":
            # Keep start and end, truncate middle
            start_keep = effective_max // 2
            end_keep = effective_max - start_keep
            truncated = tokens[:start_keep] + tokens[-end_keep:]
        else:  # "end" (default)
            # Keep the beginning, truncate the end
            truncated = tokens[:effective_max]
        
        # Add truncation marker
        truncated.append(self.special_tokens["<|truncated|>"])
        return truncated
    
    def decode(self, 
               token_ids: List[int], 
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True,
               handle_errors: str = "replace") -> str:
        """Enhanced decoding with comprehensive error handling."""
        if not token_ids:
            return ""
        
        try:
            # Validate and filter tokens
            valid_tokens = []
            special_token_positions = []
            
            for i, token_id in enumerate(token_ids):
                if not isinstance(token_id, int):
                    continue
                
                if self.is_special_token(token_id):
                    if not skip_special_tokens:
                        # Convert special tokens to readable format
                        special_text = self._reverse_special_tokens.get(token_id, f"<UNK_SPECIAL_{token_id}>")
                        # For now, we'll skip them as tiktoken can't decode them
                        special_token_positions.append((i, special_text))
                    continue
                
                # Clamp token to valid range
                if token_id < 0:
                    token_id = 0
                elif token_id >= self.base_vocab_size:
                    token_id = self.base_vocab_size - 1
                
                valid_tokens.append(token_id)
            
            if not valid_tokens:
                return ""
            
            # Decode with error handling
            if handle_errors == "strict":
                decoded = self.tokenizer.decode(valid_tokens)
            else:
                try:
                    decoded = self.tokenizer.decode(valid_tokens)
                except Exception as e:
                    logging.warning(f"Decode error (handling as {handle_errors}): {e}")
                    if handle_errors == "ignore":
                        return ""
                    elif handle_errors == "replace":
                        # Try to decode individual tokens and replace problematic ones
                        decoded_parts = []
                        for token in valid_tokens:
                            try:
                                part = self.tokenizer.decode([token])
                                decoded_parts.append(part)
                            except:
                                decoded_parts.append("�")  # Replacement character
                        decoded = "".join(decoded_parts)
                    else:
                        return "<decode_error>"
            
            # Clean up tokenization artifacts
            if clean_up_tokenization_spaces:
                # Common tiktoken cleanup
                decoded = decoded.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
                decoded = decoded.replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
                decoded = decoded.replace("( ", "(").replace(" )", ")")
                decoded = decoded.replace("[ ", "[").replace(" ]", "]")
            
            return decoded
            
        except Exception as e:
            logging.error(f"Decode error: {e}")
            if handle_errors == "strict":
                raise
            return "<decode_error>" if handle_errors == "replace" else ""
    
    def encode_batch(self, 
                    conversations: List[Dict[str, Any]], 
                    max_workers: int = 4,
                    **kwargs) -> List[List[int]]:
        """Parallel batch encoding for multiple conversations."""
        if not conversations:
            return []
        
        if len(conversations) == 1 or max_workers == 1:
            # Single-threaded for small batches
            return [self.encode_conversation(conv, **kwargs) for conv in conversations]
        
        # Multi-threaded processing
        results = [None] * len(conversations)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.encode_conversation, conv, **kwargs): i 
                for i, conv in enumerate(conversations)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logging.error(f"Batch encoding failed for conversation {index}: {e}")
                    results[index] = []
        
        return results
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token."""
        return token_id in self._reverse_special_tokens
    
    def get_role_token(self, role: str) -> int:
        """Get token ID for a role with enhanced mapping."""
        normalized_role = role.lower().strip()
        return self._role_mapping.get(normalized_role, self.special_tokens["<|user|>"])
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get all special tokens."""
        return self.special_tokens.copy()
    
    def get_vocab_size(self) -> int:
        """Get total vocabulary size including special tokens."""
        return self.vocab_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenizer usage statistics."""
        if self.thread_safe:
            with self._lock:
                stats = self.stats.copy()
        else:
            stats = self.stats.copy()
        if self.enable_caching:
            cache_info = self._cached_encode.cache_info()
            stats.update({
                'cache_hits': cache_info.hits,
                'cache_misses': cache_info.misses,
                'cache_size': cache_info.currsize,
                'cache_max_size': cache_info.maxsize,
            })
        return stats
    
    def reset_stats(self):
        """Reset usage statistics."""
        new_stats = {
            'total_conversations_processed': 0,
            'total_tokens_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'encoding_errors': 0,
        }
        if self.thread_safe:
            with self._lock:
                self.stats = new_stats
        else:
            self.stats = new_stats
        if self.enable_caching:
            self._cached_encode.cache_clear()
    
    def estimate_tokens(self, text: str) -> int:
        """Quick token count estimation without full encoding."""
        # For GPT-4 (cl100k_base), roughly ~3-4 characters per token for English
        # More accurate for length planning without full encoding cost
        return max(1, len(text) // 3)
    
    def truncate_to_limit(self, 
                         conversation: Dict[str, Any], 
                         max_tokens: int,
                         preserve_messages: int = 1) -> Dict[str, Any]:
        """
        Truncate conversation to fit within token limit while preserving structure.
        
        Args:
            conversation: Input conversation
            max_tokens: Maximum allowed tokens
            preserve_messages: Number of recent messages to always keep
        """
        messages = conversation.get('messages', [])
        if not messages:
            return conversation
        
        # Pre-encode each message once and build a suffix that fits the token budget.
        preserve_count = min(preserve_messages, len(messages))
        encoded_messages = [
            self._encode_message_tokens(message, TokenizationMode.STANDARD)
            for message in messages
        ]

        selected_indices = list(range(len(messages) - preserve_count, len(messages)))
        current_token_count = sum(len(encoded_messages[i]) for i in selected_indices)

        for idx in range(len(messages) - preserve_count - 1, -1, -1):
            message_len = len(encoded_messages[idx])
            if message_len == 0:
                continue
            if current_token_count + message_len <= max_tokens:
                selected_indices.insert(0, idx)
                current_token_count += message_len
            else:
                break

        result_messages = [messages[i] for i in selected_indices]
        
        # Create result conversation
        result = conversation.copy()
        result['messages'] = result_messages
        
        return result
    
    def __repr__(self) -> str:
        return (f"ConversationTokenizer(model='{self.model_name}', "
                f"vocab_size={self.vocab_size:,}, "
                f"max_context={self.max_context_length:,}, "
                f"pad_token_id={self.pad_token_id})")
