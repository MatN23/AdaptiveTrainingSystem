# Licensed under the Custom License below.

import json
import logging
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def process_oasst_data(input_path: str, output_path: str, max_conversations: int = None) -> int:
    """Enhanced OASST data processing with validation."""
    conversations = []
    stats = {'processed': 0, 'valid': 0, 'errors': 0}
    
    logging.info(f"Processing OASST data: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if max_conversations and stats['valid'] >= max_conversations:
                break
            
            try:
                data = json.loads(line.strip())
                stats['processed'] += 1
                
                # Extract and validate messages
                if 'messages' in data and len(data['messages']) >= 2:
                    messages = []
                    for msg in data['messages']:
                        role = msg.get('role', '').lower()
                        content = msg.get('content', '').strip()
                        
                        if not content:
                            continue
                        
                        # Normalize role names
                        if role == 'prompter':
                            role = 'user'
                        elif role not in ['user', 'assistant', 'system']:
                            role = 'user'
                        
                        messages.append({'role': role, 'content': content})
                    
                    if len(messages) >= 2:
                        conversation = {
                            'conversation_id': data.get('conversation_id', f'conv_{line_no}'),
                            'messages': messages,
                            'metadata': {
                                'source': 'oasst',
                                'processed_at': datetime.now().isoformat()
                            }
                        }
                        conversations.append(conversation)
                        stats['valid'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                if line_no <= 10:
                    logging.warning(f"Error processing line {line_no}: {e}")
            
            # Progress update
            if line_no % 10000 == 0:
                logging.info(f"Processed {line_no:,} lines, {stats['valid']:,} valid conversations")
    
    # Write processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Processing complete: {stats['valid']:,} valid conversations from {stats['processed']:,} total")
    logging.info(f"Output written to: {output_path}")
    
    return stats['valid']


def validate_data_comprehensive(data_path: str, tokenizer, max_check: int = 5000) -> Dict[str, Any]:
    """Comprehensive data validation with detailed statistics."""
    stats = {
        'file_info': {},
        'conversation_stats': {},
        'token_stats': {},
        'quality_metrics': {},
        'errors': []
    }
    
    # File information
    try:
        file_path = Path(data_path)
        if not file_path.exists():
            stats['errors'].append(f"File not found: {data_path}")
            return stats
        
        stats['file_info'] = {
            'path': str(file_path),
            'size_mb': file_path.stat().st_size / 1e6,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
    except Exception as e:
        stats['errors'].append(f"File access error: {e}")
        return stats
    
    conversation_lengths = []
    token_lengths = []
    role_counts = {'user': 0, 'assistant': 0, 'system': 0, 'other': 0}
    quality_issues = []
    
    total_lines = 0
    valid_conversations = 0
    
    logging.info(f"Validating data: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line_no > max_check:
                break
            
            total_lines += 1
            
            try:
                conversation = json.loads(line.strip())
                
                if 'messages' not in conversation:
                    quality_issues.append(f"Line {line_no}: Missing 'messages' field")
                    continue
                
                messages = conversation['messages']
                if not isinstance(messages, list) or len(messages) == 0:
                    quality_issues.append(f"Line {line_no}: Empty or invalid messages")
                    continue
                
                conversation_lengths.append(len(messages))
                
                # Analyze messages
                has_user = False
                has_assistant = False
                total_content_length = 0
                
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        quality_issues.append(f"Line {line_no}, Message {msg_idx}: Invalid message format")
                        continue
                    
                    role = msg.get('role', '').lower()
                    content = msg.get('content', '')
                    
                    if not content or not content.strip():
                        quality_issues.append(f"Line {line_no}, Message {msg_idx}: Empty content")
                        continue
                    
                    total_content_length += len(content)
                    
                    # Count roles
                    if role in ['user', 'prompter']:
                        role_counts['user'] += 1
                        has_user = True
                    elif role == 'assistant':
                        role_counts['assistant'] += 1
                        has_assistant = True
                    elif role == 'system':
                        role_counts['system'] += 1
                    else:
                        role_counts['other'] += 1
                
                if not (has_user and has_assistant):
                    quality_issues.append(f"Line {line_no}: Missing user or assistant messages")
                    continue
                
                # Tokenize and analyze
                try:
                    tokens = tokenizer.encode_conversation(conversation)
                    if tokens:
                        token_lengths.append(len(tokens))
                        valid_conversations += 1
                    else:
                        quality_issues.append(f"Line {line_no}: Tokenization failed")
                except Exception as e:
                    quality_issues.append(f"Line {line_no}: Tokenization error: {e}")
                
            except json.JSONDecodeError as e:
                quality_issues.append(f"Line {line_no}: JSON decode error: {e}")
            except Exception as e:
                quality_issues.append(f"Line {line_no}: Processing error: {e}")
            
            # Progress update
            if line_no % 1000 == 0:
                logging.info(f"Validated {line_no:,} lines...")
    
    # Compute statistics
    stats['conversation_stats'] = {
        'total_lines': total_lines,
        'valid_conversations': valid_conversations,
        'invalid_conversations': total_lines - valid_conversations,
        'avg_messages_per_conversation': np.mean(conversation_lengths) if conversation_lengths else 0,
        'max_messages': max(conversation_lengths) if conversation_lengths else 0,
        'min_messages': min(conversation_lengths) if conversation_lengths else 0,
        'role_distribution': role_counts
    }
    
    stats['token_stats'] = {
        'avg_tokens': np.mean(token_lengths) if token_lengths else 0,
        'median_tokens': np.median(token_lengths) if token_lengths else 0,
        'max_tokens': max(token_lengths) if token_lengths else 0,
        'min_tokens': min(token_lengths) if token_lengths else 0,
        'std_tokens': np.std(token_lengths) if token_lengths else 0
    }
    
    stats['quality_metrics'] = {
        'success_rate': valid_conversations / total_lines if total_lines > 0 else 0,
        'error_rate': len(quality_issues) / total_lines if total_lines > 0 else 0,
        'total_quality_issues': len(quality_issues),
        'quality_issues_sample': quality_issues[:20]  # First 20 issues
    }
    
    return stats


def create_sample_data(output_path: str, num_conversations: int = 100):
    """Create sample conversation data for testing."""
    sample_conversations = []
    
    # Templates for sample conversations
    templates = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you write a simple Python function?"},
                {"role": "assistant", "content": "Sure! Here's a simple function that adds two numbers:\n\n```python\ndef add_numbers(a, b):\n    return a + b\n```"}
            ]
        }
    ]
    
    import random
    
    for i in range(num_conversations):
        template = random.choice(templates)
        conversation = {
            'conversation_id': f'sample_{i:04d}',
            'messages': template['messages'],
            'metadata': {
                'source': 'sample_data',
                'created_at': datetime.now().isoformat()
            }
        }
        sample_conversations.append(conversation)
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + '\n')
    
    logging.info(f"Created {num_conversations} sample conversations: {output_path}")
    return num_conversations