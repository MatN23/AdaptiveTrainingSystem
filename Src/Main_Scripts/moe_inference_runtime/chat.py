#!/usr/bin/env python3
"""
Interactive chat with C++ inference engine using your custom tokenizer.
Handles proper tokenization and detokenization.
"""

import subprocess
import sys
import os
import struct
import tempfile
from pathlib import Path

# Add core directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
core_dir = os.path.join(parent_dir, 'core')

sys.path.insert(0, core_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

try:
    from tokenizer import ConversationTokenizer, TokenizationMode
    print(f" Loaded tokenizer from: {core_dir}")
except ImportError as e:
    print(f" Failed to import tokenizer: {e}")
    print(f"\nSearched in:")
    print(f"  - {core_dir}")
    print(f"  - {parent_dir}")
    print(f"  - {script_dir}")
    print(f"\nPlease ensure tokenizer.py exists at: {core_dir}/tokenizer.py")
    sys.exit(1)


class ChatInterface:
    def __init__(self, binary_path, model_path, tokenizer_model="gpt-4"):
        self.binary_path = binary_path
        self.model_path = model_path
        
        print(f" Initializing tokenizer...")
        self.tokenizer = ConversationTokenizer(
            model_name=tokenizer_model,
            max_context_length=8192,
            enable_caching=True,
            thread_safe=True
        )
        
        print(f" Tokenizer ready")
        print(f"  Vocab size: {self.tokenizer.get_vocab_size():,}")
        print(f"  Pad token: {self.tokenizer.pad_token_id}")
    
    def format_conversation(self, history):
        """Format conversation history for tokenization"""
        return {
            'messages': [
                {'role': msg['role'], 'content': msg['content']}
                for msg in history
            ]
        }
    
    def generate_with_cpp(self, prompt_tokens, max_tokens=100, temperature=1.0, debug=False):
        """
        Call C++ binary with token IDs via temporary file.
        """
        # Create temp file with token IDs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tokens', delete=False) as f:
            # Write token IDs as comma-separated integers
            f.write(','.join(map(str, prompt_tokens)))
            token_file = f.name
        
        try:
            # Call C++ binary
            cmd = [self.binary_path, self.model_path, token_file]
            
            if debug:
                print(f"\n Running: {' '.join(cmd)}")
                print(f" Token file: {token_file}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Debug: print all output
            if debug:
                if result.stdout:
                    print(f"\n--- C++ stdout ---")
                    print(result.stdout)
                    print(f"--- end stdout ---\n")
                
                if result.stderr:
                    print(f"\n--- C++ stderr ---")
                    print(result.stderr)
                    print(f"--- end stderr ---\n")
                
                if result.returncode != 0:
                    print(f"  C++ binary returned code {result.returncode}")
            
            # Parse output token IDs - handle multi-line output
            token_ids = []
            output_text = result.stdout
            
            # Find "Token IDs:" and capture everything after it
            if 'Token IDs:' in output_text:
                # Get everything after "Token IDs:"
                after_marker = output_text.split('Token IDs:', 1)[1]
                
                # Extract all numbers (including across newlines)
                import re
                numbers = re.findall(r'-?\d+', after_marker)
                
                for num_str in numbers:
                    try:
                        token_ids.append(int(num_str))
                    except ValueError:
                        pass
            
            if debug:
                print(f" Parsed {len(token_ids)} token IDs")
                if token_ids:
                    print(f"   First 10: {token_ids[:10]}")
            
            return token_ids
            
        except subprocess.TimeoutExpired:
            print("\n  Generation timed out")
            return []
        except Exception as e:
            print(f"\n  Error: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return []
        finally:
            # Clean up temp file
            try:
                os.unlink(token_file)
            except:
                pass
    
    def chat_interactive(self, debug=False):
        """Interactive chat loop"""
        print("\n" + "="*60)
        if debug:
            print(" Interactive Chat Mode (Debug)")
        else:
            print(" Interactive Chat Mode")
        print("="*60)
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit chat")
        print("  'stats' - Show tokenizer statistics")
        print("  'clear' - Clear conversation history")
        if debug:
            print("  'debug off' - Disable debug output")
        else:
            print("  'debug on' - Enable debug output")
        print()
        
        history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.tokenizer.get_stats()
                    print("\n Tokenizer Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'clear':
                    history = []
                    print(" Conversation cleared")
                    continue
                
                if user_input.lower() == 'debug on':
                    debug = True
                    print(" Debug mode enabled")
                    continue
                
                if user_input.lower() == 'debug off':
                    debug = False
                    print(" Debug mode disabled")
                    continue
                
                # Add user message to history
                history.append({'role': 'user', 'content': user_input})
                
                # Encode conversation
                conversation = self.format_conversation(history)
                prompt_tokens, stats = self.tokenizer.encode_conversation(
                    conversation,
                    mode=TokenizationMode.CHAT_OPTIMIZED,
                    return_stats=True
                )
                
                if debug:
                    print(f"\n Encoded: {stats.total_tokens} tokens ({stats.content_tokens} content + {stats.special_tokens} special)")
                
                if not prompt_tokens:
                    print("  Failed to encode conversation")
                    history.pop()  # Remove failed message
                    continue
                
                # Generate response
                print("\n\033[1;32mAssistant:\033[0m ", end='', flush=True)
                
                response_tokens = self.generate_with_cpp(prompt_tokens, debug=debug)
                
                if response_tokens:
                    # Decode response
                    response_text = self.tokenizer.decode(
                        response_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    if not debug:
                        print(response_text)
                    else:
                        print(f"\n{response_text}")
                        print(f"\n Generated {len(response_tokens)} tokens")
                    
                    # Add to history
                    history.append({'role': 'assistant', 'content': response_text})
                else:
                    print("\n  No response generated")
                    history.pop()  # Remove the user message since there was no response
                
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
        
        history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.tokenizer.get_stats()
                    print("\n Tokenizer Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'clear':
                    history = []
                    print(" Conversation cleared")
                    continue
                
                # Add user message to history
                history.append({'role': 'user', 'content': user_input})
                
                # Encode conversation
                conversation = self.format_conversation(history)
                prompt_tokens, stats = self.tokenizer.encode_conversation(
                    conversation,
                    mode=TokenizationMode.CHAT_OPTIMIZED,
                    return_stats=True
                )
                
                print(f"\n Encoded: {stats.total_tokens} tokens ({stats.content_tokens} content + {stats.special_tokens} special)")
                
                if not prompt_tokens:
                    print("  Failed to encode conversation")
                    history.pop()  # Remove failed message
                    continue
                
                # Generate response
                print("\n\033[1;32mAssistant:\033[0m ", end='', flush=True)
                
                response_tokens = self.generate_with_cpp(prompt_tokens)
                
                if response_tokens:
                    # Decode response
                    response_text = self.tokenizer.decode(
                        response_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    print(response_text)
                    print(f"\n Generated {len(response_tokens)} tokens")
                    
                    # Add to history
                    history.append({'role': 'assistant', 'content': response_text})
                else:
                    print("\n  No response generated")
                    history.pop()  # Remove the user message since there was no response
                
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                import traceback
                traceback.print_exc()
    
    def chat_single(self, prompt):
        """Single prompt mode"""
        print(f"\n Prompt: {prompt}\n")
        
        # Create conversation
        conversation = self.format_conversation([
            {'role': 'user', 'content': prompt}
        ])
        
        # Encode
        prompt_tokens, stats = self.tokenizer.encode_conversation(
            conversation,
            mode=TokenizationMode.CHAT_OPTIMIZED,
            return_stats=True
        )
        
        print(f" Encoded: {stats.total_tokens} tokens")
        print(f"   Content: {stats.content_tokens} | Special: {stats.special_tokens}")
        print(f"   Encoding time: {stats.encoding_time_ms:.2f}ms")
        
        if stats.warnings:
            print(f"  Warnings:")
            for warning in stats.warnings:
                print(f"   - {warning}")
        
        if not prompt_tokens:
            print(" Failed to encode prompt")
            return
        
        print(f"\n Token IDs (first 20): {prompt_tokens[:20]}")
        
        # Generate
        print("\n Generating response...\n")
        response_tokens = self.generate_with_cpp(prompt_tokens)
        
        if response_tokens:
            print(f"\n Generated {len(response_tokens)} tokens")
            print(f"   Token IDs (first 20): {response_tokens[:20]}")
            
            # Decode
            response_text = self.tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"\n{'='*60}")
            print("Response:")
            print(f"{'='*60}")
            print(response_text)
            print(f"{'='*60}\n")
        else:
            print(" No response generated")


def main():
    if len(sys.argv) < 3:
        print("Usage: python chat.py <binary> <model.bin> [tokenizer_model] [--debug] [prompt]")
        print("\nExamples:")
        print("  Interactive mode (clean):")
        print("    python chat.py ./moe_inference_cpu model.bin gpt-4")
        print("\n  Interactive mode (debug):")
        print("    python chat.py ./moe_inference_cpu model.bin gpt-4 --debug")
        print("\n  Single prompt (clean):")
        print("    python chat.py ./moe_inference_cpu model.bin gpt-4 'Hello!'")
        print("\n  Single prompt (debug):")
        print("    python chat.py ./moe_inference_cpu model.bin gpt-4 --debug 'Hello!'")
        print("\nTokenizer models: gpt-4, gpt-4-turbo, gpt-3.5-turbo")
        return
    
    binary = sys.argv[1]
    model = sys.argv[2]
    
    # Parse arguments
    tokenizer_model = "gpt-4"
    debug = False
    prompt = None
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == '--debug':
            debug = True
        elif arg.startswith('gpt') or arg in ['cl100k_base', 'p50k_base', 'r50k_base']:
            tokenizer_model = arg
        else:
            prompt = arg
    
    # Validate files
    if not os.path.exists(binary):
        print(f" Binary not found: {binary}")
        print("   Run ./compile.sh first")
        return
    
    if not os.path.exists(model):
        print(f" Model not found: {model}")
        print("   Create model with export script or create_test_model.py")
        return
    
    # Initialize chat interface
    try:
        chat = ChatInterface(binary, model, tokenizer_model)
        
        if prompt:
            # Single prompt mode
            chat.chat_single(prompt)
        else:
            # Interactive mode
            chat.chat_interactive(debug=debug)
            
    except Exception as e:
        print(f" Failed to initialize: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()