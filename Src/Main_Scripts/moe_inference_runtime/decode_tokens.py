#!/usr/bin/env python3
"""
Decode token IDs back to text using the model's tokenizer.
Usage: python decode_tokens.py <token_ids>
Example: python decode_tokens.py "1, 2, 3, 4, 5"
"""

import sys
from transformers import AutoTokenizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_tokens.py <token_ids>")
        print('Example: python decode_tokens.py "1, 2, 3, 4, 5"')
        return
    
    # Parse token IDs
    token_str = sys.argv[1]
    token_ids = [int(x.strip()) for x in token_str.split(',')]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Change to your tokenizer
        text = tokenizer.decode(token_ids)
        print(text)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure to install transformers: pip install transformers")
        print("And adjust the tokenizer name in the script")

if __name__ == '__main__':
    main()