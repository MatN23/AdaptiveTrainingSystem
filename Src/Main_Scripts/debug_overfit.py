# Copyright (c) 2025 MatN23. All rights reserved.

import torch
import torch.nn as nn
from core.model import DeepSeekTransformer, DeepSeekConfig
from core.tokenizer import ConversationTokenizer
from training.trainer import EnhancedConversationTrainer
import logging

def debug_training():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Create a minimal config
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=512,
        use_moe=True,
        num_experts=4,
        moe_top_k=1,
        learning_rate=1e-3, # Use a larger LR for overfitting
        seq_length=128,
        batch_size=1
    )
    
    # 2. Mock tokenizer (just need it for pad_token_id)
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.special_tokens = {"<|im_start|>": 1, "<|im_end|>": 2}
    tokenizer = MockTokenizer()
    
    # 3. Create model
    model = DeepSeekTransformer(config)
    
    # 4. Create one batch of data
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (1, config.seq_length)),
        'labels': torch.randint(0, config.vocab_size, (1, config.seq_length)),
        'attention_mask': torch.ones((1, config.seq_length)),
        'loss_weights': torch.ones((1, config.seq_length))
    }
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # 5. Initialize trainer
    trainer = EnhancedConversationTrainer(model, config, tokenizer)
    
    # 6. Overfit loop
    print("Starting overfit test (single batch)...")
    for step in range(100):
        # Manual train step to inspect gradients
        model.train()
        trainer.optimizer.zero_grad()
        
        output = model(batch['input_ids'], batch['attention_mask'])
        if isinstance(output, tuple):
            logits, aux_loss, _ = output
        else:
            logits = output
            aux_loss = 0
            
        loss_dict = trainer.compute_loss(logits, batch['labels'], batch['loss_weights'])
        loss = loss_dict['loss'] + aux_loss
        
        loss.backward()
        
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append((name, param.grad.norm().item()))
        
        # Sort by norm to see biggest/smallest
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        
        if step % 10 == 0:
            avg_grad_norm = sum(n for _, n in grad_norms) / len(grad_norms) if grad_norms else 0
            print(f"Step {step}: Loss = {loss.item():.4f}, Acc = {loss_dict['accuracy']:.2%}, Avg Grad Norm = {avg_grad_norm:.6f}")
            print(f"  Top Grads: {grad_norms[:3]}")
            print(f"  Bottom Grads: {grad_norms[-3:]}")
            
        trainer.optimizer.step()

if __name__ == "__main__":
    debug_training()