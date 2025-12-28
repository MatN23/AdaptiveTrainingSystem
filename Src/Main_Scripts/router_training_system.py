# router_training_system.py
"""
Complete system for training a 14M router model for MoE routing

TRAINING STRATEGY:
1. Collect routing data from your existing model
2. Train router to predict which experts should be used
3. Replace simple gates with trained router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)
_GLOBAL_ROUTER_CACHE = {}


class RouterGateAdapter(nn.Module):
    def __init__(self, router_checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        
        cache_key = str(router_checkpoint_path)
        
        # ✅ Check if router already loaded
        if cache_key not in _GLOBAL_ROUTER_CACHE:
            print(f"🔄 Loading router model (first time)...")
            
            # Load trained router
            checkpoint = torch.load(router_checkpoint_path, map_location=device)
            config = checkpoint['config']
            
            router = SmallRouterModel(
                hidden_size=config['hidden_size'],
                num_experts=config['num_experts'],
                top_k=config['top_k'],
                router_hidden=config.get('router_hidden', 256),
                num_layers=config.get('router_layers', 4),
                num_heads=config.get('router_heads', 4)
            )
            
            router.load_state_dict(checkpoint['model_state_dict'])
            router.eval()
            router.to(device)
            
            # ✅ Cache it globally
            _GLOBAL_ROUTER_CACHE[cache_key] = router
            print(f"✅ Router cached ({sum(p.numel() for p in router.parameters())/1e6:.1f}M params)")
        else:
            print(f"♻️ Reusing cached router model")
        
        # ✅ All adapters share the SAME router
        self.router = _GLOBAL_ROUTER_CACHE[cache_key]
        
        # Store metadata
        self.num_experts = self.router.num_experts
        self.top_k = self.router.top_k
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch*seq_len, hidden_size] or [batch, seq_len, hidden_size]
            
        Returns:
            expert_logits: [batch*seq_len, num_experts]
        """
        # Ensure 3D input for transformer
        if x.dim() == 2:
            x = x.unsqueeze(0)
            needs_squeeze = True
        else:
            needs_squeeze = False
        
        # Get router predictions (use compatibility mode)
        with torch.set_grad_enabled(self.training):
            # ✅ Use return_dict=False for gate compatibility
            expert_logits = self.router(x, return_dict=False)
        
        # Restore original shape
        if needs_squeeze:
            expert_logits = expert_logits.squeeze(0)
        
        # Flatten if needed
        if expert_logits.dim() == 3:
            batch, seq, experts = expert_logits.shape
            expert_logits = expert_logits.reshape(batch * seq, experts)
        
        return expert_logits

class RoutingDataCollector:
    """
    Collect routing decisions from your existing MoE model
    This creates training data for the router model
    """
    
    def __init__(self, model, save_dir: str = "routing_data"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.routing_examples = []
        
        # Hook into MoE layers to capture routing
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture routing decisions"""
        def routing_hook(module, input, output):
            """Capture: input hidden states + which experts were chosen"""
            if hasattr(output, 'expert_indices'):
                # Capture the data
                hidden_states = input[0].detach().cpu()
                expert_indices = output.expert_indices.detach().cpu()
                routing_weights = output.routing_weights.detach().cpu()
                
                # Store for training
                self.routing_examples.append({
                    'hidden_states': hidden_states,
                    'expert_indices': expert_indices,
                    'routing_weights': routing_weights,
                    'layer_id': module.layer_id if hasattr(module, 'layer_id') else -1
                })
        
        # Attach hooks to all MoE layers
        for name, module in self.model.named_modules():
            if 'moe' in name.lower() or 'expert' in name.lower():
                hook = module.register_forward_hook(routing_hook)
                self.hooks.append(hook)
                logger.info(f"Registered routing hook on: {name}")
    
    def collect_routing_data(
        self, 
        dataloader, 
        num_batches: int = 1000,
        save_every: int = 100
    ):
        """
        Run model on data and collect routing decisions
        
        Args:
            dataloader: Your existing training dataloader
            num_batches: How many batches to collect
            save_every: Save checkpoints every N batches
        """
        self.model.eval()
        
        logger.info(f"Collecting routing data from {num_batches} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                # Run forward pass (routing hooks will capture data)
                _ = self.model(**batch)
                
                # Periodic save
                if (batch_idx + 1) % save_every == 0:
                    self._save_checkpoint(batch_idx + 1)
                    logger.info(f"Collected {len(self.routing_examples)} routing examples")
        
        # Final save
        self._save_final_dataset()
        logger.info(f"Routing data collection complete: {len(self.routing_examples)} examples")
        
        return len(self.routing_examples)
    
    def _save_checkpoint(self, batch_idx: int):
        """Save intermediate checkpoint"""
        checkpoint_path = self.save_dir / f"routing_checkpoint_{batch_idx}.pt"
        torch.save({
            'routing_examples': self.routing_examples,
            'num_examples': len(self.routing_examples)
        }, checkpoint_path)
    
    def _save_final_dataset(self):
        """Save final routing dataset"""
        dataset_path = self.save_dir / "routing_dataset.pt"
        torch.save({
            'routing_examples': self.routing_examples,
            'num_examples': len(self.routing_examples),
            'dataset_info': {
                'num_layers': len(set(ex['layer_id'] for ex in self.routing_examples)),
                'hidden_size': self.routing_examples[0]['hidden_states'].shape[-1],
                'num_experts': self.routing_examples[0]['expert_indices'].shape[-1],
            }
        }, dataset_path)
        logger.info(f"Saved routing dataset: {dataset_path}")
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class SmallRouterModel(nn.Module):
    """
    14M parameter router model
    Takes hidden states, outputs expert selection
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 8,
        top_k: int = 2,
        router_hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Project input to router dimension
        self.input_proj = nn.Linear(hidden_size, router_hidden)
        
        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=router_hidden,
            nhead=num_heads,
            dim_feedforward=router_hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: predict expert probabilities
        self.expert_head = nn.Sequential(
            nn.Linear(router_hidden, router_hidden),
            nn.GELU(),
            nn.LayerNorm(router_hidden),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, num_experts)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Router model: {total_params/1e6:.2f}M parameters")
    
    def forward(self, hidden_states: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) or (tokens, hidden_size)
            return_dict: If False, returns only logits for compatibility
            
        Returns:
            If return_dict=True: Dictionary with all outputs
            If return_dict=False: Just expert_logits tensor
        """
        # Handle 2D input
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Project to router space
        x = self.input_proj(hidden_states)
        
        # Process with transformer
        x = self.transformer(x)
        
        # Predict expert logits
        expert_logits = self.expert_head(x)
        
        # Squeeze if needed
        if squeeze_output:
            expert_logits_flat = expert_logits.squeeze(0)
        else:
            batch, seq, experts = expert_logits.shape
            expert_logits_flat = expert_logits.reshape(batch * seq, experts)
        
        # Return format based on mode
        if not return_dict:
            return expert_logits_flat
        
        # Get top-k experts
        expert_probs = F.softmax(expert_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(expert_probs, k=self.top_k, dim=-1)
        
        # Renormalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return {
            'expert_logits': expert_logits,
            'expert_probs': expert_probs,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs
        }

class RouterTrainingDataset(Dataset):
    """Dataset for training the router model"""
    
    def __init__(self, routing_data_path: str):
        data = torch.load(routing_data_path)
        self.examples = data['routing_examples']
        self.dataset_info = data.get('dataset_info', {})
        
        logger.info(f"Loaded {len(self.examples)} routing examples")
        logger.info(f"Dataset info: {self.dataset_info}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'hidden_states': example['hidden_states'],
            'target_expert_indices': example['expert_indices'],
            'target_routing_weights': example['routing_weights'],
            'layer_id': example['layer_id']
        }

class RouterTrainer:
    """Train the router model to predict expert routing"""
    
    def __init__(
        self,
        router_model: SmallRouterModel,
        config: Dict,
        device: str = 'cuda'
    ):
        self.router_model = router_model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            router_model.parameters(),
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = None  # Set up after knowing total steps
        
        # Loss weights
        self.classification_weight = config.get('classification_weight', 1.0)
        self.distribution_weight = config.get('distribution_weight', 0.5)
        self.load_balance_weight = config.get('load_balance_weight', 0.01)
        
        # Metrics
        self.best_accuracy = 0.0
        self.global_step = 0
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Multiple objectives:
        1. Classification: predict correct expert indices
        2. Distribution matching: match routing probabilities
        3. Load balancing: encourage diverse expert usage
        """
        expert_logits = predictions['expert_logits']
        expert_probs = predictions['expert_probs']
        target_indices = targets['target_expert_indices']
        target_weights = targets['target_routing_weights']
        
        batch_size, seq_len, num_experts = expert_logits.shape
        top_k = target_indices.shape[-1]
        
        # Loss 1: Classification loss (predict correct experts)
        # Convert expert indices to one-hot and sum across top-k
        target_onehot = torch.zeros_like(expert_logits)
        for k in range(top_k):
            indices = target_indices[:, :, k]
            target_onehot.scatter_(-1, indices.unsqueeze(-1), 1.0)
        
        classification_loss = F.binary_cross_entropy_with_logits(
            expert_logits, 
            target_onehot
        )
        
        # Loss 2: Distribution matching (KL divergence)
        # Create target distribution from target weights
        target_dist = torch.zeros_like(expert_probs)
        for k in range(top_k):
            indices = target_indices[:, :, k]
            weights = target_weights[:, :, k]
            target_dist.scatter_add_(-1, indices.unsqueeze(-1), weights.unsqueeze(-1))
        
        distribution_loss = F.kl_div(
            expert_probs.log(),
            target_dist,
            reduction='batchmean'
        )
        
        # Loss 3: Load balancing (encourage uniform expert usage)
        expert_usage = expert_probs.mean(dim=[0, 1])  # Average across batch and sequence
        uniform_dist = torch.ones_like(expert_usage) / num_experts
        load_balance_loss = F.kl_div(
            expert_usage.log(),
            uniform_dist,
            reduction='batchmean'
        )
        
        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.distribution_weight * distribution_loss +
            self.load_balance_weight * load_balance_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'distribution_loss': distribution_loss,
            'load_balance_loss': load_balance_loss
        }
    
    def compute_accuracy(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute routing accuracy metrics"""
        pred_indices = predictions['top_k_indices']
        target_indices = targets['target_expert_indices']
        
        # Exact match accuracy (all top-k match)
        exact_match = (pred_indices == target_indices).all(dim=-1).float().mean()
        
        # Any match accuracy (at least one expert matches)
        any_match = torch.any(
            pred_indices.unsqueeze(-1) == target_indices.unsqueeze(-2),
            dim=[-1, -2]
        ).float().mean()
        
        # Top-1 accuracy
        top1_match = (pred_indices[:, :, 0] == target_indices[:, :, 0]).float().mean()
        
        return {
            'exact_match_acc': exact_match.item(),
            'any_match_acc': any_match.item(),
            'top1_acc': top1_match.item()
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.router_model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            hidden_states = batch['hidden_states'].to(self.device)
            target_indices = batch['target_expert_indices'].to(self.device)
            target_weights = batch['target_routing_weights'].to(self.device)
            
            # Forward pass
            predictions = self.router_model(hidden_states)
            
            # Compute loss
            losses = self.compute_loss(
                predictions,
                {
                    'target_expert_indices': target_indices,
                    'target_routing_weights': target_weights
                }
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.router_model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Compute accuracy
            accuracy = self.compute_accuracy(
                predictions,
                {
                    'target_expert_indices': target_indices,
                    'target_routing_weights': target_weights
                }
            )
            
            total_loss += losses['total_loss'].item()
            total_acc += accuracy['top1_acc']
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {losses['total_loss'].item():.4f} | "
                    f"Top-1 Acc: {accuracy['top1_acc']:.4f} | "
                    f"Exact Match: {accuracy['exact_match_acc']:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate router model"""
        self.router_model.eval()
        
        total_loss = 0.0
        total_metrics = {
            'exact_match_acc': 0.0,
            'any_match_acc': 0.0,
            'top1_acc': 0.0
        }
        num_batches = 0
        
        for batch in val_loader:
            hidden_states = batch['hidden_states'].to(self.device)
            target_indices = batch['target_expert_indices'].to(self.device)
            target_weights = batch['target_routing_weights'].to(self.device)
            
            predictions = self.router_model(hidden_states)
            
            losses = self.compute_loss(
                predictions,
                {
                    'target_expert_indices': target_indices,
                    'target_routing_weights': target_weights
                }
            )
            
            accuracy = self.compute_accuracy(
                predictions,
                {
                    'target_expert_indices': target_indices,
                    'target_routing_weights': target_weights
                }
            )
            
            total_loss += losses['total_loss'].item()
            for key in total_metrics:
                total_metrics[key] += accuracy[key]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return {'loss': avg_loss, **total_metrics}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save router model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.router_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step
        }, path)
        logger.info(f"Saved checkpoint: {path}")

def train_router_model(
    existing_model,
    train_dataloader,
    config: Dict,
    output_dir: str = "router_training"
):
    """
    Complete pipeline to train a router model
    
    Args:
        existing_model: Your existing MoE model
        train_dataloader: Your training data
        config: Router training configuration
        output_dir: Where to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("ROUTER MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Collect routing data
    logger.info("\nSTEP 1: Collecting routing data from existing model...")
    collector = RoutingDataCollector(existing_model, save_dir=str(output_dir / "routing_data"))
    num_examples = collector.collect_routing_data(
        train_dataloader,
        num_batches=config.get('collection_batches', 1000)
    )
    collector.cleanup()
    
    # Step 2: Create router model
    logger.info("\nSTEP 2: Creating router model...")
    router_model = SmallRouterModel(
        hidden_size=config['hidden_size'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        router_hidden=config.get('router_hidden', 256),
        num_layers=config.get('router_layers', 4),
        num_heads=config.get('router_heads', 4)
    )
    
    # Step 3: Create dataset
    logger.info("\nSTEP 3: Creating training dataset...")
    dataset = RouterTrainingDataset(str(output_dir / "routing_data" / "routing_dataset.pt"))
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=4
    )
    
    # Step 4: Train router
    logger.info("\nSTEP 4: Training router model...")
    trainer = RouterTrainer(router_model, config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    best_val_acc = 0.0
    num_epochs = config.get('num_epochs', 10)
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} | Val Top-1 Acc: {val_metrics['top1_acc']:.4f}")
        
        # Save best model
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            trainer.save_checkpoint(
                str(output_dir / "best_router_model.pt"),
                epoch,
                val_metrics
            )
            logger.info(f"✓ New best model saved! Val Acc: {best_val_acc:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("ROUTER TRAINING COMPLETE!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_router_model.pt'}")
    logger.info("="*80)
    
    return router_model

def replace_gates_with_router(model, router_model_path: str):
    """
    Replace simple gates in your MoE model with trained router
    
    Usage:
        trained_router = torch.load('router_training/best_router_model.pt')
        replace_gates_with_router(your_model, trained_router)
    """
    # Load trained router
    checkpoint = torch.load(router_model_path)
    router_model = SmallRouterModel(**checkpoint['config'])
    router_model.load_state_dict(checkpoint['model_state_dict'])
    router_model.eval()
    
    logger.info("Replacing gates with trained router model...")
    
    # Replace gates in all MoE layers
    num_replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and ('moe' in name.lower() or 'expert' in name.lower()):
            # Replace the gate
            module.gate = router_model
            num_replaced += 1
            logger.info(f"Replaced gate in: {name}")
    
    logger.info(f"✓ Replaced {num_replaced} gates with trained router")
    
    return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    How to use this system in your Main.py
    """
    
    # Configuration for router training
    router_config = {
        # Data collection
        'collection_batches': 1000,  # How many batches to collect routing data from
        
        # Router architecture
        'hidden_size': 768,  # Match your model
        'num_experts': 8,
        'top_k': 2,
        'router_hidden': 256,  # Smaller = fewer parameters
        'router_layers': 4,
        'router_heads': 4,
        
        # Training
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'weight_decay': 0.01,
        
        # Loss weights
        'classification_weight': 1.0,
        'distribution_weight': 0.5,
        'load_balance_weight': 0.01,
    }
    
    # Example workflow:
    """
    # 1. Train router model
    router_model = train_router_model(
        existing_model=your_trained_moe_model,
        train_dataloader=your_dataloader,
        config=router_config,
        output_dir="router_training"
    )
    
    # 2. Replace gates with trained router
    your_model = replace_gates_with_router(
        your_model,
        router_model_path="router_training/best_router_model.pt"
    )
    
    # 3. Continue training with new router (optional fine-tuning)
    # The router will now make routing decisions instead of simple gates
    """
    
    print("Router training system ready!")
    print("\nTo integrate into your Main.py:")
    print("1. Add router_config to your parameter sections")
    print("2. After initial training, run: train_router_model()")
    print("3. Replace gates: replace_gates_with_router()")
    print("4. Continue training or deploy model")