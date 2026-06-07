# Licensed under the Custom License below.

import math
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import torch
import torch.nn.functional as F
from core.dataset import create_dataloader


def train(self, train_dataset, eval_dataset=None):
    """Main training loop with comprehensive monitoring."""
    logging.info("="*80)
    logging.info("STARTING PRODUCTION TRAINING")
    logging.info("="*80)
    
    # Store eval dataset for periodic evaluation
    self.eval_dataset = eval_dataset
    
    train_dataloader = create_dataloader(train_dataset, self.config, shuffle=True)
    
    # Calculate total steps and setup scheduler
    total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
    self._setup_scheduler(total_steps)
    
    # Log training configuration
    self._log_training_config(len(train_dataloader), total_steps)
    
    training_start_time = time.time()
    
    try:
        for epoch in range(self.current_epoch, self.config.num_epochs):
            if hasattr(self, 'should_stop') and self.should_stop:
                break
            
            logging.info(f"\n{'='*60}")
            logging.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            logging.info(f"{'='*60}")
            
            epoch_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Full evaluation at epoch end
            if eval_dataset is not None:
                eval_metrics = self.evaluate(eval_dataset)
                epoch_metrics.update(eval_metrics)
                
                logging.info(f"Epoch {epoch + 1} Summary:")
                logging.info(f"  Train Loss: {epoch_metrics['avg_loss']:.6f}")
                logging.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")
                logging.info(f"  Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                
                # Early stopping check
                if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience:
                    self._check_early_stopping(eval_metrics['eval_loss'])
            
            # Log epoch metrics
            if hasattr(self, 'logger'):
                try:
                    self.logger.log_metrics(epoch_metrics, epoch, "epoch")
                except Exception as e:
                    logging.warning(f"Failed to log epoch metrics: {e}")
            
            if hasattr(self, 'checkpoint_manager'):
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.global_step, epoch + 1, self.metrics,
                    f"epoch_{epoch + 1:03d}"
                )
            
            self.current_epoch = epoch + 1
            
            # Backup checkpoint periodically
            current_time = time.time()
            if hasattr(self.config, 'backup_every_n_hours') and (current_time - getattr(self, 'last_backup_time', 0)) > self.config.backup_every_n_hours * 3600:
                self._create_backup()
                self.last_backup_time = current_time
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    finally:
        total_training_time = time.time() - training_start_time
        logging.info(f"\nTraining finished after {total_training_time / 3600:.2f} hours")
        
        if hasattr(self, 'checkpoint_manager'):
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "final"
            )
        
        self._save_training_summary(total_training_time)


def train_epoch(self, train_dataloader, epoch: int):
    """Train one epoch with comprehensive monitoring."""
    self.model.train()
    
    epoch_metrics = {
        'total_loss': 0.0,
        'total_raw_loss': 0.0,
        'total_tokens': 0,
        'num_batches': 0,
        'grad_norm_sum': 0.0
    }
    
    accumulation_metrics = {
        'loss': 0.0,
        'raw_loss': 0.0,
        'tokens': 0
    }
    
    epoch_start_time = time.time()
    last_log_time = time.time()
    
    for batch_idx, batch in enumerate(train_dataloader):
        if hasattr(self, 'should_stop') and self.should_stop:
            break
        
        step_start_time = time.time()
        
        # Training step
        step_metrics = self.train_step(batch)
        
        # Skip if loss is invalid
        if step_metrics['loss'] == 0.0 or math.isnan(step_metrics['loss']) or math.isinf(step_metrics['loss']):
            logging.warning(f"Skipping batch {batch_idx} due to invalid loss: {step_metrics['loss']}")
            continue
        
        # Accumulate metrics
        accumulation_metrics['loss'] += step_metrics['loss']
        accumulation_metrics['raw_loss'] += step_metrics['raw_loss']
        accumulation_metrics['tokens'] += step_metrics['valid_tokens']
        
        # Optimizer step after accumulation
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            opt_metrics = self.optimizer_step()
            self.global_step += 1
            
            # Update epoch metrics
            if accumulation_metrics['loss'] > 0:
                epoch_metrics['total_loss'] += accumulation_metrics['loss']
                epoch_metrics['total_raw_loss'] += accumulation_metrics['raw_loss']
                epoch_metrics['total_tokens'] += accumulation_metrics['tokens']
                epoch_metrics['num_batches'] += 1
                epoch_metrics['grad_norm_sum'] += opt_metrics['grad_norm']
            
            # Log metrics
            step_time = time.time() - step_start_time
            tokens_per_sec = accumulation_metrics['tokens'] / step_time if step_time > 0 else 0
            
            # Store metrics with safety checks
            if not math.isnan(accumulation_metrics['loss']) and not math.isinf(accumulation_metrics['loss']):
                self.metrics['train_losses'].append(accumulation_metrics['loss'])
            
            if not math.isnan(opt_metrics['lr']) and not math.isinf(opt_metrics['lr']):
                self.metrics['learning_rates'].append(opt_metrics['lr'])
            
            if not math.isnan(opt_metrics['grad_norm']) and not math.isinf(opt_metrics['grad_norm']):
                self.metrics['gradient_norms'].append(opt_metrics['grad_norm'])
            
            if not math.isnan(tokens_per_sec) and not math.isinf(tokens_per_sec):
                self.metrics['throughput'].append(tokens_per_sec)
            
            # Health monitoring
            if hasattr(self, 'health_monitor'):
                self.health_monitor.update(accumulation_metrics['loss'], opt_metrics['grad_norm'])
            
            # Periodic logging
            current_time = time.time()
            if self.global_step % 50 == 0 or current_time - last_log_time > 30:
                self._log_training_step(
                    epoch, batch_idx, len(train_dataloader),
                    accumulation_metrics, opt_metrics, tokens_per_sec
                )
                last_log_time = current_time
            
            # Log to monitoring backends
            if hasattr(self, 'logger') and self.global_step % 10 == 0:
                try:
                    self.logger.log_metrics({
                        'train_loss': accumulation_metrics['loss'],
                        'learning_rate': opt_metrics['lr'],
                        'gradient_norm': opt_metrics['grad_norm'],
                        'throughput_tokens_per_sec': tokens_per_sec,
                        'perplexity': math.exp(min(accumulation_metrics['raw_loss'], 10))
                    }, self.global_step, "train")
                except Exception as e:
                    logging.debug(f"Failed to log training metrics: {e}")
            
            # System monitoring
            if hasattr(self.config, 'health_check_interval') and self.global_step % self.config.health_check_interval == 0:
                if hasattr(self, 'logger') and hasattr(self.logger, 'log_system_stats'):
                    try:
                        self.logger.log_system_stats(self.global_step)
                    except Exception as e:
                        logging.debug(f"Failed to log system stats: {e}")
                self._log_memory_usage(f"Step {self.global_step}")
            
            # Periodic evaluation
            if (hasattr(self.config, 'eval_every_n_batches') and 
                self.config.eval_every_n_batches > 0 and 
                self.global_step % self.config.eval_every_n_batches == 0):
                self._periodic_evaluation()
            
            # Periodic checkpointing
            if (hasattr(self.config, 'save_every_n_batches') and 
                self.config.save_every_n_batches > 0 and 
                self.global_step % self.config.save_every_n_batches == 0):
                if hasattr(self, 'checkpoint_manager'):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.current_epoch, self.metrics,
                        f"step_{self.global_step:06d}"
                    )
            
            # Reset accumulation metrics
            accumulation_metrics = {'loss': 0.0, 'raw_loss': 0.0, 'tokens': 0}
    
    # Handle remaining gradients
    if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
        opt_metrics = self.optimizer_step()
        self.global_step += 1
    
    # Compute epoch statistics
    epoch_time = time.time() - epoch_start_time
    self.metrics['epoch_times'].append(epoch_time)
    
    if epoch_metrics['num_batches'] > 0:
        avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
        avg_raw_loss = epoch_metrics['total_raw_loss'] / epoch_metrics['num_batches']
        avg_grad_norm = epoch_metrics['grad_norm_sum'] / epoch_metrics['num_batches']
        avg_tokens_per_sec = epoch_metrics['total_tokens'] / epoch_time
    else:
        avg_loss = float('inf')
        avg_raw_loss = float('inf')
        avg_grad_norm = 0.0
        avg_tokens_per_sec = 0.0
    
    logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Avg Grad Norm: {avg_grad_norm:.4f} | "
                f"Throughput: {avg_tokens_per_sec:.0f} tokens/s")
    
    return {
        'avg_loss': avg_loss,
        'avg_raw_loss': avg_raw_loss,
        'avg_grad_norm': avg_grad_norm,
        'epoch_time': epoch_time,
        'throughput': avg_tokens_per_sec
    }


def _log_training_step(self, epoch: int, batch_idx: int, total_batches: int,
                      metrics, opt_metrics, tokens_per_sec: float):
    """Log training step with comprehensive information."""
    memory_info = ""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        memory_info = f" | GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
    
    # Health status
    health_info = ""
    if hasattr(self, 'health_monitor'):
        health_status = self.health_monitor.get_status()
        health_info = f" | Health: {health_status}"
    
    # Calculate perplexity with safety check
    try:
        perplexity = math.exp(min(metrics['raw_loss'], 10))
    except (ValueError, OverflowError):
        perplexity = float('inf')
    
    logging.info(
        f"Epoch {epoch+1} | Step {self.global_step:6d} | "
        f"Batch {batch_idx+1:4d}/{total_batches} | "
        f"Loss: {metrics['loss']:.6f} | "
        f"PPL: {perplexity:.2f} | "
        f"LR: {opt_metrics['lr']:.2e} | "
        f"GradNorm: {opt_metrics['grad_norm']:.4f} | "
        f"Tokens/s: {tokens_per_sec:.0f}"
        f"{memory_info}{health_info}"
    )


def _periodic_evaluation(self):
    """Perform periodic evaluation during training."""
    if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
        eval_metrics = self.evaluate(self.eval_dataset, max_batches=50)
        
        # Log evaluation metrics
        if hasattr(self, 'logger'):
            try:
                self.logger.log_metrics(eval_metrics, self.global_step, "eval")
            except Exception as e:
                logging.debug(f"Failed to log eval metrics: {e}")
        
        logging.info(f"Eval | Step {self.global_step} | "
                    f"Loss: {eval_metrics['eval_loss']:.6f} | "
                    f"PPL: {eval_metrics['eval_perplexity']:.2f}")
        
        # Early stopping check
        if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience:
            self._check_early_stopping(eval_metrics['eval_loss'])


def _check_early_stopping(self, eval_loss: float):
    """Check early stopping condition."""
    if eval_loss < self.best_eval_loss:
        self.best_eval_loss = eval_loss
        self.patience_counter = 0
        if hasattr(self, 'checkpoint_manager'):
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                "best_model"
            )
    else:
        self.patience_counter += 1
        
    if hasattr(self.config, 'early_stopping_patience') and self.patience_counter >= self.config.early_stopping_patience:
        logging.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
        self.should_stop = True


def _log_training_config(self, batches_per_epoch: int, total_steps: int):
    """Log comprehensive training configuration."""
    try:
        model_params = self._count_parameters()
    except:
        model_params = "Unknown"
    
    config_info = [
        f"Model Parameters: {model_params:,}" if isinstance(model_params, int) else f"Model Parameters: {model_params}",
        f"Epochs: {self.config.num_epochs}",
        f"Batches per epoch: {batches_per_epoch:,}",
        f"Total steps: {total_steps:,}",
        f"Effective batch size: {getattr(self.config, 'effective_batch_size', self.config.batch_size * self.config.gradient_accumulation_steps)}",
        f"Learning rate: {self.config.learning_rate:.2e}",
        f"Weight decay: {getattr(self.config, 'weight_decay', 0.01)}",
        f"Warmup ratio: {getattr(self.config, 'warmup_ratio', 0.1)}",
        f"Max grad norm: {getattr(self.config, 'max_grad_norm', 1.0)}",
        f"Precision: {getattr(self.config, 'precision', 'fp32')}",
        f"Device: {self.device}"
    ]
    
    logging.info("Training Configuration:")
    for info in config_info:
        logging.info(f"  {info}")


def _create_backup(self):
    """Create backup of current training state."""
    backup_dir = Path("backups") / getattr(self.config, 'experiment_name', 'default')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if hasattr(self, 'checkpoint_manager'):
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.global_step, self.current_epoch, self.metrics,
                f"backup_{timestamp}"
            )
            logging.info(f"Backup created at step {self.global_step}")
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")


def _save_training_summary(self, total_time: float):
    """Save comprehensive training summary."""
    try:
        # Try to convert config to dict
        try:
            model_config = asdict(self.config)
        except:
            # Fallback to manual conversion
            model_config = {
                attr: getattr(self.config, attr) 
                for attr in dir(self.config) 
                if not attr.startswith('_') and not callable(getattr(self.config, attr))
            }
        
        summary = {
            'experiment_name': getattr(self.config, 'experiment_name', 'unknown'),
            'total_training_time_hours': total_time / 3600,
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'final_metrics': {
                'best_eval_loss': self.best_eval_loss,
                'final_train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                'avg_throughput': sum(self.metrics['throughput'])/len(self.metrics['throughput']) if self.metrics['throughput'] else 0
            },
            'model_config': model_config
        }
        
        if hasattr(self, 'health_monitor'):
            summary['health_summary'] = self.health_monitor.get_summary()
        
        summary_path = Path(f"experiments/{getattr(self.config, 'experiment_name', 'default')}/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"Training summary saved: {summary_path}")
    
    except Exception as e:
        logging.error(f"Failed to save training summary: {e}")


# Patch the methods to the trainer class
try:
    from training.trainer import EnhancedConversationTrainer
    
    EnhancedConversationTrainer.train = train
    EnhancedConversationTrainer.train_epoch = train_epoch
    EnhancedConversationTrainer._log_training_step = _log_training_step
    EnhancedConversationTrainer._periodic_evaluation = _periodic_evaluation
    EnhancedConversationTrainer._check_early_stopping = _check_early_stopping
    EnhancedConversationTrainer._log_training_config = _log_training_config
    EnhancedConversationTrainer._create_backup = _create_backup
    EnhancedConversationTrainer._save_training_summary = _save_training_summary
    
    logging.info("Training loop methods patched successfully")
except ImportError as e:
    logging.error(f"Could not patch training loop methods: {e}")