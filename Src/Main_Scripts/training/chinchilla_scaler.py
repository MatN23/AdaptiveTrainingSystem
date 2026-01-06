# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

"""
training/chinchilla_scaler.py

Enhanced Chinchilla Scaling with proven research techniques.
"""

import numpy as np
import torch
import math
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ScalingMetrics:
    """Comprehensive metrics for scaling decisions."""
    step: int
    epoch: float
    loss: float
    grad_norm: float
    learning_rate: float
    tokens_seen: int
    timestamp: float
    loss_reduction_rate: float = 0.0
    grad_variance: float = 0.0
    compute_efficiency: float = 0.0
    convergence_score: float = 0.0


class ConvergenceDetector:
    """Detects training convergence using multiple signals."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        
    def update(self, loss: float, grad_norm: float):
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
    
    def detect_plateau(self, threshold: float = 0.001) -> Tuple[bool, float]:
        if len(self.loss_history) < self.window_size:
            return False, 1.0
        
        recent = list(self.loss_history)[-self.window_size//2:]
        older = list(self.loss_history)[:self.window_size//2]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        improvement_rate = (older_mean - recent_mean) / max(older_mean, 1e-8)
        is_plateau = abs(improvement_rate) < threshold
        
        return is_plateau, improvement_rate
    
    def detect_divergence(self, threshold: float = 0.1) -> Tuple[bool, float]:
        if len(self.loss_history) < 10:
            return False, 0.0
        
        recent = list(self.loss_history)[-10:]
        older = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else recent
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        divergence_rate = (recent_mean - older_mean) / max(older_mean, 1e-8)
        is_diverging = divergence_rate > threshold
        
        return is_diverging, divergence_rate
    
    def compute_gradient_variance(self) -> float:
        if len(self.grad_norm_history) < 10:
            return 1.0
        
        return float(np.std(list(self.grad_norm_history)[-20:]))
    
    def compute_convergence_score(self) -> float:
        if len(self.loss_history) < self.window_size:
            return 0.0
        
        loss_std = np.std(list(self.loss_history)[-20:])
        loss_mean = np.mean(list(self.loss_history)[-20:])
        loss_stability = 1.0 - min(loss_std / max(loss_mean, 1e-8), 1.0)
        
        _, improvement_rate = self.detect_plateau()
        improvement_score = 1.0 - min(abs(improvement_rate) * 100, 1.0)
        
        grad_var = self.compute_gradient_variance()
        grad_stability = 1.0 / (1.0 + grad_var)
        
        convergence = (
            0.4 * loss_stability +
            0.4 * improvement_score +
            0.2 * grad_stability
        )
        
        return convergence


class ComputeEfficiencyTracker:
    """Tracks compute efficiency (loss reduction per FLOP)."""
    
    def __init__(self, model_flops_per_token: Optional[float] = None):
        self.model_flops = model_flops_per_token
        self.metrics = []
        self.efficiency_history = deque(maxlen=100)
        
    def estimate_flops_per_token(self, model_params: int, seq_length: int) -> float:
        return 6 * model_params * seq_length
    
    def update(self, tokens_processed: int, loss_reduction: float):
        if self.model_flops is None:
            return
        
        flops_used = tokens_processed * self.model_flops
        efficiency = loss_reduction / max(flops_used, 1e-20)
        
        self.efficiency_history.append(efficiency)
        self.metrics.append({
            'tokens': tokens_processed,
            'loss_reduction': loss_reduction,
            'efficiency': efficiency
        })
    
    def get_current_efficiency(self) -> float:
        if not self.efficiency_history:
            return 1.0
        return float(np.mean(list(self.efficiency_history)[-20:]))
    
    def is_efficiency_declining(self, threshold: float = 0.5) -> Tuple[bool, float]:
        if len(self.efficiency_history) < 50:
            return False, 0.0
        
        recent = list(self.efficiency_history)[-25:]
        older = list(self.efficiency_history)[-50:-25]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        decline_ratio = (older_mean - recent_mean) / max(older_mean, 1e-8)
        is_declining = decline_ratio > threshold
        
        return is_declining, decline_ratio


class AdaptiveCurriculumManager:
    """Manages adaptive curriculum learning."""
    
    def __init__(self):
        self.difficulty_schedule = []
        self.learning_velocity = deque(maxlen=50)
        
    def update_learning_velocity(self, loss_reduction: float):
        self.learning_velocity.append(loss_reduction)
    
    def get_recommended_difficulty(self) -> float:
        if len(self.learning_velocity) < 10:
            return 0.3
        
        recent_velocity = np.mean(list(self.learning_velocity)[-10:])
        
        if recent_velocity > 0.01:
            return min(0.9, 0.5 + recent_velocity * 20)
        else:
            return max(0.2, 0.5 - abs(recent_velocity) * 10)


class EnhancedChinchillaScaler:
    """Advanced Chinchilla scaling with multiple proven techniques."""
    
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        
        self.chinchilla_multiplier = getattr(config, 'chinchilla_multiplier', 20)
        self.min_epochs = getattr(config, 'min_auto_epochs', 1)
        self.max_epochs = getattr(config, 'max_auto_epochs', 50)
        
        self.enable_loss_landscape = getattr(config, 'enable_loss_landscape', True)
        self.enable_compute_efficiency = getattr(config, 'enable_compute_efficiency', True)
        self.enable_adaptive_curriculum = getattr(config, 'enable_adaptive_curriculum', True)
        self.enable_early_stopping = getattr(config, 'enable_early_stopping', True)
        
        self.plateau_patience = getattr(config, 'plateau_patience', 5)
        self.efficiency_threshold = getattr(config, 'efficiency_decline_threshold', 0.3)
        self.convergence_threshold = getattr(config, 'convergence_threshold', 0.85)
        
        self.convergence_detector = ConvergenceDetector(window_size=50)
        self.compute_tracker = ComputeEfficiencyTracker()
        self.curriculum_manager = AdaptiveCurriculumManager()
        
        self.metrics_history: List[ScalingMetrics] = []
        self.initial_loss = None
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.tokens_processed = 0
        
        self.model_params = sum(p.numel() for p in model.parameters())
        self.model_params_billions = self.model_params / 1e9
        
        self.dataset_tokens = self._estimate_dataset_tokens()
        self.dataset_tokens_billions = self.dataset_tokens / 1e9
        
        seq_length = getattr(config, 'seq_length', 2048)
        self.compute_tracker.model_flops = self.compute_tracker.estimate_flops_per_token(
            self.model_params, seq_length
        )
        
        self.base_optimal_epochs = self._compute_base_chinchilla_epochs()
        self.current_optimal_epochs = self.base_optimal_epochs
        
        self._print_initialization()
    
    def _estimate_dataset_tokens(self) -> int:
        try:
            if hasattr(self.dataset, 'total_tokens'):
                return self.dataset.total_tokens
            elif hasattr(self.dataset, '__len__'):
                seq_length = getattr(self.config, 'seq_length', 2048)
                return len(self.dataset) * seq_length
            else:
                return 1_000_000_000
        except Exception as e:
            print(f"Warning: Could not estimate dataset tokens: {e}")
            return 1_000_000_000
    
    def _compute_base_chinchilla_epochs(self) -> int:
        optimal_tokens = self.chinchilla_multiplier * self.model_params
        tokens_per_epoch = self.dataset_tokens
        optimal_epochs = math.ceil(optimal_tokens / tokens_per_epoch)
        optimal_epochs = max(self.min_epochs, min(optimal_epochs, self.max_epochs))
        return optimal_epochs
    
    def _print_initialization(self):
        print("\n" + "="*80)
        print("🧠 ENHANCED CHINCHILLA SCALER INITIALIZED")
        print("="*80)
        
        print(f"\nModel Configuration:")
        print(f"  Parameters: {self.model_params:,} ({self.model_params_billions:.3f}B)")
        print(f"  FLOPs/token: {self.compute_tracker.model_flops:.2e}")
        
        print(f"\nDataset Configuration:")
        print(f"  Total tokens: {self.dataset_tokens:,} ({self.dataset_tokens_billions:.3f}B)")
        print(f"  Samples: {len(self.dataset):,}")
        
        print(f"\nChinchilla Scaling:")
        print(f"  Multiplier: {self.chinchilla_multiplier}x")
        print(f"  Optimal tokens: {self.chinchilla_multiplier * self.model_params:,}")
        print(f"  Base optimal epochs: {self.base_optimal_epochs}")
        print(f"  Epoch constraints: [{self.min_epochs}, {self.max_epochs}]")
        
        optimal_tokens_billions = (self.chinchilla_multiplier * self.model_params) / 1e9
        coverage = (self.dataset_tokens_billions * self.base_optimal_epochs) / optimal_tokens_billions * 100
        
        print(f"\nToken Budget:")
        print(f"  Chinchilla target: {optimal_tokens_billions:.3f}B tokens")
        print(f"  Training budget: {self.dataset_tokens_billions * self.base_optimal_epochs:.3f}B tokens")
        print(f"  Coverage: {coverage:.1f}%")
        
        if coverage < 50:
            print(f"  ⚠️  WARNING: Significantly under Chinchilla recommendation")
        elif coverage > 150:
            print(f"  ⚠️  WARNING: Exceeding Chinchilla recommendation")
        else:
            print(f"  ✅ Within reasonable range")
        
        print(f"\nEnhanced Features:")
        print(f"  ✅ Loss Landscape Analysis: {self.enable_loss_landscape}")
        print(f"  ✅ Compute Efficiency Tracking: {self.enable_compute_efficiency}")
        print(f"  ✅ Adaptive Curriculum: {self.enable_adaptive_curriculum}")
        print(f"  ✅ Early Stopping: {self.enable_early_stopping}")
        
        print("="*80 + "\n")
    
    def update_metrics(self, step: int, epoch: float, loss: float, 
                      grad_norm: float, learning_rate: float, batch_tokens: int):
        self.tokens_processed += batch_tokens
        
        if self.initial_loss is None:
            self.initial_loss = loss
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        loss_reduction = 0.0
        if len(self.metrics_history) > 0:
            prev_loss = self.metrics_history[-1].loss
            loss_reduction = prev_loss - loss
        
        if self.enable_loss_landscape:
            self.convergence_detector.update(loss, grad_norm)
        
        if self.enable_compute_efficiency:
            self.compute_tracker.update(batch_tokens, loss_reduction)
        
        if self.enable_adaptive_curriculum:
            self.curriculum_manager.update_learning_velocity(loss_reduction)
        
        grad_variance = self.convergence_detector.compute_gradient_variance()
        compute_efficiency = self.compute_tracker.get_current_efficiency()
        convergence_score = self.convergence_detector.compute_convergence_score()
        
        metrics = ScalingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            tokens_seen=self.tokens_processed,
            timestamp=time.time(),
            loss_reduction_rate=loss_reduction,
            grad_variance=grad_variance,
            compute_efficiency=compute_efficiency,
            convergence_score=convergence_score
        )
        
        self.metrics_history.append(metrics)
        
        if step % 500 == 0:
            self._recompute_optimal_epochs()
    
    def _recompute_optimal_epochs(self):
        if len(self.metrics_history) < 100:
            return
        
        recent_metrics = self.metrics_history[-100:]
        convergence_score = recent_metrics[-1].convergence_score
        efficiency_declining, decline_ratio = self.compute_tracker.is_efficiency_declining()
        is_plateau, improvement_rate = self.convergence_detector.detect_plateau()
        
        adjustment_factor = 1.0
        reasons = []
        
        if convergence_score > self.convergence_threshold:
            adjustment_factor *= 0.9
            reasons.append(f"High convergence ({convergence_score:.2f})")
        
        if efficiency_declining:
            adjustment_factor *= 0.85
            reasons.append(f"Efficiency decline ({decline_ratio:.1%})")
        
        if is_plateau:
            if convergence_score > 0.7:
                adjustment_factor *= 0.8
                reasons.append(f"Converged plateau")
            else:
                adjustment_factor *= 0.95
                reasons.append(f"Training plateau")
        
        if improvement_rate > 0.005 and convergence_score < 0.5:
            adjustment_factor *= 1.05
            reasons.append(f"Strong learning ({improvement_rate:.1%}/step)")
        
        new_optimal = int(self.base_optimal_epochs * adjustment_factor)
        new_optimal = max(self.min_epochs, min(new_optimal, self.max_epochs))
        
        if new_optimal != self.current_optimal_epochs:
            print(f"\n📊 Epoch Adjustment at step {recent_metrics[-1].step}:")
            print(f"   {self.current_optimal_epochs} → {new_optimal} epochs")
            print(f"   Reasons: {', '.join(reasons)}")
            self.current_optimal_epochs = new_optimal
    
    def should_stop_early(self) -> Tuple[bool, Optional[str]]:
        if not self.enable_early_stopping:
            return False, None
        
        if len(self.metrics_history) < 100:
            return False, None
        
        recent = self.metrics_history[-1]
        if recent.loss < 3.0:
        
            if recent.convergence_score > self.convergence_threshold:
                return True, f"Converged (score: {recent.convergence_score:.2f})"
            
            is_declining, decline_ratio = self.compute_tracker.is_efficiency_declining(
                threshold=self.efficiency_threshold
            )
            if is_declining and decline_ratio > 0.5:
                return True, f"Compute efficiency collapsed (decline: {decline_ratio:.1%})"
            
            is_diverging, divergence_rate = self.convergence_detector.detect_divergence()
            if is_diverging:
                return True, f"Training diverged (rate: {divergence_rate:.1%})"
            
            is_plateau, _ = self.convergence_detector.detect_plateau(threshold=0.0001)
            if is_plateau and self.plateau_counter > self.plateau_patience * 100:
                return True, f"Plateau for {self.plateau_counter} steps"
        
        return False, None
    
    def get_optimal_epochs(self) -> int:
        return self.current_optimal_epochs
    
    def get_training_phase(self) -> str:
        if len(self.metrics_history) < 10:
            return 'warmup'
        
        recent = self.metrics_history[-1]
        
        if recent.convergence_score > 0.8:
            return 'convergence'
        elif recent.convergence_score > 0.5:
            return 'main'
        elif recent.epoch > self.current_optimal_epochs * 1.2:
            return 'overtraining'
        else:
            return 'main'
    
    def get_status_report(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {'status': 'No metrics yet'}
        
        recent = self.metrics_history[-1]
        
        report = {
            'current_step': recent.step,
            'current_epoch': recent.epoch,
            'tokens_processed': self.tokens_processed,
            'tokens_processed_billions': self.tokens_processed / 1e9,
        }
        
        chinchilla_optimal_tokens = self.chinchilla_multiplier * self.model_params
        token_progress = (self.tokens_processed / chinchilla_optimal_tokens) * 100
        
        report['chinchilla'] = {
            'optimal_tokens': chinchilla_optimal_tokens,
            'optimal_tokens_billions': chinchilla_optimal_tokens / 1e9,
            'progress': token_progress,
            'base_optimal_epochs': self.base_optimal_epochs,
            'current_optimal_epochs': self.current_optimal_epochs,
        }
        
        report['training'] = {
            'current_loss': recent.loss,
            'best_loss': self.best_loss,
            'loss_reduction_from_start': self.initial_loss - recent.loss if self.initial_loss else 0,
            'convergence_score': recent.convergence_score,
            'training_phase': self.get_training_phase(),
        }
        
        if self.enable_loss_landscape:
            is_plateau, improvement = self.convergence_detector.detect_plateau()
            is_diverging, divergence = self.convergence_detector.detect_divergence()
            
            report['loss_landscape'] = {
                'is_plateau': is_plateau,
                'improvement_rate': improvement,
                'is_diverging': is_diverging,
                'divergence_rate': divergence,
                'gradient_variance': recent.grad_variance,
            }
        
        if self.enable_compute_efficiency:
            is_declining, decline = self.compute_tracker.is_efficiency_declining()
            
            report['compute_efficiency'] = {
                'current_efficiency': recent.compute_efficiency,
                'is_declining': is_declining,
                'decline_ratio': decline,
            }
        
        if self.enable_adaptive_curriculum:
            difficulty = self.curriculum_manager.get_recommended_difficulty()
            
            report['curriculum'] = {
                'recommended_difficulty': difficulty,
            }
        
        should_stop, reason = self.should_stop_early()
        report['early_stopping'] = {
            'should_stop': should_stop,
            'reason': reason,
            'plateau_counter': self.plateau_counter,
        }
        
        return report
    
    def save_state(self, filepath: str):
        state = {
            'config': {
                'chinchilla_multiplier': self.chinchilla_multiplier,
                'min_epochs': self.min_epochs,
                'max_epochs': self.max_epochs,
            },
            'model_info': {
                'params': self.model_params,
                'params_billions': self.model_params_billions,
            },
            'dataset_info': {
                'tokens': self.dataset_tokens,
                'tokens_billions': self.dataset_tokens_billions,
            },
            'training_state': {
                'tokens_processed': self.tokens_processed,
                'initial_loss': self.initial_loss,
                'best_loss': self.best_loss,
                'current_optimal_epochs': self.current_optimal_epochs,
            },
            'metrics': [asdict(m) for m in self.metrics_history[-100:]],
            'timestamp': datetime.now().isoformat(),
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Scaler state saved: {filepath}")
    
    def print_status(self):
        report = self.get_status_report()
        
        print("\n" + "="*80)
        print("📊 ENHANCED CHINCHILLA SCALING STATUS")
        print("="*80)
        
        print(f"\nProgress:")
        print(f"  Step: {report['current_step']}")
        print(f"  Epoch: {report['current_epoch']:.2f}")
        print(f"  Tokens: {report['tokens_processed_billions']:.3f}B / {report['chinchilla']['optimal_tokens_billions']:.3f}B")
        print(f"  Chinchilla progress: {report['chinchilla']['progress']:.1f}%")
        
        print(f"\nEpoch Recommendation:")
        print(f"  Base (Chinchilla): {report['chinchilla']['base_optimal_epochs']}")
        print(f"  Current optimal: {report['chinchilla']['current_optimal_epochs']}")
        
        training = report['training']
        print(f"\nTraining Metrics:")
        print(f"  Current loss: {training['current_loss']:.6f}")
        print(f"  Best loss: {training['best_loss']:.6f}")
        print(f"  Total reduction: {training['loss_reduction_from_start']:.6f}")
        print(f"  Convergence: {training['convergence_score']:.2%}")
        print(f"  Phase: {training['training_phase']}")
        
        if 'loss_landscape' in report:
            ls = report['loss_landscape']
            print(f"\nLoss Landscape:")
            print(f"  Plateau: {'Yes' if ls['is_plateau'] else 'No'} (improvement: {ls['improvement_rate']:.2%}/step)")
            print(f"  Diverging: {'Yes' if ls['is_diverging'] else 'No'}")
            print(f"  Grad variance: {ls['gradient_variance']:.4f}")
        
        if 'compute_efficiency' in report:
            ce = report['compute_efficiency']
            print(f"\nCompute Efficiency:")
            print(f"  Current: {ce['current_efficiency']:.2e}")
            print(f"  Declining: {'Yes' if ce['is_declining'] else 'No'}")
        
        es = report['early_stopping']
        if es['should_stop']:
            print(f"\n⚠️  EARLY STOPPING RECOMMENDED")
            print(f"  Reason: {es['reason']}")
        
        print("="*80 + "\n")