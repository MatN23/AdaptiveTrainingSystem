# Licensed under the Custom License below.

import time
import logging
import json
import math
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingAlert:
    """Represents a training alert."""
    timestamp: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metric: str
    value: float
    threshold: Optional[float] = None
    recommendation: Optional[str] = None


class MetricsCollector:
    """Collects and analyzes training metrics in real-time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.alerts = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Thresholds for alerts
        self.thresholds = {
            'loss_spike_factor': 2.0,  # Alert if loss > 2x recent average
            'grad_norm_spike_factor': 5.0,  # Alert if grad norm > 5x recent average
            'memory_usage_threshold': 0.95,  # Alert if GPU memory > 95%
            'nan_threshold': 1,  # Alert immediately on NaN
            'stagnation_steps': 50,  # Alert if no improvement for N steps
            'learning_rate_min': 1e-8,  # Alert if LR becomes too small
        }
        
        self.last_improvement_step = 0
        self.best_loss = float('inf')
    
    def add_metric(self, name: str, value: float, step: int):
        """Add a metric value."""
        if math.isnan(value) or math.isinf(value):
            self._create_alert('error', f'Invalid {name}: {value}', name, value)
            return
        
        self.metrics_history[name].append({
            'value': value,
            'step': step,
            'timestamp': time.time()
        })
        
        self._check_metric_alerts(name, value, step)
    
    def _check_metric_alerts(self, name: str, value: float, step: int):
        """Check if a metric value should trigger alerts."""
        history = self.metrics_history[name]
        
        if len(history) < 2:
            return
        
        # Loss-specific alerts
        if 'loss' in name.lower():
            self._check_loss_alerts(name, value, step, history)
        
        # Gradient norm alerts
        elif 'grad_norm' in name.lower():
            self._check_gradient_alerts(name, value, step, history)
        
        # Learning rate alerts
        elif 'learning_rate' in name.lower() or 'lr' in name.lower():
            self._check_learning_rate_alerts(name, value, step)
        
        # Memory alerts
        elif 'memory' in name.lower():
            self._check_memory_alerts(name, value, step)
    
    def _check_loss_alerts(self, name: str, value: float, step: int, history: deque):
        """Check loss-specific alerts."""
        if len(history) < 10:
            return
        
        recent_values = [h['value'] for h in list(history)[-10:]]
        recent_avg = statistics.mean(recent_values)
        
        # Loss spike detection
        if value > recent_avg * self.thresholds['loss_spike_factor']:
            self._create_alert(
                'warning',
                f'Loss spike detected: {value:.6f} (recent avg: {recent_avg:.6f})',
                name, value, recent_avg * self.thresholds['loss_spike_factor'],
                'Consider reducing learning rate or checking data quality'
            )
        
        # Loss improvement tracking
        if value < self.best_loss:
            self.best_loss = value
            self.last_improvement_step = step
        elif step - self.last_improvement_step > self.thresholds['stagnation_steps']:
            self._create_alert(
                'info',
                f'No loss improvement for {step - self.last_improvement_step} steps',
                name, value,
                recommendation='Consider adjusting learning rate or early stopping'
            )
    
    def _check_gradient_alerts(self, name: str, value: float, step: int, history: deque):
        """Check gradient norm alerts."""
        if len(history) < 5:
            return
        
        recent_values = [h['value'] for h in list(history)[-5:]]
        recent_avg = statistics.mean(recent_values)
        
        # Gradient explosion
        if value > recent_avg * self.thresholds['grad_norm_spike_factor']:
            self._create_alert(
                'error',
                f'Gradient explosion: {value:.4f} (recent avg: {recent_avg:.4f})',
                name, value, recent_avg * self.thresholds['grad_norm_spike_factor'],
                'Consider gradient clipping or reducing learning rate'
            )
        
        # Vanishing gradients
        elif value < 1e-6 and recent_avg > 1e-4:
            self._create_alert(
                'warning',
                f'Vanishing gradients: {value:.2e}',
                name, value,
                recommendation='Check model architecture and initialization'
            )
    
    def _check_learning_rate_alerts(self, name: str, value: float, step: int):
        """Check learning rate alerts."""
        if value < self.thresholds['learning_rate_min']:
            self._create_alert(
                'warning',
                f'Learning rate very small: {value:.2e}',
                name, value, self.thresholds['learning_rate_min'],
                'Training may be ineffective with very small learning rate'
            )
    
    def _check_memory_alerts(self, name: str, value: float, step: int):
        """Check memory usage alerts."""
        if 'percent' in name.lower():
            threshold = self.thresholds['memory_usage_threshold']
        else:
            # For absolute memory values, need context about total memory
            return
        
        if value > threshold:
            severity = 'critical' if value > 0.98 else 'warning'
            self._create_alert(
                severity,
                f'High memory usage: {value:.1%}',
                name, value, threshold,
                'Consider reducing batch size or enabling gradient checkpointing'
            )
    
    def _create_alert(self, severity: str, message: str, metric: str, value: float,
                     threshold: Optional[float] = None, recommendation: Optional[str] = None):
        """Create and log an alert."""
        alert = TrainingAlert(
            timestamp=time.time(),
            severity=severity,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            recommendation=recommendation
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.INFO)
        
        log_message = f"TRAINING ALERT [{severity.upper()}]: {message}"
        if recommendation:
            log_message += f" | Recommendation: {recommendation}"
        
        logging.log(log_level, log_message)
    
    def get_recent_alerts(self, minutes: int = 5) -> List[TrainingAlert]:
        """Get alerts from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        history = self.metrics_history[metric_name]
        if not history:
            return {}
        
        values = [h['value'] for h in history]
        
        return {
            'count': len(values),
            'latest': values[-1] if values else None,
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'trend': self._calculate_trend(values),
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Use linear regression slope
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if abs(slope) < 1e-6:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def get_health_score(self) -> float:
        """Calculate overall training health score (0-1)."""
        score = 1.0
        
        # Recent alerts penalty
        recent_alerts = self.get_recent_alerts(10)  # Last 10 minutes
        critical_alerts = sum(1 for a in recent_alerts if a.severity == 'critical')
        error_alerts = sum(1 for a in recent_alerts if a.severity == 'error')
        warning_alerts = sum(1 for a in recent_alerts if a.severity == 'warning')
        
        score -= critical_alerts * 0.3
        score -= error_alerts * 0.2
        score -= warning_alerts * 0.1
        
        # Loss trend penalty
        loss_summary = self.get_metric_summary('loss')
        if loss_summary.get('trend') == 'increasing':
            score -= 0.2
        
        # Gradient health
        grad_summary = self.get_metric_summary('grad_norm')
        if grad_summary:
            if grad_summary.get('latest', 0) < 1e-6:  # Vanishing gradients
                score -= 0.3
            elif grad_summary.get('latest', 0) > 100:  # Exploding gradients
                score -= 0.4
        
        return max(0.0, min(1.0, score))


class TrainingHealthMonitor:
    """Monitors training health with comprehensive analysis and recommendations."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.metrics_collector = MetricsCollector()
        self.log_dir = Path(log_dir) if log_dir else Path("logs/health")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_start = time.time()
        self.step_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Training phase detection
        self.training_phases = []
        self.current_phase = None
        
        # Performance benchmarks
        self.benchmarks = {
            'min_throughput_tokens_per_sec': 100,
            'max_acceptable_loss_variance': 2.0,
            'target_gradient_norm_range': (1e-4, 10.0),
            'memory_efficiency_threshold': 0.8
        }
    
    def log_step(self, metrics: Dict[str, Any]):
        """Log a training step with comprehensive analysis."""
        step = metrics.get('global_step', 0)
        timestamp = time.time()
        
        # Core metrics
        if 'loss' in metrics:
            self.metrics_collector.add_metric('loss', metrics['loss'], step)
        
        if 'learning_rate' in metrics:
            self.metrics_collector.add_metric('learning_rate', metrics['learning_rate'], step)
        elif 'lr' in metrics:
            self.metrics_collector.add_metric('learning_rate', metrics['lr'], step)
        
        if 'grad_norm' in metrics:
            self.metrics_collector.add_metric('grad_norm', metrics['grad_norm'], step)
        
        if 'tokens_per_sec' in metrics:
            self.metrics_collector.add_metric('throughput', metrics['tokens_per_sec'], step)
            self.throughput_history.append(metrics['tokens_per_sec'])
        
        # Memory metrics
        for key, value in metrics.items():
            if 'memory' in key.lower():
                self.metrics_collector.add_metric(key, value, step)
        
        # Step timing
        if len(self.step_times) > 0:
            step_duration = timestamp - self.step_times[-1] if self.step_times else 0
            self.metrics_collector.add_metric('step_duration', step_duration, step)
        
        self.step_times.append(timestamp)
        
        # Phase detection
        self._update_training_phase(metrics, step)
        
        # Periodic health checks
        if step % 50 == 0:
            self._perform_health_check(step)
    
    def _update_training_phase(self, metrics: Dict[str, Any], step: int):
        """Detect and update current training phase."""
        loss = metrics.get('loss', 0)
        
        # Simple phase detection based on loss behavior
        if len(self.metrics_collector.metrics_history['loss']) > 20:
            recent_losses = [h['value'] for h in list(self.metrics_collector.metrics_history['loss'])[-20:]]
            loss_variance = statistics.stdev(recent_losses) if len(recent_losses) > 1 else 0
            
            phase = 'unknown'
            if loss > 2.0:
                phase = 'initialization'
            elif loss_variance > 0.5:
                phase = 'volatile'
            elif loss_variance < 0.1:
                phase = 'converging'
            else:
                phase = 'training'
            
            if phase != self.current_phase:
                if self.current_phase:
                    self.training_phases.append({
                        'phase': self.current_phase,
                        'end_step': step,
                        'duration_steps': step - self.training_phases[-1].get('start_step', 0) if self.training_phases else step
                    })
                
                self.training_phases.append({
                    'phase': phase,
                    'start_step': step,
                    'start_time': time.time()
                })
                
                self.current_phase = phase
                logging.info(f"Training phase changed to: {phase} at step {step}")
    
    def _perform_health_check(self, step: int):
        """Perform comprehensive health check."""
        health_score = self.metrics_collector.get_health_score()
        
        # Log health status
        if health_score < 0.3:
            logging.error(f"Training health critical: {health_score:.2f} at step {step}")
            self._generate_emergency_recommendations()
        elif health_score < 0.6:
            logging.warning(f"Training health poor: {health_score:.2f} at step {step}")
            self._generate_improvement_recommendations()
        elif health_score > 0.9:
            logging.info(f"Training health excellent: {health_score:.2f} at step {step}")
        
        # Performance analysis
        self._analyze_performance()
    
    def _generate_emergency_recommendations(self):
        """Generate emergency recommendations for critical training health."""
        recent_alerts = self.metrics_collector.get_recent_alerts(5)
        critical_issues = [a for a in recent_alerts if a.severity in ['critical', 'error']]
        
        recommendations = []
        
        for alert in critical_issues:
            if 'memory' in alert.metric.lower():
                recommendations.append("Immediately reduce batch size or enable gradient checkpointing")
            elif 'gradient' in alert.metric.lower():
                recommendations.append("Apply gradient clipping or reduce learning rate")
            elif 'loss' in alert.metric.lower():
                recommendations.append("Check data quality and consider resetting to earlier checkpoint")
        
        if not recommendations:
            recommendations = ["Consider stopping training and reviewing configuration"]
        
        logging.critical("EMERGENCY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logging.critical(f"  {i}. {rec}")
    
    def _generate_improvement_recommendations(self):
        """Generate recommendations for improving training."""
        loss_summary = self.metrics_collector.get_metric_summary('loss')
        throughput_avg = statistics.mean(self.throughput_history) if self.throughput_history else 0
        
        recommendations = []
        
        if loss_summary.get('trend') == 'stable' and len(self.throughput_history) > 10:
            recommendations.append("Loss appears stable - consider increasing learning rate")
        
        if throughput_avg < self.benchmarks['min_throughput_tokens_per_sec']:
            recommendations.append("Low throughput detected - consider optimizing batch size or model compilation")
        
        if recommendations:
            logging.info("IMPROVEMENT RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logging.info(f"  {i}. {rec}")
    
    def _analyze_performance(self):
        """Analyze training performance against benchmarks."""
        if not self.throughput_history:
            return
        
        avg_throughput = statistics.mean(self.throughput_history)
        
        performance_metrics = {
            'average_throughput': avg_throughput,
            'throughput_efficiency': avg_throughput / self.benchmarks['min_throughput_tokens_per_sec'],
            'health_score': self.metrics_collector.get_health_score(),
            'current_phase': self.current_phase,
            'session_duration_hours': (time.time() - self.session_start) / 3600
        }
        
        # Log performance summary periodically
        logging.debug(f"Performance metrics: {performance_metrics}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        health_score = self.metrics_collector.get_health_score()
        recent_alerts = self.metrics_collector.get_recent_alerts(30)  # Last 30 minutes
        
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1
        
        loss_summary = self.metrics_collector.get_metric_summary('loss')
        throughput_avg = statistics.mean(self.throughput_history) if self.throughput_history else 0
        
        return {
            'overall_health_score': health_score,
            'health_status': self._get_health_status(health_score),
            'current_phase': self.current_phase,
            'recent_alerts': dict(alert_counts),
            'loss_trend': loss_summary.get('trend', 'unknown'),
            'avg_loss': loss_summary.get('mean', None),
            'latest_loss': loss_summary.get('latest', None),
            'avg_throughput': throughput_avg,
            'session_duration_hours': (time.time() - self.session_start) / 3600,
            'total_training_phases': len(self.training_phases)
        }
    
    def _get_health_status(self, score: float) -> str:
        """Convert health score to status string."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        elif score >= 0.3:
            return 'poor'
        else:
            return 'critical'
    
    def save_health_report(self, filename: Optional[str] = None):
        """Save comprehensive health report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"
        
        report_path = self.log_dir / filename
        
        # Comprehensive report
        report = {
            'report_timestamp': time.time(),
            'session_start': self.session_start,
            'health_summary': self.get_health_summary(),
            'training_phases': self.training_phases,
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp,
                    'severity': alert.severity,
                    'message': alert.message,
                    'metric': alert.metric,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'recommendation': alert.recommendation
                }
                for alert in self.metrics_collector.get_recent_alerts(60)
            ],
            'metric_summaries': {
                name: self.metrics_collector.get_metric_summary(name)
                for name in self.metrics_collector.metrics_history.keys()
            },
            'performance_analysis': {
                'average_throughput': statistics.mean(self.throughput_history) if self.throughput_history else 0,
                'throughput_samples': len(self.throughput_history),
                'step_samples': len(self.step_times),
                'benchmarks_met': self._check_benchmarks()
            }
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Health report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save health report: {e}")
    
    def _check_benchmarks(self) -> Dict[str, bool]:
        """Check if training meets performance benchmarks."""
        results = {}
        
        if self.throughput_history:
            avg_throughput = statistics.mean(self.throughput_history)
            results['min_throughput'] = avg_throughput >= self.benchmarks['min_throughput_tokens_per_sec']
        
        loss_summary = self.metrics_collector.get_metric_summary('loss')
        if loss_summary:
            results['loss_improving'] = loss_summary.get('trend') == 'decreasing'
        
        grad_summary = self.metrics_collector.get_metric_summary('grad_norm')
        if grad_summary and grad_summary.get('latest'):
            grad_norm = grad_summary['latest']
            min_grad, max_grad = self.benchmarks['target_gradient_norm_range']
            results['gradient_norm_healthy'] = min_grad <= grad_norm <= max_grad
        
        return results
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get detailed training diagnostics for troubleshooting."""
        return {
            'health_score': self.metrics_collector.get_health_score(),
            'active_alerts': len(self.metrics_collector.get_recent_alerts(5)),
            'training_stability': self._assess_stability(),
            'performance_efficiency': self._assess_efficiency(),
            'resource_utilization': self._assess_resource_usage(),
            'recommendations': self._get_diagnostic_recommendations()
        }
    
    def _assess_stability(self) -> Dict[str, Any]:
        """Assess training stability."""
        loss_summary = self.metrics_collector.get_metric_summary('loss')
        grad_summary = self.metrics_collector.get_metric_summary('grad_norm')
        
        stability_score = 1.0
        issues = []
        
        if loss_summary.get('trend') == 'increasing':
            stability_score -= 0.3
            issues.append("Loss trending upward")
        
        if loss_summary.get('std', 0) > loss_summary.get('mean', 1):
            stability_score -= 0.2
            issues.append("High loss variance")
        
        if grad_summary.get('latest', 0) > 100:
            stability_score -= 0.4
            issues.append("High gradient norms")
        
        return {
            'score': max(0, stability_score),
            'issues': issues,
            'status': 'stable' if stability_score > 0.7 else 'unstable'
        }
    
    def _assess_efficiency(self) -> Dict[str, Any]:
        """Assess training efficiency."""
        avg_throughput = statistics.mean(self.throughput_history) if self.throughput_history else 0
        target_throughput = self.benchmarks['min_throughput_tokens_per_sec']
        
        efficiency_ratio = avg_throughput / target_throughput if target_throughput > 0 else 0
        
        return {
            'throughput_ratio': efficiency_ratio,
            'average_throughput': avg_throughput,
            'target_throughput': target_throughput,
            'efficiency_status': 'excellent' if efficiency_ratio > 2 else 'good' if efficiency_ratio > 1 else 'poor'
        }
    
    def _assess_resource_usage(self) -> Dict[str, Any]:
        """Assess resource utilization efficiency."""
        memory_metrics = [name for name in self.metrics_collector.metrics_history.keys() if 'memory' in name.lower()]
        
        resource_status = {}
        for metric in memory_metrics:
            summary = self.metrics_collector.get_metric_summary(metric)
            if summary.get('latest'):
                resource_status[metric] = {
                    'current': summary['latest'],
                    'max': summary['max'],
                    'trend': summary['trend']
                }
        
        return resource_status
    
    def _get_diagnostic_recommendations(self) -> List[str]:
        """Get diagnostic-based recommendations."""
        recommendations = []
        
        stability = self._assess_stability()
        efficiency = self._assess_efficiency()
        
        if stability['score'] < 0.5:
            recommendations.extend([
                "Training appears unstable - consider reducing learning rate",
                "Check for data quality issues or model architecture problems"
            ])
        
        if efficiency['efficiency_status'] == 'poor':
            recommendations.append("Low training efficiency - consider optimizing batch size or hardware utilization")
        
        recent_critical = len([a for a in self.metrics_collector.get_recent_alerts(10) if a.severity == 'critical'])
        if recent_critical > 0:
            recommendations.append("Critical alerts detected - immediate intervention recommended")
        
        return recommendations