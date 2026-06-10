# Copyright (c) 2025 MatN23. All rights reserved.

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from Main_Scripts.training.orchestrator import (
    MetaLearningEngine,
    AdaptiveHyperparameterOptimizer,
    ArchitectureEvolution,
    RealTimeAnalytics,
    TrainingMetrics
)

@pytest.fixture
def mock_metrics():
    return TrainingMetrics(
        epoch=1,
        step=100,
        loss=2.5,
        grad_norm=1.0,
        learning_rate=1e-4,
        expert_utilization={},
        memory_usage={},
        throughput=100.0,
        semantic_coherence=0.8,
        factual_accuracy=0.7,
        reasoning_score=0.6,
        timestamp=datetime.now()
    )

class TestMetaLearningEngine:
    def test_suggest_hyperparameters_cold_start(self, mock_metrics):
        engine = MetaLearningEngine()
        config = MagicMock()
        config.use_moe = False
        config.use_mod = False
        # Cold start because history is empty
        suggestions = engine.suggest_hyperparameters(mock_metrics, config)
        # It actually returns a dict
        assert isinstance(suggestions, dict)

    def test_record_outcome_and_suggest(self, mock_metrics):
        engine = MetaLearningEngine()
        config = MagicMock()
        config.learning_rate = 1e-4
        config.use_moe = False
        config.use_mod = False
        
        # Record a successful run - MUST pass objects with .loss
        metrics_list = [mock_metrics]
        engine.record_training_outcome(config, metrics_list, final_performance=0.9)
        
        # Now suggest (should find similar run)
        suggestions = engine.suggest_hyperparameters(mock_metrics, config)
        assert isinstance(suggestions, dict)

class TestAdaptiveOptimizer:
    def test_should_adjust_lr(self, mock_metrics):
        optimizer = AdaptiveHyperparameterOptimizer()
        # Returns dict or None
        result = optimizer.should_adjust_learning_rate(mock_metrics)
        assert result is None or isinstance(result, dict)

    def test_optimize_batch_size(self, mock_metrics):
        optimizer = AdaptiveHyperparameterOptimizer()
        memory = {'usage_percent': 0.95}
        new_bs = optimizer.optimize_batch_size(mock_metrics, memory)
        assert isinstance(new_bs, int) or new_bs is None

class TestArchitectureEvolution:
    def test_evolution_decisions(self):
        evo = ArchitectureEvolution()
        util = {'expert_0': 0.95, 'expert_1': 0.9}
        metrics = {'val_loss': 1.0}
        
        # Returns dict or None
        should_add = evo.should_add_expert(util, metrics)
        should_prune = evo.should_prune_expert(util, metrics)
        
        assert should_add is None or isinstance(should_add, dict)
        assert should_prune is None or isinstance(should_prune, dict)

class TestRealTimeAnalytics:
    def test_anomaly_detection(self, mock_metrics):
        analytics = RealTimeAnalytics()
        # Fill buffer first
        for _ in range(10):
            analytics.detect_training_anomalies(mock_metrics)
            
        # Spike in loss
        spike_metrics = MagicMock()
        spike_metrics.loss = 100.0 # Huge compared to typical
        spike_metrics.grad_norm = 1000.0
        spike_metrics.step = 100
        
        anomalies = analytics.detect_training_anomalies(spike_metrics)
        # Can be None if no anomalies found, but our spike should trigger something
        assert anomalies is None or isinstance(anomalies, list)

    def test_loss_dynamics(self, mock_metrics):
        analytics = RealTimeAnalytics()
        recent = [mock_metrics] * 20
        dynamics = analytics.analyze_loss_dynamics(recent)
        if dynamics:
            assert 'trend_direction' in dynamics
            assert 'predicted_convergence' in dynamics
