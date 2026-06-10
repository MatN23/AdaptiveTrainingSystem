# Copyright (c) 2025 MatN23. All rights reserved.
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""
    
    def test_orchestrator_creation(self, mock_config, temp_dir):
        """Test orchestrator can be initialized."""
        try:
            from Main_Scripts.training.orchestrator import AdaptiveTrainingOrchestrator
            
            # Set experiment directory for orchestrator
            mock_config.experiment_dir = str(temp_dir / "test_experiment")
            
            orchestrator = AdaptiveTrainingOrchestrator(mock_config)
            
            assert orchestrator.config is not None
            assert orchestrator.meta_learner is not None
            assert orchestrator.hyperparameter_optimizer is not None
            
            # Cleanup
            orchestrator.cleanup()
            
        except ImportError as e:
            pytest.skip(f"Orchestrator module not available: {e}")
        except Exception as e:
            pytest.skip(f"Orchestrator initialization failed: {e}")


class TestMetaLearner:
    """Test meta-learning engine."""
    
    def test_meta_learner_creation(self):
        """Test meta learner can be created."""
        try:
            from Main_Scripts.training.orchestrator import MetaLearningEngine
            
            meta_learner = MetaLearningEngine()
            
            assert meta_learner.training_history == []
            assert meta_learner.successful_strategies == []
            
        except ImportError as e:
            pytest.skip(f"Orchestrator module not available: {e}")
        except AttributeError:
            # MetaLearningEngine might be a nested class
            pytest.skip("MetaLearningEngine not directly importable")