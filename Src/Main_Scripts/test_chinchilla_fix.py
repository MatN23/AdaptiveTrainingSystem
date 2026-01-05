import torch
import numpy as np
from collections import deque
import sys
import os

# Add the project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Src/Main_Scripts')))

# Mock torch if needed? No, torch is likely installed.
# If torch is not installed, the test will fail, but that's fine.

from training.chinchilla_scaler import EnhancedChinchillaScaler, _to_scalar

def test_to_scalar():
    print("Testing _to_scalar...")
    
    # Test with int
    assert _to_scalar(1) == 1
    
    # Test with float
    assert _to_scalar(1.5) == 1.5
    
    # Test with tensor on CPU
    t_cpu = torch.tensor(2.5)
    assert _to_scalar(t_cpu) == 2.5
    assert isinstance(_to_scalar(t_cpu), float)
    
    # Test with None
    assert _to_scalar(None) == 0.0
    
    print("✅ _to_scalar tests passed")

def test_scaler_with_tensors():
    print("Testing EnhancedChinchillaScaler with tensors...")
    
    # Mock config, model, dataset
    class MockConfig:
        chinchilla_multiplier = 20
        min_auto_epochs = 1
        max_auto_epochs = 50
        enable_loss_landscape = True
        enable_compute_efficiency = True
        enable_adaptive_curriculum = True
        enable_early_stopping = True
        seq_length = 2048
    
    class MockModel:
        def parameters(self):
            return [torch.randn(10, 10)]
    
    class MockDataset:
        def __len__(self):
            return 100
    
    config = MockConfig()
    model = MockModel()
    dataset = MockDataset()
    
    scaler = EnhancedChinchillaScaler(config, model, dataset)
    
    # Simulate update with tensors
    step = torch.tensor(1)
    epoch = torch.tensor(0.1)
    loss = torch.tensor(5.0)
    grad_norm = torch.tensor(1.2)
    lr = torch.tensor(0.001)
    tokens = torch.tensor(1024)
    
    print("Running initial update with tensors...")
    scaler.update_metrics(step, epoch, loss, grad_norm, lr, tokens)
    
    # Check if metrics are stored as scalars
    m = scaler.metrics_history[0]
    assert isinstance(m.loss, float)
    assert isinstance(m.grad_norm, float)
    assert isinstance(m.learning_rate, float)
    assert isinstance(m.tokens_seen, float) or isinstance(m.tokens_seen, int)
    
    # Run another update to trigger efficiency calculation
    print("Running subsequent updates...")
    for i in range(2, 5):
        scaler.update_metrics(
            torch.tensor(i),
            torch.tensor(0.1 * i),
            torch.tensor(5.0 - 0.1 * i),
            torch.tensor(1.2),
            torch.tensor(0.001),
            torch.tensor(1024)
        )
    
    # Check if history deques are clean
    assert all(isinstance(x, float) for x in scaler.compute_tracker.efficiency_history)
    assert all(isinstance(x, float) for x in scaler.convergence_detector.loss_history)
    
    # Trigger get_status_report (which calls np.mean)
    print("Verifying get_status_report...")
    report = scaler.get_status_report()
    print(f"Current loss from report: {report['training']['current_loss']}")
    
    print("✅ EnhancedChinchillaScaler tensor handling tests passed")

if __name__ == "__main__":
    try:
        test_to_scalar()
        test_scaler_with_tensors()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
