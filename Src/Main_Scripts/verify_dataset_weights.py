
import sys
import os
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from Src.Main_Scripts.core.dataset import FastBaseTrainingDataset, FastStreamingBaseTrainingDataset
from Src.Main_Scripts.training.cuda_kernels import FusedLoss

# Mock tokenizer and config
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.tokenizer = self
    def encode(self, text):
        return [1, 2, 3, 4, 5, 0] # Simple tokens with padding

class MockConfig:
    def __init__(self):
        self.seq_length = 128
        self.data_cache_dir = "data/cache"
        self.streaming_threshold_gb = 10.0

def verify_dataset_weights():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = MockConfig()
    tokenizer = MockTokenizer()
    
    print("="*60)
    print("VERIFYING DATASET WEIGHTS FIX")
    print("="*60)
    
    # Create a dummy file for testing
    dummy_file = "dummy_data.txt"
    with open(dummy_file, "w") as f:
        f.write("This is some dummy text for testing the dataset loader.\n" * 100)
        
    try:
        # 1. Verify FastBaseTrainingDataset
        print("\n1. Testing FastBaseTrainingDataset...")
        dataset = FastBaseTrainingDataset(dummy_file, tokenizer, config)
        
        if len(dataset) > 0:
            sample = dataset[0]
            loss_weights = sample.get('loss_weights')
            
            if loss_weights is None:
                print(" PASS: loss_weights is None")
            else:
                print(f" FAIL: loss_weights is NOT None (Type: {type(loss_weights)})")
        else:
            print(" ADVISORY: Dataset empty, could not verify sample.")

        # 2. Verify FusedLoss selection logic (Simulated)
        print("\n2. simulating FusedLoss behavior...")
        
        # Check if FusedLoss would use CUDA path with these inputs
        fused_loss = FusedLoss() # This might fail to init completely on Mac if no CUDA, but we check logic
        fused_loss.enabled = True # Force enable for simulation
        
        # Mocking the implementation methods to see which one is called
        original_cuda = fused_loss._cuda_implementation
        original_pytorch = fused_loss._pytorch_fallback
        
        called_path = None
        
        def mock_cuda(*args, **kwargs):
            nonlocal called_path
            called_path = "CUDA"
            return {"loss": 0.0}
            
        def mock_pytorch(*args, **kwargs):
            nonlocal called_path
            called_path = "PYTORCH"
            return {"loss": 0.0}
            
        fused_loss._cuda_implementation = mock_cuda
        fused_loss._pytorch_fallback = mock_pytorch
        
        # Test with None weights (our fix)
        logits = torch.randn(1, 10, 100)
        labels = torch.randint(0, 100, (1, 10))
        
        fused_loss(logits, labels, loss_weights=None)
        
        if called_path == "CUDA":
             print(" PASS: FusedLoss calls CUDA implementation when loss_weights=None")
        else:
             print(f" FAIL: FusedLoss called {called_path} instead of CUDA")
             
        # Test with Explicit weights (old behavior)
        weights = torch.ones_like(labels).float()
        fused_loss(logits, labels, loss_weights=weights)
        
        if called_path == "PYTORCH":
             print(" PASS: FusedLoss properly falls back to PyTorch when weights ARE present")
        else:
             print(f" FAIL: FusedLoss called {called_path} but expected PYTORCH fallback")

    except Exception as e:
        print(f"\n FATAL ERROR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            
if __name__ == "__main__":
    verify_dataset_weights()
