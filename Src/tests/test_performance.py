# Copyright (c) 2025 MatN23. All rights reserved.
import pytest
import torch
import time
import gc
import sys
from pathlib import Path

# Add Src directory to Python path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


class TestForwardPassPerformance:
    """Test forward pass performance."""
    
    def test_forward_speed(self, small_model):
        """Test model forward pass speed."""
        small_model.eval()
        
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = small_model(input_ids)
        
        # Benchmark
        start = time.time()
        num_iterations = 10
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = small_model(input_ids)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_iterations
        
        print(f"\\nForward pass average time: {avg_time*1000:.2f}ms")
        
        assert avg_time < 2.0, "Forward pass too slow"
    
    def test_backward_speed(self, small_model):
        """Test backward pass speed."""
        small_model.train()
        
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        
        # Warmup
        for _ in range(3):
            output = small_model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 1000),
                labels.view(-1)
            )
            loss.backward()
            small_model.zero_grad()
        
        # Benchmark
        start = time.time()
        num_iterations = 10
        
        for _ in range(num_iterations):
            output = small_model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 1000),
                labels.view(-1)
            )
            loss.backward()
            small_model.zero_grad()
        
        elapsed = time.time() - start
        avg_time = elapsed / num_iterations
        
        print(f"\\nBackward pass average time: {avg_time*1000:.2f}ms")
        
        assert avg_time < 5.0, "Backward pass too slow"


class TestMemoryUsage:
    """Test memory usage."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory(self, small_model):
        """Test GPU memory usage."""
        device = torch.device('cuda')
        small_model = small_model.to(device)
        
        torch.cuda.reset_peak_memory_stats()
        
        input_ids = torch.randint(0, 1000, (4, 32)).to(device)
        
        output = small_model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        
        print(f"\\nPeak GPU memory: {peak_memory:.2f} MB")
        
        # Cleanup
        del small_model, input_ids, output
        torch.cuda.empty_cache()
    
    def test_memory_leak(self, small_model):
        """Test for memory leaks."""
        initial_objects = len(gc.get_objects())
        
        for _ in range(10):
            input_ids = torch.randint(0, 1000, (2, 10))
            output = small_model(input_ids)
            del input_ids, output
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Allow some growth but not excessive
        assert final_objects - initial_objects < 1000, "Possible memory leak detected"