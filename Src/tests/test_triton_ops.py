
import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Setup Path
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from Main_Scripts.core.triton_ops import TritonFP8Linear, replace_linear_with_fp8, is_triton_available, triton_fp8_matmul

class TestTritonOps(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        
    def test_instantiation_cpu(self):
        """Test simple instantiation on CPU"""
        layer = TritonFP8Linear(32, 64, bias=True, device='cpu')
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight.shape, (32, 64))
        self.assertIsNotNone(layer.bias)
        
    def test_load_from_linear_cpu(self):
        """Test loading weights from standard Linear (CPU fallback path)"""
        lin = nn.Linear(32, 64)
        layer = TritonFP8Linear(32, 64, bias=True, device='cpu')
        layer.load_from_linear(lin)
        
        # Check scale
        self.assertTrue(layer.weight_scale > 0)
        # Check weights are populated (not all zeros)
        self.assertTrue(layer.weight.float().abs().sum() > 0)
        
    def test_forward_cpu(self):
        """Test forward pass on CPU (should use torch fallback)"""
        layer = TritonFP8Linear(32, 64, bias=True, device='cpu')
        x = torch.randn(10, 32)
        
        # Should work without error
        y = layer(x)
        self.assertEqual(y.shape, (10, 64))
        self.assertFalse(torch.isnan(y).any())
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_cuda_fallback(self):
        """Test forward pass on CUDA (Force fallback if Triton missing, else Kernel)"""
        device = 'cuda'
        layer = TritonFP8Linear(32, 64, bias=True, device=device)
        x = torch.randn(10, 32, device=device).half()
        
        # We need to manually populate weights properly for this test to not output garbage
        # Actually initializing with zeros (default) works for crash test.
        y = layer(x)
        self.assertEqual(y.shape, (10, 64))
        
    def test_replacement_utility(self):
        """Test replace_linear_with_fp8"""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        replace_linear_with_fp8(model)
        
        # Replacement should happen universally now (with CPU fallback)
        self.assertIsInstance(model[0], TritonFP8Linear)
        self.assertIsInstance(model[2], TritonFP8Linear)
        # Check connectivity
        x = torch.randn(2, 16)
        y = model(x)
        self.assertEqual(y.shape, (2, 16))

if __name__ == '__main__':
    unittest.main()
