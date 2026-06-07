import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    
    # Test basic operations
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    z = torch.matmul(x, y)
    
    print(" Basic tensor operations work")
    
    # Test FP16
    x_fp16 = x.to(torch.float16)
    y_fp16 = y.to(torch.float16)
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    
    print(" FP16 operations work")
    
    print(f"\nMPS device ready for training!")
else:
    print(" MPS not available on this system")