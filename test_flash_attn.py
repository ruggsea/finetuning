import torch
from flash_attn import flash_attn_func
import flash_attn

print("Flash Attention version:", flash_attn.__version__)

# Create random tensors
q = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 32, 64, device='cuda', dtype=torch.float16)

# Try running flash attention
try:
    out = flash_attn_func(q, k, v)
    print("Flash Attention test successful!")
except Exception as e:
    print("Flash Attention test failed:", str(e)) 