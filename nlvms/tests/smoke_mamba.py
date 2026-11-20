import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from models.mamba_block import BidirectionalMamba, MAMBA_SSM_AVAILABLE

print("mamba_ssm kernel available:", MAMBA_SSM_AVAILABLE, "| cuda:", torch.cuda.is_available())
dev = "cuda" if torch.cuda.is_available() else "cpu"

layer = BidirectionalMamba(dim=256, state_dim=32).to(dev)
x = torch.randn(2, 20, 256, device=dev)
mask = torch.ones(2, 20, dtype=torch.bool, device=dev)
mask[1, 15:] = False

out = layer(x, padding_mask=mask)
assert out.shape == x.shape, out.shape
assert torch.isfinite(out).all()

out.sum().backward()
assert all(p.grad is not None for p in layer.parameters())

if dev == "cuda":
    with torch.autocast("cuda", dtype=torch.float16):
        out16 = layer(x)
    assert torch.isfinite(out16).all()
print("BidirectionalMamba smoke test passed:", tuple(out.shape))
