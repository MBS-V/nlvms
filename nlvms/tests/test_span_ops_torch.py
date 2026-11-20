import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from utils.span_ops import cw_to_se, se_to_cw, elementwise_iou, pairwise_giou, pairwise_iou

a = torch.tensor([[38.0, 62.0]])
b = torch.tensor([[42.0, 58.0]])
assert abs(pairwise_iou(a, b).item() - 16 / 24) < 1e-5
assert abs(elementwise_iou(a, b).item() - 16 / 24) < 1e-5

d = torch.tensor([[0.0, 10.0]])
e = torch.tensor([[20.0, 30.0]])
assert pairwise_iou(d, e).item() == 0
assert pairwise_giou(d, e).item() < 0

x = torch.tensor([[3.0, 9.0]])
assert torch.allclose(cw_to_se(se_to_cw(x)), x)
print("torch span_ops tests passed")
