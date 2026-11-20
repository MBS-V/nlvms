import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
from utils.checkpoint import CheckpointManager

model = nn.Linear(4, 2)
opt = torch.optim.AdamW(model.parameters())
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)

with tempfile.TemporaryDirectory() as d:
    cm = CheckpointManager(d, keep_last=2)
    for ep in range(4):
        cm.save(ep, model, opt, sched, None, best_metric=float(ep), is_best=(ep == 3))
    assert sorted(p.name for p in Path(d).glob("epoch_*.pt")) == ["epoch_002.pt", "epoch_003.pt"]
    assert (Path(d) / "best.pt").exists()

    r = cm.load_latest(model, opt, sched)
    assert r["start_epoch"] == 4 and r["best_metric"] == 3.0

    (Path(d) / "epoch_003.pt").write_bytes(b"garbage")   # simulate a bad sync
    r = cm.load_latest(model, opt, sched)
    assert r["start_epoch"] == 3, r                       # fell back to epoch 2
print("checkpoint tests passed")
