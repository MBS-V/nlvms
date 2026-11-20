"""Toy training run that checks save/resume works across sessions and accounts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from utils.checkpoint import CheckpointManager
from utils.colab import RunLock, load_config

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--who", required=True)
p.add_argument("--run", default="dryrun")
p.add_argument("--drive_root", default="/content/drive/MyDrive/nlvms")
p.add_argument("--total_epochs", type=int, default=6)
p.add_argument("--max_epochs_this_session", type=int, default=3)
a = p.parse_args()

cfg = load_config(str(ROOT / "configs" / "default.yaml"), drive_root=a.drive_root)
lock = RunLock(f"{cfg.paths.drive_root}/locks/{a.run}.json", a.who)
lock.acquire()
ckpt = CheckpointManager(f"{cfg.paths.checkpoints}/{a.run}")

torch.manual_seed(0)
model = nn.Linear(8, 1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.total_epochs)
x = torch.randn(256, 8)
y = x.sum(1, keepdim=True)

resume = ckpt.load_latest(model, opt, sched)
start = resume["start_epoch"] if resume else 0

done = 0
for epoch in range(start, a.total_epochs):
    if done >= a.max_epochs_this_session:
        print("simulated session end")
        break
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    ckpt.save(epoch, model, opt, sched, None, best_metric=-loss.item())
    print(f"[{a.who}] epoch {epoch}  loss {loss.item():.4f}")
    done += 1

lock.release()
print("finished")
