"""
Creates the complete nlvms project (HM-Net reproduction + SE-TG extension scaffold).

Usage (from the folder where you want the project to appear):
    python setup_project.py                 # creates ./nlvms, skips files that already exist
    python setup_project.py --force         # overwrite existing files
    python setup_project.py --name myproj   # different folder name

Works on Windows, macOS and Linux. Only needs Python 3.9+ and the standard library.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

FILES = {}

# ---------------------------------------------------------------- root files

FILES[".gitignore"] = r'''__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
*.pt
*.pth
*.npz
*.npy
features/
checkpoints/
logs/
.DS_Store
'''

FILES["requirements.txt"] = r'''# torch comes preinstalled on Colab; do not reinstall it there.
# Locally (CPU is enough for the tests):  pip install torch
einops>=0.7
numpy
pyyaml
omegaconf>=2.3
tqdm
scipy
spacy>=3.7
nltk
'''

FILES["README.md"] = r'''# nlvms: HM-Net reproduction + SE-TG (sub-event-aware temporal grounding)

Video moment retrieval: given an untrimmed video and a sentence, predict [start, end].

## Layout
- `configs/default.yaml`   all hyperparameters
- `src/utils/`             span math, checkpointing, Colab/Drive helpers
- `src/models/`            Mamba block (HMB block, encoders, DETR head, SE-TG to come)
- `scripts/`               runnable scripts (dry-run of checkpoint resume)
- `tests/`                 CPU-runnable checks
- `notebooks/00_colab_runner.ipynb`   the notebook everyone opens in Colab

## Local checks (Windows PowerShell, from the project root)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install torch numpy omegaconf pyyaml
    python tests\check_span_math.py
    python tests\test_span_ops_torch.py
    python tests\smoke_mamba.py
    python tests\test_checkpoint.py

## Workflow
Edit in VS Code -> commit and push to GitHub -> in Colab: git pull, run -> checkpoints go to
the shared Drive folder, so any teammate's account can resume.
'''

FILES[".vscode/settings.json"] = r'''{
  "python.analysis.extraPaths": ["./src"],
  "terminal.integrated.env.windows": { "PYTHONPATH": "${workspaceFolder}\\src" },
  "files.eol": "\n",
  "editor.formatOnSave": false
}
'''

# ------------------------------------------------------------------- config

FILES["configs/default.yaml"] = r'''# HM-Net + SE-TG configuration. Paths assume Drive mounted at /content/drive.

paths:
  drive_root: /content/drive/MyDrive/nlvms
  feature_cache: ${paths.drive_root}/features
  checkpoints: ${paths.drive_root}/checkpoints
  logs: ${paths.drive_root}/logs

data:
  dataset: qvhighlights          # qvhighlights | tacos
  feature_dir: ${paths.feature_cache}   # overridden to a local copy in Colab
  max_video_len: 200
  max_query_len: 32
  clip_stride_sec: 2.0

model:
  hidden_dim: 256
  n_heads: 8
  mamba_state_dim: 32
  enhancement_layers: 2
  cofusion_layers: 2
  num_queries: 10
  positional_bias_threshold: 200
  dropout: 0.1
  droppath: 0.1

subevent:
  enabled: true
  max_subevents: 4
  connectives: [then, after, before, once, while, "and then", next]
  gate_init: 0.0                 # used as tanh(gate); 0 -> identical to HM-Net at init

loss:
  weight_ce: 6.0
  weight_l1: 10.0
  weight_giou: 1.0
  weight_sal: 1.0
  weight_rec: 0.1
  weight_sub: 0.1
  weight_div: 0.1

train:
  batch_size: 8
  grad_accum: 4                  # effective batch 32 on a T4
  lr: 2.0e-4
  warmup_epochs: 5
  epochs: 100
  weight_decay: 1.0e-4
  grad_clip: 1.0
  amp: true                      # fp16 on T4 (no bf16 on Turing)
  save_every: 1

eval:
  iou_thresholds: [0.3, 0.5, 0.7]
  map_thresholds: [0.5, 0.75]
'''

# --------------------------------------------------------------- python pkgs

for pkg in ("src", "src/data", "src/models", "src/losses", "src/engine", "src/utils"):
    FILES[f"{pkg}/__init__.py"] = ""

FILES["src/utils/span_ops.py"] = r'''"""
Conversions and overlap measures for 1D temporal spans.

Spans appear in two forms:
  (center, width)  - what the regression head predicts
  (start, end)     - what losses and metrics compare against ground truth
"""

import torch


def cw_to_se(spans):
    center, width = spans.unbind(-1)
    return torch.stack([center - 0.5 * width, center + 0.5 * width], dim=-1)


def se_to_cw(spans):
    start, end = spans.unbind(-1)
    return torch.stack([0.5 * (start + end), end - start], dim=-1)


def _inter_union(spans_a, spans_b):
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    inter_start = torch.max(start_a[:, None], start_b[None, :])
    inter_end = torch.min(end_a[:, None], end_b[None, :])
    inter = (inter_end - inter_start).clamp(min=0)
    len_a = (end_a - start_a).clamp(min=0)
    len_b = (end_b - start_b).clamp(min=0)
    union = len_a[:, None] + len_b[None, :] - inter
    return inter, union


def pairwise_iou(spans_a, spans_b):
    """(N,2) start/end, (M,2) start/end -> (N,M) IoU."""
    inter, union = _inter_union(spans_a, spans_b)
    return inter / union.clamp(min=1e-6)


def pairwise_giou(spans_a, spans_b):
    """Generalized IoU for intervals. (N,2), (M,2) -> (N,M) in [-1, 1]."""
    inter, union = _inter_union(spans_a, spans_b)
    iou = inter / union.clamp(min=1e-6)
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    enclose = (torch.max(end_a[:, None], end_b[None, :])
               - torch.min(start_a[:, None], start_b[None, :])).clamp(min=1e-6)
    return iou - (enclose - union) / enclose


def elementwise_iou(spans_a, spans_b):
    """Row-by-row IoU. (N,2), (N,2) -> (N,)."""
    start_a, end_a = spans_a.unbind(-1)
    start_b, end_b = spans_b.unbind(-1)
    inter = (torch.min(end_a, end_b) - torch.max(start_a, start_b)).clamp(min=0)
    union = (end_a - start_a).clamp(min=0) + (end_b - start_b).clamp(min=0) - inter
    return inter / union.clamp(min=1e-6)
'''

FILES["src/models/mamba_block.py"] = r'''"""
Selective state-space (Mamba) layer with bidirectional wrapping.

Two paths behind one interface, forward(x): (B, L, D) -> (B, L, D):
  - official CUDA kernel (mamba_ssm.Mamba), used when installed and a GPU is present
  - a plain PyTorch selective scan, used otherwise (slow, same math)

The fallback runs in fp32 even under autocast: the exp(delta*A) terms are
numerically touchy in fp16, and the T4 has no bf16.

Build the model after confirming the GPU is available, since the path is
chosen when each layer is constructed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _CudaMamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False


class SelectiveScanFallback(nn.Module):
    def __init__(self, dim, state_dim=32, conv_kernel=4, expand=2):
        super().__init__()
        self.state_dim = state_dim
        self.inner_dim = expand * dim

        self.in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size=conv_kernel,
            groups=self.inner_dim, padding=conv_kernel - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, self.inner_dim + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x):
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self._scan(x.float())
        return out.to(in_dtype)

    def _scan(self, x):
        batch, length, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_in = F.silu(x_in)

        delta, B, C = torch.split(
            self.x_proj(x_in), [self.inner_dim, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(delta))            # (B, L, D)

        A = -torch.exp(self.A_log)                         # (D, N)
        delta_A = torch.exp(delta.unsqueeze(-1) * A)       # (B, L, D, N)
        delta_Bx = delta.unsqueeze(-1) * B.unsqueeze(2) * x_in.unsqueeze(-1)

        h = x_in.new_zeros(batch, self.inner_dim, self.state_dim)
        ys = []
        for t in range(length):
            h = delta_A[:, t] * h + delta_Bx[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)

        y = (y + x_in * self.D) * F.silu(z)
        return self.out_proj(y)


def make_mamba_layer(dim, state_dim=32):
    if MAMBA_SSM_AVAILABLE and torch.cuda.is_available():
        return _CudaMamba(d_model=dim, d_state=state_dim)
    return SelectiveScanFallback(dim, state_dim=state_dim)


class BidirectionalMamba(nn.Module):
    """
    Two independently weighted Mamba layers, one reading forward and one
    reading the reversed sequence, summed. Separate weights because
    forward-time and backward-time dynamics are different tasks.
    """

    def __init__(self, dim, state_dim=32):
        super().__init__()
        self.forward_mamba = make_mamba_layer(dim, state_dim)
        self.backward_mamba = make_mamba_layer(dim, state_dim)

    def forward(self, x, padding_mask=None):
        """
        x: (B, L, D); padding_mask: (B, L) bool, True where real.
        Padding is zeroed before scanning. In the reversed pass the padding
        becomes leading zeros, which the scan treats as near-empty input.
        """
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)

        fwd = self.forward_mamba(x)
        bwd = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return fwd + bwd
'''

FILES["src/utils/checkpoint.py"] = r'''import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def _atomic_save(obj, path):
    """Write to a temp file, then rename, so a cut-off session never leaves a half file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


def _get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
        except Exception:
            pass


class CheckpointManager:
    """
    One file per epoch (epoch_XXX.pt), newest `keep_last` retained, plus a
    weights-only best.pt. Loading walks back from the newest file and skips
    unreadable ones, so a half-synced Drive file never blocks a resume.
    """

    def __init__(self, ckpt_dir, keep_last=2):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, epoch, model, optimizer=None, scheduler=None, scaler=None,
             best_metric=None, extra=None, is_best=False):
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
            "best_metric": best_metric,
            "extra": extra or {},
            "rng": _get_rng_state(),
        }
        path = self.dir / f"epoch_{epoch:03d}.pt"
        _atomic_save(state, path)

        pointer = self.dir / "latest.json"
        tmp = pointer.with_name("latest.json.tmp")
        tmp.write_text(json.dumps({"file": path.name, "epoch": epoch}))
        os.replace(tmp, pointer)

        if is_best:
            _atomic_save({"epoch": epoch, "model": state["model"],
                          "best_metric": best_metric}, self.dir / "best.pt")
        self._prune()
        return path

    def _prune(self):
        files = sorted(self.dir.glob("epoch_*.pt"))
        for old in files[:-self.keep_last]:
            old.unlink(missing_ok=True)

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None,
                    restore_rng=True):
        """Returns {'start_epoch', 'best_metric', 'extra'}, or None if nothing usable."""
        for path in sorted(self.dir.glob("epoch_*.pt"), reverse=True):
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"])
            except Exception as e:
                print(f"[ckpt] skipping unreadable {path.name}: {e}")
                continue
            if optimizer is not None and state.get("optimizer"):
                optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            if restore_rng and state.get("rng"):
                _set_rng_state(state["rng"])
            print(f"[ckpt] resumed from {path.name} (epoch {state['epoch']})")
            return {"start_epoch": state["epoch"] + 1,
                    "best_metric": state.get("best_metric"),
                    "extra": state.get("extra", {})}
        return None
'''

FILES["src/utils/colab.py"] = r'''import json
import time
from pathlib import Path

from omegaconf import OmegaConf


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def flush_drive():
    """Run before switching accounts so pending writes reach Drive."""
    if in_colab():
        from google.colab import drive
        drive.flush_and_unmount()


def load_config(path="configs/default.yaml", drive_root=None):
    cfg = OmegaConf.load(path)
    if drive_root:
        cfg.paths.drive_root = drive_root   # dependent paths update via interpolation
    for key in ("feature_cache", "checkpoints", "logs"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)
    return cfg


class RunLock:
    """
    Advisory lock kept in Drive so two accounts don't train into the same run.
    Considered stale after `stale_after` seconds without a heartbeat. Drive sync
    lag means this is a guard, not a guarantee.
    """

    def __init__(self, path, who, stale_after=1800):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.who = who
        self.stale_after = stale_after

    def _read(self):
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return None

    def _write(self):
        self.path.write_text(json.dumps({"who": self.who, "ts": time.time()}))

    def acquire(self, force=False):
        cur = self._read()
        if (cur and cur["who"] != self.who and not force
                and time.time() - cur["ts"] < self.stale_after):
            age = int((time.time() - cur["ts"]) / 60)
            raise RuntimeError(
                f"Run is locked by '{cur['who']}' (last heartbeat {age} min ago). "
                "Wait for them to release it, or pass force=True.")
        self._write()

    def heartbeat(self):
        self._write()

    def release(self):
        cur = self._read()
        if cur and cur["who"] == self.who:
            self.path.unlink(missing_ok=True)
'''

# ------------------------------------------------------------------ scripts

FILES["scripts/dryrun_resume.py"] = r'''"""Toy training run that checks save/resume works across sessions and accounts."""
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
'''

# -------------------------------------------------------------------- tests

FILES["tests/check_span_math.py"] = r'''"""Pure-Python check of the IoU/GIoU formulas. No dependencies."""


def iou_np(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0.0, a[1] - a[0]) + max(0.0, b[1] - b[0]) - inter
    return inter / max(union, 1e-6)


def giou_np(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0.0, a[1] - a[0]) + max(0.0, b[1] - b[0]) - inter
    enclose = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-6) - (enclose - union) / enclose


cases = [
    ((42, 58), (42, 58), 1.0, "perfect match"),
    ((38, 62), (42, 58), 16 / 24, "overlap 16, union 24"),
    ((0, 10), (20, 30), 0.0, "disjoint"),
    ((0, 10), (5, 15), 5 / 15, "half overlap"),
]
for pred, gt, expected, label in cases:
    got = iou_np(pred, gt)
    assert abs(got - expected) < 1e-6, label
    print(f"[OK] {label}: {got:.4f}")

assert giou_np((0, 10), (20, 30)) < 0
assert giou_np((0, 10), (5, 15)) <= iou_np((0, 10), (5, 15)) + 1e-9
print("all span math checks passed")
'''

FILES["tests/test_span_ops_torch.py"] = r'''import sys
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
'''

FILES["tests/smoke_mamba.py"] = r'''import sys
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
'''

FILES["tests/test_checkpoint.py"] = r'''import sys
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
'''

FILES["tests/test_config.py"] = r'''import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.colab import load_config

with tempfile.TemporaryDirectory() as d:
    cfg = load_config(str(ROOT / "configs" / "default.yaml"), drive_root=d)
    assert cfg.paths.checkpoints == f"{d}/checkpoints", cfg.paths.checkpoints
    assert cfg.data.feature_dir == f"{d}/features"
    assert cfg.subevent.gate_init == 0.0
    assert cfg.loss.weight_sub == 0.1 and cfg.loss.weight_div == 0.1
    assert Path(cfg.paths.checkpoints).is_dir()
print("config tests passed")
'''


# ----------------------------------------------------------------- notebook

def _cell(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def _md(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


NB_CELLS = [
    _md("# nlvms Colab runner\nRun top to bottom. Every cell is safe to re-run. "
        "Set `WHO`, `RUN` and `GITHUB_REPO` in the config cell."),
    _cell(r'''
# 1. GPU check (Runtime > Change runtime type > T4 GPU)
!nvidia-smi
import torch
print(torch.__version__, "cuda:", torch.cuda.is_available())
'''),
    _cell(r'''
# 2. Mount Drive and get the code. Needs a Colab Secret named GH_TOKEN
#    (key icon in the left sidebar, enable notebook access).
from google.colab import drive, userdata
drive.mount('/content/drive')

import os
GITHUB_REPO = "your-username/nlvms"          # <- edit
token = userdata.get('GH_TOKEN')
if not os.path.exists('/content/nlvms'):
    !git clone https://{token}@github.com/{GITHUB_REPO}.git /content/nlvms
%cd /content/nlvms
!git pull
'''),
    _cell(r'''
# 3. Dependencies (do not reinstall torch)
!pip install -q einops omegaconf spacy nltk
!python -m spacy download en_core_web_sm
'''),
    _cell(r'''
# 4. OPTIONAL fast Mamba kernel. Skip on day one: the PyTorch fallback works.
#    Can take ~40 min and may fail on version mismatch. Do it in its own session,
#    then restart the runtime and rerun cells 1-3.
# !pip install -q causal-conv1d mamba-ssm --no-build-isolation
'''),
    _cell(r'''
# 5. Identity, config, lock, checkpoints
import sys
sys.path.insert(0, 'src')
from utils.colab import load_config, RunLock, flush_drive
from utils.checkpoint import CheckpointManager

WHO = "diksha"          # <- each person sets their own name
RUN = "qvh_baseline"    # <- one folder per experiment

cfg = load_config("configs/default.yaml", drive_root="/content/drive/MyDrive/nlvms")
lock = RunLock(f"{cfg.paths.drive_root}/locks/{RUN}.json", who=WHO)
lock.acquire()
ckpt = CheckpointManager(f"{cfg.paths.checkpoints}/{RUN}", keep_last=2)
print(cfg.paths)
'''),
    _cell(r'''
# 6. Copy features from Drive to local disk (Drive reads are slow; later runs are quick)
!mkdir -p /content/features
!rsync -a /content/drive/MyDrive/nlvms/features/ /content/features/
cfg.data.feature_dir = "/content/features"
'''),
    _cell(r'''
# 7. Environment smoke tests on the GPU
!python tests/smoke_mamba.py
!python tests/test_checkpoint.py
!python tests/test_config.py
'''),
    _md("### Cross-account resume dry-run (toy model)\n"
        "Account A runs the first command, then cell 9. Account B runs cells 1-3 and 5, then the second command."),
    _cell(r'''
# 8a. Account A: trains epochs 0-2, then stops
!python scripts/dryrun_resume.py --who diksha
'''),
    _cell(r'''
# 8b. Account B (change --who): should print "resumed from epoch_002.pt" and finish epochs 3-5
# !python scripts/dryrun_resume.py --who bhavagna
'''),
    _cell(r'''
# 9. HAND OFF: run before stopping. Releases the lock and flushes Drive writes.
lock.release()
flush_drive()
'''),
]

NOTEBOOK = {
    "cells": NB_CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


# --------------------------------------------------------------------- main

def write(root, rel, content, force):
    path = root / rel
    if path.exists() and not force:
        print(f"  skip (exists)  {rel}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"  wrote          {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nlvms")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path.cwd() / args.name
    root.mkdir(parents=True, exist_ok=True)
    print(f"Creating project in {root}")

    for rel, content in FILES.items():
        write(root, rel, content, args.force)
    write(root, "notebooks/00_colab_runner.ipynb",
          json.dumps(NOTEBOOK, indent=1) + "\n", args.force)

    if shutil.which("git") and not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root)
        print("  git init done")

    print("\nDone. Next:")
    print(f"  cd {args.name}")
    print("  python -m venv .venv ; .venv\\Scripts\\Activate.ps1   (Windows PowerShell)")
    print("  pip install torch numpy omegaconf pyyaml")
    print("  python tests\\check_span_math.py")
    print("  python tests\\test_config.py")
    print("  python tests\\test_span_ops_torch.py")
    print("  python tests\\smoke_mamba.py")
    print("  python tests\\test_checkpoint.py")


if __name__ == "__main__":
    main()
