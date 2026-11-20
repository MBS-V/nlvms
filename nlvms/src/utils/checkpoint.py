import json
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
