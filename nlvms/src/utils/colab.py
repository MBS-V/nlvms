import json
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
