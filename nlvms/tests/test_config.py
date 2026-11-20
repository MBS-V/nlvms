import sys
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
