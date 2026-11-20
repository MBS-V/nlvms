# nlvms: HM-Net reproduction + SE-TG (sub-event-aware temporal grounding)

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
