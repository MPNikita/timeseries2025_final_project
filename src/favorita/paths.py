from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Legacy output locations kept for backwards compatibility.
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CACHE_DIR = PROJECT_ROOT / ".cache" / "favorita"

for _dir in [
    DATA_DIR,
    SUBMISSIONS_DIR,
    CHECKPOINTS_DIR,
    CACHE_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
