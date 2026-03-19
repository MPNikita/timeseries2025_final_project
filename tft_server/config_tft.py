from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_int(value: str) -> int | None:
    token = str(value).strip().lower()
    if token in {"none", "null", ""}:
        return None
    return int(token)


@dataclass
class TFTTrainConfig:
    """Runtime configuration for remote TFT training/inference."""

    project_dir: Path
    data_dir: Path
    artifacts_dir: Path
    mode: str = "final"  # "final" | "fold"
    valid_start: str | None = None

    lookback_days: int = 224
    horizon_days: int = 16
    max_encoder_length: int = 56
    max_prediction_length: int = 16
    max_series: int | None = 30_000
    min_history_points: int = 28
    batch_size: int = 256
    num_workers: int = 2

    hidden_size: int = 32
    attention_head_size: int = 4
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3

    max_epochs: int = 8
    gradient_clip_val: float = 0.1
    early_stopping_patience: int = 3
    random_seed: int = 42

    use_cache: bool = True
    force_cache: bool = False
    force_train: bool = False

    run_name: str | None = None
    resume_ckpt: Path | None = None
    init_state_dict: Path | None = None

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        self.data_dir = Path(self.data_dir).resolve()
        self.artifacts_dir = Path(self.artifacts_dir).resolve()
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"final", "fold"}:
            raise ValueError("`mode` must be either `final` or `fold`.")
        if self.resume_ckpt is not None:
            self.resume_ckpt = Path(self.resume_ckpt).resolve()
        if self.init_state_dict is not None:
            self.init_state_dict = Path(self.init_state_dict).resolve()

    @property
    def checkpoints_root(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def tensorboard_root(self) -> Path:
        return self.artifacts_dir / "tensorboard"

    @property
    def outputs_root(self) -> Path:
        return self.artifacts_dir / "outputs"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tft_{self.mode}_lb{self.lookback_days}_enc{self.max_encoder_length}_{stamp}"

    def model_params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "hidden_continuous_size": self.hidden_continuous_size,
            "dropout": self.dropout,
        }

    def trainer_params(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "gradient_clip_val": self.gradient_clip_val,
            "early_stopping_patience": self.early_stopping_patience,
            "random_seed": self.random_seed,
            "num_workers": self.num_workers,
        }

    def dataset_config(self) -> dict[str, Any]:
        return {
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "batch_size": self.batch_size,
            "min_history_points": self.min_history_points,
            "max_series": self.max_series,
            "series_sample_seed": self.random_seed,
            "num_workers": self.num_workers,
        }

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_root.mkdir(parents=True, exist_ok=True)
        self.tensorboard_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ["project_dir", "data_dir", "artifacts_dir", "resume_ckpt", "init_state_dict"]:
            value = payload.get(key)
            payload[key] = str(value) if value is not None else None
        return payload


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build CLI parser shared by data prep and training entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts_tft")
    parser.add_argument("--mode", choices=["final", "fold"], default="final")
    parser.add_argument("--valid-start", type=str, default=None)

    parser.add_argument("--lookback-days", type=int, default=224)
    parser.add_argument("--horizon-days", type=int, default=16)
    parser.add_argument("--max-encoder-length", type=int, default=56)
    parser.add_argument("--max-prediction-length", type=int, default=16)
    parser.add_argument("--max-series", type=_optional_int, default=30_000)
    parser.add_argument("--min-history-points", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--hidden-continuous-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--use-cache", dest="use_cache", action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--force-cache", action="store_true", default=False)
    parser.add_argument("--force-train", action="store_true", default=False)

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=Path, default=None)
    parser.add_argument("--init-state-dict", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TFTTrainConfig:
    """Convert parsed CLI args to a typed `TFTTrainConfig`."""
    return TFTTrainConfig(
        project_dir=args.project_dir,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        mode=args.mode,
        valid_start=args.valid_start,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        max_series=args.max_series,
        min_history_points=args.min_history_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        hidden_continuous_size=args.hidden_continuous_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.early_stopping_patience,
        random_seed=args.random_seed,
        use_cache=args.use_cache,
        force_cache=args.force_cache,
        force_train=args.force_train,
        run_name=args.run_name,
        resume_ckpt=args.resume_ckpt,
        init_state_dict=args.init_state_dict,
    )


def dump_config(config: TFTTrainConfig, output_path: Path) -> None:
    """Persist config snapshot to JSON for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
