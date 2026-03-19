from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from ..favorita_catboost import _load_recent_train_cache
from ..favorita_tft import _prepare_tft_fold_frames, _prepare_tft_test_frames

from .config_tft import TFTTrainConfig, build_parser, config_from_args
from .dataset_tft import resolve_valid_start


REQUIRED_CSV = [
    "train.csv",
    "test.csv",
    "items.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
]


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    suffix = archive_path.name.lower()
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
        return
    if suffix.endswith(".7z"):
        if shutil.which("7z") is None:
            raise RuntimeError("`7z` command is required to extract .7z archives.")
        subprocess.run(["7z", "x", "-y", str(archive_path), f"-o{output_dir}"], check=True)
        return
    if suffix.endswith(".tar") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def extract_bundle(bundle_archive: Path, data_dir: Path) -> None:
    """Extract one outer bundle and nested `*.csv.7z` files into `data_dir`."""
    if not bundle_archive.exists():
        raise FileNotFoundError(f"Bundle archive not found: {bundle_archive}")

    raw_dir = data_dir / "raw_bundle"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prep] Extracting bundle: {bundle_archive}")
    _extract_archive(bundle_archive, raw_dir)

    csv7z_files = sorted(raw_dir.rglob("*.csv.7z"))
    if not csv7z_files:
        print("[prep] No nested '*.csv.7z' found. Assuming CSV files are already extracted.")
    for csv7z in csv7z_files:
        print(f"[prep] Extracting nested archive: {csv7z.name}")
        _extract_archive(csv7z, data_dir)


def validate_csv_presence(data_dir: Path) -> None:
    """Fail fast if required competition CSV files are missing."""
    missing = [name for name in REQUIRED_CSV if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {missing}")
    print("[prep] Required CSV files are present.")


def warm_tft_cache(config: TFTTrainConfig) -> None:
    """Warm expensive caches so remote training starts fast."""
    print("[prep] Building recent train cache...")
    _load_recent_train_cache(
        data_dir=config.data_dir,
        use_cache=config.use_cache,
        force=config.force_cache,
    )
    print("[prep] Recent train cache ready.")

    if config.mode == "fold":
        valid_start = resolve_valid_start(config)
        print(f"[prep] Building fold cache for valid_start={valid_start.date()}...")
        _prepare_tft_fold_frames(
            valid_start=valid_start,
            lookback_days=config.lookback_days,
            horizon_days=config.horizon_days,
            max_encoder_length=config.max_encoder_length,
            min_history_points=config.min_history_points,
            max_series=config.max_series,
            random_seed=config.random_seed,
            data_dir=config.data_dir,
            use_cache=config.use_cache,
            force=config.force_cache,
        )
    else:
        print("[prep] Building final train/test cache...")
        _prepare_tft_test_frames(
            lookback_days=config.lookback_days,
            max_encoder_length=config.max_encoder_length,
            min_history_points=config.min_history_points,
            max_series=config.max_series,
            random_seed=config.random_seed,
            data_dir=config.data_dir,
            use_cache=config.use_cache,
            force=config.force_cache,
        )
    print("[prep] TFT cache is ready.")


def main() -> None:
    parser = build_parser("Prepare Favorita TFT data and warm caches.")
    parser.add_argument("--bundle-archive", type=Path, default=None)
    parser.add_argument("--skip-cache-warmup", action="store_true", default=False)
    args = parser.parse_args()

    config = config_from_args(args)
    config.ensure_dirs()
    config.data_dir.mkdir(parents=True, exist_ok=True)

    if args.bundle_archive is not None:
        extract_bundle(bundle_archive=args.bundle_archive, data_dir=config.data_dir)
    validate_csv_presence(config.data_dir)

    if not args.skip_cache_warmup:
        warm_tft_cache(config)
    print("[prep] Done.")


if __name__ == "__main__":
    main()
