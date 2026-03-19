from __future__ import annotations

from pathlib import Path

from .paths import CACHE_DIR, DATA_DIR


def resolve_cache_dir(data_dir: Path = DATA_DIR) -> Path:
    """Resolve cache directory for the provided data root."""
    root = Path(data_dir).resolve()
    base_cache_dir = CACHE_DIR if root == DATA_DIR.resolve() else root / ".cache" / "favorita"
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    return base_cache_dir


def cache_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Build pickle cache file path."""
    return resolve_cache_dir(data_dir=data_dir) / f"{name}.pkl"


def _cache_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Backwards-compatible alias used by legacy modules."""
    return cache_path(name=name, data_dir=data_dir)

