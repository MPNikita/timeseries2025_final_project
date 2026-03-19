from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import DATA_DIR


def build_rolling_origin_folds(
    data_dir: Path = DATA_DIR,
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
) -> pd.DataFrame:
    """Build rolling-origin fold anchors used by pseudo-public validation."""
    from .eda_utils import build_train_eda_bundle

    overview = build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]
    train_end = pd.Timestamp(overview["date_max"])
    latest_valid_start = train_end - pd.Timedelta(days=horizon_days - 1)

    rows: list[dict[str, object]] = []
    for offset in range(n_folds - 1, -1, -1):
        valid_start = latest_valid_start - pd.Timedelta(days=offset * step_days)
        valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
        rows.append(
            {
                "fold": f"fold_{len(rows) + 1}",
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )
    return pd.DataFrame(rows)

