from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from favorita_models import build_rolling_origin_folds
from favorita_tft import (
    _build_tft_datasets,
    _prepare_tft_fold_frames,
    _prepare_tft_test_frames,
)

from .config_tft import TFTTrainConfig


@dataclass
class PreparedTFTBundle:
    """Prepared data bundle used by server training/inference scripts."""

    mode: str
    fit_frame: pd.DataFrame
    horizon_frame: pd.DataFrame
    eligible_frame: pd.DataFrame
    metadata: dict[str, Any]
    dataset_bundle: dict[str, Any] | None


def resolve_valid_start(config: TFTTrainConfig) -> pd.Timestamp:
    """Resolve validation anchor date for fold mode."""
    if config.valid_start:
        return pd.Timestamp(config.valid_start)
    folds = build_rolling_origin_folds(data_dir=config.data_dir, n_folds=4)
    return pd.Timestamp(folds.iloc[-1]["valid_start"])


def _maybe_build_torch_datasets(
    fit_frame: pd.DataFrame,
    eligible_frame: pd.DataFrame,
    config: TFTTrainConfig,
) -> dict[str, Any] | None:
    if fit_frame.empty or eligible_frame.empty:
        return None
    return _build_tft_datasets(
        fit_frame=fit_frame,
        horizon_frame=eligible_frame,
        dataset_config=config.dataset_config(),
    )


def prepare_fold_bundle(config: TFTTrainConfig) -> PreparedTFTBundle:
    """Prepare one pseudo-public validation fold for TFT training/inference."""
    valid_start = resolve_valid_start(config)
    fit_frame, valid_frame, _eligible_pairs, metadata = _prepare_tft_fold_frames(
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
    valid_eligible = valid_frame.loc[valid_frame["is_tft_eligible"].eq(1)].copy()
    dataset_bundle = _maybe_build_torch_datasets(
        fit_frame=fit_frame,
        eligible_frame=valid_eligible,
        config=config,
    )
    metadata = dict(metadata)
    metadata["valid_start"] = pd.Timestamp(valid_start)
    return PreparedTFTBundle(
        mode="fold",
        fit_frame=fit_frame,
        horizon_frame=valid_frame,
        eligible_frame=valid_eligible,
        metadata=metadata,
        dataset_bundle=dataset_bundle,
    )


def prepare_final_bundle(config: TFTTrainConfig) -> PreparedTFTBundle:
    """Prepare full train/test setup for final TFT submission inference."""
    fit_frame, test_frame, _eligible_pairs, metadata = _prepare_tft_test_frames(
        lookback_days=config.lookback_days,
        max_encoder_length=config.max_encoder_length,
        min_history_points=config.min_history_points,
        max_series=config.max_series,
        random_seed=config.random_seed,
        data_dir=config.data_dir,
        use_cache=config.use_cache,
        force=config.force_cache,
    )
    test_eligible = test_frame.loc[test_frame["is_tft_eligible"].eq(1)].copy()
    dataset_bundle = _maybe_build_torch_datasets(
        fit_frame=fit_frame,
        eligible_frame=test_eligible,
        config=config,
    )
    return PreparedTFTBundle(
        mode="final",
        fit_frame=fit_frame,
        horizon_frame=test_frame,
        eligible_frame=test_eligible,
        metadata=dict(metadata),
        dataset_bundle=dataset_bundle,
    )


def prepare_bundle(config: TFTTrainConfig) -> PreparedTFTBundle:
    """Prepare training/inference data bundle according to selected mode."""
    if config.mode == "fold":
        return prepare_fold_bundle(config)
    return prepare_final_bundle(config)

