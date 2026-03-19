from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_rmsle(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
) -> float:
    """Weighted RMSLE used across all Favorita offline experiments."""
    actual = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.average((np.log1p(pred) - np.log1p(actual)) ** 2, weights=w)))

