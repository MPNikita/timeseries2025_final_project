from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _group_mean_count(
    frame: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True)["target"].agg(mean="mean", count="size").reset_index()
    grouped["mean"] = grouped["mean"].astype("float32")
    grouped["count"] = grouped["count"].astype("int32")
    return grouped


def _build_fallback_tables(fit: pd.DataFrame) -> dict[str, Any]:
    """Build hierarchical fallback lookup tables."""
    return {
        "siwo": _group_mean_count(fit, ["store_nbr", "item_nbr", "weekday", "onpromotion"]),
        "siw": _group_mean_count(fit, ["store_nbr", "item_nbr", "weekday"]),
        "fswo": _group_mean_count(fit, ["family_code", "store_nbr", "weekday", "onpromotion"]),
        "fsw": _group_mean_count(fit, ["family_code", "store_nbr", "weekday"]),
        "cswo": _group_mean_count(fit, ["class", "store_nbr", "weekday", "onpromotion"]),
        "csw": _group_mean_count(fit, ["class", "store_nbr", "weekday"]),
        "fwo": _group_mean_count(fit, ["family_code", "weekday", "onpromotion"]),
        "fw": _group_mean_count(fit, ["family_code", "weekday"]),
        "cw": _group_mean_count(fit, ["class", "weekday"]),
        "sw": _group_mean_count(fit, ["store_nbr", "weekday"]),
        "w": _group_mean_count(fit, ["weekday"]),
        "global_mean": float(fit["target"].mean()),
    }


def _apply_fallback_lookup(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    keys: list[str],
    min_count: int,
    source: str,
) -> None:
    if lookup.empty or frame["fallback_prediction"].notna().all():
        return

    columns = keys + ["mean", "count"]
    merged = frame.loc[frame["fallback_prediction"].isna(), keys].merge(lookup[columns], on=keys, how="left")
    eligible = merged["mean"].notna() & (merged["count"] >= min_count)
    if not eligible.any():
        return

    target_index = frame.index[frame["fallback_prediction"].isna()][eligible.to_numpy()]
    frame.loc[target_index, "fallback_prediction"] = merged.loc[eligible, "mean"].to_numpy()
    frame.loc[target_index, "fallback_source"] = source


def build_hierarchical_fallback_predictions(
    frame: pd.DataFrame,
    fallback_tables: dict[str, Any],
) -> pd.DataFrame:
    """Predict via hierarchical mean lookups with global fallback."""
    current = frame.copy()
    current["fallback_prediction"] = np.nan
    current["fallback_source"] = pd.Series(pd.NA, index=current.index, dtype="object")

    steps = [
        (fallback_tables["siwo"], ["store_nbr", "item_nbr", "weekday", "onpromotion"], 2, "store_item_weekday_promo"),
        (fallback_tables["siw"], ["store_nbr", "item_nbr", "weekday"], 2, "store_item_weekday"),
        (fallback_tables["fswo"], ["family_code", "store_nbr", "weekday", "onpromotion"], 5, "family_store_weekday_promo"),
        (fallback_tables["fsw"], ["family_code", "store_nbr", "weekday"], 5, "family_store_weekday"),
        (fallback_tables["cswo"], ["class", "store_nbr", "weekday", "onpromotion"], 5, "class_store_weekday_promo"),
        (fallback_tables["csw"], ["class", "store_nbr", "weekday"], 5, "class_store_weekday"),
        (fallback_tables["fwo"], ["family_code", "weekday", "onpromotion"], 10, "family_weekday_promo"),
        (fallback_tables["fw"], ["family_code", "weekday"], 10, "family_weekday"),
        (fallback_tables["cw"], ["class", "weekday"], 10, "class_weekday"),
        (fallback_tables["sw"], ["store_nbr", "weekday"], 10, "store_weekday"),
        (fallback_tables["w"], ["weekday"], 1, "weekday"),
    ]
    for lookup, keys, min_count, source in steps:
        _apply_fallback_lookup(current, lookup, keys, min_count=min_count, source=source)

    current["fallback_prediction"] = current["fallback_prediction"].fillna(fallback_tables["global_mean"]).clip(lower=0)
    current["fallback_source"] = current["fallback_source"].fillna("global_mean")
    return current[["fallback_prediction", "fallback_source"]]

