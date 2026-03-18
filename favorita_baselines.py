from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from favorita_eda_utils import DATA_DIR, _cache_path, build_train_eda_bundle, read_train_chunks


def _accumulate_group_sum_count(
    storage: dict[str, pd.DataFrame | None],
    name: str,
    frame: pd.DataFrame,
    keys: list[str],
    target: str = "target",
) -> None:
    if frame.empty:
        return
    agg = frame.groupby(keys, observed=True)[target].agg(sum="sum", count="size")
    storage[name] = agg if storage[name] is None else storage[name].add(agg, fill_value=0)


def _finalize_group_sum_count(table: pd.DataFrame | None) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame(columns=["sum", "count", "mean"])
    result = table.reset_index()
    result["count"] = result["count"].astype("int64")
    result["mean"] = result["sum"] / result["count"]
    return result


def build_baseline_validation_artifacts(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 112,
    horizon_days: int = 16,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path(
        f"baseline_validation_lb{lookback_days}_hz{horizon_days}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    train_overview = build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]
    max_date = pd.Timestamp(train_overview["date_max"])
    valid_start = max_date - pd.Timedelta(days=horizon_days - 1)
    fit_start = valid_start - pd.Timedelta(days=lookback_days)
    recent_28d_start = valid_start - pd.Timedelta(days=28)

    items = pd.read_csv(data_dir / "items.csv", usecols=["item_nbr", "family", "perishable"])
    item_to_family = items.set_index("item_nbr")["family"]
    item_to_weight = items.assign(weight=np.where(items["perishable"].eq(1), 1.25, 1.0)).set_index("item_nbr")[
        "weight"
    ]

    aggregations: dict[str, pd.DataFrame | None] = {
        "si_recent28": None,
        "item_recent28": None,
        "weekday": None,
        "siwo": None,
        "siw": None,
        "fswo": None,
        "fsw": None,
        "fwo": None,
        "fw": None,
    }
    valid_parts: list[pd.DataFrame] = []
    fit_rows = 0

    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        data_dir=data_dir,
    ):
        if chunk["date"].max() < fit_start:
            continue
        chunk = chunk[chunk["date"] >= fit_start].copy()
        if chunk.empty:
            continue

        chunk["target"] = chunk["unit_sales"].clip(lower=0).astype("float32")
        chunk["weekday"] = chunk["date"].dt.dayofweek.astype("int8")
        chunk["family"] = chunk["item_nbr"].map(item_to_family).astype("category")
        chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)

        fit = chunk[chunk["date"] < valid_start].copy()
        valid = chunk[chunk["date"] >= valid_start].copy()

        if not fit.empty:
            fit_rows += len(fit)
            fit_recent28 = fit[fit["date"] >= recent_28d_start]

            _accumulate_group_sum_count(aggregations, "si_recent28", fit_recent28, ["store_nbr", "item_nbr"])
            _accumulate_group_sum_count(aggregations, "item_recent28", fit_recent28, ["item_nbr"])
            _accumulate_group_sum_count(aggregations, "weekday", fit, ["weekday"])
            _accumulate_group_sum_count(
                aggregations,
                "siwo",
                fit,
                ["store_nbr", "item_nbr", "weekday", "onpromotion"],
            )
            _accumulate_group_sum_count(aggregations, "siw", fit, ["store_nbr", "item_nbr", "weekday"])
            _accumulate_group_sum_count(
                aggregations,
                "fswo",
                fit,
                ["family", "store_nbr", "weekday", "onpromotion"],
            )
            _accumulate_group_sum_count(aggregations, "fsw", fit, ["family", "store_nbr", "weekday"])
            _accumulate_group_sum_count(aggregations, "fwo", fit, ["family", "weekday", "onpromotion"])
            _accumulate_group_sum_count(aggregations, "fw", fit, ["family", "weekday"])

        if not valid.empty:
            valid["weight"] = valid["item_nbr"].map(item_to_weight).astype("float32")
            valid_parts.append(
                valid[
                    [
                        "date",
                        "store_nbr",
                        "item_nbr",
                        "family",
                        "weekday",
                        "onpromotion",
                        "target",
                        "weight",
                    ]
                ]
            )

    valid_frame = pd.concat(valid_parts, ignore_index=True).rename(columns={"target": "actual"})
    valid_frame = valid_frame.sort_values(["date", "store_nbr", "item_nbr"], ignore_index=True)

    finalized = {name: _finalize_group_sum_count(table) for name, table in aggregations.items()}
    weekday_table = finalized["weekday"]
    global_mean = float(weekday_table["sum"].sum() / weekday_table["count"].sum())

    artifacts = {
        "metadata": {
            "fit_start": fit_start,
            "valid_start": valid_start,
            "valid_end": max_date,
            "lookback_days": lookback_days,
            "horizon_days": horizon_days,
            "fit_rows": fit_rows,
            "valid_rows": int(len(valid_frame)),
            "global_mean": global_mean,
        },
        "aggregations": finalized,
        "valid": valid_frame,
    }
    if use_cache:
        pd.to_pickle(artifacts, cache_path)
    return artifacts


def weighted_rmsle(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
) -> float:
    actual = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.average((np.log1p(pred) - np.log1p(actual)) ** 2, weights=w)))


def _apply_lookup(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    keys: list[str],
    min_count: int,
    source: str,
) -> None:
    if lookup.empty or frame["prediction"].notna().all():
        return

    columns = keys + ["mean", "count"]
    merged = frame.loc[frame["prediction"].isna(), keys].merge(lookup[columns], on=keys, how="left")
    eligible = merged["mean"].notna() & (merged["count"] >= min_count)
    if not eligible.any():
        return

    target_index = frame.index[frame["prediction"].isna()][eligible.to_numpy()]
    frame.loc[target_index, "prediction"] = merged.loc[eligible, "mean"].to_numpy()
    frame.loc[target_index, "source"] = source


def predict_recent_mean_baseline(artifacts: dict[str, Any]) -> pd.DataFrame:
    valid = artifacts["valid"].copy()
    agg = artifacts["aggregations"]

    valid["prediction"] = np.nan
    valid["source"] = pd.Series(pd.NA, index=valid.index, dtype="object")

    _apply_lookup(valid, agg["si_recent28"], ["store_nbr", "item_nbr"], min_count=2, source="store_item_recent28")
    _apply_lookup(valid, agg["item_recent28"], ["item_nbr"], min_count=5, source="item_recent28")
    _apply_lookup(valid, agg["weekday"], ["weekday"], min_count=1, source="weekday")

    valid["prediction"] = valid["prediction"].fillna(artifacts["metadata"]["global_mean"]).clip(lower=0)
    valid["source"] = valid["source"].fillna("global_mean")
    return valid


def predict_hierarchical_baseline(artifacts: dict[str, Any]) -> pd.DataFrame:
    valid = artifacts["valid"].copy()
    agg = artifacts["aggregations"]

    valid["prediction"] = np.nan
    valid["source"] = pd.Series(pd.NA, index=valid.index, dtype="object")

    steps = [
        (agg["siwo"], ["store_nbr", "item_nbr", "weekday", "onpromotion"], 2, "store_item_weekday_promo"),
        (agg["siw"], ["store_nbr", "item_nbr", "weekday"], 2, "store_item_weekday"),
        (agg["fswo"], ["family", "store_nbr", "weekday", "onpromotion"], 5, "family_store_weekday_promo"),
        (agg["fsw"], ["family", "store_nbr", "weekday"], 5, "family_store_weekday"),
        (agg["fwo"], ["family", "weekday", "onpromotion"], 10, "family_weekday_promo"),
        (agg["fw"], ["family", "weekday"], 10, "family_weekday"),
        (agg["weekday"], ["weekday"], 1, "weekday"),
    ]
    for lookup, keys, min_count, source in steps:
        _apply_lookup(valid, lookup, keys, min_count=min_count, source=source)

    valid["prediction"] = valid["prediction"].fillna(artifacts["metadata"]["global_mean"]).clip(lower=0)
    valid["source"] = valid["source"].fillna("global_mean")
    return valid


def summarize_baseline_results(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.items():
        score = weighted_rmsle(frame["actual"], frame["prediction"], frame["weight"])
        rows.append(
            {
                "baseline": model_name,
                "weighted_rmsle": score,
                "mean_prediction": float(frame["prediction"].mean()),
                "mean_actual": float(frame["actual"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_rmsle", ignore_index=True)
