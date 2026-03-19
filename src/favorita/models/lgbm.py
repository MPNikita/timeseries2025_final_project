from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ...configs import DEFAULT_LGBM_PARAMS
from ..eda_utils import build_train_eda_bundle
from ..baselines import (
    build_baseline_validation_artifacts,
    predict_hierarchical_baseline,
    predict_recent_mean_baseline,
    weighted_rmsle,
)
from ..cache import _cache_path
from ..features.aggregates import _build_prior_aggregate_bundle, _merge_recent_and_prior_features
from ..features.holidays import _build_store_date_holiday_features
from ..features.metadata import _encode_metadata_codes
from ..io import _load_model_reference_assets, read_train_chunks
from ..paths import DATA_DIR, SUBMISSIONS_DIR
from ..validation import build_rolling_origin_folds
from .common import (
    _attach_common_features,
    _load_train_rows_between,
    _ts_feature_list,
)


HOLIDAY_FLAG_COLUMNS = [
    "is_holiday",
    "is_event",
    "is_additional",
    "is_bridge",
    "is_work_day",
]

MODEL_FEATURES = [
    "store_code",
    "item_code",
    "onpromotion",
    "weekday",
    "day",
    "month",
    "weekofyear",
    "is_month_end",
    "is_payday",
    "family_code",
    "class",
    "perishable",
    "city_code",
    "state_code",
    "type_code",
    "cluster",
    "transactions",
    "dcoilwtico",
    "is_holiday",
    "is_event",
    "is_additional",
    "is_bridge",
    "is_work_day",
    "si_recent28",
    "siw_mean",
    "fsw_mean",
    "fw_mean",
    "item_mean",
    "store_mean",
]

CATEGORICAL_FEATURES = [
    "store_code",
    "item_code",
    "weekday",
    "month",
    "family_code",
    "city_code",
    "state_code",
    "type_code",
    "cluster",
]


def _repair_cached_submission_path(
    cached_result: dict[str, Any],
    lookback_days: int,
    tx_flag: int,
) -> dict[str, Any]:
    """Repair stale absolute `submission_path` values from cached artifacts."""
    current = dict(cached_result)
    metadata = dict(current.get("metadata", {}))
    raw_path = metadata.get("submission_path")

    candidates: list[Path] = [
        SUBMISSIONS_DIR / f"submission_lgbm_tscv_lb{lookback_days}_tx{tx_flag}.csv.gz",
    ]
    if isinstance(raw_path, str) and raw_path:
        old_path = Path(raw_path)
        candidates.insert(0, old_path)
        candidates.append(SUBMISSIONS_DIR / old_path.name)

    for candidate in candidates:
        if candidate.exists():
            metadata["submission_path"] = str(candidate.resolve())
            current["metadata"] = metadata
            return current

    current["metadata"] = metadata
    return current


def _load_recent_observed_rows(
    fit_start: pd.Timestamp,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        data_dir=data_dir,
    ):
        if chunk["date"].max() < fit_start:
            continue
        recent = chunk[chunk["date"] >= fit_start].copy()
        if not recent.empty:
            parts.append(recent)

    frame = pd.concat(parts, ignore_index=True)
    frame["target"] = frame["unit_sales"].clip(lower=0).astype("float32")
    frame["onpromotion"] = frame["onpromotion"].fillna(False).astype("int8")
    frame["store_nbr"] = frame["store_nbr"].astype("int16")
    frame["item_nbr"] = frame["item_nbr"].astype("int32")
    frame = frame.drop(columns="unit_sales")
    return frame


def _prepare_feature_frames(
    data_dir: Path,
    lookback_days: int,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overview = build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]
    max_date = pd.Timestamp(overview["date_max"])
    valid_start = max_date - pd.Timedelta(days=horizon_days - 1)
    fit_start = valid_start - pd.Timedelta(days=lookback_days)
    recent_28d_start = valid_start - pd.Timedelta(days=28)

    stores = pd.read_csv(data_dir / "stores.csv")
    items = pd.read_csv(data_dir / "items.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"]).sort_values("date")
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"]).sort_values("date")

    item_meta, store_meta = _encode_metadata_codes(items, stores)
    recent = _load_recent_observed_rows(fit_start=fit_start, data_dir=data_dir)

    recent["weekday"] = recent["date"].dt.dayofweek.astype("int8")
    recent["day"] = recent["date"].dt.day.astype("int8")
    recent["month"] = recent["date"].dt.month.astype("int8")
    recent["weekofyear"] = recent["date"].dt.isocalendar().week.astype("int16")
    recent["is_month_end"] = recent["date"].dt.is_month_end.astype("int8")
    recent["is_payday"] = ((recent["date"].dt.day == 15) | recent["date"].dt.is_month_end).astype("int8")

    recent = recent.merge(item_meta, on="item_nbr", how="left")
    recent = recent.merge(store_meta, on="store_nbr", how="left")

    recent["store_code"] = pd.Categorical(recent["store_nbr"]).codes.astype("int16")
    recent["item_code"] = pd.Categorical(recent["item_nbr"]).codes.astype("int32")

    transaction_frame = transactions.copy()
    transaction_frame["transactions"] = transaction_frame["transactions"].astype("float32")
    recent = recent.merge(
        transaction_frame[["date", "store_nbr", "transactions"]],
        on=["date", "store_nbr"],
        how="left",
    )

    oil_frame = oil.drop_duplicates("date").set_index("date")
    oil_frame = oil_frame.reindex(pd.date_range(recent["date"].min(), recent["date"].max(), freq="D"))
    oil_frame["dcoilwtico"] = oil_frame["dcoilwtico"].ffill().bfill().astype("float32")
    oil_frame = oil_frame.rename_axis("date").reset_index()
    recent = recent.merge(oil_frame[["date", "dcoilwtico"]], on="date", how="left")

    holiday_frame = _build_store_date_holiday_features(
        stores=stores,
        holidays=holidays,
        min_date=recent["date"].min(),
        max_date=recent["date"].max(),
    )
    recent = recent.merge(holiday_frame, on=["date", "store_nbr"], how="left")
    for column in HOLIDAY_FLAG_COLUMNS:
        recent[column] = recent[column].fillna(0).astype("int8")

    fit = recent[recent["date"] < valid_start].copy()
    valid = recent[recent["date"] >= valid_start].copy()

    agg_recent28 = (
        fit[fit["date"] >= recent_28d_start]
        .groupby(["store_nbr", "item_nbr"], observed=True)["target"]
        .mean()
        .rename("si_recent28")
        .reset_index()
    )
    agg_siw = (
        fit.groupby(["store_nbr", "item_nbr", "weekday"], observed=True)["target"]
        .mean()
        .rename("siw_mean")
        .reset_index()
    )
    agg_fsw = (
        fit.groupby(["family_code", "store_nbr", "weekday"], observed=True)["target"]
        .mean()
        .rename("fsw_mean")
        .reset_index()
    )
    agg_fw = (
        fit.groupby(["family_code", "weekday"], observed=True)["target"]
        .mean()
        .rename("fw_mean")
        .reset_index()
    )
    agg_item = fit.groupby("item_nbr", observed=True)["target"].mean().rename("item_mean").reset_index()
    agg_store = fit.groupby("store_nbr", observed=True)["target"].mean().rename("store_mean").reset_index()

    merge_specs = [
        (agg_recent28, ["store_nbr", "item_nbr"]),
        (agg_siw, ["store_nbr", "item_nbr", "weekday"]),
        (agg_fsw, ["family_code", "store_nbr", "weekday"]),
        (agg_fw, ["family_code", "weekday"]),
        (agg_item, ["item_nbr"]),
        (agg_store, ["store_nbr"]),
    ]
    for agg_frame, keys in merge_specs:
        fit = fit.merge(agg_frame, on=keys, how="left")
        valid = valid.merge(agg_frame, on=keys, how="left")

    numeric_fill = {
        "transactions": float(transaction_frame["transactions"].median()),
        "dcoilwtico": float(oil_frame["dcoilwtico"].median()),
        "si_recent28": 0.0,
        "siw_mean": 0.0,
        "fsw_mean": 0.0,
        "fw_mean": 0.0,
        "item_mean": 0.0,
        "store_mean": 0.0,
    }
    for frame in [fit, valid]:
        for column, fill_value in numeric_fill.items():
            frame[column] = frame[column].fillna(fill_value).astype("float32")

        frame["class"] = frame["class"].astype("int32")
        frame["perishable"] = frame["perishable"].astype("int8")
        frame["cluster"] = frame["cluster"].astype("int16")
        frame["family_code"] = frame["family_code"].astype("int16")
        frame["city_code"] = frame["city_code"].astype("int16")
        frame["state_code"] = frame["state_code"].astype("int16")
        frame["type_code"] = frame["type_code"].astype("int16")

    metadata = {
        "fit_start": fit_start,
        "valid_start": valid_start,
        "valid_end": max_date,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "fit_rows": int(len(fit)),
        "valid_rows": int(len(valid)),
    }
    return fit, valid, metadata


def run_lightgbm_validation_experiment(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 112,
    horizon_days: int = 16,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_path = _cache_path(
        f"lgbm_validation_lb{lookback_days}_hz{horizon_days}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    fit, valid, metadata = _prepare_feature_frames(
        data_dir=data_dir,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
    )
    params = DEFAULT_LGBM_PARAMS.copy()
    if model_params:
        params.update(model_params)

    fit_weights = np.where(fit["perishable"].eq(1), 1.25, 1.0)
    valid_weights = np.where(valid["perishable"].eq(1), 1.25, 1.0)

    model = LGBMRegressor(**params)
    model.fit(
        fit[MODEL_FEATURES],
        np.log1p(fit["target"]),
        sample_weight=fit_weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    lightgbm_pred = np.clip(np.expm1(model.predict(valid[MODEL_FEATURES])), 0, None)

    baseline_artifacts = build_baseline_validation_artifacts(
        data_dir=data_dir,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        use_cache=True,
    )
    recent_baseline = predict_recent_mean_baseline(baseline_artifacts)[
        ["date", "store_nbr", "item_nbr", "prediction"]
    ].rename(columns={"prediction": "recent_mean_prediction"})
    hierarchical_baseline = predict_hierarchical_baseline(baseline_artifacts)[
        ["date", "store_nbr", "item_nbr", "prediction", "source"]
    ].rename(columns={"prediction": "hierarchical_prediction", "source": "hierarchical_source"})

    valid_predictions = valid[
        ["date", "store_nbr", "item_nbr", "target", "perishable", "onpromotion"]
    ].copy().rename(columns={"target": "actual"})
    valid_predictions["weight"] = valid_weights.astype("float32")
    valid_predictions["lightgbm_prediction"] = lightgbm_pred.astype("float32")

    valid_predictions = valid_predictions.merge(
        recent_baseline,
        on=["date", "store_nbr", "item_nbr"],
        how="left",
    )
    valid_predictions = valid_predictions.merge(
        hierarchical_baseline,
        on=["date", "store_nbr", "item_nbr"],
        how="left",
    )

    score_rows = [
        {
            "model": "recent_mean_28d",
            "weighted_rmsle": weighted_rmsle(
                valid_predictions["actual"],
                valid_predictions["recent_mean_prediction"],
                valid_predictions["weight"],
            ),
        },
        {
            "model": "hierarchical_weekday_promo",
            "weighted_rmsle": weighted_rmsle(
                valid_predictions["actual"],
                valid_predictions["hierarchical_prediction"],
                valid_predictions["weight"],
            ),
        },
        {
            "model": "lightgbm_feature_model",
            "weighted_rmsle": weighted_rmsle(
                valid_predictions["actual"],
                valid_predictions["lightgbm_prediction"],
                valid_predictions["weight"],
            ),
        },
    ]
    scores = pd.DataFrame(score_rows).sort_values("weighted_rmsle", ignore_index=True)

    daily_validation = (
        valid_predictions.groupby("date", as_index=False)[
            ["actual", "recent_mean_prediction", "hierarchical_prediction", "lightgbm_prediction"]
        ]
        .sum()
        .sort_values("date", ignore_index=True)
    )

    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, ignore_index=True)

    result = {
        "metadata": metadata | {"model_params": params},
        "scores": scores,
        "feature_importance": feature_importance,
        "daily_validation": daily_validation,
        "valid_predictions": valid_predictions,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def _prepare_fold_feature_frames(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int = 16,
    include_transactions: bool = False,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
    fit_start = valid_start - pd.Timedelta(days=lookback_days)

    refs = _load_model_reference_assets(data_dir=data_dir)
    recent = _load_train_rows_between(start_date=fit_start, end_date=valid_end, data_dir=data_dir)
    prepared = _attach_common_features(
        recent,
        refs=refs,
        min_date=fit_start,
        max_date=valid_end,
        include_transactions=include_transactions,
    )

    fit = prepared[prepared["date"] < valid_start].copy()
    valid = prepared[prepared["date"] >= valid_start].copy()
    prior_bundle = _build_prior_aggregate_bundle(
        cutoff_date=valid_start,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    fit, valid = _merge_recent_and_prior_features(
        fit=fit,
        valid=valid,
        prior_bundle=prior_bundle,
        include_transactions=include_transactions,
    )

    metadata = {
        "fit_start": fit_start,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "fit_rows": int(len(fit)),
        "valid_rows": int(len(valid)),
        "include_transactions": include_transactions,
    }
    return fit, valid, metadata


def run_single_fold_time_series_experiment(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int = 16,
    include_transactions: bool = False,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tx_flag = int(include_transactions)
    cache_path = _cache_path(
        f"ts_fold_{valid_start:%Y%m%d}_lb{lookback_days}_hz{horizon_days}_tx{tx_flag}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    fit, valid, metadata = _prepare_fold_feature_frames(
        valid_start=valid_start,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        include_transactions=include_transactions,
        data_dir=data_dir,
        use_cache=use_cache,
    )

    params = DEFAULT_LGBM_PARAMS.copy()
    if model_params:
        params.update(model_params)
    features = _ts_feature_list(include_transactions=include_transactions)

    fit_weights = np.where(fit["perishable"].eq(1), 1.25, 1.0)
    valid_weights = np.where(valid["perishable"].eq(1), 1.25, 1.0)

    model = LGBMRegressor(**params)
    model.fit(
        fit[features],
        np.log1p(fit["target"]),
        sample_weight=fit_weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    prediction = np.clip(np.expm1(model.predict(valid[features])), 0, None)

    valid_predictions = valid[
        ["date", "store_nbr", "item_nbr", "target", "perishable", "onpromotion"]
    ].copy().rename(columns={"target": "actual"})
    valid_predictions["weight"] = valid_weights.astype("float32")
    valid_predictions["prediction"] = prediction.astype("float32")

    score = weighted_rmsle(valid_predictions["actual"], valid_predictions["prediction"], valid_predictions["weight"])
    daily_validation = (
        valid_predictions.groupby("date", as_index=False)[["actual", "prediction"]]
        .sum()
        .sort_values("date", ignore_index=True)
    )
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, ignore_index=True)

    result = {
        "metadata": metadata | {"model_params": params},
        "score": score,
        "feature_importance": feature_importance,
        "daily_validation": daily_validation,
        "valid_predictions": valid_predictions,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_time_series_cross_validation(
    data_dir: Path = DATA_DIR,
    lookback_grid: tuple[int, ...] = (112, 224),
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
    include_transactions: bool = False,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tx_flag = int(include_transactions)
    cache_path = _cache_path(
        f"tscv_lb{'-'.join(map(str, lookback_grid))}_hz{horizon_days}_step{step_days}_folds{n_folds}_tx{tx_flag}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    folds = build_rolling_origin_folds(
        data_dir=data_dir,
        horizon_days=horizon_days,
        step_days=step_days,
        n_folds=n_folds,
    )

    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    for fold_row in folds.itertuples(index=False):
        for lookback_days in lookback_grid:
            result = run_single_fold_time_series_experiment(
                valid_start=fold_row.valid_start,
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                include_transactions=include_transactions,
                data_dir=data_dir,
                use_cache=use_cache,
                force=force,
                model_params=model_params,
            )
            fold_rows.append(
                {
                    "fold": fold_row.fold,
                    "valid_start": fold_row.valid_start,
                    "valid_end": fold_row.valid_end,
                    "lookback_days": lookback_days,
                    "fit_rows": result["metadata"]["fit_rows"],
                    "valid_rows": result["metadata"]["valid_rows"],
                    "weighted_rmsle": result["score"],
                }
            )
            fold_importance = result["feature_importance"].copy()
            fold_importance["fold"] = fold_row.fold
            fold_importance["lookback_days"] = lookback_days
            importance_rows.append(fold_importance)

    fold_scores = pd.DataFrame(fold_rows).sort_values(["lookback_days", "valid_start"], ignore_index=True)
    summary = (
        fold_scores.groupby("lookback_days", as_index=False)["weighted_rmsle"]
        .agg(mean_score="mean", std_score="std", min_score="min", max_score="max")
        .sort_values("mean_score", ignore_index=True)
    )

    if importance_rows:
        importance = pd.concat(importance_rows, ignore_index=True)
        importance_summary = (
            importance.groupby("feature", as_index=False)[["importance_gain", "importance_split"]]
            .mean()
            .sort_values("importance_gain", ascending=False, ignore_index=True)
        )
    else:
        importance_summary = pd.DataFrame(columns=["feature", "importance_gain", "importance_split"])

    best_row = summary.iloc[0]
    result = {
        "metadata": {
            "horizon_days": horizon_days,
            "step_days": step_days,
            "n_folds": n_folds,
            "lookback_grid": lookback_grid,
            "include_transactions": include_transactions,
            "best_lookback_days": int(best_row["lookback_days"]),
            "best_mean_score": float(best_row["mean_score"]),
        },
        "folds": folds,
        "fold_scores": fold_scores,
        "summary": summary,
        "feature_importance_summary": importance_summary,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def train_final_time_series_model(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 112,
    include_transactions: bool = False,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tx_flag = int(include_transactions)
    cache_path = _cache_path(
        f"final_model_lb{lookback_days}_tx{tx_flag}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        cached_result = pd.read_pickle(cache_path)
        return _repair_cached_submission_path(
            cached_result=cached_result,
            lookback_days=lookback_days,
            tx_flag=tx_flag,
        )

    refs = _load_model_reference_assets(data_dir=data_dir)
    train_end = build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]["date_max"]
    train_end = pd.Timestamp(train_end)
    test_start = refs["test"]["date"].min()
    test_end = refs["test"]["date"].max()
    fit_start = train_end - pd.Timedelta(days=lookback_days)

    train_rows = _load_train_rows_between(start_date=fit_start, end_date=train_end, data_dir=data_dir)
    train_frame = _attach_common_features(
        train_rows,
        refs=refs,
        min_date=fit_start,
        max_date=train_end,
        include_transactions=include_transactions,
    )

    test_frame = refs["test"].copy()
    test_frame["onpromotion"] = test_frame["onpromotion"].fillna(False).astype("int8")
    test_frame = _attach_common_features(
        test_frame,
        refs=refs,
        min_date=test_start,
        max_date=test_end,
        include_transactions=include_transactions,
    )

    prior_bundle = _build_prior_aggregate_bundle(
        cutoff_date=test_start,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    dummy_valid = test_frame.copy()
    train_frame, test_frame = _merge_recent_and_prior_features(
        fit=train_frame,
        valid=dummy_valid,
        prior_bundle=prior_bundle,
        include_transactions=include_transactions,
    )

    params = DEFAULT_LGBM_PARAMS.copy()
    if model_params:
        params.update(model_params)
    features = _ts_feature_list(include_transactions=include_transactions)

    train_weights = np.where(train_frame["perishable"].eq(1), 1.25, 1.0)
    model = LGBMRegressor(**params)
    model.fit(
        train_frame[features],
        np.log1p(train_frame["target"]),
        sample_weight=train_weights,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    prediction = np.clip(np.expm1(model.predict(test_frame[features])), 0, None)
    submission = refs["test"][["id"]].copy()
    submission["unit_sales"] = prediction.astype("float32")

    submission_path = SUBMISSIONS_DIR / f"submission_lgbm_tscv_lb{lookback_days}_tx{tx_flag}.csv.gz"
    submission.to_csv(submission_path, index=False, compression="gzip")

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, ignore_index=True)

    result = {
        "metadata": {
            "fit_start": fit_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "lookback_days": lookback_days,
            "include_transactions": include_transactions,
            "fit_rows": int(len(train_frame)),
            "model_params": params,
            "submission_path": str(submission_path),
        },
        "feature_importance": feature_importance,
        "submission_head": submission.head(10),
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_full_time_series_training_pipeline(
    data_dir: Path = DATA_DIR,
    lookback_grid: tuple[int, ...] = (112, 224),
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
    include_transactions: bool = False,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv_result = run_time_series_cross_validation(
        data_dir=data_dir,
        lookback_grid=lookback_grid,
        horizon_days=horizon_days,
        step_days=step_days,
        n_folds=n_folds,
        include_transactions=include_transactions,
        use_cache=use_cache,
        force=force,
        model_params=model_params,
    )
    final_result = train_final_time_series_model(
        data_dir=data_dir,
        lookback_days=cv_result["metadata"]["best_lookback_days"],
        include_transactions=include_transactions,
        use_cache=use_cache,
        force=force,
        model_params=model_params,
    )
    return {"cv": cv_result, "final": final_result}
