from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from favorita_baselines import (
    build_baseline_validation_artifacts,
    predict_hierarchical_baseline,
    predict_recent_mean_baseline,
    weighted_rmsle,
)
from favorita_eda_utils import DATA_DIR, _cache_path, build_train_eda_bundle, read_train_chunks


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

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "force_row_wise": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def _encode_metadata_codes(items: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_meta = items[["item_nbr", "class", "perishable"]].copy()
    item_meta["family_code"] = pd.Categorical(items["family"]).codes.astype("int16")
    item_meta["class"] = item_meta["class"].astype("int32")
    item_meta["perishable"] = item_meta["perishable"].astype("int8")

    store_meta = stores[["store_nbr", "cluster"]].copy()
    store_meta["city_code"] = pd.Categorical(stores["city"]).codes.astype("int16")
    store_meta["state_code"] = pd.Categorical(stores["state"]).codes.astype("int16")
    store_meta["type_code"] = pd.Categorical(stores["type"]).codes.astype("int16")
    store_meta["cluster"] = store_meta["cluster"].astype("int16")
    return item_meta, store_meta


def _build_store_date_holiday_features(
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    relevant = holidays[(holidays["date"] >= min_date) & (holidays["date"] <= max_date)].copy()
    relevant = relevant[~((relevant["type"] == "Holiday") & (relevant["transferred"]))]

    rows: list[dict[str, Any]] = []
    for row in relevant.itertuples(index=False):
        base = {
            "date": row.date,
            "is_holiday": int(row.type in ["Holiday", "Transfer"]),
            "is_event": int(row.type == "Event"),
            "is_additional": int(row.type == "Additional"),
            "is_bridge": int(row.type == "Bridge"),
            "is_work_day": int(row.type == "Work Day"),
        }
        if row.locale == "National":
            base["store_nbr"] = None
            rows.append(base)
        elif row.locale == "Regional":
            matches = stores.loc[stores["state"].eq(row.locale_name), "store_nbr"]
            for store_nbr in matches:
                current = base.copy()
                current["store_nbr"] = int(store_nbr)
                rows.append(current)
        else:
            matches = stores.loc[stores["city"].eq(row.locale_name), "store_nbr"]
            for store_nbr in matches:
                current = base.copy()
                current["store_nbr"] = int(store_nbr)
                rows.append(current)

    full_index = pd.MultiIndex.from_product(
        [pd.date_range(min_date, max_date, freq="D"), stores["store_nbr"].sort_values().unique()],
        names=["date", "store_nbr"],
    ).to_frame(index=False)

    if not rows:
        for column in HOLIDAY_FLAG_COLUMNS:
            full_index[column] = 0
        full_index["store_nbr"] = full_index["store_nbr"].astype("int16")
        return full_index

    holiday_map = pd.DataFrame(rows)
    national = holiday_map[holiday_map["store_nbr"].isna()].drop(columns="store_nbr")
    if national.empty:
        national = pd.DataFrame(columns=["date", *HOLIDAY_FLAG_COLUMNS])
    else:
        national = national.groupby("date", as_index=False).max()
    local = holiday_map[holiday_map["store_nbr"].notna()].copy()
    if local.empty:
        local = pd.DataFrame(columns=["date", "store_nbr", *HOLIDAY_FLAG_COLUMNS])
    else:
        local["store_nbr"] = local["store_nbr"].astype("int16")
        local = local.groupby(["date", "store_nbr"], as_index=False).max()

    full_index = full_index.merge(national, on="date", how="left")
    full_index = full_index.merge(local, on=["date", "store_nbr"], how="left", suffixes=("_nat", "_loc"))

    for column in HOLIDAY_FLAG_COLUMNS:
        pair = full_index[[f"{column}_nat", f"{column}_loc"]].astype("float32").fillna(0.0)
        full_index[column] = (
            pair.max(axis=1).astype("int8")
        )
        full_index = full_index.drop(columns=[f"{column}_nat", f"{column}_loc"])

    full_index["store_nbr"] = full_index["store_nbr"].astype("int16")
    return full_index


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


TS_BASE_FEATURES = [
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
    "dcoilwtico",
    "is_holiday",
    "is_event",
    "is_additional",
    "is_bridge",
    "is_work_day",
    "si_recent28",
    "si_recent56",
    "siw_mean",
    "siw_count_log",
    "fsw_mean",
    "fw_mean",
    "item_recent_mean",
    "store_recent_mean",
    "si_all_mean",
    "si_all_count_log",
    "item_all_mean",
    "item_all_count_log",
    "store_all_mean",
    "family_all_mean",
    "family_store_all_mean",
    "family_store_all_count_log",
]


def _ts_feature_list(include_transactions: bool = False) -> list[str]:
    return TS_BASE_FEATURES + (["transactions"] if include_transactions else [])


def _load_model_reference_assets(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    stores = pd.read_csv(data_dir / "stores.csv").sort_values("store_nbr").reset_index(drop=True)
    items = pd.read_csv(data_dir / "items.csv").sort_values("item_nbr").reset_index(drop=True)
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"]).sort_values("date")
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"]).sort_values("date")
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])
    test = pd.read_csv(
        data_dir / "test.csv",
        usecols=["id", "date", "store_nbr", "item_nbr", "onpromotion"],
        dtype={"id": "int64", "store_nbr": "int16", "item_nbr": "int32", "onpromotion": "boolean"},
        parse_dates=["date"],
    )

    store_meta = stores[["store_nbr", "cluster"]].copy()
    store_meta["store_code"] = np.arange(len(store_meta), dtype="int16")
    store_meta["city_code"] = pd.Categorical(stores["city"]).codes.astype("int16")
    store_meta["state_code"] = pd.Categorical(stores["state"]).codes.astype("int16")
    store_meta["type_code"] = pd.Categorical(stores["type"]).codes.astype("int16")
    store_meta["cluster"] = store_meta["cluster"].astype("int16")

    item_meta = items[["item_nbr", "class", "perishable"]].copy()
    item_meta["item_code"] = np.arange(len(item_meta), dtype="int32")
    item_meta["family_code"] = pd.Categorical(items["family"]).codes.astype("int16")
    item_meta["class"] = item_meta["class"].astype("int32")
    item_meta["perishable"] = item_meta["perishable"].astype("int8")

    return {
        "stores": stores,
        "items": items,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
        "test": test,
        "store_meta": store_meta,
        "item_meta": item_meta,
    }


def _accumulate_sum_count_tables(
    storage: dict[str, pd.DataFrame | None],
    name: str,
    frame: pd.DataFrame,
    keys: list[str],
) -> None:
    if frame.empty:
        return
    aggregated = frame.groupby(keys, observed=True)["target"].agg(sum="sum", count="size")
    storage[name] = aggregated if storage[name] is None else storage[name].add(aggregated, fill_value=0)


def _finalize_stat_table(
    table: pd.DataFrame | None,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame(columns=keys + [f"{prefix}_mean", f"{prefix}_count_log"])
    result = table.reset_index()
    result["count"] = result["count"].astype("int64")
    result[f"{prefix}_mean"] = (result["sum"] / result["count"]).astype("float32")
    result[f"{prefix}_count_log"] = np.log1p(result["count"]).astype("float32")
    return result[keys + [f"{prefix}_mean", f"{prefix}_count_log"]]


def _load_train_rows_between(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        data_dir=data_dir,
    ):
        if chunk["date"].max() < start_date:
            continue
        current = chunk[(chunk["date"] >= start_date) & (chunk["date"] <= end_date)].copy()
        if not current.empty:
            parts.append(current)

    frame = pd.concat(parts, ignore_index=True)
    frame["target"] = frame["unit_sales"].clip(lower=0).astype("float32")
    frame["onpromotion"] = frame["onpromotion"].fillna(False).astype("int8")
    frame["store_nbr"] = frame["store_nbr"].astype("int16")
    frame["item_nbr"] = frame["item_nbr"].astype("int32")
    return frame.drop(columns="unit_sales")


def _build_prior_aggregate_bundle(
    cutoff_date: pd.Timestamp,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    cache_path = _cache_path(f"prior_bundle_until_{cutoff_date:%Y%m%d}", data_dir=data_dir)
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    refs = _load_model_reference_assets(data_dir=data_dir)
    item_to_family_code = refs["item_meta"].set_index("item_nbr")["family_code"]

    storage: dict[str, pd.DataFrame | None] = {
        "si_all": None,
        "item_all": None,
        "store_all": None,
        "family_all": None,
        "family_store_all": None,
    }

    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        data_dir=data_dir,
    ):
        chunk = chunk[chunk["date"] < cutoff_date].copy()
        if chunk.empty:
            continue

        chunk["target"] = chunk["unit_sales"].clip(lower=0).astype("float32")
        chunk["family_code"] = chunk["item_nbr"].map(item_to_family_code).astype("int16")

        _accumulate_sum_count_tables(storage, "si_all", chunk, ["store_nbr", "item_nbr"])
        _accumulate_sum_count_tables(storage, "item_all", chunk, ["item_nbr"])
        _accumulate_sum_count_tables(storage, "store_all", chunk, ["store_nbr"])
        _accumulate_sum_count_tables(storage, "family_all", chunk, ["family_code"])
        _accumulate_sum_count_tables(storage, "family_store_all", chunk, ["family_code", "store_nbr"])

    bundle = {
        "si_all": _finalize_stat_table(storage["si_all"], ["store_nbr", "item_nbr"], "si_all"),
        "item_all": _finalize_stat_table(storage["item_all"], ["item_nbr"], "item_all"),
        "store_all": _finalize_stat_table(storage["store_all"], ["store_nbr"], "store_all"),
        "family_all": _finalize_stat_table(storage["family_all"], ["family_code"], "family_all"),
        "family_store_all": _finalize_stat_table(
            storage["family_store_all"],
            ["family_code", "store_nbr"],
            "family_store_all",
        ),
    }
    if use_cache:
        pd.to_pickle(bundle, cache_path)
    return bundle


def _attach_common_features(
    frame: pd.DataFrame,
    refs: dict[str, pd.DataFrame],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    include_transactions: bool,
) -> pd.DataFrame:
    current = frame.copy()
    current["weekday"] = current["date"].dt.dayofweek.astype("int8")
    current["day"] = current["date"].dt.day.astype("int8")
    current["month"] = current["date"].dt.month.astype("int8")
    current["weekofyear"] = current["date"].dt.isocalendar().week.astype("int16")
    current["is_month_end"] = current["date"].dt.is_month_end.astype("int8")
    current["is_payday"] = ((current["date"].dt.day == 15) | current["date"].dt.is_month_end).astype("int8")

    current = current.merge(refs["item_meta"], on="item_nbr", how="left")
    current = current.merge(refs["store_meta"], on="store_nbr", how="left")

    oil_frame = refs["oil"].drop_duplicates("date").set_index("date")
    oil_frame = oil_frame.reindex(pd.date_range(min_date, max_date, freq="D"))
    oil_frame["dcoilwtico"] = oil_frame["dcoilwtico"].ffill().bfill().astype("float32")
    oil_frame = oil_frame.rename_axis("date").reset_index()
    current = current.merge(oil_frame[["date", "dcoilwtico"]], on="date", how="left")

    holiday_frame = _build_store_date_holiday_features(
        stores=refs["stores"],
        holidays=refs["holidays"],
        min_date=min_date,
        max_date=max_date,
    )
    current = current.merge(holiday_frame, on=["date", "store_nbr"], how="left")
    for column in HOLIDAY_FLAG_COLUMNS:
        current[column] = current[column].fillna(0).astype("int8")

    if include_transactions:
        transactions = refs["transactions"][["date", "store_nbr", "transactions"]].copy()
        transactions["transactions"] = transactions["transactions"].astype("float32")
        current = current.merge(transactions, on=["date", "store_nbr"], how="left")

    return current


def _merge_recent_and_prior_features(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    prior_bundle: dict[str, pd.DataFrame],
    include_transactions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent_28_cutoff = fit["date"].max() - pd.Timedelta(days=27)
    recent_56_cutoff = fit["date"].max() - pd.Timedelta(days=55)

    agg_recent28 = (
        fit[fit["date"] >= recent_28_cutoff]
        .groupby(["store_nbr", "item_nbr"], observed=True)["target"]
        .mean()
        .rename("si_recent28")
        .reset_index()
    )
    agg_recent56 = (
        fit[fit["date"] >= recent_56_cutoff]
        .groupby(["store_nbr", "item_nbr"], observed=True)["target"]
        .mean()
        .rename("si_recent56")
        .reset_index()
    )
    agg_siw = (
        fit.groupby(["store_nbr", "item_nbr", "weekday"], observed=True)["target"]
        .agg(siw_mean="mean", siw_count="size")
        .reset_index()
    )
    agg_siw["siw_count_log"] = np.log1p(agg_siw["siw_count"]).astype("float32")
    agg_siw = agg_siw.drop(columns="siw_count")

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
    agg_item_recent = fit.groupby("item_nbr", observed=True)["target"].mean().rename("item_recent_mean").reset_index()
    agg_store_recent = (
        fit.groupby("store_nbr", observed=True)["target"].mean().rename("store_recent_mean").reset_index()
    )

    merge_specs = [
        (agg_recent28, ["store_nbr", "item_nbr"]),
        (agg_recent56, ["store_nbr", "item_nbr"]),
        (agg_siw, ["store_nbr", "item_nbr", "weekday"]),
        (agg_fsw, ["family_code", "store_nbr", "weekday"]),
        (agg_fw, ["family_code", "weekday"]),
        (agg_item_recent, ["item_nbr"]),
        (agg_store_recent, ["store_nbr"]),
        (prior_bundle["si_all"], ["store_nbr", "item_nbr"]),
        (prior_bundle["item_all"], ["item_nbr"]),
        (prior_bundle["store_all"], ["store_nbr"]),
        (prior_bundle["family_all"], ["family_code"]),
        (prior_bundle["family_store_all"], ["family_code", "store_nbr"]),
    ]

    for feature_frame, keys in merge_specs:
        fit = fit.merge(feature_frame, on=keys, how="left")
        valid = valid.merge(feature_frame, on=keys, how="left")

    numeric_fill = {
        "dcoilwtico": float(fit["dcoilwtico"].median()),
        "si_recent28": 0.0,
        "si_recent56": 0.0,
        "siw_mean": 0.0,
        "siw_count_log": 0.0,
        "fsw_mean": 0.0,
        "fw_mean": 0.0,
        "item_recent_mean": 0.0,
        "store_recent_mean": 0.0,
        "si_all_mean": 0.0,
        "si_all_count_log": 0.0,
        "item_all_mean": 0.0,
        "item_all_count_log": 0.0,
        "store_all_mean": 0.0,
        "family_all_mean": 0.0,
        "family_store_all_mean": 0.0,
        "family_store_all_count_log": 0.0,
    }
    if include_transactions:
        transaction_fill = float(fit["transactions"].median()) if fit["transactions"].notna().any() else 0.0
        numeric_fill["transactions"] = transaction_fill

    for current in [fit, valid]:
        for column, fill_value in numeric_fill.items():
            current[column] = current[column].fillna(fill_value).astype("float32")
        current["class"] = current["class"].astype("int32")
        current["perishable"] = current["perishable"].astype("int8")
        current["cluster"] = current["cluster"].astype("int16")
        current["family_code"] = current["family_code"].astype("int16")
        current["city_code"] = current["city_code"].astype("int16")
        current["state_code"] = current["state_code"].astype("int16")
        current["type_code"] = current["type_code"].astype("int16")
        current["store_code"] = current["store_code"].astype("int16")
        current["item_code"] = current["item_code"].astype("int32")

    return fit, valid


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


def build_rolling_origin_folds(
    data_dir: Path = DATA_DIR,
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
) -> pd.DataFrame:
    overview = build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]
    train_end = pd.Timestamp(overview["date_max"])
    latest_valid_start = train_end - pd.Timedelta(days=horizon_days - 1)

    rows: list[dict[str, Any]] = []
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
        return pd.read_pickle(cache_path)

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

    submission_path = data_dir / f"submission_lgbm_tscv_lb{lookback_days}_tx{tx_flag}.csv.gz"
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
