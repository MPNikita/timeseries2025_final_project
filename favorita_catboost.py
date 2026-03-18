from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

from favorita_baselines import weighted_rmsle
from favorita_eda_utils import DATA_DIR, _cache_path, build_train_eda_bundle, read_train_chunks
from favorita_models import (
    CATEGORICAL_FEATURES,
    TS_BASE_FEATURES,
    _attach_common_features,
    _build_prior_aggregate_bundle,
    _load_model_reference_assets,
    build_rolling_origin_folds,
)


CATBOOST_DERIVED_FEATURES = [
    "is_weekend",
    "promo_family_gap",
    "promo_store_gap",
    "recent_trend_28_56",
    "recent_ratio_28_56",
    "item_vs_family_ratio",
    "family_store_vs_store_ratio",
    "item_recent_delta",
    "store_recent_delta",
    "history_strength",
]

CATBOOST_FEATURES = TS_BASE_FEATURES + CATBOOST_DERIVED_FEATURES

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 6.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.5,
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "grow_policy": "SymmetricTree",
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}

RECENT_CACHE_MIN_DATE = pd.Timestamp("2016-08-01")


def _load_recent_train_cache(
    min_date: pd.Timestamp = RECENT_CACHE_MIN_DATE,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_path(f"train_recent_from_{min_date:%Y%m%d}", data_dir=data_dir)
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    parts: list[pd.DataFrame] = []
    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        data_dir=data_dir,
    ):
        if chunk["date"].max() < min_date:
            continue
        current = chunk[chunk["date"] >= min_date].copy()
        if not current.empty:
            parts.append(current)

    frame = pd.concat(parts, ignore_index=True)
    frame["target"] = frame["unit_sales"].clip(lower=0).astype("float32")
    frame["onpromotion"] = frame["onpromotion"].fillna(False).astype("int8")
    frame["store_nbr"] = frame["store_nbr"].astype("int16")
    frame["item_nbr"] = frame["item_nbr"].astype("int32")
    frame = frame.drop(columns="unit_sales")
    if use_cache:
        pd.to_pickle(frame, cache_path)
    return frame


def _load_train_rows_between_recent_cache(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    recent = _load_recent_train_cache(data_dir=data_dir, use_cache=use_cache)
    return recent[recent["date"].between(start_date, end_date)].copy()


def _cross_join_pairs_and_dates(
    pairs: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    panel = dates.merge(pairs, how="cross")
    panel["store_nbr"] = panel["store_nbr"].astype("int16")
    panel["item_nbr"] = panel["item_nbr"].astype("int32")
    return panel


def _add_catboost_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame.copy()
    eps = np.float32(1.0)

    current["is_weekend"] = current["weekday"].isin([5, 6]).astype("int8")
    current["promo_family_gap"] = (current["fsw_mean"] - current["fw_mean"]).astype("float32")
    current["promo_store_gap"] = (current["family_store_all_mean"] - current["store_all_mean"]).astype("float32")
    current["recent_trend_28_56"] = (current["si_recent28"] - current["si_recent56"]).astype("float32")
    current["recent_ratio_28_56"] = (current["si_recent28"] / (current["si_recent56"] + eps)).astype("float32")
    current["item_vs_family_ratio"] = (current["item_all_mean"] / (current["family_all_mean"] + eps)).astype("float32")
    current["family_store_vs_store_ratio"] = (
        current["family_store_all_mean"] / (current["store_all_mean"] + eps)
    ).astype("float32")
    current["item_recent_delta"] = (current["item_recent_mean"] - current["item_all_mean"]).astype("float32")
    current["store_recent_delta"] = (current["store_recent_mean"] - current["store_all_mean"]).astype("float32")
    current["history_strength"] = (
        current["si_all_count_log"] + current["item_all_count_log"] + current["family_store_all_count_log"]
    ).astype("float32")
    return current


def _group_mean_count(
    frame: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True)["target"].agg(mean="mean", count="size").reset_index()
    grouped["mean"] = grouped["mean"].astype("float32")
    grouped["count"] = grouped["count"].astype("int32")
    return grouped


def _build_stat_feature_tables(fit: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    return {
        "agg_recent28": agg_recent28,
        "agg_recent56": agg_recent56,
        "agg_siw": agg_siw,
        "agg_fsw": agg_fsw,
        "agg_fw": agg_fw,
        "agg_item_recent": agg_item_recent,
        "agg_store_recent": agg_store_recent,
    }


def _apply_stat_feature_tables(
    frame: pd.DataFrame,
    stat_tables: dict[str, pd.DataFrame],
    prior_bundle: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    current = frame.copy()
    merge_specs = [
        (stat_tables["agg_recent28"], ["store_nbr", "item_nbr"]),
        (stat_tables["agg_recent56"], ["store_nbr", "item_nbr"]),
        (stat_tables["agg_siw"], ["store_nbr", "item_nbr", "weekday"]),
        (stat_tables["agg_fsw"], ["family_code", "store_nbr", "weekday"]),
        (stat_tables["agg_fw"], ["family_code", "weekday"]),
        (stat_tables["agg_item_recent"], ["item_nbr"]),
        (stat_tables["agg_store_recent"], ["store_nbr"]),
        (prior_bundle["si_all"], ["store_nbr", "item_nbr"]),
        (prior_bundle["item_all"], ["item_nbr"]),
        (prior_bundle["store_all"], ["store_nbr"]),
        (prior_bundle["family_all"], ["family_code"]),
        (prior_bundle["family_store_all"], ["family_code", "store_nbr"]),
    ]
    for feature_frame, keys in merge_specs:
        current = current.merge(feature_frame, on=keys, how="left")

    numeric_fill = {
        "dcoilwtico": float(current["dcoilwtico"].median()),
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
    return current


def _build_fallback_tables(fit: pd.DataFrame) -> dict[str, Any]:
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


def _recent_biased_sample(
    frame: pd.DataFrame,
    max_rows: int | None,
    keep_recent_days: int = 28,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame

    recent_cutoff = frame["date"].max() - pd.Timedelta(days=keep_recent_days - 1)
    recent = frame[frame["date"] >= recent_cutoff]
    older = frame[frame["date"] < recent_cutoff]
    if len(recent) >= max_rows:
        return recent.sample(max_rows, random_state=random_state).sort_values(
            ["date", "store_nbr", "item_nbr"],
            ignore_index=True,
        )

    remaining = max_rows - len(recent)
    sampled_older = older.sample(remaining, random_state=random_state)
    return pd.concat([recent, sampled_older], ignore_index=True).sort_values(
        ["date", "store_nbr", "item_nbr"],
        ignore_index=True,
    )


def _sample_implicit_zero_rows(
    fit_observed: pd.DataFrame,
    refs: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size <= 0 or end_date < start_date:
        return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "target", "onpromotion"])

    pair_universe = refs["test"][["store_nbr", "item_nbr"]].drop_duplicates().sort_values(
        ["store_nbr", "item_nbr"],
        ignore_index=True,
    )
    panel = _cross_join_pairs_and_dates(pair_universe, start_date=start_date, end_date=end_date)
    observed = fit_observed.loc[
        fit_observed["date"].between(start_date, end_date),
        ["date", "store_nbr", "item_nbr", "target", "onpromotion"],
    ]
    panel = panel.merge(observed, on=["date", "store_nbr", "item_nbr"], how="left")
    zero_rows = panel[panel["target"].isna()].copy()
    if zero_rows.empty:
        return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "target", "onpromotion"])

    sample_n = min(sample_size, len(zero_rows))
    zero_rows = zero_rows.sample(sample_n, random_state=random_state).reset_index(drop=True)
    zero_rows["target"] = np.float32(0.0)
    zero_rows["onpromotion"] = zero_rows["onpromotion"].astype("boolean").fillna(False).astype("int8")
    return zero_rows


def _prepare_catboost_fold_frames(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int = 16,
    fit_feature_rows: int | None = None,
    zero_sample_size: int = 0,
    zero_sample_days: int = 28,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
    fit_start = valid_start - pd.Timedelta(days=lookback_days)

    refs = _load_model_reference_assets(data_dir=data_dir)
    pair_universe = refs["test"][["store_nbr", "item_nbr"]].drop_duplicates().sort_values(
        ["store_nbr", "item_nbr"],
        ignore_index=True,
    )

    observed = _load_train_rows_between_recent_cache(
        start_date=fit_start,
        end_date=valid_end,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    fit_observed = observed[observed["date"] < valid_start].copy()
    valid_observed = observed[observed["date"] >= valid_start].copy()

    fit_common_full = _attach_common_features(
        fit_observed,
        refs=refs,
        min_date=fit_start,
        max_date=valid_start - pd.Timedelta(days=1),
        include_transactions=False,
    )

    valid_panel = _cross_join_pairs_and_dates(pair_universe, start_date=valid_start, end_date=valid_end)
    valid_panel = valid_panel.merge(
        valid_observed[["date", "store_nbr", "item_nbr", "target", "onpromotion"]],
        on=["date", "store_nbr", "item_nbr"],
        how="left",
    )
    valid_panel["target"] = valid_panel["target"].fillna(0.0).astype("float32")
    valid_panel["onpromotion"] = valid_panel["onpromotion"].fillna(False).astype("int8")
    valid_common = _attach_common_features(
        valid_panel,
        refs=refs,
        min_date=valid_start,
        max_date=valid_end,
        include_transactions=False,
    )

    prior_bundle = _build_prior_aggregate_bundle(
        cutoff_date=valid_start,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    stat_tables = _build_stat_feature_tables(fit_common_full)
    fallback_base = fit_common_full.copy()
    fit_common = _recent_biased_sample(fit_common_full, max_rows=fit_feature_rows)
    fit_frame = _apply_stat_feature_tables(fit_common, stat_tables=stat_tables, prior_bundle=prior_bundle)
    valid_frame = _apply_stat_feature_tables(valid_common, stat_tables=stat_tables, prior_bundle=prior_bundle)

    if zero_sample_size > 0:
        zero_start = max(fit_start, valid_start - pd.Timedelta(days=zero_sample_days))
        zero_rows = _sample_implicit_zero_rows(
            fit_observed=fit_observed,
            refs=refs,
            start_date=zero_start,
            end_date=valid_start - pd.Timedelta(days=1),
            sample_size=zero_sample_size,
        )
        if not zero_rows.empty:
            zero_rows_common = _attach_common_features(
                zero_rows,
                refs=refs,
                min_date=zero_start,
                max_date=valid_start - pd.Timedelta(days=1),
                include_transactions=False,
            )
            zero_rows = _apply_stat_feature_tables(
                zero_rows_common,
                stat_tables=stat_tables,
                prior_bundle=prior_bundle,
            )
            zero_rows = _add_catboost_derived_features(zero_rows)
            fit_frame = pd.concat([fit_frame, zero_rows], ignore_index=True)

    fit_frame = _add_catboost_derived_features(fit_frame)
    valid_frame = _add_catboost_derived_features(valid_frame)

    fallback_tables = _build_fallback_tables(fallback_base)
    fallback = build_hierarchical_fallback_predictions(valid_frame, fallback_tables)
    valid_frame["fallback_prediction"] = fallback["fallback_prediction"].astype("float32")
    valid_frame["fallback_source"] = fallback["fallback_source"]
    valid_frame["unseen_item_flag"] = valid_frame["item_all_count_log"].eq(0).astype("int8")

    metadata = {
        "fit_start": fit_start,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "fit_rows": int(len(fit_frame)),
        "fit_observed_rows": int(len(fit_common_full)),
        "valid_rows": int(len(valid_frame)),
        "valid_panel_pairs": int(len(pair_universe)),
        "zero_sample_size": int(zero_sample_size),
        "zero_sample_days": int(zero_sample_days),
    }
    return fit_frame, valid_frame, metadata


def _prepare_catboost_test_frames(
    lookback_days: int,
    fit_feature_rows: int | None = None,
    zero_sample_size: int = 0,
    zero_sample_days: int = 28,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    refs = _load_model_reference_assets(data_dir=data_dir)
    train_end = pd.Timestamp(build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]["date_max"])
    test_start = refs["test"]["date"].min()
    test_end = refs["test"]["date"].max()
    fit_start = train_end - pd.Timedelta(days=lookback_days)

    fit_observed = _load_train_rows_between_recent_cache(
        start_date=fit_start,
        end_date=train_end,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    fit_common_full = _attach_common_features(
        fit_observed,
        refs=refs,
        min_date=fit_start,
        max_date=train_end,
        include_transactions=False,
    )

    test_frame = refs["test"].copy()
    test_frame["onpromotion"] = test_frame["onpromotion"].fillna(False).astype("int8")
    test_common = _attach_common_features(
        test_frame,
        refs=refs,
        min_date=test_start,
        max_date=test_end,
        include_transactions=False,
    )

    prior_bundle = _build_prior_aggregate_bundle(
        cutoff_date=test_start,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    stat_tables = _build_stat_feature_tables(fit_common_full)
    fallback_base = fit_common_full.copy()
    fit_common = _recent_biased_sample(fit_common_full, max_rows=fit_feature_rows)
    fit_frame = _apply_stat_feature_tables(fit_common, stat_tables=stat_tables, prior_bundle=prior_bundle)
    test_frame = _apply_stat_feature_tables(test_common, stat_tables=stat_tables, prior_bundle=prior_bundle)

    if zero_sample_size > 0:
        zero_start = max(fit_start, train_end - pd.Timedelta(days=zero_sample_days - 1))
        zero_rows = _sample_implicit_zero_rows(
            fit_observed=fit_observed,
            refs=refs,
            start_date=zero_start,
            end_date=train_end,
            sample_size=zero_sample_size,
        )
        if not zero_rows.empty:
            zero_rows_common = _attach_common_features(
                zero_rows,
                refs=refs,
                min_date=zero_start,
                max_date=train_end,
                include_transactions=False,
            )
            zero_rows = _apply_stat_feature_tables(
                zero_rows_common,
                stat_tables=stat_tables,
                prior_bundle=prior_bundle,
            )
            zero_rows = _add_catboost_derived_features(zero_rows)
            fit_frame = pd.concat([fit_frame, zero_rows], ignore_index=True)

    fit_frame = _add_catboost_derived_features(fit_frame)
    test_frame = _add_catboost_derived_features(test_frame)

    fallback_tables = _build_fallback_tables(fallback_base)
    fallback = build_hierarchical_fallback_predictions(test_frame, fallback_tables)
    test_frame["fallback_prediction"] = fallback["fallback_prediction"].astype("float32")
    test_frame["fallback_source"] = fallback["fallback_source"]
    test_frame["unseen_item_flag"] = test_frame["item_all_count_log"].eq(0).astype("int8")

    metadata = {
        "fit_start": fit_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "lookback_days": lookback_days,
        "fit_rows": int(len(fit_frame)),
        "fit_observed_rows": int(len(fit_common_full)),
        "test_rows": int(len(test_frame)),
        "zero_sample_size": int(zero_sample_size),
        "zero_sample_days": int(zero_sample_days),
    }
    return fit_frame, test_frame, metadata


def _build_blended_prediction(
    frame: pd.DataFrame,
    model_prediction: np.ndarray,
    history_scale: float = 3.0,
    min_model_weight: float = 1.0,
    unseen_model_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(frame["item_all_count_log"], dtype=np.float32)
    fallback = np.asarray(frame["fallback_prediction"], dtype=np.float32)
    raw_model = np.asarray(model_prediction, dtype=np.float32)
    history_strength = np.clip(history / max(history_scale, 1e-3), 0.0, 1.0)
    model_weight = min_model_weight + (1.0 - min_model_weight) * history_strength
    unseen_mask = history <= 0
    model_weight[unseen_mask] = unseen_model_weight
    blended = model_weight * raw_model + (1.0 - model_weight) * fallback
    return np.clip(blended, 0, None), model_weight.astype("float32")


def _fit_catboost_model(
    fit_frame: pd.DataFrame,
    valid_eval_frame: pd.DataFrame | None,
    model_params: dict[str, Any],
) -> CatBoostRegressor:
    model = CatBoostRegressor(**model_params)
    fit_weights = np.where(fit_frame["perishable"].eq(1), 1.25, 1.0).astype("float32")

    fit_kwargs: dict[str, Any] = {
        "X": fit_frame[CATBOOST_FEATURES],
        "y": np.log1p(fit_frame["target"]),
        "sample_weight": fit_weights,
        "cat_features": CATEGORICAL_FEATURES,
    }
    if valid_eval_frame is not None and not valid_eval_frame.empty:
        fit_kwargs["eval_set"] = (
            valid_eval_frame[CATBOOST_FEATURES],
            np.log1p(valid_eval_frame["target"]),
        )
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 50
    model.fit(**fit_kwargs)
    return model


def run_single_fold_catboost_experiment(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int = 16,
    fit_max_rows: int | None = 4_000_000,
    eval_max_rows: int | None = 800_000,
    zero_sample_size: int = 400_000,
    zero_sample_days: int = 28,
    history_scale: float = 3.0,
    min_model_weight: float = 1.0,
    unseen_model_weight: float = 1.0,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_path = _cache_path(
        f"catboost_fold_{valid_start:%Y%m%d}_lb{lookback_days}_hz{horizon_days}_zr{zero_sample_size}_fr{fit_max_rows}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    base_fit_rows = None if fit_max_rows is None else max(fit_max_rows - zero_sample_size, 1)
    fit_frame, valid_frame, metadata = _prepare_catboost_fold_frames(
        valid_start=valid_start,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        fit_feature_rows=base_fit_rows,
        zero_sample_size=zero_sample_size,
        zero_sample_days=zero_sample_days,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    valid_eval = _recent_biased_sample(valid_frame, max_rows=eval_max_rows, keep_recent_days=horizon_days)

    params = DEFAULT_CATBOOST_PARAMS.copy()
    if model_params:
        params.update(model_params)

    model = _fit_catboost_model(
        fit_frame=fit_frame,
        valid_eval_frame=valid_eval,
        model_params=params,
    )

    raw_prediction = np.clip(np.expm1(model.predict(valid_frame[CATBOOST_FEATURES])), 0, None)
    final_prediction, model_weight = _build_blended_prediction(
        frame=valid_frame,
        model_prediction=raw_prediction,
        history_scale=history_scale,
        min_model_weight=min_model_weight,
        unseen_model_weight=unseen_model_weight,
    )

    valid_weights = np.where(valid_frame["perishable"].eq(1), 1.25, 1.0).astype("float32")
    score = weighted_rmsle(valid_frame["target"], final_prediction, valid_weights)
    fallback_score = weighted_rmsle(valid_frame["target"], valid_frame["fallback_prediction"], valid_weights)
    raw_model_score = weighted_rmsle(valid_frame["target"], raw_prediction, valid_weights)

    valid_predictions = valid_frame[
        [
            "date",
            "store_nbr",
            "item_nbr",
            "target",
            "perishable",
            "onpromotion",
            "fallback_prediction",
            "fallback_source",
            "unseen_item_flag",
            "item_all_count_log",
        ]
    ].copy().rename(columns={"target": "actual"})
    valid_predictions["weight"] = valid_weights
    valid_predictions["catboost_raw_prediction"] = raw_prediction.astype("float32")
    valid_predictions["model_weight"] = model_weight
    valid_predictions["prediction"] = final_prediction.astype("float32")

    daily_validation = (
        valid_predictions.groupby("date", as_index=False)[["actual", "fallback_prediction", "catboost_raw_prediction", "prediction"]]
        .sum()
        .sort_values("date", ignore_index=True)
    )
    feature_importance = pd.DataFrame(
        {
            "feature": CATBOOST_FEATURES,
            "importance": model.get_feature_importance(type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    result = {
        "metadata": metadata
        | {
            "fit_train_rows": int(len(fit_frame)),
            "valid_eval_rows": int(len(valid_eval)),
            "model_params": params,
            "history_scale": history_scale,
            "min_model_weight": min_model_weight,
            "unseen_model_weight": unseen_model_weight,
        },
        "score": score,
        "raw_model_score": raw_model_score,
        "fallback_score": fallback_score,
        "feature_importance": feature_importance,
        "daily_validation": daily_validation,
        "valid_predictions": valid_predictions,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_catboost_optuna_search(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 224,
    horizon_days: int = 16,
    fit_max_rows: int = 1_500_000,
    eval_max_rows: int = 600_000,
    zero_sample_size: int = 250_000,
    zero_sample_days: int = 28,
    n_trials: int = 8,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path(
        f"catboost_optuna_lb{lookback_days}_hz{horizon_days}_trials{n_trials}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    folds = build_rolling_origin_folds(data_dir=data_dir, horizon_days=horizon_days, step_days=28, n_folds=4)
    latest_fold = pd.Timestamp(folds.iloc[-1]["valid_start"])
    fit_frame, valid_frame, metadata = _prepare_catboost_fold_frames(
        valid_start=latest_fold,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        fit_feature_rows=max(fit_max_rows - zero_sample_size, 1),
        zero_sample_size=zero_sample_size,
        zero_sample_days=zero_sample_days,
        data_dir=data_dir,
        use_cache=use_cache,
    )
    valid_eval = _recent_biased_sample(valid_frame, max_rows=eval_max_rows, keep_recent_days=horizon_days)
    valid_weights = np.where(valid_frame["perishable"].eq(1), 1.25, 1.0).astype("float32")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = DEFAULT_CATBOOST_PARAMS.copy()
        params.update(
            {
                "iterations": trial.suggest_int("iterations", 400, 1100),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "depth": trial.suggest_int("depth", 6, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.01, 5.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
                "border_count": trial.suggest_int("border_count", 64, 255),
            }
        )
        history_scale = trial.suggest_float("history_scale", 1.5, 6.0)
        min_model_weight = trial.suggest_float("min_model_weight", 0.85, 1.0)
        unseen_model_weight = trial.suggest_float("unseen_model_weight", 0.0, 1.0)

        model = _fit_catboost_model(
            fit_frame=fit_frame,
            valid_eval_frame=valid_eval,
            model_params=params,
        )
        raw_prediction = np.clip(np.expm1(model.predict(valid_frame[CATBOOST_FEATURES])), 0, None)
        final_prediction, _ = _build_blended_prediction(
            frame=valid_frame,
            model_prediction=raw_prediction,
            history_scale=history_scale,
            min_model_weight=min_model_weight,
            unseen_model_weight=unseen_model_weight,
        )
        score = weighted_rmsle(valid_frame["target"], final_prediction, valid_weights)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("score", float(score))
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = DEFAULT_CATBOOST_PARAMS.copy()
    best_params.update(
        {
            key: study.best_params[key]
            for key in [
                "iterations",
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
                "border_count",
            ]
        }
    )
    result = {
        "metadata": metadata
        | {
            "fit_train_rows": int(len(fit_frame)),
            "valid_eval_rows": int(len(valid_eval)),
            "n_trials": n_trials,
            "latest_fold_valid_start": latest_fold,
        },
        "best_score": float(study.best_value),
        "best_model_params": best_params,
        "best_postprocess": {
            "history_scale": float(study.best_params["history_scale"]),
            "min_model_weight": float(study.best_params["min_model_weight"]),
            "unseen_model_weight": float(study.best_params["unseen_model_weight"]),
        },
        "trials": study.trials_dataframe().sort_values("value", ignore_index=True),
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_catboost_time_series_cv(
    data_dir: Path = DATA_DIR,
    lookback_grid: tuple[int, ...] = (168, 224),
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
    fit_max_rows: int | None = 4_000_000,
    eval_max_rows: int | None = 800_000,
    zero_sample_size: int = 400_000,
    zero_sample_days: int = 28,
    model_params: dict[str, Any] | None = None,
    postprocess_params: dict[str, float] | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path(
        f"catboost_tscv_lb{'-'.join(map(str, lookback_grid))}_hz{horizon_days}_step{step_days}_folds{n_folds}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    history_scale = 3.0 if postprocess_params is None else postprocess_params["history_scale"]
    min_model_weight = 1.0 if postprocess_params is None else postprocess_params["min_model_weight"]
    unseen_model_weight = 1.0 if postprocess_params is None else postprocess_params["unseen_model_weight"]

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
            result = run_single_fold_catboost_experiment(
                valid_start=pd.Timestamp(fold_row.valid_start),
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                fit_max_rows=fit_max_rows,
                eval_max_rows=eval_max_rows,
                zero_sample_size=zero_sample_size,
                zero_sample_days=zero_sample_days,
                history_scale=history_scale,
                min_model_weight=min_model_weight,
                unseen_model_weight=unseen_model_weight,
                data_dir=data_dir,
                use_cache=use_cache,
                force=force,
                model_params=model_params,
            )
            fold_rows.append(
                {
                    "fold": fold_row.fold,
                    "valid_start": pd.Timestamp(fold_row.valid_start),
                    "valid_end": pd.Timestamp(fold_row.valid_end),
                    "lookback_days": lookback_days,
                    "weighted_rmsle": result["score"],
                    "raw_model_rmsle": result["raw_model_score"],
                    "fallback_rmsle": result["fallback_score"],
                    "fit_train_rows": result["metadata"]["fit_train_rows"],
                    "valid_rows": result["metadata"]["valid_rows"],
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
    importance = pd.concat(importance_rows, ignore_index=True)
    importance_summary = (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False, ignore_index=True)
    )

    best_row = summary.iloc[0]
    result = {
        "metadata": {
            "lookback_grid": lookback_grid,
            "horizon_days": horizon_days,
            "step_days": step_days,
            "n_folds": n_folds,
            "fit_max_rows": fit_max_rows,
            "eval_max_rows": eval_max_rows,
            "zero_sample_size": zero_sample_size,
            "zero_sample_days": zero_sample_days,
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


def train_final_catboost_model(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 224,
    fit_max_rows: int | None = 8_000_000,
    zero_sample_size: int = 800_000,
    zero_sample_days: int = 28,
    model_params: dict[str, Any] | None = None,
    postprocess_params: dict[str, float] | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path(f"catboost_final_lb{lookback_days}_fr{fit_max_rows}_zr{zero_sample_size}", data_dir=data_dir)
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    base_fit_rows = None if fit_max_rows is None else max(fit_max_rows - zero_sample_size, 1)
    fit_frame, test_frame, metadata = _prepare_catboost_test_frames(
        lookback_days=lookback_days,
        fit_feature_rows=base_fit_rows,
        zero_sample_size=zero_sample_size,
        zero_sample_days=zero_sample_days,
        data_dir=data_dir,
        use_cache=use_cache,
    )

    params = DEFAULT_CATBOOST_PARAMS.copy()
    if model_params:
        params.update(model_params)
    model = _fit_catboost_model(fit_frame=fit_frame, valid_eval_frame=None, model_params=params)

    raw_prediction = np.clip(np.expm1(model.predict(test_frame[CATBOOST_FEATURES])), 0, None)
    current_postprocess = {
        "history_scale": 3.0,
        "min_model_weight": 1.0,
        "unseen_model_weight": 1.0,
    }
    if postprocess_params:
        current_postprocess.update(postprocess_params)
    final_prediction, model_weight = _build_blended_prediction(
        frame=test_frame,
        model_prediction=raw_prediction,
        history_scale=current_postprocess["history_scale"],
        min_model_weight=current_postprocess["min_model_weight"],
        unseen_model_weight=current_postprocess["unseen_model_weight"],
    )

    refs = _load_model_reference_assets(data_dir=data_dir)
    submission = refs["test"][["id"]].copy()
    submission["unit_sales"] = final_prediction.astype("float32")
    submission_path = data_dir / f"submission_catboost_optuna_lb{lookback_days}.csv.gz"
    submission.to_csv(submission_path, index=False, compression="gzip")

    prediction_frame = refs["test"][["id", "date", "store_nbr", "item_nbr"]].copy()
    prediction_frame["catboost_raw_prediction"] = raw_prediction.astype("float32")
    prediction_frame["unit_sales"] = final_prediction.astype("float32")
    prediction_frame["model_weight"] = model_weight
    prediction_frame["fallback_prediction"] = test_frame["fallback_prediction"].to_numpy(dtype="float32")
    prediction_frame["fallback_source"] = test_frame["fallback_source"].to_numpy()
    prediction_frame["unseen_item_flag"] = test_frame["unseen_item_flag"].to_numpy(dtype="int8")

    feature_importance = pd.DataFrame(
        {
            "feature": CATBOOST_FEATURES,
            "importance": model.get_feature_importance(type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    result = {
        "metadata": metadata
        | {
            "fit_train_rows": int(len(fit_frame)),
            "submission_path": str(submission_path),
            "model_params": params,
            "postprocess_params": current_postprocess,
        },
        "feature_importance": feature_importance,
        "submission_head": submission.head(10),
        "prediction_head": prediction_frame.head(10),
        "unseen_summary": prediction_frame.groupby("unseen_item_flag", as_index=False)["unit_sales"]
        .agg(count="size", mean_unit_sales="mean"),
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_full_catboost_pipeline(
    data_dir: Path = DATA_DIR,
    optuna_trials: int = 8,
    lookback_grid: tuple[int, ...] = (168, 224),
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 4,
    tuning_fit_max_rows: int = 1_500_000,
    tuning_eval_max_rows: int = 600_000,
    cv_fit_max_rows: int = 4_000_000,
    cv_eval_max_rows: int = 800_000,
    final_fit_max_rows: int = 8_000_000,
    tuning_zero_sample_size: int = 250_000,
    cv_zero_sample_size: int = 400_000,
    final_zero_sample_size: int = 800_000,
    zero_sample_days: int = 28,
    data_dir_override: Path | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    current_data_dir = data_dir_override or data_dir
    tuning = run_catboost_optuna_search(
        data_dir=current_data_dir,
        lookback_days=max(lookback_grid),
        horizon_days=horizon_days,
        fit_max_rows=tuning_fit_max_rows,
        eval_max_rows=tuning_eval_max_rows,
        zero_sample_size=tuning_zero_sample_size,
        zero_sample_days=zero_sample_days,
        n_trials=optuna_trials,
        use_cache=use_cache,
        force=force,
    )
    cv = run_catboost_time_series_cv(
        data_dir=current_data_dir,
        lookback_grid=lookback_grid,
        horizon_days=horizon_days,
        step_days=step_days,
        n_folds=n_folds,
        fit_max_rows=cv_fit_max_rows,
        eval_max_rows=cv_eval_max_rows,
        zero_sample_size=cv_zero_sample_size,
        zero_sample_days=zero_sample_days,
        model_params=tuning["best_model_params"],
        postprocess_params=tuning["best_postprocess"],
        use_cache=use_cache,
        force=force,
    )
    final = train_final_catboost_model(
        data_dir=current_data_dir,
        lookback_days=cv["metadata"]["best_lookback_days"],
        fit_max_rows=final_fit_max_rows,
        zero_sample_size=final_zero_sample_size,
        zero_sample_days=zero_sample_days,
        model_params=tuning["best_model_params"],
        postprocess_params=tuning["best_postprocess"],
        use_cache=use_cache,
        force=force,
    )
    return {"tuning": tuning, "cv": cv, "final": final}
