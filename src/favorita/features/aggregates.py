from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import _cache_path
from ..io import _load_model_reference_assets, read_train_chunks
from ..paths import DATA_DIR


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


def _build_prior_aggregate_bundle(
    cutoff_date: pd.Timestamp,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build pre-cutoff aggregate priors used by LGBM/CatBoost/TFT."""
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


def _build_stat_feature_tables(fit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build recent-window aggregates used by CatBoost features."""
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
    """Attach precomputed stat feature tables to frame rows."""
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


def _merge_recent_and_prior_features(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    prior_bundle: dict[str, pd.DataFrame],
    include_transactions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge recent-window + prior aggregate features into fit/valid frames."""
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

