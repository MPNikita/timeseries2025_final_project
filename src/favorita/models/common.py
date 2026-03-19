from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.calendar import add_calendar_features
from ..features.external import attach_oil_feature, attach_transactions_feature
from ..features.holidays import HOLIDAY_FLAG_COLUMNS, _build_store_date_holiday_features
from ..io import read_train_chunks
from ..paths import DATA_DIR


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


def _attach_common_features(
    frame: pd.DataFrame,
    refs: dict[str, pd.DataFrame],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    include_transactions: bool,
) -> pd.DataFrame:
    """Attach shared metadata, calendar, holiday, oil and optional transactions."""
    current = add_calendar_features(frame)
    current = current.merge(refs["item_meta"], on="item_nbr", how="left")
    current = current.merge(refs["store_meta"], on="store_nbr", how="left")
    current = attach_oil_feature(current, refs["oil"], min_date=min_date, max_date=max_date)

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
        current = attach_transactions_feature(current, refs["transactions"])
    return current

