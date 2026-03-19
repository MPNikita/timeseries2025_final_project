from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import DATA_DIR


DEFAULT_CHUNKSIZE = 5_000_000

TRAIN_DTYPES = {
    "store_nbr": "int16",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "boolean",
}

TEST_DTYPES = {
    "store_nbr": "int16",
    "item_nbr": "int32",
    "onpromotion": "boolean",
}


def dataset_catalog(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.csv")):
        rows.append(
            {
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1024**2, 2),
                "loading_strategy": "chunked scan" if path.name == "train.csv" else "in-memory",
            }
        )
    return pd.DataFrame(rows).sort_values("size_mb", ascending=False, ignore_index=True)


def load_reference_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "stores": pd.read_csv(data_dir / "stores.csv"),
        "items": pd.read_csv(data_dir / "items.csv"),
        "transactions": pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"]),
        "oil": pd.read_csv(data_dir / "oil.csv", parse_dates=["date"]).sort_values("date"),
        "holidays": pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"]).sort_values("date"),
        "test": pd.read_csv(
            data_dir / "test.csv",
            usecols=["date", "store_nbr", "item_nbr", "onpromotion"],
            dtype=TEST_DTYPES,
            parse_dates=["date"],
        ),
    }


def read_train_chunks(
    usecols: list[str] | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    data_dir: Path = DATA_DIR,
):
    columns = usecols or ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    dtype = {column: TRAIN_DTYPES[column] for column in columns if column in TRAIN_DTYPES}
    parse_dates = ["date"] if "date" in columns else None
    return pd.read_csv(
        data_dir / "train.csv",
        usecols=columns,
        dtype=dtype or None,
        parse_dates=parse_dates,
        chunksize=chunksize,
    )


def _load_model_reference_assets(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Shared preloaded static assets for model pipelines."""
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

