from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / ".cache" / "favorita"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


def _cache_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / ".cache" / "favorita" / f"{name}.pkl"


def build_train_eda_bundle(
    data_dir: Path = DATA_DIR,
    chunksize: int = DEFAULT_CHUNKSIZE,
    use_cache: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path("train_eda_bundle", data_dir=data_dir)
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    items = pd.read_csv(data_dir / "items.csv", usecols=["item_nbr", "family"])
    stores = pd.read_csv(data_dir / "stores.csv")
    item_to_family = items.set_index("item_nbr")["family"]

    stats = {
        "rows": 0,
        "missing_onpromotion": 0,
        "negative_sales": 0,
        "positive_sales": 0,
        "unit_sales_sum": 0.0,
    }
    date_min = None
    date_max = None
    store_ids: set[int] = set()
    item_ids: set[int] = set()
    daily_sales: pd.DataFrame | None = None
    family_sales: pd.DataFrame | None = None
    store_sales: pd.DataFrame | None = None
    promotion_summary = {
        False: {"records": 0, "unit_sales": 0.0},
        True: {"records": 0, "unit_sales": 0.0},
    }

    for chunk in read_train_chunks(
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        chunksize=chunksize,
        data_dir=data_dir,
    ):
        chunk_date_min = chunk["date"].min()
        chunk_date_max = chunk["date"].max()
        date_min = chunk_date_min if date_min is None else min(date_min, chunk_date_min)
        date_max = chunk_date_max if date_max is None else max(date_max, chunk_date_max)

        stats["rows"] += len(chunk)
        stats["missing_onpromotion"] += int(chunk["onpromotion"].isna().sum())
        stats["negative_sales"] += int((chunk["unit_sales"] < 0).sum())
        stats["positive_sales"] += int((chunk["unit_sales"] > 0).sum())
        stats["unit_sales_sum"] += float(chunk["unit_sales"].sum())

        store_ids.update(chunk["store_nbr"].unique().tolist())
        item_ids.update(chunk["item_nbr"].unique().tolist())

        daily_chunk = (
            pd.DataFrame(
                {
                    "date": chunk["date"],
                    "unit_sales": chunk["unit_sales"].astype("float64"),
                    "records": 1,
                    "promotion_true_records": chunk["onpromotion"].fillna(False).astype("int8"),
                    "promotion_known_records": chunk["onpromotion"].notna().astype("int8"),
                    "negative_sales_records": (chunk["unit_sales"] < 0).astype("int8"),
                }
            )
            .groupby("date", observed=True)
            .sum(numeric_only=True)
        )
        daily_sales = daily_chunk if daily_sales is None else daily_sales.add(daily_chunk, fill_value=0)

        family_chunk = (
            pd.DataFrame(
                {
                    "family": chunk["item_nbr"].map(item_to_family),
                    "unit_sales": chunk["unit_sales"].astype("float64"),
                    "records": 1,
                }
            )
            .groupby("family", observed=True)
            .sum(numeric_only=True)
        )
        family_sales = family_chunk if family_sales is None else family_sales.add(family_chunk, fill_value=0)

        store_chunk = chunk.groupby("store_nbr", observed=True)["unit_sales"].agg(unit_sales="sum", records="size")
        store_sales = store_chunk if store_sales is None else store_sales.add(store_chunk, fill_value=0)

        promo_chunk = (
            chunk.dropna(subset=["onpromotion"])[["onpromotion", "unit_sales"]]
            .groupby("onpromotion", observed=True)["unit_sales"]
            .agg(["size", "sum"])
        )
        for flag, row in promo_chunk.iterrows():
            promotion_summary[bool(flag)]["records"] += int(row["size"])
            promotion_summary[bool(flag)]["unit_sales"] += float(row["sum"])

    daily_sales = daily_sales.sort_index()
    for column in ["records", "promotion_true_records", "promotion_known_records", "negative_sales_records"]:
        daily_sales[column] = daily_sales[column].astype("int64")
    daily_sales["avg_unit_sales_per_record"] = daily_sales["unit_sales"] / daily_sales["records"]
    daily_sales["promotion_share_known"] = (
        daily_sales["promotion_true_records"] / daily_sales["promotion_known_records"]
    ).where(daily_sales["promotion_known_records"] > 0)

    family_sales = family_sales.sort_values("unit_sales", ascending=False).reset_index()
    family_sales["records"] = family_sales["records"].astype("int64")
    family_sales["avg_unit_sales_per_record"] = family_sales["unit_sales"] / family_sales["records"]
    family_sales["sales_share"] = family_sales["unit_sales"] / family_sales["unit_sales"].sum()

    store_sales = store_sales.reset_index()
    store_sales["records"] = store_sales["records"].astype("int64")
    store_sales["avg_unit_sales_per_record"] = store_sales["unit_sales"] / store_sales["records"]
    store_sales["sales_share"] = store_sales["unit_sales"] / store_sales["unit_sales"].sum()
    store_sales = store_sales.merge(stores, on="store_nbr", how="left").sort_values(
        "unit_sales",
        ascending=False,
        ignore_index=True,
    )

    promotion_frame = pd.DataFrame(
        [
            {
                "onpromotion": flag,
                "records": values["records"],
                "unit_sales": values["unit_sales"],
            }
            for flag, values in promotion_summary.items()
        ]
    ).sort_values("onpromotion", ignore_index=True)
    promotion_frame["record_share_known"] = promotion_frame["records"] / promotion_frame["records"].sum()
    promotion_frame["avg_unit_sales_per_record"] = (
        promotion_frame["unit_sales"] / promotion_frame["records"]
    ).where(promotion_frame["records"] > 0)

    overview = {
        "rows": int(stats["rows"]),
        "date_min": date_min,
        "date_max": date_max,
        "n_days": int(daily_sales.index.nunique()),
        "unique_stores": int(len(store_ids)),
        "unique_items": int(len(item_ids)),
        "missing_onpromotion_rate": stats["missing_onpromotion"] / stats["rows"],
        "negative_sales_rate": stats["negative_sales"] / stats["rows"],
        "positive_sales_rate": stats["positive_sales"] / stats["rows"],
        "total_unit_sales": stats["unit_sales_sum"],
    }

    bundle = {
        "overview": overview,
        "daily_sales": daily_sales,
        "family_sales": family_sales,
        "store_sales": store_sales,
        "promotion_summary": promotion_frame,
        "train_items": sorted(item_ids),
        "train_stores": sorted(store_ids),
    }
    if use_cache:
        pd.to_pickle(bundle, cache_path)
    return bundle


def build_test_coverage(
    test: pd.DataFrame,
    train_items: list[int],
    train_stores: list[int],
) -> pd.DataFrame:
    test_items = set(test["item_nbr"].unique())
    test_stores = set(test["store_nbr"].unique())
    train_item_set = set(train_items)
    train_store_set = set(train_stores)

    rows = [
        {"metric": "test_rows", "value": f"{len(test):,}"},
        {"metric": "test_start", "value": str(test["date"].min().date())},
        {"metric": "test_end", "value": str(test["date"].max().date())},
        {"metric": "test_unique_items", "value": f"{len(test_items):,}"},
        {"metric": "new_items_in_test", "value": f"{len(test_items - train_item_set):,}"},
        {"metric": "test_unique_stores", "value": f"{len(test_stores):,}"},
        {"metric": "new_stores_in_test", "value": f"{len(test_stores - train_store_set):,}"},
    ]
    return pd.DataFrame(rows)


def build_daily_external_frame(
    daily_sales: pd.DataFrame,
    transactions: pd.DataFrame,
    oil: pd.DataFrame,
) -> pd.DataFrame:
    daily = daily_sales.copy()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily["rolling_28d"] = daily["unit_sales"].rolling(28, min_periods=7).mean()
    daily["weekday"] = daily.index.day_name()
    daily["month"] = daily.index.month_name().str[:3]
    daily["is_payday"] = (daily.index.day == 15) | daily.index.is_month_end

    transactions_daily = transactions.groupby("date", as_index=True)["transactions"].sum().rename("transactions")
    oil_daily = oil.drop_duplicates("date").sort_values("date").set_index("date")
    oil_daily = oil_daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    oil_daily["dcoilwtico"] = oil_daily["dcoilwtico"].ffill().bfill()

    daily = daily.join(transactions_daily, how="left").join(oil_daily[["dcoilwtico"]], how="left")
    return daily
