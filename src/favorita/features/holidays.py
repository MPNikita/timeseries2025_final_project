from __future__ import annotations

from typing import Any

import pandas as pd


HOLIDAY_FLAG_COLUMNS = [
    "is_holiday",
    "is_event",
    "is_additional",
    "is_bridge",
    "is_work_day",
]


def _build_store_date_holiday_features(
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build per-store holiday/event flags on a dense date grid."""
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
        full_index[column] = pair.max(axis=1).astype("int8")
        full_index = full_index.drop(columns=[f"{column}_nat", f"{column}_loc"])

    full_index["store_nbr"] = full_index["store_nbr"].astype("int16")
    return full_index

