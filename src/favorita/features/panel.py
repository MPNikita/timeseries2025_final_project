from __future__ import annotations

import pandas as pd


def _cross_join_pairs_and_dates(
    pairs: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build full daily panel for all pair combinations on [start_date, end_date]."""
    dates = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    panel = dates.merge(pairs, how="cross")
    panel["store_nbr"] = panel["store_nbr"].astype("int16")
    panel["item_nbr"] = panel["item_nbr"].astype("int32")
    return panel

