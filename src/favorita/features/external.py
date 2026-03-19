from __future__ import annotations

import pandas as pd


def attach_transactions_feature(frame: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Attach transactions by (date, store_nbr)."""
    current = frame.copy()
    transaction_frame = transactions[["date", "store_nbr", "transactions"]].copy()
    transaction_frame["transactions"] = transaction_frame["transactions"].astype("float32")
    return current.merge(transaction_frame, on=["date", "store_nbr"], how="left")


def attach_oil_feature(
    frame: pd.DataFrame,
    oil: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Attach daily oil price with dense date reindex + ffill/bfill."""
    current = frame.copy()
    oil_frame = oil.drop_duplicates("date").set_index("date")
    oil_frame = oil_frame.reindex(pd.date_range(min_date, max_date, freq="D"))
    oil_frame["dcoilwtico"] = oil_frame["dcoilwtico"].ffill().bfill().astype("float32")
    oil_frame = oil_frame.rename_axis("date").reset_index()
    return current.merge(oil_frame[["date", "dcoilwtico"]], on="date", how="left")

