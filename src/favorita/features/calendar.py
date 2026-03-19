from __future__ import annotations

import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar and business-day flags from a date column."""
    current = frame.copy()
    current["weekday"] = current[date_col].dt.dayofweek.astype("int8")
    current["day"] = current[date_col].dt.day.astype("int8")
    current["month"] = current[date_col].dt.month.astype("int8")
    current["weekofyear"] = current[date_col].dt.isocalendar().week.astype("int16")
    current["is_month_end"] = current[date_col].dt.is_month_end.astype("int8")
    current["is_payday"] = ((current[date_col].dt.day == 15) | current[date_col].dt.is_month_end).astype("int8")
    return current

