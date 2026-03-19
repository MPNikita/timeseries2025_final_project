"""Feature engineering building blocks shared by model pipelines."""

from .aggregates import (
    _accumulate_sum_count_tables,
    _apply_stat_feature_tables,
    _build_prior_aggregate_bundle,
    _build_stat_feature_tables,
    _finalize_stat_table,
    _merge_recent_and_prior_features,
)
from .calendar import add_calendar_features
from .external import attach_oil_feature, attach_transactions_feature
from .holidays import HOLIDAY_FLAG_COLUMNS, _build_store_date_holiday_features
from .metadata import _encode_metadata_codes
from .panel import _cross_join_pairs_and_dates

__all__ = [
    "_encode_metadata_codes",
    "add_calendar_features",
    "_build_store_date_holiday_features",
    "HOLIDAY_FLAG_COLUMNS",
    "attach_transactions_feature",
    "attach_oil_feature",
    "_cross_join_pairs_and_dates",
    "_accumulate_sum_count_tables",
    "_finalize_stat_table",
    "_build_prior_aggregate_bundle",
    "_build_stat_feature_tables",
    "_apply_stat_feature_tables",
    "_merge_recent_and_prior_features",
]
