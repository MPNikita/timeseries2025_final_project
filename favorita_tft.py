from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from favorita_baselines import weighted_rmsle
from favorita_catboost import (
    _build_fallback_tables,
    _cross_join_pairs_and_dates,
    _load_recent_train_cache,
    build_hierarchical_fallback_predictions,
)
from favorita_eda_utils import DATA_DIR, _cache_path, build_train_eda_bundle
from favorita_models import (
    _attach_common_features,
    _build_prior_aggregate_bundle,
    _load_model_reference_assets,
    build_rolling_origin_folds,
)


_TFT_IMPORT_ERROR: Exception | None = None
try:
    import torch

    try:
        import lightning.pytorch as pl
    except Exception:  # pragma: no cover - fallback path
        import pytorch_lightning as pl  # type: ignore[no-redef]

    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import RMSE
except Exception as exc:  # pragma: no cover - import guard for optional DL stack
    _TFT_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]
    TemporalFusionTransformer = None  # type: ignore[assignment]
    TimeSeriesDataSet = None  # type: ignore[assignment]
    GroupNormalizer = None  # type: ignore[assignment]
    NaNLabelEncoder = None  # type: ignore[assignment]
    RMSE = None  # type: ignore[assignment]


# TODO: Keep `torch`, `lightning` and `pytorch-forecasting` in requirements.txt for TFT runs.

STATIC_CATEGORICALS = [
    "store_code",
    "item_code",
    "family_code",
    "class",
    "perishable",
    "city_code",
    "state_code",
    "type_code",
    "cluster",
]

STATIC_REALS: list[str] = []

TIME_VARYING_KNOWN_CATEGORICALS = [
    "onpromotion",
    "weekday",
    "day",
    "month",
    "weekofyear",
    "is_month_end",
    "is_payday",
    "is_holiday",
    "is_event",
    "is_additional",
    "is_bridge",
    "is_work_day",
]

TIME_VARYING_KNOWN_REALS = ["dcoilwtico"]

TIME_VARYING_UNKNOWN_REALS = ["target_log"]


DEFAULT_TFT_PARAMS: dict[str, Any] = {
    "learning_rate": 1e-3,
    "hidden_size": 32,
    "attention_head_size": 4,
    "hidden_continuous_size": 16,
    "dropout": 0.1,
}

DEFAULT_TRAINER_PARAMS: dict[str, Any] = {
    "max_epochs": 8,
    "gradient_clip_val": 0.1,
    "early_stopping_patience": 3,
    "random_seed": 42,
    "num_workers": 0,
}

DEFAULT_TFT_DATASET_CONFIG: dict[str, Any] = {
    "max_encoder_length": 56,
    "max_prediction_length": 16,
    "batch_size": 256,
    "min_history_points": 28,
    "max_series": None,
    "series_sample_seed": 42,
}


def _require_tft_dependencies() -> None:
    """Raise a user-friendly error if TFT dependencies are missing."""
    if _TFT_IMPORT_ERROR is None:
        return
    raise ImportError(
        "favorita_tft requires optional packages: `torch`, `lightning` (or `pytorch-lightning`) "
        "and `pytorch-forecasting`. Install them, e.g. `pip install torch lightning pytorch-forecasting`."
    ) from _TFT_IMPORT_ERROR


def _params_digest(params: dict[str, Any]) -> str:
    """Build a short stable hash for cache keys."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def _safe_float32(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Fill missing values and cast to float32."""
    return series.fillna(fill_value).astype("float32")


def _cast_for_pytorch_forecasting_categoricals(
    fit_frame: pd.DataFrame,
    horizon_frame: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast categorical columns to string-backed pandas categories.

    `pytorch_forecasting` rejects numeric categoricals in `TimeSeriesDataSet`.
    We therefore convert all declared categorical features to string categories
    with a shared category vocabulary across fit/horizon frames.
    """
    fit_current = fit_frame.copy()
    horizon_current = horizon_frame.copy()

    fill_category = "0"
    for column in categorical_columns:
        fit_values = fit_current[column].fillna(fill_category).astype("string")
        horizon_values = horizon_current[column].fillna(fill_category).astype("string")
        categories = pd.Index(
            pd.concat(
                [fit_values, horizon_values, pd.Series([fill_category], dtype="string")],
                ignore_index=True,
            ).unique()
        )
        fit_current[column] = pd.Categorical(fit_values, categories=categories)
        horizon_current[column] = pd.Categorical(horizon_values, categories=categories)

    return fit_current, horizon_current


def _build_tft_base_frame(
    frame: pd.DataFrame,
    refs: dict[str, pd.DataFrame],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Attach shared metadata/calendar features and build TFT target columns.

    Leakage prevention:
    - only precomputed exogenous features are attached;
    - `transactions` is disabled to keep final inference identical to test-time constraints.
    """
    current = frame.copy()
    if "target" not in current.columns:
        current["target"] = np.float32(0.0)
    current["target"] = _safe_float32(current["target"], fill_value=0.0)
    current["onpromotion"] = current["onpromotion"].fillna(False).astype("int8")

    current = _attach_common_features(
        current,
        refs=refs,
        min_date=min_date,
        max_date=max_date,
        include_transactions=False,
    )

    current["target"] = _safe_float32(current["target"], fill_value=0.0)
    current["target_log"] = np.log1p(current["target"]).astype("float32")
    dcoil_fill = float(current["dcoilwtico"].median()) if current["dcoilwtico"].notna().any() else 0.0
    current["dcoilwtico"] = _safe_float32(current["dcoilwtico"], fill_value=dcoil_fill)

    categorical_columns = STATIC_CATEGORICALS + TIME_VARYING_KNOWN_CATEGORICALS
    for column in categorical_columns:
        current[column] = current[column].fillna(0).astype("int32")

    current["store_nbr"] = current["store_nbr"].astype("int16")
    current["item_nbr"] = current["item_nbr"].astype("int32")
    return current.sort_values(["date", "store_nbr", "item_nbr"], ignore_index=True)


def _add_time_idx_and_series_id(
    frame: pd.DataFrame,
    global_min_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add stable `time_idx` and `series_id` fields required by TimeSeriesDataSet."""
    current = frame.copy()
    current["time_idx"] = (current["date"] - global_min_date).dt.days.astype("int32")
    current["series_id"] = (current["store_nbr"].astype("int64") * 10_000_000 + current["item_nbr"].astype("int64")).astype(
        "int64"
    )
    return current


def _select_tft_eligible_pairs(
    fit_frame: pd.DataFrame,
    horizon_frame: pd.DataFrame,
    min_history_points: int,
    max_series: int | None,
    random_seed: int,
    seen_items: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select series that are eligible for TFT and route others to fallback.

    Eligibility logic:
    - enough observed history inside the fit window;
    - item seen before the horizon start;
    - optional series-level sampling via `max_series` (never row-level sampling).
    """
    horizon_pairs = horizon_frame[["store_nbr", "item_nbr", "series_id"]].drop_duplicates().copy()
    history_counts = (
        fit_frame.groupby(["store_nbr", "item_nbr", "series_id"], observed=True)
        .size()
        .rename("history_points")
        .reset_index()
    )
    eligible = horizon_pairs.merge(history_counts, on=["store_nbr", "item_nbr", "series_id"], how="left")
    eligible["history_points"] = eligible["history_points"].fillna(0).astype("int32")

    seen_item_set = seen_items if seen_items is not None else set(fit_frame["item_nbr"].astype("int32").unique().tolist())
    eligible["unseen_item_flag"] = (~eligible["item_nbr"].isin(seen_item_set)).astype("int8")
    eligible["enough_history_flag"] = eligible["history_points"].ge(max(int(min_history_points), 1)).astype("int8")
    eligible["is_tft_eligible"] = (
        eligible["enough_history_flag"].eq(1) & eligible["unseen_item_flag"].eq(0)
    ).astype("int8")
    eligible["sampled_out_flag"] = np.int8(0)

    if max_series is not None and max_series > 0:
        candidate = eligible.loc[eligible["is_tft_eligible"].eq(1), "series_id"]
        if len(candidate) > max_series:
            sampled = (
                eligible.loc[eligible["is_tft_eligible"].eq(1), "series_id"]
                .sample(max_series, random_state=random_seed)
                .to_numpy()
            )
            sampled_set = set(sampled.tolist())
            sampled_out_mask = eligible["is_tft_eligible"].eq(1) & ~eligible["series_id"].isin(sampled_set)
            eligible.loc[sampled_out_mask, "sampled_out_flag"] = np.int8(1)
            eligible.loc[sampled_out_mask, "is_tft_eligible"] = np.int8(0)

    summary = {
        "total_pairs": int(len(eligible)),
        "eligible_pairs": int(eligible["is_tft_eligible"].sum()),
        "fallback_pairs": int((eligible["is_tft_eligible"].eq(0)).sum()),
        "unseen_item_pairs": int(eligible["unseen_item_flag"].sum()),
        "insufficient_history_pairs": int(eligible["enough_history_flag"].eq(0).sum()),
        "sampled_out_pairs": int(eligible["sampled_out_flag"].sum()),
    }
    return eligible, summary


def _prepare_tft_fold_frames(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int,
    max_encoder_length: int,
    min_history_points: int,
    max_series: int | None,
    random_seed: int,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare fit/validation frames for one pseudo-public TFT fold."""
    cache_path = _cache_path(
        (
            f"tft_fold_frames_{valid_start:%Y%m%d}_lb{lookback_days}_hz{horizon_days}"
            f"_enc{max_encoder_length}_mh{min_history_points}_ms{max_series}_rs{random_seed}"
        ),
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
    fit_start = valid_start - pd.Timedelta(days=lookback_days)
    refs = _load_model_reference_assets(data_dir=data_dir)
    pair_universe = refs["test"][["store_nbr", "item_nbr"]].drop_duplicates().sort_values(
        ["store_nbr", "item_nbr"],
        ignore_index=True,
    )

    observed = _load_recent_train_cache(data_dir=data_dir, use_cache=use_cache)
    observed = observed[observed["date"].between(fit_start, valid_end)].copy()
    observed = observed.merge(pair_universe, on=["store_nbr", "item_nbr"], how="inner")

    fit_observed = observed[observed["date"] < valid_start].copy()
    valid_observed = observed[observed["date"] >= valid_start].copy()

    fit_frame = _build_tft_base_frame(
        fit_observed,
        refs=refs,
        min_date=fit_start,
        max_date=valid_start - pd.Timedelta(days=1),
    )

    valid_panel = _cross_join_pairs_and_dates(pair_universe, start_date=valid_start, end_date=valid_end)
    valid_panel = valid_panel.merge(
        valid_observed[["date", "store_nbr", "item_nbr", "target", "onpromotion"]],
        on=["date", "store_nbr", "item_nbr"],
        how="left",
    )
    valid_panel["target"] = valid_panel["target"].fillna(0.0).astype("float32")
    valid_panel["onpromotion"] = valid_panel["onpromotion"].fillna(False).astype("int8")

    valid_frame = _build_tft_base_frame(
        valid_panel,
        refs=refs,
        min_date=valid_start,
        max_date=valid_end,
    )

    fit_frame = _add_time_idx_and_series_id(fit_frame, global_min_date=fit_start)
    valid_frame = _add_time_idx_and_series_id(valid_frame, global_min_date=fit_start)

    prior_bundle = _build_prior_aggregate_bundle(cutoff_date=valid_start, data_dir=data_dir, use_cache=use_cache)
    seen_items = set(prior_bundle["item_all"]["item_nbr"].astype("int32").tolist())
    eligible_pairs, eligible_summary = _select_tft_eligible_pairs(
        fit_frame=fit_frame,
        horizon_frame=valid_frame,
        min_history_points=max(min_history_points, min(max_encoder_length, 7)),
        max_series=max_series,
        random_seed=random_seed,
        seen_items=seen_items,
    )

    fallback_tables = _build_fallback_tables(fit_frame)
    fallback = build_hierarchical_fallback_predictions(valid_frame, fallback_tables)
    valid_frame["fallback_prediction"] = fallback["fallback_prediction"].astype("float32")
    valid_frame["fallback_source"] = fallback["fallback_source"]

    valid_frame = valid_frame.merge(
        eligible_pairs[
            [
                "store_nbr",
                "item_nbr",
                "series_id",
                "history_points",
                "is_tft_eligible",
                "unseen_item_flag",
                "sampled_out_flag",
            ]
        ],
        on=["store_nbr", "item_nbr", "series_id"],
        how="left",
    )
    valid_frame["history_points"] = valid_frame["history_points"].fillna(0).astype("int32")
    valid_frame["is_tft_eligible"] = valid_frame["is_tft_eligible"].fillna(0).astype("int8")
    valid_frame["unseen_item_flag"] = valid_frame["unseen_item_flag"].fillna(0).astype("int8")
    valid_frame["sampled_out_flag"] = valid_frame["sampled_out_flag"].fillna(0).astype("int8")

    fit_eligible = fit_frame.merge(
        eligible_pairs.loc[eligible_pairs["is_tft_eligible"].eq(1), ["store_nbr", "item_nbr"]],
        on=["store_nbr", "item_nbr"],
        how="inner",
    )

    metadata = {
        "fit_start": fit_start,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "fit_observed_rows": int(len(fit_frame)),
        "fit_rows_eligible": int(len(fit_eligible)),
        "valid_rows": int(len(valid_frame)),
        "valid_panel_pairs": int(len(pair_universe)),
        "max_encoder_length": max_encoder_length,
        "min_history_points": min_history_points,
    } | eligible_summary

    payload = (fit_eligible, valid_frame, eligible_pairs, metadata)
    if use_cache:
        pd.to_pickle(payload, cache_path)
    return payload


def _prepare_tft_test_frames(
    lookback_days: int,
    max_encoder_length: int,
    min_history_points: int,
    max_series: int | None,
    random_seed: int,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare fit/test frames for final TFT training and Kaggle submission inference."""
    cache_path = _cache_path(
        f"tft_test_frames_lb{lookback_days}_enc{max_encoder_length}_mh{min_history_points}_ms{max_series}_rs{random_seed}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    refs = _load_model_reference_assets(data_dir=data_dir)
    train_end = pd.Timestamp(build_train_eda_bundle(data_dir=data_dir, use_cache=True)["overview"]["date_max"])
    test_start = refs["test"]["date"].min()
    test_end = refs["test"]["date"].max()
    fit_start = train_end - pd.Timedelta(days=lookback_days)

    pair_universe = refs["test"][["store_nbr", "item_nbr"]].drop_duplicates().sort_values(
        ["store_nbr", "item_nbr"],
        ignore_index=True,
    )

    observed = _load_recent_train_cache(data_dir=data_dir, use_cache=use_cache)
    observed = observed[observed["date"].between(fit_start, train_end)].copy()
    observed = observed.merge(pair_universe, on=["store_nbr", "item_nbr"], how="inner")

    fit_frame = _build_tft_base_frame(
        observed,
        refs=refs,
        min_date=fit_start,
        max_date=train_end,
    )

    test_frame = refs["test"][["id", "date", "store_nbr", "item_nbr", "onpromotion"]].copy()
    test_frame["target"] = np.float32(0.0)
    test_frame["onpromotion"] = test_frame["onpromotion"].fillna(False).astype("int8")
    test_frame = _build_tft_base_frame(test_frame, refs=refs, min_date=test_start, max_date=test_end)

    fit_frame = _add_time_idx_and_series_id(fit_frame, global_min_date=fit_start)
    test_frame = _add_time_idx_and_series_id(test_frame, global_min_date=fit_start)

    prior_bundle = _build_prior_aggregate_bundle(cutoff_date=test_start, data_dir=data_dir, use_cache=use_cache)
    seen_items = set(prior_bundle["item_all"]["item_nbr"].astype("int32").tolist())
    eligible_pairs, eligible_summary = _select_tft_eligible_pairs(
        fit_frame=fit_frame,
        horizon_frame=test_frame,
        min_history_points=max(min_history_points, min(max_encoder_length, 7)),
        max_series=max_series,
        random_seed=random_seed,
        seen_items=seen_items,
    )

    fallback_tables = _build_fallback_tables(fit_frame)
    fallback = build_hierarchical_fallback_predictions(test_frame, fallback_tables)
    test_frame["fallback_prediction"] = fallback["fallback_prediction"].astype("float32")
    test_frame["fallback_source"] = fallback["fallback_source"]

    test_frame = test_frame.merge(
        eligible_pairs[
            [
                "store_nbr",
                "item_nbr",
                "series_id",
                "history_points",
                "is_tft_eligible",
                "unseen_item_flag",
                "sampled_out_flag",
            ]
        ],
        on=["store_nbr", "item_nbr", "series_id"],
        how="left",
    )
    test_frame["history_points"] = test_frame["history_points"].fillna(0).astype("int32")
    test_frame["is_tft_eligible"] = test_frame["is_tft_eligible"].fillna(0).astype("int8")
    test_frame["unseen_item_flag"] = test_frame["unseen_item_flag"].fillna(0).astype("int8")
    test_frame["sampled_out_flag"] = test_frame["sampled_out_flag"].fillna(0).astype("int8")

    fit_eligible = fit_frame.merge(
        eligible_pairs.loc[eligible_pairs["is_tft_eligible"].eq(1), ["store_nbr", "item_nbr"]],
        on=["store_nbr", "item_nbr"],
        how="inner",
    )

    metadata = {
        "fit_start": fit_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "lookback_days": lookback_days,
        "fit_observed_rows": int(len(fit_frame)),
        "fit_rows_eligible": int(len(fit_eligible)),
        "test_rows": int(len(test_frame)),
        "max_encoder_length": max_encoder_length,
        "min_history_points": min_history_points,
    } | eligible_summary

    payload = (fit_eligible, test_frame, eligible_pairs, metadata)
    if use_cache:
        pd.to_pickle(payload, cache_path)
    return payload


def _build_tft_datasets(
    fit_frame: pd.DataFrame,
    horizon_frame: pd.DataFrame,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    """Build TimeSeriesDataSet objects and dataloaders for training/inference."""
    _require_tft_dependencies()

    fit_sorted = fit_frame.sort_values(["series_id", "time_idx"], ignore_index=True)
    horizon_sorted = horizon_frame.sort_values(["series_id", "time_idx"], ignore_index=True)
    categorical_columns = STATIC_CATEGORICALS + TIME_VARYING_KNOWN_CATEGORICALS
    fit_sorted, horizon_sorted = _cast_for_pytorch_forecasting_categoricals(
        fit_sorted,
        horizon_sorted,
        categorical_columns=categorical_columns,
    )
    inference_frame = pd.concat([fit_sorted, horizon_sorted], ignore_index=True).sort_values(
        ["series_id", "time_idx"],
        ignore_index=True,
    )

    max_encoder_length = int(dataset_config["max_encoder_length"])
    max_prediction_length = int(dataset_config["max_prediction_length"])
    batch_size = int(dataset_config["batch_size"])
    num_workers = int(dataset_config.get("num_workers", 0))

    categorical_fill: dict[str, str] = {}
    for column in TIME_VARYING_KNOWN_CATEGORICALS:
        observed = fit_sorted[column].astype("string")
        if observed.empty:
            categorical_fill[column] = "0"
            continue
        zero_present = observed.eq("0").any()
        categorical_fill[column] = "0" if zero_present else str(observed.iloc[0])

    constant_fill = {
        "target_log": 0.0,
        "dcoilwtico": float(fit_sorted["dcoilwtico"].median()) if fit_sorted["dcoilwtico"].notna().any() else 0.0,
    } | categorical_fill

    training = TimeSeriesDataSet(
        fit_sorted,
        time_idx="time_idx",
        target="target_log",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_length=1,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        allow_missing_timesteps=True,
        constant_fill_strategy=constant_fill,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        categorical_encoders={column: NaNLabelEncoder(add_nan=True) for column in categorical_columns},
    )

    internal_val = TimeSeriesDataSet.from_dataset(
        training,
        fit_sorted,
        predict=True,
        stop_randomization=True,
    )
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training,
        inference_frame,
        predict=True,
        stop_randomization=True,
    )

    pin_memory = bool(torch.cuda.is_available()) if torch is not None else False
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(num_workers > 0),
    }

    train_loader = training.to_dataloader(train=True, **dataloader_kwargs)
    val_loader = internal_val.to_dataloader(train=False, **dataloader_kwargs)
    predict_loader = prediction_dataset.to_dataloader(train=False, **dataloader_kwargs)

    prediction_index = None
    if hasattr(prediction_dataset, "decoded_index"):
        try:
            prediction_index = prediction_dataset.decoded_index.copy()
        except Exception:  # pragma: no cover - depends on pytorch-forecasting version
            prediction_index = None

    return {
        "train_dataset": training,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "predict_loader": predict_loader,
        "prediction_index": prediction_index,
        "horizon_frame": horizon_sorted,
    }


def _fit_tft_model(
    train_dataset: Any,
    train_loader: Any,
    val_loader: Any,
    tft_params: dict[str, Any],
    trainer_params: dict[str, Any],
    data_dir: Path = DATA_DIR,
) -> tuple[Any, dict[str, Any]]:
    """Fit TemporalFusionTransformer with deterministic settings and early stopping."""
    _require_tft_dependencies()

    seed = int(trainer_params["random_seed"])
    pl.seed_everything(seed, workers=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=int(trainer_params["early_stopping_patience"]),
        mode="min",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(trainer_params["max_epochs"]),
        gradient_clip_val=float(trainer_params["gradient_clip_val"]),
        callbacks=[early_stopping],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        default_root_dir=str(data_dir / ".cache" / "favorita"),
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=float(tft_params["learning_rate"]),
        hidden_size=int(tft_params["hidden_size"]),
        attention_head_size=int(tft_params["attention_head_size"]),
        hidden_continuous_size=int(tft_params["hidden_continuous_size"]),
        dropout=float(tft_params["dropout"]),
        loss=RMSE(),
        log_interval=-1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_val_loss = None
    if early_stopping.best_score is not None:
        best_val_loss = float(early_stopping.best_score.detach().cpu().item())

    model_summary = {
        "accelerator": accelerator,
        "trained_epochs": int(trainer.current_epoch + 1),
        "best_val_loss": best_val_loss,
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    return model, model_summary


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/list outputs to numpy arrays."""
    if isinstance(value, list):
        parts = [_to_numpy(part) for part in value]
        if not parts:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(parts, axis=0)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_prediction_matrix(payload: Any) -> np.ndarray:
    """Extract a 2D [n_series, horizon] matrix from TFT prediction payload."""
    if isinstance(payload, tuple):
        prediction = payload[0]
    elif hasattr(payload, "output"):
        prediction = payload.output
    else:
        prediction = payload

    matrix = _to_numpy(prediction)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim == 3:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(f"Unexpected TFT prediction shape: {matrix.shape}")
    return matrix.astype("float32")


def _predict_tft_horizon(
    model: Any,
    predict_loader: Any,
    horizon_frame: pd.DataFrame,
    prediction_index: pd.DataFrame | None,
) -> pd.DataFrame:
    """Predict `target_log` on the horizon and map outputs back to rows."""
    prediction_payload = model.predict(predict_loader, mode="prediction")
    prediction_matrix = _extract_prediction_matrix(prediction_payload)

    horizon_coords = horizon_frame[["date", "store_nbr", "item_nbr", "series_id", "time_idx"]].copy()
    horizon_coords = horizon_coords.sort_values(["series_id", "time_idx"], ignore_index=True)
    horizon_start = int(horizon_coords["time_idx"].min())
    horizon_steps = np.sort(horizon_coords["time_idx"].unique())

    n_series = prediction_matrix.shape[0]
    pred_horizon = prediction_matrix.shape[1]

    pair_order = horizon_coords[["series_id"]].drop_duplicates().sort_values("series_id", ignore_index=True)
    fallback_series_ids = pair_order["series_id"].to_numpy(dtype=np.int64)

    series_ids: np.ndarray
    start_idx: np.ndarray
    if isinstance(prediction_index, pd.DataFrame) and not prediction_index.empty and "series_id" in prediction_index.columns:
        ordered = prediction_index.copy()
        series_ids = ordered["series_id"].to_numpy(dtype=np.int64)

        start_col = None
        for candidate in ["time_idx_first_prediction", "decoder_time_idx", "time_idx"]:
            if candidate in ordered.columns:
                start_col = candidate
                break
        if start_col is None:
            start_idx = np.full(len(series_ids), horizon_start, dtype=np.int32)
        else:
            start_idx = ordered[start_col].to_numpy(dtype=np.int32)
            if start_idx.min() == horizon_start - 1:
                start_idx = start_idx + 1
    else:
        series_ids = fallback_series_ids
        start_idx = np.full(len(series_ids), horizon_start, dtype=np.int32)

    if len(series_ids) != n_series:
        if len(fallback_series_ids) != n_series:
            raise ValueError(
                f"TFT prediction/sample mismatch: model returned {n_series} series, mapping has {len(series_ids)}."
            )
        series_ids = fallback_series_ids
        start_idx = np.full(len(series_ids), horizon_start, dtype=np.int32)

    steps = np.arange(pred_horizon, dtype=np.int32)
    long_prediction = pd.DataFrame(
        {
            "series_id": np.repeat(series_ids, pred_horizon),
            "time_idx": np.repeat(start_idx, pred_horizon) + np.tile(steps, len(series_ids)),
            "predicted_target_log": prediction_matrix.reshape(-1),
        }
    )

    # Keep only requested horizon rows in case model produced shorter/longer decoders.
    long_prediction = long_prediction[long_prediction["time_idx"].isin(horizon_steps)].copy()
    merged = horizon_coords.merge(long_prediction, on=["series_id", "time_idx"], how="left")
    if merged["predicted_target_log"].notna().mean() < 0.5 and len(fallback_series_ids) == n_series:
        fallback_prediction = pd.DataFrame(
            {
                "series_id": np.repeat(fallback_series_ids, pred_horizon),
                "time_idx": np.full(n_series * pred_horizon, horizon_start, dtype=np.int32)
                + np.tile(np.arange(pred_horizon, dtype=np.int32), n_series),
                "predicted_target_log": prediction_matrix.reshape(-1),
            }
        )
        fallback_prediction = fallback_prediction[fallback_prediction["time_idx"].isin(horizon_steps)].copy()
        merged = horizon_coords.merge(fallback_prediction, on=["series_id", "time_idx"], how="left")
    merged["tft_raw_prediction"] = np.expm1(merged["predicted_target_log"].fillna(0.0)).clip(lower=0).astype("float32")
    return merged


def _build_prediction_frame(
    horizon_frame: pd.DataFrame,
    tft_prediction: pd.DataFrame | None,
) -> pd.DataFrame:
    """Blend TFT and hierarchical fallback according to eligibility flags."""
    current = horizon_frame.copy()
    current["tft_raw_prediction"] = np.nan
    if tft_prediction is not None and not tft_prediction.empty:
        current = current.merge(
            tft_prediction[["date", "store_nbr", "item_nbr", "tft_raw_prediction"]],
            on=["date", "store_nbr", "item_nbr"],
            how="left",
            suffixes=("", "_new"),
        )
        if "tft_raw_prediction_new" in current.columns:
            current["tft_raw_prediction"] = current["tft_raw_prediction_new"].astype("float32")
            current = current.drop(columns=["tft_raw_prediction_new"])

    use_tft = current["is_tft_eligible"].eq(1) & current["tft_raw_prediction"].notna()
    current["final_prediction"] = np.where(use_tft, current["tft_raw_prediction"], current["fallback_prediction"]).astype(
        "float32"
    )
    current["final_prediction"] = current["final_prediction"].clip(lower=0)
    current["used_fallback"] = (~use_tft).astype("int8")
    return current


def run_single_fold_tft_experiment(
    valid_start: pd.Timestamp,
    lookback_days: int,
    horizon_days: int = 16,
    max_encoder_length: int = 56,
    max_prediction_length: int = 16,
    max_series: int | None = None,
    min_history_points: int = 28,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    force: bool = False,
    tft_params: dict[str, Any] | None = None,
    trainer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one pseudo-public TFT fold and evaluate weighted RMSLE."""
    _require_tft_dependencies()

    model_cfg = DEFAULT_TFT_PARAMS.copy()
    if tft_params:
        model_cfg.update(tft_params)
    trainer_cfg = DEFAULT_TRAINER_PARAMS.copy()
    if trainer_params:
        trainer_cfg.update(trainer_params)

    dataset_cfg = DEFAULT_TFT_DATASET_CONFIG.copy()
    dataset_cfg.update(
        {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "max_series": max_series,
            "min_history_points": min_history_points,
            "series_sample_seed": int(trainer_cfg["random_seed"]),
            "num_workers": int(trainer_cfg.get("num_workers", DEFAULT_TRAINER_PARAMS["num_workers"])),
        }
    )

    cache_signature = _params_digest({"model": model_cfg, "trainer": trainer_cfg, "dataset": dataset_cfg})
    cache_path = _cache_path(
        (
            f"tft_fold_{valid_start:%Y%m%d}_lb{lookback_days}_hz{horizon_days}"
            f"_enc{max_encoder_length}_pred{max_prediction_length}_ms{max_series}_mh{min_history_points}_{cache_signature}"
        ),
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    fit_frame, valid_frame, _eligible_pairs, metadata = _prepare_tft_fold_frames(
        valid_start=valid_start,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        max_encoder_length=max_encoder_length,
        min_history_points=min_history_points,
        max_series=max_series,
        random_seed=int(dataset_cfg["series_sample_seed"]),
        data_dir=data_dir,
        use_cache=use_cache,
        force=force,
    )

    valid_eligible = valid_frame.loc[valid_frame["is_tft_eligible"].eq(1)].copy()
    tft_prediction = None
    model_summary: dict[str, Any] = {"status": "skipped_no_eligible_series"}

    if not fit_frame.empty and not valid_eligible.empty:
        dataset_bundle = _build_tft_datasets(
            fit_frame=fit_frame,
            horizon_frame=valid_eligible,
            dataset_config=dataset_cfg,
        )
        model, model_summary = _fit_tft_model(
            train_dataset=dataset_bundle["train_dataset"],
            train_loader=dataset_bundle["train_loader"],
            val_loader=dataset_bundle["val_loader"],
            tft_params=model_cfg,
            trainer_params=trainer_cfg,
            data_dir=data_dir,
        )
        tft_prediction = _predict_tft_horizon(
            model=model,
            predict_loader=dataset_bundle["predict_loader"],
            horizon_frame=dataset_bundle["horizon_frame"],
            prediction_index=dataset_bundle["prediction_index"],
        )

    valid_prediction_frame = _build_prediction_frame(valid_frame, tft_prediction)
    valid_prediction_frame["weight"] = np.where(valid_prediction_frame["perishable"].eq(1), 1.25, 1.0).astype("float32")

    score = weighted_rmsle(
        y_true=valid_prediction_frame["target"],
        y_pred=valid_prediction_frame["final_prediction"],
        weights=valid_prediction_frame["weight"],
    )

    daily_validation = (
        valid_prediction_frame.groupby("date", as_index=False)[
            ["target", "tft_raw_prediction", "fallback_prediction", "final_prediction"]
        ]
        .sum()
        .rename(columns={"target": "actual"})
        .sort_values("date", ignore_index=True)
    )

    eligibility_summary = pd.DataFrame(
        [
            {"metric": "total_pairs", "value": int(metadata["total_pairs"])},
            {"metric": "eligible_pairs", "value": int(metadata["eligible_pairs"])},
            {"metric": "fallback_pairs", "value": int(metadata["fallback_pairs"])},
            {"metric": "unseen_item_pairs", "value": int(metadata["unseen_item_pairs"])},
            {"metric": "insufficient_history_pairs", "value": int(metadata["insufficient_history_pairs"])},
            {"metric": "sampled_out_pairs", "value": int(metadata["sampled_out_pairs"])},
            {"metric": "eligible_rows", "value": int(valid_prediction_frame["is_tft_eligible"].sum())},
            {"metric": "rows_used_fallback", "value": int(valid_prediction_frame["used_fallback"].sum())},
        ]
    )

    valid_predictions = valid_prediction_frame[
        [
            "date",
            "store_nbr",
            "item_nbr",
            "target",
            "perishable",
            "onpromotion",
            "weight",
            "history_points",
            "is_tft_eligible",
            "sampled_out_flag",
            "unseen_item_flag",
            "tft_raw_prediction",
            "fallback_prediction",
            "fallback_source",
            "final_prediction",
            "used_fallback",
        ]
    ].copy()
    valid_predictions = valid_predictions.rename(columns={"target": "actual"})

    result = {
        "metadata": metadata
        | {
            "model_params": model_cfg,
            "trainer_params": trainer_cfg,
            "dataset_config": dataset_cfg,
        },
        "score": score,
        "daily_validation": daily_validation,
        "valid_predictions": valid_predictions,
        "eligibility_summary": eligibility_summary,
        "model_summary": model_summary,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def run_tft_backtest(
    data_dir: Path = DATA_DIR,
    lookback_grid: tuple[int, ...] = (168, 224),
    horizon_days: int = 16,
    step_days: int = 28,
    n_folds: int = 3,
    max_encoder_length: int = 56,
    max_prediction_length: int = 16,
    max_series: int | None = None,
    min_history_points: int = 28,
    use_cache: bool = True,
    force: bool = False,
    tft_params: dict[str, Any] | None = None,
    trainer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run TFT rolling-origin backtest across folds and lookback grid."""
    _require_tft_dependencies()

    model_cfg = DEFAULT_TFT_PARAMS.copy()
    if tft_params:
        model_cfg.update(tft_params)
    trainer_cfg = DEFAULT_TRAINER_PARAMS.copy()
    if trainer_params:
        trainer_cfg.update(trainer_params)

    signature = _params_digest(
        {
            "lookback_grid": lookback_grid,
            "horizon_days": horizon_days,
            "step_days": step_days,
            "n_folds": n_folds,
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "max_series": max_series,
            "min_history_points": min_history_points,
            "model": model_cfg,
            "trainer": trainer_cfg,
        }
    )
    cache_path = _cache_path(
        f"tft_backtest_{signature}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    folds = build_rolling_origin_folds(
        data_dir=data_dir,
        horizon_days=horizon_days,
        step_days=step_days,
        n_folds=n_folds,
    )

    rows: list[dict[str, Any]] = []
    for fold in folds.itertuples(index=False):
        for lookback_days in lookback_grid:
            fold_result = run_single_fold_tft_experiment(
                valid_start=pd.Timestamp(fold.valid_start),
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                max_series=max_series,
                min_history_points=min_history_points,
                data_dir=data_dir,
                use_cache=use_cache,
                force=force,
                tft_params=model_cfg,
                trainer_params=trainer_cfg,
            )
            rows.append(
                {
                    "fold": fold.fold,
                    "valid_start": pd.Timestamp(fold.valid_start),
                    "valid_end": pd.Timestamp(fold.valid_end),
                    "lookback_days": lookback_days,
                    "weighted_rmsle": float(fold_result["score"]),
                    "eligible_pairs": int(fold_result["metadata"]["eligible_pairs"]),
                    "fallback_pairs": int(fold_result["metadata"]["fallback_pairs"]),
                }
            )

    fold_scores = pd.DataFrame(rows).sort_values(["lookback_days", "valid_start"], ignore_index=True)
    summary = (
        fold_scores.groupby("lookback_days", as_index=False)["weighted_rmsle"]
        .agg(mean_score="mean", std_score="std", min_score="min", max_score="max")
        .sort_values("mean_score", ignore_index=True)
    )
    best_row = summary.iloc[0]

    result = {
        "metadata": {
            "lookback_grid": lookback_grid,
            "horizon_days": horizon_days,
            "step_days": step_days,
            "n_folds": n_folds,
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "max_series": max_series,
            "min_history_points": min_history_points,
            "best_lookback_days": int(best_row["lookback_days"]),
            "best_mean_score": float(best_row["mean_score"]),
            "model_params": model_cfg,
            "trainer_params": trainer_cfg,
        },
        "folds": folds,
        "fold_scores": fold_scores,
        "summary": summary,
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


def train_final_tft_model(
    data_dir: Path = DATA_DIR,
    lookback_days: int = 224,
    max_encoder_length: int = 56,
    max_prediction_length: int = 16,
    max_series: int | None = None,
    min_history_points: int = 28,
    use_cache: bool = True,
    force: bool = False,
    tft_params: dict[str, Any] | None = None,
    trainer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train final TFT model and generate Kaggle submission for `test.csv`."""
    _require_tft_dependencies()

    model_cfg = DEFAULT_TFT_PARAMS.copy()
    if tft_params:
        model_cfg.update(tft_params)
    trainer_cfg = DEFAULT_TRAINER_PARAMS.copy()
    if trainer_params:
        trainer_cfg.update(trainer_params)

    dataset_cfg = DEFAULT_TFT_DATASET_CONFIG.copy()
    dataset_cfg.update(
        {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "max_series": max_series,
            "min_history_points": min_history_points,
            "series_sample_seed": int(trainer_cfg["random_seed"]),
            "num_workers": int(trainer_cfg.get("num_workers", DEFAULT_TRAINER_PARAMS["num_workers"])),
        }
    )

    signature = _params_digest({"model": model_cfg, "trainer": trainer_cfg, "dataset": dataset_cfg})
    cache_path = _cache_path(
        f"tft_final_lb{lookback_days}_enc{max_encoder_length}_{signature}",
        data_dir=data_dir,
    )
    if use_cache and cache_path.exists() and not force:
        return pd.read_pickle(cache_path)

    fit_frame, test_frame, _eligible_pairs, metadata = _prepare_tft_test_frames(
        lookback_days=lookback_days,
        max_encoder_length=max_encoder_length,
        min_history_points=min_history_points,
        max_series=max_series,
        random_seed=int(dataset_cfg["series_sample_seed"]),
        data_dir=data_dir,
        use_cache=use_cache,
        force=force,
    )

    test_eligible = test_frame.loc[test_frame["is_tft_eligible"].eq(1)].copy()
    tft_prediction = None
    model_summary: dict[str, Any] = {"status": "skipped_no_eligible_series"}

    if not fit_frame.empty and not test_eligible.empty:
        dataset_bundle = _build_tft_datasets(
            fit_frame=fit_frame,
            horizon_frame=test_eligible,
            dataset_config=dataset_cfg,
        )
        model, model_summary = _fit_tft_model(
            train_dataset=dataset_bundle["train_dataset"],
            train_loader=dataset_bundle["train_loader"],
            val_loader=dataset_bundle["val_loader"],
            tft_params=model_cfg,
            trainer_params=trainer_cfg,
            data_dir=data_dir,
        )
        tft_prediction = _predict_tft_horizon(
            model=model,
            predict_loader=dataset_bundle["predict_loader"],
            horizon_frame=dataset_bundle["horizon_frame"],
            prediction_index=dataset_bundle["prediction_index"],
        )

    final_frame = _build_prediction_frame(test_frame, tft_prediction)
    submission = final_frame[["id", "final_prediction"]].rename(columns={"final_prediction": "unit_sales"}).copy()
    submission["unit_sales"] = submission["unit_sales"].astype("float32")
    submission_path = data_dir / f"submission_tft_lb{lookback_days}_enc{max_encoder_length}.csv.gz"
    submission.to_csv(submission_path, index=False, compression="gzip")

    prediction_summary = pd.DataFrame(
        [
            {"metric": "total_test_rows", "value": int(len(final_frame))},
            {"metric": "eligible_test_rows", "value": int(final_frame["is_tft_eligible"].sum())},
            {"metric": "rows_used_fallback", "value": int(final_frame["used_fallback"].sum())},
            {"metric": "unseen_item_rows", "value": int(final_frame["unseen_item_flag"].sum())},
            {"metric": "mean_final_prediction", "value": float(final_frame["final_prediction"].mean())},
        ]
    )

    result = {
        "metadata": metadata
        | {
            "model_params": model_cfg,
            "trainer_params": trainer_cfg,
            "dataset_config": dataset_cfg,
            "model_summary": model_summary,
            "submission_path": str(submission_path),
        },
        "submission_head": submission.head(10),
        "prediction_head": final_frame[
            [
                "id",
                "date",
                "store_nbr",
                "item_nbr",
                "tft_raw_prediction",
                "fallback_prediction",
                "final_prediction",
                "used_fallback",
                "unseen_item_flag",
            ]
        ].head(10),
        "prediction_summary": prediction_summary,
        "submission_path": str(submission_path),
    }
    if use_cache:
        pd.to_pickle(result, cache_path)
    return result


# Example usage:
# fold_result = run_single_fold_tft_experiment(
#     valid_start=pd.Timestamp("2017-07-26"),
#     lookback_days=224,
#     max_series=20_000,
# )
# final_result = train_final_tft_model(
#     lookback_days=224,
#     max_series=40_000,
# )
