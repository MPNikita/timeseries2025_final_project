from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..favorita_tft import _build_prediction_frame

from .config_tft import TFTTrainConfig, build_parser, config_from_args, dump_config
from .dataset_tft import prepare_bundle
from .model_tft import fit_tft_model, predict_tft_horizon, weighted_rmsle_on_frame


def _save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _build_daily_validation(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("date", as_index=False)[["target", "final_prediction", "fallback_prediction", "tft_raw_prediction"]]
        .sum()
        .rename(columns={"target": "actual"})
        .sort_values("date", ignore_index=True)
    )


def _fit_or_skip(
    config: TFTTrainConfig,
    prepared: Any,
    run_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prepared.dataset_bundle is None:
        print("[train] No eligible series for TFT; using fallback-only predictions.")
        prediction_frame = _build_prediction_frame(prepared.horizon_frame, None)
        model_summary = {"status": "skipped_no_eligible_series"}
        return prediction_frame, model_summary

    model, model_summary = fit_tft_model(
        dataset_bundle=prepared.dataset_bundle,
        config=config,
        run_name=run_name,
    )
    tft_prediction = predict_tft_horizon(model=model, dataset_bundle=prepared.dataset_bundle)
    prediction_frame = _build_prediction_frame(prepared.horizon_frame, tft_prediction)
    return prediction_frame, model_summary


def run_pipeline(config: TFTTrainConfig) -> dict[str, Any]:
    config.ensure_dirs()
    run_name = config.resolved_run_name()
    run_output_dir = config.outputs_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    dump_config(config, run_output_dir / "config.json")
    prepared = prepare_bundle(config)

    prediction_frame, model_summary = _fit_or_skip(
        config=config,
        prepared=prepared,
        run_name=run_name,
    )

    payload: dict[str, Any] = {
        "run_name": run_name,
        "mode": config.mode,
        "model_summary": model_summary,
        "metadata": prepared.metadata,
        "output_dir": str(run_output_dir),
    }

    if config.mode == "fold":
        prediction_frame["weight"] = np.where(prediction_frame["perishable"].eq(1), 1.25, 1.0).astype("float32")
        score = weighted_rmsle_on_frame(prediction_frame)
        daily_validation = _build_daily_validation(prediction_frame)

        valid_path = run_output_dir / "valid_predictions.csv.gz"
        daily_path = run_output_dir / "daily_validation.csv"
        prediction_frame.to_csv(valid_path, index=False, compression="gzip")
        daily_validation.to_csv(daily_path, index=False)

        payload |= {
            "score_weighted_rmsle": float(score),
            "valid_predictions_path": str(valid_path),
            "daily_validation_path": str(daily_path),
        }
        print(f"[train] Fold weighted_rmsle = {score:.6f}")
    else:
        submission = (
            prediction_frame[["id", "final_prediction"]]
            .rename(columns={"final_prediction": "unit_sales"})
            .astype({"unit_sales": "float32"})
        )
        submission_path = run_output_dir / (
            f"submission_tft_lb{config.lookback_days}_enc{config.max_encoder_length}.csv.gz"
        )
        predictions_path = run_output_dir / "test_predictions.csv.gz"
        submission.to_csv(submission_path, index=False, compression="gzip")
        prediction_frame.to_csv(predictions_path, index=False, compression="gzip")

        prediction_summary = {
            "total_test_rows": int(len(prediction_frame)),
            "eligible_test_rows": int(prediction_frame["is_tft_eligible"].sum()),
            "rows_used_fallback": int(prediction_frame["used_fallback"].sum()),
            "unseen_item_rows": int(prediction_frame["unseen_item_flag"].sum()),
            "mean_final_prediction": float(prediction_frame["final_prediction"].mean()),
        }
        payload |= {
            "submission_path": str(submission_path),
            "prediction_frame_path": str(predictions_path),
            "prediction_summary": prediction_summary,
        }
        print("[train] Final submission saved:", submission_path)

    _save_json(payload, run_output_dir / "run_summary.json")
    print("[train] Run summary saved:", run_output_dir / "run_summary.json")
    print("[train] TensorBoard command:")
    print(f"tensorboard --logdir {config.tensorboard_root}")
    return payload


def main() -> None:
    parser = build_parser("Train/resume TFT on remote server with TensorBoard + tqdm.")
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
