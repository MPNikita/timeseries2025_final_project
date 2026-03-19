from __future__ import annotations

from typing import Any


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "force_row_wise": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 6.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.5,
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "grow_policy": "SymmetricTree",
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
}

NOTEBOOK_CATBOOST_SINGLE_FOLD_PARAMS: dict[str, Any] = {
    "iterations": 120,
    "depth": 8,
    "learning_rate": 0.05,
    "border_count": 128,
    "verbose": False,
}

DEFAULT_CATBOOST_POSTPROCESS_PARAMS: dict[str, Any] = {
    "history_scale": 3.0,
    "min_model_weight": 1.0,
    "unseen_model_weight": 1.0,
}


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
    "min_encoder_length": 28,
    "max_prediction_length": 16,
    "batch_size": 256,
    "min_history_points": 28,
    "max_series": None,
    "series_sample_seed": 42,
}
