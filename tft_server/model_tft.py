from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import favorita_tft as ft

from .config_tft import TFTTrainConfig


def _best_score_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def build_tft_model(train_dataset: Any, config: TFTTrainConfig) -> Any:
    """Build `TemporalFusionTransformer` from prepared dataset + config."""
    ft._require_tft_dependencies()
    return ft.TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=float(config.learning_rate),
        hidden_size=int(config.hidden_size),
        attention_head_size=int(config.attention_head_size),
        hidden_continuous_size=int(config.hidden_continuous_size),
        dropout=float(config.dropout),
        loss=ft.RMSE(),
        log_interval=20,
    )


def load_state_dict_into_model(model: Any, state_dict_path: Path) -> dict[str, Any]:
    """Load `.pt`/`.ckpt` payload into TFT module as warm-start weights."""
    ft._require_tft_dependencies()
    payload = ft.torch.load(state_dict_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported state dict payload type: {type(payload)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "loaded_state_dict_path": str(state_dict_path),
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
    }


def fit_tft_model(
    dataset_bundle: dict[str, Any],
    config: TFTTrainConfig,
    run_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit TFT with checkpointing, TensorBoard logging and terminal progress bar."""
    ft._require_tft_dependencies()
    ft.pl.seed_everything(int(config.random_seed), workers=True)
    if hasattr(ft.torch, "set_float32_matmul_precision"):
        ft.torch.set_float32_matmul_precision("medium")

    train_dataset = dataset_bundle["train_dataset"]
    train_loader = dataset_bundle["train_loader"]
    val_loader = dataset_bundle["val_loader"]

    model = build_tft_model(train_dataset=train_dataset, config=config)

    init_summary: dict[str, Any] = {}
    if config.init_state_dict is not None:
        if not config.init_state_dict.exists():
            raise FileNotFoundError(f"init_state_dict not found: {config.init_state_dict}")
        init_summary = load_state_dict_into_model(model=model, state_dict_path=config.init_state_dict)

    run_ckpt_dir = config.checkpoints_root / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = ft.pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=int(config.early_stopping_patience),
        mode="min",
    )
    checkpoint_cb = ft.pl.callbacks.ModelCheckpoint(
        dirpath=str(run_ckpt_dir),
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    progress_bar = ft.pl.callbacks.TQDMProgressBar(refresh_rate=1)
    logger = ft.pl.loggers.TensorBoardLogger(save_dir=str(config.tensorboard_root), name=run_name)

    accelerator = "gpu" if ft.torch.cuda.is_available() else "cpu"
    trainer = ft.pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config.max_epochs),
        gradient_clip_val=float(config.gradient_clip_val),
        callbacks=[early_stopping, checkpoint_cb, progress_bar],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
    )

    ckpt_path = None
    if config.resume_ckpt is not None:
        if not config.resume_ckpt.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {config.resume_ckpt}")
        ckpt_path = str(config.resume_ckpt)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    best_model_path = checkpoint_cb.best_model_path
    last_model_path = str(run_ckpt_dir / "last.ckpt")
    best_val_loss = _best_score_to_float(checkpoint_cb.best_model_score)
    n_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    summary = {
        "accelerator": accelerator,
        "trained_epochs": int(trainer.current_epoch + 1),
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "tensorboard_log_dir": logger.log_dir,
        "n_parameters": n_parameters,
    } | init_summary
    return model, summary


def predict_tft_horizon(
    model: Any,
    dataset_bundle: dict[str, Any],
) -> Any:
    """Run TFT horizon prediction through shared mapping helper."""
    return ft._predict_tft_horizon(
        model=model,
        predict_loader=dataset_bundle["predict_loader"],
        horizon_frame=dataset_bundle["horizon_frame"],
        prediction_index=dataset_bundle["prediction_index"],
    )


def weighted_rmsle_on_frame(frame: Any) -> float:
    """Compute weighted RMSLE on validation frame with Favorita perishable weights."""
    current = frame.copy()
    current["weight"] = np.where(current["perishable"].eq(1), 1.25, 1.0).astype("float32")
    return float(
        ft.weighted_rmsle(
            y_true=current["target"],
            y_pred=current["final_prediction"],
            weights=current["weight"],
        )
    )

