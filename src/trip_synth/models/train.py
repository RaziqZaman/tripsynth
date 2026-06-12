from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from trip_synth.data.preprocessing import FittedPreprocessor, transform_weights
from trip_synth.data.schema import FeatureSchema
from trip_synth.data.splits import train_val_split
from trip_synth.utils.io import ensure_dir, write_json
from trip_synth.utils.progress import progress_bar, progress_iter
from trip_synth.utils.seed import set_seed

from .losses import infonce_loss, kl_divergence, reconstruction_loss
from .tabular_vae import MixedTabularVAE


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _make_positive(
    cat: torch.Tensor,
    num: torch.Tensor,
    model: MixedTabularVAE,
    preprocessor: FittedPreprocessor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.get("positive_aug", "light") == "none":
        return cat.clone(), num.clone()
    cat_pos = cat.clone()
    num_pos = num.clone()
    key_cols = {
        "o_tract_fips",
        "d_tract_fips",
        "tdate_week",
        "tdate_dow",
        "departure_time_minutes",
    }
    if cat_pos.numel():
        prob = float(config.get("positive_categorical_mask_prob", 0.03))
        for j, size in enumerate(model.category_sizes):
            col = preprocessor.categorical_columns[j]
            if col in key_cols or size <= 1:
                continue
            mask = torch.rand(cat_pos.shape[0], device=cat_pos.device) < prob
            draws = torch.randint(0, size, (cat_pos.shape[0],), device=cat_pos.device)
            cat_pos[:, j] = torch.where(mask, draws, cat_pos[:, j])
    if num_pos.numel():
        scale = float(config.get("positive_numeric_jitter", 0.02))
        for j, col in enumerate(preprocessor.numeric_columns):
            if col in key_cols:
                continue
            num_pos[:, j] = num_pos[:, j] + torch.randn_like(num_pos[:, j]) * scale
    return cat_pos, num_pos


def _make_negative(
    cat: torch.Tensor,
    num: torch.Tensor,
    model: MixedTabularVAE,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(config.get("negative_mode", "marginal"))
    cat_neg = cat.clone()
    num_neg = num.clone()
    if mode in {"shuffled", "hard_conditional"}:
        if cat_neg.shape[0] > 1:
            for j in range(cat_neg.shape[1]):
                perm = torch.randperm(cat_neg.shape[0], device=cat_neg.device)
                cat_neg[:, j] = cat_neg[perm, j]
            for j in range(num_neg.shape[1]):
                perm = torch.randperm(num_neg.shape[0], device=num_neg.device)
                num_neg[:, j] = num_neg[perm, j]
        return cat_neg, num_neg
    for j, size in enumerate(model.category_sizes):
        if size > 1:
            cat_neg[:, j] = torch.randint(0, size, (cat_neg.shape[0],), device=cat_neg.device)
    if num_neg.numel() and num_neg.shape[0] > 1:
        for j in range(num_neg.shape[1]):
            perm = torch.randperm(num_neg.shape[0], device=num_neg.device)
            num_neg[:, j] = num_neg[perm, j]
    return cat_neg, num_neg


def _tensorize(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "cat": torch.as_tensor(arrays["cat"], dtype=torch.long, device=device),
        "num": torch.as_tensor(arrays["num"], dtype=torch.float32, device=device),
        "weights": torch.as_tensor(arrays["weights"], dtype=torch.float32, device=device),
    }


def _evaluate(
    model: MixedTabularVAE,
    arrays: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        cat_logits, num_pred, mu, logvar, _ = model(arrays["cat"], arrays["num"])
        rec = reconstruction_loss(
            cat_logits,
            num_pred,
            arrays["cat"],
            arrays["num"],
            arrays["weights"],
            numeric_loss=str(config.get("numeric_loss", "mse")),
        )
        kl = kl_divergence(mu, logvar, arrays["weights"])
        loss = rec + float(config.get("beta_kl", 0.05)) * kl
    return {"loss": float(loss.detach().cpu()), "reconstruction": float(rec.cpu()), "kl": float(kl.cpu())}


def _save_history(history: list[dict[str, float]], output_dir: Path, method: str) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / f"{method}_training_loss.csv"
    fields = sorted({k for row in history for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    if history:
        plt.figure(figsize=(7, 4.5))
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["train_loss"] for row in history], label="train")
        plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{method} training curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{method}_training_loss_curve.png", dpi=200)
        plt.close()


def fit_vae_method(
    train_df: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    method: str = "contrastive_vae",
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)) if config.get("seed") is not None else None)
    output_dir = ensure_dir(output_dir)
    vae_cfg = dict(config.get("vae", {}))
    method_overrides = config.get("vae_by_method", {}).get(method, {})
    vae_cfg.update(method_overrides)
    if method == "noncontrastive_vae":
        vae_cfg["lambda_contrastive"] = 0.0
    vae_cfg.setdefault("lambda_contrastive", 0.5)
    vae_cfg.setdefault("beta_kl", 0.05)
    vae_cfg.setdefault("epochs", 10)
    vae_cfg.setdefault("batch_size", 256)

    train_part, val_part = train_val_split(train_df, seed=int(config.get("seed", 0)))
    preprocessor = FittedPreprocessor.fit(train_part, schema)
    weight_cfg = vae_cfg.get("weighting", {})
    if not weight_cfg.get("use_weight_for_training_loss", True):
        train_weights = np.ones(len(train_part), dtype="float32")
        val_weights = np.ones(len(val_part), dtype="float32")
    else:
        train_weights = transform_weights(train_part[schema.weight_column], weight_cfg)
        val_weights = transform_weights(val_part[schema.weight_column], weight_cfg)

    train_np = preprocessor.transform(train_part, train_weights)
    val_np = preprocessor.transform(val_part, val_weights)
    device = _device()
    train_t = _tensorize(train_np, device)
    val_t = _tensorize(val_np, device)

    model = MixedTabularVAE(preprocessor.category_sizes(), len(preprocessor.numeric_columns), vae_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(vae_cfg.get("lr", 0.001)))
    use_amp = device.type == "cuda" and bool(vae_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(int(config.get("seed", 0)))
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = int(vae_cfg.get("early_stopping_patience", 20))
    epochs_since_best = 0

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    write_json(preprocessor.to_dict(), output_dir / "preprocessor.json")
    write_json({"method": method, "vae": vae_cfg, "schema": schema.to_dict()}, output_dir / "model_config.json")

    total_epochs = int(vae_cfg["epochs"])
    training_budget_seconds = config.get("_training_time_budget_seconds")
    if training_budget_seconds is not None:
        total_epochs = int(vae_cfg.get("training_time_budget_max_epochs", max(total_epochs, 100000)))
    training_deadline = (
        time.monotonic() + float(training_budget_seconds)
        if training_budget_seconds is not None
        else None
    )
    stopped_by_training_time_budget = False
    with progress_bar(total_epochs, f"{method} training", unit="epoch") as epoch_bar:
        for epoch in range(1, total_epochs + 1):
            if training_deadline is not None and history and time.monotonic() >= float(training_deadline):
                stopped_by_training_time_budget = True
                break
            model.train()
            losses = []
            rec_losses = []
            kl_losses = []
            contrastive_losses = []
            batches = _batch_indices(len(train_part), int(vae_cfg["batch_size"]), rng)
            for batch in progress_iter(
                batches,
                desc=f"{method} epoch {epoch}/{total_epochs}",
                total=len(batches),
                unit="batch",
                leave=False,
            ):
                cat = train_t["cat"][batch]
                num = train_t["num"][batch]
                weights = train_t["weights"][batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    cat_logits, num_pred, mu, logvar, _ = model(cat, num)
                    rec = reconstruction_loss(
                        cat_logits,
                        num_pred,
                        cat,
                        num,
                        weights,
                        numeric_loss=str(vae_cfg.get("numeric_loss", "mse")),
                    )
                    kl = kl_divergence(mu, logvar, weights)
                    contrastive = cat.new_tensor(0.0, dtype=torch.float32)
                    lam = float(vae_cfg.get("lambda_contrastive", 0.0))
                    if lam > 0:
                        pos_cat, pos_num = _make_positive(cat, num, model, preprocessor, vae_cfg)
                        neg_cat, neg_num = _make_negative(cat, num, model, vae_cfg)
                        pos_mu, _ = model.encode(pos_cat, pos_num)
                        neg_mu, _ = model.encode(neg_cat, neg_num)
                        contrastive = infonce_loss(
                            mu,
                            pos_mu,
                            neg_mu,
                            temperature=float(vae_cfg.get("contrastive_temperature", 0.1)),
                        )
                    loss = rec + float(vae_cfg.get("beta_kl", 0.05)) * kl + lam * contrastive
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(vae_cfg.get("grad_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                rec_losses.append(float(rec.detach().cpu()))
                kl_losses.append(float(kl.detach().cpu()))
                contrastive_losses.append(float(contrastive.detach().cpu()))

            val = _evaluate(model, val_t, vae_cfg)
            row = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "train_reconstruction": float(np.mean(rec_losses)),
                "train_kl": float(np.mean(kl_losses)),
                "train_contrastive": float(np.mean(contrastive_losses)),
                "val_loss": val["loss"],
                "val_reconstruction": val["reconstruction"],
                "val_kl": val["kl"],
            }
            history.append(row)
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                epochs_since_best = 0
                torch.save(
                    {
                        "state_dict": best_state,
                        "config": vae_cfg,
                        "preprocessor": preprocessor.to_dict(),
                        "method": method,
                        "epoch": epoch,
                        "val_loss": best_loss,
                    },
                    output_dir / "best_checkpoint.pt",
                )
            else:
                epochs_since_best += 1

            if epoch % int(vae_cfg.get("checkpoint_every", 5)) == 0 or epoch == total_epochs:
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": vae_cfg,
                        "preprocessor": preprocessor.to_dict(),
                        "method": method,
                        "epoch": epoch,
                    },
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                )
            epoch_bar.set_postfix(
                train_loss=f"{row['train_loss']:.4f}",
                val_loss=f"{row['val_loss']:.4f}",
                best=f"{best_loss:.4f}",
            )
            epoch_bar.update(1)
            if training_deadline is not None and time.monotonic() >= float(training_deadline):
                stopped_by_training_time_budget = True
                break
            if training_deadline is None and epochs_since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _save_history(history, output_dir, method)
    return {
        "method": method,
        "model": model,
        "preprocessor": preprocessor,
        "config": vae_cfg,
        "history": history,
        "best_loss": best_loss,
        "output_dir": str(output_dir),
        "device": str(device),
        "stopped_by_training_time_budget": stopped_by_training_time_budget,
    }


def load_vae_artifact(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    preprocessor = FittedPreprocessor.from_dict(checkpoint["preprocessor"])
    model = MixedTabularVAE(
        preprocessor.category_sizes(),
        len(preprocessor.numeric_columns),
        checkpoint["config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "method": checkpoint.get("method", "contrastive_vae"),
        "model": model,
        "preprocessor": preprocessor,
        "config": checkpoint["config"],
        "history": [],
        "best_loss": checkpoint.get("val_loss"),
    }
