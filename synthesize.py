#!/usr/bin/env python3
"""Train a contrastive-loss VAE on tupled-survey.parquet and synthesize households.

Outputs synthesized-household-trips.parquet with row count equal to rounded sum of wthhfin.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tabular_io import StringRowWriter, read_rows
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


NUMERIC_BASES = {
    "hh_income_detailed",
    "hhsize",
    "numbicycles",
    "numvehicles",
    "numworkers",
    "age",
    "departure_time_in_minutes_after_arrival",
    "telecommute_days",
    "job_count",
    "person_tripcount",
    "reported_travel_time",
    "td_telecommute_time",
    "travelers_hh",
    "travelers_nonhh",
    "vehicle_occupancy",
    "walk_bike_loop_trips",
}


# These columns contain tuple-valued per-trip sequences in the tupled household table.
# Treating them as scalar numerics collapses each whole sequence to its mean, which
# destroys the time/trip structure that untuple.py later relies on.
TUPLED_NUMERIC_SEQUENCE_BASES = {
    "departure_time_in_minutes_after_arrival",
    "reported_travel_time",
    "travelers_hh",
    "travelers_nonhh",
    "vehicle_occupancy",
}


# These tuple-valued fields must be present for active people or downstream
# trip-time reconstruction will emit empty departure times.
REQUIRED_NON_EMPTY_TUPLED_FIELDS = {
    "departure_time_in_minutes_after_arrival",
    "reported_travel_time",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_string(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def is_numeric_column(col: str) -> bool:
    c = col.strip()
    if c == "wthhfin":
        return False
    if c in TUPLED_NUMERIC_SEQUENCE_BASES:
        return False
    if c in NUMERIC_BASES:
        return True
    if c.endswith("-age"):
        return True
    if c.endswith("-job_count"):
        return True
    if c.endswith("-person_tripcount"):
        return True
    if c.endswith("-td_telecommute_time"):
        return True
    if c.endswith("-walk_bike_loop_trips"):
        return True
    if any(c.endswith(f"-{base}") for base in TUPLED_NUMERIC_SEQUENCE_BASES):
        return False
    # user requested telecommute_days; tupled columns use j1_telecommute_days
    if c.endswith("-j1_telecommute_days") or c.endswith("-telecommute_days"):
        return True
    return False


def parse_numeric_value(raw: str) -> float:
    s = normalize_string(raw)
    if s == "":
        return float("nan")

    # Tuple-like values: use mean of all numeric tokens.
    if "(" in s or ")" in s or "," in s:
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if not nums:
            return float("nan")
        arr = np.array([float(x) for x in nums], dtype=np.float32)
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    try:
        return float(s)
    except ValueError:
        return float("nan")


def format_numeric(v: float) -> str:
    if not math.isfinite(v):
        return ""
    rounded_int = round(v)
    if abs(v - rounded_int) < 1e-6:
        return str(int(rounded_int))
    out = f"{v:.6f}".rstrip("0").rstrip(".")
    return out if out else "0"


@dataclass
class PreparedData:
    columns: List[str]
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    weights: np.ndarray
    numeric_matrix: np.ndarray
    numeric_means: np.ndarray
    numeric_stds: np.ndarray
    numeric_mins: np.ndarray
    numeric_maxs: np.ndarray
    categorical_matrix: np.ndarray
    cat_vocab_values: List[List[str]]
    cat_cardinalities: List[int]
    cat_non_empty_ids: List[np.ndarray]
    cat_non_empty_probs: List[np.ndarray]
    target_rows: int


def estimate_training_memory_gib(
    num_numeric: int,
    cat_cardinalities: Sequence[int],
    emb_dim: int,
    hidden_dim: int,
    latent_dim: int,
    batch_size: int,
) -> Tuple[float, float, float]:
    """Estimate memory footprint in GiB as (params+opt, activations, total)."""
    num_categorical = len(cat_cardinalities)
    input_dim = int(num_numeric) + int(num_categorical) * int(emb_dim)
    dec_out_dim = input_dim

    # Parameter count (embeddings + encoder + mu/logvar + decoder).
    emb_params = int(sum(int(card) * int(emb_dim) for card in cat_cardinalities))

    def linear_params(inp: int, out: int) -> int:
        return inp * out + out

    param_count = emb_params
    param_count += linear_params(input_dim, hidden_dim)
    param_count += linear_params(hidden_dim, hidden_dim)
    param_count += linear_params(hidden_dim, latent_dim)  # mu
    param_count += linear_params(hidden_dim, latent_dim)  # logvar
    param_count += linear_params(latent_dim, hidden_dim)
    param_count += linear_params(hidden_dim, hidden_dim)
    param_count += linear_params(hidden_dim, dec_out_dim)

    # Rough train-time bytes/param with AdamW: weights + grads + m + v.
    bytes_per_param_train = 16
    param_mem_bytes = float(param_count) * bytes_per_param_train

    # Rough activation memory for forward+backward (scaled by factor).
    act_units_per_sample = input_dim + (2 * hidden_dim) + latent_dim + dec_out_dim
    activation_factor = 3.0
    act_mem_bytes = float(batch_size) * float(act_units_per_sample) * 4.0 * activation_factor

    gib = 1024.0 ** 3
    param_gib = param_mem_bytes / gib
    act_gib = act_mem_bytes / gib
    total_gib = param_gib + act_gib
    return param_gib, act_gib, total_gib


def build_numeric_hist_targets(
    numeric_matrix: np.ndarray,
    weights: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_numeric = int(numeric_matrix.shape[1])
    mins = np.zeros(num_numeric, dtype=np.float32)
    maxs = np.ones(num_numeric, dtype=np.float32)
    targets = np.zeros((num_numeric, bins), dtype=np.float32)

    w = weights.astype(np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w = np.ones_like(w, dtype=np.float64)
        w_sum = float(w.sum())

    for j in range(num_numeric):
        col = numeric_matrix[:, j].astype(np.float64)
        mn = float(np.min(col))
        mx = float(np.max(col))
        if not math.isfinite(mn):
            mn = -1.0
        if not math.isfinite(mx):
            mx = 1.0
        if mx <= mn:
            mx = mn + 1.0
        hist, _ = np.histogram(col, bins=bins, range=(mn, mx), weights=w)
        total = float(hist.sum())
        if total <= 0:
            hist = np.ones(bins, dtype=np.float64) / float(bins)
        else:
            hist = hist / total
        mins[j] = mn
        maxs[j] = mx
        targets[j] = hist.astype(np.float32)

    return mins, maxs, targets


def soft_histogram_loss(
    rec_num: torch.Tensor,
    hist_mins: torch.Tensor,
    hist_maxs: torch.Tensor,
    hist_targets: torch.Tensor,
    sigma_scale: float,
) -> torch.Tensor:
    if rec_num.size(1) == 0:
        return rec_num.new_zeros(())

    bins = int(hist_targets.size(1))
    losses: List[torch.Tensor] = []
    for j in range(rec_num.size(1)):
        x = rec_num[:, j]
        mn = hist_mins[j]
        mx = hist_maxs[j]
        span = torch.clamp(mx - mn, min=1e-6)
        x = torch.clamp(x, mn, mx)
        centers = torch.linspace(mn, mx, bins, device=rec_num.device, dtype=rec_num.dtype)
        sigma = torch.clamp((span / max(bins, 1)) * max(sigma_scale, 1e-3), min=1e-4)

        # Differentiable histogram via Gaussian kernel assignment to bin centers.
        diff = (x.unsqueeze(1) - centers.unsqueeze(0)) / sigma
        soft_assign = torch.exp(-0.5 * diff * diff)
        pred = soft_assign.mean(dim=0)
        pred = pred / torch.clamp(pred.sum(), min=1e-8)

        target = hist_targets[j].to(dtype=pred.dtype)
        losses.append(F.mse_loss(pred, target))

    return torch.stack(losses).mean()


class MarginalNegativeSampler:
    def __init__(
        self,
        weights: np.ndarray,
        numeric_matrix: np.ndarray,
        categorical_matrix: np.ndarray,
        cat_cardinalities: Sequence[int],
    ) -> None:
        self.weights = weights / np.clip(weights.sum(), 1e-12, None)
        self.num_numeric = numeric_matrix.shape[1]
        self.num_categorical = categorical_matrix.shape[1]

        self.numeric_values: List[np.ndarray] = []
        self.numeric_probs: List[np.ndarray] = []
        for j in range(self.num_numeric):
            col = numeric_matrix[:, j]
            mask = np.isfinite(col)
            if not np.any(mask):
                self.numeric_values.append(np.array([0.0], dtype=np.float32))
                self.numeric_probs.append(np.array([1.0], dtype=np.float32))
                continue
            vals = col[mask]
            probs = self.weights[mask]
            probs = probs / np.clip(probs.sum(), 1e-12, None)
            self.numeric_values.append(vals.astype(np.float32))
            self.numeric_probs.append(probs.astype(np.float32))

        self.categorical_probs: List[np.ndarray] = []
        for j in range(self.num_categorical):
            card = int(cat_cardinalities[j])
            ids = categorical_matrix[:, j]
            probs = np.bincount(ids, weights=self.weights, minlength=card).astype(np.float64)
            if probs.sum() <= 0:
                probs[:] = 1.0
            probs /= probs.sum()
            self.categorical_probs.append(probs.astype(np.float32))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        num = np.zeros((batch_size, self.num_numeric), dtype=np.float32)
        cat = np.zeros((batch_size, self.num_categorical), dtype=np.int64)

        for j in range(self.num_numeric):
            vals = self.numeric_values[j]
            probs = self.numeric_probs[j]
            pick = np.random.choice(len(vals), size=batch_size, replace=True, p=probs)
            num[:, j] = vals[pick]

        for j in range(self.num_categorical):
            probs = self.categorical_probs[j]
            cat[:, j] = np.random.choice(len(probs), size=batch_size, replace=True, p=probs)

        return num, cat


class ContrastiveVAE(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Sequence[int],
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numeric = int(num_numeric)
        self.num_categorical = int(len(cat_cardinalities))
        self.emb_dim = int(emb_dim)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(card), emb_dim) for card in cat_cardinalities]
        )

        input_dim = self.num_numeric + self.num_categorical * emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        dec_out_dim = self.num_numeric + self.num_categorical * emb_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dec_out_dim),
        )

    def embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.num_categorical == 0:
            return x_cat.new_zeros((x_cat.size(0), 0), dtype=torch.float32)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        return torch.cat(embs, dim=1)

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x_num, self.embed_categorical(x_cat)], dim=1)
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(z)
        num = out[:, : self.num_numeric]
        cat = out[:, self.num_numeric :]
        if self.num_categorical > 0:
            cat = cat.view(-1, self.num_categorical, self.emb_dim)
        else:
            cat = cat.new_zeros((z.size(0), 0, self.emb_dim))
        return num, cat

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_num, x_cat)
        z = self.reparameterize(mu, logvar)
        rec_num, rec_cat = self.decode(z)
        return rec_num, rec_cat, mu, logvar, z


def prepare_data(input_csv: Path) -> PreparedData:
    columns, rows = read_rows(input_csv)
    if not columns:
        raise ValueError("Input table has no columns.")

    if "wthhfin" not in columns:
        raise ValueError("Input must include wthhfin column.")

    feature_columns = [c for c in columns if c != "wthhfin"]
    numeric_columns = [c for c in feature_columns if is_numeric_column(c)]
    categorical_columns = [c for c in feature_columns if c not in set(numeric_columns)]

    n = len(rows)
    weights = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(rows):
        try:
            w = float(normalize_string(r.get("wthhfin", "")) or "0")
        except ValueError:
            w = 0.0
        if not math.isfinite(w) or w < 0:
            w = 0.0
        weights[i] = w
    if weights.sum() <= 0:
        weights[:] = 1.0

    target_rows = int(round(float(weights.sum())))
    if target_rows <= 0:
        target_rows = len(rows)

    # Numeric matrix
    num_m = np.zeros((n, len(numeric_columns)), dtype=np.float32)
    for j, c in enumerate(numeric_columns):
        for i, r in enumerate(rows):
            num_m[i, j] = parse_numeric_value(r.get(c, ""))

    num_means = np.zeros(len(numeric_columns), dtype=np.float32)
    num_stds = np.ones(len(numeric_columns), dtype=np.float32)
    num_mins = np.zeros(len(numeric_columns), dtype=np.float32)
    num_maxs = np.zeros(len(numeric_columns), dtype=np.float32)
    for j in range(len(numeric_columns)):
        col = num_m[:, j]
        mask = np.isfinite(col)
        if not np.any(mask):
            mean = 0.0
            std = 1.0
            mn = 0.0
            mx = 0.0
            col[:] = 0.0
        else:
            finite = col[mask]
            mean = float(np.mean(finite))
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            col[~mask] = mean
            std = float(np.std(col))
            if std < 1e-6:
                std = 1.0
        num_means[j] = mean
        num_stds[j] = std
        num_mins[j] = mn
        num_maxs[j] = mx
        num_m[:, j] = (col - mean) / std

    # Categorical matrix
    cat_m = np.zeros((n, len(categorical_columns)), dtype=np.int64)
    cat_vocab_values: List[List[str]] = []
    cat_cardinalities: List[int] = []
    cat_non_empty_ids: List[np.ndarray] = []
    cat_non_empty_probs: List[np.ndarray] = []
    for j, c in enumerate(categorical_columns):
        vals = [normalize_string(r.get(c, "")) for r in rows]
        uniq = sorted(set(vals))
        if "" not in uniq:
            uniq = [""] + uniq
        stoi = {v: i for i, v in enumerate(uniq)}
        for i, v in enumerate(vals):
            cat_m[i, j] = stoi.get(v, 0)
        cat_vocab_values.append(uniq)
        cat_cardinalities.append(len(uniq))

        ids = cat_m[:, j]
        probs = np.bincount(ids, weights=weights, minlength=len(uniq)).astype(np.float64)
        non_empty = np.array([idx for idx, tok in enumerate(uniq) if tok != ""], dtype=np.int64)
        if non_empty.size == 0:
            cat_non_empty_ids.append(np.array([], dtype=np.int64))
            cat_non_empty_probs.append(np.array([], dtype=np.float64))
        else:
            p = probs[non_empty]
            if float(p.sum()) <= 0:
                p = np.ones_like(p, dtype=np.float64)
            p = p / float(p.sum())
            cat_non_empty_ids.append(non_empty)
            cat_non_empty_probs.append(p.astype(np.float64))

    return PreparedData(
        columns=columns,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        weights=weights,
        numeric_matrix=num_m,
        numeric_means=num_means,
        numeric_stds=num_stds,
        numeric_mins=num_mins,
        numeric_maxs=num_maxs,
        categorical_matrix=cat_m,
        cat_vocab_values=cat_vocab_values,
        cat_cardinalities=cat_cardinalities,
        cat_non_empty_ids=cat_non_empty_ids,
        cat_non_empty_probs=cat_non_empty_probs,
        target_rows=target_rows,
    )


def train(
    prepared: PreparedData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    emb_dim: int,
    hidden_dim: int,
    latent_dim: int,
    temperature: float,
    kl_beta: float,
    contrastive_weight: float,
    hist_bins: int,
    hist_loss_weight: float,
    hist_sigma_scale: float,
    grad_clip_norm: float,
    compile_model: bool,
) -> ContrastiveVAE:
    x_num = torch.tensor(prepared.numeric_matrix, dtype=torch.float32)
    x_cat = torch.tensor(prepared.categorical_matrix, dtype=torch.long)

    dataset = TensorDataset(x_num, x_cat)
    sample_weights = torch.tensor(prepared.weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    model = ContrastiveVAE(
        num_numeric=x_num.shape[1],
        cat_cardinalities=prepared.cat_cardinalities,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.1,
    ).to(device)

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    neg_sampler = MarginalNegativeSampler(
        weights=prepared.weights,
        numeric_matrix=prepared.numeric_matrix,
        categorical_matrix=prepared.categorical_matrix,
        cat_cardinalities=prepared.cat_cardinalities,
    )

    use_amp = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    hist_mins_np, hist_maxs_np, hist_targets_np = build_numeric_hist_targets(
        prepared.numeric_matrix,
        prepared.weights,
        bins=max(2, int(hist_bins)),
    )
    hist_mins = torch.tensor(hist_mins_np, dtype=torch.float32, device=device)
    hist_maxs = torch.tensor(hist_maxs_np, dtype=torch.float32, device=device)
    hist_targets = torch.tensor(hist_targets_np, dtype=torch.float32, device=device)

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training epochs", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        running = 0.0
        running_hist = 0.0
        n_batches = 0

        batch_iter = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=False,
            total=len(loader),
        )
        for bn, bc in batch_iter:
            bn = bn.to(device, non_blocking=True)
            bc = bc.to(device, non_blocking=True)

            neg_num_np, neg_cat_np = neg_sampler.sample(bn.size(0))
            neg_num = torch.tensor(neg_num_np, dtype=torch.float32, device=device)
            neg_cat = torch.tensor(neg_cat_np, dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                rec_num, rec_cat, mu, logvar, _ = model(bn, bc)

                # Per-sample numeric reconstruction distances.
                d_pos_num = ((rec_num - bn) ** 2).mean(dim=1)
                d_neg_num = ((rec_num - neg_num) ** 2).mean(dim=1)

                if bc.size(1) > 0:
                    target_pos_cat = torch.stack(
                        [emb(bc[:, i]) for i, emb in enumerate(model.cat_embeddings)],
                        dim=1,
                    )
                    target_neg_cat = torch.stack(
                        [emb(neg_cat[:, i]) for i, emb in enumerate(model.cat_embeddings)],
                        dim=1,
                    )
                    d_pos_cat = ((rec_cat - target_pos_cat) ** 2).mean(dim=(1, 2))
                    d_neg_cat = ((rec_cat - target_neg_cat) ** 2).mean(dim=(1, 2))
                else:
                    d_pos_cat = torch.zeros_like(d_pos_num)
                    d_neg_cat = torch.zeros_like(d_neg_num)

                d_pos = d_pos_num + d_pos_cat
                d_neg = d_neg_num + d_neg_cat

                # Contrastive objective: positive should be closer than negative.
                logits = (d_neg - d_pos) / max(temperature, 1e-6)
                ctr_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                rec_loss = d_pos.mean()
                # Keep KL in fp32 and clamp logvar to avoid exp overflow.
                mu32 = mu.float()
                logvar32 = logvar.float().clamp(min=-10.0, max=10.0)
                kl_loss = -0.5 * torch.mean(1 + logvar32 - mu32.pow(2) - logvar32.exp())
                hist_loss = soft_histogram_loss(
                    rec_num=rec_num,
                    hist_mins=hist_mins,
                    hist_maxs=hist_maxs,
                    hist_targets=hist_targets,
                    sigma_scale=hist_sigma_scale,
                )
                loss = (
                    rec_loss
                    + contrastive_weight * ctr_loss
                    + kl_beta * kl_loss
                    + hist_loss_weight * hist_loss
                )

            if not torch.isfinite(loss).item():
                batch_iter.set_postfix(loss="nan", hist="nan")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().item())
            running_hist += float(hist_loss.detach().item())
            n_batches += 1
            batch_iter.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                hist=f"{hist_loss.detach().item():.4f}",
            )

        avg = running / max(1, n_batches)
        avg_hist = running_hist / max(1, n_batches)
        epoch_iter.set_postfix(loss=f"{avg:.6f}", hist=f"{avg_hist:.6f}")

    return model


def decode_categorical_ids(model: ContrastiveVAE, rec_cat: torch.Tensor) -> torch.Tensor:
    # rec_cat: [B, C, emb_dim] -> ids [B, C]
    if rec_cat.size(1) == 0:
        return rec_cat.new_zeros((rec_cat.size(0), 0), dtype=torch.long)

    outs = []
    for j, emb in enumerate(model.cat_embeddings):
        q = rec_cat[:, j, :]  # [B, E]
        w = emb.weight  # [K, E]
        sim = F.linear(F.normalize(q, dim=1), F.normalize(w, dim=1))
        outs.append(torch.argmax(sim, dim=1))
    return torch.stack(outs, dim=1)


def synthesize(
    model: ContrastiveVAE,
    prepared: PreparedData,
    device: torch.device,
    output_csv: Path,
    sample_rows: int,
    sample_batch_size: int,
    p_empty_retain: float,
) -> None:
    model.eval()
    p_slot_re = re.compile(r"^P(\d+)-")

    num_cols = prepared.numeric_columns
    cat_cols = prepared.categorical_columns
    hhsize_num_idx = num_cols.index("hhsize") if "hhsize" in num_cols else -1
    num_p_slots: List[int] = []
    for c in num_cols:
        m = p_slot_re.match(c)
        num_p_slots.append(int(m.group(1)) if m else 0)
    cat_p_slots: List[int] = []
    for c in cat_cols:
        m = p_slot_re.match(c)
        cat_p_slots.append(int(m.group(1)) if m else 0)
    num_means = torch.tensor(prepared.numeric_means, dtype=torch.float32, device=device)
    num_stds = torch.tensor(prepared.numeric_stds, dtype=torch.float32, device=device)
    num_mins = torch.tensor(prepared.numeric_mins, dtype=torch.float32, device=device)
    num_maxs = torch.tensor(prepared.numeric_maxs, dtype=torch.float32, device=device)
    sample_probs = prepared.weights / np.clip(prepared.weights.sum(), 1e-12, None)

    with StringRowWriter(output_csv, prepared.columns, buffer_size=8192) as writer:
        with torch.no_grad():
            produced = 0
            sample_pbar = tqdm(total=sample_rows, desc="Synthesizing rows", unit="row")
            while produced < sample_rows:
                b = min(sample_batch_size, sample_rows - produced)
                src_idx = np.random.choice(len(sample_probs), size=b, replace=True, p=sample_probs)
                x_num = torch.tensor(prepared.numeric_matrix[src_idx], dtype=torch.float32, device=device)
                x_cat = torch.tensor(prepared.categorical_matrix[src_idx], dtype=torch.long, device=device)

                mu, logvar = model.encode(x_num, x_cat)
                z = model.reparameterize(mu, logvar)
                rec_num, rec_cat = model.decode(z)
                rec_num = rec_num * num_stds + num_means
                rec_num = torch.max(torch.min(rec_num, num_maxs), num_mins)
                rec_cat_ids = decode_categorical_ids(model, rec_cat)

                rec_num_np = rec_num.detach().cpu().numpy()
                rec_cat_np = rec_cat_ids.detach().cpu().numpy()
                if hhsize_num_idx >= 0:
                    hhsize_active = np.rint(rec_num_np[:, hhsize_num_idx]).astype(np.int64)
                    hhsize_active = np.clip(hhsize_active, 0, 8)
                else:
                    hhsize_active = np.full((b,), 8, dtype=np.int64)

                # Prevent empty collapse for tupled person/trip columns so untuple can emit trips.
                for j, c in enumerate(cat_cols):
                    slot = cat_p_slots[j]
                    if slot <= 0:
                        continue
                    non_empty_ids = prepared.cat_non_empty_ids[j]
                    non_empty_probs = prepared.cat_non_empty_probs[j]
                    if non_empty_ids.size == 0:
                        continue
                    empty_id = 0
                    active_mask = hhsize_active >= slot
                    empty_mask = (rec_cat_np[:, j] == empty_id) & active_mask
                    if np.any(empty_mask):
                        empty_idx = np.where(empty_mask)[0]
                        base_name = c.split("-", 1)[1] if "-" in c else c
                        retain_prob = 0.0 if base_name in REQUIRED_NON_EMPTY_TUPLED_FIELDS else float(
                            min(1.0, max(0.0, p_empty_retain))
                        )
                        retain = np.random.rand(empty_idx.size) < retain_prob
                        fill_idx = empty_idx[~retain]
                        if fill_idx.size > 0:
                            rec_cat_np[fill_idx, j] = np.random.choice(
                                non_empty_ids,
                                size=int(fill_idx.size),
                                replace=True,
                                p=non_empty_probs,
                            )

                for i in range(b):
                    out: Dict[str, str] = {c: "" for c in prepared.columns}
                    active_people = int(hhsize_active[i])

                    for j, c in enumerate(num_cols):
                        slot = num_p_slots[j]
                        if slot > 0 and slot > active_people:
                            out[c] = ""
                        else:
                            out[c] = format_numeric(float(rec_num_np[i, j]))

                    for j, c in enumerate(cat_cols):
                        slot = cat_p_slots[j]
                        if slot > 0 and slot > active_people:
                            out[c] = ""
                            continue
                        cid = int(rec_cat_np[i, j])
                        vocab = prepared.cat_vocab_values[j]
                        if cid < 0 or cid >= len(vocab):
                            cid = 0
                        out[c] = vocab[cid]

                    out["wthhfin"] = "1"
                    writer.write(out)

                produced += b
                sample_pbar.update(b)
            sample_pbar.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train contrastive VAE and synthesize household trips")
    p.add_argument("--input", type=Path, default=Path("tupled-survey.parquet"))
    p.add_argument("--output", type=Path, default=Path("synthesized-household-trips.parquet"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sample-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--emb-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=8192)
    p.add_argument("--latent-dim", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=1e-3)
    p.add_argument("--contrastive-weight", type=float, default=0.2)
    p.add_argument("--hist-bins", type=int, default=32)
    p.add_argument("--hist-loss-weight", type=float, default=0.05)
    p.add_argument("--hist-sigma-scale", type=float, default=0.5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument(
        "--p-empty-retain",
        type=float,
        default=0.2,
        help="For tupled P* categorical fields, fraction of empty predictions to keep (default: 0.2).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    prepared = prepare_data(args.input)
    print(f"rows={len(prepared.weights)} target_rows={prepared.target_rows}")
    print(
        f"features={len(prepared.feature_columns)} numeric={len(prepared.numeric_columns)} categorical={len(prepared.categorical_columns)}"
    )

    est_param_gib, est_act_gib, est_total_gib = estimate_training_memory_gib(
        num_numeric=len(prepared.numeric_columns),
        cat_cardinalities=prepared.cat_cardinalities,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )
    print(
        f"estimated_train_mem_gib total={est_total_gib:.2f} params+opt={est_param_gib:.2f} activations={est_act_gib:.2f}"
    )
    if device.type == "cuda":
        total_vram_gib = torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
        print(f"gpu_vram_gib total={total_vram_gib:.2f}")
        if est_total_gib > 0.9 * total_vram_gib:
            print(
                "WARNING: Estimated training memory is close to or above available GPU VRAM. "
                "Likely OOM; lower --hidden-dim, --latent-dim, or --batch-size."
            )

    model = train(
        prepared=prepared,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        temperature=args.temperature,
        kl_beta=args.kl_beta,
        contrastive_weight=args.contrastive_weight,
        hist_bins=args.hist_bins,
        hist_loss_weight=args.hist_loss_weight,
        hist_sigma_scale=args.hist_sigma_scale,
        grad_clip_norm=args.grad_clip_norm,
        compile_model=not args.no_compile,
    )

    synthesize(
        model=model,
        prepared=prepared,
        device=device,
        output_csv=args.output,
        sample_rows=prepared.target_rows,
        sample_batch_size=args.sample_batch_size,
        p_empty_retain=args.p_empty_retain,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
