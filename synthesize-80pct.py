#!/usr/bin/env python3
"""Train on 80% of tupled households, synthesize households, and immediately untuple them."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from holdout_split import split_rows_by_household
from synthesize import (
    PreparedData,
    decode_categorical_ids,
    estimate_training_memory_gib,
    format_numeric,
    is_numeric_column,
    parse_numeric_value,
    set_seed,
)
from tabular_io import StringRowWriter, read_rows
from untuple import untuple_households

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


ID_COL_RE = re.compile(r"(^id$|_id$|^id_|^tripid$|^tripno$|^persno$)", re.IGNORECASE)
INT_EPS = 1e-5


def normalize_string(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def is_missing_value(v: object) -> bool:
    return normalize_string(v) == ""


def is_id_column(col: str) -> bool:
    c = col.strip()
    return bool(ID_COL_RE.search(c))


@dataclass
class PreparedSplit:
    prepared: PreparedData
    holdout_rows: List[Dict[str, object]]
    dropped_training_rows: int
    critical_non_null_columns: List[str]
    integer_numeric_columns: set[str]
    train_households: int
    holdout_households: int


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
            vals = numeric_matrix[:, j].astype(np.float32)
            self.numeric_values.append(vals)
            self.numeric_probs.append(self.weights.astype(np.float32))

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
            probs = self.numeric_probs[j]
            pick = np.random.choice(len(probs), size=batch_size, replace=True, p=probs)
            num[:, j] = self.numeric_values[j][pick]

        for j in range(self.num_categorical):
            probs = self.categorical_probs[j]
            cat[:, j] = np.random.choice(len(probs), size=batch_size, replace=True, p=probs)

        return num, cat


class ContrastiveOnlyAutoencoder(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Sequence[int],
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int,
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
        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for _ in range(max(1, int(hidden_layers))):
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for _ in range(max(1, int(hidden_layers))):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.num_categorical == 0:
            return x_cat.new_zeros((x_cat.size(0), 0), dtype=torch.float32)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        return torch.cat(embs, dim=1)

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.embed_categorical(x_cat)], dim=1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(z)
        num = out[:, : self.num_numeric]
        cat = out[:, self.num_numeric :]
        if self.num_categorical > 0:
            cat = cat.view(-1, self.num_categorical, self.emb_dim)
        else:
            cat = cat.new_zeros((z.size(0), 0, self.emb_dim))
        return num, cat


def detect_integer_numeric_columns(
    rows: Sequence[Dict[str, object]],
    numeric_columns: Sequence[str],
    threshold: float,
) -> set[str]:
    integer_cols: set[str] = set()
    for col in tqdm(
        numeric_columns,
        desc="Detecting integer-like numeric columns",
        unit="col",
    ):
        total = 0
        almost_integer = 0
        for row in rows:
            x = parse_numeric_value(row.get(col, ""))
            if not math.isfinite(x):
                continue
            total += 1
            if abs(x - round(x)) <= INT_EPS:
                almost_integer += 1
        if total > 0 and (almost_integer / total) >= threshold:
            integer_cols.add(col)
    return integer_cols


def prepare_split_data(
    input_path: Path,
    holdout_fraction: float,
    seed: int,
    critical_non_null_threshold: float,
    integer_threshold: float,
) -> PreparedSplit:
    columns, rows = read_rows(input_path)
    if not columns:
        raise ValueError("Input table has no columns.")
    if "wthhfin" not in columns:
        raise ValueError("Input must include wthhfin.")

    train_rows_all, holdout_rows = split_rows_by_household(rows, holdout_fraction, seed)
    all_non_null_fraction: Dict[str, float] = {}
    for col in tqdm(columns, desc="Scanning column non-null rates", unit="col"):
        non_null = sum(1 for row in rows if not is_missing_value(row.get(col, "")))
        all_non_null_fraction[col] = non_null / max(1, len(rows))
    critical_cols = sorted(
        col for col, frac in all_non_null_fraction.items() if frac >= critical_non_null_threshold
    )

    train_rows: List[Dict[str, object]] = []
    for row in tqdm(train_rows_all, desc="Filtering training rows", unit="row"):
        if all(not is_missing_value(row.get(col, "")) for col in critical_cols):
            train_rows.append(row)
    if not train_rows:
        raise ValueError("All training rows were removed by the null-value filter.")

    feature_columns = [c for c in columns if c not in {"wthhfin"} and not is_id_column(c)]
    numeric_columns = [c for c in feature_columns if is_numeric_column(c)]
    categorical_columns = [c for c in feature_columns if c not in set(numeric_columns)]

    integer_numeric_columns = detect_integer_numeric_columns(rows, numeric_columns, integer_threshold)

    n = len(train_rows)
    weights = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(train_rows):
        try:
            w = float(normalize_string(row.get("wthhfin", "")) or "0")
        except ValueError:
            w = 0.0
        if not math.isfinite(w) or w < 0:
            w = 0.0
        weights[i] = w
    if weights.sum() <= 0:
        weights[:] = 1.0

    target_rows = int(round(float(weights.sum())))
    if target_rows <= 0:
        target_rows = n

    num_m = np.zeros((n, len(numeric_columns)), dtype=np.float32)
    for j, col in enumerate(
        tqdm(numeric_columns, desc="Encoding numeric columns", unit="col")
    ):
        for i, row in enumerate(train_rows):
            num_m[i, j] = parse_numeric_value(row.get(col, ""))

    num_means = np.zeros(len(numeric_columns), dtype=np.float32)
    num_stds = np.ones(len(numeric_columns), dtype=np.float32)
    num_mins = np.zeros(len(numeric_columns), dtype=np.float32)
    num_maxs = np.zeros(len(numeric_columns), dtype=np.float32)
    for j in tqdm(range(len(numeric_columns)), desc="Normalizing numeric columns", unit="col"):
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

    cat_m = np.zeros((n, len(categorical_columns)), dtype=np.int64)
    cat_vocab_values: List[List[str]] = []
    cat_cardinalities: List[int] = []
    cat_non_empty_ids: List[np.ndarray] = []
    cat_non_empty_probs: List[np.ndarray] = []
    for j, col in enumerate(
        tqdm(categorical_columns, desc="Encoding categorical columns", unit="col")
    ):
        vals = [normalize_string(row.get(col, "")) for row in train_rows]
        uniq = sorted(set(vals))
        if "" not in uniq:
            uniq = [""] + uniq
        stoi = {v: i for i, v in enumerate(uniq)}
        for i, val in enumerate(vals):
            cat_m[i, j] = stoi.get(val, 0)
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
            if p.sum() <= 0:
                p = np.ones_like(p, dtype=np.float64)
            p = p / p.sum()
            cat_non_empty_ids.append(non_empty)
            cat_non_empty_probs.append(p.astype(np.float64))

    prepared = PreparedData(
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

    return PreparedSplit(
        prepared=prepared,
        holdout_rows=holdout_rows,
        dropped_training_rows=len(train_rows_all) - len(train_rows),
        critical_non_null_columns=critical_cols,
        integer_numeric_columns=integer_numeric_columns,
        train_households=len(train_rows),
        holdout_households=len(holdout_rows),
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
    hidden_layers: int,
    temperature: float,
    dropout: float,
    latent_noise_std: float,
    grad_clip_norm: float,
    compile_model: bool,
) -> ContrastiveOnlyAutoencoder:
    x_num = torch.tensor(prepared.numeric_matrix, dtype=torch.float32)
    x_cat = torch.tensor(prepared.categorical_matrix, dtype=torch.long)

    dataset = TensorDataset(x_num, x_cat)
    sample_weights = torch.tensor(prepared.weights, dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=len(dataset) >= batch_size,
    )

    model = ContrastiveOnlyAutoencoder(
        num_numeric=x_num.shape[1],
        cat_cardinalities=prepared.cat_cardinalities,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
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
    epoch_iter = tqdm(range(1, epochs + 1), desc="Training epochs", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        running = 0.0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for bn, bc in batch_iter:
            bn = bn.to(device, non_blocking=True)
            bc = bc.to(device, non_blocking=True)
            neg_num_np, neg_cat_np = neg_sampler.sample(bn.size(0))
            neg_num = torch.tensor(neg_num_np, dtype=torch.float32, device=device)
            neg_cat = torch.tensor(neg_cat_np, dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                z = model.encode(bn, bc)
                if latent_noise_std > 0:
                    z = z + torch.randn_like(z) * latent_noise_std
                rec_num, rec_cat = model.decode(z)

                d_pos_num = ((rec_num - bn) ** 2).mean(dim=1)
                d_neg_num = ((rec_num - neg_num) ** 2).mean(dim=1)

                if bc.size(1) > 0:
                    pos_cat = torch.stack(
                        [emb(bc[:, i]) for i, emb in enumerate(model.cat_embeddings)],
                        dim=1,
                    )
                    neg_cat_emb = torch.stack(
                        [emb(neg_cat[:, i]) for i, emb in enumerate(model.cat_embeddings)],
                        dim=1,
                    )
                    d_pos_cat = ((rec_cat - pos_cat) ** 2).mean(dim=(1, 2))
                    d_neg_cat = ((rec_cat - neg_cat_emb) ** 2).mean(dim=(1, 2))
                else:
                    d_pos_cat = torch.zeros_like(d_pos_num)
                    d_neg_cat = torch.zeros_like(d_neg_num)

                logits = (d_neg_num + d_neg_cat - d_pos_num - d_pos_cat) / max(temperature, 1e-6)
                loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

            if not torch.isfinite(loss).item():
                batch_iter.set_postfix(loss="nan")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().item())
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.detach().item():.4f}")

        epoch_iter.set_postfix(loss=f"{(running / max(1, n_batches)):.6f}")

    return model


def format_numeric_for_column(value: float, column: str, integer_columns: set[str]) -> str:
    if not math.isfinite(value):
        return ""
    if column in integer_columns:
        return str(int(round(value)))
    return format_numeric(value)


def synthesize_households(
    model: ContrastiveOnlyAutoencoder,
    prepared: PreparedData,
    integer_numeric_columns: set[str],
    device: torch.device,
    output_path: Path,
    sample_rows: int,
    sample_batch_size: int,
    p_empty_retain: float,
    sample_noise_std: float,
) -> None:
    model.eval()
    p_slot_re = re.compile(r"^P(\d+)-")

    num_cols = prepared.numeric_columns
    cat_cols = prepared.categorical_columns
    hhsize_num_idx = num_cols.index("hhsize") if "hhsize" in num_cols else -1
    num_p_slots = [int(m.group(1)) if (m := p_slot_re.match(c)) else 0 for c in num_cols]
    cat_p_slots = [int(m.group(1)) if (m := p_slot_re.match(c)) else 0 for c in cat_cols]

    num_means = torch.tensor(prepared.numeric_means, dtype=torch.float32, device=device)
    num_stds = torch.tensor(prepared.numeric_stds, dtype=torch.float32, device=device)
    num_mins = torch.tensor(prepared.numeric_mins, dtype=torch.float32, device=device)
    num_maxs = torch.tensor(prepared.numeric_maxs, dtype=torch.float32, device=device)
    sample_probs = prepared.weights / np.clip(prepared.weights.sum(), 1e-12, None)

    with StringRowWriter(output_path, prepared.columns, buffer_size=8192) as writer:
        with torch.no_grad():
            produced = 0
            sample_pbar = tqdm(total=sample_rows, desc="Synthesizing households", unit="hh")
            while produced < sample_rows:
                b = min(sample_batch_size, sample_rows - produced)
                src_idx = np.random.choice(len(sample_probs), size=b, replace=True, p=sample_probs)
                x_num = torch.tensor(prepared.numeric_matrix[src_idx], dtype=torch.float32, device=device)
                x_cat = torch.tensor(prepared.categorical_matrix[src_idx], dtype=torch.long, device=device)

                z = model.encode(x_num, x_cat)
                if sample_noise_std > 0:
                    z = z + torch.randn_like(z) * sample_noise_std
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

                for j, col in enumerate(cat_cols):
                    slot = cat_p_slots[j]
                    if slot <= 0:
                        continue
                    non_empty_ids = prepared.cat_non_empty_ids[j]
                    non_empty_probs = prepared.cat_non_empty_probs[j]
                    if non_empty_ids.size == 0:
                        continue
                    empty_mask = (rec_cat_np[:, j] == 0) & (hhsize_active >= slot)
                    if np.any(empty_mask):
                        empty_idx = np.where(empty_mask)[0]
                        retain = np.random.rand(empty_idx.size) < float(
                            min(1.0, max(0.0, p_empty_retain))
                        )
                        fill_idx = empty_idx[~retain]
                        if fill_idx.size > 0:
                            rec_cat_np[fill_idx, j] = np.random.choice(
                                non_empty_ids,
                                size=int(fill_idx.size),
                                replace=True,
                                p=non_empty_probs,
                            )

                for i in range(b):
                    out: Dict[str, str] = {col: "" for col in prepared.columns}
                    active_people = int(hhsize_active[i])

                    for j, col in enumerate(num_cols):
                        slot = num_p_slots[j]
                        if slot > 0 and slot > active_people:
                            out[col] = ""
                        else:
                            out[col] = format_numeric_for_column(
                                float(rec_num_np[i, j]),
                                col,
                                integer_numeric_columns,
                            )

                    for j, col in enumerate(cat_cols):
                        slot = cat_p_slots[j]
                        if slot > 0 and slot > active_people:
                            continue
                        cid = int(rec_cat_np[i, j])
                        vocab = prepared.cat_vocab_values[j]
                        if cid < 0 or cid >= len(vocab):
                            cid = 0
                        out[col] = vocab[cid]

                    out["wthhfin"] = "1"
                    writer.write(out)

                produced += b
                sample_pbar.update(b)
            sample_pbar.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("tupled-survey.parquet"))
    p.add_argument("--household-output", type=Path, default=Path("synthesized-household-trips-80pct.parquet"))
    p.add_argument("--trip-output", type=Path, default=Path("synthetic_trips_80pct.parquet"))
    p.add_argument("--sample-trip-output", type=Path, default=Path("sample_synthetic_trips_80pct.csv"))
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sample-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--emb-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent-noise-std", type=float, default=0.1)
    p.add_argument("--sample-noise-std", type=float, default=0.2)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--critical-non-null-threshold", type=float, default=0.95)
    p.add_argument("--integer-threshold", type=float, default=0.98)
    p.add_argument("--p-empty-retain", type=float, default=0.2)
    p.add_argument("--sample-households", type=int, default=55440)
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

    prepared_split = prepare_split_data(
        input_path=args.input,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        critical_non_null_threshold=args.critical_non_null_threshold,
        integer_threshold=args.integer_threshold,
    )
    prepared = prepared_split.prepared

    print(
        f"train_households={prepared_split.train_households} holdout_households={prepared_split.holdout_households}"
    )
    print(
        f"dropped_training_rows={prepared_split.dropped_training_rows} critical_cols={len(prepared_split.critical_non_null_columns)}"
    )
    print(
        f"features={len(prepared.feature_columns)} numeric={len(prepared.numeric_columns)} categorical={len(prepared.categorical_columns)} id_cols_excluded={len(prepared.columns) - len(prepared.feature_columns) - 1}"
    )
    print(f"integer_numeric_columns={len(prepared_split.integer_numeric_columns)}")

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

    model = train(
        prepared=prepared,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        hidden_layers=args.hidden_layers,
        temperature=args.temperature,
        dropout=args.dropout,
        latent_noise_std=args.latent_noise_std,
        grad_clip_norm=args.grad_clip_norm,
        compile_model=not args.no_compile,
    )

    synthesize_households(
        model=model,
        prepared=prepared,
        integer_numeric_columns=prepared_split.integer_numeric_columns,
        device=device,
        output_path=args.household_output,
        sample_rows=prepared.target_rows,
        sample_batch_size=args.sample_batch_size,
        p_empty_retain=args.p_empty_retain,
        sample_noise_std=args.sample_noise_std,
    )
    print(f"Wrote {args.household_output}")

    untuple_households(
        input_path=args.household_output,
        output_path=args.trip_output,
        sample_output_path=args.sample_trip_output,
        sample_households=args.sample_households,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
