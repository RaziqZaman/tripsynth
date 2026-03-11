#!/usr/bin/env python3
"""Train on 80% of tupled households, synthesize sample-sized trips, and validate against the held-out 20%."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
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
from tabular_io import StringRowWriter, count_rows, get_fieldnames, iter_rows, read_rows, write_rows
from untuple import untuple_households
from validate import (
    NULL_LABEL,
    RunningStats,
    histogram_index,
    js_distance,
    normalized_distribution,
    normalize_value,
    parse_numeric,
    safe_filename,
    scan_dataset,
    stable_bucket,
    tv_distance,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


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


class ReconstructionContrastiveAutoencoder(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Sequence[int],
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int,
        dropout: float,
        projection_dim: int = 256,
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
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_head = nn.Linear(prev_dim, latent_dim)

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for _ in range(max(1, int(hidden_layers))):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.numeric_head = nn.Linear(prev_dim, self.num_numeric)
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(prev_dim, int(card)) for card in cat_cardinalities]
        )
        self.recon_embed_heads = nn.ModuleList(
            [nn.Linear(prev_dim, emb_dim) for _ in cat_cardinalities]
        )

        self.anchor_projector = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.positive_projector = nn.Sequential(
            nn.Linear(prev_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def embed_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.num_categorical == 0:
            return x_cat.new_zeros((x_cat.size(0), 0), dtype=torch.float32)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        return torch.cat(embs, dim=1)

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.embed_categorical(x_cat)], dim=1)
        return self.latent_head(self.encoder(x))

    def decode(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        rec_num = self.numeric_head(h)
        cat_logits = [head(h) for head in self.categorical_heads]
        if self.num_categorical > 0:
            rec_cat_embed = torch.stack([head(h) for head in self.recon_embed_heads], dim=1)
        else:
            rec_cat_embed = h.new_zeros((z.size(0), 0, self.emb_dim))
        return rec_num, cat_logits, rec_cat_embed, h


def format_numeric_for_column(value: float, column: str, integer_columns: set[str]) -> str:
    if not math.isfinite(value):
        return ""
    if column in integer_columns:
        return str(int(round(value)))
    return format_numeric(value)


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
    reconstruction_weight: float = 1.0,
    contrastive_weight: float = 0.25,
) -> ReconstructionContrastiveAutoencoder:
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

    model = ReconstructionContrastiveAutoencoder(
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
        running_recon = 0.0
        running_ctr = 0.0
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
                rec_num, cat_logits, _, dec_hidden = model.decode(z)

                num_loss = F.mse_loss(rec_num, bn)
                cat_losses = [
                    F.cross_entropy(logits, bc[:, j], reduction="mean")
                    for j, logits in enumerate(cat_logits)
                ]
                cat_loss = (
                    torch.stack(cat_losses).mean()
                    if cat_losses
                    else rec_num.new_zeros(())
                )
                recon_loss = num_loss + cat_loss

                anchor = F.normalize(model.anchor_projector(z), dim=1)
                positive = F.normalize(model.positive_projector(dec_hidden), dim=1)
                neg_z = model.encode(neg_num, neg_cat)
                negative = F.normalize(model.anchor_projector(neg_z), dim=1)

                pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)
                neg_logits = anchor @ negative.T
                logits = torch.cat([pos_logits, neg_logits], dim=1) / max(temperature, 1e-6)
                targets = torch.zeros(anchor.size(0), dtype=torch.long, device=device)
                contrastive_loss = F.cross_entropy(logits, targets)

                loss = reconstruction_weight * recon_loss + contrastive_weight * contrastive_loss

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
            running_recon += float(recon_loss.detach().item())
            running_ctr += float(contrastive_loss.detach().item())
            n_batches += 1
            batch_iter.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                recon=f"{recon_loss.detach().item():.4f}",
                ctr=f"{contrastive_loss.detach().item():.4f}",
            )

        epoch_iter.set_postfix(
            loss=f"{(running / max(1, n_batches)):.6f}",
            recon=f"{(running_recon / max(1, n_batches)):.6f}",
            ctr=f"{(running_ctr / max(1, n_batches)):.6f}",
        )

    return model


def synthesize_households(
    model: ReconstructionContrastiveAutoencoder,
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
                rec_num, _, rec_cat, _ = model.decode(z)
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


def load_synthesize_module() -> ModuleType:
    module_path = Path(__file__).with_name("synthesize-80pct.py")
    module_name = "synthesize_80pct_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load synthesis module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("tupled-survey.parquet"))
    p.add_argument("--household-output", type=Path, default=Path("synthesized-household-trips-80pct.parquet"))
    p.add_argument("--trip-output", type=Path, default=Path("synthetic_trips_80pct.parquet"))
    p.add_argument("--sample-trip-output", type=Path, default=Path("sample_synthetic_trips_80pct.csv"))
    p.add_argument("--holdout-households", type=Path, default=Path("heldout_households_20pct.parquet"))
    p.add_argument("--holdout-trips", type=Path, default=Path("heldout_trips_20pct.parquet"))
    p.add_argument("--validation-csv", type=Path, default=Path("validation_20pct.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("comparison_histograms_20pct"))
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-batch-size", type=int, default=512)
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
    p.add_argument("--sample-rows", type=int, default=10080)
    p.add_argument("--sample-households", dest="sample_rows", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--hash-buckets", type=int, default=256)
    p.add_argument("--max-categories-plot", type=int, default=25)
    p.add_argument("--numeric-threshold", type=float, default=0.98)
    p.add_argument("--max-columns", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.set_defaults(skip_plots=False)
    p.add_argument("--skip-plots", dest="skip_plots", action="store_true")
    p.add_argument("--with-plots", dest="skip_plots", action="store_false")
    return p.parse_args()


def materialize_holdout(args: argparse.Namespace) -> None:
    columns, rows = read_rows(args.input)
    _, holdout_rows = split_rows_by_household(rows, args.holdout_fraction, args.seed)
    with StringRowWriter(args.holdout_households, columns, buffer_size=8192) as writer:
        for row in tqdm(
            holdout_rows,
            desc="Writing held-out households",
            unit="hh",
            total=len(holdout_rows),
        ):
            writer.write(row)
    untuple_households(
        input_path=args.holdout_households,
        output_path=args.holdout_trips,
        sample_output_path=None,
    )


def validate_outputs(args: argparse.Namespace) -> None:
    plt = None
    if not args.skip_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as exc:
            missing = exc.name or "matplotlib"
            raise ModuleNotFoundError(
                f"Missing Python dependency: {missing}. Install plotting support with: pip install matplotlib"
            ) from exc

    synth_cols = set(get_fieldnames(args.trip_output))
    holdout_cols = set(get_fieldnames(args.holdout_trips))
    shared_cols = sorted(synth_cols & holdout_cols)
    if args.max_columns is not None:
        shared_cols = shared_cols[: args.max_columns]
    if not shared_cols:
        raise ValueError("No shared columns found between synthetic trips and holdout trips.")

    synth_total_rows = count_rows(args.trip_output)
    holdout_total_rows = count_rows(args.holdout_trips)
    synth_rows, synth_scan = scan_dataset(
        args.trip_output,
        shared_cols,
        args.max_categories_plot,
        desc="Scanning synthetic",
        total_rows_hint=synth_total_rows,
    )
    holdout_rows, holdout_scan = scan_dataset(
        args.holdout_trips,
        shared_cols,
        args.max_categories_plot,
        desc="Scanning holdout",
        total_rows_hint=holdout_total_rows,
    )

    metadata: Dict[str, dict[str, object]] = {}
    for col in tqdm(shared_cols, desc="Profiling shared columns", unit="col"):
        s = synth_scan[col]
        h = holdout_scan[col]
        s_frac = (s.numeric_count / s.non_empty) if s.non_empty > 0 else 0.0
        h_frac = (h.numeric_count / h.non_empty) if h.non_empty > 0 else 0.0
        is_numeric = (
            s.non_empty > 0
            and h.non_empty > 0
            and s_frac >= args.numeric_threshold
            and h_frac >= args.numeric_threshold
        )

        if is_numeric:
            min_value = min(s.min_value, h.min_value)
            max_value = max(s.max_value, h.max_value)
            if not math.isfinite(min_value) or not math.isfinite(max_value):
                min_value, max_value = 0.0, 1.0
            if max_value <= min_value:
                min_value -= 0.5
                max_value += 0.5
            metadata[col] = {
                "kind": "numeric",
                "min": min_value,
                "max": max_value,
                "s_hist": np.zeros(args.bins, dtype=np.int64),
                "h_hist": np.zeros(args.bins, dtype=np.int64),
                "s_stats": RunningStats(),
                "h_stats": RunningStats(),
            }
        else:
            exact = s.unique_values is not None and h.unique_values is not None
            metadata[col] = {
                "kind": "categorical",
                "mode": "exact" if exact else "hashed",
                "s_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
                "h_count": Counter() if exact else np.zeros(args.hash_buckets, dtype=np.int64),
            }

    numeric_cols = [c for c in shared_cols if metadata[c]["kind"] == "numeric"]
    categorical_exact_cols = [
        c for c in shared_cols if metadata[c]["kind"] == "categorical" and metadata[c]["mode"] == "exact"
    ]
    categorical_hashed_cols = [
        c for c in shared_cols if metadata[c]["kind"] == "categorical" and metadata[c]["mode"] == "hashed"
    ]

    def accumulate(path: Path, side: str, total_rows: int) -> None:
        label = "Accumulating synthetic" if side == "s" else "Accumulating holdout"
        for row in tqdm(iter_rows(path), total=total_rows, desc=label, unit="row"):
            for col in numeric_cols:
                info = metadata[col]
                x = parse_numeric(row.get(col))
                if x is None:
                    continue
                idx = histogram_index(float(x), float(info["min"]), float(info["max"]), args.bins)
                if side == "s":
                    info["s_hist"][idx] += 1
                    info["s_stats"].update(float(x))
                else:
                    info["h_hist"][idx] += 1
                    info["h_stats"].update(float(x))

            for col in categorical_exact_cols:
                info = metadata[col]
                token = normalize_value(row.get(col))
                if token == "":
                    token = NULL_LABEL
                if side == "s":
                    info["s_count"][token] += 1
                else:
                    info["h_count"][token] += 1

            for col in categorical_hashed_cols:
                info = metadata[col]
                token = normalize_value(row.get(col))
                if token == "":
                    token = NULL_LABEL
                bucket = stable_bucket(token, args.hash_buckets)
                if side == "s":
                    info["s_count"][bucket] += 1
                else:
                    info["h_count"][bucket] += 1

    accumulate(args.trip_output, "s", synth_rows)
    accumulate(args.holdout_trips, "h", holdout_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: List[Dict[str, str]] = []

    for col in tqdm(shared_cols, desc="Scoring columns", unit="col"):
        info = metadata[col]
        s_non_empty = synth_scan[col].non_empty
        h_non_empty = holdout_scan[col].non_empty
        s_missing_rate = 1.0 - (s_non_empty / synth_rows if synth_rows > 0 else 0.0)
        h_missing_rate = 1.0 - (h_non_empty / holdout_rows if holdout_rows > 0 else 0.0)

        if info["kind"] == "numeric":
            s_hist = normalized_distribution(info["s_hist"])
            h_hist = normalized_distribution(info["h_hist"])
            metric_js = js_distance(s_hist, h_hist)
            metric_tv = tv_distance(s_hist, h_hist)

            if not args.skip_plots:
                x_min = float(info["min"])
                x_max = float(info["max"])
                edges = np.linspace(x_min, x_max, args.bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                fig, ax = plt.subplots(figsize=(10, 5.5))
                width = edges[1] - edges[0]
                ax.bar(centers, h_hist, width=width, alpha=0.5, label="holdout", align="center")
                ax.bar(centers, s_hist, width=width, alpha=0.5, label="synthetic", align="center")
                ax.set_title(f"Histogram Match: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Probability")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                fig.savefig(args.output_dir / f"{safe_filename(col)}.png", dpi=140)
                plt.close(fig)

            s_stats: RunningStats = info["s_stats"]
            h_stats: RunningStats = info["h_stats"]
            validation_rows.append(
                {
                    "column": col,
                    "column_type": "numeric",
                    "distribution_mode": "histogram",
                    "js_distance": f"{metric_js:.6f}",
                    "tv_distance": f"{metric_tv:.6f}",
                    "synthetic_non_empty": str(s_non_empty),
                    "holdout_non_empty": str(h_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "holdout_missing_rate": f"{h_missing_rate:.6f}",
                    "synthetic_mean": f"{s_stats.mean:.6f}",
                    "holdout_mean": f"{h_stats.mean:.6f}",
                    "synthetic_std": f"{s_stats.std:.6f}",
                    "holdout_std": f"{h_stats.std:.6f}",
                }
            )
        else:
            if info["mode"] == "exact":
                s_count: Counter = info["s_count"]
                h_count: Counter = info["h_count"]
                categories = sorted(set(s_count.keys()) | set(h_count.keys()))
                s_arr = np.array([s_count.get(c, 0) for c in categories], dtype=np.float64)
                h_arr = np.array([h_count.get(c, 0) for c in categories], dtype=np.float64)
            else:
                s_arr = info["s_count"].astype(np.float64)
                h_arr = info["h_count"].astype(np.float64)

            p = normalized_distribution(s_arr)
            q = normalized_distribution(h_arr)
            metric_js = js_distance(p, q)
            metric_tv = tv_distance(p, q)

            if not args.skip_plots:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                if info["mode"] == "exact":
                    s_count = info["s_count"]
                    h_count = info["h_count"]
                    top = sorted(
                        set(s_count.keys()) | set(h_count.keys()),
                        key=lambda k: s_count.get(k, 0) + h_count.get(k, 0),
                        reverse=True,
                    )[: args.max_categories_plot]
                    xs = np.arange(len(top), dtype=np.float64)
                    s_vals = normalized_distribution(
                        np.array([s_count.get(k, 0) for k in top], dtype=np.float64)
                    )
                    h_vals = normalized_distribution(
                        np.array([h_count.get(k, 0) for k in top], dtype=np.float64)
                    )
                    width = 0.45
                    ax.bar(xs - width / 2, h_vals, width=width, label="holdout")
                    ax.bar(xs + width / 2, s_vals, width=width, label="synthetic")
                    ax.set_xticks(xs)
                    ax.set_xticklabels(top, rotation=75, ha="right", fontsize=8)
                    ax.set_xlabel("Category")
                else:
                    xs = np.arange(args.hash_buckets, dtype=np.float64)
                    width = 0.45
                    ax.bar(xs - width / 2, q, width=width, alpha=0.7, label="holdout")
                    ax.bar(xs + width / 2, p, width=width, alpha=0.7, label="synthetic")
                    ax.set_xlabel("Hashed category bucket")

                ax.set_title(f"Histogram Match: {col}")
                ax.set_ylabel("Probability")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                fig.savefig(args.output_dir / f"{safe_filename(col)}.png", dpi=140)
                plt.close(fig)

            validation_rows.append(
                {
                    "column": col,
                    "column_type": "categorical",
                    "distribution_mode": str(info["mode"]),
                    "js_distance": f"{metric_js:.6f}",
                    "tv_distance": f"{metric_tv:.6f}",
                    "synthetic_non_empty": str(s_non_empty),
                    "holdout_non_empty": str(h_non_empty),
                    "synthetic_missing_rate": f"{s_missing_rate:.6f}",
                    "holdout_missing_rate": f"{h_missing_rate:.6f}",
                    "synthetic_mean": "",
                    "holdout_mean": "",
                    "synthetic_std": "",
                    "holdout_std": "",
                }
            )

    validation_rows.sort(key=lambda row: (-float(row["js_distance"]), row["column"]))
    write_rows(
        args.validation_csv,
        [
            "column",
            "column_type",
            "distribution_mode",
            "js_distance",
            "tv_distance",
            "synthetic_non_empty",
            "holdout_non_empty",
            "synthetic_missing_rate",
            "holdout_missing_rate",
            "synthetic_mean",
            "holdout_mean",
            "synthetic_std",
            "holdout_std",
        ],
        validation_rows,
    )
    print(f"Wrote {args.validation_csv}")


def main() -> int:
    args = parse_args()
    synth_mod = load_synthesize_module()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing tupled input: {args.input}")
    if args.bins < 1:
        raise ValueError("--bins must be >= 1")
    if args.hash_buckets < 2:
        raise ValueError("--hash-buckets must be >= 2")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    prepared_split = synth_mod.prepare_split_data(
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
        sample_rows=args.sample_rows,
        sample_batch_size=args.sample_batch_size,
        p_empty_retain=args.p_empty_retain,
        sample_noise_std=args.sample_noise_std,
    )
    print(f"Wrote {args.household_output}")

    untuple_households(
        input_path=args.household_output,
        output_path=args.trip_output,
        sample_output_path=args.sample_trip_output,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )
    print(f"Wrote {args.trip_output}")

    materialize_holdout(args)
    if not args.holdout_trips.exists():
        raise FileNotFoundError(f"Failed to materialize holdout trips: {args.holdout_trips}")

    validate_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
