from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.data.load import read_survey_csv
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.models.sample import sample_vae_method
from trip_synth.models.train import fit_vae_method
from trip_synth.utils.io import ensure_dir, load_yaml
from trip_synth.utils.seed import set_seed
from trip_synth.validation.cross_marginals import validate_method_cross_marginals
from trip_synth.validation.marginals import validate_method_marginals
from trip_synth.validation.privacy import validate_method_privacy


def _trial_grid(grid: dict[str, Any], seed: int, mode: str, max_trials: int) -> list[dict[str, Any]]:
    search = grid.get("search", {})
    keys = list(search)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[search[k] for k in keys])]
    if mode == "full_grid":
        return combos
    if mode == "capped_grid":
        return combos[:max_trials]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combos))
    return [combos[i] for i in order[: min(max_trials, len(combos))]]


def _plot_hparams(results: pd.DataFrame, run_dir: Path) -> None:
    fig_dir = ensure_dir(run_dir / "figures")
    if results.empty or "score" not in results:
        return
    if {"latent_dim", "lambda_contrastive"}.issubset(results.columns):
        pivot = results.pivot_table(index="latent_dim", columns="lambda_contrastive", values="score", aggfunc="mean")
        if not pivot.empty:
            plt.figure(figsize=(8, 5))
            plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.colorbar(label="score")
            plt.xlabel("lambda_contrastive")
            plt.ylabel("latent_dim")
            plt.title("Hyperparameter score heatmap")
            plt.tight_layout()
            plt.savefig(fig_dir / "hparam_heatmap_latent_lambda.png", dpi=250)
            plt.close()
    numeric = [c for c in results.columns if c not in {"trial", "method", "status"} and pd.api.types.is_numeric_dtype(results[c])]
    if len(numeric) >= 2:
        scaled = results[numeric].copy()
        for col in numeric:
            lo, hi = scaled[col].min(), scaled[col].max()
            scaled[col] = 0.5 if hi == lo else (scaled[col] - lo) / (hi - lo)
        plt.figure(figsize=(10, 5))
        for _, row in scaled.iterrows():
            plt.plot(range(len(numeric)), row[numeric], alpha=0.45)
        plt.xticks(range(len(numeric)), numeric, rotation=30, ha="right")
        plt.title("Hyperparameter parallel coordinates")
        plt.tight_layout()
        plt.savefig(fig_dir / "hparam_parallel_coordinates.png", dpi=250)
        plt.close()


def run_hparams(config_path: str, grid_path: str) -> Path:
    base_config = load_yaml(config_path)
    grid = load_yaml(grid_path)
    seed = int(base_config.get("seed", 0))
    set_seed(seed)
    method = str(grid.get("method", "contrastive_vae"))
    trials = _trial_grid(
        grid,
        seed=seed,
        mode=str(grid.get("search_mode", "capped_sampled_grid")),
        max_trials=int(grid.get("max_trials", 40)),
    )
    run_dir = Path("outputs") / "runs" / f"{base_config.get('run_name', 'run')}_hparams"
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    real_df, _ = read_survey_csv(base_config["input_csv"], schema)
    reference = FittedPreprocessor.fit(real_df, schema)
    sample_n = min(2000, int(base_config.get("sample_n", 1000)) if base_config.get("sample_n") != "population_from_weight_sum" else 2000)
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(trials, start=1):
        config = copy.deepcopy(base_config)
        config["run_name"] = f"{base_config.get('run_name', 'run')}_hparam_{i:03d}"
        config.setdefault("vae", {}).update(params)
        config["methods"] = [method]
        trial_dir = ensure_dir(run_dir / "checkpoints" / f"trial_{i:03d}")
        try:
            artifact = fit_vae_method(real_df, schema, config, trial_dir, method=method)
            synthetic = sample_vae_method(artifact, sample_n, schema, config, run_dir / "samples", config["run_name"])
            marg = validate_method_marginals(real_df, synthetic, schema, reference, f"trial_{i:03d}", run_dir)
            cross = validate_method_cross_marginals(real_df, synthetic, schema, reference, f"trial_{i:03d}", run_dir)
            priv = validate_method_privacy(real_df, synthetic, schema, reference, f"trial_{i:03d}", run_dir)
            invalid = priv.get("invalid_category_rate", 0.0)
            copy = priv.get("exact_row_copy_rate", 0.0)
            score = (
                marg.get("summary", {}).get("marginal_similarity_score", 0.0)
                + cross.get("summary", {}).get("cross_marginal_similarity_score", 0.0)
                + priv.get("privacy_score", 0.0)
                - invalid
                - copy
            )
            row = {"trial": i, "method": method, "status": "ok", "score": float(score), **params}
        except Exception as exc:
            row = {"trial": i, "method": method, "status": f"failed: {exc}", "score": np.nan, **params}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "tables" / "hparam_results.csv", index=False)
    results = pd.DataFrame(rows)
    _plot_hparams(results, run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--grid", default="configs/hparam_grid.yaml")
    args = parser.parse_args()
    run_dir = run_hparams(args.config, args.grid)
    print(f"Completed hparam run at {run_dir}")


if __name__ == "__main__":
    main()
