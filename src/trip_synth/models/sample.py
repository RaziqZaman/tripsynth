from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from trip_synth.data.postprocessing import finalize_synthetic
from trip_synth.data.schema import FeatureSchema


def sample_vae_method(
    artifact: dict[str, Any],
    n_rows: int,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
    run_id: str,
) -> pd.DataFrame:
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    vae_cfg = artifact["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    temperature = float(vae_cfg.get("sample_temperature", config.get("sample_temperature", 1.0)))
    deterministic = bool(vae_cfg.get("deterministic_sample", False))
    chunks = []
    remaining = int(n_rows)
    chunk_size = int(vae_cfg.get("sample_chunk_size", 50000))
    with torch.no_grad():
        while remaining > 0:
            take = min(chunk_size, remaining)
            z = torch.randn(take, model.latent_dim, device=device)
            cat_logits, num = model.decode(z)
            cat_codes = []
            for logits in cat_logits:
                if deterministic:
                    codes = torch.argmax(logits, dim=1)
                else:
                    probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
                    codes = torch.multinomial(probs, 1).squeeze(1)
                cat_codes.append(codes.detach().cpu().numpy())
            if cat_codes:
                cat_arr = np.stack(cat_codes, axis=1)
            else:
                cat_arr = np.zeros((take, 0), dtype=int)
            num_arr = num.detach().cpu().numpy() if num.numel() else np.zeros((take, 0), dtype=float)
            chunks.append(preprocessor.inverse_transform(cat_arr, num_arr, vae_cfg))
            remaining -= take
    sample = pd.concat(chunks, ignore_index=True)
    return finalize_synthetic(sample, schema, preprocessor, artifact["method"], run_id)
