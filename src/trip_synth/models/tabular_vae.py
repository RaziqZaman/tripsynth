from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    return nn.ReLU()


def build_hidden_sizes(input_dim: int, config: dict[str, Any]) -> list[int]:
    explicit = config.get("hidden_sizes")
    if explicit:
        return [int(v) for v in explicit]
    depth = int(config.get("depth", 3))
    decay = float(config.get("width_decay", 0.9))
    base = int(config.get("hidden_width", max(64, min(512, input_dim * 3))))
    sizes = []
    width = base
    for _ in range(depth):
        sizes.append(max(16, int(round(width))))
        width *= decay
    return sizes


def make_mlp(input_dim: int, hidden_sizes: list[int], config: dict[str, Any]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    dropout = float(config.get("dropout", 0.0))
    norm = str(config.get("normalization", "layernorm")).lower()
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        if norm == "batchnorm":
            layers.append(nn.BatchNorm1d(width))
        elif norm == "layernorm":
            layers.append(nn.LayerNorm(width))
        layers.append(_activation(str(config.get("activation", "relu"))))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*layers)


class MixedTabularVAE(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        num_numeric: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.category_sizes = category_sizes
        self.num_numeric = num_numeric
        self.latent_dim = int(config.get("latent_dim", 64))
        self.embedding_dims = [
            int(min(32, max(2, round(size**0.25 * 4)))) if size > 1 else 1
            for size in category_sizes
        ]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(1, size), dim) for size, dim in zip(category_sizes, self.embedding_dims)]
        )
        input_dim = int(sum(self.embedding_dims) + num_numeric)
        hidden = build_hidden_sizes(input_dim, config)
        self.encoder = make_mlp(input_dim, hidden, config)
        enc_out = hidden[-1] if hidden else input_dim
        self.mu = nn.Linear(enc_out, self.latent_dim)
        self.logvar = nn.Linear(enc_out, self.latent_dim)

        decoder_hidden = list(reversed(hidden)) or [max(32, input_dim)]
        self.decoder = make_mlp(self.latent_dim, decoder_hidden, config)
        dec_out = decoder_hidden[-1]
        self.cat_heads = nn.ModuleList([nn.Linear(dec_out, size) for size in category_sizes])
        self.num_head = nn.Linear(dec_out, num_numeric) if num_numeric else None

    def encode_features(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(cat_x[:, j]))
        if num_x.numel():
            parts.append(num_x.float())
        if not parts:
            raise ValueError("VAE requires at least one feature")
        return torch.cat(parts, dim=1)

    def encode(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.encode_features(cat_x, num_x))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = self.decoder(z)
        cat_logits = [head(h) for head in self.cat_heads]
        if self.num_head is None:
            num = z.new_zeros((z.shape[0], 0))
        else:
            num = self.num_head(h)
        return cat_logits, num

    def forward(
        self, cat_x: torch.Tensor, num_x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(cat_x, num_x)
        z = self.reparameterize(mu, logvar)
        cat_logits, num = self.decode(z)
        return cat_logits, num, mu, logvar, z
