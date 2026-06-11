from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_row_mean(row_loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().view(-1)
    denom = torch.clamp(weights.sum(), min=1e-6)
    return (row_loss.view(-1) * weights).sum() / denom


def reconstruction_loss(
    cat_logits: list[torch.Tensor],
    num_pred: torch.Tensor,
    cat_target: torch.Tensor,
    num_target: torch.Tensor,
    weights: torch.Tensor,
    numeric_loss: str = "mse",
) -> torch.Tensor:
    row_loss = torch.zeros(cat_target.shape[0], device=weights.device)
    for j, logits in enumerate(cat_logits):
        if logits.numel() == 0:
            continue
        row_loss = row_loss + F.cross_entropy(logits, cat_target[:, j], reduction="none")
    if num_pred.numel() and num_target.numel():
        if numeric_loss == "huber":
            num_loss = F.smooth_l1_loss(num_pred, num_target, reduction="none")
        else:
            num_loss = F.mse_loss(num_pred, num_target, reduction="none")
        row_loss = row_loss + num_loss.sum(dim=1)
    return weighted_row_mean(row_loss, weights)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    row_kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return weighted_row_mean(row_kl, weights)


def infonce_loss(
    z: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    z_neg = F.normalize(z_neg, dim=1)
    pos = torch.sum(z * z_pos, dim=1, keepdim=True) / temperature
    neg = torch.sum(z * z_neg, dim=1, keepdim=True) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)
