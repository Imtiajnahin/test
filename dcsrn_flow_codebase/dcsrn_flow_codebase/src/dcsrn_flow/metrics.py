from __future__ import annotations

import torch

from .losses import divergence_3d


@torch.no_grad()
def velocity_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = torch.sqrt(torch.sum((pred - target) ** 2, dim=1, keepdim=True) + 1e-8)
    if mask is not None:
        denom = torch.clamp(mask.sum(), min=1.0)
        return torch.sum(mask * err) / denom
    return torch.mean(err)


@torch.no_grad()
def mean_abs_divergence(pred_velocity: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    div = torch.abs(divergence_3d(pred_velocity))
    if mask is not None:
        mask_div = mask[:, :, 1:, 1:, 1:]
        denom = torch.clamp(mask_div.sum(), min=1.0)
        return torch.sum(mask_div * div) / denom
    return torch.mean(div)
