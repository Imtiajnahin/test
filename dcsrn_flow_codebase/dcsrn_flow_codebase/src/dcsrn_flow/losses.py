from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def divergence_3d(velocity: torch.Tensor) -> torch.Tensor:
    """Compute finite-difference divergence for channel-first velocity.

    Args:
        velocity: [B, 3, X, Y, Z]
            channel 0 = u, derivative along X dimension
            channel 1 = v, derivative along Y dimension
            channel 2 = w, derivative along Z dimension

    Returns:
        div: [B, 1, X-1, Y-1, Z-1]
    """
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise ValueError(f"Expected velocity [B,3,X,Y,Z], got {tuple(velocity.shape)}")

    u = velocity[:, 0]
    v = velocity[:, 1]
    w = velocity[:, 2]

    du_dx = u[:, 1:, :, :] - u[:, :-1, :, :]
    dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
    dw_dz = w[:, :, :, 1:] - w[:, :, :, :-1]

    # Crop all terms to common [B, X-1, Y-1, Z-1]
    du_dx = du_dx[:, :, :-1, :-1]
    dv_dy = dv_dy[:, :-1, :, :-1]
    dw_dz = dw_dz[:, :-1, :-1, :]

    div = du_dx + dv_dy + dw_dz
    return div.unsqueeze(1)


class DCSRNFlowLoss(nn.Module):
    """MSE + BCE(mask) + divergence regularization."""

    def __init__(
        self,
        lambda_seg: float = 0.0016,
        lambda_div: float = 0.066,
        divergence_mode: str = "l1",
        use_masked_mse: bool = False,
    ):
        super().__init__()
        self.lambda_seg = float(lambda_seg)
        self.lambda_div = float(lambda_div)
        self.divergence_mode = str(divergence_mode).lower()
        self.use_masked_mse = bool(use_masked_mse)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target_velocity: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred_velocity = pred["velocity"]
        pred_mask = pred["mask"]

        if self.use_masked_mse:
            mask3 = target_mask.expand_as(target_velocity)
            denom = torch.clamp(mask3.sum(), min=1.0)
            mse = torch.sum(mask3 * (pred_velocity - target_velocity) ** 2) / denom
        else:
            mse = F.mse_loss(pred_velocity, target_velocity)

        bce = F.binary_cross_entropy(pred_mask.clamp(1e-6, 1.0 - 1e-6), target_mask)

        div = divergence_3d(pred_velocity)
        mask_div = target_mask[:, :, 1:, 1:, 1:]
        if self.divergence_mode == "l2":
            div_values = div ** 2
        elif self.divergence_mode == "l1":
            div_values = torch.abs(div)
        else:
            raise ValueError("divergence_mode must be 'l1' or 'l2'")

        denom = torch.clamp(mask_div.sum(), min=1.0)
        div_loss = torch.sum(mask_div * div_values) / denom

        total = mse + self.lambda_seg * bce + self.lambda_div * div_loss
        return {
            "total": total,
            "mse": mse.detach(),
            "bce": bce.detach(),
            "div": div_loss.detach(),
        }
