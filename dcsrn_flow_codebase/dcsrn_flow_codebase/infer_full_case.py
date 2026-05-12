from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dcsrn_flow import DCSRNFlow
from src.dcsrn_flow.io import load_4dflow_mat, save_4dflow_mat
from src.dcsrn_flow.utils import get_device, load_config


def starts_for_dim(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return list(dict.fromkeys(starts))


def pad_lr_volume(vol_xyz3: np.ndarray, patch_lr: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    X, Y, Z, C = vol_xyz3.shape
    px, py, pz = patch_lr
    pad_x = max(0, px - X)
    pad_y = max(0, py - Y)
    pad_z = max(0, pz - Z)
    pad_width = (
        (0, pad_x),
        (0, pad_y),
        (0, pad_z),
        (0, 0),
    )
    if pad_x or pad_y or pad_z:
        vol_xyz3 = np.pad(vol_xyz3, pad_width, mode="constant", constant_values=0)
    return vol_xyz3, (pad_x, pad_y, pad_z)


def normalize_patch(patch_xyz3: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    patch = np.nan_to_num(patch_xyz3.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.max(np.abs(patch)))
    if norm <= 1e-12:
        norm = 1.0
    patch_norm = patch / norm
    mag = np.sqrt(np.sum(patch_norm ** 2, axis=3))
    max_mag = float(np.max(mag))
    if max_mag > 1e-12:
        mag = mag / max_mag
    return patch_norm.astype(np.float32), norm, mag.astype(np.float32)


@torch.no_grad()
def super_resolve_timeframe(
    model: DCSRNFlow,
    vol_xyz3: np.ndarray,
    patch_size_hr: Tuple[int, int, int],
    scale_factor: int,
    device: torch.device,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Super-resolve one [X,Y,Z,3] low-res time frame to [sX,sY,sZ,3]."""
    patch_lr = tuple(int(p // scale_factor) for p in patch_size_hr)
    stride_lr = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_lr)

    original_shape = vol_xyz3.shape[:3]
    vol_padded, _ = pad_lr_volume(vol_xyz3, patch_lr)
    X, Y, Z, C = vol_padded.shape
    assert C == 3

    out_shape = (X * scale_factor, Y * scale_factor, Z * scale_factor)
    sr_sum = np.zeros((*out_shape, 3), dtype=np.float32)
    mask_sum = np.zeros((*out_shape, 1), dtype=np.float32)
    weight = np.zeros((*out_shape, 1), dtype=np.float32)

    xs = starts_for_dim(X, patch_lr[0], stride_lr[0])
    ys = starts_for_dim(Y, patch_lr[1], stride_lr[1])
    zs = starts_for_dim(Z, patch_lr[2], stride_lr[2])

    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                patch = vol_padded[
                    x0 : x0 + patch_lr[0],
                    y0 : y0 + patch_lr[1],
                    z0 : z0 + patch_lr[2],
                    :,
                ]
                patch_norm, norm, mag = normalize_patch(patch)
                lr_v = torch.from_numpy(np.transpose(patch_norm, (3, 0, 1, 2))).unsqueeze(0).to(device)
                lr_m = torch.from_numpy(mag[None, None, ...]).to(device)
                lr_input = torch.cat([lr_v, lr_m], dim=1)

                pred = model(lr_input)
                pred_v = pred["velocity"].squeeze(0).detach().cpu().numpy()  # [3,Xh,Yh,Zh]
                pred_m = pred["mask"].squeeze(0).detach().cpu().numpy()      # [1,Xh,Yh,Zh]

                pred_v = np.transpose(pred_v, (1, 2, 3, 0)) * norm
                pred_m = np.transpose(pred_m, (1, 2, 3, 0))

                ox0, oy0, oz0 = x0 * scale_factor, y0 * scale_factor, z0 * scale_factor
                ox1, oy1, oz1 = ox0 + patch_size_hr[0], oy0 + patch_size_hr[1], oz0 + patch_size_hr[2]

                sr_sum[ox0:ox1, oy0:oy1, oz0:oz1, :] += pred_v.astype(np.float32)
                mask_sum[ox0:ox1, oy0:oy1, oz0:oz1, :] += pred_m.astype(np.float32)
                weight[ox0:ox1, oy0:oy1, oz0:oz1, :] += 1.0

    weight = np.maximum(weight, 1e-6)
    sr = sr_sum / weight
    mask = mask_sum / weight

    # Crop away padding-derived output.
    crop_x, crop_y, crop_z = [d * scale_factor for d in original_shape]
    sr = sr[:crop_x, :crop_y, :crop_z, :]
    mask = mask[:crop_x, :crop_y, :crop_z, :]
    return sr.astype(np.float32), mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_mat", type=str, required=True)
    parser.add_argument("--output_mat", type=str, required=True)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print("Using device:", device)

    data_cfg = cfg["data"]
    data = load_4dflow_mat(args.input_mat, data_cfg["mat_var_path"], data_cfg.get("layout", "XYZCT"))
    print("Input velocity shape [X,Y,Z,3,T]:", data.shape)

    model = DCSRNFlow(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patch_size_hr = tuple(int(x) for x in data_cfg["patch_size_hr"])
    scale_factor = int(data_cfg["scale_factor"])

    X, Y, Z, C, T = data.shape
    sr_shape = (X * scale_factor, Y * scale_factor, Z * scale_factor, 3, T)
    velocity_sr = np.zeros(sr_shape, dtype=np.float32)
    mask_sr = np.zeros((sr_shape[0], sr_shape[1], sr_shape[2], 1, T), dtype=np.float32)

    for t in tqdm(range(T), desc="timeframes"):
        vol_t = data[:, :, :, :, t]
        sr_t, mask_t = super_resolve_timeframe(
            model=model,
            vol_xyz3=vol_t,
            patch_size_hr=patch_size_hr,
            scale_factor=scale_factor,
            device=device,
            overlap=args.overlap,
        )
        velocity_sr[:, :, :, :, t] = sr_t
        mask_sr[:, :, :, :, t] = mask_t

    save_4dflow_mat(
        args.output_mat,
        velocity_sr=velocity_sr,
        mask_sr=mask_sr,
        velocity_lr_original=data,
        extra={
            "scale_factor": np.asarray(scale_factor, dtype=np.int32),
            "source_checkpoint": args.checkpoint,
        },
    )
    print("Saved:", args.output_mat)
    print("velocity_sr shape:", velocity_sr.shape)


if __name__ == "__main__":
    main()
