from __future__ import annotations

import glob
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .io import load_4dflow_mat


Array5D = np.ndarray


def _as_tuple3(x: Sequence[int]) -> Tuple[int, int, int]:
    if len(x) != 3:
        raise ValueError(f"Expected length-3 sequence, got {x}")
    return int(x[0]), int(x[1]), int(x[2])


def pad_xyzct(data: np.ndarray, min_xyz: Tuple[int, int, int]) -> np.ndarray:
    """Pad [X,Y,Z,3,T] with zeros so spatial dimensions are at least min_xyz."""
    x, y, z, c, t = data.shape
    px = max(0, min_xyz[0] - x)
    py = max(0, min_xyz[1] - y)
    pz = max(0, min_xyz[2] - z)
    if px == 0 and py == 0 and pz == 0:
        return data
    pad_width = (
        (px // 2, px - px // 2),
        (py // 2, py - py // 2),
        (pz // 2, pz - pz // 2),
        (0, 0),
        (0, 0),
    )
    return np.pad(data, pad_width, mode="constant", constant_values=0)


def compute_mask_np(vel_xyz3: np.ndarray, threshold_rel: float) -> np.ndarray:
    speed = np.sqrt(np.sum(vel_xyz3.astype(np.float32) ** 2, axis=3))
    max_speed = float(np.max(speed))
    if max_speed <= 1e-12:
        return np.zeros(speed.shape, dtype=np.float32)
    return (speed > threshold_rel * max_speed).astype(np.float32)


def normalize_velocity_patch_np(patch_xyz3: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize all components by the largest absolute velocity component in the patch."""
    patch = np.nan_to_num(patch_xyz3.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(patch)))
    if scale <= 1e-12:
        scale = 1.0
    return patch / scale, scale


def velocity_to_magnitude_np(vel_xyz3_norm: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.sum(vel_xyz3_norm.astype(np.float32) ** 2, axis=3))
    max_mag = float(np.max(mag))
    if max_mag > 1e-12:
        mag = mag / max_mag
    return mag.astype(np.float32)


def synthesize_lr_from_hr(
    hr_velocity: torch.Tensor,
    hr_magnitude: torch.Tensor,
    scale_factor: int,
    velocity_noise_std: float = 0.0,
    magnitude_noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create low-resolution input from an HR patch.

    Args:
        hr_velocity: [3, X, Y, Z]
        hr_magnitude: [1, X, Y, Z]

    Returns:
        lr_velocity: [3, X/scale, Y/scale, Z/scale]
        lr_magnitude: [1, X/scale, Y/scale, Z/scale]
    """
    if hr_velocity.ndim != 4 or hr_velocity.shape[0] != 3:
        raise ValueError(f"hr_velocity must be [3,X,Y,Z], got {hr_velocity.shape}")
    if hr_magnitude.ndim != 4 or hr_magnitude.shape[0] != 1:
        raise ValueError(f"hr_magnitude must be [1,X,Y,Z], got {hr_magnitude.shape}")

    v = hr_velocity.unsqueeze(0)
    m = hr_magnitude.unsqueeze(0)
    k = int(scale_factor)
    lr_v = F.avg_pool3d(v, kernel_size=k, stride=k).squeeze(0)
    lr_m = F.avg_pool3d(m, kernel_size=k, stride=k).squeeze(0)

    if velocity_noise_std > 0:
        lr_v = lr_v + torch.randn_like(lr_v) * float(velocity_noise_std)
    if magnitude_noise_std > 0:
        lr_m = lr_m + torch.randn_like(lr_m) * float(magnitude_noise_std)
        lr_m = lr_m.clamp(0.0, 1.0)

    return lr_v.float(), lr_m.float()


class _CaseCache:
    def __init__(self, max_items: int = 2):
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        return None

    def put(self, path: str, value: np.ndarray):
        self.cache[path] = value
        self.cache.move_to_end(path)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


class FourDFlowPatchDataset(Dataset):
    """Randomly samples 3D velocity patches from 4D flow .mat files.

    Each returned item contains:
        lr_input: [4, x_lr, y_lr, z_lr] = [u,v,w,magnitude]
        hr_velocity: [3, x_hr, y_hr, z_hr]
        hr_mask: [1, x_hr, y_hr, z_hr]
        norm_factor: scalar used to normalize the patch
    """

    def __init__(
        self,
        mat_dir: str,
        mat_var_path: Any,
        layout: str = "XYZCT",
        patch_size_hr: Sequence[int] = (64, 64, 16),
        scale_factor: int = 2,
        samples_per_epoch: int = 1000,
        flow_sample_prob: float = 0.9,
        mask_threshold_rel: float = 0.03,
        velocity_noise_std: float = 0.02,
        magnitude_noise_std: float = 0.02,
        cache_items: int = 2,
    ):
        self.mat_paths = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
        if not self.mat_paths:
            raise FileNotFoundError(f"No .mat files found in {mat_dir}")

        self.mat_var_path = mat_var_path
        self.layout = layout
        self.patch_size_hr = _as_tuple3(patch_size_hr)
        self.scale_factor = int(scale_factor)
        self.samples_per_epoch = int(samples_per_epoch)
        self.flow_sample_prob = float(flow_sample_prob)
        self.mask_threshold_rel = float(mask_threshold_rel)
        self.velocity_noise_std = float(velocity_noise_std)
        self.magnitude_noise_std = float(magnitude_noise_std)
        self.cache = _CaseCache(cache_items)

        if any(p % self.scale_factor != 0 for p in self.patch_size_hr):
            raise ValueError("Every HR patch dimension must be divisible by scale_factor")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_case(self, path: str) -> np.ndarray:
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        data = load_4dflow_mat(path, self.mat_var_path, self.layout)
        data = pad_xyzct(data, self.patch_size_hr)
        self.cache.put(path, data)
        return data

    def _choose_patch_start(self, vol_xyz3: np.ndarray) -> Tuple[int, int, int]:
        X, Y, Z, _ = vol_xyz3.shape
        px, py, pz = self.patch_size_hr

        use_flow = random.random() < self.flow_sample_prob
        if use_flow:
            mask = compute_mask_np(vol_xyz3, self.mask_threshold_rel)
            coords = np.argwhere(mask > 0.5)
            if coords.size > 0:
                cx, cy, cz = coords[random.randrange(coords.shape[0])]
                x0 = int(np.clip(cx - random.randrange(px), 0, X - px))
                y0 = int(np.clip(cy - random.randrange(py), 0, Y - py))
                z0 = int(np.clip(cz - random.randrange(pz), 0, Z - pz))
                return x0, y0, z0

        x0 = random.randrange(0, X - px + 1)
        y0 = random.randrange(0, Y - py + 1)
        z0 = random.randrange(0, Z - pz + 1)
        return x0, y0, z0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.mat_paths[random.randrange(len(self.mat_paths))]
        data = self._load_case(path)  # [X,Y,Z,3,T]
        X, Y, Z, C, T = data.shape
        t = random.randrange(T)
        vol = data[:, :, :, :, t]
        x0, y0, z0 = self._choose_patch_start(vol)
        px, py, pz = self.patch_size_hr

        patch = vol[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz, :]  # [X,Y,Z,3]
        patch_norm, norm_factor = normalize_velocity_patch_np(patch)
        mag = velocity_to_magnitude_np(patch_norm)
        mask = compute_mask_np(patch_norm, self.mask_threshold_rel)

        # Convert to PyTorch channel-first [C,X,Y,Z]
        hr_velocity = torch.from_numpy(np.transpose(patch_norm, (3, 0, 1, 2))).float()
        hr_magnitude = torch.from_numpy(mag[None, ...]).float()
        hr_mask = torch.from_numpy(mask[None, ...]).float()

        lr_velocity, lr_magnitude = synthesize_lr_from_hr(
            hr_velocity,
            hr_magnitude,
            scale_factor=self.scale_factor,
            velocity_noise_std=self.velocity_noise_std,
            magnitude_noise_std=self.magnitude_noise_std,
        )
        lr_input = torch.cat([lr_velocity, lr_magnitude], dim=0)

        return {
            "lr_input": lr_input,
            "hr_velocity": hr_velocity,
            "hr_mask": hr_mask,
            "norm_factor": torch.tensor(norm_factor, dtype=torch.float32),
            "case_path": path,
            "time_index": torch.tensor(t, dtype=torch.long),
            "patch_start": torch.tensor([x0, y0, z0], dtype=torch.long),
        }


class PrecomputedFlowPatchDataset(Dataset):
    """Loads .npz files generated by scripts/prepare_dataset.py."""

    def __init__(self, npz_dir: str):
        self.paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        p = self.paths[index]
        z = np.load(p)
        return {
            "lr_input": torch.from_numpy(z["lr_input"]).float(),
            "hr_velocity": torch.from_numpy(z["hr_velocity"]).float(),
            "hr_mask": torch.from_numpy(z["hr_mask"]).float(),
            "norm_factor": torch.tensor(float(z["norm_factor"]), dtype=torch.float32),
            "case_path": str(z["case_path"]),
            "time_index": torch.tensor(int(z["time_index"]), dtype=torch.long),
            "patch_start": torch.from_numpy(z["patch_start"]).long(),
        }
