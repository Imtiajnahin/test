from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from src.dcsrn_flow.data import FourDFlowPatchDataset
from src.dcsrn_flow.utils import ensure_dir, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--num_patches", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    data_cfg = cfg["data"]

    mat_dir = data_cfg["train_mat_dir"] if args.split == "train" else data_cfg["val_mat_dir"]
    out_dir = data_cfg["train_npz_dir"] if args.split == "train" else data_cfg["val_npz_dir"]
    ensure_dir(out_dir)

    ds = FourDFlowPatchDataset(
        mat_dir=mat_dir,
        mat_var_path=data_cfg["mat_var_path"],
        layout=data_cfg.get("layout", "XYZCT"),
        patch_size_hr=data_cfg["patch_size_hr"],
        scale_factor=data_cfg["scale_factor"],
        samples_per_epoch=args.num_patches,
        flow_sample_prob=data_cfg.get("flow_sample_prob", 0.9),
        mask_threshold_rel=data_cfg.get("mask_threshold_rel", 0.03),
        velocity_noise_std=data_cfg.get("velocity_noise_std", 0.02) if args.split == "train" else 0.0,
        magnitude_noise_std=data_cfg.get("magnitude_noise_std", 0.02) if args.split == "train" else 0.0,
    )

    for i in tqdm(range(args.num_patches), desc=f"precompute-{args.split}"):
        item = ds[i]
        out_path = os.path.join(out_dir, f"{args.split}_patch_{i:06d}.npz")
        np.savez_compressed(
            out_path,
            lr_input=item["lr_input"].numpy().astype(np.float32),
            hr_velocity=item["hr_velocity"].numpy().astype(np.float32),
            hr_mask=item["hr_mask"].numpy().astype(np.float32),
            norm_factor=np.asarray(float(item["norm_factor"]), dtype=np.float32),
            case_path=str(item["case_path"]),
            time_index=np.asarray(int(item["time_index"]), dtype=np.int64),
            patch_start=item["patch_start"].numpy().astype(np.int64),
        )

    print(f"Saved {args.num_patches} patches to {out_dir}")


if __name__ == "__main__":
    main()
