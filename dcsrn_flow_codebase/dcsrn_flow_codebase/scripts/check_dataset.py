from __future__ import annotations

import argparse

from src.dcsrn_flow.data import FourDFlowPatchDataset
from src.dcsrn_flow.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    mat_dir = data_cfg["train_mat_dir"] if args.split == "train" else data_cfg["val_mat_dir"]
    samples = 1

    ds = FourDFlowPatchDataset(
        mat_dir=mat_dir,
        mat_var_path=data_cfg["mat_var_path"],
        layout=data_cfg.get("layout", "XYZCT"),
        patch_size_hr=data_cfg["patch_size_hr"],
        scale_factor=data_cfg["scale_factor"],
        samples_per_epoch=samples,
        flow_sample_prob=data_cfg.get("flow_sample_prob", 0.9),
        mask_threshold_rel=data_cfg.get("mask_threshold_rel", 0.03),
        velocity_noise_std=data_cfg.get("velocity_noise_std", 0.02),
        magnitude_noise_std=data_cfg.get("magnitude_noise_std", 0.02),
    )

    item = ds[0]
    print("Dataset item loaded successfully.")
    print("lr_input shape   :", tuple(item["lr_input"].shape), "[u,v,w,magnitude]")
    print("hr_velocity shape:", tuple(item["hr_velocity"].shape), "[u,v,w]")
    print("hr_mask shape    :", tuple(item["hr_mask"].shape))
    print("norm_factor      :", float(item["norm_factor"]))
    print("case_path        :", item["case_path"])
    print("time_index       :", int(item["time_index"]))
    print("patch_start      :", item["patch_start"].tolist())


if __name__ == "__main__":
    main()
