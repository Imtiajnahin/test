from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dcsrn_flow import DCSRNFlow, DCSRNFlowLoss, FourDFlowPatchDataset, PrecomputedFlowPatchDataset
from src.dcsrn_flow.metrics import mean_abs_divergence, velocity_rmse
from src.dcsrn_flow.utils import AverageMeter, ensure_dir, get_device, load_config, set_seed


def make_dataset(cfg: dict, split: str):
    data_cfg = cfg["data"]
    if data_cfg.get("use_precomputed", False):
        npz_dir = data_cfg["train_npz_dir"] if split == "train" else data_cfg["val_npz_dir"]
        return PrecomputedFlowPatchDataset(npz_dir)

    mat_dir = data_cfg["train_mat_dir"] if split == "train" else data_cfg["val_mat_dir"]
    samples = data_cfg["samples_per_epoch"] if split == "train" else data_cfg["val_samples"]
    noise_v = data_cfg["velocity_noise_std"] if split == "train" else 0.0
    noise_m = data_cfg["magnitude_noise_std"] if split == "train" else 0.0

    return FourDFlowPatchDataset(
        mat_dir=mat_dir,
        mat_var_path=data_cfg["mat_var_path"],
        layout=data_cfg.get("layout", "XYZCT"),
        patch_size_hr=data_cfg["patch_size_hr"],
        scale_factor=data_cfg["scale_factor"],
        samples_per_epoch=samples,
        flow_sample_prob=data_cfg.get("flow_sample_prob", 0.9),
        mask_threshold_rel=data_cfg.get("mask_threshold_rel", 0.03),
        velocity_noise_std=noise_v,
        magnitude_noise_std=noise_m,
    )


def make_loader(cfg: dict, split: str):
    ds = make_dataset(cfg, split)
    train_cfg = cfg["train"]
    return DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val: float, cfg: dict):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, mixed_precision: bool) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_meter = AverageMeter()
    mse_meter = AverageMeter()
    bce_meter = AverageMeter()
    div_meter = AverageMeter()
    rmse_meter = AverageMeter()
    div_metric_meter = AverageMeter()

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in pbar:
        lr_input = batch["lr_input"].to(device, non_blocking=True)
        hr_velocity = batch["hr_velocity"].to(device, non_blocking=True)
        hr_mask = batch["hr_mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=mixed_precision and device.type == "cuda"):
                pred = model(lr_input)
                loss_dict = criterion(pred, hr_velocity, hr_mask)
                loss = loss_dict["total"]

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_size = lr_input.shape[0]
        total_meter.update(loss.detach().item(), batch_size)
        mse_meter.update(loss_dict["mse"].item(), batch_size)
        bce_meter.update(loss_dict["bce"].item(), batch_size)
        div_meter.update(loss_dict["div"].item(), batch_size)

        with torch.no_grad():
            rmse_meter.update(velocity_rmse(pred["velocity"], hr_velocity, hr_mask).item(), batch_size)
            div_metric_meter.update(mean_abs_divergence(pred["velocity"], hr_mask).item(), batch_size)

        pbar.set_postfix(
            total=f"{total_meter.avg:.4f}",
            mse=f"{mse_meter.avg:.4f}",
            div=f"{div_meter.avg:.4f}",
        )

    return {
        "total": total_meter.avg,
        "mse": mse_meter.avg,
        "bce": bce_meter.avg,
        "div_loss": div_meter.avg,
        "rmse": rmse_meter.avg,
        "mean_abs_div": div_metric_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print("Using device:", device)

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    model = DCSRNFlow(**cfg["model"]).to(device)
    criterion = DCSRNFlowLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scaler = GradScaler(enabled=cfg["train"].get("mixed_precision", True) and device.type == "cuda")

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    ckpt_dir = cfg["train"].get("checkpoint_dir", "./checkpoints")
    ensure_dir(ckpt_dir)

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        train_stats = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, mixed_precision=cfg["train"].get("mixed_precision", True)
        )
        val_stats = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False, mixed_precision=cfg["train"].get("mixed_precision", True)
        )

        print(
            "Train: " + ", ".join([f"{k}={v:.6f}" for k, v in train_stats.items()])
        )
        print(
            "Val:   " + ", ".join([f"{k}={v:.6f}" for k, v in val_stats.items()])
        )

        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch, best_val, cfg)

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch, best_val, cfg)
            print(f"Saved new best checkpoint: val_total={best_val:.6f}")

        if epoch % int(cfg["train"].get("save_every", 10)) == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, best_val, cfg)


if __name__ == "__main__":
    main()
