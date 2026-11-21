"""
Cross-platform training script (Windows & Ubuntu/Linux) for CarComponentCNN (P0-10, P1-03, P2-01, P2-02).
Supports:
  - Automatic device selection: CUDA -> MPS -> CPU + CUDA AMP FP16
  - Configurable split protocol (T0, T1, T2) and ROI policy (legacy_fixed, full_vehicle, full_frame)
  - Optional strict deterministic mode (--deterministic) (P0-10)
  - Multi-task learning: Multi-label BCE loss + Auxiliary Azimuth (8-class) + optional Elevation (3-class) Pose CE loss
  - Automatic checkpoint metadata & model card saving
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS, ModelConfig
from dataset import CarDataset, get_dataloaders

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_AUG = ROOT_DIR / "dataset" / "augmented"
DATA_RAW = ROOT_DIR / "dataset" / "raw"
DEFAULT_DATA = str(DATA_AUG if (DATA_AUG / "annotations.json").exists() else DATA_RAW)
SAVE_DIR = ROOT_DIR / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def select_device(requested: str = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_diagnostics(device: str, use_amp: bool):
    os_info = f"{platform.system()} ({platform.release()})"
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"OS: {os_info} | Device: {device} [{gpu_name}, {vram_gb:.1f} GB VRAM] "
            f"| CUDA: {torch.version.cuda} | AMP FP16: {use_amp}"
        )
    else:
        print(f"OS: {os_info} | Device: {device} | PyTorch: {torch.__version__}")


def set_seed(seed: int = SEED, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def compute_pos_weight(train_ds: CarDataset) -> torch.Tensor:
    pos = torch.zeros(len(LABELS))
    neg = torch.zeros(len(LABELS))
    for item in train_ds.data:
        for i, label in enumerate(LABELS):
            if item["labels"][label] == 1:
                pos[i] += 1
            else:
                neg[i] += 1

    weights = neg / pos.clamp(min=1)
    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CarComponentCNN (supports Windows & Ubuntu/Linux)"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--pose-weight", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--max-aug", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="T0", choices=["T0", "T1", "T2"])
    parser.add_argument("--roi-policy", type=str, default="legacy_fixed", choices=["legacy_fixed", "full_vehicle", "full_frame", "adaptive_fg"])
    parser.add_argument("--use-elevation-pose", action="store_true")
    parser.add_argument("--use-component-attention", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-name", type=str, default="best_model.pt")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = select_device(args.device)
    use_amp = device.startswith("cuda")

    cfg = ModelConfig(
        use_elevation_pose=args.use_elevation_pose,
        use_component_attention=args.use_component_attention,
    )
    model = CarComponentCNN(config=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_device_diagnostics(device, use_amp)
    print(
        f"Model parameters: {n_params:,} | Protocol: {args.protocol} | "
        f"ROI: {args.roi_policy} | ElevPose: {args.use_elevation_pose} | "
        f"CompAttn: {args.use_component_attention}"
    )

    train_dl, val_dl = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        return_angle_train=True,
        return_elevation_train=args.use_elevation_pose,
        max_aug_per_sample=args.max_aug,
        protocol=args.protocol,
        roi_policy=args.roi_policy,
    )

    pos_weight = compute_pos_weight(train_dl.dataset).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.2,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_mean_acc = 0.0
    save_path = SAVE_DIR / args.save_name

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        for batch in train_dl:
            if args.use_elevation_pose:
                imgs, labels, angle_idx, elev_idx = batch
                elev_idx = elev_idx.to(device, non_blocking=use_amp)
            else:
                imgs, labels, angle_idx = batch
                elev_idx = None

            imgs = imgs.to(device, non_blocking=use_amp)
            labels = labels.to(device, non_blocking=use_amp)
            angle_idx = angle_idx.to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux = model(imgs, return_aux=True)
                eps = args.label_smoothing
                smooth_labels = labels * (1.0 - eps) + 0.5 * eps
                loss_bce = bce_criterion(logits, smooth_labels)
                loss_pose = ce_criterion(aux["pose_logits"], angle_idx)
                if args.use_elevation_pose and aux["elevation_logits"] is not None and elev_idx is not None:
                    loss_elev = ce_criterion(aux["elevation_logits"], elev_idx)
                    loss_pose = 0.5 * (loss_pose + loss_elev)
                loss = loss_bce + args.pose_weight * loss_pose

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_bce += loss_bce.item()

        model.eval()
        val_loss = 0.0
        per_label_acc = torch.zeros(len(LABELS))
        exact_match = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                bs = imgs.size(0)
                imgs = imgs.to(device, non_blocking=use_amp)
                labels = labels.to(device, non_blocking=use_amp)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    batch_loss = bce_criterion(logits, labels).item()
                val_loss += batch_loss * bs
                preds = (torch.sigmoid(logits) >= 0.50).float()
                per_label_acc += (preds == labels).float().sum(dim=0).cpu()
                exact_match += (preds == labels).all(dim=1).float().sum().item()
                n_val_samples += bs

        val_loss /= max(1, n_val_samples)
        per_label_acc /= max(1, n_val_samples)
        exact_match /= max(1, n_val_samples)
        mean_acc = per_label_acc.mean().item()
        dt = time.time() - t0

        short = ["FL", "FR", "RL", "RR", "Hood"]
        acc_str = " | ".join(f"{n}:{a:.1%}" for n, a in zip(short, per_label_acc))
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:4.1f}s) | "
            f"BCE: {train_bce/len(train_dl):.4f} | Val: {val_loss:.4f} | "
            f"Mean: {mean_acc:.1%} | Exact: {exact_match:.1%} | {acc_str}"
        )

        if (mean_acc > best_mean_acc) or (
            abs(mean_acc - best_mean_acc) < 1e-6 and val_loss < best_val_loss
        ):
            best_mean_acc = max(best_mean_acc, mean_acc)
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(model.state_dict(), save_path)
            print(
                f"  >> Saved best model (mean_acc={mean_acc:.2%}, val_loss={val_loss:.4f})"
            )

    ckpt_sha = hashlib.sha256(save_path.read_bytes()).hexdigest()
    card_dir = ROOT_DIR / "results" / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    card_path = card_dir / f"{save_path.stem}_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_file": str(save_path.name),
                "sha256": ckpt_sha,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "deterministic": args.deterministic,
                "protocol": args.protocol,
                "roi_policy": args.roi_policy,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_aug": args.max_aug,
                "model_config": cfg.to_dict(),
                "n_parameters": n_params,
                "best_val_mean_acc_at_050": round(best_mean_acc, 5),
                "best_val_loss": round(best_val_loss, 5),
            },
            f,
            indent=2,
        )

    print(
        f"\nTraining complete. Best val mean_acc={best_mean_acc:.2%}, "
        f"best_val_loss={best_val_loss:.4f}"
    )
    print(f"Weights saved to: {save_path} (card: {card_path})")


if __name__ == "__main__":
    train()
