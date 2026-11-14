"""
Cross-platform training script (Windows & Ubuntu/Linux) for CarComponentCNN.
Supports:
  - Automatic device selection: CUDA (Ubuntu/Windows GPU) -> MPS -> CPU
  - Automatic Mixed Precision (AMP) when running on CUDA GPUs
  - Cross-platform pathlib paths and multiprocessing worker auto-tuning
  - Multi-task learning: Multi-label BCE loss + Auxiliary 8-angle Pose CE loss
"""

import argparse
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Ensure local imports work regardless of current working directory on Windows or Ubuntu
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS
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
        if "+cpu" in torch.__version__:
            print(
                "  [TIP] PyTorch CPU build detected. To enable NVIDIA RTX GPU quickly with uv, run:\n"
                "        uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            )


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def compute_pos_weight(train_ds: CarDataset) -> torch.Tensor:
    """
    Compute per-label positive class weights for BCEWithLogitsLoss:
        pos_weight[i] = (# negative samples) / (# positive samples)
    """
    pos = torch.zeros(len(LABELS))
    neg = torch.zeros(len(LABELS))
    for item in train_ds.data:
        for i, label in enumerate(LABELS):
            if item["labels"][label] == 1:
                pos[i] += 1
            else:
                neg[i] += 1

    weights = neg / pos.clamp(min=1)
    print("pos_weight per label:")
    for name, w in zip(LABELS, weights.tolist()):
        print(f"  {name}: {w:.3f}")
    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CarComponentCNN (supports Windows & Ubuntu/Linux)"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2.5e-3, help="Peak learning rate for OneCycleLR")
    parser.add_argument("--pose-weight", type=float, default=0.35, help="Auxiliary camera pose loss weight")
    parser.add_argument("--label-smoothing", type=float, default=0.03, help="Label smoothing factor for BCE calibration")
    parser.add_argument("--max-aug", type=int, default=0, help="Offline augmented copies per sample (0 = online ColorJitter only)")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (auto: 0 on Windows, 2-4 on Ubuntu)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    use_amp = device.startswith("cuda")

    model = CarComponentCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_device_diagnostics(device, use_amp)
    print(f"Model parameters: {n_params:,} | Training on: {args.data_dir}")

    train_dl, val_dl = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        return_angle_train=True,
        max_aug_per_sample=args.max_aug,
    )

    pos_weight    = compute_pos_weight(train_dl.dataset).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_criterion  = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_bce  = 0.0
        for imgs, labels, angle_idx in train_dl:
            imgs      = imgs.to(device, non_blocking=use_amp)
            labels    = labels.to(device, non_blocking=use_amp)
            angle_idx = angle_idx.to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, pose_logits = model(imgs, return_pose=True)
                # Apply binary label smoothing: y_smooth = y * (1 - eps) + 0.5 * eps
                eps = args.label_smoothing
                smooth_labels = labels * (1.0 - eps) + 0.5 * eps
                loss_bce  = bce_criterion(logits, smooth_labels)
                loss_pose = ce_criterion(pose_logits, angle_idx)
                loss      = loss_bce + args.pose_weight * loss_pose

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_bce  += loss_bce.item()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        per_label_acc = torch.zeros(len(LABELS))
        exact_match   = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                bs = imgs.size(0)
                imgs   = imgs.to(device, non_blocking=use_amp)
                labels = labels.to(device, non_blocking=use_amp)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    batch_loss = bce_criterion(logits, labels).item()
                val_loss      += batch_loss * bs
                preds          = (torch.sigmoid(logits) > 0.5).float()
                per_label_acc += (preds == labels).float().sum(dim=0).cpu()
                exact_match   += (preds == labels).all(dim=1).float().sum().item()
                n_val_samples += bs

        val_loss      /= max(1, n_val_samples)
        per_label_acc /= max(1, n_val_samples)
        exact_match   /= max(1, n_val_samples)
        mean_acc       = per_label_acc.mean().item()
        dt             = time.time() - t0

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
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
            print(
                f"  >> Saved best model (mean_acc={mean_acc:.2%}, val_loss={val_loss:.4f})"
            )

    print(
        f"\nTraining complete. Best mean_acc={best_mean_acc:.2%}, "
        f"best_val_loss={best_val_loss:.4f}"
    )
    print(f"Weights saved to: {SAVE_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
