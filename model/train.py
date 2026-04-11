import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from architecture import CarComponentCNN, LABELS
from dataset import get_dataloaders
from pathlib import Path
import random
import numpy as np

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS   = 50
LR       = 3e-4
BATCH    = 32
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_AUG = ROOT_DIR / "dataset" / "augmented"
DATA_RAW = ROOT_DIR / "dataset" / "raw"
DATA     = str(DATA_AUG if (DATA_AUG / "annotations.json").exists() else DATA_RAW)
SAVE     = ROOT_DIR / "checkpoints"
SAVE.mkdir(exist_ok=True)
SEED     = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(data_dir: str) -> torch.Tensor:
    """
    Compute per-label positive class weights for BCEWithLogitsLoss.

    pos_weight[i] = (# negative samples) / (# positive samples)

    With the angle-stratified split the distribution is already ~50/50,
    so weights will all be ~1.0. Kept for robustness against any residual
    imbalance after augmentation.
    """
    from dataset import CarDataset
    ds = CarDataset(data_dir, split="train")
    pos = torch.zeros(len(LABELS))
    neg = torch.zeros(len(LABELS))
    for item in ds.data:
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


def accuracy_per_label(logits, targets, threshold=0.5):
    preds   = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == targets).float()
    return correct.mean(dim=0)


def train():
    set_seed()
    model = CarComponentCNN().to(DEVICE)
    print(f"Training on: {DATA}  |  device: {DEVICE}")

    train_dl, val_dl = get_dataloaders(DATA, BATCH)

    # Class-balanced loss
    pos_weight = compute_pos_weight(DATA).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        per_label_acc = torch.zeros(len(LABELS))
        exact_match   = 0.0

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits        = model(imgs)
                val_loss     += criterion(logits, labels).item()
                per_label_acc += accuracy_per_label(logits, labels).cpu()
                preds          = (torch.sigmoid(logits) > 0.5).float()
                exact_match   += (preds == labels).all(dim=1).float().mean().item()

        val_loss      /= len(val_dl)
        per_label_acc /= len(val_dl)
        exact_match   /= len(val_dl)
        scheduler.step()

        short = ["FL", "FR", "RL", "RR", "Hood"]
        acc_str = " | ".join(f"{n}:{a:.2%}" for n, a in zip(short, per_label_acc))
        print(
            f"Epoch {epoch:03d} | Train: {train_loss/len(train_dl):.4f} "
            f"| Val: {val_loss:.4f} | Exact: {exact_match:.2%} | {acc_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE / "best_model.pt")
            print(f"  >> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Weights saved to: {SAVE / 'best_model.pt'}")


if __name__ == "__main__":
    train()
