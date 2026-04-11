import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from architecture import CarComponentCNN, LABELS
from dataset import get_dataloaders
from pathlib import Path
import random
import numpy as np

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS   = 100
LR       = 3e-4
BATCH    = 32
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_AUG = ROOT_DIR / "dataset" / "augmented"
DATA_RAW = ROOT_DIR / "dataset" / "raw"
DATA     = str(DATA_AUG if (DATA_AUG / "annotations.json").exists() else DATA_RAW)
SAVE     = ROOT_DIR / "checkpoints"
SAVE.mkdir(exist_ok=True)
SEED     = 42

# ── Hyperparams ──────────────────────────────────────────────
LABEL_SMOOTHING = 0.05      # 0→0.05, 1→0.95 — improves calibration
MIXUP_ALPHA     = 0.3       # Beta distribution param for mixup
PATIENCE        = 20        # Early stopping patience


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(data_dir: str) -> torch.Tensor:
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


def smooth_labels(targets: torch.Tensor, eps: float = LABEL_SMOOTHING) -> torch.Tensor:
    """Apply label smoothing: 0 → eps, 1 → 1-eps."""
    return targets * (1 - 2 * eps) + eps


def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = MIXUP_ALPHA):
    """Mixup augmentation — interpolate random pairs within the batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(imgs.size(0), device=imgs.device)
    mixed_imgs   = lam * imgs   + (1 - lam) * imgs[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_imgs, mixed_labels


def accuracy_per_label(logits, targets, threshold=0.5):
    preds   = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == targets).float()
    return correct.mean(dim=0)


def train():
    set_seed()
    model = CarComponentCNN().to(DEVICE)
    print(f"Training on: {DATA}  |  device: {DEVICE}")
    print(f"Config: epochs={EPOCHS} lr={LR} mixup={MIXUP_ALPHA} "
          f"label_smooth={LABEL_SMOOTHING} patience={PATIENCE}")

    train_dl, val_dl = get_dataloaders(DATA, BATCH)

    pos_weight = compute_pos_weight(DATA).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR * 10,
        epochs=EPOCHS,
        steps_per_epoch=len(train_dl),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Mixup augmentation
            imgs, labels = mixup_batch(imgs, labels)

            # Label smoothing
            labels = smooth_labels(labels)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
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

        short = ["FL", "FR", "RL", "RR", "Hood"]
        acc_str = " | ".join(f"{n}:{a:.2%}" for n, a in zip(short, per_label_acc))
        print(
            f"Epoch {epoch:03d} | Train: {train_loss/len(train_dl):.4f} "
            f"| Val: {val_loss:.4f} | Exact: {exact_match:.2%} | {acc_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE / "best_model.pt")
            print(f"  >> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Weights saved to: {SAVE / 'best_model.pt'}")

    # ── Auto-run evaluation to update thresholds ─────────────────────────
    print("\n" + "=" * 65)
    print("Running evaluation to compute optimal thresholds...")
    print("=" * 65)
    from evaluate import evaluate
    evaluate(str(SAVE / "best_model.pt"), DATA)


if __name__ == "__main__":
    train()
