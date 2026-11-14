"""
Evaluation script (supports Windows & Ubuntu/Linux, NVIDIA CUDA GPU & CPU) with:
  - Per-label classification report (precision / recall / F1)
  - ROC-AUC per label
  - Exact-match accuracy
  - Per-label optimal threshold search (saved to checkpoints/thresholds.json)

Usage:
    python model/evaluate.py
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Ensure local imports work regardless of current working directory on Windows or Ubuntu
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS
from dataset import get_dataloaders

DEVICE   = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[1]


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Find the probability threshold in [0.20, 0.80] that maximises F1 on the val set.
    When multiple thresholds achieve the same peak F1 (e.g. F1 = 1.0000 across a wide
    margin), select the threshold closest to 0.50 to maximise decision margin stability.
    """
    best_f1, best_thr = -1.0, 0.50
    for thr in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thr).astype(int)
        f1    = f1_score(targets, preds, zero_division=0)
        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and abs(thr - 0.50) < abs(best_thr - 0.50)
        ):
            best_f1, best_thr = f1, thr
    return round(float(best_thr), 2)


def evaluate(checkpoint_path: str, data_dir: str):
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""
    print(f"Evaluating on OS: {platform.system()} | Device: {DEVICE}{gpu_info}")

    model = CarComponentCNN().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    _, val_dl = get_dataloaders(data_dir, batch_size=16)

    all_probs   = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs    = imgs.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            logits  = model(imgs)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    all_probs   = np.vstack(all_probs)    # (N, 5)
    all_targets = np.vstack(all_targets)  # (N, 5)

    # ── Per-label report at default threshold 0.5 ────────────────────────────
    print("=" * 65)
    print("PER-LABEL REPORT  (threshold = 0.50)")
    print("=" * 65)
    all_preds_05 = (all_probs > 0.5).astype(int)
    for i, label in enumerate(LABELS):
        print(f"\n-- {label.upper()} --")
        print(classification_report(
            all_targets[:, i], all_preds_05[:, i],
            target_names=["Closed", "Open"], zero_division=0
        ))
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            print(f"  ROC-AUC : {auc:.4f}")
        except ValueError:
            print("  ROC-AUC : N/A (only one class present)")

    exact_05 = np.all(all_preds_05 == all_targets, axis=1).mean()
    mean_acc_05 = (all_preds_05 == all_targets).mean()
    print(f"\n{'='*65}")
    print(f"Mean Per-Label Accuracy @ 0.50 : {mean_acc_05:.2%}")
    print(f"Exact Match @ 0.50             : {exact_05:.2%}")

    # ── Per-label optimal threshold search ───────────────────────────────────
    print("\n" + "=" * 65)
    print("PER-LABEL OPTIMAL THRESHOLD SEARCH  (max F1 on val set)")
    print("=" * 65)

    thresholds = {}
    all_preds_opt = np.zeros_like(all_preds_05)

    for i, label in enumerate(LABELS):
        thr = find_optimal_threshold(all_probs[:, i], all_targets[:, i])
        thresholds[label] = thr
        all_preds_opt[:, i] = (all_probs[:, i] >= thr).astype(int)
        f1  = f1_score(all_targets[:, i], all_preds_opt[:, i], zero_division=0)
        acc = (all_preds_opt[:, i] == all_targets[:, i]).mean()
        print(f"  {label:<22} optimal_thr={thr:.2f}  Acc={acc:.2%}  F1={f1:.4f}")

    exact_opt = np.all(all_preds_opt == all_targets, axis=1).mean()
    mean_acc_opt = (all_preds_opt == all_targets).mean()
    print(f"\nMean Per-Label Accuracy @ optimal thresholds : {mean_acc_opt:.2%}")
    print(f"Exact Match @ optimal thresholds             : {exact_opt:.2%}")
    print(f"Exact Match improvement over 0.50            : {exact_opt - exact_05:+.2%}")

    # ── Save thresholds so inference can load them ────────────────────────────
    thr_path = ROOT_DIR / "checkpoints" / "thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved to: {thr_path}")


if __name__ == "__main__":
    data_aug = ROOT_DIR / "dataset" / "augmented"
    data_raw = ROOT_DIR / "dataset" / "raw"
    data_dir = str(data_aug if (data_aug / "annotations.json").exists() else data_raw)

    ckpt_path = ROOT_DIR / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        print("ERROR: checkpoints/best_model.pt not found. Run train.py first.")
        sys.exit(1)

    evaluate(str(ckpt_path), data_dir)
