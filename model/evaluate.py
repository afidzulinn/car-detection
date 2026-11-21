"""
P0-03, P0-04, P0-08, P1-07, P2-07:
Leak-Free 3-Way Evaluation Script for Paper A.

Enforces:
  1. Threshold tuning (find_optimal_threshold) is performed ONLY on the Validation split ('val').
  2. Tuned thresholds (tau*) are frozen and applied ONCE to the Independent Test split ('test').
  3. Reports both default threshold (0.50) and frozen val-calibrated thresholds (tau*) on Test.
  4. Reports per-elevation Macro-F1, Worst-Elevation Macro-F1, Expected Calibration Error (ECE),
     Brier Score, and Auxiliary Pose Head accuracy (P1-07).
"""

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS
from dataset import CarDataset, get_dataloaders
from decision_policy import (
    calibrate_thresholds_on_val,
    compute_calibration_summary,
    find_optimal_threshold_on_val,
    predict_from_prob,
)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[1]

# Backward-compatible export
find_optimal_threshold = find_optimal_threshold_on_val


def _collect_split_outputs(model: torch.nn.Module, ds: CarDataset, batch_size: int = 16):
    all_probs = []
    all_targets = []
    all_pose_preds = []
    all_pose_targets = []
    all_elevs = []
    all_angles = []

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    idx_cursor = 0
    with torch.no_grad():
        for batch in dl:
            imgs, labels = batch[0], batch[1]
            bsz = imgs.size(0)
            imgs = imgs.to(DEVICE, non_blocking=(DEVICE == "cuda"))
            logits, pose_logits = model(imgs, return_pose=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            pose_pred = torch.argmax(pose_logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(labels.numpy())
            all_pose_preds.append(pose_pred)

            for b_i in range(bsz):
                rec = ds.data[idx_cursor + b_i]
                ang = int(rec.get("angle", 0)) % 360
                all_angles.append(ang)
                all_pose_targets.append(ang // 45)
                all_elevs.append(rec.get("elevation", "mid"))
            idx_cursor += bsz

    return {
        "probs": np.vstack(all_probs),
        "targets": np.vstack(all_targets),
        "pose_preds": np.concatenate(all_pose_preds),
        "pose_targets": np.array(all_pose_targets, dtype=int),
        "elevations": all_elevs,
        "angles": all_angles,
    }


def _compute_split_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds_map: dict,
    elevations: list,
    pose_preds: np.ndarray,
    pose_targets: np.ndarray,
) -> dict:
    thr_arr = np.array([float(thresholds_map[l]) for l in LABELS], dtype=np.float64)
    preds = predict_from_prob(probs, thr_arr)

    per_label = {}
    f1_list, auc_list, acc_list = [], [], []
    for i, label in enumerate(LABELS):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        y_prob = probs[:, i]
        acc = float((y_pred == y_true).mean())
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        per_label[label] = {
            "threshold": float(thresholds_map[label]),
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1": round(f1, 5),
            "roc_auc": round(auc, 5) if not np.isnan(auc) else None,
        }
        acc_list.append(acc)
        f1_list.append(f1)
        if not np.isnan(auc):
            auc_list.append(auc)

    exact_match = float(np.all(preds == targets, axis=1).mean())
    mean_acc = float(np.mean(acc_list))
    macro_f1 = float(np.mean(f1_list))
    macro_auc = float(np.mean(auc_list)) if auc_list else 0.0

    # Per-elevation breakdown & Worst-view Macro-F1
    by_elev = {}
    elev_f1s = []
    for elev_name in ("mid", "high", "top"):
        idxs = [idx for idx, e in enumerate(elevations) if e == elev_name]
        if not idxs:
            continue
        sub_t = targets[idxs]
        sub_p = preds[idxs]
        sub_f1s = [
            float(f1_score(sub_t[:, i], sub_p[:, i], zero_division=0))
            for i in range(len(LABELS))
        ]
        e_macro_f1 = float(np.mean(sub_f1s))
        e_acc = float((sub_p == sub_t).mean())
        e_exact = float(np.all(sub_p == sub_t, axis=1).mean())
        by_elev[elev_name] = {
            "n": len(idxs),
            "mean_accuracy": round(e_acc, 5),
            "macro_f1": round(e_macro_f1, 5),
            "exact_match": round(e_exact, 5),
        }
        elev_f1s.append(e_macro_f1)

    worst_elev_f1 = min(elev_f1s) if elev_f1s else macro_f1
    calibration = compute_calibration_summary(probs, targets, LABELS)
    pose_acc = float((pose_preds == pose_targets).mean()) if len(pose_targets) > 0 else 0.0

    return {
        "n_samples": int(len(targets)),
        "mean_accuracy": round(mean_acc, 5),
        "macro_f1": round(macro_f1, 5),
        "macro_roc_auc": round(macro_auc, 5),
        "exact_match": round(exact_match, 5),
        "worst_elevation_macro_f1": round(worst_elev_f1, 5),
        "pose_azimuth_accuracy": round(pose_acc, 5),
        "calibration": calibration,
        "by_elevation": by_elev,
        "per_label": per_label,
    }


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    protocol: str = "T0",
    roi_policy: str = "legacy_fixed",
):
    gpu_info = f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""
    print(f"Evaluating on OS: {platform.system()} | Device: {DEVICE}{gpu_info}")
    print(f"Protocol: {protocol} | ROI Policy: {roi_policy}")

    model = CarComponentCNN().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    ckpt_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()[:16]

    val_ds = CarDataset(data_dir, split="val", protocol=protocol, roi_policy=roi_policy)
    test_ds = CarDataset(data_dir, split="test", protocol=protocol, roi_policy=roi_policy)
    combined_192_ds = CarDataset(
        data_dir, split="val", protocol=protocol, roi_policy=roi_policy, two_way_compat=True
    )

    val_out = _collect_split_outputs(model, val_ds)
    test_out = _collect_split_outputs(model, test_ds)
    comb_out = _collect_split_outputs(model, combined_192_ds)

    default_thr = {l: 0.50 for l in LABELS}
    # P0-04: Calibrate thresholds ONLY on validation split (N=96)
    val_tuned_thr = calibrate_thresholds_on_val(val_out["probs"], val_out["targets"], LABELS)

    val_metrics_05 = _compute_split_metrics(
        val_out["probs"], val_out["targets"], default_thr,
        val_out["elevations"], val_out["pose_preds"], val_out["pose_targets"]
    )
    val_metrics_opt = _compute_split_metrics(
        val_out["probs"], val_out["targets"], val_tuned_thr,
        val_out["elevations"], val_out["pose_preds"], val_out["pose_targets"]
    )

    # Apply frozen validation thresholds ONCE to the independent test split (N=96)
    test_metrics_05 = _compute_split_metrics(
        test_out["probs"], test_out["targets"], default_thr,
        test_out["elevations"], test_out["pose_preds"], test_out["pose_targets"]
    )
    test_metrics_frozen_opt = _compute_split_metrics(
        test_out["probs"], test_out["targets"], val_tuned_thr,
        test_out["elevations"], test_out["pose_preds"], test_out["pose_targets"]
    )

    # Legacy 2-way N=192 reference (for historical continuity comparison only)
    comb_metrics_05 = _compute_split_metrics(
        comb_out["probs"], comb_out["targets"], default_thr,
        comb_out["elevations"], comb_out["pose_preds"], comb_out["pose_targets"]
    )

    print("\n" + "=" * 72)
    print(f"1. VALIDATION SPLIT (N={val_metrics_05['n_samples']}) — Threshold Calibration Split")
    print("=" * 72)
    print(f"  @ Default tau=0.50 : Mean Acc={val_metrics_05['mean_accuracy']:.2%} | Macro F1={val_metrics_05['macro_f1']:.4f} | Exact Match={val_metrics_05['exact_match']:.2%}")
    print(f"  @ Val-Tuned tau*   : Mean Acc={val_metrics_opt['mean_accuracy']:.2%} | Macro F1={val_metrics_opt['macro_f1']:.4f} | Exact Match={val_metrics_opt['exact_match']:.2%}")
    print(f"  Frozen Val tau*    : {val_tuned_thr}")

    print("\n" + "=" * 72)
    print(f"2. INDEPENDENT TEST SPLIT (N={test_metrics_05['n_samples']}) — Zero Threshold Selection Bias")
    print("=" * 72)
    print(f"  @ Default tau=0.50 : Mean Acc={test_metrics_05['mean_accuracy']:.2%} | Macro F1={test_metrics_05['macro_f1']:.4f} | Exact Match={test_metrics_05['exact_match']:.2%}")
    print(f"  @ Frozen Val tau*  : Mean Acc={test_metrics_frozen_opt['mean_accuracy']:.2%} | Macro F1={test_metrics_frozen_opt['macro_f1']:.4f} | Exact Match={test_metrics_frozen_opt['exact_match']:.2%}")
    print(f"  Macro ROC-AUC      : {test_metrics_frozen_opt['macro_roc_auc']:.4f} | Worst-Elev F1={test_metrics_frozen_opt['worst_elevation_macro_f1']:.4f}")
    print(f"  Calibration (Test) : ECE={test_metrics_frozen_opt['calibration']['macro_ece']:.4f} | Brier={test_metrics_frozen_opt['calibration']['macro_brier_score']:.4f}")
    print(f"  Pose Azimuth Acc   : {test_metrics_frozen_opt['pose_azimuth_accuracy']:.2%}")

    # Save frozen validation thresholds with audit metadata (P0-04)
    thr_path = ROOT_DIR / "checkpoints" / "thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(val_tuned_thr, f, indent=2)

    report_payload = {
        "protocol": protocol,
        "roi_policy": roi_policy,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_prefix": ckpt_sha,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_val_thresholds": val_tuned_thr,
        "validation_split_n96": {
            "at_050": val_metrics_05,
            "at_val_tuned": val_metrics_opt,
        },
        "independent_test_split_n96": {
            "at_050": test_metrics_05,
            "at_frozen_val_thresholds": test_metrics_frozen_opt,
        },
        "historical_combined_n192_at_050": comb_metrics_05,
    }
    out_dir = ROOT_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evaluation_{protocol.lower()}_{roi_policy}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nSaved auditable evaluation report to: {out_file}")
    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="T0", choices=["T0", "T1", "T2"])
    parser.add_argument("--roi-policy", type=str, default="legacy_fixed")
    parser.add_argument("--ckpt", type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"))
    args = parser.parse_args()

    data_aug = ROOT_DIR / "dataset" / "augmented"
    data_raw = ROOT_DIR / "dataset" / "raw"
    data_dir = str(data_aug if (data_aug / "annotations.json").exists() else data_raw)

    evaluate(args.ckpt, data_dir, protocol=args.protocol, roi_policy=args.roi_policy)
