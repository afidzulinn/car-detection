"""
Deep Evaluation Suite for Paper 1 (Multi-Scale PA-SCAN, N=192 across 3 Elevations):
  1. Per-Elevation Breakdown (mid, high, top)
  2. Per-Azimuth Angle Breakdown (0, 45, 90, 135, 180, 225, 270, 315 deg)
  3. Visual Corruption & Photometric Robustness Stress Test:
       - Clean (Baseline)
       - Gaussian Sensor Noise (sigma = 0.03, 0.05, 0.08)
       - Illumination Shift (Brightness 0.75x Dark, 1.25x Bright)
       - Contrast Compression (0.75x Contrast)

Outputs:
  - JSON summary: docs/tables/paper1_deep_evaluation.json
  - LaTeX tables: docs/tables/elevation_angle_breakdown.tex & docs/tables/robustness_stress_test.tex
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import f1_score
from torchvision import transforms

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS, crop_car_roi
from dataset import CarDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_thresholds():
    ckpt_path = ROOT_DIR / "checkpoints" / "best_model.pt"
    thr_path = ROOT_DIR / "checkpoints" / "thresholds.json"

    model = CarComponentCNN().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    if thr_path.exists():
        with open(thr_path, encoding="utf-8") as f:
            t_map = json.load(f)
        thresholds = np.array([float(t_map.get(l, 0.5)) for l in LABELS])
    else:
        thresholds = np.array([0.5] * len(LABELS))

    return model, thresholds


def compute_subset_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> dict:
    preds = (probs >= thresholds[None, :]).astype(int)
    per_label_acc = (preds == targets).mean(axis=0)
    mean_acc = float((preds == targets).mean())
    exact_match = float(np.all(preds == targets, axis=1).mean())
    f1s = [
        float(f1_score(targets[:, i], preds[:, i], zero_division=0))
        for i in range(len(LABELS))
    ]
    return {
        "count": int(len(targets)),
        "mean_acc": mean_acc,
        "macro_f1": float(np.mean(f1s)),
        "exact_match": exact_match,
        "per_label_acc": {LABELS[i]: float(per_label_acc[i]) for i in range(len(LABELS))},
    }


def run_deep_evaluation():
    data_aug = ROOT_DIR / "dataset" / "augmented"
    data_raw = ROOT_DIR / "dataset" / "raw"
    data_dir = str(data_aug if (data_aug / "annotations.json").exists() else data_raw)

    model, thresholds = load_model_and_thresholds()
    val_ds = CarDataset(data_dir, split="val", preload_cache=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_probs = []
    all_targets = []
    elevations = []
    angles = []
    roi_images = []

    with torch.no_grad():
        for item in val_ds.data:
            roi_img = val_ds._load_roi_image(item["file"])
            roi_images.append(roi_img)
            tensor = preprocess(roi_img).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            all_probs.append(prob)
            all_targets.append([item["labels"][l] for l in LABELS])
            elevations.append(item.get("elevation", "mid"))
            angles.append(int(item.get("angle", 0)))

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # 1. Per-Elevation Breakdown
    print("=" * 72)
    print("1. BREAKDOWN PER CAMERA ELEVATION (N = 192)")
    print("=" * 72)
    elev_results = {}
    for elev in ["mid", "high", "top"]:
        mask = np.array([e == elev for e in elevations])
        if mask.sum() == 0:
            continue
        res = compute_subset_metrics(all_probs[mask], all_targets[mask], thresholds)
        elev_results[elev] = res
        pl = res["per_label_acc"]
        print(
            f"  Elev={elev:<4} (n={res['count']:3d}) | "
            f"FL:{pl['front_left_door']*100:5.1f}% | FR:{pl['front_right_door']*100:5.1f}% | "
            f"RL:{pl['rear_left_door']*100:5.1f}% | RR:{pl['rear_right_door']*100:5.1f}% | "
            f"Hood:{pl['hood']*100:5.1f}% || Mean:{res['mean_acc']*100:6.2f}% | "
            f"F1:{res['macro_f1']:.4f} | Exact:{res['exact_match']*100:6.2f}%"
        )

    overall_res = compute_subset_metrics(all_probs, all_targets, thresholds)

    # 2. Per-Azimuth Angle Breakdown
    print("\n" + "=" * 72)
    print("2. BREAKDOWN PER AZIMUTH ANGLE (0° to 315°)")
    print("=" * 72)
    angle_results = {}
    for ang in sorted(set(angles)):
        mask = np.array([a == ang for a in angles])
        res = compute_subset_metrics(all_probs[mask], all_targets[mask], thresholds)
        angle_results[str(ang)] = res
        pl = res["per_label_acc"]
        print(
            f"  Angle={ang:3d}° (n={res['count']:2d}) | "
            f"FL:{pl['front_left_door']*100:5.1f}% | FR:{pl['front_right_door']*100:5.1f}% | "
            f"RL:{pl['rear_left_door']*100:5.1f}% | RR:{pl['rear_right_door']*100:5.1f}% | "
            f"Hood:{pl['hood']*100:5.1f}% || Mean:{res['mean_acc']*100:6.2f}% | "
            f"Exact:{res['exact_match']*100:6.2f}%"
        )

    # 3. Photometric & Sensor Noise Robustness Stress Test
    print("\n" + "=" * 72)
    print("3. PHOTOMETRIC & SENSOR NOISE ROBUSTNESS STRESS TEST (N = 192)")
    print("=" * 72)
    rng = np.random.default_rng(42)

    stress_conditions = [
        ("Clean Baseline", lambda im, t: t),
        ("Gaussian Noise (σ=0.03)", lambda im, t: t + torch.from_numpy(rng.normal(0, 0.03, size=t.shape)).float()),
        ("Gaussian Noise (σ=0.05)", lambda im, t: t + torch.from_numpy(rng.normal(0, 0.05, size=t.shape)).float()),
        ("Gaussian Noise (σ=0.08)", lambda im, t: t + torch.from_numpy(rng.normal(0, 0.08, size=t.shape)).float()),
        ("Low-Light Dimming (0.75x)", lambda im, t: preprocess(ImageEnhance.Brightness(im).enhance(0.75))),
        ("Overexposure Glare (1.25x)", lambda im, t: preprocess(ImageEnhance.Brightness(im).enhance(1.25))),
        ("Low Contrast Haze (0.75x)", lambda im, t: preprocess(ImageEnhance.Contrast(im).enhance(0.75))),
    ]

    robustness_results = []
    with torch.no_grad():
        for cond_name, perturb_fn in stress_conditions:
            c_probs = []
            for roi_img in roi_images:
                base_t = preprocess(roi_img)
                pert_t = perturb_fn(roi_img, base_t).unsqueeze(0).to(DEVICE)
                logits = model(pert_t)
                c_probs.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
            c_probs = np.vstack(c_probs)
            res = compute_subset_metrics(c_probs, all_targets, thresholds)
            res["condition"] = cond_name
            res["delta_mean_acc"] = res["mean_acc"] - overall_res["mean_acc"]
            robustness_results.append(res)
            print(
                f"  {cond_name:<28} | Mean Acc: {res['mean_acc']*100:6.2f}% "
                f"({res['delta_mean_acc']*100:+6.2f}%) | Macro F1: {res['macro_f1']:.4f} | "
                f"Exact Match: {res['exact_match']*100:6.2f}%"
            )

    # Save JSON and LaTeX tables
    out_dir = ROOT_DIR / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "overall": overall_res,
        "by_elevation": elev_results,
        "by_angle": angle_results,
        "robustness": robustness_results,
    }
    with open(out_dir / "paper1_deep_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved deep evaluation summary to: {out_dir / 'paper1_deep_evaluation.json'}")


if __name__ == "__main__":
    run_deep_evaluation()
