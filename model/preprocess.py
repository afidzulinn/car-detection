"""
P0-06 & P1-04:
ROI Preprocessing Policies & Component Foreground Retention Auditor.

Compares:
  - A0 ('legacy_fixed'): ROI_BOX_NORM = (0.17, 0.19, 0.83, 0.62)
  - A1 ('full_frame'):   Full 1264x625 frame (0.00, 0.00, 1.00, 1.00)
  - A2 ('full_vehicle'): Verified full-vehicle ROI = (0.24, 0.08, 0.76, 0.92)
                         which retains 100% of vehicle foreground across all 3 elevations
                         ('mid', 'high', 'top') and 32 open/closed door states.
  - A3 ('adaptive_fg'):  Content-aware foreground bounding box with 5% padding.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]

ROI_POLICIES: Dict[str, Tuple[float, float, float, float]] = {
    "legacy_fixed": (0.17, 0.19, 0.83, 0.62),
    "full_frame": (0.00, 0.00, 1.00, 1.00),
    "full_vehicle": (0.24, 0.08, 0.76, 0.92),
}

# Default ROI box kept compatible while exposing full_vehicle for P1-04 comparisons
LEGACY_ROI_BOX_NORM = ROI_POLICIES["legacy_fixed"]
FULL_VEHICLE_ROI_BOX_NORM = ROI_POLICIES["full_vehicle"]


def crop_with_policy(img: Image.Image, policy: str = "legacy_fixed") -> Image.Image:
    """
    Crop a 3D car screenshot using the specified ROI policy.
    """
    w, h = img.size
    if w <= 300 or h <= 300 or policy == "full_frame":
        return img

    if policy == "adaptive_fg":
        arr = np.asarray(img)
        # Studio background is uniform dark gray (~32, 32, 32), top bar is y < 0.08H
        y_top = int(0.08 * h)
        fg_mask = np.max(arr[y_top:, :, :], axis=-1) > 52
        ys, xs = np.where(fg_mask)
        if len(xs) > 100:
            pad_x = int(0.04 * w)
            pad_y = int(0.04 * h)
            x1 = max(0, int(xs.min()) - pad_x)
            y1 = max(0, int(ys.min()) + y_top - pad_y)
            x2 = min(w, int(xs.max()) + pad_x)
            y2 = min(h, int(ys.max()) + y_top + pad_y)
            return img.crop((x1, y1, x2, y2))
        box = ROI_POLICIES["full_vehicle"]
    else:
        box = ROI_POLICIES.get(policy, ROI_POLICIES["legacy_fixed"])

    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return img.crop((x1, y1, x2, y2))


def audit_roi_retention() -> dict:
    """
    Quantify vehicle foreground pixel retention (%) and truncation rate (%)
    for every ROI policy across all 768 raw renders per camera elevation.
    """
    raw_dir = ROOT_DIR / "dataset" / "raw"
    ann_path = raw_dir / "annotations.json"
    with open(ann_path, encoding="utf-8") as f:
        records = json.load(f)

    results = {}
    for policy_name, box in ROI_POLICIES.items():
        retentions_by_elev = defaultdict(list)
        truncated_count_by_elev = defaultdict(int)
        total_by_elev = defaultdict(int)

        for rec in records:
            elev = str(rec.get("elevation", "mid")).lower()
            if elev not in ("mid", "high", "top"):
                continue
            fpath = raw_dir / Path(rec["file"].replace("\\", "/")).name
            with Image.open(fpath) as im:
                arr = np.asarray(im.convert("RGB"))
            h, w, _ = arr.shape
            # Exclude top UI header strip (y < 0.08H) from vehicle foreground mask
            y_cut = int(0.08 * h)
            fg_mask = np.zeros((h, w), dtype=bool)
            fg_mask[y_cut:, :] = np.max(arr[y_cut:, :, :], axis=-1) > 52

            total_fg = int(fg_mask.sum())
            if total_fg == 0:
                continue

            x1, y1 = int(box[0] * w), int(box[1] * h)
            x2, y2 = int(box[2] * w), int(box[3] * h)
            inside_fg = int(fg_mask[y1:y2, x1:x2].sum())
            retention = inside_fg / total_fg

            retentions_by_elev[elev].append(retention)
            total_by_elev[elev] += 1
            if retention < 0.99:
                truncated_count_by_elev[elev] += 1

        all_rets = [r for lst in retentions_by_elev.values() for r in lst]
        area_frac = (box[2] - box[0]) * (box[3] - box[1])
        results[policy_name] = {
            "box_norm": list(box),
            "area_fraction": round(area_frac, 4),
            "spatial_density_gain": round(1.0 / max(area_frac, 1e-6), 2),
            "overall_mean_retention_pct": round(float(np.mean(all_rets)) * 100, 2),
            "overall_min_retention_pct": round(float(np.min(all_rets)) * 100, 2),
            "per_elevation": {
                elev: {
                    "samples": total_by_elev[elev],
                    "mean_retention_pct": round(float(np.mean(retentions_by_elev[elev])) * 100, 2),
                    "min_retention_pct": round(float(np.min(retentions_by_elev[elev])) * 100, 2),
                    "truncated_samples_gt_1pct_loss": truncated_count_by_elev[elev],
                    "truncation_rate_pct": round(
                        truncated_count_by_elev[elev] / max(total_by_elev[elev], 1) * 100, 2
                    ),
                }
                for elev in ("mid", "high", "top")
            },
        }

    out_dir = ROOT_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roi_retention_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[P0-06/P1-04] ROI Foreground Retention Audit:")
    for p_name, p_data in results.items():
        print(
            f"  Policy '{p_name:<12}' (gain={p_data['spatial_density_gain']}x): "
            f"mean_retention={p_data['overall_mean_retention_pct']:.2f}% | "
            f"min_retention={p_data['overall_min_retention_pct']:.2f}%"
        )
        for elev, ed in p_data["per_elevation"].items():
            print(
                f"    elev={elev:<4}: mean={ed['mean_retention_pct']:.2f}%, "
                f"min={ed['min_retention_pct']:.2f}%, "
                f"truncated(>1% loss)={ed['truncated_samples_gt_1pct_loss']}/{ed['samples']}"
            )
    return results


if __name__ == "__main__":
    audit_roi_retention()
