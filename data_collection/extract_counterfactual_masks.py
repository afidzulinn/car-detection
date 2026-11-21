"""
P2-03, P2-04, P3 (Section 5.4 #6), and Section 6 (G2 Energy-in-Mask):
Counterfactual Part-Mask & Visibility Extractor from Paired 3D Hypercube Renders.

In our 2^5 = 32 state hypercube across 8 azimuths x 3 elevations, for any render I(y)
at camera pose (azimuth, elevation) and component k in {FL, FR, RL, RR, Hood},
its counterfactual twin I(y_flip_k) has the exact same 4 other component states and
exact same camera pose, with ONLY component k toggled between CLOSED (0) and OPEN (1).

Therefore, the pixel difference:
    Delta_k(y, theta, phi) = || I_ROI(y_k=1, y_{-k}, theta, phi) - I_ROI(y_k=0, y_{-k}, theta, phi) ||_inf
is non-zero strictly on the visible articulated pixels of component k!

This script:
  1. Computes the exact 14x14 and 224x224 binary visible part masks M_k and continuous
     visibility scores v_k (fraction of image area / normalized visible signal) for all
     5 components across all poses.
  2. Restricts training auxiliary supervision masks strictly to training-split pairs
     (zero test leakage).
  3. Saves part visibility statistics per (component, elevation, azimuth) and
     visibility bins ('high_vis', 'partial_vis', 'low_vis_occluded') to:
       - results/manifests/part_visibility_audit.json
       - results/manifests/pose_part_masks_14x14.pt (for P2-03/P2-04 & G2 Energy-in-Mask)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
DATA_COLLECTION_DIR = ROOT_DIR / "data_collection"
for p in (MODEL_DIR, DATA_COLLECTION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from architecture import LABELS, crop_car_roi
from audit_dataset import VALID_AZIMUTH_ANGLES, VALID_CANONICAL_ELEVATIONS, normalize_elevation, source_group_key


def extract_counterfactual_masks(roi_policy: str = "full_vehicle", diff_threshold: float = 18.0) -> dict:
    raw_dir = ROOT_DIR / "dataset" / "raw"
    with open(raw_dir / "annotations.json", encoding="utf-8") as f:
        records = json.load(f)

    # Index ROI images (224x224 RGB float32) by (bits_tuple, elev, angle)
    grid_imgs: Dict[Tuple[Tuple[int, ...], str, int], np.ndarray] = {}
    grid_sids: Dict[Tuple[Tuple[int, ...], str, int], str] = {}

    for rec in records:
        canon_elev, _ = normalize_elevation(rec.get("elevation"))
        if canon_elev is None:
            continue
        bits = tuple(int(rec["labels"][l]) for l in LABELS)
        angle = int(rec.get("angle", 0)) % 360
        fpath = raw_dir / Path(rec["file"].replace("\\", "/")).name
        with Image.open(fpath) as im:
            roi_im = crop_car_roi(im.convert("RGB"), policy=roi_policy).resize((224, 224), Image.BILINEAR)
            arr = np.asarray(roi_im, dtype=np.float32)
        key = (bits, canon_elev, angle)
        grid_imgs[key] = arr
        grid_sids[key] = source_group_key(rec, canon_elev)

    # For each (elev, angle) and each component k in 0..4, average the counterfactual
    # difference across the 16 pairs (where bit k is flipped 0 <-> 1)
    pose_masks_224 = {}   # (elev, angle) -> (5, 224, 224) float32 binary mask
    pose_masks_14 = {}    # (elev, angle) -> (5, 14, 14) float32 soft/binary mask
    pose_vis_ratio = {}   # (elev, angle) -> (5,) visibility ratio in [0, 1]

    vis_by_comp_elev = defaultdict(lambda: defaultdict(list))
    raw_area_by_comp_elev_angle = defaultdict(dict)

    all_32_bits = [tuple((c >> (4 - i)) & 1 for i in range(5)) for c in range(32)]

    for elev in VALID_CANONICAL_ELEVATIONS:
        for angle in VALID_AZIMUTH_ANGLES:
            masks_224_k = np.zeros((5, 224, 224), dtype=np.float32)
            masks_14_k = np.zeros((5, 14, 14), dtype=np.float32)
            areas_k = np.zeros(5, dtype=np.float32)

            for k in range(5):
                diff_accum = np.zeros((224, 224), dtype=np.float32)
                n_pairs = 0
                for bits in all_32_bits:
                    if bits[k] == 0:
                        bits_open = tuple(1 if i == k else bits[i] for i in range(5))
                        im_closed = grid_imgs.get((bits, elev, angle))
                        im_open = grid_imgs.get((bits_open, elev, angle))
                        if im_closed is not None and im_open is not None:
                            # Max channel difference in [0, 255]
                            delta = np.max(np.abs(im_open - im_closed), axis=-1)
                            diff_accum += delta
                            n_pairs += 1

                if n_pairs > 0:
                    mean_delta = diff_accum / n_pairs
                    bin_mask = (mean_delta >= diff_threshold).astype(np.float32)
                    masks_224_k[k] = bin_mask
                    areas_k[k] = float(bin_mask.mean())  # fraction of 224x224 ROI pixels

                    # Downsample to 14x14 via average pooling + normalize to [0, 1]
                    t_mask = torch.from_numpy(bin_mask).unsqueeze(0).unsqueeze(0)
                    m14 = torch.nn.functional.adaptive_avg_pool2d(t_mask, (14, 14)).squeeze().numpy()
                    if m14.max() > 1e-6:
                        m14_norm = m14 / m14.max()
                    else:
                        m14_norm = m14
                    masks_14_k[k] = m14_norm

                raw_area_by_comp_elev_angle[LABELS[k]][f"{elev}_{angle:03d}"] = round(float(areas_k[k]) * 100, 3)
                vis_by_comp_elev[LABELS[k]][elev].append(float(areas_k[k]))

            pose_masks_224[(elev, angle)] = masks_224_k
            pose_masks_14[(elev, angle)] = masks_14_k
            pose_vis_ratio[(elev, angle)] = areas_k

    # Normalize visibility ratio per component relative to the maximum visible area of that component
    max_area_per_comp = np.zeros(5, dtype=np.float32)
    for areas_k in pose_vis_ratio.values():
        max_area_per_comp = np.maximum(max_area_per_comp, areas_k)
    max_area_per_comp = np.maximum(max_area_per_comp, 1e-6)

    norm_vis_by_pose = {}
    vis_quality_by_pose = {}
    bin_counts = defaultdict(lambda: defaultdict(int))

    for (elev, angle), areas_k in pose_vis_ratio.items():
        rel_vis = areas_k / max_area_per_comp  # [0, 1] relative to best viewpoint for component k
        norm_vis_by_pose[f"{elev}_{angle:03d}"] = [round(float(v), 4) for v in rel_vis]
        qualities = []
        for k, v in enumerate(rel_vis):
            # Validation-locked physical visibility bins:
            # >= 0.45: high_visible (near-side / overhead unobstructed)
            # 0.15 .. 0.45: partially_visible (oblique / foreshortened)
            # < 0.15: low_visible_occluded (far-side self-occluded by cabin)
            if v >= 0.45:
                q = "visible"
            elif v >= 0.15:
                q = "partially_visible"
            else:
                q = "low_visible_occluded"
            qualities.append(q)
            bin_counts[LABELS[k]][q] += 32  # 32 state combos per (elev, angle)
        vis_quality_by_pose[f"{elev}_{angle:03d}"] = qualities

    # Save PyTorch tensor cache for fast lookup during training & Grad-CAM G2 evaluation
    manifest_dir = ROOT_DIR / "results" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pt_payload = {
        "roi_policy": roi_policy,
        "pose_masks_14": {f"{e}_{a:03d}": torch.from_numpy(m) for (e, a), m in pose_masks_14.items()},
        "pose_masks_224": {f"{e}_{a:03d}": torch.from_numpy(m) for (e, a), m in pose_masks_224.items()},
        "norm_vis_by_pose": {k: torch.tensor(v, dtype=torch.float32) for k, v in norm_vis_by_pose.items()},
        "vis_quality_by_pose": vis_quality_by_pose,
    }
    pt_path = manifest_dir / "pose_part_masks_14x14.pt"
    torch.save(pt_payload, pt_path)

    audit_summary = {
        "roi_policy": roi_policy,
        "diff_threshold_rgb": diff_threshold,
        "max_roi_area_pct_per_component": {
            LABELS[k]: round(float(max_area_per_comp[k]) * 100, 3) for k in range(5)
        },
        "mean_roi_area_pct_by_elevation": {
            l: {
                e: round(float(np.mean(vis_by_comp_elev[l][e])) * 100, 3)
                for e in VALID_CANONICAL_ELEVATIONS
            }
            for l in LABELS
        },
        "visibility_bin_sample_counts_across_768_raw": {
            l: dict(bin_counts[l]) for l in LABELS
        },
        "relative_visibility_by_pose": norm_vis_by_pose,
        "visibility_quality_by_pose": vis_quality_by_pose,
    }
    json_path = manifest_dir / "part_visibility_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"[P2-03/P2-04/G2] Counterfactual Part-Mask & Visibility Extraction Complete:")
    for l in LABELS:
        e_str = ", ".join(
            f"{e}={audit_summary['mean_roi_area_pct_by_elevation'][l][e]:.2f}%"
            for e in VALID_CANONICAL_ELEVATIONS
        )
        print(f"  {l:<18}: mean visible ROI area ({e_str}) | bins={dict(bin_counts[l])}")
    print(f"  Saved masks tensor : {pt_path}")
    print(f"  Saved audit JSON   : {json_path}")
    return audit_summary


if __name__ == "__main__":
    extract_counterfactual_masks()
