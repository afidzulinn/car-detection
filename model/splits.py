"""
P0-03, P0-05, P1-02:
Disjoint 3-Way Split Manifest Generator & Verifier (Train / Validation / Independent Test).

Generates three formal experimental protocols in results/manifests/splits/:
  1. T0 ('t0_within_grid.json'):
     Paired Angle-Stratified 3-Way Split across all 32 state combinations and 3 elevations.
     - Train: 6 azimuth angles per (fg_group, elevation) -> 576 raw renders (4,608 with train-only augs)
     - Val:   1 azimuth angle  per (fg_group, elevation) ->  96 raw renders (32 mid, 32 high, 32 top; 50/50 per label)
     - Test:  1 azimuth angle  per (fg_group, elevation) ->  96 raw renders (32 mid, 32 high, 32 top; 50/50 per label)
     Strictly disjoint at the source_id level: train ∩ val = train ∩ test = val ∩ test = ∅.

  2. T1 ('t1_global_heldout_azimuth.json'):
     Global Held-Out Camera Azimuth Protocol.
     - Test:  Azimuths 90° and 270° across all combinations and elevations (192 raw renders; 0 in train/val).
     - Val:   Azimuth 135° across all combinations and elevations (96 raw renders).
     - Train: Remaining 5 azimuths {0°, 45°, 180°, 225°, 315°} (480 raw renders + train-only augs).

  3. T2 ('t2_leave_top_elevation_out.json'):
     Leave-One-Elevation-Out Protocol (Unseen Top-Down Perspective Extrapolation).
     - Test:  All 'top' elevation renders (256 raw renders; 0 'top' renders in train/val).
     - Val:   1 azimuth angle per (fg_group, elevation) from {'mid', 'high'} (64 raw renders).
     - Train: Remaining 7 azimuth angles per (fg_group, elevation) from {'mid', 'high'} (448 raw renders + train-only augs).
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_COLLECTION_DIR = ROOT_DIR / "data_collection"
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from audit_dataset import (
    LABELS,
    VALID_AZIMUTH_ANGLES,
    VALID_CANONICAL_ELEVATIONS,
    combo_bits_to_id,
    normalize_elevation,
    source_group_key,
)


def _load_canonical_sources() -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    """
    Load canonical raw renders and group augmented derivatives by source_id.
    """
    raw_ann = ROOT_DIR / "dataset" / "raw" / "annotations.json"
    aug_ann = ROOT_DIR / "dataset" / "augmented" / "annotations.json"

    with open(raw_ann, encoding="utf-8") as f:
        raw_list = json.load(f)

    sources: Dict[str, dict] = {}
    for rec in raw_list:
        canon_elev, _ = normalize_elevation(rec.get("elevation"))
        if canon_elev is None:
            continue
        src_id = source_group_key(rec, canon_elev)
        bits = tuple(int(rec["labels"][l]) for l in LABELS)
        sources[src_id] = {
            "source_id": src_id,
            "file": Path(rec["file"].replace("\\", "/")).name,
            "combo_id": combo_bits_to_id(bits),
            "combo_bits": list(bits),
            "fg_key": [bits[0], bits[1], bits[2], bits[4]],
            "rr_bit": bits[3],
            "angle": int(rec.get("angle", 0)) % 360,
            "elevation": canon_elev,
            "labels": {l: int(rec["labels"][l]) for l in LABELS},
        }

    aug_by_source: Dict[str, List[dict]] = defaultdict(list)
    if aug_ann.exists():
        with open(aug_ann, encoding="utf-8") as f:
            aug_list = json.load(f)
        for rec in aug_list:
            if "augmentation" not in rec or rec["augmentation"] in (None, "none", ""):
                continue
            canon_elev, _ = normalize_elevation(rec.get("elevation"))
            if canon_elev is None:
                continue
            src_id = source_group_key(rec, canon_elev)
            aug_by_source[src_id].append({
                "source_id": src_id,
                "file": Path(rec["file"].replace("\\", "/")).name,
                "augmentation": rec["augmentation"],
            })

    return sources, aug_by_source


def _verify_split_disjointness(
    protocol_name: str,
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
    sources: Dict[str, dict],
) -> dict:
    assert len(train_ids & val_ids) == 0, f"{protocol_name}: train ∩ val != ∅"
    assert len(train_ids & test_ids) == 0, f"{protocol_name}: train ∩ test != ∅"
    assert len(val_ids & test_ids) == 0, f"{protocol_name}: val ∩ test != ∅"
    assert len(train_ids | val_ids | test_ids) == len(sources), (
        f"{protocol_name}: union of splits ({len(train_ids | val_ids | test_ids)}) != total sources ({len(sources)})"
    )

    def _stats_for(id_set: Set[str]) -> dict:
        elev_cnt = defaultdict(int)
        angle_cnt = defaultdict(int)
        label_open = defaultdict(int)
        for sid in id_set:
            s = sources[sid]
            elev_cnt[s["elevation"]] += 1
            angle_cnt[s["angle"]] += 1
            for l in LABELS:
                if s["labels"][l] == 1:
                    label_open[l] += 1
        n = max(len(id_set), 1)
        return {
            "num_raw_sources": len(id_set),
            "by_elevation": dict(sorted(elev_cnt.items())),
            "by_azimuth": {str(k): v for k, v in sorted(angle_cnt.items())},
            "label_open_ratios": {l: round(label_open[l] / n, 4) for l in LABELS},
        }

    return {
        "disjoint_verified": True,
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "train_stats": _stats_for(train_ids),
        "val_stats": _stats_for(val_ids),
        "test_stats": _stats_for(test_ids),
    }


def build_all_split_manifests(seed: int = 42) -> Dict[str, dict]:
    sources, aug_by_source = _load_canonical_sources()
    splits_dir = ROOT_DIR / "results" / "manifests" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Group sources by (fg_key_tuple, elevation) -> dict[angle -> list of 2 RR pair source_ids]
    fg_elev_angles: Dict[Tuple[Tuple[int, ...], str], Dict[int, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sid, s in sources.items():
        fg_t = tuple(s["fg_key"])
        fg_elev_angles[(fg_t, s["elevation"])][s["angle"]].append(sid)

    all_fg_keys = sorted({tuple(s["fg_key"]) for s in sources.values()})
    angles_list = list(VALID_AZIMUTH_ANGLES)  # [0, 45, 90, 135, 180, 225, 270, 315]

    # ── Protocol T0: Within-Grid Paired Angle-Stratified 3-Way Split (576 / 96 / 96) ──
    # Rotate held-out val_angle and test_angle deterministically across the 16 fg_keys
    # and 3 elevations so that:
    #   1. Both RR=0 and RR=1 share the exact same train/val/test angles (I(RR; angle) = 0).
    #   2. Every single one of the 8 azimuth angles appears evenly in Val (12 samples/angle)
    #      and in Test (12 samples/angle).
    #   3. Every single one of the 5 labels has an exact 50.0% open / 50.0% closed balance
    #      in Train (576), Val (96), and Test (96).
    t0_train, t0_val, t0_test = set(), set(), set()
    elev_offsets = {"mid": 0, "high": 3, "top": 6}

    for fg_idx, fg_t in enumerate(all_fg_keys):
        # Pair complementary foreground states so open/closed counts remain balanced per angle
        rot_idx = (fg_idx // 2) + (fg_idx % 2) * 4
        for elev in VALID_CANONICAL_ELEVATIONS:
            angle_map = fg_elev_angles[(fg_t, elev)]
            val_a = angles_list[(rot_idx + elev_offsets[elev]) % 8]
            test_a = angles_list[(rot_idx + elev_offsets[elev] + 4) % 8]
            for a, sids in angle_map.items():
                if a == val_a:
                    t0_val.update(sids)
                elif a == test_a:
                    t0_test.update(sids)
                else:
                    t0_train.update(sids)

    # ── Protocol T1: Global Held-Out Azimuth (Test = {90, 270}, Val = {135}, Train = 5 angles) ──
    t1_train, t1_val, t1_test = set(), set(), set()
    for sid, s in sources.items():
        if s["angle"] in (90, 270):
            t1_test.add(sid)
        elif s["angle"] == 135:
            t1_val.add(sid)
        else:
            t1_train.add(sid)

    # ── Protocol T2: Leave-One-Elevation-Out (Test = 'top', Val/Train = 'mid' & 'high') ──
    t2_train, t2_val, t2_test = set(), set(), set()
    for sid, s in sources.items():
        if s["elevation"] == "top":
            t2_test.add(sid)

    for fg_idx, fg_t in enumerate(all_fg_keys):
        rot_idx = (fg_idx // 2) + (fg_idx % 2) * 4
        for elev in ("mid", "high"):
            angle_map = fg_elev_angles[(fg_t, elev)]
            val_a = angles_list[(rot_idx + elev_offsets[elev]) % 8]
            for a, sids in angle_map.items():
                if a == val_a:
                    t2_val.update(sids)
                else:
                    t2_train.update(sids)

    protocols = {
        "T0": (
            "t0_within_grid.json",
            "Within-Grid Paired Angle-Stratified 3-Way Split (576 Train / 96 Val / 96 Independent Test)",
            t0_train,
            t0_val,
            t0_test,
        ),
        "T1": (
            "t1_global_heldout_azimuth.json",
            "Global Held-Out Azimuth Protocol (Test: 90° & 270°; Val: 135°; Train: 0°,45°,180°,225°,315°)",
            t1_train,
            t1_val,
            t1_test,
        ),
        "T2": (
            "t2_leave_top_elevation_out.json",
            "Leave-One-Elevation-Out Extrapolation Protocol (Test: Top 90°; Train/Val: Mid & High)",
            t2_train,
            t2_val,
            t2_test,
        ),
    }

    summaries = {}
    for code, (fname, desc, tr_set, va_set, te_set) in protocols.items():
        verification = _verify_split_disjointness(code, tr_set, va_set, te_set, sources)
        train_aug_files = [
            aug_item["file"]
            for sid in sorted(tr_set)
            for aug_item in aug_by_source.get(sid, [])
        ]
        payload = {
            "protocol_code": code,
            "description": desc,
            "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification": verification,
            "splits": {
                "train_source_ids": sorted(tr_set),
                "val_source_ids": sorted(va_set),
                "test_source_ids": sorted(te_set),
                "train_raw_files": [sources[sid]["file"] for sid in sorted(tr_set)],
                "val_raw_files": [sources[sid]["file"] for sid in sorted(va_set)],
                "test_raw_files": [sources[sid]["file"] for sid in sorted(te_set)],
                "train_augmented_files_count": len(train_aug_files),
            },
        }
        out_file = splits_dir / fname
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summaries[code] = verification
        print(
            f"[P0-03/P0-05/P1-02] Protocol {code} ({fname}): "
            f"Train={len(tr_set)} raw (+{len(train_aug_files)} aug) | "
            f"Val={len(va_set)} raw | Test={len(te_set)} raw | Disjoint=PASS"
        )

    return summaries


if __name__ == "__main__":
    build_all_split_manifests()
