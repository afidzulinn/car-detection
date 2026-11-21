"""
CarDataset (P0-01, P0-02, P0-03, P0-05, P1-02):
  1. Canonical elevation normalization ('mid', 'high', 'top') with zero silent drops.
  2. Zero silent label mutation in __getitem__ or loader (P0-02).
  3. Explicit 3-way split support: split in {'train', 'val', 'test'} loaded from
     verified disjoint split manifests (T0, T1, T2). Unknown split names raise ValueError (P0-03).
  4. Configurable ROI policy ('legacy_fixed', 'full_vehicle', 'full_frame', 'adaptive_fg') (P0-06, P1-04).
  5. Optional pose supervision targets: return_angle (8-class azimuth) and return_elevation (3-class elevation).
"""

import json
import os
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
DATA_COLLECTION_DIR = ROOT_DIR / "data_collection"
for p in (MODEL_DIR, DATA_COLLECTION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from architecture import LABELS, crop_car_roi
from audit_dataset import normalize_elevation, source_group_key
from splits import build_all_split_manifests

ELEVATION_TO_INDEX = {"mid": 0, "high": 1, "top": 2}
VALID_SPLITS = ("train", "val", "test")
PROTOCOL_FILES = {
    "T0": "t0_within_grid.json",
    "T1": "t1_global_heldout_azimuth.json",
    "T2": "t2_leave_top_elevation_out.json",
}


class CarDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        seed: int = 42,
        val_ratio: float = 0.25,
        clean_only: bool = True,
        return_angle: bool = False,
        return_elevation: bool = False,
        max_aug_per_sample: Optional[int] = None,
        preload_cache: bool = True,
        protocol: str = "T0",
        roi_policy: str = "legacy_fixed",
        two_way_compat: bool = False,
    ):
        """
        Args:
            split: Must be one of {'train', 'val', 'test'}. Raises ValueError otherwise (P0-03).
            protocol: 'T0' (within-grid 3-way 576/96/96), 'T1' (global held-out azimuth),
                      or 'T2' (leave-top-elevation-out).
            roi_policy: 'legacy_fixed' | 'full_vehicle' | 'full_frame' | 'adaptive_fg'.
            two_way_compat: If True, merges 'val' + 'test' (N=192) when split=='val' for legacy
                            comparison against historical 576/192 runs. Default is False (strict 3-way).
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {VALID_SPLITS} (P0-03)."
            )

        self.root = Path(root_dir).resolve()
        self.split = split
        self.protocol = protocol.upper()
        self.roi_policy = roi_policy
        self.return_angle = return_angle
        self.return_elevation = return_elevation
        self.two_way_compat = two_way_compat
        self._img_cache = {}

        ann_path = self.root / "annotations.json"
        with open(ann_path, encoding="utf-8") as f:
            all_data = json.load(f)

        self.data = self._load_from_manifest_or_split(
            all_data=all_data,
            split=split,
            seed=seed,
            clean_only=clean_only,
            max_aug_per_sample=max_aug_per_sample,
        )
        self.transform = transform or self._default_transform(split)

        if preload_cache:
            for item in self.data:
                self._load_roi_image(item["file"])

    def _ensure_protocol_manifest(self, seed: int) -> dict:
        if self.protocol not in PROTOCOL_FILES:
            raise ValueError(f"Unknown protocol '{self.protocol}'. Expected one of {list(PROTOCOL_FILES)}.")
        manifest_path = ROOT_DIR / "results" / "manifests" / "splits" / PROTOCOL_FILES[self.protocol]
        if not manifest_path.exists():
            build_all_split_manifests(seed=seed)
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_from_manifest_or_split(
        self,
        all_data: list,
        split: str,
        seed: int,
        clean_only: bool = True,
        max_aug_per_sample: Optional[int] = None,
    ) -> list:
        manifest = self._ensure_protocol_manifest(seed=seed)
        splits_dict = manifest["splits"]
        train_ids = set(splits_dict["train_source_ids"])
        val_ids = set(splits_dict["val_source_ids"])
        test_ids = set(splits_dict["test_source_ids"])

        if self.two_way_compat and self.protocol == "T0":
            val_ids = val_ids | test_ids

        # Canonicalize elevation & group records by source_id without mutating labels (P0-01, P0-02)
        originals_by_sid = {}
        aug_by_sid = defaultdict(list)

        for d in all_data:
            canon_elev, _ = normalize_elevation(d.get("elevation"))
            if canon_elev is None:
                if clean_only:
                    continue
                canon_elev = "mid"

            rec = dict(d)
            rec["elevation"] = canon_elev
            rec["labels"] = {l: int(d["labels"][l]) for l in LABELS}
            sid = source_group_key(rec, canon_elev)
            rec["source_id"] = sid

            is_aug = "augmentation" in d and d["augmentation"] not in (None, "none", "")
            if is_aug:
                aug_by_sid[sid].append(rec)
            else:
                originals_by_sid[sid] = rec

        if split == "val":
            return [originals_by_sid[sid] for sid in sorted(val_ids) if sid in originals_by_sid]

        if split == "test":
            return [originals_by_sid[sid] for sid in sorted(test_ids) if sid in originals_by_sid]

        # split == "train": include train originals + augmented derivatives of TRAIN originals ONLY
        train_orig = [originals_by_sid[sid] for sid in sorted(train_ids) if sid in originals_by_sid]
        train_aug = []
        for sid in sorted(train_ids):
            aug_list = aug_by_sid.get(sid, [])
            if max_aug_per_sample is not None:
                train_aug.extend(aug_list[:max_aug_per_sample])
            else:
                train_aug.extend(aug_list)

        return train_orig + train_aug

    @staticmethod
    def _default_transform(split: str):
        if split == "train":
            return transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.20, contrast=0.20, saturation=0.20, hue=0.04
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.data)

    def _load_roi_image(self, rel_file: str) -> Image.Image:
        norm_rel = rel_file.replace("\\", "/")
        cache_key = f"{self.roi_policy}:{norm_rel}"
        cached = self._img_cache.get(cache_key)
        if cached is not None:
            return cached
        full_path = self.root / Path(norm_rel)
        if not full_path.exists():
            # Fallback to raw directory if loading raw file from augmented root
            alt_path = ROOT_DIR / "dataset" / "raw" / Path(norm_rel).name
            if alt_path.exists():
                full_path = alt_path
        with Image.open(full_path) as raw_img:
            img = raw_img.convert("RGB")
            img = crop_car_roi(img, policy=self.roi_policy).resize((224, 224), Image.BILINEAR)
        self._img_cache[cache_key] = img
        return img

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self._load_roi_image(item["file"])
        img = self.transform(img)
        label = torch.tensor(
            [item["labels"][l] for l in LABELS], dtype=torch.float32
        )
        angle_deg = int(item.get("angle", 0)) % 360
        angle_idx = torch.tensor(angle_deg // 45, dtype=torch.long)
        elev_idx = torch.tensor(ELEVATION_TO_INDEX.get(item.get("elevation", "mid"), 0), dtype=torch.long)

        if self.return_angle and self.return_elevation:
            return img, label, angle_idx, elev_idx
        if self.return_angle:
            return img, label, angle_idx
        if self.return_elevation:
            return img, label, elev_idx
        return img, label


def resolve_num_workers(requested_workers: Optional[int] = None) -> int:
    if requested_workers is not None:
        return max(0, int(requested_workers))
    if os.name == "nt" or platform.system().lower().startswith("win"):
        return 0
    cpu_cnt = os.cpu_count() or 2
    return min(4, max(1, cpu_cnt // 2))


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    return_angle_train: bool = False,
    return_elevation_train: bool = False,
    max_aug_per_sample: Optional[int] = None,
    protocol: str = "T0",
    roi_policy: str = "legacy_fixed",
    include_test: bool = False,
    two_way_compat: bool = False,
):
    workers = resolve_num_workers(num_workers)
    use_pin_memory = torch.cuda.is_available()
    use_persistent = workers > 0

    train_ds = CarDataset(
        root_dir,
        split="train",
        return_angle=return_angle_train,
        return_elevation=return_elevation_train,
        max_aug_per_sample=max_aug_per_sample,
        preload_cache=True,
        protocol=protocol,
        roi_policy=roi_policy,
        two_way_compat=two_way_compat,
    )
    val_ds = CarDataset(
        root_dir,
        split="val",
        return_angle=False,
        preload_cache=True,
        protocol=protocol,
        roi_policy=roi_policy,
        two_way_compat=two_way_compat,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )

    if include_test:
        test_ds = CarDataset(
            root_dir,
            split="test",
            return_angle=False,
            preload_cache=True,
            protocol=protocol,
            roi_policy=roi_policy,
            two_way_compat=False,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        )
        print(
            f"Dataset [{protocol} | ROI={roi_policy}] -> "
            f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)} "
            f"| workers: {workers} ({platform.system()})"
        )
        return train_dl, val_dl, test_dl

    print(
        f"Dataset [{protocol} | ROI={roi_policy}] -> "
        f"train: {len(train_ds)} | val: {len(val_ds)} "
        f"| workers: {workers} ({platform.system()})"
    )
    return train_dl, val_dl
