"""
CarDataset with:
  1. Cross-platform support for Windows (nt / spawn) and Ubuntu/Linux (posix / fork),
     including path separator normalization, Copy-on-Write RAM cache preloading,
     and persistent DataLoader workers on Ubuntu.
  2. Automatic filtering of corrupted legacy elevations (elevtop undercarriage &
     elevlow center-click raycast bugs) while supporting clean 'mid'/'high'/'top'.
  3. Ground-truth label repair for the 2 legacy Combo-0 resize artifacts
     (sample_00013_angle225_elevmid.jpg & sample_00014_angle270_elevmid.jpg).
  4. Paired Angle-Stratified Split across foreground states so that mutual information
     between camera pose/other doors and rear_right_door on the training set is 0
     (preventing 4-bit hypercube lookup leakage while testing unseen angles).
  5. Deterministic Car ROI cropping (crop_car_roi) providing a 3.52x increase
     in effective 2D spatial resolution on the car body and removing the top UI bar.
"""

import json
import os
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure local imports work regardless of current working directory on Windows or Ubuntu
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import LABELS, crop_car_roi


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
        max_aug_per_sample: int = None,
        preload_cache: bool = True,
    ):
        self.root = Path(root_dir).resolve()
        self.split = split
        self.return_angle = return_angle
        self._img_cache = {}

        ann_path = self.root / "annotations.json"
        with open(ann_path, encoding="utf-8") as f:
            all_data = json.load(f)

        self.data = self._split(
            all_data, split, val_ratio, seed,
            clean_only=clean_only,
            max_aug_per_sample=max_aug_per_sample,
        )
        self.transform = transform or self._default_transform(split)

        # Preload ROI-cropped 224x224 images into RAM before DataLoader forks workers.
        # On Ubuntu (fork), all worker processes share this cache via Copy-on-Write (0 disk I/O).
        # On Windows (num_workers=0), the main process reuses this cache across all epochs.
        if preload_cache:
            for item in self.data:
                self._load_roi_image(item["file"])

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _combo_key(item: dict) -> tuple:
        return tuple(item["labels"][l] for l in LABELS)

    @staticmethod
    def _angle_sig(item: dict) -> tuple:
        """Unique signature of the original sample an augmented item came from."""
        return (
            tuple(item["labels"][l] for l in LABELS),
            item.get("angle"),
            item.get("elevation", "mid"),
        )

    @staticmethod
    def _is_legacy_combo0_fl_open(item: dict) -> bool:
        """
        In the legacy capture, sample_00013 (angle 225) and sample_00014 (angle 270)
        in Combo 0 (all closed) had front_left_door accidentally clicked open while
        the browser window was being resized.
        """
        is_all_zero = all(item["labels"][l] == 0 for l in LABELS)
        return (
            is_all_zero
            and item.get("elevation", "mid") == "mid"
            and item.get("angle") in (225, 270)
        )

    @classmethod
    def _repair_item(cls, item: dict) -> dict:
        if cls._is_legacy_combo0_fl_open(item):
            repaired = dict(item)
            repaired["labels"] = dict(item["labels"])
            repaired["labels"]["front_left_door"] = 1
            return repaired
        return item

    def _split(
        self,
        all_data,
        split,
        val_ratio,
        seed,
        clean_only: bool = True,
        max_aug_per_sample: int = None,
    ):
        if clean_only:
            valid_elevs = {"mid", "high", "Slightly High", "Top-Down"}
            filtered = [
                d for d in all_data
                if d.get("elevation", "mid") in valid_elevs
            ]
            if filtered:
                all_data = filtered

        originals = [d for d in all_data if "augmentation" not in d]
        augmented = [d for d in all_data if "augmentation" in d]

        rng = random.Random(seed)

        groups = defaultdict(list)
        for item in originals:
            groups[self._combo_key(item)].append(item)

        val_items = []
        train_orig = []
        val_sigs = set()

        # Pair combos across rear_right_door (index 3) so that for any given
        # foreground state (FL, FR, RL, Hood) and training angle, BOTH RR=0 and
        # RR=1 are present in train (eliminating spurious 4-bit complement leakage
        # while keeping 2 unseen validation angles per combo and exact 50/50 balance).
        fg_groups = defaultdict(dict)
        for key, items in groups.items():
            fg_key = (key[0], key[1], key[2], key[4])
            fg_groups[fg_key][key[3]] = items

        for fg_key in sorted(fg_groups.keys()):
            pair_dict = fg_groups[fg_key]
            ref_items = list(next(iter(pair_dict.values())))
            slots = [(it.get("elevation", "mid"), it.get("angle")) for it in ref_items]
            rng.shuffle(slots)
            if fg_key == (0, 0, 0, 0):
                slots.sort(key=lambda s: 1 if (s[0] == "mid" and s[1] in (225, 270)) else 0)
            n_val = max(1, int(len(slots) * val_ratio))
            val_slot_set = set(slots[:n_val])

            for rr_val, items in pair_dict.items():
                for item in items:
                    slot = (item.get("elevation", "mid"), item.get("angle"))
                    if slot in val_slot_set:
                        val_items.append(self._repair_item(item))
                        val_sigs.add(self._angle_sig(item))
                    else:
                        train_orig.append(self._repair_item(item))

        if split == "val":
            return val_items

        # Training set: train originals + augmented copies of TRAIN originals only.
        train_aug_by_sig = defaultdict(list)
        for d in augmented:
            sig = self._angle_sig(d)
            if sig not in val_sigs:
                train_aug_by_sig[sig].append(self._repair_item(d))

        train_aug = []
        for sig, aug_list in train_aug_by_sig.items():
            if max_aug_per_sample is not None:
                train_aug.extend(aug_list[:max_aug_per_sample])
            else:
                train_aug.extend(aug_list)

        return train_orig + train_aug

    # ── Transforms ───────────────────────────────────────────────────────────

    @staticmethod
    def _default_transform(split):
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

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def _load_roi_image(self, rel_file: str) -> Image.Image:
        # Normalize path separators across Windows ('\') and Ubuntu/Linux ('/')
        norm_rel = rel_file.replace("\\", "/")
        cached = self._img_cache.get(norm_rel)
        if cached is not None:
            return cached
        full_path = self.root / Path(norm_rel)
        with Image.open(full_path) as raw_img:
            img = raw_img.convert("RGB")
            img = crop_car_roi(img).resize((224, 224), Image.BILINEAR)
        self._img_cache[norm_rel] = img
        return img

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self._load_roi_image(item["file"])
        img = self.transform(img)
        label = torch.tensor(
            [item["labels"][l] for l in LABELS], dtype=torch.float32
        )
        if self.return_angle:
            angle_deg = int(item.get("angle", 0)) % 360
            angle_idx = torch.tensor(angle_deg // 45, dtype=torch.long)
            return img, label, angle_idx
        return img, label


def resolve_num_workers(requested_workers: int = None) -> int:
    """
    Select optimal DataLoader worker count for Windows vs Ubuntu/Linux.
    - Windows ('nt') uses 'spawn', where num_workers=0 avoids process-spawn overhead
      and shares the in-memory PIL cache directly.
    - Ubuntu/Linux ('posix') uses 'fork' with Copy-on-Write memory sharing, so 2-4
      workers parallelize online ColorJitter/ToTensor transforms efficiently.
    """
    if requested_workers is not None:
        return max(0, int(requested_workers))
    if os.name == "nt" or platform.system().lower().startswith("win"):
        return 0
    cpu_cnt = os.cpu_count() or 2
    return min(4, max(1, cpu_cnt // 2))


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = None,
    return_angle_train: bool = False,
    max_aug_per_sample: int = None,
):
    workers = resolve_num_workers(num_workers)
    use_pin_memory = torch.cuda.is_available()
    use_persistent = workers > 0

    train_ds = CarDataset(
        root_dir,
        "train",
        return_angle=return_angle_train,
        max_aug_per_sample=max_aug_per_sample,
        preload_cache=True,
    )
    val_ds = CarDataset(
        root_dir,
        "val",
        return_angle=False,
        preload_cache=True,
    )
    print(
        f"Dataset → train: {len(train_ds)} | val: {len(val_ds)} "
        f"| workers: {workers} ({platform.system()}) "
        f"(val is angle-stratified, clean ROI, 50/50 per-label)"
    )
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        ),
    )
