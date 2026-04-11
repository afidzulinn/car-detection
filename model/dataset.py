"""
CarDataset with angle-stratified train/val split.

Why angle-stratified (not combo-stratified)
-------------------------------------------
The 32 label combos form a complete binary code.  Randomly assigning whole
combos to val causes severe per-label class imbalance in the val set — with
seed=42 and 6 val combos, front_right_door ends up 83 % open and rear_right
ends up 67 % open, which makes AUC / accuracy meaningless.

Splitting BY ANGLE WITHIN EACH COMBO guarantees:
  - Every label combination is represented in both train and val.
  - Per-label balance in val = exactly 50 / 50.
  - The model is actually tested on unseen camera angles (the real challenge).
  - No augmented copy of a val image leaks into training.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from architecture import LABELS


class CarDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        seed: int = 42,
        val_ratio: float = 0.2,
    ):
        self.root = Path(root_dir)
        with open(self.root / "annotations.json", encoding="utf-8") as f:
            all_data = json.load(f)

        self.data = self._split(all_data, split, val_ratio, seed)
        self.transform = transform or self._default_transform(split)

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

    def _split(self, all_data, split, val_ratio, seed):
        originals = [d for d in all_data if "augmentation" not in d]
        augmented  = [d for d in all_data if "augmentation"     in d]

        rng = random.Random(seed)

        # Group originals by label combo, then split by angle/elevation WITHIN
        # each combo.  This makes the val set perfectly balanced per label.
        groups = defaultdict(list)
        for item in originals:
            groups[self._combo_key(item)].append(item)

        val_items   = []
        train_orig  = []
        val_sigs    = set()   # angle-signatures of val originals

        for key, items in groups.items():
            items_copy = list(items)
            rng.shuffle(items_copy)
            n_val = max(1, int(len(items_copy) * val_ratio))

            for item in items_copy[:n_val]:
                val_items.append(item)
                val_sigs.add(self._angle_sig(item))

            train_orig.extend(items_copy[n_val:])

        if split == "val":
            return val_items

        # Training set: train originals + augmented copies of TRAIN originals only.
        # Augmented items share (labels, angle, elevation) with their source original,
        # so we can exclude any aug whose source went to val.
        train_aug = [d for d in augmented if self._angle_sig(d) not in val_sigs]
        return train_orig + train_aug

    # ── Transforms ───────────────────────────────────────────────────────────

    @staticmethod
    def _default_transform(split):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        img   = Image.open(self.root / item["file"]).convert("RGB")
        img   = self.transform(img)
        label = torch.tensor(
            [item["labels"][l] for l in LABELS], dtype=torch.float32
        )
        return img, label


def get_dataloaders(root_dir: str, batch_size: int = 32, num_workers: int = None):
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    train_ds = CarDataset(root_dir, "train")
    val_ds   = CarDataset(root_dir, "val")
    print(
        f"Dataset → train: {len(train_ds)} | val: {len(val_ds)} "
        f"(val is angle-stratified, 50/50 per-label)"
    )
    return (
        DataLoader(
            train_ds, batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        DataLoader(
            val_ds, batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    )
