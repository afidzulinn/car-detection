"""
Extended augmentation for varied lighting, quality, and viewpoint conditions.

With 768 raw samples (32 combos x 8 angles x 3 elevations) and 6 augmentation
types this produces 768 + 768*6 = 5,376 training-ready samples.

Augmentation design notes
--------------------------
* brightness_high / brightness_low  – simulate different lighting environments.
* motion_blur                        – simulate slight camera shake.
* gaussian_noise                     – simulate sensor noise / JPEG artifacts.
* hue_shift                          – simulate different monitor color profiles.
* perspective                        – simulate slight off-axis camera pose not
                                       covered by the 3D rotation (does NOT
                                       flip left/right so labels stay valid).
* jpeg_compress                      – simulate lossy transmission / screen-cap
                                       quality variation.

NOTE: horizontal flip is intentionally excluded because it swaps left↔right
doors, which would require swapping the corresponding labels too — complex and
error-prone.  Perspective distortion is used instead to add viewpoint diversity
while keeping labels unambiguous.

Requires: pip install albumentations opencv-python tqdm
"""
import cv2
import json
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("WARNING: albumentations not installed. Run: pip install albumentations")

INPUT_DIR  = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/augmented")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_augmentations():
    if not HAS_ALBUMENTATIONS:
        return {}
    return {
        # Bright scene (e.g. sun / studio light)
        "brightness_high": A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(0.2, 0.45),
                contrast_limit=(0.0, 0.2),
                p=1.0,
            )
        ]),
        # Dark scene (night / garage)
        "brightness_low": A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, -0.1),
                contrast_limit=(-0.1, 0.1),
                p=1.0,
            )
        ]),
        # Slight camera motion shake
        "motion_blur": A.Compose([
            A.MotionBlur(blur_limit=(3, 9), p=1.0)
        ]),
        # Sensor / JPEG noise
        "gaussian_noise": A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0)
        ]),
        # Different monitor / render color profile
        "hue_shift": A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            )
        ]),
        # Off-axis camera tilt not captured by the 3D orbit drag.
        # scale=(0.85,1.0) keeps most of the car visible.
        # Labels stay valid because left/right is not mirrored.
        "perspective": A.Compose([
            A.Perspective(scale=(0.04, 0.08), keep_size=True, p=1.0)
        ]),
        # Lossy compression artifacts (screen capture quality variation)
        "jpeg_compress": A.Compose([
            A.ImageCompression(quality_lower=40, quality_upper=75, p=1.0)
        ]),
    }


def augment_dataset():
    ann_path = INPUT_DIR / "annotations.json"
    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found. Run collect_data.py first.")
        return

    with open(ann_path) as f:
        annotations = json.load(f)

    augmentations = get_augmentations()
    new_annotations = []

    # ── Copy originals into OUTPUT_DIR ──────────────────────────────────────
    for item in tqdm(annotations, desc="Copying originals"):
        src = INPUT_DIR / item["file"]
        dst = OUTPUT_DIR / item["file"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        new_annotations.append(dict(item))

    # ── Generate augmented variants ─────────────────────────────────────────
    sample_id = len(new_annotations)

    for item in tqdm(annotations, desc="Augmenting"):
        img_path = INPUT_DIR / item["file"]
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for aug_name, aug in augmentations.items():
            try:
                augmented = aug(image=img_rgb)["image"]
            except Exception as exc:
                print(f"WARNING: augmentation '{aug_name}' failed on {item['file']}: {exc}")
                continue

            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            fname = f"aug_{sample_id:06d}_{aug_name}.jpg"
            cv2.imwrite(
                str(OUTPUT_DIR / fname),
                aug_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            new_annotations.append({
                "file":        fname,
                "labels":      item["labels"],
                "angle":       item.get("angle", 0),
                "elevation":   item.get("elevation", "mid"),
                "augmentation": aug_name,
            })
            sample_id += 1

    with open(OUTPUT_DIR / "annotations.json", "w") as f:
        json.dump(new_annotations, f, indent=2)

    n_orig = len(annotations)
    n_aug  = len(new_annotations) - n_orig
    print(f"\nOriginal: {n_orig}  |  Augmented: {n_aug}  |  Total: {len(new_annotations)}")


if __name__ == "__main__":
    augment_dataset()
