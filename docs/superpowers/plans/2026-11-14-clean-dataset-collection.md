# Clean Dataset Collection & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `data_collection/collect_data.py` with dynamic canvas-proportional safe dragging, `--pilot` / `--fresh` CLI modes, and signature-based resumability, paired with `data_collection/verify_dataset.py` to mathematically audit the 48-image pilot before capturing all 768 raw samples.

**Architecture:** `BrowserController.drag_canvas_safely` reads `canvas.rect` to start drags at $(+0.28\,W, +0.22\,H)$ relative to the canvas center (strictly below the car at $y = 0.72\,H > 0.62\,H$ and safe for $\Delta x = -115\text{ px}$ and $\Delta y = +85\text{ px}$). `DatasetRepository` keys records by `(combo_tuple, elevation, angle)` so `--pilot` captures 48 samples (combos `(0,0,0,0,0)` and `(1,1,1,1,1)`) and subsequent full runs seamlessly capture the remaining 720 samples. `verify_dataset.py` audits resolution uniformity, top UI bar absence, 8-angle pairwise MAE uniqueness, and closed-vs-open ROI MAE across all 3 elevations (`mid`, `high`, `top`).

**Tech Stack:** Python 3.10+, Selenium WebDriver, Pillow, NumPy.

**Spec:** `docs/superpowers/specs/2026-11-14-clean-dataset-collection-design.md`

## Global Constraints

- Safe drag start offset from canvas center: `ox = int(0.28 * canvas_w)`, `oy = int(0.22 * canvas_h)`.
- Horizontal step drag: `DRAG_PIXELS_PER_STEP = -115` across `CAMERA_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)`.
- Vertical elevations: `ELEVATIONS = ((0, "mid"), (45, "high"), (85, "top"))`.
- Pilot combinations: `((0, 0, 0, 0, 0), (1, 1, 1, 1, 1))` producing $2 \times 3 \times 8 = 48$ images.
- Full dataset: 32 combinations $\times$ 3 elevations $\times$ 8 angles $= 768$ images.

## Review Focus

- `MoveTargetOutOfBoundsException` on smaller/scaled viewports when dragging left by `-115 px` or down by `+85 px` — prevented by dynamic proportional offset `(int(0.28 * w), int(0.22 * h))` tested in Task 1.
- Filename or `sample_id` collision when resuming from `--pilot` (48 samples) to full 768 capture — prevented by deterministic canonical index `sample_id = combo_idx * 24 + elev_idx * 8 + angle_idx` tested in Task 1.
- Accidental deletion of existing `dataset/raw` without backup when `--fresh` is used — prevented by archiving to `dataset/raw_legacy_backup` if not already backed up, tested in Task 1.
- Silent raycast toggle where open and closed states produce identical images ($\text{MAE} < 1.0$) — caught by `verify_dataset.py` inter-state ROI MAE assertion in Task 2.
- Viewpoint wrap-around where `angle 270` overlaps `angle 0` — caught by `verify_dataset.py` inter-angle pairwise MAE assertion in Task 2.

---

### Task 1: Dynamic Safe Drag, Canonical Indexing, & `--pilot` / `--fresh` CLI in `collect_data.py`

**Files:**
- Modify: `data_collection/collect_data.py`
- Create: `tests/test_collect_data.py`

**Interfaces:**
- Produces:
  - `compute_safe_drag_offset(canvas_w: int, canvas_h: int) -> Tuple[int, int]`
  - `canonical_sample_id(combo: Tuple[int, ...], elev_name: str, angle: int) -> int`
  - `DatasetRepository.has_signature(combo: Tuple[int, ...], elev_name: str, angle: int) -> bool`
  - `parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace` (`--pilot`, `--fresh`, `--headless`)

- [ ] **Step 1: Write the failing unit tests in `tests/test_collect_data.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_collection"))

from collect_data import (
    compute_safe_drag_offset,
    canonical_sample_id,
    parse_args,
)


def test_compute_safe_drag_offset_stays_in_bounds_and_outside_car():
    for w, h in [(1280, 720), (1070, 805), (1024, 600)]:
        ox, oy = compute_safe_drag_offset(w, h)
        x_start, y_start = w / 2 + ox, h / 2 + oy
        # Must be strictly below car ROI (y > 0.62 * h)
        assert y_start > 0.62 * h
        # Dragging left by -115 and down by +85 must stay inside [0, w] x [0, h]
        assert 0 < x_start - 115 < w
        assert 0 < y_start + 85 < h


def test_canonical_sample_id_bijective_0_to_767():
    ids = set()
    import itertools
    for combo in itertools.product([0, 1], repeat=5):
        for elev in ("mid", "high", "top"):
            for angle in (0, 45, 90, 135, 180, 225, 270, 315):
                sid = canonical_sample_id(combo, elev, angle)
                ids.add(sid)
    assert len(ids) == 768
    assert min(ids) == 0 and max(ids) == 767
    assert canonical_sample_id((0, 0, 0, 0, 0), "mid", 0) == 0
    assert canonical_sample_id((1, 1, 1, 1, 1), "top", 315) == 767


def test_parse_args_flags():
    args = parse_args(["--pilot", "--fresh", "--headless"])
    assert args.pilot is True
    assert args.fresh is True
    assert args.headless is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with selenium --with pillow python -m unittest discover -s tests -p "test_collect_data.py" -v`
Expected: FAIL with `ImportError: cannot import name 'compute_safe_drag_offset'`

- [ ] **Step 3: Implement `compute_safe_drag_offset`, `canonical_sample_id`, `has_signature`, and CLI flags in `data_collection/collect_data.py`**

Implement:
- `compute_safe_drag_offset(canvas_w: int, canvas_h: int) -> Tuple[int, int]` returning `(int(0.28 * canvas_w), int(0.22 * canvas_h))`.
- `canonical_sample_id(combo: Tuple[int, ...], elev_name: str, angle: int) -> int` computing `combo_idx * 24 + elev_idx * 8 + angle_idx` where `combo_idx = sum(b << (4 - i) for i, b in enumerate(combo))`.
- Update `DatasetRepository` to track `_existing_sigs: set[Tuple[Tuple[int, ...], str, int]]` and sort `self.records` by `file` in `persist_annotations()`.
- Add `archive_legacy_raw(save_dir: Path)` when `--fresh` is passed (renames existing `dataset/raw` to `dataset/raw_legacy_backup` if `dataset/raw_legacy_backup` does not exist).
- Filter `combos_to_run = [all_combos[0], all_combos[-1]]` when `args.pilot` is True.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with selenium --with pillow python -m unittest discover -s tests -p "test_collect_data.py" -v`
Expected: PASS (3 tests)

---

### Task 2: Forensic Dataset Auditor (`data_collection/verify_dataset.py`) & Pilot Verification

**Files:**
- Create: `data_collection/verify_dataset.py`
- Create: `tests/test_verify_dataset.py`

**Interfaces:**
- Consumes: `crop_car_roi` from `model/architecture.py`, `dataset/raw/annotations.json`
- Produces: `audit_dataset(raw_dir: Path) -> Dict[str, any]` returning:
  - `total_samples: int`
  - `resolutions: List[Tuple[int, int]]`
  - `min_inter_angle_mae: float` (minimum ROI MAE between any two angles of the same combo & elevation; must be `> 2.5`)
  - `min_open_closed_mae_by_elev: Dict[str, float]` (minimum ROI MAE between `(0,0,0,0,0)` and `(1,1,1,1,1)` at matching angle & elevation; must be `> 6.0` for all present elevations)
  - `passed: bool`

- [ ] **Step 1: Write the failing unit test in `tests/test_verify_dataset.py`**

```python
import json
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_collection"))
from verify_dataset import audit_dataset


def test_audit_dataset_detects_identical_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        img = Image.new("RGB", (1280, 720), (220, 220, 220))
        img.save(p / "s0.jpg")
        img.save(p / "s1.jpg")
        ann = [
            {"file": "s0.jpg", "labels": {k: 0 for k in ("front_left_door","front_right_door","rear_left_door","rear_right_door","hood")}, "angle": 0, "elevation": "mid"},
            {"file": "s1.jpg", "labels": {k: 1 for k in ("front_left_door","front_right_door","rear_left_door","rear_right_door","hood")}, "angle": 0, "elevation": "mid"},
        ]
        (p / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
        report = audit_dataset(p)
        assert report["passed"] is False
        assert report["min_open_closed_mae_by_elev"]["mid"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pillow --with numpy python -m unittest discover -s tests -p "test_verify_dataset.py" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verify_dataset'`

- [ ] **Step 3: Implement `audit_dataset(raw_dir: Path) -> Dict[str, any]` and CLI summary report in `data_collection/verify_dataset.py`**

Compute ROI arrays (`crop_car_roi(img).resize((224, 224))`) for all records in `annotations.json`, calculate pairwise inter-angle MAE per `(combo, elevation)` and open-vs-closed MAE per `(elevation, angle)`, and set `passed = (len(resolutions) == 1) and (min_inter_angle_mae > 2.5) and all(v > 6.0 for v in min_open_closed_mae_by_elev.values())`.

- [ ] **Step 4: Run unit test to verify it passes**

Run: `uv run --with pillow --with numpy python -m unittest discover -s tests -p "test_verify_dataset.py" -v`
Expected: PASS
