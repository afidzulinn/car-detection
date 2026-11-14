"""
Robust & Clean Data Collector for 3D Car Interface.

Architecture:
- Clean Architecture (Separation of Concerns):
  1. Config & Domain constants (dataclasses)
  2. BrowserController (Selenium encapsulation, orbit math, raycast-free dragging)
  3. CarModelInteractions (State toggle, DOM control, button tracking)
  4. DatasetRepository (Atomic saving, canonical naming, JSON annotations, resumable state)
  5. DatasetPipeline (Orchestrator for 32 combos x 3 elevations x 8 angles)
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DataCollector")


# =====================================================================
# 1. DOMAIN CONFIGURATION & HELPER FUNCTIONS
# =====================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CAR_URL = "https://euphonious-concha-ab5c5d.netlify.app/"

COMPONENTS: Tuple[str, ...] = (
    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",
    "hood",
)

BUTTON_LABELS: Dict[str, str] = {
    "front_left_door": "Front Left Door",
    "front_right_door": "Front Right Door",
    "rear_left_door": "Rear Left Door",
    "rear_right_door": "Rear Right Door",
    "hood": "Hood",
}

# 8 Horizontal viewpoints around the car (45 deg increments)
CAMERA_ANGLES: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)

# 3 Calibrated positive vertical offsets (positive dy tilts camera upward)
ELEVATIONS: Tuple[Tuple[int, str], ...] = (
    (0, "mid"),
    (45, "high"),
    (85, "top"),
)

ELEV_INDEX: Dict[str, int] = {"mid": 0, "high": 1, "top": 2}


def compute_safe_drag_offset(canvas_w: int, canvas_h: int) -> Tuple[int, int]:
    """
    Computes proportional offset from canvas center (W/2, H/2).
    Start coordinate: (0.78 * W, 0.72 * H).
    Safely below car bounding box (y > 0.62 * H) and allows dragging
    left (-115 px) and down (+85 px) without leaving [0, W] x [0, H].
    """
    return (int(0.28 * canvas_w), int(0.22 * canvas_h))


def canonical_sample_id(combo: Tuple[int, ...], elev_name: str, angle: int) -> int:
    """
    Bijective mapping from (combo, elevation, angle) to index in 0..767.
    Ensures deterministic, gapless naming even during pilot / resume runs.
    """
    combo_idx = sum(b << (4 - i) for i, b in enumerate(combo))
    elev_idx = ELEV_INDEX[elev_name]
    angle_idx = CAMERA_ANGLES.index(angle)
    return combo_idx * 24 + elev_idx * 8 + angle_idx


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 3D car multi-view dataset safely.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode with 2 extreme combos: all-closed and all-open (48 images).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Backup legacy dataset/raw to dataset/raw_legacy_backup and start fresh.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class CollectorConfig:
    url: str = DEFAULT_CAR_URL
    save_dir: Path = field(default_factory=lambda: ROOT_DIR / "dataset" / "raw")
    debug_dir: Path = field(default_factory=lambda: ROOT_DIR / "dataset" / "debug")
    window_width: int = 1280
    window_height: int = 720
    drag_pixels_per_step: int = -115
    page_load_timeout: int = 40
    post_click_delay: float = 0.5
    post_drag_delay: float = 0.6
    image_quality: int = 92
    headless: bool = False
    pilot: bool = False
    fresh: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> CollectorConfig:
        dotenv_path = ROOT_DIR / ".env"
        url = DEFAULT_CAR_URL
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("CAR_URL="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        url = val
        url = os.environ.get("CAR_URL", url).strip().strip('"').strip("'")
        is_headless = args.headless or (os.environ.get("HEADLESS", "").lower() in {"1", "true", "yes"})
        return cls(url=url, headless=is_headless, pilot=args.pilot, fresh=args.fresh)


# =====================================================================
# 2. BROWSER & INTERACTION CONTROLLER (Selenium Infrastructure)
# =====================================================================

class BrowserController:
    """Encapsulates WebDriver setup, canvas dragging, and screenshot capture."""

    def __init__(self, config: CollectorConfig):
        self.cfg = config
        self.driver: Optional[webdriver.Chrome] = None

    def start(self) -> None:
        opts = webdriver.ChromeOptions()
        opts.add_argument(f"--window-size={self.cfg.window_width},{self.cfg.window_height}")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        if self.cfg.headless:
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=opts)
        self.driver.set_window_size(self.cfg.window_width, self.cfg.window_height)
        logger.info(f"Chrome WebDriver initialized ({self.cfg.window_width}x{self.cfg.window_height}, Headless={self.cfg.headless})")

    def stop(self) -> None:
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            logger.info("Chrome WebDriver closed.")

    def navigate_fresh(self) -> None:
        """Loads URL fresh and waits for 3D canvas and action buttons."""
        if not self.driver:
            raise RuntimeError("Driver not started.")
        self.driver.get(self.cfg.url)
        wait = WebDriverWait(self.driver, self.cfg.page_load_timeout)
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
        wait.until(lambda d: all(lbl in self._get_button_texts(d) for lbl in BUTTON_LABELS.values()))

        # Enforce fixed window dimensions across initial and subsequent page loads
        self.driver.set_window_size(self.cfg.window_width, self.cfg.window_height)
        # Ensure WebGL render loop has started and finished compiling shaders
        time.sleep(2.0)

    @staticmethod
    def _get_button_texts(driver: webdriver.Chrome) -> set[str]:
        return {btn.text.strip() for btn in driver.find_elements(By.TAG_NAME, "button") if btn.text.strip()}

    def get_button_elements(self) -> Dict[str, any]:
        mapping = {}
        for btn in self.driver.find_elements(By.TAG_NAME, "button"):
            txt = btn.text.strip()
            if txt:
                mapping[txt] = btn
        return mapping

    def drag_canvas_safely(self, dx: int, dy: int) -> None:
        """
        Executes click-and-drag starting strictly from an empty bottom-right corner
        proportional to canvas size to avoid raycast collisions with the 3D car mesh.
        """
        canvas = self.driver.find_element(By.TAG_NAME, "canvas")
        rect = canvas.rect
        w, h = rect["width"], rect["height"]
        ox, oy = compute_safe_drag_offset(w, h)
        ActionChains(self.driver).move_to_element_with_offset(
            canvas, ox, oy
        ).click_and_hold().move_by_offset(
            dx, dy
        ).release().perform()
        time.sleep(self.cfg.post_drag_delay)

    def capture_screenshot(self) -> Image.Image:
        png_bytes = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# =====================================================================
# 3. 3D CAR STATE MANAGER (DOM & UI Controls)
# =====================================================================

class CarStateManager:
    """Applies door/hood target combinations and hides UI overlays."""

    def __init__(self, browser: BrowserController, config: CollectorConfig):
        self.browser = browser
        self.cfg = config

    def apply_state(self, target_state: Dict[str, int]) -> None:
        """Clicks only the buttons whose target state is open (1)."""
        btn_map = self.browser.get_button_elements()
        for comp_name, is_open in target_state.items():
            if is_open:
                lbl = BUTTON_LABELS[comp_name]
                btn = btn_map.get(lbl)
                if not btn:
                    raise RuntimeError(f"Button for {comp_name} ({lbl}) not found in DOM.")
                self.browser.driver.execute_script("arguments[0].click();", btn)
                time.sleep(self.cfg.post_click_delay)

    def hide_ui_containers(self) -> None:
        """Hides buttons and their parent containers so screenshots contain only the car & canvas."""
        self.browser.driver.execute_script(
            """
            document.querySelectorAll('button').forEach((b) => {
                b.style.visibility = 'hidden';
                b.style.pointerEvents = 'none';
                if (b.parentElement && b.parentElement.tagName !== 'BODY') {
                    b.parentElement.style.display = 'none';
                    b.parentElement.style.visibility = 'hidden';
                    b.parentElement.style.pointerEvents = 'none';
                }
            });
            """
        )


# =====================================================================
# 4. DATASET REPOSITORY (Storage & Resumable Indexing)
# =====================================================================

class DatasetRepository:
    """Handles thread-safe/resumable saving of JPEG images and annotations.json."""

    def __init__(self, save_dir: Path, fresh: bool = False):
        self.save_dir = save_dir
        if fresh and self.save_dir.exists():
            backup_dir = self.save_dir.parent / "raw_legacy_backup"
            if not backup_dir.exists():
                logger.info(f"Backing up existing dataset to {backup_dir}...")
                shutil.move(str(self.save_dir), str(backup_dir))
            else:
                logger.warning(f"Backup {backup_dir} already exists. Removing current {save_dir}...")
                shutil.rmtree(str(self.save_dir))

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_file = self.save_dir / "annotations.json"
        self.records: List[dict] = []
        self._existing_sigs: Set[Tuple[Tuple[int, ...], str, int]] = set()
        self._existing_files: Set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if self.annotations_file.exists():
            try:
                with open(self.annotations_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                for r in self.records:
                    self._existing_files.add(r["file"])
                    sig = (
                        tuple(r["labels"][k] for k in COMPONENTS),
                        r["elevation"],
                        int(r["angle"]),
                    )
                    self._existing_sigs.add(sig)
                logger.info(f"Loaded existing annotations: {len(self.records)} records found.")
            except Exception as e:
                logger.warning(f"Could not read existing annotations.json: {e}. Starting fresh.")
                self.records = []
                self._existing_sigs = set()
                self._existing_files = set()

    def has_signature(self, combo: Tuple[int, ...], elev_name: str, angle: int) -> bool:
        return (combo, elev_name, angle) in self._existing_sigs

    def save_sample(
        self,
        image: Image.Image,
        filename: str,
        labels: Dict[str, int],
        angle: int,
        elevation: str,
        quality: int = 92,
    ) -> None:
        img_path = self.save_dir / filename
        image.save(img_path, quality=quality, optimize=True)
        record = {
            "file": filename,
            "labels": labels,
            "angle": angle,
            "elevation": elevation,
        }
        self.records.append(record)
        self._existing_files.add(filename)
        sig = (tuple(labels[k] for k in COMPONENTS), elevation, angle)
        self._existing_sigs.add(sig)

    def persist_annotations(self) -> None:
        """Atomic write of annotations.json using a temp file."""
        # Sort deterministically by filename
        self.records.sort(key=lambda r: r["file"])
        tmp_path = self.save_dir / "annotations.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        tmp_path.replace(self.annotations_file)
        logger.info(f"Persisted {len(self.records)} annotations to {self.annotations_file}")


# =====================================================================
# 5. DATA COLLECTION PIPELINE (Orchestrator)
# =====================================================================

class DataCollectionPipeline:
    """Coordinates the full collection process with pilot & resume support."""

    def __init__(self, config: CollectorConfig):
        self.cfg = config
        self.browser = BrowserController(config)
        self.car_mgr = CarStateManager(self.browser, config)
        self.repo = DatasetRepository(config.save_dir, fresh=config.fresh)

    def run(self) -> None:
        all_combos = list(itertools.product([0, 1], repeat=len(COMPONENTS)))
        if self.cfg.pilot:
            # 2 extreme combinations: all-closed and all-open
            combos_to_run = [all_combos[0], all_combos[-1]]
            mode_desc = f"PILOT MODE ({len(combos_to_run)} combos x {len(ELEVATIONS)} elev x 8 angles = 48 samples)"
        else:
            combos_to_run = all_combos
            mode_desc = f"FULL MODE ({len(combos_to_run)} combos x {len(ELEVATIONS)} elev x 8 angles = 768 samples)"

        total_runs = len(combos_to_run) * len(ELEVATIONS)

        logger.info("=" * 65)
        logger.info("STARTING ROBUST DATA COLLECTION")
        logger.info(mode_desc)
        logger.info(f"Save Directory: {self.cfg.save_dir}")
        logger.info("=" * 65)

        self.browser.start()
        run_idx = 0

        try:
            for combo in combos_to_run:
                target_state = dict(zip(COMPONENTS, combo))

                for elev_offset, elev_name in ELEVATIONS:
                    run_idx += 1
                    # Check if all angles for this combo & elevation are already captured
                    all_done = all(self.repo.has_signature(combo, elev_name, a) for a in CAMERA_ANGLES)
                    if all_done:
                        logger.info(f"Run {run_idx:03d}/{total_runs} | Combo={combo} | Elev={elev_name} [ALREADY COMPLETE, SKIPPING]")
                        continue

                    logger.info(f"Run {run_idx:03d}/{total_runs} | Combo={combo} | Elev={elev_name}...")

                    # Fresh load per run guarantees a known origin camera viewpoint
                    self.browser.navigate_fresh()
                    self.car_mgr.apply_state(target_state)

                    if elev_offset != 0:
                        self.browser.drag_canvas_safely(0, elev_offset)

                    self.car_mgr.hide_ui_containers()
                    time.sleep(0.3)

                    for angle_idx, angle in enumerate(CAMERA_ANGLES):
                        if angle_idx > 0:
                            self.browser.drag_canvas_safely(self.cfg.drag_pixels_per_step, 0)

                        if self.repo.has_signature(combo, elev_name, angle):
                            continue

                        sid = canonical_sample_id(combo, elev_name, angle)
                        filename = f"sample_{sid:05d}_angle{angle}_elev{elev_name}.jpg"

                        img = self.browser.capture_screenshot()
                        self.repo.save_sample(
                            image=img,
                            filename=filename,
                            labels=target_state,
                            angle=angle,
                            elevation=elev_name,
                            quality=self.cfg.image_quality,
                        )

                    # Persist metadata incrementally after each angle sweep
                    self.repo.persist_annotations()

            logger.info("=" * 65)
            logger.info(f"COLLECTION SWEEP COMPLETED: {len(self.repo.records)} total samples stored.")
            logger.info("=" * 65)

        finally:
            self.browser.stop()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    config = CollectorConfig.from_args(args)
    pipeline = DataCollectionPipeline(config)
    pipeline.run()
