"""
Automated dataset collection from the current 3D car web page.
Captures all 2^5 = 32 state combinations x 8 horizontal angles x 3 elevations.
Total raw samples: 32 x 8 x 3 = 768
"""
import io
import itertools
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse


from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CAR_URL = "https://euphonious-concha-ab5c5d.netlify.app/"
SAVE_DIR  = ROOT_DIR / "dataset" / "raw"
DEBUG_DIR = ROOT_DIR / "dataset" / "debug"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = [
    "front_left_door",
    "front_right_door",
    "rear_left_door",
    "rear_right_door",
    "hood",
]
BUTTON_LABELS = {
    "front_left_door": "Front Left Door",
    "front_right_door": "Front Right Door",
    "rear_left_door": "Rear Left Door",
    "rear_right_door": "Rear Right Door",
    "hood": "Hood",
}

# 8 horizontal rotation steps (each step ≈ 45°, drag=-140px on 1280-wide canvas)
CAMERA_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# 3 vertical elevation levels (pixels to drag vertically after page load).
# Negative = drag downward on canvas = tilt camera downward (bird-eye view).
# Positive = drag upward  on canvas = tilt camera upward  (worm-eye view).
# 0 = default eye-level.
ELEVATION_OFFSETS = [
    (-70, "top"),    # slightly top-down view
    (0,   "mid"),    # eye-level (default)
    (70,  "low"),    # slightly low-angle view
]

PAGE_LOAD_TIMEOUT = 45
POST_CLICK_DELAY = 0.6
POST_DRAG_DELAY  = 0.7
# Pixels per 45° horizontal step (calibrated for 1280-wide window)
DRAG_PIXELS_PER_STEP = -140


def _read_car_url_from_dotenv():
    dotenv_path = ROOT_DIR / ".env"
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "CAR_URL":
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None


def resolve_car_url():
    value = os.environ.get("CAR_URL") or _read_car_url_from_dotenv() or DEFAULT_CAR_URL
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().strip('"').strip("'")
    if not value:
        value = DEFAULT_CAR_URL
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
        parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid CAR_URL: {value!r}")
    if not value.endswith("/"):
        value += "/"
    return value


CAR_URL = resolve_car_url()


def setup_driver():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--window-size=1280,720")
    if os.environ.get("HEADLESS", "").lower() in {"1", "true", "yes"}:
        opts.add_argument("--headless=new")
    return webdriver.Chrome(options=opts)


def save_debug_artifacts(driver, prefix):
    screenshot_path = DEBUG_DIR / f"{prefix}.png"
    html_path = DEBUG_DIR / f"{prefix}.html"
    try:
        driver.save_screenshot(str(screenshot_path))
    except Exception:
        pass
    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
    except Exception:
        pass


def get_button_map(driver):
    mapping = {}
    for button in driver.find_elements(By.TAG_NAME, "button"):
        label = button.text.strip()
        if label:
            mapping[label] = button
    return mapping


def wait_for_scene_ready(driver):
    try:
        driver.get(CAR_URL)
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "canvas"))
        )
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            lambda d: all(label in get_button_map(d) for label in BUTTON_LABELS.values())
        )
    except TimeoutException as exc:
        save_debug_artifacts(driver, "page_load_timeout")
        available = sorted(get_button_map(driver).keys())
        raise RuntimeError(
            "Timed out waiting for the current 3D car page to finish loading. "
            f"Available buttons: {available or 'none'}; title={driver.title!r}; url={driver.current_url}"
        ) from exc

    time.sleep(1.5)


def click_component_button(driver, component):
    label = BUTTON_LABELS[component]
    button = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        lambda d: get_button_map(d).get(label)
    )
    driver.execute_script("arguments[0].click();", button)
    time.sleep(POST_CLICK_DELAY)


def hide_controls_for_capture(driver):
    driver.execute_script(
        """
        document.querySelectorAll('button').forEach((button) => {
            button.style.visibility = 'hidden';
            button.style.pointerEvents = 'none';
        });
        """
    )


def drag_canvas(driver, dx: int, dy: int):
    """Drag the canvas by (dx, dy) pixels to rotate/tilt the camera."""
    canvas = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        EC.presence_of_element_located((By.TAG_NAME, "canvas"))
    )
    ActionChains(driver).move_to_element(canvas).click_and_hold().move_by_offset(
        dx, dy
    ).release().perform()
    time.sleep(POST_DRAG_DELAY)


def rotate_camera_horizontal(driver):
    """Rotate camera one step (~45°) to the right."""
    drag_canvas(driver, DRAG_PIXELS_PER_STEP, 0)


def apply_elevation(driver, vertical_offset: int):
    """Tilt camera vertically. Negative = bird-eye, positive = worm-eye."""
    if vertical_offset != 0:
        drag_canvas(driver, 0, vertical_offset)


def capture_screenshot(driver):
    png = driver.get_screenshot_as_png()
    return Image.open(io.BytesIO(png)).convert("RGB")


def apply_component_state(driver, target_state):
    for component in COMPONENTS:
        if target_state[component]:
            click_component_button(driver, component)


def collect_dataset():
    driver = setup_driver()
    annotations = []
    sample_id = 0
    all_combos = list(itertools.product([0, 1], repeat=len(COMPONENTS)))

    total_runs = len(all_combos) * len(ELEVATION_OFFSETS)
    run_idx = 0

    try:
        for combo in all_combos:
            target = dict(zip(COMPONENTS, [bool(v) for v in combo]))

            for elev_offset, elev_name in ELEVATION_OFFSETS:
                run_idx += 1
                # Reload page and set up state fresh for each (combo, elevation) pair.
                # This guarantees a known starting camera pose every time.
                wait_for_scene_ready(driver)
                apply_component_state(driver, target)
                apply_elevation(driver, elev_offset)
                hide_controls_for_capture(driver)
                time.sleep(0.4)

                for angle_idx, angle in enumerate(CAMERA_ANGLES):
                    if angle_idx > 0:
                        rotate_camera_horizontal(driver)

                    image = capture_screenshot(driver)
                    filename = f"sample_{sample_id:05d}_angle{angle}_elev{elev_name}.jpg"
                    image.save(SAVE_DIR / filename, quality=90)
                    annotations.append(
                        {
                            "file": filename,
                            "labels": {
                                component: int(value)
                                for component, value in target.items()
                            },
                            "angle": angle,
                            "elevation": elev_name,
                        }
                    )
                    sample_id += 1

                print(
                    f"Run {run_idx:03d}/{total_runs} combo={combo} elev={elev_name} "
                    f"done ({sample_id} samples so far)"
                )
    finally:
        driver.quit()

    with open(SAVE_DIR / "annotations.json", "w", encoding="utf-8") as handle:
        json.dump(annotations, handle, indent=2)

    print(f"Done. Total: {sample_id} images collected.")


if __name__ == "__main__":
    collect_dataset()
