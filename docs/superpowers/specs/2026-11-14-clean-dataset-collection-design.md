# Clean Dataset Collection & Verification Design (Paper 1 & Paper 2 Foundation)

**Date:** 2026-11-14  
**Scope:** Small Change (Data Collection Calibration, Pilot Verification, & Automated Integrity Audit)

## 1. Intent & Why
Provide a 100% mathematically verified, raycast-free, multi-elevation 3D car dataset ($2^5 \text{ states} \times 3 \text{ elevations} \times 8 \text{ angles} = 768 \text{ raw samples}$, plus $5,376$ augmented samples) to serve as the ground-truth benchmark for:
- **Paper 1:** *Multi-Scale PA-SCAN Custom CNN from Scratch for Real-Time 3D Car Component Detection*
- **Paper 2:** *Comparative Study of Lightweight Custom CNN vs. Fine-Tuned PaliGemma VLM vs. Zero-Shot Grounding DINO*

## 2. Goals, Non-Goals, & Anti-Goals
- **Goals:**
  - Eliminate `MoveTargetOutOfBoundsException` and accidental raycast clicks by using dynamic canvas-proportional bottom-right origin offsets $(o_x, o_y) = (+0.28\,W_{\text{canvas}}, +0.22\,H_{\text{canvas}})$.
  - Support a `--pilot` CLI mode in `data_collection/collect_data.py` that captures 2 extreme combinations (`(0,0,0,0,0)` all-closed and `(1,1,1,1,1)` all-open $\times$ 3 elevations $\times$ 8 angles = 48 images) in ~1 minute.
  - Provide an automated forensic verification script (`data_collection/verify_dataset.py`) that computes inter-state $\text{MAE}$, inter-angle uniqueness (confirming no $270^\circ \approx 0^\circ$ viewpoint overlap), resolution consistency, and UI-bar removal before running the remaining 30 combinations.
  - Back up the legacy `dataset/raw` metadata before full collection and support seamless auto-resume from the pilot run to the full 768-image dataset.
- **Non-Goals:** Changing the 5 target component labels (`FL, FR, RL, RR, Hood`) or modifying the external Netlify 3D web app.
- **Anti-Goals:** Capturing 768 images blindly without verifying pilot angle calibration and raycast safety first.

## 3. Constraints
- Must run reliably on both Windows and Ubuntu (headless or visible Chrome) via Selenium WebDriver.
- Must keep all mouse drag trajectories strictly inside the `<canvas>` viewport $[0, W] \times [0, H]$ and strictly outside the 3D car bounding box $B_{\text{car}} = [0.17\,W, 0.83\,W] \times [0.19\,H, 0.62\,H]$.

## 4. Approaches Considered
1. **Approach A (Selected — Pilot Verification $\to$ Automated Pixel Audit $\to$ Resumable Full 768 Capture):**
   Run a 2-combo pilot (48 images, ~1 min), run `verify_dataset.py` to mathematically confirm $\text{MAE} > 0$ across all states and angles, then auto-resume `collect_data.py` for the remaining 30 combos (720 images). *Trade-off:* Adds a 1-minute verification gate upfront, eliminating the risk of wasting 20 minutes on miscalibrated captures.
2. **Approach B (Direct Full 768 Capture):**
   Run all 96 sweeps in one shot without a pilot check. *Rejected:* Any browser rendering or DPI issue would only be discovered after 20 minutes.
3. **Approach C (2-Elevation Subset — 512 Images):**
   Drop the `top` elevation to save collection time. *Rejected:* Weakens the 3D multi-elevation experimental claims for both Paper 1 and Paper 2.

## 5. Technical Design Decisions
- **`data_collection/collect_data.py`:**
  - Compute safe drag start offset dynamically from `canvas.rect`: `ox = int(0.28 * width)`, `oy = int(0.22 * height)` so start position is $(0.78\,W, 0.72\,H)$ — strictly below the car ($y > 0.62\,H$) and with ample room to drag left ($\Delta x = -115\text{ px}$) and down ($\Delta y = +85\text{ px}$).
  - Add CLI flags: `--pilot` (runs combos `(0,0,0,0,0)` and `(1,1,1,1,1)`), `--fresh` (archives legacy `dataset/raw` to `dataset/raw_legacy_backup` before starting), and `--headless`.
  - Key resumability by `(tuple(combo), elevation, angle)` signature rather than sequential `sample_id` so resuming after `--pilot` automatically fills in the remaining 30 combinations and sorts `annotations.json` cleanly.
- **`data_collection/verify_dataset.py`:**
  - Verifies image dimensions, checks top UI bar absence, computes pairwise inter-angle $\text{MAE}$ within each combo (ensuring all 8 angles are distinct), and computes inter-state $\text{MAE}$ inside `crop_car_roi` between closed and open states across all 3 elevations (`mid`, `high`, `top`).
