# Car Component AI Detector

Real-time deep learning system that detects open/closed states of car doors and hood
by capturing the user's browser tab showing a 3D car web interface via the Screen Capture API.

---

## Objectives

| # | Task | Type | Status |
|---|------|------|--------|
| 1 | Custom CNN from scratch — real-time detection via Screen Capture | Mandatory | Done |
| 2 | Fine-tune VLM (SmolVLM-500M-Instruct) — natural language description | Bonus | Ready to run |
| 3 | Visual Grounding (Grounding DINO) — text-prompted bounding boxes | Bonus | Done |

---

## Quick Start

```bash
./start.sh
```

This starts the backend and opens `http://localhost:8000` in your browser. That's it.

---

## Project Structure

```
car-detection/
├── data_collection/
│   ├── collect_data.py        # Selenium: captures 32×8×3 = 768 images automatically
│   └── augment_data.py        # 7-type augmentation → ~5,376 total samples
├── model/
│   ├── architecture.py        # Custom CNN + SE-Attention (built from scratch, no pretrained)
│   ├── dataset.py             # Angle-stratified train/val split (50/50 per-label balance)
│   ├── train.py               # AdamW + OneCycleLR, mixup, label smoothing, early stopping
│   ├── evaluate.py            # Per-label accuracy, ROC-AUC, exact-match, threshold search
│   └── change_detector.py     # 5-frame temporal buffer, per-label hysteresis thresholds
├── backend/
│   ├── app.py                 # FastAPI — WebSocket broker + REST endpoints + static frontend
│   ├── vlm_module.py          # SmolVLM-500M + LoRA lazy-loader
│   └── requirements.txt
├── frontend/
│   └── index.html             # Single-page web app — all 3 objectives in one page
├── vlm/
│   ├── finetune.py            # QLoRA fine-tuning script (optimised for 4 GB VRAM)
│   └── inference.py           # Standalone VLM CLI for testing
├── visual_grounding/
│   └── inference.py           # Grounding DINO wrapper + prompt expansion
├── checkpoints/
│   ├── best_model.pt          # Trained CNN weights (auto-created by train.py)
│   └── thresholds.json        # Per-label optimal thresholds (auto-created by evaluate.py)
├── docs/
│   ├── PRESENTATION_SLIDE_DECK.md
│   ├── DEMO_VIDEO_GUIDE.md
│   └── SUBMISSION_CHECKLIST.md
├── start.sh                   # One-command launcher: backend + browser
├── .env                       # CAR_URL setting
├── Dockerfile
└── docker-compose.yml
```

---

## System Architecture

```
  USER'S BROWSER
  (tab: 3D car model page)
         │
         │  Screen Capture API (getDisplayMedia)
         ▼
  Web App (localhost:8000)
         │
         │  WebSocket  ws://localhost:8000/ws/extension
         │  JPEG frames @ 1 FPS
         ▼
  FastAPI Backend (backend/app.py)
         │
    ┌────┴──────────────────────────┐
    │                               │
[CNN + ChangeDetector]     POST /api/vlm/describe
(model/ + change_detector)  POST /api/grounding/locate
    │
    │  WebSocket  ws://localhost:8000/ws/dashboard
    ▼
  Web App — tab-based UI
  [Detection] [VLM Description] [Visual Grounding]
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Recommended |
| Google Chrome | Latest | Screen Capture API support |
| ChromeDriver | Must match Chrome | Needed only for data collection |
| CUDA GPU | 4 GB+ VRAM | Only for VLM fine-tuning (Objective 2) |

### Install ChromeDriver (Ubuntu/Debian)
```bash
# Check your Chrome version first
google-chrome --version

# Install matching chromedriver
sudo apt install chromium-chromedriver
# OR download manually from https://chromedriver.chromium.org/downloads
```

---

## Step 1 — Install Python Dependencies

Run from the project root:

```bash
pip install -r backend/requirements.txt
```

For **Objective 3** (Visual Grounding), install Grounding DINO separately:

```bash
pip install groundingdino-py

# Download pretrained weights (~700 MB, one-time)
mkdir -p weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O weights/groundingdino_swint_ogc.pth
```

---

## Step 2 — Collect Training Data

> Skip if `dataset/raw/annotations.json` already exists.

The script opens Chrome, automatically clicks every button combination (32 states),
rotates the camera through 8 angles at 3 elevation levels, and saves each screenshot.

```bash
python data_collection/collect_data.py
```

**Output:** `dataset/raw/` — 768 JPEG images + `annotations.json`

- 32 state combos × 8 horizontal angles × 3 elevations (top / eye-level / low)
- Controls are hidden during capture so they do not appear in training images

> Set `HEADLESS=1` to run Chrome without a visible window.

---

## Step 3 — Augment the Dataset

> Skip if `dataset/augmented/annotations.json` already exists.

```bash
python data_collection/augment_data.py
```

**Output:** `dataset/augmented/` — ~5,376 images + `annotations.json`

7 augmentation types per image:
`brightness_high` · `brightness_low` · `motion_blur` · `gaussian_noise` ·
`hue_shift` · `perspective` · `jpeg_compress`

> Horizontal flip is intentionally excluded — it would swap left/right door labels.

---

## Step 4 — Train the CNN Model

> Skip if `checkpoints/best_model.pt` already exists.

```bash
python model/train.py
```

**Output:** `checkpoints/best_model.pt` + `checkpoints/thresholds.json`

Training configuration:
- Architecture: 5-block CNN + SE-Attention, built fully from scratch (~3.2 M params)
- Loss: BCEWithLogitsLoss with class-balanced pos_weight
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: OneCycleLR (warmup 10%, cosine annealing)
- Regularisation: Mixup (alpha=0.3), Label Smoothing (0.05), Dropout
- Early stopping: patience=20 epochs
- Max epochs: 100 (stops early when converged)
- Auto-evaluation: thresholds are computed and saved after training

---

## Step 5 — Evaluate the Model *(optional)*

```bash
python model/evaluate.py
```

Prints per-component precision/recall/F1, ROC-AUC, exact-match accuracy,
and saves per-label optimal thresholds to `checkpoints/thresholds.json`.

---

## Step 6 — Fine-Tune VLM *(Objective 2 only — needs GPU)*

> Skip if doing Objective 1 only.
> Requires GPU with 4 GB+ VRAM. Takes ~1.5 hours on RTX 3050 Laptop.

```bash
python vlm/finetune.py
```

**Output:** `vlm/vlm_checkpoints/final/` — LoRA adapter weights

> If skipped, the backend falls back to base SmolVLM-500M-Instruct (still functional, just not fine-tuned).

---

## Step 7 — Run the App

### Option A — Quick Start (recommended)

```bash
./start.sh
```

Starts the backend and opens `http://localhost:8000` in your browser.

Custom port: `PORT=3000 ./start.sh`

### Option B — Manual

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser.

### Option C — Docker

```bash
docker-compose up --build
```

---

## Step 8 — Use the Web App

> **No Chrome Extension needed.** The web app uses the Screen Capture API built into
> modern browsers — just share a browser tab.

1. Open the 3D car model page in Chrome:
   `https://euphonious-concha-ab5c5d.netlify.app/`

2. Go to `http://localhost:8000` (opened automatically by `start.sh`).

3. Click **Start Capture** and select the tab showing the 3D car model.

4. The app streams frames at 1 FPS. Use the tabs to switch between objectives:

### Objective 1 — Real-Time Detection
- Component states update live as you click buttons on the 3D car page.
- The dashboard shows Open/Closed badges with confidence percentages.
- Yellow "Changed" alerts appear when a component switches state.

### Objective 2 — VLM Description
- Click **Capture Current Frame & Describe**.
- The captured frame is sent to the SmolVLM backend which returns a natural language
  description like: *"The car currently has: open: front left door, hood. closed: ..."*

### Objective 3 — Visual Grounding
- Type a prompt (e.g. `front left door`) or use the quick-suggestion chips.
- Click **Generate** — the backend returns an annotated image with bounding boxes
  drawn around matched components.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web app (single-page frontend) |
| `/health` | GET | Backend health check |
| `/ws/extension` | WebSocket | Receives JPEG frames from the web app |
| `/ws/dashboard` | WebSocket | Pushes predictions to dashboard clients |
| `/api/vlm/describe` | POST | VLM natural language description |
| `/api/grounding/locate` | POST | Grounding DINO bounding box detection |

### POST `/api/vlm/describe`
```json
Request:  { "image_b64": "<base64 JPEG or data URL>" }
Response: { "description": "The car currently has: ...", "success": true }
```

### POST `/api/grounding/locate`
```json
Request:  { "image_b64": "<base64 JPEG>", "prompt": "front left door" }
Response: { "annotated_image": "<base64 JPEG>",
            "boxes": [[cx, cy, w, h], ...],
            "detected_labels": ["front left door"],
            "success": true }
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ws://localhost:8000` refused | Backend not running | Run `./start.sh` |
| Screen capture not working | Browser doesn't support Screen Capture API | Use latest Chrome or Edge |
| Dashboard never updates | Capture not started | Click "Start Capture" and share the 3D car tab |
| `WARNING: No checkpoint found` | Model not trained yet | Run Steps 2-4 |
| `WARNING: LoRA weights not found` | VLM not fine-tuned | Run Step 6, or ignore (base SmolVLM used) |
| Grounding DINO `FileNotFoundError` | Weights not downloaded | Run the wget command in Step 1 |
| Selenium `TimeoutException` | ChromeDriver version mismatch | Match ChromeDriver to your Chrome version |
| CUDA OOM on VLM fine-tune | Not enough VRAM | Already optimised for 4 GB; reduce LoRA rank if needed |

---

## Model Details

### Custom CNN — Objective 1

```
Input: 224 x 224 RGB
  |
Conv Block 1:  3  ->  32 ch  (Conv3x3 -> BN -> GELU -> Conv3x3 -> BN -> GELU -> MaxPool2x2 -> Dropout)
Conv Block 2:  32 ->  64 ch
Conv Block 3:  64 -> 128 ch
Conv Block 4: 128 -> 256 ch
Conv Block 5: 256 -> 512 ch
  |
Squeeze-and-Excitation attention  (512 -> 64 -> 512 -> sigmoid)
  |
AdaptiveAvgPool2d(1) -> Flatten
  |
FC 512 -> 256 -> 64 -> 5  (GELU + Dropout between layers)
  |
5 independent sigmoid outputs  (FL door, FR door, RL door, RR door, Hood)
```

**Parameters:** ~3.2 M
**Inference:** ~8 ms CPU / ~2 ms GPU per frame
**No transfer learning** — all weights trained from random initialisation

### Training Improvements

| Technique | Purpose |
|---|---|
| OneCycleLR | Aggressive warmup + cosine decay for better convergence |
| Mixup (alpha=0.3) | Regularisation via interpolated training pairs |
| Label Smoothing (0.05) | Better probability calibration (thresholds closer to 0.5) |
| Early Stopping (patience=20) | Auto-stop when converged, prevents overfitting |
| Per-label Optimal Thresholds | Maximise F1 per component instead of using fixed 0.5 |

### Change Detector

Each frame -> CNN -> probabilities averaged over a **5-frame rolling buffer**.
Per-label hysteresis prevents flickering at ambiguous angles:

- p >= threshold + 0.12 -> **Open**
- p <= threshold - 0.12 -> **Closed**
- Otherwise -> keep previous state

Thresholds are loaded from `checkpoints/thresholds.json` (produced by `evaluate.py`).

### VLM — Objective 2

| Setting | Value |
|---|---|
| Base model | HuggingFaceTB/SmolVLM-500M-Instruct |
| Parameters | 507 M (vs 3 B for PaliGemma — 6x lighter) |
| Method | QLoRA (4-bit NF4 quantisation) |
| LoRA rank | r=4, alpha=8 |
| Target modules | q_proj, v_proj |
| Optimiser | AdamW 8-bit (bitsandbytes) |
| Training | 3 epochs, lr=2e-4, gradient checkpointing |
| VRAM | ~3.5 GB (fits on 4 GB GPU) |

### Visual Grounding — Objective 3

| Setting | Value |
|---|---|
| Model | Grounding DINO SwinT-OGC |
| Mode | Zero-shot (no fine-tuning needed) |
| Box threshold | 0.35 |
| Text threshold | 0.25 |
| Prompt expansion | `"front left door"` -> `"front left door . door"` |

---

## Evaluation Criteria

| Criteria | How it is met |
|---|---|
| Model Architecture Efficiency | From-scratch CNN, 3.2 M params, ~8 ms/frame inference |
| Model Accuracy | Per-label BCE loss, ROC-AUC, exact-match — run `python model/evaluate.py` |
| Real-Time Integration | Screen Capture -> WebSocket -> FastAPI -> live dashboard at 1 FPS |
| Presentation & Reporting | README + `docs/` slide deck + per-label evaluation report |
