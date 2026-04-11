# Car Component AI Detector

Real-time deep learning system that detects open/closed states of car doors and hood
by capturing the user's currently active browser tab showing a 3D car web interface.

---

## Objectives

| # | Task | Type | Status |
|---|------|------|--------|
| 1 | Custom CNN from scratch — real-time detection via browser capture | Mandatory | Done |
| 2 | Fine-tune VLM (SmolVLM-500M-Instruct) — natural language description | Bonus | Ready to run |
| 3 | Visual Grounding (Grounding DINO) — text-prompted bounding boxes | Bonus | Done |

---

## Project Structure

```
car-detection/
├── data_collection/
│   ├── collect_data.py        # Selenium: captures 32×8×3 = 768 images automatically
│   └── augment_data.py        # 7-type augmentation → ~5,376 total samples
├── model/
│   ├── architecture.py        # Custom CNN + SE-Attention (built from scratch, no pretrained)
│   ├── dataset.py             # Combo-stratified train/val split
│   ├── train.py               # AdamW + CosineAnnealingLR, 80 epochs
│   ├── evaluate.py            # Per-label accuracy, ROC-AUC, exact-match
│   └── change_detector.py     # 3-frame temporal buffer, hysteresis thresholds
├── backend/
│   ├── app.py                 # FastAPI — WebSocket broker + REST endpoints
│   ├── vlm_module.py          # SmolVLM-500M + LoRA lazy-loader
│   └── requirements.txt
├── extension/
│   ├── manifest.json          # Chrome Extension Manifest V3
│   ├── background.js          # Streams active tab frames via WebSocket (1 FPS)
│   └── popup.html             # Start / Stop buttons
├── frontend/
│   ├── index.html             # Objective 1 — live status dashboard
│   ├── vlm.html               # Objective 2 — capture & describe page
│   └── grounding.html         # Objective 3 — text-prompted grounding page
├── vlm/
│   ├── finetune.py            # QLoRA fine-tuning script (run once, needs GPU)
│   └── inference.py           # Standalone VLM CLI for testing
├── visual_grounding/
│   └── inference.py           # Grounding DINO wrapper + prompt expansion
├── checkpoints/
│   └── best_model.pt          # Trained CNN weights (auto-created by train.py)
├── docs/
│   ├── PRESENTATION_SLIDE_DECK.md
│   ├── DEMO_VIDEO_GUIDE.md
│   └── SUBMISSION_CHECKLIST.md
├── .env                       # CAR_URL setting
├── Dockerfile
└── docker-compose.yml
```

---

## System Architecture

```
  USER'S CHROME BROWSER
  (tab: 3D car model page)
         │
         │  captureVisibleTab() — no new browser opened
         ▼
  Chrome Extension (extension/)
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
frontend/index.html        frontend/vlm.html    frontend/grounding.html
  (live status)            (NL description)      (bounding boxes)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | Recommended |
| Google Chrome | Latest | For the extension |
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

> Horizontal flip is intentionally excluded — it would swap left↔right door labels.

---

## Step 4 — Train the CNN Model

> Skip if `checkpoints/best_model.pt` already exists.

```bash
python model/train.py
```

**Output:** `checkpoints/best_model.pt`

Training configuration:
- Architecture: 5-block CNN + SE-Attention, built fully from scratch (~3.2 M params)
- Loss: BCEWithLogitsLoss (multi-label, 5 independent binary outputs)
- Optimizer: AdamW — lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=80 epochs
- Split: combo-stratified 80/20 (val = only original images from held-out combos)

---

## Step 5 — Evaluate the Model *(optional)*

```bash
python model/evaluate.py
```

Prints per-component precision/recall/F1, ROC-AUC, and exact-match accuracy.

---

## Step 6 — Fine-Tune VLM *(Objective 2 only — needs GPU)*

> Skip if doing Objective 1 only.  
> Requires GPU with ≥ 4 GB VRAM. Takes ~15–20 minutes (SmolVLM-500M is much lighter than PaliGemma-3B).

```bash
python vlm/finetune.py
```

**Output:** `vlm/vlm_checkpoints/final/` — LoRA adapter weights

> If skipped, the backend falls back to base SmolVLM-500M-Instruct (still functional, just not fine-tuned).

---

## Step 7 — Start the Backend

### Option A — Direct (development)

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify: open `http://localhost:8000/health` — should return `{"status":"ok",...}`

### Option B — Docker

```bash
docker-compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend (nginx): `http://localhost:3000`

---

## Step 8 — Run the Web App (All Objectives)

> **No Chrome Extension needed.** The web app uses the Screen Capture API built into
> modern browsers — just share a browser tab.

1. Backend must be running (Step 7).

2. Open the 3D car model page in Chrome:
   `https://euphonious-concha-ab5c5d.netlify.app/`

3. Open `frontend/index.html` in a **second tab** (or `http://localhost:3000` via Docker).

4. Click **Start Capture** and select the tab showing the 3D car model.

5. The app streams frames at 1 FPS — switch between the three tabs inside the app:

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
| `/health` | GET | Backend health check |
| `/ws/extension` | WebSocket | Receives JPEG frames from the Chrome extension |
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
| `ws://localhost:8000` refused | Backend not running | Run Step 7 first |
| Extension ID not found | Extension not loaded | Go to `chrome://extensions/` and load unpacked |
| Dashboard never updates | Extension not streaming | Click ▶ Start Detection in the extension popup |
| Extension captures wrong tab | Active tab ≠ 3D car page | Switch to the 3D car tab, then click Start |
| `WARNING: No checkpoint found` | Model not trained yet | Run Steps 2–4 |
| `WARNING: LoRA weights not found` | VLM not fine-tuned | Run Step 6, or ignore (base SmolVLM used) |
| Grounding DINO `FileNotFoundError` | Weights not downloaded | Run the wget command in Step 1 |
| Selenium `TimeoutException` | ChromeDriver version mismatch | Match ChromeDriver to your Chrome version |
| CUDA OOM on VLM fine-tune | Not enough VRAM | Set `per_device_train_batch_size=1` in `vlm/finetune.py` |

---

## Model Details

### Custom CNN — Objective 1

```
Input: 224 × 224 RGB
  ↓
Conv Block 1:  3  →  32 ch  (Conv3×3 → BN → GELU → Conv3×3 → BN → GELU → MaxPool2×2 → Dropout)
Conv Block 2:  32 →  64 ch
Conv Block 3:  64 → 128 ch
Conv Block 4: 128 → 256 ch
Conv Block 5: 256 → 512 ch
  ↓
Squeeze-and-Excitation attention  (512 → 64 → 512 → sigmoid)
  ↓
AdaptiveAvgPool2d(1) → Flatten
  ↓
FC 512 → 256 → 64 → 5  (GELU + Dropout between layers)
  ↓
5 independent sigmoid outputs  (FL door · FR door · RL door · RR door · Hood)
```

**Parameters:** ~3.2 M  
**Inference:** ~8 ms CPU / ~2 ms GPU per frame  
**No transfer learning** — all weights trained from random initialisation

### Change Detector

Each frame → CNN → probabilities averaged over a **3-frame rolling buffer**.  
Hysteresis prevents flickering at ambiguous angles:

- p ≥ 0.60 → **Open**
- p ≤ 0.40 → **Closed**
- 0.40 < p < 0.60 → keep previous state

### VLM — Objective 2

| Setting | Value |
|---|---|
| Base model | HuggingFaceTB/SmolVLM-500M-Instruct |
| Parameters | 507 M (vs 3 B for PaliGemma — 6× lighter) |
| Method | QLoRA (4-bit NF4 quantisation) |
| LoRA rank | r=16, alpha=32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Epochs | 3 · lr=2e-4 · batch 2 + grad accum 8 |

### Visual Grounding — Objective 3

| Setting | Value |
|---|---|
| Model | Grounding DINO SwinT-OGC |
| Mode | Zero-shot (no fine-tuning needed) |
| Box threshold | 0.35 |
| Text threshold | 0.25 |
| Prompt expansion | `"front left door"` → `"front left door . door"` |

---

## Evaluation Criteria

| Criteria | How it is met |
|---|---|
| Model Architecture Efficiency | From-scratch CNN, 3.2 M params, ~8 ms/frame inference |
| Model Accuracy | Per-label BCE loss · ROC-AUC · exact-match — run `python model/evaluate.py` |
| Real-Time Integration | Extension → WebSocket → FastAPI → live dashboard at 1 FPS |
| Presentation & Reporting | README + `docs/` slide deck + per-label evaluation report |
