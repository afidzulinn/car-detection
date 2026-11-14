# Car Component AI Detector

Real-time deep learning system that detects open/closed states of 5 car components
(`front_left_door`, `front_right_door`, `rear_left_door`, `rear_right_door`, `hood`)
by capturing the user's currently active browser tab showing a 3D car web interface.

---

## Objectives

| # | Task | Type | Status |
|---|------|------|--------|
| 1 | Custom CNN from scratch (Multi-Scale PA-SCAN) — real-time detection via browser capture | Mandatory | Done (**96.56% Mean Acc / 0.9934 AUC**) |
| 2 | Fine-tune VLM (PaliGemma) — natural language description | Bonus | Ready to run |
| 3 | Visual Grounding (Grounding DINO) — text-prompted bounding boxes | Bonus | Done |

---

## Project Structure

```
car-detection/
├── data_collection/
│   ├── collect_data.py        # Raycast-safe Selenium collector (32×8×3 = 768 images)
│   └── augment_data.py        # 7-type geometry-safe augmentation → ~6,144 total samples
├── model/
│   ├── architecture.py        # Multi-Scale PA-SCAN Custom CNN (from scratch, no pretrained)
│   ├── dataset.py             # Paired angle-stratified split + ROI crop + RAM cache
│   ├── train.py               # Cross-platform (Windows/Ubuntu) + CUDA AMP FP16 trainer
│   ├── evaluate.py            # Per-label P/R/F1, ROC-AUC, exact-match & threshold search
│   └── change_detector.py     # 3-frame rolling buffer + per-label hysteresis thresholds
├── backend/
│   ├── app.py                 # FastAPI — WebSocket broker + ROI inference + REST endpoints
│   ├── vlm_module.py          # PaliGemma + LoRA lazy-loader
│   └── requirements.txt       # Includes PyTorch CUDA 12.4 index for Windows & Ubuntu
├── extension/
│   ├── manifest.json          # Chrome Extension Manifest V3
│   ├── background.js          # Streams active tab frames via WebSocket (1 FPS)
│   └── popup.html             # Start / Stop controls
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
│   ├── best_model.pt          # Trained Custom CNN weights (auto-created by train.py)
│   └── thresholds.json        # Per-label optimal F1 thresholds (auto-created by evaluate.py)
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
[ROI Crop + PA-SCAN CNN]    POST /api/vlm/describe
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
| Python | 3.10 or 3.11 | Windows & Ubuntu/Linux supported |
| Google Chrome | Latest | For Chrome Extension & Selenium capture |
| ChromeDriver | Matching Chrome | Auto-managed by Selenium 4.18+ |
| NVIDIA GPU *(optional)* | RTX 3050 / CUDA 12.x | Supported with FP16 AMP acceleration (`cu124`) |

---

## Step 1 — Virtual Environment & Dependencies dengan `uv` (Windows & Ubuntu)

Proyek ini sudah dilengkapi konfigurasi [`uv.toml`](uv.toml) (`index-strategy = "unsafe-best-match"`) agar `uv` otomatis mengunduh wheel **PyTorch CUDA 12.4 (`+cu124`)** untuk GPU NVIDIA RTX (bukan versi CPU bawaan PyPI).

### A. Buat & Aktifkan `.venv` dengan `uv`

Jalankan dari folder root proyek (`E:\car-detection`):

```powershell
# 1. Buat virtual environment (.venv) dengan uv (sangat cepat, <1 detik)
uv venv --python 3.10
```

**2. Aktifkan `.venv` sesuai terminal Anda:**
- **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows Command Prompt (`cmd.exe`):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Windows Git Bash (`MINGW64`):**
  ```bash
  source .venv/Scripts/activate
  ```
- **Ubuntu / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### B. Instal Dependensi + PyTorch GPU dengan `uv`

```powershell
# Instal seluruh paket di backend/requirements.txt (otomatis menarik PyTorch CUDA 12.4)
uv pip install -r backend/requirements.txt --index-strategy unsafe-best-match
```

Jika di environment Anda sebelumnya sudah terinstal `torch` versi `+cpu`, paksa ganti ke versi GPU CUDA 12.4 dengan `uv`:

```powershell
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verifikasi bahwa **NVIDIA GeForce RTX 3050 Laptop GPU** sudah aktif:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Tips `uv run`:** Anda juga bisa langsung menjalankan script tanpa aktivasi manual menggunakan `uv run python model/train.py` atau `uv run python model/evaluate.py`.

For **Objective 3** (Visual Grounding), install Grounding DINO separately:

```bash
pip install groundingdino-py

# Download pretrained weights (~700 MB, one-time)
mkdir -p weights
curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o weights/groundingdino_swint_ogc.pth
```

---

## Step 2 — Collect Training Data

> Skip if `dataset/raw/annotations.json` already exists.

The script opens Chrome, clicks every component combination ($2^5 = 32$ states),
rotates the camera through 8 horizontal angles ($0^\circ..315^\circ$) at 3 elevation levels,
and saves each screenshot.

```bash
python data_collection/collect_data.py
```

**Key calibration & anti-corruption safeguards in `collect_data.py`:**
- **Raycast-Free Corner Drag:** Camera drag starts at offset `(-450, -200)` from the `<canvas>` center so the cursor never touches the 3D car mesh or accidentally toggles doors.
- **Complete UI Container Hiding:** Sets `button.parentElement.style.display = 'none'` so the top white button bar is completely removed during capture.
- **Above-Ground Elevation Offsets:** Uses `(0, "mid")`, `(+45, "high")`, and `(+85, "top")` (avoiding negative vertical offsets that orbit underneath the car floor).

---

## Step 3 — Augment the Dataset

> Skip if `dataset/augmented/annotations.json` already exists.

```bash
python data_collection/augment_data.py
```

7 photometric & sensor augmentations per image:
`brightness_high` · `brightness_low` · `motion_blur` · `gaussian_noise` ·
`hue_shift` · `perspective` · `jpeg_compress`

> Horizontal flip is intentionally excluded so left (`FL, RL`) and right (`FR, RR`) door geometry is preserved.

---

## Step 4 — Train the Custom CNN Model (Windows & Ubuntu, GPU / CPU)

```bash
python model/train.py
```

Optional CLI flags:

```bash
# Train on NVIDIA GPU (auto-detected) or specify batch size / epochs / workers:
python model/train.py --epochs 40 --batch-size 16 --lr 0.0025
```

**Output:** `checkpoints/best_model.pt`

Training highlights:
- **Cross-Platform OS & Device Support:** Automatically detects `cuda` (with FP16 Automatic Mixed Precision `torch.amp.autocast` and `cudnn.benchmark`), `mps`, or `cpu`. Automatically selects optimal DataLoader workers (`workers=0` with RAM cache on Windows `spawn`; `workers=2..4` with Copy-on-Write RAM cache and `persistent_workers=True` on Ubuntu `fork`).
- **Deterministic ROI Cropping (`crop_car_roi`):** Crops `ROI_BOX_NORM = (0.17, 0.19, 0.83, 0.62)`, removing $71.6\%$ of empty background/UI and increasing effective 2D spatial pixel density on the car by **$3.52\times$**.
- **Paired Angle-Stratified Split:** Holds out 2 unseen camera angles per combination ($50\%$ Open / $50\%$ Closed balance per label) while pairing foreground states so mutual information $I(Y_{\text{RR}} ; \theta, Y_{\text{-RR}}) = 0$ on the training set.
- **Multi-Task Loss:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}, y) + 0.35\,\mathcal{L}_{\text{CE}}(\hat{p}(\theta), \theta^*)$ with `AdamW` and `OneCycleLR`.

---

## Step 5 — Evaluate the Model

```bash
python model/evaluate.py
```

Prints per-component Precision, Recall, F1-Score, ROC-AUC, and Exact-Match accuracy, and saves optimal per-label thresholds to `checkpoints/thresholds.json`.

### Measured Validation Benchmark (Unseen Camera Angles)

| Component (`LABELS`) | Accuracy (@0.50) | Accuracy (@Optimal Thr) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **`front_left_door` (`FL`)** | 100.00% | **100.00%** | **1.0000** | **1.0000** |
| **`front_right_door` (`FR`)** | 96.88% | **100.00%** | **1.0000** | **1.0000** |
| **`rear_left_door` (`RL`)** | 85.94% | **89.06%** | **0.8955** | **0.9746** |
| **`rear_right_door` (`RR`)** | 100.00% | **100.00%** | **1.0000** | **1.0000** |
| **`hood` (`Hood`)** | 90.62% | **93.75%** | **0.9412** | **0.9922** |
| **Mean Across All 5 Components** | **94.69%** | **96.56%** | **0.9673** | **0.9934** |
| **Exact Match (All 5 Correct Simultaneously)** | **73.44%** | **82.81%** | — | — |

---

## Step 6 — Fine-Tune VLM *(Objective 2 only — needs GPU)*

> Skip if doing Objective 1 only. Requires GPU with ≥ 8 GB VRAM.

```bash
python vlm/finetune.py
```

**Output:** `vlm/vlm_checkpoints/final/` — LoRA adapter weights.

---

## Step 7 — Start the Backend

### Option A — Direct (Development)

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify: open `http://localhost:8000/health` — returns `{"status":"ok", "model_loaded": true, ...}`

### Option B — Docker

```bash
docker-compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend (nginx): `http://localhost:3000`

---

## Step 8 — Install the Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode** (toggle, top-right)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. Copy the **Extension ID** shown on the card — needed for Objectives 2 & 3

---

## Step 9 — Objective 1: Real-Time Detection

1. Open the 3D car model page in Chrome:
   `https://euphonious-concha-ab5c5d.netlify.app/`
2. Open the dashboard in a **second tab**:
   - Docker: `http://localhost:3000/index.html`
   - Manual: open `frontend/index.html` in Chrome
3. Click the **Car Detector** extension icon → **▶ Start Detection**
4. Switch to the 3D car tab and click any component button (e.g. "Front Left Door")
5. The dashboard badge updates **Open/Closed** in real time and highlights changed components.

---

## Step 10 — Objective 2: VLM Description

1. Ensure the backend is running (Step 7).
2. Open `frontend/vlm.html` in Chrome and paste your **Extension ID**.
3. Set any combination of doors/hood on the 3D car tab.
4. Click **📸 Capture Current 3D View & Describe**.

---

## Step 11 — Objective 3: Visual Grounding

1. Ensure the backend is running and Grounding DINO weights are placed at `weights/groundingdino_swint_ogc.pth`.
2. Open `frontend/grounding.html` in Chrome and paste your **Extension ID**.
3. Type a prompt such as `open front left door` and click **Generate Grounding**.

---

## Model Details

### Custom CNN (Multi-Scale PA-SCAN) — Objective 1

Built **100% from scratch** (random initialization, zero pretrained weights / zero transfer learning):

```
Full Browser Frame (1070 × 805 RGB)
  ↓  crop_car_roi (ROI_BOX_NORM = 0.17, 0.19, 0.83, 0.62 → 3.52× pixel density gain)
Cropped ROI (224 × 224 RGB) + CoordConv (X, Y ∈ [-1, 1]) → 5 × 224 × 224
  ↓
Stage 1..4 (ResidualConvBlock, stride=2): 5 → 32 → 64 → 128 → 256 ch  (F4: 256 × 14 × 14)
  ↓
Stage 5 (ResidualConvBlock, stride=2):    256 → 512 ch                (F5: 512 × 7 × 7)
  ↓
CBAM Attention: Squeeze-and-Excitation (Channel) + 3×3 Spatial Attention
  ├──► Branch 1 (Global Pose Context): AdaptiveAvgPool2d(1) → 512-d → Pose Head (8 angles)
  └──► Branch 2 (Multi-Scale Spatial Grid):
         Concat[F4 (14×14), Upsample(F5) (14×14)] → 1×1 Conv (64 ch)
         0.5 * AdaptiveAvgPool2d(6×6) + 0.5 * AdaptiveMaxPool2d(6×6) → 2304-d
  ↓
Pose-Conditioned FiLM Gate: s_gated = (1 + tanh(γ(θ))) ⊙ s_feat + β(θ)
  ↓
5 Disentangled Component Heads (ModuleList of 5 independent Linear(2304 → 96 → 1) MLPs)
  ↓
5 Sigmoid Outputs (FL door · FR door · RL door · RR door · Hood)
```

- **Backbone Efficiency:** Strided $3\times 3$ residual blocks reduce backbone compute from $3.28\text{ GMACs}$ to **$0.855\text{ GMACs}$** ($3.84\times$ fewer MACs).
- **Temporal Change Detector:** 3-frame rolling probability average with per-label hysteresis bands around the optimal F1 thresholds loaded from `checkpoints/thresholds.json`.
