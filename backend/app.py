import io, base64, torch, asyncio, json
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
import numpy as np
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
sys.path.insert(0, str(MODEL_DIR))
from architecture import CarComponentCNN, LABELS
from change_detector import ChangeDetectorWrapper

# ── Setup ────────────────────────────────────────────────────
app = FastAPI(title="Car Component Detector API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CNN model
cnn_model = CarComponentCNN().to(DEVICE)
ckpt = ROOT_DIR / "checkpoints" / "best_model.pt"
if ckpt.exists():
    cnn_model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"Model loaded from {ckpt}")
else:
    print("WARNING: No checkpoint found. Model weights are random.")
cnn_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global state broker
latest_prediction: dict = {label: {"status": "unknown", "confidence": 0.0} for label in LABELS}
previous_prediction: dict = {}
dashboard_clients: list[WebSocket] = []
change_detector = ChangeDetectorWrapper(cnn_model)

# ── Utilities ────────────────────────────────────────────────
def decode_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    return preprocess(img).unsqueeze(0).to(DEVICE)

def run_cnn_inference(img: Image.Image, threshold=0.5) -> dict:
    tensor = image_to_tensor(img)
    with torch.no_grad():
        logits = cnn_model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    preds = (probs > threshold).astype(int)
    return {
        LABELS[i]: {"status": "open" if preds[i] else "closed", "confidence": float(probs[i])}
        for i in range(len(LABELS))
    }

def run_smoothed_inference(img: Image.Image) -> tuple[dict, list]:
    tensor = image_to_tensor(img)
    confirmed_state, changed_idx, avg_probs = change_detector.update(tensor)
    prediction = {
        LABELS[i]: {
            "status": "open" if confirmed_state[i] else "closed",
            "confidence": float(avg_probs[i]),
        }
        for i in range(len(LABELS))
    }
    return prediction, [LABELS[i] for i in changed_idx]

def detect_changes(old: dict, new: dict) -> list:
    changed = []
    for label in LABELS:
        if old and old.get(label, {}).get("status") != new[label]["status"]:
            changed.append(label)
    return changed

async def broadcast_to_dashboards(payload: dict):
    disconnected = []
    for client in dashboard_clients:
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.append(client)
    for c in disconnected:
        dashboard_clients.remove(c)


# ── Objective 1: Extension WebSocket ─────────────────────────
@app.websocket("/ws/extension")
async def extension_ws(ws: WebSocket):
    """Receives frames from Chrome Extension."""
    global latest_prediction, previous_prediction
    await ws.accept()
    change_detector.reset()
    print("Extension connected")
    try:
        while True:
            b64_frame = await ws.receive_text()
            img = decode_image(b64_frame)
            new_pred, changed = run_smoothed_inference(img)
            previous_prediction = {**latest_prediction}
            latest_prediction = new_pred
            payload = {
                "predictions": new_pred,
                "changed_components": changed,
                "timestamp": asyncio.get_event_loop().time()
            }
            await broadcast_to_dashboards(payload)
    except WebSocketDisconnect:
        change_detector.reset()
        print("Extension disconnected")


# ── Objective 1: Dashboard WebSocket ─────────────────────────
@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    """Sends predictions to dashboard frontend."""
    await ws.accept()
    dashboard_clients.append(ws)
    await ws.send_json({"predictions": latest_prediction, "changed_components": [], "timestamp": 0})
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"ping": True})
    except (WebSocketDisconnect, Exception):
        if ws in dashboard_clients:
            dashboard_clients.remove(ws)


# ── Objective 2: VLM Endpoint ─────────────────────────────────
class ImageRequest(BaseModel):
    image_b64: str

@app.post("/api/vlm/describe")
async def vlm_describe(req: ImageRequest):
    try:
        img = decode_image(req.image_b64)
        description = run_vlm_inference(img)
        return {"description": description, "success": True}
    except Exception as e:
        return {"description": f"Error: {str(e)}", "success": False}

def run_vlm_inference(img: Image.Image) -> str:
    try:
        from vlm_module import describe_car_state
        return describe_car_state(img)
    except ImportError:
        pred = run_cnn_inference(img)
        name_map = {
            "front_left_door":  "front left door",
            "front_right_door": "front right door",
            "rear_left_door":   "rear left door",
            "rear_right_door":  "rear right door",
            "hood":             "hood"
        }
        parts = [f"{name_map[label]} is {result['status']}" for label, result in pred.items()]
        return "The car has the following states: " + ", ".join(parts) + "."


# ── Objective 3: Visual Grounding Endpoint ────────────────────
class GroundingRequest(BaseModel):
    image_b64: str
    prompt: str

@app.post("/api/grounding/locate")
async def grounding_locate(req: GroundingRequest):
    try:
        img = decode_image(req.image_b64)
        annotated_b64, boxes, labels = run_grounding_inference(img, req.prompt)
        return {"annotated_image": annotated_b64, "boxes": boxes, "detected_labels": labels, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

_grounding_model = None

def get_grounding_model():
    global _grounding_model
    if _grounding_model is None:
        from groundingdino.util.inference import load_model

        config_path = ROOT_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        if not config_path.exists():
            try:
                import groundingdino  # type: ignore
                pkg_dir = Path(groundingdino.__file__).resolve().parent
                candidate = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
                if candidate.exists():
                    config_path = candidate
            except Exception:
                config_path = Path("groundingdino/config/GroundingDINO_SwinT_OGC.py")

        weights_path = ROOT_DIR / "weights" / "groundingdino_swint_ogc.pth"
        if not weights_path.exists():
            weights_path = Path("weights/groundingdino_swint_ogc.pth")

        _grounding_model = load_model(
            str(config_path),
            str(weights_path)
        )
    return _grounding_model

def run_grounding_inference(img: Image.Image, prompt: str):
    import cv2
    from groundingdino.util.inference import predict, annotate

    img_np = np.array(img)
    prompt_clean = prompt.strip().rstrip(".") + " ."
    model = get_grounding_model()

    boxes, logits, phrases = predict(
        model=model,
        image=torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0,
        caption=prompt_clean,
        box_threshold=0.35,
        text_threshold=0.25
    )

    annotated = annotate(image_source=img_np, boxes=boxes, logits=logits, phrases=phrases)
    _, buffer = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64, boxes.tolist(), phrases


# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "dashboard_clients": len(dashboard_clients),
        "model_loaded": ckpt.exists()
    }
