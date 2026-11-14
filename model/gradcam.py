"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Multi-Scale PA-SCAN.
Generates publication-quality saliency heatmaps for multi-label car component detection.
Uses pure NumPy & PIL for zero-dependency, ultra-fast visualization.

Usage:
    python model/gradcam.py --num-samples 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from architecture import CarComponentCNN, LABELS, crop_car_roi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = Path(__file__).resolve().parents[1]


def jet_colormap(val: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of the Jet colormap.
    val: float array in [0, 1] of shape (H, W).
    Returns: uint8 array of shape (H, W, 3).
    """
    v = np.clip(val, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class ModelGradCAM:
    """
    Grad-CAM extractor targeting the final convolutional stage (stage5) before pooling.
    Hooks forward activation and backward gradients to compute importance weights.
    """
    def __init__(self, model: CarComponentCNN):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.stage5  # Final ResidualConvBlock (7x7 @ 512ch)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a normalized 2D Grad-CAM heatmap in [0, 1] for class_idx.
        input_tensor: shape (1, 3, 224, 224)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of forward activation maps
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(pooled_grads):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap_on_image(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """
    Resize heatmap to image dimensions and overlay with Jet colormap.
    """
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        img_pil.size, resample=Image.Resampling.BILINEAR
    )
    heatmap_norm = np.array(heatmap_img, dtype=np.float32) / 255.0
    colored_heatmap = jet_colormap(heatmap_norm)

    img_np = np.array(img_pil, dtype=np.float32)
    overlay = (alpha * colored_heatmap + (1.0 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


def render_panel(
    roi_img: Image.Image,
    overlays: list[tuple[str, str, float, Image.Image]],
    angle_deg: int,
    elevation: str,
) -> Image.Image:
    """
    Stitch 1 Input panel + 5 Grad-CAM panels horizontally into a composite figure.
    """
    thumb_w, thumb_h = 224, 224
    header_h = 44
    n_cols = 1 + len(overlays)
    total_w = n_cols * thumb_w + (n_cols + 1) * 8
    total_h = thumb_h + header_h + 16

    composite = Image.new("RGB", (total_w, total_h), (245, 245, 248))
    draw = ImageDraw.Draw(composite)

    # Panel 0: Input ROI
    x_offset = 8
    composite.paste(roi_img.resize((thumb_w, thumb_h)), (x_offset, header_h))
    draw.rectangle([x_offset, header_h, x_offset + thumb_w, header_h + thumb_h], outline=(180, 180, 190), width=1)
    draw.text((x_offset + 6, 8), f"Input ROI (deg={angle_deg})", fill=(20, 20, 30))
    draw.text((x_offset + 6, 24), f"Elevation: {elevation}", fill=(90, 90, 100))

    # Panels 1..5: Grad-CAM
    for label_name, gt_status, prob, cam_overlay in overlays:
        x_offset += thumb_w + 8
        composite.paste(cam_overlay, (x_offset, header_h))
        is_match = (gt_status == "OPEN" and prob >= 0.5) or (gt_status == "CLOSED" and prob < 0.5)
        border_col = (46, 125, 50) if is_match else (211, 47, 47)
        draw.rectangle([x_offset, header_h, x_offset + thumb_w, header_h + thumb_h], outline=border_col, width=2)

        short_title = label_name.replace("_", " ").title()
        draw.text((x_offset + 6, 8), short_title, fill=(20, 20, 30))
        status_text = f"GT: {gt_status} | P: {prob:.2f}"
        draw.text((x_offset + 6, 24), status_text, fill=border_col)

    return composite


def run_gradcam_inspection(checkpoint_path: str, output_dir: str, num_samples: int = 6):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    model = CarComponentCNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    gradcam = ModelGradCAM(model)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    raw_dir = ROOT_DIR / "dataset" / "raw"
    with open(raw_dir / "annotations.json", encoding="utf-8") as f:
        annotations = json.load(f)

    # Pick diverse test samples with open components across different angles
    open_samples = [item for item in annotations if any(item["labels"].values())]
    if not open_samples:
        open_samples = annotations

    step = max(1, len(open_samples) // num_samples)
    selected = open_samples[::step][:num_samples]

    for idx, item in enumerate(selected):
        img_file = raw_dir / Path(item["file"]).name
        if not img_file.exists():
            continue

        raw_img = Image.open(img_file).convert("RGB")
        roi_img = crop_car_roi(raw_img)
        input_tensor = preprocess(roi_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        roi_resized = roi_img.resize((224, 224))
        overlays = []
        for c_idx, label in enumerate(LABELS):
            heatmap = gradcam.generate_heatmap(input_tensor, c_idx)
            overlay = overlay_heatmap_on_image(roi_resized, heatmap, alpha=0.55)
            gt_status = "OPEN" if item["labels"][label] == 1 else "CLOSED"
            pred_prob = float(probs[c_idx])
            overlays.append((label, gt_status, pred_prob, overlay))

        composite_img = render_panel(
            roi_img, overlays, item.get("angle", 0), item.get("elevation", "mid")
        )
        save_file = out_path / f"gradcam_sample_{idx:02d}_{item.get('angle', 0)}deg.png"
        composite_img.save(save_file, quality=95)
        print(f"[{idx+1}/{len(selected)}] Saved Grad-CAM figure: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM generator for PA-SCAN model.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "docs" / "figures" / "gradcam"))
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    run_gradcam_inspection(args.checkpoint, args.output_dir, args.num_samples)
