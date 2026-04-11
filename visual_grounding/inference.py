"""
Visual Grounding using Grounding DINO (open-source, zero-shot).
Takes text prompt + image -> bounding boxes around matching objects.

Setup:
  pip install groundingdino-py
  mkdir -p weights
  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth

Usage:
  python visual_grounding/inference.py --image img.jpg --prompt "open front left door"
"""
import argparse
import base64
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

CONFIG_PATH  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

# Car-component-aware prompt expansion
PROMPT_ALIASES = {
    "front left door":  "front left door . door",
    "front right door": "front right door . door",
    "rear left door":   "rear left door . door",
    "rear right door":  "rear right door . door",
    "hood":             "hood . bonnet . engine cover",
    "open doors":       "open door . front door . rear door",
    "all open":         "open door . open hood",
}


def load_grounding_model():
    from groundingdino.util.inference import load_model
    return load_model(CONFIG_PATH, WEIGHTS_PATH)


def expand_prompt(prompt: str) -> str:
    """Map common user prompts to Grounding DINO-friendly phrases."""
    lower = prompt.lower().strip()
    for key, expanded in PROMPT_ALIASES.items():
        if key in lower:
            return expanded
    return prompt.strip().rstrip(".") + " ."


def run_grounding(
    image_input,             # str path or PIL Image
    prompt: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    save_path: str = None,
):
    """
    Returns:
        annotated_image : np.ndarray (BGR)
        boxes           : list of [cx, cy, w, h] normalized
        phrases         : list of matched phrase strings
        scores          : list of confidence floats
    """
    from groundingdino.util.inference import predict, annotate

    # Load image
    if isinstance(image_input, str):
        image_pil = Image.open(image_input).convert("RGB")
    else:
        image_pil = image_input.convert("RGB")

    image_np = np.array(image_pil)

    # Prepare tensor
    image_tensor = (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .float() / 255.0
    )

    expanded = expand_prompt(prompt)
    print(f"Prompt expanded: '{prompt}' -> '{expanded}'")

    model = load_grounding_model()
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=expanded,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    scores = logits.tolist()

    # Annotate
    annotated = annotate(
        image_source=image_np,
        boxes=boxes,
        logits=logits,
        phrases=phrases,
    )

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image to {save_path}")

    return annotated, boxes.tolist(), phrases, scores


def image_to_base64(image_np: np.ndarray) -> str:
    """Convert annotated numpy image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Text grounding prompt")
    parser.add_argument("--output", default="grounding_result.jpg", help="Output image path")
    parser.add_argument("--box-threshold",  type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    args = parser.parse_args()

    annotated, boxes, phrases, scores = run_grounding(
        args.image, args.prompt,
        args.box_threshold, args.text_threshold,
        save_path=args.output
    )

    print(f"\n=== Grounding Results ===")
    print(f"Detected {len(boxes)} object(s):")
    for phrase, score, box in zip(phrases, scores, boxes):
        print(f"  [{phrase}] confidence={score:.3f} box={[round(b,3) for b in box]}")
