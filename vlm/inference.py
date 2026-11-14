"""
Standalone VLM inference script for testing.
Usage: python vlm/inference.py --image path/to/image.jpg
"""
import argparse
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

BASE_MODEL   = "google/paligemma-3b-pt-224"
LORA_WEIGHTS = Path("vlm/vlm_checkpoints/final")


def load_model(lora_path: Path = LORA_WEIGHTS):
    processor = AutoProcessor.from_pretrained(
        str(lora_path) if lora_path.exists() else BASE_MODEL
    )

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path.exists():
        model = PeftModel.from_pretrained(base, str(lora_path))
        print(f"Loaded fine-tuned model from {lora_path}")
    else:
        model = base
        print("WARNING: LoRA weights not found. Using base PaliGemma.")

    model.eval()
    return model, processor


def describe(image_path: str, model=None, processor=None) -> str:
    if model is None:
        model, processor = load_model()

    img    = Image.open(image_path).convert("RGB")
    prompt = "Describe the open/closed status of each car component."

    inputs = processor(
        text=f"<image>{prompt}",
        images=img,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    decoded = processor.decode(output[0], skip_special_tokens=True)
    if prompt in decoded:
        decoded = decoded.split(prompt)[-1].strip()

    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to car image")
    parser.add_argument("--lora",  default=str(LORA_WEIGHTS), help="LoRA weights path")
    args = parser.parse_args()

    model, processor = load_model(Path(args.lora))
    result = describe(args.image, model, processor)
    print("\n=== VLM Description ===")
    print(result)
