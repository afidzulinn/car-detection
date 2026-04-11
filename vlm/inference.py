"""
Standalone SmolVLM-500M-Instruct inference script for testing.
Usage: python vlm/inference.py --image path/to/image.jpg
"""
import argparse
import torch
from PIL import Image
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _ModelClass = AutoModelForVision2Seq
except ImportError:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _ModelClass = AutoModelForImageTextToText

from peft import PeftModel

BASE_MODEL   = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_WEIGHTS = Path("vlm/vlm_checkpoints/final")


def load_model(lora_path: Path = LORA_WEIGHTS):
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = _ModelClass.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_path.exists():
        model = PeftModel.from_pretrained(base, str(lora_path))
        print(f"Loaded fine-tuned model from {lora_path}")
    else:
        model = base
        print("WARNING: LoRA weights not found. Using base SmolVLM.")

    model.eval()
    return model, processor


def describe(image_path: str, model=None, processor=None) -> str:
    if model is None:
        model, processor = load_model()

    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the open/closed status of each car component."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to car image")
    parser.add_argument("--lora",  default=str(LORA_WEIGHTS), help="LoRA weights path")
    args = parser.parse_args()

    model, processor = load_model(Path(args.lora))
    result = describe(args.image, model, processor)
    print("\n=== VLM Description ===")
    print(result)
