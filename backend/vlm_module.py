"""
Fine-tuned PaliGemma inference module.
Loaded lazily to avoid OOM when only CNN is needed.
"""
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

BASE_MODEL   = "google/paligemma-3b-pt-224"
LORA_WEIGHTS = Path(__file__).resolve().parents[1] / "vlm" / "vlm_checkpoints" / "final"

_processor = None
_vlm_model = None

def _load_model():
    global _processor, _vlm_model
    if _vlm_model is not None:
        return

    _processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if LORA_WEIGHTS.exists():
        _vlm_model = PeftModel.from_pretrained(base, str(LORA_WEIGHTS))
        print("Loaded fine-tuned VLM with LoRA weights")
    else:
        _vlm_model = base
        print("WARNING: LoRA weights not found -- using base model")

    _vlm_model.eval()


def describe_car_state(img: Image.Image) -> str:
    _load_model()

    prompt = "Describe the open/closed status of each car component."
    inputs = _processor(
        text=f"<image>{prompt}",
        images=img,
        return_tensors="pt"
    ).to(_vlm_model.device)

    with torch.no_grad():
        output = _vlm_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=1.0
        )

    decoded = _processor.decode(output[0], skip_special_tokens=True)
    if prompt in decoded:
        decoded = decoded.split(prompt)[-1].strip()

    return decoded
