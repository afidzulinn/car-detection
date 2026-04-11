"""
SmolVLM-500M-Instruct inference module.
Loaded lazily to avoid OOM when only CNN is needed.
"""
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
LORA_WEIGHTS = Path(__file__).resolve().parents[1] / "vlm" / "vlm_checkpoints" / "final"

_processor = None
_vlm_model = None


def _load_model():
    global _processor, _vlm_model
    if _vlm_model is not None:
        return

    _processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = _ModelClass.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if LORA_WEIGHTS.exists():
        _vlm_model = PeftModel.from_pretrained(base, str(LORA_WEIGHTS))
        print("Loaded fine-tuned VLM with LoRA weights")
    else:
        _vlm_model = base
        print("WARNING: LoRA weights not found -- using base SmolVLM")

    _vlm_model.eval()


def describe_car_state(img: Image.Image) -> str:
    _load_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the open/closed status of each car component."},
            ],
        }
    ]
    prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(text=prompt, images=[img], return_tensors="pt")
    inputs = {k: v.to(_vlm_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = _vlm_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    return _processor.decode(new_tokens, skip_special_tokens=True).strip()
