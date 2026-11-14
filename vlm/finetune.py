"""
Fine-tune PaliGemma-3B using QLoRA for car component description.
Requirements: transformers, peft, bitsandbytes, trl, datasets, accelerate

Run: python vlm/finetune.py
"""
import torch
import json
from pathlib import Path
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── Config ─────────────────────────────────────────────────────
MODEL_ID  = "google/paligemma-3b-pt-224"
DATA_DIR  = Path("dataset/augmented")
SAVE_DIR  = Path("vlm/vlm_checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAME_MAP = {
    "front_left_door":  "front left door",
    "front_right_door": "front right door",
    "rear_left_door":   "rear left door",
    "rear_right_door":  "rear right door",
    "hood":             "hood",
}

# ── Dataset ────────────────────────────────────────────────────
def build_description(labels: dict) -> str:
    """Convert label dict to natural language description."""
    open_comps   = [LABEL_NAME_MAP[k] for k, v in labels.items() if v == 1]
    closed_comps = [LABEL_NAME_MAP[k] for k, v in labels.items() if v == 0]

    desc = "The car currently has: "
    if open_comps:
        desc += "open: " + ", ".join(open_comps) + ". "
    if closed_comps:
        desc += "closed: " + ", ".join(closed_comps) + "."
    return desc.strip()


def load_vlm_dataset() -> Dataset:
    with open(DATA_DIR / "annotations.json") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        description = build_description(item["labels"])
        records.append({
            "image_path": str(DATA_DIR / item["file"]),
            "prompt":     "Describe the open/closed status of each car component.",
            "answer":     description,
        })

    return Dataset.from_list(records)


# ── Training ───────────────────────────────────────────────────
def finetune():
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_vlm_dataset()
    print(f"Dataset size: {len(dataset)} samples")

    def collate_fn(batch):
        images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
        texts  = [
            f"<image>{q}\n{a}"
            for q, a in zip(batch["prompt"], batch["answer"])
        ]
        return processor(
            text=texts, images=images,
            padding=True, return_tensors="pt"
        )

    args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(SAVE_DIR / "final")
    processor.save_pretrained(SAVE_DIR / "final")
    print(f"Model saved to {SAVE_DIR / 'final'}")


if __name__ == "__main__":
    finetune()
