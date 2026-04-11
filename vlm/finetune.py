"""
Fine-tune SmolVLM-500M-Instruct using QLoRA for car component description.
Requirements: transformers>=4.45.0, peft, bitsandbytes, trl, datasets, accelerate

Run: python vlm/finetune.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# Ampere TF32 — lossless speedup for fp32 ops remaining in mixed-precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import json
from pathlib import Path
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer

try:
    from transformers import AutoModelForVision2Seq
    _ModelClass = AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText
    _ModelClass = AutoModelForImageTextToText

# ── Config ─────────────────────────────────────────────────────
MODEL_ID  = "HuggingFaceTB/SmolVLM-500M-Instruct"
# VLM sudah pretrained — raw data cukup, augmentasi hanya perlu untuk CNN from-scratch
DATA_DIR  = Path("dataset/raw")
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
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = _ModelClass.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # Freeze vision encoder — LoRA only touches the language model
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = load_vlm_dataset()
    print(f"Dataset size: {len(dataset)} samples")

    def collate_fn(batch):
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        full_texts = []
        prompt_texts = []
        for item in batch:
            # Full conversation (user + assistant) — used as input_ids
            full_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}],
                },
            ]
            full_texts.append(
                processor.apply_chat_template(full_msgs, add_generation_prompt=False)
            )
            # Prompt-only (user turn + generation prefix) — to measure what to mask
            prompt_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                },
            ]
            prompt_texts.append(
                processor.apply_chat_template(prompt_msgs, add_generation_prompt=True)
            )

        enc = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Mask prompt + padding in labels — only train on assistant answer tokens
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for i, pt in enumerate(prompt_texts):
            prompt_len = len(processor.tokenizer(pt, add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = -100
        enc["labels"] = labels
        return enc

    args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(SAVE_DIR / "final")
    processor.save_pretrained(SAVE_DIR / "final")
    print(f"Model saved to {SAVE_DIR / 'final'}")


if __name__ == "__main__":
    finetune()
