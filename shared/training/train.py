import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of Qwen with LoRA adapters.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B", help="HF model hub path or local path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training dataset (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-agent", help="Where to store the fine-tuned model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU training (slow)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    return parser.parse_args()


@dataclass
class SFTDataset:
    file_path: str
    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __post_init__(self):
        logger.info(f"Loading dataset from {self.file_path}")
        raw_ds = load_dataset("json", data_files=self.file_path, split="train")
        self.data = raw_ds.map(self._tokenize_function, batched=False, remove_columns=raw_ds.column_names)
        logger.info(f"Loaded {len(self.data)} samples")

    def _tokenize_function(self, example: Dict[str, str]):
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        text = prompt + response
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # standard causal LM loss
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|extra_pad|>"})
    tokenizer.padding_side = "left"  # Qwen uses left padding

    # Device setup
    device_map = "cpu" if args.cpu_only else "auto"
    load_8bit = not args.cpu_only  # Use 8-bit quantization when on GPU

    # Load base model
    logger.info("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_8bit=load_8bit,
        trust_remote_code=True,
    )

    # Resize token embeddings if tokenizer was expanded
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "c_attn", "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("LoRA adapters applied")

    # Dataset
    train_dataset = SFTDataset(args.train_file, tokenizer)

    # Data collator (pad to longest in batch)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not args.cpu_only,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")

    # Save final adapter & tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model & tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main() 