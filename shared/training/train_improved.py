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
    parser = argparse.ArgumentParser(description="Improved supervised fine-tuning of Qwen with LoRA adapters for tool-aware behavior.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B", help="HF model hub path or local path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training dataset (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-agent-improved", help="Where to store the fine-tuned model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)  # Increased for better gradients
    parser.add_argument("--num_train_epochs", type=int, default=5)  # Increased for better learning
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # Reduced for stability
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU training (slow)")
    parser.add_argument("--lora_r", type=int, default=16)  # Increased for better adaptation
    parser.add_argument("--lora_alpha", type=int, default=64)  # Increased for stronger signal
    parser.add_argument("--lora_dropout", type=float, default=0.1)  # Slightly increased for regularization
    parser.add_argument("--max_length", type=int, default=4096)  # Increased for complex workflows
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=50)  # Added warmup
    parser.add_argument("--weight_decay", type=float, default=0.01)  # Added weight decay
    return parser.parse_args()


@dataclass
class ImprovedSFTDataset:
    file_path: str
    tokenizer: AutoTokenizer
    max_length: int = 4096

    def __post_init__(self):
        logger.info(f"Loading dataset from {self.file_path}")
        raw_ds = load_dataset("json", data_files=self.file_path, split="train")
        self.data = raw_ds.map(self._tokenize_function, batched=False, remove_columns=raw_ds.column_names)
        logger.info(f"Loaded {len(self.data)} samples")

    def _tokenize_function(self, example: Dict[str, str]):
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        
        # Add proper conversation formatting
        text = f"{prompt}{response}<|im_end|>"
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding is better
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        
        # Create labels with proper masking (only train on response)
        labels = input_ids.clone()
        
        # Find where the response starts (after the last <user> tag)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_tokens) < len(input_ids):
            # Mask the prompt tokens so we only train on the response
            labels[:len(prompt_tokens)] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"][0],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer with proper configuration
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Changed to right padding for better training

    # Device setup
    device_map = "cpu" if args.cpu_only else "auto"
    load_8bit = not args.cpu_only

    # Load base model with better configuration
    logger.info("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_8bit=load_8bit,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.cpu_only else torch.float32,
    )

    # Resize token embeddings if tokenizer was expanded
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA with improved configuration
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            # Targeting more modules for better tool awareness
            "c_attn", "c_proj", "w1", "w2", "c_fc",
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # Print parameter count
    logger.info("LoRA adapters applied")

    # Dataset
    train_dataset = ImprovedSFTDataset(args.train_file, tokenizer, args.max_length)

    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # For better GPU utilization
    )

    # Improved training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=not args.cpu_only,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        dataloader_drop_last=True,
        gradient_checkpointing=True,  # Save memory
        remove_unused_columns=False,
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        # Better optimization
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting improved training")
    trainer.train()
    logger.info("Training complete")

    # Save final adapter & tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model & tokenizer saved to {args.output_dir}")

    # Save training metrics
    with open(f"{args.output_dir}/training_metrics.json", "w") as f:
        json.dump({
            "total_parameters": model.num_parameters(),
            "trainable_parameters": model.num_parameters(only_trainable=True),
            "training_args": vars(args),
        }, f, indent=2)


if __name__ == "__main__":
    main() 