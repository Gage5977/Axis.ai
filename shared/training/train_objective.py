import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Objective supervised fine-tuning of Qwen with LoRA adapters.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B")
    parser.add_argument("--train_file", type=str, default="data/objective_training.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-objective")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=2048)
    return parser.parse_args()


@dataclass
class ObjectiveDataset:
    file_path: str
    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __post_init__(self):
        logger.info(f"Loading dataset from {self.file_path}")
        raw_ds = load_dataset("json", data_files=self.file_path, split="train")
        self.data = raw_ds.map(self._tokenize_function, batched=False, remove_columns=raw_ds.column_names)
        logger.info(f"Loaded {len(self.data)} training samples")

    def _tokenize_function(self, example: Dict[str, str]):
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        
        # Format as instruction-following conversation
        text = f"{prompt}\n{response}"
        
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
            "labels": input_ids.clone(),
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
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    
    tokenizer.padding_side = "left"
    
    # Device configuration
    device_map = "cpu" if args.cpu_only else "auto"
    load_8bit = not args.cpu_only
    
    # Load base model
    logger.info("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_8bit=load_8bit,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.cpu_only else torch.float32,
    )
    
    # Resize embeddings if tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "c_attn", "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("LoRA configuration applied")
    
    # Prepare dataset
    train_dataset = ObjectiveDataset(args.train_file, tokenizer, args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Training configuration
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
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Execute training
    logger.info("Starting objective training")
    trainer.train()
    logger.info("Training completed")
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metadata
    metadata = {
        "model_name": args.model_name_or_path,
        "training_file": args.train_file,
        "num_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "training_samples": len(train_dataset)
    }
    
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model and metadata saved to {args.output_dir}")


if __name__ == "__main__":
    main()