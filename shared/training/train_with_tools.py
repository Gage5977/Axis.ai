#!/usr/bin/env python3
"""
Enhanced training script that includes tool usage examples.
"""

import argparse
import json
import logging
import os
from datasets import Dataset

from train_objective import (
    parse_args as base_parse_args,
    ObjectiveDataset,
    main as base_main
)

def parse_args():
    parser = argparse.ArgumentParser(description="Tool-enhanced training for Qwen")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B")
    parser.add_argument("--objective_file", type=str, default="data/objective_training.jsonl")
    parser.add_argument("--tool_file", type=str, default="data/tool_training.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen-tools")
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

def combine_datasets(objective_file: str, tool_file: str, output_file: str):
    """Combine objective and tool training datasets."""
    combined_data = []
    
    # Load objective training data
    if os.path.exists(objective_file):
        with open(objective_file, 'r') as f:
            for line in f:
                combined_data.append(json.loads(line.strip()))
    
    # Load tool training data
    if os.path.exists(tool_file):
        with open(tool_file, 'r') as f:
            for line in f:
                combined_data.append(json.loads(line.strip()))
    
    # Write combined dataset
    with open(output_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Combined {len(combined_data)} training examples")
    return output_file

def main():
    args = parse_args()
    
    # Combine datasets
    combined_file = "data/combined_training.jsonl"
    combine_datasets(args.objective_file, args.tool_file, combined_file)
    
    # Update args to use combined file
    args.train_file = combined_file
    
    # Call base training function with modified args
    import sys
    sys.argv = [
        "train_objective.py",
        "--model_name_or_path", args.model_name_or_path,
        "--train_file", args.train_file,
        "--output_dir", args.output_dir,
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--max_length", str(args.max_length),
        "--logging_steps", str(args.logging_steps),
        "--save_steps", str(args.save_steps)
    ]
    
    if args.cpu_only:
        sys.argv.append("--cpu_only")
    
    # Import and run base training
    from train_objective import main as train_main
    train_main()

if __name__ == "__main__":
    main()