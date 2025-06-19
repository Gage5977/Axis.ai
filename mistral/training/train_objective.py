import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Objective supervised fine-tuning of Mistral 7B with LoRA adapters.")
parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="HF model hub path or local path")
parser.add_argument("--output_dir", type=str, default="outputs/mistral-objective", help="Where to store the fine-tuned model")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True) 