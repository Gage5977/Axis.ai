import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Tool-enabled inference with objective Mistral 7B model")
parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model path or HF hub id")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) 