import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Simple inference with Mistral 7B model")
parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side="left")
base_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True) 