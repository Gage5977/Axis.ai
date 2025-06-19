import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Inference with objective Mistral 7B model")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) 