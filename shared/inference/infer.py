import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Qwen agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned adapter or merged model")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to generate a response for")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main() 