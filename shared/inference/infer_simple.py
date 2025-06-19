import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Qwen agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.8B", help="Base model path")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to generate a response for")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side="left")
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu", 
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    
    print("Generating response...")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            do_sample=True, 
            temperature=args.temperature, 
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)


if __name__ == "__main__":
    main() 