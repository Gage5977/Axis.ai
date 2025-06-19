#!/usr/bin/env python3
"""
Inference script for objective Qwen model.
Loads trained LoRA adapter and generates responses.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with objective Qwen model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-7B",
                       help="Base model name")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt for inference")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str, base_model: str):
    """Load the trained model and tokenizer."""
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading base model {base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights into base model
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512, 
                     temperature: float = 0.1, top_p: float = 0.9):
    """Generate response using the model."""
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove input prompt)
    if prompt in response:
        response = response.split(prompt, 1)[1].strip()
    
    return response

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    # Generate response
    print(f"Prompt: {args.prompt}")
    print("Generating response...")
    
    response = generate_response(
        model, 
        tokenizer, 
        args.prompt, 
        args.max_length, 
        args.temperature, 
        args.top_p
    )
    
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    main()