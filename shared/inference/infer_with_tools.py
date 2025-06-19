#!/usr/bin/env python3
"""
Tool-enabled inference for objective Qwen model.
Allows model to call external tools and functions.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tools.tool_interface import ToolManager, execute_tool_calls

def parse_args():
    parser = argparse.ArgumentParser(description="Tool-enabled inference with objective Qwen model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-7B",
                       help="Base model name")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt for inference")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
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
    model = model.merge_and_unload()
    
    return model, tokenizer

def create_tool_enhanced_prompt(user_prompt: str, tool_manager: ToolManager) -> str:
    """Create prompt that includes tool information."""
    tools_info = tool_manager.format_tools_for_prompt()
    
    system_prompt = f"""<system>You are an objective AI assistant with access to tools. Provide accurate, factual responses without emotional language or bias. When you need to perform actions or access external information, use the available tools.

{tools_info}

Always respond objectively and factually. Use tools when necessary to complete tasks or gather information.</system>"""
    
    return f"{system_prompt}<user>{user_prompt}</user><assistant>"

def generate_with_tools(model, tokenizer, prompt: str, tool_manager: ToolManager, 
                       max_length: int = 1024, temperature: float = 0.1):
    """Generate response with tool support."""
    model.eval()
    
    # Create tool-enhanced prompt
    enhanced_prompt = create_tool_enhanced_prompt(prompt, tool_manager)
    
    # Tokenize
    inputs = tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated part
    if enhanced_prompt in full_response:
        response = full_response.split(enhanced_prompt, 1)[1].strip()
    else:
        response = full_response
    
    # Execute any tool calls in the response
    final_response = execute_tool_calls(response, tool_manager)
    
    return final_response

def interactive_mode(model, tokenizer, tool_manager: ToolManager, args):
    """Run in interactive mode."""
    print("Interactive mode - Type 'quit' to exit")
    print("Available tools:", [tool['name'] for tool in tool_manager.get_available_tools()])
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue
            
            print("Assistant:", end=" ")
            response = generate_with_tools(
                model, tokenizer, user_input, tool_manager,
                args.max_length, args.temperature
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    # Initialize tool manager
    tool_manager = ToolManager()
    print(f"Loaded {len(tool_manager.tools)} tools")
    
    if args.interactive:
        interactive_mode(model, tokenizer, tool_manager, args)
    else:
        # Single inference
        print(f"Prompt: {args.prompt}")
        print("Generating response...")
        
        response = generate_with_tools(
            model, tokenizer, args.prompt, tool_manager,
            args.max_length, args.temperature
        )
        
        print(f"Response: {response}")

if __name__ == "__main__":
    main()