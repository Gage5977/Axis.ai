#!/usr/bin/env python3
"""
Test Ollama model with tool integration.
Tests why current models can't use tools and demonstrates proper tool usage.
"""

import requests
import json
from tools.tool_interface import ToolManager, execute_tool_calls

def test_ollama_direct():
    """Test direct Ollama API call."""
    print("Testing direct Ollama API...")
    
    url = "http://localhost:11434/api/generate"
    
    # Test with simple prompt
    simple_prompt = "What is 2 + 2?"
    payload = {
        "model": "qwen3:14b",
        "prompt": simple_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        print(f"Simple response: {data.get('response', 'No response')[:200]}...")
    except Exception as e:
        print(f"Direct API error: {e}")

def test_ollama_with_tool_prompt():
    """Test Ollama with tool-aware prompt."""
    print("\nTesting Ollama with tool prompt...")
    
    tm = ToolManager()
    tools_info = tm.format_tools_for_prompt()
    
    tool_prompt = f"""You are an AI assistant with access to tools. Use tools when needed.

{tools_info}

User: List files in the current directory.
Assistant:"""
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3:14b",
        "prompt": tool_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    try:
        print("Sending tool-aware prompt to Ollama...")
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        model_response = data.get('response', '')
        
        print(f"Model response: {model_response}")
        
        # Check if model attempted to use tools
        if "TOOL_CALL:" in model_response:
            print("✓ Model attempted to use tools!")
            final_response = execute_tool_calls(model_response, tm)
            print(f"Final response with tool execution:\n{final_response}")
        else:
            print("✗ Model did not use tools - needs training")
            
    except Exception as e:
        print(f"Tool prompt error: {e}")

def test_finance_assistant():
    """Test the finance-assistant model which might have some tool training."""
    print("\nTesting finance-assistant model...")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "finance-assistant:latest", 
        "prompt": "List the files in the current directory using available tools.",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        print(f"Finance assistant response: {data.get('response', 'No response')[:300]}...")
    except Exception as e:
        print(f"Finance assistant error: {e}")

def demonstrate_working_tools():
    """Demonstrate that our tool system works."""
    print("\nDemonstrating working tool system...")
    
    tm = ToolManager()
    
    # Simulate what a trained model should output
    simulated_response = 'TOOL_CALL: {"tool": "bash", "parameters": {"command": "ls -la"}}'
    
    print(f"Simulated model response: {simulated_response}")
    
    final_response = execute_tool_calls(simulated_response, tm)
    print(f"Tool execution result:\n{final_response}")

def show_training_need():
    """Show why current models need tool training."""
    print("\nWhy current models can't use tools:")
    print("1. No training on TOOL_CALL format")
    print("2. No understanding of tool schemas")
    print("3. No examples of tool usage patterns")
    print("4. No reinforcement for structured output")
    print("\nSolution: Train with our comprehensive tool dataset")

if __name__ == "__main__":
    test_ollama_direct()
    test_ollama_with_tool_prompt()
    test_finance_assistant()
    demonstrate_working_tools()
    show_training_need()