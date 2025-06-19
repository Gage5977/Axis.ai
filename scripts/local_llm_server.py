#!/usr/bin/env python3
"""
Local LLM Server - Ollama Replacement
Lightweight server to run Qwen locally without external dependencies
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import threading
import time
import os
from pathlib import Path

class LocalLLMServer:
    def __init__(self, model_path="Qwen/Qwen2.5-14B-Instruct"):
        self.app = Flask(__name__)
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.is_loaded = False
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes compatible with Ollama"""
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            if not self.is_loaded:
                return jsonify({"error": "Model not loaded"}), 503
                
            data = request.json
            messages = data.get('messages', [])
            
            # Convert messages to prompt
            prompt = self.messages_to_prompt(messages)
            
            # Generate response
            response = self.generate_response(prompt)
            
            return jsonify({
                "message": {"content": response},
                "done": True
            })
        
        @self.app.route('/api/tags', methods=['GET'])
        def list_models():
            return jsonify({
                "models": [{
                    "name": "qwen-local:latest",
                    "size": "9.3GB",
                    "modified": "2024-06-18"
                }]
            })
        
        @self.app.route('/api/generate', methods=['POST'])
        def generate():
            if not self.is_loaded:
                return jsonify({"error": "Model not loaded"}), 503
                
            data = request.json
            prompt = data.get('prompt', '')
            
            response = self.generate_response(prompt)
            
            return jsonify({
                "response": response,
                "done": True
            })
    
    def messages_to_prompt(self, messages):
        """Convert chat messages to prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def load_model(self):
        """Load the Qwen model"""
        print("Loading Qwen model...")
        
        try:
            # Check if model exists locally
            local_model_path = Path("./models/qwen-local")
            
            if local_model_path.exists():
                model_path = str(local_model_path)
                print(f"Loading local model from {model_path}")
            else:
                model_path = self.model_path
                print(f"Downloading model {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.is_loaded = True
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU-only mode...")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
                
                self.is_loaded = True
                print("Model loaded in CPU mode")
                
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                self.is_loaded = False
    
    def generate_response(self, prompt, max_length=2048):
        """Generate response from the model"""
        if not self.is_loaded:
            return "Error: Model not loaded"
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated text
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Generation error: {e}"
    
    def start_server(self, host='localhost', port=11434):
        """Start the server"""
        print(f"Starting Local LLM Server on {host}:{port}")
        
        # Load model in background thread
        model_thread = threading.Thread(target=self.load_model)
        model_thread.daemon = True
        model_thread.start()
        
        # Start Flask server
        self.app.run(host=host, port=port, debug=False)

def main():
    """Main function"""
    print("Local LLM Server - Ollama Replacement")
    print("Compatible with your existing Enhanced Qwen setup")
    
    # Create server instance
    server = LocalLLMServer()
    
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("Running on CPU (slower but still functional)")
    
    # Start server
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()