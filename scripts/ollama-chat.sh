#!/bin/bash

# Quick Ollama Chat Script
echo "🤖 Ollama Chat Assistant"
echo "Available models:"
ollama list

echo -n "Choose model (default: qwen3:14b): "
read model
model=${model:-qwen3:14b}

echo "Starting chat with $model... (Type 'exit' to quit)"
ollama run $model 