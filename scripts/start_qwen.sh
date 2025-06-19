#!/bin/bash

# Enhanced Qwen Startup Script
# Provides real system access to Qwen through function calling

echo "Starting Enhanced Qwen with System Access..."

# Check if Ollama is running
if ! pgrep -f "ollama" > /dev/null; then
    echo "Warning: Ollama not running. Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Check if Qwen model is available
if ! ollama list | grep -q "qwen3:14b"; then
    echo "Error: Qwen model not found. Please run: ollama pull qwen3:14b"
    exit 1
fi

echo "Ollama is running"
echo "Qwen model available"

# Create training data directory if it doesn't exist
mkdir -p /Users/axisthornllc/QWEN_TRAINING_DOCS/current_system_state

echo "Training data directory ready"

# Make Python scripts executable
chmod +x qwen_system_agent.py
chmod +x qwen_enhanced.py

echo "Scripts are executable"

# Start the enhanced Qwen
echo "Launching Enhanced Qwen..."
echo "   - Use 'help' to see available functions"
echo "   - Use 'status' to check system status"
echo "   - Use 'quit' to exit"
echo "   - Try: 'list files in current directory'"
echo "   - Try: 'show me git status'"
echo "   - Try: 'what processes are running?'"
echo ""

python3 qwen_enhanced.py