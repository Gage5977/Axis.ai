#!/bin/bash

echo "🚀 AI Terminal Setup & Launch"
echo "============================"

# Check if we're in the right directory
if [ ! -f "terminal_ui.py" ]; then
    echo "❌ Error: Please run this script from the AI-Local-Models directory"
    exit 1
fi

# Check Python
echo "🔍 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check Ollama
echo "🔍 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. AI features will be limited."
    echo "   Install from: https://ollama.ai"
else
    echo "✅ Ollama found"
    echo "📦 Available models:"
    ollama list | head -5
fi

# Create app if needed
if [ ! -d "AI Terminal.app" ]; then
    echo ""
    echo "🛠  Creating macOS app..."
    python3 launch_ai_app.py
fi

# Launch options
echo ""
echo "🎯 How would you like to launch?"
echo "1) Simple Terminal UI (terminal_ui.py)"
echo "2) Advanced Split-Screen UI (advanced_terminal_ui.py)"
echo "3) Create macOS App"
echo "4) Exit"
echo ""
read -p "Choose (1-4): " choice

case $choice in
    1)
        echo "🚀 Launching Simple Terminal UI..."
        python3 terminal_ui.py
        ;;
    2)
        echo "🚀 Launching Advanced Terminal UI..."
        python3 advanced_terminal_ui.py
        ;;
    3)
        echo "🎯 Opening AI Terminal app..."
        open "AI Terminal.app"
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac