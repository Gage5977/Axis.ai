#!/bin/bash

echo "🚀 AI Terminal Launcher"
echo "====================="

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip3 install flask flask-cors
fi

# Start the server in background
echo "🔌 Starting AI server..."
python3 local_ai_server.py &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Open the terminal in browser
echo "🌐 Opening terminal in browser..."
open terminal_web.html

echo ""
echo "✅ Terminal is running!"
echo "📡 Server PID: $SERVER_PID"
echo ""
echo "To stop the server, run: kill $SERVER_PID"
echo "Or press Ctrl+C to stop everything"

# Wait for user to stop
wait $SERVER_PID