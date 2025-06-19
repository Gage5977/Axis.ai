#!/usr/bin/env python3
"""
Simple Flask web app for Ollama chat
Run with: python3 ollama_web_app.py
"""

from flask import Flask, render_template_string, request, jsonify
import requests
import json

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Local Ollama Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-box { height: 400px; border: 1px solid #ddd; padding: 10px; overflow-y: scroll; margin: 20px 0; }
        .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; margin-left: 10px; }
        select { padding: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🤖 Local AI Assistant</h1>
    
    <label>Model: </label>
    <select id="model">
        <option value="qwen3:14b">Qwen3 14B</option>
        <option value="llama3.2:latest">Llama 3.2</option>
    </select>
    
    <div id="chat" class="chat-box"></div>
    
    <input type="text" id="message" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
    <button onclick="sendMessage()">Send</button>
    
    <script>
        async function sendMessage() {
            const messageInput = document.getElementById('message');
            const chatBox = document.getElementById('chat');
            const modelSelect = document.getElementById('model');
            
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message
            chatBox.innerHTML += `<div class="message user"><strong>You:</strong> ${message}</div>`;
            messageInput.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message,
                        model: modelSelect.value 
                    })
                });
                
                const data = await response.json();
                chatBox.innerHTML += `<div class="message assistant"><strong>AI:</strong> ${data.response}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
                
            } catch (error) {
                chatBox.innerHTML += `<div class="message assistant"><strong>Error:</strong> ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    model = data.get('model', 'qwen3:14b')
    
    try:
        # Send request to Ollama
        response = requests.post('http://localhost:11434/api/generate', json={
            'model': model,
            'prompt': message,
            'stream': False
        })
        
        result = response.json()
        return jsonify({'response': result.get('response', 'No response')})
        
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Ollama Web App...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(host='localhost', port=5000, debug=True) 