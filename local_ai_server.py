#!/usr/bin/env python3
"""
Local AI Server - Connects web terminal to Ollama and instant models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import sys
import os

# Add capabilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'capabilities'))

try:
    from instant_models import InstantModelOrchestrator
    from unified_ai_system import UnifiedAISystem, TaskType
    instant_models = InstantModelOrchestrator()
    unified_system = UnifiedAISystem()
except:
    instant_models = None
    unified_system = None
    print("Warning: Instant models not available")

app = Flask(__name__)
CORS(app)  # Allow browser to connect

# Current model
current_model = "mistral:latest"

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from terminal"""
    try:
        data = request.json
        message = data.get('message', '')
        mode = data.get('mode', 'chat')
        
        # Try instant models first for specific modes
        if instant_models and mode in ['finance', 'code', 'logic']:
            try:
                result = instant_models.process(message, mode if mode != 'logic' else 'reasoning')
                if result and not isinstance(result, dict) or not result.get('error'):
                    if isinstance(result, dict):
                        # Format dict responses nicely
                        if 'result' in result:
                            response = f"{result.get('calculation', 'Result')}: {result['result']}"
                        elif 'reasoning_steps' in result:
                            response = "Reasoning:\n" + "\n".join(f"- {step}" for step in result['reasoning_steps'])
                        elif 'metrics' in result:
                            metrics = result['metrics']
                            response = f"Analysis:\nLines: {metrics['lines']}\nComplexity: {metrics['complexity']}"
                        else:
                            response = json.dumps(result, indent=2)
                    else:
                        response = str(result)
                    
                    return jsonify({'response': response, 'model': 'instant'})
            except Exception as e:
                print(f"Instant model error: {e}")
        
        # Fall back to Ollama
        result = subprocess.run(
            ['ollama', 'run', current_model, message],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return jsonify({
                'response': result.stdout.strip(),
                'model': current_model
            })
        else:
            return jsonify({
                'response': f"Error: {result.stderr}",
                'model': current_model
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({'response': 'Request timed out', 'model': current_model}), 500
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}', 'model': current_model}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    models.append({
                        'name': parts[0],
                        'size': parts[2] if len(parts) > 2 else 'unknown'
                    })
            return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch current model"""
    global current_model
    data = request.json
    model = data.get('model')
    
    # Verify model exists
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if model in result.stdout:
        current_model = model
        return jsonify({'success': True, 'model': current_model})
    else:
        return jsonify({'success': False, 'error': 'Model not found'}), 404

@app.route('/status', methods=['GET'])
def status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'current_model': current_model,
        'instant_models': instant_models is not None,
        'unified_system': unified_system is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Local AI Server...")
    print("📡 Server running at http://localhost:5000")
    print("🔌 Connect web terminal to this server")
    print("✨ Instant models:", "Available" if instant_models else "Not available")
    print("\nPress Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=False)