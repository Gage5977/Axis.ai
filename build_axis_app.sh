#!/bin/bash

echo "Building AXIS Terminal for local AI APIs..."

# Create standalone Python app
cat > axis_terminal_app.py << 'EOF'
#!/usr/bin/env python3

import sys
import os
import json
import urllib.request
import urllib.parse
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time

class AxisTerminal:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AXIS Terminal")
        self.window.geometry("1000x700")
        self.window.configure(bg='#000000')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = tk.Frame(self.window, bg='#0A0A0A', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="AXIS TERMINAL", 
                        font=('SF Mono', 18, 'normal'),
                        fg='#E0E0E0', bg='#0A0A0A')
        title.pack(side='left', padx=24, pady=16)
        
        self.status = tk.Label(header, text="● Connected", 
                              font=('SF Mono', 10),
                              fg='#00FF88', bg='#0A0A0A')
        self.status.pack(side='right', padx=24)
        
        # Messages area
        self.messages = scrolledtext.ScrolledText(
            self.window,
            font=('SF Mono', 12),
            bg='#050505',
            fg='#E0E0E0',
            insertbackground='#00FF88',
            wrap='word',
            padx=20,
            pady=20
        )
        self.messages.pack(fill='both', expand=True)
        
        # Input area
        input_frame = tk.Frame(self.window, bg='#080808')
        input_frame.pack(fill='x', padx=20, pady=20)
        
        self.input_field = tk.Entry(
            input_frame,
            font=('SF Mono', 14),
            bg='#0F0F0F',
            fg='#E0E0E0',
            insertbackground='#00FF88',
            relief='flat',
            bd=10
        )
        self.input_field.pack(side='left', fill='x', expand=True)
        self.input_field.bind('<Return>', lambda e: self.send_message())
        
        self.send_btn = tk.Button(
            input_frame,
            text='→',
            font=('SF Mono', 20),
            bg='#080808',
            fg='#00FF88',
            activebackground='#080808',
            activeforeground='#00FFAA',
            relief='flat',
            bd=0,
            padx=20,
            command=self.send_message
        )
        self.send_btn.pack(side='right')
        
        # Initial message
        self.add_message("AXIS", "Terminal initialized. Connecting to local AI services...")
        
        # Check connections
        self.check_apis()
        
    def add_message(self, sender, text):
        timestamp = time.strftime("%H:%M:%S")
        
        self.messages.insert('end', f"\n[{timestamp}] ", 'timestamp')
        self.messages.insert('end', f"{sender}\n", sender.lower())
        self.messages.insert('end', f"{text}\n", 'content')
        
        # Configure tags
        self.messages.tag_config('timestamp', foreground='#606060')
        self.messages.tag_config('user', foreground='#00FF88')
        self.messages.tag_config('axis', foreground='#00AAFF')
        self.messages.tag_config('content', foreground='#E0E0E0')
        
        self.messages.see('end')
        
    def check_apis(self):
        """Check for available local AI APIs"""
        apis = [
            ("Ollama", "http://localhost:11434/api/tags"),
            ("Local API (5000)", "http://localhost:5000/"),
            ("Local API (8080)", "http://localhost:8080/"),
            ("Local API (3000)", "http://localhost:3000/"),
        ]
        
        found = False
        for name, url in apis:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        self.add_message("AXIS", f"✓ Found {name}")
                        found = True
                        break
            except:
                pass
        
        if not found:
            self.add_message("AXIS", "No local AI APIs detected. Please start Ollama or your local AI service.")
            self.status.config(text="● Disconnected", fg='#FF4444')
        else:
            self.add_message("AXIS", "Ready for input.")
            
    def send_message(self):
        text = self.input_field.get().strip()
        if not text:
            return
            
        self.add_message("USER", text)
        self.input_field.delete(0, 'end')
        
        # Process in thread
        threading.Thread(target=self.process_message, args=(text,)).start()
        
    def process_message(self, prompt):
        self.window.after(0, lambda: self.add_message("AXIS", "..."))
        
        try:
            # Try Ollama first
            data = json.dumps({
                "model": "mistral:latest",
                "prompt": prompt,
                "stream": False
            }).encode('utf-8')
            
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                response_text = result.get('response', 'No response')
                
                # Update last message
                self.window.after(0, lambda: self.update_last_message(response_text))
                
        except Exception as e:
            error_msg = f"Error: Could not connect to AI service. Make sure Ollama is running.\n\nTry: ollama serve"
            self.window.after(0, lambda: self.update_last_message(error_msg))
            
    def update_last_message(self, new_text):
        # Delete the "..." message
        self.messages.delete('end-2l', 'end-1l')
        self.add_message("AXIS", new_text)
        
    def run(self):
        self.input_field.focus()
        self.window.mainloop()

if __name__ == '__main__':
    app = AxisTerminal()
    app.run()
EOF

# Make executable
chmod +x axis_terminal_app.py

# Create app bundle
APP_NAME="AXIS Terminal Local"
rm -rf "$APP_NAME.app"
mkdir -p "$APP_NAME.app/Contents/MacOS"
mkdir -p "$APP_NAME.app/Contents/Resources"

# Create launcher script
cat > "$APP_NAME.app/Contents/MacOS/AXIS Terminal Local" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(dirname "$(dirname "$(dirname "$DIR")")")"
/usr/bin/python3 axis_terminal_app.py
EOF

chmod +x "$APP_NAME.app/Contents/MacOS/AXIS Terminal Local"

# Create Info.plist
cat > "$APP_NAME.app/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>AXIS Terminal Local</string>
    <key>CFBundleExecutable</key>
    <string>AXIS Terminal Local</string>
    <key>CFBundleIdentifier</key>
    <string>com.axisthorn.terminal.local</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF

echo "✓ Created axis_terminal_app.py - Python GUI app"
echo "✓ Created $APP_NAME.app - macOS app bundle"
echo ""
echo "You can run it with:"
echo "  python3 axis_terminal_app.py"
echo "Or double-click '$APP_NAME.app'"