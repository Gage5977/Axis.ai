#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, scrolledtext
import requests
import threading
import json
import time

class AxisLocalTerminal:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AXIS Terminal - Local AI")
        self.window.geometry("1000x700")
        self.window.configure(bg='#000000')
        
        # Track available endpoints
        self.endpoints = []
        self.current_endpoint = None
        
        # Header
        header = tk.Frame(self.window, bg='#0A0A0A', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="AXIS TERMINAL", 
                        font=('SF Mono', 18, 'normal'),
                        fg='#E0E0E0', bg='#0A0A0A')
        title.pack(side='left', padx=24, pady=16)
        
        self.status = tk.Label(header, text="● Scanning...", 
                              font=('SF Mono', 10),
                              fg='#FFAA00', bg='#0A0A0A')
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
            pady=20,
            relief='flat',
            borderwidth=0
        )
        self.messages.pack(fill='both', expand=True)
        
        # Configure text tags
        self.messages.tag_config('user', foreground='#00FF88')
        self.messages.tag_config('ai', foreground='#00AAFF')
        self.messages.tag_config('error', foreground='#FF4444')
        self.messages.tag_config('system', foreground='#808080')
        self.messages.tag_config('timestamp', foreground='#606060')
        
        # Input area
        input_frame = tk.Frame(self.window, bg='#080808', height=80)
        input_frame.pack(fill='x')
        input_frame.pack_propagate(False)
        
        input_container = tk.Frame(input_frame, bg='#080808')
        input_container.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.input_field = tk.Entry(
            input_container,
            font=('SF Mono', 14),
            bg='#0F0F0F',
            fg='#E0E0E0',
            insertbackground='#00FF88',
            relief='flat',
            bd=12
        )
        self.input_field.pack(side='left', fill='x', expand=True)
        self.input_field.bind('<Return>', lambda e: self.send_message())
        
        self.send_btn = tk.Button(
            input_container,
            text='→',
            font=('SF Mono', 24),
            bg='#080808',
            fg='#00FF88',
            activebackground='#080808',
            activeforeground='#00FFAA',
            relief='flat',
            bd=0,
            padx=20,
            command=self.send_message
        )
        self.send_btn.pack(side='right', padx=(10, 0))
        
        # Start scanning for services
        self.add_message("SYSTEM", "Scanning for local AI services...")
        threading.Thread(target=self.scan_services, daemon=True).start()
        
    def add_message(self, sender, text):
        timestamp = time.strftime("%H:%M:%S")
        
        # Add timestamp
        self.messages.insert('end', f"[{timestamp}] ", 'timestamp')
        
        # Add sender
        tag = sender.lower()
        self.messages.insert('end', f"{sender}: ", tag)
        
        # Add message
        self.messages.insert('end', f"{text}\n\n", 'content')
        
        self.messages.see('end')
        
    def scan_services(self):
        """Scan for available local AI services"""
        services = [
            # Your running services
            {"name": "Node API", "port": 3000, "endpoints": [
                "/api/chat", "/api/generate", "/api/completion", "/chat", "/"
            ]},
            {"name": "Python API", "port": 8000, "endpoints": [
                "/chat", "/api/chat", "/v1/chat", "/generate", "/"
            ]},
            {"name": "Python API", "port": 8080, "endpoints": [
                "/api/chat", "/chat", "/v1/chat/completions", "/generate", "/"
            ]},
            # Finance ETL
            {"name": "Finance ETL", "port": 5001, "endpoints": [
                "/api/chat", "/chat", "/"
            ]},
        ]
        
        found_services = []
        
        for service in services:
            for endpoint in service['endpoints']:
                try:
                    url = f"http://localhost:{service['port']}{endpoint}"
                    response = requests.get(url, timeout=1)
                    if response.status_code < 500:  # Any non-error response
                        found_services.append({
                            'name': f"{service['name']} ({service['port']})",
                            'url': f"http://localhost:{service['port']}",
                            'endpoint': endpoint
                        })
                        self.add_message("SYSTEM", f"✓ Found {service['name']} on port {service['port']}")
                        break
                except:
                    pass
        
        if found_services:
            self.endpoints = found_services
            self.current_endpoint = found_services[0]
            self.status.config(text=f"● Connected to {self.current_endpoint['name']}", fg='#00FF88')
            self.add_message("SYSTEM", f"Using {self.current_endpoint['name']} at {self.current_endpoint['url']}")
        else:
            self.status.config(text="● No services found", fg='#FF4444')
            self.add_message("ERROR", "No local AI services detected on ports 3000, 5001, 8000, or 8080")
            
    def send_message(self):
        text = self.input_field.get().strip()
        if not text or not self.current_endpoint:
            return
            
        self.add_message("USER", text)
        self.input_field.delete(0, 'end')
        
        # Process in thread
        threading.Thread(target=self.process_message, args=(text,), daemon=True).start()
        
    def process_message(self, prompt):
        self.add_message("AI", "Processing...")
        
        try:
            # Try different request formats based on the endpoint
            url = self.current_endpoint['url'] + self.current_endpoint['endpoint']
            
            # Common request formats
            payloads = [
                {"prompt": prompt},  # Simple format
                {"message": prompt},
                {"text": prompt},
                {"query": prompt},
                {"input": prompt},
                {"messages": [{"role": "user", "content": prompt}]},  # OpenAI format
                {"question": prompt},
            ]
            
            response_text = None
            
            for payload in payloads:
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract response from various formats
                        response_text = (
                            data.get('response') or
                            data.get('text') or
                            data.get('message') or
                            data.get('output') or
                            data.get('result') or
                            data.get('answer') or
                            data.get('choices', [{}])[0].get('message', {}).get('content') or
                            data.get('data') or
                            str(data)
                        )
                        break
                except:
                    continue
            
            if response_text:
                # Update last message
                self.messages.delete('end-3c', 'end-1c')
                self.messages.insert('end', response_text + '\n\n')
            else:
                raise Exception("Could not get valid response from API")
                
        except Exception as e:
            self.messages.delete('end-3c', 'end-1c')
            self.messages.insert('end', f"Error: {str(e)}\n\n", 'error')
            
    def run(self):
        self.input_field.focus()
        self.window.mainloop()

if __name__ == '__main__':
    app = AxisLocalTerminal()
    app.run()