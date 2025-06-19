#!/usr/bin/env python3
"""
Terminal-style AI Interface with App-like Experience
"""

import os
import sys
import time
import subprocess
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Any

# Terminal colors and styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CLEAR = '\033[2J\033[H'
    
    # Box drawing characters
    TL = '╔'
    TR = '╗'
    BL = '╚'
    BR = '╝'
    H = '═'
    V = '║'
    T = '╦'
    B = '╩'
    L = '╠'
    R = '╣'
    X = '╬'

class TerminalUI:
    def __init__(self):
        self.width = os.get_terminal_size().columns
        self.height = os.get_terminal_size().lines
        self.messages = []
        self.current_model = "mistral:latest"
        self.available_models = []
        self.mode = "chat"  # chat, code, finance, reasoning
        self.typing_speed = 0.01  # For typewriter effect
        self.status = "Ready"
        
        # Import our capabilities
        sys.path.append('capabilities')
        try:
            from instant_models import InstantModelOrchestrator
            self.instant_models = InstantModelOrchestrator()
        except:
            self.instant_models = None
            
    def clear_screen(self):
        """Clear terminal and move cursor to top"""
        print(Colors.CLEAR, end='')
        
    def draw_box(self, x, y, width, height, title=""):
        """Draw a box at position with title"""
        # Move cursor and draw top border
        print(f'\033[{y};{x}H', end='')
        print(Colors.CYAN + Colors.TL + Colors.H * (width-2) + Colors.TR + Colors.ENDC)
        
        # Draw title if provided
        if title:
            title_pos = x + (width - len(title)) // 2
            print(f'\033[{y};{title_pos}H', end='')
            print(Colors.BOLD + Colors.YELLOW + title + Colors.ENDC)
            
        # Draw sides
        for i in range(1, height-1):
            print(f'\033[{y+i};{x}H', end='')
            print(Colors.CYAN + Colors.V + Colors.ENDC, end='')
            print(f'\033[{y+i};{x+width-1}H', end='')
            print(Colors.CYAN + Colors.V + Colors.ENDC)
            
        # Draw bottom border
        print(f'\033[{y+height-1};{x}H', end='')
        print(Colors.CYAN + Colors.BL + Colors.H * (width-2) + Colors.BR + Colors.ENDC)
        
    def typewriter_effect(self, text, x, y, color=""):
        """Print text with typewriter effect"""
        print(f'\033[{y};{x}H', end='')
        for char in text:
            print(color + char + Colors.ENDC, end='', flush=True)
            if self.typing_speed > 0:
                time.sleep(self.typing_speed)
                
    def draw_header(self):
        """Draw the header section"""
        self.draw_box(1, 1, self.width, 5, "🤖 AI TERMINAL")
        
        # Status line
        print(f'\033[3;3H', end='')
        print(Colors.GREEN + f"Mode: {self.mode.upper()}" + Colors.ENDC, end='')
        
        print(f'\033[3;20H', end='')
        print(Colors.BLUE + f"Model: {self.current_model}" + Colors.ENDC, end='')
        
        print(f'\033[3;{self.width-20}H', end='')
        print(Colors.YELLOW + f"Status: {self.status}" + Colors.ENDC, end='')
        
    def draw_menu_bar(self):
        """Draw bottom menu bar"""
        y = self.height - 3
        self.draw_box(1, y, self.width, 3)
        
        # Menu items
        menu_items = [
            ("F1", "Chat"),
            ("F2", "Code"),
            ("F3", "Finance"),
            ("F4", "Logic"),
            ("F5", "Models"),
            ("ESC", "Exit")
        ]
        
        x_pos = 3
        print(f'\033[{y+1};{x_pos}H', end='')
        for key, label in menu_items:
            print(Colors.BOLD + Colors.CYAN + f"[{key}]" + Colors.ENDC, end='')
            print(f" {label}  ", end='')
            
    def draw_chat_area(self):
        """Draw the main chat area"""
        chat_height = self.height - 8
        self.draw_box(1, 5, self.width, chat_height, "CONVERSATION")
        
        # Display messages
        y_offset = 6
        max_messages = chat_height - 3
        display_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        for msg in display_messages:
            if msg['role'] == 'user':
                print(f'\033[{y_offset};3H', end='')
                print(Colors.BOLD + Colors.GREEN + "You: " + Colors.ENDC, end='')
                self.wrap_text(msg['content'], 8, y_offset, self.width - 6, Colors.GREEN)
            else:
                print(f'\033[{y_offset};3H', end='')
                print(Colors.BOLD + Colors.CYAN + "AI: " + Colors.ENDC, end='')
                self.wrap_text(msg['content'], 7, y_offset, self.width - 6, Colors.CYAN)
            y_offset += self.count_wrapped_lines(msg['content'], self.width - 10) + 1
            
    def wrap_text(self, text, x_start, y_start, max_width, color=""):
        """Wrap and print text within bounds"""
        words = text.split()
        current_line = ""
        y_offset = 0
        
        for word in words:
            if len(current_line) + len(word) + 1 > max_width - x_start:
                print(f'\033[{y_start + y_offset};{x_start}H', end='')
                print(color + current_line + Colors.ENDC)
                current_line = word + " "
                y_offset += 1
            else:
                current_line += word + " "
                
        if current_line:
            print(f'\033[{y_start + y_offset};{x_start}H', end='')
            print(color + current_line.strip() + Colors.ENDC)
            
    def count_wrapped_lines(self, text, max_width):
        """Count how many lines text will take when wrapped"""
        words = text.split()
        current_line = ""
        lines = 1
        
        for word in words:
            if len(current_line) + len(word) + 1 > max_width:
                lines += 1
                current_line = word + " "
            else:
                current_line += word + " "
                
        return lines
        
    def get_ollama_models(self):
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            self.available_models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    self.available_models.append(model_name)
        except:
            self.available_models = ["mistral:latest"]
            
    def process_with_instant_models(self, query):
        """Process query with instant models"""
        if self.instant_models:
            try:
                if self.mode == "finance":
                    return self.instant_models.process(query, 'finance')
                elif self.mode == "code":
                    return self.instant_models.process(query, 'code')
                elif self.mode == "reasoning":
                    return self.instant_models.process(query, 'reasoning')
            except:
                pass
        return None
        
    def process_with_ollama(self, query):
        """Process query with Ollama"""
        try:
            self.status = "Processing..."
            self.draw_header()
            
            cmd = ['ollama', 'run', self.current_model, query]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Request timed out"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            self.status = "Ready"
            
    def show_model_selector(self):
        """Show model selection screen"""
        self.clear_screen()
        self.get_ollama_models()
        
        # Draw model selection box
        self.draw_box(10, 5, self.width - 20, len(self.available_models) + 6, "SELECT MODEL")
        
        y = 7
        for i, model in enumerate(self.available_models):
            print(f'\033[{y};15H', end='')
            if model == self.current_model:
                print(Colors.GREEN + Colors.BOLD + f"▶ {model}" + Colors.ENDC)
            else:
                print(f"  {model}")
            y += 1
            
        print(f'\033[{y+1};15H', end='')
        print(Colors.YELLOW + "Press number to select, ESC to cancel" + Colors.ENDC)
        
        # Wait for selection
        import termios, tty
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            
            if key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(self.available_models):
                    self.current_model = self.available_models[idx]
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    def run(self):
        """Main UI loop"""
        self.clear_screen()
        
        # Welcome animation
        welcome = "INITIALIZING AI TERMINAL..."
        self.typewriter_effect(welcome, (self.width - len(welcome)) // 2, self.height // 2, Colors.CYAN + Colors.BOLD)
        time.sleep(0.5)
        
        self.clear_screen()
        self.get_ollama_models()
        
        while True:
            # Draw UI
            self.draw_header()
            self.draw_chat_area()
            self.draw_menu_bar()
            
            # Get input
            input_y = self.height - 5
            print(f'\033[{input_y};3H', end='')
            print(Colors.BOLD + "> " + Colors.ENDC, end='', flush=True)
            
            try:
                user_input = input()
                
                # Handle function keys
                if user_input.lower() in ['f1', '/chat']:
                    self.mode = "chat"
                    continue
                elif user_input.lower() in ['f2', '/code']:
                    self.mode = "code"
                    continue
                elif user_input.lower() in ['f3', '/finance']:
                    self.mode = "finance"
                    continue
                elif user_input.lower() in ['f4', '/logic']:
                    self.mode = "reasoning"
                    continue
                elif user_input.lower() in ['f5', '/models']:
                    self.show_model_selector()
                    continue
                elif user_input.lower() in ['exit', 'quit', 'esc']:
                    break
                    
                if user_input.strip():
                    # Add user message
                    self.messages.append({'role': 'user', 'content': user_input})
                    
                    # Process with appropriate model
                    response = None
                    
                    # Try instant models first for specific modes
                    if self.mode in ["finance", "code", "reasoning"]:
                        response = self.process_with_instant_models(user_input)
                        
                    # Fall back to Ollama
                    if not response:
                        response = self.process_with_ollama(user_input)
                        
                    # Format response
                    if isinstance(response, dict):
                        response = self.format_dict_response(response)
                        
                    self.messages.append({'role': 'assistant', 'content': str(response)})
                    
            except KeyboardInterrupt:
                break
                
        # Cleanup
        self.clear_screen()
        print(Colors.BOLD + Colors.CYAN + "Thanks for using AI Terminal!" + Colors.ENDC)
        
    def format_dict_response(self, response):
        """Format dictionary responses nicely"""
        if 'calculation' in response:
            return f"{response.get('calculation', '')}: {response.get('result', '')}"
        elif 'metrics' in response:
            metrics = response['metrics']
            return f"Lines: {metrics['lines']}, Complexity: {metrics['complexity']}"
        elif 'reasoning_steps' in response:
            return "Reasoning: " + " → ".join(response['reasoning_steps'])
        else:
            return str(response)

if __name__ == "__main__":
    # Enable ANSI escape sequences on Windows
    if sys.platform == "win32":
        os.system("color")
        
    ui = TerminalUI()
    ui.run()