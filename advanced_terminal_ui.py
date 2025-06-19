#!/usr/bin/env python3
"""
Advanced Terminal UI - Split-screen with real-time updates
"""

import os
import sys
import time
import threading
import queue
import subprocess
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import select
import termios
import tty

# Add our capabilities
sys.path.append('capabilities')
try:
    from instant_models import InstantModelOrchestrator
    from unified_ai_system import UnifiedAISystem, TaskType
except:
    InstantModelOrchestrator = None
    UnifiedAISystem = None

class Colors:
    # Enhanced color palette
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright versions
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    
    # Reset
    RESET = '\033[0m'
    CLEAR = '\033[2J\033[H'
    
    # Box drawing (double lines)
    DTL = '╔'
    DTR = '╗'
    DBL = '╚'
    DBR = '╝'
    DH = '═'
    DV = '║'
    DT = '╦'
    DB = '╩'
    DL = '╠'
    DR = '╣'
    DX = '╬'
    
    # Box drawing (single lines)
    STL = '┌'
    STR = '┐'
    SBL = '└'
    SBR = '┘'
    SH = '─'
    SV = '│'
    ST = '┬'
    SB = '┴'
    SL = '├'
    SR = '┤'
    SX = '┼'

class Panel:
    """Base class for UI panels"""
    def __init__(self, x: int, y: int, width: int, height: int, title: str = ""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.content = []
        self.scroll_position = 0
        
    def draw_border(self, style="double"):
        """Draw panel border"""
        if style == "double":
            tl, tr, bl, br = Colors.DTL, Colors.DTR, Colors.DBL, Colors.DBR
            h, v = Colors.DH, Colors.DV
        else:
            tl, tr, bl, br = Colors.STL, Colors.STR, Colors.SBL, Colors.SBR
            h, v = Colors.SH, Colors.SV
            
        # Top border
        print(f'\033[{self.y};{self.x}H', end='')
        print(Colors.BRIGHT_CYAN + tl + h * (self.width - 2) + tr + Colors.RESET)
        
        # Title
        if self.title:
            title_x = self.x + (self.width - len(self.title) - 2) // 2
            print(f'\033[{self.y};{title_x}H', end='')
            print(Colors.BRIGHT_YELLOW + f" {self.title} " + Colors.RESET)
            
        # Sides
        for i in range(1, self.height - 1):
            print(f'\033[{self.y + i};{self.x}H', end='')
            print(Colors.BRIGHT_CYAN + v + Colors.RESET)
            print(f'\033[{self.y + i};{self.x + self.width - 1}H', end='')
            print(Colors.BRIGHT_CYAN + v + Colors.RESET)
            
        # Bottom border
        print(f'\033[{self.y + self.height - 1};{self.x}H', end='')
        print(Colors.BRIGHT_CYAN + bl + h * (self.width - 2) + br + Colors.RESET)
        
    def clear_content(self):
        """Clear panel content area"""
        for i in range(1, self.height - 1):
            print(f'\033[{self.y + i};{self.x + 1}H', end='')
            print(' ' * (self.width - 2))
            
    def add_line(self, text: str, color: str = ""):
        """Add a line to the panel content"""
        self.content.append((text, color))
        if len(self.content) > self.height - 2:
            self.scroll_position = len(self.content) - (self.height - 2)
            
    def render_content(self):
        """Render panel content with scrolling"""
        self.clear_content()
        visible_lines = self.content[self.scroll_position:self.scroll_position + self.height - 2]
        
        for i, (text, color) in enumerate(visible_lines):
            y_pos = self.y + i + 1
            # Truncate text if too long
            max_width = self.width - 4
            if len(text) > max_width:
                text = text[:max_width - 3] + "..."
                
            print(f'\033[{y_pos};{self.x + 2}H', end='')
            print(color + text + Colors.RESET)

class ChatPanel(Panel):
    """Panel for chat messages"""
    def add_message(self, role: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "user":
            self.add_line(f"[{timestamp}] You:", Colors.BRIGHT_GREEN + Colors.BOLD)
            # Word wrap message
            words = message.split()
            line = "  "
            for word in words:
                if len(line) + len(word) + 1 > self.width - 6:
                    self.add_line(line, Colors.GREEN)
                    line = "  " + word
                else:
                    line += " " + word if line != "  " else word
            if line.strip():
                self.add_line(line, Colors.GREEN)
                
        elif role == "assistant":
            self.add_line(f"[{timestamp}] AI:", Colors.BRIGHT_CYAN + Colors.BOLD)
            # Word wrap response
            if isinstance(message, dict):
                message = json.dumps(message, indent=2)
                
            lines = message.split('\n')
            for line in lines:
                words = line.split()
                wrapped_line = "  "
                for word in words:
                    if len(wrapped_line) + len(word) + 1 > self.width - 6:
                        self.add_line(wrapped_line, Colors.CYAN)
                        wrapped_line = "  " + word
                    else:
                        wrapped_line += " " + word if wrapped_line != "  " else word
                if wrapped_line.strip():
                    self.add_line(wrapped_line, Colors.CYAN)
                    
        self.add_line("", "")  # Empty line after message

class StatusPanel(Panel):
    """Panel for status information"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = {
            'mode': 'chat',
            'model': 'mistral:latest',
            'requests': 0,
            'avg_time': 0,
            'status': 'Ready'
        }
        
    def update_stats(self, **kwargs):
        """Update status statistics"""
        self.stats.update(kwargs)
        self.render_stats()
        
    def render_stats(self):
        """Render statistics"""
        self.clear_content()
        self.content = []
        
        self.add_line(f"Mode: {self.stats['mode'].upper()}", Colors.BRIGHT_GREEN)
        self.add_line(f"Model: {self.stats['model']}", Colors.BRIGHT_BLUE)
        self.add_line(f"Status: {self.stats['status']}", Colors.BRIGHT_YELLOW)
        self.add_line("", "")
        self.add_line(f"Requests: {self.stats['requests']}", Colors.WHITE)
        self.add_line(f"Avg Time: {self.stats['avg_time']:.2f}s", Colors.WHITE)
        
        self.render_content()

class ModelInfoPanel(Panel):
    """Panel showing model capabilities"""
    def show_model_info(self, mode: str):
        """Show information about current mode"""
        self.clear_content()
        self.content = []
        
        info = {
            'chat': [
                ("General Chat", Colors.BRIGHT_WHITE),
                ("", ""),
                ("Models:", Colors.BRIGHT_YELLOW),
                ("• Mistral", Colors.WHITE),
                ("• Llama 3.2", Colors.WHITE),
                ("• Qwen3 14B", Colors.WHITE),
                ("", ""),
                ("Best for:", Colors.BRIGHT_GREEN),
                ("• Conversations", Colors.WHITE),
                ("• Q&A", Colors.WHITE),
                ("• General tasks", Colors.WHITE)
            ],
            'code': [
                ("Code Assistant", Colors.BRIGHT_WHITE),
                ("", ""),
                ("Instant:", Colors.BRIGHT_YELLOW),
                ("• Code analysis", Colors.WHITE),
                ("• Complexity check", Colors.WHITE),
                ("• Style review", Colors.WHITE),
                ("", ""),
                ("With Claude:", Colors.BRIGHT_GREEN),
                ("• Generation", Colors.WHITE),
                ("• Debugging", Colors.WHITE),
                ("• Architecture", Colors.WHITE)
            ],
            'finance': [
                ("Finance Calculator", Colors.BRIGHT_WHITE),
                ("", ""),
                ("Instant:", Colors.BRIGHT_YELLOW),
                ("• ROI calc", Colors.WHITE),
                ("• Compound int", Colors.WHITE),
                ("• P/E ratios", Colors.WHITE),
                ("", ""),
                ("With Model:", Colors.BRIGHT_GREEN),
                ("• Analysis", Colors.WHITE),
                ("• Forecasting", Colors.WHITE),
                ("• Reports", Colors.WHITE)
            ],
            'reasoning': [
                ("Logic Engine", Colors.BRIGHT_WHITE),
                ("", ""),
                ("Instant:", Colors.BRIGHT_YELLOW),
                ("• Prop. logic", Colors.WHITE),
                ("• Inference", Colors.WHITE),
                ("• Proofs", Colors.WHITE),
                ("", ""),
                ("Advanced:", Colors.BRIGHT_GREEN),
                ("• Math proofs", Colors.WHITE),
                ("• Complex logic", Colors.WHITE),
                ("• Planning", Colors.WHITE)
            ]
        }
        
        for text, color in info.get(mode, []):
            self.add_line(text, color)
            
        self.render_content()

class AdvancedTerminalUI:
    def __init__(self):
        self.running = True
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Terminal dimensions
        self.update_dimensions()
        
        # UI Panels
        self.setup_panels()
        
        # Model system
        self.unified_system = UnifiedAISystem() if UnifiedAISystem else None
        self.current_mode = "chat"
        
        # Input handling
        self.input_buffer = ""
        self.input_history = []
        self.history_index = -1
        
        # Performance tracking
        self.request_times = []
        
    def update_dimensions(self):
        """Update terminal dimensions"""
        size = os.get_terminal_size()
        self.width = size.columns
        self.height = size.lines
        
    def setup_panels(self):
        """Setup UI panels with dynamic sizing"""
        # Calculate panel dimensions
        chat_width = int(self.width * 0.6)
        side_width = self.width - chat_width - 1
        
        # Main chat panel
        self.chat_panel = ChatPanel(1, 1, chat_width, self.height - 6, "CONVERSATION")
        
        # Status panel (top right)
        status_height = 10
        self.status_panel = StatusPanel(chat_width + 1, 1, side_width, status_height, "STATUS")
        
        # Model info panel (middle right)
        info_height = self.height - status_height - 7
        self.info_panel = ModelInfoPanel(chat_width + 1, status_height + 1, side_width, info_height, "MODE INFO")
        
        # Input panel (bottom)
        self.input_panel = Panel(1, self.height - 5, self.width, 5, "INPUT")
        
    def draw_ui(self):
        """Draw all UI elements"""
        # Clear screen
        print(Colors.CLEAR, end='')
        
        # Draw panels
        self.chat_panel.draw_border()
        self.status_panel.draw_border()
        self.info_panel.draw_border()
        self.input_panel.draw_border()
        
        # Draw menu bar
        self.draw_menu_bar()
        
        # Initial content
        self.info_panel.show_model_info(self.current_mode)
        self.status_panel.render_stats()
        
    def draw_menu_bar(self):
        """Draw menu bar at top of input panel"""
        y = self.height - 4
        print(f'\033[{y};3H', end='')
        
        menu_items = [
            ("F1", "Chat", self.current_mode == "chat"),
            ("F2", "Code", self.current_mode == "code"),
            ("F3", "Finance", self.current_mode == "finance"),
            ("F4", "Logic", self.current_mode == "reasoning"),
            ("F5", "Clear", False),
            ("ESC", "Exit", False)
        ]
        
        for key, label, active in menu_items:
            if active:
                print(Colors.BG_CYAN + Colors.BLACK + f"[{key}]" + Colors.RESET, end='')
                print(Colors.BRIGHT_CYAN + f" {label} " + Colors.RESET, end='')
            else:
                print(Colors.BRIGHT_BLUE + f"[{key}]" + Colors.RESET, end='')
                print(f" {label} ", end='')
                
    def update_input_display(self):
        """Update input area display"""
        # Clear input line
        y = self.height - 3
        print(f'\033[{y};3H', end='')
        print(' ' * (self.width - 6))
        
        # Display prompt and input
        print(f'\033[{y};3H', end='')
        prompt = f"{Colors.BRIGHT_GREEN}❯{Colors.RESET} "
        print(prompt + self.input_buffer, end='', flush=True)
        
        # Position cursor
        cursor_x = 5 + len(self.input_buffer)
        print(f'\033[{y};{cursor_x}H', end='', flush=True)
        
    def process_input(self, key: str):
        """Process keyboard input"""
        if key == '\x1b':  # ESC
            self.running = False
        elif key == '\r' or key == '\n':  # Enter
            if self.input_buffer.strip():
                self.handle_command(self.input_buffer)
                self.input_history.append(self.input_buffer)
                self.history_index = -1
                self.input_buffer = ""
        elif key == '\x7f' or key == '\b':  # Backspace
            self.input_buffer = self.input_buffer[:-1]
        elif key == '\x1b[A':  # Up arrow
            if self.history_index < len(self.input_history) - 1:
                self.history_index += 1
                self.input_buffer = self.input_history[-(self.history_index + 1)]
        elif key == '\x1b[B':  # Down arrow
            if self.history_index > -1:
                self.history_index -= 1
                if self.history_index == -1:
                    self.input_buffer = ""
                else:
                    self.input_buffer = self.input_history[-(self.history_index + 1)]
        elif key >= ' ':  # Printable characters
            self.input_buffer += key
            
        self.update_input_display()
        
    def handle_command(self, command: str):
        """Handle user commands"""
        # Check for function keys
        if command.lower() == 'f1':
            self.switch_mode('chat')
        elif command.lower() == 'f2':
            self.switch_mode('code')
        elif command.lower() == 'f3':
            self.switch_mode('finance')
        elif command.lower() == 'f4':
            self.switch_mode('reasoning')
        elif command.lower() == 'f5':
            self.chat_panel.content = []
            self.chat_panel.render_content()
        else:
            # Process as query
            self.process_query(command)
            
    def switch_mode(self, mode: str):
        """Switch operation mode"""
        self.current_mode = mode
        self.status_panel.update_stats(mode=mode)
        self.info_panel.show_model_info(mode)
        self.draw_menu_bar()
        
    def process_query(self, query: str):
        """Process user query"""
        # Add to chat
        self.chat_panel.add_message("user", query)
        self.chat_panel.render_content()
        
        # Update status
        self.status_panel.update_stats(status="Processing...")
        
        # Process in background thread
        thread = threading.Thread(target=self._process_async, args=(query,))
        thread.start()
        
    def _process_async(self, query: str):
        """Process query asynchronously"""
        start_time = time.time()
        
        try:
            if self.unified_system:
                # Map mode to task type
                task_map = {
                    'chat': TaskType.GENERAL_CHAT,
                    'code': TaskType.CODE_ANALYSIS,
                    'finance': TaskType.FINANCIAL_CALC,
                    'reasoning': TaskType.REASONING
                }
                
                task_type = task_map.get(self.current_mode, TaskType.GENERAL_CHAT)
                result = self.unified_system.process(query, task_type, prefer_local=True)
                
                # Format response
                if isinstance(result, dict):
                    if 'response' in result:
                        response = result['response']
                    elif 'result' in result:
                        response = f"{result.get('calculation', 'Result')}: {result['result']}"
                    elif 'reasoning_steps' in result:
                        response = "Reasoning:\n" + "\n".join(f"• {step}" for step in result['reasoning_steps'])
                    else:
                        response = json.dumps(result, indent=2)
                else:
                    response = str(result)
            else:
                # Fallback to simple response
                response = f"Processed: {query} (no AI system available)"
                
        except Exception as e:
            response = f"Error: {str(e)}"
            
        # Update performance stats
        elapsed = time.time() - start_time
        self.request_times.append(elapsed)
        avg_time = sum(self.request_times[-10:]) / len(self.request_times[-10:])
        
        # Update UI from main thread
        self.output_queue.put(('response', response))
        self.output_queue.put(('stats', {
            'requests': len(self.request_times),
            'avg_time': avg_time,
            'status': 'Ready'
        }))
        
    def input_thread(self):
        """Thread for handling keyboard input"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    # Handle escape sequences
                    if key == '\x1b':
                        seq = key
                        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                            seq += sys.stdin.read(1)
                            if seq[-1] == '[':
                                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                                    seq += sys.stdin.read(1)
                        key = seq
                        
                    self.input_queue.put(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    def run(self):
        """Main UI loop"""
        # Initial setup
        self.draw_ui()
        self.update_input_display()
        
        # Start input thread
        input_thread = threading.Thread(target=self.input_thread)
        input_thread.daemon = True
        input_thread.start()
        
        # Main loop
        while self.running:
            # Handle input
            try:
                key = self.input_queue.get_nowait()
                self.process_input(key)
            except queue.Empty:
                pass
                
            # Handle output
            try:
                msg_type, data = self.output_queue.get_nowait()
                if msg_type == 'response':
                    self.chat_panel.add_message('assistant', data)
                    self.chat_panel.render_content()
                elif msg_type == 'stats':
                    self.status_panel.update_stats(**data)
            except queue.Empty:
                pass
                
            time.sleep(0.01)
            
        # Cleanup
        print(Colors.CLEAR, end='')
        print(Colors.BRIGHT_CYAN + "Thanks for using Advanced AI Terminal!" + Colors.RESET)

def main():
    """Entry point"""
    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("color")
        
    try:
        ui = AdvancedTerminalUI()
        ui.run()
    except KeyboardInterrupt:
        print(Colors.CLEAR + "\nExiting...")
    except Exception as e:
        print(Colors.CLEAR + f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()