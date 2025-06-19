#!/usr/bin/env python3
"""
Clean Terminal UI - Better text flow and formatting
"""

import os
import sys
import subprocess
import textwrap
import shutil

# Add capabilities
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'capabilities'))

try:
    from instant_models import InstantModelOrchestrator
    instant_models = InstantModelOrchestrator()
except:
    instant_models = None

class CleanTerminal:
    def __init__(self):
        self.mode = "chat"
        self.model = "mistral:latest"
        self.update_width()
        
    def update_width(self):
        """Update terminal width dynamically"""
        try:
            self.width = shutil.get_terminal_size().columns
        except:
            self.width = 80
        # Ensure consistent wrapping with margin
        wrap_width = max(40, min(self.width - 4, 100))
        self.wrapper = textwrap.TextWrapper(
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False
        )
        
    def clear(self):
        os.system('clear' if os.name != 'nt' else 'cls')
        
    def print_wrapped(self, text, prefix="", color=""):
        """Print text with proper wrapping"""
        # Update width before each print
        self.update_width()
        
        if not text:
            print()
            return
            
        if prefix:
            # Split prefix for alignment
            prefix_len = len(prefix)
            indent = ' ' * prefix_len
            
            lines = self.wrapper.wrap(text)
            if lines:
                print(f"{color}{prefix}{lines[0]}")
                for line in lines[1:]:
                    print(f"{indent}{line}")
        else:
            for line in self.wrapper.wrap(text):
                print(f"{color}{line}")
        print()  # Add space after
                
    def process_instant(self, query):
        """Try instant processing first"""
        if not instant_models:
            return None
            
        mode_map = {
            'finance': 'finance',
            'code': 'code', 
            'logic': 'reasoning'
        }
        
        if self.mode in mode_map:
            try:
                result = instant_models.process(query, mode_map[self.mode])
                if isinstance(result, dict):
                    if 'result' in result:
                        return f"{result.get('calculation', 'Result')}: {result['result']}"
                    elif 'metrics' in result:
                        m = result['metrics']
                        return f"Lines: {m['lines']}, Functions: {m.get('functions', 0)}, Complexity: {m['complexity']}"
                return str(result)
            except:
                pass
        return None
        
    def process_ollama(self, query):
        """Process with Ollama"""
        try:
            cmd = ['ollama', 'run', self.model, query]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Request timed out (30s)"
        except Exception as e:
            return f"Ollama error: {str(e)}"
            
    def run(self):
        """Main loop"""
        self.clear()
        print("AI Terminal\n")
        print("Commands: help, mode <name>, models, clear, exit")
        print(f"Current: {self.mode} mode, {self.model}\n")
        
        while True:
            try:
                # Update width before prompt
                self.update_width()
                
                # Prompt
                prompt = f"[{self.mode}] > "
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'clear':
                    self.clear()
                    continue
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  help          - Show this help")
                    print("  clear         - Clear screen")
                    print("  mode <name>   - Switch mode (chat, code, finance, logic)")
                    print("  models        - List available models")
                    print("  use <model>   - Switch model")
                    print("  exit          - Quit\n")
                    continue
                elif user_input.lower().startswith('mode '):
                    new_mode = user_input[5:].lower()
                    if new_mode in ['chat', 'code', 'finance', 'logic']:
                        self.mode = new_mode
                        print(f"\nSwitched to {self.mode} mode\n")
                    else:
                        print("\nInvalid mode. Choose: chat, code, finance, logic\n")
                    continue
                elif user_input.lower() == 'models':
                    try:
                        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                        print("\n" + result.stdout)
                    except:
                        print("\nCould not list models\n")
                    continue
                elif user_input.lower().startswith('use '):
                    self.model = user_input[4:].strip()
                    print(f"\nSwitched to {self.model}\n")
                    continue
                    
                # Process query
                print()  # Space before response
                
                # Try instant first
                response = self.process_instant(user_input)
                source = "instant"
                
                # Fall back to Ollama
                if not response:
                    response = self.process_ollama(user_input)
                    source = "ollama"
                    
                # Print response with wrapping
                if response:
                    # Handle multi-line responses
                    lines = response.split('\n')
                    for line in lines:
                        self.print_wrapped(line)
                    
                    if source == "instant":
                        print("[Instant calculation]\n")
                else:
                    print("No response\n")
                    
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit\n")
            except EOFError:
                break
                
        print("\nGoodbye!\n")

if __name__ == "__main__":
    terminal = CleanTerminal()
    terminal.run()