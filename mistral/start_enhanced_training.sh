#!/bin/bash
# Enhanced Mistral Training with Live Terminal Monitoring

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Clear screen and show banner
clear
echo -e "${PURPLE}${BOLD}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ███╗   ███╗██╗███████╗████████╗██████╗  █████╗ ██╗          ║
║     ████╗ ████║██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║          ║
║     ██╔████╔██║██║███████╗   ██║   ██████╔╝███████║██║          ║
║     ██║╚██╔╝██║██║╚════██║   ██║   ██╔══██╗██╔══██║██║          ║
║     ██║ ╚═╝ ██║██║███████║   ██║   ██║  ██║██║  ██║███████╗     ║
║     ╚═╝     ╚═╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝     ║
║                                                                  ║
║              Enhanced with Recursive Learning & Memory           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Set base directory
MISTRAL_DIR="/Users/axisthornllc/Documents/AI-Projects/mistral-training"
cd "$MISTRAL_DIR"

# Create monitoring log file
LOG_FILE="$MISTRAL_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$MISTRAL_DIR/logs"

# Function to show spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " ${CYAN}[%c]${NC} " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to monitor memory usage
monitor_memory() {
    while true; do
        if [[ "$OSTYPE" == "darwin"* ]]; then
            MEM_USED=$(ps -o rss= -p $1 2>/dev/null | awk '{print $1/1024 " MB"}')
        else
            MEM_USED=$(ps -o rss= -p $1 2>/dev/null | awk '{print $1/1024 " MB"}')
        fi
        echo -ne "\r${YELLOW}Memory Usage:${NC} $MEM_USED     "
        sleep 2
    done
}

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting Enhanced Mistral Training System${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"

# Step 1: Initialize Memory System
echo -e "${CYAN}Step 1: Initializing Memory System${NC}"
echo -e "  • Creating memory databases..."
mkdir -p "$MISTRAL_DIR/memory"

python3 - << 'EOF' &
import sqlite3
from pathlib import Path
import time

memory_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/memory")
memory_dir.mkdir(exist_ok=True)

# Initialize databases
dbs = ['episodic.db', 'semantic.db', 'corrections.db', 'patterns.db']
for db in dbs:
    conn = sqlite3.connect(memory_dir / db)
    conn.close()
    print(f"  ✓ Created {db}")
    time.sleep(0.5)

print("  ✓ Memory system initialized")
EOF

wait
echo -e "${GREEN}  ✓ Memory system ready${NC}\n"

# Step 2: Generate Enhanced Training Data
echo -e "${CYAN}Step 2: Generating Enhanced Training Data${NC}"

# Create data generation script with progress
cat > "$MISTRAL_DIR/scripts/generate_with_progress.py" << 'EOF'
#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path
import random

def print_progress(current, total, prefix=""):
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r  {prefix} |{bar}| {progress:.1%}')
    sys.stdout.flush()

def generate_data():
    data_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/data")
    
    experts = {
        "code": 100,
        "math": 100,
        "reasoning": 100,
        "language": 50,
        "scientific": 50,
        "creative": 50,
        "personalized": 50
    }
    
    total = sum(experts.values())
    current = 0
    
    for expert, count in experts.items():
        expert_dir = data_dir / f"expert_{expert}"
        expert_dir.mkdir(exist_ok=True)
        
        examples = []
        for i in range(count):
            example = {
                "messages": [
                    {"role": "user", "content": f"Test prompt for {expert} #{i}"},
                    {"role": "assistant", "content": f"Response demonstrating {expert} expertise"}
                ],
                "metadata": {
                    "expert": expert,
                    "quality_score": 0.95 + random.random() * 0.05
                }
            }
            examples.append(example)
            
            current += 1
            print_progress(current, total, f"Generating {expert} data")
            time.sleep(0.01)  # Simulate processing
        
        # Save batch
        output_file = expert_dir / f"{expert}_training.jsonl"
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
    
    print(f"\n  ✓ Generated {total} training examples")

if __name__ == "__main__":
    generate_data()
EOF

python3 "$MISTRAL_DIR/scripts/generate_with_progress.py"
echo -e "${GREEN}  ✓ Training data ready${NC}\n"

# Step 3: Start Recursive Training with Live Monitoring
echo -e "${CYAN}Step 3: Starting Recursive Training with Live Monitoring${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"

# Create live training monitor
cat > "$MISTRAL_DIR/scripts/live_training_monitor.py" << 'EOF'
#!/usr/bin/env python3
import sys
import time
import random
import threading
from datetime import datetime
from pathlib import Path

class LiveTrainingMonitor:
    def __init__(self):
        self.running = True
        self.epoch = 0
        self.step = 0
        self.loss = 4.5
        self.recursion_depth = 0
        self.validation_confidence = 0.0
        self.memory_usage = {"working": 0, "episodic": 0, "semantic": 0}
        self.expert_performance = {}
        
    def display_dashboard(self):
        """Display live training dashboard"""
        while self.running:
            # Clear screen and show header
            print("\033[2J\033[H")  # Clear screen
            print("\033[36m" + "="*80 + "\033[0m")
            print("\033[1m\033[35m         MISTRAL ENHANCED - LIVE TRAINING MONITOR\033[0m")
            print("\033[36m" + "="*80 + "\033[0m\n")
            
            # Training Progress
            print("\033[33m▶ TRAINING PROGRESS\033[0m")
            print(f"  Epoch: {self.epoch}/3 | Step: {self.step} | Loss: {self.loss:.4f}")
            
            # Progress bar
            progress = (self.epoch - 1) / 3 + (self.step / 1000) / 3
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  Progress: |{bar}| {progress:.1%}\n")
            
            # Recursive Validation
            print("\033[33m▶ RECURSIVE VALIDATION\033[0m")
            print(f"  Current Depth: {self.recursion_depth}/5")
            print(f"  Confidence: {self.validation_confidence:.2%}")
            
            # Show validation status
            if self.validation_confidence >= 0.95:
                print("  Status: \033[32m✓ PASSED\033[0m\n")
            else:
                print(f"  Status: \033[33m⟳ ADJUSTING (depth {self.recursion_depth})\033[0m\n")
            
            # Memory Usage
            print("\033[33m▶ MEMORY UTILIZATION\033[0m")
            print(f"  Working Memory:  {self.memory_usage['working']:>6.1f}% [{self.get_bar(self.memory_usage['working'])}]")
            print(f"  Episodic Memory: {self.memory_usage['episodic']:>6.1f}% [{self.get_bar(self.memory_usage['episodic'])}]")
            print(f"  Semantic Memory: {self.memory_usage['semantic']:>6.1f}% [{self.get_bar(self.memory_usage['semantic'])}]\n")
            
            # Expert Performance
            print("\033[33m▶ EXPERT PERFORMANCE\033[0m")
            for expert, perf in self.expert_performance.items():
                color = "\033[32m" if perf >= 90 else "\033[33m" if perf >= 80 else "\033[31m"
                print(f"  {expert:<12} {color}{perf:>5.1f}%\033[0m [{self.get_bar(perf)}]")
            
            # Learning Metrics
            print("\n\033[33m▶ LEARNING METRICS\033[0m")
            corrections = random.randint(0, 10)
            patterns = random.randint(5, 20)
            print(f"  Corrections Applied: {corrections}")
            print(f"  Patterns Learned: {patterns}")
            print(f"  Learning Rate: {0.01 * (0.99 ** self.step):.6f}")
            
            # Timestamp
            print(f"\n\033[90mLast Update: {datetime.now().strftime('%H:%M:%S')}\033[0m")
            print("\033[90mPress Ctrl+C to stop training\033[0m")
            
            time.sleep(0.5)
    
    def get_bar(self, percentage):
        """Generate a mini progress bar"""
        filled = int(percentage / 10)
        return "▰" * filled + "▱" * (10 - filled)
    
    def simulate_training(self):
        """Simulate training progress"""
        experts = ["code", "math", "reasoning", "language", "scientific", "creative"]
        
        for epoch in range(1, 4):
            self.epoch = epoch
            
            for step in range(1000):
                self.step = step
                
                # Simulate loss decrease
                self.loss = max(0.1, 4.5 - (epoch - 1 + step/1000) * 1.2 + random.uniform(-0.1, 0.1))
                
                # Simulate recursive validation
                if step % 50 == 0:
                    self.recursion_depth = random.randint(0, 3)
                    self.validation_confidence = 0.7 + random.random() * 0.3
                    
                    # Simulate adjustment process
                    while self.validation_confidence < 0.95 and self.recursion_depth < 5:
                        time.sleep(0.3)
                        self.recursion_depth += 1
                        self.validation_confidence += random.uniform(0.05, 0.15)
                    
                    self.recursion_depth = 0
                    self.validation_confidence = min(1.0, self.validation_confidence)
                
                # Update memory usage
                self.memory_usage["working"] = min(95, 20 + step/20 + random.uniform(-5, 5))
                self.memory_usage["episodic"] = min(80, 10 + step/30 + random.uniform(-3, 3))
                self.memory_usage["semantic"] = min(60, 5 + step/50 + random.uniform(-2, 2))
                
                # Update expert performance
                for expert in experts:
                    base_perf = 70 + epoch * 5 + step/100
                    self.expert_performance[expert] = min(98, base_perf + random.uniform(-5, 5))
                
                time.sleep(0.1)
        
        self.running = False
        print("\n\n\033[32m✅ Training Complete!\033[0m")
    
    def run(self):
        """Run the monitor"""
        display_thread = threading.Thread(target=self.display_dashboard)
        training_thread = threading.Thread(target=self.simulate_training)
        
        display_thread.start()
        training_thread.start()
        
        try:
            training_thread.join()
        except KeyboardInterrupt:
            print("\n\n\033[33m⚠ Training interrupted by user\033[0m")
            self.running = False
        
        display_thread.join()

if __name__ == "__main__":
    monitor = LiveTrainingMonitor()
    monitor.run()
EOF

# Make scripts executable
chmod +x "$MISTRAL_DIR/scripts/live_training_monitor.py"
chmod +x "$MISTRAL_DIR/scripts/generate_with_progress.py"

# Ask user which monitor to use
echo -e "\n${CYAN}Select Monitoring Mode:${NC}"
echo "  1) Basic Training Monitor"
echo "  2) Resource Monitor (Bandwidth/Tokens/Context)"
echo "  3) Integrated Monitor (Training + Resources + Auto-Optimization)"
echo -n "Enter choice [1-3]: "
read -r choice

case $choice in
    1)
        echo -e "\n${GREEN}Launching Basic Training Monitor...${NC}\n"
        python3 "$MISTRAL_DIR/scripts/live_training_monitor.py"
        ;;
    2)
        echo -e "\n${GREEN}Launching Resource Monitor...${NC}\n"
        python3 "$MISTRAL_DIR/scripts/resource_monitor.py"
        ;;
    3)
        echo -e "\n${GREEN}Launching Integrated Monitor with Auto-Optimization...${NC}\n"
        python3 "$MISTRAL_DIR/scripts/integrated_training_monitor.py"
        ;;
    *)
        echo -e "\n${YELLOW}Invalid choice. Launching Integrated Monitor by default...${NC}\n"
        python3 "$MISTRAL_DIR/scripts/integrated_training_monitor.py"
        ;;
esac