#!/usr/bin/env python3
"""
Integrated Training Monitor with Resource Optimization
Combines training progress with bandwidth/token/context monitoring
"""

import os
import sys
import time
import threading
import queue
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import resource monitor components
sys.path.append(str(Path(__file__).parent))
from resource_monitor import TokenContextMonitor, BandwidthOptimizer, ResourceMetrics

class IntegratedTrainingMonitor:
    """Combined training and resource monitoring with automatic optimization"""
    
    def __init__(self):
        self.running = True
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.loss = 4.5
        self.learning_rate = 0.01
        
        # Resource monitors
        self.token_monitor = TokenContextMonitor(max_context=262144)
        self.bandwidth_optimizer = BandwidthOptimizer()
        
        # Optimization state
        self.auto_optimize = True
        self.optimization_history = deque(maxlen=100)
        self.bottleneck_alerts = queue.Queue()
        
        # Performance metrics
        self.performance_history = deque(maxlen=1000)
        self.resource_efficiency = 1.0
        
    def display_integrated_dashboard(self):
        """Display comprehensive training and resource dashboard"""
        last_optimization = time.time()
        
        while self.running:
            # Clear screen
            print("\033[2J\033[H")
            
            # Header
            print("\033[36m" + "="*100 + "\033[0m")
            print("\033[1m\033[35m                    MISTRAL ENHANCED - INTEGRATED TRAINING & RESOURCE MONITOR\033[0m")
            print("\033[36m" + "="*100 + "\033[0m\n")
            
            # Create two-column layout
            self._display_training_column()
            print("\033[15A")  # Move cursor up 15 lines
            print("\033[52C")  # Move cursor right 52 columns
            self._display_resource_column()
            print("\033[15B")  # Move cursor down 15 lines
            
            # Bottom section - Optimizations and Alerts
            print("\n" + "\033[36m" + "-"*100 + "\033[0m")
            self._display_optimization_section()
            
            # Auto-optimization check
            current_time = time.time()
            if self.auto_optimize and current_time - last_optimization > 10:
                self._perform_auto_optimization()
                last_optimization = current_time
            
            # Footer
            print(f"\n\033[90mAuto-Optimization: {'ON' if self.auto_optimize else 'OFF'} | ")
            print(f"Resource Efficiency: {self.resource_efficiency:.1%} | ")
            print(f"Last Update: {datetime.now().strftime('%H:%M:%S')}\033[0m")
            print("\033[90mPress 'o' to toggle optimization, Ctrl+C to stop\033[0m")
            
            time.sleep(0.5)
    
    def _display_training_column(self):
        """Display training metrics (left column)"""
        print("\033[33m▶ TRAINING PROGRESS\033[0m")
        print(f"  Epoch: {self.epoch}/3 | Step: {self.step:>5}")
        print(f"  Loss: {self.loss:.4f} | LR: {self.learning_rate:.6f}")
        
        # Progress bar
        progress = (self.epoch - 1) / 3 + (self.step / 1000) / 3
        self._print_progress_bar("Progress", progress, 40)
        
        print("\n\033[33m▶ MODEL PERFORMANCE\033[0m")
        # Simulate model metrics
        accuracy = min(0.98, 0.7 + progress * 0.3)
        perplexity = max(1.2, 10 - progress * 8)
        print(f"  Accuracy:   {accuracy:.1%}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        print("\n\033[33m▶ EXPERT UTILIZATION\033[0m")
        experts = ["Code", "Math", "Reason", "Lang"]
        for expert in experts:
            usage = np.random.uniform(60, 95)
            self._print_mini_bar(f"  {expert:<7}", usage)
    
    def _display_resource_column(self):
        """Display resource metrics (right column)"""
        # Collect current metrics
        bandwidth = self.bandwidth_optimizer.measure_bandwidth()
        tokens_per_sec = np.random.uniform(3000, 5000)
        context_usage = self.token_monitor.current_context / self.token_monitor.max_context * 100
        
        print("\033[52C\033[33m▶ BANDWIDTH & TOKENS\033[0m")
        print(f"\033[52C  Network:  {bandwidth['network_mbps']:>6.1f} Mbps")
        print(f"\033[52C  Tokens/s: {tokens_per_sec:>6.0f}")
        print(f"\033[52C  Memory:   {bandwidth['memory_gbps']:>6.1f} GB/s")
        
        print("\033[52C")
        print("\033[52C\033[33m▶ CONTEXT WINDOW\033[0m")
        print(f"\033[52C  Usage: {context_usage:>6.1f}%")
        print(f"\033[52C  Free:  {100-context_usage:>6.1f}%")
        self._print_context_visualization(context_usage)
        
        print("\033[52C")
        print("\033[52C\033[33m▶ BOTTLENECKS\033[0m")
        bottlenecks = self._detect_bottlenecks(bandwidth, tokens_per_sec, context_usage)
        if bottlenecks:
            for i, b in enumerate(bottlenecks[:3]):
                print(f"\033[52C  \033[31m⚠ {b}\033[0m")
        else:
            print("\033[52C  \033[32m✓ No bottlenecks\033[0m")
    
    def _display_optimization_section(self):
        """Display optimization actions and alerts"""
        print("\033[33m▶ ACTIVE OPTIMIZATIONS\033[0m")
        
        # Show recent optimizations
        if self.optimization_history:
            recent = list(self.optimization_history)[-3:]
            for opt in recent:
                print(f"  • {opt['action']} → {opt['result']}")
        else:
            print("  No optimizations applied yet")
        
        # Show alerts
        alerts = []
        while not self.bottleneck_alerts.empty():
            alerts.append(self.bottleneck_alerts.get())
        
        if alerts:
            print("\n\033[33m▶ ALERTS\033[0m")
            for alert in alerts[-3:]:
                print(f"  \033[33m! {alert}\033[0m")
    
    def _perform_auto_optimization(self):
        """Automatically optimize based on current metrics"""
        # Collect metrics
        bandwidth = self.bandwidth_optimizer.measure_bandwidth()
        tokens_per_sec = np.random.uniform(3000, 5000)
        context_usage = self.token_monitor.current_context / self.token_monitor.max_context * 100
        
        optimization_applied = False
        
        # Context window optimization
        if context_usage > 85:
            self._optimize_context_window()
            optimization_applied = True
        
        # Bandwidth optimization
        if bandwidth['network_mbps'] > 800:
            self._optimize_bandwidth()
            optimization_applied = True
        
        # Token throughput optimization
        if tokens_per_sec < 3500:
            self._optimize_token_throughput()
            optimization_applied = True
        
        # Update efficiency score
        if optimization_applied:
            self._update_efficiency_score()
    
    def _optimize_context_window(self):
        """Optimize context window usage"""
        # Simulate context optimization
        freed = np.random.randint(5000, 15000)
        self.token_monitor.current_context = max(0, self.token_monitor.current_context - freed)
        
        optimization = {
            'timestamp': time.time(),
            'action': 'Context compression',
            'result': f'Freed {freed:,} tokens'
        }
        self.optimization_history.append(optimization)
        self.bottleneck_alerts.put(f"Context optimized: {freed:,} tokens freed")
    
    def _optimize_bandwidth(self):
        """Optimize bandwidth usage"""
        reduction = np.random.uniform(10, 30)
        
        optimization = {
            'timestamp': time.time(),
            'action': 'Bandwidth compression',
            'result': f'Reduced by {reduction:.0f}%'
        }
        self.optimization_history.append(optimization)
    
    def _optimize_token_throughput(self):
        """Optimize token processing"""
        improvement = np.random.uniform(200, 500)
        
        optimization = {
            'timestamp': time.time(),
            'action': 'Batch size adjustment',
            'result': f'+{improvement:.0f} tokens/s'
        }
        self.optimization_history.append(optimization)
    
    def _update_efficiency_score(self):
        """Update overall resource efficiency"""
        # Calculate based on various factors
        bandwidth_eff = min(1.0, 500 / self.bandwidth_optimizer.measure_bandwidth()['network_mbps'])
        context_eff = 1.0 - (self.token_monitor.current_context / self.token_monitor.max_context)
        
        self.resource_efficiency = (bandwidth_eff + context_eff) / 2
    
    def _detect_bottlenecks(self, bandwidth: Dict, tokens_per_sec: float, 
                           context_usage: float) -> List[str]:
        """Detect current bottlenecks"""
        bottlenecks = []
        
        if bandwidth['network_mbps'] > 900:
            bottlenecks.append("High network bandwidth")
        
        if tokens_per_sec < 3000:
            bottlenecks.append("Low token throughput")
        
        if context_usage > 90:
            bottlenecks.append("Context near capacity")
        
        if bandwidth['memory_gbps'] > 45:
            bottlenecks.append("Memory bandwidth saturated")
        
        return bottlenecks
    
    def _print_progress_bar(self, label: str, progress: float, width: int = 40):
        """Print a progress bar"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        print(f"  {label}: |{bar}| {progress:.1%}")
    
    def _print_mini_bar(self, label: str, percentage: float):
        """Print a mini percentage bar"""
        filled = int(percentage / 10)
        bar = '▰' * filled + '▱' * (10 - filled)
        color = '\033[32m' if percentage > 80 else '\033[33m' if percentage > 60 else '\033[31m'
        print(f"{label} {color}{bar} {percentage:>5.1f}%\033[0m")
    
    def _print_context_visualization(self, usage: float):
        """Visualize context window usage"""
        # Create a visual representation of context window
        total_blocks = 20
        used_blocks = int(usage / 100 * total_blocks)
        
        print("\033[52C  [", end='')
        for i in range(total_blocks):
            if i < used_blocks:
                if usage > 90:
                    print("\033[31m█\033[0m", end='')
                elif usage > 70:
                    print("\033[33m█\033[0m", end='')
                else:
                    print("\033[32m█\033[0m", end='')
            else:
                print("░", end='')
        print("]")
    
    def simulate_training(self):
        """Simulate training progress"""
        for epoch in range(1, 4):
            self.epoch = epoch
            
            for step in range(1000):
                self.step = step
                
                # Simulate training metrics
                self.loss = max(0.1, 4.5 - (epoch - 1 + step/1000) * 1.2 + np.random.uniform(-0.1, 0.1))
                self.learning_rate = 0.01 * (0.99 ** (epoch * 1000 + step))
                
                # Simulate token processing
                tokens = ['token'] * np.random.randint(100, 500)
                importance = np.random.uniform(0.5, 1.0)
                self.token_monitor.add_tokens(tokens, importance)
                
                # Simulate bandwidth usage
                data = b'x' * np.random.randint(1000, 10000)
                self.bandwidth_optimizer.optimize(data)
                
                time.sleep(0.05)
                
                if not self.running:
                    return
        
        self.running = False

def main():
    """Run integrated monitor"""
    monitor = IntegratedTrainingMonitor()
    
    # Start training simulation in background
    training_thread = threading.Thread(target=monitor.simulate_training)
    training_thread.daemon = True
    training_thread.start()
    
    try:
        monitor.display_integrated_dashboard()
    except KeyboardInterrupt:
        print("\n\n\033[33mStopping integrated monitor...\033[0m")
        monitor.running = False
        training_thread.join(timeout=1)
        
        # Show final summary
        print("\n\033[36m" + "="*60 + "\033[0m")
        print("\033[1mTRAINING SUMMARY\033[0m")
        print(f"  Final Loss: {monitor.loss:.4f}")
        print(f"  Resource Efficiency: {monitor.resource_efficiency:.1%}")
        print(f"  Optimizations Applied: {len(monitor.optimization_history)}")
        print("\033[36m" + "="*60 + "\033[0m")

if __name__ == "__main__":
    main()