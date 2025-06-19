#!/usr/bin/env python3
"""
Resource Monitor for Enhanced Mistral
Tracks bandwidth, tokens, context usage, and provides optimization
"""

import time
import psutil
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import threading
import json

@dataclass
class ResourceMetrics:
    """Container for resource metrics"""
    timestamp: float
    bandwidth_mbps: float
    tokens_per_second: float
    context_usage_percent: float
    memory_bandwidth_gbps: float
    gpu_utilization: float
    token_efficiency: float

class TokenContextMonitor:
    """Monitors token usage and context window efficiency"""
    
    def __init__(self, max_context: int = 262144):  # 256K tokens
        self.max_context = max_context
        self.current_context = 0
        self.token_buffer = deque(maxlen=1000)
        self.context_map = {}
        self.fragmentation = 0.0
        
    def add_tokens(self, tokens: List[str], importance: float = 1.0):
        """Add tokens with importance scoring"""
        token_count = len(tokens)
        
        # Check if we need to evict tokens
        if self.current_context + token_count > self.max_context:
            self._evict_tokens(token_count)
        
        # Add tokens with metadata
        entry = {
            'tokens': tokens,
            'count': token_count,
            'importance': importance,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        self.token_buffer.append(entry)
        self.current_context += token_count
        
        # Update fragmentation metric
        self._calculate_fragmentation()
        
    def _evict_tokens(self, needed_space: int):
        """Intelligent token eviction based on importance and recency"""
        evicted = 0
        
        # Sort by importance and age
        candidates = sorted(
            self.token_buffer,
            key=lambda x: x['importance'] * (1 / (time.time() - x['timestamp'] + 1))
        )
        
        while evicted < needed_space and candidates:
            entry = candidates.pop(0)
            self.current_context -= entry['count']
            evicted += entry['count']
            self.token_buffer.remove(entry)
    
    def _calculate_fragmentation(self):
        """Calculate context window fragmentation"""
        if not self.token_buffer:
            self.fragmentation = 0.0
            return
            
        # Check for gaps in importance/usage patterns
        importances = [e['importance'] for e in self.token_buffer]
        if len(importances) > 1:
            variance = np.var(importances)
            self.fragmentation = min(variance * 10, 1.0)
    
    def get_efficiency(self) -> float:
        """Calculate context window efficiency"""
        if self.max_context == 0:
            return 0.0
            
        usage_ratio = self.current_context / self.max_context
        fragmentation_penalty = self.fragmentation * 0.3
        
        # Calculate weighted access frequency
        if self.token_buffer:
            avg_access = np.mean([e['access_count'] for e in self.token_buffer])
            access_bonus = min(avg_access / 10, 0.2)
        else:
            access_bonus = 0.0
            
        efficiency = usage_ratio * (1 - fragmentation_penalty) + access_bonus
        return min(max(efficiency, 0.0), 1.0)

class BandwidthOptimizer:
    """Optimizes data transfer and memory bandwidth"""
    
    def __init__(self):
        self.transfer_history = deque(maxlen=100)
        self.optimization_strategies = {
            'compression': self._apply_compression,
            'batching': self._apply_batching,
            'caching': self._apply_caching,
            'pruning': self._apply_pruning
        }
        self.cache = {}
        
    def measure_bandwidth(self) -> Dict[str, float]:
        """Measure current bandwidth usage"""
        net_io = psutil.net_io_counters()
        
        # Calculate network bandwidth
        time.sleep(0.1)
        net_io_2 = psutil.net_io_counters()
        
        bytes_sent = net_io_2.bytes_sent - net_io.bytes_sent
        bytes_recv = net_io_2.bytes_recv - net_io.bytes_recv
        
        bandwidth_mbps = (bytes_sent + bytes_recv) * 8 / 1024 / 1024 / 0.1
        
        # Estimate memory bandwidth (simplified)
        mem_bandwidth_gbps = self._estimate_memory_bandwidth()
        
        return {
            'network_mbps': bandwidth_mbps,
            'memory_gbps': mem_bandwidth_gbps,
            'efficiency': self._calculate_efficiency()
        }
    
    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth usage"""
        # Simplified estimation based on CPU usage and memory access patterns
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Rough estimation: higher CPU usage typically means more memory bandwidth
        estimated_gbps = (cpu_percent / 100) * 50  # Assume 50 GB/s max
        
        return estimated_gbps
    
    def _calculate_efficiency(self) -> float:
        """Calculate bandwidth efficiency"""
        if not self.transfer_history:
            return 1.0
            
        # Analyze transfer patterns
        sizes = [t['size'] for t in self.transfer_history]
        times = [t['time'] for t in self.transfer_history]
        
        if len(sizes) < 2:
            return 1.0
            
        # Check for inefficient small transfers
        small_transfer_ratio = sum(1 for s in sizes if s < 1024) / len(sizes)
        
        # Check for burst patterns
        time_gaps = np.diff(times)
        burst_penalty = np.std(time_gaps) / (np.mean(time_gaps) + 1)
        
        efficiency = 1.0 - (small_transfer_ratio * 0.3) - (burst_penalty * 0.2)
        return max(efficiency, 0.0)
    
    def optimize(self, data: bytes, strategy: str = 'auto') -> Tuple[bytes, Dict]:
        """Apply optimization strategy to data transfer"""
        metrics = {'original_size': len(data)}
        
        if strategy == 'auto':
            # Choose strategy based on current conditions
            if len(data) > 1024 * 1024:  # > 1MB
                strategy = 'compression'
            elif len(self.transfer_history) > 50:
                strategy = 'batching'
            else:
                strategy = 'caching'
        
        if strategy in self.optimization_strategies:
            optimized_data, strategy_metrics = self.optimization_strategies[strategy](data)
            metrics.update(strategy_metrics)
        else:
            optimized_data = data
            
        metrics['final_size'] = len(optimized_data)
        metrics['reduction'] = 1 - (metrics['final_size'] / metrics['original_size'])
        
        # Record transfer
        self.transfer_history.append({
            'size': len(optimized_data),
            'time': time.time(),
            'strategy': strategy
        })
        
        return optimized_data, metrics
    
    def _apply_compression(self, data: bytes) -> Tuple[bytes, Dict]:
        """Apply compression to reduce bandwidth"""
        import zlib
        compressed = zlib.compress(data, level=6)
        return compressed, {'compression_ratio': len(compressed) / len(data)}
    
    def _apply_batching(self, data: bytes) -> Tuple[bytes, Dict]:
        """Batch multiple small transfers"""
        # In real implementation, would batch multiple transfers
        return data, {'batched': False}
    
    def _apply_caching(self, data: bytes) -> Tuple[bytes, Dict]:
        """Cache frequently accessed data"""
        data_hash = hash(data)
        
        if data_hash in self.cache:
            return b'', {'cache_hit': True}
        
        self.cache[data_hash] = data
        # Limit cache size
        if len(self.cache) > 1000:
            self.cache.pop(list(self.cache.keys())[0])
            
        return data, {'cache_hit': False}
    
    def _apply_pruning(self, data: bytes) -> Tuple[bytes, Dict]:
        """Prune unnecessary data"""
        # In real implementation, would remove redundant information
        return data, {'pruned': False}

class ResourceMonitor:
    """Main resource monitoring system"""
    
    def __init__(self):
        self.token_monitor = TokenContextMonitor()
        self.bandwidth_optimizer = BandwidthOptimizer()
        self.metrics_history = deque(maxlen=1000)
        self.running = True
        
        # Thresholds for alerts
        self.thresholds = {
            'bandwidth_mbps': 1000,
            'tokens_per_second': 5000,
            'context_usage': 0.9,
            'memory_bandwidth': 40,
            'gpu_utilization': 0.95
        }
        
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        bandwidth = self.bandwidth_optimizer.measure_bandwidth()
        
        # Simulate token processing (in real implementation, would hook into model)
        tokens_per_second = np.random.uniform(3000, 6000)
        
        # Get context usage
        context_usage = self.token_monitor.current_context / self.token_monitor.max_context
        
        # Simulate GPU utilization
        gpu_utilization = np.random.uniform(0.7, 0.95)
        
        # Calculate token efficiency
        token_efficiency = self.token_monitor.get_efficiency()
        
        metrics = ResourceMetrics(
            timestamp=time.time(),
            bandwidth_mbps=bandwidth['network_mbps'],
            tokens_per_second=tokens_per_second,
            context_usage_percent=context_usage * 100,
            memory_bandwidth_gbps=bandwidth['memory_gbps'],
            gpu_utilization=gpu_utilization,
            token_efficiency=token_efficiency
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_bottlenecks(self, metrics: ResourceMetrics) -> List[Dict]:
        """Identify resource bottlenecks"""
        bottlenecks = []
        
        # Check bandwidth bottleneck
        if metrics.bandwidth_mbps > self.thresholds['bandwidth_mbps']:
            bottlenecks.append({
                'type': 'bandwidth',
                'severity': 'high',
                'value': metrics.bandwidth_mbps,
                'threshold': self.thresholds['bandwidth_mbps'],
                'remedy': 'Enable compression or batching'
            })
        
        # Check token processing bottleneck
        if metrics.tokens_per_second < 1000:
            bottlenecks.append({
                'type': 'token_processing',
                'severity': 'critical',
                'value': metrics.tokens_per_second,
                'remedy': 'Reduce batch size or optimize model'
            })
        
        # Check context window bottleneck
        if metrics.context_usage_percent > 90:
            bottlenecks.append({
                'type': 'context_window',
                'severity': 'high',
                'value': metrics.context_usage_percent,
                'threshold': 90,
                'remedy': 'Implement sliding window or context compression'
            })
        
        # Check memory bandwidth
        if metrics.memory_bandwidth_gbps > self.thresholds['memory_bandwidth']:
            bottlenecks.append({
                'type': 'memory_bandwidth',
                'severity': 'medium',
                'value': metrics.memory_bandwidth_gbps,
                'threshold': self.thresholds['memory_bandwidth'],
                'remedy': 'Optimize memory access patterns'
            })
        
        return bottlenecks
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on metrics"""
        if len(self.metrics_history) < 10:
            return ["Insufficient data for recommendations"]
        
        recent_metrics = list(self.metrics_history)[-10:]
        recommendations = []
        
        # Analyze trends
        avg_bandwidth = np.mean([m.bandwidth_mbps for m in recent_metrics])
        avg_tokens = np.mean([m.tokens_per_second for m in recent_metrics])
        avg_context = np.mean([m.context_usage_percent for m in recent_metrics])
        
        if avg_bandwidth > 800:
            recommendations.append("Enable adaptive compression for high bandwidth usage")
        
        if avg_tokens < 3000:
            recommendations.append("Consider using smaller batch sizes for better latency")
        
        if avg_context > 80:
            recommendations.append("Implement sliding window attention to manage context")
        
        if avg_context < 30:
            recommendations.append("Increase batch size to better utilize context window")
        
        # Check for inefficient patterns
        token_variance = np.var([m.tokens_per_second for m in recent_metrics])
        if token_variance > 1000000:
            recommendations.append("Stabilize token processing rate with dynamic batching")
        
        return recommendations

class ResourceDashboard:
    """Terminal dashboard for resource monitoring"""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.running = True
        
    def display(self):
        """Display resource monitoring dashboard"""
        while self.running:
            # Clear screen
            print("\033[2J\033[H")
            
            # Collect metrics
            metrics = self.monitor.collect_metrics()
            bottlenecks = self.monitor.check_bottlenecks(metrics)
            recommendations = self.monitor.get_optimization_recommendations()
            
            # Header
            print("\033[36m" + "="*80 + "\033[0m")
            print("\033[1m\033[35m       MISTRAL RESOURCE MONITOR - BANDWIDTH & TOKEN OPTIMIZATION\033[0m")
            print("\033[36m" + "="*80 + "\033[0m\n")
            
            # Bandwidth Section
            print("\033[33m▶ BANDWIDTH METRICS\033[0m")
            print(f"  Network:        {metrics.bandwidth_mbps:>8.1f} Mbps")
            print(f"  Memory:         {metrics.memory_bandwidth_gbps:>8.1f} GB/s")
            self._print_gauge("Network Load", metrics.bandwidth_mbps / 1000 * 100)
            print()
            
            # Token Processing Section
            print("\033[33m▶ TOKEN PROCESSING\033[0m")
            print(f"  Tokens/Second:  {metrics.tokens_per_second:>8.0f}")
            print(f"  Efficiency:     {metrics.token_efficiency:>8.1%}")
            self._print_gauge("Processing Rate", metrics.tokens_per_second / 5000 * 100)
            print()
            
            # Context Window Section
            print("\033[33m▶ CONTEXT WINDOW\033[0m")
            print(f"  Usage:          {metrics.context_usage_percent:>8.1f}%")
            print(f"  Fragmentation:  {self.monitor.token_monitor.fragmentation:>8.1%}")
            self._print_gauge("Context Usage", metrics.context_usage_percent)
            print()
            
            # Bottlenecks Section
            if bottlenecks:
                print("\033[33m▶ BOTTLENECKS DETECTED\033[0m")
                for bottleneck in bottlenecks[:3]:
                    severity_color = {
                        'critical': '\033[31m',
                        'high': '\033[33m',
                        'medium': '\033[36m'
                    }.get(bottleneck['severity'], '\033[0m')
                    
                    print(f"  {severity_color}⚠ {bottleneck['type'].upper()}\033[0m")
                    print(f"    Current: {bottleneck['value']:.1f}, Threshold: {bottleneck.get('threshold', 'N/A')}")
                    print(f"    Remedy: {bottleneck['remedy']}")
                print()
            
            # Recommendations Section
            print("\033[33m▶ OPTIMIZATION RECOMMENDATIONS\033[0m")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
            
            # GPU Section
            print("\n\033[33m▶ COMPUTE UTILIZATION\033[0m")
            print(f"  GPU Usage:      {metrics.gpu_utilization:>8.1%}")
            self._print_gauge("GPU Load", metrics.gpu_utilization * 100)
            
            # Timestamp
            print(f"\n\033[90mLast Update: {time.strftime('%H:%M:%S')}\033[0m")
            print("\033[90mPress Ctrl+C to stop monitoring\033[0m")
            
            time.sleep(1)
    
    def _print_gauge(self, label: str, percentage: float):
        """Print a visual gauge"""
        percentage = min(max(percentage, 0), 100)
        filled = int(percentage / 2)
        bar = '█' * filled + '░' * (50 - filled)
        
        # Color based on percentage
        if percentage > 90:
            color = '\033[31m'  # Red
        elif percentage > 70:
            color = '\033[33m'  # Yellow
        else:
            color = '\033[32m'  # Green
            
        print(f"  {label:<16} {color}|{bar}| {percentage:>5.1f}%\033[0m")
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False

def main():
    """Run resource monitor"""
    monitor = ResourceMonitor()
    dashboard = ResourceDashboard(monitor)
    
    try:
        dashboard.display()
    except KeyboardInterrupt:
        print("\n\n\033[33mStopping resource monitor...\033[0m")
        dashboard.stop()

if __name__ == "__main__":
    main()