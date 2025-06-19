#!/usr/bin/env python3
"""
Unified AI System - Combines Claude, Local Models, and Programmed Logic
No massive downloads required!
"""

import os
import subprocess
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    FINANCIAL_CALC = "financial_calculation"
    REASONING = "reasoning"
    GENERAL_CHAT = "general_chat"
    VISION = "vision"
    CREATIVE = "creative"

class UnifiedAISystem:
    """
    Intelligent routing between:
    1. Claude API (best quality)
    2. Local Ollama models (privacy/offline)
    3. Programmed logic (instant/transparent)
    """
    
    def __init__(self):
        # Import our instant models
        from instant_models import InstantModelOrchestrator
        from hybrid_model_system import HybridModelSystem
        
        self.instant_models = InstantModelOrchestrator()
        self.hybrid_system = HybridModelSystem()
        
        # Model routing preferences
        self.routing_rules = {
            TaskType.CODE_GENERATION: [
                ("claude_api", 0.9),      # Claude is best at code generation
                ("qwen2.5-coder", 0.7),   # If available locally
                ("programmed", 0.3)       # Basic templates as fallback
            ],
            TaskType.CODE_ANALYSIS: [
                ("programmed", 0.8),      # Our analyzer is instant and good
                ("claude_api", 0.9),      # Claude for complex analysis
                ("local", 0.6)            # Local models as backup
            ],
            TaskType.FINANCIAL_CALC: [
                ("programmed", 0.95),     # Instant and accurate for calculations
                ("finance-assistant", 0.8), # Our fine-tuned model
                ("claude_api", 0.7)       # Claude for complex scenarios
            ],
            TaskType.REASONING: [
                ("deepseek-r1", 0.9),     # If we have it locally
                ("programmed", 0.6),      # Basic logic solver
                ("claude_api", 0.95)      # Claude for complex reasoning
            ]
        }
        
        # Track performance for adaptive routing
        self.performance_history = {}
        
    def process(self, 
                query: str, 
                task_type: Optional[TaskType] = None,
                prefer_local: bool = True,
                require_fast: bool = False) -> Dict[str, Any]:
        """
        Main entry point - intelligently routes queries
        """
        
        # Detect task type if not specified
        if not task_type:
            task_type = self._detect_task_type(query)
            
        # Get routing preferences
        routes = self.routing_rules.get(task_type, [("claude_api", 0.5)])
        
        # Adjust based on requirements
        if require_fast:
            routes = [(r, s) for r, s in routes if r == "programmed"]
        elif prefer_local:
            routes = [(r, s) for r, s in routes if r != "claude_api"]
            
        # Try routes in order of preference
        for route, confidence in sorted(routes, key=lambda x: x[1], reverse=True):
            try:
                result = self._execute_route(query, route, task_type)
                if result and not result.get('error'):
                    result['route_used'] = route
                    result['confidence'] = confidence
                    self._record_performance(task_type, route, True)
                    return result
            except Exception as e:
                self._record_performance(task_type, route, False)
                continue
                
        return {"error": "All routes failed", "task_type": task_type.value}
    
    def _detect_task_type(self, query: str) -> TaskType:
        """Detect the type of task from the query"""
        query_lower = query.lower()
        
        # Code-related
        if any(word in query_lower for word in ['function', 'code', 'implement', 'debug', 'class', 'def']):
            if any(word in query_lower for word in ['analyze', 'review', 'check']):
                return TaskType.CODE_ANALYSIS
            return TaskType.CODE_GENERATION
            
        # Financial
        if any(word in query_lower for word in ['roi', 'investment', 'compound', 'profit', 'cost']):
            return TaskType.FINANCIAL_CALC
            
        # Reasoning
        if any(word in query_lower for word in ['if', 'therefore', 'conclude', 'prove', 'logic']):
            return TaskType.REASONING
            
        # Vision
        if any(word in query_lower for word in ['image', 'picture', 'see', 'ocr']):
            return TaskType.VISION
            
        return TaskType.GENERAL_CHAT
    
    def _execute_route(self, query: str, route: str, task_type: TaskType) -> Dict[str, Any]:
        """Execute a specific route"""
        
        if route == "programmed":
            # Use our instant models
            if task_type == TaskType.FINANCIAL_CALC:
                return self.instant_models.process(query, 'finance')
            elif task_type == TaskType.CODE_ANALYSIS:
                return self.instant_models.process(query, 'code')
            elif task_type == TaskType.REASONING:
                return self.instant_models.process(query, 'reasoning')
            else:
                return {"error": "No programmed model for this task type"}
                
        elif route == "claude_api":
            # Placeholder for Claude API integration
            return {
                "info": "Claude API would handle this",
                "recommendation": "Use Claude for best results on " + task_type.value
            }
            
        elif route in ["deepseek-r1", "qwen2.5-coder", "finance-assistant"]:
            # Try local Ollama model
            return self._run_ollama(query, route)
            
        else:
            # Try any available local model
            return self._run_ollama(query, "mistral")
    
    def _run_ollama(self, query: str, model: str) -> Dict[str, Any]:
        """Run a query through Ollama"""
        try:
            # Check if model exists
            list_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            if model not in list_result.stdout:
                return {"error": f"Model {model} not installed"}
                
            # Run the query
            result = subprocess.run(
                ["ollama", "run", model, query],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "response": result.stdout.strip(),
                "model": model,
                "type": "ollama"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _record_performance(self, task_type: TaskType, route: str, success: bool):
        """Track performance for adaptive routing"""
        key = f"{task_type.value}_{route}"
        if key not in self.performance_history:
            self.performance_history[key] = {"success": 0, "failure": 0}
            
        if success:
            self.performance_history[key]["success"] += 1
        else:
            self.performance_history[key]["failure"] += 1

# Smart Model Builder - Create models on demand

class SmartModelBuilder:
    """Build specialized models without downloading large weights"""
    
    @staticmethod
    def create_minimal_model(task: str) -> str:
        """Generate code for a minimal model for the task"""
        
        templates = {
            "text_classifier": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalTextClassifier(nn.Module):
    def __init__(self, vocab_size=10000, num_classes=2):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, 100)
        self.fc = nn.Linear(100, num_classes)
        
    def forward(self, text, offsets):
        embedded = self.embed(text, offsets)
        return self.fc(embedded)

# Training data can be generated programmatically
# Model trains in minutes, not hours
""",
            "sequence_tagger": """
import torch
import torch.nn as nn

class MinimalSequenceTagger(nn.Module):
    def __init__(self, vocab_size=10000, tagset_size=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, tagset_size)
        
    def forward(self, sentence):
        embeds = self.embed(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.fc(lstm_out)
        return F.log_softmax(tag_space, dim=2)
""",
            "feature_extractor": """
# No neural network needed - use programmed feature extraction
import numpy as np
from collections import Counter

class ProgrammedFeatureExtractor:
    def extract_features(self, text):
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'unique_words': len(set(text.lower().split())),
            'avg_word_length': np.mean([len(w) for w in text.split()]),
            'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text)
        }
        return features
"""
        }
        
        return templates.get(task, "# No template available for this task")

# Example usage script

def demonstrate_system():
    """Show how the unified system works"""
    
    system = UnifiedAISystem()
    
    # Example queries
    queries = [
        ("Calculate ROI for $10000 investment returning $12500", TaskType.FINANCIAL_CALC),
        ("Analyze this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", TaskType.CODE_ANALYSIS),
        ("If A implies B, and B implies C, does A imply C?", TaskType.REASONING),
        ("Write a Python function to reverse a string", TaskType.CODE_GENERATION)
    ]
    
    for query, task_type in queries:
        print(f"\nQuery: {query}")
        print(f"Task Type: {task_type.value}")
        
        # Try fast programmed models first
        result = system.process(query, task_type, require_fast=True)
        if not result.get('error'):
            print(f"Result (fast): {result}")
        else:
            # Fall back to best available
            result = system.process(query, task_type)
            print(f"Result (best): {result}")

if __name__ == "__main__":
    # Create a model on demand
    builder = SmartModelBuilder()
    print("Creating text classifier code:")
    print(builder.create_minimal_model("text_classifier"))
    
    # Demonstrate the unified system
    demonstrate_system()