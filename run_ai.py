#!/usr/bin/env python3
"""
Quick launcher for the Unified AI System
"""

import sys
sys.path.append('capabilities')

from unified_ai_system import UnifiedAISystem, TaskType

def main():
    system = UnifiedAISystem()
    
    print("Unified AI System Ready!")
    print("=" * 50)
    print("Available capabilities:")
    print("- Financial calculations (instant, no download)")
    print("- Code analysis (instant, no download)")
    print("- Logic reasoning (instant, no download)")
    print("- General chat (via Ollama models)")
    print("- Code generation (best with Claude API)")
    print("=" * 50)
    
    while True:
        query = input("\nEnter query (or 'quit'): ").strip()
        if query.lower() in ['quit', 'exit']:
            break
            
        # Let system auto-detect task type
        result = system.process(query, prefer_local=True)
        
        print(f"\nRoute used: {result.get('route_used', 'unknown')}")
        print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    import json
    main()
