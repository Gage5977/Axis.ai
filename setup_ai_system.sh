#!/bin/bash

echo "Setting up Unified AI System..."

# Create necessary directories
mkdir -p capabilities/{coding,reasoning,general,finance,vision,embedding,creative}

# Make scripts executable
chmod +x capabilities/unified_ai_system.py

# Create a simple launcher
cat > run_ai.py << 'EOF'
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
EOF

chmod +x run_ai.py

echo "Testing instant models..."
python3 -c "
import sys
sys.path.append('capabilities')
from instant_models import InstantFinanceModel, InstantCodeAnalyzer

# Test finance
finance = InstantFinanceModel()
result = finance.analyze('Calculate ROI for investment of 10000 with return of 12500')
print('Finance test:', result)

# Test code analysis
analyzer = InstantCodeAnalyzer()
code = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)'''
result = analyzer.analyze(code)
print('Code analysis test:', result)
"

echo -e "\n✅ Setup complete!"
echo "You now have:"
echo "1. Instant financial calculator (no download needed)"
echo "2. Instant code analyzer (no download needed)"
echo "3. Instant logic reasoner (no download needed)"
echo "4. Ability to use local Ollama models when available"
echo "5. Framework to integrate Claude API when needed"
echo ""
echo "Run './run_ai.py' to start the unified system"