#!/usr/bin/env python3
"""Quick start data generation for Enhanced Mistral"""

import json
from pathlib import Path
from datetime import datetime

def generate_initial_data():
    """Generate small high-quality dataset to start"""
    
    data_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/data")
    
    # Code expert examples
    code_examples = [
        {
            "messages": [
                {"role": "user", "content": "Implement a binary search tree in Python with insert and search methods."},
                {"role": "assistant", "content": """I'll implement a binary search tree with insert and search methods.

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

# Example usage
bst = BinarySearchTree()
values = [5, 3, 7, 1, 9, 4, 6]
for val in values:
    bst.insert(val)

print(bst.search(4))  # True
print(bst.search(8))  # False
```

This implementation provides:
- O(log n) average case for insert and search
- O(n) worst case (unbalanced tree)
- Clean recursive structure
- Easy to extend with additional methods"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a function to detect memory leaks in JavaScript."},
                {"role": "assistant", "content": """I'll create a memory leak detector for JavaScript applications.

```javascript
class MemoryLeakDetector {
    constructor() {
        this.snapshots = [];
        this.leakThreshold = 50 * 1024 * 1024; // 50MB
        this.checkInterval = 10000; // 10 seconds
    }
    
    startMonitoring() {
        if (!performance.memory) {
            console.warn('Memory API not available. Use Chrome with --enable-precise-memory-info flag');
            return;
        }
        
        this.intervalId = setInterval(() => {
            this.takeSnapshot();
            this.analyzeMemoryTrend();
        }, this.checkInterval);
    }
    
    takeSnapshot() {
        const snapshot = {
            timestamp: Date.now(),
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
        };
        
        this.snapshots.push(snapshot);
        
        // Keep only last 10 snapshots
        if (this.snapshots.length > 10) {
            this.snapshots.shift();
        }
    }
    
    analyzeMemoryTrend() {
        if (this.snapshots.length < 3) return;
        
        const recent = this.snapshots.slice(-3);
        const avgGrowth = this.calculateAverageGrowth(recent);
        
        if (avgGrowth > this.leakThreshold / 10) {
            this.reportPotentialLeak(avgGrowth);
        }
    }
    
    calculateAverageGrowth(snapshots) {
        let totalGrowth = 0;
        for (let i = 1; i < snapshots.length; i++) {
            totalGrowth += snapshots[i].usedJSHeapSize - snapshots[i-1].usedJSHeapSize;
        }
        return totalGrowth / (snapshots.length - 1);
    }
    
    reportPotentialLeak(growthRate) {
        console.error(`⚠️ Potential memory leak detected!`);
        console.error(`Memory growing at ${(growthRate / 1024 / 1024).toFixed(2)} MB per interval`);
        console.error(`Current heap usage: ${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        
        // Trigger garbage collection if available
        if (global.gc) {
            console.log('Forcing garbage collection...');
            global.gc();
        }
    }
    
    findLeakyObjects() {
        // Advanced leak detection using WeakMap
        const tracker = new WeakMap();
        const suspects = [];
        
        // Override common leak sources
        const originalAddEventListener = EventTarget.prototype.addEventListener;
        EventTarget.prototype.addEventListener = function(type, listener, options) {
            suspects.push({
                type: 'event_listener',
                target: this,
                event: type,
                timestamp: Date.now()
            });
            
            return originalAddEventListener.call(this, type, listener, options);
        };
        
        return suspects;
    }
    
    stopMonitoring() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
}

// Usage example
const detector = new MemoryLeakDetector();
detector.startMonitoring();

// Example of a memory leak
let leakyArray = [];
setInterval(() => {
    // This creates a memory leak
    leakyArray.push(new Array(1000000).fill('leak'));
}, 1000);
```

This detector provides:
- Real-time memory monitoring
- Trend analysis to detect gradual leaks
- Automatic alerts when thresholds exceeded
- Integration points for identifying leak sources"""}
            ]
        }
    ]
    
    # Math expert examples
    math_examples = [
        {
            "messages": [
                {"role": "user", "content": "Prove that the sum of angles in a triangle equals 180 degrees."},
                {"role": "assistant", "content": """I'll prove that the sum of angles in any triangle equals 180°.

**Proof using parallel lines:**

Given: Triangle ABC with angles α, β, and γ

Step 1: Draw a line through vertex A parallel to side BC
Let's call this line DE, where D and E are points on either side of A

Step 2: Identify angle relationships
- Angle DAB = angle ABC (alternate interior angles with parallel lines)
- Angle EAC = angle ACB (alternate interior angles with parallel lines)

Step 3: Observe angles at vertex A
The three angles at vertex A form a straight line:
- Angle DAB (= β)
- Angle BAC (= α) 
- Angle CAE (= γ)

Step 4: Apply the straight angle theorem
Since these three angles form a straight line at A:
β + α + γ = 180°

Therefore: The sum of angles in triangle ABC equals 180°

**Alternative proof using exterior angles:**

Step 1: Extend side BC beyond C to point D
The exterior angle ACD = α + β (exterior angle theorem)

Step 2: Note that angles ACB and ACD are supplementary
γ + (α + β) = 180°

Therefore: α + β + γ = 180°

This fundamental property holds for all triangles in Euclidean geometry. □"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Find the maximum value of f(x) = -x² + 6x - 5 using calculus."},
                {"role": "assistant", "content": """I'll find the maximum value of f(x) = -x² + 6x - 5 using calculus.

**Step 1: Find the derivative**
f(x) = -x² + 6x - 5
f'(x) = -2x + 6

**Step 2: Find critical points**
Set f'(x) = 0:
-2x + 6 = 0
-2x = -6
x = 3

**Step 3: Verify it's a maximum using the second derivative test**
f''(x) = -2

Since f''(x) = -2 < 0 for all x, the critical point at x = 3 is indeed a maximum.

**Step 4: Find the maximum value**
f(3) = -(3)² + 6(3) - 5
     = -9 + 18 - 5
     = 4

**Verification using completing the square:**
f(x) = -x² + 6x - 5
     = -(x² - 6x) - 5
     = -(x² - 6x + 9 - 9) - 5
     = -(x² - 6x + 9) + 9 - 5
     = -(x - 3)² + 4

This confirms:
- Maximum occurs at x = 3
- Maximum value is 4

The parabola opens downward (negative leading coefficient), confirming this is a global maximum."""}
            ]
        }
    ]
    
    # Reasoning expert examples  
    reasoning_examples = [
        {
            "messages": [
                {"role": "user", "content": "A company's sales increased by 20% after hiring a new marketing manager. Can we conclude the manager caused the increase?"},
                {"role": "assistant", "content": """Let me analyze this causal claim systematically.

**Initial Claim**: New marketing manager → 20% sales increase

**Critical Analysis:**

1. **Correlation vs Causation**
   - We observe correlation: Manager hired, sales increased
   - But correlation ≠ causation without further evidence

2. **Alternative Explanations**
   - Seasonal trends (holiday shopping, back-to-school)
   - Market conditions (economic recovery, competitor issues)
   - Product changes (new releases, quality improvements)
   - Other marketing efforts already in progress
   - Industry-wide growth trends

3. **Required Evidence for Causation**
   - Historical data: What were sales trends before?
   - Control comparison: How did similar companies perform?
   - Specific actions: What did the manager actually do?
   - Timeline analysis: When did changes occur relative to hiring?
   - Isolation of variables: Were other changes made simultaneously?

4. **Statistical Considerations**
   - Sample size: Is one period enough?
   - Variance: Normal fluctuation or significant change?
   - Regression to mean: Were sales unusually low before?

**Conclusion**: We cannot definitively conclude the manager caused the increase without:
- Controlling for other variables
- Longer observation period
- Comparison with control groups
- Analysis of specific interventions

**Recommendation**: Conduct a proper analysis using:
- A/B testing for future campaigns
- Time-series analysis with multiple data points
- Attribution modeling for marketing channels
- Difference-in-differences analysis if possible"""}
            ]
        }
    ]
    
    # Save examples
    for expert, examples in [("code", code_examples), ("math", math_examples), ("reasoning", reasoning_examples)]:
        expert_dir = data_dir / f"expert_{expert}"
        expert_dir.mkdir(exist_ok=True)
        
        output_file = expert_dir / f"{expert}_starter.jsonl"
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✓ Created {len(examples)} examples for {expert} expert")
    
    # Create config for Mistral training
    config = {
        "model": "mistral-enhanced",
        "training": {
            "data_path": str(data_dir),
            "output_path": str(data_dir.parent / "models"),
            "num_epochs": 3,
            "batch_size": 1,
            "learning_rate": 5e-5,
            "warmup_steps": 100
        },
        "experts": ["code", "math", "reasoning"],
        "generated_at": datetime.now().isoformat()
    }
    
    with open(data_dir.parent / "configs" / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Initial data generation complete!")
    print(f"📁 Data saved in: {data_dir}")
    print("🚀 Ready to start training Enhanced Mistral")

if __name__ == "__main__":
    generate_initial_data()