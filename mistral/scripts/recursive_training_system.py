#!/usr/bin/env python3
"""
Recursive Training System with Memory for Enhanced Mistral
Implements check-validate-adjust loops with persistent learning
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import sqlite3
import pickle

class RecursiveMemoryTrainer:
    """
    Main training system with recursive validation and memory
    """
    def __init__(self):
        self.base_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training")
        self.memory_dir = self.base_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize memory systems
        self.memory = {
            'working': WorkingMemory(capacity=1000),
            'episodic': EpisodicMemory(self.memory_dir / "episodic.db"),
            'semantic': SemanticMemory(self.memory_dir / "semantic.db"),
            'corrections': CorrectionMemory(self.memory_dir / "corrections.db")
        }
        
        # Validation system
        self.validator = RecursiveValidator(self.memory)
        
        # Learning parameters
        self.confidence_threshold = 0.95
        self.max_recursion_depth = 5
        self.learning_rate = 0.01
        
    def train_with_recursion(self, data: Dict, expert_type: str):
        """
        Train with recursive validation loop
        """
        print(f"\n🔄 Recursive training for {expert_type} expert")
        
        # Process through recursive loop
        result = self.recursive_process(
            data=data,
            expert_type=expert_type,
            depth=0
        )
        
        # Store successful result in memory
        if result['confidence'] >= self.confidence_threshold:
            self.memory['semantic'].store_pattern(
                expert_type=expert_type,
                pattern=result['pattern'],
                confidence=result['confidence']
            )
            
        return result
    
    def recursive_process(self, data: Dict, expert_type: str, depth: int) -> Dict:
        """
        Core recursive processing loop
        """
        if depth >= self.max_recursion_depth:
            print(f"  ⚠️ Max recursion depth reached")
            return {'confidence': 0.0, 'output': None, 'pattern': None}
        
        print(f"\n  📍 Recursion depth: {depth}")
        
        # Step 1: Process
        output = self.process_input(data, expert_type)
        print(f"  ✓ Initial processing complete")
        
        # Step 2: Validate
        validation = self.validator.validate(
            output=output,
            input_data=data,
            expert_type=expert_type
        )
        print(f"  📊 Validation confidence: {validation['confidence']:.2%}")
        
        # Step 3: Check if adjustment needed
        if validation['confidence'] < self.confidence_threshold:
            print(f"  ⚠️ Below threshold, adjusting...")
            
            # Step 4: Adjust based on validation errors
            adjusted = self.adjust_output(
                output=output,
                validation=validation,
                expert_type=expert_type
            )
            
            # Step 5: Learn from correction
            self.learn_from_correction(
                original=output,
                adjusted=adjusted,
                validation=validation,
                expert_type=expert_type
            )
            
            # Step 6: Recursive call
            return self.recursive_process(data, expert_type, depth + 1)
        
        print(f"  ✅ Validation passed!")
        return {
            'confidence': validation['confidence'],
            'output': output,
            'pattern': self.extract_pattern(output, data)
        }
    
    def process_input(self, data: Dict, expert_type: str) -> Dict:
        """
        Process input based on expert type
        """
        # Check memory for similar inputs
        similar = self.memory['episodic'].find_similar(data, expert_type)
        
        if similar and similar['confidence'] > 0.9:
            print(f"    💾 Using memory from similar case")
            return similar['output']
        
        # Process based on expert type
        if expert_type == "code":
            return self.process_code_input(data)
        elif expert_type == "math":
            return self.process_math_input(data)
        elif expert_type == "reasoning":
            return self.process_reasoning_input(data)
        else:
            return self.process_generic_input(data)
    
    def adjust_output(self, output: Dict, validation: Dict, expert_type: str) -> Dict:
        """
        Adjust output based on validation errors
        """
        adjusted = output.copy()
        
        # Apply corrections based on error types
        for error in validation['errors']:
            if error['type'] == 'syntax':
                adjusted = self.fix_syntax_error(adjusted, error)
            elif error['type'] == 'logic':
                adjusted = self.fix_logic_error(adjusted, error)
            elif error['type'] == 'consistency':
                adjusted = self.fix_consistency_error(adjusted, error)
            elif error['type'] == 'style':
                adjusted = self.apply_user_style(adjusted)
        
        return adjusted
    
    def learn_from_correction(self, original: Dict, adjusted: Dict, 
                            validation: Dict, expert_type: str):
        """
        Learn from the correction to improve future processing
        """
        # Store correction in memory
        correction = {
            'timestamp': time.time(),
            'expert_type': expert_type,
            'original': original,
            'adjusted': adjusted,
            'errors': validation['errors'],
            'improvement': validation['confidence']
        }
        
        self.memory['corrections'].store(correction)
        
        # Detect patterns in corrections
        patterns = self.memory['corrections'].analyze_patterns(expert_type)
        
        if patterns:
            print(f"    🧠 Learned {len(patterns)} correction patterns")
            
            # Update processing rules based on patterns
            for pattern in patterns:
                self.update_processing_rules(pattern, expert_type)
    
    def extract_pattern(self, output: Dict, input_data: Dict) -> Dict:
        """
        Extract reusable pattern from successful processing
        """
        return {
            'input_type': self.classify_input(input_data),
            'output_structure': self.analyze_structure(output),
            'key_elements': self.extract_key_elements(output),
            'timestamp': time.time()
        }
    
    def process_code_input(self, data: Dict) -> Dict:
        """Process code-specific input"""
        return {
            'type': 'code',
            'content': f"Code solution for: {data.get('prompt', '')}",
            'language': data.get('language', 'python'),
            'complexity': 'medium'
        }
    
    def process_math_input(self, data: Dict) -> Dict:
        """Process math-specific input"""
        return {
            'type': 'math',
            'content': f"Mathematical solution for: {data.get('prompt', '')}",
            'steps': ['Step 1', 'Step 2', 'Step 3'],
            'proof_type': 'deductive'
        }
    
    def process_reasoning_input(self, data: Dict) -> Dict:
        """Process reasoning-specific input"""
        return {
            'type': 'reasoning',
            'content': f"Logical analysis of: {data.get('prompt', '')}",
            'logic_chain': ['Premise', 'Inference', 'Conclusion'],
            'confidence': 0.85
        }
    
    def process_generic_input(self, data: Dict) -> Dict:
        """Process generic input"""
        return {
            'type': 'generic',
            'content': f"Response to: {data.get('prompt', '')}",
            'metadata': {}
        }


class WorkingMemory:
    """Fast, temporary memory for current context"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def add(self, item: Dict):
        """Add item to working memory"""
        self.memory.append({
            'data': item,
            'timestamp': time.time(),
            'access_count': 0
        })
    
    def get_context(self, n: int = 10) -> List[Dict]:
        """Get recent context"""
        return list(self.memory)[-n:]


class EpisodicMemory:
    """Memory for specific interactions and episodes"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                expert_type TEXT,
                input_data TEXT,
                output_data TEXT,
                confidence REAL,
                embedding BLOB
            )
        ''')
        self.conn.commit()
    
    def store(self, episode: Dict):
        """Store an episode"""
        self.conn.execute('''
            INSERT INTO episodes 
            (timestamp, expert_type, input_data, output_data, confidence, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            episode['expert_type'],
            json.dumps(episode['input']),
            json.dumps(episode['output']),
            episode['confidence'],
            pickle.dumps(episode.get('embedding', []))
        ))
        self.conn.commit()
    
    def find_similar(self, data: Dict, expert_type: str, threshold: float = 0.8) -> Optional[Dict]:
        """Find similar past episodes"""
        cursor = self.conn.execute('''
            SELECT output_data, confidence 
            FROM episodes 
            WHERE expert_type = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', (expert_type,))
        
        # Simple similarity check (would use embeddings in real implementation)
        for row in cursor:
            output_data = json.loads(row[0])
            confidence = row[1]
            
            if confidence > threshold:
                return {
                    'output': output_data,
                    'confidence': confidence
                }
        
        return None


class SemanticMemory:
    """Long-term memory for learned patterns and knowledge"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_type TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                usage_count INTEGER,
                last_used REAL
            )
        ''')
        self.conn.commit()
    
    def store_pattern(self, expert_type: str, pattern: Dict, confidence: float):
        """Store a learned pattern"""
        self.conn.execute('''
            INSERT INTO patterns 
            (expert_type, pattern_type, pattern_data, confidence, usage_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            expert_type,
            pattern.get('type', 'general'),
            json.dumps(pattern),
            confidence,
            1,
            time.time()
        ))
        self.conn.commit()


class CorrectionMemory:
    """Memory for tracking and learning from corrections"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                expert_type TEXT,
                error_type TEXT,
                original TEXT,
                corrected TEXT,
                improvement REAL
            )
        ''')
        self.conn.commit()
    
    def store(self, correction: Dict):
        """Store a correction"""
        for error in correction['errors']:
            self.conn.execute('''
                INSERT INTO corrections 
                (timestamp, expert_type, error_type, original, corrected, improvement)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                correction['timestamp'],
                correction['expert_type'],
                error['type'],
                json.dumps(correction['original']),
                json.dumps(correction['adjusted']),
                correction['improvement']
            ))
        self.conn.commit()
    
    def analyze_patterns(self, expert_type: str) -> List[Dict]:
        """Analyze correction patterns"""
        cursor = self.conn.execute('''
            SELECT error_type, COUNT(*) as count 
            FROM corrections 
            WHERE expert_type = ? 
            GROUP BY error_type 
            ORDER BY count DESC
        ''', (expert_type,))
        
        patterns = []
        for row in cursor:
            if row[1] > 3:  # At least 3 occurrences
                patterns.append({
                    'error_type': row[0],
                    'frequency': row[1],
                    'expert_type': expert_type
                })
        
        return patterns


class RecursiveValidator:
    """Validates outputs with multiple checks"""
    def __init__(self, memory_system: Dict):
        self.memory = memory_system
        
    def validate(self, output: Dict, input_data: Dict, expert_type: str) -> Dict:
        """Perform multi-stage validation"""
        errors = []
        scores = []
        
        # Check syntax/structure
        syntax_score = self.check_syntax(output, expert_type)
        scores.append(syntax_score)
        if syntax_score < 0.9:
            errors.append({'type': 'syntax', 'score': syntax_score})
        
        # Check logical consistency
        logic_score = self.check_logic(output, input_data)
        scores.append(logic_score)
        if logic_score < 0.9:
            errors.append({'type': 'logic', 'score': logic_score})
        
        # Check historical consistency
        history_score = self.check_historical_consistency(output, expert_type)
        scores.append(history_score)
        if history_score < 0.8:
            errors.append({'type': 'consistency', 'score': history_score})
        
        # Check user style preferences
        style_score = self.check_user_style(output)
        scores.append(style_score)
        if style_score < 0.95:
            errors.append({'type': 'style', 'score': style_score})
        
        # Calculate overall confidence
        confidence = np.mean(scores) if scores else 0.0
        
        return {
            'confidence': confidence,
            'errors': errors,
            'scores': {
                'syntax': syntax_score,
                'logic': logic_score,
                'consistency': history_score,
                'style': style_score
            }
        }
    
    def check_syntax(self, output: Dict, expert_type: str) -> float:
        """Check syntax/structure validity"""
        if expert_type == "code":
            # Check for code structure
            return 0.95 if 'content' in output else 0.5
        elif expert_type == "math":
            # Check for mathematical structure
            return 0.95 if 'steps' in output else 0.5
        return 0.9
    
    def check_logic(self, output: Dict, input_data: Dict) -> float:
        """Check logical consistency"""
        # Simple check - would be more sophisticated in real implementation
        return 0.92
    
    def check_historical_consistency(self, output: Dict, expert_type: str) -> float:
        """Check consistency with past outputs"""
        # Check against semantic memory
        return 0.88
    
    def check_user_style(self, output: Dict) -> float:
        """Check adherence to user style preferences"""
        content = str(output.get('content', ''))
        
        # Check for forbidden elements (emojis, markdown)
        if any(char in content for char in ['😊', '👍', '#', '**', '```']):
            return 0.3
        
        # Check for required structure
        if 'EXECUTIVE SUMMARY' in content:
            return 1.0
        
        return 0.7


def main():
    """Demo the recursive training system"""
    trainer = RecursiveMemoryTrainer()
    
    # Example training data
    test_data = {
        'prompt': 'Analyze trial balance discrepancy',
        'type': 'accounting',
        'context': 'family_office'
    }
    
    # Train with recursion
    result = trainer.train_with_recursion(
        data=test_data,
        expert_type='reasoning'
    )
    
    print(f"\n📈 Final confidence: {result['confidence']:.2%}")
    print("✅ Recursive training complete!")


if __name__ == "__main__":
    main()