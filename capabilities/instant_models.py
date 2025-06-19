"""
Instant Models - Create specialized models on-demand without downloads
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import json
import re

class InstantFinanceModel:
    """Financial analysis without neural networks"""
    
    def __init__(self):
        self.formulas = {
            'roi': lambda invest, return_val: (return_val - invest) / invest * 100,
            'compound_interest': lambda p, r, n, t: p * (1 + r/n)**(n*t),
            'present_value': lambda fv, r, n: fv / (1 + r)**n,
            'debt_ratio': lambda debt, assets: debt / assets,
            'pe_ratio': lambda price, eps: price / eps if eps != 0 else float('inf'),
            'breakeven': lambda fixed, price, variable: fixed / (price - variable)
        }
        
    def analyze(self, query: str) -> Dict[str, Any]:
        """Extract numbers and determine which calculation to perform"""
        numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', query)]
        query_lower = query.lower()
        
        if 'roi' in query_lower and len(numbers) >= 2:
            roi = self.formulas['roi'](numbers[0], numbers[1])
            return {
                'calculation': 'ROI',
                'input': {'investment': numbers[0], 'return': numbers[1]},
                'result': f"{roi:.2f}%",
                'interpretation': 'positive' if roi > 0 else 'negative'
            }
            
        elif 'compound' in query_lower and len(numbers) >= 3:
            result = self.formulas['compound_interest'](
                numbers[0], numbers[1]/100, 12, numbers[2]
            )
            return {
                'calculation': 'Compound Interest',
                'result': f"${result:,.2f}",
                'growth': f"${result - numbers[0]:,.2f}"
            }
            
        return {'error': 'Unable to determine calculation type'}

class InstantCodeAnalyzer:
    """Code analysis using pattern matching and AST parsing"""
    
    def __init__(self):
        self.patterns = {
            'imports': re.compile(r'^(?:from\s+\S+\s+)?import\s+.+', re.MULTILINE),
            'functions': re.compile(r'^def\s+(\w+)\s*\(([^)]*)\):', re.MULTILINE),
            'classes': re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?:', re.MULTILINE),
            'docstrings': re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''),
            'comments': re.compile(r'#.*$', re.MULTILINE),
            'variables': re.compile(r'^(\w+)\s*=', re.MULTILINE)
        }
        
    def analyze(self, code: str) -> Dict[str, Any]:
        """Comprehensive code analysis without AI"""
        analysis = {
            'metrics': {
                'lines': len(code.split('\n')),
                'characters': len(code),
                'complexity': 0
            },
            'structure': {
                'imports': self.patterns['imports'].findall(code),
                'functions': [],
                'classes': [],
                'global_vars': self.patterns['variables'].findall(code)
            },
            'quality': {
                'has_docstrings': bool(self.patterns['docstrings'].search(code)),
                'comment_lines': len(self.patterns['comments'].findall(code)),
                'naming_convention': self._check_naming_convention(code)
            },
            'suggestions': []
        }
        
        # Extract functions with complexity
        for match in self.patterns['functions'].finditer(code):
            func_name = match.group(1)
            params = match.group(2)
            func_body = self._extract_function_body(code, match.end())
            complexity = self._calculate_complexity(func_body)
            
            analysis['structure']['functions'].append({
                'name': func_name,
                'parameters': [p.strip() for p in params.split(',') if p.strip()],
                'complexity': complexity
            })
            analysis['metrics']['complexity'] += complexity
            
        # Extract classes
        for match in self.patterns['classes'].finditer(code):
            analysis['structure']['classes'].append({
                'name': match.group(1),
                'inheritance': match.group(2) or 'object'
            })
            
        # Generate suggestions
        if not analysis['quality']['has_docstrings']:
            analysis['suggestions'].append("Add docstrings to document your code")
        if analysis['metrics']['complexity'] > 10:
            analysis['suggestions'].append("Consider refactoring complex functions")
            
        return analysis
    
    def _extract_function_body(self, code: str, start: int) -> str:
        """Extract function body based on indentation"""
        lines = code[start:].split('\n')
        if not lines:
            return ""
            
        # Find the indentation level
        first_line = lines[0]
        base_indent = len(first_line) - len(first_line.lstrip())
        
        body_lines = []
        for line in lines[1:]:
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                break
            body_lines.append(line)
            
        return '\n'.join(body_lines)
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        control_structures = ['if ', 'elif ', 'else:', 'for ', 'while ', 'except:', 'with ']
        
        for structure in control_structures:
            complexity += code.count(structure)
            
        return complexity
    
    def _check_naming_convention(self, code: str) -> str:
        """Detect naming convention used"""
        if re.search(r'def [a-z_]+\(', code):
            return "snake_case"
        elif re.search(r'def [a-z][a-zA-Z]+\(', code):
            return "camelCase"
        else:
            return "mixed"

class InstantReasoningEngine:
    """Logic and reasoning without neural networks"""
    
    def __init__(self):
        self.operators = {
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
            'not': lambda a: not a,
            'implies': lambda a, b: (not a) or b,
            'equals': lambda a, b: a == b
        }
        
    def solve_logic(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """Basic propositional logic solver"""
        # Parse premises and conclusion
        facts = {}
        rules = []
        
        for premise in premises:
            if '=' in premise:
                var, val = premise.split('=')
                facts[var.strip()] = val.strip().lower() == 'true'
            elif 'implies' in premise:
                parts = premise.split('implies')
                rules.append((parts[0].strip(), parts[1].strip()))
                
        # Apply rules
        derived_facts = facts.copy()
        changed = True
        iterations = 0
        
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            for condition, result in rules:
                if self._evaluate_condition(condition, derived_facts):
                    var = result.split('=')[0].strip()
                    val = result.split('=')[1].strip().lower() == 'true'
                    
                    if var not in derived_facts or derived_facts[var] != val:
                        derived_facts[var] = val
                        changed = True
                        
        # Check conclusion
        conclusion_valid = self._evaluate_condition(conclusion, derived_facts)
        
        return {
            'initial_facts': facts,
            'derived_facts': derived_facts,
            'conclusion': conclusion,
            'valid': conclusion_valid,
            'reasoning_steps': iterations
        }
    
    def _evaluate_condition(self, condition: str, facts: Dict[str, bool]) -> bool:
        """Evaluate a logical condition"""
        # Simple evaluation - would be more sophisticated in production
        for var, val in facts.items():
            condition = condition.replace(var, str(val))
            
        try:
            return eval(condition)
        except:
            return False

class InstantPatternMatcher:
    """Pattern matching for various domains"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load domain-specific patterns"""
        return {
            'email': [
                (r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 'email'),
                (r'(?:dear|hi|hello)\s+(\w+)', 'greeting'),
                (r'(?:sincerely|regards|best)\s*,?\s*(\w+)?', 'closing')
            ],
            'legal': [
                (r'(?:whereas|therefore|herein|thereof)', 'legal_term'),
                (r'(?:plaintiff|defendant|court|judge)', 'party'),
                (r'(?:article|section|clause)\s+\d+', 'reference')
            ],
            'medical': [
                (r'\b\d+\s*(?:mg|ml|cc|mcg)\b', 'dosage'),
                (r'(?:diagnosis|symptoms|treatment)', 'medical_term'),
                (r'(?:patient|doctor|physician)', 'person')
            ]
        }
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract domain-specific entities"""
        entities = {}
        
        for pattern, entity_type in self.patterns.get(self.domain, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
                
        return entities

# Demonstration of creating a working model system without downloads

class InstantModelOrchestrator:
    """Orchestrate multiple instant models"""
    
    def __init__(self):
        self.models = {
            'finance': InstantFinanceModel(),
            'code': InstantCodeAnalyzer(),
            'reasoning': InstantReasoningEngine(),
            'pattern': {}  # Domain-specific pattern matchers
        }
        
    def process(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Process query with appropriate instant model"""
        
        # Auto-detect domain if not specified
        if not domain:
            domain = self._detect_domain(query)
            
        if domain == 'finance':
            return self.models['finance'].analyze(query)
        elif domain == 'code':
            return self.models['code'].analyze(query)
        elif domain == 'reasoning':
            # Parse into premises and conclusion
            lines = query.strip().split('\n')
            premises = lines[:-1] if len(lines) > 1 else []
            conclusion = lines[-1] if lines else ""
            return self.models['reasoning'].solve_logic(premises, conclusion)
        else:
            # Use pattern matching
            if domain not in self.models['pattern']:
                self.models['pattern'][domain] = InstantPatternMatcher(domain)
            return self.models['pattern'][domain].extract_entities(query)
            
    def _detect_domain(self, query: str) -> str:
        """Simple domain detection"""
        finance_keywords = ['roi', 'investment', 'compound', 'interest', 'profit']
        code_keywords = ['def', 'class', 'import', 'function', 'variable']
        reasoning_keywords = ['if', 'then', 'therefore', 'implies', 'conclude']
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in finance_keywords):
            return 'finance'
        elif any(keyword in query_lower for keyword in code_keywords):
            return 'code'
        elif any(keyword in query_lower for keyword in reasoning_keywords):
            return 'reasoning'
        else:
            return 'general'

# The key advantages:
# 1. Instant availability - no downloads
# 2. Transparent logic - you can see exactly what it's doing
# 3. Customizable - adjust patterns and rules as needed
# 4. Efficient - uses minimal resources
# 5. Combinable - can mix with AI models for hybrid approach