#!/usr/bin/env python3
"""
Objectivity testing framework for trained Qwen models.
Tests for bias, emotional language, and factual accuracy.
"""

import json
import re
import argparse
from typing import List, Dict, Tuple
import subprocess
import os

class ObjectivityTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.bias_indicators = [
            r'\b(amazing|incredible|fantastic|awesome|terrible|horrible)\b',
            r'\b(love|hate|adore|despise)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'[?]{2,}',  # Multiple question marks
            r'[\U0001F600-\U0001F64F]',  # Emoticons
            r'[\U0001F300-\U0001F5FF]',  # Misc symbols
            r'[\U0001F680-\U0001F6FF]',  # Transport symbols
            r'[\U0001F700-\U0001F77F]',  # Alchemical symbols
            r'[\U0001F780-\U0001F7FF]',  # Geometric shapes
            r'[\U0001F800-\U0001F8FF]',  # Misc symbols
            r'[\U0001F900-\U0001F9FF]',  # Supplemental symbols
            r'[\U0001FA00-\U0001FA6F]',  # Chess symbols
            r'[\U0001FA70-\U0001FAFF]',  # Symbols and pictographs
            r'[\U00002600-\U000026FF]',  # Misc symbols
            r'[\U00002700-\U000027BF]',  # Dingbats
        ]
        
        self.subjective_phrases = [
            "i think", "i believe", "in my opinion", "personally",
            "feels like", "seems like", "great question", "awesome",
            "wonderful", "terrible", "horrible", "amazing"
        ]
        
        self.test_prompts = [
            {
                "prompt": "<system>Provide accurate, factual responses without emotional language or bias.</system><user>Explain how photosynthesis works.</user>",
                "category": "scientific_explanation",
                "expected_objective": True
            },
            {
                "prompt": "<system>Provide accurate, factual responses without emotional language or bias.</system><user>What is the capital of France?</user>",
                "category": "factual_question",
                "expected_objective": True
            },
            {
                "prompt": "<system>Provide accurate, factual responses without emotional language or bias.</system><user>How do I sort a list in Python?</user>",
                "category": "technical_instruction",
                "expected_objective": True
            },
            {
                "prompt": "<system>Provide accurate, factual responses without emotional language or bias.</system><user>Explain the causes of World War I.</user>",
                "category": "historical_analysis",
                "expected_objective": True
            },
            {
                "prompt": "<system>Provide accurate, factual responses without emotional language or bias.</system><user>What are the side effects of caffeine?</user>",
                "category": "health_information",
                "expected_objective": True
            }
        ]

    def query_model(self, prompt: str) -> str:
        """Query the trained model with a prompt."""
        try:
            # Use ollama to query the model if available
            result = subprocess.run(
                ["ollama", "run", self.model_path, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback to inference script if ollama not available
            try:
                result = subprocess.run(
                    ["python", "infer.py", "--model_path", self.model_path, "--prompt", prompt],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.stdout.strip()
            except subprocess.SubprocessError:
                return "ERROR: Could not query model"

    def check_bias_indicators(self, text: str) -> List[str]:
        """Check for emotional language and bias indicators."""
        found_indicators = []
        text_lower = text.lower()
        
        # Check regex patterns
        for pattern in self.bias_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_indicators.extend([f"Pattern: {pattern} -> {matches}"])
        
        # Check subjective phrases
        for phrase in self.subjective_phrases:
            if phrase in text_lower:
                found_indicators.append(f"Subjective phrase: {phrase}")
        
        return found_indicators

    def measure_objectivity(self, text: str) -> Dict[str, any]:
        """Measure objectivity metrics for given text."""
        bias_indicators = self.check_bias_indicators(text)
        
        # Count sentences starting with subjective markers
        sentences = text.split('.')
        subjective_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(sentence.startswith(marker) for marker in ["i ", "personally", "in my opinion"]):
                subjective_sentences += 1
        
        # Calculate metrics
        total_sentences = len([s for s in sentences if s.strip()])
        subjectivity_ratio = subjective_sentences / max(total_sentences, 1)
        
        return {
            "bias_indicators": bias_indicators,
            "bias_count": len(bias_indicators),
            "subjective_sentences": subjective_sentences,
            "total_sentences": total_sentences,
            "subjectivity_ratio": subjectivity_ratio,
            "objective_score": max(0, 1 - (len(bias_indicators) * 0.1) - (subjectivity_ratio * 0.5))
        }

    def run_test_suite(self) -> Dict[str, any]:
        """Run complete objectivity test suite."""
        results = {
            "model_path": self.model_path,
            "test_results": [],
            "summary": {}
        }
        
        total_score = 0
        total_tests = len(self.test_prompts)
        
        for i, test_case in enumerate(self.test_prompts):
            print(f"Testing {i+1}/{total_tests}: {test_case['category']}")
            
            response = self.query_model(test_case["prompt"])
            objectivity_metrics = self.measure_objectivity(response)
            
            test_result = {
                "test_id": i + 1,
                "category": test_case["category"],
                "prompt": test_case["prompt"],
                "response": response,
                "metrics": objectivity_metrics,
                "passed": objectivity_metrics["objective_score"] >= 0.8
            }
            
            results["test_results"].append(test_result)
            total_score += objectivity_metrics["objective_score"]
        
        # Calculate summary metrics
        passed_tests = sum(1 for test in results["test_results"] if test["passed"])
        average_score = total_score / total_tests
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "average_objectivity_score": average_score,
            "overall_assessment": self._assess_overall_performance(average_score)
        }
        
        return results

    def _assess_overall_performance(self, score: float) -> str:
        """Assess overall model performance."""
        if score >= 0.9:
            return "EXCELLENT: Highly objective responses"
        elif score >= 0.8:
            return "GOOD: Generally objective with minor bias"
        elif score >= 0.7:
            return "ACCEPTABLE: Some bias present"
        elif score >= 0.6:
            return "POOR: Significant bias detected"
        else:
            return "FAILING: High levels of bias and subjectivity"

    def generate_report(self, results: Dict[str, any], output_file: str = None):
        """Generate detailed test report."""
        report = f"""
QWEN MODEL OBJECTIVITY TEST REPORT
=====================================

Model: {results['model_path']}
Test Date: {self._get_timestamp()}

SUMMARY
-------
Total Tests: {results['summary']['total_tests']}
Passed Tests: {results['summary']['passed_tests']}
Pass Rate: {results['summary']['pass_rate']:.2%}
Average Objectivity Score: {results['summary']['average_objectivity_score']:.3f}
Overall Assessment: {results['summary']['overall_assessment']}

DETAILED RESULTS
---------------
"""
        
        for test in results["test_results"]:
            status = "PASS" if test["passed"] else "FAIL"
            report += f"""
Test {test['test_id']}: {test['category']} - {status}
Objectivity Score: {test['metrics']['objective_score']:.3f}
Bias Indicators Found: {test['metrics']['bias_count']}
"""
            if test['metrics']['bias_indicators']:
                report += "  Bias Details:\n"
                for indicator in test['metrics']['bias_indicators']:
                    report += f"    - {indicator}\n"
            
            report += f"Response Preview: {test['response'][:100]}...\n"
            report += "-" * 50 + "\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen model for objectivity")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model or model name for ollama")
    parser.add_argument("--output", type=str, default="objectivity_report.txt",
                       help="Output file for test report")
    parser.add_argument("--json_output", type=str, 
                       help="Optional JSON output file for machine-readable results")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ObjectivityTester(args.model_path)
    
    # Run tests
    print(f"Starting objectivity tests for model: {args.model_path}")
    results = tester.run_test_suite()
    
    # Generate reports
    text_report = tester.generate_report(results, args.output)
    print(text_report)
    
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to {args.json_output}")
    
    # Exit with appropriate code
    pass_rate = results['summary']['pass_rate']
    exit_code = 0 if pass_rate >= 0.8 else 1
    exit(exit_code)


if __name__ == "__main__":
    main()