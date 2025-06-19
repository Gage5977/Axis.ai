#!/usr/bin/env python3
"""
Convert Qwen-formatted training data to Mistral format with quality filtering
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MistralDataConverter:
    """Convert Qwen format to Mistral format with quality checks"""
    
    def __init__(self):
        self.quality_issues = []
        self.conversion_stats = {
            'total': 0,
            'converted': 0,
            'filtered': 0,
            'tool_calls': 0,
            'simple_qa': 0
        }
    
    def convert_to_mistral_format(self, qwen_data: Dict) -> Optional[Dict]:
        """Convert single example from Qwen to Mistral format"""
        try:
            prompt = qwen_data['prompt']
            response = qwen_data['response']
            
            # Extract system and user messages
            system_match = re.search(r'<system>(.*?)</system>', prompt, re.DOTALL)
            user_match = re.search(r'<user>(.*?)</user>', prompt, re.DOTALL)
            
            if not user_match:
                self.quality_issues.append(f"No user message found in: {prompt[:100]}...")
                return None
            
            # Build Mistral format
            mistral_prompt = ""
            if system_match:
                system_msg = system_match.group(1).strip()
                mistral_prompt = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
                mistral_prompt += f"{user_match.group(1).strip()} [/INST]"
            else:
                mistral_prompt = f"<s>[INST] {user_match.group(1).strip()} [/INST]"
            
            # Convert tool calls
            mistral_response = self._convert_tool_calls(response)
            
            return {
                "messages": [
                    {"role": "user", "content": user_match.group(1).strip()},
                    {"role": "assistant", "content": mistral_response}
                ],
                "system": system_match.group(1).strip() if system_match else None
            }
            
        except Exception as e:
            self.quality_issues.append(f"Conversion error: {e}")
            return None
    
    def _convert_tool_calls(self, response: str) -> str:
        """Convert custom tool call format to Mistral function calling"""
        if "TOOL_CALL:" in response:
            self.conversion_stats['tool_calls'] += 1
            
            # Extract tool call JSON
            tool_match = re.search(r'TOOL_CALL:\s*({.*})', response, re.DOTALL)
            if tool_match:
                try:
                    tool_data = json.loads(tool_match.group(1))
                    
                    # Convert to Mistral function calling format
                    function_call = {
                        "function": {
                            "name": tool_data.get("tool", "unknown"),
                            "arguments": json.dumps(tool_data.get("args", {}))
                        }
                    }
                    
                    # Build response with function call
                    prefix = response[:tool_match.start()].strip()
                    if prefix:
                        return f"{prefix}\n\n[Function Call]\n{json.dumps(function_call, indent=2)}"
                    else:
                        return f"[Function Call]\n{json.dumps(function_call, indent=2)}"
                        
                except json.JSONDecodeError:
                    self.quality_issues.append(f"Invalid tool call JSON: {tool_match.group(1)[:100]}...")
                    
        else:
            self.conversion_stats['simple_qa'] += 1
            
        return response
    
    def quality_check(self, example: Dict) -> Tuple[bool, List[str]]:
        """Check quality of training example"""
        issues = []
        
        # Check for incomplete prompts
        if 'messages' in example:
            user_msg = example['messages'][0]['content']
            if len(user_msg) < 10 or "I'm trying to some" in user_msg:
                issues.append("Incomplete or malformed prompt")
            
            # Check response quality
            response = example['messages'][1]['content']
            if len(response) < 5:
                issues.append("Response too short")
            
            # Check for obvious errors
            if "10*5" in user_msg and "4" in response:
                issues.append("Mathematical error in response")
            
            # Check for mismatched file operations
            if re.search(r'create.*?\.py', user_msg, re.IGNORECASE) and '.txt' in response:
                issues.append("File extension mismatch")
        
        return len(issues) == 0, issues
    
    def convert_file(self, input_path: Path, output_path: Path):
        """Convert entire JSONL file with quality filtering"""
        converted_examples = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                self.conversion_stats['total'] += 1
                
                try:
                    qwen_data = json.loads(line)
                    mistral_data = self.convert_to_mistral_format(qwen_data)
                    
                    if mistral_data:
                        # Quality check
                        is_quality, issues = self.quality_check(mistral_data)
                        
                        if is_quality:
                            converted_examples.append(mistral_data)
                            self.conversion_stats['converted'] += 1
                        else:
                            self.conversion_stats['filtered'] += 1
                            self.quality_issues.extend([f"Line {line_num}: {issue}" for issue in issues])
                    else:
                        self.conversion_stats['filtered'] += 1
                        
                except json.JSONDecodeError as e:
                    self.quality_issues.append(f"Line {line_num}: JSON decode error - {e}")
                    self.conversion_stats['filtered'] += 1
        
        # Write converted data
        with open(output_path, 'w') as f:
            for example in converted_examples:
                f.write(json.dumps(example) + '\n')
        
        return self.conversion_stats
    
    def generate_report(self) -> str:
        """Generate conversion report"""
        report = f"""
# Mistral Data Conversion Report

## Statistics
- Total examples: {self.conversion_stats['total']}
- Successfully converted: {self.conversion_stats['converted']}
- Filtered out: {self.conversion_stats['filtered']}
- Examples with tool calls: {self.conversion_stats['tool_calls']}
- Simple Q&A examples: {self.conversion_stats['simple_qa']}
- Conversion rate: {self.conversion_stats['converted'] / max(1, self.conversion_stats['total']) * 100:.1f}%

## Quality Issues Found
"""
        
        # Sample first 10 issues
        for issue in self.quality_issues[:10]:
            report += f"- {issue}\n"
        
        if len(self.quality_issues) > 10:
            report += f"\n... and {len(self.quality_issues) - 10} more issues\n"
        
        return report


def main():
    """Run conversion on all training files"""
    converter = MistralDataConverter()
    
    data_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/data")
    output_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/mistral_formatted")
    output_dir.mkdir(exist_ok=True)
    
    # Files to convert
    files_to_convert = [
        "agentic_training.jsonl",
        "tool_training.jsonl",
        "objective_training.jsonl",
        "comprehensive_training.jsonl",
        "complete_training_dataset.jsonl",
        "massive_training.jsonl",
        "enhanced_training.jsonl"
    ]
    
    all_stats = {}
    
    for filename in files_to_convert:
        input_path = data_dir / filename
        output_path = output_dir / f"mistral_{filename}"
        
        if input_path.exists():
            print(f"Converting {filename}...")
            stats = converter.convert_file(input_path, output_path)
            all_stats[filename] = stats.copy()
            
            # Reset stats for next file
            converter.conversion_stats = {
                'total': 0,
                'converted': 0,
                'filtered': 0,
                'tool_calls': 0,
                'simple_qa': 0
            }
    
    # Generate final report
    report = converter.generate_report()
    
    # Add file-by-file breakdown
    report += "\n## File-by-File Breakdown\n"
    for filename, stats in all_stats.items():
        report += f"\n### {filename}\n"
        report += f"- Converted: {stats['converted']}/{stats['total']}\n"
        report += f"- Tool calls: {stats['tool_calls']}\n"
    
    # Save report
    report_path = output_dir / "conversion_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nConversion complete! Report saved to: {report_path}")
    print(f"Converted data saved to: {output_dir}")


if __name__ == "__main__":
    main()