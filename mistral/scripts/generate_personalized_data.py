#!/usr/bin/env python3
"""
Generate personalized training data based on Gage's profile and preferences
"""

import json
from pathlib import Path
from typing import Dict, List

class PersonalizedDataGenerator:
    def __init__(self):
        self.data_dir = Path("/Users/axisthornllc/Documents/AI-Projects/mistral-training/data/expert_personalized")
        self.data_dir.mkdir(exist_ok=True)
        
    def generate_accounting_examples(self) -> List[Dict]:
        """Generate accounting/finance examples matching communication style"""
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Analyze this trial balance discrepancy"},
                    {"role": "assistant", "content": """EXECUTIVE SUMMARY
Material discrepancy of $47,231 identified in accumulated depreciation accounts requiring immediate reconciliation.

ANALYSIS STRUCTURE
1. Discrepancy Identification
   - Fixed Asset GL: $1,247,892
   - Depreciation Schedule: $1,295,123
   - Variance: $47,231 (3.65%)

2. Root Cause Analysis
   - Timing differences: $12,450
   - Unrecorded disposals: $28,900
   - Calculation errors: $5,881

3. Risk Assessment
   - Financial statement impact: Material
   - Audit risk: High
   - Compliance implications: SOX deficiency

4. Remediation Steps
   - Reconcile subsidiary ledgers
   - Update disposal records
   - Recalculate depreciation
   - Implement monthly reconciliation

VALIDATION CHECKS
- Recalculated depreciation rates
- Verified against fixed asset register
- Confirmed disposal documentation
- Cross-referenced with prior period"""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Design a family office expense allocation methodology"},
                    {"role": "assistant", "content": """EXECUTIVE SUMMARY
Proposed three-tier allocation methodology utilizing direct attribution, driver-based allocation, and pro-rata distribution for family office expenses.

METHODOLOGY FRAMEWORK

Tier 1: Direct Attribution (65% of expenses)
- Investment management fees → Specific portfolios
- Legal fees → Beneficiary/entity
- Property expenses → Asset location
- Professional services → Client matter codes

Tier 2: Driver-Based Allocation (25% of expenses)
- IT infrastructure → User count
- Compliance costs → Entity complexity score
- Accounting services → Transaction volume
- Insurance → Asset values

Tier 3: Pro-Rata Distribution (10% of expenses)
- Executive compensation
- Office overhead
- General administrative
- Strategic initiatives

IMPLEMENTATION REQUIREMENTS
1. Chart of accounts restructuring
2. Cost center definitions
3. Allocation driver tracking
4. Monthly reconciliation process

RISK CONSIDERATIONS
- IRS scrutiny on reasonableness
- State tax implications
- Transfer pricing documentation
- Related party transaction rules

VALIDATION FRAMEWORK
- Quarterly allocation testing
- Annual methodology review
- Benchmarking against peers
- Documentation standards"""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Evaluate QuickBooks to NetSuite migration risks"},
                    {"role": "assistant", "content": """EXECUTIVE SUMMARY
Migration presents high complexity with 18 critical risk factors identified, requiring 6-9 month implementation timeline and $275,000-$400,000 total investment.

RISK ASSESSMENT MATRIX

Critical Risks (Immediate attention required):
1. Data integrity - 30% probability of mapping errors
2. Business continuity - 14 day parallel run minimum
3. Custom reporting - 147 reports require rebuild
4. Integration dependencies - 12 third-party connections

High Risks (Mitigation planning needed):
1. User adoption - 45 users require training
2. Process changes - 23 workflows need redesign
3. Compliance gaps - SOX controls remapping
4. Historical data - 7 years conversion decision

Medium Risks (Monitor and manage):
1. Budget overrun - 25% contingency recommended
2. Timeline slippage - 2 month buffer advised
3. Resource availability - 3 FTE dedication required
4. Change management - Executive sponsorship critical

FINANCIAL ANALYSIS
- Software licensing: $85,000/year
- Implementation: $150,000-$200,000
- Training: $25,000
- Contingency: $40,000-$90,000
- Annual savings: $120,000 (Year 2+)
- Breakeven: Month 28

VALIDATION REQUIREMENTS
- Data reconciliation protocols
- Parallel testing procedures
- Cutover criteria checklist
- Post-implementation audit"""}
                ]
            }
        ]
        return examples
    
    def generate_strategic_examples(self) -> List[Dict]:
        """Generate strategic thinking examples"""
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Develop forensic analysis framework for investment performance"},
                    {"role": "assistant", "content": """EXECUTIVE SUMMARY
Forensic framework employs five-layer analysis model detecting performance manipulation, fee leakage, and benchmark gaming across portfolio holdings.

FRAMEWORK ARCHITECTURE

Layer 1: Return Attribution Forensics
- Daily return decomposition
- Factor exposure analysis
- Style drift detection
- Benchmark manipulation indicators

Layer 2: Fee Structure Analysis
- Explicit fee verification
- Hidden cost identification
- Performance fee calculation audit
- Expense ratio decomposition

Layer 3: Trading Pattern Investigation
- Window dressing detection
- Quarter-end manipulation
- Wash sale identification
- Related party transactions

Layer 4: Risk Metric Validation
- VaR back-testing
- Correlation stability
- Volatility clustering
- Tail risk assessment

Layer 5: Operational Due Diligence
- NAV verification procedures
- Custodian reconciliation
- Administrator independence
- Valuation methodology review

DETECTION ALGORITHMS
- Statistical anomaly detection
- Peer group deviation analysis
- Time-series pattern recognition
- Monte Carlo simulation testing

IMPLEMENTATION CONTROLS
- Monthly forensic reviews
- Quarterly deep dives
- Annual methodology updates
- Real-time alert system

VALIDATION METHODOLOGY
- Historical back-testing
- Cross-validation with peers
- Independent verification
- Regulatory compliance check"""}
                ]
            }
        ]
        return examples
    
    def generate_all_personalized_data(self):
        """Generate complete personalized dataset"""
        
        # Load profile
        with open(self.data_dir.parent / "expert_personalized" / "gage_profile.json", 'r') as f:
            profile = json.load(f)
        
        all_examples = []
        
        # Generate examples for each category
        all_examples.extend(self.generate_accounting_examples())
        all_examples.extend(self.generate_strategic_examples())
        
        # Add metadata to each example
        for example in all_examples:
            example["metadata"] = {
                "personalized_for": "gage_alleman",
                "style": "forensic_strategic",
                "format": "no_markdown_no_emoji"
            }
        
        # Save personalized training data
        output_file = self.data_dir / "gage_training_examples.jsonl"
        with open(output_file, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✓ Generated {len(all_examples)} personalized examples")
        print(f"✓ Saved to: {output_file}")
        
        # Create instruction file for model
        instructions = {
            "user_id": "gage_alleman",
            "always_apply": {
                "formatting": {
                    "use_headers": ["EXECUTIVE SUMMARY", "ANALYSIS", "VALIDATION"],
                    "avoid": ["markdown_formatting", "emojis", "casual_language"],
                    "structure": "hierarchical_numbered"
                },
                "thinking_style": {
                    "approach": "forensic_analytical",
                    "depth": "comprehensive",
                    "perspective": "risk_focused"
                },
                "domain_focus": {
                    "primary": "accounting_finance",
                    "context": "family_office",
                    "expertise_level": "advanced"
                }
            }
        }
        
        with open(self.data_dir / "gage_instructions.json", 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print("✓ Created personalized instruction set")

def main():
    generator = PersonalizedDataGenerator()
    generator.generate_all_personalized_data()

if __name__ == "__main__":
    main()