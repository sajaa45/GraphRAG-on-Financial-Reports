#!/usr/bin/env python3
"""
Validate and normalize extracted financial metrics using strict rule-based validation.
Reads output/extracted_financial.json and produces a cleaned, standardized version.
NO LLM - pure Python rules to avoid semantic errors.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

# Standard metric names with STRICT matching rules
STANDARD_METRICS = {
    "Revenue": {
        "exact_matches": ["revenue", "net revenues", "net sales", "total revenues", "total net revenues", "sales"],
        "reject_if_contains": ["per share", "segment", "product line"],
    },
    "EBITDA": {
        "exact_matches": ["ebitda", "earnings before interest tax depreciation amortization", "adjusted ebitda"],
        "reject_if_contains": ["margin", "per share"],
    },
    "EBIT": {
        "exact_matches": ["ebit", "earnings before interest and tax", "operating income"],
        "reject_if_contains": ["margin", "per share"],
    },
    "Net Income": {
        "exact_matches": ["net income", "net profit", "net earnings", "profit for the year", "profit attributable", "net income attributable"],
        "reject_if_contains": ["per share", "diluted", "basic"],
    },
    "Operating Cash Flow": {
        "exact_matches": ["operating cash flow", "cash flow from operations", "cash provided from operations", "cash from operating activities"],
        "reject_if_contains": ["free cash flow", "per share"],
    },
    "Free Cash Flow": {
        "exact_matches": ["free cash flow", "fcf"],
        "reject_if_contains": ["per share", "operating"],
    },
    "Total Debt": {
        "exact_matches": ["total debt", "total borrowings", "total financial debt", "total indebtedness"],
        "reject_if_contains": ["short-term", "long-term", "maturity", "due in", "net debt"],
    },
    "Net Debt": {
        "exact_matches": ["net debt", "net financial debt", "net borrowings"],
        "reject_if_contains": ["ratio", "leverage"],
    },
    "Interest Expense": {
        "exact_matches": ["interest expense", "interest cost", "interest costs", "interest paid"],
        "reject_if_contains": ["coverage", "ratio", "capitalized"],
    },
    "Cash & Equivalents": {
        "exact_matches": ["cash and equivalents", "cash and cash equivalents", "cash"],
        "reject_if_contains": ["short-term investments", "flow", "ratio"],
    },
    "Current Assets": {
        "exact_matches": ["current assets", "total current assets"],
        "reject_if_contains": ["cash", "receivables", "inventory", "ratio"],
    },
    "Current Liabilities": {
        "exact_matches": ["current liabilities", "total current liabilities"],
        "reject_if_contains": ["current liability", "ratio", "short-term"],  # Reject singular form
    },
    "CapEx": {
        "exact_matches": ["capital expenditure", "capital expenditures", "capex", "capital spending", "purchases of property plant and equipment"],
        "reject_if_contains": ["capitalized interest", "ratio"],
    },
}

# Metrics that should NEVER be extracted (they're derived/calculated)
REJECT_METRICS = {
    "Debt / Equity": "This is a calculated ratio, not a direct metric",
    "Current Ratio": "This is a calculated ratio, not a direct metric",
    "FCF / Debt": "This is a calculated ratio, not a direct metric",
    "Working Capital": "This is calculated (Current Assets - Current Liabilities), not a direct metric",
    "Debt maturity breakdown": "This is a schedule, not a single metric",
}


class MetricValidator:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file

    def validate_and_normalize(self) -> Dict[str, Any]:
        """Main validation and normalization process - RULE-BASED."""
        print(f"\n{'='*80}")
        print("METRIC VALIDATION (Rule-based, no LLM)")
        print(f"{'='*80}\n")
        
        # Load input data
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        main_company = data.get("main_company", "Unknown")
        print(f"Company: {main_company}")
        
        # Get HAS_METRIC relations
        raw_metrics = data.get("relations", {}).get("HAS_METRIC", [])
        print(f"Input metrics: {len(raw_metrics)}\n")
        
        if not raw_metrics:
            print("⚠ No metrics found in input file")
            return {"main_company": main_company, "standardized_metrics": {}}
        
        # Validate each metric
        validated_metrics = []
        rejected_metrics = []
        
        for m in raw_metrics:
            result = self._validate_metric(m)
            if result["is_valid"]:
                validated_metrics.append(result)
            else:
                rejected_metrics.append(result)
        
        print(f"✓ Validated: {len(validated_metrics)} metrics")
        print(f"✗ Rejected: {len(rejected_metrics)} metrics\n")
        
        # Show rejection reasons
        if rejected_metrics:
            print("Rejection reasons:")
            rejection_counts = {}
            for r in rejected_metrics:
                reason = r["reason"]
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
                print(f"  • {reason}: {count}")
            print()
        
        # Organize by standard metric names (keep most recent year)
        result = {
            "main_company": main_company,
            "standardized_metrics": self._organize_by_standard_names(validated_metrics)
        }
        
        # Save output
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {self.output_file}\n")
        
        return result

    def _validate_metric(self, metric_relation: Dict) -> Dict[str, Any]:
        """Validate a single metric using strict rules."""
        # Extract data from relation
        target = metric_relation.get("target") or metric_relation.get("tgt", {})
        props = target.get("properties", {})
        
        original_name = props.get("metric_type", "").strip()
        value = props.get("value", "").strip()
        unit = props.get("unit", "").strip()
        year = props.get("year", "").strip()
        
        # Preserve ALL original metadata
        metadata = {
            "chunk_text": metric_relation.get("chunk_text", ""),
            "similarity": metric_relation.get("similarity"),
            "section_title": metric_relation.get("section_title", ""),
            "source_page": metric_relation.get("source_page"),
        }
        
        # Basic validation
        if not original_name or not value:
            return {
                "is_valid": False,
                "original_name": original_name,
                "reason": "Missing metric name or value"
            }
        
        # Check if it's a rejected metric type
        original_lower = original_name.lower()
        for reject_name, reject_reason in REJECT_METRICS.items():
            if reject_name.lower() in original_lower:
                return {
                    "is_valid": False,
                    "original_name": original_name,
                    "reason": reject_reason
                }
        
        # Try to match to standard metrics
        matched_standard = None
        for standard_name, rules in STANDARD_METRICS.items():
            # Check exact matches
            if any(exact.lower() == original_lower for exact in rules["exact_matches"]):
                # Check rejection rules
                if any(reject.lower() in original_lower for reject in rules["reject_if_contains"]):
                    return {
                        "is_valid": False,
                        "original_name": original_name,
                        "reason": f"Contains rejected substring for {standard_name}"
                    }
                matched_standard = standard_name
                break
        
        if not matched_standard:
            return {
                "is_valid": False,
                "original_name": original_name,
                "reason": "Does not match any standard metric"
            }
        
        # Additional validation for specific metrics
        validation_result = self._validate_specific_metric(matched_standard, original_name, value, unit)
        if not validation_result["valid"]:
            return {
                "is_valid": False,
                "original_name": original_name,
                "reason": validation_result["reason"]
            }
        
        # Valid metric
        return {
            "is_valid": True,
            "standard_name": matched_standard,
            "original_name": original_name,
            "value": value,
            "unit": unit,
            "year": year,
            "metadata": metadata
        }

    def _validate_specific_metric(self, standard_name: str, original_name: str, value: str, unit: str) -> Dict:
        """Additional validation rules for specific metrics."""
        original_lower = original_name.lower()
        
        # Current Assets: must be "total" or just "current assets"
        if standard_name == "Current Assets":
            if "cash" in original_lower and "total" not in original_lower:
                return {"valid": False, "reason": "Cash is a component, not total current assets"}
            if "receivable" in original_lower and "total" not in original_lower:
                return {"valid": False, "reason": "Receivables is a component, not total current assets"}
        
        # Current Liabilities: must be plural "liabilities" not singular "liability"
        if standard_name == "Current Liabilities":
            if "liability" in original_lower and "liabilities" not in original_lower:
                return {"valid": False, "reason": "Singular 'liability' is a line item, not total"}
        
        # Total Debt: reject if it's a component
        if standard_name == "Total Debt":
            if "short-term" in original_lower or "long-term" in original_lower:
                return {"valid": False, "reason": "This is a debt component, not total debt"}
            if "maturity" in original_lower or "due in" in original_lower:
                return {"valid": False, "reason": "This is a maturity schedule, not total debt"}
        
        # Net Debt: must explicitly say "net debt"
        if standard_name == "Net Debt":
            if "net debt" not in original_lower and "net borrowings" not in original_lower:
                return {"valid": False, "reason": "Not explicitly labeled as net debt"}
        
        # Interest Expense: reject capitalized interest
        if standard_name == "Interest Expense":
            if "capitalized" in original_lower:
                return {"valid": False, "reason": "Capitalized interest is not interest expense"}
        
        return {"valid": True}

    def _organize_by_standard_names(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Organize metrics by standard names, keeping most recent year."""
        organized = {}
        
        for metric in metrics:
            std_name = metric.get("standard_name")
            if not std_name:
                continue
            
            # If metric already exists, keep the one with most recent year
            if std_name in organized:
                existing_year = organized[std_name].get("year", "")
                new_year = metric.get("year", "")
                if new_year > existing_year:
                    organized[std_name] = {
                        "value": metric.get("value"),
                        "unit": metric.get("unit"),
                        "year": metric.get("year"),
                        "original_name": metric.get("original_name"),
                        "metadata": metric.get("metadata", {})
                    }
            else:
                organized[std_name] = {
                    "value": metric.get("value"),
                    "unit": metric.get("unit"),
                    "year": metric.get("year"),
                    "original_name": metric.get("original_name"),
                    "metadata": metric.get("metadata", {})
                }
        
        return organized


def main():
    parser = argparse.ArgumentParser(description="Validate and normalize extracted financial metrics")
    parser.add_argument(
        "--input",
        default="output/extracted_financial.json",
        help="Input JSON file with extracted metrics"
    )
    parser.add_argument(
        "--output",
        default="output/validated_metrics.json",
        help="Output JSON file with validated metrics"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"✗ Input file not found: {args.input}")
        sys.exit(1)
    
    validator = MetricValidator(args.input, args.output)
    result = validator.validate_and_normalize()
    
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Company: {result['main_company']}")
    print(f"Standardized metrics: {len(result['standardized_metrics'])}")
    print("\nMetrics found:")
    for name, data in result['standardized_metrics'].items():
        print(f"  • {name}: {data['value']} {data['unit']} ({data['year']})")
        print(f"    Original: {data['original_name']}")
        if data.get('metadata', {}).get('section_title'):
            print(f"    Section: {data['metadata']['section_title']}")
    print()


if __name__ == "__main__":
    main()
