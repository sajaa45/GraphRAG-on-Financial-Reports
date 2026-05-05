#!/usr/bin/env python3
"""
Validate and normalize extracted financial metrics using LLM verification.
Reads output/extracted_financial.json and produces a cleaned, standardized version.
"""

import os
import sys
import json
import argparse
import time
import boto3
import re
import tiktoken
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Batch size for processing grouped metrics
METRICS_PER_BATCH = 15  # Process ~15 metrics per LLM call

# Financial Ontology: Required keywords for each metric class
METRIC_ONTOLOGY = {
    # Debt metrics - MUST contain debt-specific terms
    "Total Debt": {
        "required_any": ["debt", "borrowing", "loan", "note payable", "bond"],
        "forbidden": ["liability", "obligation", "provision", "pension", "retirement", "deferred", "lease"]
    },
    "Long-term Debt": {
        "required_any": ["debt", "borrowing", "loan", "note payable", "bond", "notes"],
        "required_all": ["long-term"],
        "forbidden": ["liability", "obligation", "provision", "pension", "retirement", "deferred", "lease"]
    },
    "Short-term Debt": {
        "required_any": ["debt", "borrowing", "loan", "note payable", "bond"],
        "required_all": ["short-term", "current"],
        "forbidden": ["liability", "obligation", "provision", "pension", "retirement", "deferred", "lease"]
    },
    
    # Cash metrics - MUST contain cash terms
    "Cash & Equivalents": {
        "required_any": ["cash", "cash equivalent"],
        "forbidden": ["restricted", "investment", "marketable security"]
    },
    
    # Income metrics - MUST contain income/earnings terms
    "Net Income": {
        "required_any": ["net income", "net profit", "net earning", "profit attributable"],
        "forbidden": ["per share", "diluted", "basic"]
    },
    "EBIT": {
        "required_any": ["operating income", "income from operation", "ebit"],
        "forbidden": ["per share", "segment"]
    },
    "EBITDA": {
        "required_any": ["ebitda", "earnings before interest"],
        "forbidden": ["per share", "segment"]
    },
    
    # Interest metrics - MUST be expense, not rate
    "Interest Expense": {
        "required_any": ["interest expense", "interest cost", "interest paid", "finance cost"],
        "forbidden": ["interest rate", "rate per annum", "applicable margin", "basis point", "weighted average rate"]
    },
    
    # Asset/Liability totals - MUST say "total"
    "Current Assets": {
        "required_all": ["total", "current asset"],
        "forbidden": ["segment", "pension", "deferred"]
    },
    "Current Liabilities": {
        "required_all": ["total", "current liabilit"],
        "forbidden": ["segment", "pension", "deferred", "other"]
    },
    "Total Assets": {
        "required_all": ["total asset"],
        "forbidden": ["segment", "pension", "deferred"]
    },
    "Total Liabilities": {
        "required_all": ["total liabilit"],
        "forbidden": ["segment", "pension", "deferred"]
    },
}

# Standard metric names we want to keep
STANDARD_METRICS = {
    # Income Statement
    "Revenue": ["revenue", "net revenues", "net sales", "total revenues", "total net revenues", "sales"],
    "EBITDA": ["ebitda", "earnings before interest tax depreciation amortization", "adjusted ebitda"],
    "EBIT": ["ebit", "earnings before interest and tax", "operating income"],
    "Net Income": ["net income", "net profit", "net earnings", "profit for the year", "profit attributable"],
    
    # Cash Flow
    "Operating Cash Flow": ["operating cash flow", "cash flow from operations", "cash provided from operations"],
    "Free Cash Flow": ["free cash flow", "fcf", "levered free cash flow"],
    
    # Balance Sheet - Debt
    "Total Debt": ["total debt", "total borrowings", "total financial debt", "total indebtedness"],
    "Long-term Debt": ["long-term debt", "non-current debt", "long term borrowings"],
    "Short-term Debt": ["short-term debt", "current debt", "short-term borrowings"],
    "Net Debt": ["net debt", "net financial debt"],
    
    # Balance Sheet - Assets
    "Cash & Equivalents": ["cash and cash equivalents","cash & cash equivalents"],
    "Cash + Short-Term Investments": ["cash, cash equivalents and short-term investments","cash and short-term investments"],
    "Current Assets": ["current assets", "total current assets"],
    "Total Assets": ["total assets"],
    "Liquidity": ["total liquidity", "available liquidity"],
"Revolving Credit Availability": ["availability under revolving credit facility", "revolver availability"],
    # Balance Sheet - Liabilities & Equity
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Total Liabilities": ["total liabilities"],
    "Total Equity": ["total equity", "shareholders equity", "stockholders equity"],
    
    # Credit Metrics
    "Interest Expense": ["interest expense", "interest cost", "finance costs"],
    "Interest Paid": ["interest paid"],
    "Interest Coverage": ["interest coverage", "interest coverage ratio", "times interest earned"],
    "Debt Service Coverage": ["debt service coverage", "dscr"],
    "Debt Maturities": ["maturities of long-term debt", "debt maturity schedule"],
    # Ratios (if pre-calculated)
    "Debt / Equity": ["debt to equity", "debt equity ratio", "gearing ratio", "leverage ratio"],
    "Current Ratio": ["current ratio", "liquidity ratio"],
    "Quick Ratio": ["quick ratio", "acid test ratio"],
    "FCF / Debt": ["fcf to debt", "free cash flow to debt"],
    
    # Other
    "CapEx": ["capital expenditure", "capex", "capital spending", "purchases of property plant and equipment"],
    "Working Capital": ["working capital", "net working capital"],
    "Depreciation": ["depreciation", "depreciation expense"],
    "Amortization": ["amortization", "amortization expense"],
}

_JSON_BLOCK_RE = re.compile(r'\[.*?\]', re.DOTALL)
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


class MetricValidator:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        
        # Initialize Bedrock client - reads model from .env
        self.bedrock_model = os.getenv("BEDROCK_MODEL")
        if not self.bedrock_model:
            raise ValueError("BEDROCK_MODEL not set in .env file")
        
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not aws_key or not aws_secret:
            raise ValueError("AWS credentials not set in .env file")
        
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
        )
        print(f"✓ Bedrock client ready (model: {self.bedrock_model}, region: {region})")
        
        # Initialize tokenizer for chunking
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            print("⚠ tiktoken not available, using character-based estimation")
            self.tokenizer = None
    def _post_validate(self, metric):
        name = metric.get("standard_name")

        # Reject ratios incorrectly mapped
        if "ratio" in metric.get("original_name", "").lower() and name in ["Net Debt", "Total Debt"]:
            return False

        # Reject zeros or empty
        if not metric.get("value") or metric["value"] in ["0", "—"]:
            return False

        return True
    def _group_by_chunk_text(self, metrics: List[Dict]) -> List[List[Dict]]:
        """Group metrics by chunk_text so metrics from same source are validated together."""
        from collections import defaultdict
        
        # Group by chunk_text hash (use first 100 chars as key to handle long texts)
        groups = defaultdict(list)
        for metric in metrics:
            chunk_key = metric.get("chunk_text", "")[:100]  # Use first 100 chars as key
            groups[chunk_key].append(metric)
        
        # Convert to list of groups, sorted by group size (larger groups first for efficiency)
        grouped = sorted(groups.values(), key=len, reverse=True)
        
        # Show grouping stats
        multi_metric_groups = [g for g in grouped if len(g) > 1]
        if multi_metric_groups:
            total_in_groups = sum(len(g) for g in multi_metric_groups)
            print(f"  ✓ {len(multi_metric_groups)} chunks contain multiple metrics ({total_in_groups} total)")
            print(f"    Largest group: {len(multi_metric_groups[0])} metrics from same source")
        
        return grouped

    def _batch_metrics(self, metrics: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Simple batching: split metrics into fixed-size batches.
        Keeps grouped metrics together when possible."""
        batches = []
        current_batch = []
        
        for metric in metrics:
            # Remove metadata before sending to LLM (only send metric data)
            metric_for_llm = {
                "metric": metric["metric"],
                "value": metric["value"],
                "unit": metric["unit"],
                "year": metric["year"],
            }
            
            current_batch.append(metric_for_llm)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining metrics
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _check_ontology_constraints(self, metric: Dict) -> tuple[bool, str]:
        """Check if metric satisfies financial ontology constraints.
        
        Returns: (is_valid, reason)
        """
        standard_name = metric.get("standard_name", "")
        original_name = metric.get("original_name", "").lower()
        chunk_text = metric.get("chunk_text", "").lower()
        
        if standard_name not in METRIC_ONTOLOGY:
            return True, "No ontology constraints"
        
        constraints = METRIC_ONTOLOGY[standard_name]
        
        # Check required_any: at least ONE must be present in original_name
        if "required_any" in constraints:
            if not any(term in original_name for term in constraints["required_any"]):
                missing = ", ".join(constraints["required_any"])
                return False, f"Missing required term (need one of: {missing})"
        
        # Check required_all: ALL must be present in original_name
        if "required_all" in constraints:
            missing_terms = [term for term in constraints["required_all"] if term not in original_name]
            if missing_terms:
                return False, f"Missing required terms: {', '.join(missing_terms)}"
        
        # Check forbidden: NONE should be present in original_name
        if "forbidden" in constraints:
            found_forbidden = [term for term in constraints["forbidden"] if term in original_name]
            if found_forbidden:
                return False, f"Contains forbidden term: {found_forbidden[0]}"
        
        return True, "Ontology constraints satisfied"

    def _post_validate_metric(self, metric: Dict) -> bool:
        """Post-validation checks using chunk_text for grounding.
        
        Critical checks:
        0. Financial ontology constraints (FIRST - most important)
        1. Value must exist EXACTLY in chunk_text (ungrounded values)
        2. Detect maturity schedules (future years)
        3. Detect segment-specific tables (pension, region)
        4. Semantic meaning checks
        """
        # CHECK 0: Financial Ontology (CRITICAL - prevents semantic hallucinations)
        is_valid, reason = self._check_ontology_constraints(metric)
        if not is_valid:
            print(f"    [ONTOLOGY] {metric.get('standard_name')} rejected: {reason}")
            return False
        
        chunk_text = metric.get("chunk_text", "").lower()
        value = str(metric.get("value", "")).strip()
        year = str(metric.get("year", "")).strip()
        standard_name = metric.get("standard_name", "")
        original_name = metric.get("original_name", "").lower()
        
        if not chunk_text or not value:
            return False
        
        # CHECK 0b: Composite metric — must contain ALL component terms in original_name
        if standard_name == "Cash + Short-Term Investments":
            if "cash" not in original_name or "short-term invest" not in original_name:
                print(f"    [PARTIAL MATCH] 'Cash + Short-Term Investments' requires both 'cash' and 'short-term invest' in metric name")
                return False

        # CHECK 0c: Availability metric — reject if text is about utilization, not availability
        if standard_name == "Revolving Credit Availability":
            utilization_terms = ["outstanding", "drawn", "used", "borrowed", "letters of credit"]
            if any(t in original_name for t in utilization_terms):
                print(f"    [SEMANTIC ERROR] 'Revolving Credit Availability' from utilization term: '{original_name}'")
                return False

        # CHECK 1: Value grounding - MOST CRITICAL
        # Value must exist in chunk_text (with some flexibility for formatting)
        value_clean = value.replace(",", "").replace(".", "")
        chunk_clean = chunk_text.replace(",", "").replace(".", "").replace("\n", " ")
        
        # Check if value appears in text (as number)
        if value_clean and not value_clean in chunk_clean:
            # Try with decimal point variations
            value_with_decimal = value.replace(",", "")
            if value_with_decimal not in chunk_text.replace("\n", " "):
                print(f"    [UNGROUNDED] Value '{value}' not found in source text")
                return False
        
        # CHECK 2: Maturity schedule detection
        if standard_name in ["Total Debt", "Long-term Debt", "Short-term Debt"]:
            # Detect future year patterns
            maturity_indicators = [
                "due in", "maturing in", "thereafter", "maturity", "maturities",
                "payable in", "payments due"
            ]
            if any(ind in chunk_text for ind in maturity_indicators):
                # Check if multiple future years mentioned
                import re
                future_years = re.findall(r'\b(202[5-9]|203[0-9])\b', chunk_text)
                if len(future_years) >= 2:
                    print(f"    [MATURITY SCHEDULE] Multiple future years detected: {future_years[:3]}")
                    return False
        
        # CHECK 3: Segment-specific tables
        segment_indicators = [
            "pension", "retirement", "postretirement",
            "segment", "geographic", "by region", "by country",
            "discontinued", "held for sale"
        ]
        if any(ind in chunk_text for ind in segment_indicators):
            # Only reject if it's clearly a breakdown, not a total
            if "total" not in original_name and standard_name in ["Current Liabilities", "Total Liabilities", "Total Assets"]:
                print(f"    [SEGMENT TABLE] Segment-specific breakdown detected")
                return False
        
        # CHECK 4: Semantic meaning checks (now redundant with ontology, but keep as backup)
        semantic_rejects = {
            "Interest Expense": [
                "interest rate", "rate per annum", "applicable margin", "basis points",
                "weighted average interest rate"  # Rate %, not expense $
            ],
            "Total Debt": [
                "revolving credit availability", "credit facility availability", 
                "unused", "undrawn", "available"
            ],
            "Cash & Equivalents": [
                "restricted cash" if "restricted" in chunk_text and "unrestricted" not in chunk_text else None
            ],
        }
        
        if standard_name in semantic_rejects:
            reject_patterns = [p for p in semantic_rejects[standard_name] if p]
            # Only reject if the EXACT reject pattern is in the original name
            if any(pattern in original_name for pattern in reject_patterns):
                print(f"    [SEMANTIC ERROR] Wrong meaning: {standard_name} from '{original_name}'")
                return False
        
        return True

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

    def _batch_metrics(self, metrics: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Simple batching: split metrics into fixed-size batches.
        Keeps grouped metrics together when possible."""
        batches = []
        current_batch = []
        
        for metric in metrics:
            # Remove metadata before sending to LLM (only send metric data)
            metric_for_llm = {
                "metric": metric["metric"],
                "value": metric["value"],
                "unit": metric["unit"],
                "year": metric["year"],
            }
            
            current_batch.append(metric_for_llm)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining metrics
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _call_llm(self, prompt: str) -> str:
        """Call Bedrock LLM with retry logic and longer timeout."""
        for attempt in range(5):
            try:
                response = self.bedrock.converse(
                    modelId=self.bedrock_model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 8192, "temperature": 0.1},  # Increased from 4096
                )
                text = response["output"]["message"]["content"][0]["text"]
                text = _THINK_RE.sub('', text).strip()
                return text
            except Exception as e:
                err = str(e)
                is_throttle = "ThrottlingException" in err or "429" in err
                is_overload = "ServiceUnavailableException" in err or "503" in err
                if (is_throttle or is_overload) and attempt < 4:
                    wait = 60 if is_overload else 30 * (attempt + 1)
                    print(f"    ⚠ Bedrock {'overloaded' if is_overload else 'rate limit'}... retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

    def validate_and_normalize(self) -> Dict[str, Any]:
        """Main validation and normalization process - BATCHED LLM CALLS."""
        print(f"\n{'='*80}")
        print("METRIC VALIDATION (Batched LLM calls)")
        print(f"{'='*80}\n")
        
        # Load input data
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        main_company = data.get("main_company", "Unknown")
        print(f"Company: {main_company}")
        
        # Get HAS_METRIC relations
        raw_metrics = data.get("relations", {}).get("HAS_METRIC", [])
        print(f"Input metrics: {len(raw_metrics)}")
        
        if not raw_metrics:
            print("⚠ No metrics found in input file")
            return {"main_company": main_company, "standardized_metrics": {}}
        

        # Prepare metrics with ALL metadata for validation and preservation
        metrics_with_context = []
        for m in raw_metrics:
            # Support both formats: 'target' or 'tgt'
            target = m.get("target") or m.get("tgt", {})
            props = target.get("properties", {})
            
            metric_data = {
                "metric": props.get("metric_type", ""),
                "value": props.get("value", ""),
                "unit": props.get("unit", ""),
                "year": props.get("year", ""),
                # Preserve ALL original metadata
                "chunk_text": m.get("chunk_text", ""),
                "section_title": m.get("section_title", ""),
                "source_page": m.get("source_page"),
                "similarity": m.get("similarity"),
            }
            
            # Only add if at least metric name exists
            if metric_data["metric"]:
                metrics_with_context.append(metric_data)
        
        if not metrics_with_context:
            print("⚠ No valid metric data found (all fields empty)")
            return {"main_company": main_company, "standardized_metrics": {}}
        
        print(f"Valid metrics to validate: {len(metrics_with_context)}")
        
        # Group metrics by chunk_text (metrics from same source together)
        grouped_metrics = self._group_by_chunk_text(metrics_with_context)
        print(f"✓ Grouped into {len(grouped_metrics)} unique source chunks")
        
        # Flatten back to list but keep grouping order
        metrics_for_validation = []
        for group in grouped_metrics:
            metrics_for_validation.extend(group)
        
        # Debug: show first few metrics
        print("\nSample metrics:")
        for i, m in enumerate(metrics_for_validation[:3]):
            print(f"  {i+1}. {m['metric']}: {m['value']} {m['unit']} ({m['year']})")
        if len(metrics_for_validation) > 3:
            print(f"  ... and {len(metrics_for_validation) - 3} more")
        
        # Batch the grouped metrics (simple count-based batching)
        print(f"\nBatching into groups of ~{METRICS_PER_BATCH} metrics...")
        metric_batches = self._batch_metrics(metrics_for_validation, METRICS_PER_BATCH)
        print(f"✓ Split into {len(metric_batches)} batches")
        for i, batch in enumerate(metric_batches, 1):
            print(f"  Batch {i}: {len(batch)} metrics")
        
        # Process each batch and remap metadata back
        all_validated_metrics = []
        for i, batch in enumerate(metric_batches, 1):
            print(f"\nBatch {i}/{len(metric_batches)}: Processing {len(batch)} metrics...")
            
            prompt = self._create_validation_prompt(batch, main_company)
            llm_output = self._call_llm(prompt)
            validated = self._parse_llm_response(llm_output)
            
            # Remap metadata back to validated metrics AND apply post-validation checks
            validated_with_metadata = []
            for v_metric in validated:
                original_name = v_metric.get("original_name", "")
                # Find the original metric with this name to get metadata
                for orig_metric in metrics_for_validation:
                    if orig_metric["metric"] == original_name:
                        # Add all metadata back
                        v_metric["chunk_text"] = orig_metric.get("chunk_text", "")
                        v_metric["section_title"] = orig_metric.get("section_title", "")
                        v_metric["source_page"] = orig_metric.get("source_page")
                        v_metric["similarity"] = orig_metric.get("similarity")
                        
                        # CRITICAL: Post-validation checks with chunk_text
                        if self._post_validate_metric(v_metric):
                            validated_with_metadata.append(v_metric)
                        else:
                            print(f"  ⚠ Post-validation rejected: {v_metric.get('standard_name')} = {v_metric.get('value')}")
                        break
            
            all_validated_metrics.extend(validated_with_metadata)
            print(f"  ✓ Validated: {len(validated_with_metadata)} metrics from this batch")
        
        print(f"\n✓ Total validated: {len(all_validated_metrics)} metrics")
        
        # Organize by standard metric names
        result = {
            "main_company": main_company,
            "standardized_metrics": self._organize_by_standard_names(all_validated_metrics)
        }
        
        # Save output
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {self.output_file}\n")
        
        return result

    def _create_validation_prompt(self, metrics: List[Dict], company: str) -> str:
        """Create validation prompt for credit risk metrics - balanced strictness."""
        metrics_json = json.dumps(metrics, indent=2)
        
        standard_list = ", ".join(STANDARD_METRICS.keys())
        
        prompt = f"""Validate and normalize financial metrics for credit risk analysis. Map to standard names or REJECT if clearly wrong.

STANDARD NAMES: {standard_list}

ACCEPT if metric name contains:
- For DEBT: "debt", "borrowing", "loan", "note payable"
- For CASH: "cash", "cash equivalent"
- For INCOME: "net income", "operating income", "ebitda"
- For INTEREST: "interest expense", "interest cost" (NOT "interest rate")
- For TOTALS: Must say "total" explicitly

REJECT if:
1. Non-debt liabilities:
   - "Pension liabilities", "Asset retirement obligations"
   - "Deferred compensation", "Post-retirement benefits"
   - "Other liabilities", "Provisions"
   → Unless explicitly "debt" or "borrowings"
   
2. Rates (not expenses):
   - "Interest RATE", "Weighted average rate"
   
3. Availability (not debt):
   - "Credit availability", "Unused credit"
   
4. Maturity schedules:
   - "Debt due in 2025", "Maturities 2026-2030"
   
5. Per-share metrics:
   - "Earnings per share"
   
6. Subcomponents without "total":
   - "Other current liabilities" (need "Total current liabilities")
   - "Current portion of liability" (ambiguous - what liability?)

FLEXIBLE:
- "Interest costs" = "Interest expense" ✓
- "Cash and cash equivalents" = "Cash & Equivalents" ✓
- "Short-term borrowings" = "Short-term Debt" ✓ (if contains "borrowing")
- "Income from operations" = "EBIT" ✓ (if contains "operating income")

METRICS:
{metrics_json}

Return JSON array:
[
  {{"standard_name": "Revenue", "value": "2118.5", "unit": "$ million", "year": "2024", "original_name": "Net sales", "is_valid": true, "reason": "Valid"}}
]

REJECT examples:
{{"standard_name": null, "original_name": "Debt due 2028", "is_valid": false, "reason": "Future maturity schedule"}}
{{"standard_name": null, "original_name": "Interest rate", "is_valid": false, "reason": "Rate % not Expense $"}}
"""
        return prompt

    def _parse_llm_response(self, llm_output: str) -> List[Dict]:
        """Parse and extract JSON from LLM response."""
        m = _JSON_BLOCK_RE.search(llm_output)
        if not m:
            print(f"⚠ LLM returned no JSON block. Output:\n{llm_output[:500]}")
            return []
        
        raw_json = m.group()
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try to clean markdown fences
            cleaned = re.sub(r'```(?:json)?', '', raw_json).strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"⚠ Failed to parse JSON: {e}")
                print(f"Raw output:\n{llm_output[:1000]}")
                return []
        
        # Filter only valid metrics
        valid_metrics = [m for m in data if m.get("is_valid", False) and m.get("standard_name") and self._post_validate(m)]
        
        # Print rejected metrics
        rejected = [m for m in data if not m.get("is_valid", False)]
        if rejected:
            print(f"\n⚠ Rejected {len(rejected)} metrics:")
            for r in rejected:
                print(f"  - {r.get('original_name', 'Unknown')}: {r.get('reason', 'No reason')}")
        
        return valid_metrics
    def _to_int_year(self, y):
        try:
            return int(y)
        except:
            return 0

    def _metric_priority(self, metric: Dict) -> float:
        """Score a candidate when two metrics map to the same standard name and year.
        Higher = preferred. Prefer totals over components, then higher similarity."""
        orig = metric.get("original_name", "").lower()
        score = metric.get("similarity", 0.0)
        if "total" in orig:
            score += 2.0
        # Prefer broader phrases (longer name = less likely to be a sub-component)
        score += len(orig) * 0.005
        return score

    def _metric_entry(self, metric: Dict) -> Dict:
        return {
            "value": metric.get("value"),
            "unit": metric.get("unit"),
            "year": metric.get("year"),
            "original_name": metric.get("original_name"),
            "chunk_text": metric.get("chunk_text", ""),
            "section_title": metric.get("section_title", ""),
            "source_page": metric.get("source_page"),
            "similarity": metric.get("similarity"),
        }

    def _organize_by_standard_names(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Organize metrics by standard names, keeping most recent year.
        Ties on year are broken by priority: prefer totals, broader phrases, higher similarity."""
        organized = {}

        for metric in metrics:
            std_name = metric.get("standard_name")
            if not std_name:
                continue

            new_year = self._to_int_year(metric.get("year", ""))

            if std_name in organized:
                existing_year = self._to_int_year(organized[std_name].get("year", ""))
                if new_year > existing_year:
                    organized[std_name] = self._metric_entry(metric)
                elif new_year == existing_year:
                    # Same year: prefer the higher-priority candidate
                    if self._metric_priority(metric) > self._metric_priority(organized[std_name]):
                        organized[std_name] = self._metric_entry(metric)
            else:
                organized[std_name] = self._metric_entry(metric)

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
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Company: {result['main_company']}")
    print(f"Standardized metrics: {len(result['standardized_metrics'])}")
    print("\nMetrics found:")
    for name, data in result['standardized_metrics'].items():
        print(f"  • {name}: {data['value']} {data['unit']} ({data['year']})")
    print()


if __name__ == "__main__":
    main()
