#!/usr/bin/env python3
"""Validate and normalize extracted financial metrics."""

import os
import sys
import json
import argparse
import time
import re
import boto3
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

METRICS_PER_BATCH = 15

# Minimal ontology: rules for metric classes where LLMs commonly make mistakes.
# All checks run against the *original metric name* (lowercased).
METRIC_ONTOLOGY: Dict[str, Dict] = {
    "Interest Expense": {
        "forbidden": ["rate", "per annum", "applicable margin", "basis point"]
    },
    "Total Debt": {
        "required_any": ["debt", "borrowing", "loan", "note", "bond"],
        "forbidden": ["pension", "deferred", "lease", "available", "unused", "undrawn"]
    },
    "Long-term Debt": {
        "required_any": ["debt", "borrowing", "loan", "note", "bond"],
        "required_all": ["long-term"],
        "forbidden": ["pension", "deferred", "lease"]
    },
    "Short-term Debt": {
        "required_any": ["debt", "borrowing", "loan", "note", "bond"],
        "required_all": ["short-term"],
        "forbidden": ["pension", "deferred", "lease"]
    },
    "Net Income": {
        "required_any": ["net income", "net profit", "net earning"],
        "forbidden": ["per share"]
    },
}

STANDARD_METRICS: Dict[str, List[str]] = {
    "Revenue": ["revenue", "net revenues", "net sales", "total revenues", "total net revenues", "sales"],
    "EBITDA": ["ebitda", "earnings before interest tax depreciation amortization", "adjusted ebitda"],
    "EBIT": ["ebit", "earnings before interest and tax", "operating income"],
    "Net Income": ["net income", "net profit", "net earnings", "profit for the year", "profit attributable"],
    "Operating Cash Flow": ["operating cash flow", "cash flow from operations", "cash provided from operations"],
    "Free Cash Flow": ["free cash flow", "fcf", "levered free cash flow"],
    "Total Debt": ["total debt", "total borrowings", "total financial debt", "total indebtedness"],
    "Long-term Debt": ["long-term debt", "non-current debt", "long term borrowings"],
    "Short-term Debt": ["short-term debt", "current debt", "short-term borrowings"],
    "Net Debt": ["net debt", "net financial debt"],
    "Cash & Equivalents": ["cash and cash equivalents", "cash & cash equivalents"],
    "Cash + Short-Term Investments": ["cash, cash equivalents and short-term investments", "cash and short-term investments"],
    "Current Assets": ["current assets", "total current assets"],
    "Total Assets": ["total assets"],
    "Liquidity": ["total liquidity", "available liquidity"],
    "Revolving Credit Availability": ["availability under revolving credit facility", "revolver availability"],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Total Liabilities": ["total liabilities"],
    "Total Equity": ["total equity", "shareholders equity", "stockholders equity"],
    "Interest Expense": ["interest expense", "interest cost", "finance costs"],
    "Interest Paid": ["interest paid"],
    "Interest Coverage": ["interest coverage", "interest coverage ratio", "times interest earned"],
    "Debt Service Coverage": ["debt service coverage", "dscr"],
    "Debt / Equity": ["debt to equity", "debt equity ratio", "gearing ratio", "leverage ratio"],
    "Current Ratio": ["current ratio", "liquidity ratio"],
    "Quick Ratio": ["quick ratio", "acid test ratio"],
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

    def _call_llm(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = self.bedrock.converse(
                    modelId=self.bedrock_model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 8192, "temperature": 0.1},
                )
                text = response["output"]["message"]["content"][0]["text"]
                return _THINK_RE.sub('', text).strip()
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

    def _post_validate_metric(self, metric: Dict) -> bool:
        """Rule-based checks applied after LLM standardization, using source chunk_text."""
        standard_name = metric.get("standard_name", "")
        original_name = metric.get("original_name", "").lower()
        chunk_text = metric.get("chunk_text", "").lower()
        value = str(metric.get("value", "")).strip()

        if not chunk_text or not value:
            return False

        # 1. Ontology: per-metric required/forbidden terms checked against original name
        if standard_name in METRIC_ONTOLOGY:
            rules = METRIC_ONTOLOGY[standard_name]
            if "required_any" in rules and not any(t in original_name for t in rules["required_any"]):
                print(f"    [ONTOLOGY] {standard_name}: missing required term in '{original_name}'")
                return False
            if "required_all" in rules and any(t not in original_name for t in rules["required_all"]):
                print(f"    [ONTOLOGY] {standard_name}: missing required term in '{original_name}'")
                return False
            if "forbidden" in rules and any(t in original_name for t in rules["forbidden"]):
                print(f"    [ONTOLOGY] {standard_name}: forbidden term in '{original_name}'")
                return False

        # 2. Composite metric (e.g. "Cash + Short-Term Investments"):
        #    each component of the standard name must appear in the original name.
        if re.search(r'[+&]| and ', standard_name, re.IGNORECASE):
            for part in re.split(r'[+&]| and ', standard_name.lower()):
                key_words = [w for w in part.split() if len(w) > 3]
                if key_words and not any(w in original_name for w in key_words):
                    print(f"    [PARTIAL MATCH] '{standard_name}': component '{part.strip()}' missing from '{original_name}'")
                    return False

        # 3. Availability metrics must not come from utilization/drawn language
        if "availability" in standard_name.lower():
            if any(t in original_name for t in ["outstanding", "drawn", "used", "borrowed", "letters of credit"]):
                print(f"    [SEMANTIC ERROR] Availability metric mapped from utilization: '{original_name}'")
                return False

        # 4. Value grounding: numeric value must appear in source chunk
        value_digits = re.sub(r'[^\d]', '', value)
        if value_digits and value_digits not in re.sub(r'[^\d]', '', chunk_text):
            print(f"    [UNGROUNDED] Value '{value}' not found in source text")
            return False

        # 5. Maturity schedule: debt chunks listing multiple future years are schedules, not balances
        if standard_name in ("Total Debt", "Long-term Debt", "Short-term Debt"):
            schedule_triggers = ["thereafter", "maturity", "maturities", "due in", "maturing in", "payable in"]
            if any(t in chunk_text for t in schedule_triggers):
                if len(re.findall(r'\b(202[5-9]|203\d)\b', chunk_text)) >= 2:
                    print(f"    [MATURITY SCHEDULE] Multiple future years in debt chunk")
                    return False

        return True

    def _batch(self, metrics: List[Dict], size: int) -> List[List[Dict]]:
        return [metrics[i:i + size] for i in range(0, len(metrics), size)]

    def _create_validation_prompt(self, metrics: List[Dict], company: str) -> str:
        standard_list = ", ".join(STANDARD_METRICS.keys())
        return f"""Validate and normalize financial metrics for credit risk analysis.

STANDARD NAMES: {standard_list}

ACCEPT (flexible matching):
- Net sales / net revenues → Revenue
- Interest cost / finance costs → Interest Expense  (NOT "interest rate %")
- Cash and cash equivalents → Cash & Equivalents
- Short-term borrowings → Short-term Debt

REJECT:
1. Non-debt liabilities: pension, deferred compensation, lease obligations, provisions
2. Rates not expenses: "interest rate", "weighted average rate", "applicable margin"
3. Availability / unused credit (not debt)
4. Maturity schedules: "debt due in 2025", "maturities 2026–2030"
5. Per-share metrics
6. Sub-totals without "total" for balance sheet line items

METRICS:
{json.dumps(metrics, indent=2)}

Return a JSON array:
[
  {{"standard_name": "Revenue", "value": "2118.5", "unit": "$ million", "year": "2024", "original_name": "Net sales", "is_valid": true, "reason": "Valid"}}
]

Reject examples:
{{"standard_name": null, "original_name": "Interest rate 6.9%", "is_valid": false, "reason": "Rate not expense"}}
{{"standard_name": null, "original_name": "Debt due 2028", "is_valid": false, "reason": "Maturity schedule"}}
"""

    def _parse_llm_response(self, llm_output: str) -> List[Dict]:
        m = _JSON_BLOCK_RE.search(llm_output)
        if not m:
            print(f"⚠ LLM returned no JSON.\n{llm_output[:500]}")
            return []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            cleaned = re.sub(r'```(?:json)?', '', m.group()).strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"⚠ JSON parse failed: {e}")
                return []

        valid = [x for x in data if x.get("is_valid") and x.get("standard_name") and x.get("value") not in ("0", "—", None, "")]
        rejected = [x for x in data if not x.get("is_valid")]
        if rejected:
            print(f"  LLM rejected {len(rejected)}: {[r.get('original_name') for r in rejected]}")
        return valid

    def _to_int_year(self, y) -> int:
        try:
            return int(y)
        except Exception:
            return 0

    def _metric_priority(self, metric: Dict) -> float:
        """Score for same-name same-year tie-breaking: prefer totals, broader names, higher similarity."""
        orig = metric.get("original_name", "").lower()
        score = metric.get("similarity", 0.0)
        if "total" in orig:
            score += 2.0
        score += len(orig) * 0.005
        return score

    def _metric_entry(self, metric: Dict) -> Dict:
        return {k: metric.get(k) for k in ("value", "unit", "year", "original_name", "chunk_text", "section_title", "source_page", "similarity")}

    def _organize_by_standard_names(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Keep most recent year; break same-year ties by priority (prefer totals over sub-components)."""
        organized: Dict[str, Any] = {}
        for metric in metrics:
            name = metric.get("standard_name")
            if not name:
                continue
            new_year = self._to_int_year(metric.get("year", ""))
            if name in organized:
                existing_year = self._to_int_year(organized[name].get("year", ""))
                if new_year > existing_year or (
                    new_year == existing_year
                    and self._metric_priority(metric) > self._metric_priority(organized[name])
                ):
                    organized[name] = self._metric_entry(metric)
            else:
                organized[name] = self._metric_entry(metric)
        return organized

    def validate_and_normalize(self) -> Dict[str, Any]:
        print(f"\n{'='*80}\nMETRIC VALIDATION\n{'='*80}\n")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_company = data.get("main_company", "Unknown")
        raw_metrics = data.get("relations", {}).get("HAS_METRIC", [])
        print(f"Company: {main_company} | Input metrics: {len(raw_metrics)}")

        if not raw_metrics:
            return {"main_company": main_company, "standardized_metrics": {}}

        metrics_with_context = []
        for m in raw_metrics:
            target = m.get("target") or m.get("tgt", {})
            props = target.get("properties", {})
            metric_type = props.get("metric_type", "")
            if metric_type:
                metrics_with_context.append({
                    "metric": metric_type,
                    "value": props.get("value", ""),
                    "unit": props.get("unit", ""),
                    "year": props.get("year", ""),
                    "chunk_text": m.get("chunk_text", ""),
                    "section_title": m.get("section_title", ""),
                    "source_page": m.get("source_page"),
                    "similarity": m.get("similarity"),
                })

        print(f"Metrics to validate: {len(metrics_with_context)}")
        llm_payloads = [{"metric": m["metric"], "value": m["value"], "unit": m["unit"], "year": m["year"]} for m in metrics_with_context]
        batches = self._batch(llm_payloads, METRICS_PER_BATCH)
        print(f"Batches: {len(batches)} × ~{METRICS_PER_BATCH}")

        all_validated: List[Dict] = []
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}/{len(batches)}...")
            for v in self._parse_llm_response(self._call_llm(self._create_validation_prompt(batch, main_company))):
                source = next((m for m in metrics_with_context if m["metric"] == v.get("original_name")), None)
                if source:
                    v.update({k: source[k] for k in ("chunk_text", "section_title", "source_page", "similarity")})
                    if self._post_validate_metric(v):
                        all_validated.append(v)
                    else:
                        print(f"  ⚠ Post-validation rejected: {v.get('standard_name')} = {v.get('value')}")

        print(f"\n✓ Validated metrics: {len(all_validated)}")
        result = {
            "main_company": main_company,
            "standardized_metrics": self._organize_by_standard_names(all_validated),
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to: {self.output_file}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Validate and normalize extracted financial metrics")
    parser.add_argument("--input", default="output/extracted_financial.json")
    parser.add_argument("--output", default="output/validated_metrics.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"✗ Input file not found: {args.input}")
        sys.exit(1)

    validator = MetricValidator(args.input, args.output)
    result = validator.validate_and_normalize()

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"Company: {result['main_company']}")
    print(f"Standardized metrics: {len(result['standardized_metrics'])}")
    for name, data in result['standardized_metrics'].items():
        print(f"  • {name}: {data['value']} {data['unit']} ({data['year']})")


if __name__ == "__main__":
    main()
