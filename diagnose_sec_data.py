#!/usr/bin/env python3
"""
Configuration for multi-relation extraction pipeline
"""

import re
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

# Pre-compiled regex patterns used in parse_metric_entity (avoids per-call recompilation)
_OCR_FIX_RE = re.compile(r'(\d),\s+(\d)')
_CLEAN_VALUE_RE = re.compile(r'[,₹#$€£¥\s]')
_TRAILING_X_RE = re.compile(r'[xX]$')
_OCR_PERIOD_RE = re.compile(r'^\d{1,3}\.\d{3}$')
_ACCOUNTING_NEG_RE = re.compile(r'^\(([0-9.]+)\)$')


@dataclass(slots=True)
class RelationConfig:
    name: str
    source_entity_type: str
    target_entity_type: str
    relationship_type: str
    section_keywords: str
    chunk_keywords: str
    extraction_prompt_template: str
    entity_parser: Callable[..., Optional[Dict]]
    # Optional extra kwargs forwarded to entity_parser (e.g. main_company for OPERATES_IN)
    entity_parser_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Per-relation retrieval tuning (override global defaults)
    n_sections: int = 2
    n_chunks_per_section: int = 3
    # Minimum cosine similarity for a chunk to be passed to the LLM.
    # Lower this for relations whose content lives in dense tables (e.g. HAS_METRIC).
    chunk_similarity_threshold: float = 0.3
    # If non-empty, one Qdrant query is issued per entry; results are union-deduplicated.
    # Overrides chunk_keywords when set.
    chunk_keywords_list: List[str] = field(default_factory=list)
    # When False, each keyword in chunk_keywords_list retrieves its own fresh chunk set.
    # Entity-level dedup still applies. Set False for dense tables (e.g. HAS_METRIC)
    # where a single chunk contains multiple distinct metrics.
    deduplicate_chunks_across_keywords: bool = True


# ============================================================================
# ENTITY PARSERS
# ============================================================================

def parse_metric_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    metric = str(entity.get('metric', '')).strip()
    value = str(entity.get('value', '')).strip()
    unit = str(entity.get('unit', 'ratio')).strip()
    year = str(entity.get('year', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not metric or not value:
        return None

    # Fix OCR-spaced thousands ("216, 642" → "216,642"), then strip commas/currency/whitespace in one pass
    value = _OCR_FIX_RE.sub(r'\1,\2', value)
    value_clean = _CLEAN_VALUE_RE.sub('', value)
    value_clean = _TRAILING_X_RE.sub('', value_clean)
    # Normalize OCR period-as-comma artifact: "393.891" → "393891"
    if _OCR_PERIOD_RE.match(value_clean):
        value_clean = value_clean.replace('.', '')
    # Convert accounting negatives: (78078) → -78078
    m = _ACCOUNTING_NEG_RE.match(value_clean)
    if m:
        value_clean = '-' + m.group(1)

    try:
        float(value_clean)
    except ValueError:
        return None

    metric_name = f"{metric} ({year})" if year else metric

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Metric',
            'name': metric_name,
            'properties': {
                'value': value_clean,
                'unit': unit,
                'year': year,
                'metric_type': metric
            }
        },
        'relationship': 'HAS_METRIC',
        'properties': {}
    }


# ============================================================================
# RELATION CONFIGURATIONS
# ============================================================================

RELATION_CONFIGS: Dict[str, RelationConfig] = {

    'HAS_METRIC': RelationConfig(
        name='HAS_METRIC',
        source_entity_type='Company',
        target_entity_type='Metric',
        relationship_type='HAS_METRIC',
        section_keywords='financial performance results earnings statements',
        chunk_keywords='',  
        chunk_keywords_list=[
            'gearing ratio net debt total equity leverage debt ratio',
            'free cash flow operating cash flow capital expenditure investment',
            'EBITDA earnings before interest tax depreciation amortization operating profit',
            'ROACE return on average capital employed return on equity',
            'net income profit loss attributable earnings per share',
            'revenue net revenues net sales total revenues total net revenues',
            'interest coverage ratio EBIT debt service fixed charge interest expense',
            'current ratio quick ratio liquidity cash equivalents short-term current assets current liabilities',
            'total debt borrowings long-term debt bonds notes payable senior notes',
        ],
        n_sections=3,
        n_chunks_per_section=7,
        chunk_similarity_threshold=0.15,
        deduplicate_chunks_across_keywords=False,
        extraction_prompt_template="""You are a strict data extraction auditor. Your ONLY job is to copy financial metrics that appear EXPLICITLY in the text below. You must NOT use any external knowledge or training data about this company.

CRITICAL GROUNDING RULES:
1. If a metric name or value does NOT appear in the text below, you MUST return [] for that metric
2. You MUST quote the exact text snippet where you found each metric (we will verify this)
3. DO NOT use your knowledge about {main_company} - ONLY extract what is written in this specific text
4. If you are unsure whether the text contains a metric, return [] - it is better to miss a metric than to hallucinate one

IMPORTANT: You are a strict auditor. If the chunk_text does NOT contain the financial metric with an explicit numeric value, do not invent one. Return an empty list []. Do not provide values from your own knowledge; if you cannot find the metric and its value in the text, do not return it.

ROW-LABEL → STANDARD NAME MAPPING
Match the row label from the table (case-insensitive). Use the standard name shown on the right.

  Row label variants                                                  → Standard metric name
  "Gearing" / "Gearing ratio"                                         → "Gearing Ratio"
  "Net debt" / "Net debt (cash)"                                       → "Net Debt"
  "Free cash flow"                                                     → "Free Cash Flow"
  "EBITDA" / "Adjusted EBITDA"                                         → "EBITDA"
  "ROACE"                                                              → "ROACE"
  "Capital expenditures" / "Capital expenditures - cash basis"
    / "Purchases of property, plant and equipment"
    / "Acquisition of property, plant and equipment"                   → "Capital Expenditure"
  "Net income" / "Net income attributable to the ordinary shareholders" → "Net Income"
  "Earnings per share" / "Basic earnings per share"
    / "Diluted earnings per share"                                     → "Earnings per Share"
  "Revenue" / "Total revenues" / "External revenue" (Consolidated only)
    / "Net revenues" / "Net sales" / "Total net revenues"
    / "Sales" / "Net revenue"                                          → "Revenue"
  "Earnings (losses) before interest, income taxes and zakat"
    / "Operating income" / "EBIT"                                      → "EBIT"
  "Total borrowings" / "Total borrowings (current and non-current)"
    / "Long-term debt" / "Total long-term debt"
    / "Long-term debt, net" / "Total debt"
    / "Long-term debt, including current portion"                      → "Total Debt"
  "Interest expense" / "Interest expense, net"
    / "Interest and debt expense"                                      → "Interest Expense"
  "Total current assets"                                               → "Current Assets"
  "Total current liabilities"                                          → "Current Liabilities"
  "Cash and cash equivalents"                                          → "Cash and Equivalents"
  "Short-term investments"                                             → "Short-term Investments"

SOURCE TABLE RULES:
  ✓ Extract Net Income and EBIT ONLY from tables titled (or headed):
      "Consolidated Statements of Income", "Consolidated Statements of Operations",
      "Statements of Income", or "Statements of Operations".
  ✗ DO NOT extract Net Income or EBIT from the Balance Sheet, Statement of Financial Position,
      or Statement of Shareholders' Equity — values there represent cumulative or different
      accounting snapshots and are NOT the period income figures.
  ✗ If you cannot identify the source table title from the surrounding text, skip Net Income and EBIT.

CRITICAL — DO NOT confuse these:
  ✗ "Acquisition of right-of-use assets"                      ≠  Capital Expenditure (lease asset addition)
  ✗ "Net cash provided by operating activities"               ≠  Free Cash Flow
  ✗ "Net cash used in investing activities"                   ≠  any listed metric
  ✗ "Net cash used in financing activities"                   ≠  any listed metric
  ✗ "Total liabilities"                                       ≠  Net Debt
  ✗ "Total equity" / "Total assets"                           ≠  any listed metric
  ✗ "Net income attributable to non-controlling interests"    ≠  Net Income (it is a subset/deduction)
  ✗ "Net income attributable to [subsidiary/segment]"         ≠  Net Income (use the consolidated total)
  ✓ Net Income = the TOTAL "Net income" line from the Consolidated Statements of Income/Operations.
     If you see two figures for Net Income in the same year, you are reading the wrong line —
     use the larger consolidated total, NOT the amount attributable to a subset.
  ✗ Upstream / Downstream / Corporate column values in a segment table — skip them;
     extract ONLY from the "Consolidated" (rightmost total) column.
  ✗ Do NOT write negative values for asset metrics (Cash and Equivalents, Short-term Investments,
     Current Assets). These appear as deductions inside gearing calculations but are always
     positive assets — strip the minus sign and write the absolute value.
  ✗ "Long-term debt" from the Balance Sheet may include the current portion — use the TOTAL
     long-term debt line, not the current-portion-only line.
  ✗ Interest Expense: use the standalone "Interest expense" line from the Income Statement.
     Do NOT use "Net interest expense" if it nets against interest income — prefer the gross line.
  ✓ Current Assets = "Total current assets" from the Balance Sheet.
  ✓ Current Liabilities = "Total current liabilities" from the Balance Sheet.

OCR ARTIFACT RULE:
  Values like "393.891" or "452.753" (1–3 digits, period, exactly 3 digits) are large financial
  amounts where the PDF comma was misread as a period. Write "393.891" as "393891", etc.

==================== TEXT TO ANALYZE ====================
{text}
=========================================================

The company in this document is: {main_company}

VERIFICATION REQUIREMENT: For each metric you extract, you MUST be able to point to the exact sentence or table row in the text above where it appears. If you cannot find it in the text, DO NOT extract it.

Return ONLY a valid JSON array (no other text):
[
  {{"metric": "Revenue", "value": "1850.3", "unit": "$ million", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "EBITDA", "value": "374.9", "unit": "$ million", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "Total Debt", "value": "1200.0", "unit": "$ million", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "Interest Expense", "value": "45.2", "unit": "$ million", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "Current Assets", "value": "800.5", "unit": "$ million", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "Current Liabilities", "value": "420.1", "unit": "$ million", "year": "2024", "organization": "{main_company}"}}
]

If you cannot find any metrics with explicit values in the text above, return: []

STRICT Rules:
- Only extract a metric when its row label matches one of the mappings above AND appears in the text
- value: Take the most recent year column. Strip commas. Apply the OCR artifact rule.
  Values in parentheses are negative: "(216,642)" → "-216642".
- unit: Combine the currency symbol AND the scale from the document into one string:
  · "$" or "USD" + "millions" → "$ million"   · "€" + "thousands" → "€ thousand"
  · No currency symbol but scale present → "million" or "thousand" (no currency prefix)
  · No currency and no scale stated → "ratio"
  Read ONLY what is explicitly written in the text. Do NOT assume any currency.
  Use "%" for Gearing Ratio and ROACE only.
  For Earnings per Share: EPS is a per-share dollar amount, NEVER scaled by millions.
    Write just the currency symbol (e.g. "$"). NEVER write "million per share" — that is always wrong.
  When both local-currency and USD columns exist, use the local-currency column.
- year: Most recent year shown IN THE TEXT ABOVE.
- organization: ALWAYS use "{main_company}".
- Return [] if no mapped row label appears in the text.
- DO NOT use external knowledge or your training data about {main_company}.
- ONLY extract what is explicitly written in the text between the === markers above.
""",
        entity_parser=parse_metric_entity,
        entity_parser_kwargs={}
    )
}


def get_relation_config(relation_name: str) -> RelationConfig:
    """Get configuration for a specific relation type"""
    return RELATION_CONFIGS.get(relation_name.upper())


def set_main_company(company_name: str):
    """Inject the main company name into all relation configs at runtime."""
    for cfg in RELATION_CONFIGS.values():
        cfg.entity_parser_kwargs['main_company'] = company_name


def list_available_relations() -> List[str]:
    """List all available relation types"""
    return list(RELATION_CONFIGS.keys())
