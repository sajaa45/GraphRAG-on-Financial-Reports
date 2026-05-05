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
    # Fix D: ordered list of section title substrings (lowercase). Earlier entries get a
    # larger similarity boost so the most structured financial sections surface first.
    section_priority_tiers: List[str] = field(default_factory=list)


# ============================================================================
# ENTITY PARSERS
# ============================================================================

def parse_person_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    person = str(entity.get('person', '')).strip()
    role = str(entity.get('role', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company
    is_current = entity.get('is_current', False)

    if not person or len(person) <= 2 or not is_current:
        return None

    name_parts = person.split()
    if len(name_parts) < 2:
        return None

    role_lower = role.lower()
    if 'ceo' in role_lower or 'chief executive' in role_lower:
        rel_type = 'CEO_OF'
    elif 'cfo' in role_lower or 'chief financial' in role_lower:
        rel_type = 'CFO_OF'
    elif 'board' in role_lower or 'director' in role_lower:
        rel_type = 'BOARD_MEMBER_OF'
    else:
        rel_type = 'WORKS_AT'

    return {
        'source': {'type': 'Person', 'name': person},
        'target': {'type': 'Company', 'name': org},
        'relationship': rel_type,
        'properties': {'role': role}
    }


def parse_metric_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    metric = str(entity.get('metric', '')).strip()
    value = str(entity.get('value', '')).strip()
    unit = str(entity.get('unit', 'ratio')).strip()
    year = str(entity.get('year', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not metric or not value:
        return None

    # FIX: Reject em-dash and similar placeholder values
    if value in ('—', '–', '-', '—', 'N/A', 'n/a', 'NA', 'na'):
        print(f"[FILTER] Rejected placeholder value '{value}' for metric '{metric}'")
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
        numeric_value = float(value_clean)
    except ValueError:
        return None

    # FIX: Reject zero values (often placeholder or irrelevant)
    if numeric_value == 0.0:
        print(f"[FILTER] Rejected zero value for metric '{metric}'")
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


def parse_risk_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    risk_name = str(entity.get('risk_name', '')).strip()
    description = str(entity.get('description', '')).strip()
    why = str(entity.get('why', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not risk_name:
        return None

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Risk',
            'name': risk_name,
            'properties': {
                'description': description,
                'why': why,
            }
        },
        'relationship': 'FACES_RISK',
        'properties': {}
    }


def parse_industry_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    industry = str(entity.get('industry', '')).strip()
    sector = str(entity.get('sector', '')).strip()
    # Allow entity itself to carry the company name as a fallback
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not industry:
        return None

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Industry',
            'name': industry,
            'properties': {'sector': sector}
        },
        'relationship': 'OPERATES_IN',
        'properties': {}
    }


# ============================================================================
# RELATION CONFIGURATIONS
# ============================================================================

RELATION_CONFIGS: Dict[str, RelationConfig] = {
    'CEO': RelationConfig(
        name='CEO',
        source_entity_type='Person',
        target_entity_type='Company',
        relationship_type='CEO_OF',
        section_keywords='corporate governance executive officers overview introduction',
        chunk_keywords='president and ceo chief executive officer ceo president chief executive information about our executive officers named executive',
        extraction_prompt_template="""/no_think
Extract ONLY the CURRENT CEO from this text. Ignore board members, CFOs, former executives.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"person": "Full Name", "role": "CEO", "organization": "{main_company}", "is_current": true}}
]

STRICT Rules:
- Extract ONLY the person explicitly called "President and CEO", "Chief Executive Officer", or "CEO" who currently holds the position.
- Ignore anyone with titles like "Chairman", "Director", "CFO", "Executive Vice President", or past tense.
- organization: ALWAYS use "{main_company}".
- Extract exactly ONE person.
- Return empty array [] if no current CEO is clearly identified.
""",
        entity_parser=parse_person_entity,
        entity_parser_kwargs={}
    ),

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
        n_chunks_per_section=10,
        chunk_similarity_threshold=0.15,
        deduplicate_chunks_across_keywords=False,
        section_priority_tiers=[
            # Tier 1 — structured statement tables (highest boost)
            "consolidated statements of income",
            "consolidated statements of operations",
            "statements of income",
            "statements of operations",
            "consolidated balance sheet",
            "consolidated statement of financial position",
            "statement of financial position",
            "consolidated statements of cash flows",
            "statements of cash flows",
            # Tier 2 — narrative sections with financial numbers
            "results of operations",
            "financial highlights",
            "selected financial data",
            "selected consolidated financial data",
            # Tier 3 — broad fallback
            "management discussion",
            "financial performance",
        ],
        extraction_prompt_template="""/no_think
You are a strict data extraction auditor. Your ONLY job is to copy financial metrics that appear EXPLICITLY in the text below. You must NOT use any external knowledge or training data about this company.

CRITICAL GROUNDING RULES:
1. If a metric name or value does NOT appear in the text below, you MUST return [] for that metric
2. You MUST quote the exact text snippet where you found each metric (we will verify this)
3. DO NOT use your knowledge about {main_company} - ONLY extract what is written in this specific text

IMPORTANT: You are a strict auditor. If the chunk_text does NOT contain the financial metric with an explicit numeric value, do not invent one. Do not provide values from your own knowledge; if you cannot find the metric and its value in the text, do not return it.

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
  ✓ Extract Net Income and EBIT from tables titled (or headed):
      "Consolidated Statements of Income", "Consolidated Statements of Operations",
      "Statements of Income", or "Statements of Operations".
  ✓ Also extract Net Income and EBIT from clearly labeled narrative text, e.g.:
      "Net income for [year] was $X million" or "EBIT increased to $X million in [year]".
      Narrative sentences that name the metric and provide an explicit numeric value are valid sources.
  ✗ DO NOT extract Net Income or EBIT from the Balance Sheet, Statement of Financial Position,
      or Statement of Shareholders' Equity — values there represent cumulative or different
      accounting snapshots and are NOT the period income figures.
  ✗ Skip Net Income and EBIT ONLY when values appear in an unlabeled table with no surrounding
      context identifying the table as an income/operations statement or a narrative sentence.
NARRATIVE PARAGRAPH CAUTION:
  Dense narrative paragraphs often contain many different numbers in close proximity.
  Each number belongs to the specific label that directly precedes or follows it in the same
  sentence. Do NOT "borrow" a number from one sentence and assign it to a label from a
  different sentence. If a paragraph mentions restructuring charges, margins, interest rates,
  and net income in consecutive sentences, extract ONLY the sentence where the metric label
  and value appear together explicitly.
  Example of correct extraction:
    ✓ "Net income was $X million in [year]" → Net Income = X ([year])
  Example of wrong extraction:
    ✗ A value described as "impairment charges" or "restructuring charges" → do NOT label as EBIT
    ✗ A percentage described as "of sales" or "of revenue" → do NOT label as Revenue

CRITICAL — DO NOT confuse these:
  ✗ "Acquisition of right-of-use assets"                      ≠  Capital Expenditure (lease asset addition)
  ✗ "Net cash provided by operating activities" / "Cash flow from operations" /
      "Operating cash flow" / "Cash provided by operating activities"
                                                              ≠  Free Cash Flow
      These labels describe OPERATING cash flow, not free cash flow.
      Free Cash Flow = Operating Cash Flow MINUS Capital Expenditure.
      ONLY extract "Free Cash Flow" if the document explicitly uses that exact label
      (or "FCF") on a pre-computed line item. If no such label appears, do NOT extract it.
  ✗ "Net cash used in investing activities"                   ≠  any listed metric
  ✗ "Net cash used in financing activities"                   ≠  any listed metric
  ✗ "Cash flow from operations" / "Cash provided from continuing operations" /
      "Cash flows provided from operations" / "Operating cash flow"   ≠  Capital Expenditure
      These are operating cash flow figures. Capital Expenditure requires an explicit label such as
      "Capital expenditures", "Purchases of property, plant and equipment", or "Capex".
  ✗ Revenue must be an absolute monetary amount (e.g. a dollar or currency figure in millions).
      Do NOT extract a percentage as Revenue — any value followed by "%" or described as
      "of sales", "of revenue", or "margin" is a ratio, not a revenue figure.
  ✗ Restructuring charges, impairment charges, write-downs, and debt extinguishment expenses
      are NOT Net Income, EBIT, or any listed metric — skip any value described as a "charge",
      "write-down", "expense", or "loss" unless it is explicitly labeled as one of the mapped metrics.
  ✗ "Cash, cash equivalents and short-term investments were $X million" is a COMBINED figure.
      Do NOT assign this combined total to either Cash and Equivalents or Short-term Investments
      individually. Each must come from a separately labeled line item or sentence.
  ✗ Short-term Investments MUST come from a Balance Sheet line item explicitly labeled
      "Short-term investments" or "Marketable securities" (current). Do NOT extract from:
      · Narrative sentences that combine cash and investments
      · Any line that says "cash and investments" or "cash, cash equivalents and short-term investments"
      · Any value that is not a standalone Balance Sheet line item
  ✗ "Borrowings" / "Short-term borrowings" / "Current portion of long-term debt" ≠ Short-term Investments.
      These are LIABILITIES (debt), not assets. Never extract a borrowing as an investment.
  ✗ Net interest expense / Net interest deductions ≠ Interest Expense.
      Use only the standalone gross "Interest expense" line. If the text only mentions
      "net interest expense", do not extract it as Interest Expense.
  ✗ Loans outstanding under a revolving facility / letters of credit outstanding ≠ Net Debt.
      Net Debt requires an explicit "Net debt" label. Do not derive it.
  ✗ Bond or loan issuance sentences describe a historical transaction, not a current balance.
      If the sentence says "the Company issued $X", "the Company entered into $X",
      or "aggregate principal amount of $X", skip it — this is the original face value
      at issuance, not the current outstanding balance.
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
  ✗ DEBT MATURITY SCHEDULE: A table or sentence listing amounts due in future periods
      is a repayment schedule, not the current debt balance. Signs you are reading a
      maturity schedule:
      · The surrounding text describes when debt falls due: "due in [year]", "maturing in
        [year]", "in [year]", "thereafter", "after [year]"
      · Values are organized by future calendar years (any year after the reporting date)
      · The table has column or row headers that are calendar years
      Do NOT extract ANY such value as "Total Debt". Total Debt must come ONLY from the
      Balance Sheet "Long-term debt" / "Total debt" total line.
  ✗ Do NOT sum, calculate, or derive any value. If the exact figure is not stated as a single
     number in the text, skip that metric entirely. Only copy numbers that are explicitly printed.

OCR DECIMAL RULE:
  PDF text extraction sometimes splits a decimal number across lines, e.g.:
    "337" on one line and ".1" or "1" on the next.
  When you see a whole number immediately followed by a decimal fragment in the surrounding text,
  reconstruct the full number with the decimal: "337" + ".1" → "337.1".
  NEVER concatenate without the decimal point ("3371" from "337.1" is always wrong).

OCR ARTIFACT RULE:
  Values like "393.891" or "452.753" (1–3 digits, period, exactly 3 digits) are large financial
  amounts where the PDF comma was misread as a period. Write "393.891" as "393891", etc.

==================== TEXT TO ANALYZE ====================
{text}
=========================================================

The company in this document is: {main_company}

VERIFICATION REQUIREMENT: For each metric you extract, you MUST be able to point to the exact sentence or table row in the text above where it appears. If you cannot find it in the text, DO NOT extract it.

Return ONLY a valid JSON array. If you found 1 metric, return 1 object. If you found 3, return 3.
Example of correct output (only metrics actually found in the text):
[
  {{"metric": "Revenue", "value": "1234.5", "unit": "$ million", "year": "20XX", "organization": "{main_company}"}},
  {{"metric": "Net Income", "value": "98.7", "unit": "$ million", "year": "20XX", "organization": "{main_company}"}}
]
If no metrics found: []

JSON FORMAT RULES — these are hard requirements:
- Return ONLY the metrics you actually found with a concrete numeric value. Do NOT return objects
  for metrics that are absent from the text. If a metric is not in the text, simply omit it.
- Return EACH metric AT MOST ONCE — the most recent year only. Do NOT return the same metric
  name for 2024, 2023, and 2022 as three separate objects. Pick the most recent year and stop.
- Every field value MUST be a JSON string (double-quoted). NEVER use arrays, objects, or null:
    ✗ WRONG: "value": []    ✗ WRONG: "value": null    ✗ WRONG: "value": {{}}
    ✓ RIGHT: omit the object entirely when the value is not found in the text.
- Do NOT include comments of any kind: no // comments, no /* */ comments.
- Do NOT add trailing commas after the last element in the array.
- The output must parse correctly with Python's json.loads() — test it mentally before responding.

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
- year: Most recent REPORTED year shown IN THE TEXT as an explicit four-digit calendar year.
  Do NOT extract values from sentences using relative phrases such as "prior year",
  "previous year", "compared with the prior year", or "compared to last year".
  Do NOT use any year after the reporting period of this document — such years indicate
  forecasts, maturities, or future obligations, not reported results.
  Do NOT use a year that refers to a transaction date (e.g. a bond issuance year) rather
  than a reporting period.
  Only extract values paired with an explicit four-digit past or current reporting year.
- organization: ALWAYS use "{main_company}".
- Return [] if no mapped row label appears in the text.
- DO NOT use external knowledge or your training data about {main_company}.
- ONLY extract what is explicitly written in the text between the === markers above.
""",
        entity_parser=parse_metric_entity,
        entity_parser_kwargs={}
    ),

    'FACES_RISK': RelationConfig(
        name='FACES_RISK',
        source_entity_type='Company',
        target_entity_type='Risk',
        relationship_type='FACES_RISK',
        section_keywords='risk management exposure factors financial operational',
        chunk_keywords='could materially adversely affect business financial condition operations results',
        n_sections=3,
        n_chunks_per_section=5,
        extraction_prompt_template="""/no_think
Extract ALL risks explicitly disclosed in this text.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{
    "risk_name": "Crude oil supply and demand fluctuations",
    "description": "concise factual description using the text's own wording, 80-160 chars",
    "why": "specific factual evidence or mechanism stated in the text explaining why this is a risk, 80-200 chars",
    "organization": "{main_company}"
  }}
]

STRICT Rules:
- risk_name: Use the exact heading or short title from the text. Do NOT invent category labels
  like "Geopolitical_Risk" or "Market_Risk" — use what the document actually says.
- description: Paraphrase the text's own wording — 80-160 characters. State WHAT the risk is.
- why: The factual basis from the document — state WHY this is a risk using specific facts,
  numbers, mechanisms, or conditions mentioned in the text (e.g. "Company derives 60% of revenue
  from a single customer, making it vulnerable to that customer's financial health").
  Do NOT write generic statements like "this could affect the business". Be specific and factual.
- Do NOT add a severity field.
- organization: ALWAYS use "{main_company}".
- Extract each distinct risk as a separate object. Merge near-duplicates into one.
- Return [] if no risk is explicitly described in the text.
- DO NOT use external knowledge — only what is written.
""",
        entity_parser=parse_risk_entity,
        entity_parser_kwargs={}
    ),

    'OPERATES_IN': RelationConfig(
        name='OPERATES_IN',
        source_entity_type='Company',
        target_entity_type='Industry',
        relationship_type='OPERATES_IN',
        section_keywords='overview strategy introduction',
        chunk_keywords='oil gas energy refining petrochemicals chemicals upstream downstream renewable energy marketing distribution production exploration',
        extraction_prompt_template="""/no_think
Extract the PRIMARY industry of {main_company} ONLY.

Text: {text}

Return ONLY a valid JSON array:
[
  {{"industry": "Oil & Gas", "sector": "Energy"}}
]

Rules:
- ONLY extract the industry of {main_company} — ignore all other companies, subsidiaries, or partners
- Return exactly ONE entry
- If unclear, return []
""",
        entity_parser=parse_industry_entity,
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
