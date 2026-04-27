#!/usr/bin/env python3
"""
Configuration for multi-relation extraction pipeline
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
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
    # If non-empty, one Qdrant query is issued per entry; results are union-deduplicated.
    # Overrides chunk_keywords when set.
    chunk_keywords_list: List[str] = field(default_factory=list)


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
        'target': {'type': 'Organization', 'name': org},
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

    import re
    # Strip internal spaces from OCR-spaced thousands (e.g. "216, 642" → "216,642")
    value = re.sub(r'(\d),\s+(\d)', r'\1,\2', value)
    # Strip commas, currency symbols, and trailing 'x'/'times'
    value_clean = value.replace(',', '').strip()
    value_clean = re.sub(r'[₹#$€£¥\s]', '', value_clean)
    value_clean = re.sub(r'[xX]$', '', value_clean).strip()
    # Normalize OCR period-as-comma artifact: "393.891" → "393891"
    # Pattern: 1-3 digits, period, exactly 3 digits (OCR comma in large round numbers)
    if re.match(r'^\d{1,3}\.\d{3}$', value_clean):
        value_clean = value_clean.replace('.', '')
    # Convert accounting negatives: (78078) → -78078
    m = re.match(r'^\(([0-9.]+)\)$', value_clean)
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


def parse_risk_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    risk_name = str(entity.get('risk_name', '')).strip()
    description = str(entity.get('description', '')).strip()
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
        target_entity_type='Organization',
        relationship_type='CEO_OF',
        section_keywords='corporate governance overview introduction',
        chunk_keywords='president and ceo chief executive officer ceo president chief executive',
        extraction_prompt_template="""Extract ONLY the CURRENT CEO from this text. Ignore board members, CFOs, former executives.

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
        chunk_keywords='',  # Overridden by chunk_keywords_list below
        chunk_keywords_list=[
            'gearing ratio net debt total equity leverage debt ratio',
            'free cash flow operating cash flow capital expenditure investment',
            'EBITDA earnings before interest tax depreciation amortization operating profit',
            'ROACE return on average capital employed return on equity',
            'net income profit loss attributable earnings per share',
            'revenue total sales income turnover',
            'interest coverage ratio EBIT debt service fixed charge',
            'current ratio quick ratio liquidity cash equivalents short-term',
            'total debt borrowings long-term debt bonds notes payable',
        ],
        n_sections=3,
        n_chunks_per_section=3,
        extraction_prompt_template="""Extract ALL financial metrics that appear with explicit numeric values in the text.

ROW-LABEL → STANDARD NAME MAPPING
Match the row label from the table (case-insensitive). Use the standard name shown on the right.

  Row label variants                                                  → Standard metric name
  "Gearing" / "Gearing ratio"                                         → "Gearing Ratio"
  "Net debt" / "Net debt (cash)"                                       → "Net Debt"
  "Free cash flow"                                                     → "Free Cash Flow"
  "EBITDA" / "Adjusted EBITDA"                                         → "EBITDA"
  "ROACE"                                                              → "ROACE"
  "Capital expenditures" / "Capital expenditures - cash basis"         → "Capital Expenditure"
  "Net income" / "Net income attributable to the ordinary shareholders" → "Net Income"
  "Earnings per share" / "Basic earnings per share"                    → "Earnings per Share"
  "Revenue" / "Total revenues" / "External revenue" (Consolidated only)→ "Revenue"
  "Earnings (losses) before interest, income taxes and zakat"
    / "Operating income" / "EBIT"                                      → "EBIT"
  "Total borrowings" / "Total borrowings (current and non-current)"    → "Total Debt"
  "Cash and cash equivalents"                                          → "Cash and Equivalents"
  "Short-term investments"                                             → "Short-term Investments"

CRITICAL — DO NOT confuse these:
  ✗ "Acquisition of right-of-use assets"         ≠  Capital Expenditure (it is a lease asset addition)
  ✗ "Net cash provided by operating activities"  ≠  Free Cash Flow
  ✗ "Net cash used in investing activities"      ≠  any listed metric
  ✗ "Net cash used in financing activities"      ≠  any listed metric
  ✗ "Total liabilities"                          ≠  Net Debt
  ✗ "Total equity" / "Total assets"              ≠  any listed metric
  ✗ Upstream / Downstream / Corporate column values in a segment table — skip them;
     extract ONLY from the "Consolidated" (rightmost total) column.
  ✗ Do NOT write negative values for asset metrics (Cash and Equivalents, Short-term Investments).
     These appear as deductions inside gearing calculations but are always positive assets —
     strip the minus sign and write the absolute value.

OCR ARTIFACT RULE:
  Values like "393.891" or "452.753" (1–3 digits, period, exactly 3 digits) are large financial
  amounts where the PDF comma was misread as a period. Write "393.891" as "393891", etc.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"metric": "Gearing Ratio", "value": "4.5", "unit": "%", "year": "2024", "organization": "{main_company}"}},
  {{"metric": "Net Debt", "value": "78078", "unit": "SAR million", "year": "2024", "organization": "{main_company}"}}
]

STRICT Rules:
- Only extract a metric when its row label matches one of the mappings above.
- value: Take the most recent year column. Strip commas. Apply the OCR artifact rule.
  Values in parentheses are negative: "(216,642)" → "-216642".
- unit: Read from the header note "All amounts in X unless otherwise stated".
  Use "%" for Gearing Ratio and ROACE.
  For Earnings per Share use "<currency> per share" (e.g. "SAR per share").
  Net Income in the EPS table is in SAR millions (same scale as consolidated statements).
  When both local-currency and USD columns exist, use the local-currency column.
  If no header unit is present, write "million" without a currency prefix.
- year: Most recent year shown.
- organization: ALWAYS use "{main_company}".
- Return [] if no mapped row label appears in the text.
- DO NOT use external knowledge.
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
        extraction_prompt_template="""Extract ALL risks explicitly disclosed in this text.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"risk_name": "Crude oil supply and demand fluctuations", "description": "concise description using the text's own wording, 80-160 chars", "organization": "{main_company}"}}
]

STRICT Rules:
- risk_name: Use the exact heading or short title from the text (e.g. "Terrorism and armed conflict",
  "Regulatory changes", "Climate change and GHG emissions targets"). Do NOT invent category labels
  like "Geopolitical_Risk" or "Market_Risk" — use what the document actually says.
- description: Paraphrase the text's own wording — 80-160 characters. No generic filler.
- Do NOT add a severity field. The report does not assign severity ratings.
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
        extraction_prompt_template="""Extract the PRIMARY industry of {main_company} ONLY.

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
