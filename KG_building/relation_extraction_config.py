

import re
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

#clean the numeric values
_OCR_FIX_RE = re.compile(r'(\d),\s+(\d)') #remove , : 1,000->1000
_CLEAN_VALUE_RE = re.compile(r'[,₹#$€£¥\s]') #remove currencies: €500->500
_TRAILING_X_RE = re.compile(r'[xX]$') #remove x and X: 10x->10
_OCR_PERIOD_RE = re.compile(r'^\d{1,3}\.\d{3}$') #remove . when there are 3 digits after . (.252) and before is not 0 and not >3 digits
_ACCOUNTING_NEG_RE = re.compile(r'^\(([0-9.]+)\)$') #add negative if it is between (): (123.45)-> -123.45


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
    entity_parser_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_sections: int = 2
    n_chunks_per_section: int = 3
    chunk_similarity_threshold: float = 0.3
    chunk_keywords_list: List[str] = field(default_factory=list)
    deduplicate_chunks_across_keywords: bool = True
    section_priority_tiers: List[str] = field(default_factory=list)


#entities
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

    # reject when value=placeholder
    if value in ('—', '–', '-', '—', 'N/A', 'n/a', 'NA', 'na'):
        print(f"[FILTER] Rejected placeholder value '{value}' for metric '{metric}'")
        return None

    # "216, 642" → "216,642"
    value = _OCR_FIX_RE.sub(r'\1,\2', value)
    value_clean = _CLEAN_VALUE_RE.sub('', value)
    value_clean = _TRAILING_X_RE.sub('', value_clean)
    # "393.891" → "393891"
    if _OCR_PERIOD_RE.match(value_clean):
        value_clean = value_clean.replace('.', '')
    # (78078) → -78078
    m = _ACCOUNTING_NEG_RE.match(value_clean)
    if m:
        value_clean = '-' + m.group(1)

    try:
        numeric_value = float(value_clean)
    except ValueError:
        return None

    # reject zero values
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


#prmpts

RELATION_CONFIGS: Dict[str, RelationConfig] = {
    'CEO': RelationConfig(
        name='CEO',
        source_entity_type='Person',
        target_entity_type='Company',
        relationship_type='CEO_OF',
        section_keywords='',
        chunk_keywords='chief executive officer ceo president chairman executive officers names ages positions table registrant',
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
        entity_parser_kwargs={},
        n_sections=3,
        n_chunks_per_section=4,
        chunk_similarity_threshold=0.15
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
            "consolidated statements of income",
            "consolidated statements of operations",
            "statements of income",
            "statements of operations",
            "consolidated balance sheet",
            "consolidated statement of financial position",
            "statement of financial position",
            "consolidated statements of cash flows",
            "statements of cash flows",
            "results of operations",
            "financial highlights",
            "selected financial data",
            "selected consolidated financial data",
            "management discussion",
            "financial performance",
        ],
        extraction_prompt_template="""/no_think

            Extract financial metrics explicitly stated in the text.

            Rules:
            - Copy the metric label exactly as written.
            - Extract only metrics with explicit numeric values.
            - Never infer, calculate, combine, or rename metrics.
            - In narrative text, the metric label and value must appear in the same sentence.
            - If multiple years appear, extract only the most recent reported year.
            - Do not extract from unlabeled tables.

            Important exclusions:
            - Operating cash flow ≠ Free Cash Flow
            - Net leverage ratio ≠ Net Debt
            - Capitalized interest ≠ Capital Expenditure
            - Debt maturity schedules ≠ Total Debt

            Return ONLY a valid JSON array.

            Format:
            [
            {{
                "metric": "Net sales",
                "value": "2118.5",
                "unit": "$ million",
                "year": "2024",
                "organization": "{main_company}"
            }}
            ]

            If nothing valid is found:
            []

            TEXT:
            {text}""",
        entity_parser=parse_metric_entity,
        entity_parser_kwargs={}
    ),

    'FACES_RISK': RelationConfig(
        name='FACES_RISK',
        source_entity_type='Company',
        target_entity_type='Risk',
        relationship_type='FACES_RISK',
        section_keywords='risk factors',
        chunk_keywords='could materially adversely affect business financial condition operations results',
        n_sections=2,
        n_chunks_per_section=10,
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
        section_keywords='item 1. business',
        chunk_keywords='company primary business activity manufacturing production products principal products operations producer supplier processing industrial services ',
        n_sections=1,
        n_chunks_per_section=3,
        chunk_similarity_threshold=0.3,
        extraction_prompt_template="""/no_think

Extract the PRIMARY industry of {main_company} ONLY.

Text:
{text}

Return ONLY a valid JSON array:
[
{{
"industry": "Industry Name",
"sector": "Sector Name"
}}
]

Rules:

* ONLY classify {main_company}
* Return exactly ONE entry
* Use a stable industrial/business category
* Prefer industry labels commonly associated with SEC/SIC-style classifications
* Avoid niche, marketing, or overly specialized labels
* Avoid overly broad parent sectors
* The industry should reasonably map to a commonly used SIC category
* Prefer practical categories used across multiple public companies
* If uncertain, choose the broader established industry
* If unclear, return []

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
