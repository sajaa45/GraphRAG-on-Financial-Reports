import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base import RelationConfig, OCR_FIX_RE, CLEAN_VALUE_RE, TRAILING_X_RE, OCR_PERIOD_RE, ACCOUNTING_NEG_RE
from typing import Dict, Optional


def parse_metric_entity(entity: Dict, main_company: str = 'the Company') -> Optional[Dict]:
    metric       = str(entity.get('metric', '')).strip()
    value        = str(entity.get('value',  '')).strip()
    unit         = str(entity.get('unit',   'ratio')).strip()
    year         = str(entity.get('year',   '')).strip()
    org          = str(entity.get('organization', main_company)).strip() or main_company
    category     = str(entity.get('category', '')).strip()
    gaap_concept = str(entity.get('gaap_concept', '') or metric).strip()

    if not metric or not value:
        return None

    if value in ('—', '–', '-', '—', 'N/A', 'n/a', 'NA', 'na'):
        print(f"[FILTER] Rejected placeholder value '{value}' for metric '{metric}'")
        return None

    value = OCR_FIX_RE.sub(r'\1,\2', value)
    value_clean = CLEAN_VALUE_RE.sub('', value)
    value_clean = TRAILING_X_RE.sub('', value_clean)
    if OCR_PERIOD_RE.match(value_clean):
        value_clean = value_clean.replace('.', '')
    m = ACCOUNTING_NEG_RE.match(value_clean)
    if m:
        value_clean = '-' + m.group(1)

    try:
        numeric_value = float(value_clean)
    except ValueError:
        return None

    if numeric_value == 0.0:
        print(f"[FILTER] Rejected zero value for metric '{metric}'")
        return None

    metric_name = f"{metric} ({year})" if year else metric
    
    # citation_id will be assigned during Neo4j building phase
    # when we have access to a global counter

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Metric',
            'name': metric_name,
            'properties': {
                'value': value_clean,
                'unit': unit,
                'year': year,
                'metric_type': metric,
                'gaap_concept': gaap_concept,
                'category': category,
            },
        },
        'relationship': 'HAS_METRIC',
        'properties': {},
    }


CONFIG = RelationConfig(
    name='HAS_METRIC',
    source_entity_type='Company',
    target_entity_type='Metric',
    relationship_type='HAS_METRIC',
    required_fields=['metric', 'value', 'unit', 'year', 'category'],
    section_queries=[
        "results of operations",
        "liquidity and capital resources",
        "quantitative and qualitative disclosures about market risk",
        "consolidated statements of income",
        "consolidated statements of operations",
        "consolidated statements of earnings",
        "consolidated statements of cash flows",
        "consolidated balance sheets",
        "consolidated statements of financial position",
        "Financial Statements and Supplementary Data",
        "segment information",
        "non-gaap",
        "financing",
        "credit ratings",
        "cash flow activities",
        "capital requirements",
        "interest expense",
        "income taxes",
        "revenue",
    ],
    extraction_prompt_template="""/no_think

            Extract financial metrics explicitly stated in the text.

            Rules:
            - Copy the metric label exactly as written into "metric".
            - Extract only metrics with explicit numeric values.
            - Never infer, calculate, combine, or rename metrics.
            - In narrative text, the metric label and value must appear in the same sentence.
            - If multiple years appear, extract only the most recent reported year.
            - Do not extract from unlabeled tables.
            - If multiple figures map to the same GAAP concept, extract only the one
              that is the consolidated company-level total — skip subtotals, breakdowns,
              or partial figures for the same concept.
            - Add "gaap_concept": the closest standard US GAAP line-item name for this metric
              (e.g. "Revenue", "Operating Income (Loss)", "Net Income (Loss)", "Total Debt",
              "Interest Expense", "EBITDA", "Cash and Cash Equivalents").
              Use the standard GAAP name — not the company's custom phrasing.
              If no standard GAAP concept exists for this metric, set "gaap_concept": "no gaap".

            Important exclusions:
            - Operating cash flow ≠ Free Cash Flow
            - Net leverage ratio ≠ Net Debt
            - Capitalized interest ≠ Capital Expenditure
            - Debt maturity schedules ≠ Total Debt

            For each metric, assign exactly one category from the five below.
            Use the descriptions to decide — do NOT use the example labels as an exhaustive list.

            Categories:
            - "Leverage": Metrics that measure debt, liabilities, or obligations relative to earnings, equity, or assets. Includes total debt, long-term debt, total liabilities, debt-to-equity ratios, net debt, and leverage ratios. ALWAYS categorize liability and debt amounts here.
            - "Coverage": Metrics that assess whether a company generates enough earnings or cash flow to meet its interest and debt obligations. Includes interest coverage ratios, EBITDA coverage, debt service coverage, and fixed charge coverage ratios.
            - "Liquidity": Metrics that reflect a company's ability to meet near-term obligations using available cash, liquid assets, or short-term cash generation. Includes cash and cash equivalents, working capital, current ratio, quick ratio, and operating cash flow.
            - "Profitability": Metrics that capture the company's ability to generate positive earnings from its operations. ONLY include revenue, sales, gross profit, operating income, net income, EBITDA, EBIT, and margin percentages. EXCLUDE losses, expenses, liabilities, and debt.
            - "Debt Structure": Metrics that describe how the company's debt is composed or organized — its maturity profile, cost of debt, debt type mix, credit standing, and covenant flexibility. Includes interest rates, debt maturities, credit ratings, and covenant terms.

            Return ONLY a valid JSON array.

            Format:
            [
            {{
                "metric": "Net sales",
                "gaap_concept": "Revenue",
                "value": "2118.5",
                "unit": "$ million",
                "year": "2024",
                "category": "Profitability",
                "organization": "{main_company}"
            }}
            ]

            If nothing valid is found:
            []

            TEXT:
            {text}""",
    entity_parser=parse_metric_entity,
)