import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base import RelationConfig
from typing import Dict, Optional


def parse_risk_entity(entity: Dict, main_company: str = 'the Company') -> Optional[Dict]:
    risk_name   = str(entity.get('risk_name',   '')).strip()
    description = str(entity.get('description', '')).strip()
    why         = str(entity.get('why',         '')).strip()
    org         = str(entity.get('organization', main_company)).strip() or main_company

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
            },
        },
        'relationship': 'FACES_RISK',
        'properties': {},
    }


CONFIG = RelationConfig(
    name='FACES_RISK',
    source_entity_type='Company',
    target_entity_type='Risk',
    relationship_type='FACES_RISK',
    required_fields=['risk_name', 'description', 'why'],
    section_queries=[
        "risk factors",
    ],
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
)
