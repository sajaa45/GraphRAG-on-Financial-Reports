import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from relation_extraction_config import RelationConfig
from typing import Dict, Optional


def parse_industry_entity(entity: Dict, main_company: str = 'the Company') -> Optional[Dict]:
    industry = str(entity.get('industry',  '')).strip()
    sector   = str(entity.get('sector',    '')).strip()
    sic_code = str(entity.get('sic_code',  '')).strip()
    org      = str(entity.get('organization', main_company)).strip() or main_company

    if not industry:
        return None

    return {
        'source': {'type': 'TargetCompany', 'name': org},
        'target': {
            'type': 'Industry',
            'name': industry,
            'properties': {'sector': sector, 'sic_code': sic_code},
        },
        'relationship': 'OPERATES_IN',
        'properties': {},
    }


CONFIG = RelationConfig(
    name='OPERATES_IN',
    source_entity_type='TargetCompany',
    target_entity_type='Industry',
    relationship_type='OPERATES_IN',
    required_fields=['industry', 'sector'],
    section_queries=[
        'item 1. business',
    ],
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
)