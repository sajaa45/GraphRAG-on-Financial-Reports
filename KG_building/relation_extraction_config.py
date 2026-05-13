import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relations'))

from typing import Dict, List

# Re-export RelationConfig so existing imports keep working
from base import RelationConfig

from HAS_METRIC.entities_extraction  import CONFIG as _HAS_METRIC
from FACES_RISK.entities_extraction  import CONFIG as _FACES_RISK
from OPERATES_IN.entities_extraction import CONFIG as _OPERATES_IN

RELATION_CONFIGS: Dict[str, RelationConfig] = {
    'HAS_METRIC':  _HAS_METRIC,
    'FACES_RISK':  _FACES_RISK,
    'OPERATES_IN': _OPERATES_IN,
}


def get_relation_config(relation_name: str) -> RelationConfig:
    return RELATION_CONFIGS.get(relation_name.upper())


def set_main_company(company_name: str):
    for cfg in RELATION_CONFIGS.values():
        cfg.entity_parser_kwargs['main_company'] = company_name


def list_available_relations() -> List[str]:
    return list(RELATION_CONFIGS.keys())