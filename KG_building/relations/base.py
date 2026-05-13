import re
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

# Shared numeric-cleaning regexes used by entity parsers
OCR_FIX_RE       = re.compile(r'(\d),\s+(\d)')
CLEAN_VALUE_RE   = re.compile(r'[,₹#$€£¥\s]')
TRAILING_X_RE    = re.compile(r'[xX]$')
OCR_PERIOD_RE    = re.compile(r'^\d{1,3}\.\d{3}$')
ACCOUNTING_NEG_RE = re.compile(r'^\(([0-9.]+)\)$')


@dataclass(slots=True)
class RelationConfig:
    name: str
    source_entity_type: str
    target_entity_type: str
    relationship_type: str
    extraction_prompt_template: str
    entity_parser: Callable[..., Optional[Dict]]
    entity_parser_kwargs: Dict[str, Any] = field(default_factory=dict)
    section_queries: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)