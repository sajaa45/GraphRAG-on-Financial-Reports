"""
Relation extraction configuration registry.
Dynamically imports relation configs from the relations/ directory.
"""

import os
import sys
import importlib.util
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relations'))

from base import RelationConfig

# Global registry
_RELATION_CONFIGS: Dict[str, RelationConfig] = {}
_MAIN_COMPANY = "the Company"


def _discover_relations():
    """Auto-discover relation configs from relations/ subdirectories."""
    relations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relations')
    
    for item in os.listdir(relations_dir):
        item_path = os.path.join(relations_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # Look for entities_extraction.py in each subdirectory
        extraction_file = os.path.join(item_path, 'entities_extraction.py')
        if not os.path.exists(extraction_file):
            continue
        
        # Import the module
        module_name = f"{item}_entities_extraction"
        spec = importlib.util.spec_from_file_location(module_name, extraction_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the CONFIG object
            if hasattr(module, 'CONFIG'):
                config = module.CONFIG
                if isinstance(config, RelationConfig):
                    _RELATION_CONFIGS[config.name.upper()] = config


def get_relation_config(relation_name: str) -> RelationConfig:
    """Get a relation config by name."""
    if not _RELATION_CONFIGS:
        _discover_relations()
    
    relation_name = relation_name.upper()
    if relation_name not in _RELATION_CONFIGS:
        raise ValueError(f"Unknown relation: {relation_name}. Available: {list(_RELATION_CONFIGS.keys())}")
    
    return _RELATION_CONFIGS[relation_name]


def list_available_relations() -> List[str]:
    """List all available relation names."""
    if not _RELATION_CONFIGS:
        _discover_relations()
    
    return sorted(_RELATION_CONFIGS.keys())


def set_main_company(company_name: str):
    """Set the main company name for entity parsing."""
    global _MAIN_COMPANY
    _MAIN_COMPANY = company_name


def get_main_company() -> str:
    """Get the current main company name."""
    return _MAIN_COMPANY


# Auto-discover on import
_discover_relations()
