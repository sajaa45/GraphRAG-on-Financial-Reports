

import re
import os
from typing import Dict, List, Set, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from industry_node_to_sic import get_sic_code



_TOKEN_CLEAN_RE = re.compile(r'[^a-z0-9 ]')
_STOP_WORDS = frozenset({
    'the', 'and', 'for', 'of', 'in', 'a', 'an',
    'company', 'corporation', 'incorporated', 'limited', 'group', 'holdings',
})


class CompanyDetector:
    """Detects and normalizes company names from document text."""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, 
                 embedding_fn, llm_fn=None):
       
        self.client = qdrant_client
        self.collection_name = collection_name
        self._embed = embedding_fn
        self._call_llm = llm_fn
        self.company_aliases: Set[str] = set()
        self.main_company: str = "the Company"
    
    def detect_main_company(self) -> str:
        
        print("  Auto-detecting main company from vector store...")
        
        #  top chunks about company overview
        try:
            query_vec = self._embed("annual report company overview")
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))]),
                limit=4,
                with_payload=True
            ).points
            docs = [r.payload["text"] for r in results]
        except Exception:
            self.company_aliases = set()
            return "the Company"

        context = "\n\n---\n\n".join(docs)

        #  llm detection 
        if self._call_llm and os.getenv("AWS_ACCESS_KEY_ID"):
            try:
                prompt = (
                    "Read the following excerpts from an annual report and return ONLY "
                    "the full legal name of the main company this report is about. "
                    "No explanation, just the name.\n\n" + context
                )
                name = self._call_llm(prompt).strip().strip('"').strip("'")
                if name and len(name) > 3:
                    print(f"  ✓ Detected main company: {name}")
                    self.company_aliases = {name}
                    self.main_company = name
                    return name
            except Exception as e:
                print(f"  ⚠ Bedrock detection failed ({e}), falling back to regex")
        
        #fallback to regex
        return self._detect_via_regex(docs)
    
    def _detect_via_regex(self, docs: List[str]) -> str:
        candidates: Dict[str, int] = {}
        ORG_PATTERNS = [
            r'\b((?:[A-Z][A-Za-z0-9&\'\-\.]+\s+){1,8}(?:Corporation|Incorporated|Limited|Company|Corp\.|Inc\.|Ltd\.|plc|LLC|L\.L\.C\.|S\.A\.|N\.V\.|AG|SE|GmbH))\b',
            r'\b((?:[A-Z][A-Za-z0-9&\'\-\.]+\s+){1,6}(?:Group|Holdings|Holding|Bancorp|Financial|Energy|Capital|Resources|Technologies|Industries|International|Enterprises|Partners))\b',
        ]
        
        for doc in docs:
            for pattern in ORG_PATTERNS:
                for match in re.finditer(pattern, doc):
                    name = match.group(1).strip()
                    if len(name) >= 8 and name.lower() not in ('the company', 'this company'):
                        candidates[name] = candidates.get(name, 0) + 1

        if not candidates:
            print("  ⚠ Could not detect company name — using 'the Company'")
            self.company_aliases = set()
            self.main_company = "the Company"
            return "the Company"

        # keep only the longest versions
        names = list(candidates.keys())
        filtered = {n for n in names if not any(n != o and n in o for o in names)}
        candidates = {n: v for n, v in candidates.items() if n in filtered}
        
        #pick most frequent (and longest if tied)
        canonical = max(candidates, key=lambda k: (candidates[k], len(k)))

        # build aliases based on token overlap
        canonical_tokens = set(self._significant_tokens(canonical))
        self.company_aliases = {
            n for n in candidates if canonical_tokens & set(self._significant_tokens(n))
        }
        
        print(f"  ✓ Detected main company: {canonical}")
        self.main_company = canonical
        return canonical
    
    #this agrees on one name when multiple ones are being extracted
    @staticmethod #method doesn't use self
    def _significant_tokens(name: str) -> List[str]:
        #keep only letters and digits
        return [t for t in _TOKEN_CLEAN_RE.sub('', name.lower()).split()
                #excludes stop words in company name
                if len(t) > 3 and t not in _STOP_WORDS]
    
    def normalize_company_name(self, name: str) -> str:
       
        if not name or name.lower() in ('the company', 'this company', ''):
            return self.main_company
        
        if name in self.company_aliases:
            return self.main_company
        
        # Check token overlap
        name_tokens = set(self._significant_tokens(name))
        canonical_tokens = set(self._significant_tokens(self.main_company))
        
        if name_tokens and canonical_tokens:
            overlap = len(name_tokens & canonical_tokens) / min(len(name_tokens), len(canonical_tokens))
            if overlap >= 0.5:
                return self.main_company
        
        return name


class SICLookup:
    
    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}
    
    def lookup(self, sector: str) -> Optional[str]:
        
        if not sector:
            return None
        
        key = sector.strip().lower()
        if key not in self._cache:
            try:
                code = get_sic_code(sector)
                self._cache[key] = str(code).strip()
                print(f"    ✓ SIC lookup: '{sector}' → {self._cache[key]}")
            except Exception as e:
                print(f"    ⚠ SIC lookup failed for '{sector}': {e}")
                self._cache[key] = None
        
        return self._cache[key]
