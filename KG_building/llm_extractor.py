
import os
import sys
import json
import argparse
import time
import re
import boto3
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, QueryRequest

from relation_extraction_config import (
    RelationConfig,
    get_relation_config,
    list_available_relations,
    set_main_company,
)
from industry_node_to_sic import get_sic_code

_JSON_BLOCK_RE = re.compile(r'\[.*\]', re.DOTALL)

_STOP_WORDS = frozenset({
    'the', 'and', 'for', 'of', 'in', 'a', 'an',
    'company', 'corporation', 'incorporated', 'limited', 'group', 'holdings',
})
_TOKEN_CLEAN_RE = re.compile(r'[^a-z0-9 ]')

TOP_N_SECTIONS = 2
TOP_N_CHUNKS_PER_SECTION = 3
SECTION_SIMILARITY_THRESHOLD = 0.25
CHUNK_SIMILARITY_THRESHOLD = 0.3
MAX_CHUNKS_PER_LLM_BATCH = 8

# Fix A: synonym map for keyword expansion.
# Keys are lowercase substrings matched against each query keyword.
# Values list extra query variants whose embeddings are averaged in.
METRIC_SYNONYMS: Dict[str, List[str]] = {
    "ebitda": ["earnings before interest tax depreciation", "operating income before depreciation", "adjusted ebitda"],
    "roace": ["return on capital employed", "roce", "return on average capital employed"],
    "revenue": ["net sales", "net revenues", "total revenues", "turnover", "total net revenues"],
    "free cash flow": ["fcf", "levered free cash flow", "unlevered free cash flow"],
    "gearing": ["net debt to equity", "leverage ratio", "debt to equity ratio"],
    "net income": ["net profit", "net earnings", "profit for the year", "profit attributable"],
    "total debt": ["total borrowings", "long-term debt", "total financial debt", "total indebtedness"],
    "interest coverage": ["interest cover ratio", "ebit to interest", "fixed charge coverage ratio"],
    "current ratio": ["liquidity ratio", "quick ratio", "current assets liabilities"],
    "capital expenditure": ["capex", "purchases of property plant and equipment", "capital spending"],
}


class LLMExtractor:
    """Retrieves relevant chunks from Qdrant, calls Bedrock LLM to extract entities,
    and saves structured results to JSON for downstream Neo4j ingestion."""

    def __init__(self,
                 collection_name: str = "financial",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 output_dir: str = "/app/output",
                 main_company: str = "the Company",
                 source_file: str = ""):

        self.embedding_model = SentenceTransformer(embedding_model)
        self._embed_cache: Dict[str, list] = {}
        print(f"✓ Loaded embedding model: {embedding_model}")

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        print(f"✓ Connected to Qdrant collection: {collection_name}")

        self.bedrock_model = os.getenv("BEDROCK_MODEL", "meta.llama3-70b-instruct-v1:0")
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        print(f"✓ Bedrock client ready (model: {self.bedrock_model}, region: {region})")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.main_company = main_company
        if main_company == "the Company":
            self.main_company = self.detect_main_company()
        set_main_company(self.main_company)
        print(f"✓ Main company: {self.main_company}")

        self._sic_cache: Dict[str, str] = {}

        base = os.path.splitext(os.path.basename(source_file))[0] if source_file else collection_name
        self.log_file = os.path.join(self.output_dir, f"relationships_{base}.txt")
        self.json_file = os.path.join(self.output_dir, f"extracted_{base}.json")
        self.log_buffer = []
        self.start_time = time.time()

        print(f"✓ Logging to  : {self.log_file}")
        print(f"✓ JSON output : {self.json_file}")

    # ========================================================================
    # EMBEDDING / CACHING
    # ========================================================================
    def _embed(self, text: str) -> list:
        if text not in self._embed_cache:
            self._embed_cache[text] = self.embedding_model.encode([text])[0].tolist()
        return self._embed_cache[text]

    def _expand_keyword(self, keyword: str) -> List[str]:
        """Return keyword plus synonym variants from METRIC_SYNONYMS."""
        kw_lower = keyword.lower()
        for key, variants in METRIC_SYNONYMS.items():
            if key in kw_lower:
                return [keyword] + [v for v in variants if v.lower() not in kw_lower]
        return [keyword]

    def _embed_expanded(self, keyword: str) -> list:
        """Embed keyword with synonym expansion; return L2-normalised average vector."""
        variants = self._expand_keyword(keyword)
        if len(variants) == 1:
            return self._embed(variants[0])
        vecs = np.array([self._embed(v) for v in variants], dtype=np.float32)
        avg = vecs.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-9:
            avg = avg / norm
        return avg.tolist()

    def _rerank_chunks(self, chunks: List[Dict], keyword: str = "") -> List[Dict]:
        """Fix C: Re-score chunks by keyword density + numeric density + table bonus."""
        if not chunks:
            return chunks
        kw_words = {w.lower() for w in keyword.split() if len(w) > 3} if keyword else set()

        def _score(chunk: Dict) -> float:
            text = chunk["text"].lower()
            kw_hit = sum(1 for w in kw_words if w in text) / max(len(kw_words), 1) if kw_words else 0.0
            num_count = len(re.findall(r'\b\d[\d,\.]*\b', text))
            num_score = min(num_count / 15.0, 1.0)
            return chunk["similarity"] * 0.4 + kw_hit * 0.3 + num_score * 0.2

        return sorted(chunks, key=_score, reverse=True)

    def _apply_section_priority_boost(self, candidates, priority_tiers: List[str], keep_n: int):
        """Fix D: Re-rank section candidates using tiered title boost, return top keep_n."""
        scored = []
        for hit in candidates:
            title = hit.payload.get("title", "").lower()
            boost = 0.0
            for i, tier in enumerate(priority_tiers):
                if tier in title:
                    boost = 0.3 * (1.0 - i / len(priority_tiers))
                    break
            scored.append((hit, hit.score + boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [hit for hit, _ in scored[:keep_n]]
        if result:
            top_title = result[0].payload.get("title", "")
            print(f"  Section priority boost applied — top: '{top_title}'")
        return result

    # ========================================================================
    # LOGGING
    # ========================================================================
    def _log(self, message: str):
        self.log_buffer.append(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def _save_log(self):
        if self.log_buffer:
            elapsed = time.time() - self.start_time
            header = (
                f"Started : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n"
                f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Elapsed : {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
                f"{'='*80}\n\n"
            )
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write('\n'.join(self.log_buffer))
            print(f"\n✓ Relationships log saved to: {self.log_file}")

    # ========================================================================
    # COMPANY DETECTION / NORMALISATION
    # ========================================================================
    def detect_main_company(self) -> str:
        print("  Auto-detecting main company from vector store...")
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

        if os.getenv("AWS_ACCESS_KEY_ID"):
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
                    return name
            except Exception as e:
                print(f"  ⚠ Bedrock detection failed ({e}), falling back to regex")

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
            return "the Company"

        names = list(candidates.keys())
        filtered = {n for n in names if not any(n != o and n in o for o in names)}
        candidates = {n: v for n, v in candidates.items() if n in filtered}
        canonical = max(candidates, key=lambda k: (candidates[k], len(k)))

        canonical_tokens = set(self._significant_tokens(canonical))
        self.company_aliases = {
            n for n in candidates if canonical_tokens & set(self._significant_tokens(n))
        }
        print(f"  ✓ Detected main company: {canonical}")
        return canonical

    @staticmethod
    def _significant_tokens(name: str) -> List[str]:
        return [t for t in _TOKEN_CLEAN_RE.sub('', name.lower()).split()
                if len(t) > 3 and t not in _STOP_WORDS]

    def normalize_company_name(self, name: str) -> str:
        if not name or name.lower() in ('the company', 'this company', ''):
            return self.main_company
        if name in getattr(self, 'company_aliases', set()):
            return self.main_company
        name_tokens = set(self._significant_tokens(name))
        canonical_tokens = set(self._significant_tokens(self.main_company))
        if name_tokens and canonical_tokens:
            overlap = len(name_tokens & canonical_tokens) / min(len(name_tokens), len(canonical_tokens))
            if overlap >= 0.5:
                return self.main_company
        return name

    # ========================================================================
    # SIC LOOKUP
    # ========================================================================
    def _lookup_sic(self, sector: str) -> str:
        if not sector:
            return None
        key = sector.strip().lower()
        if key not in self._sic_cache:
            try:
                code = get_sic_code(sector)
                self._sic_cache[key] = str(code).strip()
                print(f"    ✓ SIC lookup: '{sector}' → {self._sic_cache[key]}")
            except Exception as e:
                print(f"    ⚠ SIC lookup failed for '{sector}': {e}")
                self._sic_cache[key] = None
        return self._sic_cache[key]

    # ========================================================================
    # HIERARCHICAL RETRIEVAL
    # ========================================================================
    def hierarchical_retrieval(self, relation_config,
                               n_sections=TOP_N_SECTIONS,
                               n_chunks_per_section=TOP_N_CHUNKS_PER_SECTION,
                               section_threshold=SECTION_SIMILARITY_THRESHOLD):
        print(f"\n{'='*60}")
        print(f"Hierarchical Retrieval: {relation_config.name}")
        print(f"{'='*60}")

        n_sections = relation_config.n_sections
        n_chunks_per_section = relation_config.n_chunks_per_section

        kw_list = relation_config.chunk_keywords_list or [relation_config.chunk_keywords]
        chunk_embeddings = [self._embed(kw) for kw in kw_list]
        print(f"  Using {len(chunk_embeddings)} chunk query vector(s)")

        section_embedding = self._embed(relation_config.section_keywords)
        priority_tiers = getattr(relation_config, 'section_priority_tiers', [])
        # Fix D: fetch a wider candidate pool so priority boost can re-rank meaningfully
        section_fetch_limit = n_sections * 3 if priority_tiers else n_sections
        section_results = self.client.query_points(
            collection_name=self.collection_name,
            query=section_embedding,
            query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="section"))]),
            limit=section_fetch_limit,
            with_payload=True
        ).points

        if priority_tiers and section_results:
            section_results = self._apply_section_priority_boost(section_results, priority_tiers, n_sections)

        use_sections = False
        if section_results:
            best_sim = section_results[0].score
            print(f"  Best section similarity: {best_sim:.3f}")
            if best_sim >= section_threshold:
                use_sections = True
                print(f"  ✓ Using {len(section_results)} relevant sections")

        seen_ids: set = set()
        all_chunks = []

        def _add_chunk(r, section_title, section_id):
            if r.id in seen_ids:
                return
            seen_ids.add(r.id)
            chunk_data = {
                "section_title": section_title,
                "section_id": section_id,
                "chunk_index": r.payload.get("chunk_index"),
                "similarity": r.score,
                "text": r.payload["text"],
            }
            if "source_page" in r.payload:
                chunk_data["source_page"] = r.payload["source_page"]
            all_chunks.append(chunk_data)

        if use_sections:
            requests, meta = [], []
            for hit in section_results:
                section_id = hit.payload["section_id"]
                section_title = hit.payload["title"]
                for emb in chunk_embeddings:
                    requests.append(QueryRequest(
                        query=emb,
                        filter=Filter(must=[
                            FieldCondition(key="type", match=MatchValue(value="chunk")),
                            FieldCondition(key="section_id", match=MatchValue(value=section_id)),
                        ]),
                        limit=n_chunks_per_section,
                        with_payload=True,
                    ))
                    meta.append((section_title, section_id))
            for (section_title, section_id), result in zip(
                meta, self.client.query_batch_points(self.collection_name, requests)
            ):
                for r in result.points:
                    _add_chunk(r, section_title, section_id)
        else:
            requests = [
                QueryRequest(
                    query=emb,
                    filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))]),
                    limit=n_sections * n_chunks_per_section,
                    with_payload=True,
                )
                for emb in chunk_embeddings
            ]
            for result in self.client.query_batch_points(self.collection_name, requests):
                for r in result.points:
                    _add_chunk(r, r.payload.get("section_title", "Unknown"), r.payload.get("section_id"))

        print(f"  ✓ Retrieved {len(all_chunks)} chunks total (deduplicated)")
        return all_chunks

    # ========================================================================
    # LLM CALLS
    # ========================================================================
    def _call_llm(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = self.bedrock.converse(
                    modelId=self.bedrock_model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
                )
                return response["output"]["message"]["content"][0]["text"].strip()
            except Exception as e:
                err = str(e)
                is_throttle = "ThrottlingException" in err or "429" in err
                is_overload = "ServiceUnavailableException" in err or "503" in err
                if (is_throttle or is_overload) and attempt < 4:
                    wait = 60 if is_overload else 30 * (attempt + 1)
                    print(f"    ⚠ Bedrock {'overloaded' if is_overload else 'rate limit'}... retrying in {wait}s ({attempt+1}/4)")
                    time.sleep(wait)
                else:
                    raise

    # ========================================================================
    # ENTITY EXTRACTION
    # ========================================================================
    def _extract_entities_batch(self, chunks: List[Dict], relation_config) -> List[tuple]:
        chunks_text = "\n\n---CHUNK SEPARATOR---\n\n".join(
            f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
        )
        prompt = relation_config.extraction_prompt_template.format(
            text=chunks_text,
            main_company=self.main_company
        )
        try:
            llm_output = self._call_llm(prompt)
            m = _JSON_BLOCK_RE.search(llm_output)
            if not m:
                print(f"[DEBUG] LLM returned no JSON block for llm! Output was: {llm_output[:500]}")

                return []
            entities_data = json.loads(m.group())
            results = []
            for e in entities_data:
                # For metrics, anchor to the chunk that actually contains the value
                # string — word-overlap is unreliable for dense numeric tables.
                value_str = str(e.get('value', '')).strip()
                if value_str:
                    best_chunk = next(
                        (c for c in chunks if value_str in c['text']),
                        max(chunks, key=lambda c: c['similarity'])
                    )
                else:
                    best_chunk = max(chunks, key=lambda c: sum(
                        1 for w in str(e).lower().split() if w in c['text'].lower()
                    ))
                
                # Validate against the specific chunk (not combined text)
                if not self._validate_entity_in_text(e, best_chunk['text'], relation_config.name, best_chunk):
                    continue
                    
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    for node_key in ('source', 'target'):
                        node = p[node_key]
                        if node['type'] in ('Company', 'Organization'):
                            node['name'] = self.normalize_company_name(node['name'])
                    results.append((p, best_chunk))
            return results
        except Exception as e:
            print(f"    ✗ Batch extraction error: {e}")
            return []

    def extract_entities_with_llm(self, text: str, relation_config):
        prompt = relation_config.extraction_prompt_template.format(
            text=text,
            main_company=self.main_company
        )
        try:
            llm_output = self._call_llm(prompt)
            m = _JSON_BLOCK_RE.search(llm_output)
            if not m:
                print(f"[DEBUG] LLM returned no JSON block! Output was: {llm_output[:500]}")
                return []
            entities_data = json.loads(m.group())
            parsed = []
            for e in entities_data:
                # No chunk info available in this method
                if not self._validate_entity_in_text(e, text, relation_config.name, chunk_info=None):
                    continue
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    for node_key in ('source', 'target'):
                        node = p[node_key]
                        if node['type'] in ('Company', 'Organization'):
                            node['name'] = self.normalize_company_name(node['name'])
                    parsed.append(p)
            return parsed
        except Exception as e:
            print(f"    ✗ Extraction error: {e}")
            return []

    def _validate_entity_in_text(self, entity: Dict, text: str, relation_type: str, chunk_info: Optional[Dict] = None) -> bool:
        text_lower = text.lower()
        metric = str(entity.get('metric', '')).strip()
        value = str(entity.get('value', '')).strip()
        
        # --- CHUNK ID NOTIFICATION ---
        if chunk_info:
            chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
            section_title = chunk_info.get('section_title', 'Unknown Section')
            print(f"\n[VALIDATION] Chunk ID: {chunk_id} | Section: {section_title}")
            print(f"[VALIDATION] Evaluating Entity: Metric='{metric}', Value='{value}'")

        if relation_type == 'CEO':
            person = str(entity.get('person', '')).strip()
            if not person or len(person) <= 2:
                return False
            name_parts = person.split()
            if len(name_parts) < 2:
                return False
            if name_parts[-1].lower() not in text_lower:
                return False
            role = str(entity.get('role', '')).lower()
            if 'ceo' in role or 'chief executive' in role:
                if not any(kw in text_lower for kw in ['ceo', 'chief executive', 'president']):
                    return False
            return True

        elif relation_type == 'HAS_METRIC':
            if not metric or not value:
                if chunk_info:
                    chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
                    print(f"[VALIDATION] Chunk {chunk_id}: FAILED - Missing metric/value")
                return False
                
            # Check 1: Direct Substring (Case insensitive)
            if value.lower() in text.lower():
                if chunk_info:
                    chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
                    print(f"[VALIDATION] Chunk {chunk_id}: PASSED - '{value}' found in text")
                return True

            # Check 2: Digit extraction
            core_digits = re.sub(r'[^\d]', '', value)[:5]
            if len(core_digits) >= 3:
                if core_digits in re.sub(r'[^\d]', '', text):
                    if chunk_info:
                        chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
                        print(f"[VALIDATION] Chunk {chunk_id}: PASSED - Digits '{core_digits}' found in text")
                    return True

            # Check 3: Patterns
            patterns = [
                rf'\b{re.escape(value)}\b',
                rf'[₹#$€£¥]\s*{re.escape(value)}',
                rf'{re.escape(value)}\s*[₹#$€£¥]',
                rf'\({re.escape(value)}\)',
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                if chunk_info:
                    chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
                    print(f"[VALIDATION] Chunk {chunk_id}: PASSED - Pattern match found")
                return True

            # --- VALIDATION FAILED ---
            if chunk_info:
                chunk_id = chunk_info.get('_id', chunk_info.get('chunk_index', 'Unknown'))
                print(f"[VALIDATION] Chunk {chunk_id}: FAILED - Could not find '{value}' in chunk text")
            return False
        elif relation_type == 'FACES_RISK':
            risk_name = str(entity.get('risk_name', '')).strip()
            if not risk_name:
                return False
            description = str(entity.get('description', '')).strip()
            combined = (risk_name + ' ' + description).lower()
            words = [w for w in re.split(r'\W+', combined) if len(w) > 4]
            if words:
                matches = sum(1 for w in words if w in text_lower)
                if matches < max(1, len(words) // 5):
                    print(f"      ⚠ Risk validation failed: not grounded in text ({matches}/{len(words)} words matched)")
                    return False
            return True

        elif relation_type == 'OPERATES_IN':
            industry = str(entity.get('industry', '')).strip()
            if not industry:
                return False
            if not any(kw in text_lower for kw in industry.lower().split() if len(kw) > 2):
                return False
            return True

        return True

    # ========================================================================
    # EXTRACTION ORCHESTRATION
    # ========================================================================
    def extract_relation(self, relation_name: str) -> List[Dict]:
        """Run LLM extraction for one relation and return the write_queue (list of items)."""
        relation_config = get_relation_config(relation_name)
        if not relation_config:
            print(f"✗ Unknown relation: {relation_name}")
            return []

        relation_start = time.time()
        self._log(f"\n{'='*80}")
        self._log(f"EXTRACTING RELATION: {relation_config.name}")
        self._log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*80}")
        self._log(f"Source: {relation_config.source_entity_type}")
        self._log(f"Target: {relation_config.target_entity_type}")
        self._log(f"Relationship: {relation_config.relationship_type}")
        self._log("")

        print(f"\n{'='*80}\nEXTRACTING RELATION: {relation_config.name}\n{'='*80}")
        print(f"Source: {relation_config.source_entity_type} → {relation_config.relationship_type} → {relation_config.target_entity_type}")

        write_queue = []
        ceo_candidates = []
        created = set()

        if relation_config.chunk_keywords_list:
            n_per_kw = relation_config.n_chunks_per_section
            total_kw = len(relation_config.chunk_keywords_list)
            threshold = relation_config.chunk_similarity_threshold
            fallback_threshold = threshold / 2
            print(f"\n  Per-keyword batch mode: {total_kw} keywords × {n_per_kw} chunks each (threshold: {threshold})")
            self._log(f"\n  Per-keyword batch mode: {total_kw} keywords × {n_per_kw} chunks each (threshold: {threshold})")

            # NEW: Pre-filter sections using priority tiers if configured
            priority_tiers = getattr(relation_config, 'section_priority_tiers', [])
            allowed_section_ids = None
            
            if priority_tiers:
                print(f"  ✓ Applying section priority filter with {len(priority_tiers)} tiers")
                self._log(f"  ✓ Applying section priority filter with {len(priority_tiers)} tiers")
                
                # Fetch candidate sections
                section_embedding = self._embed(relation_config.section_keywords)
                section_fetch_limit = relation_config.n_sections * 5  # Wider pool for filtering
                section_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=section_embedding,
                    query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="section"))]),
                    limit=section_fetch_limit,
                    with_payload=True
                ).points
                
                # Apply priority boost and select top sections
                if section_results:
                    section_results = self._apply_section_priority_boost(
                        section_results, 
                        priority_tiers, 
                        relation_config.n_sections
                    )
                    allowed_section_ids = {hit.payload["section_id"] for hit in section_results}
                    section_titles = [hit.payload.get("title", "Unknown") for hit in section_results[:3]]
                    print(f"  ✓ Filtered to {len(allowed_section_ids)} priority sections")
                    print(f"    Top sections: {', '.join(section_titles)}")
                    self._log(f"  ✓ Filtered to {len(allowed_section_ids)} priority sections: {section_titles}")

            global_seen_ids: set = set()

            def _to_chunk(r):
                return {
                    "section_title": r.payload.get("section_title", "Unknown"),
                    "section_id": r.payload.get("section_id"),
                    "chunk_index": r.payload.get("chunk_index"),
                    "source_page": r.payload.get("source_page"),
                    "similarity": r.score,
                    "text": r.payload["text"],
                    "_id": r.id,
                }

            for kw_idx, kw in enumerate(relation_config.chunk_keywords_list, 1):
                # Build query filter with optional section restriction
                if allowed_section_ids:
                    # Filter chunks to only those in priority sections using MatchAny
                    query_filter = Filter(
                        must=[
                            FieldCondition(key="type", match=MatchValue(value="chunk")),
                            FieldCondition(
                                key="section_id",
                                match=MatchAny(any=list(allowed_section_ids))
                            )
                        ]
                    )
                else:
                    query_filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))])
                
                kw_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=self._embed_expanded(kw),  # Fix A: synonym-averaged embedding
                    query_filter=query_filter,
                    limit=n_per_kw,
                    with_payload=True,
                ).points

                # Per-keyworad dedup set: always prevents duplicates within one batch.
                # Cross-keyword dedup is skipped when deduplicate_chunks_across_keywords=False
                # (e.g. HAS_METRIC) because a single financial table chunk contains many
                # distinct metrics — letting each keyword re-process it finds them all.
                dedup_ids = global_seen_ids if relation_config.deduplicate_chunks_across_keywords else set()

                kw_chunks = [
                    _to_chunk(r)
                    for r in kw_results
                    if r.score >= threshold and r.id not in dedup_ids
                ]

                # Fallback: if nothing passed the threshold, retry with a softer cut
                if not kw_chunks:
                    kw_chunks = [
                        _to_chunk(r)
                        for r in kw_results
                        if r.score >= fallback_threshold and r.id not in dedup_ids
                    ]
                    if kw_chunks:
                        print(f"  ↓ Fallback threshold ({fallback_threshold}) used for keyword {kw_idx}")
                        self._log(f"    ↓ Fallback threshold ({fallback_threshold}) used")

                # Update global seen set only when cross-keyword dedup is active.
                if relation_config.deduplicate_chunks_across_keywords:
                    for r in kw_results:
                        if r.score >= threshold:
                            global_seen_ids.add(r.id)

                print(f"\n  Keyword {kw_idx}/{total_kw}: '{kw[:60]}' → {len(kw_chunks)} chunks (best score: {kw_results[0].score:.3f} {('↓fallback' if kw_chunks and kw_chunks[0]['similarity'] < threshold else '')})" if kw_results else f"\n  Keyword {kw_idx}/{total_kw}: '{kw[:60]}' → 0 results from Qdrant")
                self._log(f"\nKeyword {kw_idx}/{total_kw}: {kw}")

                if not kw_chunks:
                    self._log("    ✗ No chunks above threshold (even with fallback)")
                    print("    ✗ No chunks above threshold")
                    continue

                kw_chunks = self._rerank_chunks(kw_chunks, kw)  # Fix C
                
                # Display and log chunk texts for this batch
                separator = f"\n  {'='*70}"
                header = f"  CHUNKS FOR KEYWORD {kw_idx}/{total_kw}"
                
                print(separator)
                print(header)
                print(f"  {'='*70}")
                self._log(separator)
                self._log(header)
                self._log(f"  {'='*70}")
                
                for chunk_idx, chunk in enumerate(kw_chunks, 1):
                    chunk_header = f"\n  Chunk {chunk_idx}/{len(kw_chunks)} (similarity: {chunk['similarity']:.3f})"
                    section_info = f"    Section: {chunk['section_title']}"
                    page_info = f"    Page: {chunk.get('source_page', 'N/A')}"
                    text_separator = f"    {'-'*68}"
                    
                    print(chunk_header)
                    print(section_info)
                    print(page_info)
                    print(text_separator)
                    
                    self._log(chunk_header)
                    self._log(section_info)
                    self._log(page_info)
                    self._log(text_separator)
                    self._log(f"{chunk['text']}")
                    self._log(text_separator)
                
                print(f"  {'='*70}\n")
                self._log(f"  {'='*70}\n")
                
                batch_results = self._extract_entities_batch(kw_chunks, relation_config)
                if not batch_results:
                    self._log("    ✗ No entities extracted")

                for entity, chunk in batch_results:
                    src = entity['source']
                    tgt = entity['target']
                    rel = entity['relationship']
                    key = (src['name'], rel, tgt['name'])
                    if key in created:
                        continue

                    log_msg = f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})"
                    if tgt.get('properties'):
                        log_msg += f"\n      Properties: {json.dumps(tgt['properties'], indent=8)}"
                    log_msg += f"\n      Metadata: page={chunk.get('source_page')} | section={chunk.get('section_title')}"
                    self._log(log_msg)
                    print(f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")

                    write_queue.append({
                        'src': src, 'tgt': tgt, 'rel': rel,
                        'props': entity.get('properties', {}),
                        'chunk_text': chunk['text'],
                        'similarity': chunk['similarity'],
                        'section_title': chunk.get('section_title'),
                        'source_page': chunk.get('source_page'),
                        'sic': None,
                    })
                    created.add(key)

        else:
            chunks = self.hierarchical_retrieval(relation_config)
            threshold = relation_config.chunk_similarity_threshold
            filtered_chunks = [c for c in chunks if c['similarity'] >= threshold]
            filtered_chunks = self._rerank_chunks(filtered_chunks)  # Fix C
            print(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {threshold})")
            self._log(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {threshold})")

            for i, chunk in enumerate(filtered_chunks, 1):
                self._log(f"\nChunk {i}/{len(filtered_chunks)} (similarity: {chunk['similarity']:.3f})")
                self._log(f"  Section: {chunk['section_title']}")
                self._log(f"  Section ID: {chunk['section_id']}")
                self._log(f"  Chunk Index: {chunk.get('chunk_index', 'N/A')}")
                if 'source_page' in chunk:
                    self._log(f"  Page: {chunk['source_page']}")
                self._log(f"  Text preview: {chunk['text'][:200]}...")
                self._log(f"  Full text:")
                self._log("-" * 80)
                self._log(chunk['text'])
                self._log("-" * 80)

                print(f"\nChunk {i}/{len(filtered_chunks)} | Similarity: {chunk['similarity']:.3f} | Section: {chunk['section_title']}")

                entities = self.extract_entities_with_llm(chunk['text'], relation_config)
                if not entities:
                    self._log("    ✗ No entities extracted")

                for entity in entities:
                    src = entity['source']
                    tgt = entity['target']
                    rel = entity['relationship']
                    key = (src['name'], rel, tgt['name'])

                    if rel == 'CEO_OF':
                        ceo_candidates.append({
                            'entity': entity,
                            'similarity': chunk['similarity'],
                            'text': chunk['text'],
                            'section_title': chunk.get('section_title'),
                            'source_page': chunk.get('source_page'),
                        })
                        self._log(f"    ~ CEO candidate: {src['name']} (confidence: {chunk['similarity']:.3f})")
                        print(f"    ~ CEO candidate: {src['name']} (sim: {chunk['similarity']:.3f})")
                        continue

                    if key in created:
                        self._log(f"    ⊘ Skipping duplicate: ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")
                        print(f"    ⊘ Skipping duplicate")
                        continue

                    log_msg = f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})"
                    if tgt.get('properties'):
                        log_msg += f"\n      Properties: {json.dumps(tgt['properties'], indent=8)}"
                    log_msg += f"\n      Metadata: page={chunk.get('source_page')} | section={chunk.get('section_title')}"
                    self._log(log_msg)
                    print(f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")

                    sic = None
                    if rel == 'OPERATES_IN':
                        industry_name = tgt['name']
                        sector = tgt.get('properties', {}).get('sector', '')
                        sic_code = self._lookup_sic(industry_name)
                        if sic_code:
                            sic = {
                                'code': sic_code,
                                'industry': industry_name,
                                'sector': sector,
                                'src_type': tgt['type'],
                                'src_name': tgt['name'],
                            }
                            self._log(f"      SIC: ({tgt['type']}: {industry_name}) --[HAS_SIC_CODE]--> (SICCode: {sic_code})")
                            print(f"      SIC: {industry_name} --[HAS_SIC_CODE]--> {sic_code}")

                    write_queue.append({
                        'src': src, 'tgt': tgt, 'rel': rel,
                        'props': entity.get('properties', {}),
                        'chunk_text': chunk['text'],
                        'similarity': chunk['similarity'],
                        'section_title': chunk.get('section_title'),
                        'source_page': chunk.get('source_page'),
                        'sic': sic,
                    })
                    created.add(key)

        # CEO scoring
        if ceo_candidates:
            self._log(f"\n  Processing {len(ceo_candidates)} CEO candidates...")
            print(f"\n  Processing {len(ceo_candidates)} CEO candidates...")

            def score_ceo_candidate(candidate):
                text = candidate['text'].lower()
                name = candidate['entity']['source']['name'].lower()
                score = 0
                if 'president and ceo' in text or 'president & ceo' in text:
                    score += 100
                if 'chief executive officer' in text:
                    score += 80
                if f"{name.split()[0].lower()}" in text and 'ceo' in text:
                    score += 50
                if any(phrase in text for phrase in ['was appointed', 'serves as', 'is the']):
                    score += 30
                if any(phrase in text for phrase in ['board member', 'director', 'chairman', 'cfo', 'formerly', 'previous']):
                    score -= 50
                if 'executive vice president' in text and 'ceo' not in text:
                    score -= 40
                name_parts = name.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    if last_name in text:
                        idx = text.find(last_name)
                        context = text[max(0, idx - 100):min(len(text), idx + 100)]
                        if 'ceo' not in context and 'chief executive' not in context:
                            score -= 30
                return score

            for candidate in ceo_candidates:
                candidate['semantic_score'] = score_ceo_candidate(candidate)
                log_msg = f"    ~ {candidate['entity']['source']['name']}: semantic_score={candidate['semantic_score']}, similarity={candidate['similarity']:.3f}"
                self._log(log_msg)
                print(log_msg)

            ceo_candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
            best = ceo_candidates[0]
            if best['semantic_score'] > 0:
                entity = best['entity']
                src, tgt, rel = entity['source'], entity['target'], entity['relationship']
                log_msg = f"  ✓ Selected CEO: {src['name']} (semantic_score: {best['semantic_score']}, similarity: {best['similarity']:.3f})"
                self._log(log_msg)
                print(log_msg)
                write_queue.append({
                    'src': src, 'tgt': tgt, 'rel': rel, 'props': {},
                    'chunk_text': best['text'], 'similarity': best['similarity'],
                    'section_title': best.get('section_title'),
                    'source_page': best.get('source_page'),
                    'sic': None,
                })
            else:
                log_msg = f"  ✗ No valid CEO candidate found (best score: {best['semantic_score']})"
                self._log(log_msg)
                print(log_msg)

        relation_elapsed = time.time() - relation_start
        summary = (
            f"\n{'='*80}\n"
            f"EXTRACTION COMPLETE: {relation_name}\n"
            f"{'='*80}\n"
            f"Entities     : {len(write_queue)}\n"
            f"Duration     : {relation_elapsed:.1f}s ({relation_elapsed/60:.1f} min)\n"
        )
        self._log(summary)
        print(summary)
        return write_queue

    def extract_multiple_relations(self, relation_names: List[str]) -> str:
        """Extract all relations and save results to JSON. Returns the JSON file path."""
        print(f"\n{'='*60}\nMULTI-RELATION EXTRACTION\n{'='*60}")
        output = {
            "main_company": self.main_company,
            "relations": {},
        }
        for rel in relation_names:
            items = self.extract_relation(rel)
            output["relations"][rel] = items

        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Extracted data saved to: {self.json_file}")
        return self.json_file

    def close(self):
        self._save_log()


def main():
    parser = argparse.ArgumentParser(description="LLM Extractor — retrieves chunks and extracts entities to JSON")
    parser.add_argument("relations", nargs="*", help="Relations to extract")
    parser.add_argument("--all", action="store_true", help="Extract all relations")
    parser.add_argument("--list", action="store_true", help="List available relations")
    parser.add_argument("--collection", default="financial", help="Qdrant collection name")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--main-company", default=None, help="Main company name")
    parser.add_argument("--source-file", default="", help="Source document filename (used to name output files)")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")

    args = parser.parse_args()

    if args.list:
        print("Available relations:", list_available_relations())
        return

    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR", default_output)

    relations = list_available_relations() if args.all else [r.upper() for r in args.relations]

    extractor = LLMExtractor(
        collection_name=args.collection,
        qdrant_host=args.qdrant_host or os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=args.qdrant_port or int(os.getenv("QDRANT_PORT", "6333")),
        embedding_model=args.embedding_model,
        output_dir=output_dir,
        main_company=args.main_company or os.getenv("MAIN_COMPANY", "the Company"),
        source_file=args.source_file or os.getenv("SOURCE_FILE", ""),
    )

    try:
        json_path = extractor.extract_multiple_relations(relations)
        print(f"\n✓ Done. JSON saved to: {json_path}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
