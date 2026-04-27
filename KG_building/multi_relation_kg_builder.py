
import os
import sys
import json
import argparse
import time
import re
import boto3
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Allow running from any working directory (e.g. project root or KG_building/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from the project root (one level above KG_building/)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, QueryRequest

from relation_extraction_config import (
    RelationConfig,
    get_relation_config,
    list_available_relations,
    set_main_company
)
from industry_node_to_sic import get_sic_code

_JSON_BLOCK_RE = re.compile(r'\[.*\]', re.DOTALL)

# D: Module-level constants for _significant_tokens (avoids per-call recreation)
_STOP_WORDS = frozenset({
    'the', 'and', 'for', 'of', 'in', 'a', 'an',
    'company', 'corporation', 'incorporated', 'limited', 'group', 'holdings',
})
_TOKEN_CLEAN_RE = re.compile(r'[^a-z0-9 ]')

# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_N_SECTIONS = 2
TOP_N_CHUNKS_PER_SECTION = 3
SECTION_SIMILARITY_THRESHOLD = 0.25
CHUNK_SIMILARITY_THRESHOLD = 0.3
MAX_CHUNKS_PER_LLM_BATCH = 8  # Keep well within 8192-token context window  
# ============================================================================


class MultiRelationKGBuilder:
    """Build Neo4j knowledge graph with multiple relation types"""
    
    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 collection_name: str = "financial_docs",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 output_dir: str = "/app/output",
                 main_company: str = "the Company",
                 source_file: str = ""):

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"✓ Connected to Neo4j at {neo4j_uri}")

        self.embedding_model = SentenceTransformer(embedding_model)
        self._embed_cache: Dict[str, list] = {}  # C: reuse vectors for repeated strings
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

        # Stamp the node so it can always be found with a simple query
        self._stamp_target_company()

        self._sic_cache: Dict[str, str] = {}

        # Derive log filename from source_file or fall back to collection name
        base = os.path.splitext(os.path.basename(source_file))[0] if source_file else collection_name
        self.log_file = os.path.join(self.output_dir, f"relationships_{base}.txt")
        self.log_buffer = []
        self.start_time = time.time()

        print(f"✓ Logging to: {self.log_file}")
    
    def _stamp_target_company(self):
        """Set is_target=true on the main company node so it can always be found trivially.
        Also removes stale placeholder and case-variant duplicate Company nodes from prior runs."""
        if self.main_company.lower() in ('the company', 'this company', ''):
            print("⚠ Skipping stamp — company name not yet resolved")
            return
        try:
            with self.driver.session() as session:
                # Remove the unresolved placeholder left by runs where detection failed
                session.run(
                    "MATCH (c:Company) WHERE toLower(c.name) IN ['the company', 'this company'] DETACH DELETE c"
                )
                # Remove case-variant duplicates (e.g. "Newmarket" vs "NewMarket") by comparing
                # lowercased significant tokens — safe to DETACH DELETE because the canonical name
                # will be stamped next and relationships will be re-created on the next extraction run
                canonical_tokens = set(self._significant_tokens(self.main_company))
                if canonical_tokens:
                    result = session.run(
                        "MATCH (c:Company) WHERE c.name <> $name RETURN c.name AS name",
                        {"name": self.main_company}
                    )
                    for record in result:
                        variant = record["name"]
                        if canonical_tokens == set(self._significant_tokens(variant)):
                            session.run(
                                "MATCH (c:Company {name: $variant}) DETACH DELETE c",
                                {"variant": variant}
                            )
                            print(f"  ✓ Removed case-variant duplicate: '{variant}'")
                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    SET c.is_target = true, c.target_since = datetime()
                    """,
                    {"name": self.main_company},
                )
            print(f"✓ Stamped is_target=true on: {self.main_company}")
        except Exception as e:
            print(f"⚠ Could not stamp target company: {e}")

    def close(self):
        self.driver.close()
        self._save_log()

    def _embed(self, text: str) -> list:
        """C: Cache-backed encode — avoids re-encoding the same string across calls."""
        if text not in self._embed_cache:
            self._embed_cache[text] = self.embedding_model.encode([text])[0].tolist()
        return self._embed_cache[text]
    
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
    
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

    def _lookup_sic(self, sector: str) -> str:
        """Return SIC code for a sector string, cached to avoid repeat API calls."""
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

    def detect_main_company(self) -> str:
        """
        Use Gemini to identify the main company from document chunks.
        Falls back to regex if Gemini is unavailable.
        """
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

        # Regex fallback
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
        """Lowercase tokens >3 chars, excluding common stop words."""
        return [t for t in _TOKEN_CLEAN_RE.sub('', name.lower()).split()
                if len(t) > 3 and t not in _STOP_WORDS]

    def normalize_company_name(self, name: str) -> str:
        """Map any alias of the main company to the canonical name."""
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
    # HIERARCHICAL RETRIEVAL
    # ========================================================================
    def hierarchical_retrieval(self, relation_config, n_sections=TOP_N_SECTIONS,
                               n_chunks_per_section=TOP_N_CHUNKS_PER_SECTION,
                               section_threshold=SECTION_SIMILARITY_THRESHOLD):
        print(f"\n{'='*60}")
        print(f"Hierarchical Retrieval: {relation_config.name}")
        print(f"{'='*60}")

        # Use per-relation overrides when set
        n_sections = relation_config.n_sections
        n_chunks_per_section = relation_config.n_chunks_per_section

        # Build list of chunk embeddings: one per keyword string, or one from chunk_keywords
        kw_list = relation_config.chunk_keywords_list or [relation_config.chunk_keywords]
        chunk_embeddings = [self._embed(kw) for kw in kw_list]  # C: cached
        print(f"  Using {len(chunk_embeddings)} chunk query vector(s)")

        # Step 1: find relevant sections
        section_embedding = self._embed(relation_config.section_keywords)  # C: cached
        section_results = self.client.query_points(
            collection_name=self.collection_name,
            query=section_embedding,
            query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="section"))]),
            limit=n_sections,
            with_payload=True
        ).points

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
                "text": r.payload["text"]
            }
            if "source_page" in r.payload:
                chunk_data["source_page"] = r.payload["source_page"]
            all_chunks.append(chunk_data)

        if use_sections:
            # A: batch all (section × embedding) queries in one network round-trip
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
            # A: batch all embedding queries in one network round-trip
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
    # LLM EXTRACTION
    # ========================================================================
    def _call_llm(self, prompt: str) -> str:
        """Call AWS Bedrock via the Converse API (uses shared client initialised in __init__)."""
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

    def _extract_entities_batch(self, chunks: List[Dict], relation_config) -> List[tuple]:
        """
        Single Gemini call for all chunks combined.
        Returns list of (entity, chunk) tuples so validation still has the source text.
        """
        # Build one prompt with all chunks separated
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
                return []
            entities_data = json.loads(m.group())
            results = []
            for e in entities_data:
                # Validate against full batch text (numbers won't word-match a single chunk)
                if not self._validate_entity_in_text(e, chunks_text, relation_config.name):
                    continue
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    # Normalize company names to prevent duplicate nodes
                    for node_key in ('source', 'target'):
                        node = p[node_key]
                        if node['type'] in ('Company', 'Organization'):
                            node['name'] = self.normalize_company_name(node['name'])
                    # Associate with the chunk whose text best overlaps (for logging/relationships)
                    best_chunk = max(chunks, key=lambda c: sum(
                        1 for w in str(e).lower().split() if w in c['text'].lower()
                    ))
                    results.append((p, best_chunk))
            return results
        except Exception as e:
            print(f"    ✗ Batch extraction error: {e}")
            return []

    def extract_entities_with_llm(self, text: str, relation_config):
        """Single-chunk extraction."""
        prompt = relation_config.extraction_prompt_template.format(
            text=text,
            main_company=self.main_company
        )
        try:
            llm_output = self._call_llm(prompt)
            m = _JSON_BLOCK_RE.search(llm_output)
            if not m:
                return []
            entities_data = json.loads(m.group())
            parsed = []
            for e in entities_data:
                if not self._validate_entity_in_text(e, text, relation_config.name):
                    continue
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    # Normalize company names to prevent duplicate nodes
                    for node_key in ('source', 'target'):
                        node = p[node_key]
                        if node['type'] in ('Company', 'Organization'):
                            node['name'] = self.normalize_company_name(node['name'])
                    parsed.append(p)
            return parsed
        except Exception as e:
            print(f"    ✗ Extraction error: {e}")
            return []
    
    def _validate_entity_in_text(self, entity: Dict, text: str, relation_type: str) -> bool:
        """Validate that extracted entity is actually grounded in the source text"""
        text_lower = text.lower()
        
        if relation_type == 'CEO':
            person = str(entity.get('person', '')).strip()
            if not person or len(person) <= 2:
                return False
            # Check if person name appears in text
            name_parts = person.split()
            if len(name_parts) < 2:
                return False
            # Must have at least last name in text
            if name_parts[-1].lower() not in text_lower:
                return False
            # Verify CEO context
            role = str(entity.get('role', '')).lower()
            if 'ceo' in role or 'chief executive' in role:
                # Must have CEO-related keywords near the name
                if not any(kw in text_lower for kw in ['ceo', 'chief executive', 'president']):
                    return False
            return True
        
        elif relation_type == 'HAS_METRIC':
            metric = str(entity.get('metric', '')).strip()
            value = str(entity.get('value', '')).strip()
            if not metric or not value:
                return False

            # Clean value for matching (remove commas, decimals)
            value_clean = value.replace(',', '').replace('.', '')

            # Check if the numeric value appears in text
            # Handle both regular numbers and currency symbols (₹, #, $)
            # (re is imported at the top of the module)

            # More flexible patterns - look for the core digits
            # Extract just the significant digits (first 4-5 digits)
            core_digits = re.sub(r'[^\d]', '', value)[:5]  # First 5 digits
            
            if len(core_digits) >= 3:
                # Look for these digits anywhere in the text
                if core_digits in re.sub(r'[^\d]', '', text):
                    return True
            
            # Fallback: look for the full value with various formats
            patterns = [
                rf'\b{re.escape(value)}\b',  # Exact match
                rf'[₹#$€£¥]\s*{re.escape(value)}',  # With currency prefix
                rf'{re.escape(value)}\s*[₹#$€£¥]',  # With currency suffix
                rf'\({re.escape(value)}\)',  # In parentheses
            ]
            
            found_number = any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
            if not found_number:
                print(f"      ⚠ Metric validation failed: {metric}={value} not found in text")
                return False
            
            return True
        
        elif relation_type == 'FACES_RISK':
            risk_name = str(entity.get('risk_name', '')).strip()
            if not risk_name:
                return False

            # Validate that at least one word from the risk name or description appears in text
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
            # Industry keywords must appear
            industry_keywords = industry.lower().split()
            if not any(kw in text_lower for kw in industry_keywords if len(kw) > 2):
                return False
            return True
        
        return True
    
    # ========================================================================
    # NEO4J OPERATIONS (unchanged)
    # ========================================================================
    def create_node(self, session, node_type: str, name: str, properties: Dict = None):
        props = properties or {}
        props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        ON CREATE SET n.created_at = datetime() {', ' + props_str if props_str else ''}
        ON MATCH SET {props_str if props_str else 'n.updated_at = datetime()'}
        RETURN n
        """
        session.run(query, {"name": name, **props})
    
    def create_relationship(self, session, source_type, source_name, target_type,
                            target_name, rel_type, properties=None, source_chunk=None,
                            similarity=None, section_title=None, source_page=None):
        props = dict(properties or {})
        if source_chunk:
            props['source_chunk'] = source_chunk[:200]
        if similarity is not None:
            props['confidence'] = round(float(similarity), 4)
        if section_title:
            props['section_title'] = section_title
        if source_page is not None:
            props['source_page'] = source_page

        # Build SET clause dynamically so all props are actually written to the relationship.
        # Without this, parameters passed to session.run() are silently unused.
        props_set = ", ".join(f"r.{k} = ${k}" for k in props)
        on_create = f"r.created_at = datetime(){', ' + props_set if props_set else ''}"
        on_match  = f"r.updated_at = datetime(){', ' + props_set if props_set else ''}"

        query = f"""
        MATCH (s:{source_type} {{name: $source_name}})
        MATCH (t:{target_type} {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET {on_create}
        ON MATCH SET {on_match}
        RETURN r
        """
        session.run(query, {"source_name": source_name, "target_name": target_name, **props})
    
    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================
    def extract_relation(self, relation_name: str):
        relation_config = get_relation_config(relation_name)
        if not relation_config:
            print(f"✗ Unknown relation: {relation_name}")
            return {"error": "Unknown relation"}
        
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
        
        self._log("="*80)
        self._log("EXTRACTING ENTITIES WITH LLM")
        self._log("="*80)

        # ── Phase 1: LLM extraction (no Neo4j session held during slow LLM calls) ──
        write_queue = []   # items: {src, tgt, rel, props, chunk_text, similarity, sic}
        ceo_candidates = []
        created = set()

        if relation_config.chunk_keywords_list:
            # Per-keyword batch path: one LLM call per keyword with its top-N chunks.
            # Used by HAS_METRIC — avoids one LLM call per chunk (up to 81 calls → 9 calls).
            n_per_kw = relation_config.n_chunks_per_section
            total_kw = len(relation_config.chunk_keywords_list)
            print(f"\n  Per-keyword batch mode: {total_kw} keywords × {n_per_kw} chunks each")
            self._log(f"\n  Per-keyword batch mode: {total_kw} keywords × {n_per_kw} chunks each")

            for kw_idx, kw in enumerate(relation_config.chunk_keywords_list, 1):
                kw_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=self._embed(kw),
                    query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))]),
                    limit=n_per_kw,
                    with_payload=True,
                ).points

                kw_chunks = [
                    {
                        "section_title": r.payload.get("section_title", "Unknown"),
                        "section_id": r.payload.get("section_id"),
                        "chunk_index": r.payload.get("chunk_index"),
                        "source_page": r.payload.get("source_page"),
                        "similarity": r.score,
                        "text": r.payload["text"],
                    }
                    for r in kw_results
                    if r.score >= CHUNK_SIMILARITY_THRESHOLD
                ]

                print(f"\n  Keyword {kw_idx}/{total_kw}: '{kw[:60]}' → {len(kw_chunks)} chunks")
                self._log(f"\nKeyword {kw_idx}/{total_kw}: {kw}")

                if not kw_chunks:
                    self._log("    ✗ No chunks above threshold")
                    continue

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
            # Hierarchical retrieval + per-chunk extraction for all other relations.
            chunks = self.hierarchical_retrieval(relation_config)
            filtered_chunks = [c for c in chunks if c['similarity'] >= CHUNK_SIMILARITY_THRESHOLD]
            print(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {CHUNK_SIMILARITY_THRESHOLD})")
            self._log(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {CHUNK_SIMILARITY_THRESHOLD})")

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

                    # SIC lookup happens here (cached, no DB) so it doesn't hold a transaction open
                    sic = None
                    if rel == 'OPERATES_IN':
                        industry_name = tgt['name']
                        sector = tgt.get('properties', {}).get('sector', '')
                        sic_code = self._lookup_sic(industry_name)
                        if sic_code:
                            sic = {'code': sic_code, 'industry': industry_name, 'sector': sector,
                                   'src_type': tgt['type'], 'src_name': tgt['name']}
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

        # CEO scoring (still Phase 1 — no DB)
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

        # ── Phase 2: single Neo4j transaction for all writes (B) ──────────────
        def _write_batch(tx):
            for item in write_queue:
                # Build a single metadata dict from chunk provenance for all nodes in this item
                raw_meta = {}
                if item.get('source_page') is not None:
                    raw_meta['source_page'] = item['source_page']
                if item.get('section_title'):
                    raw_meta['section_title'] = item['section_title']
                if item.get('similarity') is not None:
                    raw_meta['confidence'] = round(float(item['similarity']), 4)
                if item.get('chunk_text'):
                    raw_meta['source_text'] = item['chunk_text']
                metadata_str = json.dumps(raw_meta) if raw_meta else None

                # Source node (Company / Person / Organization) — add metadata property
                src_props = dict(item['src'].get('properties', {}))
                if metadata_str:
                    src_props['metadata'] = metadata_str
                self.create_node(tx, item['src']['type'], item['src']['name'], src_props)

                # Target node (Metric / Risk / Industry / …) — merge own props + metadata
                tgt_props = dict(item['tgt'].get('properties', {}))
                if metadata_str:
                    tgt_props['metadata'] = metadata_str
                self.create_node(tx, item['tgt']['type'], item['tgt']['name'], tgt_props)

                self.create_relationship(tx, item['src']['type'], item['src']['name'],
                                         item['tgt']['type'], item['tgt']['name'],
                                         item['rel'], item['props'],
                                         item['chunk_text'], item['similarity'],
                                         item.get('section_title'), item.get('source_page'))
                if item['sic']:
                    s = item['sic']
                    self.create_node(tx, 'SICCode', s['code'],
                                     {'code': s['code'], 'industry': s['industry'], 'sector': s['sector']})
                    self.create_relationship(tx, s['src_type'], s['src_name'], 'SICCode', s['code'], 'HAS_SIC_CODE')

        if write_queue:
            with self.driver.session() as session:
                session.execute_write(_write_batch)

        total_entities = len(write_queue)
        total_relationships = len(write_queue)
        
        relation_elapsed = time.time() - relation_start
        summary = (
            f"\n{'='*80}\n"
            f"EXTRACTION COMPLETE: {relation_name}\n"
            f"{'='*80}\n"
            f"Entities     : {total_entities}\n"
            f"Relationships: {total_relationships}\n"
            f"Duration     : {relation_elapsed:.1f}s ({relation_elapsed/60:.1f} min)\n"
        )
        self._log(summary)
        print(summary)
        return {"relation": relation_name, "entities": total_entities, "relationships": total_relationships}
    def extract_multiple_relations(self, relation_names: List[str]):
        print(f"\n{'='*60}\nMULTI-RELATION EXTRACTION\n{'='*60}")
        results = {}
        for rel in relation_names:
            results[rel] = self.extract_relation(rel)
        return results

    def show_graph_stats(self):
        """Display graph statistics"""
        with self.driver.session() as session:
            print(f"\n{'='*60}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'='*60}")
            
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("\nNodes:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\nRelationships:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Relation Knowledge Graph Builder")
    parser.add_argument("relations", nargs="*", help="Relations to extract")
    parser.add_argument("--all", action="store_true", help="Extract all relations")
    parser.add_argument("--list", action="store_true", help="List available relations")
    parser.add_argument("--clear", action="store_true", help="Clear database first")
    parser.add_argument("--collection", default="financial_docs", help="Qdrant collection name")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--main-company", default=None, help="Main company name for OPERATES_IN extraction")
    parser.add_argument("--source-file", default="", help="Source document filename used to name the relationships log")
    parser.add_argument("--output-dir", default=None, help="Directory for relationship logs (default: output/ next to this script)")

    args = parser.parse_args()

    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR", default_output)
    
    if args.list:
        print("Available relations:", list_available_relations())
        return
    
    relations = list_available_relations() if args.all else [r.upper() for r in args.relations]
    
    builder = MultiRelationKGBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "Lexical12345"),
        collection_name=args.collection,
        qdrant_host=args.qdrant_host or os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=args.qdrant_port or int(os.getenv("QDRANT_PORT", "6333")),
        embedding_model=args.embedding_model,
        output_dir=output_dir,
        main_company=args.main_company or os.getenv("MAIN_COMPANY", "the Company"),
        source_file=args.source_file or os.getenv("SOURCE_FILE", "")
    )
    
    try:
        if args.clear:
            builder.clear_database()
        builder.extract_multiple_relations(relations)
        builder.show_graph_stats()
    finally:
        builder.close()


if __name__ == "__main__":
    main()