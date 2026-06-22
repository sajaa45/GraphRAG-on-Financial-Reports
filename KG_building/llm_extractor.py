#this is the orchestrator for the nodes extraction
import os
import sys
import json
import importlib.util
import time
import re
import boto3
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from relation_extraction_config import get_relation_config, set_main_company
from company_utils import CompanyDetector, SICLookup

_JSON_BLOCK_RE = re.compile(r'\[.*\]', re.DOTALL)
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

#call llm system 
def make_llm_fn(client, model: str):
    """Return a prompt→str callable backed by a boto3 bedrock-runtime client."""
    def llm_fn(prompt: str) -> str:
        response = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 4096, "temperature": 0.0},
        )
        text = response["output"]["message"]["content"][0]["text"]
        return _THINK_RE.sub('', text).strip()
    return llm_fn


#body for reading the sections json and getting the matched  sections + save entities in json 
class LLMExtractor:

    def __init__(self,
                 parsed_sections_file: str,
                 output_dir: str = "/app/output",
                 main_company: str = "the Company",
                 source_file: str = "",
                 bedrock_client=None):

        with open(parsed_sections_file, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        self._flat_sections = self._flatten_sections(parsed_data.get('sections', []))
        print(f"✓ Loaded {len(self._flat_sections)} sections from {parsed_sections_file}")

        self.bedrock_model = os.getenv("BEDROCK_MODEL", "")
        if bedrock_client is not None:
            self.bedrock = bedrock_client
       
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.company_detector = CompanyDetector(llm_fn=self._call_llm)
        self.sic_lookup = SICLookup()

        self.main_company = main_company
        if main_company == "the Company":
            self.main_company = self.company_detector.detect_from_sections(self._flat_sections)
        else:
            self.company_detector.main_company = main_company
        set_main_company(self.main_company)
        print(f"✓ Main company: {self.main_company}")

        self._base = os.path.splitext(os.path.basename(source_file))[0] if source_file else "output"
        self._relations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relations')
        self.log_file = os.path.join(self.output_dir, f"relationships_{self._base}.txt")
        self.log_buffer = []
        self.start_time = time.time()
        open(self.log_file, 'w', encoding='utf-8').close()

        print(f"✓ Logging to  : {self.log_file}")
    #remove the hierarchy between the sections
    def _flatten_sections(self, sections: list) -> list:
        flat = []
        for section in sections:
            raw_pages = section.get('page_contents') or []
            raw_chunks = section.get('chunk_contents') or []
            if raw_chunks and not raw_pages:
                raw_pages = [
                    {'page_number': c.get('chunk_index', i + 1), 'content': c.get('content', '')}
                    for i, c in enumerate(raw_chunks)
                ]
            if raw_pages:
                flat.append({
                    'title': section['title'],
                    'level': section.get('level', 1),
                    'start_page': section.get('start_page'),
                    'end_page': section.get('end_page'),
                    'page_contents': raw_pages,
                })
            if 'subsections' in section:
                flat.extend(self._flatten_sections(section['subsections']))
        return flat

    def _find_matching_sections_bm25(self, section_queries: List[str]) -> list:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 is not installed. Run: pip install rank-bm25")

        def _section_tokens(s: dict) -> list[str]:
            return s['title'].lower().split()

        tokenized_titles = [_section_tokens(s) for s in self._flat_sections]
        bm25 = BM25Okapi(tokenized_titles)

        seen_indices: set = set()
        matched: list = []  

        for query in section_queries:
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for idx in ranked:
                if scores[idx] <= 1.3:
                    break
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    matched.append((self._flat_sections[idx], float(scores[idx])))
                    print(f"    BM25 '{query}' → '{self._flat_sections[idx]['title']}' (score: {scores[idx]:.2f})")
                    break

        matched.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in matched]

        if not result:
            raise ValueError(f"BM25 found no matching sections for queries: {section_queries}")

        return result

    def _log(self, message: str):
        self.log_buffer.append(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def _save_log(self):
        elapsed = time.time() - self.start_time
        header = (
            f"Started : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n"
            f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Elapsed : {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
            f"{'='*80}\n\n"
        )
        with open(self.log_file, 'r', encoding='utf-8') as f:
            existing = f.read()
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header + existing)

    def _call_llm(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = self.bedrock.converse(
                    modelId=self.bedrock_model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
                )
                text = response["output"]["message"]["content"][0]["text"]
                text = _THINK_RE.sub('', text).strip()
                return text
            except Exception as e:
                err = str(e)
                is_throttle = "ThrottlingException" in err or "429" in err
                is_overload = "ServiceUnavailableException" in err or "503" in err
                if not (is_throttle or is_overload) or attempt == 4:
                    raise
                wait = 60 if is_overload else 30 * (attempt + 1)
                print(f"    ⚠ Bedrock {'overloaded' if is_overload else 'rate limit'}... retrying in {wait}s ({attempt+1}/4)")
                time.sleep(wait)

    def _extract_entities_from_page(self, page_text: str, page_number: int,
                                     section_title: str, relation_config) -> list:
        page_info = {
            'text': page_text,
            'source_page': page_number,
            'section_title': section_title,
        }

        prompt = relation_config.extraction_prompt_template.format(
            text=page_text,
            main_company=self.main_company
        )

        try:
            llm_output = self._call_llm(prompt)
            m = _JSON_BLOCK_RE.search(llm_output)
            if not m:
                print(f"[DEBUG] LLM returned no JSON block for page {page_number}. Output: {llm_output[:500]}")
                return []

            raw_json = m.group()
            try:
                entities_data = json.loads(raw_json)
            except json.JSONDecodeError as je:
                print(f"[DEBUG] JSON parse failed ({je}). Raw LLM output (first 1000 chars):\n{llm_output[:1000]}")
                return []

            results = []
            for e in entities_data:
                for _f in ('metric', 'value', 'unit', 'year', 'organization'):
                    raw = e.get(_f)
                    if not isinstance(raw, str):
                        e[_f] = '' if raw is None or isinstance(raw, (list, dict)) else str(raw)

                missing = [f for f in relation_config.required_fields if not str(e.get(f, '')).strip()]
                if missing:
                    print(f"[VALIDATION] Entity rejected — missing {missing}: {e}")
                    continue

                kwargs = {**relation_config.entity_parser_kwargs, 'main_company': self.main_company}
                p = relation_config.entity_parser(e, **kwargs)
                if p:
                    results.append((p, page_info))

            return results

        except Exception as ex:
            print(f"    ✗ Page {page_number} extraction error: {ex}")
            return []

    def _load_validator(self, rel_name: str):
        validator_path = os.path.join(self._relations_dir, rel_name, 'validate_entity.py')
        if not os.path.exists(validator_path):
            return None
        spec = importlib.util.spec_from_file_location(f"validator_{rel_name}", validator_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, f"validate_{rel_name.lower()}", None)

    def extract_relations(self, relation_names: List[str]) -> List[str]:
        print(f"\n{'='*60}\nMULTI-RELATION EXTRACTION\n{'='*60}")
        json_paths = []
        for relation_name in relation_names:
            relation_config = get_relation_config(relation_name)
            if not relation_config:
                print(f"✗ Unknown relation: {relation_name}")
                continue

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

            print(f"  Using BM25 section search ({len(relation_config.section_queries)} queries)")
            matched_sections = self._find_matching_sections_bm25(relation_config.section_queries)

            print(f"\n  Matched {len(matched_sections)} section(s):")
            for s in matched_sections:
                print(f"    - '{s['title']}' (pages {s.get('start_page')}–{s.get('end_page')})")
                self._log(f"  Section: '{s['title']}' (pages {s.get('start_page')}–{s.get('end_page')})")

            write_queue = []
            created = set()

            for section in matched_sections:
                section_title = section['title']
                pages = section['page_contents']

                self._log(f"\n{'='*70}")
                self._log(f"SECTION: {section_title}")
                self._log(f"{'='*70}")

                for page in pages:
                    page_number = page['page_number']
                    page_text = page['content']

                    print(f"\n  Page {page_number} | Section: {section_title}")
                    self._log(f"\n  Page {page_number} | Section: {section_title}")
                    self._log(f"  {'-'*68}")
                    self._log(page_text)
                    self._log(f"  {'-'*68}")

                    batch_results = self._extract_entities_from_page(
                        page_text, page_number, section_title, relation_config
                    )

                    if not batch_results:
                        self._log("    ✗ No entities extracted from this page")

                    for entity, page_info in batch_results:
                        src = entity['src']
                        tgt = entity['tgt']
                        rel = entity['rel']
                        key = (src['name'], rel, tgt['name'])

                        if key in created:
                            self._log(f"    ⊘ Skipping duplicate: ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")
                            print(f"    ⊘ Skipping duplicate")
                            continue

                        log_msg = f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})"
                        if tgt.get('properties'):
                            log_msg += f"\n      Properties: {json.dumps(tgt['properties'], indent=8)}"
                        log_msg += f"\n      Metadata: page={page_number} | section={section_title}"
                        self._log(log_msg)
                        print(f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")

                        sic = None
                        if rel == 'OPERATES_IN':
                            industry_name = tgt['name']
                            sector = tgt.get('properties', {}).get('sector', '')
                            sic_code = self.sic_lookup.lookup(industry_name)
                            if sic_code:
                                if 'properties' not in tgt:
                                    tgt['properties'] = {}
                                tgt['properties']['sic_code'] = sic_code
                                sic = {
                                    'code': sic_code,
                                    'industry': industry_name,
                                    'sector': sector,
                                    'src_type': tgt['type'],
                                    'src_name': tgt['name'],
                                }
                                self._log(f"      SIC: ({tgt['type']}: {industry_name}) --[HAS_SIC_CODE]--> (SICCode: {sic_code})")
                                print(f"      SIC: {industry_name} --[HAS_SIC_CODE]--> {sic_code}")

                        relation_entry = {
                            'src': src, 'tgt': tgt, 'rel': rel,
                            'props': entity.get('properties', {}),
                            'chunk_text': page_text,
                            'section_title': section_title,
                            'source_page': page_number,
                        }
                        if sic is not None:
                            relation_entry['sic'] = sic

                        write_queue.append(relation_entry)
                        created.add(key)

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

            rel_dir = os.path.join(self.output_dir, relation_name)
            os.makedirs(rel_dir, exist_ok=True)
            json_path = os.path.join(rel_dir, f"extracted_{self._base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"main_company": self.main_company, "relations": {relation_name: write_queue}}, f, indent=2, ensure_ascii=False)

            validator = self._load_validator(relation_name)
            if validator:
                print(f"\n{'='*60}\nVALIDATING: {relation_name}\n{'='*60}")
                validated = validator(json_path, llm_fn=make_llm_fn(self.bedrock, self.bedrock_model))
                if validated is not None:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(validated, f, indent=2, ensure_ascii=False)
                    print(f"✓ {relation_name} validated and saved to: {json_path}")
                else:
                    print(f"⚠ {relation_name} validation returned None, keeping raw extraction")
            else:
                print(f"\n✓ {relation_name} saved to: {json_path}")

            json_paths.append(json_path)
        return json_paths

    def close(self):
        self._save_log()