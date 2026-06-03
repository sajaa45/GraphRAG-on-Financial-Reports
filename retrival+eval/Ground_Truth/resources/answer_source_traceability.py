"""
Ground Truth Evaluation — Answer-Source Traceability (Phase 1)

For each atomic claim in the narrative answer:
  1. Identify which source(s) it should come from (risk chunk / metric node)
  2. Check whether the pipeline used the correct source to generate that claim
  3. Store as {claim_id, claim_text, correct_source_id,
               pipeline_used_source_id, is_correct_source}

This answers: "Did the pipeline pick the right data to answer this question?"

Input files (from retrival_results/):
  - extraction_result.json  — retrieved graph data (risks + metrics)
  - answer.txt              — the narrative answer produced by the pipeline

Output:
  - answer_source_traceability.csv
"""

import csv
import json
import os
import re
import sys
from typing import Any

import boto3
from dotenv import load_dotenv

# Fix Windows console encoding — guard against double-wrapping when loaded
# multiple times from the API (codecs StreamWriter has no .buffer attribute).
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

# ---------------------------------------------------------------------------
# Source registry builder
# ---------------------------------------------------------------------------

def _make_metric_id(metric: dict, cik: str) -> str:
    tag = metric.get('xbrl_tag') or metric.get('metric_type') or metric.get('name', 'unknown')
    year = metric.get('year', '')
    return f"{cik}__{tag}__{year}"


def build_source_registry(extraction_data: dict) -> dict[str, dict]:
    """
    Build a flat registry of every retrieved source item.

    Keys are citation_ids directly (e.g., 'TARGET_M0024', 'PEER_R0042').
    
    Each entry keeps:
      citation_id, type, company, role, summary, full (for LLM context)
    """
    registry: dict[str, dict] = {}

    for row in extraction_data.get('results', []):
        target_name = row.get('target', 'Target')
        peer_name   = row.get('peer', '')

        # ── target risks ────────────────────────────────────────────────
        for r in (row.get('target_risks') or []):
            citation_id = r.get('citation_id', '')
            if not citation_id:
                continue  # Skip if no citation_id
            
            registry[citation_id] = {
                'citation_id': citation_id,
                'type':        'risk',
                'company':     target_name,
                'role':        'target',
                'summary':     f"{r.get('name', '')} — {str(r.get('description', ''))[:120]}",
                'full':        {
                    'name':        r.get('name', ''),
                    'description': r.get('description', ''),
                    'why':         r.get('why', ''),
                    'filing_date': r.get('filing_date', ''),
                    'document_url':r.get('document_url', ''),
                },
            }

        # ── peer risks ───────────────────────────────────────────────────
        if peer_name:
            for r in (row.get('peer_risks') or []):
                citation_id = r.get('citation_id', '')
                if not citation_id:
                    continue
                
                registry[citation_id] = {
                    'citation_id': citation_id,
                    'type':        'risk',
                    'company':     peer_name,
                    'role':        'peer',
                    'summary':     f"{r.get('name', '')} — {str(r.get('description', ''))[:120]}",
                    'full':        {
                        'name':        r.get('name', ''),
                        'description': r.get('description', ''),
                        'why':         r.get('why', ''),
                        'filing_date': r.get('filing_date', ''),
                        'document_url':r.get('document_url', ''),
                    },
                }

        # ── target metrics ───────────────────────────────────────────────
        for m in (row.get('target_metrics') or []):
            citation_id = m.get('citation_id', '')
            if not citation_id:
                continue
            
            year = m.get('year', '')
            registry[citation_id] = {
                'citation_id': citation_id,
                'type':        'metric',
                'company':     target_name,
                'role':        'target',
                'summary':     (
                    f"{m.get('label') or m.get('name', '')} = {m.get('value', '')} "
                    f"{m.get('unit', '')} ({year})"
                ),
                'full': {
                    'label':       m.get('label', ''),
                    'name':        m.get('name', ''),
                    'value':       m.get('value', ''),
                    'unit':        m.get('unit', ''),
                    'year':        year,
                    'metric_type': m.get('metric_type', ''),
                },
            }

        # ── peer metrics ─────────────────────────────────────────────────
        if peer_name:
            for m in (row.get('peer_metrics') or []):
                citation_id = m.get('citation_id', '')
                if not citation_id:
                    continue
                
                registry[citation_id] = {
                    'citation_id': citation_id,
                    'type':        'metric',
                    'company':     peer_name,
                    'role':        'peer',
                    'summary':     (
                        f"{m.get('label') or m.get('name', '')} = {m.get('value', '')} "
                        f"{m.get('unit', '')} ({m.get('year', '')})"
                    ),
                    'full': {
                        'label':       m.get('label', ''),
                        'name':        m.get('name', ''),
                        'value':       m.get('value', ''),
                        'unit':        m.get('unit', ''),
                        'year':        m.get('year', ''),
                        'metric_type': m.get('metric_type', ''),
                    },
                }

    return registry


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_call(bedrock_client, model_id: str, prompt: str, max_tokens: int = 2048) -> str:
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return response['output']['message']['content'][0]['text'].strip()


# ---------------------------------------------------------------------------
# Citation parsing — inline [CITE:...] tags
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r'\[CITE:([^\]]+)\]', re.IGNORECASE)


def _resolve_citation(raw_citation: str, registry: dict[str, dict]) -> str:
    """
    Direct lookup of citation_id in registry.
    Returns the citation_id if found, or '?<raw>' if not found.
    """
    raw = raw_citation.strip()
    
    # Direct lookup by citation_id (which is now the registry key)
    if raw in registry:
        return raw
    
    return f'?{raw}'


def _strip_citations(text: str) -> str:
    """Remove all inline citation tags from a claim string."""
    return _CITATION_RE.sub('', text).strip()


# ---------------------------------------------------------------------------
# Step 1 — Extract atomic claims (with their inline citations) from the answer
# ---------------------------------------------------------------------------

_ABBREV_RE = re.compile(
    r'\b(Inc|Corp|Ltd|Co|vs|Dr|Mr|Mrs|Ms|MTI|NL|VALHI|No|Vol|Fig|Dept|Div)\.',
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r'\[[^\]]*\]')   # matches [CITE:...]


def _safe_sentence_split(text: str) -> list[str]:
    """
    Split on sentence-ending punctuation while:
      1. Protecting content inside [...] brackets (citations may contain "Inc." etc.)
      2. Not splitting on known abbreviations (Inc., Corp., MTI., etc.)
    """
    # Step 1: Replace brackets with placeholders so their content is invisible to splitter
    placeholders: dict[str, str] = {}
    def _protect(m: re.Match) -> str:
        key = f"\x00CITE{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key
    protected = _BRACKET_RE.sub(_protect, text)

    # Step 2: Temporarily hide abbreviation periods (replace "Inc." → "Inc\x01")
    protected = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + '\x01', protected)

    # Step 3: Split on real sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\$\-])', protected)

    # Step 4: Restore placeholders and abbreviation periods
    sentences = []
    for part in parts:
        part = part.replace('\x01', '.')
        for key, original in placeholders.items():
            part = part.replace(key, original)
        sentences.append(part)
    return sentences


def _split_by_citations(sentence: str) -> list[tuple[str, str, list[str]]]:
    """
    Split a sentence into (fragment, citation, co_citations) triples.

    co_citations lists the OTHER citation IDs that share the same fragment
    (back-to-back citations annotating the same fact).

    Example:
      "A reported $108M [CITE:M1] and margin 21% [CITE:M2]"
      → [("A reported $108M", "M1", []), ("and margin 21%", "M2", [])]

      "...insolvency [CITE:R1][CITE:R2]"
      → [("...insolvency", "R1", ["R2"]), ("...insolvency", "R2", ["R1"])]
    """
    parts = _CITATION_RE.split(sentence)
    if len(parts) == 1:
        return [(sentence, 'NONE', [])]

    raw_pairs: list[tuple[str, str]] = []
    last_fragment = ''
    for i in range(0, len(parts) - 1, 2):
        fragment = parts[i].strip()
        cite_id  = parts[i + 1].strip()
        if fragment:
            last_fragment = fragment
        anchor = last_fragment or _strip_citations(sentence)
        raw_pairs.append((anchor, cite_id))

    result = []
    for idx, (anchor, cite_id) in enumerate(raw_pairs):
        co = [c for j, (a, c) in enumerate(raw_pairs) if a == anchor and j != idx]
        result.append((anchor, cite_id, co))

    return result if result else [(sentence, 'NONE', [])]


def extract_claims(answer_text: str, registry: dict[str, dict]) -> list[dict]:
    """
    Split the answer into sentences, then further split each sentence on its
    inline [CITE:...] tags so each citation is evaluated against only the
    specific text fragment it annotates.

    Returns [{claim_id, claim_text, resolved_citation}, ...]
    """
    sentences = _safe_sentence_split(answer_text)

    claims = []
    claim_num = 0
    skip_prefixes = ('**sources', 'sources:', '- [', '#')

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(sentence.lower().startswith(p) for p in skip_prefixes):
            continue

        citation_matches = _CITATION_RE.findall(sentence)

        # Skip sentences with no citations — they are summary/conclusion statements
        if not citation_matches:
            continue

        if len(citation_matches) > 1:
            # Multiple citations: split sentence into per-citation fragments
            fragments = _split_by_citations(sentence)
            for fragment, raw_cite, co_cites in fragments:
                bare = _strip_citations(fragment).strip()
                if len(bare.split()) < 4:
                    continue
                citation_id = _resolve_citation(raw_cite, registry) if raw_cite != 'NONE' else 'NONE'
                claim_num += 1
                claims.append({
                    'claim_id':          f"C{claim_num:03d}",
                    'claim_text':        bare,
                    'claim_text_raw':    fragment,
                    'raw_citation':      raw_cite,
                    'resolved_citation': citation_id,
                    'co_citations':      co_cites,
                    'sentence_context':  _strip_citations(sentence),
                })
        else:
            bare = _strip_citations(sentence).strip()
            if len(bare.split()) < 6:
                continue
            raw_citation = citation_matches[0]
            citation_id  = _resolve_citation(raw_citation, registry)
            claim_num += 1
            claims.append({
                'claim_id':          f"C{claim_num:03d}",
                'claim_text':        bare,
                'claim_text_raw':    sentence,
                'raw_citation':      raw_citation,
                'resolved_citation': citation_id,
                'sentence_context':  bare,
            })

    return claims


# ---------------------------------------------------------------------------
# Step 2 — Judge whether the cited source is correct (one LLM call per claim)
# ---------------------------------------------------------------------------

_MAX_SOURCE_TEXT_CHARS = 600   # per source in the LLM prompt


def _source_content(src: dict) -> str:
    """Return a human-readable content block for one registry entry."""
    full = src['full']
    if src['type'] == 'risk':
        name  = full.get('name', '')
        desc  = (full.get('description') or '')[:_MAX_SOURCE_TEXT_CHARS]
        why   = (full.get('why') or '')[:_MAX_SOURCE_TEXT_CHARS]
        url   = full.get('document_url', '')
        lines = [f"  Risk: {name}", f"  Company: {src['company']} ({src['role']})"]
        if desc:
            lines.append(f"  Description: {desc}")
        if why:
            lines.append(f"  Why relevant: {why}")
        if url:
            lines.append(f"  Source URL: {url}")
    else:
        label = full.get('label') or full.get('name', '')
        value = full.get('value', '')
        unit  = full.get('unit', '')
        year  = full.get('year', '')
        lines = [
            f"  Metric: {label}",
            f"  Company: {src['company']} ({src['role']})",
            f"  Value: {value} {unit} ({year})",
        ]
    return "\n".join(lines)


def _resolve_to_content(citation_id: str, registry: dict[str, dict]) -> str:
    """Convert a citation_id to a human-readable source description for CSV output."""
    if citation_id in ('NONE', 'FABRICATED', 'ERROR', 'UNKNOWN') or citation_id.startswith('?'):
        return citation_id
    src = registry.get(citation_id)
    if not src:
        return citation_id
    full = src['full']
    if src['type'] == 'risk':
        name = full.get('name', '')
        desc = (full.get('description') or '')[:200]
        return f"[{citation_id}] {src['company']} — {name}: {desc}"
    else:
        label = full.get('label') or full.get('name', '')
        value = full.get('value', '')
        unit  = full.get('unit', '')
        year  = full.get('year', '')
        return f"[{citation_id}] {src['company']} — {label} = {value} {unit} ({year})"


def _build_source_listing(registry: dict[str, dict]) -> str:
    """Full source content listing for LLM prompts — includes description and source text."""
    blocks = []
    for sid, src in registry.items():
        blocks.append(f"[{sid}]\n{_source_content(src)}")
    return "\n\n".join(blocks)


def _parse_batch_response(
    raw: str,
    claims: list[dict],
    registry: dict[str, dict],
    valid_ids: set[str],
) -> dict[str, dict]:
    """
    Parse a batched LLM response into a dict keyed by claim_id.
    Expected format per claim block:
        CLAIM_ID: Cxxx
        USED_SOURCE: <id or NONE>
        CORRECT_SOURCE: YES | NO:<better_id> | FABRICATED
        EXPLANATION: ...
        ---
    """
    parsed: dict[str, dict] = {}
    current: dict = {}

    for line in raw.splitlines():
        line  = line.strip()
        upper = line.upper()

        if upper.startswith('CLAIM_ID:'):
            if current.get('claim_id'):
                parsed[current['claim_id']] = current
            current = {'claim_id': line.split(':', 1)[1].strip()}

        elif upper.startswith('USED_SOURCE:') and current:
            val = line.split(':', 1)[1].strip().upper()
            current['used_sid'] = val if val in valid_ids else ('NONE' if val == 'NONE' else f'?{val}')

        elif upper.startswith('CORRECT_SOURCE:') and current:
            val = line.split(':', 1)[1].strip()
            if val.upper() == 'YES':
                current['is_correct']  = True
                current['correct_sid'] = current.get('used_sid', 'UNKNOWN')
            elif val.upper() == 'FABRICATED':
                current['is_correct']  = False
                current['correct_sid'] = 'FABRICATED'
            elif val.upper().startswith('NO:'):
                better = val[3:].strip().upper()
                current['is_correct']  = False
                current['correct_sid'] = better if better in valid_ids else f'?{better}'
            else:
                candidate = val.strip().upper()
                current['is_correct']  = False
                current['correct_sid'] = candidate if candidate in valid_ids else f'?{candidate}'

        elif upper.startswith('EXPLANATION:') and current:
            current['explanation'] = line.split(':', 1)[1].strip()

        elif line == '---' and current.get('claim_id'):
            parsed[current['claim_id']] = current
            current = {}

    if current.get('claim_id'):
        parsed[current['claim_id']] = current

    return parsed


def attribute_claims(
    claims: list[dict],
    registry: dict[str, dict],
    question: str,
    bedrock_client,
    model_id: str,
) -> list[dict]:
    """
    Judge whether the pipeline used the correct source for every claim in one
    batched LLM call (all claims + all sources in a single prompt).

    Falls back to an error record per claim if the LLM call fails.
    """
    if not registry:
        for c in claims:
            c.update({
                'pipeline_used_source': 'NONE',
                'correct_source':       'NONE',
                'is_correct_source':    False,
                'explanation':          'No sources retrieved by pipeline',
            })
        return claims

    source_listing = _build_source_listing(registry)
    valid_ids = set(registry.keys())

    # ── Build per-claim sections ──────────────────────────────────────────
    claim_sections: list[str] = []
    for claim in claims:
        cid      = claim['claim_id']
        ctext    = claim['claim_text']
        cited_id = claim['resolved_citation']
        is_resolved   = cited_id != 'NONE' and not cited_id.startswith('?')
        is_unresolved = cited_id.startswith('?')

        if is_resolved:
            cited_block  = _source_content(registry[cited_id])
            sentence_ctx = claim.get('sentence_context', ctext)
            co_cites     = claim.get('co_citations', [])
            co_note = ''
            if co_cites:
                co_ids = ', '.join(co_cites)
                co_note = (
                    f"  CO-CITATIONS: {co_ids} (co-cited for the same fact; "
                    f"answer YES if [{cited_id}] partially supports it)\n"
                )
            claim_sections.append(
                f"[{cid}] TYPE: resolved  CITED: {cited_id}\n"
                f"  FACT: {ctext}\n"
                f"  SENTENCE CONTEXT: {sentence_ctx}\n"
                f"{co_note}"
                f"  CITED SOURCE CONTENT:\n{cited_block}"
            )
        elif is_unresolved:
            raw_citation = cited_id[1:]
            claim_sections.append(
                f"[{cid}] TYPE: unresolved  CITED: \"{raw_citation}\" (not found in sources)\n"
                f"  FACT: {ctext}"
            )
        else:
            claim_sections.append(
                f"[{cid}] TYPE: uncited\n"
                f"  FACT: {ctext}"
            )

    claims_block = "\n\n".join(claim_sections)

    prompt = (
        f"You are auditing an AI pipeline that answered a credit-risk question.\n\n"
        f"QUESTION: {question}\n\n"
        f"ALL RETRIEVED SOURCES:\n{source_listing}\n\n"
        f"CLAIMS TO EVALUATE:\n{claims_block}\n\n"
        "For EACH claim output EXACTLY this block (no extra text between blocks):\n"
        "CLAIM_ID: <id>\n"
        "USED_SOURCE: <source_id or NONE>\n"
        "CORRECT_SOURCE: YES | NO:<better_id> | FABRICATED\n"
        "EXPLANATION: <one sentence referencing the source text>\n"
        "---\n\n"
        "Rules:\n"
        "  - resolved:   USED_SOURCE = the CITED id; judge YES / NO:<id> / FABRICATED\n"
        "  - unresolved: USED_SOURCE = best matching source id (or NONE); "
        "CORRECT_SOURCE = that id or FABRICATED\n"
        "  - uncited:    infer the most likely USED_SOURCE; then judge CORRECT_SOURCE\n"
        "  - YES means the used source directly supports the specific fact\n"
        "  - NO:<better_id> means a different source is clearly better and the used source is irrelevant\n"
        "  - FABRICATED means no source in the list supports this fact\n"
    )

    # ── Single LLM call ───────────────────────────────────────────────────
    max_tokens = max(300, len(claims) * 80)
    try:
        raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"    ✗ Batch LLM call failed: {e}")
        return [
            {
                **c,
                'pipeline_used_source': _resolve_to_content(c['resolved_citation'], registry),
                'correct_source':       'ERROR',
                'is_correct_source':    False,
                'explanation':          str(e),
            }
            for c in claims
        ]

    parsed = _parse_batch_response(raw, claims, registry, valid_ids)

    # ── Assemble results, falling back gracefully for missing claim blocks ──
    results = []
    for claim in claims:
        cid      = claim['claim_id']
        cited_id = claim['resolved_citation']
        is_resolved   = cited_id != 'NONE' and not cited_id.startswith('?')
        is_unresolved = cited_id.startswith('?')

        block = parsed.get(cid)
        if not block:
            print(f"    ✗ No response block for {cid} — marking ERROR")
            results.append({
                **claim,
                'pipeline_used_source': _resolve_to_content(cited_id, registry),
                'correct_source':       'ERROR',
                'is_correct_source':    False,
                'explanation':          'Missing from batch LLM response',
            })
            continue

        # For resolved/unresolved, the used source is already known
        if is_resolved:
            used_sid = cited_id
        elif is_unresolved:
            used_sid = block.get('used_sid', cited_id[1:])
        else:
            used_sid = block.get('used_sid', 'NONE')

        correct_sid = block.get('correct_sid', 'UNKNOWN')
        is_correct  = block.get('is_correct', False)

        # For resolved YES, correct_sid should point to the used (cited) source
        if is_correct and correct_sid == 'UNKNOWN':
            correct_sid = used_sid

        # Unresolved: is_correct is true only if LLM found the cited source itself
        if is_unresolved and is_correct:
            is_correct = False

        results.append({
            **claim,
            'pipeline_used_source': _resolve_to_content(used_sid, registry),
            'correct_source':       _resolve_to_content(correct_sid, registry),
            'is_correct_source':    is_correct,
            'explanation':          block.get('explanation', ''),
        })

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def evaluate_traceability(
    extraction_result_path: str,
    answer_path: str,
    output_csv_path: str,
):
    log_path = output_csv_path.replace('.csv', '_log.txt')

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.__stdout__
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, msg):
            # Ensure msg is properly encoded for terminal
            try:
                self.terminal.write(msg)
            except UnicodeEncodeError:
                # Fallback: replace problematic characters
                self.terminal.write(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            self.log.write(msg)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def close(self):
            self.log.close()

    original_stdout = sys.stdout
    logger = TeeLogger(log_path)
    sys.stdout = logger
    try:
        _run(extraction_result_path, answer_path, output_csv_path)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_path}")


def _run(extraction_result_path: str, answer_path: str, output_csv_path: str):
    print("=" * 70)
    print("Answer-Source Traceability — Phase 1 Ground Truth Evaluation")
    print("=" * 70)

    # ── Load inputs ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading inputs")
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    with open(answer_path, 'r', encoding='utf-8') as f:
        answer_text = f.read().strip()

    question = extraction_data.get('question', '')
    print(f"  Question    : {question[:100]}")
    print(f"  Answer chars: {len(answer_text)}")
    print(f"  Result rows : {extraction_data.get('record_count', len(extraction_data.get('results', [])))}")

    # ── Build source registry ─────────────────────────────────────────────
    print(f"\n[2/5] Building source registry")
    registry = build_source_registry(extraction_data)
    risk_count   = sum(1 for s in registry.values() if s['type'] == 'risk')
    metric_count = sum(1 for s in registry.values() if s['type'] == 'metric')
    print(f"  Total sources: {len(registry)}  ({risk_count} risks, {metric_count} metrics)")
    for sid, src in list(registry.items())[:5]:
        print(f"    [{sid}] {src['summary'][:80]}")
    if len(registry) > 5:
        print(f"    ... and {len(registry) - 5} more")

    # ── Init Bedrock ──────────────────────────────────────────────────────
    print(f"\n[3/5] Initialising LLM (AWS Bedrock)")
    aws_region  = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL_EVAL", "us.meta.llama3-3-70b-instruct-v1:0")
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
    )
    print(f"  Region: {aws_region}  |  Model: {model_id}")

    # ── Extract claims + parse inline citations ───────────────────────────
    print(f"\n[4/5] Parsing claims and inline citations from answer")
    claims = extract_claims(answer_text, registry)
    cited   = sum(1 for c in claims if c['resolved_citation'] != 'NONE' and not c['resolved_citation'].startswith('?'))
    unresolved = sum(1 for c in claims if c['resolved_citation'].startswith('?'))
    uncited = sum(1 for c in claims if c['resolved_citation'] == 'NONE')
    print(f"  → {len(claims)} claims  |  {cited} cited  |  {unresolved} unresolved  |  {uncited} uncited")
    for c in claims[:5]:
        # Show the citation and whether it resolved
        raw_cite = c.get('raw_citation', 'NONE')
        resolved = c.get('resolved_citation', 'NONE')
        if raw_cite != 'NONE':
            if resolved.startswith('?'):
                print(f"    {c['claim_id']}: {c['claim_text'][:70]}  [CITE:{raw_cite}] ✗ NOT FOUND")
            else:
                print(f"    {c['claim_id']}: {c['claim_text'][:70]}  [CITE:{raw_cite}] ✓")
        else:
            print(f"    {c['claim_id']}: {c['claim_text'][:70]}  [NO CITATION]")
    if len(claims) > 5:
        print(f"    ... and {len(claims) - 5} more")

    if not claims:
        print("\n✗ No claims found — aborting")
        return

    # ── Judge source correctness (LLM) ────────────────────────────────────
    print(f"\n[5/5] Judging source correctness ({len(claims)} claims — 1 batched LLM call)")
    records = attribute_claims(claims, registry, question, bedrock_client, model_id)

    # ── Metrics ───────────────────────────────────────────────────────────
    total         = len(records)
    correct       = sum(1 for r in records if r['is_correct_source'])
    fabricated    = sum(1 for r in records if r.get('correct_source') == 'FABRICATED')
    wrong_source  = total - correct - fabricated
    traceability  = correct / total * 100 if total else 0

    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    print(f"  Total claims       : {total}")
    print(f"  Correct source     : {correct}  ({traceability:.1f}%)")
    print(f"  Wrong source       : {wrong_source}")
    print(f"  Fabricated (no src): {fabricated}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        'claim_id', 'claim_text',
        'pipeline_used_source', 'correct_source',
        'is_correct_source', 'explanation',
        'claim_text_raw',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f"\n✓ Results written to {output_csv_path}")
    print("=" * 70)
    print(f"Source Traceability Score: {traceability:.1f}%")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--out-dir', default=None)
    args, _ = ap.parse_known_args()
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    retrival_dir = os.path.join(script_dir, '..', '..', 'retrival_results')
    out_dir      = args.out_dir or os.path.join(script_dir, '..')
    ext_path     = os.path.join(out_dir, 'extraction_result.json') if args.out_dir else os.path.join(retrival_dir, 'extraction_result.json')
    ans_path     = os.path.join(out_dir, 'answer.txt') if args.out_dir else os.path.join(retrival_dir, 'answer.txt')

    evaluate_traceability(
        extraction_result_path=ext_path,
        answer_path=ans_path,
        output_csv_path=os.path.join(out_dir, 'answer_source_traceability.csv'),
    )
