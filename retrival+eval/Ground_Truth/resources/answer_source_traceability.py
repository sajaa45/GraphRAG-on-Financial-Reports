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
import time
from typing import Any

import boto3
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

# ---------------------------------------------------------------------------
# Target company document URL
# Set this to the local HTML file path or remote URL for the target company's
# 10-K filing. Used as the source reference for target metrics (which have
# no source_url in the graph since they come from HTML parsing, not XBRL).
# ---------------------------------------------------------------------------
TARGET_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/891014/000089101425000018/form10k.htm"  # e.g. "file:///C:/path/to/filing.htm" or "https://..."

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
                    'source_url':  TARGET_DOCUMENT_URL,
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
                        'source_url':  m.get('source_url', ''),
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
        url   = full.get('source_url', '')
        lines = [
            f"  Metric: {label}",
            f"  Company: {src['company']} ({src['role']})",
            f"  Value: {value} {unit} ({year})",
        ]
        if url:
            lines.append(f"  Source URL: {url}")
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
        url   = full.get('source_url', '')
        return f"[{citation_id}] {src['company']} — {label} = {value} {unit} ({year}) | {url}"


def _build_source_listing(registry: dict[str, dict]) -> str:
    """Full source content listing for LLM prompts — includes description and source text."""
    blocks = []
    for sid, src in registry.items():
        blocks.append(f"[{sid}]\n{_source_content(src)}")
    return "\n\n".join(blocks)


def attribute_claims(
    claims: list[dict],
    registry: dict[str, dict],
    question: str,
    bedrock_client,
    model_id: str,
) -> list[dict]:
    """
    For each claim judge whether the pipeline used the correct source.
    - Resolved citation  → show the actual source text; ask only whether it's correct.
    - Unresolved citation → pipeline fabricated a source; ask LLM for the right one.
    - No citation        → full inference against source content.

    CSV output stores actual source content, not registry IDs.
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

    results = []
    for claim in claims:
        cid           = claim['claim_id']
        ctext         = claim['claim_text']
        cited_id      = claim['resolved_citation']  # citation_id, '?<raw>', or 'NONE'
        is_resolved   = cited_id != 'NONE' and not cited_id.startswith('?')
        is_unresolved = cited_id.startswith('?')

        if is_resolved:
            cited_block   = _source_content(registry[cited_id])
            sentence_ctx  = claim.get('sentence_context', ctext)
            co_cites      = claim.get('co_citations', [])
            co_note = ''
            if co_cites:
                co_ids = ', '.join(co_cites)
                co_note = (
                    f"\nNOTE: The pipeline also co-cited [{co_ids}] for this same fact. "
                    f"Each co-citation only needs to partially support the claim. "
                    f"Answer YES if [{cited_id}] validly contributes to supporting the fact, "
                    f"even if a co-citation covers a different aspect of it.\n"
                )
            prompt = (
                f"You are auditing an AI pipeline that answered a credit-risk question.\n\n"
                f"QUESTION: {question}\n\n"
                f"FULL SENTENCE (for context): {sentence_ctx}\n\n"
                f"SPECIFIC FACT BEING CITED: {ctext}\n"
                f"(The citation tag annotates only this specific fact, not the full sentence.){co_note}\n\n"
                f"The pipeline cited this source [{cited_id}]:\n{cited_block}\n\n"
                f"ALL RETRIEVED SOURCES:\n{source_listing}\n\n"
                "Judge: does the cited source support the SPECIFIC FACT above?\n"
                "   - YES if the source text contains the value, figure, or statement in the specific fact.\n"
                "   - NO:<better_id> if a DIFFERENT source in the list is a clearly better match and the cited source is irrelevant.\n"
                "   - FABRICATED if no source in the list supports this specific fact.\n\n"
                "Output EXACTLY two lines:\n"
                "CORRECT_SOURCE: YES | NO:<better_id> | FABRICATED\n"
                "EXPLANATION: <one sentence referencing the source text>\n"
            )
        elif is_unresolved:
            raw_citation = cited_id[1:]
            prompt = (
                f"You are auditing an AI pipeline that answered a credit-risk question.\n\n"
                f"QUESTION: {question}\n\n"
                f"CLAIM: {ctext}\n\n"
                f"The pipeline cited \"{raw_citation}\" but this does not match any retrieved source.\n\n"
                f"ALL RETRIEVED SOURCES:\n{source_listing}\n\n"
                "Read the sources above. Which one actually supports this claim?\n"
                "   - Output the source ID (e.g. M005) if a source supports it.\n"
                "   - FABRICATED if no source in the list supports this claim.\n\n"
                "Output EXACTLY two lines:\n"
                "CORRECT_SOURCE: <ID> | FABRICATED\n"
                "EXPLANATION: <one sentence referencing the source text>\n"
            )
        else:
            prompt = (
                f"You are auditing an AI pipeline that answered a credit-risk question.\n\n"
                f"QUESTION: {question}\n\n"
                f"CLAIM: {ctext}\n"
                f"(No inline citation was provided.)\n\n"
                f"ALL RETRIEVED SOURCES:\n{source_listing}\n\n"
                "Read the sources above.\n"
                "1. USED_SOURCE: Which source did the pipeline most likely draw on? "
                "Output the ID or NONE.\n"
                "2. CORRECT_SOURCE: Is that the correct/best source?\n"
                "   - YES | NO:<better_id> | FABRICATED\n\n"
                "Output EXACTLY three lines:\n"
                "USED_SOURCE: <ID or NONE>\n"
                "CORRECT_SOURCE: YES | NO:<better_id> | FABRICATED\n"
                "EXPLANATION: <one sentence referencing the source text>\n"
            )

        try:
            raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=300)
        except Exception as e:
            print(f"    ✗ LLM error for {cid}: {e}")
            results.append({**claim,
                'pipeline_used_source': _resolve_to_content(cited_id, registry),
                'correct_source':       'ERROR',
                'is_correct_source':    False,
                'explanation':          str(e),
            })
            time.sleep(1)
            continue

        # Parse IDs from response, then resolve to content for CSV
        used_sid    = cited_id if is_resolved else (cited_id[1:] if is_unresolved else 'NONE')
        correct_sid = 'UNKNOWN'
        is_correct  = False
        explanation = ''

        for line in raw.splitlines():
            line  = line.strip()
            upper = line.upper()
            if upper.startswith('USED_SOURCE:') and not (is_resolved or is_unresolved):
                val = line.split(':', 1)[1].strip().upper()
                used_sid = val if val in valid_ids else ('NONE' if val == 'NONE' else f'?{val}')
            elif upper.startswith('CORRECT_SOURCE:'):
                val = line.split(':', 1)[1].strip()
                if val.upper() == 'YES' and not is_unresolved:
                    is_correct  = True
                    correct_sid = used_sid
                elif val.upper() == 'FABRICATED':
                    correct_sid = 'FABRICATED'
                elif val.upper().startswith('NO:'):
                    better = val[3:].strip().upper()
                    correct_sid = better if better in valid_ids else f'?{better}'
                else:
                    candidate = val.strip().upper()
                    correct_sid = candidate if candidate in valid_ids else f'?{candidate}'
            elif upper.startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[1].strip()

        results.append({
            **claim,
            'pipeline_used_source': _resolve_to_content(used_sid, registry),
            'correct_source':       _resolve_to_content(correct_sid, registry),
            'is_correct_source':    is_correct,
            'explanation':          explanation,
        })
        time.sleep(0.3)

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
    model_id    = os.getenv("BEDROCK_MODEL", "us.meta.llama3-2-90b-instruct-v1:0")
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
    print(f"\n[5/5] Judging source correctness ({len(claims)} LLM calls)")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    retrival_dir = os.path.join(script_dir, '..', '..', 'retrival_results')

    evaluate_traceability(
        extraction_result_path=os.path.join(retrival_dir, 'extraction_result.json'),
        answer_path=os.path.join(retrival_dir, 'answer.txt'),
        output_csv_path=os.path.join(script_dir, '..', 'answer_source_traceability.csv'),
    )
