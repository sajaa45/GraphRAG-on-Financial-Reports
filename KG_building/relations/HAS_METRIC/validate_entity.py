import sys
import os
import json
import re
import hashlib
import argparse
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import boto3
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
load_dotenv(env_path)

MODEL = os.getenv("BEDROCK_MODEL", "qwen.qwen3-next-80b-a3b")
PHASE1_BATCH = 30    # max metrics per within-chunk LLM call
CATEGORY_BATCH = 50  # max metrics per per-category LLM call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_json_array(raw: str) -> List[Dict]:
    raw = _strip_think(raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _call_llm(client, prompt: str, max_tokens: int = 1500) -> List[Dict]:
    response = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": max_tokens},
    )
    raw = response['output']['message']['content'][0]['text']
    return _parse_json_array(raw)


def _normalize_value(val: str) -> Optional[str]:
    try:
        return str(round(float(str(val).replace(',', '')), 4))
    except (ValueError, TypeError):
        return None


def _metric_compact(m: Dict) -> str:
    p = m['tgt']['properties']
    return (
        f"[{m['_idx']}] metric_type=\"{p.get('metric_type', m['tgt']['name'])}\" "
        f"value={p.get('value')} unit=\"{p.get('unit')}\" "
        f"year={p.get('year')} category={p.get('category')}"
    )


def _apply_merge_decisions(metrics: List[Dict], decisions: List[Dict]) -> List[Dict]:
    dec_map = {d['idx']: d for d in decisions if 'idx' in d}
    idx_to_m = {m['_idx']: m for m in metrics}

    kept_indices = set()
    for m in metrics:
        idx = m['_idx']
        action = dec_map.get(idx, {}).get('action', 'KEEP').upper()
        if action == 'KEEP':
            kept_indices.add(idx)
        elif action == 'MERGE':
            target = dec_map[idx].get('merge_into', idx)
            kept_indices.add(target if target in idx_to_m else idx)

    return [idx_to_m[i] for i in sorted(kept_indices) if i in idx_to_m]


# ---------------------------------------------------------------------------
# Phase 1 — Within-chunk: hallucination filter + value/unit correction
# ---------------------------------------------------------------------------

def _build_phase1_prompt(company_name: str, chunk_text: str, metrics: List[Dict]) -> str:
    metrics_str = "\n".join(_metric_compact(m) for m in metrics)
    return f"""/no_think

You are validating financial metrics extracted from {company_name}'s 10-K filing.

SOURCE TEXT:
{chunk_text}

EXTRACTED METRICS:
{metrics_str}

For each metric, return exactly one of:
- KEEP   → value and unit are correct and the metric can be verified in the source text
- REMOVE → the metric or its specific numeric value is NOT present in the source text (hallucination)
- FIX    → metric is real but value or unit is wrong (e.g. "2119 $ billion" should be "2.119 $ billion"); provide corrected value and unit
- MERGE  → this metric is a duplicate of another metric in this list (same concept, same year); provide merge_into idx

Rules:
- REMOVE only if the value is completely absent from the source text.
- FIX when: value is a unit-scale error (billions vs millions), wrong sign, or unit label is incorrect.
  Always derive the corrected value and unit directly from the source text.
- MERGE when two metrics measure the same concept for the same year, even with different names.
  Prefer the more standard/canonical name as the merge target.
- Year-only values (e.g. maturity year "2029") are valid if the source text mentions them.
- Do NOT remove a metric just because another broader metric exists.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "REMOVE"}},
  {{"idx": 3, "action": "FIX", "value": "2.119", "unit": "$ billion"}},
  {{"idx": 4, "action": "MERGE", "merge_into": 1}}
]"""


def _within_chunk_pass(
    client,
    company_name: str,
    groups: Dict[str, List[Dict]],
    chunk_texts: Dict[str, str],
) -> List[Dict]:
    kept: List[Dict] = []

    for cid, group in groups.items():
        chunk = chunk_texts[cid]
        print(f"  Chunk {cid} ({len(group)} metrics, {len(chunk)} chars) ...")

        if len(group) == 1:
            kept.append(group[0])
            continue

        batches = [group[i:i + PHASE1_BATCH] for i in range(0, len(group), PHASE1_BATCH)]
        chunk_kept: Dict[int, Dict] = {}

        for b_num, batch in enumerate(batches, 1):
            if len(batches) > 1:
                print(f"    Batch {b_num}/{len(batches)} ...")

            prompt = _build_phase1_prompt(company_name, chunk, batch)
            decisions = _call_llm(client, prompt)
            dec_map = {d['idx']: d for d in decisions if 'idx' in d}

            for m in batch:
                idx = m['_idx']
                d = dec_map.get(idx, {'action': 'KEEP'})
                action = d.get('action', 'KEEP').upper()

                if action == 'REMOVE':
                    print(f"    REMOVE [{idx}] {m['tgt']['name']}")

                elif action == 'FIX':
                    old_val = m['tgt']['properties']['value']
                    old_unit = m['tgt']['properties']['unit']
                    m['tgt']['properties']['value'] = str(d.get('value', old_val))
                    m['tgt']['properties']['unit'] = d.get('unit', old_unit)
                    print(f"    FIX    [{idx}] {m['tgt']['name']}: {old_val} {old_unit} → {m['tgt']['properties']['value']} {m['tgt']['properties']['unit']}")
                    chunk_kept[idx] = m

                elif action == 'MERGE':
                    target = d.get('merge_into', idx)
                    t = next((x for x in group if x['_idx'] == target), None)
                    if t:
                        chunk_kept[target] = t
                        print(f"    MERGE  [{idx}] → [{target}]")
                    else:
                        chunk_kept[idx] = m  # orphaned merge → keep

                else:  # KEEP or unrecognised
                    chunk_kept[idx] = m

        kept.extend(chunk_kept.values())
        removed = len(group) - len(chunk_kept)
        if removed:
            print(f"    → kept={len(chunk_kept)}, removed/merged={removed}")

    return kept


# ---------------------------------------------------------------------------
# Phase 2a — Per-category cross-chunk dedup (same concept, same value+year)
# ---------------------------------------------------------------------------

def _build_phase2a_prompt(company_name: str, category: str, metrics: List[Dict]) -> str:
    lines = "\n".join(
        f"[{m['_idx']}] \"{m['tgt']['properties'].get('metric_type', m['tgt']['name'])}\" "
        f"= {m['tgt']['properties'].get('value')} {m['tgt']['properties'].get('unit')} "
        f"(year={m['tgt']['properties'].get('year')})"
        for m in metrics
    )
    return f"""/no_think

These are {category} metrics extracted from different sections of {company_name}'s 10-K filing.
Some may be duplicates: same metric concept, same year, same (or equivalent) numeric value, but named differently.

METRICS:
{lines}

Identify duplicates where the metric names are close synonyms AND the value and year match.
Examples of duplicates:
- "Net income" / "Net income attributable to [Company]" with the same value → duplicates
- "Consolidated income from operations" / "Income from operations" with the same value → duplicates
- "Net sales" / "Total net revenues" with the same value → duplicates

Examples that are NOT duplicates even with similar names:
- Same metric type but different years → NOT duplicates
- Same metric type but different values (e.g. segment vs consolidated total) → NOT duplicates
- Related but distinct concepts (e.g. "Operating income" vs "Net income") → NOT duplicates

For each duplicate, return MERGE into the more standard or canonical entry. For all others, return KEEP.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}}
]"""


def _per_category_pass(client, company_name: str, metrics: List[Dict]) -> List[Dict]:
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    uncategorised: List[Dict] = []

    for m in metrics:
        cat = m['tgt']['properties'].get('category', '').strip()
        if cat:
            by_category[cat].append(m)
        else:
            uncategorised.append(m)

    surviving = list(uncategorised)

    for cat, cat_metrics in by_category.items():
        print(f"  Category '{cat}': {len(cat_metrics)} metrics ...")

        if len(cat_metrics) <= 1:
            surviving.extend(cat_metrics)
            continue

        all_kept: List[Dict] = []
        batches = [cat_metrics[i:i + CATEGORY_BATCH] for i in range(0, len(cat_metrics), CATEGORY_BATCH)]

        for b_num, batch in enumerate(batches, 1):
            if len(batches) > 1:
                print(f"    Batch {b_num}/{len(batches)} ...")
            prompt = _build_phase2a_prompt(company_name, cat, batch)
            decisions = _call_llm(client, prompt, max_tokens=512)
            all_kept.extend(_apply_merge_decisions(batch, decisions))

        removed = len(cat_metrics) - len(all_kept)
        if removed:
            print(f"    → kept={len(all_kept)}, merged={removed}")
        surviving.extend(all_kept)

    return surviving


# ---------------------------------------------------------------------------
# Phase 2b — Cross-category dedup (same value+year, different category)
# ---------------------------------------------------------------------------

def _find_cross_category_candidates(metrics: List[Dict]) -> List[List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for m in metrics:
        p = m['tgt']['properties']
        norm_val = _normalize_value(p.get('value', ''))
        if norm_val is None:
            continue
        key = (p.get('year', ''), norm_val)
        groups[key].append(m)

    return [
        g for g in groups.values()
        if len(g) > 1
        and len({m['tgt']['properties'].get('category', '') for m in g}) > 1
    ]


def _build_phase2b_prompt(company_name: str, candidate_groups: List[List[Dict]]) -> str:
    lines = []
    for group in candidate_groups:
        lines.append("--- Group (same value+year) ---")
        for m in group:
            p = m['tgt']['properties']
            lines.append(
                f"  [{m['_idx']}] \"{p.get('metric_type', m['tgt']['name'])}\" "
                f"= {p.get('value')} {p.get('unit')} "
                f"year={p.get('year')} category={p.get('category')}"
            )
    groups_str = "\n".join(lines)

    return f"""/no_think

These metrics from {company_name}'s 10-K share the same numeric value and year but were assigned different categories.
Decide whether each group contains true duplicates (same underlying metric, miscategorised) or genuinely distinct metrics that happen to share a value.

CANDIDATE GROUPS:
{groups_str}

For duplicates within a group: MERGE the less appropriate ones into the best-categorised entry.
For non-duplicates (same value is a coincidence): KEEP all.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}}
]"""


def _cross_category_pass(client, company_name: str, metrics: List[Dict]) -> List[Dict]:
    candidate_groups = _find_cross_category_candidates(metrics)
    if not candidate_groups:
        print("  No cross-category candidates found.")
        return metrics

    total = sum(len(g) for g in candidate_groups)
    print(f"  {len(candidate_groups)} candidate groups ({total} metrics) ...")

    MAX_GROUPS_PER_CALL = 15
    all_decisions: List[Dict] = []
    for i in range(0, len(candidate_groups), MAX_GROUPS_PER_CALL):
        batch = candidate_groups[i:i + MAX_GROUPS_PER_CALL]
        prompt = _build_phase2b_prompt(company_name, batch)
        all_decisions.extend(_call_llm(client, prompt, max_tokens=512))

    result = _apply_merge_decisions(metrics, all_decisions)
    removed = len(metrics) - len(result)
    if removed:
        print(f"  → merged {removed} cross-category duplicates")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_has_metric(extracted_json_path: str) -> Dict:
    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data.get('main_company', 'the Company')
    relations = data.get('relations', {}).get('HAS_METRIC', [])

    if not relations:
        print("No HAS_METRIC relations found.")
        return {'main_company': company_name, 'relations': {'HAS_METRIC': []}}

    print(f"Loaded {len(relations)} raw metrics for '{company_name}'")

    # Fix wrong src.name — extraction sometimes records "Anchor" instead of the company
    for rel in relations:
        if rel.get('src', {}).get('name', '') != company_name:
            rel['src']['name'] = company_name

    # Exact-name + same-year dedup (free, no LLM)
    seen: set = set()
    unique: List[Dict] = []
    for rel in relations:
        key = (rel['tgt']['name'].lower().strip(), rel['tgt']['properties'].get('year', ''))
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    print(f"After exact-name dedup: {len(unique)} metrics")

    # Assign global indices and group by chunk
    chunk_texts: Dict[str, str] = {}
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for i, rel in enumerate(unique, 1):
        rel['_idx'] = i
        chunk = rel.get('chunk_text', '')
        cid = _chunk_id(chunk)
        chunk_texts[cid] = chunk
        rel['_chunk_id'] = cid
        groups[cid].append(rel)

    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )

    # Phase 1: within-chunk validation
    print(f"\n[Phase 1] Within-chunk validation ({len(groups)} chunks) ...")
    surviving = _within_chunk_pass(client, company_name, groups, chunk_texts)
    print(f"After phase 1: {len(surviving)} metrics")

    # Phase 2a: per-category cross-chunk dedup
    print(f"\n[Phase 2a] Per-category deduplication ...")
    surviving = _per_category_pass(client, company_name, surviving)
    print(f"After phase 2a: {len(surviving)} metrics")

    # Phase 2b: cross-category dedup
    print(f"\n[Phase 2b] Cross-category deduplication ...")
    surviving = _cross_category_pass(client, company_name, surviving)
    print(f"After phase 2b: {len(surviving)} metrics")

    # Clean internal bookkeeping keys
    final_relations = []
    for rel in sorted(surviving, key=lambda r: r['_idx']):
        clean = {k: v for k, v in rel.items() if not k.startswith('_')}
        final_relations.append(clean)

    removed = len(relations) - len(final_relations)
    print(f"\nSummary: {len(relations)} input → {len(final_relations)} kept ({removed} removed/merged)")

    return {'main_company': company_name, 'relations': {'HAS_METRIC': final_relations}}


def main():
    parser = argparse.ArgumentParser(description="Validate HAS_METRIC relations.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--input',
        default=os.path.join(script_dir, 'extracted_output.json'),
        help="Path to extracted HAS_METRIC JSON file",
    )
    parser.add_argument(
        '--output',
        default=os.path.join(script_dir, 'validated_metrics.json'),
        help="Path to write the validated JSON file",
    )
    args = parser.parse_args()

    result = validate_has_metric(args.input)

    print("\nFinal metric list:")
    for rel in result['relations']['HAS_METRIC']:
        p = rel['tgt']['properties']
        print(f"  [{p.get('category','?')}] {rel['tgt']['name']} = {p.get('value')} {p.get('unit')}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == '__main__':
    main()
