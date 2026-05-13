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


def _call_llm(client, prompt: str, max_tokens: int = 1024) -> List[Dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "You are a precise JSON validator. Return only valid JSON arrays, no explanation.\n\n" + prompt
                }
            ]
        }
    ]
    
    response = client.converse(
        modelId=MODEL,
        messages=messages,
        inferenceConfig={
            "temperature": 0.0,
            "maxTokens": max_tokens
        }
    )
    
    content = response['output']['message']['content'][0]['text']
    return _parse_json_array(content)




# ---------------------------------------------------------------------------
# Phase 1 — Within-chunk deduplication + hallucination filtering
# ---------------------------------------------------------------------------

def _build_within_chunk_prompt(company_name: str, chunk_text: str, risks: List[Dict]) -> str:
    risk_lines = "\n".join(
        f"[{r['_idx']}] name=\"{r['tgt']['name']}\" | "
        f"desc=\"{r['tgt']['properties'].get('description', '')}\" | "
        f"why=\"{r['tgt']['properties'].get('why', '')}\""
        for r in risks
    )
    return f"""/no_think

You are reviewing risks extracted from {company_name}'s 10-K filing.

SOURCE TEXT (the actual passage these risks were extracted from):
{chunk_text}

EXTRACTED RISKS:
{risk_lines}

For each risk, decide:
- KEEP   → distinct risk genuinely supported by the source text
- MERGE X → near-duplicate of risk X (same underlying risk, different wording); X is kept
- REMOVE  → hallucination: not actually described in the source text

Rules:
- A risk is a hallucination if neither its name nor description can be found anywhere in the source text.
- Two risks are near-duplicates if they describe the same root cause and impact — even with different names.
- When merging, prefer the more descriptive or specific risk as the target (MERGE into it).
- Risks on different aspects of the same broad theme (e.g. "debt covenants" vs "refinancing") are NOT duplicates.
- KEEP all risks that are genuinely distinct, even if from the same paragraph.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}},
  {{"idx": 3, "action": "REMOVE"}}
]"""


def _within_chunk_pass(
    client,
    company_name: str,
    groups: Dict[str, List[Dict]],
    chunk_texts: Dict[str, str],
) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Run per-chunk validation. Returns (surviving_relations, idx_to_removal_reason).
    """
    kept: List[Dict] = []
    removed_reason: Dict[int, str] = {}

    for cid, group in groups.items():
        full_chunk = chunk_texts[cid]
        print(f"  Chunk {cid} ({len(group)} risks, {len(full_chunk)} chars) ...")

        if len(group) == 1:
            kept.append(group[0])
            continue

        prompt = _build_within_chunk_prompt(company_name, full_chunk, group)
        decisions = _call_llm(client, prompt)

        dec_map = {d['idx']: d for d in decisions if 'idx' in d}

        # Default missing decisions to KEEP
        for r in group:
            if r['_idx'] not in dec_map:
                dec_map[r['_idx']] = {'idx': r['_idx'], 'action': 'KEEP'}

        # First pass: collect kept and merge targets
        group_kept = {}
        merge_map = {}  # idx → merge_into
        for r in group:
            idx = r['_idx']
            d = dec_map.get(idx, {'action': 'KEEP'})
            action = d.get('action', 'KEEP').upper()
            if action == 'KEEP':
                group_kept[idx] = r
            elif action == 'MERGE':
                merge_map[idx] = d.get('merge_into', idx)
            elif action == 'REMOVE':
                removed_reason[idx] = 'hallucination'

        # Resolve merge targets: ensure merge_into is kept
        for idx, target in merge_map.items():
            r = next((x for x in group if x['_idx'] == idx), None)
            if r is None:
                continue
            t = next((x for x in group if x['_idx'] == target), None)
            if t is None:
                # Orphaned merge → keep original
                group_kept[idx] = r
            else:
                # Ensure the target is kept
                group_kept[target] = t
                removed_reason[idx] = f'merged into {target}'

        kept.extend(group_kept.values())
        removed = [r for r in group if r['_idx'] not in group_kept]
        print(f"    kept={len(group_kept)}, removed/merged={len(removed)}")

    return kept, removed_reason


# ---------------------------------------------------------------------------
# Phase 2 — Cross-chunk deduplication (names + descriptions only, no chunk text)
# ---------------------------------------------------------------------------

def _build_cross_chunk_prompt(company_name: str, risks: List[Dict]) -> str:
    risk_lines = "\n".join(
        f"[{r['_idx']}] name=\"{r['tgt']['name']}\" | "
        f"desc=\"{r['tgt']['properties'].get('description', '')[:120]}\" | "
        f"why=\"{r['tgt']['properties'].get('why', '')[:120]}\""
        for r in risks
    )
    return f"""/no_think

These risks were extracted from different sections of {company_name}'s 10-K filing.
Some risks that appeared in separate sections may describe the same underlying risk.

RISKS:
{risk_lines}

Identify any cross-section near-duplicates and decide which to merge.
Two risks are near-duplicates only if they share the SAME root cause (desc) AND the same mechanism/impact (why).
Risks with overlapping themes but different mechanisms or impacts are NOT duplicates.

For each risk that should be removed as a duplicate, return MERGE. For all others, return KEEP.
When merging, prefer the more detailed or specific entry as the target.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}}
]"""


def _cross_chunk_pass(client, company_name: str, risks: List[Dict]) -> List[Dict]:
    if len(risks) <= 1:
        return risks

    print(f"  Cross-chunk dedup: {len(risks)} risks ...")
    prompt = _build_cross_chunk_prompt(company_name, risks)
    decisions = _call_llm(client, prompt, max_tokens=512)

    dec_map = {d['idx']: d for d in decisions if 'idx' in d}
    idx_to_rel = {r['_idx']: r for r in risks}

    kept_indices = set()
    for r in risks:
        idx = r['_idx']
        d = dec_map.get(idx, {'action': 'KEEP'})
        action = d.get('action', 'KEEP').upper()
        if action == 'KEEP':
            kept_indices.add(idx)
        elif action == 'MERGE':
            target = d.get('merge_into', idx)
            kept_indices.add(target)
            # Ensure target exists
            if target not in idx_to_rel:
                kept_indices.add(idx)

    result = [idx_to_rel[i] for i in sorted(kept_indices) if i in idx_to_rel]
    print(f"    kept={len(result)}, merged/removed={len(risks) - len(result)}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_faces_risk(extracted_json_path: str) -> Dict:
    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data.get('main_company', 'the Company')
    relations = data.get('relations', {}).get('FACES_RISK', [])

    if not relations:
        print("No FACES_RISK relations found.")
        return {'main_company': company_name, 'relations': {'FACES_RISK': []}}

    print(f"Loaded {len(relations)} raw risks for '{company_name}'")

    # Assign global indices and collect chunk texts
    chunk_texts: Dict[str, str] = {}
    groups: Dict[str, List[Dict]] = defaultdict(list)
    seen_names: set = set()
    global_idx = 0

    for rel in relations:
        name_norm = rel.get('tgt', {}).get('name', '').strip().lower()
        if name_norm in seen_names:
            continue  # exact-name dedup (free, no LLM needed)
        seen_names.add(name_norm)

        chunk = rel.get('chunk_text', '')
        cid = _chunk_id(chunk)
        chunk_texts[cid] = chunk

        global_idx += 1
        rel['_idx'] = global_idx
        rel['_chunk_id'] = cid
        groups[cid].append(rel)

    total_after_exact = sum(len(g) for g in groups.values())
    print(f"After exact-name dedup: {total_after_exact} risks across {len(groups)} unique chunks")

    # Initialize AWS Bedrock client
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

    # Phase 1: within-chunk dedup + hallucination filter
    print("\n[Phase 1] Within-chunk validation ...")
    surviving, removed_reasons = _within_chunk_pass(client, company_name, groups, chunk_texts)
    print(f"After phase 1: {len(surviving)} risks remain")

    # Phase 2: cross-chunk dedup
    print("\n[Phase 2] Cross-chunk deduplication ...")
    final_risks = _cross_chunk_pass(client, company_name, surviving)
    print(f"After phase 2: {len(final_risks)} final risks")

    # Clean internal bookkeeping keys before output
    final_relations = []
    for rel in sorted(final_risks, key=lambda r: r['_idx']):
        clean = {k: v for k, v in rel.items() if not k.startswith('_')}
        final_relations.append(clean)

    removed_count = len(relations) - len(final_relations)
    print(f"\nSummary: {len(relations)} input → {len(final_relations)} kept "
          f"({removed_count} duplicates/hallucinations removed)")

    return {
        'main_company': company_name,
        'relations': {'FACES_RISK': final_relations},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate FACES_RISK relations: deduplicate and remove hallucinations."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--input',
        default=os.path.join(script_dir, 'extracted_output.json'),
        help="Path to extracted FACES_RISK JSON file",
    )
    parser.add_argument(
        '--output',
        default=os.path.join(script_dir, 'validated_risks.json'),
        help="Path to write the validated JSON file",
    )
    args = parser.parse_args()

    result = validate_faces_risk(args.input)

    print("\nFinal risk list:")
    for rel in result['relations']['FACES_RISK']:
        print(f"  - {rel['tgt']['name']}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == '__main__':
    main()
