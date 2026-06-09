import os
import json
import re
import hashlib
import argparse
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Callable

import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

MODEL = os.getenv("BEDROCK_MODEL", "qwen.qwen3-next-80b-a3b")
PHASE1_BATCH = 30
CATEGORY_BATCH = 50


def _chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _parse_json_array(raw: str) -> List[Dict]:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []



def _normalize_value(val: str) -> Optional[str]:
    try:
        return str(round(float(str(val).replace(',', '')), 4))
    except (ValueError, TypeError):
        return None


def _metric_compact(m: Dict) -> str:
    p = m['tgt']['properties']
    return (
        f"[{m['_idx']}] metric_type=\"{p.get('metric_type', m['tgt']['name'])}\" "
        f"gaap_concept=\"{p.get('gaap_concept', '')}\" "
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


def _build_phase1_prompt(company_name: str, chunk_text: str, metrics: List[Dict]) -> str:
    metrics_str = "\n".join(_metric_compact(m) for m in metrics)
    return f"""/no_think

You are validating financial metrics extracted from {company_name}'s 10-K filing.

SOURCE TEXT:
{chunk_text}

EXTRACTED METRICS:
{metrics_str}

For each metric, return exactly one of:
- KEEP   → value, unit, and gaap_concept are all correct and the metric can be verified in the source text
- REMOVE → the metric or its specific numeric value is NOT present in the source text (hallucination)
- FIX    → metric is real but value, unit, or gaap_concept is wrong; provide the corrected fields
- MERGE  → this metric is a duplicate of another metric in this list (same concept, same year); provide merge_into idx

Rules:
- REMOVE only if the value is completely absent from the source text.
- FIX when: value is a unit-scale error (billions vs millions), wrong sign, unit label is incorrect,
  OR gaap_concept does not match the standard GAAP name for this metric_type.
  Always derive corrected value and unit directly from the source text.
  For gaap_concept: provide the closest standard US GAAP line-item name
  (e.g. "Revenue", "Operating Income (Loss)", "Net Income (Loss)", "Total Debt",
  "Interest Expense", "EBITDA", "Cash and Cash Equivalents").
- MERGE when two metrics measure the same concept for the same year, even with different names.
  Prefer the entry whose gaap_concept is more precise as the merge target.
- Year-only values (e.g. maturity year "2029") are valid if the source text mentions them.
- Do NOT remove a metric just because another broader metric exists.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "REMOVE"}},
  {{"idx": 3, "action": "FIX", "value": "2.119", "unit": "$ billion", "gaap_concept": "Revenue"}},
  {{"idx": 4, "action": "MERGE", "merge_into": 1}}
]"""


def _within_chunk_pass(llm_fn: Callable, company_name: str, groups: Dict[str, List[Dict]], chunk_texts: Dict[str, str]) -> List[Dict]:
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
            decisions = _parse_json_array(llm_fn(_build_phase1_prompt(company_name, chunk, batch)))
            dec_map = {d['idx']: d for d in decisions if 'idx' in d}
            for m in batch:
                idx = m['_idx']
                d = dec_map.get(idx, {'action': 'KEEP'})
                action = d.get('action', 'KEEP').upper()
                if action == 'REMOVE':
                    print(f"    REMOVE [{idx}] {m['tgt']['name']}")
                elif action == 'FIX':
                    old_val  = m['tgt']['properties']['value']
                    old_unit = m['tgt']['properties']['unit']
                    old_gaap = m['tgt']['properties'].get('gaap_concept', '')
                    m['tgt']['properties']['value'] = str(d.get('value', old_val))
                    m['tgt']['properties']['unit']  = d.get('unit', old_unit)
                    if d.get('gaap_concept'):
                        m['tgt']['properties']['gaap_concept'] = d['gaap_concept']
                    new_gaap = m['tgt']['properties'].get('gaap_concept', '')
                    fix_parts = []
                    if m['tgt']['properties']['value'] != old_val or m['tgt']['properties']['unit'] != old_unit:
                        fix_parts.append(f"{old_val} {old_unit} → {m['tgt']['properties']['value']} {m['tgt']['properties']['unit']}")
                    if new_gaap != old_gaap:
                        fix_parts.append(f"gaap_concept: \"{old_gaap}\" → \"{new_gaap}\"")
                    print(f"    FIX    [{idx}] {m['tgt']['name']}: {' | '.join(fix_parts) or '(no change)'}")
                    chunk_kept[idx] = m
                elif action == 'MERGE':
                    target = d.get('merge_into', idx)
                    t = next((x for x in group if x['_idx'] == target), None)
                    if t:
                        chunk_kept[target] = t
                        print(f"    MERGE  [{idx}] → [{target}]")
                    else:
                        chunk_kept[idx] = m
                else:
                    chunk_kept[idx] = m
        kept.extend(chunk_kept.values())
        removed = len(group) - len(chunk_kept)
        if removed:
            print(f"    → kept={len(chunk_kept)}, removed/merged={removed}")
    return kept


def _build_gaap_mapping_prompt(company_name: str, metrics: List[Dict]) -> str:
    lines = "\n".join(
        f"[{m['_idx']}] metric_type=\"{m['tgt']['properties'].get('metric_type', m['tgt']['name'])}\" "
        f"gaap_concept=\"{m['tgt']['properties'].get('gaap_concept', '')}\" "
        f"value={m['tgt']['properties'].get('value')} unit=\"{m['tgt']['properties'].get('unit')}\" "
        f"category={m['tgt']['properties'].get('category')}"
        for m in metrics
    )
    return f"""/no_think

You are validating whether each metric from {company_name}'s 10-K has been correctly mapped to a standard US GAAP concept.

METRICS:
{lines}

For each metric, decide:
- KEEP   → gaap_concept is a valid, appropriate standard US GAAP line-item name for this metric_type
           (it does not have to be an exact match — company-specific labels like "Production margin",
           "Worldwide sales", "Income from operations" can map correctly to standard GAAP concepts)
- REMOVE → gaap_concept is absent, nonsensical, or clearly wrong for this metric_type AND
           no standard GAAP concept exists that could reasonably represent this metric
           (e.g. internal KPIs, non-financial operating metrics with no GAAP equivalent)

Be generous: remove only when there is genuinely no valid GAAP mapping, not when the label is non-standard.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "REMOVE", "reason": "one-sentence reason"}}
]"""


def _gaap_mapping_pass(llm_fn: Callable, company_name: str, metrics: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    gaap_ok: List[Dict] = []
    no_gaap: List[Dict] = []
    batches = [metrics[i:i + CATEGORY_BATCH] for i in range(0, len(metrics), CATEGORY_BATCH)]
    for b_num, batch in enumerate(batches, 1):
        if len(batches) > 1:
            print(f"  Batch {b_num}/{len(batches)} ...")
        dec_map = {d['idx']: d for d in _parse_json_array(llm_fn(_build_gaap_mapping_prompt(company_name, batch))) if 'idx' in d}
        for m in batch:
            idx = m['_idx']
            if m['tgt']['properties'].get('gaap_concept', '').strip().lower() == 'no gaap':
                print(f"  NO-GAAP [{idx}] \"{m['tgt']['properties'].get('metric_type', '')}\" → flagged by extraction")
                no_gaap.append(m)
            elif dec_map.get(idx, {}).get('action', 'KEEP').upper() == 'REMOVE':
                print(f"  NO-GAAP [{idx}] \"{m['tgt']['properties'].get('metric_type', '')}\" → {dec_map[idx].get('reason', '')}")
                no_gaap.append(m)
            else:
                gaap_ok.append(m)
    return gaap_ok, no_gaap


_CATEGORY_DEFINITIONS = """
- "Leverage"     : Debt, liabilities, or obligations relative to earnings/equity/assets.
                   Includes total debt, long-term debt, debt-to-equity, net debt, leverage ratios.
- "Coverage"     : Whether earnings/cash flow cover interest and debt obligations.
                   Includes interest coverage, EBITDA coverage, debt service coverage.
- "Liquidity"    : Ability to meet near-term obligations. Cash, working capital, current ratio,
                   quick ratio, operating cash flow.
- "Profitability": Revenue, gross profit, operating income, net income, EBITDA, EBIT, margins.
                   EXCLUDE shares, tax items, non-recurring charges, balance sheet items.
- "Debt Structure": Debt composition, maturity profile, cost of debt, credit ratings,
                   interest rates, covenant terms.
"""


def _build_category_validation_prompt(company_name: str, metrics: List[Dict]) -> str:
    lines = "\n".join(
        f"[{m['_idx']}] gaap_concept=\"{m['tgt']['properties'].get('gaap_concept', '')}\" "
        f"metric_type=\"{m['tgt']['properties'].get('metric_type', m['tgt']['name'])}\" "
        f"current_category={m['tgt']['properties'].get('category', '?')}"
        for m in metrics
    )
    return f"""/no_think

Validate and correct the category assignment for each metric from {company_name}'s 10-K.
Use the gaap_concept as the primary signal for correct category assignment.

CATEGORY DEFINITIONS:
{_CATEGORY_DEFINITIONS}

METRICS:
{lines}

For each metric return KEEP if the current_category is correct, or FIX with the correct category.
Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "FIX", "category": "Leverage"}}
]"""


def _category_validation_pass(llm_fn: Callable, company_name: str, metrics: List[Dict]) -> List[Dict]:
    batches = [metrics[i:i + CATEGORY_BATCH] for i in range(0, len(metrics), CATEGORY_BATCH)]
    for b_num, batch in enumerate(batches, 1):
        if len(batches) > 1:
            print(f"  Batch {b_num}/{len(batches)} ...")
        dec_map = {d['idx']: d for d in _parse_json_array(llm_fn(_build_category_validation_prompt(company_name, batch))) if 'idx' in d}
        for m in batch:
            d = dec_map.get(m['_idx'], {})
            if d.get('action', 'KEEP').upper() == 'FIX' and d.get('category'):
                old_cat = m['tgt']['properties'].get('category', '?')
                new_cat = d['category']
                if old_cat != new_cat:
                    m['tgt']['properties']['category'] = new_cat
                    print(f"  CAT-FIX [{m['_idx']}] \"{m['tgt']['properties'].get('metric_type', '')}\" {old_cat} → {new_cat}")
    return metrics


def _build_phase2a_prompt(company_name: str, category: str, metrics: List[Dict]) -> str:
    lines = "\n".join(
        f"[{m['_idx']}] metric_type=\"{m['tgt']['properties'].get('metric_type', m['tgt']['name'])}\" "
        f"gaap_concept=\"{m['tgt']['properties'].get('gaap_concept', '')}\" "
        f"= {m['tgt']['properties'].get('value')} {m['tgt']['properties'].get('unit')} "
        f"(year={m['tgt']['properties'].get('year')})"
        for m in metrics
    )
    return f"""/no_think

These are {category} metrics extracted from different sections of {company_name}'s 10-K filing.
Some may be duplicates: same metric concept, same year, same (or equivalent) numeric value, but named differently.

METRICS:
{lines}

Duplicate detection rules — apply in order:
1. STRONGEST signal: same gaap_concept + same year → almost certainly duplicates regardless of metric_type wording.
2. STRONG signal: metric_type names are close synonyms (e.g. "Net income" / "Net income attributable to [Company]") + same value + same year → duplicates.
3. WEAK signal: same value + same year but different gaap_concept → NOT duplicates unless names are also synonyms.

Examples of duplicates:
- gaap_concept="Net Income (Loss)" for both, same value → duplicates
- "Consolidated income from operations" / "Income from operations", same gaap_concept, same value → duplicates
- "Net sales" / "Total net revenues", gaap_concept="Revenue" for both, same value → duplicates

Examples that are NOT duplicates:
- Same metric_type but different years → NOT duplicates
- Same metric_type but different values (e.g. segment vs consolidated total) → NOT duplicates
- Related but distinct gaap_concepts (e.g. "Operating Income (Loss)" vs "Net Income (Loss)") → NOT duplicates

For each duplicate, MERGE into the entry whose gaap_concept is most precise and standard.
For all others, return KEEP.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}}
]"""


def _per_category_pass(llm_fn: Callable, company_name: str, metrics: List[Dict]) -> List[Dict]:
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    uncategorised: List[Dict] = []
    for m in metrics:
        cat = m['tgt']['properties'].get('category', '').strip()
        (by_category[cat] if cat else uncategorised).append(m)

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
            all_kept.extend(_apply_merge_decisions(batch, _parse_json_array(llm_fn(_build_phase2a_prompt(company_name, cat, batch)))))
        removed = len(cat_metrics) - len(all_kept)
        if removed:
            print(f"    → kept={len(all_kept)}, merged={removed}")
        surviving.extend(all_kept)
    return surviving


def _find_cross_category_candidates(metrics: List[Dict]) -> List[List[Dict]]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for m in metrics:
        p = m['tgt']['properties']
        norm_val = _normalize_value(p.get('value', ''))
        if norm_val is not None:
            groups[(p.get('year', ''), norm_val)].append(m)
    return [g for g in groups.values() if len(g) > 1 and len({m['tgt']['properties'].get('category', '') for m in g}) > 1]


def _build_phase2b_prompt(company_name: str, candidate_groups: List[List[Dict]]) -> str:
    lines = []
    for group in candidate_groups:
        lines.append("--- Group (same value+year) ---")
        for m in group:
            p = m['tgt']['properties']
            lines.append(
                f"  [{m['_idx']}] \"{p.get('metric_type', m['tgt']['name'])}\" "
                f"gaap_concept=\"{p.get('gaap_concept', '')}\" "
                f"= {p.get('value')} {p.get('unit')} "
                f"year={p.get('year')} category={p.get('category')}"
            )
    return f"""/no_think

These metrics from {company_name}'s 10-K share the same numeric value and year but were assigned different categories.
Decide whether each group contains true duplicates (same underlying metric, miscategorised) or genuinely distinct metrics that happen to share a value.

CANDIDATE GROUPS:
{chr(10).join(lines)}

For duplicates within a group: MERGE the less appropriate ones into the best-categorised entry.
For non-duplicates (same value is a coincidence): KEEP all.

Return ONLY a valid JSON array:
[
  {{"idx": 1, "action": "KEEP"}},
  {{"idx": 2, "action": "MERGE", "merge_into": 1}}
]"""


def _cross_category_pass(llm_fn: Callable, company_name: str, metrics: List[Dict]) -> List[Dict]:
    candidate_groups = _find_cross_category_candidates(metrics)
    if not candidate_groups:
        print("  No cross-category candidates found.")
        return metrics
    print(f"  {len(candidate_groups)} candidate groups ({sum(len(g) for g in candidate_groups)} metrics) ...")
    all_decisions: List[Dict] = []
    for i in range(0, len(candidate_groups), 15):
        all_decisions.extend(_parse_json_array(llm_fn(_build_phase2b_prompt(company_name, candidate_groups[i:i+15]))))
    result = _apply_merge_decisions(metrics, all_decisions)
    removed = len(metrics) - len(result)
    if removed:
        print(f"  → merged {removed} cross-category duplicates")
    return result


def validate_has_metric(extracted_json_path: str, llm_fn: Callable = None) -> Dict:
    if llm_fn is None:
        raise ValueError("An llm_fn callable must be provided.")

    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data.get('main_company', 'the Company')
    relations = data.get('relations', {}).get('HAS_METRIC', [])

    if not relations:
        print("No HAS_METRIC relations found.")
        return {'main_company': company_name, 'relations': {'HAS_METRIC': []}}

    print(f"Loaded {len(relations)} raw metrics for '{company_name}'")

    for rel in relations:
        if rel.get('src', {}).get('name', '') != company_name:
            rel['src']['name'] = company_name

    seen: set = set()
    unique: List[Dict] = []
    for rel in relations:
        key = (rel['tgt']['name'].lower().strip(), rel['tgt']['properties'].get('year', ''))
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    print(f"After exact-name dedup: {len(unique)} metrics")

    chunk_texts: Dict[str, str] = {}
    for i, rel in enumerate(unique, 1):
        rel['_idx'] = i
        chunk = rel.get('chunk_text', '')
        cid = _chunk_id(chunk)
        chunk_texts[cid] = chunk
        rel['_chunk_id'] = cid

    print(f"\n[Phase 1b] GAAP-mapping validation ({len(unique)} metrics) ...")
    surviving, no_gaap = _gaap_mapping_pass(llm_fn, company_name, unique)
    print(f"After phase 1b: {len(surviving)} valid, {len(no_gaap)} no-GAAP removed")

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for rel in surviving:
        groups[rel['_chunk_id']].append(rel)

    print(f"\n[Phase 1] Within-chunk validation ({len(groups)} chunks) ...")
    surviving = _within_chunk_pass(llm_fn, company_name, groups, chunk_texts)
    print(f"After phase 1: {len(surviving)} metrics")

    print(f"\n[Phase 1c] Category validation ({len(surviving)} metrics) ...")
    surviving = _category_validation_pass(llm_fn, company_name, surviving)

    print(f"\n[Phase 2a] Per-category deduplication ...")
    surviving = _per_category_pass(llm_fn, company_name, surviving)
    print(f"After phase 2a: {len(surviving)} metrics")

    print(f"\n[Phase 2b] Cross-category deduplication ...")
    surviving = _cross_category_pass(llm_fn, company_name, surviving)
    print(f"After phase 2b: {len(surviving)} metrics")

    gaap_best: Dict[tuple, Dict] = {}
    for rel in surviving:
        p = rel['tgt']['properties']
        gaap = p.get('gaap_concept', '').strip()
        year = p.get('year', '').strip()
        if not gaap or gaap.lower() == 'no gaap':
            continue
        key = (gaap.lower(), year)
        if key not in gaap_best:
            gaap_best[key] = rel
        else:
            gaap_tokens = set(re.split(r'\W+', gaap.lower())) - {'', 'loss', 'and', 'or', 'of', 'the'}
            existing_score = len(gaap_tokens & set(re.split(r'\W+', gaap_best[key]['tgt']['properties'].get('metric_type', '').lower())))
            new_score = len(gaap_tokens & set(re.split(r'\W+', p.get('metric_type', '').lower())))
            if new_score > existing_score:
                gaap_best[key] = rel

    winner_ids = {id(v) for v in gaap_best.values()}
    surviving = [
        rel for rel in surviving
        if not rel['tgt']['properties'].get('gaap_concept', '').strip()
        or rel['tgt']['properties'].get('gaap_concept', '').strip().lower() == 'no gaap'
        or id(rel) in winner_ids
    ]
    print(f"\nAfter gaap_concept dedup: {len(surviving)} metrics")

    final_relations = [
        {k: v for k, v in rel.items() if not k.startswith('_')}
        for rel in sorted(surviving, key=lambda r: r['_idx'])
    ]
    print(f"Summary: {len(relations)} input → {len(final_relations)} kept ({len(relations) - len(final_relations)} removed/merged), {len(no_gaap)} no-GAAP")

    return {'main_company': company_name, 'relations': {'HAS_METRIC': final_relations}}


