"""
Ground Truth Validation — Target Company Extraction Accuracy (Risks + Metrics)

Tests whether the pipeline correctly extracted information from source chunks:

  Risks   : Does the source_text actually support the extracted risk name
            and description? (extraction faithfulness)
  Metrics : Is the extracted metric name, value, and unit internally
            consistent and plausible for that metric type? (extraction accuracy)

Note: target metrics have no source_text (HTML-parsed), so metric validation
uses LLM knowledge of typical financial statement values/units.

Output CSV columns:
  {citation_id, type, extracted_name, extracted_value, extracted_unit,
   source_preview, is_correctly_extracted, explanation, notes}
"""

import csv
import json
import os
import re
import sys
import time

import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

MAX_SOURCE_CHARS   = 6000
MAX_EVALS_PER_CALL = 8

def _llm_call(bedrock_client, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return response['output']['message']['content'][0]['text'].strip()


# ---------------------------------------------------------------------------
# Collect target chunks
# ---------------------------------------------------------------------------

def collect_target_chunks(extraction_data: dict) -> list[dict]:
    chunks = []
    seen   = set()

    for row in extraction_data.get('results', []):
        target_name = row.get('target', 'Target')
        category    = row.get('category', '')

        # ── target risks ────────────────────────────────────────────────
        for r in (row.get('target_risks') or []):
            cid = r.get('citation_id', '')
            if not cid or cid in seen:
                continue
            seen.add(cid)

            source = (r.get('source_text') or '').strip()
            source_origin = 'source_text'
            # Reject TOC-like or too-short source_text and fall back to description+why
            _toc_hints = ('table of content', 'table of content', 'contents')
            if not source or len(source) < 80 or source.lower().startswith(_toc_hints):
                desc = (r.get('description') or '').strip()
                why  = (r.get('why') or '').strip()
                source = ' | '.join(filter(None, [desc, why]))
                source_origin = 'description_fallback'

            chunks.append({
                'citation_id':     cid,
                'type':            'risk',
                'company':         target_name,
                'extracted_name':  r.get('name', ''),
                'extracted_desc':  r.get('description', ''),
                'extracted_why':   r.get('why', ''),
                'extracted_value': '',
                'extracted_unit':  '',
                'source_text':     source[:MAX_SOURCE_CHARS],
                'source_length':   len(source),
                'source_origin':   source_origin,
                'category':        '',
            })

        # ── target metrics ───────────────────────────────────────────────
        for m in (row.get('target_metrics') or []):
            cid = m.get('citation_id', '')
            if not cid or cid in seen:
                continue
            seen.add(cid)

            label        = m.get('label') or m.get('metric_type') or m.get('name', '')
            xbrl_tag     = m.get('xbrl_tag') or ''
            value        = m.get('value', '')
            unit         = m.get('unit', '')
            gaap_concept = m.get('gaap_concept', '')
            # Use source_text stored as a flat property (same as risks)
            metric_source = (m.get('source_text') or '').strip()
            if not metric_source:
                section     = m.get('section_title', '')
                source_page = m.get('source_page', '')
                parts = []
                if section:
                    parts.append(f"Section: {section}")
                if source_page:
                    parts.append(f"Page: {source_page}")
                if xbrl_tag:
                    parts.append(f"XBRL tag: {xbrl_tag}")
                metric_source = " | ".join(parts) if parts else (f"XBRL-extracted: {value} {unit}".strip() if value else '')

            chunks.append({
                'citation_id':     cid,
                'type':            'metric',
                'company':         target_name,
                'extracted_name':  label,
                'extracted_desc':  '',
                'extracted_why':   '',
                'extracted_value': value,
                'extracted_unit':  unit,
                'gaap_concept':    gaap_concept,
                'source_text':     metric_source,
                'source_length':   len(metric_source),
                'source_origin':   'xbrl',
                'category':        category or m.get('metric_type', ''),
            })

    return chunks


# ---------------------------------------------------------------------------
# LLM batch judges — separate prompts for risks vs metrics
# ---------------------------------------------------------------------------

def judge_risk_batch(
    batch: list[dict],
    bedrock_client,
    model_id: str,
) -> list[tuple[bool, str]]:
    """
    For each risk: does the source_text actually support the extracted name + description?
    Groups risks by source text so the full text is shown once per chunk, not repeated.
    Falls back to internal consistency check when source_text is absent.
    """
    # Group by source_text so each unique chunk is shown once in full
    from collections import OrderedDict
    groups: OrderedDict[str, list] = OrderedDict()
    no_source: list = []
    for i, c in enumerate(batch, 1):
        c['_n'] = i
        if c['source_text']:
            groups.setdefault(c['source_text'], []).append(c)
        else:
            no_source.append(c)

    body = ""
    for src_text, chunks in groups.items():
        body += f"SOURCE TEXT:\n\"{src_text}\"\n\n"
        body += "RISKS EXTRACTED FROM THIS TEXT:\n"
        for c in chunks:
            body += (
                f"  {c['_n']}. Risk name: \"{c['extracted_name']}\"\n"
                f"     Description: \"{c['extracted_desc']}\"\n"
            )
        body += "\n"

    for c in no_source:
        body += (
            f"{c['_n']}. Risk name: \"{c['extracted_name']}\"\n"
            f"   Description: \"{c['extracted_desc']}\"\n"
            f"   Why relevant: \"{c['extracted_why']}\"\n"
            f"   (No source text — judge internal consistency only)\n\n"
        )

    prompt = (
        "You are auditing a pipeline that extracts risk factors from SEC 10-K filings.\n\n"
        "For risks with a source text: judge whether the extracted risk name and description "
        "are accurately supported by that source text.\n"
        "For risks without source text: judge whether the name, description, and 'why' "
        "are internally consistent and plausible for a 10-K risk factor.\n\n"
        "For EACH numbered item output exactly one line:\n"
        "  <N>: YES | NO — <one-sentence explanation>\n\n"
        "  YES — the extraction is accurate / internally consistent\n"
        "  NO  — the name or description contradicts or misrepresents the source\n\n"
        "Output ONLY the numbered lines — no extra text.\n\n"
        f"{body}"
    )

    max_tokens = max(300, len(batch) * 100)
    try:
        raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=max_tokens)
    except Exception as e:
        return [(False, f"LLM error: {e}")] * len(batch)

    return _parse_verdicts(raw, len(batch))


def judge_metric_batch(
    batch: list[dict],
    bedrock_client,
    model_id: str,
) -> list[tuple[bool, str]]:
    """
    For each metric: is the extracted name, value, and unit consistent with
    what that metric type should look like in a corporate 10-K filing?
    """
    body = ""
    for i, c in enumerate(batch, 1):
        gaap = c.get('gaap_concept') or c['extracted_name']
        body += (
            f"{i}. GAAP concept : \"{gaap}\"\n"
            f"   Value        : {c['extracted_value']}\n"
            f"   Unit         : {c['extracted_unit']}\n\n"
        )

    prompt = (
        "You are auditing a pipeline that extracts financial metrics from SEC 10-K annual reports.\n\n"
        "For each item, judge whether the (value, unit) PAIR is correct and plausible "
        "for the given GAAP concept. Evaluate them together, not separately:\n"
        "  - Does the unit match the GAAP concept? "
        "(e.g. monetary amounts in USD/thousands/millions, ratios as pure numbers or %, "
        "EPS in USD per share, share counts in shares)\n"
        "  - Given the unit, is the numeric value in a plausible range for a real company's annual filing? "
        "(e.g. Revenue reported as 1.2 billion USD is plausible; Revenue as 1.2 % is not)\n"
        "  - Do the value and unit together represent the same scale? "
        "(e.g. value=1500000 unit=USD is fine; value=1500000 unit=USD millions would mean $1.5 quadrillion — flag that)\n\n"
        "For EACH numbered item output exactly one line:\n"
        "  <N>: YES | NO — <one-sentence explanation citing both value and unit>\n\n"
        "  YES — the (value, unit) pair is correct and plausible for this GAAP concept\n"
        "  NO  — there is a clear error in the pair (wrong unit for this concept, implausible magnitude, scale mismatch)\n\n"
        "Output ONLY the numbered lines — no extra text.\n\n"
        f"{body}"
    )

    max_tokens = max(300, len(batch) * 100)
    try:
        raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=max_tokens)
    except Exception as e:
        return [(False, f"LLM error: {e}")] * len(batch)

    return _parse_verdicts(raw, len(batch))


def _parse_verdicts(raw: str, n: int) -> list[tuple[bool, str]]:
    results = [None] * n
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)\s*[:\.\)]\s*(YES|NO)\s*[-—–:]\s*(.*)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                results[idx] = (m.group(2).upper() == 'YES', m.group(3).strip())
    for i, r in enumerate(results):
        if r is None:
            results[i] = (False, 'Could not parse LLM response.')
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def validate_target(
    extraction_result_path: str,
    output_csv_path: str,
):
    log_path = output_csv_path.replace('.csv', '_log.txt')

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.__stdout__
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, msg):
            try:
                self.terminal.write(msg)
            except UnicodeEncodeError:
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
        _run(extraction_result_path, output_csv_path)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_path}")


def _run(extraction_result_path: str, output_csv_path: str):
    print("=" * 70)
    print("Target Company — Extraction Accuracy Validation (Risks + Metrics)")
    print("=" * 70)

    print(f"\n[1/4] Loading extraction results")
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    print(f"  Question : {extraction_data.get('question', '')[:100]}")

    print(f"\n[2/4] Collecting target chunks")
    chunks = collect_target_chunks(extraction_data)
    risks   = [c for c in chunks if c['type'] == 'risk']
    metrics = [c for c in chunks if c['type'] == 'metric']
    print(f"  Total : {len(chunks)}  ({len(risks)} risks, {len(metrics)} metrics)")

    if not chunks:
        print("\n✗ No target chunks found — aborting")
        return

    print(f"\n[3/4] Initialising LLM (AWS Bedrock)")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL_EVAL", "us.meta.llama3-3-70b-instruct-v1:0")
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
    )
    print(f"  Region: {aws_region}  |  Model: {model_id}")

    print(f"\n[4/4] Judging extraction accuracy")
    verdict_map: dict[str, tuple[bool, str]] = {}

    # ── risks ─────────────────────────────────────────────────────────────
    if risks:
        n_batches = (len(risks) + MAX_EVALS_PER_CALL - 1) // MAX_EVALS_PER_CALL
        print(f"  Risks — {len(risks)} items in {n_batches} batch(es)")
        for i in range(0, len(risks), MAX_EVALS_PER_CALL):
            batch = risks[i:i + MAX_EVALS_PER_CALL]
            bn    = i // MAX_EVALS_PER_CALL + 1
            print(f"    Batch [{bn}/{n_batches}] ...")
            verdicts = judge_risk_batch(batch, bedrock_client, model_id)
            for c, v in zip(batch, verdicts):
                verdict_map[c['citation_id']] = v
            time.sleep(0.4)

    # ── metrics ───────────────────────────────────────────────────────────
    if metrics:
        n_batches = (len(metrics) + MAX_EVALS_PER_CALL - 1) // MAX_EVALS_PER_CALL
        print(f"  Metrics — {len(metrics)} items in {n_batches} batch(es)")
        for i in range(0, len(metrics), MAX_EVALS_PER_CALL):
            batch = metrics[i:i + MAX_EVALS_PER_CALL]
            bn    = i // MAX_EVALS_PER_CALL + 1
            print(f"    Batch [{bn}/{n_batches}] ...")
            verdicts = judge_metric_batch(batch, bedrock_client, model_id)
            for c, v in zip(batch, verdicts):
                verdict_map[c['citation_id']] = v
            time.sleep(0.4)

    # ── Build results table ───────────────────────────────────────────────
    rows = []
    for c in chunks:
        is_correct, expl = verdict_map.get(c['citation_id'], (False, 'Not evaluated'))
        rows.append({
            'citation_id':          c['citation_id'],
            'type':                 c['type'],
            'extracted_name':       c['extracted_name'],
            'extracted_value':      c['extracted_value'],
            'extracted_unit':       c['extracted_unit'],
            'source_preview':       c['source_text'][:150] if c['source_text'] else '(no source text)',
            'source_text':          c['source_text'] if c['source_text'] else '(no source text)',
            'is_correctly_extracted': is_correct,
            'explanation':          expl,
            'notes': {
                'xbrl':                'XBRL-extracted — no filing source_text',
                'description_fallback': 'source_text absent — used description/why',
                'source_text':          '',
            }.get(c.get('source_origin', ''), ''),
        })

    # ── Summary ───────────────────────────────────────────────────────────
    total       = len(rows)
    correct     = sum(1 for r in rows if r['is_correctly_extracted'] is True)
    risk_ok     = sum(1 for r in rows if r['type'] == 'risk'   and r['is_correctly_extracted'] is True)
    metric_ok   = sum(1 for r in rows if r['type'] == 'metric' and r['is_correctly_extracted'] is True)
    accuracy    = correct / total if total else 0.0

    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    print(f"  Total             : {total}")
    print(f"  Correctly extracted: {correct}  ({accuracy*100:.1f}%)")
    print(f"  Risk accuracy     : {risk_ok}/{len(risks)}")
    print(f"  Metric accuracy   : {metric_ok}/{len(metrics)}")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        'citation_id', 'type', 'extracted_name', 'extracted_value', 'extracted_unit',
        'source_preview', 'source_text', 'is_correctly_extracted', 'explanation', 'notes',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")
    print("=" * 70)
    print(f"Extraction Accuracy : {accuracy:.2f}  (risks {risk_ok}/{len(risks)}, metrics {metric_ok}/{len(metrics)})")
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

    validate_target(
        extraction_result_path=ext_path,
        output_csv_path=os.path.join(out_dir, 'target_validation_results.csv'),
    )
