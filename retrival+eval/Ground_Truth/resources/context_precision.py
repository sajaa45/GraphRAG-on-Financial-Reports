"""
Ground Truth Evaluation — Context Precision (RAGAS)

For every chunk retrieved by the pipeline (risk or metric), judges whether
it is relevant to the original question — not to its own label or category.

Score = relevant_chunks / total_chunks

This answers: "Did the pipeline fetch the right data for this question?"

Input files (from retrival_results/):
  - extraction_result.json  — question + all retrieved risks and metrics

Output:
  - context_precision_results.csv
"""

import csv
import json
import os
import re
import sys
import time

import boto3
from dotenv import load_dotenv

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

# Max items per LLM call — keeps prompts manageable
MAX_EVALS_PER_CALL = 10


def _llm_call(bedrock_client, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    text = response['output']['message']['content'][0]['text'].strip()
    return _THINK_RE.sub('', text).strip()


# ---------------------------------------------------------------------------
# Build flat list of chunks from extraction_result
# ---------------------------------------------------------------------------

def collect_chunks(extraction_data: dict) -> list[dict]:
    """
    Flatten all risks and metrics out of the nested results rows into a
    uniform list. Keeps only the fields needed for relevance judgement
    (no source_text — too large for batch prompts).
    """
    chunks = []
    seen_citations = set()
    seen_gaap = set()  # (company, gaap_concept) — dedup same concept from multiple XBRL tags

    for row in extraction_data.get('results', []):
        target_name = row.get('target', 'Target')
        peer_name   = row.get('peer', '')
        category    = row.get('category', '')

        # ── target risks ────────────────────────────────────────────────
        for r in (row.get('target_risks') or []):
            cid = r.get('citation_id', '')
            if not cid or cid in seen_citations:
                continue
            seen_citations.add(cid)
            chunks.append({
                'citation_id': cid,
                'type':        'risk',
                'role':        'target',
                'company':     target_name,
                'summary':     f"{r.get('name', '')} — {(r.get('description') or r.get('why') or '')[:200]}",
            })

        # ── peer risks ───────────────────────────────────────────────────
        for r in (row.get('peer_risks') or []):
            cid = r.get('citation_id', '')
            if not cid or cid in seen_citations:
                continue
            seen_citations.add(cid)
            chunks.append({
                'citation_id': cid,
                'type':        'risk',
                'role':        'peer',
                'company':     peer_name,
                'summary':     f"{r.get('name', '')} — {(r.get('description') or r.get('why') or '')[:200]}",
            })

        # ── target metrics ───────────────────────────────────────────────
        for m in (row.get('target_metrics') or []):
            cid = m.get('citation_id', '')
            if not cid or cid in seen_citations:
                continue
            label = m.get('gaap_concept') or m.get('label') or m.get('name', '')
            gaap_key = (target_name, label)
            if gaap_key in seen_gaap:
                continue
            seen_citations.add(cid)
            seen_gaap.add(gaap_key)
            chunks.append({
                'citation_id': cid,
                'type':        'metric',
                'role':        'target',
                'company':     target_name,
                'summary':     f"[{category or 'Metric'}] {label} = {m.get('value', '')} {m.get('unit', '')} ({m.get('year', '')})",
            })

        # ── peer metrics ─────────────────────────────────────────────────
        for m in (row.get('peer_metrics') or []):
            cid = m.get('citation_id', '')
            if not cid or cid in seen_citations:
                continue
            label = m.get('gaap_concept') or m.get('label') or m.get('name', '')
            gaap_key = (peer_name, label)
            if gaap_key in seen_gaap:
                continue
            seen_citations.add(cid)
            seen_gaap.add(gaap_key)
            chunks.append({
                'citation_id': cid,
                'type':        'metric',
                'role':        'peer',
                'company':     peer_name,
                'summary':     f"[{category or 'Metric'}] {label} = {m.get('value', '')} {m.get('unit', '')} ({m.get('year', '')})",
            })

    return chunks


# ---------------------------------------------------------------------------
# LLM batch judge
# ---------------------------------------------------------------------------

def judge_batch(
    question: str,
    batch: list[dict],   # list of chunk dicts
    bedrock_client,
    model_id: str,
) -> list[tuple[bool, str]]:
    """
    Ask the LLM whether each chunk in the batch is relevant to the question.
    Returns [(is_relevant, explanation), ...] parallel to batch.
    """
    # Build company context so the LLM knows which names are target vs peers
    target_names = sorted({c['company'] for c in batch if c['role'] == 'target' and c['company']})
    peer_names   = sorted({c['company'] for c in batch if c['role'] == 'peer'   and c['company']})
    company_context = ""
    if target_names:
        company_context += f"TARGET company (the one being analyzed): {', '.join(target_names)}\n"
    if peer_names:
        company_context += f"PEER companies (competitors): {', '.join(peer_names)}\n"
    if company_context:
        company_context = "COMPANY ROLES:\n" + company_context + "\n"

    items_block = ""
    for i, chunk in enumerate(batch, 1):
        items_block += (
            f"{i}. [{chunk['citation_id']}] {chunk['company']} ({chunk['role']}, {chunk['type']})\n"
            f"   {chunk['summary']}\n\n"
        )

    prompt = (
        "You are evaluating whether retrieved data chunks are relevant to a credit-risk question.\n\n"
        f"{company_context}"
        f"QUESTION: {question}\n\n"
        "RELEVANCE RULES:\n"
        "- TARGET chunks (role=target): mark YES if a credit analyst would use this data\n"
        "  when answering the question — including direct answers, cost items, tax expense,\n"
        "  balance sheet items, and prior-year comparisons. When in doubt, mark YES.\n"
        "- PEER chunks (role=peer): mark YES if this data helps COMPARE the peer against\n"
        "  the target for the question asked. Peer risks and metrics are always relevant\n"
        "  context for comparative credit analysis, even if they are not about the target.\n"
        "  Only mark NO if the chunk is completely unrelated to the question domain.\n\n"
        "DATA CHUNKS:\n"
        f"{items_block}"
        "For each numbered chunk output exactly one line:\n"
        "  <N>: YES | NO — <one-sentence explanation>\n\n"
        "Output ONLY the numbered lines — no extra text."
    )

    max_tokens = max(500, len(batch) * 120)
    try:
        raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=max_tokens)
    except Exception as e:
        return [(False, f"LLM error: {e}")] * len(batch)

    results = [None] * len(batch)
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)\s*[:\.\)]\s*(YES|NO)\s*[-—–:]\s*(.*)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(batch):
                results[idx] = (m.group(2).upper() == 'YES', m.group(3).strip())

    # Fill unparsed
    for i, r in enumerate(results):
        if r is None:
            results[i] = (False, 'Could not parse LLM response.')

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def evaluate_context_precision(
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
    print("Context Precision — RAGAS-style Ground Truth Evaluation")
    print("=" * 70)

    # ── Load inputs ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading extraction results")
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)

    question = extraction_data.get('question', '')
    if not question:
        print("✗ No question found in extraction_result.json — aborting")
        return
    print(f"  Question : {question[:120]}")

    # ── Collect chunks ────────────────────────────────────────────────────
    print(f"\n[2/4] Collecting retrieved chunks")
    chunks = collect_chunks(extraction_data)

    risk_count   = sum(1 for c in chunks if c['type'] == 'risk')
    metric_count = sum(1 for c in chunks if c['type'] == 'metric')
    print(f"  Total chunks : {len(chunks)}  ({risk_count} risks, {metric_count} metrics)")
    for c in chunks[:4]:
        print(f"    [{c['citation_id']}] {c['company']} ({c['role']}) — {c['summary'][:70]}")
    if len(chunks) > 4:
        print(f"    ... and {len(chunks) - 4} more")

    if not chunks:
        print("\n✗ No chunks found — aborting")
        return

    # ── Init Bedrock ──────────────────────────────────────────────────────
    print(f"\n[3/4] Initialising LLM (AWS Bedrock)")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL", "us.meta.llama3-2-90b-instruct-v1:0")
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
    )
    print(f"  Region: {aws_region}  |  Model: {model_id}")

    # ── Judge relevance in batches ────────────────────────────────────────
    n_batches = (len(chunks) + MAX_EVALS_PER_CALL - 1) // MAX_EVALS_PER_CALL
    print(f"\n[4/4] Judging relevance — {len(chunks)} chunks in {n_batches} batch(es)")

    verdicts: list[tuple[bool, str]] = []
    for i in range(0, len(chunks), MAX_EVALS_PER_CALL):
        batch = chunks[i:i + MAX_EVALS_PER_CALL]
        batch_num = i // MAX_EVALS_PER_CALL + 1
        print(f"  Batch [{batch_num}/{n_batches}] — {len(batch)} chunks ...")
        batch_verdicts = judge_batch(question, batch, bedrock_client, model_id)
        verdicts.extend(batch_verdicts)
        parsed = sum(1 for _, e in batch_verdicts if 'Could not parse' not in e and 'LLM error' not in e)
        print(f"    → parsed {parsed}/{len(batch)}")
        time.sleep(0.4)

    # ── Metrics ───────────────────────────────────────────────────────────
    total     = len(chunks)
    relevant  = sum(1 for v, _ in verdicts if v)
    precision = relevant / total if total else 0.0

    # Break down by type
    risk_rel    = sum(1 for c, (v, _) in zip(chunks, verdicts) if c['type'] == 'risk'   and v)
    metric_rel  = sum(1 for c, (v, _) in zip(chunks, verdicts) if c['type'] == 'metric' and v)

    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    print(f"  Total chunks     : {total}")
    print(f"  Relevant         : {relevant}  ({precision*100:.1f}%)")
    print(f"  Irrelevant       : {total - relevant}")
    print(f"  Risk precision   : {risk_rel}/{risk_count}")
    print(f"  Metric precision : {metric_rel}/{metric_count}")

    quality = "PASS" if precision >= 0.7 else ("WARNING" if precision >= 0.5 else "FAIL")
    print(f"  Quality gate     : {quality}  (threshold: 0.70)")

    # ── Write CSV ─────────────────────────────────────────────────────────
    rows = []
    for chunk, (is_relevant, explanation) in zip(chunks, verdicts):
        rows.append({
            'citation_id':  chunk['citation_id'],
            'type':         chunk['type'],
            'role':         chunk['role'],
            'company':      chunk['company'],
            'summary':      chunk['summary'],
            'is_relevant':  is_relevant,
            'explanation':  explanation,
        })

    # Summary row
    rows.append({
        'citation_id':  '** SUMMARY **',
        'type':         '',
        'role':         '',
        'company':      '',
        'summary':      f"relevant={relevant}/{total}  risks={risk_rel}/{risk_count}  metrics={metric_rel}/{metric_count}",
        'is_relevant':  precision,
        'explanation':  quality,
    })

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = ['citation_id', 'type', 'role', 'company', 'summary', 'is_relevant', 'explanation']
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")
    print("=" * 70)
    print(f"Context Precision Score : {precision:.2f}  [{quality}]")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    retrival_dir = os.path.join(script_dir, '..', '..', 'retrival_results')

    evaluate_context_precision(
        extraction_result_path=os.path.join(retrival_dir, 'extraction_result.json'),
        output_csv_path=os.path.join(script_dir, '..', 'context_precision_results.csv'),
    )
