"""
Ground Truth Evaluation — Answer Relevancy (RAGAS)

Measures whether the generated answer actually addresses the question asked.

Approach (RAGAS-style):
  1. Decompose the question into N specific aspects/sub-questions it requires.
  2. For each aspect, judge whether the answer covers it (YES/PARTIAL/NO).
  3. answer_relevancy = fully_covered / total_aspects
  4. Also produce an overall LLM-as-judge relevancy score (0.0–1.0).

Input files (from retrival_results/):
  - extraction_result.json  — contains the original question
  - answer.txt              — the narrative answer produced by the pipeline

Output:
  - answer_relevancy_results.csv
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


def _llm_call(bedrock_client, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    text = response['output']['message']['content'][0]['text'].strip()
    return _THINK_RE.sub('', text).strip()


# ---------------------------------------------------------------------------
# Step 1 — Decompose the question into evaluable aspects
# ---------------------------------------------------------------------------

def decompose_question(question: str, bedrock_client, model_id: str) -> list[str]:
    """
    Ask the LLM to break the question into specific, independently-verifiable
    aspects. Each aspect is a yes/no question the answer should address.
    """
    prompt = (
        "You are evaluating whether an AI-generated answer addresses a credit-risk question.\n\n"
        "PIPELINE CONTEXT:\n"
        "- Data source: ONE target company's SEC 10-K annual report + up to 3 peer companies' 10-K filings.\n"
        "- What the pipeline can do: retrieve financial metrics (profitability, leverage, liquidity, coverage, "
        "debt structure) and risk factor text from those filings, then compare target vs peers.\n"
        "- What it CANNOT do: access broad market data, industry-wide benchmarks, analyst reports, "
        "pricing data, cost breakdowns, strategic advice, forecasts, or any source outside those filings.\n\n"
        f"QUESTION: {question}\n\n"
        "Break this question into aspects focused on the INTENT of the question, not on specific metric names.\n"
        "The answer may use any relevant profitability/risk metric available in the filings — "
        "do not demand a particular metric (e.g. ROE, revenues) unless the question explicitly names it.\n"
        "Every aspect must be answerable solely from the 10-K filings of 1 target + up to 3 peers.\n"
        "Do NOT generate aspects that require market data, industry benchmarks, strategic recommendations, "
        "forecasts, cost-structure analysis, or anything beyond what annual filings contain.\n\n"
        "Rules:\n"
        "- Output between 3 and 5 aspects — keep it tight.\n"
        "- Each aspect must be answerable with YES, PARTIAL, or NO by reading the answer.\n"
        "- Phrase aspects as intent checks, e.g. 'Does the answer report at least one profitability metric "
        "for the target company?' not 'Does the answer report the exact ROE?'\n"
        "- Start each line with a number and a period: '1. ...'\n"
        "- Output ONLY the numbered list — no intro, no commentary.\n\n"
        "Aspects:"
    )
    raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=512)
    aspects = []
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.+)', line)
        if m:
            aspects.append(m.group(1).strip())
    return aspects if aspects else [question]


# ---------------------------------------------------------------------------
# Step 2 — Judge each aspect against the answer
# ---------------------------------------------------------------------------

def judge_aspects(
    question: str,
    answer: str,
    aspects: list[str],
    bedrock_client,
    model_id: str,
) -> list[dict]:
    """
    For each aspect, ask the LLM: does the answer cover it?
    Returns a list of {aspect, verdict, explanation}.
    """
    aspect_block = "\n".join(f"{i+1}. {a}" for i, a in enumerate(aspects))
    prompt = (
        "You are evaluating whether an AI-generated answer covers specific aspects of a question.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"ASPECTS TO CHECK:\n{aspect_block}\n\n"
        "For each numbered aspect, output exactly one line:\n"
        "  <N>: YES | PARTIAL | NO — <one-sentence explanation>\n\n"
        "Verdicts:\n"
        "  YES     — the answer clearly and directly addresses this aspect\n"
        "  PARTIAL — the answer touches on it but incompletely or vaguely\n"
        "  NO      — the answer does not address this aspect at all\n\n"
        "Output ONLY the numbered lines — no extra text."
    )
    raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=max(512, len(aspects) * 120))

    results = [None] * len(aspects)
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)\s*[:\.\)]\s*(YES|PARTIAL|NO)\s*[-—–:]\s*(.*)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(aspects):
                results[idx] = {
                    'aspect':      aspects[idx],
                    'verdict':     m.group(2).upper(),
                    'explanation': m.group(3).strip(),
                }

    # Fill any unparsed slots with defaults
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                'aspect':      aspects[i],
                'verdict':     'NO',
                'explanation': 'Could not parse LLM response for this aspect.',
            }
    return results


# ---------------------------------------------------------------------------
# Step 3 — Overall relevancy score
# ---------------------------------------------------------------------------

def overall_relevancy_score(
    question: str,
    answer: str,
    bedrock_client,
    model_id: str,
) -> tuple[float, str]:
    """
    Ask the LLM for a holistic relevancy score (0.0–1.0) and a one-sentence verdict.
    """
    prompt = (
        "You are a credit-risk analyst evaluating the quality of an AI-generated answer.\n\n"
        "PIPELINE CONTEXT:\n"
        "- Data source: ONE target company's SEC 10-K annual report + up to 3 peer companies' 10-K filings.\n"
        "- The pipeline retrieves financial metrics and risk factors from those filings and compares target vs peers.\n"
        "- It has NO access to broad market data, industry benchmarks, pricing data, or anything outside those filings.\n"
        "- Do NOT penalise the answer for lacking market-wide context, strategic advice, or forecasts.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Rate how well the answer addresses the question given the pipeline's data scope "
        "(1 target + up to 3 peers from 10-K filings only).\n\n"
        "  1.0 — fully addresses the question with specific, relevant information\n"
        "  0.5 — partially addresses it (misses key aspects or is too generic)\n"
        "  0.0 — does not address the question at all\n\n"
        "Output EXACTLY two lines:\n"
        "SCORE: <number between 0.0 and 1.0>\n"
        "REASON: <one sentence>\n"
    )
    raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=200)
    score = 0.0
    reason = ''
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith('SCORE:'):
            try:
                score = float(line.split(':', 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif line.upper().startswith('REASON:'):
            reason = line.split(':', 1)[1].strip()
    return score, reason


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def evaluate_relevancy(
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
        _run(extraction_result_path, answer_path, output_csv_path)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_path}")


def _run(extraction_result_path: str, answer_path: str, output_csv_path: str):
    print("=" * 70)
    print("Answer Relevancy — RAGAS-style Ground Truth Evaluation")
    print("=" * 70)

    # ── Load inputs ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading inputs")
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    with open(answer_path, 'r', encoding='utf-8') as f:
        answer_text = f.read().strip()

    question = extraction_data.get('question', '')
    if not question:
        print("✗ No question found in extraction_result.json — aborting")
        return

    print(f"  Question    : {question[:120]}")
    print(f"  Answer chars: {len(answer_text)}")

    # ── Init Bedrock ──────────────────────────────────────────────────────
    print(f"\n[2/5] Initialising LLM (AWS Bedrock)")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL", "us.meta.llama3-2-90b-instruct-v1:0")
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
    )
    print(f"  Region: {aws_region}  |  Model: {model_id}")

    # ── Decompose question ────────────────────────────────────────────────
    print(f"\n[3/5] Decomposing question into verifiable aspects")
    aspects = decompose_question(question, bedrock_client, model_id)
    print(f"  → {len(aspects)} aspects identified:")
    for i, a in enumerate(aspects, 1):
        print(f"    {i}. {a}")

    # ── Judge aspects ─────────────────────────────────────────────────────
    print(f"\n[4/5] Judging answer coverage per aspect")
    time.sleep(0.3)
    aspect_results = judge_aspects(question, answer_text, aspects, bedrock_client, model_id)

    yes_count     = sum(1 for r in aspect_results if r['verdict'] == 'YES')
    partial_count = sum(1 for r in aspect_results if r['verdict'] == 'PARTIAL')
    no_count      = sum(1 for r in aspect_results if r['verdict'] == 'NO')
    total         = len(aspect_results)

    for r in aspect_results:
        symbol = {'YES': '✓', 'PARTIAL': '~', 'NO': '✗'}.get(r['verdict'], '?')
        print(f"    [{symbol}] {r['verdict']:<7}  {r['aspect'][:70]}")
        if r['explanation']:
            print(f"             {r['explanation'][:90]}")

    # Aspect-level score: YES = 1.0, PARTIAL = 0.5, NO = 0.0
    aspect_score = (yes_count + 0.5 * partial_count) / total if total else 0.0

    # ── Overall score ─────────────────────────────────────────────────────
    print(f"\n[5/5] Computing overall relevancy score")
    time.sleep(0.3)
    overall_score, overall_reason = overall_relevancy_score(
        question, answer_text, bedrock_client, model_id
    )
    print(f"  Overall score : {overall_score:.2f}")
    print(f"  Reason        : {overall_reason}")

    # Combined relevancy = average of aspect-level and holistic scores
    combined = (aspect_score + overall_score) / 2

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    print(f"  Aspects total   : {total}")
    print(f"  Fully covered   : {yes_count}  (YES)")
    print(f"  Partially covered: {partial_count}  (PARTIAL)")
    print(f"  Not covered     : {no_count}  (NO)")
    print(f"  Aspect score    : {aspect_score:.2f}")
    print(f"  Holistic score  : {overall_score:.2f}")
    print(f"  Combined        : {combined:.2f}")

    quality = "PASS" if combined >= 0.7 else ("WARNING" if combined >= 0.5 else "FAIL")
    print(f"  Quality gate    : {quality}  (threshold: 0.70)")

    # ── Write CSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    rows = []

    # One row per aspect
    for r in aspect_results:
        rows.append({
            'question':       question,
            'aspect':         r['aspect'],
            'verdict':        r['verdict'],
            'explanation':    r['explanation'],
            'aspect_score':   {'YES': 1.0, 'PARTIAL': 0.5, 'NO': 0.0}.get(r['verdict'], 0.0),
            'overall_score':  '',
            'combined_score': '',
            'quality_gate':   '',
        })

    # Summary row
    rows.append({
        'question':       question,
        'aspect':         '** SUMMARY **',
        'verdict':        f"YES={yes_count} PARTIAL={partial_count} NO={no_count}",
        'explanation':    overall_reason,
        'aspect_score':   round(aspect_score, 4),
        'overall_score':  round(overall_score, 4),
        'combined_score': round(combined, 4),
        'quality_gate':   quality,
    })

    fieldnames = [
        'question', 'aspect', 'verdict', 'explanation',
        'aspect_score', 'overall_score', 'combined_score', 'quality_gate',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")
    print("=" * 70)
    print(f"Answer Relevancy Score : {combined:.2f}  [{quality}]")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    retrival_dir = os.path.join(script_dir, '..', '..', 'retrival_results')

    evaluate_relevancy(
        extraction_result_path=os.path.join(retrival_dir, 'extraction_result.json'),
        answer_path=os.path.join(retrival_dir, 'answer.txt'),
        output_csv_path=os.path.join(script_dir, '..', 'answer_relevancy_results.csv'),
    )
