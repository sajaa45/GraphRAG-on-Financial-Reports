"""
Ground Truth Evaluation — Answer Relevancy (RAGAS)

True RAGAS answer relevancy algorithm:
  1. Generate N questions from the answer using an LLM.
  2. Check whether the answer is non-committal ("I don't know", refusal, etc.).
     If yes → score = 0.0.
  3. Embed the original question and all N generated questions.
  4. Compute cosine similarity between the original and each generated question.
  5. answer_relevancy = mean(cosine_similarities)

Reference: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023).

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
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

N_QUESTIONS = 5

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model



def _llm_call(bedrock_client, model_id: str, prompt: str, max_tokens: int = 1024) -> str:
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return response['output']['message']['content'][0]['text'].strip()


# ---------------------------------------------------------------------------
# Step 1 — Generate N questions from the answer
# ---------------------------------------------------------------------------

def generate_questions(answer: str, n: int, bedrock_client, model_id: str) -> list[str]:
    """
    Ask the LLM to produce N questions that the given answer could plausibly
    be responding to. This is the reverse-generation step at the heart of RAGAS.
    """
    prompt = (
        "You are evaluating an AI-generated answer from a financial analysis pipeline.\n\n"
        "PIPELINE CONTEXT:\n"
        "- The answer was generated from SEC 10-K filings of one target company and up to 3 peer companies.\n"
        "- It covers financial metrics, risk factors, and peer comparisons drawn solely from those filings.\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"Generate exactly {n} distinct questions that capture the CENTRAL INTENT of this answer.\n\n"
        "Rules:\n"
        "- Focus on the MAIN TOPIC of the answer, "
        "not on specific supporting details, individual metrics, or side facts mentioned in passing.\n"
        "- Each question must be fully answerable from the information present in the answer above.\n"
        "- Questions should vary in phrasing and angle — do not repeat the same question differently.\n"
        "- Do NOT generate questions about narrow accounting details, single data points, or "
        "background facts that are only mentioned to support the main argument.\n"
        f"- Output ONLY {n} numbered lines: '1. ...'  '2. ...'  etc. No extra text."
    )
    raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=768)
    questions = []
    for line in raw.splitlines():
        m = re.match(r'^\d+\.\s+(.+)', line.strip())
        if m:
            questions.append(m.group(1).strip())
    return questions[:n]


# ---------------------------------------------------------------------------
# Step 2 — Non-committal check
# ---------------------------------------------------------------------------

def is_noncommittal(answer: str, bedrock_client, model_id: str) -> bool:
    """
    Returns True if the answer refuses to answer, says it doesn't know,
    or otherwise fails to provide substantive information.
    RAGAS sets the relevancy score to 0 for non-committal answers.
    """
    prompt = (
        "Does the following answer refuse to answer, express that it doesn't know, "
        "or fail to provide any substantive information?\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with exactly one word: YES or NO."
    )
    raw = _llm_call(bedrock_client, model_id, prompt, max_tokens=10).upper()
    return raw.startswith('YES')


# ---------------------------------------------------------------------------
# Step 3 — Embeddings via sentence-transformers (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Embed text using all-MiniLM-L6-v2 — the model RAGAS was calibrated on."""
    return _get_embed_model().encode(text, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Step 4 — Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


# ---------------------------------------------------------------------------
# Orchestration
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
    print("Answer Relevancy — True RAGAS Evaluation")
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

    # ── Init Bedrock + embedding model ───────────────────────────────────
    print(f"\n[2/5] Initialising LLM (AWS Bedrock) + embedding model (local)")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL_EVAL", "us.meta.llama3-3-70b-instruct-v1:0")
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region,
    )
    print(f"  Region      : {aws_region}")
    print(f"  LLM model   : {model_id}")
    print(f"  Embed model : {EMBED_MODEL_NAME}  (local, sentence-transformers)")
    _get_embed_model()  # warm up / download on first run

    # ── Non-committal check ───────────────────────────────────────────────
    print(f"\n[3/5] Checking for non-committal answer")
    noncommittal = is_noncommittal(answer_text, bedrock_client, model_id)
    print(f"  Non-committal: {noncommittal}")
    if noncommittal:
        score = 0.0
        print("  → Answer is non-committal; relevancy score = 0.0")
        _write_csv(output_csv_path, question, [], [], score, noncommittal)
        _print_summary(score, [], noncommittal)
        return

    # ── Generate questions from answer ────────────────────────────────────
    print(f"\n[4/5] Generating {N_QUESTIONS} questions from the answer")
    time.sleep(0.3)
    generated_qs = generate_questions(answer_text, N_QUESTIONS, bedrock_client, model_id)

    if not generated_qs:
        print("  ✗ No questions generated — aborting")
        return

    print(f"  Generated {len(generated_qs)} question(s):")
    for i, q in enumerate(generated_qs, 1):
        print(f"    {i}. {q}")

    # ── Embed and compute similarities ────────────────────────────────────
    print(f"\n[5/5] Embedding questions and computing cosine similarities")

    # Extract company name from extraction data to normalize generated questions.
    # The stored question uses "the target company" (normalized), but the answer
    # still contains the real company name, so reverse-generated questions do too.
    # Replacing the real name with "the target company" removes this spurious
    # vocabulary gap before embedding.
    target_name = extraction_data.get('results', [{}])[0].get('target', '') if extraction_data.get('results') else ''

    def _normalize_q(text: str) -> str:
        if target_name:
            return re.sub(re.escape(target_name), 'the target company', text, flags=re.IGNORECASE)
        return text

    orig_emb = get_embedding(question)
    similarities = []
    for i, gq in enumerate(generated_qs, 1):
        gen_emb = get_embedding(_normalize_q(gq))
        sim = cosine_similarity(orig_emb, gen_emb)
        similarities.append(sim)
        print(f"    Q{i} similarity: {sim:.4f}  |  {gq[:70]}")

    top3 = sorted(similarities, reverse=True)[:3]
    score = float(np.mean(top3))

    _write_csv(output_csv_path, question, generated_qs, similarities, score, noncommittal)
    _print_summary(score, similarities, noncommittal)


def _print_summary(score: float, similarities: list[float], noncommittal: bool):
    quality = "PASS" if score >= 0.7 else ("WARNING" if score >= 0.5 else "FAIL")
    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    if noncommittal:
        print(f"  Non-committal answer → score forced to 0.0")
    elif similarities:
        print(f"  Individual similarities : {[round(s, 4) for s in similarities]}")
        print(f"  Mean similarity         : {score:.4f}")
    print(f"  Answer Relevancy Score  : {score:.4f}")
    print(f"  Quality gate            : {quality}  (threshold: 0.70)")
    print("=" * 70)
    print(f"Answer Relevancy Score : {score:.4f}  [{quality}]")
    print("=" * 70)


def _write_csv(
    output_csv_path: str,
    question: str,
    generated_qs: list[str],
    similarities: list[float],
    final_score: float,
    noncommittal: bool,
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    quality = "PASS" if final_score >= 0.7 else ("WARNING" if final_score >= 0.5 else "FAIL")
    rows = []

    if noncommittal:
        rows.append({
            'question':              question,
            'generated_question':    '',
            'cosine_similarity':     '',
            'noncommittal':          True,
            'answer_relevancy_score': 0.0,
            'quality_gate':          'FAIL',
        })
    else:
        for gq, sim in zip(generated_qs, similarities):
            rows.append({
                'question':              question,
                'generated_question':    gq,
                'cosine_similarity':     round(sim, 6),
                'noncommittal':          False,
                'answer_relevancy_score': '',
                'quality_gate':          '',
            })
        # Summary row
        rows.append({
            'question':              question,
            'generated_question':    '** TOP-3 MEAN **',
            'cosine_similarity':     '',
            'noncommittal':          False,
            'answer_relevancy_score': round(final_score, 6),
            'quality_gate':          quality,
        })

    fieldnames = [
        'question', 'generated_question', 'cosine_similarity',
        'noncommittal', 'answer_relevancy_score', 'quality_gate',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")


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

    evaluate_relevancy(
        extraction_result_path=ext_path,
        answer_path=ans_path,
        output_csv_path=os.path.join(out_dir, 'answer_relevancy_results.csv'),
    )
