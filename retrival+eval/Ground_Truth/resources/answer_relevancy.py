"""
Ground Truth Evaluation: Adapted Answer Relevancy (RAGAS):
True RAGAS answer relevancy algorithm:
  1. Generate 5 questions from the answer using an LLM.
  2. Check whether the answer is non-committal ("I don't know", refusal, etc.).
     If yes → score = 0.0.
  3. Embed the original question and all N generated questions.
  4. Compute cosine similarity between the original and each generated question.
  5. answer_relevancy = mean(cosine_similarities)
"""
import csv
import json
import os
import re
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

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



def get_embedding(text: str) -> np.ndarray:
    """Embed text using all-MiniLM-L6-v2 — the model RAGAS was calibrated on."""
    return _get_embed_model().encode(text, convert_to_numpy=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)

def evaluate_relevancy(
    extraction_result_path: str,
    answer_path: str,
    output_csv_path: str,
    bedrock_client,
    model_id: str,
):
    log_path = output_csv_path.replace('.csv', '_log.txt')

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.__stdout__
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, msg):
            self.terminal.write(msg)
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
        _run(extraction_result_path, answer_path, output_csv_path, bedrock_client, model_id)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_path}")


def _run(extraction_result_path: str, answer_path: str, output_csv_path: str, bedrock_client, model_id: str):
    print("=" * 70)
    print("Answer Relevancy — True RAGAS Evaluation")
    print("=" * 70)

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

    print(f"\n[2/5] Initialising embedding model (local)")
    print(f"  LLM model   : {model_id}")
    print(f"  Embed model : {EMBED_MODEL_NAME}  (local, sentence-transformers)")
    _get_embed_model() 

    print(f"\n[3/4] Generating {N_QUESTIONS} questions from the answer")
    time.sleep(0.3)
    generated_qs = generate_questions(answer_text, N_QUESTIONS, bedrock_client, model_id)

    if not generated_qs:
        print("  ✗ No questions generated — aborting")
        return

    print(f"  Generated {len(generated_qs)} question(s):")
    for i, q in enumerate(generated_qs, 1):
        print(f"    {i}. {q}")

    print(f"\n[4/4] Embedding questions and computing cosine similarities")
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

    _write_csv(output_csv_path, question, generated_qs, similarities, score)
    _print_summary(score, similarities)


def _print_summary(score: float, similarities: list[float]):
    quality = "PASS" if score >= 0.7 else ("WARNING" if score >= 0.5 else "FAIL")
    print(f"\n{'─'*70}")
    print(f"Results")
    print(f"{'─'*70}")
    if similarities:
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
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    quality = "PASS" if final_score >= 0.7 else ("WARNING" if final_score >= 0.5 else "FAIL")
    rows = []

    for gq, sim in zip(generated_qs, similarities):
        rows.append({
            'question':               question,
            'generated_question':     gq,
            'cosine_similarity':      round(sim, 6),
            'answer_relevancy_score': '',
            'quality_gate':           '',
        })
    rows.append({
        'question':               question,
        'generated_question':     '** TOP-3 MEAN **',
        'cosine_similarity':      '',
        'answer_relevancy_score': round(final_score, 6),
        'quality_gate':           quality,
    })

    fieldnames = [
        'question', 'generated_question', 'cosine_similarity',
        'answer_relevancy_score', 'quality_gate',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")


