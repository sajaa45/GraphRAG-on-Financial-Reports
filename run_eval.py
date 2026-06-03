"""
End-to-end evaluation runner — loops over a list of questions.

For each question:
  1. Runs the RAG pipeline (credit_risk_qa.py)
  2. Runs all evaluation scripts, writing directly into Ground_Truth/results/Q<N>_<slug>/

Run:  python run_eval.py
"""

import csv
import os
import re
import shutil
import subprocess
import sys

# ── Questions to evaluate ────────────────────────────────────────────────────
QUESTIONS = [
    # ── Risk-focused ──────────────────────────────────────────────────────
    "What are the main operational and business risks facing the target company?",
    "What cybersecurity or technology risks does the target company disclose and do its peers face similar threats?",
    "What regulatory and legal risks could materially impact the target company's business?",
    "Does the target company face any going-concern or solvency risks compared to its peers?",
    "Which risk categories appear exclusively in the target company's filings but not in any peer filing?",

    # ── Metrics-focused ───────────────────────────────────────────────────
    "How does the target company's liquidity position compare to its peers?",
    "What is the target company's leverage ratio and how does it rank among its peers?",
    "How does the target company's interest coverage compare to its peers?",
    "What are the target company's revenue and net income trends and how do they compare to peers?",
    "How does the target company's debt maturity profile compare to its peers?",

    # ── Cross-domain (risks + metrics) ────────────────────────────────────
    "Does the target company carry more financial risk relative to its operating performance than its peers?",
    "Is the target company generating enough cash flow to cover its debt obligations?",
    "What leverage and debt structure risks does the target company face?",
    "How exposed is the target company to interest rate risk compared to its peers?",

    # ── Synthesis ─────────────────────────────────────────────────────────
    "Which company in the peer group represents the strongest overall credit profile?",
    "Based on its risk disclosures and financial metrics, what is the target company's key credit weakness?",
]
# ─────────────────────────────────────────────────────────────────────────────

BASE        = os.path.dirname(os.path.abspath(__file__))
GROUND_DIR  = os.path.join(BASE, "retrival+eval", "Ground_Truth")
RESULTS_DIR = os.path.join(GROUND_DIR, "results")

EVAL_SCRIPTS = [
    os.path.join(GROUND_DIR, "resources", "answer_relevancy.py"),
    os.path.join(GROUND_DIR, "resources", "answer_source_traceability.py"),
    os.path.join(GROUND_DIR, "resources", "context_precision.py"),
    os.path.join(GROUND_DIR, "resources", "num_values_peers.py"),
    os.path.join(GROUND_DIR, "resources", "risk_peers.py"),
    os.path.join(GROUND_DIR, "resources", "target_validation.py"),
    os.path.join(GROUND_DIR, "resources", "overall_score.py"),
]


# ---------------------------------------------------------------------------
# Score readers — one per eval output CSV
# ---------------------------------------------------------------------------

def _csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100*num//den}%)" if den else "0/0"


def _score_answer_relevancy(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "answer_relevancy_results.csv"))
    scores = [float(r["answer_relevancy_score"]) for r in rows
              if r.get("answer_relevancy_score") not in (None, "", "N/A")]
    if not scores:
        return "N/A"
    avg = sum(scores) / len(scores)
    return f"{avg:.3f} (avg over {len(scores)} q)"


def _score_source_traceability(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "answer_source_traceability.csv"))
    if not rows:
        return "N/A"
    correct = sum(1 for r in rows if r.get("is_correct_source", "").strip().lower() == "true")
    return _pct(correct, len(rows))


def _score_context_precision(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "context_precision_results.csv"))
    if not rows:
        return "N/A"
    relevant = sum(1 for r in rows if r.get("is_relevant", "").strip().lower() == "true")
    return _pct(relevant, len(rows))


def _score_metrics_validation(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "metrics_validation_results.csv"))
    if not rows:
        return "N/A"
    match      = sum(1 for r in rows if r.get("match") == "MATCH")
    mismatch   = sum(1 for r in rows if r.get("match") == "MISMATCH")
    verifiable = match + mismatch
    return _pct(match, verifiable) if verifiable else f"0 verified / {len(rows)} total"


def _score_risk_peers(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "risks_validation_results.csv"))
    if not rows:
        return "N/A"
    relevant = sum(1 for r in rows
                   if r.get("is_semantically_relevant", "").strip().lower() == "true")
    return _pct(relevant, len(rows))


def _score_target_validation(q_dir: str) -> str:
    rows = _csv_rows(os.path.join(q_dir, "target_validation_results.csv"))
    if not rows:
        return "N/A"
    correct = sum(1 for r in rows
                  if r.get("is_correctly_extracted", "").strip().lower() == "true")
    return _pct(correct, len(rows))


def _score_overall(q_dir: str) -> str:
    path = os.path.join(q_dir, "overall_score.csv")
    if not os.path.exists(path):
        return "N/A"
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if parts and "OVERALL" in parts[0].upper():
                    score  = parts[2] if len(parts) > 2 else "?"
                    status = parts[3] if len(parts) > 3 else ""
                    return f"{score}  ({status})" if status else score
    except Exception:
        pass
    return "N/A"


# Map script basename → score reader
_SCORE_READERS = {
    "answer_relevancy":          _score_answer_relevancy,
    "answer_source_traceability": _score_source_traceability,
    "context_precision":         _score_context_precision,
    "num_values_peers":          _score_metrics_validation,
    "risk_peers":                _score_risk_peers,
    "target_validation":         _score_target_validation,
    "overall_score":             _score_overall,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return s[:max_len]


def run(cmd: list, label: str) -> int:
    print(f"  [{label}] ...", end="", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=BASE, capture_output=True, env=env)
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\r  [{label}] {status:<30}")
    return result.returncode


def prepare_q_dir(q_dir: str, question: str):
    """Create the per-question directory and copy retrieval artefacts into it."""
    os.makedirs(q_dir, exist_ok=True)
    with open(os.path.join(q_dir, "question.txt"), "w", encoding="utf-8") as f:
        f.write(question + "\n")
    for fname in ("extraction_result.json", "answer.txt"):
        src = os.path.join(BASE, "retrival+eval", "retrival_results", fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(q_dir, fname))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []

    total_steps = 1 + len(EVAL_SCRIPTS)

    for idx, question in enumerate(QUESTIONS, start=1):
        q_label = f"Q{idx:02d}"
        q_dir   = os.path.join(RESULTS_DIR, f"{q_label}_{_slug(question)}")

        print(f"\n{q_label}/{len(QUESTIONS)}: {question[:80]}")

        failed = 0

        # Step 1 — RAG query (writes to retrival_results/)
        rc = run(
            [sys.executable,
             os.path.join(BASE, "retrival+eval", "credit_risk_qa.py"),
             question],
            f"1/{total_steps} RAG query",
        )
        failed += rc != 0

        # Copy retrieval artefacts into q_dir so they are archived per-question
        prepare_q_dir(q_dir, question)

        # Steps 2–N — eval scripts write directly into q_dir
        overall_script = os.path.join(GROUND_DIR, "resources", "overall_score.py")
        for i, script in enumerate(EVAL_SCRIPTS, start=2):
            name = os.path.basename(script).replace(".py", "")
            extra = ["--out-dir", q_dir]
            if script == overall_script:
                extra = ["--in-dir", q_dir, "--out-dir", q_dir]
            rc = run(
                [sys.executable, script] + extra,
                f"{i}/{total_steps} {name}",
            )
            failed += rc != 0

        print(f"  {'─'*52}")
        for name, reader in _SCORE_READERS.items():
            score = reader(q_dir)
            print(f"    {name:<34} {score}")
        overall = _score_overall(q_dir)
        print(f"  {'─'*52}")
        run_status = "OK" if failed == 0 else f"{failed} step(s) failed"
        print(f"  Overall: {overall}  |  {run_status}")

        summary.append((question, q_dir, failed, overall))

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    for question, q_dir, failed, overall in summary:
        status = "OK" if failed == 0 else f"{failed} failed"
        print(f"  {os.path.basename(q_dir)[:45]:<45}  {overall}  {status}")
    print(f"\n  Results in: {RESULTS_DIR}")
    print(f"{'='*60}\n")
