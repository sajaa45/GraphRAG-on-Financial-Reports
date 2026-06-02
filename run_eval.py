"""
End-to-end evaluation runner.
Edit QUESTION below, then run:  python run_eval.py
"""

import subprocess
import sys
import os

# ── Configure your question here ─────────────────────────────────────────────
QUESTION = " What are the main risks facing the target company and how do they compare to its peers?"
# ─────────────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

EVAL_SCRIPTS = [
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "answer_relevancy.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "answer_source_traceability.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "context_precision.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "num_values_peers.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "risk_peers.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "target_validation.py"),
    os.path.join(BASE, "retrival+eval", "Ground_Truth", "resources", "overall_score.py"),
]


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n[WARNING] '{label}' exited with code {result.returncode}")


if __name__ == "__main__":
    # Step 1 – RAG query
    run(
        [sys.executable,
         os.path.join(BASE, "retrival+eval", "credit_risk_qa.py"),
         QUESTION,  "-v"],
        "Step 1: RAG Query",
    )

    # Step 2 – Evaluation scripts
    for i, script in enumerate(EVAL_SCRIPTS, start=2):
        run([sys.executable, script], f"Step {i}: {os.path.basename(script)}")

    print(f"\n{'='*60}")
    print("  All steps complete.")
    print(f"{'='*60}\n")
