"""
Overall RAG Pipeline Performance Score

Aggregates the five individual evaluation CSVs into a single weighted score.
This is NOT a standard RAGAS score — it is a custom pipeline quality metric.

Weights:
  Adapted Faithfulness       (answer_source_traceability) : 30%
  Adapted Context Precision  (context_precision_results)  : 25%
  Adapted Answer Relevancy   (answer_relevancy_results)   : 25%
  ExtractionQuality          (avg of risks + metrics +
                              target validation)           : 20%
   
Faithfulness scoring:
  correct source      → 1.0
  wrong-but-real source → 0.5  (pipeline used a real source, just the wrong one)
  fabricated citation → 0.0  (no source in the graph supports the claim)

Missing components score 0.0 (not excluded) to avoid silent score inflation.

Output: prints a dashboard and writes overall_score.csv
"""

import csv
import os
import sys


# ---------------------------------------------------------------------------
# CSV readers — each returns a float in [0, 1]
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list[dict]:
    import sys
    csv.field_size_limit(min(sys.maxsize, 10_000_000))
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def score_answer_relevancy(path: str) -> float:
    for row in _read_csv(path):
        if row.get('generated_question', '').strip() in ('** MEAN **', '** TOP-3 MEAN **'):
            return float(row['answer_relevancy_score'])
    raise ValueError(f"No ** MEAN ** summary row found in {path}")


def score_context_precision(path: str) -> float:
    for row in _read_csv(path):
        if row.get('citation_id', '').strip() == '** SUMMARY **':
            return float(row['is_relevant'])
    raise ValueError(f"No ** SUMMARY ** row found in {path}")


def score_faithfulness(path: str) -> float:
    rows = [r for r in _read_csv(path) if r.get('claim_id', '').strip()]
    if not rows:
        raise ValueError(f"No claim rows found in {path}")
    total = 0.0
    for r in rows:
        if r['is_correct_source'].strip().lower() == 'true':
            total += 1.0
        elif r.get('correct_source', '').strip().upper() == 'FABRICATED':
            total += 0.0  # no source in graph supports this claim
        else:
            total += 0.5  # wrong-but-real source cited
    return total / len(rows)


def score_risks_validation(path: str) -> float:
    rows = [r for r in _read_csv(path) if r.get('company_name', '').strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    relevant = sum(1 for r in rows if str(r['is_semantically_relevant']).strip().lower() == 'true')
    return relevant / len(rows)


def score_metrics_validation(path: str) -> float:
    rows = [r for r in _read_csv(path) if r.get('xbrl_tag', '').strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    matched = sum(1 for r in rows if r['match'].strip().upper() == 'MATCH')
    return matched / len(rows)


def score_target_validation(path: str) -> float:
    rows = [r for r in _read_csv(path) if r.get('citation_id', '').strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    correct = sum(1 for r in rows if str(r['is_correctly_extracted']).strip().lower() == 'true')
    return correct / len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_overall(ground_truth_dir: str, output_csv_path: str):
    print("=" * 70)
    print("Overall RAG Pipeline Performance Score")
    print("=" * 70)

    def p(label: str) -> str:
        return os.path.join(ground_truth_dir, label)

    # ── Individual scores ─────────────────────────────────────────────────
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    checks = [
        ("Adapted Answer Relevancy",   score_answer_relevancy,  p("answer_relevancy_results.csv")),
        ("Adapted Context Precision",  score_context_precision, p("context_precision_results.csv")),
        ("Adapted Faithfulness",       score_faithfulness,      p("answer_source_traceability.csv")),
        ("Risks Validation",           score_risks_validation,  p("risks_validation_results.csv")),
        ("Metrics Validation",         score_metrics_validation, p("metrics_validation_results.csv")),
        ("Target Validation",          score_target_validation, p("target_validation_results.csv")),
    ]

    print()
    for name, fn, path in checks:
        if not os.path.exists(path):
            errors[name] = f"File not found: {path}"
            print(f"  {'✗':<3} {name:<22} — FILE NOT FOUND")
            continue
        try:
            s = fn(path)
            scores[name] = s
            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            print(f"  {'✓':<3} {name:<22}  {bar}  {s:.1%}")
        except Exception as e:
            errors[name] = str(e)
            print(f"  {'✗':<3} {name:<22} — ERROR: {e}")

    # ── Extraction Quality = average of available validators ─────────────
    # NOTE: this is NOT RAGAS Context Recall — it measures extraction accuracy
    # and data integrity, not retrieval completeness vs a ground-truth answer.
    # Components that errored or have no data are SKIPPED (not scored as 0.0)
    # so that e.g. having no peer risks doesn't silently deflate the score.
    eq_components = ["Risks Validation", "Metrics Validation", "Target Validation"]
    eq_available = {k: scores[k] for k in eq_components if k in scores}
    if eq_available:
        extraction_quality = sum(eq_available.values()) / len(eq_available)
    else:
        extraction_quality = 0.0

    skipped = [k for k in eq_components if k not in scores]
    bar = "█" * int(extraction_quality * 20) + "░" * (20 - int(extraction_quality * 20))
    print(f"\n  {'~':<3} {'ExtractionQuality':<22}  {bar}  {extraction_quality:.1%}")
    detail = "  ".join(
        f"{k.split()[0]} {scores[k]:.1%}" for k in eq_components if k in scores
    )
    print(f"       (avg of: {detail})", end="")
    if skipped:
        print(f"  — skipped: {', '.join(skipped)} (no data)", end="")
    print()

    # ── Weighted overall ──────────────────────────────────────────────────
    weights = {
        "Adapted Faithfulness":        0.30,
        "Adapted Context Precision":   0.25,
        "Adapted Answer Relevancy":    0.25,
        "ExtractionQuality":           0.20,
    }

    # Missing components score 0.0 — weights are NOT renormalised to prevent
    # a missing file from silently inflating the overall score.
    component_scores = {
        "Adapted Faithfulness":       scores.get("Adapted Faithfulness", 0.0),
        "Adapted Context Precision":  scores.get("Adapted Context Precision", 0.0),
        "Adapted Answer Relevancy":   scores.get("Adapted Answer Relevancy", 0.0),
        "ExtractionQuality":          extraction_quality,
    }

    if not any(k in scores for k in ("Adapted Faithfulness", "Adapted Context Precision", "Adapted Answer Relevancy")) \
            and not any(scores.get(k) for k in eq_components):
        print("\n✗ No scores computed — cannot produce overall score")
        return

    total_weight = sum(weights.values())  # always 1.0
    overall = sum(weights[k] * v for k, v in component_scores.items())

    quality = "PASS" if overall >= 0.80 else ("WARNING" if overall >= 0.65 else "FAIL")
    bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))

    print(f"\n{'─' * 70}")
    print(f"  {'OVERALL SCORE':<22}  {bar}  {overall:.1%}  [{quality}]")
    print(f"  Threshold: PASS ≥ 80%  |  WARNING ≥ 65%  |  FAIL < 65%")
    print(f"{'─' * 70}")

    if errors:
        print(f"\n  ⚠ {len(errors)} component(s) could not be scored:")
        for name, msg in errors.items():
            print(f"    - {name}: {msg}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    rows = []
    for name, score in component_scores.items():
        rows.append({
            "component":   name,
            "weight":      f"{weights[name]:.0%}",
            "score":       f"{score:.4f}",
            "weighted_contribution": f"{weights[name] * score:.4f}",
        })
    rows.append({
        "component":   "** OVERALL **",
        "weight":      "100%",
        "score":       f"{overall:.4f}",
        "weighted_contribution": quality,
    })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "weight", "score", "weighted_contribution"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Results written to {output_csv_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir',  default=None)
    ap.add_argument('--out-dir', default=None)
    args, _ = ap.parse_known_args()

    script_dir       = os.path.dirname(os.path.abspath(__file__))
    in_dir           = args.in_dir  or os.path.join(script_dir, "..")
    out_dir          = args.out_dir or in_dir
    output_csv_path  = os.path.join(out_dir, "overall_score.csv")

    compute_overall(in_dir, output_csv_path)
