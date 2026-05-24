"""
Overall RAG Pipeline Performance Score

Aggregates the five individual evaluation CSVs into a single weighted score.

Weights:
  Faithfulness      (answer_source_traceability) : 30%
  Context Precision (context_precision_results)  : 25%
  Answer Relevancy  (answer_relevancy_results)   : 25%
  Context Recall    (avg of risks + metrics +
                     target validation)           : 20%

Output: prints a dashboard and writes overall_score.csv
"""

import csv
import os
import sys


# ---------------------------------------------------------------------------
# CSV readers — each returns a float in [0, 1]
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def score_answer_relevancy(path: str) -> float:
    for row in _read_csv(path):
        if row.get('aspect', '').strip() == '** SUMMARY **':
            return float(row['combined_score'])
    raise ValueError(f"No ** SUMMARY ** row found in {path}")


def score_context_precision(path: str) -> float:
    for row in _read_csv(path):
        if row.get('citation_id', '').strip() == '** SUMMARY **':
            return float(row['is_relevant'])
    raise ValueError(f"No ** SUMMARY ** row found in {path}")


def score_faithfulness(path: str) -> float:
    rows = [r for r in _read_csv(path) if r.get('claim_id', '').strip()]
    if not rows:
        raise ValueError(f"No claim rows found in {path}")
    correct = sum(1 for r in rows if r['is_correct_source'].strip().lower() == 'true')
    return correct / len(rows)


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
        ("Answer Relevancy",   score_answer_relevancy,  p("answer_relevancy_results.csv")),
        ("Context Precision",  score_context_precision, p("context_precision_results.csv")),
        ("Faithfulness",       score_faithfulness,      p("answer_source_traceability.csv")),
        ("Risks Validation",   score_risks_validation,  p("risks_validation_results.csv")),
        ("Metrics Validation", score_metrics_validation, p("metrics_validation_results.csv")),
        ("Target Validation",  score_target_validation, p("target_validation_results.csv")),
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

    # ── Context Recall = average of the three validators ─────────────────
    recall_components = ["Risks Validation", "Metrics Validation", "Target Validation"]
    recall_scores = [scores[k] for k in recall_components if k in scores]
    context_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

    if context_recall is not None:
        bar = "█" * int(context_recall * 20) + "░" * (20 - int(context_recall * 20))
        print(f"\n  {'~':<3} {'Context Recall (avg)':<22}  {bar}  {context_recall:.1%}")
        print(f"       (avg of: Risks {scores.get('Risks Validation', 0):.1%}  "
              f"Metrics {scores.get('Metrics Validation', 0):.1%}  "
              f"Target {scores.get('Target Validation', 0):.1%})")

    # ── Weighted overall ──────────────────────────────────────────────────
    weights = {
        "Faithfulness":     0.30,
        "Context Precision": 0.25,
        "Answer Relevancy": 0.25,
        "Context Recall":   0.20,
    }

    component_scores = {
        "Faithfulness":      scores.get("Faithfulness"),
        "Context Precision": scores.get("Context Precision"),
        "Answer Relevancy":  scores.get("Answer Relevancy"),
        "Context Recall":    context_recall,
    }

    available = {k: v for k, v in component_scores.items() if v is not None}
    if not available:
        print("\n✗ No scores computed — cannot produce overall score")
        return

    # Re-normalise weights if any component is missing
    total_weight = sum(weights[k] for k in available)
    overall = sum(weights[k] * v / total_weight for k, v in available.items())

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
            "score":       f"{score:.4f}" if score is not None else "N/A",
            "weighted_contribution": f"{weights[name] * score / total_weight:.4f}" if score is not None else "N/A",
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
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    ground_truth_dir = os.path.join(script_dir, "..")
    output_csv_path  = os.path.join(ground_truth_dir, "overall_score.csv")

    compute_overall(ground_truth_dir, output_csv_path)
