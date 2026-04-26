import json

MAX_CONTEXT = 8042
TOKENS_PER_WORD = 1.33
OUTPUT_PER_RISK = 80

BATCH_PROMPT = """Analyze the following {n} risk factors from a 10-K filing.
For EACH risk, extract:
1. A concise risk name (max 10 words)
2. A clear description (1-2 sentences)
3. A category from: [Operational, Financial, Market, Regulatory, Strategic, Environmental, Legal, Technology, Competitive, Other]

{risks_text}

Return ONLY a JSON array with exactly {n} objects:
[
  {{"risk_name": "...", "description": "...", "category": "..."}},
  ...
]"""


def make_batches(risks):
    batches, current, current_tokens = [], [], 0
    for risk in risks:
        risk_tokens = len(risk.split()) * TOKENS_PER_WORD
        if current and current_tokens + risk_tokens + OUTPUT_PER_RISK > MAX_CONTEXT:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(risk)
        current_tokens += risk_tokens + OUTPUT_PER_RISK
    if current:
        batches.append(current)
    return batches


def preview_llm_inputs(input_file="all_companies_risks.json", num_examples=1):
    with open(input_file, "r", encoding="utf-8") as f:
        companies = json.load(f)

    company = next((c for c in companies if c["risk_count"] > 0), None)
    if not company:
        print("No companies with risks found.")
        return

    risks = company["individual_risks"]
    batches = make_batches(risks)
    first_batch = batches[0]

    print(f"COMPANY: CIK {company['cik']} | Risks: {len(risks)} | Filed: {company['filing_date']}")
    print(f"Dynamic batches: {len(batches)} (budget: {MAX_CONTEXT} total tokens/batch)\n")

    for i in range(min(num_examples, len(batches))):
        batch = batches[i]
        risks_text = "\n\n".join(f"RISK #{j+1}:\n{r}" for j, r in enumerate(batch))
        input_tokens = int(sum(len(r.split()) * TOKENS_PER_WORD for r in batch))
        output_tokens = len(batch) * OUTPUT_PER_RISK
        print(f"--- Batch {i+1}: {len(batch)} risks | ~{input_tokens} input tokens | ~{output_tokens} output tokens ---")
        print(BATCH_PROMPT.format(n=len(batch), risks_text=risks_text))
        print()

    print("--- SUMMARY (all companies) ---")
    total_risks = total_calls = 0
    for c in companies:
        if c["risk_count"] == 0:
            continue
        b = make_batches(c["individual_risks"])
        sizes = [len(x) for x in b]
        est_tokens = [int(sum(len(r.split()) * TOKENS_PER_WORD for r in x)) for x in b]
        print(f"  CIK {c['cik']}: {c['risk_count']} risks → {len(b)} calls "
              f"(batch sizes: {sizes}, input tokens: {est_tokens})")
        total_risks += c["risk_count"]
        total_calls += len(b)

    print(f"\nTotal risks: {total_risks}")
    print(f"Total LLM calls (dynamic batching): {total_calls}")


if __name__ == "__main__":
    preview_llm_inputs(input_file="all_companies_risks.json", num_examples=1)
