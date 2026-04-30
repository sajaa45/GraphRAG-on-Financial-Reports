import json
import re
import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

MODEL_ID = "meta.llama3-70b-instruct-v1:0"
MAX_CONTEXT = 8042       # 8192 - 150 prompt overhead
MAX_BATCH_RISKS = 10     # cap for reliable JSON output from Llama
TOKENS_PER_WORD = 1.33
OUTPUT_PER_RISK = 80     # estimated output tokens per extracted risk

# Configure AWS client with credentials from .env
client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

SYSTEM_PROMPT = "You are a financial analyst expert at extracting structured risk information from SEC filings."

# Fix 3: ask the model to echo source_index so results can be mapped back
# to their original input text regardless of how many objects the model returns.
BATCH_PROMPT = """Analyze the following {n} risk factors from a 10-K filing.
For EACH risk, extract:
1. A concise risk name (max 10 words)
2. A clear description (1-2 sentences)
3. A category from: [Operational, Financial, Market, Regulatory, Strategic, Environmental, Legal, Technology, Competitive, Other]

{risks_text}

Return ONLY a JSON array. Include the original risk number as "source_index" so each result can be matched back to its input:
[
  {{"source_index": 1, "risk_name": "...", "description": "...", "category": "..."}},
  ...
]"""


def call_llama(prompt, n_risks):
    # Fix 4: prime the assistant turn with '[' to force JSON array output and
    # prevent the model from prepending conversational filler like "Here is...".
    formatted = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n["
    )
    max_gen_len = n_risks * OUTPUT_PER_RISK + 100
    body = json.dumps({"prompt": formatted, "max_gen_len": max_gen_len, "temperature": 0.3, "top_p": 0.9})
    response = client.invoke_model(modelId=MODEL_ID, body=body)
    raw = json.loads(response["body"].read()).get("generation", "").strip()
    return "[" + raw  # re-attach the opening bracket we primed


def extract_json_array(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON array found in response")


def _map_results_to_batch(results, batch):
    """
    Fix 3: use source_index to map each result back to its original input text.
    Returns a list of (result, source_text) pairs, or None if mapping fails.
    """
    indexed = {
        r.get("source_index"): r
        for r in results
        if isinstance(r.get("source_index"), int)
    }
    if len(indexed) == len(batch) and all(i + 1 in indexed for i in range(len(batch))):
        return [(indexed[i + 1], batch[i]) for i in range(len(batch))]
    return None


def extract_single_risk(risk_text):
    """Fallback: process one risk individually."""
    single_prompt = BATCH_PROMPT.format(n=1, risks_text=f"RISK #1:\n{risk_text}")
    try:
        text = call_llama(single_prompt, 1)
        results = extract_json_array(text)
        if isinstance(results, list) and results:
            return results[0]
    except Exception as e:
        print(f"    single-risk fallback failed: {e}")
    return None


def extract_batch(risks, batch_num):
    risks_text = "\n\n".join(f"RISK #{i+1}:\n{r}" for i, r in enumerate(risks))
    prompt = BATCH_PROMPT.format(n=len(risks), risks_text=risks_text)
    try:
        text = call_llama(prompt, len(risks))
        results = extract_json_array(text)
        if not isinstance(results, list):
            return None

        # Fix 2 + Fix 3: try source_index mapping regardless of count match
        mapped = _map_results_to_batch(results, risks)
        if mapped:
            return mapped  # list of (result_dict, source_text)

        # Exact count match without valid source_index: fall back to positional
        if len(results) == len(risks):
            print(f"  Batch {batch_num}: no source_index, using positional mapping")
            return list(zip(results, risks))

        # Fix 2: count mismatch with no reliable mapping — signal caller to fall back
        print(f"  Batch {batch_num}: got {len(results)} results for {len(risks)} risks — will retry individually")
        return None
    except Exception as e:
        print(f"  Batch {batch_num} failed: {e}")
        return None


def make_batches(risks):
    """Group risks by token budget and cap at MAX_BATCH_RISKS for reliability."""
    batches, current, current_tokens = [], [], 0
    for risk in risks:
        cost = len(risk.split()) * TOKENS_PER_WORD + OUTPUT_PER_RISK
        if current and (current_tokens + cost > MAX_CONTEXT or len(current) >= MAX_BATCH_RISKS):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(risk)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


def process_all_risks(input_file="all_companies_risks.json", output_file="structured_risks.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        companies_data = json.load(f)

    structured_results = []

    for company in companies_data:
        cik = company["cik"]
        if company["risk_count"] == 0:
            continue

        risks = company["individual_risks"]
        batches = make_batches(risks)
        print(f"\n[CIK {cik}] {len(risks)} risks → {len(batches)} batches (dynamic)")

        company_risks = []

        for b, batch in enumerate(batches):
            print(f"  Batch {b+1}/{len(batches)} ({len(batch)} risks)...", end=" ")

            mapped = extract_batch(batch, b + 1)

            if mapped:
                for result, source_text in mapped:
                    result.pop("source_index", None)
                    result["source_text"] = source_text
                    # Fix 1: ID is always len(company_risks)+1 so it stays
                    # contiguous even when earlier batches were dropped.
                    result["risk_id"] = f"{cik}_risk_{len(company_risks) + 1}"
                    company_risks.append(result)
                print("✓")
            else:
                # Fix 2: per-risk fallback instead of silently dropping the batch
                print("✗ falling back to per-risk processing")
                for risk_text in batch:
                    single = extract_single_risk(risk_text)
                    if single:
                        single.pop("source_index", None)
                        single["source_text"] = risk_text
                        single["risk_id"] = f"{cik}_risk_{len(company_risks) + 1}"
                        company_risks.append(single)
                        print(f"    → risk_{len(company_risks)} ✓")
                    else:
                        print(f"    → skipped (model returned nothing)")

        structured_results.append({
            "cik": cik,
            "company_name": company.get("company_name", ""),
            "filing_date": company["filing_date"],
            "document_url": company["document_url"],
            "total_risks": len(company_risks),
            "risks": company_risks,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(structured_results)} companies saved to {output_file}")
    return structured_results


if __name__ == "__main__":
    process_all_risks()
