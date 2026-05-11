import json
import re
import os
import time
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

MODEL_ID = "meta.llama3-70b-instruct-v1:0"
MAX_CONTEXT = 8042       
MAX_BATCH_RISKS = 10     
TOKENS_PER_WORD = 1.33
OUTPUT_PER_RISK = 150
client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

SYSTEM_PROMPT = "You are a financial analyst expert at extracting structured risk information from SEC filings."

BATCH_PROMPT = """/no_think
Extract ALL risks explicitly disclosed in the following {n} risk factors from a 10-K filing.

The company in this document is: {main_company}

{risks_text}

Return ONLY a valid JSON array. Include the original risk number as "source_index" so each result can be matched back to its input:
[
  {{
    "source_index": 1,
    "risk_name": "Crude oil supply and demand fluctuations",
    "description": "concise factual description using the text's own wording, 80-160 chars",
    "why": "specific factual evidence or mechanism stated in the text explaining why this is a risk, 80-200 chars",
    "organization": "{main_company}"
  }}
]

STRICT Rules:
- risk_name: Use the exact heading or short title from the text. Do NOT invent category labels
  like "Geopolitical_Risk" or "Market_Risk" — use what the document actually says.
- description: Paraphrase the text's own wording — 80-160 characters. State WHAT the risk is.
- why: The factual basis from the document — state WHY this is a risk using specific facts,
  numbers, mechanisms, or conditions mentioned in the text (e.g. "Company derives 60% of revenue
  from a single customer, making it vulnerable to that customer's financial health").
  Do NOT write generic statements like "this could affect the business". Be specific and factual.
- Do NOT add a severity or category field.
- organization: ALWAYS use "{main_company}".
- Extract each distinct risk as a separate object. Merge near-duplicates into one.
- Return [] if no risk is explicitly described in the text.
- DO NOT use external knowledge — only what is written."""


def call_llama(prompt, n_risks, max_retries=5):
    formatted = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n["
    )
    max_gen_len = n_risks * OUTPUT_PER_RISK + 100
    body = json.dumps({"prompt": formatted, "max_gen_len": max_gen_len, "temperature": 0.3, "top_p": 0.9})
    
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            raw = json.loads(response["body"].read()).get("generation", "").strip()
            return "[" + raw
        except Exception as e:
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff: 1s, 2.5s, 5s, 9s, 17s
                print(f"    ⚠ Throttled, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries due to throttling") 


def validate_and_fix_result(result, main_company):
    """Validate and fix common issues in LLM results."""
    if not isinstance(result, dict):
        return None
    
    # Fix organization field
    org = result.get("organization", "").strip()
    if not org or org.lower() in ("n/a", "na", "the company", "this company", ""):
        result["organization"] = main_company
    
    # Validate required fields
    if not result.get("risk_name") or not result.get("description"):
        return None
    
    # Ensure description and why are reasonable length
    desc = result.get("description", "")
    why = result.get("why", "")
    
    if len(desc) < 20 or len(desc) > 500:
        return None
    
    if why and (len(why) < 20 or len(why) > 500):
        result["why"] = desc  # Fallback to description if why is invalid
    
    return result


def _repair_truncated_array(text):
    """Salvage all complete JSON objects from a truncated array."""
    start = text.find('[')
    if start == -1:
        return None
    fragment = text[start:]
    # Collect complete {...} objects one by one
    objects = []
    i = 0
    while i < len(fragment):
        if fragment[i] == '{':
            depth, j = 0, i
            in_str, escape = False, False
            while j < len(fragment):
                ch = fragment[j]
                if escape:
                    escape = False
                elif ch == '\\' and in_str:
                    escape = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                objects.append(json.loads(fragment[i:j+1]))
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
        i += 1
    return objects if objects else None


def extract_json_array(text):
    """Extract JSON array from LLM response with multiple fallback strategies."""
    # Remove markdown code blocks first
    text = re.sub(r'```json\s*|\s*```', '', text)

    # Strategy 1: Find complete JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON decode error: {e}")

    # Strategy 2: Repair truncated array — salvage complete objects
    repaired = _repair_truncated_array(text)
    if repaired:
        print(f"    ⚠ Truncated response repaired: recovered {len(repaired)} object(s)")
        return repaired

    # Strategy 3: Check if response is just empty array indicator
    if '[]' in text or 'no risk' in text.lower() or 'not found' in text.lower():
        return []

    print(f"    ⚠ Could not extract JSON. Response preview: {text[:1000]}")
    raise ValueError("No JSON array found in response")


def _map_results_to_batch(results, batch):
    
    indexed = {
        r.get("source_index"): r
        for r in results
        if isinstance(r.get("source_index"), int)
    }
    if len(indexed) == len(batch) and all(i + 1 in indexed for i in range(len(batch))):
        return [(indexed[i + 1], batch[i]) for i in range(len(batch))]
    return None


def extract_single_risk(risk_text, main_company="the Company", max_retries=2):
    """Extract a single risk with retry logic for better reliability."""
    single_prompt = BATCH_PROMPT.format(n=1, risks_text=f"RISK #1:\n{risk_text}", main_company=main_company)
    
    for attempt in range(max_retries):
        try:
            text = call_llama(single_prompt, 1)
            results = extract_json_array(text)
            if isinstance(results, list) and results:
                # Validate and fix the result
                fixed = validate_and_fix_result(results[0], main_company)
                if fixed:
                    return fixed
                elif attempt < max_retries - 1:
                    print(f"    ⚠ Result failed validation, retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(0.5)
                    continue
            elif isinstance(results, list) and not results:
                # Empty array returned - risk might be invalid
                if attempt < max_retries - 1:
                    print(f"    ⚠ Empty result, retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(0.5)
                    continue
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠ Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(0.5)
            else:
                print(f"    ✗ Single-risk extraction failed after {max_retries} attempts: {e}")
    return None


def extract_batch(risks, batch_num, main_company="the Company"):
    risks_text = "\n\n".join(f"RISK #{i+1}:\n{r}" for i, r in enumerate(risks))
    prompt = BATCH_PROMPT.format(n=len(risks), risks_text=risks_text, main_company=main_company)
    try:
        text = call_llama(prompt, len(risks))
        results = extract_json_array(text)
        if not isinstance(results, list):
            return None

        # Validate and fix each result
        validated_results = []
        for result in results:
            fixed = validate_and_fix_result(result, main_company)
            if fixed:
                validated_results.append(fixed)
        
        if not validated_results:
            print(f"  Batch {batch_num}: all results failed validation")
            return None

        mapped = _map_results_to_batch(validated_results, risks)
        if mapped:
            return mapped  

        if len(validated_results) == len(risks):
            print(f"  Batch {batch_num}: no source_index, using positional mapping")
            return list(zip(validated_results, risks))

        print(f"  Batch {batch_num}: got {len(validated_results)} valid results for {len(risks)} risks — will retry individually")
        return None
    except Exception as e:
        print(f"  Batch {batch_num} failed: {e}")
        return None


def make_batches(risks):
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


def process_all_risks(input_file="all_companies_risks.json", output_file="structured_risks.json", 
                     batch_delay=1.0, fallback_delay=0.5):
    """
    Process risks with rate limiting to avoid throttling.
    
    Args:
        batch_delay: Seconds to wait between batch API calls (default: 1.0)
        fallback_delay: Seconds to wait between individual risk API calls (default: 0.5)
    """
    with open(input_file, "r", encoding="utf-8") as f:
        companies_data = json.load(f)

    structured_results = []

    for company in companies_data:
        cik = company["cik"]
        if company["risk_count"] < 10:
            print(f"[CIK {cik}] Skipping — only {company['risk_count']} individual risks (likely unsplit section)")
            continue

        risks = company["individual_risks"]
        batches = make_batches(risks)
        print(f"\n[CIK {cik}] {len(risks)} risks → {len(batches)} batches (dynamic)")

        company_risks = []

        company_name = company.get("company_name", "the Company") or "the Company"

        for b, batch in enumerate(batches):
            print(f"  Batch {b+1}/{len(batches)} ({len(batch)} risks)...", end=" ")

            mapped = extract_batch(batch, b + 1, main_company=company_name)

            if mapped:
                for result, source_text in mapped:
                    result.pop("source_index", None)
                    result["source_text"] = source_text
                    result["risk_id"] = f"{cik}_risk_{len(company_risks) + 1}"
                    company_risks.append(result)
                print("✓")
            else:
                print("✗ falling back to per-risk processing")
                skipped_count = 0
                for i, risk_text in enumerate(batch, 1):
                    single = extract_single_risk(risk_text, main_company=company_name)
                    if single:
                        single.pop("source_index", None)
                        single["source_text"] = risk_text
                        single["risk_id"] = f"{cik}_risk_{len(company_risks) + 1}"
                        company_risks.append(single)
                        print(f"    → risk_{len(company_risks)} ✓")
                    else:
                        skipped_count += 1
                        print(f"    → risk {i}/{len(batch)} skipped (no valid response)")
                    
                    # Rate limit between individual risk calls
                    time.sleep(fallback_delay)
                
                if skipped_count > 0:
                    print(f"    ⚠ Warning: {skipped_count}/{len(batch)} risks skipped in this batch")
            
            # Rate limit between batches
            if b < len(batches) - 1:  # Don't sleep after last batch
                time.sleep(batch_delay)

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
