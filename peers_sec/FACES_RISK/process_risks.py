import json
import re
import os
import time
import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

MODEL_ID = "qwen.qwen3-next-80b-a3b"
MIN_BATCH_CHARS = 5000

client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

SYSTEM_PROMPT = "You are a financial analyst expert at extracting structured risk information from SEC filings."

EXTRACT_PROMPT = """/no_think
The following are numbered paragraphs from the Risk Factors section of a 10-K filing.
The company is: {main_company}

{text}

Extract ONLY the distinct risks that are explicitly described in these paragraphs.
Return a valid JSON array. If no actual risk is described, return [].

[
  {{
    "risk_name": "Short title using the document's own wording — no invented category labels",
    "description": "Concise factual description of what the risk is, 80-160 chars, paraphrasing the text",
    "why": "Specific factual evidence or mechanism from the text explaining why it is a risk, 80-200 chars — use numbers, conditions, or specific facts. Do NOT write generic statements.",
    "organization": "{main_company}",
    "source_index": 0
  }}
]

Rules:
- Extract each distinct risk as a separate object. Merge near-duplicates.
- source_index must be the 0-based index of the paragraph the risk came from (0, 1, 2, …).
- Do NOT use external knowledge — only what is written.
- Return [] for preamble, introductory text, or passages with no concrete risk described."""


def call_llm(prompt: str, max_retries: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    body = json.dumps({
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9,
    })

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(modelId=MODEL_ID, body=body)
            resp = json.loads(response["body"].read())

            raw = None
            if "output" in resp:
                raw = resp["output"].get("text", "") if isinstance(resp["output"], dict) else resp["output"]
            elif "content" in resp:
                raw = resp["content"][0].get("text", "") if isinstance(resp["content"], list) else resp["content"]
            elif "completion" in resp:
                raw = resp["completion"]
            elif "choices" in resp:
                raw = resp["choices"][0].get("message", {}).get("content", "")
            else:
                raise ValueError(f"Unknown response format: {list(resp.keys())}")

            if raw:
                return raw.strip()
            raise ValueError("Empty text in response")

        except Exception as e:
            if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                wait = (2 ** attempt) + (attempt * 0.5)
                print(f"    ⚠ Throttled, waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def _repair_truncated_array(text: str):
    start = text.find('[')
    if start == -1:
        return None
    fragment = text[start:]
    objects, i = [], 0
    while i < len(fragment):
        if fragment[i] == '{':
            depth, j, in_str, escape = 0, i, False, False
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
                                objects.append(json.loads(fragment[i:j + 1]))
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
        i += 1
    return objects or None


def extract_json_array(text: str):
    text = re.sub(r'```json\s*|\s*```', '', text)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_array(text)
    if repaired:
        return repaired
    if '[]' in text or 'no risk' in text.lower():
        return []
    raise ValueError(f"No JSON array found. Preview: {text[:500]}")


def validate_risk(risk: dict, company_name: str):
    if not isinstance(risk, dict):
        return None
    org = risk.get("organization", "").strip()
    if not org or org.lower() in ("n/a", "na", "the company", ""):
        risk["organization"] = company_name
    if not risk.get("risk_name") or not risk.get("description"):
        return None
    desc = risk.get("description", "")
    if len(desc) < 20 or len(desc) > 600:
        return None
    return risk


def make_batches(paragraphs: list) -> list:
    batches, current, current_chars = [], [], 0
    for para in paragraphs:
        chars = len(para)
        if chars >= MIN_BATCH_CHARS:
            # Large paragraph: flush any accumulated small ones first, then send alone
            if current:
                batches.append(current)
                current, current_chars = [], 0
            batches.append([para])
        else:
            current.append(para)
            current_chars += chars
            if current_chars >= MIN_BATCH_CHARS:
                batches.append(current)
                current, current_chars = [], 0
    if current:
        batches.append(current)
    return batches


def extract_batch(paragraphs: list, company_name: str) -> list:
    # Number each paragraph so the LLM can report which one each risk came from
    numbered = "\n\n".join(f"[{i}]\n{p}" for i, p in enumerate(paragraphs))
    prompt = EXTRACT_PROMPT.format(main_company=company_name, text=numbered)
    try:
        raw = call_llm(prompt)
        results = extract_json_array(raw)
        if not isinstance(results, list):
            return []
        validated = []
        for r in results:
            risk = validate_risk(r, company_name)
            if risk:
                idx = risk.pop("source_index", None)
                if isinstance(idx, int) and 0 <= idx < len(paragraphs):
                    risk["_source_paragraph"] = paragraphs[idx]
                else:
                    # Fallback: use the full batch text if index is missing/invalid
                    risk["_source_paragraph"] = "\n\n".join(paragraphs)
                validated.append(risk)
        return validated
    except Exception as e:
        print(f"    ✗ Batch failed: {e}")
        return []


def process_all_risks(
    input_file: str = "peers_sec/FACES_RISK/companies_risks.json",
    output_file: str = "peers_sec/FACES_RISK/structured_risks.json",
    batch_delay: float = 1.0,
):
    with open(input_file, "r", encoding="utf-8") as f:
        companies_data = json.load(f)

    structured_results = []
    
    # Global counter for peer risks
    peer_risk_counter = 0

    for company in companies_data:
        cik = company["cik"]
        company_name = company.get("company_name", "") or "the Company"


        paragraphs = company["individual_risks"]
        batches = make_batches(paragraphs)
        print(f"\n[CIK {cik}] {company_name} — {len(paragraphs)} paragraphs → {len(batches)} batches")

        company_risks = []

        for b, batch in enumerate(batches):
            print(f"  Batch {b + 1}/{len(batches)} ({len(batch)} paragraphs)...", end=" ")
            found = extract_batch(batch, company_name)

            for risk in found:
                # Get the specific source paragraph for this risk
                source_text = risk.pop("_source_paragraph", "\n\n".join(batch))
                
                risk_id = f"{cik}_risk_{len(company_risks) + 1}"
                
                # Generate citation_id using global counter
                peer_risk_counter += 1
                citation_id = f"PEER_R{peer_risk_counter:04d}"
                
                risk["risk_id"] = risk_id
                risk["citation_id"] = citation_id
                risk["metadata"] = {
                    "document_url": company.get("document_url", ""),
                    "source_text": source_text,
                }
                company_risks.append(risk)

            print(f"→ {len(found)} risk(s) found")

            if b < len(batches) - 1:
                time.sleep(batch_delay)

        print(f"  Total: {len(company_risks)} risks extracted")

        structured_results.append({
            "cik": cik,
            "company_name": company_name,
            "period_of_report": company.get("period_of_report"),
            "filing_date": company["filing_date"],
            "document_url": company["document_url"],
            "total_risks": len(company_risks),
            "risks": company_risks,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)

    total = sum(r["total_risks"] for r in structured_results)
    print(f"\nDone: {len(structured_results)} companies, {total} total risks → {output_file}")
    return structured_results


if __name__ == "__main__":
    process_all_risks()
