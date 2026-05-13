import sys
import os
import json
import re
import argparse
import time
import urllib3
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import boto3
from dotenv import load_dotenv

# Load .env from project root (3 levels up from this file)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
load_dotenv(env_path)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL = os.getenv("BEDROCK_MODEL", "qwen.qwen3-next-80b-a3b")

EDGAR_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "YourName your_email@example.com")}
_edgar_sic_cache: Dict[str, bool] = {}


def _sic_exists_in_edgar(sic_code: str) -> bool:
    """Return True if SEC EDGAR recognises this SIC code (has at least one 10-K filer)."""
    if sic_code in _edgar_sic_cache:
        return _edgar_sic_cache[sic_code]
    try:
        time.sleep(0.3)  # respect SEC rate limit
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={"action": "getcompany", "SIC": sic_code, "type": "10-K",
                    "owner": "include", "count": "1", "output": "atom"},
            headers=EDGAR_HEADERS,
            timeout=10,
            verify=False,
        )
        if resp.status_code != 200:
            print(f"    EDGAR returned HTTP {resp.status_code} for SIC {sic_code} — keeping candidate")
            _edgar_sic_cache[sic_code] = True
            return True

        # Parse the Atom feed the same way fetch_and_extract_risks.py does
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        exists = len(entries) > 0
        _edgar_sic_cache[sic_code] = exists
        return exists
    except ET.ParseError:
        # EDGAR returned HTML (e.g. error page) instead of XML — treat as not found
        print(f"    EDGAR returned non-XML for SIC {sic_code} — marking as invalid")
        _edgar_sic_cache[sic_code] = False
        return False
    except Exception as e:
        print(f"  ⚠ Could not verify SIC {sic_code} against EDGAR: {e} — keeping candidate")
        _edgar_sic_cache[sic_code] = True
        return True  # optimistic fallback


def _build_prompt(company_name: str, candidates: List[Dict], chunk_texts: List[str]) -> str:
    # Combine chunk texts, capped to avoid token limits
    combined_text = "\n\n---\n\n".join(chunk[:1200] for chunk in chunk_texts)[:7000]

    candidates_str = "\n".join(
        f"{i+1}. industry=\"{c['industry']}\", sector=\"{c['sector']}\", sic_code=\"{c['sic_code']}\""
        for i, c in enumerate(candidates)
    )

    return f"""/no_think

You are an expert in SEC EDGAR SIC classification. Given excerpts from {company_name}'s 10-K filing and a list of candidate industry classifications, choose the ONE candidate whose SIC code SEC EDGAR would most likely assign to this company.

COMPANY EXCERPTS:
{combined_text}

CANDIDATE INDUSTRIES:
{candidates_str}

Rules:
- Choose exactly one candidate number from the list above
- Prioritise the SIC code that SEC EDGAR actually uses for companies with this primary business activity — not just the closest description
- Prefer a specific, narrow SIC code over a broad parent category when the specific one fits
- Prefer the classification that captures the company's core revenue-generating activity as SEC EDGAR would record it
- Avoid generic or catch-all SIC codes (e.g. "Services-Not Elsewhere Classified") when a more specific code is present

Return ONLY the candidate number (e.g. "2"). No explanation, no text."""


def validate_operates_in(extracted_json_path: str) -> Optional[Dict]:
    """
    Reads an OPERATES_IN extracted JSON, examines all chunk_texts and SIC candidates,
    selects the single best industry/SIC entry, and returns the complete relation with metadata.
    """
    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data.get('main_company', 'the Company')
    relations = data.get('relations', {}).get('OPERATES_IN', [])

    if not relations:
        print("No OPERATES_IN relations found.")
        return None

    # Collect unique candidates (deduplicated by sic_code) and all chunk texts
    # Also keep track of which relation each candidate came from
    seen_sic: set = set()
    candidates: List[Dict] = []
    chunk_texts: List[str] = []
    candidate_to_relation: Dict[int, Dict] = {}  # Map candidate index to full relation

    for rel in relations:
        sic = rel.get('sic', {})
        code = str(sic.get('code', '')).strip()
        industry = (sic.get('industry') or rel.get('tgt', {}).get('name', '')).strip()
        sector = (sic.get('sector') or rel.get('tgt', {}).get('properties', {}).get('sector', '')).strip()

        if code and code not in seen_sic:
            seen_sic.add(code)
            candidate_idx = len(candidates)
            candidates.append({'industry': industry, 'sector': sector, 'sic_code': code})
            candidate_to_relation[candidate_idx] = rel  # Store the full relation

        chunk = rel.get('chunk_text', '').strip()
        if chunk:
            chunk_texts.append(chunk)

    if not candidates:
        print("No SIC-backed candidates found.")
        return None

    # --- Filter out SIC codes not recognised by SEC EDGAR ---
    print(f"Verifying {len(candidates)} candidate SIC codes against SEC EDGAR...")
    valid_candidates: List[Dict] = []
    valid_candidate_to_relation: Dict[int, Dict] = {}
    for old_idx, cand in enumerate(candidates):
        if _sic_exists_in_edgar(cand['sic_code']):
            new_idx = len(valid_candidates)
            valid_candidates.append(cand)
            valid_candidate_to_relation[new_idx] = candidate_to_relation[old_idx]
            print(f"  ✓ SIC {cand['sic_code']} ({cand['industry']}) — confirmed in EDGAR")
        else:
            print(f"  ✗ SIC {cand['sic_code']} ({cand['industry']}) — NOT found in EDGAR, removed")

    if not valid_candidates:
        print("⚠ All candidates removed by EDGAR validation — falling back to original list.")
        valid_candidates = candidates
        valid_candidate_to_relation = candidate_to_relation

    candidates = valid_candidates
    candidate_to_relation = valid_candidate_to_relation
    # --------------------------------------------------------

    if len(candidates) == 1:
        print(f"Single candidate after EDGAR validation — keeping: {candidates[0]}")
        return {
            'main_company': company_name,
            'validated_relation': candidate_to_relation[0]
        }

    print(f"Found {len(candidates)} candidate industries for '{company_name}'. Asking LLM to pick the best one...")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c['industry']} | sector={c['sector']} | sic={c['sic_code']}")

    # Initialize AWS Bedrock client
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    prompt = _build_prompt(company_name, candidates, chunk_texts)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "You are an expert in SEC SIC classification. Return only the candidate number, nothing else.\n\n" + prompt
                }
            ]
        }
    ]
    
    response = client.converse(
        modelId=MODEL,
        messages=messages,
        inferenceConfig={
            "temperature": 0.0,
            "maxTokens": 10
        }
    )

    raw = response['output']['message']['content'][0]['text'].strip()
    match = re.search(r'\d+', raw)

    if not match:
        print(f"LLM returned unexpected output: {raw!r}. Defaulting to candidate 1.")
        chosen_idx = 0
    else:
        chosen_idx = int(match.group()) - 1
        if not (0 <= chosen_idx < len(candidates)):
            print(f"LLM chose out-of-range index {chosen_idx + 1}. Defaulting to candidate 1.")
            chosen_idx = 0

    winner = candidates[chosen_idx]
    winner_relation = candidate_to_relation[chosen_idx]
    
    print(f"\nSelected: industry='{winner['industry']}', sector='{winner['sector']}', sic_code='{winner['sic_code']}'")
    
    return {
        'main_company': company_name,
        'validated_relation': winner_relation
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and select the best OPERATES_IN industry/SIC for a company."
    )
    
    # Default paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, 'extracted_output.json')
    default_output = os.path.join(script_dir, 'validated_industry.json')
    
    parser.add_argument('--input', default=default_input, help=f"Path to extracted OPERATES_IN JSON file (default: {default_input})")
    parser.add_argument('--output', default=default_output, help=f"Path to write the result JSON (default: {default_output})")
    args = parser.parse_args()

    result = validate_operates_in(args.input)
    if result:
        print("\nFinal validated industry:")
        print(json.dumps(result, indent=2))

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
