import os
import json
import re
import time
import argparse
import urllib3
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EDGAR_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "YourName your_email@example.com")}
_edgar_sic_cache: Dict[str, bool] = {}


def _sic_exists_in_edgar(sic_code: str) -> bool:
    if sic_code in _edgar_sic_cache:
        return _edgar_sic_cache[sic_code]
    time.sleep(0.3)
    resp = requests.get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={"action": "getcompany", "SIC": sic_code, "type": "10-K",
                "owner": "include", "count": "1", "output": "atom"},
        headers=EDGAR_HEADERS,
        timeout=30,
        verify=False,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"EDGAR returned HTTP {resp.status_code} for SIC {sic_code}")
    root = ET.fromstring(resp.content)
    exists = len(root.findall("atom:entry", {"atom": "http://www.w3.org/2005/Atom"})) > 0
    _edgar_sic_cache[sic_code] = exists
    return exists


def _build_prompt(company_name: str, candidates: List[Dict], chunk_texts: List[str]) -> str:
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


def validate_operates_in(extracted_json_path: str, llm_fn=None) -> Optional[Dict]:
    if llm_fn is None:
        raise ValueError("An llm_fn callable must be provided.")

    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data.get('main_company', 'the Company')
    relations = data.get('relations', {}).get('OPERATES_IN', [])

    if not relations:
        print("No OPERATES_IN relations found.")
        return None

    seen_sic: set = set()
    candidates: List[tuple] = []  # (cand_dict, relation)
    chunk_texts: List[str] = []

    for rel in relations:
        sic = rel.get('sic', {})
        code = str(sic.get('code', '')).strip()
        if code and code not in seen_sic:
            seen_sic.add(code)
            industry = (sic.get('industry') or rel['tgt']['name']).strip()
            sector = (sic.get('sector') or rel['tgt'].get('properties', {}).get('sector', '')).strip()
            candidates.append(({'industry': industry, 'sector': sector, 'sic_code': code}, rel))
        chunk = rel.get('chunk_text', '').strip()
        if chunk:
            chunk_texts.append(chunk)

    if not candidates:
        raise ValueError("No SIC-backed candidates found.")

    print(f"Verifying {len(candidates)} candidate SIC codes against SEC EDGAR...")
    valid = [(c, r) for c, r in candidates if _sic_exists_in_edgar(c['sic_code'])]
    for c, _ in valid:
        print(f"  ✓ SIC {c['sic_code']} ({c['industry']})")
    removed = len(candidates) - len(valid)
    if removed:
        print(f"  ✗ {removed} SIC code(s) not found in EDGAR, removed")

    if not valid:
        raise ValueError("All SIC candidates were rejected by EDGAR validation.")

    if len(valid) == 1:
        print(f"Single candidate — keeping: {valid[0][0]}")
        return {'main_company': company_name, 'validated_relation': valid[0][1]}

    cands = [c for c, _ in valid]
    print(f"Found {len(cands)} candidates. Asking LLM to pick the best one...")
    for i, c in enumerate(cands, 1):
        print(f"  {i}. {c['industry']} | sector={c['sector']} | sic={c['sic_code']}")

    raw = llm_fn(_build_prompt(company_name, cands, chunk_texts)).strip()
    match = re.search(r'\d+', raw)
    if not match:
        raise ValueError(f"LLM returned unexpected output: {raw!r}")

    chosen_idx = int(match.group()) - 1
    if not (0 <= chosen_idx < len(valid)):
        raise ValueError(f"LLM chose out-of-range index {chosen_idx + 1} (have {len(valid)} candidates)")

    winner_cand, winner_rel = valid[chosen_idx]
    print(f"Selected: industry='{winner_cand['industry']}', sic_code='{winner_cand['sic_code']}'")
    return {'main_company': company_name, 'validated_relation': winner_rel}



