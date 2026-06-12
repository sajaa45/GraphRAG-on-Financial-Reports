import os
import requests
import re
import json
import time
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
import urllib3
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

# Load .env from project root (two levels up from this file)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "YourName your_email@example.com"}

# Configuration
FISCAL_YEAR = int(os.getenv("FISCAL_YEAR", 2024))


def get_sic_from_neo4j(driver, target_company: str = None) -> list:
    with driver.session() as session:
        if target_company:
            records = list(session.run(
                """
                MATCH (c:TargetCompany {name: $name})-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """,
                {"name": target_company},
            ))
        else:
            records = list(session.run(
                """
                MATCH (c:TargetCompany)-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """
            ))
    codes = [str(r["sic_code"]).strip() for r in records if r["sic_code"]]
    if not codes:
        raise ValueError("No SIC codes found in Neo4j for the target company.")
    print(f"✓ SIC code(s) from Neo4j: {codes}")
    return codes


def get_companies_from_api(sic_codes=['1311'], fiscal_year=FISCAL_YEAR, size=100):
    """Fetch companies with 10-K filing info directly from EDGAR search index."""
    if isinstance(sic_codes, str):
        sic_codes = [sic_codes]

    fiscal_year = int(fiscal_year)
    start_date = f"{fiscal_year}-07-01"
    end_date = f"{fiscal_year + 1}-06-30"

    print(f"  Querying EDGAR search index for SIC codes: {sic_codes}, fiscal year {fiscal_year}")

    params = {
        "q": "",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": "10-K",
        "sics": sic_codes,
        "size": size,
    }

    response = requests.get(
        "https://efts.sec.gov/LATEST/search-index",
        params=params,
        headers=HEADERS,
        verify=False,
    )
    response.raise_for_status()
    hits = response.json().get("hits", {}).get("hits", [])

    seen_ciks = set()
    companies = []
    for hit in hits:
        source = hit.get("_source", {})

        # Skip non-10-K forms that slip through the filter
        if source.get("form") != "10-K":
            continue

        cik = source.get("ciks", [None])[0]
        if not cik or cik in seen_ciks:
            continue
        seen_ciks.add(cik)

        # _id format: "{adsh}:{primary_document}" — adsh is the real company accession
        hit_id = hit.get("_id", "")
        if ":" not in hit_id:
            continue
        _, primary_document = hit_id.rsplit(":", 1)
        adsh = source.get("adsh", "")
        if not adsh:
            continue

        name = re.sub(r'\s*\(CIK\s+\d+\)\s*$', '', source.get("display_names", ["Unknown"])[0]).strip()
        ticker = source.get("tickers", ["N/A"])[0] if source.get("tickers") else "N/A"
        cik_padded = str(cik).zfill(10)
        accession = adsh.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{accession}/{primary_document}"

        companies.append({
            "name": name,
            "cik": cik_padded,
            "ticker": ticker,
            "filing_date": source.get("file_date", "N/A"),
            "period_of_report": source.get("period_ending"),
            "document_url": doc_url,
        })

    print(f"  Total unique companies found: {len(companies)}")
    return companies


def extract_risk_factors(doc_url):
    resp = requests.get(doc_url, headers=HEADERS, verify=False)
    resp.raise_for_status()
    text = BeautifulSoup(resp.content, "html.parser").get_text(separator="\n")
    pattern = re.compile(
        r'(item\s+1a[\.\s]*risk\s+factors)(.*?)(item\s+1b|item\s+2)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(text)
    if matches:
        return max(matches, key=lambda m: len(m[1]))[1].strip()
    return "Risk factors section not found."


def split_risks(text):
    return [r.strip() for r in re.split(r'\n\s*\n', text) if len(r.split()) > 30]


def get_risk_factor_data(company: dict):
    risk_text = extract_risk_factors(company["document_url"])
    individual_risks = split_risks(risk_text)
    print(f"  period_of_report={company['period_of_report']}, filing_date={company['filing_date']}, chars={len(risk_text)}, words={len(risk_text.split())}, risks={len(individual_risks)}")
    return {
        "company_name": company["name"],
        "cik": company["cik"],
        "period_of_report": company["period_of_report"],
        "filing_date": company["filing_date"],
        "document_url": company["document_url"],
        "section": "Item 1A - Risk Factors",
        "text": risk_text,
        "individual_risks": individual_risks,
        "risk_count": len(individual_risks),
        "length": {"characters": len(risk_text), "words": len(risk_text.split())},
    }


def is_target_company(company_name: str, target_name: str, threshold: float = 15.0) -> bool:
    if not target_name or not company_name:
        return False
    bm25 = BM25Okapi([target_name.lower().split()])
    score = bm25.get_scores(company_name.lower().split())[0]
    return score >= threshold


def get_target_name(driver) -> str | None:
    with driver.session() as session:
        record = session.run("MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1").single()
        if record and record["name"]:
            return record["name"].strip().lower()
    return None


def process_companies_from_api(driver, sic_codes=['1311'], fiscal_year=FISCAL_YEAR, size=100, delay=0.5,
                               output_file='peers_sec/FACES_RISK/companies_risks.json'):
    fiscal_year = int(fiscal_year)
    print(f"Filtering for fiscal year: {fiscal_year}")
    companies = get_companies_from_api(sic_codes, fiscal_year, size)
    print(f"Fetched {len(companies)} companies for SIC code(s): {sic_codes} [fiscal year {fiscal_year}]")

    target_name = get_target_name(driver)
    if target_name:
        print(f"Excluding target company: {target_name}")
        companies = [c for c in companies if not is_target_company(c['name'], target_name)]

    with open('peers_sec/FACES_RISK/companies_list.json', 'w', encoding='utf-8') as f:
        json.dump({"total_count": len(companies), "companies": companies, "sic_codes": sic_codes, "fiscal_year": fiscal_year}, f, indent=2, ensure_ascii=False)

    results = []
    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] CIK {company['cik']} — {company['name']}")
        try:
            data = get_risk_factor_data(company)
            if is_target_company(data['company_name'], target_name):
                print(f"  Skipped — this is the target company: {data['company_name']}")
                continue
            if data['risk_count'] < 7:
                print(f"  Skipped — only {data['risk_count']} risks")
            elif data['length']['words'] > 4800:
                results.append(data)
                print(f"  Added ({len(results)}/3)")
                if len(results) >= 3:
                    break
            else:
                print(f"  Skipped — only {data['length']['words']} words")
        except Exception as e:
            print(f"  Error: {e}")

        if i < len(companies):
            time.sleep(delay)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} companies saved to {output_file}")
    for i, c in enumerate(results, 1):
        print(f"  {i}. {c['company_name']} — {c['length']['words']} words, {c['risk_count']} risks")
    return results


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    try:
        sic_codes = get_sic_from_neo4j(driver)
        process_companies_from_api(driver, sic_codes=sic_codes, fiscal_year=FISCAL_YEAR, size=100, delay=0.5)
    finally:
        driver.close()
