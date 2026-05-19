import os
import requests
import re
import json
import time
from bs4 import BeautifulSoup
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
FISCAL_YEAR = 2024  # Extract only this fiscal year


def get_sic_from_neo4j() -> list:
    """Query Neo4j for the SIC code(s) of the target company (is_target=true)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Try multiple query patterns
            queries = [
                # Pattern 1: Direct path through Industry
                """
                MATCH (c:Company {is_target: true})-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """,
                # Pattern 2: Direct HAS_SIC_CODE from Company
                """
                MATCH (c:Company {is_target: true})-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """,
                # Pattern 3: Any Company with SIC code
                """
                MATCH (c:Company)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """,
                # Pattern 4: Any SIC code in the database
                """
                MATCH (s:SICCode)
                RETURN DISTINCT s.code AS sic_code
                """
            ]
            
            for i, query in enumerate(queries, 1):
                result = session.run(query)
                records = list(result)
                if records:
                    codes = [str(record["sic_code"]).strip() for record in records if record["sic_code"]]
                    if codes:
                        print(f"✓ SIC code(s) from Neo4j (pattern {i}): {codes}")
                        driver.close()
                        return codes
        
        driver.close()
    except Exception as e:
        print(f"⚠ Could not connect to Neo4j: {e}")
    
    print("⚠ No SIC code found in Neo4j, using default: ['1311'] (Oil & Gas)")
    return ["1311"]


def get_companies_from_api(sic_codes=['1311'], fiscal_year=FISCAL_YEAR, size=100):
    """Fetch companies from EDGAR full-text search index filtered by SIC code and fiscal year."""
    if isinstance(sic_codes, str):
        sic_codes = [sic_codes]

    # Convert fiscal year to date range for API (filings are typically made in the following year)
    start_date = f"{fiscal_year}-01-01"
    end_date = f"{fiscal_year}-12-31"
    
    print(f"  Querying EDGAR search index for SIC codes: {sic_codes}, fiscal year {fiscal_year}")

    params = {
        "q": "",
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "form": "10-K",
        "sics": sic_codes,
        "size": size,
    }

    try:
        response = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            verify=False,
        )
        print(f"  Status: {response.status_code}")
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])

        seen_ciks = set()
        companies = []
        for hit in hits:
            source = hit.get("_source", {})
            cik = source.get("ciks", [None])[0]
            if not cik or cik in seen_ciks:
                continue
            seen_ciks.add(cik)
            name = source.get("display_names", ["Unknown"])[0]
            ticker = source.get("tickers", ["N/A"])[0] if source.get("tickers") else "N/A"
            companies.append({
                "name": name,
                "cik": str(cik).zfill(10),
                "ticker": ticker,
                "filing_date": source.get("file_date", "N/A"),
            })

        print(f"  Total unique companies found: {len(companies)}")
        return companies

    except Exception as e:
        print(f"  ✗ Error fetching companies: {e}")
        return []


def get_10k_filings(cik: str, fiscal_year: int = None, limit: int = 1):
    cik_padded = cik.zfill(10)
    data = requests.get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json", headers=HEADERS, verify=False).json()
    company_name = data.get("name", "N/A")
    filings = data["filings"]["recent"]
    
    results = []
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            filing_date = filings["filingDate"][i]
            
            # Filter by fiscal year if provided
            # Extract year from filing date (format: YYYY-MM-DD)
            if fiscal_year:
                filing_year = int(filing_date.split('-')[0])
                if filing_year != fiscal_year:
                    continue
            
            results.append({
                "accession_raw": filings["accessionNumber"][i],
                "date": filing_date,
                "primary_document": filings["primaryDocument"][i],
            })
            
            if len(results) >= limit:
                break
    
    return results, cik_padded, company_name


def get_doc_url(cik_padded, accession_raw, primary_document):
    accession = accession_raw.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{accession}/{primary_document}"


def extract_risk_factors(doc_url):
    resp = requests.get(doc_url, headers=HEADERS, verify=False)
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


def get_risk_factor_data(cik: str, company_name: str, fiscal_year: int = None):
    filings, cik_padded, real_name = get_10k_filings(cik, fiscal_year)
    if not filings:
        return None

    if real_name and real_name != "N/A":
        company_name = real_name

    filing = filings[0]
    doc_url = get_doc_url(cik_padded, filing["accession_raw"], filing["primary_document"])
    risk_text = extract_risk_factors(doc_url)
    individual_risks = split_risks(risk_text)

    print(f"  filing_date={filing['date']}, chars={len(risk_text)}, words={len(risk_text.split())}, risks={len(individual_risks)}")
    return {
        "company_name": company_name,
        "cik": cik,
        "filing_date": filing["date"],
        "document_url": doc_url,
        "section": "Item 1A - Risk Factors",
        "text": risk_text,
        "individual_risks": individual_risks,
        "risk_count": len(individual_risks),
        "length": {"characters": len(risk_text), "words": len(risk_text.split())},
    }


def is_target_company(company_name: str, target_name: str, threshold: float = 15.0) -> bool:
    """
    Check if company_name matches target_name using BM25 similarity.
    Returns True if the BM25 score exceeds the threshold.
    
    Args:
        company_name: Name to check
        target_name: Target company name from Neo4j
        threshold: BM25 score threshold (default 15.0, higher = stricter matching)
    """
    if not target_name or not company_name:
        return False
    
    # Tokenize both names
    target_tokens = target_name.lower().split()
    company_tokens = company_name.lower().split()
    
    # Create BM25 index with just the target name
    bm25 = BM25Okapi([target_tokens])
    
    # Score the company name against target
    scores = bm25.get_scores(company_tokens)
    score = scores[0] if len(scores) > 0 else 0.0
    
    return score >= threshold


def get_target_name() -> str | None:
    """Return the name of the target company so it can be excluded from peers."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            for query in [
                "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1",
                "MATCH (c:Company {is_target: true}) RETURN c.name AS name LIMIT 1",
            ]:
                record = session.run(query).single()
                if record and record["name"]:
                    return record["name"].strip().lower()
        driver.close()
    except Exception:
        pass
    return None


def process_companies_from_api(sic_codes=['1311'], fiscal_year=FISCAL_YEAR, size=100, delay=0.5, 
                               output_file='peers_sec/FACES_RISK/companies_risks.json'):
    """Process companies from SEC API for given SIC code(s) and fiscal year."""
    print(f"Filtering for fiscal year: {fiscal_year}")
    companies = get_companies_from_api(sic_codes, fiscal_year, size)
    print(f"Fetched {len(companies)} companies for SIC code(s): {sic_codes} [fiscal year {fiscal_year}]")

    target_name = get_target_name()
    if target_name:
        print(f"Excluding target company: {target_name}")
        companies = [c for c in companies if not is_target_company(c['name'], target_name, threshold=15.0)]

    with open('peers_sec/FACES_RISK/companies_list.json', 'w', encoding='utf-8') as f:
        json.dump({"total_count": len(companies), "companies": companies, "sic_codes": sic_codes, "fiscal_year": fiscal_year}, f, indent=2, ensure_ascii=False)

    results = []
    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] CIK {company['cik']}")
        try:
            data = get_risk_factor_data(company['cik'], company['name'], fiscal_year)
            if data:
                # Double-check: exclude target company using BM25 after fetching real company name
                if target_name and is_target_company(data['company_name'], target_name, threshold=15.0):
                    print(f"  Skipped — this is the target company: {data['company_name']}")
                    continue
                
                if data['risk_count'] < 7:
                    print(f"  Skipped — only {data['risk_count']} risks (minimum 10 required)")
                elif data['length']['words'] > 4800:
                    results.append(data)
                    print(f"  Added ({len(results)}/3)")
                    if len(results) >= 3:
                        break
                else:
                    print(f"  Skipped — only {data['length']['words']} words")
            else:
                print(f"  Skipped — no 10-K filing found for fiscal year {fiscal_year}")
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
    sic_codes = get_sic_from_neo4j()
    process_companies_from_api(sic_codes=sic_codes, fiscal_year=FISCAL_YEAR, size=100, delay=0.5)
