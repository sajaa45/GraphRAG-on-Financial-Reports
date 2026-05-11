import os
import requests
import re
import json
import time
from bs4 import BeautifulSoup
import urllib3
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "YourName your_email@example.com"}


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


def get_companies_from_api(sic_codes=['1311'], start_date='2023-01-01', end_date='2024-01-01', size=100):
    """Fetch companies from EDGAR company browse endpoint for given SIC code(s)."""
    if isinstance(sic_codes, str):
        sic_codes = [sic_codes]

    print(f"  Querying EDGAR company browse for SIC codes: {sic_codes}")

    seen_ciks = set()
    companies = []

    for sic in sic_codes:
        try:
            params = {
                "action": "getcompany",
                "SIC": sic,
                "type": "10-K",
                "dateb": "",
                "owner": "include",
                "count": size,
                "search_text": "",
                "output": "atom",
            }
            response = requests.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params=params,
                headers=HEADERS,
                verify=False,
            )
            print(f"  SIC {sic} — status: {response.status_code}")

            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            print(f"  SIC {sic} — found {len(entries)} entries")

            for entry in entries:
                # CIK is embedded in the <id> tag in format: urn:tag:www.sec.gov:cik=0001311901
                id_text = entry.findtext("atom:id", default="", namespaces=ns)
                cik_match = re.search(r"cik[=:](\d+)", id_text, re.IGNORECASE)
                if not cik_match:
                    continue
                cik = cik_match.group(1)
                if cik in seen_ciks:
                    continue
                seen_ciks.add(cik)

                # Company name is in the <content><company-info name="..."> structure
                name = "N/A"
                content = entry.find("atom:content", ns)
                if content is not None:
                    company_info = content.find("company-info")
                    if company_info is not None:
                        name = company_info.get("name", "N/A")
                
                # Fallback to title if name not found
                if name == "N/A" or name.startswith("ARRAY("):
                    title = entry.findtext("atom:title", default="", namespaces=ns)
                    if title and not title.startswith("ARRAY("):
                        name = title.split(" (")[0]

                companies.append({
                    "name": name,
                    "cik": cik,
                    "ticker": "N/A",
                    "filing_date": "N/A",
                    "sic": sic,
                })

        except Exception as e:
            print(f"  ✗ Error fetching SIC {sic}: {e}")

    print(f"  Total unique companies found: {len(companies)}")
    return companies


def get_10k_filings(cik: str, limit: int = 1):
    cik_padded = cik.zfill(10)
    data = requests.get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json", headers=HEADERS, verify=False).json()
    filings = data["filings"]["recent"]
    results = [
        {
            "accession_raw": filings["accessionNumber"][i],
            "date": filings["filingDate"][i],
            "primary_document": filings["primaryDocument"][i],
        }
        for i, form in enumerate(filings["form"])
        if form == "10-K"
    ][:limit]
    return results, cik_padded


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


def get_risk_factor_data(cik: str, company_name: str):
    filings, cik_padded = get_10k_filings(cik)
    if not filings:
        return None

    filing = filings[0]
    doc_url = get_doc_url(cik_padded, filing["accession_raw"], filing["primary_document"])
    risk_text = extract_risk_factors(doc_url)
    individual_risks = split_risks(risk_text)

    print(f"  chars={len(risk_text)}, words={len(risk_text.split())}, risks={len(individual_risks)}")
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


def process_companies_from_api(sic_codes=['1311'], start_date='2023-01-01', end_date='2024-01-01',
                               size=100, delay=0.5, output_file='all_companies_risks.json'):
    """Process companies from SEC API for given SIC code(s)."""
    companies = get_companies_from_api(sic_codes, start_date, end_date, size)
    print(f"Fetched {len(companies)} companies for SIC code(s): {sic_codes}")

    with open('companies_list.json', 'w', encoding='utf-8') as f:
        json.dump({"total_count": len(companies), "companies": companies, "sic_codes": sic_codes}, f, indent=2, ensure_ascii=False)

    results = []
    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] {company['name']}")
        try:
            data = get_risk_factor_data(company['cik'], company['name'])
            if data and data['length']['words'] > 5000:
                results.append(data)
                print(f"  Added ({len(results)}/3)")
                if len(results) >= 3:
                    break
            elif data:
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
    sic_codes = get_sic_from_neo4j()
    process_companies_from_api(sic_codes=sic_codes, start_date='2023-01-01', end_date='2024-01-01', size=100, delay=0.5)
