import requests
import re
import json
import time
from bs4 import BeautifulSoup
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "YourName your_email@example.com"}


def get_companies_from_api(sic_code='1311', start_date='2023-01-01', end_date='2024-01-01', size=100):
    params = {
        "forms": "10-K",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "sics": [sic_code],
        "size": size,
    }
    data = requests.get("https://efts.sec.gov/LATEST/search-index", params=params, headers=HEADERS, verify=False).json()
    companies = []
    for hit in data.get("hits", {}).get("hits", []):
        s = hit.get("_source", {})
        companies.append({
            "name": (s.get("display_names") or ["N/A"])[0],
            "cik": (s.get("ciks") or ["N/A"])[0],
            "ticker": (s.get("tickers") or ["N/A"])[0],
            "filing_date": s.get("file_date", "N/A"),
        })
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


def process_companies_from_api(sic_code='1311', start_date='2023-01-01', end_date='2024-01-01',
                               size=100, delay=0.5, output_file='all_companies_risks.json'):
    companies = get_companies_from_api(sic_code, start_date, end_date, size)
    print(f"Fetched {len(companies)} companies")

    with open('companies_list.json', 'w', encoding='utf-8') as f:
        json.dump({"total_count": len(companies), "companies": companies}, f, indent=2, ensure_ascii=False)

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
    process_companies_from_api(sic_code='1311', start_date='2023-01-01', end_date='2024-01-01', size=100, delay=0.5)
