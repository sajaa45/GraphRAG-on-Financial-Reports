import os
import requests
import json
import time
from datetime import datetime
import urllib3
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define financial covenant tag groups
debt_ebitda_tags = [
    'DebtInstrumentCovenantComplianceRatioOfConsolidatedNetDebtToConsolidatedEBITDA',
    'NetDebtToEBITDARatio',
    'DebtToEbitda',
    'ActualDebtToEbitdaRatio',
    'DebtInstrumentCovenantDebtToEBITDARatio',
    'DebtInstrumentCovenantMaximumDebtToEBITDARatio',
    'DebtInstrumentCovenantNetDebtToEBITDARatio',
    'RatioOfIndebtednessToEBITDA',
    'DebtInstrumentCovenantConsolidatedIndebtednessToConsolidatedEBITDAMaximum',
    'DebtToEbitdaRatioMaximum'
]

interest_coverage_ratio_tags = [
    'InterestCoverageRatio',
    'ConsolidatedInterestCoverageRatio',
    'DebtInstrumentCovenantActualInterestCoverageRatio',
    'DebtInstrumentCovenantComplianceConsolidatedInterestCoverageRatio',
    'DebtInstrumentCovenantInterestCoverageRatio',
    'DebtInstrumentCovenantMinimumInterestCoverageRatio',
    'DebtInstrumentConsolidatedInterestCoverageRatio',
    'DebtInstrumentCovenantComplianceMinimumInterestCoverageRatio',
    'LineofCreditFacilityCovenantComplianceActualInterestCoverageRatio',
    'DebtInstrumentCovenantConsolidatedInterestCoverageRatioMinimum'
]

current_ratio_tags = [
    'CurrentRatio',
    'DebtServiceCoverageRatioCurrentFiscalYear',
    'DebtInstrumentCovenantComplianceCurrentRatio',
    'DebtInstrumentCovenantCurrentRatio',
    'MinimumCurrentRatioPerCreditFacility',
    'MinimumCurrentRatioRequired',
    'DebtCovenantCurrentRatio',
    'LineOfCreditFacilityCovenantTermsMinimumCurrentRatio',
    'DebtInstrumentCovenantCurrentRatioMinimum',
    'FinancialCovenantsCurrentAssetsToCurrentLiabilitiesRatio',
    'DebtInstrumentCovenantComplianceCurrentRatio'
]

debt_to_equity_tags = [
    'DebtToEquityRatio',
    'DebtEquityRatio',
    'NetDebtToEquityRatio',
    'DebtInstrumentCovenantDebtToEquityRatio',
    'DebtInstrumentCovenantDebtToEquityRatioMaximum',
    'RatioOfDebtToDebtPlusEquity',
    'DebtCovenantRatioOfDebtToDebtPlusEquity',
    'TargetedMaximumDebtToEquityRatio',
    'LineofCreditFacilityCovenantTermsMaximumDebttoEquityRatio',
    'DebtInstrumentCovenantDebtToAllowanceAndEquityRatioMaximum'
]

credit_risk_tags = [
    'MaximumExposureToCreditRisk',
    'NetExposureToCreditRisk',
    'ConcentrationRiskCreditRiskFinancialInstrumentMaximumExposure',
    'MaximumExposureToCreditRiskOfFinancialAssets',
    'CreditRiskExposureToBankingAndFinancialSectorPercentage'
]

# Configuration
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
FORM_TYPE = "10-K"
TARGET_FISCAL_YEAR = 2024 
MAX_COMPANIES = 3  

headers = {"User-Agent": "User (your_email@example.com)"}


def get_companies_from_neo4j():
    """Get peer companies already in the Neo4j graph (excluding target company)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)-[:FACES_RISK]->(r:Risk)
                WHERE c.is_peer = true OR (NOT c:TargetCompany AND NOT c.is_target = true)
                WITH c, count(r) AS risk_count
                WHERE risk_count >= 10
                RETURN c.name AS name, c.cik AS cik, c.ticker AS ticker, risk_count
                ORDER BY risk_count DESC
                """
            )
            companies = []
            for record in result:
                name = record.get("name")
                cik = record.get("cik")
                ticker = record.get("ticker", "N/A")
                
                if name:
                    # If CIK is not stored, we'll need to look it up or skip
                    companies.append({
                        'name': name,
                        'cik': str(cik).zfill(10) if cik else None,
                        'ticker': ticker or 'N/A',
                        'source': 'neo4j'
                    })
            
            if companies:
                print(f"✓ Found {len(companies)} peer companies in Neo4j graph:")
                for c in companies:
                    cik_str = c['cik'] if c['cik'] else 'No CIK'
                    print(f"  - {c['name']} ({c['ticker']}) - {cik_str}")
                print()
            else:
                print("⚠ No peer companies found in Neo4j graph\n")
            
            return companies
    except Exception as e:
        print(f"⚠ Error querying Neo4j for companies: {e}\n")
        return []
    finally:
        driver.close()


def get_sic_from_neo4j() -> str:
    """Query Neo4j for the SIC code of the target company (is_target=true)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Try multiple query patterns
            queries = [
                # Pattern 1: TargetCompany node type
                """
                MATCH (c:TargetCompany)-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 2: Company with is_target property
                """
                MATCH (c:Company {is_target: true})-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 3: Direct HAS_SIC_CODE from TargetCompany
                """
                MATCH (c:TargetCompany)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 4: Direct HAS_SIC_CODE from Company
                """
                MATCH (c:Company {is_target: true})-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 5: Any Company with SIC code
                """
                MATCH (c:Company)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 4: Any SIC code in the database
                """
                MATCH (s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """
            ]
            
            for i, query in enumerate(queries, 1):
                result = session.run(query)
                record = result.single()
                if record and record["sic_code"]:
                    code = str(record["sic_code"]).strip()
                    print(f"✓ SIC code from Neo4j (pattern {i}): {code}")
                    driver.close()
                    return code
        
        driver.close()
    except Exception as e:
        print(f"⚠ Could not connect to Neo4j: {e}")
    
    # Fallback to default SIC code for Oil & Gas
    print("⚠ No SIC code found in Neo4j, using default: 1311 (Oil & Gas)")
    return "1311"

def get_target_company_metrics():
    """Query Neo4j for metrics found in the target company."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            # Try both TargetCompany and Company node types
            result = session.run(
                """
                MATCH (c)-[:HAS_METRIC]->(m:Metric)
                WHERE c:TargetCompany OR (c:Company AND c.is_target = true)
                RETURN m.name AS metric_name, m.metric_type AS metric_type, m.year AS year
                """
            )
            metrics = []
            seen_types = set()
            
            for record in result:
                metric_name = record["metric_name"]
                metric_type = record["metric_type"]
                year = record.get("year")
                
                # If metric_type is not set, try to extract it from the name
                if not metric_type and metric_name:
                    # Parse "EBITDA (2024)" -> type: "EBITDA", year: "2024"
                    if '(' in metric_name:
                        metric_type = metric_name.split('(')[0].strip()
                        year_part = metric_name.split('(')[1].split(')')[0].strip()
                        if year_part.isdigit():
                            year = int(year_part)
                    else:
                        metric_type = metric_name
                
                # Convert year to int if it's a string
                if year and isinstance(year, str) and year.isdigit():
                    year = int(year)
                
                # Only add unique metric types (not each year)
                if metric_type and metric_type not in seen_types:
                    metrics.append({
                        "name": metric_name,
                        "type": metric_type,
                        "year": year  # Will be None if not found, then use default
                    })
                    seen_types.add(metric_type)
            
            if metrics:
                print(f"✓ Found {len(metrics)} unique metric types in target company:")
                for m in metrics:
                    year_str = f" (year: {m['year']})" if m['year'] else " (using default year)"
                    print(f"  - {m['type']}{year_str}")
                print()
            else:
                print("⚠ No metrics found in target company\n")
            
            return metrics
    finally:
        driver.close()

# Combine all tag groups
all_covenant_tags = {
    'Debt/EBITDA Ratio': debt_ebitda_tags,
    'Interest Coverage Ratio': interest_coverage_ratio_tags,
    'Current Ratio': current_ratio_tags,
    'Debt to Equity': debt_to_equity_tags,
    'Credit Risk': credit_risk_tags
}


def fetch_companies_by_sic(sic_codes, start_date, end_date, form_type, size=100):
    """Fetch companies from SEC EDGAR search index by SIC code and date range."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": "",
        "forms": form_type,
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "form": form_type,
        "sics": sic_codes,
        "size": size
    }
    
    print(f"Fetching companies with SIC codes: {sic_codes}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Form type: {form_type}\n")
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        hits = data.get('hits', {}).get('hits', [])
        companies = []
        
        for hit in hits:
            source = hit.get('_source', {})
            cik = source.get('ciks', [None])[0]
            if cik:
                companies.append({
                    'cik': str(cik).zfill(10),
                    'name': source.get('display_names', ['Unknown'])[0],
                    'ticker': source.get('tickers', ['N/A'])[0] if source.get('tickers') else 'N/A',
                    'filing_date': source.get('file_date', 'N/A')
                })
        
        print(f"Found {len(companies)} companies\n")
        return companies
    
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return []


def analyze_company_covenants(cik, company_name, target_metrics=None, default_fiscal_year=2024):
    """Analyze a single company for financial covenant tags and target metrics.
    
    Args:
        cik: Company CIK number
        company_name: Company name
        target_metrics: List of target metrics to match (with optional year per metric)
        default_fiscal_year: Default fiscal year to use if metric doesn't specify one
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {})
        
        results = {}
        target_metric_results = {}
        
        # Search through all taxonomies
        for taxonomy_name, taxonomy_data in facts.items():
            for tag_name, tag_content in taxonomy_data.items():
                # Check if this tag matches any of our covenant tags
                for covenant_type, tag_list in all_covenant_tags.items():
                    if tag_name in tag_list:
                        if covenant_type not in results:
                            results[covenant_type] = {}
                        
                        # Extract the data
                        units = tag_content.get('units', {})
                        tag_data = {
                            'taxonomy': taxonomy_name,
                            'tag': tag_name,
                            'label': tag_content.get('label', ''),
                            'description': tag_content.get('description', ''),
                            'units': {}
                        }
                        
                        # Get all unit data and filter by fiscal year and period
                        for unit_name, entries in units.items():
                            filtered_entries = []
                            for entry in entries:
                                # Filter: only FY (full year) and matching fiscal year
                                fp = entry.get('fp', '')
                                fy = entry.get('fy')
                                form = entry.get('form', '')
                                if fp == 'FY' and form == '10-K' and fy == default_fiscal_year:
                                    filtered_entries.append({
                                        'end': entry.get('end'),
                                        'val': entry.get('val'),
                                        'accn': entry.get('accn'),
                                        'fy': fy,
                                        'fp': fp,
                                        'form': form,
                                        'filed': entry.get('filed', '')
                                    })
                            
                            if filtered_entries:
                                tag_data['units'][unit_name] = filtered_entries
                        
                        if tag_data['units']:
                            results[covenant_type][tag_name] = tag_data
                
                if target_metrics:
                    for metric in target_metrics:
                        metric_type = metric.get('type', '')
                        if not metric_type:
                            continue
                        
                        target_year = metric.get('year') or default_fiscal_year
                        
                        normalized_metric = ''.join(c.lower() for c in metric_type if c.isalnum())
                        normalized_tag = tag_name.lower()
                        
                        match = False
                        if normalized_metric in normalized_tag or normalized_tag in normalized_metric:
                            match = True
                        elif len(normalized_metric) > 3:
                            metric_words = [w for w in metric_type.lower().split() if len(w) > 2]
                            if metric_words and all(w in normalized_tag for w in metric_words):
                                match = True
                        
                        if match:
                            if metric_type not in target_metric_results:
                                target_metric_results[metric_type] = []
                            
                            units = tag_content.get('units', {})
                            tag_data = {
                                'taxonomy': taxonomy_name,
                                'tag': tag_name,
                                'label': tag_content.get('label', ''),
                                'description': tag_content.get('description', ''),
                                'metric_type': metric_type,
                                'target_year': target_year,
                                'units': {}
                            }
                            
                            # Get all unit data and filter by fiscal year and period
                            for unit_name, entries in units.items():
                                filtered_entries = []
                                for entry in entries:
                                    # Filter: only FY (full year) and matching fiscal year
                                    fp = entry.get('fp', '')
                                    fy = entry.get('fy')
                                    form = entry.get('form', '')
                                    
                                    if fp == 'FY' and form == '10-K' and fy == target_year:
                                        filtered_entries.append({
                                            'end': entry.get('end'),
                                            'val': entry.get('val'),
                                            'accn': entry.get('accn'),
                                            'fy': fy,
                                            'fp': fp,
                                            'form': form,
                                            'filed': entry.get('filed', '')
                                        })
                                
                                if filtered_entries:
                                    tag_data['units'][unit_name] = filtered_entries
                            
                            if tag_data['units']:
                                target_metric_results[metric_type].append(tag_data)
        
        return results, target_metric_results
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None  # Company data not available
        print(f"  HTTP Error for {company_name}: {e}")
        return None
    except Exception as e:
        print(f"  Error analyzing {company_name}: {e}")
        return None


def main():
    """Main execution function."""
    print("=" * 80)
    print("FINANCIAL COVENANT ANALYSIS - BATCH PROCESSING")
    print("=" * 80)
    print()
    
    # Get target company metrics
    try:
        target_metrics = get_target_company_metrics()
    except Exception as e:
        print(f"Error getting target metrics: {e}\n")
        target_metrics = []
    
    print("=" * 80)
    print("STEP 1: CHECKING NEO4J GRAPH FOR PEER COMPANIES")
    print("=" * 80)
    print()
    
    companies = get_companies_from_neo4j()
    
    companies_with_cik = [c for c in companies if c.get('cik')]
    companies_without_cik = [c for c in companies if not c.get('cik')]
    
    if companies_without_cik:
        print(f"⚠ Skipping {len(companies_without_cik)} companies without CIK:")
        for c in companies_without_cik:
            print(f"  - {c['name']}")
        print()
    
    if len(companies_with_cik) < MAX_COMPANIES:
        needed = MAX_COMPANIES - len(companies_with_cik)
        print("=" * 80)
        print(f"STEP 2: FETCHING ADDITIONAL COMPANIES FROM SEC (need {needed} more)")
        print("=" * 80)
        print()
        
        try:
            sic_code = get_sic_from_neo4j()
            sic_codes = [sic_code]
        except Exception as e:
            print(f"Error getting SIC from Neo4j: {e}")
            print("Falling back to default SIC code: 1311\n")
            sic_codes = ['1311']
        
        sec_companies = fetch_companies_by_sic(sic_codes, START_DATE, END_DATE, FORM_TYPE, needed)
        
        for c in sec_companies:
            c['source'] = 'sec'
        
        existing_ciks = {c['cik'] for c in companies_with_cik}
        new_companies = [c for c in sec_companies if c['cik'] not in existing_ciks]
        
        companies_with_cik.extend(new_companies[:needed])
        
        print(f"✓ Added {len(new_companies[:needed])} companies from SEC")
        print(f"✓ Total companies to analyze: {len(companies_with_cik)}\n")
    else:
        print(f"✓ Using {len(companies_with_cik)} companies from Neo4j graph (limit: {MAX_COMPANIES})\n")
        companies_with_cik = companies_with_cik[:MAX_COMPANIES]
    
    companies = companies_with_cik
    
    if not companies:
        print("No companies to analyze. Exiting.")
        return
    
    all_results = {}
    companies_with_covenants = []
    companies_without_covenants = []
    companies_with_errors = []
    companies_with_target_metrics = []
    
    print("=" * 80)
    print("ANALYZING COMPANIES")
    print("=" * 80)
    print()
    
    for idx, company in enumerate(companies, 1):
        cik = company['cik']
        name = company['name']
        ticker = company['ticker']
        
        print(f"[{idx}/{len(companies)}] {name} ({ticker}) - CIK: {cik}")
        
        covenant_results, metric_results = analyze_company_covenants(
            cik, name, target_metrics, TARGET_FISCAL_YEAR
        )
        
        if covenant_results is None and metric_results is None:
            companies_with_errors.append(company)
            print(f"  ⚠ Data not available or error occurred\n")
        else:
            has_covenants = bool(covenant_results)
            has_metrics = bool(metric_results)
            
            if has_covenants or has_metrics:
                # Count total tags found
                total_covenant_tags = sum(len(tags) for tags in covenant_results.values()) if covenant_results else 0
                total_metric_matches = sum(len(matches) for matches in metric_results.values()) if metric_results else 0
                
                company_data = {
                    'company': company,
                    'covenant_count': total_covenant_tags,
                    'covenants': covenant_results,
                    'target_metric_count': total_metric_matches,
                    'target_metrics': metric_results
                }
                
                companies_with_covenants.append(company_data)
                
                if has_covenants:
                    print(f"  ✓ Found {total_covenant_tags} covenant tag(s)")
                    for covenant_type, tags in covenant_results.items():
                        if tags:
                            print(f"    - {covenant_type}: {len(tags)} tag(s)")
                
                if has_metrics:
                    print(f"  ✓ Found {total_metric_matches} target metric match(es)")
                    for metric_name, matches in metric_results.items():
                        if matches:
                            print(f"    - {metric_name}: {len(matches)} match(es)")
                    companies_with_target_metrics.append(company_data)
                
                print()
            else:
                companies_without_covenants.append(company)
                print(f"  ✗ No covenant tags or target metrics found\n")
        
        time.sleep(0.2)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    neo4j_count = sum(1 for c in companies if c.get('source') == 'neo4j')
    sec_count = sum(1 for c in companies if c.get('source') == 'sec')
    print(f"Companies from Neo4j graph: {neo4j_count}")
    print(f"Companies from SEC EDGAR: {sec_count}")
    print(f"Total companies analyzed: {len(companies)}")
    print(f"Companies with covenant tags or metrics: {len(companies_with_covenants)}")
    print(f"Companies with target metrics: {len(companies_with_target_metrics)}")
    print(f"Companies without covenant tags: {len(companies_without_covenants)}")
    print(f"Companies with errors: {len(companies_with_errors)}")
    print()
    
    if companies_with_covenants:
        print("=" * 80)
        print("COMPANIES WITH FINANCIAL COVENANTS")
        print("=" * 80)
        print()
        
        for item in companies_with_covenants:
            company = item['company']
            print(f"{company['name']} ({company['ticker']}) - CIK: {company['cik']}")
            print(f"  Total covenant tags: {item['covenant_count']}")
            
            for covenant_type, tags in item['covenants'].items():
                if tags:
                    print(f"  {covenant_type}:")
                    for tag_name, tag_data in tags.items():
                        print(f"    - {tag_name}")
                        # Show latest value
                        for unit_name, entries in tag_data['units'].items():
                            if entries:
                                latest = entries[-1]
                                print(f"      Latest: {latest['val']} ({unit_name}) as of {latest['end']}")
                                break
            print()
    
    # Save comprehensive results
    output_data = {
        'metadata': {
            'neo4j_companies': neo4j_count,
            'sec_companies': sec_count,
            'target_metrics_searched': [m['name'] for m in target_metrics] if target_metrics else [],
            'date_range': {'start': START_DATE, 'end': END_DATE},
            'form_type': FORM_TYPE,
            'analysis_date': datetime.now().isoformat(),
            'total_companies': len(companies),
            'companies_with_covenants': len(companies_with_covenants),
            'companies_with_target_metrics': len(companies_with_target_metrics),
            'companies_without_covenants': len(companies_without_covenants),
            'companies_with_errors': len(companies_with_errors)
        },
        'companies_with_covenants': companies_with_covenants,
        'companies_with_target_metrics': companies_with_target_metrics,
        'companies_without_covenants': companies_without_covenants,
        'companies_with_errors': companies_with_errors
    }
    
    output_file = f"financial_covenants_analysis_{START_DATE}_to_{END_DATE}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Complete results saved to: {output_file}")


if __name__ == "__main__":
    main()
