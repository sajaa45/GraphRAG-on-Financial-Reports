import requests
import json
import time
from datetime import datetime
import urllib3

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
SIC_CODES = ['1311']  # Change as needed
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
FORM_TYPE = "10-K"
MAX_COMPANIES = 100

headers = {"User-Agent": "User (your_email@example.com)"}

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


def analyze_company_covenants(cik, company_name):
    """Analyze a single company for financial covenant tags."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {})
        
        results = {}
        
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
                        
                        # Get all unit data
                        for unit_name, entries in units.items():
                            tag_data['units'][unit_name] = []
                            for entry in entries:
                                tag_data['units'][unit_name].append({
                                    'end': entry.get('end'),
                                    'val': entry.get('val'),
                                    'accn': entry.get('accn'),
                                    'fy': entry.get('fy'),
                                    'fp': entry.get('fp'),
                                    'form': entry.get('form'),
                                    'filed': entry.get('filed')
                                })
                        
                        results[covenant_type][tag_name] = tag_data
        
        return results
    
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
    
    # Fetch companies
    companies = fetch_companies_by_sic(SIC_CODES, START_DATE, END_DATE, FORM_TYPE, MAX_COMPANIES)
    
    if not companies:
        print("No companies found. Exiting.")
        return
    
    all_results = {}
    companies_with_covenants = []
    companies_without_covenants = []
    companies_with_errors = []
    
    print("=" * 80)
    print("ANALYZING COMPANIES")
    print("=" * 80)
    print()
    
    for idx, company in enumerate(companies, 1):
        cik = company['cik']
        name = company['name']
        ticker = company['ticker']
        
        print(f"[{idx}/{len(companies)}] {name} ({ticker}) - CIK: {cik}")
        
        results = analyze_company_covenants(cik, name)
        
        if results is None:
            companies_with_errors.append(company)
            print(f"  ⚠ Data not available or error occurred\n")
        elif results:
            # Count total tags found
            total_tags = sum(len(tags) for tags in results.values())
            companies_with_covenants.append({
                'company': company,
                'covenant_count': total_tags,
                'covenants': results
            })
            print(f"  ✓ Found {total_tags} covenant tag(s)")
            
            # Show which covenant types were found
            for covenant_type, tags in results.items():
                if tags:
                    print(f"    - {covenant_type}: {len(tags)} tag(s)")
            print()
        else:
            companies_without_covenants.append(company)
            print(f"  ✗ No covenant tags found\n")
        
        # Rate limiting - be respectful to SEC servers
        time.sleep(0.2)
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total companies analyzed: {len(companies)}")
    print(f"Companies with covenant tags: {len(companies_with_covenants)}")
    print(f"Companies without covenant tags: {len(companies_without_covenants)}")
    print(f"Companies with errors: {len(companies_with_errors)}")
    print()
    
    # Detailed results for companies with covenants
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
            'sic_codes': SIC_CODES,
            'date_range': {'start': START_DATE, 'end': END_DATE},
            'form_type': FORM_TYPE,
            'analysis_date': datetime.now().isoformat(),
            'total_companies': len(companies),
            'companies_with_covenants': len(companies_with_covenants),
            'companies_without_covenants': len(companies_without_covenants),
            'companies_with_errors': len(companies_with_errors)
        },
        'companies_with_covenants': companies_with_covenants,
        'companies_without_covenants': companies_without_covenants,
        'companies_with_errors': companies_with_errors
    }
    
    output_file = f"financial_covenants_sic_{'_'.join(SIC_CODES)}_{START_DATE}_to_{END_DATE}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Complete results saved to: {output_file}")


if __name__ == "__main__":
    main()
