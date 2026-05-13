"""
Test file for get_companies_from_api function
You can modify the parameters below to test different scenarios
"""

import sys
import os

# Add the parent directory to the path so we can import from peers_sec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peers_sec.fetch_and_extract_risks import get_companies_from_api
import json


def test_get_companies():
    sic_codes = ['2869']  
    size = 50  
    print("Testing get_companies_from_api function")
    print(f"  SIC Codes: {sic_codes}")
    print(f"  Size: {size}")
    print()
    
    # Call the function
    companies = get_companies_from_api(
        sic_codes=sic_codes,
        size=size
    )
    
    # Display results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total companies found: {len(companies)}")
    print()
    
    if companies:
        print("First 10 companies:")
        print("-" * 60)
        for i, company in enumerate(companies[:10], 1):
            print(f"{i}. {company['name']}")
            print(f"   CIK: {company['cik']}")
            print(f"   SIC: {company['sic']}")
            print()
        
        # Save to file
        output_file = 'test_companies_output.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "parameters": {
                    "sic_codes": sic_codes,
                    "size": size
                },
                "total_count": len(companies),
                "companies": companies
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Full results saved to: {output_file}")
    else:
        print("No companies found!")
    
    print("=" * 60)
    
    return companies


if __name__ == "__main__":
    # Run the test
    companies = test_get_companies()
