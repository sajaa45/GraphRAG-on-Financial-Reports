"""
Debug script to see what the SEC API is actually returning
"""

import requests
import xml.etree.ElementTree as ET
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "YourName your_email@example.com"}

sic = '1311'
params = {
    "action": "getcompany",
    "SIC": sic,
    "type": "10-K",
    "dateb": "",
    "owner": "include",
    "count": 10,  # Just get 10 for debugging
    "search_text": "",
    "output": "atom",
}

response = requests.get(
    "https://www.sec.gov/cgi-bin/browse-edgar",
    params=params,
    headers=HEADERS,
    verify=False,
)

print(f"Status: {response.status_code}")
print("\n" + "="*60)
print("RAW XML RESPONSE (first 2000 chars):")
print("="*60)
print(response.text[:2000])
print("\n" + "="*60)
print("PARSING XML:")
print("="*60)

root = ET.fromstring(response.content)
ns = {"atom": "http://www.w3.org/2005/Atom"}
entries = root.findall("atom:entry", ns)

print(f"\nFound {len(entries)} entries")

if entries:
    print("\n" + "="*60)
    print("FIRST ENTRY DETAILS:")
    print("="*60)
    entry = entries[0]
    
    # Print all child elements
    for child in entry:
        tag = child.tag.replace("{http://www.w3.org/2005/Atom}", "")
        print(f"\n{tag}: {child.text}")
    
    # Try to find CIK
    print("\n" + "="*60)
    print("LOOKING FOR CIK:")
    print("="*60)
    
    id_text = entry.findtext("atom:id", default="", namespaces=ns)
    print(f"ID field: {id_text}")
    
    # Try different ways to find CIK
    import re
    cik_match = re.search(r"CIK=(\d+)", id_text)
    if cik_match:
        print(f"✓ Found CIK via regex: {cik_match.group(1)}")
    else:
        print("✗ No CIK found via regex in ID field")
    
    # Check for edgar:cik or other namespaces
    for child in entry:
        if 'cik' in child.tag.lower():
            print(f"Found CIK-related tag: {child.tag} = {child.text}")
