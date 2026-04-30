from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import json

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)

# Load structured_risks.json to see what should be there
with open('structured_risks.json', 'r', encoding='utf-8') as f:
    structured_risks = json.load(f)

print(f"Companies in structured_risks.json: {len(structured_risks)}")
print(f"Total risks in JSON: {sum(len(c['risks']) for c in structured_risks)}")

with driver.session() as session:
    # Companies WITH risks
    result = session.run('''
        MATCH (c:Company)-[:FACES_RISK]->()
        RETURN c.name as name, c.is_target as is_target, c.is_peer as is_peer
        ORDER BY name
    ''')
    companies_with_risks = list(result)
    print(f"\nCompanies WITH FACES_RISK in Neo4j: {len(companies_with_risks)}")
    
    # Companies WITHOUT risks
    result = session.run('''
        MATCH (c:Company)
        WHERE NOT (c)-[:FACES_RISK]->()
        RETURN c.name as name, c.is_target as is_target, c.is_peer as is_peer
        ORDER BY name
        LIMIT 20
    ''')
    companies_without_risks = list(result)
    print(f"Companies WITHOUT FACES_RISK in Neo4j: {len(companies_without_risks)}")
    
    if companies_without_risks:
        print("\nSample companies WITHOUT risks:")
        for rec in companies_without_risks[:10]:
            print(f"  - {rec['name']} (target: {rec['is_target']}, peer: {rec['is_peer']})")
    
    # Check for name mismatches
    print("\n" + "="*60)
    print("Checking for name mismatches...")
    print("="*60)
    
    json_company_names = {c['company_name'] for c in structured_risks}
    
    result = session.run('MATCH (c:Company) RETURN c.name as name')
    neo4j_company_names = {rec['name'] for rec in result}
    
    # Companies in JSON but not in Neo4j
    missing_in_neo4j = json_company_names - neo4j_company_names
    if missing_in_neo4j:
        print(f"\nCompanies in JSON but NOT in Neo4j: {len(missing_in_neo4j)}")
        for name in list(missing_in_neo4j)[:5]:
            print(f"  - {name}")
    
    # Companies in Neo4j but not in JSON (likely from companies_list.json)
    extra_in_neo4j = neo4j_company_names - json_company_names
    if extra_in_neo4j:
        print(f"\nCompanies in Neo4j but NOT in structured_risks.json: {len(extra_in_neo4j)}")
        print("(These are likely from companies_list.json)")
        for name in list(extra_in_neo4j)[:5]:
            print(f"  - {name}")

driver.close()
