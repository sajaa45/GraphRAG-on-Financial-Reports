import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def write_metrics_to_neo4j(json_file: str):
    """Write covenant metrics from JSON to Neo4j as HAS_METRIC relationships."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    companies_with_covenants = data.get('companies_with_metrics', data.get('companies_with_covenants', []))
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    total_metrics = 0
    total_companies = 0
    total_competitors = 0
    
    print("=" * 80)
    print("WRITING METRICS TO NEO4J")
    print("=" * 80)
    print()
    
    try:
        with driver.session() as session:
            # First, ensure target company exists as TargetCompany
            session.run(
                """
                MERGE (tc:TargetCompany {name: $name})
                SET tc.updated_at = datetime()
                """,
                {"name": target_company_name}
            )
            print(f"  ✓ Target company created/updated: {target_company_name}")
            print()
            
            for item in companies_with_covenants:
                company = item['company']
                company_name = company['name']
                cik = company['cik']
                ticker = company['ticker']
                
                # Create or update Company node with CIK and ticker
                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    SET c.cik = $cik, c.ticker = $ticker, c.updated_at = datetime()
                    """,
                    {"name": company_name, "cik": cik, "ticker": ticker}
                )
                
                # Create COMPETES_WITH relationship to target company
                session.run(
                    """
                    MATCH (c:Company {name: $company_name})
                    MATCH (tc:TargetCompany {name: $target_name})
                    MERGE (c)-[r:COMPETES_WITH]->(tc)
                    SET r.updated_at = datetime()
                    """,
                    {"company_name": company_name, "target_name": target_company_name}
                )
                total_competitors += 1
                
                company_metrics = 0
                
                # Process covenant metrics
                for covenant_type, tags in item.get('covenants', {}).items():
                    for tag_name, tag_data in tags.items():
                        # Get the most recent value for each unit
                        for unit_name, entries in tag_data.get('units', {}).items():
                            if entries:
                                # Sort by end date and get latest
                                sorted_entries = sorted(entries, key=lambda x: x.get('end', ''), reverse=True)
                                latest = sorted_entries[0]
                                
                                # Extract year from end date
                                end_date = latest.get('end', '')
                                year = end_date.split('-')[0] if end_date else 'Unknown'
                                
                                # Create metric name
                                metric_name = f"{covenant_type} ({year})"
                                
                                # Create Metric node and HAS_METRIC relationship
                                session.run(
                                    """
                                    MATCH (c:Company {name: $company_name})
                                    MERGE (m:Metric {name: $metric_name, company: $company_name})
                                    SET m.value = $value,
                                        m.unit = $unit,
                                        m.year = $year,
                                        m.metric_type = $metric_type,
                                        m.xbrl_tag = $xbrl_tag,
                                        m.taxonomy = $taxonomy,
                                        m.label = $label,
                                        m.updated_at = datetime()
                                    MERGE (c)-[r:HAS_METRIC]->(m)
                                    SET r.source = 'SEC_XBRL',
                                        r.filing_date = $filed,
                                        r.form = $form,
                                        r.accn = $accn,
                                        r.updated_at = datetime()
                                    """,
                                    {
                                        "company_name": company_name,
                                        "metric_name": metric_name,
                                        "value": str(latest.get('val', '')),
                                        "unit": unit_name,
                                        "year": year,
                                        "metric_type": covenant_type,
                                        "xbrl_tag": tag_name,
                                        "taxonomy": tag_data.get('taxonomy', ''),
                                        "label": tag_data.get('label', ''),
                                        "filed": latest.get('filed', ''),
                                        "form": latest.get('form', ''),
                                        "accn": latest.get('accn', '')
                                    }
                                )
                                company_metrics += 1
                
                # Process target metric matches
                for metric_type, matches in item.get('target_metrics', {}).items():
                    for match in matches:
                        # Get the most recent value for each unit
                        for unit_name, entries in match.get('units', {}).items():
                            if entries:
                                sorted_entries = sorted(entries, key=lambda x: x.get('end', ''), reverse=True)
                                latest = sorted_entries[0]
                                
                                end_date = latest.get('end', '')
                                year = latest.get('fy', end_date.split('-')[0] if end_date else 'Unknown')
                                
                                # Use the metric_type with year
                                full_metric_name = f"{metric_type} ({year})"
                                
                                session.run(
                                    """
                                    MATCH (c:Company {name: $company_name})
                                    MERGE (m:Metric {name: $metric_name, company: $company_name})
                                    SET m.value = $value,
                                        m.unit = $unit,
                                        m.year = $year,
                                        m.metric_type = $metric_type,
                                        m.xbrl_tag = $xbrl_tag,
                                        m.taxonomy = $taxonomy,
                                        m.label = $label,
                                        m.updated_at = datetime()
                                    MERGE (c)-[r:HAS_METRIC]->(m)
                                    SET r.source = 'SEC_XBRL',
                                        r.filing_date = $filed,
                                        r.form = $form,
                                        r.accn = $accn,
                                        r.matched_target_metric = true,
                                        r.updated_at = datetime()
                                    """,
                                    {
                                        "company_name": company_name,
                                        "metric_name": full_metric_name,
                                        "value": str(latest.get('val', '')),
                                        "unit": unit_name,
                                        "year": str(year),
                                        "metric_type": metric_type,
                                        "xbrl_tag": match.get('tag', ''),
                                        "taxonomy": match.get('taxonomy', ''),
                                        "label": match.get('label', ''),
                                        "filed": latest.get('filed', ''),
                                        "form": latest.get('form', ''),
                                        "accn": latest.get('accn', '')
                                    }
                                )
                                company_metrics += 1
                
                if company_metrics > 0:
                    total_metrics += company_metrics
                    total_companies += 1
                    print(f"  ✓ {company_name}: {company_metrics} metrics written")
        
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Target company: {target_company_name}")
        print(f"Total competitor companies: {total_competitors}")
        print(f"Total companies with metrics: {total_companies}")
        print(f"Total metrics: {total_metrics}")
        print()
        
        return total_metrics, total_companies
        
    except Exception as e:
        print(f"⚠ Error writing to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0
    finally:
        driver.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python write_covenants_to_neo4j.py <json_file>")
        print("Example: python write_covenants_to_neo4j.py financial_covenants_analysis_2023-01-01_to_2024-01-01.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    write_metrics_to_neo4j(json_file)
