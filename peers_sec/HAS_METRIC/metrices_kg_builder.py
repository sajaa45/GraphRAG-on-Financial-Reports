import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))


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
            # Auto-detect target company from existing graph
            record = session.run(
                "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1"
            ).single()
            if not record:
                record = session.run(
                    "MATCH (c:Company {is_target: true}) RETURN c.name AS name LIMIT 1"
                ).single()
            target_company_name = record["name"] if record else None

            if not target_company_name:
                print("⚠ No target company found in Neo4j — COMPETES_WITH edges will be skipped")
            else:
                print(f"  ✓ Target company: {target_company_name}")

            # Build metric_type → category map from target company's existing graph
            category_map = {}
            if target_company_name:
                rows = session.run(
                    """
                    MATCH (c)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                    WHERE c:TargetCompany OR (c:Company AND c.is_target = true)
                    RETURN m.metric_type AS metric_type, mc.name AS category
                    """
                )
                for row in rows:
                    if row["metric_type"]:
                        category_map[row["metric_type"]] = row["category"]
                print(f"  ✓ Loaded {len(category_map)} category mappings from target company")
            print()

            for item in companies_with_covenants:
                company = item['company']
                company_name = company['name']
                cik = company['cik']
                ticker = company['ticker']

                # Create or update Company node
                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    SET c.cik = $cik, c.ticker = $ticker, c.updated_at = datetime()
                    """,
                    {"name": company_name, "cik": cik, "ticker": ticker}
                )

                # COMPETES_WITH edge to target
                if target_company_name and company_name != target_company_name:
                    session.run(
                        """
                        MATCH (c:Company {name: $company_name})
                        MATCH (tc) WHERE (tc:TargetCompany OR tc:Company) AND tc.name = $target_name
                        MERGE (c)-[r:COMPETES_WITH]->(tc)
                        SET r.updated_at = datetime()
                        """,
                        {"company_name": company_name, "target_name": target_company_name}
                    )
                total_competitors += 1

                company_metrics = 0

                def write_metric(metric_type, metric_name, latest, unit_name, year,
                                 xbrl_tag, taxonomy, label, metadata=None):
                    category = category_map.get(metric_type, "Other")
                    
                    # Extract metadata fields
                    source_url = ""
                    metric_cik = cik  # fallback to company CIK
                    if metadata:
                        source_url = metadata.get("source_url", "")
                        metric_cik = metadata.get("cik", cik)
                    
                    session.run(
                        """
                        MATCH (c:Company {name: $company_name})
                        MERGE (mc:MetricCategory {name: $category, company: $company_name})
                        MERGE (c)-[:HAS_METRIC_CATEGORY]->(mc)
                        MERGE (m:Metric {name: $metric_name, company: $company_name})
                        SET m.value = $value,
                            m.unit = $unit,
                            m.year = $year,
                            m.metric_type = $metric_type,
                            m.xbrl_tag = $xbrl_tag,
                            m.taxonomy = $taxonomy,
                            m.label = $label,
                            m.source_url = $source_url,
                            m.cik = $cik,
                            m.updated_at = datetime()
                        MERGE (mc)-[:HAS_METRIC]->(m)
                        """,
                        {
                            "company_name": company_name,
                            "category": category,
                            "metric_name": metric_name,
                            "value": str(latest.get('val', '')),
                            "unit": unit_name,
                            "year": str(year),
                            "metric_type": metric_type,
                            "xbrl_tag": xbrl_tag,
                            "taxonomy": taxonomy,
                            "label": label,
                            "source_url": source_url,
                            "cik": metric_cik,
                        }
                    )

                # Process covenant metrics
                for covenant_type, tags in item.get('covenants', {}).items():
                    for tag_name, tag_data in tags.items():
                        for unit_name, entries in tag_data.get('units', {}).items():
                            if entries:
                                latest = sorted(entries, key=lambda x: x.get('end', ''), reverse=True)[0]
                                end_date = latest.get('end', '')
                                year = end_date.split('-')[0] if end_date else 'Unknown'
                                metadata = tag_data.get('metadata', {})
                                write_metric(
                                    covenant_type, f"{covenant_type} ({year})",
                                    latest, unit_name, year,
                                    tag_name, tag_data.get('taxonomy', ''), tag_data.get('label', ''),
                                    metadata
                                )
                                company_metrics += 1

                # Process target metric matches
                for metric_type, matches in item.get('target_metrics', {}).items():
                    for match in matches:
                        for unit_name, entries in match.get('units', {}).items():
                            if entries:
                                latest = sorted(entries, key=lambda x: x.get('end', ''), reverse=True)[0]
                                end_date = latest.get('end', '')
                                year = latest.get('fy', end_date.split('-')[0] if end_date else 'Unknown')
                                metadata = match.get('metadata', {})
                                write_metric(
                                    metric_type, f"{metric_type} ({year})",
                                    latest, unit_name, year,
                                    match.get('tag', ''), match.get('taxonomy', ''), match.get('label', ''),
                                    metadata
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

    _default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies_metrices.json")
    json_file = sys.argv[1] if len(sys.argv) > 1 else _default

    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)

    write_metrics_to_neo4j(json_file)
