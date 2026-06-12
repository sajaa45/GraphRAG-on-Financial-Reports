import os
import sys
import json
import traceback
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

_KNOWN_CATEGORIES = {'Leverage', 'Coverage', 'Liquidity', 'Profitability', 'Debt Structure'}


def _write_metric(session, company_name, cik, category_map, peer_metric_counter,
                  metric_type, metric_name, latest, unit_name, year,
                  xbrl_tag, taxonomy, label, metadata=None, gaap_concept=None):
    raw_cat = category_map.get(metric_type, "Other")
    category = raw_cat if raw_cat in _KNOWN_CATEGORIES else "Other"

    source_url = ""
    metric_cik = cik
    if metadata:
        source_url = metadata.get("source_url", "")
        metric_cik = metadata.get("cik", cik)

    raw_val = latest.get('val', '')
    if unit_name == "USD" and raw_val != '':
        try:
            stored_value = str(round(float(raw_val) / 1_000_000, 4))
            stored_unit = "$ million"
        except (ValueError, TypeError):
            stored_value = str(raw_val)
            stored_unit = unit_name
    else:
        stored_value = str(raw_val)
        stored_unit = unit_name

    peer_metric_counter[0] += 1
    citation_id = f"PEER_M{peer_metric_counter[0]:04d}"

    session.run(
        """
        MATCH (c:Company {name: $company_name})
        MERGE (mc:MetricCategory {name: $category, company: $company_name})
        MERGE (c)-[:HAS_METRIC_CATEGORY]->(mc)
        MERGE (m:Metric {name: $metric_name, company: $company_name})
        SET m.citation_id = $citation_id,
            m.value = $value,
            m.unit = $unit,
            m.year = $year,
            m.metric_type = $metric_type,
            m.gaap_concept = $gaap_concept,
            m.category = $category,
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
            "citation_id": citation_id,
            "value": stored_value,
            "unit": stored_unit,
            "year": str(year),
            "metric_type": metric_type,
            "gaap_concept": gaap_concept or label or metric_type,
            "xbrl_tag": xbrl_tag,
            "taxonomy": taxonomy,
            "label": label,
            "source_url": source_url,
            "cik": metric_cik,
        }
    )


def write_metrics_to_neo4j(driver, json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    companies_with_metrics = data.get('companies_with_metrics', [])

    total_metrics = 0
    total_companies = 0
    peer_metric_counter = [0]  # mutable int for module-level helper

    print("=" * 80)
    print("WRITING METRICS TO NEO4J")
    print("=" * 80)
    print()

    try:
        with driver.session() as session:
            record = session.run(
                "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1"
            ).single()
            target_company_name = record["name"] if record else None

            if not target_company_name:
                print("⚠ No target company found in Neo4j — COMPETES_WITH edges will be skipped")
            else:
                print(f"  ✓ Target company: {target_company_name}")

            category_map = {}
            if target_company_name:
                rows = session.run(
                    """
                    MATCH (c:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                    RETURN m.metric_type AS metric_type, mc.name AS category
                    """
                )
                for row in rows:
                    if row["metric_type"]:
                        category_map[row["metric_type"]] = row["category"]
                print(f"  ✓ Loaded {len(category_map)} category mappings from target company")
            print()

            for item in companies_with_metrics:
                company = item['company']
                company_name = company['name']
                cik = company['cik']
                ticker = company['ticker']

                if target_company_name and company_name.lower() == target_company_name.lower():
                    print(f"  Skipped — this is the target company: {company_name}")
                    continue

                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    SET c.cik = $cik, c.ticker = $ticker, c.updated_at = datetime()
                    """,
                    {"name": company_name, "cik": cik, "ticker": ticker}
                )

                if target_company_name:
                    session.run(
                        """
                        MATCH (c:Company {name: $company_name})
                        MATCH (tc:TargetCompany {name: $target_name})
                        MERGE (c)-[r:COMPETES_WITH]->(tc)
                        SET r.updated_at = datetime()
                        """,
                        {"company_name": company_name, "target_name": target_company_name}
                    )

                company_metrics = 0

                for metric_type, matches in item.get('target_metrics', {}).items():
                    for match in matches:
                        for unit_name, entries in match.get('units', {}).items():
                            if entries:
                                latest = sorted(entries, key=lambda x: x.get('end', ''), reverse=True)[0]
                                end_date = latest.get('end', '')
                                year = latest.get('fy', end_date.split('-')[0] if end_date else 'Unknown')
                                peer_label = match.get('label', '') or match.get('tag', metric_type)
                                _write_metric(
                                    session, company_name, cik, category_map, peer_metric_counter,
                                    metric_type, f"{peer_label} ({year})",
                                    latest, unit_name, year,
                                    match.get('tag', ''), match.get('taxonomy', ''), peer_label,
                                    match.get('metadata', {}),
                                    gaap_concept=peer_label,
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
        print(f"Target company:              {target_company_name}")
        print(f"Companies with metrics:      {total_companies}")
        print(f"Total metrics written:       {total_metrics}")
        print()

        return total_metrics, total_companies

    except Exception as e:
        print(f"⚠ Error writing to Neo4j: {e}")
        traceback.print_exc()
        return 0, 0


if __name__ == "__main__":
    _default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies_metrices.json")
    json_file = sys.argv[1] if len(sys.argv) > 1 else _default

    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    try:
        write_metrics_to_neo4j(driver, json_file)
    finally:
        driver.close()
