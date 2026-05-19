"""
Extract target company metrics from SEC XBRL API to populate label and xbrl_tag fields.
This ensures target and peer companies have consistent data structure for matching.
"""
import os
import sys
import json
import requests
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

headers = {"User-Agent": "User (your_email@example.com)"}


def get_target_company_from_neo4j():
    """Get target company info from Neo4j."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (tc:TargetCompany)
                RETURN tc.name AS name, tc.cik AS cik, tc.ticker AS ticker
                LIMIT 1
                """
            )
            record = result.single()
            if record:
                return {
                    'name': record['name'],
                    'cik': str(record['cik']).zfill(10) if record['cik'] else None,
                    'ticker': record.get('ticker', 'N/A')
                }
            return None
    finally:
        driver.close()


def get_target_metrics_from_neo4j():
    """Get target company's existing metrics (from LLM extraction)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                RETURN DISTINCT m.metric_type AS metric_type, m.year AS year
                WHERE m.metric_type IS NOT NULL
                """
            )
            metrics = []
            for record in result:
                metric_type = record['metric_type']
                year = record.get('year')
                if metric_type:
                    metrics.append({
                        'type': metric_type,
                        'year': int(year) if year and str(year).isdigit() else 2024
                    })
            return metrics
    finally:
        driver.close()


def fetch_xbrl_data(cik):
    """Fetch XBRL company facts from SEC API."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching XBRL data: {e}")
        return None


def update_target_metrics_in_neo4j(cik, company_name, xbrl_data, target_metrics):
    """Update target company metrics with XBRL label and tag information."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    updated_count = 0
    
    try:
        with driver.session() as session:
            facts = xbrl_data.get('facts', {})
            
            # Build a lookup of XBRL tags by their labels
            xbrl_lookup = {}
            for taxonomy_name, taxonomy_data in facts.items():
                for tag_name, tag_content in taxonomy_data.items():
                    label = tag_content.get('label', '')
                    if label:
                        xbrl_lookup[label.lower()] = {
                            'tag': tag_name,
                            'label': label,
                            'taxonomy': taxonomy_name,
                            'description': tag_content.get('description', '')
                        }
            
            # For each target metric, try to find matching XBRL data
            for metric in target_metrics:
                metric_type = metric['type']
                year = metric['year']
                
                # Try to find matching XBRL tag by searching for similar labels
                best_match = None
                metric_lower = metric_type.lower()
                
                for label_lower, xbrl_info in xbrl_lookup.items():
                    if metric_lower in label_lower or label_lower in metric_lower:
                        best_match = xbrl_info
                        break
                
                if best_match:
                    # Update the metric with XBRL information
                    session.run(
                        """
                        MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                        WHERE m.metric_type = $metric_type AND m.year = $year
                        SET m.label = $label,
                            m.xbrl_tag = $xbrl_tag,
                            m.taxonomy = $taxonomy,
                            m.source_url = $source_url,
                            m.updated_at = datetime()
                        """,
                        {
                            'metric_type': metric_type,
                            'year': str(year),
                            'label': best_match['label'],
                            'xbrl_tag': best_match['tag'],
                            'taxonomy': best_match['taxonomy'],
                            'source_url': f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                        }
                    )
                    updated_count += 1
                    print(f"  ✓ Updated: {metric_type} → {best_match['label']}")
                else:
                    print(f"  ⚠ No XBRL match found for: {metric_type}")
        
        return updated_count
    finally:
        driver.close()


def main():
    print("=" * 80)
    print("EXTRACT TARGET COMPANY METRICS FROM SEC XBRL API")
    print("=" * 80)
    print()
    
    # Get target company info
    target = get_target_company_from_neo4j()
    if not target:
        print("⚠ No target company found in Neo4j")
        return
    
    if not target['cik']:
        print(f"⚠ Target company '{target['name']}' has no CIK")
        return
    
    print(f"Target company: {target['name']} (CIK: {target['cik']})")
    print()
    
    # Get existing metrics
    target_metrics = get_target_metrics_from_neo4j()
    print(f"Found {len(target_metrics)} existing metrics in Neo4j")
    print()
    
    # Fetch XBRL data
    print("Fetching XBRL data from SEC API...")
    xbrl_data = fetch_xbrl_data(target['cik'])
    if not xbrl_data:
        print("⚠ Failed to fetch XBRL data")
        return
    
    print("✓ XBRL data fetched")
    print()
    
    # Update metrics
    print("Updating metrics with XBRL labels and tags...")
    updated_count = update_target_metrics_in_neo4j(
        target['cik'],
        target['name'],
        xbrl_data,
        target_metrics
    )
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total metrics: {len(target_metrics)}")
    print(f"Updated with XBRL data: {updated_count}")
    print()


if __name__ == "__main__":
    main()
