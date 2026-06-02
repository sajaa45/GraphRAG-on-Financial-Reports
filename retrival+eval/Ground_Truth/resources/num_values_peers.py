"""
Ground Truth Validation for XBRL Metrics

This script validates metrics extracted by the pipeline against the SEC XBRL API.
For every (CIK, year, XBRL tag) used in the pipeline:
1. Fetches ground truth from data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
2. Navigates to facts > us-gaap > {tag} > units > USD (or shares) > []
3. Filters by form: "10-K" and matches the fiscal year
4. Records the val field as ground truth
5. Compares pipeline_value == expected_value with tolerance for floats
6. Computes metric_accuracy = correct_matches / total_metrics_pulled

Output: CSV table with columns:
{cik, company_name, xbrl_tag, fiscal_year, gt_value, pipeline_value, units, match}
"""

import json
import os
import requests
import time
from typing import Dict, List, Optional, Tuple
import csv
from pathlib import Path


def load_extraction_result(json_path: str) -> dict:
    """Load the extraction_result.json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_cik_from_source_url(source_url: str) -> Optional[str]:
    """Extract CIK from source_url like https://data.sec.gov/api/xbrl/companyfacts/CIK0001817232.json"""
    if not source_url or 'CIK' not in source_url:
        return None
    try:
        # Extract CIK from URL
        cik_part = source_url.split('CIK')[1].split('.')[0]
        return cik_part.lstrip('0')  # Remove leading zeros for comparison
    except (IndexError, AttributeError):
        return None


def fetch_sec_company_facts(cik: str, cache_dir: str = "sec_cache") -> Optional[dict]:
    """
    Fetch company facts from SEC API with caching.
    
    Args:
        cik: Company CIK (without leading zeros)
        cache_dir: Directory to cache API responses
    
    Returns:
        Company facts JSON or None if fetch fails
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Pad CIK to 10 digits for API call
    cik_padded = cik.zfill(10)
    cache_file = os.path.join(cache_dir, f"CIK{cik_padded}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        print(f"  → Using cached data for CIK {cik}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fetch from SEC API
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {
        'User-Agent': 'Research Project contact@example.com',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'data.sec.gov'
    }
    
    try:
        print(f"  → Fetching from SEC API: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Cache the response
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Be respectful to SEC API - rate limit
        time.sleep(0.2)
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to fetch CIK {cik}: {e}")
        return None


def find_ground_truth_value(
    company_facts: dict,
    xbrl_tag: str,
    fiscal_year: str,
    unit: str = "USD"
) -> Optional[Tuple[float, str, str]]:
    """
    Find ground truth value from SEC company facts.
    
    Args:
        company_facts: SEC company facts JSON
        xbrl_tag: XBRL tag name (e.g., "NetIncomeLoss")
        fiscal_year: Fiscal year as string (e.g., "2024")
        unit: Unit type (e.g., "USD", "shares", "pure")
    
    Returns:
        Tuple of (value, actual_unit, filing_date) or None if not found
    """
    try:
        # Navigate to facts > us-gaap > {tag}
        facts = company_facts.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        if xbrl_tag not in us_gaap:
            # Try with different taxonomy prefixes
            for taxonomy in ['dei', 'ifrs-full', 'srt']:
                taxonomy_facts = facts.get(taxonomy, {})
                if xbrl_tag in taxonomy_facts:
                    us_gaap = taxonomy_facts
                    break
            else:
                return None
        
        tag_data = us_gaap.get(xbrl_tag, {})
        units_data = tag_data.get('units', {})
        
        # Try to find the unit - be flexible with unit matching
        unit_values = None
        actual_unit = unit
        
        if unit in units_data:
            unit_values = units_data[unit]
        elif unit == "USD" and "USD" not in units_data:
            # Try other currency-like units
            for u in units_data.keys():
                if u.upper() in ["USD", "DOLLARS", "CURRENCY"]:
                    unit_values = units_data[u]
                    actual_unit = u
                    break
        elif unit == "pure" and "pure" not in units_data:
            # Try dimensionless units
            for u in units_data.keys():
                if u.lower() in ["pure", "number", "ratio"]:
                    unit_values = units_data[u]
                    actual_unit = u
                    break
        
        if not unit_values:
            return None
        
        # Filter by form: "10-K" (or amended 10-K/A) and fiscal year.
        # SEC XBRL entries sometimes lack the 'fy' field, so fall back to
        # matching by the year embedded in the 'end' date (e.g. "2024-12-31").
        candidates = []
        for entry in unit_values:
            if entry.get('form') not in ('10-K', '10-K/A'):
                continue
            fy = entry.get('fy')
            end_year = (entry.get('end') or '')[:4]
            if str(fy) == str(fiscal_year) or end_year == str(fiscal_year):
                candidates.append(entry)
        
        if not candidates:
            return None
        
        # If multiple entries exist, prefer the one with the latest 'end' date
        # (this is typically the annual value, not a quarterly restatement)
        # Also prefer entries with 'filed' date (most recent filing)
        candidates.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
        
        best_entry = candidates[0]
        return (best_entry.get('val'), actual_unit, best_entry.get('filed', 'N/A'))
        
    except (KeyError, TypeError, AttributeError) as e:
        return None


def compare_values(pipeline_val: str, gt_val: float, tolerance: float = 0.01) -> bool:
    """
    Compare pipeline value with ground truth value.
    
    Args:
        pipeline_val: Value from pipeline (string)
        gt_val: Ground truth value (float)
        tolerance: Relative tolerance for float comparison
    
    Returns:
        True if values match within tolerance
    """
    try:
        pipeline_float = float(pipeline_val)
        
        # Exact match for zero
        if gt_val == 0 and pipeline_float == 0:
            return True
        
        # Relative difference for non-zero values
        if gt_val != 0:
            rel_diff = abs(pipeline_float - gt_val) / abs(gt_val)
            return rel_diff <= tolerance
        
        return False
        
    except (ValueError, TypeError):
        return False


def validate_metrics(extraction_result_path: str, output_csv_path: str, debug: bool = False):
    """
    Main validation function.
    
    Args:
        extraction_result_path: Path to extraction_result.json
        output_csv_path: Path to output CSV file
        debug: If True, print detailed debug information
    """
    # Set up logging to both console and file
    log_file_path = output_csv_path.replace('.csv', '_log.txt')
    
    class TeeLogger:
        """Write to both console and file"""
        def __init__(self, filename):
            import sys
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    import sys
    original_stdout = sys.stdout
    logger = TeeLogger(log_file_path)
    sys.stdout = logger
    
    try:
        _run_validation(extraction_result_path, output_csv_path, debug)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_file_path}")


def _run_validation(extraction_result_path: str, output_csv_path: str, debug: bool = False):
    """
    Main validation function.
    
    Args:
        extraction_result_path: Path to extraction_result.json
        output_csv_path: Path to output CSV file
        debug: If True, print detailed debug information
    """
    print("="*70)
    print("XBRL Metrics Ground Truth Validation")
    print("="*70)
    
    # Load extraction results
    print(f"\n[1/4] Loading extraction results from {extraction_result_path}")
    data = load_extraction_result(extraction_result_path)
    
    # Collect all metrics to validate
    metrics_to_validate = []
    
    def _normalise_metric(raw_unit: str, raw_value: str) -> tuple:
        """Convert pipeline-stored units back to SEC API units + raw value.
        metrices_kg_builder.py stores USD amounts in millions with unit "$ million"
        and thousands with "$ thousand". SEC API always uses "USD" with raw values.
        """
        u = (raw_unit or '').strip().lower()
        if u == '$ million':
            try:
                return 'USD', str(round(float(raw_value) * 1_000_000))
            except (ValueError, TypeError):
                pass
        elif u == '$ thousand':
            try:
                return 'USD', str(round(float(raw_value) * 1_000))
            except (ValueError, TypeError):
                pass
        return raw_unit or 'USD', raw_value

    for result in data.get('results', []):
        # Check for peer_metrics
        if 'peer_metrics' in result and result['peer_metrics']:
            peer_name = result.get('peer', 'Unknown')
            for metric in result['peer_metrics']:
                if metric.get('xbrl_tag') and metric.get('source_url'):
                    cik = extract_cik_from_source_url(metric['source_url'])
                    if cik:
                        unit, value = _normalise_metric(
                            metric.get('unit', 'USD'), metric.get('value', '')
                        )
                        metrics_to_validate.append({
                            'company_name': peer_name,
                            'cik': cik,
                            'xbrl_tag': metric.get('xbrl_tag'),
                            'fiscal_year': metric.get('year'),
                            'pipeline_value': value,
                            'unit': unit,
                        })

        # Check for target_metrics
        if 'target_metrics' in result and result['target_metrics']:
            target_name = result.get('target', 'Unknown')
            for metric in result['target_metrics']:
                if metric.get('xbrl_tag') and metric.get('source_url'):
                    cik = extract_cik_from_source_url(metric.get('source_url', ''))
                    if cik:
                        unit, value = _normalise_metric(
                            metric.get('unit', 'USD'), metric.get('value', '')
                        )
                        metrics_to_validate.append({
                            'company_name': target_name,
                            'cik': cik,
                            'xbrl_tag': metric.get('xbrl_tag'),
                            'fiscal_year': metric.get('year'),
                            'pipeline_value': value,
                            'unit': unit,
                        })
    
    print(f"  → Found {len(metrics_to_validate)} metrics to validate")
    
    if not metrics_to_validate:
        print("\n✗ No metrics with xbrl_tag found in extraction results")
        return
    
    # Fetch ground truth and compare
    print(f"\n[2/4] Fetching ground truth from SEC API")
    
    results_table = []
    company_facts_cache = {}
    
    for i, metric in enumerate(metrics_to_validate, 1):
        print(f"\n[{i}/{len(metrics_to_validate)}] {metric['company_name']} - {metric['xbrl_tag']} ({metric['fiscal_year']})")
        
        cik = metric['cik']
        
        # Fetch company facts (with caching)
        if cik not in company_facts_cache:
            company_facts_cache[cik] = fetch_sec_company_facts(cik)
        
        company_facts = company_facts_cache[cik]
        
        if not company_facts:
            results_table.append({
                **metric,
                'gt_value': 'N/A',
                'match': 'ERROR',
                'notes': 'Failed to fetch company facts'
            })
            continue
        
        # Find ground truth value
        gt_result = find_ground_truth_value(
            company_facts,
            metric['xbrl_tag'],
            metric['fiscal_year'],
            metric['unit']
        )
        
        if gt_result is None:
            results_table.append({
                **metric,
                'gt_value': 'N/A',
                'gt_filing_date': 'N/A',
                'match': 'NOT_FOUND',
                'notes': f"Tag {metric['xbrl_tag']} not found in 10-K for FY {metric['fiscal_year']}"
            })
            print(f"  ✗ Ground truth not found")
            continue
        
        gt_value, actual_unit, gt_filing_date = gt_result
        
        # Compare values
        match = compare_values(metric['pipeline_value'], gt_value)
        
        results_table.append({
            **metric,
            'gt_value': gt_value,
            'actual_unit': actual_unit,
            'gt_filing_date': gt_filing_date,
            'match': 'MATCH' if match else 'MISMATCH',
            'notes': ''
        })
        
        if match:
            print(f"  ✓ Match: pipeline={metric['pipeline_value']}, gt={gt_value}")
        else:
            print(f"  ✗ Mismatch: pipeline={metric['pipeline_value']}, gt={gt_value}")
    
    # Calculate accuracy
    print(f"\n[3/4] Computing accuracy metrics")
    
    total_metrics = len(results_table)
    matches = sum(1 for r in results_table if r['match'] == 'MATCH')
    mismatches = sum(1 for r in results_table if r['match'] == 'MISMATCH')
    not_found = sum(1 for r in results_table if r['match'] == 'NOT_FOUND')
    errors = sum(1 for r in results_table if r['match'] == 'ERROR')
    
    accuracy = (matches / total_metrics * 100) if total_metrics > 0 else 0
    
    print(f"  Total metrics:  {total_metrics}")
    print(f"  Matches:        {matches}")
    print(f"  Mismatches:     {mismatches}")
    print(f"  Not found:      {not_found}")
    print(f"  Errors:         {errors}")
    print(f"  Accuracy:       {accuracy:.2f}%")
    
    # Write results to CSV
    print(f"\n[4/4] Writing results to {output_csv_path}")
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'company_name', 'cik', 'xbrl_tag', 'fiscal_year',
            'pipeline_value', 'gt_value', 'unit', 'actual_unit', 'gt_filing_date', 'match', 'notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    
    print(f"  ✓ Results saved to {output_csv_path}")
    
    print("\n" + "="*70)
    print(f"Validation Complete - Accuracy: {accuracy:.2f}%")
    print("="*70)


if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extraction_result_path = os.path.join(
        script_dir, '..', '..', 'retrival_results', 'extraction_result.json'
    )
    output_csv_path = os.path.join(
        script_dir, '..', 'metrics_validation_results.csv'
    )
    
    validate_metrics(extraction_result_path, output_csv_path)
