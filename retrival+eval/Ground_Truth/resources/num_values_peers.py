"""
For every peer metric that has an xbrl_tag in extraction_result.json:
1. Fetches ground truth from data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
2. Matches by xbrl_tag + fiscal year (10-K/10-K/A forms only)
3. Compares pipeline_value == ground_truth_value with 2% tolerance
4. Computes accuracy = matches / total_verified

"""

import csv
import json
import os
import sys
import time
from typing import Optional

import requests

def _cik_from_source_url(source_url: str) -> Optional[str]:
    """Extract CIK from a URL like https://data.sec.gov/api/xbrl/companyfacts/CIK0001801602.json"""
    if not source_url or 'CIK' not in source_url:
        return None
    try:
        cik_part = source_url.split('CIK')[1].split('.')[0]
        return cik_part.lstrip('0') or '0'
    except (IndexError, AttributeError):
        return None


def _fetch_company_facts(cik: str, cache_dir: str) -> Optional[dict]:
    os.makedirs(cache_dir, exist_ok=True)
    cik_padded = cik.zfill(10)
    cache_file = os.path.join(cache_dir, f"CIK{cik_padded}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding='utf-8') as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {
        'User-Agent': 'Research Project research@example.com',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'data.sec.gov',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        time.sleep(0.2)
        return data
    except Exception as e:
        print(f"    EDGAR fetch failed for CIK {cik}: {e}")
        return None


def _find_gt_value(facts: dict, xbrl_tag: str, fiscal_year: str, unit: str = 'USD') -> Optional[tuple]:
    """
    Return (value, actual_unit, filed_date) for the best matching 10-K entry,
    or None if not found.
    """
    us_gaap = facts.get('facts', {}).get('us-gaap', {})
    if xbrl_tag not in us_gaap:
        for taxonomy in ('dei', 'ifrs-full', 'srt'):
            alt = facts.get('facts', {}).get(taxonomy, {})
            if xbrl_tag in alt:
                us_gaap = alt
                break
        else:
            return None

    units_data = us_gaap[xbrl_tag].get('units', {})
    # Try requested unit first, then any available unit
    unit_values = units_data.get(unit)
    actual_unit = unit
    if not unit_values:
        for u, vals in units_data.items():
            unit_values = vals
            actual_unit = u
            break
    if not unit_values:
        return None

    candidates = [
        e for e in unit_values
        if e.get('form') in ('10-K', '10-K/A')
        and (str(e.get('fy', '')) == str(fiscal_year)
             or (e.get('end') or '')[:4] == str(fiscal_year))
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
    best = candidates[0]
    return best.get('val'), actual_unit, best.get('filed', 'N/A')



def _to_float(raw_unit: str, raw_value: str) -> tuple[str, Optional[float]]:
    u = (raw_unit or '').strip().lower()
    try:
        v = float(raw_value)
    except (ValueError, TypeError):
        return raw_unit, None
    if u in ('$ million', 'million', '$ millions'):
        return 'USD_million', v * 1_000_000
    if u in ('$ thousand', 'thousand', '$ thousands'):
        return 'USD_thousand', v * 1_000
    return u, v


def _values_match(pipeline_val: str, pipeline_unit: str,
                  gt_val: float, tolerance: float = 0.02) -> bool:
    _, pv = _to_float(pipeline_unit, pipeline_val)
    if pv is None or gt_val is None:
        return False
    if gt_val == 0 and pv == 0:
        return True
    if gt_val == 0:
        return False
    return abs(pv - gt_val) / abs(gt_val) <= tolerance



def _collect_peer_metrics(data: dict) -> list[dict]:
    seen: set[str] = set()
    metrics: list[dict] = []

    for row in data.get('results', []):
        peer_name = row.get('peer', '')
        if not peer_name:
            continue

        for m in (row.get('peer_metrics') or []):
            xbrl_tag = (m.get('xbrl_tag') or '').strip()
            if not xbrl_tag:
                continue

            cid = m.get('citation_id', '')
            if cid in seen:
                continue
            seen.add(cid)

            cik = _cik_from_source_url(m.get('source_url', ''))
            if not cik:
                continue

            metrics.append({
                'company':      peer_name,
                'citation_id':  cid,
                'cik':          cik,
                'xbrl_tag':     xbrl_tag,
                'gaap_concept': m.get('gaap_concept') or m.get('label') or '',
                'year':         str(m.get('year', '')),
                'value':        str(m.get('value', '')),
                'unit':         m.get('unit', ''),
            })

    return metrics



def validate_metrics(extraction_result_path: str, output_csv_path: str, debug: bool = False):
    log_path = output_csv_path.replace('.csv', '_log.txt')

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.__stdout__
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def close(self):
            self.log.close()

    original_stdout = sys.stdout
    logger = TeeLogger(log_path)
    sys.stdout = logger
    try:
        _run_validation(extraction_result_path, output_csv_path, debug)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_path}")


def _run_validation(extraction_result_path: str, output_csv_path: str, debug: bool = False):
    print("=" * 70)
    print("Peer XBRL Metrics Ground Truth Validation (SEC EDGAR)")
    print("=" * 70)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', '..', 'sec_cache')

    print(f"\n[1/4] Loading extraction results from {extraction_result_path}")
    with open(extraction_result_path, encoding='utf-8') as f:
        data = json.load(f)

    question = data.get('question', '')
    print(f"  Question: {question[:100]}")

    metrics = _collect_peer_metrics(data)
    print(f"  Peer metrics with xbrl_tag: {len(metrics)}")

    if not metrics:
        print("\n✗ No peer metrics with xbrl_tag found in extraction results")
        print("  (xbrl_tag is populated by metrices_kg_builder.py for peer metrics)")
        return

    # ── EDGAR lookup ──────────────────────────────────────────────────────
    print(f"\n[2/4] Fetching ground truth from SEC EDGAR ({len(metrics)} metrics)")

    facts_cache: dict[str, Optional[dict]] = {}
    results_table = []

    for m in metrics:
        cik = m['cik']
        if cik not in facts_cache:
            facts_cache[cik] = _fetch_company_facts(cik, cache_dir)

        facts = facts_cache[cik]
        if not facts:
            results_table.append({**m, 'gt_value': 'N/A', 'gt_unit': '',
                                   'gt_filed': '', 'match': 'ERROR',
                                   'notes': 'EDGAR fetch failed'})
            continue

        gt = _find_gt_value(facts, m['xbrl_tag'], m['year'])
        if gt is None:
            results_table.append({**m, 'gt_value': 'N/A', 'gt_unit': '',
                                   'gt_filed': '', 'match': 'NOT_FOUND',
                                   'notes': f"Tag {m['xbrl_tag']} not in 10-K FY{m['year']}"})
            if debug:
                print(f"  NOT_FOUND  {m['citation_id']}  {m['xbrl_tag']} ({m['year']})")
            continue

        gt_val, gt_unit, gt_filed = gt
        match = _values_match(m['value'], m['unit'], gt_val)
        results_table.append({**m, 'gt_value': gt_val, 'gt_unit': gt_unit,
                               'gt_filed': gt_filed,
                               'match': 'MATCH' if match else 'MISMATCH',
                               'notes': ''})
        if debug:
            status = 'MATCH' if match else 'MISMATCH'
            print(f"  {status}  {m['citation_id']}  pipeline={m['value']} {m['unit']}  gt={gt_val} {gt_unit}")

    total     = len(results_table)
    matches   = sum(1 for r in results_table if r['match'] == 'MATCH')
    mismatches= sum(1 for r in results_table if r['match'] == 'MISMATCH')
    not_found = sum(1 for r in results_table if r['match'] == 'NOT_FOUND')
    errors    = sum(1 for r in results_table if r['match'] == 'ERROR')
    verified  = matches + mismatches
    accuracy  = (matches / verified * 100) if verified > 0 else 0.0

    print(f"\n[3/4] Results")
    print(f"  Total        : {total}")
    print(f"  Matches      : {matches}")
    print(f"  Mismatches   : {mismatches}")
    print(f"  Not found    : {not_found}")
    print(f"  Errors       : {errors}")
    print(f"  Accuracy     : {accuracy:.1f}%")

    # ── Write CSV ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Writing results to {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = ['company', 'citation_id', 'cik', 'xbrl_tag', 'gaap_concept',
                  'year', 'value', 'unit', 'gt_value', 'gt_unit', 'gt_filed',
                  'match', 'notes']
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results_table)

    print(f"  ✓ Saved to {output_csv_path}")
    print("\n" + "=" * 70)
    print(f"Validation Complete — Accuracy: {accuracy:.1f}%")
    print("=" * 70)
