import os
import re
import requests
import json
import time
from datetime import datetime, timedelta
import urllib3
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


FISCAL_YEAR = 2024  # Extract only this fiscal year
# Companies file their FY annual report months after the fiscal year ends.
# Span second half of FY through mid-next-year to capture all fiscal year-end dates.
START_DATE = f"{FISCAL_YEAR}-07-01"
END_DATE = f"{FISCAL_YEAR + 1}-06-30"
FORM_TYPE = "10-K"
MAX_COMPANIES = 3

headers = {"User-Agent": "User (your_email@example.com)"}


def get_companies_from_neo4j():
    """Get peer companies already in the Neo4j graph (excluding target company)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)-[:FACES_RISK]->(r:Risk)
                WHERE c.is_peer = true OR (NOT c:TargetCompany AND NOT c.is_target = true)
                WITH c, count(r) AS risk_count
                WHERE risk_count >= 10
                RETURN c.name AS name, c.cik AS cik, c.ticker AS ticker, risk_count
                ORDER BY risk_count DESC
                """
            )
            companies = []
            for record in result:
                name = record.get("name")
                cik = record.get("cik")
                ticker = record.get("ticker", "N/A")
                
                if name:
                    # If CIK is not stored, we'll need to look it up or skip
                    companies.append({
                        'name': name,
                        'cik': str(cik).zfill(10) if cik else None,
                        'ticker': ticker or 'N/A',
                        'source': 'neo4j'
                    })
            
            if companies:
                print(f"✓ Found {len(companies)} peer companies in Neo4j graph:")
                for c in companies:
                    cik_str = c['cik'] if c['cik'] else 'No CIK'
                    print(f"  - {c['name']} ({c['ticker']}) - {cik_str}")
                print()
            else:
                print("⚠ No peer companies found in Neo4j graph\n")
            
            return companies
    except Exception as e:
        print(f"⚠ Error querying Neo4j for companies: {e}\n")
        return []
    finally:
        driver.close()


def get_sic_from_neo4j() -> str:
    """Query Neo4j for the SIC code of the target company (is_target=true)."""
    print("  → Querying Neo4j for SIC code...")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Try multiple query patterns
            queries = [
                # Pattern 1: TargetCompany node type
                """
                MATCH (c:TargetCompany)-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 2: Company with is_target property
                """
                MATCH (c:Company {is_target: true})-[:OPERATES_IN]->(:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 3: Direct HAS_SIC_CODE from TargetCompany
                """
                MATCH (c:TargetCompany)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 4: Direct HAS_SIC_CODE from Company
                """
                MATCH (c:Company {is_target: true})-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 5: SIC code stored directly on the Industry node
                """
                MATCH (c)-[:OPERATES_IN]->(i:Industry)
                WHERE c:TargetCompany OR (c:Company AND c.is_target = true)
                AND i.sic_code IS NOT NULL
                RETURN i.sic_code AS sic_code
                LIMIT 1
                """,
                # Pattern 6: Any company with a SIC code (same broad fallback used by risk extractor)
                """
                MATCH (c:Company)-[:HAS_SIC_CODE]->(s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """,
                # Pattern 7: Any SIC code node in the DB
                """
                MATCH (s:SICCode)
                RETURN s.code AS sic_code
                LIMIT 1
                """
            ]

            for i, query in enumerate(queries, 1):
                result = session.run(query)
                record = result.single()
                if record and record["sic_code"]:
                    code = str(record["sic_code"]).strip()
                    print(f"✓ SIC code from Neo4j (pattern {i}): {code}")
                    driver.close()
                    return code

        driver.close()
    except Exception as e:
        print(f"⚠ Could not connect to Neo4j: {e}")

    print("⚠ No SIC code found in Neo4j, using default: 1311 (Oil & Gas)")
    return "1311"

def get_target_company_metrics():
    """Query Neo4j for metrics found in the target company."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            # Updated query to match new structure: Company -> MetricCategory -> Metric
            result = session.run(
                """
                MATCH (c)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                WHERE c:TargetCompany OR (c:Company AND c.is_target = true)
                RETURN m.name AS metric_name, m.metric_type AS metric_type,
                       m.gaap_concept AS gaap_concept, m.year AS year, mc.name AS category
                """
            )
            metrics = []
            seen_types = set()

            for record in result:
                metric_name  = record["metric_name"]
                metric_type  = record["metric_type"]
                gaap_concept = record.get("gaap_concept") or ''
                year         = record.get("year")
                category     = record.get("category", "Uncategorised")

                # If metric_type is not set, try to extract it from the name
                if not metric_type and metric_name:
                    if '(' in metric_name:
                        metric_type = metric_name.split('(')[0].strip()
                        year_part = metric_name.split('(')[1].split(')')[0].strip()
                        if year_part.isdigit():
                            year = int(year_part)
                    else:
                        metric_type = metric_name

                # Convert year to int if it's a string
                if year and isinstance(year, str) and year.isdigit():
                    year = int(year)

                # Only add unique metric types (not each year)
                if metric_type and metric_type not in seen_types:
                    metrics.append({
                        "name":         metric_name,
                        "type":         metric_type,
                        "gaap_concept": gaap_concept or metric_type,
                        "year":         year,
                        "category":     category,
                    })
                    seen_types.add(metric_type)

            if metrics:
                print(f"✓ Found {len(metrics)} unique metric types in target company:")
                for m in metrics:
                    year_str = f" (year: {m['year']})" if m['year'] else " (using default year)"
                    category_str = f" [{m['category']}]" if m.get('category') else ""
                    gaap_str = f" ← {m['gaap_concept']}" if m['gaap_concept'] != m['type'] else ""
                    print(f"  - {m['type']}{gaap_str}{year_str}{category_str}")
                print()
            else:
                print("⚠ No metrics found in target company\n")
            
            return metrics
    finally:
        driver.close()



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


BM25_TOP_K = 1
BM25_MIN_SCORE = 3.0

TAXONOMY_URL = "https://xbrl.fasb.org/us-gaap/2024/elts/us-gaap-lab-2024.xml"
TAXONOMY_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".taxonomy_cache.json")
TAXONOMY_CACHE_DAYS = 30
GLOBAL_BM25_TOP_K = 5   # candidate tag names to try per metric against company facts
GLOBAL_BM25_MIN_SCORE = 2.0  # lower threshold since taxonomy labels are authoritative

_global_taxonomy = None  # (tag_list[(name, label)], BM25Okapi) — built once per run


def _tokenize_xbrl_tag(tag: str) -> list:
    """Split a PascalCase/camelCase XBRL tag into lowercase tokens, preserving acronyms.

    EBITDACoverageRatio -> ['ebitda', 'coverage', 'ratio']
    NetIncomeLoss       -> ['net', 'income', 'loss']
    """
    # Separate acronym from next capitalised word: EBITDACoverage -> EBITDA Coverage
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tag)
    # Separate camelCase boundary: netIncome -> net Income
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return [t.lower() for t in s.split() if len(t) >= 2]


def _tokenize_metric(metric: str) -> list:
    """Tokenize a human-readable metric name into lowercase tokens."""
    return [t.lower() for t in re.split(r'[\s\-_/]+', metric) if len(t) >= 2]


def _build_global_taxonomy_index():
    """Fetch the full US-GAAP label linkbase once, cache to disk, build a BM25 index.

    Returns (tags, bm25) where tags is a list of (tag_name, label) tuples.
    Falls back to an empty index on any failure so callers can degrade gracefully.
    """
    global _global_taxonomy
    if _global_taxonomy is not None:
        return _global_taxonomy

    # Load from disk cache if fresh enough
    if os.path.exists(TAXONOMY_CACHE_FILE):
        try:
            with open(TAXONOMY_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_at < timedelta(days=TAXONOMY_CACHE_DAYS):
                tag_data = cached['tags']
                tags = [(d['tag'], d['label']) for d in tag_data]
                tokenized = [
                    _tokenize_metric(label) if label else _tokenize_xbrl_tag(tag)
                    for tag, label in tags
                ]
                bm25 = BM25Okapi(tokenized)
                print(f"✓ Global taxonomy loaded from cache ({len(tags)} concepts)")
                _global_taxonomy = (tags, bm25)
                return _global_taxonomy
        except Exception as e:
            print(f"⚠ Taxonomy cache unreadable ({e}), re-fetching...")

    print("Fetching full US-GAAP taxonomy labels (one-time download, will be cached)...")
    try:
        resp = requests.get(TAXONOMY_URL, headers=headers, verify=False, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠ Could not fetch taxonomy ({e}). Global lookup disabled — falling back to per-company BM25.")
        _global_taxonomy = ([], None)
        return _global_taxonomy

    XLINK = 'http://www.w3.org/1999/xlink'
    LINK  = 'http://www.xbrl.org/2003/linkbase'

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"⚠ Taxonomy XML parse error ({e}). Global lookup disabled.")
        _global_taxonomy = ([], None)
        return _global_taxonomy

    tag_label_map = {}  # concept_name -> label text

    for label_link in root.iter(f'{{{LINK}}}labelLink'):
        # loc xlink:label -> concept name
        loc_to_concept = {}
        for loc in label_link.findall(f'{{{LINK}}}loc'):
            href  = loc.get(f'{{{XLINK}}}href', '')
            lbl   = loc.get(f'{{{XLINK}}}label', '')
            if '#' in href and lbl:
                concept = href.split('#')[-1]
                # strip namespace prefix: us-gaap_OperatingIncomeLoss -> OperatingIncomeLoss
                if '_' in concept:
                    concept = concept.split('_', 1)[1]
                loc_to_concept[lbl] = concept

        # label xlink:label -> text  (standard label role only)
        lbl_to_text = {}
        for lbl_elem in label_link.findall(f'{{{LINK}}}label'):
            role = lbl_elem.get(f'{{{XLINK}}}role', '')
            lbl  = lbl_elem.get(f'{{{XLINK}}}label', '')
            text = (lbl_elem.text or '').strip()
            if role.endswith('/label') and lbl and text:
                lbl_to_text[lbl] = text

        # wire together via arcs
        for arc in label_link.findall(f'{{{LINK}}}labelArc'):
            arcrole  = arc.get(f'{{{XLINK}}}arcrole', '')
            from_lbl = arc.get(f'{{{XLINK}}}from', '')
            to_lbl   = arc.get(f'{{{XLINK}}}to', '')
            if (arcrole.endswith('/concept-label')
                    and from_lbl in loc_to_concept
                    and to_lbl in lbl_to_text):
                tag_label_map[loc_to_concept[from_lbl]] = lbl_to_text[to_lbl]

    if not tag_label_map:
        print("⚠ Taxonomy parsed but no concepts found. Global lookup disabled.")
        _global_taxonomy = ([], None)
        return _global_taxonomy

    # Persist to disk
    try:
        payload = {
            'cached_at': datetime.now().isoformat(),
            'tags': [{'tag': t, 'label': l} for t, l in tag_label_map.items()],
        }
        with open(TAXONOMY_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"⚠ Could not write taxonomy cache ({e})")

    tags = list(tag_label_map.items())
    tokenized = [
        _tokenize_metric(label) if label else _tokenize_xbrl_tag(tag)
        for tag, label in tags
    ]
    bm25 = BM25Okapi(tokenized)
    print(f"✓ Global taxonomy built: {len(tags)} concepts, cached to disk")
    _global_taxonomy = (tags, bm25)
    return _global_taxonomy


_STOP_WORDS = {'the', 'and', 'or', 'of', 'in', 'to', 'a', 'an', 'for', 'net', 'total'}


def _resolve_metric_to_tags(metric_type: str) -> list:
    """Return up to GLOBAL_BM25_TOP_K XBRL tag names that best match metric_type
    according to the full US-GAAP taxonomy index.  Returns [] if taxonomy unavailable.
    """
    tags, bm25 = _build_global_taxonomy_index()
    if not tags or bm25 is None:
        return []

    raw_tokens = _tokenize_metric(metric_type)
    query_tokens = [t for t in raw_tokens if t not in _STOP_WORDS] or raw_tokens
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:GLOBAL_BM25_TOP_K]
    return [tags[i][0] for i in top_indices if scores[i] >= GLOBAL_BM25_MIN_SCORE]


def analyze_company_covenants(cik, company_name, target_metrics=None, default_fiscal_year=2024):
    """Analyze a single company for target metrics.
    
    Args:
        cik: Company CIK number
        company_name: Company name
        target_metrics: List of target metrics to match (with optional year per metric)
        default_fiscal_year: Default fiscal year to use if metric doesn't specify one
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {})
        
        target_metric_results = {}

        if not target_metrics:
            return target_metric_results

        # Phase 1: collect every (taxonomy, tag_name, tag_content) tuple once
        #          and build a fast name→(taxonomy, content) lookup
        all_tags = [
            (taxonomy_name, tag_name, tag_content)
            for taxonomy_name, taxonomy_data in facts.items()
            for tag_name, tag_content in taxonomy_data.items()
        ]

        if not all_tags:
            return target_metric_results

        facts_by_tag = {tag_name: (taxonomy_name, tag_content)
                        for taxonomy_name, tag_name, tag_content in all_tags}

        # Phase 2 (fallback): per-company BM25 index — only used when global lookup misses
        def _tag_tokens(tag_name, tag_content):
            label = tag_content.get('label', '')
            return _tokenize_metric(label) if label else _tokenize_xbrl_tag(tag_name)

        tokenized_tags = [_tag_tokens(tn, tc) for _, tn, tc in all_tags]
        bm25_fallback = BM25Okapi(tokenized_tags)

        def _extract_tag_data(taxonomy_name, tag_name, tag_content, metric_type, target_year, score, match_source, gaap_concept=None):
            units = tag_content.get('units', {})
            entry = {
                'taxonomy': taxonomy_name,
                'tag': tag_name,
                'label': tag_content.get('label', ''),
                'description': tag_content.get('description', ''),
                'metric_type': metric_type,
                'gaap_concept': gaap_concept or metric_type,
                'target_year': target_year,
                'match_source': match_source,
                'bm25_score': round(float(score), 4),
                'units': {},
                'metadata': {'cik': cik, 'source_url': url},
            }
            for unit_name, entries in units.items():
                filtered = [
                    {
                        'end': e.get('end'),
                        'val': e.get('val'),
                        'accn': e.get('accn'),
                        'fy': e.get('fy'),
                        'fp': e.get('fp'),
                        'form': e.get('form'),
                        'filed': e.get('filed', ''),
                    }
                    for e in entries
                    if e.get('fp') == 'FY'
                    and e.get('form') == '10-K'
                    and START_DATE <= (e.get('end') or '') <= END_DATE
                ]
                if filtered:
                    entry['units'][unit_name] = filtered
            return entry if entry['units'] else None

        # Phase 3: for each metric, try global taxonomy lookup first; fall back to per-company BM25
        for metric in target_metrics:
            metric_type = metric.get('type', '')
            if not metric_type:
                continue

            gaap_concept = metric.get('gaap_concept') or metric_type
            target_year = metric.get('year') or default_fiscal_year
            matched = False

            # --- Global taxonomy path ---
            # Prefer gaap_concept (standardised GAAP name) over the verbatim metric_type
            # so that "Net income attributable to MTI" is resolved as "Net Income (Loss)"
            search_term = gaap_concept if gaap_concept != metric_type else metric_type
            candidate_tags = _resolve_metric_to_tags(search_term)
            # If gaap_concept search returned nothing, retry with raw metric_type
            if not candidate_tags and search_term != metric_type:
                candidate_tags = _resolve_metric_to_tags(metric_type)
            for tag_name in candidate_tags:
                if tag_name in facts_by_tag:
                    taxonomy_name, tag_content = facts_by_tag[tag_name]
                    entry = _extract_tag_data(taxonomy_name, tag_name, tag_content,
                                              metric_type, target_year, 0.0, 'global_taxonomy',
                                              gaap_concept=gaap_concept)
                    if entry:
                        target_metric_results.setdefault(metric_type, []).append(entry)
                        matched = True
                        break  # first candidate with data wins

            if matched:
                continue

            # --- Per-company BM25 fallback ---
            raw_tokens = _tokenize_metric(search_term)
            query_tokens = [t for t in raw_tokens if t not in _STOP_WORDS] or raw_tokens
            if not query_tokens:
                continue

            scores = bm25_fallback.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]
            top_indices = [i for i in top_indices if scores[i] >= BM25_MIN_SCORE]

            for idx in top_indices:
                taxonomy_name, tag_name, tag_content = all_tags[idx]
                entry = _extract_tag_data(taxonomy_name, tag_name, tag_content,
                                          metric_type, target_year, scores[idx], 'bm25_fallback',
                                          gaap_concept=gaap_concept)
                if entry:
                    target_metric_results.setdefault(metric_type, []).append(entry)

        return target_metric_results
    
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
    print("TARGET METRICS ANALYSIS - BATCH PROCESSING")
    print("=" * 80)
    print()
    
    # Get target company metrics
    try:
        target_metrics = get_target_company_metrics()
    except Exception as e:
        print(f"Error getting target metrics: {e}\n")
        target_metrics = []
    
    print("=" * 80)
    print("STEP 1: CHECKING NEO4J GRAPH FOR PEER COMPANIES")
    print("=" * 80)
    print()
    
    companies = get_companies_from_neo4j()
    
    companies_with_cik = [c for c in companies if c.get('cik')]
    companies_without_cik = [c for c in companies if not c.get('cik')]
    
    if companies_without_cik:
        print(f"⚠ Skipping {len(companies_without_cik)} companies without CIK:")
        for c in companies_without_cik:
            print(f"  - {c['name']}")
        print()
    
    if len(companies_with_cik) < MAX_COMPANIES:
        needed = MAX_COMPANIES - len(companies_with_cik)
        print("=" * 80)
        print(f"STEP 2: FETCHING ADDITIONAL COMPANIES FROM SEC (need {needed} more)")
        print("=" * 80)
        print()
        
        try:
            sic_code = get_sic_from_neo4j()
            print(f"✓ Using SIC code: {sic_code}\n")
            sic_codes = [sic_code]
        except Exception as e:
            print(f"Error getting SIC from Neo4j: {e}")
            print("Falling back to default SIC code: 1311\n")
            sic_codes = ['1311']
        
        sec_companies = fetch_companies_by_sic(sic_codes, START_DATE, END_DATE, FORM_TYPE, needed)
        
        for c in sec_companies:
            c['source'] = 'sec'
        
        existing_ciks = {c['cik'] for c in companies_with_cik}
        new_companies = [c for c in sec_companies if c['cik'] not in existing_ciks]
        
        companies_with_cik.extend(new_companies[:needed])
        
        print(f"✓ Added {len(new_companies[:needed])} companies from SEC")
        print(f"✓ Total companies to analyze: {len(companies_with_cik)}\n")
    else:
        print(f"✓ Using {len(companies_with_cik)} companies from Neo4j graph (limit: {MAX_COMPANIES})\n")
        companies_with_cik = companies_with_cik[:MAX_COMPANIES]
    
    companies = companies_with_cik
    
    if not companies:
        print("No companies to analyze. Exiting.")
        return
    
    all_results = {}
    companies_with_metrics = []
    companies_without_metrics = []
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
        
        metric_results = analyze_company_covenants(
            cik, name, target_metrics, FISCAL_YEAR
        )
        
        if metric_results is None:
            companies_with_errors.append(company)
            print(f"  ⚠ Data not available or error occurred\n")
        else:
            has_metrics = bool(metric_results)
            
            if has_metrics:
                # Count total tags found
                total_metric_matches = sum(len(matches) for matches in metric_results.values())
                
                company_data = {
                    'company': company,
                    'target_metric_count': total_metric_matches,
                    'target_metrics': metric_results
                }
                
                companies_with_metrics.append(company_data)
                
                print(f"  ✓ Found {total_metric_matches} target metric match(es)")
                for metric_name, matches in metric_results.items():
                    if matches:
                        print(f"    - {metric_name}: {len(matches)} match(es)")
                
                print()
            else:
                companies_without_metrics.append(company)
                print(f"  ✗ No target metrics found\n")
        
        time.sleep(0.2)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    neo4j_count = sum(1 for c in companies if c.get('source') == 'neo4j')
    sec_count = sum(1 for c in companies if c.get('source') == 'sec')
    print(f"Companies from Neo4j graph: {neo4j_count}")
    print(f"Companies from SEC EDGAR: {sec_count}")
    print(f"Total companies analyzed: {len(companies)}")
    print(f"Companies with target metrics: {len(companies_with_metrics)}")
    print(f"Companies without target metrics: {len(companies_without_metrics)}")
    print(f"Companies with errors: {len(companies_with_errors)}")
    print()
    
    if companies_with_metrics:
        print("=" * 80)
        print("COMPANIES WITH TARGET METRICS")
        print("=" * 80)
        print()
        
        for item in companies_with_metrics:
            company = item['company']
            print(f"{company['name']} ({company['ticker']}) - CIK: {company['cik']}")
            print(f"  Total metric matches: {item['target_metric_count']}")
            
            for metric_type, matches in item['target_metrics'].items():
                if matches:
                    print(f"  {metric_type}:")
                    for match in matches:
                        print(f"    - {match['tag']}")
                        # Show latest value
                        for unit_name, entries in match['units'].items():
                            if entries:
                                latest = entries[-1]
                                print(f"      Latest: {latest['val']} ({unit_name}) as of {latest['end']}")
                                break
            print()
    
    # Save comprehensive results
    output_data = {
        'metadata': {
            'neo4j_companies': neo4j_count,
            'sec_companies': sec_count,
            'target_metrics_searched': [m['name'] for m in target_metrics] if target_metrics else [],
            'date_range': {'start': START_DATE, 'end': END_DATE},
            'form_type': FORM_TYPE,
            'analysis_date': datetime.now().isoformat(),
            'total_companies': len(companies),
            'companies_with_metrics': len(companies_with_metrics),
            'companies_without_metrics': len(companies_without_metrics),
            'companies_with_errors': len(companies_with_errors)
        },
        'companies_with_metrics': companies_with_metrics,
        'companies_without_metrics': companies_without_metrics,
        'companies_with_errors': companies_with_errors
    }
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies_metrices.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Complete results saved to: {output_file}")


if __name__ == "__main__":
    main()
