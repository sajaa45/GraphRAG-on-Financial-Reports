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

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FISCAL_YEAR = int(os.getenv("FISCAL_YEAR", 2024))
START_DATE = f"{FISCAL_YEAR}-07-01"
END_DATE = f"{FISCAL_YEAR + 1}-06-30"

headers = {"User-Agent": "User (your_email@example.com)"}


def get_companies_from_neo4j(driver):
    """Get peer companies from the Neo4j graph via COMPETES_WITH → TargetCompany edges."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Company {is_peer: true})-[:COMPETES_WITH]->(tc:TargetCompany)
            RETURN c.name AS name, c.cik AS cik, c.ticker AS ticker
            ORDER BY c.name
            """
        )
        companies = []
        for record in result:
            name = record.get("name")
            cik = record.get("cik")
            if name:
                companies.append({
                    'name': name,
                    'cik': str(cik).zfill(10) if cik else None,
                    'ticker': record.get("ticker") or 'N/A',
                })

    if companies:
        print(f"✓ Found {len(companies)} peer companies in Neo4j graph:")
        for c in companies:
            print(f"  - {c['name']} ({c['ticker']}) - {c['cik'] or 'No CIK'}")
        print()
    else:
        print("⚠ No peer companies found in Neo4j graph\n")

    return companies


def get_target_company_metrics(driver, company_name: str = None):
    """Query Neo4j for metrics of the target company."""
    with driver.session() as session:
        if company_name:
            result = session.run(
                """
                MATCH (c:TargetCompany {name: $name})-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory {company: $name})-[:HAS_METRIC]->(m:Metric {company: $name})
                RETURN m.name AS metric_name, m.metric_type AS metric_type,
                       m.gaap_concept AS gaap_concept, m.year AS year, mc.name AS category
                """,
                {"name": company_name},
            )
        else:
            result = session.run(
                """
                MATCH (c:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
                WHERE mc.company = c.name AND m.company = c.name
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

            if not metric_type and metric_name:
                if '(' in metric_name:
                    metric_type = metric_name.split('(')[0].strip()
                    year_part = metric_name.split('(')[1].split(')')[0].strip()
                    if year_part.isdigit():
                        year = int(year_part)
                else:
                    metric_type = metric_name

            if year and isinstance(year, str) and year.isdigit():
                year = int(year)

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


BM25_TOP_K = 1
BM25_MIN_SCORE = 3.0

TAXONOMY_URL = "https://xbrl.fasb.org/us-gaap/2024/elts/us-gaap-lab-2024.xml"
TAXONOMY_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".taxonomy_cache.json")
TAXONOMY_CACHE_DAYS = 30
GLOBAL_BM25_TOP_K = 15
GLOBAL_BM25_MIN_SCORE = 2.0

_global_taxonomy = None


def _tokenize_xbrl_tag(tag: str) -> list:
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tag)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return [t.lower() for t in s.split() if len(t) >= 2]


def _stem(token: str) -> str:
    for suffix in ('ings', 'ing', 'ies', 'es', 's'):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _tokenize_metric(metric: str) -> list:
    return [_stem(t.lower()) for t in re.split(r'[\s\-_/]+', metric) if len(t) >= 2]


def _build_global_taxonomy_index():
    global _global_taxonomy
    if _global_taxonomy is not None:
        return _global_taxonomy

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
        print(f"⚠ Could not fetch taxonomy ({e}). Global lookup disabled.")
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

    tag_label_map = {}

    for label_link in root.iter(f'{{{LINK}}}labelLink'):
        loc_to_concept = {}
        for loc in label_link.findall(f'{{{LINK}}}loc'):
            href  = loc.get(f'{{{XLINK}}}href', '')
            lbl   = loc.get(f'{{{XLINK}}}label', '')
            if '#' in href and lbl:
                concept = href.split('#')[-1]
                if '_' in concept:
                    concept = concept.split('_', 1)[1]
                loc_to_concept[lbl] = concept

        lbl_to_text = {}
        for lbl_elem in label_link.findall(f'{{{LINK}}}label'):
            role = lbl_elem.get(f'{{{XLINK}}}role', '')
            lbl  = lbl_elem.get(f'{{{XLINK}}}label', '')
            text = (lbl_elem.text or '').strip()
            if role.endswith('/label') and lbl and text:
                lbl_to_text[lbl] = text

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


_STOP_WORDS = {'the', 'and', 'or', 'of', 'in', 'to', 'a', 'an', 'for'}

_CONCEPT_TAG_MAP: dict[str, list[str]] = {
    "revenue":                          ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net sales":                        ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net revenue":                      ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cost of goods sold":               ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "cost of sales":                    ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross profit":                     ["GrossProfit"],
    "operating income":                 ["OperatingIncomeLoss"],
    "operating income (loss)":          ["OperatingIncomeLoss"],
    "income from operations":           ["OperatingIncomeLoss"],
    "net income":                       ["NetIncomeLoss"],
    "net income (loss)":                ["NetIncomeLoss"],
    "net income attributable to non-controlling interests": ["NetIncomeLossAttributableToNoncontrollingInterest"],
    "net income attributable to noncontrolling interests":  ["NetIncomeLossAttributableToNoncontrollingInterest"],
    "equity in earnings of affiliates": ["IncomeLossFromEquityMethodInvestments"],
    "equity in earnings":               ["IncomeLossFromEquityMethodInvestments"],
    "selling general and administrative expenses": ["SellingGeneralAndAdministrativeExpense"],
    "marketing and administrative expenses":       ["SellingGeneralAndAdministrativeExpense"],
    "research and development expenses": ["ResearchAndDevelopmentExpense"],
    "research and development":          ["ResearchAndDevelopmentExpense"],
    "income tax expense":               ["IncomeTaxExpenseBenefit"],
    "income tax expense (benefit)":     ["IncomeTaxExpenseBenefit"],
    "provision for income taxes":       ["IncomeTaxExpenseBenefit"],
    "provision for taxes on income":    ["IncomeTaxExpenseBenefit"],
    "total tax provision":              ["IncomeTaxExpenseBenefit"],
    "interest expense":                 ["InterestExpense", "InterestAndDebtExpense"],
    "interest expense net":             ["InterestExpense", "InterestAndDebtExpense"],
    "depreciation and amortization":    ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "depreciation and depletion":       ["DepreciationDepletionAndAmortization", "Depreciation"],
    "amortization expense":             ["AmortizationOfIntangibleAssets", "DepreciationDepletionAndAmortization"],
    "earnings per share (basic)":       ["EarningsPerShareBasic"],
    "earnings per share (diluted)":     ["EarningsPerShareDiluted"],
    "basic earnings per share":         ["EarningsPerShareBasic"],
    "diluted earnings per share":       ["EarningsPerShareDiluted"],
    "weighted average shares outstanding":           ["WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding"],
    "weighted average shares outstanding (diluted)": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "long-term debt":                   ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short-term debt":                  ["ShortTermBorrowings", "NotesPayableCurrent"],
    "current maturities of long-term debt": ["LongTermDebtCurrent"],
    "debt extinguishment costs":        ["GainsLossesOnExtinguishmentOfDebt"],
    "total assets":                     ["Assets"],
    "total liabilities":                ["Liabilities"],
    "total equity":                     ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total shareholders equity":        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "stockholders equity":              ["StockholdersEquity"],
    "retained earnings":                ["RetainedEarningsAccumulatedDeficit"],
    "comprehensive income":             ["ComprehensiveIncomeNetOfTax", "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest"],
    "goodwill":                         ["Goodwill"],
    "intangible assets":                ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "property plant and equipment net": ["PropertyPlantAndEquipmentNet"],
    "property plant and equipment":     ["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentGross"],
    "cash and cash equivalents":        ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "total inventories":                ["InventoryNet"],
    "capital expenditures":             ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "capital expenditure":              ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "net cash provided by operating activities": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash dividends paid":              ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "dividends paid":                   ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "deferred tax assets":              ["DeferredTaxAssetsGross", "DeferredTaxAssetsNet"],
    "deferred tax liabilities":         ["DeferredTaxLiabilities", "DeferredIncomeTaxLiabilitiesNet"],
    "net deferred tax liability":       ["DeferredIncomeTaxLiabilitiesNet", "DeferredTaxLiabilities"],
    "unrecognized tax benefits":        ["UnrecognizedTaxBenefits"],
    "restructuring charges":            ["RestructuringCharges"],
    "operating lease cost":             ["OperatingLeaseCost", "LeaseCost"],
    "operating lease liability":        ["OperatingLeaseLiability"],
    "stock based compensation expense": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "pension expense":                  ["DefinedBenefitPlanNetPeriodicBenefitCost"],
    "projected benefit obligation":     ["DefinedBenefitPlanBenefitObligation"],
    "plan assets":                      ["DefinedBenefitPlanFairValueOfPlanAssets"],
    "impairment loss":                  ["AssetImpairmentCharges", "GoodwillImpairmentLoss"],
    "accumulated other comprehensive loss": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
}


def _lookup_hardcoded(concept: str) -> list[str]:
    key = re.sub(r'\s+', ' ', concept.lower().strip())
    if key in _CONCEPT_TAG_MAP:
        return _CONCEPT_TAG_MAP[key]
    key2 = re.sub(r'\s*\(.*?\)', '', key).strip()
    if key2 in _CONCEPT_TAG_MAP:
        return _CONCEPT_TAG_MAP[key2]
    if key.endswith('s') and key[:-1] in _CONCEPT_TAG_MAP:
        return _CONCEPT_TAG_MAP[key[:-1]]
    return []


def _resolve_metric_to_tags(metric_type: str) -> list:
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
    fy = int(default_fiscal_year)
    start_date = f"{fy}-07-01"
    end_date   = f"{fy + 1}-06-30"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {})

        target_metric_results = {}

        if not target_metrics:
            return target_metric_results

        all_tags = [
            (taxonomy_name, tag_name, tag_content)
            for taxonomy_name, taxonomy_data in facts.items()
            for tag_name, tag_content in taxonomy_data.items()
        ]

        if not all_tags:
            return target_metric_results

        facts_by_tag = {tag_name: (taxonomy_name, tag_content)
                        for taxonomy_name, tag_name, tag_content in all_tags}

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
                    and start_date <= (e.get('end') or '') <= end_date
                ]
                if filtered:
                    entry['units'][unit_name] = filtered
            return entry if entry['units'] else None

        for metric in target_metrics:
            metric_type = metric.get('type', '')
            if not metric_type:
                continue

            gaap_concept = metric.get('gaap_concept') or metric_type
            target_year = metric.get('year') or default_fiscal_year
            matched = False

            search_term = gaap_concept if gaap_concept != metric_type else metric_type

            candidate_tags = _lookup_hardcoded(search_term)
            if not candidate_tags and search_term != metric_type:
                candidate_tags = _lookup_hardcoded(metric_type)

            if not candidate_tags:
                candidate_tags = _resolve_metric_to_tags(search_term)
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
                        break

            if matched:
                continue

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
            return None
        print(f"  HTTP Error for {company_name}: {e}")
        return None
    except Exception as e:
        print(f"  Error analyzing {company_name}: {e}")
        return None


def main():
    print("=" * 80)
    print("TARGET METRICS ANALYSIS - BATCH PROCESSING")
    print("=" * 80)
    print()

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )

    try:
        target_metrics = get_target_company_metrics(driver)
        companies = get_companies_from_neo4j(driver)
    finally:
        driver.close()

    companies = [c for c in companies if c.get('cik')]

    if not companies:
        print("No companies with CIK to analyze. Exiting.")
        return

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

        metric_results = analyze_company_covenants(cik, name, target_metrics, FISCAL_YEAR)

        if metric_results is None:
            companies_with_errors.append(company)
            print(f"  ⚠ Data not available or error occurred\n")
        elif metric_results:
            total_matches = sum(len(m) for m in metric_results.values())
            companies_with_metrics.append({
                'company': company,
                'target_metric_count': total_matches,
                'target_metrics': metric_results,
            })
            print(f"  ✓ Found {total_matches} target metric match(es)")
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
    print(f"Total companies analyzed:        {len(companies)}")
    print(f"Companies with target metrics:   {len(companies_with_metrics)}")
    print(f"Companies without target metrics:{len(companies_without_metrics)}")
    print(f"Companies with errors:           {len(companies_with_errors)}")
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
                        for unit_name, entries in match['units'].items():
                            if entries:
                                latest = entries[-1]
                                print(f"      Latest: {latest['val']} ({unit_name}) as of {latest['end']}")
                                break
            print()

    output_data = {
        'metadata': {
            'target_metrics_searched': [m['name'] for m in target_metrics] if target_metrics else [],
            'date_range': {'start': START_DATE, 'end': END_DATE},
            'fiscal_year': FISCAL_YEAR,
            'analysis_date': datetime.now().isoformat(),
            'total_companies': len(companies),
            'companies_with_metrics': len(companies_with_metrics),
            'companies_without_metrics': len(companies_without_metrics),
            'companies_with_errors': len(companies_with_errors),
        },
        'companies_with_metrics': companies_with_metrics,
        'companies_without_metrics': companies_without_metrics,
        'companies_with_errors': companies_with_errors,
    }

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "companies_metrices.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
