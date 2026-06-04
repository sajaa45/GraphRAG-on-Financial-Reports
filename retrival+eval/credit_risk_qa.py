import os
import re
import json
import time
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo4j import GraphDatabase
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# LangSmith tracing — activate by setting LANGCHAIN_API_KEY in .env
os.environ.setdefault("LANGCHAIN_TRACING_V2",
                      "true" if os.getenv("LANGCHAIN_API_KEY") else "false")
os.environ.setdefault("LANGCHAIN_PROJECT", "PeersGraphRAG")
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

# ---------------------------------------------------------------------------
# Graph schema (injected into every Cypher-generation prompt)
# ---------------------------------------------------------------------------

GRAPH_SCHEMA = """
Node labels and key properties:
  - TargetCompany   {name, cik, ticker, is_target:true, filing_date, document_url}
  - Company         {name, cik, ticker, is_peer:true, filing_date, document_url}
  - MetricCategory  {name}
  - Metric          {citation_id, name, value, unit, year, metric_type, gaap_concept, xbrl_tag, label, source_url, cik, source_page, section_title, source_text}
                    NOTE: name = "{metric_type} ({year})" — use gaap_concept to align target vs peer metrics
                    CRITICAL: Always return citation_id for citations
                    CRITICAL: When comparing target vs peer metrics, match by gaap_concept — not by name or label
                    CRITICAL: Always include source_page, section_title, and source_text in collect() for target_metrics
  - Risk            {citation_id, risk_id, name, description, why, source_text, document_url, filing_date, section_title, source_page}
                    CRITICAL: Always return citation_id for citations
  - Industry        {name, sector}
  - SICCode         {code, industry, sector}

Relationships:
  (TargetCompany|Company)-[:HAS_METRIC_CATEGORY]->(:MetricCategory)
  (:MetricCategory)-[:HAS_METRIC]->(:Metric)
  (TargetCompany|Company)-[:FACES_RISK]->(:Risk)
  (:Company {is_peer:true})-[:COMPETES_WITH]->(:TargetCompany)
  (TargetCompany|Company)-[:OPERATES_IN]->(:Industry)
  (:Industry)-[:HAS_SIC_CODE]->(:SICCode)

Key rules:
  - TargetCompany is the company being analyzed. Peers are Company nodes with is_peer:true
    linked via (:Company)-[:COMPETES_WITH]->(:TargetCompany).
  - Metric.value is a string — use toFloat() for numeric comparisons.
  - Risk comparison across companies: match by risk.name, NOT risk_id.
  - For risk text search, always check ALL four fields: name, description, why, source_text.
  - Use properties(node) to return all node properties instead of hardcoding field names.
  - Always use DISTINCT on risk rows to avoid duplicates from OPTIONAL MATCH joins.
  - CRITICAL: WHERE must follow MATCH/OPTIONAL MATCH directly, not come after WITH.
  - CRITICAL: Every variable used after WITH must be carried in that WITH clause.
  - CRITICAL: Always return citation_id for every Risk and Metric node for citation traceability.
"""

# ---------------------------------------------------------------------------
# Prompts — tune these to change how the LLM generates and answers queries
# ---------------------------------------------------------------------------

KNOWN_CATEGORIES = {"Leverage", "Coverage", "Liquidity", "Profitability", "Debt Structure"}

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "categories", "target_company", "question"],
    template="""You are a Neo4j Cypher expert for a credit-risk knowledge graph.
Generate a single valid Cypher query that retrieves all data needed to answer the question.

Schema:
{schema}

Known MetricCategory names (use EXACT spelling):
{categories}

Target company in this graph: {target_company}
CRITICAL: Never filter TargetCompany by name. The graph contains exactly ONE TargetCompany node.
Always match it as (tc:TargetCompany) — no {{name: ...}} or other property filter.

Rules:
1. Output ONLY the Cypher query — no markdown, no comments, no explanation.
2. No write operations (CREATE, MERGE, SET, DELETE).
3. Choose LIMIT based on what the query returns:
   - Queries that use COLLECT to aggregate (one row per company): LIMIT 20
   - Queries returning individual metric rows (no COLLECT): LIMIT 50
   - Queries returning individual risk rows (no COLLECT): LIMIT 100
   - Broad "list all" questions explicitly asking for everything: LIMIT 200
   Never omit LIMIT — always include it at the end of the query.
4. Use OPTIONAL MATCH for peer data so the query works even with no peers.
5. Use TWO separate OPTIONAL MATCHes for peers: first find the peer via COMPETES_WITH, then find its data in a second OPTIONAL MATCH.
6. NEVER chain COMPETES_WITH and FACES_RISK in one path — that returns target risks, not peer risks.
7. Use COLLECT to aggregate peer data into lists — never produce cartesian products between target rows and peer rows.
8. Use toFloat() for numeric comparisons on Metric.value.
9. For risks, search all four fields: name, description, why, source_text.
10. For metrics, fetch target metrics and peer metrics independently — both filtered by the same MetricCategory.
    Do NOT try to join them row-by-row in Cypher. Collect each side into a list and return them separately.
    The LLM will align target and peer metrics by label/name after retrieval.
11. When filtering MetricCategory, map the question intent to the closest known category name.
12. CRITICAL — risk keyword rules:
    a. NEVER use a company name (target or peer) as a WHERE keyword for risks. Company names do NOT appear
       in the risk text of a competitor's filing. Using one as a filter will return zero peer risks.
    b. Only use TOPIC keywords extracted from the question (e.g. 'debt', 'credit', 'liquidity', 'climate').
    c. If the question asks for ALL risks or MAIN risks without specifying a topic keyword, OMIT the WHERE
       clause entirely on both target and peer risks — return every risk node matched by the relationship.
13. CRITICAL — always cap collected arrays to avoid returning the entire graph:
    - Use collect(r)[0..15] for target risks and collect(pr)[0..10] for peer risks.
    - Use collect(m)[0..20] for target metrics and collect(pm)[0..15] for peer metrics.
    Apply the slice INSIDE collect: collect({{...}})[0..N] — never slice after a WITH.

Reference patterns:

Risks with peer comparison — topic-filtered (e.g. question mentions "debt"):
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE toLower(r.description) CONTAINS 'debt' OR toLower(r.name) CONTAINS 'debt'
           OR toLower(r.why) CONTAINS 'debt' OR toLower(r.source_text) CONTAINS 'debt'
    WITH tc, collect({{citation_id: r.citation_id, risk_id: r.risk_id, name: r.name, description: r.description, why: r.why, source_text: r.source_text, document_url: r.document_url, section_title: r.section_title, source_page: r.source_page}})[0..15] AS target_risks
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WHERE toLower(pr.description) CONTAINS 'debt' OR toLower(pr.name) CONTAINS 'debt'
           OR toLower(pr.why) CONTAINS 'debt' OR toLower(pr.source_text) CONTAINS 'debt'
    WITH tc, target_risks, peer, collect({{citation_id: pr.citation_id, risk_id: pr.risk_id, name: pr.name, description: pr.description, why: pr.why, source_text: pr.source_text, document_url: pr.document_url, section_title: pr.section_title, source_page: pr.source_page}})[0..10] AS peer_risks
    RETURN tc.name AS target, target_risks, peer.name AS peer, peer_risks
    LIMIT 20

Risks with peer comparison — ALL risks (e.g. question asks "what are the main risks"):
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WITH tc, collect({{citation_id: r.citation_id, risk_id: r.risk_id, name: r.name, description: r.description, why: r.why, source_text: r.source_text, document_url: r.document_url, section_title: r.section_title, source_page: r.source_page}})[0..15] AS target_risks
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WITH tc, target_risks, peer, collect({{citation_id: pr.citation_id, risk_id: pr.risk_id, name: pr.name, description: pr.description, why: pr.why, source_text: pr.source_text, document_url: pr.document_url, section_title: pr.section_title, source_page: pr.source_page}})[0..10] AS peer_risks
    RETURN tc.name AS target, target_risks, peer.name AS peer, peer_risks
    LIMIT 20

Metrics with peer comparison (fetch independently by category — let the LLM align by label):
    MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
    WHERE mc.name = 'Profitability'
    WITH tc, mc, collect({{citation_id: m.citation_id, name: m.name, label: m.label, gaap_concept: m.gaap_concept, value: m.value, unit: m.unit, year: m.year, metric_type: m.metric_type, xbrl_tag: m.xbrl_tag, source_page: m.source_page, section_title: m.section_title, source_text: m.source_text}}) AS target_metrics
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:HAS_METRIC_CATEGORY]->(pmc:MetricCategory {{name: mc.name}})-[:HAS_METRIC]->(pm:Metric)
    WITH tc, mc, target_metrics, peer, collect({{citation_id: pm.citation_id, name: pm.name, label: pm.label, gaap_concept: pm.gaap_concept, value: pm.value, unit: pm.unit, year: pm.year, metric_type: pm.metric_type, xbrl_tag: pm.xbrl_tag, source_url: pm.source_url}}) AS peer_metrics
    RETURN tc.name AS target, mc.name AS category, target_metrics, peer.name AS peer, peer_metrics
    LIMIT 20

Question: {question}

Cypher query:""",
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "target_company"],
    template="""You are a senior credit-risk analyst providing KYC assessments.
Answer the question using only the graph results below.

The TargetCompany in this knowledge graph is: {target_company}
CRITICAL: If the question mentions a different company name, treat it as referring to {target_company} — that is the subject of the analysis.

- Structure your response in this EXACT order:
  1. **Sources** — list one document URL per company, taken verbatim from the graph results. Omit any company with no URL. This section comes FIRST, before any analysis.
  2. The analysis body.
  Do NOT repeat or reference URLs anywhere in the analysis body or conclusion — URLs belong only in the Sources section at the top.
- Use ONLY values present in the graph results. Never infer or derive a metric that is not explicitly returned.
- If a metric is missing for a peer, state "not available" rather than estimating it.
- Directly answer the question; compare target vs peers where data exists.
- Flag any elevated risks or concerning metrics.
- SIGN RULE: Report each metric's sign independently from the data. A negative OperatingIncomeLoss is an operating loss; a positive NetIncomeLoss is net income — these are separate line items and one does NOT override the other. Never use one metric's sign to "correct" another.
- INDEPENDENCE RULE: Every metric value must be stated exactly as it appears in the data (sign included). Do NOT reconcile or adjust a value to fit a narrative built from other metrics.
- PROFITABILITY RULE: "Unprofitable" only applies to the specific metric being discussed. A company with negative operating income but positive net income is NOT simply "unprofitable" — state both figures and let the data speak.
- ATTRIBUTION RULE: Each metric belongs to the company in its surrounding context (target row vs peer row). Do NOT infer company ownership from words embedded in the metric name or label.
- Plain language suitable for a credit committee; ≤ 300 words unless data requires more.
- Place a citation IMMEDIATELY after each individual claim — never group multiple citations at the end of a sentence.
  WRONG: "Company X faces risk A and risk B [CITE:ID1][CITE:ID2]"
  RIGHT:  "Company X faces risk A [CITE:ID1] and risk B [CITE:ID2]"
  - Risk claim:   [CITE:<citation_id>]   e.g. [CITE:TARGET_RISK_Competitive_industries] or [CITE:PEER_RISK_1234567_risk_3]
  - Metric claim: [CITE:<citation_id>]   e.g. [CITE:TARGET_METRIC_Net_income_2024] or [CITE:PEER_METRIC_891014_NetIncome_2024]

CRITICAL: The citation_id field is provided in every risk and metric object in the graph results. Use it EXACTLY as shown — do not construct or modify it.

**Sources:**
- [Company Name]: [document_url]

Graph results:
{context}

Question: {question}

Answer:""",
)

REASONING_PROMPT = PromptTemplate(
    input_variables=["context", "question", "queries", "target_company"],
    template="""You are a senior credit-risk analyst with access to structured 10-K filing data.
The TargetCompany in this knowledge graph is: {target_company}
CRITICAL: If the question mentions a different company name, treat it as referring to {target_company}.
A set of graph queries has been executed and the raw results are provided below.
Your task is to produce a transparent, step-by-step reasoning trace that shows exactly
how you arrived at your conclusions, and to cite every source you rely on.

─────────────────────────────────────────
QUERIES EXECUTED
─────────────────────────────────────────
{queries}

─────────────────────────────────────────
RAW GRAPH RESULTS
─────────────────────────────────────────
{context}

─────────────────────────────────────────
QUESTION
─────────────────────────────────────────
{question}

─────────────────────────────────────────
STRICT RULES — follow these before writing anything:
─────────────────────────────────────────
- Use ONLY values explicitly present in the graph results. Never infer, calculate, or derive a value that is not directly in the data.
- If a metric is missing for a company, say "not available" — do not substitute a related value.
- If a document_url is missing for a company, omit it entirely — do not guess or construct one.
- SIGN RULE: Report each metric's sign independently from the data. A negative OperatingIncomeLoss is an operating loss; a positive NetIncomeLoss is net income — these are different line items and one does NOT override the other. Never use one metric's sign to "correct" another metric.
- INDEPENDENCE RULE: State every metric value exactly as it appears in the data. Do NOT adjust or reconcile a value to fit a narrative you built from other metrics.
- PROFITABILITY RULE: "Unprofitable" only applies to the specific metric under discussion. Negative operating income alongside positive net income is a valid financial state (e.g. driven by non-operating gains or tax items) — report both figures separately, do NOT collapse them into a single "unprofitable" label.
- ATTRIBUTION RULE: A metric belongs to the company in its surrounding context (target row vs peer row). Do NOT infer company ownership from words embedded in the metric name or label.

─────────────────────────────────────────
INSTRUCTIONS — produce the sections below in order:
─────────────────────────────────────────

## 1. Understanding the Question
State in one sentence what is being asked and what type of data (risks / metrics /
comparison / other) is needed to answer it.

## 2. Data Retrieved
For each query result, list what was found:
  - Company name and role (target / peer)
  - Data type (risk name, metric name + value + year, etc.)
  - Filing date (use Risk.filing_date or Company.filing_date when available)
  - Document URL (Risk.document_url or Company.document_url)

## 3. Reasoning Steps
Walk through your analysis step by step:
  Step 1 — <what you observed in the raw data>
  Step 2 — <how you compared target vs peers>
  Step 3 — <what pattern or gap stands out>
  … (add as many steps as the data warrants)

## 4. Sources & References
List document URLs grouped by company, using ONLY URLs present in the graph results.
If no document_url exists for a company, omit that company from this section — do NOT invent or guess URLs.

**Target Company:**
  [Company Name] — Filing: <filing_date> (omit URL line if not in results)

**Peer Companies:**
  [Peer 1 Name] — URL: <document_url> (omit if not in results)
  [Peer 2 Name] — URL: <document_url> (omit if not in results)

## 5. Final Answer
A concise credit-committee-ready answer (≤ 200 words) that directly responds to
the question, highlights peer comparisons, and flags any elevated risks or
concerning metrics.
After every claim drawn from a specific data point, append an inline citation using
the citation_id from the graph results:
  - Risk claim:   [CITE:<citation_id>]   e.g. [CITE:TARGET_RISK_Competitive_industries] or [CITE:PEER_RISK_1234567_risk_3]
  - Metric claim: [CITE:<citation_id>]   e.g. [CITE:TARGET_METRIC_Net_income_2024] or [CITE:PEER_METRIC_891014_NetIncome_2024]

CRITICAL: The citation_id field is provided in every risk and metric object. Use it EXACTLY as shown.
""",
)

# Each strategy instructs the LLM to fetch a DIFFERENT type of data so queries
# don't overlap. "risks" → FACES_RISK only. "metrics" → HAS_METRIC only.
# Add or remove strategies here to control what gets queried.
QUERY_STRATEGIES = {
    "risks": (
        "Using ONLY the FACES_RISK relationship, fetch risk nodes for the TargetCompany "
        "AND its peers (Company {{is_peer:true}}). "
        "If the question asks for specific risk TOPICS (e.g. 'debt risk', 'credit risk', 'liquidity'), "
        "filter by those topic keywords — but NEVER filter by the company name itself. "
        "If the question asks for 'all risks', 'main risks', or does not specify a topic, "
        "return ALL risk nodes with NO WHERE clause. "
        "Original question: {question}. "
        "CRITICAL: Use TWO separate OPTIONAL MATCHes — first match peers via COMPETES_WITH, "
        "then match each peer's own risks in a second OPTIONAL MATCH from the peer node. "
        "NEVER chain COMPETES_WITH → FACES_RISK in one path (that returns target risks, not peer risks). "
        "CRITICAL: Use COLLECT to group peer risks — return one row per peer, not one row per "
        "(target_risk × peer_risk) combination. Include ALL peers even if they have no matching risks. "
        "Do NOT fetch any Metric nodes in this query."
    ),
    "metrics": (
        "Using ONLY HAS_METRIC_CATEGORY → HAS_METRIC relationships, fetch metric nodes "
        "for the TargetCompany AND its peers. "
        "Filter MetricCategory to ONLY the categories most relevant to: {question}. "
        "Map the question to the closest category names from: "
        "Leverage, Coverage, Liquidity, Profitability, Debt Structure. "
        "Fetch target metrics and peer metrics independently — both filtered by the same MetricCategory. "
        "Do NOT join them row-by-row in Cypher. "
        "Collect target metrics into a list (including xbrl_tag, source_page, section_title, source_text for validation), "
        "collect each peer's metrics into a separate list (including xbrl_tag for validation), "
        "and return them side by side. Include ALL peers even if they have no data. "
        "Do NOT fetch any Risk nodes in this query."
    ),
}


class _DirectNeo4jGraph:
    """Raw neo4j driver wrapper — no APOC, no LangChain graph abstraction needed."""

    def __init__(self, url: str, username: str, password: str):
        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self._driver.verify_connectivity()

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def close(self):
        self._driver.close()


class CreditRiskQA:
    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")

        # Raw-driver wrapper — no APOC required
        self.graph = _DirectNeo4jGraph(
            url=neo4j_uri,
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )

        self.llm = ChatBedrockConverse(
            model=os.getenv("BEDROCK_MODEL", "qwen.qwen3-next-80b-a3b"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            max_tokens=2048,
            temperature=0.1,
        )
        self._cypher_cache: dict[str, str] = {}

        # Resolve the actual TargetCompany name from the graph once at startup
        try:
            rows = self.graph.query("MATCH (tc:TargetCompany) RETURN tc.name AS name LIMIT 1")
            self._target_company = rows[0]["name"] if rows and rows[0]["name"] else "unknown"
        except Exception:
            self._target_company = "unknown"

        print(f"✓ Neo4j connected  : {neo4j_uri}")
        print(f"✓ LLM model        : {os.getenv('BEDROCK_MODEL', 'qwen.qwen3-next-80b-a3b')}")
        print(f"✓ Target company   : {self._target_company}")

    _CYPHER_FENCE_RE = re.compile(r'```(?:cypher)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)

    def _generate_cypher(self, question: str, error_context: str = "") -> str:
        """Call the LLM to generate (or fix) a Cypher query."""
        if not error_context and question in self._cypher_cache:
            return self._cypher_cache[question]

        base_prompt = CYPHER_GENERATION_PROMPT.format(
            schema=GRAPH_SCHEMA,
            categories=", ".join(sorted(KNOWN_CATEGORIES)),
            target_company=self._target_company,
            question=question,
        )
        if error_context:
            base_prompt += (
                f"\n\nThe previous attempt produced this Neo4j error — fix it:\n{error_context}"
                "\n\nCorrected Cypher query:"
            )
        raw = self._call_llm_raw(base_prompt)
        m = self._CYPHER_FENCE_RE.search(raw)
        cypher = m.group(1).strip() if m else raw.strip()
        # Safety net: prevent full-graph scans if LLM forgets LIMIT
        if 'LIMIT' not in cypher.upper():
            fallback = 20 if 'COLLECT' in cypher.upper() else 100
            cypher = cypher.rstrip(';').rstrip() + f'\nLIMIT {fallback}'
        if not error_context:
            self._cypher_cache[question] = cypher
        return cypher

    def _run_pipeline(self, question: str, max_retries: int = 2) -> dict:
        """
        Step 1 — generate Cypher (with up to max_retries correction attempts on syntax errors).
        Step 2 — execute against Neo4j.
        Returns {cypher, results}. Answer generation is deferred to ask_multi()
        so results from all strategies are combined before a single QA call.
        """
        cypher = self._generate_cypher(question)
        last_error = ""

        for attempt in range(max_retries):
            try:
                results = self.graph.query(cypher)
                break  # success
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    print(f"             → Cypher error (attempt {attempt + 1}), retrying: {last_error[:120]}")
                    cypher = self._generate_cypher(question, error_context=last_error)
                else:
                    print(f"             → Cypher failed after {max_retries} attempts: {last_error[:120]}")
                    return {"cypher": cypher, "results": [], "error": last_error}

        return {"cypher": cypher, "results": results}

    # ------------------------------------------------------------------
    # Multi-strategy mode: one pipeline call per strategy, unified answer.
    # ------------------------------------------------------------------

    def _select_strategies(self, question: str) -> list[str]:
        """
        Ask the LLM to classify whether the question requires metrics, risks, or both.
        Note: 'risks' always includes metrics because financial metrics are the
        quantitative evidence used to identify and substantiate credit risks.
        Falls back to running all strategies if the call fails.
        """
        prompt = (
            "You are routing a credit-risk question to the correct data source.\n\n"
            "Classify the question below as ONE of:\n"
            "  metrics — needs ONLY financial figures, ratios, or quantitative performance data, with no risk analysis\n"
            "  both    — needs risk factors OR risk analysis (metrics are always fetched alongside risks\n"
            "            because financial figures are the evidence behind every credit risk)\n\n"
            "Output ONLY one word: metrics | both\n\n"
            f"Question: {question}"
        )
        try:
            answer = self._call_llm_raw(prompt).strip().lower().split()[0]
            if answer == "metrics":
                return ["metrics"]
        except Exception:
            pass
        return list(QUERY_STRATEGIES.keys())

    def ask_multi(self, question: str, verbose: bool = False) -> dict:
        all_results = []
        queries_run = []

        active_strategies = self._select_strategies(question)
        print(f"      [strategy routing] selected: {active_strategies}")

        def _run_strategy(strategy: str) -> tuple[str, dict]:
            sub_question = QUERY_STRATEGIES[strategy].format(question=question)
            return strategy, self._run_pipeline(sub_question)

        skipped = [s for s in QUERY_STRATEGIES if s not in active_strategies]
        for s in skipped:
            print(f"      [{s}] skipped")

        with ThreadPoolExecutor(max_workers=len(active_strategies)) as pool:
            futures = {pool.submit(_run_strategy, s): s for s in active_strategies}
            strategy_results: dict[str, dict] = {}
            for fut in as_completed(futures):
                strategy, data = fut.result()
                strategy_results[strategy] = data

        for strategy in active_strategies:
            data = strategy_results[strategy]
            rows = data["results"]
            print(f"      [{strategy}] Cypher: {data['cypher'][:120].replace(chr(10), ' ')}")
            print(f"             → {len(rows)} records")
            for row in rows:
                row["_strategy"] = strategy
            all_results.extend(rows)
            if data["cypher"]:
                queries_run.append({"strategy": strategy, "cypher": data["cypher"]})

        

        seen = set()
        unique = []
        for row in all_results:
            key = tuple(sorted(
                (k, str(v)) for k, v in row.items() if k != "_strategy"
            ))
            if key not in seen:
                seen.add(key)
                unique.append(row)

        unique = self._dedup_list_fields(unique)

        # Generate unified answer from combined results using the QA prompt
        clean = self._clean_metadata(unique[:100])
        context = json.dumps(self._truncate(clean), indent=2, default=str)
        answer_msg = QA_PROMPT.format(context=context, question=question, target_company=self._target_company)
        unified_answer = self._call_llm_raw(answer_msg)

        return {
            "answer": unified_answer,
            "queries": queries_run,
            "results": unique,  # Return original uncleaned results for saving
        }

    # ------------------------------------------------------------------
    # Main entry point — choose single or multi strategy
    # ------------------------------------------------------------------

    def _normalize_question(self, question: str) -> str:
        """Replace the known target company name with 'the target company'."""
        if self._target_company and self._target_company.lower() != "unknown":
            question = re.sub(re.escape(self._target_company), "the target company", question, flags=re.IGNORECASE)
        return question

    def ask(self, question: str, verbose: bool = False, reasoning: bool = False) -> str:
        """
        Runs one pipeline call per strategy, then generates a unified answer.
        reasoning=True  → also run REASONING_PROMPT for a cited chain-of-thought trace.
        """
        question = self._normalize_question(question)
        print(f"\n{'='*70}")
        print(f"Question : {question}")
        print(f"Mode     : multi{'  +reasoning' if reasoning else ''}")
        print('='*70)

        print("\n[1/2] Generating Cypher and querying graph per strategy …")
        data = self.ask_multi(question, verbose=verbose)
        print(f"      → {len(data['results'])} unique records across {len(data['queries'])} queries")
        print("\n[2/2] Unified answer generated.")
        answer = data["answer"]
        queries_run = data["queries"]
        results = data["results"]

        answer = _THINK_RE.sub('', answer).strip()

        if results:
            self._save_extraction_result(question, queries_run, results)
        
        self._save_answer(answer)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)

        if reasoning and results:
            print(f"\n{'='*70}")
            print("Reasoning Trace:")
            print('='*70)
            trace = self.generate_reasoning_trace(question, queries_run, results)
            trace = _THINK_RE.sub('', trace).strip()
            print(trace)
            self._save_reasoning(trace)
            return answer + "\n\n---\n\n" + trace

        return answer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_list_fields(
        rows: list,
        key_fields: tuple = ("risk_id", "gaap_concept", "name", "xbrl_tag"),
    ) -> list:
        """
        Deduplicate items inside list-valued fields across rows.

        - target_* fields: global dedup — a risk that appears in every peer row
          is kept only in the first row that contains it.
        - peer_* fields: per-peer dedup — scoped by the row's 'peer' value so
          two different peers can legitimately share the same risk/metric name.
        """
        seen: dict[str, set] = {}  # namespace -> set of seen item keys
        result = []
        for row in rows:
            new_row = dict(row)
            peer_ns = str(row.get("peer", ""))
            for field, val in row.items():
                if not isinstance(val, list):
                    continue
                ns = f"{field}::{peer_ns}" if field.startswith("peer") else field
                seen.setdefault(ns, set())
                deduped = []
                for item in val:
                    if not isinstance(item, dict):
                        deduped.append(item)
                        continue
                    item_key = next(
                        (str(item[k]) for k in key_fields if k in item and item[k] is not None), None
                    )
                    if item_key is None or item_key not in seen[ns]:
                        deduped.append(item)
                        if item_key:
                            seen[ns].add(item_key)
                new_row[field] = deduped
            result.append(new_row)
        return result

    @staticmethod
    def _clean_metadata(results: list) -> list:
        """
        Strip source_text and cik before sending to LLM.
        xbrl_tag is kept so it is available in the saved JSON for eval.
        Deep-copies every row so the originals (saved to JSON) are never mutated.
        """
        import copy

        def _strip(obj):
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if k not in ('source_text', 'cik')}
            if isinstance(obj, list):
                return [_strip(i) for i in obj]
            return obj

        return [_strip(dict(row)) for row in results]

    def _call_llm_raw(self, user_msg: str) -> str:
        """Direct LLM call for cases where we bypass the chain."""
        response = self.llm.invoke(user_msg)
        text = response.content if hasattr(response, "content") else str(response)
        return _THINK_RE.sub('', text).strip()

    @staticmethod
    def _truncate(obj, max_str: int = 400):
        if isinstance(obj, str):
            return obj[:max_str] + "…" if len(obj) > max_str else obj
        if isinstance(obj, dict):
            return {k: CreditRiskQA._truncate(v, max_str) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CreditRiskQA._truncate(i, max_str) for i in obj]
        return obj

    def generate_reasoning_trace(self, question: str, queries_run: list, results: list) -> str:
        """
        Run REASONING_PROMPT over already-retrieved graph data.
        Produces a structured chain-of-thought with cited sources.

        Qwen3 (and similar models) emit reasoning inside <think> tags and a
        short conclusion outside.  We want the DETAILED content, so we prefer
        the outside text when it is substantive; otherwise we fall back to the
        <think> block itself.
        """
        queries_text = "\n".join(
            f"[{q.get('strategy', '?')}]\n{q.get('cypher', '')}"
            for q in queries_run
        ) if queries_run else "N/A"

        # Strip source_text — only page number and section title are needed for citations.
        clean = self._clean_metadata(results[:150])
        context = json.dumps(clean, indent=2, default=str)

        prompt = REASONING_PROMPT.format(queries=queries_text, context=context, question=question, target_company=self._target_company)
        response = self.llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract content outside <think> blocks
        outside = _THINK_RE.sub('', raw).strip()

        # If the model put all the structured reasoning inside <think> (Qwen3
        # thinking mode), fall back to that content so the trace is not empty.
        if len(outside) < 100:
            think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
            if think_match:
                return think_match.group(1).strip()

        return outside

    def _save_extraction_result(self, question: str, queries: list, results: list):
        """Save extraction results with question and queries to retrival_results/extraction_result.json"""
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "extraction_result.json")
        clean = self._clean_metadata(results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "question": question,
                "queries": queries,
                "record_count": len(clean),
                "results": clean
            }, f, indent=2, default=str)
        print(f"      → Saved: retrival_results/extraction_result.json")

    def _save_answer(self, answer: str):
        """Save answer text to retrival_results/answer.txt"""
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "answer.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(answer)
        print(f"      → Saved: retrival_results/answer.txt")

    def _save_reasoning(self, reasoning: str):
        """Save reasoning trace to retrival_results/reasoning.txt"""
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "reasoning.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(reasoning)
        print(f"      → Saved: retrival_results/reasoning.txt")

    def close(self):
        self.graph.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Credit Risk Q&A — natural language → Cypher (via LangChain) → answer"
    )
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--reasoning", "-r", action="store_true",
        help="After answering, run the reasoning-trace prompt showing step-by-step "
             "analysis with source citations (filing dates, document URLs, source text).",
    )
    args = parser.parse_args()

    qa = CreditRiskQA()
    try:
        if args.question:
            qa.ask(args.question, verbose=args.verbose, reasoning=args.reasoning)
        else:
            print("\nCredit Risk Q&A — interactive mode  (type 'exit' to quit)\n")
            while True:
                try:
                    question = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    break
                qa.ask(question, verbose=args.verbose, reasoning=args.reasoning)
    finally:
        qa.close()


if __name__ == "__main__":
    main()
