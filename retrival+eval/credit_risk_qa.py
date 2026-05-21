import os
import re
import json
import time
import boto3
from neo4j import GraphDatabase
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

# ---------------------------------------------------------------------------
# Graph schema (injected into every Cypher-generation prompt)
# ---------------------------------------------------------------------------

GRAPH_SCHEMA = """
Node labels and key properties:
  - TargetCompany   {name, cik, ticker, is_target:true, filing_date, document_url}
  - Company         {name, cik, ticker, is_peer:true, filing_date, document_url}
  - MetricCategory  {name}
  - Metric          {citation_id, name, value, unit, year, metric_type, xbrl_tag, label, source_url, cik}
                    NOTE: name = "{metric_type} ({year})" — use xbrl_tag to identify the exact concept
                    CRITICAL: Always return citation_id for citations
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

# Exact MetricCategory.name values in the graph — injected into every Cypher prompt
# so the LLM uses correct spellings instead of guessing.
KNOWN_CATEGORIES = {"Leverage", "Coverage", "Liquidity", "Profitability", "Debt Structure"}

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "categories", "question"],
    template="""You are a Neo4j Cypher expert for a credit-risk knowledge graph.
Generate a single valid Cypher query that retrieves all data needed to answer the question.

Schema:
{schema}

Known MetricCategory names (use EXACT spelling):
{categories}

Rules:
1. Output ONLY the Cypher query — no markdown, no comments, no explanation.
2. No write operations (CREATE, MERGE, SET, DELETE).
3. Always LIMIT 300.
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

Reference patterns:

Risks with peer comparison:
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE toLower(r.description) CONTAINS 'keyword' OR toLower(r.name) CONTAINS 'keyword'
           OR toLower(r.why) CONTAINS 'keyword' OR toLower(r.source_text) CONTAINS 'keyword'
    WITH tc, collect({{citation_id: r.citation_id, risk_id: r.risk_id, name: r.name, description: r.description, why: r.why, source_text: r.source_text, document_url: r.document_url, section_title: r.section_title, source_page: r.source_page}}) AS target_risks
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WHERE toLower(pr.description) CONTAINS 'keyword' OR toLower(pr.name) CONTAINS 'keyword'
           OR toLower(pr.why) CONTAINS 'keyword' OR toLower(pr.source_text) CONTAINS 'keyword'
    WITH tc, target_risks, peer, collect({{citation_id: pr.citation_id, risk_id: pr.risk_id, name: pr.name, description: pr.description, why: pr.why, source_text: pr.source_text, document_url: pr.document_url, section_title: pr.section_title, source_page: pr.source_page}}) AS peer_risks
    RETURN tc.name AS target, target_risks, peer.name AS peer, peer_risks
    LIMIT 300

Metrics with peer comparison (fetch independently by category — let the LLM align by label):
    MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
    WHERE mc.name = 'Profitability'
    WITH tc, mc, collect({{citation_id: m.citation_id, name: m.name, label: m.label, value: m.value, unit: m.unit, year: m.year, metric_type: m.metric_type, xbrl_tag: m.xbrl_tag}}) AS target_metrics
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:HAS_METRIC_CATEGORY]->(pmc:MetricCategory {{name: mc.name}})-[:HAS_METRIC]->(pm:Metric)
    WITH tc, mc, target_metrics, peer, collect({{citation_id: pm.citation_id, name: pm.name, label: pm.label, value: pm.value, unit: pm.unit, year: pm.year, metric_type: pm.metric_type, xbrl_tag: pm.xbrl_tag, source_url: pm.source_url}}) AS peer_metrics
    RETURN tc.name AS target, mc.name AS category, target_metrics, peer.name AS peer, peer_metrics
    LIMIT 300

Question: {question}

Cypher query:""",
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior credit-risk analyst providing KYC assessments.
Answer the question using only the graph results below.

- Lead with a **Sources** section listing one document URL per company — only include URLs that appear in the graph results. Omit any company with no URL.
- Use ONLY values present in the graph results. Never infer or derive a metric that is not explicitly returned.
- If a metric is missing for a peer, state "not available" rather than estimating it.
- Directly answer the question; compare target vs peers where data exists.
- Flag any elevated risks or concerning metrics.
- Plain language suitable for a credit committee; ≤ 300 words unless data requires more.
- After every claim drawn from a specific data point, append an inline citation using the citation_id from the graph results:
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

# Shown when --reasoning is enabled.  The LLM must think step-by-step,

# name what it found, and cite every source document it draws from.
REASONING_PROMPT = PromptTemplate(
    input_variables=["context", "question", "queries"],
    template="""You are a senior credit-risk analyst with access to structured 10-K filing data.
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

# ---------------------------------------------------------------------------
# Query strategies — each focuses the chain on a different aspect.
# The user can add / remove / rename strategies, or adjust the sub-question
# templates below, to change what gets queried.
# ---------------------------------------------------------------------------

# Each strategy instructs the LLM to fetch a DIFFERENT type of data so queries
# don't overlap. "risks" → FACES_RISK only. "metrics" → HAS_METRIC only.
# Add or remove strategies here to control what gets queried.
QUERY_STRATEGIES = {
    "risks": (
        "Using ONLY the FACES_RISK relationship, fetch risk nodes for the TargetCompany "
        "AND its peers (Company {{is_peer:true}}). Filter risks relevant to: {question}. "
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
        "Collect target metrics into a list (including xbrl_tag for validation), "
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

        print(f"✓ Neo4j connected  : {neo4j_uri}")
        print(f"✓ LLM model        : {os.getenv('BEDROCK_MODEL', 'qwen.qwen3-next-80b-a3b')}")

    # ------------------------------------------------------------------
    # Core pipeline: NL → Cypher (LLM) → execute → answer (LLM)
    # Replaces GraphCypherQAChain with a direct two-step call.
    # ------------------------------------------------------------------

    _CYPHER_FENCE_RE = re.compile(r'```(?:cypher)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)

    def _generate_cypher(self, question: str, error_context: str = "") -> str:
        """Call the LLM to generate (or fix) a Cypher query."""
        base_prompt = CYPHER_GENERATION_PROMPT.format(
            schema=GRAPH_SCHEMA,
            categories=", ".join(sorted(KNOWN_CATEGORIES)),
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
            cypher = cypher.rstrip(';').rstrip() + '\nLIMIT 300'
        return cypher

    def _run_pipeline(self, question: str, max_retries: int = 2) -> dict:
        """
        Step 1 — generate Cypher (with up to max_retries correction attempts on syntax errors).
        Step 2 — execute against Neo4j.
        Step 3 — generate answer via QA_PROMPT.
        Returns {answer, cypher, results}.
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
                    return {"answer": "", "cypher": cypher, "results": [], "error": last_error}

        # QA answer — strip source_text so it doesn't bloat the prompt
        clean = self._clean_metadata(results[:50])
        context = json.dumps(self._truncate(clean), indent=2, default=str)
        answer = self._call_llm_raw(QA_PROMPT.format(context=context, question=question))

        return {"answer": answer, "cypher": cypher, "results": results}

    # ------------------------------------------------------------------
    # Multi-strategy mode: one pipeline call per strategy, unified answer.
    # ------------------------------------------------------------------

    def ask_multi(self, question: str, verbose: bool = False) -> dict:
        all_results = []
        queries_run = []

        for strategy, template in QUERY_STRATEGIES.items():
            sub_question = template.format(question=question)
            print(f"      [{strategy}] {sub_question[:80]}")
            data = self._run_pipeline(sub_question)
            rows = data["results"]
            print(f"             Cypher: {data['cypher'][:120].replace(chr(10), ' ')}")
            print(f"             → {len(rows)} records")
            for row in rows:
                row["_strategy"] = strategy
            all_results.extend(rows)
            if data["cypher"]:
                queries_run.append({"strategy": strategy, "cypher": data["cypher"]})

        

        # Row-level dedup — catches identical rows across strategies.
        seen = set()
        unique = []
        for row in all_results:
            key = tuple(sorted(
                (k, str(v)) for k, v in row.items() if k != "_strategy"
            ))
            if key not in seen:
                seen.add(key)
                unique.append(row)

        # List-field dedup — target_* lists are deduplicated globally (same
        # risk appears in every peer row); peer_* lists are deduplicated per
        # peer so two peers can share the same risk name without one being dropped.
        unique = self._dedup_list_fields(unique)

        # Generate unified answer from combined results using the QA prompt
        clean = self._clean_metadata(unique[:100])
        context = json.dumps(self._truncate(clean), indent=2, default=str)
        answer_msg = QA_PROMPT.format(context=context, question=question)
        unified_answer = self._call_llm_raw(answer_msg)

        return {
            "answer": unified_answer,
            "queries": queries_run,
            "results": unique,  # Return original uncleaned results for saving
        }

    # ------------------------------------------------------------------
    # Main entry point — choose single or multi strategy
    # ------------------------------------------------------------------

    def ask(self, question: str, verbose: bool = False, reasoning: bool = False) -> str:
        """
        Runs one pipeline call per strategy, then generates a unified answer.
        reasoning=True  → also run REASONING_PROMPT for a cited chain-of-thought trace.
        """
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
        key_fields: tuple = ("risk_id", "name", "xbrl_tag"),
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
                        (str(item[k]) for k in key_fields if k in item), None
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
        Strip source_text, cik, and xbrl_tag before sending to LLM.
        Deep-copies every row so the originals (saved to JSON) are never mutated.
        """
        import copy

        def _strip(obj):
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if k not in ('source_text', 'cik', 'xbrl_tag')}
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
        """
        queries_text = "\n".join(
            f"[{q.get('strategy', '?')}]\n{q.get('cypher', '')}"
            for q in queries_run
        ) if queries_run else "N/A"

        # Strip source_text — only page number and section title are needed for citations.
        clean = self._clean_metadata(results[:150])
        context = json.dumps(clean, indent=2, default=str)

        return self._call_llm_raw(
            REASONING_PROMPT.format(queries=queries_text, context=context, question=question)
        )

    def _save_extraction_result(self, question: str, queries: list, results: list):
        """Save extraction results with question and queries to retrival_results/extraction_result.json"""
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "extraction_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "question": question,
                "queries": queries,
                "record_count": len(results),
                "results": results
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
