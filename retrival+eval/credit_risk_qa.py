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
  - Metric          {name, value, unit, year, metric_type, xbrl_tag, label}
  - Risk            {risk_id, name, description, why, source_text, filing_date}
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
"""

# ---------------------------------------------------------------------------
# Prompts — tune these to change how the LLM generates and answers queries
# ---------------------------------------------------------------------------

# Exact MetricCategory.name values in the graph — injected into every Cypher prompt
# so the LLM uses correct spellings instead of guessing.
KNOWN_CATEGORIES = {"Leverage", "Coverage", "Liquidity", "Profitability", "Debt Structure"}

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "categories", "question"],
    template="""You are a Neo4j Cypher expert for a KYC / credit-risk knowledge graph.
Generate a single, valid Cypher query that retrieves all data needed to answer the question.
Include peer comparison data where relevant to the question.

Schema:
{schema}

Known MetricCategory names (use EXACT spelling when filtering by category):
{categories}

Rules:
1. Output ONLY the Cypher query. No explanation, no markdown fences, no comments.
2. Always match the TargetCompany node when the question is about the analyzed company.
3. Use OPTIONAL MATCH for peer data — queries must work even when no peers exist.
4. No write operations (CREATE, MERGE, SET, DELETE, DETACH DELETE).
5. Use DISTINCT to avoid duplicates.
6. Use toFloat() for numeric comparisons on Metric.value.
7. Return readable aliases: tc.name AS company, m.value AS value, m.year AS year, etc.
8. Use properties(node) to return full node data rather than hardcoding property names.
9. For risk topics, search ALL text fields: name, description, why, source_text.
10. For metric topics, traverse: Company -> HAS_METRIC_CATEGORY -> MetricCategory -> HAS_METRIC -> Metric.
11. When filtering by MetricCategory, use ONLY the known category names listed above.
    Match the question intent to the closest category — e.g. "leverage" → "Leverage",
    "debt" → "Leverage" or "Debt Structure", "liquidity" → "Liquidity".

Rules (CRITICAL — these prevent common failure modes):
11. NEVER chain COMPETES_WITH and FACES_RISK in a single path. WRONG: (p)-[:COMPETES_WITH]->(tc)-[:FACES_RISK]->(pr). This makes pr a target risk, not a peer risk.
12. NEVER create cartesian products between target rows and peer rows. Always use COLLECT to aggregate peer data.
13. Always add LIMIT 300 at the end of every query to prevent full-graph scans.
14. For peers: use TWO separate OPTIONAL MATCHes — one to find peers, one to find their data.

Correct patterns:

Risks with peer comparison (COLLECT avoids cartesian explosion, includes all peers even with no matches):
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE toLower(r.description) CONTAINS 'keyword' OR toLower(r.name) CONTAINS 'keyword'
           OR toLower(r.why) CONTAINS 'keyword' OR toLower(r.source_text) CONTAINS 'keyword'
    WITH tc, collect(properties(r)) AS target_risks
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WHERE toLower(pr.description) CONTAINS 'keyword' OR toLower(pr.name) CONTAINS 'keyword'
           OR toLower(pr.why) CONTAINS 'keyword' OR toLower(pr.source_text) CONTAINS 'keyword'
    WITH tc, target_risks, peer, collect(properties(pr)) AS peer_risks
    RETURN tc.name AS target, target_risks, peer.name AS peer, peer_risks
    LIMIT 300

Metrics by category — filter to ONLY relevant categories, COLLECT peer metrics per row:
    MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
    WHERE mc.name IN ['Leverage', 'Debt Structure']
    OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (peer)-[:HAS_METRIC_CATEGORY]->(pmc:MetricCategory {{name: mc.name}})-[:HAS_METRIC]->(pm:Metric)
    WHERE pm.year = m.year
    WITH tc, peer, mc, m, collect({{name: pm.name, value: pm.value, year: pm.year}}) AS peer_metrics
    RETURN tc.name AS target, peer.name AS peer, mc.name AS category,
           m.name AS target_metric, m.value AS target_value, m.year AS year,
           peer_metrics
    LIMIT 300

Risks unique to target:
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE NOT EXISTS {{ MATCH (p:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc) MATCH (p)-[:FACES_RISK]->(pr:Risk) WHERE pr.name = r.name }}
    RETURN DISTINCT tc.name AS company, r.name AS risk_name, r.why AS reason
    LIMIT 300

Risk counts:
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WITH tc, count(r) AS tc_count
    MATCH (p:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    OPTIONAL MATCH (p)-[:FACES_RISK]->(pr:Risk)
    RETURN tc.name AS company, tc_count, p.name AS peer, count(pr) AS peer_count
    LIMIT 300

Question: {question}

Cypher query:""",
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior credit-risk analyst providing KYC assessments.
Using the graph query results below, write a clear and concise answer.

Requirements:
- Directly answer the question.
- Highlight how the target company compares to its peers where data is available.
- Flag elevated risks or concerning metrics.
- Use plain language suitable for a credit committee.
- Stay under 300 words unless the data requires more detail.

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
  - Source reference:
      • For TARGET risks → quote up to 2 sentences from Risk.source_text AND include
        metadata.document_url if present. Cite as: [document_url] — "<quote>"
      • For PEER risks   → note the peer name, risk name, and metadata.document_url if present
      • For companies    → include Company.document_url or TargetCompany.document_url
      • For metrics      → note the MetricCategory, Metric.year, and metadata.source_url if present

## 3. Reasoning Steps
Walk through your analysis step by step:
  Step 1 — <what you observed in the raw data>
  Step 2 — <how you compared target vs peers>
  Step 3 — <what pattern or gap stands out>
  … (add as many steps as the data warrants)

## 4. Sources & References
List every document or data point cited, one per line.

For TARGET company risks use this format:
  [Company | Target] Filing: <filing_date> | URL: <metadata.document_url or N/A> | Source: "<excerpt from source_text, or N/A>"

For PEER company risks use this format:
  [Company | Peer] Risk: <risk name> | URL: <metadata.document_url or N/A>

For metrics use this format:
  [Company | Role] Category: <MetricCategory> | Metric: <name> | Value: <value> | Year: <year>

## 5. Final Answer
A concise credit-committee-ready answer (≤ 200 words) that directly responds to
the question, highlights peer comparisons, and flags any elevated risks or
concerning metrics.
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
        "Filter MetricCategory to ONLY the categories MOST RELEVANT to: {question}. "
        "Map the question to the closest category names from: "
        "Leverage, Coverage, Liquidity, Profitability, Debt Structure. "
        "Do NOT fetch all categories — choose only those that match the question intent. "
        "CRITICAL: Use COLLECT to avoid row explosion — return one row per (target_metric, peer) pair "
        "by collecting peer metrics into a list. Include ALL peers even if they have no data. "
        "Add LIMIT 300. "
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
    # Single-query mode
    # ------------------------------------------------------------------

    def ask_single(self, question: str, verbose: bool = False) -> dict:
        data = self._run_pipeline(question)


        return data

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

        

        # Deduplicate — sort all values as strings so minor type differences
        # (float vs int) and alias differences don't create false uniques.
        seen = set()
        unique = []
        for row in all_results:
            key = tuple(sorted(
                (k, str(v)) for k, v in row.items() if k != "_strategy"
            ))
            if key not in seen:
                seen.add(key)
                unique.append(row)

        # Generate unified answer from combined results using the QA prompt
        context = json.dumps(self._truncate(unique[:100]), indent=2, default=str)
        answer_msg = QA_PROMPT.format(context=context, question=question)
        unified_answer = self._call_llm_raw(answer_msg)

        return {
            "answer": unified_answer,
            "queries": queries_run,
            "results": unique,
        }

    # ------------------------------------------------------------------
    # Main entry point — choose single or multi strategy
    # ------------------------------------------------------------------

    def ask(self, question: str, verbose: bool = False,
            mode: str = "multi", reasoning: bool = False) -> str:
        """
        mode="single"   → one pipeline call end-to-end.
        mode="multi"    → one pipeline call per strategy, unified answer (default).
        reasoning=True  → also run REASONING_PROMPT for a cited chain-of-thought trace.
        """
        print(f"\n{'='*70}")
        print(f"Question : {question}")
        print(f"Mode     : {mode}{'  +reasoning' if reasoning else ''}")
        print('='*70)

        if mode == "single":
            print("\n[1/2] Generating Cypher and querying graph (single) …")
            data = self.ask_single(question, verbose=verbose)
            print(f"      → Cypher: {data['cypher'][:120]}")
            print(f"      → {len(data['results'])} records")
            print("\n[2/2] Answer generated.")
            answer = data["answer"]
            queries_run = [{"cypher": data["cypher"]}]
            results = data["results"]

        else:  # multi
            print("\n[1/2] Generating Cypher and querying graph per strategy …")
            data = self.ask_multi(question, verbose=verbose)
            print(f"      → {len(data['results'])} unique records across {len(data['queries'])} queries")
            print("\n[2/2] Unified answer generated.")
            answer = data["answer"]
            queries_run = data["queries"]
            results = data["results"]

        answer = _THINK_RE.sub('', answer).strip()

        if results:
            self._save(question, queries_run, results)

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
            return answer + "\n\n---\n\n" + trace

        return answer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_meta(node) -> dict | None:
        """Drop source_text from metadata (too large for LLM context); keep document_url, cik, source_url."""
        if not isinstance(node, dict):
            return node
        node = dict(node)
        raw_meta = node.get("metadata")
        keep_keys = ("document_url", "cik", "source_url")
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta)
                node["metadata"] = {k: meta[k] for k in keep_keys if k in meta}
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(raw_meta, dict):
            node["metadata"] = {k: raw_meta[k] for k in keep_keys if k in raw_meta}
        return node

    @staticmethod
    def _clean_metadata(results: list) -> list:
        """
        Strip source_text from every risk/peer_risk metadata in results,
        keeping only source_page and section_title.
        Returns a new list (originals are not mutated).
        """
        cleaned = []
        for row in results:
            row = dict(row)
            if "risk" in row:
                row["risk"] = CreditRiskQA._strip_meta(row["risk"])
            if "peer_risk" in row:
                row["peer_risk"] = CreditRiskQA._strip_meta(row["peer_risk"])
            cleaned.append(row)
        return cleaned

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

    def _save(self, question: str, queries: list, results: list):
        from datetime import datetime
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(out_dir, exist_ok=True)
        slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:60].strip('_')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"query_results_{slug}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"question": question, "queries": queries,
                       "record_count": len(results),
                       "results": results},  # full metadata preserved (incl. source_text, document_url, source_url)
                      f, indent=2, default=str)
        print(f"      → Saved: retrival_results/{os.path.basename(path)}")

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
        "--mode", choices=["single", "multi"], default="multi",
        help="single: one chain call; multi: one call per strategy (default: multi)",
    )
    parser.add_argument(
        "--reasoning", "-r", action="store_true",
        help="After answering, run the reasoning-trace prompt showing step-by-step "
             "analysis with source citations (filing dates, document URLs, source text).",
    )
    args = parser.parse_args()

    qa = CreditRiskQA()
    try:
        if args.question:
            qa.ask(args.question, verbose=args.verbose, mode=args.mode, reasoning=args.reasoning)
        else:
            print(f"\nCredit Risk Q&A — interactive mode  [mode={args.mode}]  (type 'exit' to quit)\n")
            while True:
                try:
                    question = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    break
                qa.ask(question, verbose=args.verbose, mode=args.mode, reasoning=args.reasoning)
    finally:
        qa.close()


if __name__ == "__main__":
    main()
