import os
import re
import json
import time
import boto3
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_CYPHER_BLOCK_RE = re.compile(r'```(?:cypher)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Graph schema (category names are injected dynamically at startup)
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
  - Person          {name}

Relationships:
  (TargetCompany|Company)-[:HAS_METRIC_CATEGORY]->(:MetricCategory)
  (:MetricCategory)-[:HAS_METRIC]->(:Metric)
  (TargetCompany|Company)-[:FACES_RISK]->(:Risk)
  (:Company {{is_peer:true}})-[:COMPETES_WITH]->(:TargetCompany)
  (TargetCompany|Company)-[:OPERATES_IN]->(:Industry)
  (:Industry)-[:HAS_SIC_CODE]->(:SICCode)
  (:Person)-[:CEO_OF|CFO_OF|BOARD_MEMBER_OF|WORKS_AT]->(TargetCompany|Company)

Notes:
  - The company being analyzed is always TargetCompany (use label TargetCompany in MATCH).
  - Peers are Company nodes with is_peer:true that have a COMPETES_WITH edge to TargetCompany.
  - Metrics are organized by category: Company -[:HAS_METRIC_CATEGORY]-> MetricCategory -[:HAS_METRIC]-> Metric
  - Metric.value is stored as a string; cast with toFloat() when comparing.
  - Metric.metric_type holds the covenant / financial category (e.g. "Debt", "Revenue").
  - MetricCategory.name groups related metrics (e.g. "Financial Ratios", "Revenue Metrics").
  - FACES_RISK relationship connects a company to a specific Risk node.
  - Risk.risk_id is company-specific (format: CIK_risk_N). To compare risks across companies,
    match by risk.name, NOT risk_id.
  - Risk.why contains the specific factual evidence or mechanism explaining why this is a risk.
  - Risk.source_text contains the original risk disclosure text from the 10-K.
  - When filtering risks by topic, ALWAYS search ALL four text fields: name, description, why, source_text.
  - When returning node properties, use `properties(r)` or `properties(m)` to capture ALL properties
    that actually exist on the node — never hardcode a fixed list of property names.
  - When the question is about a financial topic (interest rates, debt, revenue, liquidity, etc.),
    ALWAYS include BOTH the related risks AND the related metrics in the same query (two MATCH clauses
    or UNION, returning results labelled by type).
  - Always return enough fields for a meaningful comparison (company name, metric values, risk details).
  - Always use DISTINCT when returning risk rows to avoid duplicates from OPTIONAL MATCH joins.
"""

CYPHER_SYSTEM_PROMPT_BASE = """You are a Neo4j Cypher expert for a KYC / credit-risk knowledge graph.
Given a natural-language question about credit risk, generate a single valid Cypher query
that retrieves the data needed to answer it — including a peer comparison where relevant.

Graph schema:
{schema}

{name_hints}

Rules:
1. Output ONLY the Cypher query inside a ```cypher ... ``` code block. No explanation.
2. Always include the TargetCompany and its peers in comparisons.
3. Use OPTIONAL MATCH for peer data so the query works even if no peers exist.
4. No limit.
5. Prefer readable aliases (e.g. tc.name AS company, m.value AS value, m.year AS year).
6. Never use DETACH DELETE or any write operation.
7. CRITICAL: When using WITH, include ALL variables you need in subsequent clauses.
8. CRITICAL: WHERE clauses must come immediately after MATCH/OPTIONAL MATCH, not after WITH.
9. Use COLLECT() and list operations carefully — ensure variables are in scope.
10. CRITICAL: Use ONLY metric names, risk names, and category names from the lists provided in
    the name hints above. Do NOT invent or guess names. If matching multiple, use CONTAINS or IN.

Common patterns:
- Compare target to peers: MATCH (tc:TargetCompany)-[rel]->(node) OPTIONAL MATCH (peer:Company {{is_peer:true}})-[peer_rel]->(peer_node) RETURN ...
- Get metrics by category: MATCH (c:Company)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric) WHERE mc.name = 'Revenue' RETURN c.name, m.name, m.value, m.year
- Compare metrics across companies: MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc)-[:HAS_METRIC]->(m) OPTIONAL MATCH (peer:Company {{is_peer:true}})-[:HAS_METRIC_CATEGORY]->(mc)-[:HAS_METRIC]->(pm:Metric {{name: m.name, year: m.year}}) RETURN tc.name, peer.name, m.name, m.value, pm.value
- Filter risks by keyword in why: MATCH (c)-[:FACES_RISK]->(r:Risk) WHERE toLower(r.why) CONTAINS 'financial' RETURN ...
- Count risks per company: MATCH (c)-[:FACES_RISK]->(r:Risk) RETURN c.name, count(r) AS risk_count
- Return ALL node properties (never hardcode fields): RETURN c.name AS company, properties(r) AS risk, properties(m) AS metric
- Combined risks + metrics for a financial topic — use resolved names from the hints above:
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE r.name IN [<resolved risk names>]
       OR toLower(r.description) CONTAINS '<keyword>' OR toLower(r.why) CONTAINS '<keyword>'
    WITH tc, COLLECT(DISTINCT properties(r)) AS risks
    OPTIONAL MATCH (tc)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)-[:HAS_METRIC]->(m:Metric)
    WHERE mc.name IN [<resolved category names>] OR m.name IN [<resolved metric names>]
    RETURN tc.name AS company, risks, COLLECT(DISTINCT properties(m)) AS metrics
- Risks target faces that peers DO NOT (by name):
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WHERE NOT EXISTS {{
      MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
      MATCH (peer)-[:FACES_RISK]->(pr:Risk)
      WHERE pr.name = r.name
    }}
    RETURN DISTINCT tc.name AS company, r.risk_id, r.name, r.description, r.why
- Risks peers share but target does NOT:
    MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc:TargetCompany)
    MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WHERE NOT EXISTS {{
      MATCH (tc)-[:FACES_RISK]->(tr:Risk) WHERE tr.name = pr.name
    }}
    RETURN DISTINCT peer.name AS peer, pr.name, pr.why
- Risk comparison across all companies (use COLLECT to avoid row explosion):
    MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
    WITH tc, COLLECT(DISTINCT r.name) AS target_risks
    MATCH (peer:Company {{is_peer:true}})-[:COMPETES_WITH]->(tc)
    MATCH (peer)-[:FACES_RISK]->(pr:Risk)
    WITH tc, target_risks, peer, COLLECT(DISTINCT pr.name) AS peer_risks
    RETURN tc.name, target_risks, peer.name AS peer, peer_risks
"""

ANSWER_SYSTEM_PROMPT = """You are a senior credit-risk analyst providing KYC assessments.
Given the original question, the Cypher query used, and the raw graph results,
write a clear, concise answer that:
- Directly answers the question.
- Highlights how the target company compares to its peers.
- Flags any elevated risks or concerning metrics.
- Uses plain language suitable for a credit committee.
Keep the answer under 300 words unless the data demands more detail.
"""

FALLBACK_ANSWER_PROMPT = """You are a senior credit-risk analyst.
The graph query returned no results, but here are the most semantically relevant risk documents
retrieved from the knowledge base. Use these to answer the question as best you can, and note
that the answer is based on document retrieval rather than a direct graph query.
"""


class CreditRiskQA:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        region = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.model_id = os.getenv("BEDROCK_MODEL", "qwen.qwen3-next-80b-a3b")
        print(f"✓ Neo4j connected  : {uri}")
        print(f"✓ Bedrock model    : {self.model_id}")

        print("  Loading graph name index …")
        self._load_graph_index()
        print("  ✓ Name index ready")

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------
    def _load_graph_index(self):
        """Pull all real names from the graph and build embedding + BM25 indexes."""
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        with self.driver.session() as session:
            self.metric_names = [
                r["name"] for r in session.run(
                    "MATCH (m:Metric) RETURN DISTINCT m.name AS name"
                ) if r["name"]
            ]
            self.category_names = [
                r["name"] for r in session.run(
                    "MATCH (mc:MetricCategory) RETURN DISTINCT mc.name AS name"
                ) if r["name"]
            ]
            risk_rows = [
                dict(r) for r in session.run(
                    "MATCH (r:Risk) RETURN DISTINCT r.name AS name, "
                    "r.description AS description, r.why AS why"
                )
            ]

        self.risk_names = [r["name"] for r in risk_rows if r["name"]]
        self._risk_docs = [
            f"{r.get('name','')} {r.get('description','')} {r.get('why','')}"
            for r in risk_rows if r["name"]
        ]

        # Embeddings (encode once, cache as numpy arrays)
        if self.metric_names:
            self._metric_emb = self._embed_model.encode(
                self.metric_names, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            self._metric_emb = np.empty((0, 384))

        if self.category_names:
            self._category_emb = self._embed_model.encode(
                self.category_names, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            self._category_emb = np.empty((0, 384))

        if self.risk_names:
            self._risk_emb = self._embed_model.encode(
                self._risk_docs, normalize_embeddings=True, show_progress_bar=False
            )
            self._bm25 = BM25Okapi([doc.split() for doc in self._risk_docs])
        else:
            self._risk_emb = np.empty((0, 384))
            self._bm25 = None

        print(f"    ✓ Categories ({len(self.category_names)}): {self.category_names}")
        print(f"    ✓ Metrics    ({len(self.metric_names)}): "
              f"{self.metric_names[:5]}{' …' if len(self.metric_names) > 5 else ''}")
        print(f"    ✓ Risks      ({len(self.risk_names)} total, sample): "
              f"{self.risk_names[:3]}{' …' if len(self.risk_names) > 3 else ''}")

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------
    def _top_k_embedding(self, query_emb: np.ndarray, corpus_emb: np.ndarray,
                          names: list, k: int, min_score: float = 0.0) -> list[str]:
        if not names or corpus_emb.shape[0] == 0:
            return []
        scores = (query_emb @ corpus_emb.T)[0]
        idx = scores.argsort()[-k:][::-1]
        return [names[i] for i in idx if scores[i] >= min_score]

    def resolve_names(self, query: str, top_k: int = 5) -> dict:
        """Return top-k real graph names most relevant to the query."""
        q_emb = self._embed_model.encode([query], normalize_embeddings=True,
                                         show_progress_bar=False)

        top_metrics = self._top_k_embedding(q_emb, self._metric_emb,
                                             self.metric_names, top_k)
        # Categories: only top-2 with similarity >= 0.3 to avoid fetching every category
        top_categories = self._top_k_embedding(q_emb, self._category_emb,
                                                self.category_names, k=2, min_score=0.3)

        # Risks: hybrid BM25 (keyword) + embedding (semantic)
        if self._bm25 and self.risk_names:
            bm25_scores = np.array(self._bm25.get_scores(query.split()), dtype=float)
            emb_scores = (q_emb @ self._risk_emb.T)[0]
            bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
            combined = 0.4 * bm25_norm + 0.6 * emb_scores
            idx = combined.argsort()[-top_k:][::-1]
            top_risks = [self.risk_names[i] for i in idx]
        else:
            top_risks = self._top_k_embedding(q_emb, self._risk_emb,
                                               self.risk_names, top_k)

        return {"metrics": top_metrics, "risks": top_risks, "categories": top_categories}

    def _build_name_hints(self, resolved: dict) -> str:
        lines = ["Based on semantic similarity to your question, the most relevant real names in the graph are:"]
        if resolved["categories"]:
            lines.append(f"  Available MetricCategory names (use exact spelling): {resolved['categories']}")
        if resolved["metrics"]:
            lines.append(f"  Top matching Metric names: {resolved['metrics']}")
        if resolved["risks"]:
            lines.append(f"  Top matching Risk names: {resolved['risks']}")
        lines += [
            "",
            "Instructions for using these names in Cypher:",
            "  - For risk queries: filter with `r.name IN [<resolved risk names>]` first.",
            "    If those names don't match the question well, do a broad text search across ALL",
            "    risk fields: `toLower(r.name) CONTAINS '...' OR toLower(r.description) CONTAINS",
            "    '...' OR toLower(r.why) CONTAINS '...' OR toLower(r.source_text) CONTAINS '...'`",
            "  - For metric queries: use `m.name IN [<resolved metric names>]` or `mc.name IN [<resolved categories>]`.",
            "  - Do NOT invent names. Do NOT search only one field.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fallback: retrieve risk docs when Cypher returns nothing
    # ------------------------------------------------------------------
    def _retrieve_risk_docs(self, query: str, top_k: int = 5) -> list[str]:
        if not self._risk_docs:
            return []
        q_emb = self._embed_model.encode([query], normalize_embeddings=True,
                                          show_progress_bar=False)
        if self._bm25:
            bm25_scores = np.array(self._bm25.get_scores(query.split()), dtype=float)
            emb_scores = (q_emb @ self._risk_emb.T)[0]
            bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
            combined = 0.4 * bm25_norm + 0.6 * emb_scores
            idx = combined.argsort()[-top_k:][::-1]
        else:
            emb_scores = (q_emb @ self._risk_emb.T)[0]
            idx = emb_scores.argsort()[-top_k:][::-1]
        return [self._risk_docs[i] for i in idx]

    # ------------------------------------------------------------------
    def _call_llm(self, system: str, user_msg: str, max_tokens: int = 2048) -> str:
        prompt = f"{system}\n\nUser: {user_msg}"
        for attempt in range(4):
            try:
                response = self.bedrock.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
                )
                text = response["output"]["message"]["content"][0]["text"]
                return _THINK_RE.sub('', text).strip()
            except Exception as e:
                err = str(e)
                if ("ThrottlingException" in err or "429" in err) and attempt < 3:
                    wait = 30 * (attempt + 1)
                    print(f"  ⚠ Rate-limited, retrying in {wait}s …")
                    time.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    def translate_to_cypher(self, question: str, max_retries: int = 2) -> str:
        resolved = self.resolve_names(question)
        name_hints = self._build_name_hints(resolved)
        cypher_prompt = CYPHER_SYSTEM_PROMPT_BASE.format(
            schema=GRAPH_SCHEMA,
            name_hints=name_hints,
        )

        last_error = None
        for attempt in range(max_retries):
            if attempt == 0:
                user_msg = question
            else:
                user_msg = (
                    f"{question}\n\n"
                    f"Previous attempt failed with syntax error:\n{last_error}\n\n"
                    f"Please generate a corrected Cypher query that avoids this error."
                )

            raw = self._call_llm(cypher_prompt, user_msg)
            match = _CYPHER_BLOCK_RE.search(raw)
            if match:
                cypher = match.group(1).strip()
            else:
                lines = [l for l in raw.splitlines() if not l.strip().startswith('#') and l.strip()]
                cypher = "\n".join(lines)

            if attempt < max_retries - 1:
                try:
                    with self.driver.session() as session:
                        session.run(f"EXPLAIN {cypher}")
                    return cypher
                except Exception as e:
                    last_error = str(e)
                    print(f"  ⚠ Attempt {attempt + 1} generated invalid Cypher, retrying…")
                    continue

            return cypher

    # ------------------------------------------------------------------
    def run_cypher(self, query: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    def _deduplicate_results(self, results: list[dict], max_records: int = 100) -> list[dict]:
        seen = set()
        unique = []
        for row in results:
            key = json.dumps(row, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(row)
                if len(unique) >= max_records:
                    break
        return unique

    # ------------------------------------------------------------------
    def generate_answer(self, question: str, cypher: str, results: list[dict]) -> str:
        if not results:
            results_text = "No data returned from the graph."
        else:
            deduped = self._deduplicate_results(results)
            truncated = len(deduped) < len(results)
            results_text = json.dumps(
                self._truncate_for_llm(deduped), indent=2, default=str
            )
            if truncated:
                results_text += (
                    f"\n\n[Note: {len(results)} total records; "
                    f"showing {len(deduped)} unique rows after deduplication.]"
                )

        user_msg = (
            f"Question: {question}\n\n"
            f"Cypher query used:\n```cypher\n{cypher}\n```\n\n"
            f"Graph results ({len(results)} records):\n{results_text}"
        )
        return self._call_llm(ANSWER_SYSTEM_PROMPT, user_msg, max_tokens=1024)

    @staticmethod
    def _truncate_for_llm(results: list, max_str: int = 400) -> list:
        """Truncate long string values so the prompt stays within context limits."""
        def _trim(obj):
            if isinstance(obj, str):
                return obj[:max_str] + "…" if len(obj) > max_str else obj
            if isinstance(obj, dict):
                return {k: _trim(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_trim(i) for i in obj]
            return obj
        return _trim(results)

    # Fixed set of known MetricCategory names
    KNOWN_CATEGORIES = {"Leverage", "Coverage", "Liquidity", "Profitability", "Debt Structure"}

    def _fetch_peer_risks(self, resolved: dict) -> list[dict]:
        """Fetch risks for peer companies matching the resolved risk names / keyword."""
        risk_names = resolved.get("risks", [])
        if not risk_names:
            return []
        keyword = risk_names[0].split()[0].lower() if risk_names else ""
        with self.driver.session() as session:
            res = session.run(
                """
                MATCH (c:Company {is_peer: true})-[:FACES_RISK]->(r:Risk)
                WHERE r.name IN $risk_names
                   OR toLower(r.name) CONTAINS $keyword
                   OR toLower(r.description) CONTAINS $keyword
                   OR toLower(r.why) CONTAINS $keyword
                RETURN c.name AS company, 'peer' AS role, properties(r) AS risk
                ORDER BY c.name, r.name
                """,
                {"risk_names": risk_names, "keyword": keyword},
            )
            return [dict(r) for r in res]

    def _fetch_resolved_metrics(self, resolved: dict) -> list[dict]:
        """
        Fetch metrics by category match (primary) + semantic similarity to resolved
        metric names (secondary filter, threshold 0.25).  The literal keyword no longer
        needs to appear in m.name — category membership is the main gate.
        """
        metric_names = resolved.get("metrics", [])
        category_names = [c for c in resolved.get("categories", []) if c in self.KNOWN_CATEGORIES]

        if not metric_names and not category_names:
            return []

        rows = []
        with self.driver.session() as session:
            res = session.run(
                """
                MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)
                      -[:HAS_METRIC]->(m:Metric)
                WHERE mc.name IN $categories
                RETURN tc.name AS company, 'target' AS role,
                       mc.name AS category, properties(m) AS metric
                ORDER BY mc.name, m.name
                """,
                {"categories": category_names},
            )
            rows += [dict(r) for r in res]

            res = session.run(
                """
                MATCH (c:Company)-[:HAS_METRIC_CATEGORY]->(mc:MetricCategory)
                      -[:HAS_METRIC]->(m:Metric)
                WHERE mc.name IN $categories
                RETURN c.name AS company, 'peer' AS role,
                       mc.name AS category, properties(m) AS metric
                ORDER BY c.name, mc.name, m.name
                """,
                {"categories": category_names},
            )
            rows += [dict(r) for r in res]

        # Secondary filter: keep only metrics whose name is semantically close to
        # at least one resolved metric name (cosine similarity >= 0.25).
        # Skip filtering if no resolved metric names are available.
        if metric_names and rows:
            # Strip year suffixes for cleaner embedding comparison
            clean_refs = [n.rsplit("(", 1)[0].strip() for n in metric_names]
            ref_embs = self._embed_model.encode(clean_refs, normalize_embeddings=True)

            kept = []
            for row in rows:
                m_name = (row.get("metric") or {}).get("name", "")
                if not m_name:
                    kept.append(row)
                    continue
                candidate = m_name.rsplit("(", 1)[0].strip()
                cand_emb = self._embed_model.encode([candidate], normalize_embeddings=True)[0]
                sim = float(np.max(ref_embs @ cand_emb))
                if sim >= 0.40:
                    kept.append(row)
            rows = kept

        return rows

    def _generate_fallback_answer(self, question: str, docs: list[str]) -> str:
        docs_text = "\n\n---\n\n".join(docs) if docs else "No relevant documents found."
        user_msg = f"Question: {question}\n\nRelevant risk documents:\n{docs_text}"
        return self._call_llm(FALLBACK_ANSWER_PROMPT, user_msg, max_tokens=1024)

    # ------------------------------------------------------------------
    def ask(self, question: str, verbose: bool = False) -> str:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)

        print("\n[1/3] Resolving names & translating to Cypher …")
        resolved = self.resolve_names(question)
        print(f"      → categories : {resolved['categories']}")
        print(f"      → metrics    : {resolved['metrics'][:3]}")
        print(f"      → risks      : {resolved['risks'][:3]}")
        cypher = self.translate_to_cypher(question)
        print(f"\nGenerated Cypher:\n{cypher}")

        print("\n[2/3] Querying Neo4j …")
        try:
            results = self.run_cypher(cypher)
            deduped_count = len(self._deduplicate_results(results))
            dup_note = f" ({deduped_count} unique)" if deduped_count < len(results) else ""
            print(f"      → {len(results)} graph records returned{dup_note}")
        except Exception as e:
            print(f"  ⚠ Cypher error: {e}")
            return f"Could not execute the generated Cypher query.\nError: {e}\nQuery:\n{cypher}"

        # Fetch peer risks programmatically (LLM Cypher usually only covers target)
        peer_risk_results = self._fetch_peer_risks(resolved)
        if peer_risk_results:
            print(f"      → {len(peer_risk_results)} peer risk records fetched")
            results = results + peer_risk_results

        # Fetch resolved metrics programmatically using proper graph traversal
        metric_results = self._fetch_resolved_metrics(resolved)
        if metric_results:
            print(f"      → {len(metric_results)} metric records fetched")
            results = results + metric_results

        if verbose and results:
            print("\nRaw results (first 3):")
            for r in results[:3]:
                print(" ", r)

        if results:
            self._save_query_results(question, cypher, results)

        print("\n[3/3] Generating answer …")
        if results:
            answer = self.generate_answer(question, cypher, results)
        else:
            print("      → No graph results — falling back to semantic document retrieval …")
            docs = self._retrieve_risk_docs(question)
            answer = self._generate_fallback_answer(question, docs)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)
        return answer

    # ------------------------------------------------------------------
    def _save_query_results(self, question: str, cypher: str, results: list[dict]):
        """Save the raw query results (all node properties) to retrival_results/."""
        from datetime import datetime
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrival_results")
        os.makedirs(results_dir, exist_ok=True)
        slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:60].strip('_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{slug}_{timestamp}.json"
        path = os.path.join(results_dir, filename)
        payload = {
            "question": question,
            "cypher": cypher,
            "record_count": len(results),
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"      → Results saved to: retrival_results/{filename}")

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Credit Risk Q&A — natural language → Cypher → peer-comparative answer"
    )
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print raw graph results")
    args = parser.parse_args()

    qa = CreditRiskQA()
    try:
        if args.question:
            qa.ask(args.question, verbose=args.verbose)
        else:
            print("\nCredit Risk Q&A — interactive mode (type 'exit' to quit)\n")
            while True:
                try:
                    question = input("Question> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    break
                qa.ask(question, verbose=args.verbose)
    finally:
        qa.close()


if __name__ == "__main__":
    main()
