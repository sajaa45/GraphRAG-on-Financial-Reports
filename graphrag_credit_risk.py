#!/usr/bin/env python3
"""
GraphRAG credit-risk analyzer.

Queries the Neo4j knowledge graph (main company + peers, built by
`domain_multi_relation_kg_builder.py` and `peers_kg_builder.py`) via
LangChain's GraphCypherQAChain. The LLM is AWS Bedrock (model chosen via
BEDROCK_MODEL env var, defaults to a DeepSeek / Llama Bedrock model).

Usage:
    python graphrag_credit_risk.py --query "What is the Debt-to-EBITDA ratio?"
    python graphrag_credit_risk.py --analyze
    python graphrag_credit_risk.py --analyze --main-company "Saudi Aramco"
"""

import os
import sys
import argparse

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Schema description fed to the LLM so it can write valid Cypher.
# ---------------------------------------------------------------------------
SCHEMA_DESCRIPTION = """
The Neo4j graph contains financial data for a MAIN company and its SIC-industry PEERS.

Node labels and key properties:
- Company {{name, sic, cik}}
    The main company node has no `source_company`. Peers merged from SEC XBRL
    CSVs have `sic` and `cik`. Peers are linked to the main company by
    (:Company)-[:COMPETES_WITH]->(:Company).
- Metric {{name, value, unit, year, metric_type, source_company, adsh, date, qtrs}}
    Main-company metrics (extracted from the annual report) use human-readable
    `metric_type` values such as "Debt-to-EBITDA", "Interest Coverage Ratio",
    "Current Ratio", "Operating Cash Flow to Total Debt", "Debt-to-Equity".
    Peer metrics (from XBRL) use `name` = XBRL tag (e.g.
    "DebtLongtermAndShorttermCombinedAmount") and have `source_company` set.
- Risk {{name, description, severity, source_company, adsh, date}}
    Main-company risks use `name` in {{Credit_Risk, Liquidity_Risk, Risk_factor,
    Risk_credit}}. Peer risks come from XBRL with tag-style names.
- Industry {{name, sic}}
- SICCode {{name, code}}
- Person {{name}}

Relationships:
- (Company)-[:HAS_METRIC]->(Metric)
- (Company)-[:FACES_RISK]->(Risk)
- (Company)-[:OPERATES_IN]->(Industry)
- (Industry)-[:HAS_SIC_CODE]->(SICCode)
- (:Company)-[:COMPETES_WITH {{sic}}]->(:Company)   # main -> peer
- (Person)-[:CEO_OF|CFO_OF|BOARD_MEMBER_OF|WORKS_AT]->(Company)

Useful patterns:
- Peers of the main company:
    MATCH (main:Company)-[:COMPETES_WITH]->(peer:Company) RETURN peer.name
- Main company's credit-risk ratios:
    MATCH (main:Company)-[:HAS_METRIC]->(m:Metric)
    WHERE m.metric_type IN ['Debt-to-EBITDA','Interest Coverage Ratio',
                            'Current Ratio','Operating Cash Flow to Total Debt',
                            'Debt-to-Equity']
    RETURN m.metric_type, m.value, m.unit, m.year
- Main company's risks:
    MATCH (main:Company)-[:FACES_RISK]->(r:Risk)
    WHERE r.source_company IS NULL OR r.source_company = main.name
    RETURN r.name, r.description, r.severity
- Peer metrics for the same SIC:
    MATCH (main:Company)-[:COMPETES_WITH]->(peer:Company)-[:HAS_METRIC]->(m:Metric)
    RETURN peer.name, m.name, m.value, m.date
"""


CYPHER_GENERATION_TEMPLATE = """You are an expert Cypher query writer for a financial knowledge graph.

Schema:
{schema}

Additional schema context (authoritative — prefer these patterns when they apply):
""" + SCHEMA_DESCRIPTION + """

Rules:
- Output ONLY the Cypher query, no explanations, no markdown fences.
- Use the exact labels and relationship types from the schema.
- NEVER use Python/JavaScript list comprehensions like [x FOR x IN list]. Use standard Cypher only.
- NEVER use subqueries with collect() and list comprehensions together. Use MATCH + WITH + RETURN instead.
- When filtering by a list of values, use: WHERE x.name IN ['val1', 'val2'] — not list comprehensions.
- When you need to collect peer names first, use a WITH clause:
    MATCH (main:Company)-[:COMPETES_WITH]->(peer:Company)
    WITH collect(peer.name) AS peerNames
    MATCH ...
    WHERE something IN peerNames
- When a question is about the "main company" or "our company" without a
  specific name, match Company nodes that have at least one outgoing
  COMPETES_WITH relationship (that identifies the main company).
- When asking about peers/competitors, traverse COMPETES_WITH from the main company.
- Prefer `m.metric_type` for main-company ratio filters; use `m.name` for
  peer XBRL tags.
- Return only the fields needed to answer the question; limit to 50 rows
  unless the question implies aggregation.
- Always test that your Cypher uses valid Neo4j 5.x syntax.

Question: {question}

Cypher:"""


QA_TEMPLATE = """You are a credit-risk analyst. Use the query results below,
which come from a Neo4j knowledge graph, to answer the user's question.
Focus on credit-risk indicators: leverage (Debt-to-EBITDA, Debt-to-Equity),
coverage (Interest Coverage Ratio), liquidity (Current Ratio, Operating Cash
Flow to Total Debt), and qualitative risks (Credit_Risk, Liquidity_Risk).

When comparing the main company to peers, be explicit about whether the
company looks better, worse, or in-line with peers, and name the peers when
possible. If the data is insufficient, say so — do not invent numbers.

Question: {question}

Graph query results:
{context}

Answer:"""


CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE,
)




class CreditRiskAnalyzer:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        bedrock_model: str,
        aws_region: str,
        temperature: float = 0.0,
    ):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
        )
        self.graph.refresh_schema()

        self.llm = ChatBedrockConverse(
            model=bedrock_model,
            region_name=aws_region,
            temperature=temperature,
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=CYPHER_PROMPT,
            qa_prompt=QA_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    def ask(self, question: str) -> dict:
        return self._invoke_with_retry(question)

    def _invoke_with_retry(self, question: str, max_retries: int = 2) -> dict:
        """
        Invoke the chain for ad-hoc questions. On CypherSyntaxError, ask the
        LLM to fix the query and retry up to max_retries times.
        """
        last_exc = None
        bad_cypher = None

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    return self.chain.invoke({"query": question})
                else:
                    fix_prompt = (
                        f"The following Cypher query is invalid for Neo4j 5.x:\n\n"
                        f"{bad_cypher}\n\n"
                        f"Error: {last_exc}\n\n"
                        f"Rewrite it as valid Neo4j Cypher. "
                        f"Do NOT use Python list comprehensions like [x FOR x IN list]. "
                        f"Use only simple MATCH/WITH/RETURN patterns. "
                        f"Output ONLY the corrected Cypher query, nothing else."
                    )
                    fixed = self.llm.invoke(fix_prompt).content.strip()
                    if fixed.startswith("```"):
                        fixed = "\n".join(
                            ln for ln in fixed.splitlines()
                            if not ln.strip().startswith("```")
                        ).strip()
                    print(f"\n--- Retry {attempt}: fixed Cypher ---\n{fixed}\n")
                    rows = self.graph.query(fixed)
                    qa_text = QA_TEMPLATE.format(question=question, context=rows)
                    answer = self.llm.invoke(qa_text).content.strip()
                    return {"result": answer, "intermediate_steps": [{"query": fixed}, {"context": rows}]}
            except Exception as e:
                err_str = str(e)
                if "SyntaxError" in err_str or "Invalid input" in err_str or "not defined" in err_str:
                    last_exc = err_str
                    # Pull the bad cypher from intermediate_steps if available
                    if hasattr(e, 'args') and bad_cypher is None:
                        bad_cypher = f"(unknown — question: {question})"
                    print(f"⚠ CypherSyntaxError on attempt {attempt + 1}, retrying...")
                else:
                    raise
        raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_exc}")

    def analyze_credit_risk(self, main_company: str) -> dict:
        """
        Fetch ALL metrics and risks for the main company and ALL peer metrics
        and risks, then let the LLM do the comparison — no assumptions about
        what's stored.
        """
        print(f"\n{'='*60}")
        print(f"Gathering graph data for: {main_company}")
        print(f"{'='*60}")

        main_metrics = self.graph.query(
            "MATCH (c:Company {name: $name})-[:HAS_METRIC]->(m:Metric) "
            "RETURN m.name AS name, m.metric_type AS metric_type, m.value AS value, "
            "m.unit AS unit, m.year AS year, m.date AS date",
            {"name": main_company},
        )

        main_risks = self.graph.query(
            "MATCH (c:Company {name: $name})-[:FACES_RISK]->(r:Risk) "
            "RETURN r.name AS name, r.description AS description, r.severity AS severity",
            {"name": main_company},
        )

        peer_metrics = self.graph.query(
            "MATCH (c:Company {name: $name})-[:COMPETES_WITH]->(peer:Company)-[:HAS_METRIC]->(m:Metric) "
            "RETURN peer.name AS peer, m.name AS name, m.value AS value, "
            "m.unit AS unit, m.date AS date "
            "ORDER BY peer.name LIMIT 300",
            {"name": main_company},
        )

        peer_risks = self.graph.query(
            "MATCH (c:Company {name: $name})-[:COMPETES_WITH]->(peer:Company)-[:FACES_RISK]->(r:Risk) "
            "RETURN peer.name AS peer, r.name AS name, r.description AS description "
            "LIMIT 150",
            {"name": main_company},
        )

        context = (
            f"=== {main_company} Metrics ===\n{main_metrics}\n\n"
            f"=== {main_company} Risks ===\n{main_risks}\n\n"
            f"=== Industry Peer Metrics ===\n{peer_metrics}\n\n"
            f"=== Industry Peer Risks ===\n{peer_risks}\n"
        )

        question = (
            f"You have all available metrics and risks for {main_company} and its industry peers below.\n\n"
            f"IMPORTANT — Currency: {main_company} reports in SAR (Saudi Riyals). "
            f"The peer companies are US-listed and report in USD. "
            f"Do NOT directly compare absolute monetary values across currencies. "
            f"For absolute figures, note the currency and convert where needed "
            f"(approximate rate: 1 USD ≈ 3.75 SAR). "
            f"Ratios (e.g. Debt-to-EBITDA, Current Ratio) are dimensionless and can be compared directly.\n\n"
            f"1. List every metric {main_company} has with its value and currency.\n"
            f"2. List every risk {main_company} faces.\n"
            f"3. Compare {main_company}'s metrics to what the industry peers report — "
            f"use ratios for direct comparison; for absolute figures state both currencies explicitly.\n"
            f"4. Note any risks that appear across multiple peers vs unique to {main_company}.\n"
            f"5. Give an overall credit-risk conclusion."
        )

        qa_text = QA_TEMPLATE.format(question=question, context=context)
        answer = self.llm.invoke(qa_text).content.strip()

        print("\n=== Answer ===")
        print(answer)

        return {
            "result": answer,
            "intermediate_steps": [
                {"main_metrics": main_metrics},
                {"main_risks": main_risks},
                {"peer_metrics": peer_metrics},
                {"peer_risks": peer_risks},
            ],
        }

    def detect_main_company(self) -> str:
        """Return the target company — checks is_target flag first, then COMPETES_WITH."""
        rows = self.graph.query(
            "MATCH (c:Company {is_target: true}) RETURN c.name AS name LIMIT 1"
        )
        if rows:
            return rows[0]["name"]
        rows = self.graph.query(
            "MATCH (c:Company)-[:COMPETES_WITH]->() RETURN c.name AS name LIMIT 1"
        )
        if rows:
            return rows[0]["name"]
        rows = self.graph.query("MATCH (c:Company) RETURN c.name AS name LIMIT 1")
        return rows[0]["name"] if rows else ""


def _print_result(result: dict, output_dir: str = "output", main_company: str = "") -> None:
    steps = result.get("intermediate_steps") or []
    for step in steps:
        if isinstance(step, dict) and "query" in step:
            print("\n--- Generated Cypher ---")
            print(step["query"])
        elif isinstance(step, dict) and "context" in step:
            print("\n--- Query Results ---")
            print(step["context"])

    answer = result.get("result", result)
    print("\n=== Answer ===")
    print(answer)

    # Save to file
    import re as _re
    os.makedirs(output_dir, exist_ok=True)
    safe = _re.sub(r'[^\w\-]', '_', main_company).strip('_') if main_company else "credit_risk"
    out_path = os.path.join(output_dir, f"credit_risk_{safe}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Credit Risk Analysis: {main_company}\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(answer))
    print(f"\n✓ Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG credit-risk analyzer")
    parser.add_argument("--query", help="Natural-language question to ask the graph")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run the full credit-risk comparison against peers",
    )
    parser.add_argument(
        "--main-company",
        default=os.getenv("MAIN_COMPANY", ""),
        help="Main company name (auto-detected from the graph if omitted)",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument(
        "--bedrock-model",
        default=os.getenv("BEDROCK_MODEL", "meta.llama3-70b-instruct-v1:0"),
    )
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "output"))
    args = parser.parse_args()

    if not args.query and not args.analyze:
        parser.error("Provide either --query '...' or --analyze")

    analyzer = CreditRiskAnalyzer(
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        bedrock_model=args.bedrock_model,
        aws_region=args.aws_region,
    )

    if args.analyze:
        main_company = args.main_company or analyzer.detect_main_company()
        if not main_company:
            print("ERROR: could not determine main company. Pass --main-company.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Running credit-risk analysis for: {main_company}")
        result = analyzer.analyze_credit_risk(main_company)
    else:
        main_company = args.main_company
        result = analyzer.ask(args.query)

    _print_result(result, output_dir=args.output_dir, main_company=main_company)


if __name__ == "__main__":
    main()
