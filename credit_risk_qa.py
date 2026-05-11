import os
import re
import json
import time
import boto3
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_CYPHER_BLOCK_RE = re.compile(r'```(?:cypher)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Graph schema injected into every LLM prompt so it knows what to query
# ---------------------------------------------------------------------------
GRAPH_SCHEMA = """
Node labels and key properties:
  - TargetCompany   {name, cik, ticker, is_target:true, filing_date, document_url}
  - Company         {name, cik, ticker, is_peer:true, filing_date, document_url}
  - Metric          {name, value, unit, year, metric_type, xbrl_tag, label}
  - Risk            {risk_id, name, description, why, source_text, filing_date}
  - Industry        {name, sector}
  - SICCode         {code, industry, sector}
  - Person          {name}

Relationships:
  (TargetCompany|Company)-[:HAS_METRIC]->(:Metric)
  (TargetCompany|Company)-[:FACES_RISK]->(:Risk)
  (:Company {is_peer:true})-[:COMPETES_WITH]->(:TargetCompany)
  (TargetCompany|Company)-[:OPERATES_IN]->(:Industry)
  (:Industry)-[:HAS_SIC_CODE]->(:SICCode)
  (:Person)-[:CEO_OF|CFO_OF|BOARD_MEMBER_OF|WORKS_AT]->(TargetCompany|Company)

Notes:
  - The company being analyzed is always TargetCompany (use label TargetCompany in MATCH).
  - Peers are Company nodes with is_peer:true that have a COMPETES_WITH edge to TargetCompany.
  - Metric.value is stored as a string; cast with toFloat() when comparing.
  - Metric.metric_type holds the covenant / financial category (e.g. "Debt", "Revenue").
  - HAS_METRIC relationship has: source='SEC_XBRL', filing_date, form, accn.
  - FACES_RISK relationship connects a company to a specific Risk node.
  - IMPORTANT: Risk.risk_id is company-specific (format: CIK_risk_N). Each company has its own
    Risk nodes, so the same risk topic will appear as different nodes with different risk_ids.
    To compare risks across companies, match by risk.name, NOT risk_id.
  - Risk.why contains the specific factual evidence or mechanism explaining why this is a risk.
  - Risk.source_text contains the original risk disclosure text from the 10-K.
  - Always return enough fields for a meaningful comparison (company name, metric values, years).
  - Always use DISTINCT when returning risk rows to avoid duplicates from OPTIONAL MATCH joins.
"""

CYPHER_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert for a KYC / credit-risk knowledge graph.
Given a natural-language question about credit risk, generate a single valid Cypher query
that retrieves the data needed to answer it — including a peer comparison where relevant.

Graph schema:
{GRAPH_SCHEMA}

Rules:
1. Output ONLY the Cypher query inside a ```cypher ... ``` code block. No explanation.
2. Always include the TargetCompany and its peers in comparisons.
3. Use OPTIONAL MATCH for peer data so the query works even if no peers exist.
4. No limit.
5. Prefer readable aliases (e.g. tc.name AS company, m.value AS value, m.year AS year).
6. Never use DETACH DELETE or any write operation.
7. CRITICAL: When using WITH, include ALL variables you need in subsequent clauses.
8. CRITICAL: WHERE clauses must come immediately after MATCH/OPTIONAL MATCH, not after WITH.
9. Use COLLECT() and list operations carefully - ensure variables are in scope.
10. For filtering after aggregation, use WHERE in the WITH clause itself.

Common patterns:
- Compare target to peers: MATCH (tc:TargetCompany)-[rel]->(node) OPTIONAL MATCH (peer:Company {{is_peer:true}})-[peer_rel]->(peer_node) RETURN ...
- Filter risks by keyword in why: MATCH (c)-[:FACES_RISK]->(r:Risk) WHERE toLower(r.why) CONTAINS 'financial' RETURN ...
- Count risks per company: MATCH (c)-[:FACES_RISK]->(r:Risk) RETURN c.name, count(r) AS risk_count
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
        """Generate Cypher with retry on syntax errors."""
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
            
            raw = self._call_llm(CYPHER_SYSTEM_PROMPT, user_msg)
            match = _CYPHER_BLOCK_RE.search(raw)
            if match:
                cypher = match.group(1).strip()
            else:
                # fallback: return whatever came back stripped of obvious prose
                lines = [l for l in raw.splitlines() if not l.strip().startswith('#') and l.strip()]
                cypher = "\n".join(lines)
            
            # Quick syntax validation
            if attempt < max_retries - 1:
                try:
                    # Try to run EXPLAIN to validate syntax without executing
                    with self.driver.session() as session:
                        session.run(f"EXPLAIN {cypher}")
                    return cypher  # Valid syntax
                except Exception as e:
                    last_error = str(e)
                    print(f"  ⚠ Attempt {attempt + 1} generated invalid Cypher, retrying...")
                    continue
            
            return cypher

    # ------------------------------------------------------------------
    def run_cypher(self, query: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    def _deduplicate_results(self, results: list[dict], max_records: int = 100) -> list[dict]:
        """Remove duplicate rows and cap total records sent to the LLM."""
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
            results_text = json.dumps(deduped, indent=2, default=str)
            if truncated:
                results_text += f"\n\n[Note: {len(results)} total records returned; showing {len(deduped)} unique rows after deduplication.]"

        user_msg = (
            f"Question: {question}\n\n"
            f"Cypher query used:\n```cypher\n{cypher}\n```\n\n"
            f"Graph results ({len(results)} records):\n{results_text}"
        )
        return self._call_llm(ANSWER_SYSTEM_PROMPT, user_msg, max_tokens=1024)

    # ------------------------------------------------------------------
    def ask(self, question: str, verbose: bool = False) -> str:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)

        print("\n[1/3] Translating to Cypher …")
        cypher = self.translate_to_cypher(question)
        print(f"\nGenerated Cypher:\n{cypher}")

        print("\n[2/3] Querying Neo4j …")
        try:
            results = self.run_cypher(cypher)
            deduped_count = len(self._deduplicate_results(results))
            dup_note = f" ({deduped_count} unique)" if deduped_count < len(results) else ""
            print(f"      → {len(results)} records returned{dup_note}")
        except Exception as e:
            print(f"  ⚠ Cypher error: {e}")
            return f"Could not execute the generated Cypher query.\nError: {e}\nQuery:\n{cypher}"

        if verbose and results:
            print("\nRaw results (first 3):")
            for r in results[:3]:
                print(" ", r)

        print("\n[3/3] Generating answer …")
        answer = self.generate_answer(question, cypher, results)

        print(f"\n{'='*70}")
        print("Answer:")
        print('='*70)
        print(answer)
        return answer

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
