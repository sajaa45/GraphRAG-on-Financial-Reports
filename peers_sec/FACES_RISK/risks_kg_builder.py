
import os
import sys
import json
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

from neo4j import GraphDatabase


class RisksKGBuilder:
    """
    Builds a Neo4j knowledge graph from:
      - structured_risks.json  → Company, Risk nodes
      - companies_list.json    → peer Company nodes with COMPETES_WITH edges

    Graph schema
    ------------
    (:Company)-[:FACES_RISK]->(:Risk)
    (:Company {is_peer:true})-[:COMPETES_WITH]->(:Company {is_target:true})
    """

    def __init__(self, neo4j_uri: str = "", neo4j_user: str = "", neo4j_password: str = "",
                 target_company_name: str = "", driver=None):
        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            print(f"✓ Connected to Neo4j at {neo4j_uri}")
        self.target = target_company_name.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_target(self, session) -> bool:
        """
        If self.target is empty, auto-detect it from the TargetCompany node
        created by neo4j_builder.py.  Returns True when a target is available.
        """
        if not self.target or self.target.lower() in ('', 'the company', 'this company'):
            # First try TargetCompany node
            result = session.run(
                "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1"
            )
            record = result.single()
            if record:
                self.target = record["name"]
                print(f"✓ Auto-detected target company: '{self.target}'")
            else:
                # Fallback to Company with is_target flag
                result = session.run(
                    "MATCH (c:Company {is_target: true}) RETURN c.name AS name LIMIT 1"
                )
                record = result.single()
                if record:
                    self.target = record["name"]
                    print(f"✓ Auto-detected target company: '{self.target}'")
                else:
                    print("⚠ No target company found in Neo4j. "
                          "Run neo4j_builder.py first or pass --target-company.")
                    return False

        # Check if target exists as TargetCompany or Company
        result = session.run(
            "MATCH (c) WHERE (c:TargetCompany OR c:Company) AND c.name = $name RETURN c LIMIT 1",
            {"name": self.target},
        )
        found = result.single() is not None
        if not found:
            print(f"⚠ Target company '{self.target}' not found in Neo4j — "
                  f"run neo4j_builder.py first.")
        return found

    def _verify_target(self, session) -> bool:
        return self._resolve_target(session)

    def _add_competes_with(self, session, peer_name: str):
        """Add COMPETES_WITH from peer → target. Target must already exist."""
        if not self.target or self.target.lower() in ('', 'the company', 'this company'):
            return
        if peer_name == self.target:
            return
        session.run(
            """
            MATCH (peer:Company {name: $peer})
            MATCH (tgt) WHERE (tgt:TargetCompany OR tgt:Company) AND tgt.name = $target
            MERGE (peer)-[:COMPETES_WITH]->(tgt)
            """,
            {"peer": peer_name, "target": self.target},
        )

    # ------------------------------------------------------------------
    # Public: structured_risks.json
    # ------------------------------------------------------------------
    def build_from_structured_risks(self, json_file: str) -> int:
        """
        Load structured_risks.json and write:
          Company → FACES_RISK → Risk
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON array in {json_file}")

        total_risks = 0
        
        # Global counter for peer risks (in case citation_id is missing)
        peer_risk_counter = 0

        print(f"\n{'='*60}")
        print(f"Loading structured risks from: {json_file}")
        print(f"{'='*60}")

        with self.driver.session() as session:
            target_found = self._verify_target(session)

            for entry in entries:
                company_name = entry.get("company_name", "").strip()
                cik = entry.get("cik", "")
                filing_date = entry.get("filing_date", "")
                document_url = entry.get("document_url", "")
                risks = entry.get("risks", [])

                if not company_name:
                    continue

                is_target = bool(self.target) and company_name == self.target
                node_type = "TargetCompany" if is_target else "Company"
                role_flag = "is_target" if is_target else "is_peer"

                # Merge on name - use TargetCompany for target, Company for peers
                session.run(
                    f"""
                    MERGE (c:{node_type} {{name: $name}})
                    ON CREATE SET c.cik = $cik, c.filing_date = $filing_date,
                                  c.document_url = $document_url, c.{role_flag} = true,
                                  c.created_at = datetime()
                    ON MATCH  SET c.cik = $cik, c.filing_date = $filing_date,
                                  c.document_url = $document_url, c.{role_flag} = true,
                                  c.updated_at = datetime()
                    """,
                    {"name": company_name, "cik": cik,
                     "filing_date": filing_date, "document_url": document_url},
                )

                if target_found:
                    self._add_competes_with(session, peer_name=company_name)

                for risk in risks:
                    risk_id = risk.get("risk_id", "")
                    citation_id = risk.get("citation_id", "")
                    
                    # If citation_id is missing, generate one
                    if not citation_id:
                        peer_risk_counter += 1
                        citation_id = f"PEER_R{peer_risk_counter:04d}"
                    
                    risk_name = risk.get("risk_name", "Unnamed Risk")
                    description = risk.get("description", "")
                    why = risk.get("why", "")
                    
                    # Extract metadata fields
                    metadata = risk.get("metadata", {})
                    source_text = metadata.get("source_text", "")
                    risk_document_url = metadata.get("document_url", document_url)  # fallback to company doc_url

                    if not risk_id:
                        continue

                    # Upsert Risk node with metadata fields
                    session.run(
                        """
                        MERGE (r:Risk {risk_id: $risk_id})
                        ON CREATE SET r.citation_id = $citation_id,
                                      r.name = $risk_name, r.description = $description,
                                      r.why = $why,
                                      r.source_text = $source_text,
                                      r.document_url = $document_url,
                                      r.filing_date = $filing_date,
                                      r.created_at = datetime()
                        ON MATCH  SET r.citation_id = $citation_id,
                                      r.name = $risk_name, r.description = $description,
                                      r.why = $why,
                                      r.source_text = $source_text,
                                      r.document_url = $document_url,
                                      r.updated_at = datetime()
                        """,
                        {"risk_id": risk_id, "citation_id": citation_id, "risk_name": risk_name,
                         "description": description, "why": why,
                         "source_text": source_text, "document_url": risk_document_url,
                         "filing_date": filing_date},
                    )

                    # Company/TargetCompany -[FACES_RISK]-> Risk  (always look up by name)
                    session.run(
                        """
                        MATCH (c) WHERE (c:Company OR c:TargetCompany) AND c.name = $name
                        MATCH (r:Risk {risk_id: $risk_id})
                        MERGE (c)-[:FACES_RISK]->(r)
                        """,
                        {"name": company_name, "risk_id": risk_id},
                    )

                    total_risks += 1

                print(f"  ✓ {company_name} — {len(risks)} risks")

        print(f"\n✓ Structured risks: {len(entries)} companies, {total_risks} risk nodes written")
        return total_risks

    # ------------------------------------------------------------------
    # Public: companies_list.json
    # ------------------------------------------------------------------
    def build_peer_companies(self, json_file: str) -> int:
        """
        Load companies_list.json and create Company peer nodes.
        Attaches COMPETES_WITH edges to the target company if one is set.
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        companies = data.get("companies", data) if isinstance(data, dict) else data

        print(f"\n{'='*60}")
        print(f"Loading peer companies from: {json_file}")
        print(f"{'='*60}")

        with self.driver.session() as session:
            target_found = self._verify_target(session)

            for company in companies:
                name = company.get("name", "").strip()
                cik = company.get("cik", "")
                ticker = company.get("ticker", "")
                filing_date = company.get("filing_date", "")

                if not name:
                    continue

                is_target = target_found and name == self.target
                node_type = "TargetCompany" if is_target else "Company"
                role_flag = "is_target" if is_target else "is_peer"

                session.run(
                    f"""
                    MERGE (c:{node_type} {{name: $name}})
                    ON CREATE SET c.cik = $cik, c.ticker = $ticker,
                                  c.filing_date = $filing_date, c.{role_flag} = true,
                                  c.created_at = datetime()
                    ON MATCH  SET c.cik = $cik, c.ticker = $ticker,
                                  c.filing_date = $filing_date, c.{role_flag} = true,
                                  c.updated_at = datetime()
                    """,
                    {"name": name, "cik": cik, "ticker": ticker, "filing_date": filing_date},
                )

                if target_found:
                    self._add_competes_with(session, peer_name=name)

        print(f"✓ Peer companies: {len(companies)} nodes written")
        return len(companies)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def show_graph_stats(self):
        with self.driver.session() as session:
            print(f"\n{'='*60}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'='*60}")
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC"
            )
            print("\nNodes:")
            for r in result:
                print(f"  {r['type']}: {r['count']}")
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
            )
            print("\nRelationships:")
            for r in result:
                print(f"  {r['type']}: {r['count']}")

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

    def close(self):
        self.driver.close()


# ----------------------------------------------------------------------
# CLI  — all arguments have defaults so bare `python risks_kg_builder.py` works
# ----------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        description="Build a Neo4j KG from structured_risks.json"
    )
    parser.add_argument("--risks-json",
                        default=os.path.join(_ROOT, "structured_risks.json"),
                        help="Path to structured_risks.json (default: FACES_RISK/structured_risks.json)")
    parser.add_argument("--target-company", default="",
                        help="Target company name — auto-detected from Neo4j if omitted")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the entire database before loading")

    args = parser.parse_args()

    builder = RisksKGBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        target_company_name=args.target_company,
    )

    try:
        if args.clear:
            builder.clear_database()

        if os.path.exists(args.risks_json):
            builder.build_from_structured_risks(args.risks_json)
        else:
            print(f"⚠ risks-json not found: {args.risks_json}")

        builder.show_graph_stats()
    finally:
        builder.close()


if __name__ == "__main__":
    main()
