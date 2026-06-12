
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

from neo4j import GraphDatabase


class RisksKGBuilder:

    def __init__(self, neo4j_uri: str = "", neo4j_user: str = "", neo4j_password: str = "",
                 target_company_name: str = "", driver=None):
        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            print(f"✓ Connected to Neo4j at {neo4j_uri}")
        self.target = target_company_name.strip()

    def _resolve_target(self, session) -> bool:
        if not self.target or self.target.lower() in ('', 'the company', 'this company'):
            record = session.run(
                "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1"
            ).single()
            if record:
                self.target = record["name"]
                print(f"✓ Auto-detected target company: '{self.target}'")
            else:
                print("⚠ No TargetCompany found in Neo4j. Run neo4j_builder.py first.")
                return False

        found = session.run(
            "MATCH (c:TargetCompany {name: $name}) RETURN c LIMIT 1",
            {"name": self.target},
        ).single() is not None

        if not found:
            print(f"⚠ Target company '{self.target}' not found — run neo4j_builder.py first.")
        return found

    def _add_competes_with(self, session, peer_name: str):
        if not self.target or peer_name == self.target:
            return
        session.run(
            """
            MATCH (peer:Company {name: $peer})
            MATCH (tgt:TargetCompany {name: $target})
            MERGE (peer)-[:COMPETES_WITH]->(tgt)
            """,
            {"peer": peer_name, "target": self.target},
        )

    def build_from_structured_risks(self, json_file: str) -> int:
        with open(json_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON array in {json_file}")

        total_risks = 0

        print(f"\n{'='*60}")
        print(f"Loading structured risks from: {json_file}")
        print(f"{'='*60}")

        with self.driver.session() as session:
            target_found = self._resolve_target(session)

            for entry in entries:
                company_name = entry.get("company_name", "").strip()
                cik = entry.get("cik", "")
                filing_date = entry.get("filing_date", "")
                document_url = entry.get("document_url", "")
                risks = entry.get("risks", [])

                if not company_name:
                    continue

                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    ON CREATE SET c.cik = $cik, c.filing_date = $filing_date,
                                  c.document_url = $document_url, c.is_peer = true,
                                  c.created_at = datetime()
                    ON MATCH  SET c.cik = $cik, c.filing_date = $filing_date,
                                  c.document_url = $document_url, c.is_peer = true,
                                  c.updated_at = datetime()
                    """,
                    {"name": company_name, "cik": cik,
                     "filing_date": filing_date, "document_url": document_url},
                )

                if target_found:
                    self._add_competes_with(session, peer_name=company_name)

                for risk in risks:
                    risk_id = risk.get("risk_id", "")
                    if not risk_id:
                        continue

                    metadata = risk.get("metadata", {})
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
                        {
                            "risk_id": risk_id,
                            "citation_id": risk.get("citation_id", ""),
                            "risk_name": risk.get("risk_name", "Unnamed Risk"),
                            "description": risk.get("description", ""),
                            "why": risk.get("why", ""),
                            "source_text": metadata.get("source_text", ""),
                            "document_url": metadata.get("document_url", document_url),
                            "filing_date": filing_date,
                        },
                    )

                    session.run(
                        """
                        MATCH (c:Company {name: $name})
                        MATCH (r:Risk {risk_id: $risk_id})
                        MERGE (c)-[:FACES_RISK]->(r)
                        """,
                        {"name": company_name, "risk_id": risk_id},
                    )

                    total_risks += 1

                print(f"  ✓ {company_name} — {len(risks)} risks")

        print(f"\n✓ {len(entries)} companies, {total_risks} risk nodes written")
        return total_risks

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


_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="Build Neo4j KG from structured_risks.json")
    parser.add_argument("--risks-json",
                        default=os.path.join(_ROOT, "structured_risks.json"),
                        help="Path to structured_risks.json")
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
