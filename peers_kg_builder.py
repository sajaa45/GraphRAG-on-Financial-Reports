"""
Builds a knowledge graph from SEC XBRL filing CSVs:
  - data/text_filtered_data.csv  → COMPETES_WITH + FACES_RISK  (tag=risk name, value=description)
  - data/filtered_data.csv       → COMPETES_WITH + HAS_METRIC  (tag=metric name, value=numeric)

Both CSVs are filtered to TARGET_SIC so only same-industry peers are loaded.
The main company node is merged (not duplicated) and linked to every peer via COMPETES_WITH.
"""

import os
import re
import time
import argparse
import pandas as pd
from neo4j import GraphDatabase

# ============================================================================
# CONFIGURATION – edit these before running
# ============================================================================
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

MAIN_COMPANY = os.getenv("MAIN_COMPANY", "")   # auto-detected from Neo4j if empty
TARGET_SIC   = os.getenv("TARGET_SIC",   "")   # auto-detected from Neo4j if empty

TEXT_CSV    = "data/text_filtered_data.csv"
METRICS_CSV = "data/filtered_data.csv"
OUTPUT_DIR  = "output"
# ============================================================================


class CompetitorKGBuilder:
    """Load XBRL peer filings into Neo4j as competitors, risks, and metrics."""

    def __init__(self, uri, user, password, main_company, target_sic, output_dir=OUTPUT_DIR):
        self.driver     = GraphDatabase.driver(uri, auth=(user, password))
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log_buffer = []
        self.start_time = time.time()

        print(f"✓ Connected to Neo4j at {uri}")

        # Auto-detect main company and SIC from the graph built by the previous pipeline
        if not main_company or not target_sic:
            detected_company, detected_sic = self._detect_from_graph()
            main_company = main_company or detected_company
            target_sic   = target_sic   or detected_sic

        if not main_company:
            raise ValueError("Could not detect main company. Pass --main-company explicitly.")
        if not target_sic:
            raise ValueError("Could not detect SIC code. Pass --sic explicitly.")

        self.main_company = main_company
        self.target_sic   = int(target_sic)

        # Stamp the node so future runs can detect it instantly
        self._stamp_target_company()

        safe_name = re.sub(r'[^\w\-]', '_', self.main_company).strip('_')
        self.log_file = os.path.join(output_dir, f"peers_kg_{safe_name}.txt")

        print(f"✓ Main company : {self.main_company}")
        print(f"✓ Target SIC   : {self.target_sic}")

    # ------------------------------------------------------------------ utils

    def _stamp_target_company(self):
        """Persist is_target=true on the main company node."""
        try:
            with self.driver.session() as session:
                session.run(
                    "MERGE (c:Company {name: $name}) SET c.is_target = true",
                    {"name": self.main_company},
                )
        except Exception as e:
            print(f"⚠ Could not stamp target company: {e}")

    def _detect_from_graph(self):
        """
        Detect the main company from the graph using multiple signals, in order:
        1. is_target=true property (set by any previous pipeline run)
        2. Company with outgoing COMPETES_WITH + OPERATES_IN
        3. Company that a Person has CEO_OF relationship to
        4. Company with outgoing COMPETES_WITH only
        Returns (company_name, sic_code) or (None, None) on failure.
        """
        try:
            with self.driver.session() as session:
                # 1. is_target flag — fastest and most reliable
                rec = session.run(
                    """
                    MATCH (c:Company {is_target: true})
                    OPTIONAL MATCH (c)-[:OPERATES_IN]->(i:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                    RETURN c.name AS company, s.code AS sic, c.sic AS sic_prop LIMIT 1
                    """
                ).single()
                if rec:
                    sic = rec["sic"] or rec["sic_prop"]
                    print(f"✓ Auto-detected (is_target) → {rec['company']}, SIC: {sic}")
                    return rec["company"], sic

                # 2. COMPETES_WITH + OPERATES_IN
                rec = session.run(
                    """
                    MATCH (c:Company)-[:COMPETES_WITH]->()
                    MATCH (c)-[:OPERATES_IN]->(i:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                    RETURN c.name AS company, s.code AS sic LIMIT 1
                    """
                ).single()
                if rec:
                    print(f"✓ Auto-detected (COMPETES_WITH+OPERATES_IN) → {rec['company']}, SIC: {rec['sic']}")
                    return rec["company"], rec["sic"]

                # 3. CEO_OF signal
                rec = session.run(
                    """
                    MATCH (p:Person)-[:CEO_OF]->(c:Company)
                    OPTIONAL MATCH (c)-[:OPERATES_IN]->(i:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
                    RETURN c.name AS company, s.code AS sic, c.sic AS sic_prop LIMIT 1
                    """
                ).single()
                if rec:
                    sic = rec["sic"] or rec["sic_prop"]
                    print(f"✓ Auto-detected (CEO_OF) → {rec['company']}, SIC: {sic}")
                    return rec["company"], sic

                # 4. COMPETES_WITH only
                rec = session.run(
                    """
                    MATCH (c:Company)-[r:COMPETES_WITH]->()
                    RETURN c.name AS company, r.sic AS sic LIMIT 1
                    """
                ).single()
                if rec:
                    print(f"✓ Auto-detected (COMPETES_WITH) → {rec['company']}, SIC: {rec['sic']}")
                    return rec["company"], rec["sic"]

        except Exception as e:
            print(f"⚠ Graph auto-detection failed: {e}")
        return None, None

    def close(self):
        self.driver.close()
        self._save_log()

    def _log(self, msg: str):
        self.log_buffer.append(msg)
        print(msg)

    def _save_log(self):
        elapsed = time.time() - self.start_time
        header = (
            f"Started : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n"
            f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Elapsed : {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
            f"{'='*80}\n\n"
        )
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header + '\n'.join(self.log_buffer))
        print(f"\n✓ Log saved to: {self.log_file}")

    # -------------------------------------------------------- Neo4j primitives

    def _merge_company(self, session, name: str, sic=None, cik=None):
        props = {}
        if sic is not None:
            props['sic'] = str(sic)
        if cik is not None:
            props['cik'] = str(cik)
        set_clause = (
            ", ".join([f"n.{k} = ${k}" for k in props])
            if props else "n.updated_at = datetime()"
        )
        session.run(
            f"""
            MERGE (n:Company {{name: $name}})
            ON CREATE SET n.created_at = datetime(){(', ' + set_clause) if props else ''}
            ON MATCH  SET {set_clause}
            """,
            {"name": name, **props},
        )

    def _merge_competes_with(self, session, competitor: str):
        session.run(
            """
            MERGE (main:Company {name: $main})
            MERGE (comp:Company {name: $comp})
            MERGE (main)-[r:COMPETES_WITH]->(comp)
            ON CREATE SET r.created_at = datetime(), r.sic = $sic
            """,
            {"main": self.main_company, "comp": competitor, "sic": str(self.target_sic)},
        )

    def _merge_operates_in(self, session, company: str):
        """
        Link a peer company to the same Industry + SICCode nodes that the main
        company already has in the graph (written by domain_multi_relation_kg_builder).
        Falls back to creating bare Industry/SICCode nodes if none exist yet.
        """
        session.run(
            """
            MATCH (main:Company {name: $main})-[:OPERATES_IN]->(i:Industry)-[:HAS_SIC_CODE]->(s:SICCode)
            MERGE (peer:Company {name: $peer})
            MERGE (peer)-[:OPERATES_IN]->(i)
            """,
            {"main": self.main_company, "peer": company},
        )
        # Fallback: if the main company has no OPERATES_IN yet, create minimal nodes
        session.run(
            """
            MATCH (peer:Company {name: $peer})
            WHERE NOT (peer)-[:OPERATES_IN]->()
            MERGE (i:Industry {name: $industry})
              ON CREATE SET i.sic = $sic
            MERGE (s:SICCode  {name: $sic})
              ON CREATE SET s.code = $sic
            MERGE (i)-[:HAS_SIC_CODE]->(s)
            MERGE (peer)-[:OPERATES_IN]->(i)
            """,
            {"peer": company, "industry": f"SIC-{self.target_sic}", "sic": str(self.target_sic)},
        )

    def _merge_risk(self, session, company: str, tag: str, description: str,
                    adsh: str = "", ddate: str = ""):
        """(Company)-[:FACES_RISK]->(Risk)  keyed on tag + company."""
        session.run(
            """
            MERGE (n:Risk {name: $name, source_company: $company})
            ON CREATE SET n.created_at = datetime(),
                          n.description = $description,
                          n.adsh        = $adsh,
                          n.date        = $ddate
            ON MATCH  SET n.description = $description,
                          n.adsh        = $adsh,
                          n.date        = $ddate
            """,
            {"name": tag, "company": company,
             "description": description[:500], "adsh": adsh, "ddate": ddate},
        )
        session.run(
            """
            MATCH (c:Company {name: $company})
            MATCH (r:Risk    {name: $name, source_company: $company})
            MERGE (c)-[rel:FACES_RISK]->(r)
            ON CREATE SET rel.created_at = datetime()
            """,
            {"company": company, "name": tag},
        )

    def _merge_metric(self, session, company: str, tag: str, value: str,
                      adsh: str = "", ddate: str = "", qtrs: str = ""):
        """(Company)-[:HAS_METRIC]->(Metric)  keyed on tag + company + adsh
           so that multiple values for the same tag (same filing) stay distinct."""
        session.run(
            """
            MERGE (n:Metric {name: $name, source_company: $company, adsh: $adsh})
            ON CREATE SET n.created_at = datetime(),
                          n.value = $value,
                          n.date  = $ddate,
                          n.qtrs  = $qtrs
            ON MATCH  SET n.value = $value,
                          n.date  = $ddate,
                          n.qtrs  = $qtrs
            """,
            {"name": tag, "company": company, "adsh": adsh,
             "value": value, "ddate": ddate, "qtrs": qtrs},
        )
        session.run(
            """
            MATCH (c:Company {name: $company})
            MATCH (m:Metric  {name: $name, source_company: $company, adsh: $adsh})
            MERGE (c)-[rel:HAS_METRIC]->(m)
            ON CREATE SET rel.created_at = datetime()
            """,
            {"company": company, "name": tag, "adsh": adsh},
        )

    # --------------------------------------------------------- build methods

    def _collect_peers(self, *csv_paths) -> dict:
        """
        Return {company_name: cik} for all peers across every CSV,
        filtered to TARGET_SIC. Deduplicates across files.
        """
        peers = {}
        for path in csv_paths:
            try:
                df = pd.read_csv(path, dtype=str)
                df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
                filtered = df[df['sic'] == self.target_sic]
                for company in filtered['name'].dropna().unique():
                    company = company.strip()
                    if company and company not in peers:
                        cik = filtered[filtered['name'].str.strip() == company].iloc[0].get('cik', '')
                        peers[company] = str(cik)
            except FileNotFoundError:
                self._log(f"  ⚠ File not found, skipping: {path}")
        return peers

    def _setup_peers(self, session, peers: dict):
        """
        Merge all peer Company nodes, COMPETES_WITH, and OPERATES_IN in one pass
        so companies appearing in both CSVs are never duplicated.
        """
        self._merge_company(session, self.main_company, sic=self.target_sic)
        for company, cik in peers.items():
            self._merge_company(session, company, sic=self.target_sic, cik=cik or None)
            if company.upper() != self.main_company.upper():
                self._merge_competes_with(session, company)
                self._merge_operates_in(session, company)
                self._log(f"  COMPETES_WITH + OPERATES_IN → {company}")

    def build_from_text_csv(self, session, csv_path: str = TEXT_CSV):
        """text_filtered_data.csv → FACES_RISK (company nodes already set up)."""
        df = pd.read_csv(csv_path, dtype=str)
        df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
        filtered = df[df['sic'] == self.target_sic].copy()

        self._log(f"\n[TEXT CSV] {csv_path}")
        self._log(f"  Rows after SIC={self.target_sic} filter : {len(filtered)}")

        for company in [c.strip() for c in filtered['name'].dropna().unique() if c.strip()]:
            rows = filtered[filtered['name'].str.strip() == company]
            risk_count = 0
            for _, row in rows.iterrows():
                tag   = str(row.get('tag',   '')).strip()
                value = str(row.get('value', '')).strip()
                adsh  = str(row.get('adsh',  '')).strip()
                ddate = str(row.get('ddate', '')).strip()
                if tag and value:
                    self._merge_risk(session, company, tag, value, adsh, ddate)
                    risk_count += 1
            self._log(f"    {company} → FACES_RISK: {risk_count}")

    def build_from_metrics_csv(self, session, csv_path: str = METRICS_CSV):
        """filtered_data.csv → HAS_METRIC (company nodes already set up)."""
        df = pd.read_csv(csv_path, dtype=str)
        df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
        filtered = df[df['sic'] == self.target_sic].copy()

        self._log(f"\n[METRICS CSV] {csv_path}")
        self._log(f"  Rows after SIC={self.target_sic} filter : {len(filtered)}")

        for company in [c.strip() for c in filtered['name'].dropna().unique() if c.strip()]:
            rows = filtered[filtered['name'].str.strip() == company]
            metric_count = 0
            for _, row in rows.iterrows():
                tag   = str(row.get('tag',      '')).strip()
                value = str(row.get('value',    '')).strip()
                adsh  = str(row.get('adsh',     '')).strip()
                ddate = str(row.get('ddate',    '')).strip()
                qtrs  = str(row.get('qtrs',     '')).strip()
                if tag and value:
                    self._merge_metric(session, company, tag, value, adsh, ddate, qtrs)
                    metric_count += 1
            self._log(f"    {company} → HAS_METRIC: {metric_count}")

    def build_all(self, text_csv: str = TEXT_CSV, metrics_csv: str = METRICS_CSV):
        self._log("=" * 60)
        self._log(f"Competitor KG  |  {self.main_company}  |  SIC {self.target_sic}")
        self._log("=" * 60)

        peers = self._collect_peers(text_csv, metrics_csv)
        self._log(f"\n  Total unique peers across both CSVs: {len(peers)}")

        with self.driver.session() as session:
            self._setup_peers(session, peers)
            self.build_from_text_csv(session, text_csv)
            self.build_from_metrics_csv(session, metrics_csv)

        self._log("\n✓ Done.")


# --------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(
        description="Build competitor KG from XBRL CSV filings"
    )
    parser.add_argument("--neo4j-uri",      default=NEO4J_URI)
    parser.add_argument("--neo4j-user",     default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    parser.add_argument("--main-company",   default=MAIN_COMPANY or None,
                        help="Company node name already in Neo4j (auto-detected if omitted)")
    parser.add_argument("--sic",            default=TARGET_SIC or None,
                        help="SIC code to filter peers by (auto-detected if omitted)")
    parser.add_argument("--text-csv",       default=TEXT_CSV)
    parser.add_argument("--metrics-csv",    default=METRICS_CSV)
    parser.add_argument("--output-dir",     default=OUTPUT_DIR)
    args = parser.parse_args()

    builder = CompetitorKGBuilder(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        main_company=args.main_company or "",
        target_sic=args.sic or "",
        output_dir=args.output_dir,
    )
    try:
        builder.build_all(args.text_csv, args.metrics_csv)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
