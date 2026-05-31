
import os
import sys
import json
import argparse
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from neo4j import GraphDatabase
from industry_node_to_sic import get_sic_code
from company_utils import CompanyDetector


class Neo4jBuilder:
    """Reads extracted-entity JSON produced by LLMExtractor and writes the
    corresponding nodes and relationships into Neo4j."""

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 main_company: str = ""):

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"✓ Connected to Neo4j at {neo4j_uri}")

        self.main_company = main_company
        self._sic_cache: Dict[str, str] = {}
        
        # Citation ID counters for target company
        self._target_metric_counter = 0
        self._target_risk_counter = 0

    def stamp_target_company(self, main_company: str):
        """Convert Company node to TargetCompany node type for the main company."""
        if main_company.lower() in ('the company', 'this company', ''):
            print("⚠ Skipping stamp — company name not yet resolved")
            return
        try:
            with self.driver.session() as session:
                session.run(
                    "MATCH (c:Company) WHERE toLower(c.name) IN ['the company', 'this company'] DETACH DELETE c"
                )
                
                canonical_tokens = set(CompanyDetector._significant_tokens(main_company))
                if canonical_tokens:
                    result = session.run(
                        "MATCH (c:Company) WHERE c.name <> $name RETURN c.name AS name",
                        {"name": main_company}
                    )
                    for record in result:
                        variant = record["name"]
                        if canonical_tokens == set(CompanyDetector._significant_tokens(variant)):
                            session.run(
                                "MATCH (c:Company {name: $variant}) DETACH DELETE c",
                                {"variant": variant}
                            )
                            print(f"  ✓ Removed case-variant duplicate: '{variant}'")
                
                result = session.run(
                    "MATCH (c:Company {name: $name}) RETURN c",
                    {"name": main_company}
                )
                
                if result.single():
                    # Convert existing Company node to TargetCompany by recreating with relationships
                    session.run(
                        """
                        MATCH (old:Company {name: $name})
                        OPTIONAL MATCH (old)-[r_out]->(target)
                        OPTIONAL MATCH (source)-[r_in]->(old)
                        WITH old, 
                             collect(DISTINCT {type: type(r_out), props: properties(r_out), target: target}) as outRels,
                             collect(DISTINCT {type: type(r_in), props: properties(r_in), source: source}) as inRels,
                             properties(old) as oldProps
                        CREATE (new:TargetCompany {name: $name})
                        SET new = oldProps, new.is_target = true
                        WITH new, old, outRels, inRels
                        UNWIND outRels as outRel
                        FOREACH (dummy in CASE WHEN outRel.target IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (new)-[r:DUMMY]->(outRel.target)
                        )
                        WITH new, old, inRels
                        UNWIND inRels as inRel
                        FOREACH (dummy in CASE WHEN inRel.source IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (inRel.source)-[r:DUMMY]->(new)
                        )
                        WITH old
                        DETACH DELETE old
                        """,
                        {"name": main_company}
                    )
                    print(f"✓ Converted Company → TargetCompany: {main_company}")
                else:
                    #  new TargetCompany node
                    session.run(
                        """
                        MERGE (c:TargetCompany {name: $name})
                        SET c.is_target = true
                        """,
                        {"name": main_company}
                    )
                    print(f"✓ Created TargetCompany node: {main_company}")
                    
        except Exception as e:
            print(f"⚠ Could not stamp target company: {e}")

    def _lookup_sic(self, sector: str) -> str:
        if not sector:
            return None
        key = sector.strip().lower()
        if key not in self._sic_cache:
            try:
                code = get_sic_code(sector)
                self._sic_cache[key] = str(code).strip()
                print(f"    ✓ SIC lookup: '{sector}' → {self._sic_cache[key]}")
            except Exception as e:
                print(f"    ⚠ SIC lookup failed for '{sector}': {e}")
                self._sic_cache[key] = None
        return self._sic_cache[key]

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

    def create_node(self, session, node_type: str, name: str, properties: Dict = None):
        # If this is the main company and node_type is Company, use TargetCompany instead
        if node_type == "Company" and name == self.main_company:
            node_type = "TargetCompany"
            properties = properties or {}
            properties['is_target'] = True

        props = properties or {}

        # Target MetricCategory and Metric nodes have no `company` property.
        # Peer versions of these nodes carry `company: peer_name`.
        # A plain MERGE on {name} would match peer nodes on re-runs (Neo4j MERGE
        # matches any node whose *specified* properties match, ignoring extras).
        # Use OPTIONAL MATCH + FOREACH to create only when a target-scoped node is absent,
        # then MATCH + SET to update properties.
        if node_type in ("MetricCategory", "Metric") and not props.get("company"):
            session.run(
                f"""
                OPTIONAL MATCH (existing:{node_type} {{name: $name}})
                WHERE existing.company IS NULL
                WITH existing
                FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
                    CREATE (:{node_type} {{name: $name, created_at: datetime()}})
                )
                """,
                {"name": name},
            )
            # Set any additional properties (e.g. metric_type, value, year, citation_id)
            extra = {k: v for k, v in props.items() if k != "name"}
            if extra:
                extra_str = ", ".join(f"n.{k} = ${k}" for k in extra)
                session.run(
                    f"""
                    MATCH (n:{node_type} {{name: $name}}) WHERE n.company IS NULL
                    SET {extra_str}
                    """,
                    {"name": name, **extra},
                )
            return

        props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        ON CREATE SET n.created_at = datetime() {', ' + props_str if props_str else ''}
        ON MATCH SET {props_str if props_str else 'n.updated_at = datetime()'}
        RETURN n
        """
        session.run(query, {"name": name, **props})

    def create_relationship(self, session, source_type, source_name, target_type,
                            target_name, rel_type, properties=None, source_chunk=None,
                            similarity=None, section_title=None, source_page=None):
        # If source or target is the main company and type is Company, use TargetCompany instead
        if source_type == "Company" and source_name == self.main_company:
            source_type = "TargetCompany"
        if target_type == "Company" and target_name == self.main_company:
            target_type = "TargetCompany"
            
        props = dict(properties or {})
        if source_chunk:
            props['source_chunk'] = source_chunk[:200]
        if similarity is not None:
            props['confidence'] = round(float(similarity), 4)
        if section_title:
            props['section_title'] = section_title
        if source_page is not None:
            props['source_page'] = source_page

        props_set = ", ".join(f"r.{k} = ${k}" for k in props)
        on_create = f"r.created_at = datetime(){', ' + props_set if props_set else ''}"
        on_match  = f"r.updated_at = datetime(){', ' + props_set if props_set else ''}"

        # Peer MetricCategory and Metric nodes carry a `company` property; target ones do not.
        # Guard both ends so target-side queries never touch peer-scoped nodes.
        _TARGET_SCOPED = ("MetricCategory", "Metric")

        if source_type in _TARGET_SCOPED:
            match_s = f"MATCH (s:{source_type} {{name: $source_name}}) WHERE s.company IS NULL"
        else:
            match_s = f"MATCH (s:{source_type} {{name: $source_name}})"

        if target_type in _TARGET_SCOPED:
            match_t = f"MATCH (t:{target_type} {{name: $target_name}}) WHERE t.company IS NULL"
        else:
            match_t = f"MATCH (t:{target_type} {{name: $target_name}})"

        query = f"""
        {match_s}
        {match_t}
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET {on_create}
        ON MATCH SET {on_match}
        RETURN r
        """
        session.run(query, {"source_name": source_name, "target_name": target_name, **props})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_validated_file(filepath: str) -> Optional[Dict]:
        """Load a validated JSON file and normalise to {main_company, relations: {REL: [...]}}."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_company = data.get('main_company', '')

        # OPERATES_IN format: single item under 'validated_relation'
        if 'validated_relation' in data:
            item = data['validated_relation']
            rel_type = item.get('rel', '')
            if not rel_type:
                print(f"  ⚠ No 'rel' key in validated_relation in {filepath}")
                return None
            return {'main_company': main_company, 'relations': {rel_type: [item]}}

        # Standard format: already has 'relations' dict
        if 'relations' in data:
            return data

        print(f"  ⚠ Unrecognised validated file format: {filepath}")
        return None

    def _build_from_data(self, data: Dict) -> int:
        """Write all entities/relationships from a normalised data dict to Neo4j."""
        relations = data.get("relations", {})
        total_written = 0

        # Only keep peers that actually have risks or metrics — skip bare OPERATES_IN-only peers.
        valid_peers: set = set()
        for rel_name in ("FACES_RISK", "HAS_METRIC"):
            for item in relations.get(rel_name, []):
                src_name = item["src"]["name"]
                if src_name != self.main_company:
                    valid_peers.add(src_name)

        for relation_name, items in relations.items():
            if not items:
                continue

            # Drop items whose source is a peer company not in valid_peers
            filtered = []
            for item in items:
                src_name = item["src"]["name"]
                src_type = item["src"]["type"]
                if src_type == "Company" and src_name != self.main_company and src_name not in valid_peers:
                    continue
                filtered.append(item)

            if not filtered:
                print(f"  ⚠ All {relation_name} items filtered out (no qualifying peers) — skipping")
                continue
            skipped = len(items) - len(filtered)
            if skipped:
                print(f"  ↳ Skipped {skipped} {relation_name} item(s) from peers with no risks/metrics")
            items = filtered
            print(f"\n{'='*60}")
            print(f"Building Neo4j graph for: {relation_name} ({len(items)} items)")
            print(f"{'='*60}")

            # For HAS_METRIC: wipe all target-scoped MetricCategory nodes before rebuilding.
            # This guarantees no duplicate categories or stale HAS_METRIC / HAS_METRIC_CATEGORY
            # edges from previous runs. Peer MetricCategory nodes (company IS NOT NULL) are untouched.
            # Also collapse any duplicate target Metric nodes (same name, no company) into one.
            if relation_name == "HAS_METRIC":
                with self.driver.session() as _cleanup_session:
                    _cleanup_session.run(
                        "MATCH (mc:MetricCategory) WHERE mc.company IS NULL DETACH DELETE mc"
                    )
                    _cleanup_session.run(
                        """
                        MATCH (m:Metric) WHERE m.company IS NULL
                        WITH m.name AS nm, collect(m) AS dupes
                        WHERE size(dupes) > 1
                        FOREACH (d IN tail(dupes) | DETACH DELETE d)
                        """
                    )

            def _write_batch(tx, batch=items):
                for item in batch:
                    raw_meta = {}
                    if item.get('source_page') is not None:
                        raw_meta['source_page'] = item['source_page']
                    if item.get('section_title'):
                        raw_meta['section_title'] = item['section_title']
                    if item.get('similarity') is not None:
                        raw_meta['confidence'] = round(float(item['similarity']), 4)
                    if item.get('chunk_text'):
                        raw_meta['source_text'] = item['chunk_text']
                    metadata_str = json.dumps(raw_meta) if raw_meta else None

                    src_props = dict(item['src'].get('properties', {}))
                    if metadata_str:
                        src_props['metadata'] = metadata_str
                    self.create_node(tx, item['src']['type'], item['src']['name'], src_props)

                    tgt_props = dict(item['tgt'].get('properties', {}))
                    if metadata_str:
                        tgt_props['metadata'] = metadata_str

                    if item.get('rel') == 'HAS_METRIC':
                        # Assign citation_id for target company metrics
                        if item['src']['name'] == self.main_company:
                            self._target_metric_counter += 1
                            tgt_props['citation_id'] = f"TARGET_M{self._target_metric_counter:04d}"

                        # Ensure gaap_concept is always populated — fall back to metric_type
                        # so peer comparison always has a canonical GAAP name to align on.
                        if not tgt_props.get('gaap_concept'):
                            tgt_props['gaap_concept'] = (
                                tgt_props.get('metric_type')
                                or item['tgt'].get('name', '')
                            )

                        # Store source metadata as flat properties — same pattern as FACES_RISK.
                        tgt_props.pop('metadata', None)
                        if item.get('source_page') is not None:
                            tgt_props['source_page'] = item['source_page']
                        if item.get('section_title'):
                            tgt_props['section_title'] = item['section_title']
                        if item.get('chunk_text'):
                            tgt_props['source_text'] = item['chunk_text']

                        # Structure:
                        #   Company -[HAS_METRIC_CATEGORY]-> MetricCategory -[HAS_METRIC]-> Metric
                        category = tgt_props.get('category', None) or 'Uncategorised'

                        self.create_node(tx, 'MetricCategory', category, {'name': category})
                        self.create_relationship(
                            tx,
                            item['src']['type'], item['src']['name'],
                            'MetricCategory', category,
                            'HAS_METRIC_CATEGORY',
                        )
                        self.create_node(tx, item['tgt']['type'], item['tgt']['name'], tgt_props)
                        self.create_relationship(
                            tx,
                            'MetricCategory', category,
                            item['tgt']['type'], item['tgt']['name'],
                            'HAS_METRIC', item.get('props', {}),
                            item.get('chunk_text'), item.get('similarity'),
                            item.get('section_title'), item.get('source_page'),
                        )
                    elif item.get('rel') == 'FACES_RISK':
                        # Assign citation_id for target company risks
                        if item['src']['name'] == self.main_company:
                            self._target_risk_counter += 1
                            tgt_props['citation_id'] = f"TARGET_R{self._target_risk_counter:04d}"

                        # Store source metadata as flat properties (not nested JSON string)
                        # so Cypher can access r.source_page, r.section_title, r.source_text directly
                        tgt_props.pop('metadata', None)
                        if item.get('source_page') is not None:
                            tgt_props['source_page'] = item['source_page']
                        if item.get('section_title'):
                            tgt_props['section_title'] = item['section_title']
                        if item.get('chunk_text'):
                            tgt_props['source_text'] = item['chunk_text']

                        self.create_node(tx, item['tgt']['type'], item['tgt']['name'], tgt_props)
                        self.create_relationship(
                            tx,
                            item['src']['type'], item['src']['name'],
                            item['tgt']['type'], item['tgt']['name'],
                            item['rel'], item.get('props', {}),
                            item.get('chunk_text'), item.get('similarity'),
                            item.get('section_title'), item.get('source_page'),
                        )
                    else:
                        self.create_node(tx, item['tgt']['type'], item['tgt']['name'], tgt_props)
                        self.create_relationship(
                            tx,
                            item['src']['type'], item['src']['name'],
                            item['tgt']['type'], item['tgt']['name'],
                            item['rel'], item.get('props', {}),
                            item.get('chunk_text'), item.get('similarity'),
                            item.get('section_title'), item.get('source_page'),
                        )

                    sic = item.get('sic')
                    if not sic and item.get('rel') == 'OPERATES_IN':
                        industry_name = item['tgt']['name']
                        sector = item['tgt'].get('properties', {}).get('sector', '')
                        sic_code = self._lookup_sic(industry_name)
                        if sic_code:
                            sic = {
                                'code': sic_code,
                                'industry': industry_name,
                                'sector': sector,
                                'src_type': item['tgt']['type'],
                                'src_name': item['tgt']['name'],
                            }
                    if sic:
                        s = sic
                        self.create_node(tx, 'SICCode', s['code'],
                                         {'code': s['code'], 'industry': s['industry'], 'sector': s['sector']})
                        self.create_relationship(tx, s['src_type'], s['src_name'], 'SICCode', s['code'], 'HAS_SIC_CODE')

            with self.driver.session() as session:
                session.execute_write(_write_batch)

            total_written += len(items)
            print(f"  ✓ Written {len(items)} items for {relation_name}")

        print(f"\n✓ Total items written to Neo4j: {total_written}")
        return total_written

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def build_from_json(self, json_file: str, clear: bool = False):
        """Read a raw extracted JSON file and write to Neo4j."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_company = data.get("main_company", self.main_company)
        self.main_company = main_company
        self.stamp_target_company(main_company)

        if clear:
            self.clear_database()

        return self._build_from_data(data)

    def build_from_validated_dirs(self, relations_dir: str, clear: bool = False):
        """
        Scan each subdirectory of relations_dir for a validated_*.json file,
        merge all relations across files, and write to Neo4j.

        Expected layout:
            relations_dir/
                OPERATES_IN/validated_industry.json
                FACES_RISK/validated_risks.json
                HAS_METRIC/validated_metrics.json
        """
        if not os.path.isdir(relations_dir):
            raise FileNotFoundError(f"Relations directory not found: {relations_dir}")

        combined_relations: Dict[str, List] = {}
        main_company = self.main_company

        for entry in sorted(os.listdir(relations_dir)):
            sub_dir = os.path.join(relations_dir, entry)
            if not os.path.isdir(sub_dir):
                continue

            validated_files = sorted(
                f for f in os.listdir(sub_dir)
                if f.startswith('validated_') and f.endswith('.json')
            )

            if not validated_files:
                print(f"  ⚠ No validated_*.json found in {sub_dir} — skipping")
                continue

            for fname in validated_files:
                fpath = os.path.join(sub_dir, fname)
                print(f"  Loading {os.path.join(entry, fname)} ...")
                normalised = self._normalise_validated_file(fpath)
                if normalised is None:
                    continue

                if not main_company:
                    main_company = normalised.get('main_company', '')

                for rel_type, items in normalised.get('relations', {}).items():
                    combined_relations.setdefault(rel_type, []).extend(items)
                    print(f"    ✓ {len(items)} {rel_type} items")

        if not main_company:
            raise ValueError("Could not determine main_company from any validated file.")

        self.main_company = main_company
        self.stamp_target_company(main_company)

        if clear:
            self.clear_database()

        total = sum(len(v) for v in combined_relations.values())
        print(f"\nMerged {total} total items across {len(combined_relations)} relation type(s): "
              f"{', '.join(combined_relations)}")

        return self._build_from_data({'main_company': main_company, 'relations': combined_relations})

    def show_graph_stats(self):
        with self.driver.session() as session:
            print(f"\n{'='*60}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'='*60}")
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("\nNodes:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\nRelationships:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")

    def close(self):
        self.driver.close()


def main():
    parser = argparse.ArgumentParser(description="Neo4j Builder — loads extracted/validated JSON into Neo4j")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "json_file",
        nargs="?",
        help="Path to a single raw extracted JSON file (legacy mode)",
    )
    mode.add_argument(
        "--validated-dir",
        metavar="DIR",
        help=(
            "Directory containing relation subdirs with validated_*.json files "
            "(e.g. KG_building/relations). "
            "Defaults to <this script's dir>/relations when flag is given without a value."
        ),
    )

    parser.add_argument("--clear", action="store_true", help="Clear database before loading")

    args = parser.parse_args()

    builder = Neo4jBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "Lexical12345*"),
    )

    try:
        if args.validated_dir:
            relations_dir = args.validated_dir
            builder.build_from_validated_dirs(relations_dir, clear=args.clear)
        else:
            builder.build_from_json(args.json_file, clear=args.clear)

        builder.show_graph_stats()
    finally:
        builder.close()


if __name__ == "__main__":
    main()
