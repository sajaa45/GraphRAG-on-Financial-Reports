
import os
import sys
import json
import argparse
import re
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from neo4j import GraphDatabase
from industry_node_to_sic import get_sic_code

_STOP_WORDS = frozenset({
    'the', 'and', 'for', 'of', 'in', 'a', 'an',
    'company', 'corporation', 'incorporated', 'limited', 'group', 'holdings',
})
_TOKEN_CLEAN_RE = re.compile(r'[^a-z0-9 ]')


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

    # ========================================================================
    # COMPANY STAMP
    # ========================================================================
    @staticmethod
    def _significant_tokens(name: str) -> List[str]:
        return [t for t in _TOKEN_CLEAN_RE.sub('', name.lower()).split()
                if len(t) > 3 and t not in _STOP_WORDS]

    def stamp_target_company(self, main_company: str):
        """Convert Company node to TargetCompany node type for the main company."""
        if main_company.lower() in ('the company', 'this company', ''):
            print("⚠ Skipping stamp — company name not yet resolved")
            return
        try:
            with self.driver.session() as session:
                # Remove generic placeholder companies
                session.run(
                    "MATCH (c:Company) WHERE toLower(c.name) IN ['the company', 'this company'] DETACH DELETE c"
                )
                
                # Remove case-variant duplicates
                canonical_tokens = set(self._significant_tokens(main_company))
                if canonical_tokens:
                    result = session.run(
                        "MATCH (c:Company) WHERE c.name <> $name RETURN c.name AS name",
                        {"name": main_company}
                    )
                    for record in result:
                        variant = record["name"]
                        if canonical_tokens == set(self._significant_tokens(variant)):
                            session.run(
                                "MATCH (c:Company {name: $variant}) DETACH DELETE c",
                                {"variant": variant}
                            )
                            print(f"  ✓ Removed case-variant duplicate: '{variant}'")
                
                # Check if Company node exists and convert it to TargetCompany
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
                    # Create new TargetCompany node
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

    # ========================================================================
    # SIC LOOKUP (fallback for items where sic was not resolved during extraction)
    # ========================================================================
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

    # ========================================================================
    # GRAPH OPERATIONS
    # ========================================================================
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

        query = f"""
        MATCH (s:{source_type} {{name: $source_name}})
        MATCH (t:{target_type} {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET {on_create}
        ON MATCH SET {on_match}
        RETURN r
        """
        session.run(query, {"source_name": source_name, "target_name": target_name, **props})

    # ========================================================================
    # BUILD FROM JSON
    # ========================================================================
    def build_from_json(self, json_file: str, clear: bool = False):
        """Read extracted JSON and write all entities/relationships to Neo4j."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_company = data.get("main_company", self.main_company)
        self.main_company = main_company
        self.stamp_target_company(main_company)

        if clear:
            self.clear_database()

        relations = data.get("relations", {})
        total_written = 0

        for relation_name, items in relations.items():
            if not items:
                continue
            print(f"\n{'='*60}")
            print(f"Building Neo4j graph for: {relation_name} ({len(items)} items)")
            print(f"{'='*60}")

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
                    # Fallback: resolve SIC now if the extractor didn't store it
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

    # ========================================================================
    # STATS
    # ========================================================================
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
    parser = argparse.ArgumentParser(description="Neo4j Builder — loads extracted JSON into Neo4j")
    parser.add_argument("json_file", help="Path to the extracted JSON file produced by llm_extractor.py")
    parser.add_argument("--clear", action="store_true", help="Clear database before loading")

    args = parser.parse_args()

    builder = Neo4jBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "Lexical12345*"),
    )

    try:
        builder.build_from_json(args.json_file, clear=args.clear)
        builder.show_graph_stats()
    finally:
        builder.close()


if __name__ == "__main__":
    main()
