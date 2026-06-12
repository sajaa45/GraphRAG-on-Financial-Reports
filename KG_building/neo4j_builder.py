
import os
import sys
import json
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from neo4j import GraphDatabase


class Neo4jBuilder:
    """Reads extracted-entity JSON produced by LLMExtractor and writes the
    corresponding nodes and relationships into Neo4j."""

    def __init__(self,neo4j_uri: str = "",neo4j_user: str = "",neo4j_password: str = "",main_company: str = "",driver=None):

        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            print(f"✓ Connected to Neo4j at {neo4j_uri}")

        self.main_company = main_company

        # Citation ID counters for target company
        self._target_metric_counter = 0
        self._target_risk_counter = 0

    def _set_main_company(self, name: str):
        if not name or name.lower() in ('the company', 'this company'):
            raise ValueError(f"Company name could not be extracted (got: {name!r}). Aborting build.")
        self.main_company = name

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

    def create_node(self, session, node_type: str, name: str, properties: Dict = None):
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
                            target_name, rel_type, properties=None,
                            section_title=None, source_page=None):
        props = dict(properties or {})
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

        for relation_name, items in relations.items():
            if not items:
                continue

            print(f"\n{'='*60}")
            print(f"Building Neo4j graph for: {relation_name} ({len(items)} items)")
            print(f"{'='*60}")

            if relation_name == "FACES_RISK":
                with self.driver.session() as _cleanup_session:
                    _cleanup_session.run("MATCH (r:Risk) DETACH DELETE r")

            if relation_name == "HAS_METRIC":
                with self.driver.session() as _cleanup_session:
                    _cleanup_session.run("MATCH (mc:MetricCategory) DETACH DELETE mc")
                    _cleanup_session.run(
                        """
                        MATCH (m:Metric)
                        WITH m.name AS nm, collect(m) AS dupes
                        WHERE size(dupes) > 1
                        FOREACH (d IN tail(dupes) | DETACH DELETE d)
                        """
                    )

            def _write_batch(tx, batch=items):
                for item in batch:
                    src_props = dict(item['src'].get('properties', {}))
                    self.create_node(tx, item['src']['type'], item['src']['name'], src_props)

                    tgt_props = dict(item['tgt'].get('properties', {}))

                    if item.get('rel') == 'HAS_METRIC':
                        self._target_metric_counter += 1
                        tgt_props['citation_id'] = f"TARGET_M{self._target_metric_counter:04d}"

                        if not tgt_props.get('gaap_concept'):
                            tgt_props['gaap_concept'] = (
                                tgt_props.get('metric_type')
                                or item['tgt'].get('name', '')
                            )

                        if item.get('source_page') is not None:
                            tgt_props['source_page'] = item['source_page']
                        if item.get('section_title'):
                            tgt_props['section_title'] = item['section_title']
                        if item.get('chunk_text'):
                            tgt_props['source_text'] = item['chunk_text']

                        # Structure: TargetCompany -[HAS_METRIC_CATEGORY]-> MetricCategory -[HAS_METRIC]-> Metric
                        _raw_cat = (tgt_props.get('category') or '').strip()
                        _KNOWN = {'Leverage', 'Coverage', 'Liquidity', 'Profitability', 'Debt Structure'}
                        category = _raw_cat if _raw_cat in _KNOWN else 'Other'

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
                            item.get('section_title'), item.get('source_page'),
                        )
                    elif item.get('rel') == 'FACES_RISK':
                        self._target_risk_counter += 1
                        tgt_props['citation_id'] = f"TARGET_R{self._target_risk_counter:04d}"

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
                            item.get('section_title'), item.get('source_page'),
                        )
                    else:
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
                            item.get('section_title'), item.get('source_page'),
                        )

                    sic = item.get('sic')
                    if sic:
                        self.create_node(tx, 'SICCode', sic['code'],
                                         {'code': sic['code'], 'industry': sic['industry'], 'sector': sic['sector']})
                        self.create_relationship(tx, sic['src_type'], sic['src_name'], 'SICCode', sic['code'], 'HAS_SIC_CODE')

            with self.driver.session() as session:
                session.execute_write(_write_batch)

            total_written += len(items)
            print(f"  ✓ Written {len(items)} items for {relation_name}")

        print(f"\n✓ Total items written to Neo4j: {total_written}")
        return total_written

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def build_from_validated_dirs(self, relations_dir: str, clear: bool = False):
        """
        Scan each subdirectory of relations_dir for a extracted_*.json file,
        merge all relations across files, and write to Neo4j.
        """
        if not os.path.isdir(relations_dir):
            raise FileNotFoundError(f"Relations directory not found: {relations_dir}")

        combined_relations: Dict[str, List] = {}
        main_company = self.main_company

        for entry in sorted(os.listdir(relations_dir)):
            if entry.startswith('_'):
                continue
            sub_dir = os.path.join(relations_dir, entry)
            if not os.path.isdir(sub_dir):
                continue

            validated_files = sorted(
                f for f in os.listdir(sub_dir)
                if f.startswith('extracted_') and f.endswith('.json')
            )

            if not validated_files:
                print(f"  ⚠ No extracted_*.json found in {sub_dir} — skipping")
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

        self._set_main_company(main_company)

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
