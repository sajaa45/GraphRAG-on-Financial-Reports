
"""
Orchestrator: runs LLMExtractor (→ JSON) then Neo4jBuilder (JSON → Neo4j).

Can also be used as a library:
    from multi_relation_kg_builder import MultiRelationKGBuilder
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from relation_extraction_config import list_available_relations
from llm_extractor import LLMExtractor
from neo4j_builder import Neo4jBuilder


class MultiRelationKGBuilder:
    """Convenience wrapper that chains LLMExtractor and Neo4jBuilder."""

    def __init__(self,
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 parsed_sections_file: str,
                 output_dir: str = "/app/output",
                 main_company: str = "the Company",
                 source_file: str = ""):

        self.extractor = LLMExtractor(
            parsed_sections_file=parsed_sections_file,
            output_dir=output_dir,
            main_company=main_company,
            source_file=source_file,
        )

        self.builder = Neo4jBuilder(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )

    def clear_database(self):
        self.builder.clear_database()

    def extract_multiple_relations(self, relation_names):
        for json_path in self.extractor.extract_multiple_relations(relation_names):
            self.builder.build_from_json(json_path)

    def show_graph_stats(self):
        self.builder.show_graph_stats()

    def close(self):
        self.extractor.close()
        self.builder.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Relation Knowledge Graph Builder")
    parser.add_argument("relations", nargs="*", help="Relations to extract")
    parser.add_argument("--all", action="store_true", help="Extract all relations")
    parser.add_argument("--list", action="store_true", help="List available relations")
    parser.add_argument("--clear", action="store_true", help="Clear database first")
    parser.add_argument("--parsed-sections-file", default=None, help="Path to parsed_sections_html.json")
    parser.add_argument("--main-company", default=None, help="Main company name")
    parser.add_argument("--source-file", default="", help="Source document filename used to name output files")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")

    args = parser.parse_args()

    if args.list:
        print("Available relations:", list_available_relations())
        return

    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR", default_output)

    default_sections = os.path.join(default_output, "parsed_sections_html.json")
    parsed_sections_file = (
        args.parsed_sections_file
        or os.getenv("PARSED_SECTIONS_FILE", default_sections)
    )

    relations = list_available_relations() if args.all else [r.upper() for r in args.relations]

    kg = MultiRelationKGBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "Lexical12345*"),
        parsed_sections_file=parsed_sections_file,
        output_dir=output_dir,
        main_company=args.main_company or os.getenv("MAIN_COMPANY", "the Company"),
        source_file=args.source_file or os.getenv("SOURCE_FILE", ""),
    )

    try:
        if args.clear:
            kg.clear_database()
        kg.extract_multiple_relations(relations)
        kg.show_graph_stats()
    finally:
        kg.close()


if __name__ == "__main__":
    main()
