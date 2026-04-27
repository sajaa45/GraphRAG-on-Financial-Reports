#!/usr/bin/env python3
"""
Simple script to build knowledge graph locally
Run this after you have Neo4j running locally
"""

import os
import sys

def main():
    print("Knowledge Graph Builder")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        ("output/SemanticSplitterNodeParser_chunks.json", "output/llamaindex_chunks_with_pages.json"),
        ("output/saudi-aramco-ara-2024-english_sections.json", "output/pdf2_sections.json"),
        ("saudi-aramco-ara-2024-english.json", "pdf2.json")
    ]
    
    missing_files = []
    found_files = []
    
    for file_options in required_files:
        found = False
        for file_path in (file_options if isinstance(file_options, tuple) else (file_options,)):
            if os.path.exists(file_path):
                found_files.append(file_path)
                found = True
                break
        
        if not found:
            missing_files.append(file_options[0] if isinstance(file_options, tuple) else file_options)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run:")
        print("  1. sections_merging_pages.py (to generate sections)")
        print("  2. chunking.py (to generate chunks with embeddings)")
        return
    
    print("All required files found:")
    for file_path in found_files:
        print(f"  ✓ {file_path}")
    
    # Set default Neo4j connection if not set
    if not os.getenv("NEO4J_URI"):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    if not os.getenv("NEO4J_USERNAME"):
        os.environ["NEO4J_USERNAME"] = "neo4j"
    if not os.getenv("NEO4J_PASSWORD"):
        print("\nNeo4j password not set. Please set NEO4J_PASSWORD environment variable")
        print("Or run with: NEO4J_PASSWORD=your_password python lexical_wrapper_kg.py")
        return
    
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print(f"Neo4j Username: {os.getenv('NEO4J_USERNAME')}")
    
    # Import and run the knowledge graph builder
    try:
        from lexical_kG_building import main as build_kg
        build_kg()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install neo4j: pip install neo4j")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()