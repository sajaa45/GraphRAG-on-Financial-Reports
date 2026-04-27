#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Builder for Document Chunks
Creates nodes for Pages, Sections, and Chunks with their relationships
"""

import json
import os
import sys
from typing import Dict, List, Any
from neo4j import GraphDatabase
import hashlib

class DocumentKnowledgeGraph:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_constraints(self):
        """Create unique constraints for better performance"""
        constraints = [
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE", 
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
    
    def create_page_nodes(self, original_pages: Dict[str, str]):
        """
        Create Page nodes from original JSON pages
        
        Args:
            original_pages: Dictionary with page numbers as keys and text as values
        """
        print(f"Creating {len(original_pages)} page nodes...")
        
        with self.driver.session() as session:
            for page_num, page_text in original_pages.items():
                # Create a hash of the content for deduplication
                content_hash = hashlib.md5(page_text.encode()).hexdigest()[:12]
                
                session.run("""
                    MERGE (p:Page {page_id: $page_id})
                    SET p.page_number = $page_number,
                        p.content = $content,
                        p.length = $length,
                        p.content_hash = $content_hash,
                        p.created_at = datetime()
                """, 
                page_id=f"page_{page_num}",
                page_number=int(page_num),
                content=page_text,
                length=len(page_text),
                content_hash=content_hash
                )
        
        print(f"Created {len(original_pages)} page nodes")
    
    def create_section_nodes(self, sections: List[Dict]):
        """
        Create Section nodes and relationships to Pages
        
        Args:
            sections: List of section dictionaries from sections JSON
        """
        print(f"Creating {len(sections)} section nodes...")
        
        with self.driver.session() as session:
            for section in sections:
                # Create section node
                session.run("""
                    MERGE (s:Section {section_id: $section_id})
                    SET s.content = $content,
                        s.length = $length,
                        s.page_range = $page_range,
                        s.page_count = $page_count,
                        s.created_at = datetime()
                """,
                section_id=f"section_{section['section_id']}",
                content=section['text'],
                length=section['length'],
                page_range=section['page_range'],
                page_count=len(section['pages'])
                )
                
                # Create relationships to pages
                for page_num in section['pages']:
                    session.run("""
                        MATCH (p:Page {page_id: $page_id})
                        MATCH (s:Section {section_id: $section_id})
                        MERGE (p)-[:BELONGS_TO_SECTION]->(s)
                    """,
                    page_id=f"page_{page_num}",
                    section_id=f"section_{section['section_id']}"
                    )
        
        print(f"Created {len(sections)} section nodes with page relationships")
    
    def create_chunk_nodes(self, chunks: List[Dict]):
        """
        Create Chunk nodes and relationships to Sections
        
        Args:
            chunks: List of chunk dictionaries from LlamaIndex JSON
        """
        print(f"Creating {len(chunks)} chunk nodes...")
        
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks, 1):
                # Create unique chunk ID
                chunk_id = f"chunk_{i:04d}"
                
                # Create chunk node
                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content,
                        c.length = $length,
                        c.method = $method,
                        c.chunk_index_in_section = $chunk_index,
                        c.total_chunks_in_section = $total_chunks,
                        c.page_range = $page_range,
                        c.source_pages = $source_pages,
                        c.embedding = $embedding,
                        c.created_at = datetime()
                """,
                chunk_id=chunk_id,
                content=chunk['text'],
                length=chunk['length'],
                method=chunk['method'],
                chunk_index=chunk['chunk_index_in_section'],
                total_chunks=chunk['total_chunks_in_section'],
                page_range=chunk['page_range'],
                source_pages=chunk['source_pages'],
                embedding=chunk.get('embedding', [])
                )
                
                # Create relationship to section
                session.run("""
                    MATCH (s:Section {section_id: $section_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (s)-[:CONTAINS_CHUNK]->(c)
                """,
                section_id=f"section_{chunk['section_id']}",
                chunk_id=chunk_id
                )
                
                # Create relationships to source pages
                for page_num in chunk['source_pages']:
                    session.run("""
                        MATCH (p:Page {page_id: $page_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    page_id=f"page_{page_num}",
                    chunk_id=chunk_id
                    )
        
        print(f"Created {len(chunks)} chunk nodes with relationships")
    
    def create_semantic_relationships(self):
        """Create additional semantic relationships based on content similarity"""
        print("Creating semantic relationships...")
        
        with self.driver.session() as session:
            # Connect consecutive chunks within the same section (NEXT_CHUNK)
            session.run("""
                MATCH (s:Section)-[:CONTAINS_CHUNK]->(c1:Chunk)
                MATCH (s)-[:CONTAINS_CHUNK]->(c2:Chunk)
                WHERE c1.chunk_index_in_section = c2.chunk_index_in_section - 1
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
            """)
            
            # Connect consecutive chunks within the same section (PREVIOUS_CHUNK)
            session.run("""
                MATCH (s:Section)-[:CONTAINS_CHUNK]->(c1:Chunk)
                MATCH (s)-[:CONTAINS_CHUNK]->(c2:Chunk)
                WHERE c2.chunk_index_in_section = c1.chunk_index_in_section - 1
                MERGE (c1)-[:PREVIOUS_CHUNK]->(c2)
            """)
            
            # Connect sections that share pages (multi-page sections)
            session.run("""
                MATCH (p:Page)-[:BELONGS_TO_SECTION]->(s1:Section)
                MATCH (p)-[:BELONGS_TO_SECTION]->(s2:Section)
                WHERE s1 <> s2
                MERGE (s1)-[:SHARES_PAGE]->(s2)
            """)
            
        print("Created semantic relationships (NEXT_CHUNK, PREVIOUS_CHUNK, SHARES_PAGE)")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the created graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Node counts
            result = session.run("MATCH (p:Page) RETURN count(p) as count")
            stats['pages'] = result.single()['count']
            
            result = session.run("MATCH (s:Section) RETURN count(s) as count")
            stats['sections'] = result.single()['count']
            
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            stats['chunks'] = result.single()['count']
            
            # Relationship counts
            result = session.run("MATCH ()-[r:BELONGS_TO_SECTION]->() RETURN count(r) as count")
            stats['page_to_section_rels'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:CONTAINS_CHUNK]->() RETURN count(r) as count")
            stats['section_to_chunk_rels'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) as count")
            stats['page_to_chunk_rels'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) as count")
            stats['next_chunk_rels'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:PREVIOUS_CHUNK]->() RETURN count(r) as count")
            stats['previous_chunk_rels'] = result.single()['count']
            
            result = session.run("MATCH ()-[r:SHARES_PAGE]->() RETURN count(r) as count")
            stats['shares_page_rels'] = result.single()['count']
            
            return stats

def load_json_file(file_path: str) -> Dict:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def main():
    """Main function to build the knowledge graph"""
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # File paths
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    input_dir = "/app/input" if os.path.exists("/app/input") else "."
    
    # Try to find chunks file (support both old and new naming)
    chunks_file_options = [
        os.path.join(output_dir, "SemanticSplitterNodeParser_chunks.json"),
        os.path.join(output_dir, "llamaindex_chunks_with_pages.json"),
    ]
    
    chunks_file = None
    for file_path in chunks_file_options:
        if os.path.exists(file_path):
            chunks_file = file_path
            break
    
    # Try to find sections file (support multiple naming patterns)
    sections_file_options = [
        os.path.join(output_dir, "saudi-aramco-ara-2024-english_sections.json"),
        os.path.join(output_dir, "pdf2_sections.json"),
    ]
    
    sections_file = None
    for file_path in sections_file_options:
        if os.path.exists(file_path):
            sections_file = file_path
            break
    
    # Try to find original JSON file
    original_pages_file_options = [
        os.path.join(input_dir, "saudi-aramco-ara-2024-english.json"),
        os.path.join(input_dir, "pdf2.json"),
    ]
    
    original_pages_file = None
    for file_path in original_pages_file_options:
        if os.path.exists(file_path):
            original_pages_file = file_path
            break
    
    # Check if files exist
    if not chunks_file:
        print(f"Chunks file not found in: {output_dir}")
        print("Tried:")
        for file_path in chunks_file_options:
            print(f"  - {file_path}")
        print("\nPlease run chunking.py first to generate chunk files")
        return
    
    if not sections_file:
        print(f"Sections file not found in: {output_dir}")
        print("Tried:")
        for file_path in sections_file_options:
            print(f"  - {file_path}")
        print("\nPlease run sections_merging_pages.py first to generate section files")
        return
    
    if not original_pages_file:
        print(f"Original pages file not found in: {input_dir}")
        print("Tried:")
        for file_path in original_pages_file_options:
            print(f"  - {file_path}")
        return
    
    print(f"Using files:")
    print(f"  Chunks: {chunks_file}")
    print(f"  Sections: {sections_file}")
    print(f"  Original: {original_pages_file}")
    
    print("Building Neo4j Knowledge Graph for Document Structure")
    print("=" * 60)
    
    # Load data
    print("Loading data files...")
    chunks_data = load_json_file(chunks_file)
    sections_data = load_json_file(sections_file)
    original_pages = load_json_file(original_pages_file)
    
    if not chunks_data or not sections_data or not original_pages:
        print("Failed to load required data files")
        return
    
    chunks = chunks_data.get('chunks', [])
    sections = sections_data.get('sections', [])
    
    print(f"Loaded:")
    print(f"  - {len(original_pages)} original pages")
    print(f"  - {len(sections)} sections")
    print(f"  - {len(chunks)} chunks")
    
    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            kg = DocumentKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            
            # Test connection
            with kg.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                print(result.single()['message'])
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to Neo4j after {max_retries} attempts: {e}")
                print("Make sure Neo4j is running and credentials are correct")
                print("You can set environment variables:")
                print("  NEO4J_URI=bolt://localhost:7687")
                print("  NEO4J_USERNAME=neo4j")
                print("  NEO4J_PASSWORD=your_password")
                return
    
    try:
        # Ask user if they want to clear the database (skip in Docker)
        clear_db = os.getenv("CLEAR_DB", "n").lower().strip()
        if clear_db == "" and sys.stdin.isatty():  # Only ask if running interactively
            clear_db = input("\nClear existing graph data? (y/N): ").lower().strip()
        
        if clear_db == 'y':
            kg.clear_database()
        else:
            print("\nSkipping database clear (keeping existing data)")
        
        # Create constraints
        print("\nCreating database constraints...")
        kg.create_constraints()
        
        # Build the graph
        print("\nBuilding knowledge graph...")
        
        # Step 1: Create page nodes
        kg.create_page_nodes(original_pages)
        
        # Step 2: Create section nodes and page->section relationships
        kg.create_section_nodes(sections)
        
        # Step 3: Create chunk nodes and section->chunk relationships
        kg.create_chunk_nodes(chunks)
        
        # Step 4: Create semantic relationships
        kg.create_semantic_relationships()
        
        # Get statistics
        print("\nGraph Statistics:")
        print("=" * 40)
        stats = kg.get_graph_statistics()
        
        print(f"Nodes:")
        print(f"  Pages: {stats['pages']}")
        print(f"  Sections: {stats['sections']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Total: {stats['pages'] + stats['sections'] + stats['chunks']}")
        
        print(f"\nRelationships:")
        print(f"  Page -> Section: {stats['page_to_section_rels']}")
        print(f"  Section -> Chunk: {stats['section_to_chunk_rels']}")
        print(f"  Page -> Chunk: {stats['page_to_chunk_rels']}")
        print(f"  Next Chunk: {stats['next_chunk_rels']}")
        print(f"  Shares Page: {stats['shares_page_rels']}")
        print(f"  Total: {sum(stats[k] for k in stats if k.endswith('_rels'))}")
        
        print("\nKnowledge graph created successfully!")
        print("\nSample Cypher queries to explore your graph:")
        print("=" * 50)
        
        print("\n1. Find all chunks from a specific page:")
        print("   MATCH (p:Page {page_number: 1})-[:HAS_CHUNK]->(c:Chunk)")
        print("   RETURN p.page_number, c.content LIMIT 5")
        
        print("\n2. Find the path from page to chunks through sections:")
        print("   MATCH path = (p:Page)-[:BELONGS_TO_SECTION]->(s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)")
        print("   WHERE p.page_number = 3")
        print("   RETURN path LIMIT 3")
        
        print("\n3. Find sections with multiple pages:")
        print("   MATCH (s:Section)")
        print("   WHERE s.page_count > 1")
        print("   RETURN s.section_id, s.page_range, s.page_count")
        
        print("\n4. Find consecutive chunks in a section:")
        print("   MATCH (c1:Chunk)-[:NEXT_CHUNK]->(c2:Chunk)")
        print("   RETURN c1.chunk_id, c2.chunk_id, c1.content[..50] + '...' as preview")
        print("   LIMIT 5")
        
        print("\n5. Get section statistics:")
        print("   MATCH (s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)")
        print("   RETURN s.section_id, s.page_range, count(c) as chunk_count,")
        print("          avg(c.length) as avg_chunk_length")
        print("   ORDER BY chunk_count DESC LIMIT 10")
        
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        
    finally:
        kg.close()
        print("\nConnection closed")

if __name__ == "__main__":
    main()