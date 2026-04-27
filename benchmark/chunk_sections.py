#!/usr/bin/env python3
"""
Chunk sections with page and hierarchy tracking
Processes hierarchical sections and creates chunks with metadata about their location
"""

import json
import argparse
from typing import List, Dict, Any
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer


def get_section_path(section, parent_path=""):
    """
    Get the full hierarchical path of a section
    """
    current_path = f"{parent_path} > {section['title']}" if parent_path else section['title']
    return current_path


def chunk_page(page_text, page_num, section_path, chunker, chunk_id_start):
    """
    Chunk a single page and add metadata
    """
    chunks = []
    
    if not page_text or not page_text.strip():
        return chunks, chunk_id_start
    
    # Chunk the page text
    page_chunks = chunker.chunk(page_text)
    
    for chunk in page_chunks:
        chunks.append({
            "chunk_id": chunk_id_start,
            "text": chunk.text,
            "page": page_num,
            "section_path": section_path,
            "char_count": len(chunk.text),
            "token_count": chunk.token_count if hasattr(chunk, 'token_count') else None
        })
        chunk_id_start += 1
    
    return chunks, chunk_id_start


def process_section(section, chunker, chunk_id_start, parent_path=""):
    """
    Recursively process a section and its subsections
    """
    all_chunks = []
    
    # Get current section path
    section_path = get_section_path(section, parent_path)
    
    print(f"Processing: {section_path} (pages {section['start_page']}-{section['end_page']})")
    
    # Split section text by pages (assuming text has page markers or we process page by page)
    # For now, we'll chunk the entire section text and assign to the page range
    section_text = section.get('text', '')
    
    if section_text and section_text.strip():
        # Chunk the section
        section_chunks = chunker.chunk(section_text)
        
        # Calculate pages per chunk (distribute chunks across page range)
        total_pages = section['end_page'] - section['start_page'] + 1
        chunks_per_page = len(section_chunks) / total_pages if total_pages > 0 else len(section_chunks)
        
        for i, chunk in enumerate(section_chunks):
            # Estimate which page this chunk belongs to
            estimated_page = section['start_page'] + int(i / chunks_per_page) if chunks_per_page > 0 else section['start_page']
            estimated_page = min(estimated_page, section['end_page'])
            
            all_chunks.append({
                "chunk_id": chunk_id_start,
                "text": chunk.text,
                "page": estimated_page,
                "section_path": section_path,
                "section_level": section['level'],
                "char_count": len(chunk.text),
                "token_count": chunk.token_count if hasattr(chunk, 'token_count') else None
            })
            chunk_id_start += 1
    
    # Process subsections recursively
    if 'subsections' in section and section['subsections']:
        for subsection in section['subsections']:
            subsection_chunks, chunk_id_start = process_section(
                subsection, chunker, chunk_id_start, section_path
            )
            all_chunks.extend(subsection_chunks)
    
    return all_chunks, chunk_id_start


def chunk_sections(sections_file, output_file, chunk_size=512, similarity_threshold=0.5):
    """
    Process sections file and create chunks with metadata
    """
    print(f"Loading sections from: {sections_file}")
    
    with open(sections_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data.get('sections', [])
    
    if not sections:
        print("No sections found in file")
        return
    
    print(f"Found {len(sections)} top-level sections")
    print(f"Initializing chunker (size={chunk_size}, threshold={similarity_threshold})")
    
    # Initialize chunker
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        similarity_threshold=similarity_threshold
    )
    
    # Process all sections
    all_chunks = []
    chunk_id = 1
    
    for section in sections:
        section_chunks, chunk_id = process_section(section, chunker, chunk_id)
        all_chunks.extend(section_chunks)
    
    # Create output
    result = {
        "source_file": sections_file,
        "total_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "similarity_threshold": similarity_threshold,
        "chunks": all_chunks
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created {len(all_chunks)} chunks")
    print(f"Output saved to: {output_file}")
    
    # Print statistics
    pages_with_chunks = set(chunk['page'] for chunk in all_chunks)
    sections_with_chunks = set(chunk['section_path'] for chunk in all_chunks)
    
    print(f"\nStatistics:")
    print(f"  Pages with chunks: {len(pages_with_chunks)}")
    print(f"  Unique section paths: {len(sections_with_chunks)}")
    print(f"  Avg chunks per page: {len(all_chunks) / len(pages_with_chunks):.1f}")
    
    # Show sample chunks
    print(f"\nSample chunks:")
    for chunk in all_chunks[:3]:
        print(f"  Chunk {chunk['chunk_id']} - Page {chunk['page']}")
        print(f"    Section: {chunk['section_path']}")
        print(f"    Text: {chunk['text'][:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Chunk hierarchical sections with page and section tracking"
    )
    parser.add_argument(
        "sections_file",
        help="Path to sections JSON file (from sections_parser_pdf.py)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default="output/chunked_sections.json"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for semantic chunking (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    chunk_sections(
        args.sections_file,
        args.output,
        args.chunk_size,
        args.threshold
    )


if __name__ == "__main__":
    main()
