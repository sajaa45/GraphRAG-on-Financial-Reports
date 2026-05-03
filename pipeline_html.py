#!/usr/bin/env python
"""
Unified pipeline for processing HTML files:
1. Parse HTML and extract sections with logical page numbers
2. Chunk sections using semantic splitter
3. Store in Qdrant vector database

Usage:
    python pipeline_html.py input/NYSE_MTX_2024.htm
"""

import argparse
import asyncio
import sys
import os

# Add parsing directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parsing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chunking'))

from parsing_sections_html import sections_parser_html, generate_page_index_from_html
from pipeline import UnifiedPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Process HTML file: parse sections, chunk, and store in vector DB"
    )
    parser.add_argument(
        "html_path",
        nargs='?',
        default="input/NYSE_MTX_2024.htm",
        help="Path to HTML file (default: input/NYSE_MTX_2024.htm)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        default="output",
        help="Output directory for intermediate files (default: output)"
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_docs",
        help="Qdrant collection name (default: financial_docs)"
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1,
        help="LlamaIndex buffer size (default: 1)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=70,
        help="LlamaIndex threshold (default: 70)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete and recreate the Qdrant collection"
    )
    parser.add_argument(
        "--no-chunk-files",
        action="store_true",
        help="Skip saving chunk JSON/TXT files"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    sections_file = os.path.join(args.output_dir, "parsed_sections_html.json")
    page_index_file = os.path.join(args.output_dir, "page_index_html.json")
    
    print("=" * 70)
    print("HTML PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input HTML: {args.html_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Qdrant collection: {args.collection}")
    print("=" * 70)
    print()
    
    # Step 1: Parse HTML and extract sections
    print("STEP 1: Parsing HTML and extracting sections...")
    print("-" * 70)
    
    # Generate page index
    page_index = generate_page_index_from_html(args.html_path, page_index_file)
    if not page_index:
        print("✗ Failed to generate page index")
        return 1
    
    # Parse sections
    result = sections_parser_html(args.html_path, sections_file)
    if not result:
        print("✗ Failed to parse sections")
        return 1
    
    print(f"\n✓ Parsed {result['num_sections']} sections from {result['num_pages']} logical pages")
    print()
    
    # Step 2: Chunk and store in vector database
    print("STEP 2: Chunking sections and storing in Qdrant...")
    print("-" * 70)
    
    async def run_pipeline():
        pipeline = UnifiedPipeline(
            collection_name=args.collection,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            buffer_size=args.buffer_size,
            threshold=args.threshold,
            clear=args.clear,
            workers=args.workers,
        )
        await pipeline.process_sections_file(
            sections_file,
            page_index_file=page_index_file,
            save_chunks=not args.no_chunk_files,
        )
    
    asyncio.run(run_pipeline())
    
    print()
    print("=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Sections file: {sections_file}")
    print(f"Page index file: {page_index_file}")
    print(f"Qdrant collection: {args.collection}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
