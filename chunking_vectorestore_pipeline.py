
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import os

# Import chunking functions from chunking.py
from chunking import llamaindex_chunker, get_embedding_model

# Vector store
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)


class UnifiedPipeline:
    """Unified pipeline that chunks and stores simultaneously"""
    
    def __init__(self,
                 collection_name: str = "financial_docs",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 buffer_size: int = 1,
                 threshold: int = 70,
                 clear: bool = False):
        """
        Initialize unified pipeline
        
        Args:
            collection_name: Qdrant collection name
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            buffer_size: LlamaIndex buffer size
            threshold: LlamaIndex threshold
        """
        print("Initializing Unified Pipeline...")
        print(f"  Collection: {collection_name}")
        print(f"  Qdrant: {qdrant_host}:{qdrant_port}")
        
        # Store chunking parameters
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        # Initialize embedding model (from chunking.py)
        print("  Loading embedding model...")
        self.embed_model = get_embedding_model()
        if self.embed_model:
            print("  ✓ Embedding model loaded")
        else:
            print("  ✗ Failed to load embedding model")
            raise Exception("Embedding model required for pipeline")
        
        # Determine vector size from a test encode
        sample_vec = self.embed_model.encode(["test"])[0]
        self.vector_size = len(sample_vec)

        # Initialize Qdrant
        print("  Connecting to Qdrant...")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        # Get or create collection
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing:
            if clear:
                self.client.delete_collection(collection_name)
                print(f"  ✓ Cleared existing collection: {collection_name}")
            else:
                print(f"  ✓ Using existing collection: {collection_name}")
        if collection_name not in existing or clear:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"  ✓ Created new collection: {collection_name}")
        
        print("✓ Pipeline initialized\n")
    
    def process_section(self, section: Dict, section_idx: int, total_sections: int):
        """
        Process a single section: chunk its text and store in Qdrant immediately.
        Also stores a section-level embedding for hierarchical search.
        Sections with explicit page_contents are chunked page-by-page for exact page tracking;
        all others (including those whose text was pre-populated from a pages JSON) are
        chunked as a single unit.
        """
        section_title = section.get('title', 'Untitled')
        section_text = section.get('text', '')

        print(f"[{section_idx}/{total_sections}] Processing: {section_title}")

        has_page_contents = bool(section.get('page_contents'))
        use_page_by_page = has_page_contents

        if not use_page_by_page and not section_text.strip():
            print(f"  ⚠ Empty section, skipping")
            return 0

        # Store section-level embedding for hierarchical search
        if section_text and len(section_text) > 50:
            section_embedding = self.embed_model.encode([section_title])[0].tolist()
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=section_idx,
                    vector=section_embedding,
                    payload={
                        "type": "section",
                        "section_id": section_idx,
                        "title": section_title,
                        "text": section_text[:1000],
                        "level": section.get('level', 0),
                        "start_page": section.get('start_page', 0),
                        "end_page": section.get('end_page', 0),
                        "char_count": len(section_text)
                    }
                )]
            )

        chunk_texts = []
        chunk_ids = []
        chunk_metadatas = []
        chunk_embeddings = []
        total_filtered = 0

        if use_page_by_page:
            print(f"  Processing {len(section['page_contents'])} pages individually...")
            for chunk_counter, page_content in enumerate(section['page_contents'], 1):
                page_num = page_content['page_number']
                page_text = page_content.get('content', '')
                if not page_text.strip():
                    continue

                filtered_chunks, filtering_stats = llamaindex_chunker(
                    text=page_text,
                    buffer_size=self.buffer_size,
                    threshold=self.threshold,
                    embed_model=self.embed_model,
                    debug=False
                )
                total_filtered += sum(filtering_stats.values())

                for i, chunk_data in enumerate(filtered_chunks, 1):
                    chunk_id = f"section_{section_idx}_chunk_{chunk_counter}_{i}"
                    chunk_texts.append(chunk_data['text'])
                    chunk_ids.append(chunk_id)
                    chunk_metadatas.append({
                        "type": "chunk",
                        "section_id": section_idx,
                        "section_title": section_title,
                        "section_level": section.get('level', 0),
                        "source_page": page_num,
                        "start_page": section.get('start_page', 0),
                        "end_page": section.get('end_page', 0),
                        "chunk_index": chunk_counter,
                        "chunk_index_in_page": i,
                        "total_chunks_in_page": len(filtered_chunks),
                        "char_count": len(chunk_data['text'])
                    })
                    chunk_embeddings.append(chunk_data['embedding'])

            if total_filtered > 0:
                print(f"  Filtered: {total_filtered} chunks across all pages")
        else:
            filtered_chunks, filtering_stats = llamaindex_chunker(
                text=section_text,
                buffer_size=self.buffer_size,
                threshold=self.threshold,
                embed_model=self.embed_model,
                debug=False
            )

            if not filtered_chunks:
                print(f"  ⚠ No chunks after filtering")
                return 0

            total_filtered = sum(filtering_stats.values())
            if total_filtered > 0:
                print(f"  Filtered: {total_filtered} chunks (small:{filtering_stats.get('small',0)}, "
                      f"decorative:{filtering_stats.get('decorative',0)}, "
                      f"TOC:{filtering_stats.get('toc',0)}, "
                      f"meaningless:{filtering_stats.get('meaningless',0)}, "
                      f"repetitive:{filtering_stats.get('repetitive',0)})")

            for i, chunk_data in enumerate(filtered_chunks, 1):
                chunk_id = f"section_{section_idx}_chunk_{i}"
                chunk_texts.append(chunk_data['text'])
                chunk_ids.append(chunk_id)
                chunk_metadatas.append({
                    "type": "chunk",
                    "section_id": section_idx,
                    "section_title": section_title,
                    "section_level": section.get('level', 0),
                    "start_page": section.get('start_page', 0),
                    "end_page": section.get('end_page', 0),
                    "chunk_index": i,
                    "total_chunks": len(filtered_chunks),
                    "char_count": len(chunk_data['text'])
                })
                chunk_embeddings.append(chunk_data['embedding'])
        
        if not chunk_texts:
            print(f"  ⚠ No chunks after filtering")
            return 0
        
        # Store directly in Qdrant (no JSON intermediate)
        # Chunk IDs are strings like "section_1_chunk_3" — hash them to ints for Qdrant
        points = [
            PointStruct(
                id=abs(hash(chunk_ids[i])) % (2**63),
                vector=chunk_embeddings[i],
                payload={**chunk_metadatas[i], "text": chunk_texts[i]}
            )
            for i in range(len(chunk_texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        
        print(f"  ✓ Stored section embedding + {len(chunk_texts)} filtered chunks in vector DB")
        return len(chunk_texts)
    
    def process_sections_file(self, sections_file: str, save_chunks: bool = True):
        """
        Process entire sections file
        
        Args:
            sections_file: Path to sections JSON file
            save_metadata: Save lightweight metadata JSON (without embeddings)
            save_chunks: Save detailed chunk files (JSON and TXT without embeddings)
        """
        start_time = time.time()
        
        print("="*70)
        print("UNIFIED PIPELINE: CHUNK → VECTOR STORE")
        print("="*70)
        
        # Load sections
        print(f"\nLoading sections from: {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', [])
        if not sections:
            print("✗ No sections found!")
            return
        
        print(f"✓ Found {len(sections)} sections")
        
        # Check if sections have meaningful text (not just placeholders)
        has_text = any(section.get('text') and len(section.get('text', '')) > 100 for section in sections)
        
        # Try to load pages_data for page-by-page chunking
        pages_data = {}
        source_filename = data.get('filename', '')
        
        if source_filename:
            base_name = os.path.splitext(os.path.basename(source_filename))[0]
            
            # Try multiple locations
            possible_paths = [
                Path(sections_file).parent.parent / f"{base_name}.json",  # ../filename.json
                Path(sections_file).parent / f"{base_name}.json",  # output/filename.json
                Path("/app/input") / f"{base_name}.json",  # /app/input/filename.json (Docker)
                Path(".") / f"{base_name}.json",  # ./filename.json
            ]
            
            for path in possible_paths:
                if path.exists():
                    print(f"✓ Loading page data for page-by-page chunking from: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            pages_data = json.load(f)
                        print(f"✓ Loaded {len(pages_data)} pages")
                        break
                    except Exception as e:
                        print(f"⚠ Error loading pages: {e}")
        
        # If sections don't have meaningful text, extract it from pages_data
        if not has_text:
            print("⚠ Sections don't have text content")

            if not pages_data:
                print("✗ No pages_data available - cannot proceed")
                return

            print("✓ Extracting section text from pages data (start_page → end_page)...")
            sections = self._add_text_to_sections(sections, pages_data)
            populated = sum(1 for s in sections if len(s.get('text', '')) > 100)
            print(f"✓ Populated text for {populated}/{len(sections)} sections")
        
        print()
        
        # Process each section
        total_chunks = 0
        all_chunk_details = []

        for idx, section in enumerate(sections, 1):
            chunks_count = self.process_section(section, idx, len(sections))
            total_chunks += chunks_count
            
            # Collect detailed chunk info for chunk files (if enabled)
            if save_chunks and chunks_count > 0:
                # Get only the chunks we just stored from Qdrant
                section_chunks, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="type", match=MatchValue(value="chunk")),
                        FieldCondition(key="section_id", match=MatchValue(value=idx))
                    ]),
                    limit=chunks_count,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in section_chunks:
                    meta = point.payload
                    doc = meta.get("text", "")
                    chunk_detail = {
                        'text': doc,
                        'length': len(doc),
                        'section_id': meta['section_id'],
                        'section_title': meta['section_title'],
                        'section_level': meta['section_level'],
                        'chunk_index': meta['chunk_index'],
                        'method': 'SemanticSplitterNodeParser'
                    }
                    
                    if 'source_page' in meta:
                        chunk_detail['source_page'] = meta['source_page']
                        chunk_detail['section_page_range'] = meta.get('section_page_range', '')
                        chunk_detail['chunk_index_in_page'] = meta.get('chunk_index_in_page', 1)
                        chunk_detail['total_chunks_in_page'] = meta.get('total_chunks_in_page', 1)
                    else:
                        chunk_detail['start_page'] = meta['start_page']
                        chunk_detail['end_page'] = meta['end_page']
                        chunk_detail['total_chunks'] = meta.get('total_chunks', 0)
                    
                    all_chunk_details.append(chunk_detail)
        
        # Save detailed chunk files (JSON and TXT without embeddings)
        if save_chunks and all_chunk_details:
            output_dir = Path(sections_file).parent
            
            # Save JSON file (without embeddings)
            chunks_json_file = output_dir / "SemanticSplitterNodeParser_chunks.json"
            chunks_data = {
                "method": "SemanticSplitterNodeParser",
                "status": "completed",
                "total_sections": len(sections),
                "total_chunks": len(all_chunk_details),
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "chunks": all_chunk_details
            }
            
            with open(chunks_json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved detailed chunks JSON to: {chunks_json_file}")
            print(f"  (No embeddings - for human review)")
            
            # Save TXT file (human-readable)
            chunks_txt_file = output_dir / "SemanticSplitterNodeParser_chunks.txt"
            
            with open(chunks_txt_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SemanticSplitterNodeParser Chunks - Detailed View\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total sections: {len(sections)}\n")
                f.write(f"Total chunks: {len(all_chunk_details)}\n")
                f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Embeddings stored in: Qdrant vector store\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk in enumerate(all_chunk_details, 1):
                    f.write(f"CHUNK {i:04d}\n")
                    f.write(f"Length: {chunk['length']} characters\n")
                    f.write(f"Method: {chunk['method']}\n")
                    f.write(f"Section: {chunk['section_id']} - {chunk['section_title']}\n")
                    f.write(f"Section Level: {chunk['section_level']}\n")
                    
                    # Show page information (prefer source_page if available)
                    if 'source_page' in chunk:
                        f.write(f"Source Page: {chunk['source_page']}\n")
                        f.write(f"Section Pages: {chunk.get('section_page_range', 'N/A')}\n")
                        f.write(f"Chunk {chunk.get('chunk_index_in_page', 1)} of {chunk.get('total_chunks_in_page', 1)} on this page\n")
                    else:
                        f.write(f"Pages: {chunk.get('start_page', 'N/A')}-{chunk.get('end_page', 'N/A')}\n")
                        f.write(f"Chunk {chunk['chunk_index']} of {chunk.get('total_chunks', 0)} in this section\n")
                    
                    f.write("-" * 40 + "\n")
                    f.write(chunk['text'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"✓ Saved detailed chunks TXT to: {chunks_txt_file}")
            print(f"  (Human-readable format)")
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"Sections processed: {len(sections)}")
        print(f"Chunks created: {total_chunks}")
        if total_chunks > 0:
            print(f"Processing speed: {total_chunks/total_time:.1f} chunks/sec")
        print(f"\n✓ All embeddings stored in Qdrant")
        print(f"✓ Detailed chunk files saved (without embeddings)")
        print(f"✓ Ready for fast semantic search!")
    
    def _add_text_to_sections(self, sections: List[Dict], pages_data: Dict) -> List[Dict]:
        """
        Recursively add text content to sections from pages data
        
        Args:
            sections: List of section dictionaries
            pages_data: Dictionary mapping page numbers to text
            
        Returns:
            Sections with text content added
        """
        for section in sections:
            section_text = ""
            page_contents = []
            for page_num in range(section['start_page'], section['end_page'] + 1):
                # pages_data keys may be strings or ints depending on the source
                page_text = pages_data.get(str(page_num)) or pages_data.get(page_num)
                if page_text:
                    section_text += page_text + "\n"
                    page_contents.append({'page_number': page_num, 'content': page_text})

            section['text'] = section_text
            if page_contents:
                section['page_contents'] = page_contents
            
            # Process subsections recursively
            if 'subsections' in section and section['subsections']:
                section['subsections'] = self._add_text_to_sections(
                    section['subsections'], 
                    pages_data
                )
        
        return sections


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Chunk and store in vector DB simultaneously"
    )
    parser.add_argument(
        "sections_file",
        help="Path to sections JSON file"
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
        "--no-chunk-files",
        action="store_true",
        help="Don't save detailed chunk JSON/TXT files"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete and recreate the Qdrant collection before processing"
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = UnifiedPipeline(
        collection_name=args.collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        buffer_size=args.buffer_size,
        threshold=args.threshold,
        clear=args.clear
    )
    
    pipeline.process_sections_file(
        args.sections_file,
        save_chunks=not args.no_chunk_files
    )


if __name__ == "__main__":
    main()
