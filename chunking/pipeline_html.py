#!/usr/bin/env python
"""
HTML Chunking Pipeline - processes pre-parsed HTML sections:
1. Load sections from parsed_sections_html.json
2. Chunk sections using semantic splitter
3. Store in Qdrant vector database

Usage:
    # First parse the HTML:
    python parsing/parsing_sections_html.py input/NYSE_MTX_2024.htm
    
    # Then chunk and store:
    python chunking/pipeline_html.py output/parsed_sections_html.json
"""

import argparse
import asyncio
import sys
import os
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from chunker import get_embedding_model, llamaindex_chunker

_CHUNK_NS = uuid.NAMESPACE_URL


def _point_id(key: str) -> str:
    """Deterministic, collision-resistant UUID5 from a string key."""
    return str(uuid.uuid5(_CHUNK_NS, key))


def flatten_sections(sections: List[Dict]) -> List[Dict]:
    """Recursively flatten nested sections + subsections into a single list."""
    result = []
    for section in sections:
        result.append(section)
        if section.get('subsections'):
            result.extend(flatten_sections(section['subsections']))
    return result


class HTMLPipeline:
    """Chunks HTML sections and stores them in Qdrant."""

    def __init__(
        self,
        collection_name: str = "financial",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        buffer_size: int = 1,
        threshold: int = 70,
        clear: bool = False,
        workers: int = 4,
    ):
        print("Initializing HTML Pipeline...")
        print(f"  Collection: {collection_name}")
        print(f"  Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"  Workers: {workers}")

        self.collection_name = collection_name
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.workers = workers
        self.clear = clear

        print("  Loading embedding model...")
        self.embed_model = get_embedding_model()
        if not self.embed_model:
            raise RuntimeError("Embedding model required for pipeline")
        print("  ✓ Embedding model loaded")

        self.vector_size = len(self.embed_model.encode(["test"])[0])
        self.client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        print("✓ Pipeline initialized\n")

    async def _ensure_collection(self):
        existing = [c.name for c in (await self.client.get_collections()).collections]
        if self.collection_name in existing and self.clear:
            await self.client.delete_collection(self.collection_name)
            print(f"  ✓ Cleared existing collection: {self.collection_name}")
        if self.collection_name not in existing or self.clear:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"  ✓ Created new collection: {self.collection_name}")
        else:
            print(f"  ✓ Using existing collection: {self.collection_name}")

    def _chunk_section(
        self,
        section: Dict,
        section_idx: int,
        total_sections: int,
        title_embedding: list,
        source_html: str,
    ) -> Tuple[List[PointStruct], List[Dict]]:
        """CPU-bound: chunk + embed text, build PointStructs."""
        section_title = section.get('title', 'Untitled')
        section_text = section.get('text', '')
        has_page_contents = bool(section.get('page_contents'))

        print(f"[{section_idx}/{total_sections}] Processing: {section_title}")

        # Always create a section point for metadata/navigation
        points: List[PointStruct] = []
        points.append(PointStruct(
            id=_point_id(f"section_{section_idx}"),
            vector=title_embedding,
            payload={
                "type": "section",
                "section_id": section_idx,
                "title": section_title,
                "original_title": section.get('original_title', section_title),
                "text": section_text[:1000] if section_text else "",
                "level": section.get('level', 0),
                "start_page": section.get('start_page', 0),
                "end_page": section.get('end_page', 0),
                "char_count": len(section_text),
                "word_count": section.get('word_count', len(section_text.split())),
                "source_html": source_html,
                "has_content": has_page_contents or bool(section_text.strip()),
            },
        ))

        # If no content to chunk, return just the section point
        if not has_page_contents and not section_text.strip():
            print(f"  [{section_idx}] ℹ Parent section (metadata only)")
            return points, []

        chunk_texts, chunk_ids, chunk_metadatas, chunk_embeddings = [], [], [], []

        if has_page_contents:
            print(f"  [{section_idx}] Processing {len(section['page_contents'])} pages...")
            for chunk_counter, page_content in enumerate(section['page_contents'], 1):
                page_num = page_content['page_number']
                page_text = page_content['content']  # Direct access to content field
                if not page_text or not page_text.strip():
                    continue

                chunks, _ = llamaindex_chunker(
                    text=page_text,
                    buffer_size=self.buffer_size,
                    threshold=self.threshold,
                    embed_model=self.embed_model,
                )

                for i, chunk in enumerate(chunks, 1):
                    chunk_texts.append(chunk['text'])
                    chunk_ids.append(f"section_{section_idx}_page_{page_num}_chunk_{i}")
                    chunk_metadatas.append({
                        "type": "chunk",
                        "section_id": section_idx,
                        "section_title": section_title,
                        "original_title": section.get('original_title', section_title),
                        "section_level": section.get('level', 0),
                        "source_page": page_num,
                        "start_page": section.get('start_page', 0),
                        "end_page": section.get('end_page', 0),
                        "chunk_index": chunk_counter,
                        "chunk_index_in_page": i,
                        "total_chunks_in_page": len(chunks),
                        "char_count": len(chunk['text']),
                        "word_count": len(chunk['text'].split()),
                        "source_html": source_html,
                    })
                    chunk_embeddings.append(chunk['embedding'])
        else:
            chunks, _ = llamaindex_chunker(
                text=section_text,
                buffer_size=self.buffer_size,
                threshold=self.threshold,
                embed_model=self.embed_model,
            )

            if not chunks:
                print(f"  [{section_idx}] ⚠ No chunks created")
                return points, []

            for i, chunk in enumerate(chunks, 1):
                chunk_texts.append(chunk['text'])
                chunk_ids.append(f"section_{section_idx}_chunk_{i}")
                chunk_metadatas.append({
                    "type": "chunk",
                    "section_id": section_idx,
                    "section_title": section_title,
                    "original_title": section.get('original_title', section_title),
                    "section_level": section.get('level', 0),
                    "start_page": section.get('start_page', 0),
                    "end_page": section.get('end_page', 0),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk['text']),
                    "word_count": len(chunk['text'].split()),
                    "source_html": source_html,
                })
                chunk_embeddings.append(chunk['embedding'])

        if not chunk_texts:
            print(f"  [{section_idx}] ⚠ No chunks created")
            return points, []

        for i in range(len(chunk_texts)):
            points.append(PointStruct(
                id=_point_id(chunk_ids[i]),
                vector=chunk_embeddings[i],
                payload={**chunk_metadatas[i], "text": chunk_texts[i]},
            ))

        print(f"  [{section_idx}] ✓ Built {len(chunk_texts)} chunks")

        details = []
        for text, meta in zip(chunk_texts, chunk_metadatas):
            detail = {
                'text': text,
                'length': len(text),
                'word_count': meta.get('word_count', len(text.split())),
                'section_id': meta['section_id'],
                'section_title': meta['section_title'],
                'original_title': meta.get('original_title', meta['section_title']),
                'section_level': meta['section_level'],
                'chunk_index': meta['chunk_index'],
                'method': 'SemanticSplitterNodeParser',
                'source_html': meta.get('source_html', 'unknown'),
            }
            if 'source_page' in meta:
                detail['source_page'] = meta['source_page']
                detail['chunk_index_in_page'] = meta.get('chunk_index_in_page', 1)
                detail['total_chunks_in_page'] = meta.get('total_chunks_in_page', 1)
            else:
                detail['start_page'] = meta['start_page']
                detail['end_page'] = meta['end_page']
                detail['total_chunks'] = meta.get('total_chunks', 0)
            details.append(detail)

        return points, details

    async def process_sections_file(
        self,
        sections_file: str,
        save_chunks: bool = True,
    ):
        start_time = time.time()

        print("=" * 70)
        print("HTML PIPELINE: CHUNK → EMBED → VECTOR STORE (parallel)")
        print("=" * 70)

        await self._ensure_collection()

        print(f"\nLoading sections from: {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            top_level_sections = data.get('sections', [])
            source_html = data.get('source_html') or data.get('filename', 'unknown')
            total_pages = data.get('total_pages') or data.get('num_pages', 0)

        if not top_level_sections:
            print("✗ No sections found!")
            return

        sections = flatten_sections(top_level_sections)
        print(f"✓ Found {len(top_level_sections)} top-level sections → {len(sections)} total (including subsections)")
        print(f"✓ Source HTML: {source_html}")
        print(f"✓ Total pages: {total_pages}")

        # Check if sections have text or page_contents
        has_content = any(
            len(s.get('text', '')) > 100 or s.get('page_contents')
            for s in sections
        )
        if not has_content:
            print("✗ Sections have no text or page_contents - cannot proceed")
            return

        # Batch-encode all section titles
        section_titles = [s.get('title', 'Untitled') for s in sections]
        print(f"\nBatch-encoding {len(section_titles)} section titles...")
        title_embeddings = self.embed_model.encode(section_titles)
        print("✓ Title embeddings ready")

        print(f"\nProcessing {len(sections)} sections with {self.workers} workers...\n")

        results: Dict[int, Tuple[List[PointStruct], List[Dict]]] = {}

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self._chunk_section,
                    section, idx, len(sections),
                    title_embeddings[idx - 1].tolist(),
                    source_html,
                ): idx
                for idx, section in enumerate(sections, 1)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    points, details = future.result()
                    results[idx] = (points, details)
                except Exception as e:
                    print(f"  [section {idx}] ✗ Error: {e}")

        all_points = [p for idx in sorted(results) for p in results[idx][0]]
        all_chunk_details = [d for idx in sorted(results) for d in results[idx][1]]
        total_chunks = sum(
            sum(1 for p in pts if p.payload.get('type') == 'chunk')
            for pts, _ in results.values()
        )

        if all_points:
            print(f"\nUploading {len(all_points)} points to Qdrant in one batch...")
            await self.client.upsert(
                collection_name=self.collection_name,
                points=all_points,
            )
            print("✓ Upload complete")

        if save_chunks and all_chunk_details:
            self._save_chunk_files(sections_file, sections, all_chunk_details, source_html)

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Total time:        {total_time:.2f}s")
        print(f"Source HTML:       {source_html}")
        print(f"Sections:          {len(sections)}")
        print(f"Chunks created:    {total_chunks}")
        if total_chunks > 0:
            print(f"Speed:             {total_chunks / total_time:.1f} chunks/sec")
        print("\n✓ All embeddings stored in Qdrant")
        print("✓ Ready for semantic search!")

    def _save_chunk_files(self, sections_file: str, sections: List[Dict], chunks: List[Dict], source_html: str = "unknown"):
        output_dir = Path(sections_file).parent
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        json_path = output_dir / "SemanticSplitterNodeParser_chunks_html.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "method": "SemanticSplitterNodeParser",
                "status": "completed",
                "source_html": source_html,
                "total_sections": len(sections),
                "total_chunks": len(chunks),
                "created_at": timestamp,
                "chunks": chunks,
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved chunks JSON to: {json_path}")

        txt_path = output_dir / "SemanticSplitterNodeParser_chunks_html.txt"
        sep = "=" * 80
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{sep}\nSemanticSplitterNodeParser Chunks (HTML)\n{sep}\n\n")
            f.write(f"Source HTML: {source_html}\n")
            f.write(f"Total sections: {len(sections)}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Created at: {timestamp}\n{sep}\n\n")

            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK {i:04d}\n")
                f.write(f"Length: {chunk['length']} characters\n")
                f.write(f"Words: {chunk.get('word_count', 'N/A')}\n")
                f.write(f"Section: {chunk['section_id']} - {chunk['section_title']}\n")
                f.write(f"Section Level: {chunk.get('section_level', 'N/A')}\n")
                if 'source_page' in chunk:
                    f.write(f"Source Page: {chunk['source_page']}\n")
                    f.write(f"Chunk {chunk.get('chunk_index_in_page', 'N/A')}/{chunk.get('total_chunks_in_page', 'N/A')} on page\n")
                else:
                    f.write(f"Pages: {chunk.get('start_page', 'N/A')}-{chunk.get('end_page', 'N/A')}\n")
                    f.write(f"Chunk {chunk.get('chunk_index', 'N/A')}/{chunk.get('total_chunks', 'N/A')} in section\n")
                f.write(f"Source HTML: {chunk.get('source_html', 'unknown')}\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'])
                f.write(f"\n\n{sep}\n\n")
        print(f"✓ Saved chunks TXT to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk HTML sections and store in vector DB (uses pre-parsed sections JSON)"
    )
    parser.add_argument(
        "sections_file",
        nargs='?',
        default="output/parsed_sections_html.json",
        help="Path to parsed sections JSON (default: output/parsed_sections_html.json)"
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial",
        help="Qdrant collection name (default: financial)"
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
    
    print("=" * 70)
    print("HTML CHUNKING PIPELINE")
    print("=" * 70)
    print(f"Sections file: {args.sections_file}")
    print(f"Qdrant collection: {args.collection}")
    print("=" * 70)
    print()
    
    async def run_pipeline():
        pipeline = HTMLPipeline(
            collection_name=args.collection,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            buffer_size=args.buffer_size,
            threshold=args.threshold,
            clear=args.clear,
            workers=args.workers,
        )
        await pipeline.process_sections_file(
            args.sections_file,
            save_chunks=not args.no_chunk_files,
        )
    
    asyncio.run(run_pipeline())
    
    print()
    print("=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Sections file: {args.sections_file}")
    print(f"Qdrant collection: {args.collection}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
