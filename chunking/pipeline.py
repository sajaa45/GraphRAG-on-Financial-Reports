import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from chunker import get_embedding_model, llamaindex_chunker


class UnifiedPipeline:
    """Chunks sections and stores them in Qdrant."""

    def __init__(
        self,
        collection_name: str = "financial_docs",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        buffer_size: int = 1,
        threshold: int = 70,
        clear: bool = False,
        workers: int = 4,
    ):
        print("Initializing Unified Pipeline...")
        print(f"  Collection: {collection_name}")
        print(f"  Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"  Workers: {workers}")

        self.buffer_size = buffer_size
        self.threshold = threshold
        self.workers = workers

        print("  Loading embedding model...")
        self.embed_model = get_embedding_model()
        if not self.embed_model:
            raise RuntimeError("Embedding model required for pipeline")
        print("  ✓ Embedding model loaded")

        self.vector_size = len(self.embed_model.encode(["test"])[0])

        print("  Connecting to Qdrant...")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing and clear:
            self.client.delete_collection(collection_name)
            print(f"  ✓ Cleared existing collection: {collection_name}")

        if collection_name not in existing or clear:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"  ✓ Created new collection: {collection_name}")
        else:
            print(f"  ✓ Using existing collection: {collection_name}")

        print("✓ Pipeline initialized\n")

    def process_section(
        self, section: Dict, section_idx: int, total_sections: int, source_pdf: str = None
    ) -> Tuple[int, List[Dict]]:
        """Chunk a section, embed, store in Qdrant. Returns (chunk_count, chunk_details)."""
        section_title = section.get('title', 'Untitled')
        section_text = section.get('text', '')
        has_page_contents = bool(section.get('page_contents'))

        print(f"[{section_idx}/{total_sections}] Processing: {section_title}")

        if not has_page_contents and not section_text.strip():
            print(f"  [{section_idx}] ⚠ Empty section, skipping")
            return 0, []

        # Store section metadata
        if section_text and len(section_text) > 50:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=section_idx,
                    vector=self.embed_model.encode([section_title])[0].tolist(),
                    payload={
                        "type": "section",
                        "section_id": section_idx,
                        "title": section_title,
                        "original_title": section.get('original_title', section_title),
                        "text": section_text[:1000],
                        "level": section.get('level', 0),
                        "start_page": section.get('start_page', 0),
                        "end_page": section.get('end_page', 0),
                        "char_count": len(section_text),
                        "word_count": section.get('word_count', len(section_text.split())),
                        "source_pdf": source_pdf or "unknown",
                    },
                )],
            )

        chunk_texts, chunk_ids, chunk_metadatas, chunk_embeddings = [], [], [], []

        if has_page_contents:
            print(f"  [{section_idx}] Processing {len(section['page_contents'])} pages...")
            for chunk_counter, page_content in enumerate(section['page_contents'], 1):
                page_num = page_content['page_number']
                page_text = page_content.get('content', '')
                if not page_text.strip():
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
                        "source_pdf": source_pdf or "unknown",
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
                return 0, []

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
                    "source_pdf": source_pdf or "unknown",
                })
                chunk_embeddings.append(chunk['embedding'])

        if not chunk_texts:
            print(f"  [{section_idx}] ⚠ No chunks created")
            return 0, []

        points = [
            PointStruct(
                id=abs(hash(chunk_ids[i])) % (2**63),
                vector=chunk_embeddings[i],
                payload={**chunk_metadatas[i], "text": chunk_texts[i]},
            )
            for i in range(len(chunk_texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"  [{section_idx}] ✓ Stored {len(chunk_texts)} chunks")

        # Build details from what we already have — no Qdrant scroll needed
        details = []
        for i, (text, meta) in enumerate(zip(chunk_texts, chunk_metadatas)):
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
                'source_pdf': meta.get('source_pdf', 'unknown'),
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

        return len(chunk_texts), details

    @staticmethod
    def _flatten_sections(sections: List[Dict]) -> List[Dict]:
        """Recursively flatten nested sections + subsections into a single list."""
        result = []
        for section in sections:
            result.append(section)
            if section.get('subsections'):
                result.extend(UnifiedPipeline._flatten_sections(section['subsections']))
        return result

    def process_sections_file(
        self,
        sections_file: str,
        page_index_file: Optional[str] = None,
        save_chunks: bool = True,
    ):
        start_time = time.time()

        print("=" * 70)
        print("UNIFIED PIPELINE: CHUNK → EMBED → VECTOR STORE (parallel)")
        print("=" * 70)

        print(f"\nLoading sections from: {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            top_level_sections = data.get('sections', [])
            # Support both 'source_pdf' (page_index style) and 'filename' (parsed_sections style)
            source_pdf = data.get('source_pdf') or data.get('filename', 'unknown')
            total_pages = data.get('total_pages') or data.get('num_pages', 0)

        if not top_level_sections:
            print("✗ No sections found!")
            return

        # Flatten all nested subsections into a single list
        sections = self._flatten_sections(top_level_sections)
        print(f"✓ Found {len(top_level_sections)} top-level sections → {len(sections)} total (including subsections)")
        print(f"✓ Source PDF: {source_pdf}")
        print(f"✓ Total pages: {total_pages}")

        # Always use page_index for PDF-sourced text when available
        if page_index_file:
            print(f"✓ Loading page index from: {page_index_file}")
            page_index_data = self._load_page_index(page_index_file)
            if page_index_data:
                print("✓ Populating section text from PDF page index...")
                sections = self._add_text_from_page_index(sections, page_index_data)
                populated = sum(1 for s in sections if len(s.get('text', '')) > 100)
                print(f"✓ Populated text for {populated}/{len(sections)} sections")
        else:
            has_text = any(len(s.get('text', '')) > 100 for s in sections)
            if not has_text:
                print("✗ Sections have no text and no page index provided - cannot proceed")
                return

        print(f"\nProcessing {len(sections)} sections with {self.workers} workers...\n")

        total_chunks = 0
        # results keyed by section_idx to preserve order in output files
        results: Dict[int, List[Dict]] = {}

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.process_section, section, idx, len(sections), source_pdf): idx
                for idx, section in enumerate(sections, 1)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    count, details = future.result()
                    total_chunks += count
                    if save_chunks and details:
                        results[idx] = details
                except Exception as e:
                    print(f"  [section {idx}] ✗ Error: {e}")

        # Flatten details in section order
        all_chunk_details = [d for idx in sorted(results) for d in results[idx]]

        if save_chunks and all_chunk_details:
            self._save_chunk_files(sections_file, sections, all_chunk_details, source_pdf)

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)
        print(f"Total time:        {total_time:.2f}s")
        print(f"Source PDF:        {source_pdf}")
        print(f"Sections:          {len(sections)}")
        print(f"Chunks created:    {total_chunks}")
        if total_chunks > 0:
            print(f"Speed:             {total_chunks / total_time:.1f} chunks/sec")
        print("\n✓ All embeddings stored in Qdrant")
        print("✓ Ready for semantic search!")

    def _load_page_index(self, page_index_file: str) -> Dict:
        """Load page index data."""
        print(f"✓ Loading page index from: {page_index_file}")
        try:
            with open(page_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pages = data.get('pages', [])
            print(f"✓ Loaded {len(pages)} pages")
            # Convert list to dict for easier lookup
            return {p['page']: p for p in pages}
        except FileNotFoundError:
            print(f"✗ Page index file not found: {page_index_file}")
            return {}
        except Exception as e:
            print(f"✗ Error loading page index file: {e}")
            return {}

    def _add_text_from_page_index(self, sections: List[Dict], page_index: Dict) -> List[Dict]:
        """Add text to sections from page index."""
        for section in sections:
            section_text = ""
            page_contents = []
            
            for page_num in range(section['start_page'], section['end_page'] + 1):
                page_data = page_index.get(page_num)
                if page_data and page_data.get('text'):
                    page_text = page_data['text']
                    section_text += page_text + "\n"
                    page_contents.append({
                        'page_number': page_num,
                        'content': page_text,
                        'sections': page_data.get('sections', [])
                    })

            section['text'] = section_text.strip()
            section['text_length'] = len(section_text)
            section['word_count'] = len(section_text.split())
            if page_contents:
                section['page_contents'] = page_contents

            if section.get('subsections'):
                section['subsections'] = self._add_text_from_page_index(
                    section['subsections'], 
                    page_index
                )

        return sections

    def _load_pages_data(self, pages_file: str) -> Dict:
        """Legacy method for loading old-style pages data."""
        print(f"✓ Loading page data from: {pages_file}")
        try:
            with open(pages_file, 'r', encoding='utf-8') as f:
                pages_data = json.load(f).get('pages', {})
            print(f"✓ Loaded {len(pages_data)} pages")
            return pages_data
        except FileNotFoundError:
            print(f"✗ Pages file not found: {pages_file}")
            return {}
        except Exception as e:
            print(f"✗ Error loading pages file: {e}")
            return {}

    def _add_text_to_sections(self, sections: List[Dict], pages_data: Dict) -> List[Dict]:
        for section in sections:
            section_text = ""
            page_contents = []
            for page_num in range(section['start_page'], section['end_page'] + 1):
                page_text = pages_data.get(str(page_num)) or pages_data.get(page_num)
                if page_text:
                    section_text += page_text + "\n"
                    page_contents.append({'page_number': page_num, 'content': page_text})

            section['text'] = section_text
            if page_contents:
                section['page_contents'] = page_contents

            if section.get('subsections'):
                section['subsections'] = self._add_text_to_sections(section['subsections'], pages_data)

        return sections

    def _save_chunk_files(self, sections_file: str, sections: List[Dict], chunks: List[Dict], source_pdf: str = "unknown"):
        output_dir = Path(sections_file).parent
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        json_path = output_dir / "SemanticSplitterNodeParser_chunks.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "method": "SemanticSplitterNodeParser",
                "status": "completed",
                "source_pdf": source_pdf,
                "total_sections": len(sections),
                "total_chunks": len(chunks),
                "created_at": timestamp,
                "chunks": chunks,
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved chunks JSON to: {json_path}")

        txt_path = output_dir / "SemanticSplitterNodeParser_chunks.txt"
        sep = "=" * 80
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{sep}\nSemanticSplitterNodeParser Chunks\n{sep}\n\n")
            f.write(f"Source PDF: {source_pdf}\n")
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
                f.write(f"Source PDF: {chunk.get('source_pdf', 'unknown')}\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'])
                f.write(f"\n\n{sep}\n\n")
        print(f"✓ Saved chunks TXT to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified pipeline: Chunk and store in vector DB")
    parser.add_argument("sections_file", nargs="?", default="output/parsed_sections.json",
                        help="Path to sections JSON (default: output/parsed_sections.json)")
    parser.add_argument("--page-index", default="output/page_index.json",
                        help="Path to page index JSON (default: output/page_index.json)")
    parser.add_argument("--collection", "-c", default="financial_docs", help="Qdrant collection name")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--buffer-size", type=int, default=1, help="LlamaIndex buffer size")
    parser.add_argument("--threshold", type=int, default=70, help="LlamaIndex threshold")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--no-chunk-files", action="store_true", help="Skip saving chunk JSON/TXT files")
    parser.add_argument("--clear", action="store_true", help="Delete and recreate the Qdrant collection")
    args = parser.parse_args()

    pipeline = UnifiedPipeline(
        collection_name=args.collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        buffer_size=args.buffer_size,
        threshold=args.threshold,
        clear=args.clear,
        workers=args.workers,
    )
    pipeline.process_sections_file(
        args.sections_file,
        page_index_file=args.page_index,
        save_chunks=not args.no_chunk_files,
    )


if __name__ == "__main__":
    main()
