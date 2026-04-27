import time
import os
from typing import List, Dict, Any
import statistics
import re
import numpy as np

def load_json_sections(file_path: str) -> List[Dict]:
    """Load the processed sections from JSON for section-based chunking"""
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract sections list
        if 'sections' in data:
            return data['sections']
        else:
            return []
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please run sections_merging_pages.py first to process JSON files")
        return None
    except Exception as e:
        print(f"Error loading JSON sections: {e}")
        return None
        return None

def apply_chunking_to_sections(sections: List[Dict], chunker_func, method_name: str, **kwargs) -> List[Dict]:
    """
    Apply a chunking method to each section individually and track page origins
    
    Args:
        sections: List of page sections
        chunker_func: The chunking function to apply
        method_name: Name of the chunking method
        **kwargs: Arguments for the chunker function
    
    Returns:
        List of chunks with page tracking information
    """
    
    all_chunks = []
    
    for section in sections:
        print(f"  Processing section {section['section_id']} (pages {section['page_range']}) with {method_name}")
        
        # Apply chunking method to this section
        section_chunks = chunker_func(section['text'], **kwargs)
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(section_chunks):
            chunk_info = {
                'text': chunk_text,
                'length': len(chunk_text),
                'section_id': section['section_id'],
                'source_pages': section['pages'],
                'page_range': section['page_range'],
                'chunk_index_in_section': i + 1,
                'total_chunks_in_section': len(section_chunks),
                'method': method_name
            }
            all_chunks.append(chunk_info)
    
    return all_chunks

def langchain_chunker(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """LangChain RecursiveCharacterTextSplitter implementation"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        return chunks
    
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return []

def llamaindex_chunker(text: str) -> List[str]:
    """LlamaIndex SemanticSplitterNodeParser with free embeddings"""
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        
        # Create document
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        
        chunks = [node.text for node in nodes]
        return chunks
    
    except ImportError:
        print("LlamaIndex or HuggingFace embeddings not installed.")
        print("Install with: pip install llama-index llama-index-embeddings-huggingface sentence-transformers")
        return []
    except Exception as e:
        print(f"LlamaIndex error: {e}")
        return []

def chonkie_chunker(text: str) -> List[str]:
    """Chonkie semantic chunker implementation"""
    try:
        from chonkie import SemanticChunker
        
        chunker = SemanticChunker()
        chunks = chunker.chunk(text)
        
        # Extract text from chunk objects
        chunk_texts = [chunk.text for chunk in chunks]
        return chunk_texts
    
    except ImportError:
        print("Chonkie not installed. Install with: pip install chonkie")
        return []

def calculate_quality_metrics(chunks: List[str]) -> Dict[str, float]:
    """Calculate chunking quality metrics"""
    if not chunks:
        return {
            "sentence_completeness": 0.0,
            "paragraph_preservation": 0.0,
            "size_consistency": 0.0,
            "content_overlap_ratio": 0.0
        }
    
    # 1. Sentence completeness - how many chunks end with proper sentence endings
    sentence_endings = ['.', '!', '?', ':', ';']
    complete_sentences = sum(1 for chunk in chunks 
                           if any(chunk.strip().endswith(ending) for ending in sentence_endings))
    sentence_completeness = complete_sentences / len(chunks) if chunks else 0
    
    # 2. Paragraph preservation - chunks that start/end at paragraph boundaries
    paragraph_chunks = sum(1 for chunk in chunks 
                          if chunk.strip().startswith('\n') or chunk.strip().endswith('\n\n'))
    paragraph_preservation = paragraph_chunks / len(chunks) if chunks else 0
    
    # 3. Size consistency - coefficient of variation (lower is better)
    lengths = [len(chunk) for chunk in chunks]
    if len(lengths) > 1:
        mean_length = statistics.mean(lengths)
        std_length = statistics.stdev(lengths)
        size_consistency = 1 - (std_length / mean_length) if mean_length > 0 else 0
        size_consistency = max(0, size_consistency)  # Ensure non-negative
    else:
        size_consistency = 1.0
    
    # 4. Content overlap detection (for overlapping chunkers)
    overlap_chars = 0
    total_chars = sum(len(chunk) for chunk in chunks)
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Find overlap by checking if end of current chunk appears in start of next
        for j in range(min(100, len(current_chunk)), 0, -1):  # Check up to 100 chars
            suffix = current_chunk[-j:]
            if suffix in next_chunk[:j*2]:  # Look in first part of next chunk
                overlap_chars += j
                break
    
    content_overlap_ratio = overlap_chars / total_chars if total_chars > 0 else 0
    
    return {
        "sentence_completeness": round(sentence_completeness, 3),
        "paragraph_preservation": round(paragraph_preservation, 3),
        "size_consistency": round(size_consistency, 3),
        "content_overlap_ratio": round(content_overlap_ratio, 3)
    }

def calculate_semantic_coherence(chunks: List[str]) -> float:
    """Calculate semantic coherence using sentence transformers (free)"""
    try:
        from sentence_transformers import SentenceTransformer
        
        if not chunks or len(chunks) < 2:
            return 0.0
        
        # Use a lightweight, free model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Take sample of chunks to avoid memory issues
        sample_size = min(20, len(chunks))
        sample_chunks = chunks[:sample_size]
        
        # Get embeddings
        embeddings = model.encode(sample_chunks)
        
        # Calculate average cosine similarity between consecutive chunks
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(similarity)
        
        return round(np.mean(similarities), 3) if similarities else 0.0
    
    except ImportError:
        print("Sentence Transformers not available for semantic coherence calculation")
        return 0.0
    except Exception as e:
        print(f"Error calculating semantic coherence: {e}")
        return 0.0
def apply_chunking_to_sections(sections: List[Dict], chunker_func, method_name: str, **kwargs) -> List[Dict]:
    """
    Apply a chunking method to each section individually and track page origins
    This matches the two-step approach from sections_merging_pages.py
    
    Args:
        sections: List of page sections from sections_merging_pages
        chunker_func: The chunking function to apply
        method_name: Name of the chunking method
        **kwargs: Arguments for the chunker function
    
    Returns:
        List of chunks with page tracking information
    """
    
    all_chunks = []
    
    for section in sections:
        print(f"  Processing section {section['section_id']} (pages {section['page_range']}) with {method_name}")
        
        # Apply chunking method to this section's text
        section_chunks = chunker_func(section['text'], **kwargs)
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(section_chunks):
            chunk_info = {
                'text': chunk_text,
                'length': len(chunk_text),
                'section_id': section['section_id'],
                'source_pages': section['pages'],
                'page_range': section['page_range'],
                'chunk_index_in_section': i + 1,
                'total_chunks_in_section': len(section_chunks),
                'method': method_name
            }
            all_chunks.append(chunk_info)
    
    return all_chunks
    """Analyze chunk statistics and quality metrics"""
    if not chunks:
        return {
            "method": method_name,
            "total_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "median_length": 0,
            "total_characters": 0,
            "quality_metrics": {
                "sentence_completeness": 0,
                "paragraph_preservation": 0,
                "size_consistency": 0,
                "content_overlap_ratio": 0,
                "semantic_coherence": 0
            }
        }
    
    lengths = [len(chunk) for chunk in chunks]
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(chunks)
    
    # Add semantic coherence (this might take a moment)
    print(f"  Calculating semantic coherence for {method_name}...")
    quality_metrics["semantic_coherence"] = calculate_semantic_coherence(chunks)
    
    return {
        "method": method_name,
        "total_chunks": len(chunks),
        "avg_length": round(statistics.mean(lengths), 2),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": statistics.median(lengths),
        "total_characters": sum(lengths),
        "quality_metrics": quality_metrics
    }

def benchmark_chunker_on_sections(sections: List[Dict], chunker_func, method_name: str, **kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
    """Benchmark a chunking method on sections"""
    print(f"\n--- Testing {method_name} on {len(sections)} sections ---")
    
    start_time = time.time()
    chunks_with_metadata = apply_chunking_to_sections(sections, chunker_func, method_name, **kwargs)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Extract just the text for analysis
    chunk_texts = [chunk['text'] for chunk in chunks_with_metadata]
    
    stats = analyze_chunks(chunk_texts, method_name)
    stats["processing_time"] = round(processing_time, 3)
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average chunk length: {stats['avg_length']} characters")
    
    return stats, chunks_with_metadata

def save_sample_chunks_with_metadata(chunks_with_metadata: List[Dict], method_name: str, num_samples: int = 3):
    """Save sample chunks with page metadata to file for manual inspection"""
    if not chunks_with_metadata:
        return
    
    # Determine output directory - prioritize samples mount, then output, then current
    output_dir = "."
    if os.path.exists("/app/samples") and os.access("/app/samples", os.W_OK):
        output_dir = "/app/samples"
    elif os.path.exists("/app/output") and os.access("/app/output", os.W_OK):
        output_dir = "/app/output"
    elif os.path.exists("output") and os.access("output", os.W_OK):
        output_dir = "output"
    
    filename = os.path.join(output_dir, f"sample_chunks_{method_name.lower().replace(' ', '_')}.txt")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== Sample Chunks from {method_name} (Section-Based) ===\n\n")
            
            # Take samples from beginning, middle, and end
            indices = []
            if len(chunks_with_metadata) >= num_samples:
                indices = [0, len(chunks_with_metadata)//2, len(chunks_with_metadata)-1]
            else:
                indices = list(range(len(chunks_with_metadata)))
            
            for i, idx in enumerate(indices):
                chunk = chunks_with_metadata[idx]
                f.write(f"--- Sample Chunk {i+1} (Index {idx}) ---\n")
                f.write(f"Length: {chunk['length']} characters\n")
                f.write(f"Section: {chunk['section_id']}\n")
                f.write(f"Source Pages: {chunk['page_range']}\n")
                f.write(f"Page Numbers: {', '.join(chunk['source_pages'])}\n")
                f.write(f"Chunk {chunk['chunk_index_in_section']} of {chunk['total_chunks_in_section']} in this section\n\n")
                f.write(chunk['text'])
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"    Saved: {filename}")
    except Exception as e:
        print(f"    Failed to save {filename}: {e}")

def save_all_chunks_with_metadata(chunks_with_metadata: List[Dict], method_name: str):
    """Save ALL chunks with metadata to file for complete inspection"""
    if not chunks_with_metadata:
        return
    
    # Determine output directory - prioritize output, then samples, then current
    output_dir = "."
    if os.path.exists("/app/output") and os.access("/app/output", os.W_OK):
        output_dir = "/app/output"
    elif os.path.exists("/app/samples") and os.access("/app/samples", os.W_OK):
        output_dir = "/app/samples"
    elif os.path.exists("output") and os.access("output", os.W_OK):
        output_dir = "output"
    
    filename = os.path.join(output_dir, f"all_chunks_{method_name.lower().replace(' ', '_')}.txt")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== ALL CHUNKS from {method_name} (Section-Based) ===\n")
            f.write(f"Total chunks: {len(chunks_with_metadata)}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for i, chunk in enumerate(chunks_with_metadata, 1):
                f.write(f"CHUNK {i:04d}\n")
                f.write(f"Length: {chunk['length']} characters\n")
                f.write(f"Method: {chunk['method']}\n")
                f.write(f"Section: {chunk['section_id']}\n")
                f.write(f"Source Pages: {chunk['page_range']}\n")
                f.write(f"Page Numbers: {', '.join(chunk['source_pages'])}\n")
                f.write(f"Chunk {chunk['chunk_index_in_section']} of {chunk['total_chunks_in_section']} in this section\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'])
                f.write("\n\n" + "="*80 + "\n\n")
        
        print(f"    Saved ALL chunks: {filename}")
        
        # Also save as JSON for programmatic access
        json_filename = os.path.join(output_dir, f"all_chunks_{method_name.lower().replace(' ', '_')}.json")
        try:
            import json
            chunk_data = {
                "method": method_name,
                "total_chunks": len(chunks_with_metadata),
                "generated_on": time.strftime('%Y-%m-%d %H:%M:%S'),
                "chunks": chunks_with_metadata
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            print(f"    Saved JSON: {json_filename}")
        except Exception as e:
            print(f"    Failed to save JSON {json_filename}: {e}")
            
    except Exception as e:
        print(f"    Failed to save {filename}: {e}")

def create_chunk_comparison_summary(all_chunks_with_metadata: Dict[str, List[Dict]], results: List[Dict]):
    """Create a summary file comparing all chunking methods with section-based approach"""
    
    output_dir = "."
    if os.path.exists("/app/output") and os.access("/app/output", os.W_OK):
        output_dir = "/app/output"
    elif os.path.exists("output") and os.access("output", os.W_OK):
        output_dir = "output"
    
    filename = os.path.join(output_dir, "chunking_methods_comparison.txt")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SECTION-BASED CHUNKING METHODS COMPARISON SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Note: This comparison uses section-based chunking (two-step process)\n")
            f.write("Step 1: Create page sections based on custom logic\n")
            f.write("Step 2: Apply chunking methods to each section individually\n\n")
            
            # Performance comparison table
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<35} {'Chunks':<8} {'Avg Len':<8} {'Time(s)':<8} {'Min':<6} {'Max':<6}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                if result['total_chunks'] > 0:
                    f.write(f"{result['method']:<35} {result['total_chunks']:<8} "
                          f"{result['avg_length']:<8} {result['processing_time']:<8} "
                          f"{result['min_length']:<6} {result['max_length']:<6}\n")
            
            # Quality metrics table
            f.write("\n\nQUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<35} {'Sentence':<10} {'Paragraph':<10} {'Size Cons.':<10} {'Semantic':<8}\n")
            f.write(f"{'':35} {'Complete':<10} {'Preserve':<10} {'(0-1)':<10} {'Coherence':<8}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                if result['total_chunks'] > 0:
                    qm = result['quality_metrics']
                    f.write(f"{result['method']:<35} {qm['sentence_completeness']:<10} "
                          f"{qm['paragraph_preservation']:<10} {qm['size_consistency']:<10} "
                          f"{qm['semantic_coherence']:<8}\n")
            
            # Overall scores
            f.write("\n\nOVERALL SCORES (0-1 scale, higher is better)\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                if result['total_chunks'] > 0:
                    qm = result['quality_metrics']
                    overall_score = (
                        qm['sentence_completeness'] * 0.3 +
                        qm['paragraph_preservation'] * 0.2 +
                        qm['size_consistency'] * 0.2 +
                        qm['semantic_coherence'] * 0.3
                    )
                    f.write(f"{result['method']:<35} Overall Score: {overall_score:.3f}\n")
            
            # Section-based analysis
            f.write("\n\nSECTION-BASED ANALYSIS\n")
            f.write("-" * 80 + "\n")
            
            for method_key, chunks_with_metadata in all_chunks_with_metadata.items():
                if chunks_with_metadata:
                    method_name = method_key.replace('_', ' ').title()
                    f.write(f"\n{method_name}:\n")
                    
                    # Analyze chunks per section
                    sections_analysis = {}
                    for chunk in chunks_with_metadata:
                        section_id = chunk['section_id']
                        if section_id not in sections_analysis:
                            sections_analysis[section_id] = {
                                'chunks': 0,
                                'total_length': 0,
                                'page_range': chunk['page_range'],
                                'source_pages': chunk['source_pages']
                            }
                        sections_analysis[section_id]['chunks'] += 1
                        sections_analysis[section_id]['total_length'] += chunk['length']
                    
                    f.write(f"  Sections processed: {len(sections_analysis)}\n")
                    f.write(f"  Total chunks: {len(chunks_with_metadata)}\n")
                    f.write(f"  Avg chunks per section: {len(chunks_with_metadata)/len(sections_analysis):.1f}\n")
                    
                    # Show section breakdown
                    f.write(f"  Section breakdown:\n")
                    for section_id, analysis in sorted(sections_analysis.items()):
                        f.write(f"    Section {section_id} (pages {analysis['page_range']}): {analysis['chunks']} chunks\n")
            
            # First few chunks from each method for quick comparison
            f.write("\n\n" + "="*80 + "\n")
            f.write("FIRST CHUNK FROM EACH METHOD (for quick comparison)\n")
            f.write("="*80 + "\n")
            
            for method_key, chunks_with_metadata in all_chunks_with_metadata.items():
                if chunks_with_metadata:
                    method_name = method_key.replace('_', ' ').title()
                    chunk = chunks_with_metadata[0]
                    f.write(f"\n--- {method_name} (First Chunk) ---\n")
                    f.write(f"Length: {chunk['length']} characters\n")
                    f.write(f"Section: {chunk['section_id']}, Pages: {chunk['page_range']}\n")
                    f.write(f"Chunk {chunk['chunk_index_in_section']} of {chunk['total_chunks_in_section']} in section\n")
                    f.write("-" * 40 + "\n")
                    f.write(chunk['text'][:500])  # First 500 chars
                    if len(chunk['text']) > 500:
                        f.write("...\n[TRUNCATED - see all_chunks_*.txt for complete chunks]")
                    f.write("\n\n")
        
        print(f"    Saved comparison summary: {filename}")
        
    except Exception as e:
        print(f"    Failed to save comparison summary: {e}")


def main():
    # Load processed JSON sections
    # Look for JSON section files in output directory
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    
    # Find JSON section files
    section_files = []
    if os.path.exists(output_dir):
        section_files = [f for f in os.listdir(output_dir) if f.endswith('_sections.json')]
    
    if not section_files:
        print("No JSON section files found. Please run sections_merging_pages.py first.")
        return
    
    # Check if comparison files already exist
    comparison_files = [
        "chunking_methods_comparison.txt",
        "sample_chunks_langchain.txt",
        "sample_chunks_llamaindex.txt", 
        "sample_chunks_chonkie.txt",
        "all_chunks_langchain.txt",
        "all_chunks_llamaindex.txt",
        "all_chunks_chonkie.txt"
    ]
    
    all_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in comparison_files)
    
    if all_exist:
        print("All chunking comparison files already exist, skipping comparison")
        print("   Delete comparison files if you want to regenerate them")
        return
    
    # Use the first section file found
    section_file = os.path.join(output_dir, section_files[0])
    sections = load_json_sections(section_file)
    
    if not sections:
        return
    
    print(f"Loaded {len(sections)} sections from JSON")
    print(f"Source: {section_files[0]}")
    
    # Calculate total text length for reference
    total_text_length = sum(len(section['text']) for section in sections)
    print(f"Total text across all sections: {total_text_length:,} characters")
    
    # Show section breakdown
    print(f"\nSection breakdown:")
    for section in sections[:5]:  # Show first 5 sections
        print(f"  Section {section['section_id']}: {len(section['text'])} chars, pages {section['page_range']}")
    if len(sections) > 5:
        print(f"  ... and {len(sections) - 5} more sections")
    
    print("="*60)
    
    results = []
    all_chunks_with_metadata = {}
    
    # Test LangChain RecursiveCharacterTextSplitter on sections
    stats, chunks_with_metadata = benchmark_chunker_on_sections(
        sections,
        langchain_chunker, 
        "LangChain RecursiveCharacterTextSplitter",
        chunk_size=512,
        overlap=64
    )
    results.append(stats)
    all_chunks_with_metadata["langchain"] = chunks_with_metadata
    
    # Test LlamaIndex SemanticSplitterNodeParser on sections
    stats, chunks_with_metadata = benchmark_chunker_on_sections(
        sections,
        llamaindex_chunker,
        "LlamaIndex SemanticSplitterNodeParser"
    )
    results.append(stats)
    all_chunks_with_metadata["llamaindex"] = chunks_with_metadata
    
    # Test Chonkie on sections
    stats, chunks_with_metadata = benchmark_chunker_on_sections(
        sections,
        chonkie_chunker,
        "Chonkie SemanticChunker"
    )
    results.append(stats)
    all_chunks_with_metadata["chonkie"] = chunks_with_metadata
    
    # Print comparison table
    print("\n" + "="*100)
    print("SECTION-BASED CHUNKING PERFORMANCE COMPARISON")
    print("="*100)
    
    print(f"{'Method':<35} {'Chunks':<8} {'Avg Len':<8} {'Time(s)':<8} {'Min':<6} {'Max':<6}")
    print("-" * 100)
    
    for result in results:
        if result['total_chunks'] > 0:
            print(f"{result['method']:<35} {result['total_chunks']:<8} "
                  f"{result['avg_length']:<8} {result['processing_time']:<8} "
                  f"{result['min_length']:<6} {result['max_length']:<6}")
    
    # Print quality metrics table
    print("\n" + "="*100)
    print("SECTION-BASED CHUNKING QUALITY COMPARISON")
    print("="*100)
    
    print(f"{'Method':<35} {'Sentence':<10} {'Paragraph':<10} {'Size Cons.':<10} {'Overlap':<8} {'Semantic':<8}")
    print(f"{'':35} {'Complete':<10} {'Preserve':<10} {'(higher=better)':<10} {'Ratio':<8} {'Coherence':<8}")
    print("-" * 100)
    
    for result in results:
        if result['total_chunks'] > 0:
            qm = result['quality_metrics']
            print(f"{result['method']:<35} {qm['sentence_completeness']:<10} "
                  f"{qm['paragraph_preservation']:<10} {qm['size_consistency']:<10} "
                  f"{qm['content_overlap_ratio']:<8} {qm['semantic_coherence']:<8}")
    
    # Calculate and display overall scores
    print("\n" + "="*100)
    print("OVERALL QUALITY SCORES (0-1 scale, higher is better)")
    print("="*100)
    
    for result in results:
        if result['total_chunks'] > 0:
            qm = result['quality_metrics']
            # Calculate weighted overall score
            overall_score = (
                qm['sentence_completeness'] * 0.3 +
                qm['paragraph_preservation'] * 0.2 +
                qm['size_consistency'] * 0.2 +
                qm['semantic_coherence'] * 0.3
            )
            print(f"{result['method']:<35} Overall Score: {overall_score:.3f}")
            
            # Speed score (inverse of time, normalized)
            max_time = max(r['processing_time'] for r in results if r['total_chunks'] > 0)
            speed_score = 1 - (result['processing_time'] / max_time) if max_time > 0 else 1
            print(f"{'':35} Speed Score:   {speed_score:.3f}")
            print(f"{'':35} Combined:      {(overall_score + speed_score) / 2:.3f}")
            print()
    
    # Save sample chunks for manual inspection
    print("\n" + "="*60)
    print("SAVING SAMPLE CHUNKS FOR MANUAL INSPECTION")
    print("="*60)
    
    for method_key, chunks_with_metadata in all_chunks_with_metadata.items():
        if chunks_with_metadata:
            method_name = method_key.replace('_', ' ').title()
            save_sample_chunks_with_metadata(chunks_with_metadata, method_name)
    
    # Save ALL chunks for complete inspection
    print("\n" + "="*60)
    print("SAVING ALL CHUNKS FOR COMPLETE INSPECTION")
    print("="*60)
    
    for method_key, chunks_with_metadata in all_chunks_with_metadata.items():
        if chunks_with_metadata:
            method_name = method_key.replace('_', ' ').title()
            save_all_chunks_with_metadata(chunks_with_metadata, method_name)
    
    # Create comparison summary
    print("\n" + "="*60)
    print("CREATING COMPARISON SUMMARY")
    print("="*60)
    create_chunk_comparison_summary(all_chunks_with_metadata, results)
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. LangChain RecursiveCharacterTextSplitter:")
    print("   - Best for: General purpose, fast processing")
    print("   - Pros: Fast, consistent chunk sizes, good overlap")
    print("   - Cons: May split mid-sentence or mid-concept")
    
    print("\n2. LlamaIndex SemanticSplitterNodeParser:")
    print("   - Best for: High-quality semantic boundaries")
    print("   - Pros: Respects semantic meaning, cleaner splits")
    print("   - Cons: Slower, requires embeddings model, variable chunk sizes")
    
    print("\n3. Chonkie SemanticChunker:")
    print("   - Best for: Balance of speed and semantic awareness")
    print("   - Pros: Faster than LlamaIndex, semantic boundaries")
    print("   - Cons: Newer library, less established")
    
    print(f"\nOriginal text length: {total_text_length:,} characters across {len(sections)} sections")
    print("Check the sample_chunks_*.txt files to manually evaluate chunk quality!")
    print("Check the all_chunks_*.txt files to see EVERY chunk from each method!")
    print("Check the all_chunks_*.json files for programmatic access to all chunks!")
    
    print("\n" + "="*60)
    print("SECTION-BASED CHUNKING APPROACH")
    print("="*60)
    print("This comparison uses your custom two-step chunking approach:")
    print("Step 1: Create page sections based on custom logic")
    print("        (combine pages if current doesn't start with uppercase AND previous doesn't end with period)")
    print("Step 2: Apply chunking methods to EACH SECTION INDIVIDUALLY")
    print("        (preserves page tracking and respects section boundaries)")
    print("")
    print("This is different from traditional chunking that processes the entire document as one unit.")
    print("Your approach maintains better context boundaries and page tracking.")
    
    print("\n" + "="*60)
    print("FILES GENERATED")
    print("="*60)
    print("Comparison summary:")
    print("   - chunking_methods_comparison.txt")
    print("\nSample files (3 chunks each with page metadata):")
    print("   - sample_chunks_langchain.txt")
    print("   - sample_chunks_llamaindex.txt") 
    print("   - sample_chunks_chonkie.txt")
    print("\nComplete files (ALL chunks with page metadata):")
    print("   - all_chunks_langchain.txt")
    print("   - all_chunks_llamaindex.txt")
    print("   - all_chunks_chonkie.txt")
    print("\nJSON files (programmatic access with metadata):")
    print("   - all_chunks_langchain.json")
    print("   - all_chunks_llamaindex.json")
    print("   - all_chunks_chonkie.json")

if __name__ == "__main__":
    main()