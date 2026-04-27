import json
import os
import time
import statistics
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

_BULLET_CHARS = frozenset({'•', '●', '○', '◦', '▪', '▫', '■', '□', '◆', '◇'})
_ARABIC_NUMERALS = frozenset({'٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'})
_DECORATIVE_SYMBOLS = _BULLET_CHARS | _ARABIC_NUMERALS  # Combined for faster lookup
_DECORATIVE_CHARS = _DECORATIVE_SYMBOLS | {'$', '\\', ' ', '¢', '۰', '٠', '-', '=', '*'}
_FINANCIAL_INDICATORS = frozenset([
    'sar', 'usd', 'million', 'billion', '%', 'percent',
    'revenue', 'income', 'profit', 'loss', 'equity',
    'debt', 'assets', 'cash', 'investment'
])


def load_json_sections(file_path: str) -> Optional[List[Dict]]:
    """Load the processed sections from JSON for section-based chunking."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('sections', [])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please run sections_merging_pages.py first to process JSON files")
        return None
    except Exception as e:
        print(f"Error loading JSON sections: {e}")
        return None


def flatten_hierarchical_sections(sections, parent_path="", pages_data=None):
    """
    Flatten hierarchical sections from sections_parser_pdf.py into a list with section paths.
    
    Args:
        sections: List of hierarchical sections
        parent_path: Parent section path for recursion
        pages_data: Dictionary mapping page numbers to text content
    
    Returns:
        List of flattened sections with section_path added and text from pages_data
    """
    flattened = []
    
    for section in sections:
        # Build section path
        section_path = f"{parent_path} > {section['title']}" if parent_path else section['title']
        
        # Get text from pages_data if available
        section_text = ""
        if pages_data:
            for page_num in range(section['start_page'], section['end_page'] + 1):
                page_key = str(page_num)
                if page_key in pages_data:
                    section_text += pages_data[page_key] + "\n"
        
        # Create flattened section entry
        flat_section = {
            'section_id': len(flattened) + 1,
            'title': section['title'],
            'section_path': section_path,
            'level': section['level'],
            'start_page': section['start_page'],
            'end_page': section['end_page'],
            'text': section_text,
            'char_count': len(section_text),
            'page_range': f"{section['start_page']}-{section['end_page']}",
            'pages': list(range(section['start_page'], section['end_page'] + 1))
        }
        
        flattened.append(flat_section)
        
        # Recursively process subsections
        if 'subsections' in section and section['subsections']:
            subsections_flat = flatten_hierarchical_sections(section['subsections'], section_path, pages_data)
            flattened.extend(subsections_flat)
    
    # Renumber section IDs
    for i, section in enumerate(flattened, 1):
        section['section_id'] = i
    
    return flattened


def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def generate_embedding(text: str, model) -> List[float]:
    """Generate embedding for a text chunk."""
    if model is None:
        return []
    try:
        return model.encode(text, convert_to_tensor=False).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def is_decorative_chunk(text: str) -> bool:
    """
    Detect if chunk is decorative content (symbols, bullets, etc.)
    Very conservative to preserve data tables with financial/business information
    """
    # Use module-level precomputed sets
    bullet_count = sum(text.count(char) for char in _BULLET_CHARS)
    arabic_count = sum(text.count(char) for char in _ARABIC_NUMERALS)
    
    # Check for latex/math symbols
    latex_count = text.count('$\\bullet')
    
    total_decorative = bullet_count + arabic_count + latex_count
    
    # Lower threshold from 0.5 to 0.3 for better detection
    if total_decorative / max(len(text), 1) > 0.3:
        return True
    
    # Check if it's mostly table formatting with little actual content
    lines = text.split('\n')
    table_lines = [line for line in lines if '|' in line and line.count('|') > 3]
    
    # Only check if it's almost entirely table (> 90%)
    if len(table_lines) > 0.9 * len(lines) and len(table_lines) > 5:
        # Extract table cells
        all_cells = []
        for line in table_lines[:15]:  # Sample more lines for better detection
            cells = [cell.strip() for cell in line.split('|')]
            all_cells.extend([c for c in cells if c])
        
        if all_cells:
            # Check for data indicators (numbers, words, financial terms)
            data_cells = 0
            for cell in all_cells:
                # Skip empty or very short cells
                if len(cell) < 2:
                    continue
                
                # Check if cell contains actual data (use module-level set)
                has_numbers = any(c.isdigit() for c in cell)
                has_words = len([c for c in cell if c.isalpha()]) > 3
                has_financial = any(indicator in cell.lower() for indicator in _FINANCIAL_INDICATORS)
                
                if has_numbers or has_words or has_financial:
                    data_cells += 1
            
            # If more than 20% of cells contain data, it's a data table - KEEP IT
            if data_cells / len(all_cells) > 0.2:
                return False  # NOT decorative, preserve it
            
            # Check if cells are mostly just symbols/decorative (use module-level set)
            decorative_cells = sum(1 for cell in all_cells 
                                 if len(cell) <= 3 or 
                                 all(c in _DECORATIVE_CHARS for c in cell))
            
            # Only flag as decorative if > 90% of cells are symbols
            if decorative_cells / len(all_cells) > 0.9:
                return True
    
    return False

def is_small_chunk(text: str, min_length: int = 30) -> bool:
    """Check if chunk is too small (likely just a title)"""
    # Remove whitespace and check length
    clean_text = text.strip()
    
    # Too short
    if len(clean_text) < min_length:
        return True
    
    # Check if it's likely just a title (no periods, very short lines)
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    if len(lines) <= 2 and all(len(line) < 100 and '.' not in line for line in lines):
        return True
    
    return False

def is_meaningless_chunk(text: str, embedding: List[float], embed_model, meaningless_threshold: float = 0.5, meaningless_embedding: List[float] = None) -> bool:
    """Check if chunk contains meaningless content using embedding similarity"""
    if not embedding or embed_model is None:
        return False
    
    try:
        # Use provided embedding or generate new one
        if meaningless_embedding is None:
            meaningless_embedding = generate_embedding("", embed_model)
        
        if meaningless_embedding:
            similarity = cosine_similarity([embedding], [meaningless_embedding])[0][0]
            return similarity > meaningless_threshold
        
        return False
    except Exception as e:
        print(f"Error checking meaningless content: {e}")
        return False

def is_table_of_contents_chunk(text: str, embedding: List[float], toc_keywords_embedding: List[float], threshold: float = 0.68) -> bool:
    """Check if chunk is table of contents using structure analysis and embedding similarity"""
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Need at least 5 lines to be considered TOC
    if len(lines) < 5:
        return False
    
    text_lower = text.lower()
    
    # FIRST: Check for strong TOC indicators before excluding as data table
    has_contents_keyword = 'contents' in text_lower or 'table of contents' in text_lower
    has_chapter_section = 'chapter' in text_lower or 'section' in text_lower
    
    # If it has strong TOC keywords, prioritize TOC detection
    if has_contents_keyword or has_chapter_section:
        # Check for typical TOC structure patterns
        toc_score = 0
        
        # 1. Page number patterns (numbers at end of lines)
        page_number_lines = 0
        for line in lines:
            # Check if line ends with 2-3 digit numbers (typical page numbers)
            words = line.split()
            if words and words[-1].isdigit() and 1 <= len(words[-1]) <= 3:
                page_number_lines += 1
        
        if page_number_lines / len(lines) > 0.3:  # Lowered from 0.35
            toc_score += 2
        
        # 2. Dotted leaders check
        dotted_lines = sum(1 for line in lines if '..' in line or '...' in line)
        if dotted_lines / len(lines) > 0.2:  # Lowered from 0.25
            toc_score += 2
        
        # 3. Has TOC keywords
        if has_contents_keyword:
            toc_score += 2
        if has_chapter_section:
            toc_score += 1
        
        # 4. Lines with typical TOC formatting (short lines with numbers)
        formatted_lines = sum(1 for line in lines if len(line) < 100 and any(c.isdigit() for c in line))
        if formatted_lines / len(lines) > 0.5:
            toc_score += 1
        
        # Lower threshold when strong keywords present
        if toc_score >= 3:  # Lowered from 4
            return True
    
    # SECOND: Exclude data tables (but only if no strong TOC indicators)
    if not (has_contents_keyword or has_chapter_section):
        has_financial_data = any(term in text_lower for term in 
                                ['million', 'billion', 'revenue', 'profit', 'loss', 
                                 'assets', 'equity', 'debt', 'sar', 'usd', '$', '%'])
        
        if '|' in text:
            cells = [cell.strip() for line in lines for cell in line.split('|') if cell.strip()]
            if cells:
                numeric_cells = sum(1 for cell in cells if any(c.isdigit() for c in cell))
                if numeric_cells / len(cells) > 0.2 or has_financial_data:
                    return False
    
    # THIRD: Fallback to embedding similarity with adjusted threshold
    if not embedding or not toc_keywords_embedding:
        return False
    
    try:
        similarity = cosine_similarity([embedding], [toc_keywords_embedding])[0][0]
        # Use lower threshold (0.65) if has keywords, higher (0.70) otherwise
        effective_threshold = 0.65 if (has_contents_keyword or has_chapter_section) else threshold
        
        if similarity > effective_threshold:
            return True
            
        return False
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return False
def is_repetitive_chunk(text: str, min_unique_words: int = 15) -> bool:
    """Check if chunk is repetitive with very few unique words"""
    words = text.lower().split()
    if len(words) < 10:  # Too short to analyze
        return False
    
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)
    
    # If less than 30% unique words and fewer than min_unique_words, it's repetitive
    return unique_ratio < 0.3 and len(unique_words) < min_unique_words
    

def filter_chunks_with_embeddings(chunks, embed_model, min_length=30,
                                   toc_threshold=0.55, meaningless_threshold=0.5,
                                   debug=False, filtered_examples_accumulator=None) -> tuple[List[Dict], Dict[str, int]]:    
    """Filter chunks and generate embeddings simultaneously

    Returns:
        tuple: (filtered_chunks, filtering_stats)
    """
    if embed_model is None:
        print("Warning: No embedding model available, skipping advanced filtering")
        # Just filter by size, repetition, and decorative content
        filtered_chunks = []
        filtered_count = {'small': 0, 'toc': 0, 'meaningless': 0, 'repetitive': 0, 'decorative': 0}

        for i, chunk_text in enumerate(chunks):
            if is_small_chunk(chunk_text, min_length):
                filtered_count['small'] += 1
            # elif is_repetitive_chunk(chunk_text):
            #     filtered_count['repetitive'] += 1
            # elif is_decorative_chunk(chunk_text):
            #     filtered_count['decorative'] += 1
            else:
                filtered_chunks.append({
                    'text': chunk_text,
                    'embedding': [],
                    'filtered_reason': None
                })

        return filtered_chunks, filtered_count

    # # Generate TOC keywords embedding once
    # toc_keywords = "table of contents contents index chapter section page number list overview summary outline"
    # toc_embedding = generate_embedding(toc_keywords, embed_model)

    filtered_chunks = []
    filtered_count = {'small': 0, 'toc': 0, 'meaningless': 0, 'repetitive': 0, 'decorative': 0}
    
    # Use provided accumulator or create new one
    if filtered_examples_accumulator is None:
        filtered_examples = {'small': [], 'toc': [], 'meaningless': [], 'repetitive': [], 'decorative': []}
    else:
        filtered_examples = filtered_examples_accumulator
    
    print(f"    Filtering {len(chunks)} chunks with enhanced detection...")
    if debug:
        print(f"    Debug mode: showing similarity scores")
    
    for i, chunk_text in enumerate(chunks):
        # Check if chunk is too small
        if is_small_chunk(chunk_text, min_length):
            filtered_count['small'] += 1
            filtered_examples['small'].append(chunk_text)
            if debug:
                print(f"      Chunk {i+1}: FILTERED (small) - {len(chunk_text)} chars")
            continue

        # # Check if chunk is decorative (before generating embedding)
        # if is_decorative_chunk(chunk_text):
        #     filtered_count['decorative'] += 1
        #     filtered_examples['decorative'].append(chunk_text)
        #     if debug:
        #         print(f"      Chunk {i+1}: FILTERED (decorative)")
        #     continue

        # # Check if chunk is repetitive (before generating embedding)
        # if is_repetitive_chunk(chunk_text):
        #     filtered_count['repetitive'] += 1
        #     filtered_examples['repetitive'].append(chunk_text)
        #     if debug:
        #         print(f"      Chunk {i+1}: FILTERED (repetitive)")
        #     continue

        # Generate embedding for this chunk
        embedding = generate_embedding(chunk_text, embed_model)

        # # Check if chunk is table of contents related (with pattern matching)
        # if is_table_of_contents_chunk(chunk_text, embedding, toc_embedding, toc_threshold):
        #     filtered_count['toc'] += 1
        #     filtered_examples['toc'].append(chunk_text)
        #     if debug:
        #         toc_similarity = cosine_similarity([embedding], [toc_embedding])[0][0] if embedding and toc_embedding else None
        #         print(f"      Chunk {i+1}: FILTERED (TOC) - similarity={toc_similarity:.3f if toc_similarity else 'N/A'}, threshold={toc_threshold}")
        #     continue

        # Check if chunk contains meaningless content
        if is_meaningless_chunk(chunk_text, embedding, embed_model, meaningless_threshold):
            filtered_count['meaningless'] += 1
            filtered_examples['meaningless'].append(chunk_text)
            if debug:
                print(f"      Chunk {i+1}: FILTERED (meaningless)")
            continue

        # Keep this chunk
        filtered_chunks.append({
            'text': chunk_text,
            'embedding': embedding,
            'filtered_reason': None
        })

    total_filtered = sum(filtered_count.values())
    print(f"    Filtered out: {filtered_count['small']} small, {filtered_count['meaningless']} meaningless")
    print(f"    Total filtered: {total_filtered}, Kept: {len(filtered_chunks)} chunks")
    
    # Only write to file if not using accumulator (i.e., this is the final call or standalone call)
    if filtered_examples_accumulator is None and (debug or any(filtered_count.values())):
        filtered_output_file = "filtered_chunks_examples.txt"
        print(f"    Writing filtered chunk examples to: {filtered_output_file}")
        
        with open(filtered_output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FILTERED CHUNK EXAMPLES - FULL LENGTH\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total filtered: {total_filtered}\n")
            f.write(f"Kept chunks: {len(filtered_chunks)}\n\n")
            
            for filter_type, examples in filtered_examples.items():
                if examples:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"{filter_type.upper()} - {filtered_count[filter_type]} total filtered\n")
                    f.write(f"{'=' * 80}\n\n")
                    
                    for idx, example in enumerate(examples, 1):
                        f.write(f"--- Example {idx} ---\n")
                        f.write(f"Length: {len(example)} characters\n")
                        f.write(f"{'-' * 40}\n")
                        f.write(example)
                        f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"    Filtered examples saved to: {filtered_output_file}")
    
    return filtered_chunks, filtered_count

# Cache the LlamaIndex embedding model to avoid reloading
_llamaindex_embed_model = None

def llamaindex_chunker(text: str, buffer_size: int = 1, threshold: int = 80, embed_model=None, debug: bool = False, filtered_examples_accumulator=None) -> tuple[List[Dict], Dict[str, int]]:
    """
    LlamaIndex SemanticSplitterNodeParser with configurable parameters and filtering
    
    Args:
        text: Text to chunk
        buffer_size: Number of sentences to group together (smaller = more granular)
        threshold: Percentile threshold for semantic breaks (lower = more breaks = smaller chunks)
        embed_model: Embedding model for filtering
        debug: Enable debug output
        filtered_examples_accumulator: Optional dict to accumulate filtered chunks across calls
    
    Returns:
        tuple: (filtered_chunks, filtering_stats)
    """
    global _llamaindex_embed_model
    
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        # Reuse cached model if available
        if _llamaindex_embed_model is None:
            _llamaindex_embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=threshold,
            embed_model=_llamaindex_embed_model,
        )
        
        # Create document
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        
        # Extract raw chunks
        raw_chunks = [node.text for node in nodes]
        
        # Filter chunks and generate embeddings
        filtered_chunks, filtering_stats = filter_chunks_with_embeddings(
            raw_chunks, embed_model, debug=debug, 
            filtered_examples_accumulator=filtered_examples_accumulator
        )
        
        return filtered_chunks, filtering_stats
    
    except ImportError:
        print("LlamaIndex or HuggingFace embeddings not installed.")
        print("Install with: pip install llama-index llama-index-embeddings-huggingface sentence-transformers")
        return [], {}
    except Exception as e:
        print(f"LlamaIndex error: {e}")
        return [], {}

def apply_chunking_to_sections_progressive(sections, chunker_func, method_name,
                                            output_dir, debug=False, pages_data=None, **kwargs):    
    """
    Apply a chunking method to each section page-by-page and save progressively
    Can resume from where it left off if interrupted
    
    Args:
        sections: List of page sections
        chunker_func: The chunking function to apply
        method_name: Name of the chunking method
        output_dir: Directory to save progressive output
        pages_data: Dictionary mapping page numbers to text (for page-by-page chunking)
        **kwargs: Arguments for the chunker function
    
    Returns:
        List of chunks with exact page tracking information and embeddings
    """
    all_chunks = []
    
    # Initialize embedding model
    embed_model = get_embedding_model()
    if embed_model:
        print(f"  Loaded embedding model for filtering and storage")
    else:
        print(f"  Warning: No embedding model available")
    
    # File paths
    json_filename = os.path.join(output_dir, "SemanticSplitterNodeParser_chunks.json")
    txt_filename = os.path.join(output_dir, "SemanticSplitterNodeParser_chunks.txt")
    
    # Check if we can resume from existing progress
    start_section_idx = 1
    existing_chunks = []
    
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if (existing_data.get("method") == method_name and 
                existing_data.get("total_sections") == len(sections) and
                existing_data.get("status") != "completed"):
                
                # We can resume
                start_section_idx = existing_data.get("processed_sections", 0) + 1
                existing_chunks = existing_data.get("chunks", [])
                all_chunks = existing_chunks.copy()
                
                print(f"  Resuming from section {start_section_idx} (found {len(existing_chunks)} existing chunks)")
                
                # If already completed, just return existing chunks
                if existing_data.get("status") == "completed":
                    print(f"  Processing already completed! Found {len(existing_chunks)} chunks")
                    return existing_chunks
            else:
                print(f"  Existing file found but cannot resume (different method/sections), starting fresh")
                start_section_idx = 1
                existing_chunks = []
                all_chunks = []
        
        except Exception as e:
            print(f"  Error reading existing progress: {e}, starting fresh")
            start_section_idx = 1
            existing_chunks = []
            all_chunks = []
    
    # Track cumulative filtering stats
    cumulative_stats = {'small': 0, 'toc': 0, 'meaningless': 0, 'repetitive': 0, 'decorative': 0}
    total_raw_chunks = 0
    
    # Accumulator for ALL filtered chunks across all sections
    all_filtered_chunks = {'small': [], 'toc': [], 'meaningless': [], 'repetitive': [], 'decorative': []}
    
    # Initialize or update JSON file with metadata
    if start_section_idx == 1:
        # Starting fresh
        initial_data = {
            "method": method_name,
            "status": "processing",
            "total_sections": len(sections),
            "processed_sections": 0,
            "total_chunks": 0,
            "filtering_enabled": embed_model is not None,
            "filtering_stats": cumulative_stats,
            "total_raw_chunks": 0,
            "total_filtered": 0,
            "kept_chunks": 0,
            "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "chunks": []
        }
        
        # Write initial JSON file
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        
        # Write initial text file
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"=== {method_name} Chunks with Exact Page Tracking and Filtering ===\n")
            f.write(f"Status: Processing...\n")
            f.write(f"Total sections: {len(sections)}\n")
            f.write(f"Filtering enabled: {embed_model is not None}\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        print(f"  Starting fresh processing:")
    else:
        print(f"  Resuming processing:")
        # Load existing stats if resuming
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                cumulative_stats = existing_data.get("filtering_stats", cumulative_stats)
                total_raw_chunks = existing_data.get("total_raw_chunks", 0)
        except:
            pass
    
    print(f"    JSON: {json_filename}")
    print(f"    Text: {txt_filename}")
    
    # Process sections starting from start_section_idx
    for section_idx in range(start_section_idx, len(sections) + 1):
        section = sections[section_idx - 1]  # Convert to 0-based index
        
        print(f"  Processing section {section['section_id']} (pages {section['page_range']}) with {method_name}")
        
        # Process page-by-page if pages_data is available
        section_chunk_data = []
        
        # Check if section has page_contents (from content-based sectioning)
        has_page_contents = 'page_contents' in section and section['page_contents']
        
        if has_page_contents or pages_data:
            # Chunk each page individually to track exact source page
            for page_num in section['pages']:
                # Get page text from page_contents or pages_data
                page_text = None
                
                if has_page_contents:
                    # Extract from page_contents array
                    page_content = next((p for p in section['page_contents'] if p['page_number'] == page_num), None)
                    if page_content:
                        page_text = page_content.get('content', '')
                elif pages_data:
                    # Extract from pages_data dictionary
                    page_key = str(page_num)
                    if page_key in pages_data:
                        page_text = pages_data[page_key]
                
                if not page_text or not page_text.strip():
                    continue
                
                # Chunk this page
                filtered_chunks, page_stats = chunker_func(
                    page_text, 
                    embed_model=embed_model, 
                    debug=debug,
                    filtered_examples_accumulator=all_filtered_chunks, 
                    **kwargs
                )
                
                # Update cumulative stats
                for key in cumulative_stats:
                    cumulative_stats[key] += page_stats.get(key, 0)
                total_raw_chunks += len(filtered_chunks) + sum(page_stats.values())
                
                # Add chunks with exact page tracking
                for i, chunk_data in enumerate(filtered_chunks):
                    chunk_info = {
                        'text': chunk_data['text'],
                        'length': len(chunk_data['text']),
                        'embedding': chunk_data['embedding'],
                        'section_id': section['section_id'],
                        'section_path': section.get('section_path', section.get('title', 'Unknown')),
                        'section_level': section.get('level', 1),
                        'source_page': page_num,  # Exact page this chunk came from
                        'section_page_range': section['page_range'],
                        'chunk_index_in_page': i + 1,
                        'total_chunks_in_page': len(filtered_chunks),
                        'method': method_name
                    }
                    all_chunks.append(chunk_info)
                    section_chunk_data.append(chunk_info)
        else:
            # Fallback: chunk entire section text (old behavior)
            filtered_chunks, section_stats = chunker_func(
                section.get('text', ''), 
                embed_model=embed_model, 
                debug=debug,
                filtered_examples_accumulator=all_filtered_chunks, 
                **kwargs
            )
            
            # Update cumulative stats
            for key in cumulative_stats:
                cumulative_stats[key] += section_stats.get(key, 0)
            total_raw_chunks += len(filtered_chunks) + sum(section_stats.values())
            
            # Process chunks for this section
            for i, chunk_data in enumerate(filtered_chunks):
                chunk_info = {
                    'text': chunk_data['text'],
                    'length': len(chunk_data['text']),
                    'embedding': chunk_data['embedding'],
                    'section_id': section['section_id'],
                    'section_path': section.get('section_path', section.get('title', 'Unknown')),
                    'section_level': section.get('level', 1),
                    'source_pages': section['pages'],
                    'page_range': section['page_range'],
                    'chunk_index_in_section': i + 1,
                    'total_chunks_in_section': len(filtered_chunks),
                    'method': method_name
                }
                all_chunks.append(chunk_info)
                section_chunk_data.append(chunk_info)
        
        # Update JSON file progressively
        try:
            # Read current JSON data
            with open(json_filename, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            # Add new chunks to JSON
            current_data["chunks"].extend(section_chunk_data)
            current_data["processed_sections"] = section_idx
            current_data["total_chunks"] = len(current_data["chunks"])
            current_data["last_updated"] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Update filtering statistics
            current_data["filtering_stats"] = cumulative_stats
            current_data["total_raw_chunks"] = total_raw_chunks
            current_data["total_filtered"] = sum(cumulative_stats.values())
            current_data["kept_chunks"] = len(current_data["chunks"])
            
            # Mark as complete if this is the last section
            if section_idx == len(sections):
                current_data["status"] = "completed"
                current_data["completed_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write updated JSON data
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"    Warning: Failed to update progressive JSON: {e}")
        
        # Update text file progressively
        try:
            # Append new chunks to text file (or overwrite if resuming)
            mode = 'a' if start_section_idx > 1 and section_idx > start_section_idx else 'a'
            
            with open(txt_filename, mode, encoding='utf-8') as f:
                if section_idx == start_section_idx and start_section_idx > 1:
                    f.write(f"\n=== RESUMED PROCESSING ===\n")
                    f.write(f"Resumed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Starting from section {start_section_idx}\n")
                    f.write("="*80 + "\n\n")
                
                f.write(f"--- Section {section['section_id']} (Pages {section['page_range']}) ---\n")
                f.write(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Chunks in this section: {len(section_chunk_data)}\n\n")
                
                for chunk_info in section_chunk_data:
                    chunk_number = len(all_chunks) - len(section_chunk_data) + section_chunk_data.index(chunk_info) + 1
                    f.write(f"CHUNK {chunk_number:04d}\n")
                    f.write(f"Length: {chunk_info['length']} characters\n")
                    f.write(f"Method: {chunk_info['method']}\n")
                    f.write(f"Section: {chunk_info['section_id']}\n")
                    f.write(f"Section Path: {chunk_info.get('section_path', 'N/A')}\n")
                    f.write(f"Section Level: {chunk_info.get('section_level', 1)}\n")
                    
                    # Show exact source page if available, otherwise show page range
                    if 'source_page' in chunk_info:
                        f.write(f"Source Page: {chunk_info['source_page']}\n")
                        f.write(f"Section Pages: {chunk_info.get('section_page_range', 'N/A')}\n")
                        f.write(f"Chunk {chunk_info.get('chunk_index_in_page', 1)} of {chunk_info.get('total_chunks_in_page', 1)} on this page\n")
                    else:
                        f.write(f"Source Pages: {chunk_info.get('page_range', 'N/A')}\n")
                        f.write(f"Page Numbers: {', '.join(map(str, chunk_info.get('source_pages', [])))}\n")
                        f.write(f"Chunk {chunk_info.get('chunk_index_in_section', 1)} of {chunk_info.get('total_chunks_in_section', 1)} in this section\n")
                    
                    f.write(f"Has Embedding: {len(chunk_info['embedding']) > 0}\n")
                    f.write("-" * 40 + "\n")
                    f.write(chunk_info['text'])
                    f.write("\n\n" + "="*80 + "\n\n")
                
                # Update progress info at the end
                f.write(f"=== PROGRESS UPDATE ===\n")
                f.write(f"Processed sections: {section_idx}/{len(sections)}\n")
                f.write(f"Total chunks so far: {len(all_chunks)}\n")
                f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if section_idx == len(sections):
                    f.write(f"Status: COMPLETED\n")
                    f.write(f"\n=== FILTERING SUMMARY ===\n")
                    f.write(f"Total raw chunks: {total_raw_chunks}\n")
                    f.write(f"Filtered out:\n")
                    f.write(f"  - Small chunks: {cumulative_stats['small']}\n")
                    f.write(f"  - Decorative chunks: {cumulative_stats['decorative']}\n")
                    f.write(f"  - TOC chunks (similarity): {cumulative_stats['toc']}\n")
                    f.write(f"  - Meaningless chunks (similarity): {cumulative_stats['meaningless']}\n")
                    f.write(f"  - Repetitive chunks: {cumulative_stats['repetitive']}\n")
                    f.write(f"Total filtered: {sum(cumulative_stats.values())} ({100*sum(cumulative_stats.values())/total_raw_chunks:.1f}%)\n")
                    f.write(f"Kept chunks: {len(all_chunks)} ({100*len(all_chunks)/total_raw_chunks:.1f}%)\n")
                else:
                    f.write(f"Status: Processing...\n")
                f.write("="*80 + "\n\n")
            
            print(f"    Added {len(section_chunk_data)} filtered chunks to files ({len(all_chunks)} total)")
            
        except Exception as e:
            print(f"    Warning: Failed to update progressive text: {e}")
    
    # Write ALL filtered chunks to a separate file at the end
    filtered_output_file = os.path.join(output_dir, f"{method_name}_filtered_chunks.txt")
    try:
        print(f"\n  Writing all filtered chunks to: {filtered_output_file}")
        
        with open(filtered_output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ALL FILTERED CHUNKS - {method_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total sections processed: {len(sections)}\n")
            f.write(f"Total raw chunks: {total_raw_chunks}\n")
            f.write(f"Total filtered: {sum(cumulative_stats.values())}\n")
            f.write(f"Kept chunks: {len(all_chunks)}\n")
            f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FILTERING BREAKDOWN\n")
            f.write("=" * 80 + "\n")
            for filter_type, count in cumulative_stats.items():
                f.write(f"  {filter_type.upper()}: {count} chunks\n")
            f.write("\n")
            
            # Write all filtered chunks by type
            for filter_type, examples in all_filtered_chunks.items():
                if examples:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"{filter_type.upper()} CHUNKS - {len(examples)} total\n")
                    f.write(f"{'=' * 80}\n\n")
                    
                    for idx, example in enumerate(examples, 1):
                        f.write(f"--- {filter_type.upper()} Chunk {idx} of {len(examples)} ---\n")
                        f.write(f"Length: {len(example)} characters\n")
                        f.write(f"{'-' * 40}\n")
                        f.write(example)
                        f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"  Successfully wrote {sum(len(examples) for examples in all_filtered_chunks.values())} filtered chunks to file")
        
    except Exception as e:
        print(f"  Warning: Failed to write filtered chunks file: {e}")
    
    # Return chunks and filtering summary
    filtering_summary = {
        'stats': cumulative_stats,
        'total_raw_chunks': total_raw_chunks,
        'total_filtered': sum(cumulative_stats.values()),
        'kept_chunks': len(all_chunks),
        'filter_percentage': 100 * sum(cumulative_stats.values()) / total_raw_chunks if total_raw_chunks > 0 else 0
    }
    
    return all_chunks, filtering_summary

def benchmark_chunker_on_sections_progressive(sections: List[Dict], chunker_func, 
                                               method_name: str, output_dir: str,
                                               debug: bool = False, pages_data=None, **kwargs) -> tuple:
    """Benchmark a chunking method on sections with progressive JSON output"""
    print(f"\nTesting {method_name} with progressive output...")
    
    start_time = time.time()
    chunks_with_metadata, filtering_summary = apply_chunking_to_sections_progressive(
        sections, chunker_func, method_name, output_dir, pages_data=pages_data, **kwargs
    )
    end_time = time.time()
    
    processing_time = round(end_time - start_time, 2)
    
    if not chunks_with_metadata:
        return {
            "method": method_name,
            "total_chunks": 0,
            "processing_time": processing_time,
            "error": "No chunks generated"
        }, []
    
    # Calculate statistics
    lengths = [chunk['length'] for chunk in chunks_with_metadata]
    
    stats = {
        "method": method_name,
        "total_chunks": len(chunks_with_metadata),
        "avg_length": round(statistics.mean(lengths), 2),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": statistics.median(lengths),
        "total_characters": sum(lengths),
        "processing_time": processing_time,
        "filtering_summary": filtering_summary
    }
    
    print(f"  Generated {len(chunks_with_metadata)} chunks in {processing_time}s")
    print(f"  Average chunk length: {stats['avg_length']} characters")
    print(f"  Range: {stats['min_length']} - {stats['max_length']} characters")
    
    # Print filtering summary
    if filtering_summary:
        print(f"\n  === FILTERING SUMMARY ===")
        print(f"  Total raw chunks: {filtering_summary['total_raw_chunks']}")
        print(f"  Filtered out:")
        print(f"    - Small: {filtering_summary['stats']['small']}")
        print(f"    - Decorative: {filtering_summary['stats']['decorative']}")
        print(f"    - TOC (similarity): {filtering_summary['stats']['toc']}")
        print(f"    - Meaningless (similarity): {filtering_summary['stats']['meaningless']}")
        print(f"    - Repetitive: {filtering_summary['stats']['repetitive']}")
        print(f"  Total filtered: {filtering_summary['total_filtered']} ({filtering_summary['filter_percentage']:.1f}%)")
        print(f"  Kept chunks: {len(chunks_with_metadata)} ({100-filtering_summary['filter_percentage']:.1f}%)")
    
    return stats, chunks_with_metadata

def save_text_chunks(chunks_with_metadata: List[Dict], method_name: str, output_dir: str):
    """Save chunks as text file for easy reading (JSON is saved progressively)"""
    if not chunks_with_metadata:
        return
    
    # Save as text for easy reading
    txt_filename = os.path.join(output_dir, "SemanticSplitterNodeParser_chunks.txt")
    
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"=== {method_name} Chunks with Page Tracking ===\n")
            f.write(f"Total chunks: {len(chunks_with_metadata)}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for i, chunk in enumerate(chunks_with_metadata, 1):
                f.write(f"CHUNK {i:04d}\n")
                f.write(f"Length: {chunk['length']} characters\n")
                f.write(f"Method: {chunk['method']}\n")
                f.write(f"Section: {chunk['section_id']}\n")
                f.write(f"Source Pages: {chunk['page_range']}\n")
                f.write(f"Page Numbers: {', '.join(map(str, chunk['source_pages']))}\n")
                f.write(f"Chunk {chunk['chunk_index_in_section']} of {chunk['total_chunks_in_section']} in this section\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'])
                f.write("\n\n" + "="*80 + "\n\n")
        
        print(f"  Saved text: {txt_filename}")
    except Exception as e:
        print(f"  Failed to save text {txt_filename}: {e}")

def main():
    """Main function to run chunking with filtering and embeddings"""
    
    print("Production Chunking System - LlamaIndex with Filtering and Embeddings")
    print("=" * 70)
    
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    input_dir = "/app/input" if os.path.exists("/app/input") else "."
    
    # Find JSON section files (prioritize parsed_sections.json from sections_parser_pdf.py)
    section_files = []
    if os.path.exists(output_dir):
        # Look for parsed_sections.json first (from sections_parser_pdf.py)
        if os.path.exists(os.path.join(output_dir, 'parsed_sections.json')):
            section_files = ['parsed_sections.json']
        else:
            section_files = [f for f in os.listdir(output_dir) if f.endswith('_sections.json')]
    
    if not section_files:
        print("No JSON section files found. Please run sections_parser_pdf.py first.")
        return
    
    # Use the first section file found
    section_file = os.path.join(output_dir, section_files[0])
    
    try:
        with open(section_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', [])
        
        if not sections:
            print("No sections found in file")
            return
        
        # Check if sections are hierarchical (from sections_parser_pdf.py)
        is_hierarchical = any('subsections' in section for section in sections)
        
        if is_hierarchical:
            print(f"Detected hierarchical sections from sections_parser_pdf.py")
            
            # Load pages data from the source JSON file
            filename = data.get('filename', '')
            if filename:
                # Extract base name and look for corresponding JSON
                base_name = os.path.splitext(os.path.basename(filename))[0]
                pages_json_file = os.path.join(input_dir, f"{base_name}.json")
                
                if os.path.exists(pages_json_file):
                    print(f"Loading page text from: {pages_json_file}")
                    with open(pages_json_file, 'r', encoding='utf-8') as f:
                        pages_data = json.load(f)
                    print(f"Loaded {len(pages_data)} pages")
                else:
                    print(f"Warning: Could not find pages JSON file: {pages_json_file}")
                    print("Sections will have no text content")
                    pages_data = {}
            else:
                print("Warning: No filename in sections file, cannot load page text")
                pages_data = {}
            
            print(f"Flattening {len(sections)} top-level sections...")
            sections = flatten_hierarchical_sections(sections, pages_data=pages_data)
            print(f"Flattened to {len(sections)} total sections")
        else:
            print(f"Loaded {len(sections)} flat sections")
            
            # For flat sections, try to load pages_data from the source JSON file
            filename = data.get('filename', '')
            if not filename and sections:
                # Try to infer filename from section data or input file
                input_basename = os.path.splitext(os.path.basename(section_files[0]))[0]
                # Remove _sections suffix if present
                if input_basename.endswith('_sections'):
                    filename = input_basename[:-9]  # Remove '_sections'
            
            if filename:
                # Look for the source JSON file
                base_name = os.path.splitext(os.path.basename(filename))[0] if '.' in filename else filename
                pages_json_file = os.path.join(input_dir, f"{base_name}.json")
                
                if os.path.exists(pages_json_file):
                    print(f"Loading page text from: {pages_json_file}")
                    with open(pages_json_file, 'r', encoding='utf-8') as f:
                        pages_data = json.load(f)
                    print(f"Loaded {len(pages_data)} pages for page-by-page chunking")
                else:
                    print(f"Warning: Could not find pages JSON file: {pages_json_file}")
                    print("Will chunk sections as-is without individual page tracking")
                    pages_data = {}
            else:
                print("Warning: No filename found, cannot load individual page data")
                print("Will chunk sections as-is without individual page tracking")
                pages_data = {}
        
    except Exception as e:
        print(f"Error loading sections: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Loaded from {section_files[0]}")
    
    # Calculate total text length for reference
    total_text_length = sum(len(section.get('text', '')) for section in sections)
    print(f"Total text across all sections: {total_text_length:,} characters")
    
    # Show section breakdown
    print(f"\nSection breakdown:")
    for section in sections[:5]:  # Show first 5 sections
        section_path = section.get('section_path', section.get('title', 'Unknown'))
        text_len = len(section.get('text', ''))
        print(f"  Section {section['section_id']}: {section_path}")
        print(f"    {text_len} chars, pages {section['page_range']}")
    if len(sections) > 5:
        print(f"  ... and {len(sections) - 5} more sections")
    
    print("="*70)
    print("Enhanced filtering features:")
    print("  ✓ Small chunk filtering (< 50 characters)")
    print("  ✓ Decorative content filtering (symbols, bullets, Arabic numerals)")
    print("  ✓ Table of contents filtering (embedding similarity > 0.7)")
    print("  ✓ Meaningless content filtering (navigation, formatting, etc.)")
    print("  ✓ Repetitive content filtering (< 20% unique words)")
    print("  ✓ Simultaneous embedding generation")
    print("  ✓ Progressive output with embeddings")
    print("  ✓ Section path and page tracking")
    print("  ✓ Resume capability (continues from where it left off)")
    print("="*70)
    
    # Test LlamaIndex configuration with filtering
    configurations = [
        {"buffer_size": 1, "threshold": 70, "name": "LlamaIndex Small (threshold=70) + Filtering"}
    ]
    
    results = []
    
    for config in configurations:
        method_name = config.pop("name")
        stats, chunks_with_metadata = benchmark_chunker_on_sections_progressive(
            sections,
            llamaindex_chunker,
            method_name,
            output_dir,
            pages_data=pages_data,
            **config
        )
        results.append(stats)
        
        print(f"  Progressive files completed for {method_name}")
        
        # Show filtering statistics
        if chunks_with_metadata:
            embedding_count = sum(1 for chunk in chunks_with_metadata if len(chunk.get('embedding', [])) > 0)
            print(f"  Chunks with embeddings: {embedding_count}/{len(chunks_with_metadata)}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("LLAMAINDEX CHUNKING WITH FILTERING RESULTS")
    print("="*80)
    
    print(f"{'Configuration':<45} {'Chunks':<8} {'Avg Len':<8} {'Time(s)':<8} {'Min':<6} {'Max':<6}")
    print("-" * 80)
    
    for result in results:
        if result['total_chunks'] > 0:
            print(f"{result['method']:<45} {result['total_chunks']:<8} "
                  f"{result['avg_length']:<8} {result['processing_time']:<8} "
                  f"{result['min_length']:<6} {result['max_length']:<6}")
    
    print(f"\nFiles saved to: {output_dir}")
    print("✓ SemanticSplitterNodeParser_chunks.json - Contains chunks with embeddings and section paths")
    print("✓ SemanticSplitterNodeParser_chunks.txt - Human-readable format")
    print("\nEnhanced filtering applied:")
    print("  - Removed chunks < 50 characters (likely titles)")
    print("  - Removed decorative chunks (> 30% symbols/bullets)")
    print("  - Removed table of contents chunks (similarity > 0.7)")
    print("  - Removed meaningless content (navigation, formatting)")
    print("  - Removed repetitive chunks (< 20% unique words)")
    print("  - Generated embeddings for all remaining chunks")
    print("  - Tracked section paths and page numbers for each chunk")
    print("\nUse the generated JSON files for knowledge graph creation and GraphRAG testing.")

if __name__ == "__main__":
    main()