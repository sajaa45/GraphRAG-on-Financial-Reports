# HTML Processing Pipeline

This pipeline processes HTML files (like SEC filings) and extracts sections, chunks them semantically, and stores them in a vector database for retrieval.

## Overview

The pipeline consists of three main steps:

1. **Parse HTML** - Extract sections and create logical page breaks
2. **Chunk Content** - Split sections into semantic chunks using LlamaIndex
3. **Store Vectors** - Embed and store chunks in Qdrant vector database

## Features

- ✅ Extracts sections from HTML (PART I, PART II, Item 1, Item 2, etc.)
- ✅ Uses actual page breaks from HTML (BRPFPageHeader divs)
- ✅ Preserves real page number sources for each chunk
- ✅ Semantic chunking using LlamaIndex SemanticSplitterNodeParser
- ✅ Parallel processing for faster chunking
- ✅ Stores embeddings in Qdrant for semantic search

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- beautifulsoup4
- llama-index
- sentence-transformers
- qdrant-client

## Usage

### Quick Start

Process an HTML file with default settings:

```bash
python pipeline_html.py input/NYSE_MTX_2024.htm
```

### Full Pipeline

```bash
python pipeline_html.py input/NYSE_MTX_2024.htm \
    --output-dir output \
    --collection financial_docs \
    --workers 4 \
    --threshold 70
```

### Step-by-Step

#### 1. Parse HTML Only

```bash
python parsing/parsing_sections_html.py input/NYSE_MTX_2024.htm \
    --output output/parsed_sections_html.json \
    --page-index output/page_index_html.json
```

This creates:
- `parsed_sections_html.json` - Hierarchical sections with text
- `page_index_html.json` - Logical pages with text content

#### 2. Chunk and Store

```bash
python chunking/pipeline.py output/parsed_sections_html.json \
    --page-index output/page_index_html.json \
    --collection financial_docs \
    --workers 4
```

## Output Files

### parsed_sections_html.json
Contains hierarchical section structure:
```json
{
  "filename": "input/NYSE_MTX_2024.htm",
  "num_pages": 19,
  "num_sections": 30,
  "sections": [
    {
      "title": "PART I",
      "level": 1,
      "start_page": 1,
      "end_page": 5,
      "text": "...",
      "page_contents": [...]
    }
  ]
}
```

### page_index_html.json
Contains logical pages:
```json
{
  "source_html": "input/NYSE_MTX_2024.htm",
  "total_pages": 19,
  "pages": [
    {
      "page": 1,
      "text": "...",
      "word_count": 3000
    }
  ]
}
```

### SemanticSplitterNodeParser_chunks.json
Contains all chunks with metadata:
```json
{
  "chunks": [
    {
      "text": "...",
      "section_title": "Item 1. Business",
      "source_page": 3,
      "chunk_index": 1
    }
  ]
}
```

## Configuration Options

### Pipeline Options

- `--output-dir` - Directory for output files (default: `output`)
- `--collection` - Qdrant collection name (default: `financial_docs`)
- `--workers` - Number of parallel workers (default: `4`)
- `--clear` - Delete and recreate Qdrant collection
- `--no-chunk-files` - Skip saving chunk JSON/TXT files

### Chunking Options

- `--buffer-size` - LlamaIndex buffer size (default: `1`)
- `--threshold` - Semantic similarity threshold (default: `70`)

### Qdrant Options

- `--qdrant-host` - Qdrant server host (default: `localhost`)
- `--qdrant-port` - Qdrant server port (default: `6333`)

## How It Works

### 1. HTML Parsing

The parser:
1. Removes hidden XBRL data
2. Extracts all text content
3. Creates logical pages (~3000 words each)
4. Detects section headers (PART I, Item 1, etc.)
5. Assigns page ranges to each section

### 2. Semantic Chunking

For each section:
1. Extracts text from page contents
2. Uses LlamaIndex SemanticSplitterNodeParser
3. Creates chunks based on semantic similarity
4. Preserves page number metadata

### 3. Vector Storage

For each chunk:
1. Generates embeddings using sentence-transformers
2. Stores in Qdrant with metadata:
   - Section title and level
   - Source page number
   - Chunk index
   - Word count

## Comparison with PDF Pipeline

| Feature | PDF Pipeline | HTML Pipeline |
|---------|-------------|---------------|
| Page Detection | Real PDF pages | Logical pages (~3000 words) |
| Section Detection | TOC metadata | Pattern matching (PART, Item) |
| Text Extraction | PyMuPDF | BeautifulSoup |
| Chunking | Same (LlamaIndex) | Same (LlamaIndex) |
| Vector Storage | Same (Qdrant) | Same (Qdrant) |

## Example: Processing SEC 10-K Filing

```bash
# Process the HTML file
python pipeline_html.py input/NYSE_MTX_2024.htm

# Output:
# ✓ Parsed 30 sections from 19 logical pages
# ✓ Created 156 chunks
# ✓ Stored in Qdrant collection 'financial_docs'
```

## Troubleshooting

### No sections found
- Check if HTML contains PART/Item headers
- Verify HTML structure with BeautifulSoup

### Qdrant connection error
- Ensure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`
- Check host/port settings

### Out of memory
- Reduce `--workers` count
- Process smaller HTML files
- Increase system RAM

## Next Steps

After processing, you can:
1. Query the vector database for semantic search
2. Build a knowledge graph from the chunks
3. Use RAG (Retrieval Augmented Generation) for Q&A

## Related Files

- `parsing/parsing_sections.py` - PDF parser (similar structure)
- `chunking/chunker.py` - Chunking utilities
- `chunking/pipeline.py` - Vector storage pipeline
