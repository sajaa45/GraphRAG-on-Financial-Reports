# JSON Text Processing + Chunking Comparison + Knowledge Graph Pipeline

An automated Docker pipeline that processes JSON files with page-based text content, compares different chunking strategies, and builds Neo4j knowledge graphs.

##  Quick Start

### Automated Pipeline (Recommended)

1. **Place your JSON files in the project directory**
   - JSON should have page numbers as keys (e.g., "1", "2", "3")
   - Each page should contain text content as the value
2. **Run the complete pipeline:**
```bash
docker-compose up --build
```

**What it does automatically:**
-  Detects JSON files in your directory
-  Processes pages with smart chunking logic
-  Runs chunking comparison with 3 methods
-  Generates performance and quality metrics
-  Creates sample chunk files for inspection
-  Provides recommendations

### Knowledge Graph (Neo4j)

3. **Build knowledge graph from chunks:**
```bash
# Start Neo4j and build graph
docker-compose up neo4j knowledge-graph

# Or run locally (requires Neo4j running)
NEO4J_PASSWORD=your_password python lexical_wrapper_kg.py
```

**Knowledge Graph Features:**
-  **Page nodes** - Original document pages
-  **Section nodes** - Smart page combinations
-  **Chunk nodes** - LlamaIndex semantic chunks
-  **Relationships** - Page→Section→Chunk hierarchy
-  **Semantic links** - Next chunk, shared pages

### Manual Steps (Alternative)

1. **Test components first:**
```bash
python test_pipeline.py
```

2. **Process JSON files only:**
```bash
python sections_merging_pages.py
```

3. **Run chunking comparison only:**
```bash
python chunking_comparison.py
```

4. **Build knowledge graph only:**
```bash
python lexical_kG_building.py
```

##  JSON Format Expected

Your JSON file should look like this:
```json
{
  "1": "# Page 1 Title\n\nThis is the content of page 1...",
  "2": "This continues from page 1 without proper ending",
  "3": "# New Section\n\nThis starts a new section...",
  "4": "more content that continues..."
}
```

##  Smart Chunking Logic

The pipeline uses intelligent page combination rules:

**Pages are combined when:**
- Current page doesn't start with an uppercase letter
- AND previous page doesn't end with a period

**Example:**
- Page 1: "This is a sentence" (no period)
- Page 2: "that continues here." (lowercase start)
- → **Combined into one chunk**

- Page 3: "This is complete." (ends with period)  
- Page 4: "New section starts" (uppercase start)
- → **Separate chunks**

##  What You Get

### Performance Metrics
- Processing time for each method
- Number of chunks created
- Average chunk sizes
- Speed scores

### Quality Metrics
- Sentence completeness (how many chunks end properly)
- Paragraph preservation
- Size consistency
- Semantic coherence
- Content overlap ratios

### Output Files
- `*_chunks.txt` - Human-readable chunk analysis
- `*_chunks.json` - Programmatic access to chunks
- `chunking_methods_comparison.txt` - Side-by-side comparison summary

### Sample Files (3 chunks each method)
- `sample_chunks_langchain.txt` - LangChain examples
- `sample_chunks_llamaindex.txt` - LlamaIndex examples  
- `sample_chunks_chonkie.txt` - Chonkie examples

### Complete Files (ALL chunks from each method)
- `all_chunks_langchain.txt` - Every LangChain chunk
- `all_chunks_llamaindex.txt` - Every LlamaIndex chunk
- `all_chunks_chonkie.txt` - Every Chonkie chunk

### JSON Files (Programmatic access to all chunks)
- `all_chunks_langchain.json` - LangChain chunks as JSON
- `all_chunks_llamaindex.json` - LlamaIndex chunks as JSON
- `all_chunks_chonkie.json` - Chonkie chunks as JSON

##  GraphRAG System

The GraphRAG system uses your Neo4j knowledge graph for intelligent document retrieval and connects to local Ollama models for question answering.

### Four Retrieval Approaches

**Approach 1: Single Chunk**
- Finds the most semantically similar chunk
- Passes only that chunk to the LLM
- Fastest, most focused responses

**Approach 2: Sequential Chunks**  
- Finds the best chunk + the next chunk in sequence
- Uses Neo4j `NEXT_CHUNK` relationships
- Better context continuity

**Approach 3: Section-based**
- Finds the best chunk, then retrieves all chunks from that section
- Uses Neo4j `CONTAINS_CHUNK` relationships  
- Most comprehensive context

**Approach 4: Context Window**
- Finds the best chunk + previous chunk + next chunk
- Uses Neo4j `PREVIOUS_CHUNK` and `NEXT_CHUNK` relationships
- Balanced context with surrounding information

### Setup Requirements

**1. Neo4j Knowledge Graph**
```bash
# Make sure your knowledge graph is built first
docker-compose up neo4j knowledge-graph
```

**2. Ollama with Local Model**
```bash
# Install Ollama: https://ollama.ai
# Pull a model (e.g., Llama 3.2)
ollama pull mistral:lates

# Or other models:
ollama pull mistral
ollama pull codellama
```

### Running GraphRAG

**Option 1: Local Ollama + Docker GraphRAG (Recommended)**
```bash
# 1. Make sure Ollama is running locally
ollama pull mistral:latest

# 2. Run the automated setup
python run_graphrag_local_ollama.py
```

**Option 2: Full Docker Stack (Everything Containerized)**
```bash
# Runs Neo4j + Ollama + GraphRAG all in Docker
python run_graphrag_docker.py
```

**Option 3: Manual Docker Commands**
```bash
# With local Ollama
docker-compose up graphrag-test

# Or with containerized Ollama (after pulling models)
docker-compose up ollama neo4j knowledge-graph graphrag-test
```

**Option 4: Local Development**
```bash
# Set environment variables
export NEO4J_PASSWORD=Lexical12345
export OLLAMA_URL=http://localhost:11434

# Run the test
python test_graphrag.py
```

### Sample Test Queries

The system automatically tests these queries:
- "What are the main financial highlights?"
- "Who are the directors of the company?"  
- "What is the company's registered office address?"
- "What are the key business segments?"
- "What sustainability initiatives does the company have?"

### Output Files

**`graphrag_comparison_results.json`**
- Complete results from all approaches
- Response times, similarity scores
- Source page references
- Full LLM responses

### Performance Comparison

The system compares:
- **Response Quality**: How well each approach answers questions
- **Response Time**: Speed of each retrieval method
- **Context Relevance**: Similarity scores and source tracking
- **Chunk Usage**: Number of chunks used per approach

### Example Output

```
=== Approach 1: Single Chunk Retrieval ===
Best match: chunk_0042 (similarity: 0.847)
Pages: 15
Chunks used: 1
Response time: 2.3s

=== Approach 2: Sequential Chunks Retrieval ===  
Best match: chunk_0042 (similarity: 0.847)
Next chunk: chunk_0043
Chunks used: 2
Response time: 3.1s

=== Approach 3: Section-based Retrieval ===
Best match: chunk_0042 (similarity: 0.847)
Section section_15 has 8 chunks
Chunks used: 8
Response time: 4.7s
```

## 🕸️ Knowledge Graph Structure

The Neo4j knowledge graph creates a hierarchical representation of your document:

### Node Types

**📄 Page Nodes**
- Original document pages from JSON
- Properties: `page_number`, `content`, `length`, `content_hash`

**📑 Section Nodes** 
- Smart page combinations based on your logic
- Properties: `section_id`, `content`, `page_range`, `page_count`

**🧩 Chunk Nodes**
- LlamaIndex semantic chunks with metadata
- Properties: `chunk_id`, `content`, `method`, `chunk_index_in_section`

### Relationships

- `Page -[:BELONGS_TO_SECTION]-> Section`
- `Section -[:CONTAINS_CHUNK]-> Chunk`  
- `Page -[:HAS_CHUNK]-> Chunk`
- `Chunk -[:NEXT_CHUNK]-> Chunk` (sequential)
- `Section -[:SHARES_PAGE]-> Section` (multi-page sections)

### Sample Cypher Queries

**Find all chunks from page 3:**
```cypher
MATCH (p:Page {page_number: 3})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.content LIMIT 5
```

**Trace path from page to chunks:**
```cypher
MATCH path = (p:Page)-[:BELONGS_TO_SECTION]->(s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)
WHERE p.page_number = 1
RETURN path LIMIT 3
```

**Find multi-page sections:**
```cypher
MATCH (s:Section)
WHERE s.page_count > 1
RETURN s.section_id, s.page_range, s.page_count
```

**Get section statistics:**
```cypher
MATCH (s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)
RETURN s.section_id, s.page_range, count(c) as chunk_count,
       avg(c.length) as avg_chunk_length
ORDER BY chunk_count DESC LIMIT 10
```

### Neo4j Setup

**Using Docker (Recommended):**
```bash
# Start Neo4j + build graph
docker-compose up neo4j knowledge-graph

# Access Neo4j Browser: http://localhost:7474
# Username: neo4j, Password: knowledge123
```

**Local Setup:**
```bash
# Install Neo4j locally, then:
NEO4J_PASSWORD=your_password python lexical_wrapper_kg.py
```

##  Chunking Methods Compared

### 1. Custom JSON Page-Based Chunking
- **Best for**: Document structure preservation
- **Logic**: Smart page combination based on punctuation and capitalization
- **Use case**: Maintaining document flow and natural breaks

### 2. LangChain RecursiveCharacterTextSplitter
- **Best for**: Production RAG systems
- **Speed**: ⚡ Fast (10-15 seconds)
- **Quality**: Balanced, consistent sizes
- **Use case**: General purpose, fast processing

### 3. LlamaIndex SemanticSplitterNodeParser
- **Best for**: High-quality document analysis
- **Speed**: 🐌 Slow (150+ seconds)
- **Quality**: Highest (98%+ sentence completeness)
- **Use case**: Research, detailed analysis

### 4. Chonkie SemanticChunker
- **Best for**: Real-time semantic processing
- **Speed**: ⚡ Fast (20-25 seconds)
- **Quality**: Moderate, many small chunks
- **Use case**: Real-time applications

##  File Structure

```
.
├── pdf2.json                       # Your JSON file(s)
├── docker-compose.yml              # Easy pipeline execution + Neo4j
├── Dockerfile                      # Container definition
├── main_pipeline.py                # Automated pipeline
├── sections_merging_pages.py          # JSON → chunks conversion
├── chunking_comparison.py          # Chunking analysis
├── lexical_kG_building.py        # Knowledge graph builder
├── lexical_wrapper_kg.py        # Local KG builder script
├── requirements.txt                # Dependencies (includes neo4j)
├── output/                         # Generated files
│   ├── pdf2_sections.json         # Page sections with metadata
│   ├── pdf2_sections.txt          # Human-readable sections
│   ├── llamaindex_chunks_with_pages.json  # Chunks with page tracking
│   ├── langchain_chunks_with_pages.json   # LangChain chunks
│   ├── chonkie_chunks_with_pages.json     # Chonkie chunks
│   └── chunking_methods_comparison.txt    # Side-by-side comparison
├── sample_chunks_langchain.txt     # LangChain samples
├── sample_chunks_llamaindex.txt    # LlamaIndex samples
├── sample_chunks_chonkie.txt       # Chonkie samples
├── all_chunks_langchain.txt        # ALL LangChain chunks
├── all_chunks_llamaindex.txt       # ALL LlamaIndex chunks
└── all_chunks_chonkie.txt          # ALL Chonkie chunks
```

##  How to Examine the Full Chunks

After running the pipeline, you'll have several ways to examine the chunks:

### 1. Quick Overview
- **`chunking_methods_comparison.txt`** - Side-by-side comparison with first chunk from each method

### 2. Sample Inspection (Recommended first step)
- **`sample_chunks_langchain.txt`** - 3 representative chunks (beginning, middle, end)
- **`sample_chunks_llamaindex.txt`** - 3 representative chunks
- **`sample_chunks_chonkie.txt`** - 3 representative chunks

### 3. Complete Inspection (For detailed analysis)
- **`all_chunks_langchain.txt`** - Every single chunk from LangChain method
- **`all_chunks_llamaindex.txt`** - Every single chunk from LlamaIndex method  
- **`all_chunks_chonkie.txt`** - Every single chunk from Chonkie method

### 4. Programmatic Access
- **`all_chunks_*.json`** - JSON format for scripts/analysis tools

### Example: Examining LangChain chunks
```bash
# Quick look at sample chunks
cat sample_chunks_langchain.txt

# See all chunks (might be very long!)
cat all_chunks_langchain.txt

# Count total chunks
grep "^CHUNK" all_chunks_langchain.txt | wc -l

# See just chunk headers and lengths
grep -E "^CHUNK|^Length:" all_chunks_langchain.txt
```

##  Configuration

### For Different Document Types

**Annual Reports/Financial Documents:**
- JSON chunking preserves section boundaries
- Use LlamaIndex for highest quality on combined text

**General Documents:**
- Use LangChain with 512 chunk size
- 64 character overlap recommended

**Real-time Applications:**
- Use JSON chunking for structure preservation
- Use Chonkie for speed + semantic awareness

##  Troubleshooting

### Common Issues

**"No JSON files found"**
- Ensure JSON files are in the project directory
- Check file extensions (.json)

**"Failed to load data from JSON"**
- Verify JSON format is valid
- Ensure page numbers are strings ("1", "2", etc.)

**Import errors locally**
- Use Docker instead: `docker-compose up --build`
- Or install: `pip install -r requirements.txt`

### Docker Issues

**Build failures:**
- Try: `docker-compose down && docker-compose up --build`
- Check Docker has enough memory (4GB+ recommended)

##  Performance Expectations

| Document Size | JSON Processing | LangChain | LlamaIndex | Chonkie |
|---------------|----------------|-----------|------------|---------|
| Small (50KB)  | <1s           | 2-5s      | 20-40s     | 5-10s   |
| Medium (500KB)| 1-2s          | 10-20s    | 100-200s   | 20-40s  |
| Large (5MB)   | 5-10s         | 60-120s   | 800-1200s  | 120-240s|

##  Recommendations

### Choose JSON Page-Based Chunking if:
-  You want to preserve document structure
-  Pages have natural content boundaries
-  Need fast processing with logical breaks
-  Working with structured documents (reports, books)

### Choose LangChain if:
-  You need fast, consistent processing
-  Building production RAG systems
-  Processing multiple documents
-  Speed is more important than perfect boundaries

### Choose LlamaIndex if:
-  Quality is paramount
-  Processing critical documents
-  Time is not a constraint
-  Need perfect semantic boundaries

### Choose Chonkie if:
-  Need real-time processing
-  Want semantic awareness with speed
-  Working with shorter documents
-  Building interactive applications

---

*Automated pipeline for JSON text processing and chunking comparison*