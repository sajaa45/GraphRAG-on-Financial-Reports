# PeersGraphRAG

A credit-risk analysis pipeline that extracts financial metrics and risk factors from SEC 10-K filings, builds a Neo4j knowledge graph, retrieves peer company data from EDGAR, and answers credit-risk questions using retrieval-augmented generation.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Neo4j (local or remote)
- AWS account with Bedrock access (for LLM extraction and QA)

Copy `.env.example` to `.env` and fill in:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_MODEL=...
BEDROCK_MODEL_EVAL=us.meta.llama3-3-70b-instruct-v1:0
MAX_PEER_COMPANIES=3
```

## Running the API

```bash
pip install -r requirements.txt
python api.py
```

The API runs on `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` (or the port shown in the terminal).

## Pipeline steps

1. Parse the uploaded 10-K HTML/PDF into sections
2. Extract target company entities (metrics, risks, industry) via LLM
3. Write target entities to Neo4j
4. Find peer companies via EDGAR (same SIC code), retrieve their metrics and risks
5. Write peer data to Neo4j

Once the graph is built, use the QA endpoint to ask credit-risk questions in natural language.

## Code map

### Entry point

| File | Description |
|------|-------------|
| `api.py` | FastAPI application. Exposes all HTTP endpoints, orchestrates the five pipeline steps, and routes evaluation requests to the appropriate module. |

### parsing/

| File | Description |
|------|-------------|
| `parsing_sections_markdown.py` | Parses an HTML or PDF 10-K filing into a structured JSON of titled sections and their page contents. |

### KG_building/

| File | Description |
|------|-------------|
| `llm_extractor.py` | Reads the parsed sections, uses BM25 to find relevant sections per relation type, calls the LLM page-by-page, and writes extracted entities to JSON. |
| `neo4j_builder.py` | Reads the extracted-entity JSON files and writes nodes and relationships for the target company into Neo4j. |
| `company_utils.py` | Detects the main company name from document text via LLM, and looks up SIC codes via the Groq API. |
| `relation_extraction_config.py` | Defines the `RelationConfig` dataclass and a registry of relation configs loaded dynamically from each relation's `entities_extraction.py`. |
| `relations/FACES_RISK/entities_extraction.py` | Prompt template, entity parser, and config for extracting risk factors. |
| `relations/FACES_RISK/validate_entity.py` | LLM-based validator that checks extracted risk entities before they are written to the graph. |
| `relations/HAS_METRIC/entities_extraction.py` | Prompt template, entity parser, and config for extracting financial metrics. |
| `relations/HAS_METRIC/validate_entity.py` | LLM-based validator for extracted metric entities. |
| `relations/OPERATES_IN/entities_extraction.py` | Prompt template, entity parser, and config for extracting industry/sector information. |
| `relations/OPERATES_IN/validate_entity.py` | LLM-based validator for extracted industry entities. |

### peers_sec/FACES_RISK/

| File | Description |
|------|-------------|
| `fetch_and_extract_risks.py` | Queries the EDGAR full-text search index for peer 10-K filings by SIC code and extracts the Item 1A risk-factor section from each filing. |
| `process_risks.py` | Calls the LLM to structure raw risk-factor text into named risk objects (name, description, why). |
| `risks_kg_builder.py` | Writes structured peer risk nodes and `FACES_RISK` relationships into Neo4j. |

### peers_sec/HAS_METRIC/

| File | Description |
|------|-------------|
| `extract_metrices.py` | Fetches peer company financials from the SEC XBRL API (data.sec.gov) and matches them against the target company's metric types using BM25 and taxonomy lookup. |
| `metrices_kg_builder.py` | Writes peer metric nodes, `HAS_METRIC_CATEGORY`, and `HAS_METRIC` relationships into Neo4j, and links peers to the target via `COMPETES_WITH`. |

### retrival+eval/

| File | Description |
|------|-------------|
| `credit_risk_qa.py` | The RAG engine. Classifies the question, generates Cypher queries via LLM, executes them against Neo4j, deduplicates results, and produces a cited natural-language answer using LangChain and AWS Bedrock. |

### retrival+eval/Ground_Truth/resources/

| File | Description |
|------|-------------|
| `answer_relevancy.py` | Adapted RAGAS answer relevancy: generates N reverse questions from the answer, embeds them, and scores cosine similarity against the original question. |
| `answer_source_traceability.py` | Adapted faithfulness: extracts inline-cited claims from the answer and judges whether each citation actually supports the claim via a batched LLM call. |
| `context_precision.py` | Adapted context precision: judges whether each retrieved risk/metric chunk is relevant to the question via a batched LLM call. |
| `target_validation.py` | Judges whether the target company's extracted risks and metrics are accurate — risks against their source text, metrics against plausibility for the GAAP concept. |
| `risk_peers.py` | Judges whether peer risk chunks are semantically relevant to their assigned risk theme via a grouped LLM call. |
| `num_values_peers.py` | Validates peer metric values against ground truth from the SEC XBRL API (data.sec.gov), with 2% tolerance. |
| `overall_score.py` | Aggregates the five evaluation CSVs into a single weighted pipeline quality score. |
