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
FISCAL_YEAR=2024
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
