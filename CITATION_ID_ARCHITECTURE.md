# Citation ID Architecture

## Overview

This document describes the citation ID system implemented across the knowledge graph to enable deterministic source traceability in LLM-generated answers.

## Citation ID Format

All citations use the format: `[CITE:<citation_id>]`

### Citation ID Patterns

#### Target Company (from HTML parsing)

**Metrics:**
```
TARGET_METRIC_<metric_type>_<year>
```
Examples:
- `TARGET_METRIC_Net_income_2024`
- `TARGET_METRIC_Total_debt_2024`
- `TARGET_METRIC_EBITDA_2023`

**Risks:**
```
TARGET_RISK_<sanitized_risk_name>
```
Examples:
- `TARGET_RISK_Competitive_industries`
- `TARGET_RISK_Regulatory_compliance`
- `TARGET_RISK_Supply_chain_disruptions`

Note: Risk names are sanitized (special chars → `_`, max 50 chars)

#### Peer Companies (from SEC XBRL/filings)

**Metrics:**
```
PEER_METRIC_<CIK>_<xbrl_tag>_<year>
```
Examples:
- `PEER_METRIC_0000891014_NetIncomeLoss_2024`
- `PEER_METRIC_0001234567_LongTermDebt_2024`
- `PEER_METRIC_0000059255_EBITDA_2023`

**Risks:**
```
PEER_RISK_<CIK>_risk_<number>
```
Examples:
- `PEER_RISK_0000891014_risk_1`
- `PEER_RISK_0001234567_risk_5`
- `PEER_RISK_0000059255_risk_12`

## Implementation Points

### 1. Data Ingestion

#### Target Metrics
**File:** `KG_building/relations/HAS_METRIC/entities_extraction.py`
```python
citation_id = f"TARGET_METRIC_{metric.replace(' ', '_')}_{year}"
```

#### Target Risks
**File:** `KG_building/relations/FACES_RISK/entities_extraction.py`
```python
sanitized_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in risk_name)
sanitized_name = sanitized_name[:50]
citation_id = f"TARGET_RISK_{sanitized_name}"
```

#### Peer Metrics
**File:** `peers_sec/HAS_METRIC/metrices_kg_builder.py`
```python
citation_id = f"PEER_METRIC_{metric_cik}_{xbrl_tag}_{year}"
```

#### Peer Risks
**File:** `peers_sec/FACES_RISK/process_risks_simple.py`
```python
risk_id = f"{cik}_risk_{len(company_risks) + 1}"
citation_id = f"PEER_RISK_{risk_id}"
```

**File:** `peers_sec/FACES_RISK/risks_kg_builder.py`
```python
citation_id = f"PEER_RISK_{risk_id}"
```

### 2. Neo4j Storage

All Risk and Metric nodes have a `citation_id` property:

```cypher
// Risk node
(:Risk {
  citation_id: "TARGET_RISK_Competitive_industries",
  risk_id: "...",
  name: "Competitive industries",
  description: "...",
  ...
})

// Metric node
(:Metric {
  citation_id: "TARGET_METRIC_Net_income_2024",
  name: "Net income (2024)",
  value: "1000000",
  year: "2024",
  ...
})
```

### 3. Cypher Queries

**File:** `retrival+eval/credit_risk_qa.py`

All Cypher queries return `citation_id`:

```cypher
// Risks
MATCH (tc:TargetCompany)-[:FACES_RISK]->(r:Risk)
WITH tc, collect({
  citation_id: r.citation_id,
  risk_id: r.risk_id,
  name: r.name,
  ...
}) AS target_risks

// Metrics
MATCH (tc:TargetCompany)-[:HAS_METRIC_CATEGORY]->(mc)-[:HAS_METRIC]->(m:Metric)
WITH tc, mc, collect({
  citation_id: m.citation_id,
  name: m.name,
  value: m.value,
  ...
}) AS target_metrics
```

### 4. LLM Prompts

**File:** `retrival+eval/credit_risk_qa.py`

#### QA_PROMPT
```
After every claim, append an inline citation using the citation_id from the graph results:
  - Risk claim:   [CITE:<citation_id>]   e.g. [CITE:TARGET_RISK_Competitive_industries]
  - Metric claim: [CITE:<citation_id>]   e.g. [CITE:TARGET_METRIC_Net_income_2024]

CRITICAL: The citation_id field is provided in every risk and metric object. 
Use it EXACTLY as shown — do not construct or modify it.
```

### 5. Validation

**File:** `retrival+eval/Ground_Truth/resources/answer_source_traceability.py`

#### Citation Parsing
```python
_CITATION_RE = re.compile(r'\[CITE:([^\]]+)\]', re.IGNORECASE)
```

#### Citation Resolution
```python
def _resolve_citation(raw_citation: str, registry: dict[str, dict]) -> str:
    """Direct lookup by citation_id - no fuzzy matching needed"""
    raw = raw_citation.strip()
    
    # Direct lookup by citation_id
    for sid, src in registry.items():
        if src.get('citation_id') == raw:
            return sid
    
    return f'?{raw}'  # Unresolved
```

## Benefits

### 1. Deterministic Citations
- LLM copies exact ID from graph results
- No ambiguity or fuzzy matching needed
- Citations are stable across runs

### 2. Fast Validation
- Direct dictionary lookup (O(1))
- No expensive LLM calls for validation
- Can validate thousands of citations instantly

### 3. Direct Traceability
- `citation_id` → Neo4j node lookup
- Can query graph directly: `MATCH (n {citation_id: "TARGET_METRIC_Net_income_2024"})`
- Easy debugging and auditing

### 4. Target vs Peer Distinction
- Citation ID prefix clearly indicates source type
- `TARGET_*` = from HTML parsing (target company)
- `PEER_*` = from SEC XBRL/filings (peer companies)

## Migration Checklist

To implement this system:

- [x] Add `citation_id` generation in target metric extraction
- [x] Add `citation_id` generation in target risk extraction
- [x] Add `citation_id` generation in peer metric ingestion
- [x] Add `citation_id` generation in peer risk processing
- [x] Update Neo4j schema to include `citation_id` property
- [x] Update Cypher queries to return `citation_id`
- [x] Update LLM prompts to use `[CITE:<citation_id>]` format
- [x] Update validation script to parse `[CITE:...]` tags
- [x] Update validation script to use direct lookup
- [ ] Re-ingest all data with citation IDs
- [ ] Test end-to-end citation flow
- [ ] Validate citation accuracy

## Example Flow

### 1. Data Ingestion
```python
# Target metric extracted from HTML
{
  "metric": "Net income",
  "value": "1000000",
  "year": "2024",
  "citation_id": "TARGET_METRIC_Net_income_2024"  # ← Generated
}
```

### 2. Neo4j Storage
```cypher
CREATE (m:Metric {
  citation_id: "TARGET_METRIC_Net_income_2024",
  name: "Net income (2024)",
  value: "1000000",
  year: "2024"
})
```

### 3. Cypher Query Result
```json
{
  "target_metrics": [
    {
      "citation_id": "TARGET_METRIC_Net_income_2024",
      "name": "Net income (2024)",
      "value": "1000000",
      "year": "2024"
    }
  ]
}
```

### 4. LLM Answer
```
The target company reported net income of $1M in 2024 [CITE:TARGET_METRIC_Net_income_2024].
```

### 5. Validation
```python
# Parse citation
citation_id = "TARGET_METRIC_Net_income_2024"

# Direct lookup in registry
if citation_id in registry:
    status = "VALID"
else:
    status = "INVALID"
```

## Notes

- Citation IDs are generated at ingestion time, not query time
- They are stored as properties in Neo4j nodes
- The LLM receives them in query results and copies them verbatim
- Validation is a simple dictionary lookup
- No LLM calls needed for validation
