"""
PeersGraphRAG Pipeline API
--------------------------
POST /pipeline/run          — upload a document with fiscal year and start the pipeline; returns {job_id}
                              Required: file (multipart/form-data), fiscal_year (form field, e.g., "2024")
GET  /pipeline/{id}/stream  — SSE stream of step events (status + summary per step)
GET  /pipeline/{id}/status  — polling fallback: current job state
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import os
import re
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

# Ensure all sub-packages are importable
for _subdir in [
    "parsing",
    "KG_building",
    os.path.join("peers_sec", "FACES_RISK"),
    os.path.join("peers_sec", "HAS_METRIC"),
    "retrival+eval",
]:
    _path = os.path.join(ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

UPLOAD_DIR = os.path.join(ROOT, "uploads")
OUTPUT_DIR = os.path.join(ROOT, "pipeline_output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=4)

_RELATIONS_DIR = os.path.join(ROOT, "KG_building", "relations")


def _import_validator(rel_name: str, fn_name: str):
    """Load a validate_entity.py by relation name and return the named function.
    Uses importlib because all three files share the same module name."""
    path = os.path.join(_RELATIONS_DIR, rel_name, "validate_entity.py")
    spec = importlib.util.spec_from_file_location(f"validator_{rel_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PeersGraphRAG Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {"id": 1, "name": "parse_document",         "label": "Parse & convert document"},
    {"id": 2, "name": "extract_target_entities", "label": "Extract target company entities & build KG"},
    {"id": 3, "name": "find_peers",              "label": "Identify SIC code & find peers via EDGAR"},
    {"id": 4, "name": "retrieve_peer_metrics",   "label": "Retrieve peer metrics via XBRL"},
    {"id": 5, "name": "retrieve_peer_risks",     "label": "Retrieve peer risk factors from HTM filings"},
    {"id": 6, "name": "build_peer_kg",           "label": "Build & populate peer knowledge graph"},
]

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "queue": asyncio.Queue(),
        "steps": {},
        "completed": False,
        "failed": False,
        "error": None,
    }
    return job_id


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def _emit(job_id: str, step: int, status: str, **kwargs) -> None:
    """Push a step-level event onto the job's SSE queue."""
    job = _jobs.get(job_id)
    if not job:
        return
    step_meta = next((s for s in STEPS if s["id"] == step), {})
    payload = {
        "step": step,
        "name": step_meta.get("name", ""),
        "label": step_meta.get("label", ""),
        "status": status,
        **kwargs,
    }
    job["steps"][step] = payload
    job["queue"].put_nowait(payload)


def _pipeline_done(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job:
        job["completed"] = True
        job["queue"].put_nowait({"type": "pipeline_complete"})


def _pipeline_fail(job_id: str, error: str) -> None:
    job = _jobs.get(job_id)
    if job:
        job["failed"] = True
        job["error"] = error
        job["queue"].put_nowait({"type": "pipeline_failed", "error": error})


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def _run_pipeline(
    job_id: str,
    file_path: str,
    fiscal_year: str,
    start_from_step: int = 1,
    prev_job_id: Optional[str] = None,
) -> None:
    loop = asyncio.get_event_loop()

    # When resuming, reuse the previous job's directory so intermediate files are available.
    job_dir = os.path.join(
        OUTPUT_DIR,
        prev_job_id if (start_from_step > 1 and prev_job_id) else job_id,
    )
    os.makedirs(job_dir, exist_ok=True)

    parsed_sections_path  = os.path.join(job_dir, "parsed_sections.json")
    companies_risks_path  = os.path.join(job_dir, "companies_risks.json")
    structured_risks_path = os.path.join(job_dir, "structured_risks.json")
    peer_metrics_path     = os.path.join(job_dir, "peer_metrics.json")
    peer_companies_path   = os.path.join(job_dir, "peer_companies.json")

    ctx: Dict[str, Any] = {"fiscal_year": fiscal_year}

    # Variables referenced by step closures — pre-populated when steps are skipped.
    main_company: str = ""
    peer_companies: list = []

    # Reconstruct context from Neo4j for skipped steps (start_from_step > 2 means
    # steps 1+2 are done and the target company is already in the graph).
    if start_from_step > 2:
        def _reconstruct_ctx():
            from neo4j import GraphDatabase
            neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
            neo_pass = os.getenv("NEO4J_PASSWORD",  "")
            driver   = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pass))
            with driver.session() as s:
                rec = s.run(
                    "MATCH (c:TargetCompany) RETURN c.name AS name LIMIT 1"
                ).single()
            driver.close()
            return rec["name"] if rec else ""
        main_company = await loop.run_in_executor(_executor, _reconstruct_ctx)
        ctx["main_company"] = main_company

    # Load peer companies saved by step 3 so step 4's closure can reference them.
    if start_from_step > 3 and os.path.exists(peer_companies_path):
        with open(peer_companies_path, "r", encoding="utf-8") as fh:
            peer_companies = json.load(fh)
        ctx["peer_companies"] = peer_companies

    try:
        # ------------------------------------------------------------------ Step 1
        if start_from_step <= 1:
            _emit(job_id, 1, "running", message="Parsing HTML document…")

            def _step1() -> Dict:
                from parsing_sections_html import sections_parser_html
                result = sections_parser_html(file_path, parsed_sections_path)
                if result is None:
                    raise RuntimeError("Parser returned no sections — check the HTML format")
                return result

            r1 = await loop.run_in_executor(_executor, _step1)
            _emit(job_id, 1, "done",
                  summary=f"Extracted {r1['num_sections']} sections across {r1['num_pages']} pages")

        # ------------------------------------------------------------------ Step 2
        if start_from_step <= 2:
            _emit(job_id, 2, "running", message="Extracting entities with LLM and writing target company KG…")

            def _step2() -> tuple:
                from llm_extractor import LLMExtractor
                from neo4j_builder import Neo4jBuilder

                extractor = LLMExtractor(
                    parsed_sections_file=parsed_sections_path,
                    output_dir=job_dir,
                    source_file=file_path,
                )
                json_paths = extractor.extract_multiple_relations(["HAS_METRIC", "FACES_RISK", "OPERATES_IN"])
                _mc = extractor.main_company
                extractor.close()

                _validators = {
                    "FACES_RISK": _import_validator("FACES_RISK", "validate_faces_risk"),
                    "HAS_METRIC": _import_validator("HAS_METRIC", "validate_has_metric"),
                    "OPERATES_IN": _import_validator("OPERATES_IN", "validate_operates_in"),
                }

                neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
                neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
                neo_pass = os.getenv("NEO4J_PASSWORD",  "")
                builder  = Neo4jBuilder(neo_uri, neo_user, neo_pass, main_company=_mc)
                total_written = 0
                counts: Dict[str, int] = {}

                # Accumulate all items per relation type across every extracted file
                # before writing to Neo4j — mirrors build_from_validated_dirs so that
                # the HAS_METRIC cleanup runs exactly once, not once per chunk file.
                combined_relations: Dict[str, list] = {}
                for jp in json_paths:
                    rel_name  = os.path.basename(os.path.dirname(jp))
                    validator = _validators.get(rel_name)
                    if validator:
                        print(f"\n[Validation] Running {rel_name} validator on {jp} ...")
                        validated = validator(jp)
                        if validated is None:
                            print(f"  ⚠ Validator returned None for {rel_name} — skipping")
                            continue
                        if "validated_relation" in validated:
                            item     = validated["validated_relation"]
                            rel_type = item.get("rel", rel_name)
                            combined_relations.setdefault(rel_type, []).append(item)
                        else:
                            for rel_type, items in validated.get("relations", {}).items():
                                combined_relations.setdefault(rel_type, []).extend(items)
                    else:
                        with open(jp, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        for rel_type, items in data.get("relations", {}).items():
                            combined_relations.setdefault(rel_type, []).extend(items)

                if combined_relations:
                    normalised_all: Dict = {
                        "main_company": _mc,
                        "relations": combined_relations,
                    }
                    builder.clear_database()
                    builder.stamp_target_company(_mc)
                    total_written = builder._build_from_data(normalised_all)
                    counts = {rel: len(items) for rel, items in combined_relations.items()}

                builder.driver.close()
                return _mc, counts, total_written

            _mc, entity_counts, total_written = await loop.run_in_executor(_executor, _step2)
            main_company = _mc
            ctx["main_company"] = main_company
            parts = [f"{v} {k.replace('_', ' ').lower()}" for k, v in entity_counts.items()]
            _emit(job_id, 2, "done",
                  summary=f"Extracted {', '.join(parts)} for {main_company}. "
                          f"{total_written} nodes written to Neo4j.")

        # ------------------------------------------------------------------ Step 3
        if start_from_step <= 3:
            _emit(job_id, 3, "running", message="Reading SIC code from Neo4j and querying EDGAR for peer companies…")

            def _step3() -> tuple:
                from fetch_and_extract_risks import get_sic_from_neo4j, get_companies_from_api
                sic = get_sic_from_neo4j()
                if isinstance(sic, str):
                    sic = [sic]
                companies = get_companies_from_api(sic, fiscal_year=ctx["fiscal_year"])
                return sic, companies

            _sic, _peers = await loop.run_in_executor(_executor, _step3)
            peer_companies = _peers
            ctx["sic_codes"]      = _sic
            ctx["peer_companies"] = peer_companies
            # Persist so a future resume can reload peer_companies without re-querying EDGAR.
            with open(peer_companies_path, "w", encoding="utf-8") as fh:
                json.dump(peer_companies, fh, ensure_ascii=False)
            _emit(job_id, 3, "done",
                  summary=f"SIC {', '.join(_sic)} → Found {len(peer_companies)} peers in EDGAR")

        # ------------------------------------------------------------------ Step 4
        if start_from_step <= 4:
            _emit(job_id, 4, "running", message="Fetching peer financial metrics via XBRL…")

            def _step4() -> tuple:
                from extract_metrices import (
                    get_target_company_metrics,
                    analyze_company_covenants,
                    MAX_COMPANIES,
                )
                target_metrics       = get_target_company_metrics()
                # Exclude the target company itself from peer analysis
                companies_to_analyze = [
                    c for c in peer_companies[:MAX_COMPANIES]
                    if c.get("name") != main_company
                ]
                fy = ctx["fiscal_year"]

                companies_with_metrics = []
                for company in companies_to_analyze:
                    cik    = company["cik"]
                    name   = company["name"]
                    ticker = company.get("ticker", "N/A")
                    metric_results = analyze_company_covenants(cik, name, target_metrics, fy)
                    if metric_results:
                        total_matches = sum(len(v) for v in metric_results.values())
                        companies_with_metrics.append({
                            "company":       {"cik": cik, "name": name, "ticker": ticker},
                            "metrics":       metric_results,
                            "total_matches": total_matches,
                        })

                output = {"companies_with_metrics": companies_with_metrics, "fiscal_year": fy}
                with open(peer_metrics_path, "w", encoding="utf-8") as fh:
                    json.dump(output, fh, indent=2, ensure_ascii=False)

                return len(companies_with_metrics), len(target_metrics)

            n_metric_cos, n_metric_types = await loop.run_in_executor(_executor, _step4)
            _emit(job_id, 4, "done",
                  summary=f"Retrieved metrics for {n_metric_cos} peers across {n_metric_types} metric types")

        # ------------------------------------------------------------------ Step 5
        if start_from_step <= 5:
            _emit(job_id, 5, "running",
                  message="Downloading HTM filings for each peer and extracting risks with LLM…")

            def _step5() -> tuple:
                from fetch_and_extract_risks import get_sic_from_neo4j, process_companies_from_api
                from process_risks import process_all_risks

                sic = get_sic_from_neo4j()
                if isinstance(sic, str):
                    sic = [sic]

                fy = ctx["fiscal_year"]
                companies_data = process_companies_from_api(
                    sic_codes=sic,
                    fiscal_year=fy,
                    output_file=companies_risks_path,
                )
                n_companies = len(companies_data)

                if n_companies > 0:
                    structured  = process_all_risks(
                        input_file=companies_risks_path,
                        output_file=structured_risks_path,
                    )
                    total_risks = sum(c.get("total_risks", 0) for c in structured)
                else:
                    total_risks = 0

                return n_companies, total_risks

            n_risk_cos, n_total_risks = await loop.run_in_executor(_executor, _step5)
            _emit(job_id, 5, "done",
                  summary=f"Extracted {n_total_risks} structured risks from {n_risk_cos} peer filings")

        # ------------------------------------------------------------------ Step 6
        _emit(job_id, 6, "running",
              message="Writing peer risks and metrics into the Neo4j knowledge graph…")

        def _step6() -> int:
            neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
            neo_pass = os.getenv("NEO4J_PASSWORD",  "")

            from risks_kg_builder   import RisksKGBuilder
            from metrices_kg_builder import write_metrics_to_neo4j

            n_risks = 0
            if os.path.exists(structured_risks_path):
                risk_builder = RisksKGBuilder(neo_uri, neo_user, neo_pass)
                n_risks      = risk_builder.build_from_structured_risks(structured_risks_path)
                risk_builder.driver.close()

            if os.path.exists(peer_metrics_path):
                write_metrics_to_neo4j(peer_metrics_path)

            return n_risks

        n_kg_risks = await loop.run_in_executor(_executor, _step6)
        _emit(job_id, 6, "done",
              summary=f"Knowledge graph complete. {n_kg_risks} peer risks and metrics committed to Neo4j.")

        _pipeline_done(job_id)

    except Exception as exc:
        print(f"[Pipeline ERROR] job={job_id}\n{traceback.format_exc()}")
        _pipeline_fail(job_id, str(exc))


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _sse_generator(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Job not found'})}\n\n"
        return

    queue: asyncio.Queue = job["queue"]

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            # Send a heartbeat so the connection stays alive
            yield ": heartbeat\n\n"
            continue

        yield f"data: {json.dumps(event)}\n\n"

        event_type = event.get("type")
        if event_type in ("pipeline_complete", "pipeline_failed"):
            break

        # Also stop if the step had a terminal error
        if event.get("status") == "error":
            # Let the pipeline_failed event arrive naturally; keep listening
            pass


# ---------------------------------------------------------------------------
# QA singleton
# ---------------------------------------------------------------------------

_qa_instance = None


def _get_qa():
    global _qa_instance
    if _qa_instance is None:
        from credit_risk_qa import CreditRiskQA
        _qa_instance = CreditRiskQA()
    return _qa_instance


def _enrich_metric_source_info(citations: Dict[str, Any]) -> None:
    """Stamp source_page / section_title onto target-metric citations that are missing them.

    Existing Metric nodes store these values inside a JSON metadata blob (m.metadata)
    rather than as flat properties.  We query by citation_id and parse the blob so this
    works without requiring a re-ingest.  New nodes (after the neo4j_builder fix) expose
    them as flat properties — both paths are handled by the single query below.
    """
    target_metric_ids = [
        cid for cid, info in citations.items()
        if info.get("type") == "metric" and info.get("role") == "target"
        and (info.get("source_page") is None and info.get("section_title") is None)
    ]
    if not target_metric_ids:
        return

    neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
    neo_pass = os.getenv("NEO4J_PASSWORD",  "")
    try:
        from neo4j import GraphDatabase as _GD
        driver = _GD.driver(neo_uri, auth=(neo_user, neo_pass))
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (m:Metric)
                WHERE m.company IS NULL AND m.citation_id IN $ids
                RETURN m.citation_id AS cid,
                       m.source_page   AS source_page,
                       m.section_title AS section_title,
                       m.metadata      AS metadata
                """,
                {"ids": target_metric_ids},
            )
            for row in rows:
                cid          = row["cid"]
                source_page  = row["source_page"]
                section_title = row["section_title"]
                # Fall back to the metadata JSON blob for nodes ingested before the fix
                if (source_page is None or not section_title) and row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                        source_page   = source_page   or meta.get("source_page")
                        section_title = section_title or meta.get("section_title", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                if cid in citations:
                    citations[cid]["source_page"]   = source_page
                    citations[cid]["section_title"]  = section_title or None
        driver.close()
    except Exception:
        pass  # enrichment is best-effort — don't break the QA response


def _build_citations(results: list) -> Dict[str, Any]:
    citations: Dict[str, Any] = {}
    for row in results:
        target = row.get("target", "")
        peer = row.get("peer", "")
        for r in (row.get("target_risks") or []):
            cid = r.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "risk", "company": target, "role": "target",
                    "document_url": r.get("document_url") or None,
                    "source_page": r.get("source_page") or None,
                    "section_title": r.get("section_title") or None,
                    "summary": r.get("name", ""),
                }
        for r in (row.get("peer_risks") or []):
            cid = r.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "risk", "company": peer, "role": "peer",
                    "document_url": r.get("document_url") or None,
                    "source_page": r.get("source_page") or None,
                    # section_title isn't stored in Neo4j for peer risks — it's always this
                    "section_title": r.get("section_title") or "Item 1A – Risk Factors",
                    "summary": r.get("name", ""),
                }
        for m in (row.get("target_metrics") or []):
            cid = m.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "metric", "company": target, "role": "target",
                    "document_url": None,
                    "source_page": m.get("source_page") or None,
                    "section_title": m.get("section_title") or None,
                    "summary": (
                        f"{m.get('label') or m.get('name', '')} = "
                        f"{m.get('value', '')} {m.get('unit', '')} ({m.get('year', '')})"
                    ).strip(),
                }
        for m in (row.get("peer_metrics") or []):
            cid = m.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "metric", "company": peer, "role": "peer",
                    "document_url": m.get("source_url") or None,
                    "summary": (
                        f"{m.get('label') or m.get('name', '')} = "
                        f"{m.get('value', '')} {m.get('unit', '')} ({m.get('year', '')})"
                    ).strip(),
                }
    return citations


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    question: str
    reasoning: bool = False


class ResumeRequest(BaseModel):
    prev_job_id: str
    start_from_step: int   # 3–6 (steps 1-2 require the original uploaded file)
    fiscal_year: str


class EvalRequest(BaseModel):
    test_type: str  # answer_relevancy | context_precision | answer_source_traceability
                    # | target_validation | risk_peers_validation | overall_score


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

_RETRIVAL_DIR = os.path.join(ROOT, "retrival+eval", "retrival_results")
_GROUND_TRUTH_DIR = os.path.join(ROOT, "retrival+eval", "Ground_Truth")
_RESOURCES_DIR = os.path.join(_GROUND_TRUTH_DIR, "resources")


def _load_eval_module(name: str):
    path = os.path.join(_RESOURCES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _csv_to_scorecard(rows: list, test_type: str) -> Dict[str, Any]:
    """Map CSV rows from any eval test into the {rows, weighted} Scorecard shape."""
    if test_type == "answer_relevancy":
        summary = next((r for r in rows if r.get("aspect", "").strip() == "** SUMMARY **"), None)
        weighted = float(summary["combined_score"]) if summary else 0.0
        items = [
            {
                "dimension": r["aspect"],
                "score": float(r.get("aspect_score") or 0),
                "note": r.get("explanation", ""),
            }
            for r in rows if r.get("aspect", "").strip() != "** SUMMARY **"
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "context_precision":
        summary = next((r for r in rows if r.get("citation_id", "").strip() == "** SUMMARY **"), None)
        weighted = float(summary["is_relevant"]) if summary else 0.0
        items = [
            {
                "dimension": r.get("citation_id", ""),
                "score": 1.0 if str(r.get("is_relevant", "")).lower() == "true" else 0.0,
                "note": r.get("explanation", ""),
            }
            for r in rows if r.get("citation_id", "").strip() != "** SUMMARY **"
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type in ("answer_source_traceability", "faithfulness"):
        data_rows = [r for r in rows if r.get("claim_id", "").strip()]
        correct = sum(1 for r in data_rows if str(r.get("is_correct_source", "")).lower() == "true")
        weighted = correct / len(data_rows) if data_rows else 0.0
        items = [
            {
                "dimension": r["claim_id"],
                "score": 1.0 if str(r.get("is_correct_source", "")).lower() == "true" else 0.0,
                "note": r.get("explanation", ""),
            }
            for r in data_rows
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "target_validation":
        data_rows = [r for r in rows if r.get("citation_id", "").strip()]
        correct = sum(1 for r in data_rows if str(r.get("is_correctly_extracted", "")).lower() == "true")
        weighted = correct / len(data_rows) if data_rows else 0.0
        items = [
            {
                "dimension": f"{r.get('type','?')} · {r.get('extracted_name','')}",
                "score": 1.0 if str(r.get("is_correctly_extracted", "")).lower() == "true" else 0.0,
                "note": r.get("explanation", ""),
            }
            for r in data_rows
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "risk_peers_validation":
        data_rows = [r for r in rows if r.get("company_name", "").strip()]
        relevant = sum(1 for r in data_rows if str(r.get("is_semantically_relevant", "")).lower() == "true")
        weighted = relevant / len(data_rows) if data_rows else 0.0
        items = [
            {
                "dimension": f"{r.get('company_name','')} · {r.get('risk_theme','')}",
                "score": 1.0 if str(r.get("is_semantically_relevant", "")).lower() == "true" else 0.0,
                "note": r.get("relevance_explanation", ""),
            }
            for r in data_rows
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "overall_score":
        overall_row = next((r for r in rows if r.get("component", "").strip() == "** OVERALL **"), None)
        try:
            weighted = float(overall_row["score"]) if overall_row else 0.0
        except (TypeError, ValueError):
            weighted = 0.0
        items = []
        for r in rows:
            comp = r.get("component", "").strip()
            if comp == "** OVERALL **":
                continue
            try:
                score = float(r.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            items.append({
                "dimension": comp,
                "score": score,
                "note": f"weight {r.get('weight', '')}",
            })
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    return {"test_type": test_type, "rows": [], "weighted": 0.0}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/pipeline/run")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fiscal_year: str = Form(...),
):
    """Upload an HTML/PDF annual report and start the processing pipeline.
    
    Args:
        file: The annual report file (HTML or PDF)
        fiscal_year: The fiscal year for the report (e.g., "2024")
    """
    job_id = _new_job()

    # Save uploaded file
    ext = os.path.splitext(file.filename or "report.htm")[1] or ".htm"
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    content = await file.read()
    with open(file_path, "wb") as fh:
        fh.write(content)

    background_tasks.add_task(_run_pipeline, job_id, file_path, fiscal_year)

    return {"job_id": job_id, "filename": file.filename, "fiscal_year": fiscal_year, "steps": STEPS}


@app.post("/pipeline/resume")
async def resume_pipeline(request: ResumeRequest, background_tasks: BackgroundTasks):
    """Resume a previously failed pipeline from a specific step (3–6).

    The previous job's intermediate files (peer_companies.json, peer_metrics.json,
    structured_risks.json) are reused from its output directory, so only the
    failed step(s) are re-executed.
    """
    if not (3 <= request.start_from_step <= 6):
        raise HTTPException(
            status_code=400,
            detail="start_from_step must be between 3 and 6 "
                   "(steps 1-2 require the original file — start a fresh run instead)",
        )
    prev_job_dir = os.path.join(OUTPUT_DIR, request.prev_job_id)
    if not os.path.isdir(prev_job_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Previous job directory not found: {request.prev_job_id}",
        )

    job_id = _new_job()
    background_tasks.add_task(
        _run_pipeline,
        job_id,
        "",                        # file_path unused when start_from_step >= 3
        request.fiscal_year,
        request.start_from_step,
        request.prev_job_id,
    )
    return {"job_id": job_id, "steps": STEPS, "start_from_step": request.start_from_step}


@app.get("/pipeline/{job_id}/stream")
async def stream_pipeline(job_id: str):
    """Server-Sent Events stream.  Each event is a JSON object with:
    - step (int), name, label, status ("running"|"done"|"error"), message/summary/error
    - or type "pipeline_complete" | "pipeline_failed"
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/pipeline/{job_id}/status")
async def get_status(job_id: str):
    """Polling alternative to SSE — returns the current state of all steps."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id":    job_id,
        "completed": job["completed"],
        "failed":    job["failed"],
        "error":     job["error"],
        "steps":     job["steps"],
    }


@app.post("/qa/run")
async def run_qa(request: QARequest):
    """Run a question against the knowledge graph.

    Returns the answer with inline [CITE:id] tags, a citations map for rendering
    clickable links, the Cypher queries that were executed, and (optionally) a
    reasoning trace.
    """
    loop = asyncio.get_event_loop()

    def _run():
        qa = _get_qa()
        data = qa.ask_multi(request.question)
        answer = _THINK_RE.sub('', data["answer"]).strip()
        queries = data["queries"]
        results = data["results"]

        citations = _build_citations(results)

        # Enrich target-metric citations with source_page / section_title.
        # The LLM-generated Cypher may not request these fields, and existing
        # Metric nodes store them inside a JSON metadata blob rather than as
        # flat properties. Query Neo4j directly by citation_id so this works
        # with all existing data without requiring a re-ingest.
        _enrich_metric_source_info(citations)

        qa._save_extraction_result(request.question, queries, results)
        qa._save_answer(answer)

        reasoning_trace: Optional[str] = None
        if request.reasoning and results:
            reasoning_trace = qa.generate_reasoning_trace(request.question, queries, results)
            reasoning_trace = reasoning_trace or None
            if reasoning_trace:
                qa._save_reasoning(reasoning_trace)

        return {
            "answer": answer,
            "reasoning_trace": reasoning_trace,
            "queries": queries,
            "citations": citations,
        }

    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/eval/run")
async def run_eval(request: EvalRequest):
    """Run an evaluation test against the last QA answer.

    test_type must be one of:
      answer_relevancy | context_precision | answer_source_traceability
      | target_validation | risk_peers_validation | overall_score
    """
    valid_types = {
        "answer_relevancy", "context_precision", "answer_source_traceability",
        "faithfulness", "context_recall",
        "target_validation", "risk_peers_validation", "overall_score",
    }
    if request.test_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Unknown test_type: {request.test_type}")

    extraction_path = os.path.join(_RETRIVAL_DIR, "extraction_result.json")
    answer_path = os.path.join(_RETRIVAL_DIR, "answer.txt")

    no_extraction_needed = {"overall_score", "context_recall"}
    if request.test_type not in no_extraction_needed:
        for p in (extraction_path, answer_path):
            if not os.path.exists(p):
                raise HTTPException(
                    status_code=404,
                    detail="No QA results found — ask a question first.",
                )

    loop = asyncio.get_event_loop()

    def _run():
        tt = request.test_type

        if tt == "answer_relevancy":
            mod = _load_eval_module("answer_relevancy")
            out = os.path.join(_GROUND_TRUTH_DIR, "answer_relevancy_results.csv")
            mod.evaluate_relevancy(extraction_path, answer_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "context_precision":
            mod = _load_eval_module("context_precision")
            out = os.path.join(_GROUND_TRUTH_DIR, "context_precision_results.csv")
            # context_precision only needs extraction_result.json
            mod.evaluate_context_precision(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt in ("answer_source_traceability", "faithfulness"):
            mod = _load_eval_module("answer_source_traceability")
            out = os.path.join(_GROUND_TRUTH_DIR, "answer_source_traceability.csv")
            mod.evaluate_traceability(extraction_path, answer_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "context_recall":
            os_mod = _load_eval_module("overall_score")
            def _try(fn, path):
                try:
                    return fn(path) if os.path.exists(path) else None
                except Exception:
                    return None
            gt = _GROUND_TRUTH_DIR
            components = {
                "Target Validation": _try(os_mod.score_target_validation,  os.path.join(gt, "target_validation_results.csv")),
                "Peer Risk Recall":  _try(os_mod.score_risks_validation,   os.path.join(gt, "risks_validation_results.csv")),
                "Metric Accuracy":   _try(os_mod.score_metrics_validation, os.path.join(gt, "metrics_validation_results.csv")),
            }
            available = {k: v for k, v in components.items() if v is not None}
            recall = sum(available.values()) / len(available) if available else 0.0
            items = [{"dimension": k, "score": v, "note": ""} for k, v in available.items()]
            missing = [k for k, v in components.items() if v is None]
            if missing:
                items.append({"dimension": f"⚠ missing: {', '.join(missing)}", "score": 0.0, "note": "run the individual tests first"})
            return {"test_type": "context_recall", "rows": items, "weighted": recall}

        if tt == "target_validation":
            mod = _load_eval_module("target_validation")
            out = os.path.join(_GROUND_TRUTH_DIR, "target_validation_results.csv")
            mod.validate_target(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "risk_peers_validation":
            mod = _load_eval_module("risk_peers")
            out = os.path.join(_GROUND_TRUTH_DIR, "risks_validation_results.csv")
            mod.validate_risk_chunks(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "overall_score":
            mod = _load_eval_module("overall_score")
            out = os.path.join(_GROUND_TRUTH_DIR, "overall_score.csv")
            mod.compute_overall(_GROUND_TRUTH_DIR, out)
            return _csv_to_scorecard(_read_csv(out), tt)

    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/qa/ready")
async def qa_ready():
    """Return {ready: bool} — true if Neo4j already contains TargetCompany or Company nodes.
    The frontend calls this on startup so the QA section unlocks even when the pipeline
    was run outside the UI (e.g. via the CLI)."""
    loop = asyncio.get_event_loop()

    def _check():
        try:
            from neo4j import GraphDatabase
            neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
            neo_pass = os.getenv("NEO4J_PASSWORD",  "")
            driver = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pass))
            with driver.session() as s:
                cnt = s.run(
                    "MATCH (n) WHERE n:TargetCompany OR n:Company "
                    "RETURN count(n) AS cnt LIMIT 1"
                ).single()["cnt"]
            driver.close()
            return {"ready": cnt > 0, "node_count": cnt}
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    return await loop.run_in_executor(_executor, _check)


@app.get("/health")
async def health():
    return {"status": "ok"}
