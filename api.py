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
import importlib.util
import json
import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
]:
    _path = os.path.join(ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

async def _run_pipeline(job_id: str, file_path: str, fiscal_year: str) -> None:
    loop = asyncio.get_event_loop()
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Shared file paths for inter-step data
    parsed_sections_path  = os.path.join(job_dir, "parsed_sections.json")
    companies_risks_path  = os.path.join(job_dir, "companies_risks.json")
    structured_risks_path = os.path.join(job_dir, "structured_risks.json")
    peer_metrics_path     = os.path.join(job_dir, "peer_metrics.json")

    # Mutable state shared between step closures
    ctx: Dict[str, Any] = {"fiscal_year": fiscal_year}

    try:
        # ------------------------------------------------------------------ Step 1
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
            main_company = extractor.main_company
            extractor.close()

            # Validate extracted entities before writing to Neo4j
            _validators = {
                "FACES_RISK": _import_validator("FACES_RISK", "validate_faces_risk"),
                "HAS_METRIC": _import_validator("HAS_METRIC", "validate_has_metric"),
                "OPERATES_IN": _import_validator("OPERATES_IN", "validate_operates_in"),
            }

            neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
            neo_pass = os.getenv("NEO4J_PASSWORD",  "")
            builder = Neo4jBuilder(neo_uri, neo_user, neo_pass, main_company=main_company)
            total_written = 0
            counts: Dict[str, int] = {}

            for jp in json_paths:
                rel_name = os.path.basename(os.path.dirname(jp))
                validator = _validators.get(rel_name)

                if validator:
                    print(f"\n[Validation] Running {rel_name} validator on {jp} ...")
                    validated = validator(jp)
                    if validated is None:
                        print(f"  ⚠ Validator returned None for {rel_name} — skipping")
                        continue
                    # OPERATES_IN returns {"validated_relation": {...}}; normalise to standard shape
                    if "validated_relation" in validated:
                        item = validated["validated_relation"]
                        rel_type = item.get("rel", rel_name)
                        normalised: Dict = {
                            "main_company": validated.get("main_company", main_company),
                            "relations": {rel_type: [item]},
                        }
                    else:
                        normalised = validated
                    total_written += builder._build_from_data(normalised)
                    for rel, items in normalised.get("relations", {}).items():
                        counts[rel] = len(items)
                else:
                    total_written += builder.build_from_json(jp)
                    with open(jp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    for rel, items in data.get("relations", {}).items():
                        counts[rel] = len(items)

            builder.driver.close()

            return main_company, counts, total_written

        main_company, entity_counts, total_written = await loop.run_in_executor(_executor, _step2)
        ctx["main_company"] = main_company
        parts = [f"{v} {k.replace('_', ' ').lower()}" for k, v in entity_counts.items()]
        _emit(job_id, 2, "done",
              summary=f"Extracted {', '.join(parts)} for {main_company}. "
                      f"{total_written} nodes written to Neo4j.")

        # ------------------------------------------------------------------ Step 3
        _emit(job_id, 3, "running", message="Reading SIC code from Neo4j and querying EDGAR for peer companies…")

        def _step3() -> tuple:
            # Import from the FACES_RISK module (has both get_sic and get_companies_from_api)
            from fetch_and_extract_risks import (
                get_sic_from_neo4j,
                get_companies_from_api,
            )
            sic_codes = get_sic_from_neo4j()
            if isinstance(sic_codes, str):
                sic_codes = [sic_codes]
            companies = get_companies_from_api(sic_codes)
            return sic_codes, companies

        sic_codes, peer_companies = await loop.run_in_executor(_executor, _step3)
        ctx["sic_codes"]     = sic_codes
        ctx["peer_companies"] = peer_companies
        _emit(job_id, 3, "done",
              summary=f"SIC {', '.join(sic_codes)} → Found {len(peer_companies)} peers in EDGAR")

        # ------------------------------------------------------------------ Step 4
        _emit(job_id, 4, "running", message="Fetching peer financial metrics via XBRL…")

        def _step4() -> tuple:
            from extract_metrices import (
                get_target_company_metrics,
                analyze_company_covenants,
                MAX_COMPANIES,
            )
            target_metrics = get_target_company_metrics()
            companies_to_analyze = peer_companies[:MAX_COMPANIES]
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
                        "cik":           cik,
                        "name":          name,
                        "ticker":        ticker,
                        "metrics":       metric_results,
                        "total_matches": total_matches,
                    })

            output = {
                "companies_with_metrics": companies_with_metrics,
                "fiscal_year": fy,
            }
            with open(peer_metrics_path, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False)

            return len(companies_with_metrics), len(target_metrics)

        n_metric_cos, n_metric_types = await loop.run_in_executor(_executor, _step4)
        _emit(job_id, 4, "done",
              summary=f"Retrieved metrics for {n_metric_cos} peers across {n_metric_types} metric types")

        # ------------------------------------------------------------------ Step 5
        _emit(job_id, 5, "running",
              message="Downloading HTM filings for each peer and extracting risks with LLM…")

        def _step5() -> tuple:
            from fetch_and_extract_risks import (
                get_sic_from_neo4j,
                process_companies_from_api,
            )
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
                structured = process_all_risks(
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

        def _step6() -> tuple:
            neo_uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            neo_user = os.getenv("NEO4J_USERNAME",  "neo4j")
            neo_pass = os.getenv("NEO4J_PASSWORD",  "")

            from risks_kg_builder import RisksKGBuilder
            from metrices_kg_builder import write_metrics_to_neo4j

            n_risks = 0
            if os.path.exists(structured_risks_path):
                risk_builder = RisksKGBuilder(neo_uri, neo_user, neo_pass)
                n_risks = risk_builder.build_from_structured_risks(structured_risks_path)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
