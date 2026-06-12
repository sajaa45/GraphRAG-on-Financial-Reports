"""
PeersGraphRAG — REST API

POST /pipeline/run                      upload HTML/PDF filing + fiscal year, run all 6 steps
POST /pipeline/{job_id}/step/{step_id}  run or re-run a single step (1–6)
POST /pipeline/{job_id}/run             run all remaining incomplete steps
GET  /pipeline/{job_id}/stream          live step progress via server-sent events
GET  /pipeline/{job_id}/status          JSON status snapshot (polling alternative)
GET  /pipeline/jobs                     list all jobs saved on disk
POST /qa/run                            ask a question against the knowledge graph
POST /eval/run                          run an evaluation test against the last QA answer
GET  /qa/ready                          check whether the graph has data
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
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

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

import boto3 as _boto3
from neo4j import GraphDatabase as _GraphDatabase

_bedrock_client = _boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

_neo4j_driver = _GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)

_RELATIONS_DIR = os.path.join(ROOT, "KG_building", "relations")


def _import_validator(rel_name: str, fn_name: str):
    path = os.path.join(_RELATIONS_DIR, rel_name, "validate_entity.py")
    spec = importlib.util.spec_from_file_location(f"validator_{rel_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


app = FastAPI(title="PeersGraphRAG Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STEPS = [
    {"id": 1, "name": "parse_document",          "label": "Parse & convert document"},
    {"id": 2, "name": "extract_target_entities",  "label": "Extract target company entities & build KG"},
    {"id": 3, "name": "find_peers",               "label": "Identify SIC code & find peers via EDGAR"},
    {"id": 4, "name": "retrieve_peer_metrics",    "label": "Retrieve peer metrics via XBRL"},
    {"id": 5, "name": "retrieve_peer_risks",      "label": "Retrieve peer risk factors from HTM filings"},
    {"id": 6, "name": "build_peer_kg",            "label": "Build & populate peer knowledge graph"},
]

# In-memory SSE state: job_id → {queue, steps, completed, failed, error}
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Job state — persisted to disk so steps can be resumed after server restart
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> str:
    return os.path.join(OUTPUT_DIR, job_id)

def _state_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "state.json")

def _load_state(job_id: str) -> dict:
    path = _state_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_state(state: dict) -> None:
    with open(_state_path(state["job_id"]), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def _create_state(job_id: str, fiscal_year: str, file_path: str) -> dict:
    state = {
        "job_id":          job_id,
        "fiscal_year":     fiscal_year,
        "file_path":       file_path,
        "main_company":    "",
        "sic_codes":       [],
        "completed_steps": [],
        "created_at":      datetime.utcnow().isoformat(),
    }
    os.makedirs(_job_dir(job_id), exist_ok=True)
    _save_state(state)
    return state

def _mark_step_complete(state: dict, step_id: int, **updates) -> None:
    if step_id not in state["completed_steps"]:
        state["completed_steps"].append(step_id)
    state.update(updates)
    _save_state(state)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _ensure_memory_job(job_id: str) -> dict:
    if job_id not in _jobs:
        _jobs[job_id] = {
            "queue":     asyncio.Queue(),
            "steps":     {},
            "completed": False,
            "failed":    False,
            "error":     None,
        }
    return _jobs[job_id]

def _emit(job_id: str, step: int, status: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    step_meta = next((s for s in STEPS if s["id"] == step), {})
    payload = {
        "step":   step,
        "name":   step_meta.get("name", ""),
        "label":  step_meta.get("label", ""),
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
        job["error"]  = error
        job["queue"].put_nowait({"type": "pipeline_failed", "error": error})


# ---------------------------------------------------------------------------
# Individual step implementations
# ---------------------------------------------------------------------------

def _run_step1(job_id: str, state: dict) -> dict:
    from parsing_sections_markdown import sections_parser_markdown
    out = os.path.join(_job_dir(job_id), "parsed_sections.json")
    result = sections_parser_markdown(state["file_path"], out)
    if result is None:
        raise RuntimeError("Parser returned no sections — check the HTML format")
    return result


def _run_step2(job_id: str, state: dict) -> tuple:
    from llm_extractor import LLMExtractor
    from neo4j_builder import Neo4jBuilder

    parsed_path = os.path.join(_job_dir(job_id), "parsed_sections.json")
    extractor = LLMExtractor(
        parsed_sections_file=parsed_path,
        output_dir=_job_dir(job_id),
        source_file=state["file_path"],
        bedrock_client=_bedrock_client,
    )
    json_paths = extractor.extract_multiple_relations(["HAS_METRIC", "FACES_RISK", "OPERATES_IN"])
    main_company = extractor.main_company
    extractor.close()

    combined_relations: Dict[str, list] = {}
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "validated_relation" in data:
            item = data["validated_relation"]
            combined_relations.setdefault(item.get("rel", ""), []).append(item)
        else:
            for rel_type, items in data.get("relations", {}).items():
                combined_relations.setdefault(rel_type, []).extend(items)

    builder = Neo4jBuilder(main_company=main_company, driver=_neo4j_driver)
    builder.clear_database()
    total_written = builder._build_from_data({"main_company": main_company, "relations": combined_relations})
    counts = {rel: len(items) for rel, items in combined_relations.items()}
    return main_company, counts, total_written


def _run_step3(job_id: str, state: dict) -> tuple:
    from fetch_and_extract_risks import get_sic_from_neo4j, get_companies_from_api
    sic = get_sic_from_neo4j(_neo4j_driver)
    if isinstance(sic, str):
        sic = [sic]
    companies = get_companies_from_api(sic, fiscal_year=state["fiscal_year"])
    out = os.path.join(_job_dir(job_id), "peer_companies.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(companies, f, ensure_ascii=False)
    return sic, companies


def _run_step4(job_id: str, state: dict) -> tuple:
    from extract_metrices import get_target_company_metrics, analyze_company_covenants
    peers_path = os.path.join(_job_dir(job_id), "peer_companies.json")
    with open(peers_path, "r", encoding="utf-8") as f:
        peer_companies = json.load(f)

    target_metrics = get_target_company_metrics(_neo4j_driver)
    fy = state["fiscal_year"]
    main_company = state["main_company"]

    companies_with_metrics = []
    for company in peer_companies:
        if company.get("name") == main_company:
            continue
        cik    = company["cik"]
        name   = company["name"]
        ticker = company.get("ticker", "N/A")
        results = analyze_company_covenants(cik, name, target_metrics, fy)
        if results:
            companies_with_metrics.append({
                "company":       {"cik": cik, "name": name, "ticker": ticker},
                "target_metrics": results,
                "total_matches": sum(len(v) for v in results.values()),
            })

    out = os.path.join(_job_dir(job_id), "peer_metrics.json")
    with open(out, "w", encoding="utf-8", indent=2) as f:
        json.dump({"companies_with_metrics": companies_with_metrics, "fiscal_year": fy}, f, ensure_ascii=False)
    return len(companies_with_metrics), len(target_metrics)


def _run_step5(job_id: str, state: dict) -> tuple:
    from fetch_and_extract_risks import process_companies_from_api
    from process_risks import process_all_risks

    risks_path      = os.path.join(_job_dir(job_id), "companies_risks.json")
    structured_path = os.path.join(_job_dir(job_id), "structured_risks.json")

    companies_data = process_companies_from_api(
        _neo4j_driver,
        sic_codes=state["sic_codes"],
        fiscal_year=state["fiscal_year"],
        output_file=risks_path,
    )
    total_risks = 0
    if companies_data:
        structured  = process_all_risks(_bedrock_client, input_file=risks_path, output_file=structured_path)
        total_risks = sum(c.get("total_risks", 0) for c in structured)
    return len(companies_data), total_risks


def _run_step6(job_id: str, state: dict) -> int:
    from risks_kg_builder    import RisksKGBuilder
    from metrices_kg_builder import write_metrics_to_neo4j

    structured_path = os.path.join(_job_dir(job_id), "structured_risks.json")
    metrics_path    = os.path.join(_job_dir(job_id), "peer_metrics.json")

    n_risks = 0
    if os.path.exists(structured_path):
        builder = RisksKGBuilder(driver=_neo4j_driver)
        n_risks = builder.build_from_structured_risks(structured_path)

    if os.path.exists(metrics_path):
        write_metrics_to_neo4j(_neo4j_driver, metrics_path)

    return n_risks


# ---------------------------------------------------------------------------
# Step dispatcher — runs one step, emits SSE, updates state
# ---------------------------------------------------------------------------

async def _execute_step(job_id: str, step_id: int, state: dict, loop) -> bool:
    """Run step_id, emit SSE events, update state. Returns True on success."""
    _ensure_memory_job(job_id)

    step_meta = next(s for s in STEPS if s["id"] == step_id)
    _emit(job_id, step_id, "running", message=f"{step_meta['label']}…")

    try:
        if step_id == 1:
            r = await loop.run_in_executor(_executor, _run_step1, job_id, state)
            _mark_step_complete(state, 1)
            _emit(job_id, 1, "done",
                  summary=f"Extracted {r['num_sections']} sections")

        elif step_id == 2:
            main_company, counts, total = await loop.run_in_executor(_executor, _run_step2, job_id, state)
            _mark_step_complete(state, 2, main_company=main_company)
            parts = [f"{v} {k.replace('_', ' ').lower()}" for k, v in counts.items()]
            _emit(job_id, 2, "done",
                  summary=f"Extracted {', '.join(parts)} for {main_company}. {total} nodes written.")

        elif step_id == 3:
            sic, companies = await loop.run_in_executor(_executor, _run_step3, job_id, state)
            _mark_step_complete(state, 3, sic_codes=sic)
            _emit(job_id, 3, "done",
                  summary=f"SIC {', '.join(sic)} → {len(companies)} peers found in EDGAR")

        elif step_id == 4:
            n_cos, n_types = await loop.run_in_executor(_executor, _run_step4, job_id, state)
            _mark_step_complete(state, 4)
            _emit(job_id, 4, "done",
                  summary=f"Metrics retrieved for {n_cos} peers across {n_types} metric types")

        elif step_id == 5:
            n_cos, n_risks = await loop.run_in_executor(_executor, _run_step5, job_id, state)
            _mark_step_complete(state, 5)
            _emit(job_id, 5, "done",
                  summary=f"Extracted {n_risks} structured risks from {n_cos} peer filings")

        elif step_id == 6:
            n_risks = await loop.run_in_executor(_executor, _run_step6, job_id, state)
            _mark_step_complete(state, 6)
            _emit(job_id, 6, "done",
                  summary=f"Knowledge graph complete. {n_risks} peer risks committed to Neo4j.")

        return True

    except Exception as exc:
        print(f"[Step {step_id} ERROR] job={job_id}\n{traceback.format_exc()}")
        _emit(job_id, step_id, "failed", error=str(exc))
        return False


async def _run_pipeline(job_id: str, from_step: int = 1) -> None:
    """Background task: run all steps from from_step, stopping on first failure."""
    loop = asyncio.get_running_loop()
    state = _load_state(job_id)
    _ensure_memory_job(job_id)

    try:
        for step_id in range(from_step, len(STEPS) + 1):
            if step_id in state["completed_steps"]:
                continue
            ok = await _execute_step(job_id, step_id, state, loop)
            if not ok:
                _pipeline_fail(job_id, f"Step {step_id} failed")
                return
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

    while True:
        try:
            event = await asyncio.wait_for(job["queue"].get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield ": heartbeat\n\n"
            continue
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") in ("pipeline_complete", "pipeline_failed"):
            break


# ---------------------------------------------------------------------------
# QA helpers
# ---------------------------------------------------------------------------

_qa_instance = None
_qa_lock = asyncio.Lock()


def _get_qa():
    global _qa_instance
    if _qa_instance is None:
        from credit_risk_qa import CreditRiskQA
        _qa_instance = CreditRiskQA()
    return _qa_instance


def _enrich_metric_source_info(citations: Dict[str, Any]) -> None:
    target_metric_ids = [
        cid for cid, info in citations.items()
        if info.get("type") == "metric" and info.get("role") == "target"
        and (info.get("source_page") is None and info.get("section_title") is None)
    ]
    if not target_metric_ids:
        return
    try:
        with _neo4j_driver.session() as session:
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
                cid           = row["cid"]
                source_page   = row["source_page"]
                section_title = row["section_title"]
                if (source_page is None or not section_title) and row["metadata"]:
                    try:
                        meta          = json.loads(row["metadata"])
                        source_page   = source_page   or meta.get("source_page")
                        section_title = section_title or meta.get("section_title", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                if cid in citations:
                    citations[cid]["source_page"]  = source_page
                    citations[cid]["section_title"] = section_title or None
    except Exception:
        pass


def _build_citations(results: list) -> Dict[str, Any]:
    citations: Dict[str, Any] = {}
    for row in results:
        target = row.get("target", "")
        peer   = row.get("peer", "")
        for r in (row.get("target_risks") or []):
            cid = r.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "risk", "company": target, "role": "target",
                    "document_url":  r.get("document_url") or None,
                    "source_page":   r.get("source_page") or None,
                    "section_title": r.get("section_title") or None,
                    "summary":       r.get("name", ""),
                }
        for r in (row.get("peer_risks") or []):
            cid = r.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "risk", "company": peer, "role": "peer",
                    "document_url":  r.get("document_url") or None,
                    "source_page":   r.get("source_page") or None,
                    "section_title": r.get("section_title") or "Item 1A – Risk Factors",
                    "summary":       r.get("name", ""),
                }
        for m in (row.get("target_metrics") or []):
            cid = m.get("citation_id", "")
            if cid and cid not in citations:
                citations[cid] = {
                    "type": "metric", "company": target, "role": "target",
                    "document_url":  None,
                    "source_page":   m.get("source_page") or None,
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
# Request/response models
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    question: str
    reasoning: bool = False

class EvalRequest(BaseModel):
    test_type: str


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/pipeline/run")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fiscal_year: str = Form(...),
):
    """Upload an annual report and run all 6 pipeline steps. Returns a job_id."""
    job_id = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename or "report.htm")[1] or ".htm"
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    with open(file_path, "wb") as fh:
        fh.write(await file.read())

    _create_state(job_id, fiscal_year, file_path)
    _ensure_memory_job(job_id)
    background_tasks.add_task(_run_pipeline, job_id, 1)
    return {"job_id": job_id, "filename": file.filename, "fiscal_year": fiscal_year, "steps": STEPS}


@app.post("/pipeline/{job_id}/run")
async def resume_pipeline(job_id: str, background_tasks: BackgroundTasks):
    """Run all remaining incomplete steps for an existing job."""
    state = _load_state(job_id)
    done  = set(state["completed_steps"])
    next_step = next((s["id"] for s in STEPS if s["id"] not in done), None)
    if next_step is None:
        return {"job_id": job_id, "message": "All steps already completed."}
    _ensure_memory_job(job_id)
    background_tasks.add_task(_run_pipeline, job_id, next_step)
    return {"job_id": job_id, "resuming_from_step": next_step}


@app.post("/pipeline/{job_id}/step/{step_id}")
async def run_step(job_id: str, step_id: int, background_tasks: BackgroundTasks):
    """Run or re-run a single step for an existing job."""
    if step_id not in {s["id"] for s in STEPS}:
        raise HTTPException(status_code=400, detail=f"Invalid step_id: {step_id}. Must be 1–6.")

    state = _load_state(job_id)

    # Verify prerequisites: all previous steps must be complete
    for prereq in range(1, step_id):
        if prereq not in state["completed_steps"]:
            raise HTTPException(
                status_code=400,
                detail=f"Step {prereq} must be completed before running step {step_id}.",
            )

    _ensure_memory_job(job_id)

    async def _run():
        loop = asyncio.get_running_loop()
        ok = await _execute_step(job_id, step_id, state, loop)
        if ok:
            _pipeline_done(job_id) if step_id == 6 else None
        else:
            _pipeline_fail(job_id, f"Step {step_id} failed")

    background_tasks.add_task(_run)
    return {"job_id": job_id, "step_id": step_id, "status": "started"}


@app.get("/pipeline/jobs")
async def list_jobs():
    """List all jobs saved on disk."""
    jobs: List[dict] = []
    if not os.path.isdir(OUTPUT_DIR):
        return {"jobs": jobs}
    for name in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, name, "state.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                jobs.append({
                    "job_id":          s["job_id"],
                    "fiscal_year":     s.get("fiscal_year"),
                    "main_company":    s.get("main_company"),
                    "completed_steps": s.get("completed_steps", []),
                    "created_at":      s.get("created_at"),
                })
            except Exception:
                pass
    return {"jobs": jobs}


@app.get("/pipeline/{job_id}/stream")
async def stream_pipeline(job_id: str):
    """Open an SSE stream for live step updates."""
    _load_state(job_id)  # raises 404 if unknown
    _ensure_memory_job(job_id)
    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pipeline/{job_id}/status")
async def get_status(job_id: str):
    """Return the current state of all steps as a JSON snapshot."""
    state = _load_state(job_id)
    job   = _jobs.get(job_id, {})
    return {
        "job_id":          job_id,
        "main_company":    state.get("main_company"),
        "fiscal_year":     state.get("fiscal_year"),
        "completed_steps": state.get("completed_steps", []),
        "completed":       job.get("completed", len(state.get("completed_steps", [])) == len(STEPS)),
        "failed":          job.get("failed", False),
        "error":           job.get("error"),
        "steps":           job.get("steps", {}),
    }


# ---------------------------------------------------------------------------
# QA endpoints
# ---------------------------------------------------------------------------

@app.post("/qa/run")
async def run_qa(request: QARequest):
    """Ask a question against the knowledge graph."""
    loop = asyncio.get_running_loop()

    def _run():
        qa      = _get_qa()
        data    = qa.ask_multi(request.question)
        answer  = _THINK_RE.sub('', data["answer"]).strip()
        queries = data["queries"]
        results = data["results"]

        citations = _build_citations(results)
        _enrich_metric_source_info(citations)

        qa._save_extraction_result(request.question, queries, results)
        qa._save_answer(answer)

        reasoning_trace: Optional[str] = None
        if request.reasoning and results:
            reasoning_trace = qa.generate_reasoning_trace(request.question, queries, results) or None
            if reasoning_trace:
                qa._save_reasoning(reasoning_trace)

        return {"answer": answer, "reasoning_trace": reasoning_trace,
                "queries": queries, "citations": citations}

    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/qa/ready")
async def qa_ready():
    """Return {ready: bool} — true if Neo4j already has graph data."""
    loop = asyncio.get_running_loop()

    def _check():
        try:
            with _neo4j_driver.session() as s:
                cnt = s.run(
                    "MATCH (n) WHERE n:TargetCompany OR n:Company RETURN count(n) AS cnt LIMIT 1"
                ).single()["cnt"]
            return {"ready": cnt > 0, "node_count": cnt}
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    return await loop.run_in_executor(_executor, _check)


# ---------------------------------------------------------------------------
# Eval endpoints
# ---------------------------------------------------------------------------

_RETRIVAL_DIR     = os.path.join(ROOT, "retrival+eval", "retrival_results")
_GROUND_TRUTH_DIR = os.path.join(ROOT, "retrival+eval", "Ground_Truth")
_RESOURCES_DIR    = os.path.join(_GROUND_TRUTH_DIR, "resources")


def _load_eval_module(name: str):
    path = os.path.join(_RESOURCES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    csv.field_size_limit(min(sys.maxsize, 10_000_000))
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _csv_to_scorecard(rows: list, test_type: str) -> Dict[str, Any]:
    if test_type == "answer_relevancy":
        summary  = next((r for r in rows if r.get("generated_question", "").strip() == "** TOP-3 MEAN **"), None)
        weighted = float(summary["answer_relevancy_score"]) if summary else 0.0
        items = [
            {"dimension": r["generated_question"], "score": float(r.get("cosine_similarity") or 0), "note": ""}
            for r in rows if r.get("generated_question", "").strip() != "** TOP-3 MEAN **"
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "context_precision":
        summary  = next((r for r in rows if r.get("citation_id", "").strip() == "** SUMMARY **"), None)
        weighted = float(summary["is_relevant"]) if summary else 0.0
        items = [
            {"dimension": r.get("citation_id", ""),
             "score": 1.0 if str(r.get("is_relevant", "")).lower() == "true" else 0.0,
             "note": r.get("explanation", "")}
            for r in rows if r.get("citation_id", "").strip() != "** SUMMARY **"
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type in ("answer_source_traceability", "faithfulness"):
        data_rows = [r for r in rows if r.get("claim_id", "").strip()]
        total = sum(
            1.0 if str(r.get("is_correct_source", "")).lower() == "true"
            else 0.0 if r.get("correct_source", "").strip().upper() == "FABRICATED"
            else 0.5
            for r in data_rows
        )
        weighted = total / len(data_rows) if data_rows else 0.0
        items = [
            {"dimension": r["claim_id"],
             "score": 1.0 if str(r.get("is_correct_source", "")).lower() == "true" else 0.0,
             "note": r.get("explanation", "")}
            for r in data_rows
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "target_validation":
        data_rows = [r for r in rows if r.get("citation_id", "").strip()]
        correct   = sum(1 for r in data_rows if str(r.get("is_correctly_extracted", "")).lower() == "true")
        weighted  = correct / len(data_rows) if data_rows else 0.0
        items = [
            {"dimension": f"{r.get('type','?')} · {r.get('extracted_name','')}",
             "score": 1.0 if str(r.get("is_correctly_extracted", "")).lower() == "true" else 0.0,
             "note": r.get("explanation", "")}
            for r in data_rows
        ]
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    if test_type == "risk_peers_validation":
        data_rows = [r for r in rows if r.get("company_name", "").strip()]
        relevant  = sum(1 for r in data_rows if str(r.get("is_semantically_relevant", "")).lower() == "true")
        weighted  = relevant / len(data_rows) if data_rows else 0.0
        items = [
            {"dimension": f"{r.get('company_name','')} · {r.get('risk_theme','')}",
             "score": 1.0 if str(r.get("is_semantically_relevant", "")).lower() == "true" else 0.0,
             "note": r.get("relevance_explanation", "")}
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
            if r.get("component", "").strip() == "** OVERALL **":
                continue
            try:
                score = float(r.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            items.append({"dimension": r.get("component", ""), "score": score,
                          "note": f"weight {r.get('weight', '')}"})
        return {"test_type": test_type, "rows": items, "weighted": weighted}

    return {"test_type": test_type, "rows": [], "weighted": 0.0}


@app.post("/eval/run")
async def run_eval(request: EvalRequest):
    """Run an evaluation test and return a scorecard."""
    valid_types = {
        "answer_relevancy", "context_precision", "answer_source_traceability",
        "faithfulness", "context_recall", "target_validation",
        "risk_peers_validation", "overall_score",
    }
    if request.test_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Unknown test_type: {request.test_type}")

    extraction_path = os.path.join(_RETRIVAL_DIR, "extraction_result.json")
    answer_path     = os.path.join(_RETRIVAL_DIR, "answer.txt")

    if request.test_type not in {"overall_score", "context_recall"}:
        for p in (extraction_path, answer_path):
            if not os.path.exists(p):
                raise HTTPException(status_code=404, detail="No QA results found — ask a question first.")

    loop = asyncio.get_running_loop()

    def _run():
        tt = request.test_type
        gt = _GROUND_TRUTH_DIR

        if tt == "answer_relevancy":
            mod = _load_eval_module("answer_relevancy")
            out = os.path.join(gt, "answer_relevancy_results.csv")
            mod.evaluate_relevancy(extraction_path, answer_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "context_precision":
            mod = _load_eval_module("context_precision")
            out = os.path.join(gt, "context_precision_results.csv")
            mod.evaluate_context_precision(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt in ("answer_source_traceability", "faithfulness"):
            mod = _load_eval_module("answer_source_traceability")
            out = os.path.join(gt, "answer_source_traceability.csv")
            mod.evaluate_traceability(extraction_path, answer_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "context_recall":
            os_mod = _load_eval_module("overall_score")
            def _try(fn, path):
                try:
                    return fn(path) if os.path.exists(path) else None
                except Exception:
                    return None
            components = {
                "Target Validation": _try(os_mod.score_target_validation,  os.path.join(gt, "target_validation_results.csv")),
                "Peer Risk Recall":  _try(os_mod.score_risks_validation,   os.path.join(gt, "risks_validation_results.csv")),
                "Metric Accuracy":   _try(os_mod.score_metrics_validation, os.path.join(gt, "metrics_validation_results.csv")),
            }
            scores = [v if v is not None else 0.0 for v in components.values()]
            recall = sum(scores) / len(scores)
            items  = [{"dimension": k, "score": v if v is not None else 0.0,
                       "note": "" if v is not None else "not run yet"} for k, v in components.items()]
            return {"test_type": "context_recall", "rows": items, "weighted": recall}

        if tt == "target_validation":
            mod = _load_eval_module("target_validation")
            out = os.path.join(gt, "target_validation_results.csv")
            mod.validate_target(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "risk_peers_validation":
            mod = _load_eval_module("risk_peers")
            out = os.path.join(gt, "risks_validation_results.csv")
            mod.validate_risk_chunks(extraction_path, out)
            return _csv_to_scorecard(_read_csv(out), tt)

        if tt == "overall_score":
            mod = _load_eval_module("overall_score")
            out = os.path.join(gt, "overall_score.csv")
            mod.compute_overall(gt, out)
            return _csv_to_scorecard(_read_csv(out), tt)

    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
