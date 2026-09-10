"""
New API surface for the multi-agent pipeline. Mounted additively into
fastapp1.py (app.include_router(agents_router)) - none of fastapp1.py's
existing routes are modified or removed.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import agents.db as db
import agents.orchestrator as orchestrator
from agents.auth import require_api_key
from agents.ingestion_agent import MAX_RESULTS, ApifyError, apify_config, stream_posts
import agents.scrape_jobs as scrape_jobs
from urllib.parse import parse_qs, urlparse
from agents.state import PipelineState

# `import agents.orchestrator as orchestrator` (not `from ... import run_pipeline`)
# is deliberate: fastapp1.py imports this router at the bottom of its own module
# body, and agents.orchestrator imports agents.classify_agent, which imports
# fastapp1 - when agents.orchestrator is the entry point (e.g. the eval harness,
# `python -m eval.run_eval`), that cycle reaches this file while orchestrator.py
# is still mid-import, before run_pipeline is defined. Binding the module object
# instead of the name defers the attribute lookup (orchestrator.run_pipeline) to
# call time, by which point the whole import chain has finished.

router = APIRouter()


class ProcessRequest(BaseModel):
    text: str
    username: str = "Anonymous"
    platform: str = "Web"


class IngestRequest(BaseModel):
    limit: int = 20
    source: str = "auto"  # "auto" | "apify" | "sample"
    urls: Optional[List[str]] = None  # public Facebook page URLs; defaults to APIFY_START_URLS


class DecisionRequest(BaseModel):
    decision: str  # confirm_violation | dismiss | escalate_external
    decided_by: Optional[str] = None
    notes: Optional[str] = None


def _state_to_response(state: PipelineState) -> Dict[str, Any]:
    flagged = bool(state.get("requires_human_review"))
    sarcasm = None
    if state.get("sarcasm_score") is not None:
        sarcasm = {
            "score": state.get("sarcasm_score"),
            "flag": state.get("sarcasm_flag"),
            "note": state.get("sarcasm_note"),
        }
    cluster = None
    if state.get("cluster_id") is not None:
        cluster = {
            "cluster_id": state.get("cluster_id"),
            "campaign_flag": state.get("campaign_flag"),
            "cluster_size": state.get("cluster_size"),
        }
    return {
        "error": bool(state.get("error", False)),
        "message": state.get("message"),
        "category": state.get("category"),
        "confidence": state.get("confidence"),
        "language": state.get("language"),
        "scores": state.get("scores"),
        "sarcasm": sarcasm,
        "cluster": cluster,
        "legal_matches": state.get("legal_matches") if flagged else None,
        "requires_human_review": flagged,
        "case_file_id": state.get("case_file_id"),
        "review_queue_id": state.get("review_queue_id"),
        "final_tier": state.get("final_tier", "none"),
    }


@router.post("/process", dependencies=[Depends(require_api_key)])
async def process_text(payload: ProcessRequest):
    state = await orchestrator.run_pipeline(
        text=payload.text, username=payload.username, platform=payload.platform, source="manual"
    )
    return JSONResponse(content=_state_to_response(state))


@router.get("/ingest/status", dependencies=[Depends(require_api_key)])
async def ingest_status():
    """Whether live ingestion is configured - never returns the token itself."""
    token, actor, urls = apify_config()
    return JSONResponse(
        content={
            "error": False,
            "apify_configured": bool(token),
            "actor": actor,
            "start_urls": urls,
            "max_results_per_page": MAX_RESULTS,
        }
    )


@router.post("/ingest-and-process", dependencies=[Depends(require_api_key)])
async def ingest_and_process(payload: IngestRequest):
    results = []
    flagged_count = 0
    source_used = None
    try:
        posts = list(stream_posts(limit=payload.limit, source=payload.source, urls=payload.urls))
    except ApifyError as e:
        return JSONResponse(
            content={
                "error": True,
                "message": str(e),
                "source_used": payload.source,
                "processed": 0,
                "flagged": 0,
                "results": [],
            },
            status_code=502,
        )
    for post in posts:
        source_used = post.get("source", source_used)
        state = await orchestrator.run_pipeline(
            text=post["text"], username=post["username"], platform=post["platform"], source=source_used
        )
        item = _state_to_response(state)
        item["text"] = post["text"]
        item["username"] = post["username"]
        item["platform"] = post["platform"]
        if item["requires_human_review"]:
            flagged_count += 1
        results.append(item)

    return JSONResponse(
        content={
            "error": False,
            "source_used": source_used or "sample",
            "processed": len(results),
            "flagged": flagged_count,
            "results": results,
        }
    )


@router.get("/review-queue", dependencies=[Depends(require_api_key)])
async def get_review_queue(
    status: str = "open", platform: Optional[str] = None, limit: int = 50, offset: int = 0
):
    total, rows = db.get_review_queue(status=status, platform=platform, limit=limit, offset=offset)
    return JSONResponse(content={"error": False, "total": total, "data": rows})


@router.post("/review-queue/{review_queue_id}/decision", dependencies=[Depends(require_api_key)])
async def decide_review_queue_item(review_queue_id: int, payload: DecisionRequest):
    ok = db.record_review_decision(
        review_queue_id=review_queue_id,
        decision=payload.decision,
        decided_by=payload.decided_by,
        notes=payload.notes,
    )
    if not ok:
        return JSONResponse(
            content={"error": True, "message": f"review_queue id {review_queue_id} not found"},
            status_code=404,
        )
    return JSONResponse(
        content={
            "error": False,
            "review_queue_id": review_queue_id,
            "status": "closed",
            "decision": payload.decision,
        }
    )


@router.get("/stats/districts", dependencies=[Depends(require_api_key)])
async def stats_districts():
    return JSONResponse(content={"error": False, "data": db.get_district_stats()})


@router.get("/stats/platforms", dependencies=[Depends(require_api_key)])
async def stats_platforms():
    return JSONResponse(content={"error": False, "data": db.get_platform_stats()})


@router.get("/legal-reference")
async def legal_reference():
    """Read-only listing of the static legal provision reference table (agents/legal_reference.json),
    so the frontend shows the same canonical source the Legal Mapping Agent matches against,
    instead of duplicating it."""
    from agents.legal_mapping_agent import LEGAL_REFERENCE

    return JSONResponse(content={"error": False, "data": LEGAL_REFERENCE})


# --- Monitored targets --------------------------------------------------------

FACEBOOK_HOSTS = {"facebook.com", "www.facebook.com", "m.facebook.com", "web.facebook.com"}


class TargetRequest(BaseModel):
    profile_url: str
    name: Optional[str] = None
    district: Optional[str] = None  # officer-assigned GB district for this page


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    ids = parse_qs(parsed.query).get("id")
    if ids:
        return f"fb_{ids[0]}"
    segments = [s for s in parsed.path.split("/") if s]
    return segments[0] if segments else url


@router.post("/targets", dependencies=[Depends(require_api_key)])
async def add_target(payload: TargetRequest):
    url = payload.profile_url.strip()
    if urlparse(url).netloc.lower() not in FACEBOOK_HOSTS:
        return JSONResponse(
            content={"error": True, "message": "Enter a public Facebook page URL, e.g. https://www.facebook.com/PageName/"},
            status_code=400,
        )
    name = (payload.name or "").strip() or _name_from_url(url)
    user_id = db.add_target(name, url, (payload.district or "").strip() or "Unknown")
    return JSONResponse(content={"error": False, "id": user_id, "username": name})


@router.post("/targets/{user_id}/scrape", dependencies=[Depends(require_api_key)])
async def scrape_target(user_id: int, limit: int = 10):
    if not db.get_target(user_id):
        return JSONResponse(content={"error": True, "message": "Target not found."}, status_code=404)
    try:
        run_id = scrape_jobs.start_target_scrape(user_id, limit)
    except ApifyError as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=400)
    return JSONResponse(content={"error": False, "scrape_run_id": run_id, "status": "RUNNING"})


@router.get("/scrape-runs", dependencies=[Depends(require_api_key)])
async def scrape_runs(user_id: Optional[int] = None, limit: int = 20):
    return JSONResponse(content={"error": False, "data": db.list_scrape_runs(user_id, limit)})


@router.get("/targets/{user_id}/posts", dependencies=[Depends(require_api_key)])
async def target_posts(user_id: int, limit: int = 300):
    """Posts and comments scraped from a monitored page."""
    if not db.get_target(user_id):
        return JSONResponse(content={"error": True, "message": "Target not found."}, status_code=404)
    return JSONResponse(content={"error": False, "data": db.get_target_items(user_id, limit)})
