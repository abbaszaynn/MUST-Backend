"""
New API surface for the multi-agent pipeline. Mounted additively into
fastapp1.py (app.include_router(agents_router)) - none of fastapp1.py's
existing routes are modified or removed.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import agents.db as db
import agents.orchestrator as orchestrator
from agents.auth import require_api_key
from agents.ingestion_agent import stream_posts
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


@router.post("/ingest-and-process", dependencies=[Depends(require_api_key)])
async def ingest_and_process(payload: IngestRequest):
    results = []
    flagged_count = 0
    source_used = None
    for post in stream_posts(limit=payload.limit, source=payload.source):
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
