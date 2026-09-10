"""
requires_human_review is ALWAYS True for every case file this module writes.
It is a hardcoded literal in agents/db.py's insert_case_file() SQL text, never
a variable derived from any input or branch here - there is no parameter, no
config flag, and no code path in this file that can produce False. This is
backed by a DB-level CHECK (requires_human_review = 1) constraint on the
case_files table (see agents/db.py), so even a future bug or a raw SQL edit
elsewhere cannot silently insert a case file that skips human review. This is
the accountability boundary: no agent in this pipeline ever takes autonomous
enforcement action.
"""
import agents.db as db
from agents.state import PipelineState
from agents.tiering import compute_tier

PRIORITY_BY_TIER = {"high": "high", "medium": "medium", "low": "low"}


async def escalation_node(state: PipelineState) -> PipelineState:
    # Recomputed here, not trusted from routing-function state mutation - see
    # agents/tiering.py's docstring for why. This is what actually persists.
    tier = compute_tier(state)
    state["final_tier"] = tier

    case_file_id = db.insert_case_file(
        text=state["text"],
        category=state.get("category"),
        confidence=state.get("confidence"),
        language=state.get("language"),
        username=state.get("username", "Anonymous"),
        platform=state.get("platform", "Web"),
        district=state.get("district") or "Unknown",
        cluster_id=state.get("cluster_id"),
        campaign_flag=state.get("campaign_flag", False),
        legal_provisions=state.get("legal_matches", []),
        sarcasm_score=state.get("sarcasm_score"),
        sarcasm_flag=state.get("sarcasm_flag", False),
    )

    review_queue_id = db.insert_review_queue_entry(
        case_file_id=case_file_id, priority=PRIORITY_BY_TIER.get(tier, "low")
    )

    state["requires_human_review"] = True
    state["case_file_id"] = case_file_id
    state["review_queue_id"] = review_queue_id
    return state
