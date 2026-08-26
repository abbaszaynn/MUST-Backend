"""
Wraps the existing, unmodified hate-speech classifier. Does not reimplement or
touch the model - calls fastapp1.predict_text() exactly as fastapp1.py's own
/analyze route does, and normalizes the result into PipelineState. Also logs
via fastapp1.log_analysis() (the same untouched helper /analyze and the Apify
webhook already use) so /trends, /flagged, /live-feed, /monitoring, and
/stats/platforms stay meaningful for text processed through this pipeline too
- otherwise the rest of the existing dashboard would stay blind to anything
analyzed via /process or /ingest-and-process.
"""
import fastapp1
from agents.state import PipelineState


async def classify_node(state: PipelineState) -> PipelineState:
    result = await fastapp1.predict_text(state["text"])

    if result.get("error"):
        state["error"] = True
        state["message"] = result.get("message")
        return state

    state["error"] = False
    state["category"] = result["category"]
    state["confidence"] = result["confidence"]
    state["confidence_frac"] = result["confidence"] / 100.0
    state["language"] = result["language"]
    state["scores"] = result["scores"]
    state["text"] = result["text"]

    fastapp1.log_analysis(
        result["text"],
        result["category"],
        result["confidence"],
        result["language"],
        state.get("username", "Anonymous"),
        state.get("platform", "Web"),
    )
    return state
