"""
HEURISTIC PLACEHOLDER - replace with a trained sarcasm classifier when available.
Current approach: rule-based pattern matching, not ML. sarcasm_score is NOT a
calibrated probability and should not be presented to a human reviewer as a
confident signal - it only exists to give the orchestration graph a real
secondary signal for classifications that land in the ambiguous confidence band.
"""
import re

from agents.state import PipelineState

SARCASM_MARKERS = [
    "/s",
    "yeah right",
    "oh sure",
    "totally",
    "as if",
    "wow, great job",
    "thanks a lot",
    "just wonderful",
    "can't wait",
    "love that for",
]

CONTRAST_PATTERN = re.compile(
    r"(?:so (?:smart|great|brilliant|nice)).{0,40}(?:idiot|stupid|hate|kill)"
)

MARKER_WEIGHT = 0.2
CONTRAST_WEIGHT = 0.5
FLAG_THRESHOLD = 0.4


def score_text(text: str) -> dict:
    """Run the heuristic over one string.

    Shared by the graph node and the /sarcasm/test endpoint, so what the
    dashboard demonstrates is the rule the pipeline actually applies rather
    than a re-implementation of it that could drift.
    """
    lowered = (text or "").lower()
    matched = [m for m in SARCASM_MARKERS if m in lowered]
    contrast_hit = bool(CONTRAST_PATTERN.search(lowered))

    score = min(1.0, MARKER_WEIGHT * len(matched) + (CONTRAST_WEIGHT if contrast_hit else 0.0))
    return {
        "score": round(score, 2),
        "flag": score >= FLAG_THRESHOLD,
        "matched_markers": matched,
        "contrast_pattern": contrast_hit,
        "note": f"heuristic markers={len(matched)}, contrast_pattern={contrast_hit}",
    }


async def sarcasm_node(state: PipelineState) -> PipelineState:
    result = score_text(state["text"])
    state["sarcasm_score"] = result["score"]
    state["sarcasm_flag"] = result["flag"]
    state["sarcasm_note"] = result["note"]
    return state
