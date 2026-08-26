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


async def sarcasm_node(state: PipelineState) -> PipelineState:
    text = state["text"].lower()
    marker_hits = sum(1 for m in SARCASM_MARKERS if m in text)
    contrast_hit = bool(CONTRAST_PATTERN.search(text))

    score = min(1.0, 0.2 * marker_hits + (0.5 if contrast_hit else 0.0))
    state["sarcasm_score"] = round(score, 2)
    state["sarcasm_flag"] = score >= 0.4
    state["sarcasm_note"] = f"heuristic markers={marker_hits}, contrast_pattern={contrast_hit}"
    return state
