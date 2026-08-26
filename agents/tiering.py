"""
Shared, pure tiering logic used both for graph routing (orchestrator.py) and
for computing the persisted final_tier at escalation time (escalation_agent.py).

Kept in one place deliberately: LangGraph's conditional-edge routing functions
do not have their state mutations merged back into the graph's state (only a
node's return value is merged) - so tier must be (re)computed inside a real
node (escalation_node) rather than trusted from a routing-function side effect.
Centralizing the logic here means routing and persistence can never disagree
about what counts as "flagged" or which confidence band maps to which tier.
"""
from agents.state import PipelineState

SARCASM_LOW, SARCASM_HIGH = 0.4, 0.7  # ambiguous confidence band -> run sarcasm heuristic
TIER_MEDIUM_MIN = 0.5  # clustering + legal mapping fire at/above this
TIER_HIGH_MIN = 0.85


def is_flagged(state: PipelineState) -> bool:
    return state.get("category") in ("hate", "offensive") or bool(state.get("sarcasm_flag"))


def compute_tier(state: PipelineState) -> str:
    if not is_flagged(state):
        return "none"
    cf = state.get("confidence_frac", 0.0)
    if cf >= TIER_HIGH_MIN:
        return "high"
    if cf >= TIER_MEDIUM_MIN:
        return "medium"
    return "low"
