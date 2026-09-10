"""
Builds and compiles the LangGraph StateGraph wiring all agents with
conditional routing. Routing thresholds live in agents/tiering.py (shared
with escalation_agent.py, which recomputes and persists final_tier itself -
see tiering.py's docstring for why routing functions can't do that directly).
"""
from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.tiering import SARCASM_LOW, SARCASM_HIGH, compute_tier
from agents.classify_agent import classify_node
from agents.sarcasm_agent import sarcasm_node
from agents.clustering_agent import clustering_node
from agents.legal_mapping_agent import legal_mapping_node
from agents.escalation_agent import escalation_node


def _route_by_tier(state: PipelineState) -> str:
    tier = compute_tier(state)
    if tier == "none":
        return END
    if tier in ("medium", "high"):
        return "clustering"  # clustering -> legal_mapping -> escalation (fixed edges)
    return "escalation"  # low tier: flagged but under 0.5 - skip enrichment, still escalate


def route_after_classify(state: PipelineState) -> str:
    if state.get("error"):
        return END
    cf = state.get("confidence_frac", 0.0)
    if SARCASM_LOW <= cf <= SARCASM_HIGH:
        return "sarcasm"
    return _route_by_tier(state)


def route_after_sarcasm(state: PipelineState) -> str:
    return _route_by_tier(state)


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("classify", classify_node)
    graph.add_node("sarcasm", sarcasm_node)
    graph.add_node("clustering", clustering_node)
    graph.add_node("legal_mapping", legal_mapping_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"sarcasm": "sarcasm", "clustering": "clustering", "escalation": "escalation", END: END},
    )
    graph.add_conditional_edges(
        "sarcasm",
        route_after_sarcasm,
        {"clustering": "clustering", "escalation": "escalation", END: END},
    )
    graph.add_edge("clustering", "legal_mapping")
    graph.add_edge("legal_mapping", "escalation")
    graph.add_edge("escalation", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_pipeline(
    text: str,
    username: str = "Anonymous",
    platform: str = "Web",
    source: str = "manual",
    district: str = "Unknown",
) -> PipelineState:
    initial_state: PipelineState = {
        "text": text,
        "username": username,
        "platform": platform,
        "source": source,
        "district": district or "Unknown",
    }
    return await get_graph().ainvoke(initial_state)
