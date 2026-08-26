"""
Maps flagged content to actual legal provisions rather than just a risk score.
Matching is keyword/category-based against the classified text - deliberately
not dependent on LIME/SHAP explanation features (fastapp2.py's LIME/SHAP path
has known reliability issues and fastapp1.py is being kept untouched/stable).

This is an assistive drafting aid for a human reviewer (ideally with legal
training) to confirm, never an authoritative legal determination. If no
provision matches a hate/offensive item, an explicit "unmapped" entry is
returned rather than fabricating or omitting a citation.
"""
import json
import os

from agents.state import PipelineState

_REF_PATH = os.path.join(os.path.dirname(__file__), "legal_reference.json")
with open(_REF_PATH, "r", encoding="utf-8") as f:
    LEGAL_REFERENCE = json.load(f)


async def legal_mapping_node(state: PipelineState) -> PipelineState:
    text = state["text"].lower()
    category = state.get("category")

    matches = []
    for provision in LEGAL_REFERENCE:
        if category not in provision.get("categories", []):
            continue
        hits = [kw for kw in provision["keywords"] if kw in text]
        if hits:
            confidence = round(min(1.0, 0.3 + 0.15 * len(hits)), 2)
            matches.append(
                {
                    "id": provision["id"],
                    "law": provision["law"],
                    "section": provision["section"],
                    "title": provision["title"],
                    "jurisdiction": provision["jurisdiction"],
                    "matched_keywords": hits,
                    "confidence": confidence,
                }
            )

    if not matches:
        matches.append(
            {
                "id": "unmapped",
                "law": None,
                "section": "NOT YET MAPPED",
                "title": None,
                "jurisdiction": None,
                "matched_keywords": [],
                "confidence": 0.0,
                "notes": "No keyword match; requires manual legal review. Do not fabricate a citation.",
            }
        )

    state["legal_matches"] = matches
    return state
