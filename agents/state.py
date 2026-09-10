from typing import TypedDict, Optional, List, Dict, Any


class PipelineState(TypedDict, total=False):
    # --- input ---
    text: str
    username: str
    platform: str
    source: str  # "manual" | "apify" | "sample"
    district: str  # GB district assigned to the monitored target; "Unknown" otherwise

    # --- classify_agent output ---
    error: bool
    message: Optional[str]
    category: Optional[str]  # "neutral" | "offensive" | "hate"
    confidence: Optional[float]  # 0-100
    confidence_frac: Optional[float]  # 0-1, derived
    language: Optional[str]
    scores: Optional[Dict[str, float]]

    # --- sarcasm_agent output ---
    sarcasm_score: Optional[float]
    sarcasm_flag: Optional[bool]
    sarcasm_note: Optional[str]

    # --- clustering_agent output ---
    cluster_id: Optional[int]
    campaign_flag: Optional[bool]
    cluster_size: Optional[int]

    # --- legal_mapping_agent output ---
    legal_matches: Optional[List[Dict[str, Any]]]

    # --- escalation_agent output ---
    requires_human_review: bool  # ALWAYS True once set - see escalation_agent.py
    case_file_id: Optional[int]
    review_queue_id: Optional[int]
    final_tier: Optional[str]  # "none" | "low" | "medium" | "high"
