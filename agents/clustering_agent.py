"""
SIMPLIFIED CLUSTERING - placeholder for ULTRA integration. Current approach:
multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2) +
incremental nearest-centroid cosine-similarity grouping, one item at a time -
this fits the per-item graph invocation model. A batch method like DBSCAN would
need the whole point set up front and isn't a good fit for a streaming node;
it's left as a possible future offline/periodic re-clustering job, not built here.
"""
from agents.state import PipelineState
import agents.db as db

SIMILARITY_THRESHOLD = 0.80
CAMPAIGN_MIN_MEMBERS = 3

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder


async def clustering_node(state: PipelineState) -> PipelineState:
    embedder = _get_embedder()
    vec = embedder.encode([state["text"]])[0].tolist()

    cluster_id, size = db.assign_to_nearest_cluster(
        vec, state["text"], state.get("platform"), similarity_threshold=SIMILARITY_THRESHOLD
    )
    campaign_flag = size >= CAMPAIGN_MIN_MEMBERS
    if campaign_flag:
        db.mark_cluster_campaign(cluster_id)

    state["cluster_id"] = cluster_id
    state["campaign_flag"] = campaign_flag
    state["cluster_size"] = size
    return state
